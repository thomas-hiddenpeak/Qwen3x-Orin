#include "q3x/kernels/sm87_a4w4_factorized_lane_quantize.h"
#include "q3x/kernels/sm87_a4w4_gateup_factorized_lane.h"
#include "q3x/kernels/sm87_a4w4_gateup_r1_product_finalize.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

inline constexpr std::size_t kLogicalTokens = 1'853U;
inline constexpr std::size_t kGateLaunchTokens = 1'920U;
inline constexpr std::size_t kDownLaunchTokens = 2'048U;
inline constexpr std::size_t kInputSize =
    kernels::kSm87A4W4GateUpFactorizedModelInput;
inline constexpr std::size_t kIntermediateSize =
    kernels::kSm87A4W4GateUpFactorizedModelIntermediate;
inline constexpr std::size_t kPrimarySize =
    kernels::kSm87A4W4GateUpFactorizedPrimaryWidth;
inline constexpr std::size_t kPrimaryStride =
    kernels::kSm87A4W4GateUpFactorizedPrimaryStride;
inline constexpr std::size_t kSecondarySize =
    kernels::kSm87A4W4GateUpFactorizedSecondaryWidth;
inline constexpr std::size_t kSecondaryStride =
    kernels::kSm87A4W4GateUpFactorizedSecondaryStride;
inline constexpr std::size_t kPartialTiles =
    kernels::kSm87A4W4GateUpFactorizedR1ProductPartialTiles;
inline constexpr float kClipRatio = 0.875F;

inline constexpr std::size_t kWordGuard = 8U;
inline constexpr std::size_t kFloatGuard = 4U;
inline constexpr std::size_t kByteGuard = 64U;
inline constexpr std::uint16_t kProductSentinel = 0xa55aU;
inline constexpr std::uint16_t kScaleSentinel = 0xbeefU;
inline constexpr std::uint8_t kPackedSentinel = 0xa5U;
inline constexpr float kPartialSentinel = -12'345.0F;

static_assert(kInputSize == 5'120U);
static_assert(kIntermediateSize == 17'408U);
static_assert(kPrimarySize == 12'288U);
static_assert(kPrimaryStride == kPrimarySize);
static_assert(kSecondarySize == 5'120U);
static_assert(kSecondaryStride == 6'144U);
static_assert(kPartialTiles == 136U);
static_assert(kernels::sm87_a4w4_gateup_factorized_launch_token_count(
                  kLogicalTokens) == kGateLaunchTokens);
static_assert(kernels::sm87_a4w4_down_factorized_launch_token_count(
                  kLogicalTokens) == kDownLaunchTokens);

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t elements) noexcept {
    return elements != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&data_),
                      elements * sizeof(T)) == cudaSuccess;
  }

  [[nodiscard]] T* get() const noexcept { return data_; }

 private:
  T* data_{};
};

class Stream final {
 public:
  Stream() = default;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] bool create() noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) ==
           cudaSuccess;
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_{};
};

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const std::string& operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << ": " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ")\n";
  return false;
}

[[nodiscard]] bool launch_ok(const int status, const std::string& operation) {
  return cuda_ok(static_cast<cudaError_t>(status), operation);
}

[[nodiscard]] int target_device_status() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      count == 0) {
    (void)cudaGetLastError();
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  if (!cuda_ok(status, "cudaGetDeviceCount")) {
    return 1;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the 16-SM SM87 target\n";
    return 77;
  }
  return 0;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

struct HostInputs final {
  std::vector<std::uint8_t> packed_a;
  std::vector<std::uint16_t> a_scales;
  std::vector<std::uint8_t> packed_gate;
  std::vector<std::uint16_t> gate_scales;
  std::vector<std::uint8_t> packed_up;
  std::vector<std::uint16_t> up_scales;
  std::vector<float> down_inverse_alpha;
};

[[nodiscard]] HostInputs make_inputs() {
  const std::size_t a_packed_capacity =
      kernels::sm87_a4w4_gateup_factorized_packed_capacity_bytes(
          kGateLaunchTokens, kInputSize);
  const std::size_t a_scale_capacity =
      kernels::sm87_a4w4_gateup_factorized_scale_capacity_elements(
          kGateLaunchTokens);
  const std::size_t weight_packed_capacity =
      kernels::sm87_a4w4_gateup_factorized_packed_capacity_bytes(
          kIntermediateSize, kInputSize);
  const std::size_t weight_scale_capacity =
      kernels::sm87_a4w4_gateup_factorized_scale_capacity_elements(
          kIntermediateSize);
  HostInputs result{
      std::vector<std::uint8_t>(a_packed_capacity, 0U),
      std::vector<std::uint16_t>(a_scale_capacity, encode_bf16(1.0F)),
      std::vector<std::uint8_t>(weight_packed_capacity, 0x21U),
      std::vector<std::uint16_t>(weight_scale_capacity),
      std::vector<std::uint8_t>(weight_packed_capacity, 0x11U),
      std::vector<std::uint16_t>(weight_scale_capacity),
      std::vector<float>(kIntermediateSize)};

  const std::size_t input_k64_groups = kInputSize / 64U;
  for (std::size_t row = 1U; row < kLogicalTokens; ++row) {
    const int even_code = 1 + static_cast<int>(row & 1U);
    const int odd_code = 1;
    const std::uint8_t packed =
        kernels::sm87_a4w4_pack_signed_pair(even_code, odd_code);
    for (std::size_t group = 0U; group < input_k64_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        result.packed_a[kernels::sm87_a4w4_consumer_packed_offset(
            row, group, byte, input_k64_groups)] = packed;
      }
    }
    const float scale = (row & 2U) == 0U ? 0.015625F : 0.03125F;
    result.a_scales[kernels::sm87_a4w4_gateup_factorized_scale_offset(row)] =
        encode_bf16(scale);
  }

  for (std::size_t n = 0U; n < kIntermediateSize; ++n) {
    float gate_scale = ((n / 128U) & 1U) == 0U ? 0.0078125F : 0.015625F;
    float up_scale = ((n / 128U) & 2U) == 0U ? 0.00390625F : 0.0078125F;
    if (n + 2U >= kPrimarySize && n < kPrimarySize) {
      gate_scale = 0.00390625F;
      up_scale = 0.00390625F;
    } else if (n >= kPrimarySize && n < kPrimarySize + 2U) {
      gate_scale = 0.03125F;
      up_scale = 0.015625F;
    }
    const std::size_t offset =
        kernels::sm87_a4w4_gateup_factorized_scale_offset(n);
    result.gate_scales[offset] = encode_bf16(gate_scale);
    result.up_scales[offset] = encode_bf16(up_scale);
  }

  for (std::size_t k = 0U; k < kIntermediateSize; ++k) {
    constexpr float kValues[] = {0.25F, 0.5F, 1.0F, 2.0F};
    result.down_inverse_alpha[k] = kValues[(k / 128U) & 3U];
  }
  result.down_inverse_alpha[kPrimarySize - 2U] = 0.125F;
  result.down_inverse_alpha[kPrimarySize - 1U] = 0.125F;
  result.down_inverse_alpha[kPrimarySize] = 4.0F;
  result.down_inverse_alpha[kPrimarySize + 1U] = 4.0F;
  return result;
}

struct DeviceInputs final {
  DeviceBuffer<std::uint8_t> packed_a;
  DeviceBuffer<std::uint16_t> a_scales;
  DeviceBuffer<std::uint8_t> packed_gate;
  DeviceBuffer<std::uint16_t> gate_scales;
  DeviceBuffer<std::uint8_t> packed_up;
  DeviceBuffer<std::uint16_t> up_scales;
  DeviceBuffer<float> down_inverse_alpha;
};

[[nodiscard]] bool upload_inputs(const HostInputs& host,
                                 DeviceInputs* const device) {
  if (device == nullptr || !device->packed_a.allocate(host.packed_a.size()) ||
      !device->a_scales.allocate(host.a_scales.size()) ||
      !device->packed_gate.allocate(host.packed_gate.size()) ||
      !device->gate_scales.allocate(host.gate_scales.size()) ||
      !device->packed_up.allocate(host.packed_up.size()) ||
      !device->up_scales.allocate(host.up_scales.size()) ||
      !device->down_inverse_alpha.allocate(host.down_inverse_alpha.size())) {
    std::cerr << "input allocation failed\n";
    return false;
  }
  return cuda_ok(cudaMemcpy(device->packed_a.get(), host.packed_a.data(),
                            host.packed_a.size(), cudaMemcpyHostToDevice),
                 "copy packed A") &&
         cuda_ok(cudaMemcpy(device->a_scales.get(), host.a_scales.data(),
                            host.a_scales.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "copy A scales") &&
         cuda_ok(cudaMemcpy(device->packed_gate.get(), host.packed_gate.data(),
                            host.packed_gate.size(), cudaMemcpyHostToDevice),
                 "copy Gate weights") &&
         cuda_ok(cudaMemcpy(device->gate_scales.get(), host.gate_scales.data(),
                            host.gate_scales.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "copy Gate scales") &&
         cuda_ok(cudaMemcpy(device->packed_up.get(), host.packed_up.data(),
                            host.packed_up.size(), cudaMemcpyHostToDevice),
                 "copy Up weights") &&
         cuda_ok(cudaMemcpy(device->up_scales.get(), host.up_scales.data(),
                            host.up_scales.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "copy Up scales") &&
         cuda_ok(cudaMemcpy(device->down_inverse_alpha.get(),
                            host.down_inverse_alpha.data(),
                            host.down_inverse_alpha.size() * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "copy Down inverse-alpha");
}

struct DeviceOutputs final {
  DeviceBuffer<std::uint16_t> baseline_primary;
  DeviceBuffer<std::uint16_t> baseline_secondary;
  DeviceBuffer<std::uint16_t> candidate_primary;
  DeviceBuffer<std::uint16_t> candidate_secondary;
  DeviceBuffer<std::uint8_t> baseline_packed;
  DeviceBuffer<std::uint8_t> candidate_packed;
  DeviceBuffer<std::uint16_t> baseline_scales;
  DeviceBuffer<std::uint16_t> candidate_scales;
  DeviceBuffer<float> partials;
};

struct Capacities final {
  std::size_t primary_elements{};
  std::size_t secondary_elements{};
  std::size_t packed_bytes{};
  std::size_t scale_elements{};
  std::size_t partial_elements{};
};

[[nodiscard]] Capacities capacities() {
  const auto plan = kernels::sm87_a4w4_gateup_r1_product_finalize_plan(
      kLogicalTokens, kDownLaunchTokens);
  return {kLogicalTokens * kPrimaryStride,
          kLogicalTokens * kSecondaryStride,
          plan.packed_capacity_bytes,
          plan.scale_capacity_elements,
          plan.partial_capacity_elements};
}

template <typename T>
[[nodiscard]] bool initialize_buffer(DeviceBuffer<T>* const buffer,
                                     const std::size_t payload_elements,
                                     const std::size_t guard_elements,
                                     const T value,
                                     const std::string& label) {
  const std::size_t total = payload_elements + 2U * guard_elements;
  if (buffer == nullptr || !buffer->allocate(total)) {
    std::cerr << label << " allocation failed\n";
    return false;
  }
  const std::vector<T> initial(total, value);
  return cuda_ok(cudaMemcpy(buffer->get(), initial.data(), total * sizeof(T),
                            cudaMemcpyHostToDevice),
                 label + " initialize");
}

[[nodiscard]] bool allocate_outputs(const Capacities& cap,
                                    DeviceOutputs* const outputs) {
  return outputs != nullptr &&
         initialize_buffer(&outputs->baseline_primary, cap.primary_elements,
                           kWordGuard, kProductSentinel,
                           "baseline primary") &&
         initialize_buffer(&outputs->baseline_secondary,
                           cap.secondary_elements, kWordGuard,
                           kProductSentinel, "baseline secondary") &&
         initialize_buffer(&outputs->candidate_primary, cap.primary_elements,
                           kWordGuard, kProductSentinel,
                           "candidate primary") &&
         initialize_buffer(&outputs->candidate_secondary,
                           cap.secondary_elements, kWordGuard,
                           kProductSentinel, "candidate secondary") &&
         initialize_buffer(&outputs->baseline_packed, cap.packed_bytes,
                           kByteGuard, kPackedSentinel, "baseline packed") &&
         initialize_buffer(&outputs->candidate_packed, cap.packed_bytes,
                           kByteGuard, kPackedSentinel, "candidate packed") &&
         initialize_buffer(&outputs->baseline_scales, cap.scale_elements,
                           kWordGuard, kScaleSentinel, "baseline scales") &&
         initialize_buffer(&outputs->candidate_scales, cap.scale_elements,
                           kWordGuard, kScaleSentinel, "candidate scales") &&
         initialize_buffer(&outputs->partials, cap.partial_elements,
                           kFloatGuard, kPartialSentinel, "tile maxima");
}

[[nodiscard]] bool query_resource_contract() {
  kernels::Sm87A4W4FactorizedLaneQuantizeResources quantize{};
  kernels::Sm87A4W4GateUpFactorizedLaneResources gate{};
  kernels::Sm87A4W4GateUpR1ProductFinalizeResources finalize{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_factorized_lane_quantize_resources_cuda(
              &quantize),
          "query incumbent quantizer resources") ||
      !launch_ok(
          kernels::query_sm87_a4w4_gateup_factorized_lane_r1_tile_max_resources_cuda(
              &gate),
          "query Gate tile-max resources") ||
      !launch_ok(
          kernels::query_sm87_a4w4_gateup_r1_product_finalize_resources_cuda(
              &finalize),
          "query product finalizer resources")) {
    return false;
  }
  const bool valid =
      quantize.registers_per_thread > 0 && quantize.local_bytes == 0U &&
      quantize.active_blocks_per_sm >= 2 &&
      quantize.maximum_threads_per_block >= 256 &&
      gate.registers_per_thread > 0 &&
      gate.registers_per_thread <=
          static_cast<int>(
              kernels::kSm87A4W4GateUpFactorizedMaximumRegisters) &&
      gate.local_bytes == 0U && gate.active_blocks_per_sm >= 1 &&
      gate.maximum_threads_per_block >= 512 &&
      finalize.registers_per_thread > 0 &&
      finalize.registers_per_thread <=
          static_cast<int>(
              kernels::kSm87A4W4GateUpR1ProductFinalizeMaximumRegisters) &&
      finalize.local_bytes == 0U && finalize.active_blocks_per_sm >= 2 &&
      finalize.maximum_threads_per_block >= 256 &&
      finalize.compute_major == 8 && finalize.compute_minor == 7 &&
      finalize.multiprocessor_count == 16;
  if (!valid) {
    std::cerr << "resource contract failed: quant(regs="
              << quantize.registers_per_thread
              << ",local=" << quantize.local_bytes
              << ",active=" << quantize.active_blocks_per_sm
              << ") gate(regs=" << gate.registers_per_thread
              << ",local=" << gate.local_bytes
              << ",active=" << gate.active_blocks_per_sm
              << ") finalize(regs=" << finalize.registers_per_thread
              << ",local=" << finalize.local_bytes
              << ",active=" << finalize.active_blocks_per_sm << ")\n";
    return false;
  }
  std::cout << "resources PASS: gate_tile_max_regs="
            << gate.registers_per_thread
            << " finalizer_regs=" << finalize.registers_per_thread
            << " finalizer_active=" << finalize.active_blocks_per_sm << '\n';
  return true;
}

[[nodiscard]] bool launch_paths(const DeviceInputs& inputs,
                                const Capacities& cap,
                                DeviceOutputs* const outputs,
                                const cudaStream_t stream) {
  if (outputs == nullptr) {
    return false;
  }
  std::uint16_t* const baseline_primary =
      outputs->baseline_primary.get() + kWordGuard;
  std::uint16_t* const baseline_secondary =
      outputs->baseline_secondary.get() + kWordGuard;
  std::uint16_t* const candidate_primary =
      outputs->candidate_primary.get() + kWordGuard;
  std::uint16_t* const candidate_secondary =
      outputs->candidate_secondary.get() + kWordGuard;
  std::uint8_t* const baseline_packed =
      outputs->baseline_packed.get() + kByteGuard;
  std::uint8_t* const candidate_packed =
      outputs->candidate_packed.get() + kByteGuard;
  std::uint16_t* const baseline_scales =
      outputs->baseline_scales.get() + kWordGuard;
  std::uint16_t* const candidate_scales =
      outputs->candidate_scales.get() + kWordGuard;
  float* const partials = outputs->partials.get() + kFloatGuard;

  const std::size_t packed_a_capacity =
      kernels::sm87_a4w4_gateup_factorized_packed_capacity_bytes(
          kGateLaunchTokens, kInputSize);
  const std::size_t a_scale_capacity =
      kernels::sm87_a4w4_gateup_factorized_scale_capacity_elements(
          kGateLaunchTokens);
  const std::size_t packed_weight_capacity =
      kernels::sm87_a4w4_gateup_factorized_packed_capacity_bytes(
          kIntermediateSize, kInputSize);
  const std::size_t weight_scale_capacity =
      kernels::sm87_a4w4_gateup_factorized_scale_capacity_elements(
          kIntermediateSize);

  return launch_ok(
             kernels::launch_sm87_a4w4_gateup_factorized_lane_bf16_cuda(
                 inputs.packed_a.get(), packed_a_capacity,
                 inputs.a_scales.get(), a_scale_capacity,
                 inputs.packed_gate.get(), packed_weight_capacity,
                 inputs.gate_scales.get(), weight_scale_capacity,
                 inputs.packed_up.get(), packed_weight_capacity,
                 inputs.up_scales.get(), weight_scale_capacity,
                 kLogicalTokens, kGateLaunchTokens, kIntermediateSize,
                 kInputSize, baseline_primary, kPrimaryStride,
                 cap.primary_elements, baseline_secondary, kSecondaryStride,
                 cap.secondary_elements, stream),
             "baseline GateUp") &&
         launch_ok(
             kernels::launch_sm87_a4w4_factorized_lane_quantize_bf16_split_cuda(
                 baseline_primary, kPrimaryStride, cap.primary_elements,
                 kPrimarySize, baseline_secondary, kSecondaryStride,
                 cap.secondary_elements, kSecondarySize,
                 inputs.down_inverse_alpha.get(), kIntermediateSize,
                 kLogicalTokens, kDownLaunchTokens, 1U, kClipRatio,
                 baseline_packed, cap.packed_bytes, baseline_scales,
                 cap.scale_elements, stream),
             "incumbent split quantizer") &&
         launch_ok(
             kernels::launch_sm87_a4w4_gateup_factorized_lane_r1_tile_max_bf16_cuda(
                 inputs.packed_a.get(), packed_a_capacity,
                 inputs.a_scales.get(), a_scale_capacity,
                 inputs.packed_gate.get(), packed_weight_capacity,
                 inputs.gate_scales.get(), weight_scale_capacity,
                 inputs.packed_up.get(), packed_weight_capacity,
                 inputs.up_scales.get(), weight_scale_capacity,
                 kLogicalTokens, kGateLaunchTokens, kIntermediateSize,
                 kInputSize, candidate_primary, kPrimaryStride,
                 cap.primary_elements, candidate_secondary, kSecondaryStride,
                 cap.secondary_elements, inputs.down_inverse_alpha.get(),
                 kIntermediateSize, partials, cap.partial_elements, stream),
             "candidate GateUp tile-max") &&
         launch_ok(
             kernels::launch_sm87_a4w4_gateup_r1_product_finalize_cuda(
                 candidate_primary, kPrimaryStride, cap.primary_elements,
                 candidate_secondary, kSecondaryStride,
                 cap.secondary_elements, inputs.down_inverse_alpha.get(),
                 kIntermediateSize, partials, cap.partial_elements,
                 kLogicalTokens, kDownLaunchTokens, kClipRatio,
                 candidate_packed, cap.packed_bytes, candidate_scales,
                 cap.scale_elements, stream),
             "candidate product finalizer");
}

template <typename T>
[[nodiscard]] bool copy_device_vector(const DeviceBuffer<T>& source,
                                      const std::size_t elements,
                                      std::vector<T>* const output,
                                      const std::string& label) {
  if (output == nullptr) {
    return false;
  }
  output->resize(elements);
  return cuda_ok(cudaMemcpy(output->data(), source.get(),
                            elements * sizeof(T), cudaMemcpyDeviceToHost),
                 label);
}

template <typename T>
[[nodiscard]] bool compare_payload(const std::string& label,
                                   const T* const baseline,
                                   const T* const candidate,
                                   const std::size_t elements) {
  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < elements; ++index) {
    if (baseline[index] != candidate[index]) {
      ++mismatches;
      if (mismatches <= 8U) {
        std::cerr << label << " mismatch index=" << index << " baseline=";
        if constexpr (std::is_same_v<T, std::uint8_t>) {
          std::cerr << static_cast<unsigned int>(baseline[index]);
        } else {
          std::cerr << baseline[index];
        }
        std::cerr << " candidate=";
        if constexpr (std::is_same_v<T, std::uint8_t>) {
          std::cerr << static_cast<unsigned int>(candidate[index]);
        } else {
          std::cerr << candidate[index];
        }
        std::cerr << '\n';
      }
    }
  }
  if (mismatches != 0U) {
    std::cerr << label << " FAIL mismatches=" << mismatches << '\n';
    return false;
  }
  return true;
}

template <typename T>
[[nodiscard]] bool guards_equal(const std::vector<T>& values,
                                const std::size_t guard,
                                const T sentinel,
                                const std::string& label) {
  if (values.size() < 2U * guard ||
      !std::all_of(values.begin(), values.begin() + guard,
                   [sentinel](const T value) { return value == sentinel; }) ||
      !std::all_of(values.end() - guard, values.end(),
                   [sentinel](const T value) { return value == sentinel; })) {
    std::cerr << label << " guard changed\n";
    return false;
  }
  return true;
}

struct HostOutputs final {
  std::vector<std::uint16_t> baseline_primary;
  std::vector<std::uint16_t> baseline_secondary;
  std::vector<std::uint16_t> candidate_primary;
  std::vector<std::uint16_t> candidate_secondary;
  std::vector<std::uint8_t> baseline_packed;
  std::vector<std::uint8_t> candidate_packed;
  std::vector<std::uint16_t> baseline_scales;
  std::vector<std::uint16_t> candidate_scales;
  std::vector<float> partials;
};

[[nodiscard]] bool download_outputs(const Capacities& cap,
                                    const DeviceOutputs& device,
                                    HostOutputs* const host) {
  return host != nullptr &&
         copy_device_vector(device.baseline_primary,
                            cap.primary_elements + 2U * kWordGuard,
                            &host->baseline_primary,
                            "copy baseline primary") &&
         copy_device_vector(device.baseline_secondary,
                            cap.secondary_elements + 2U * kWordGuard,
                            &host->baseline_secondary,
                            "copy baseline secondary") &&
         copy_device_vector(device.candidate_primary,
                            cap.primary_elements + 2U * kWordGuard,
                            &host->candidate_primary,
                            "copy candidate primary") &&
         copy_device_vector(device.candidate_secondary,
                            cap.secondary_elements + 2U * kWordGuard,
                            &host->candidate_secondary,
                            "copy candidate secondary") &&
         copy_device_vector(device.baseline_packed,
                            cap.packed_bytes + 2U * kByteGuard,
                            &host->baseline_packed,
                            "copy baseline packed") &&
         copy_device_vector(device.candidate_packed,
                            cap.packed_bytes + 2U * kByteGuard,
                            &host->candidate_packed,
                            "copy candidate packed") &&
         copy_device_vector(device.baseline_scales,
                            cap.scale_elements + 2U * kWordGuard,
                            &host->baseline_scales,
                            "copy baseline scales") &&
         copy_device_vector(device.candidate_scales,
                            cap.scale_elements + 2U * kWordGuard,
                            &host->candidate_scales,
                            "copy candidate scales") &&
         copy_device_vector(device.partials,
                            cap.partial_elements + 2U * kFloatGuard,
                            &host->partials, "copy tile maxima");
}

[[nodiscard]] bool verify_zero_row_and_padding(
    const Capacities& cap, const HostOutputs& host) {
  const std::uint16_t* const primary =
      host.candidate_primary.data() + kWordGuard;
  const std::uint16_t* const secondary =
      host.candidate_secondary.data() + kWordGuard;
  const std::uint8_t* const packed =
      host.candidate_packed.data() + kByteGuard;
  const std::uint16_t* const scales =
      host.candidate_scales.data() + kWordGuard;
  const float* const partials = host.partials.data() + kFloatGuard;
  const std::uint16_t one = encode_bf16(1.0F);
  bool ok = true;

  for (std::size_t k = 0U; k < kPrimarySize; ++k) {
    if (primary[k] != 0U) {
      std::cerr << "zero row primary product is nonzero at K=" << k << '\n';
      ok = false;
      break;
    }
  }
  for (std::size_t k = 0U; k < kSecondarySize; ++k) {
    if (secondary[k] != 0U) {
      std::cerr << "zero row secondary product is nonzero at K=" << k << '\n';
      ok = false;
      break;
    }
  }
  for (std::size_t k = kSecondarySize; k < kSecondaryStride; ++k) {
    if (secondary[k] != kProductSentinel) {
      std::cerr << "secondary row-stride padding changed at K=" << k << '\n';
      ok = false;
      break;
    }
  }
  for (std::size_t tile = 0U; tile < kPartialTiles; ++tile) {
    if (partials[tile] != 0.0F) {
      std::cerr << "zero row tile maximum is nonzero at tile=" << tile
                << '\n';
      ok = false;
      break;
    }
  }

  const std::size_t physical_groups = kIntermediateSize / 64U;
  const auto row_is_zero = [&](const std::size_t row) {
    for (std::size_t group = 0U; group < physical_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        if (packed[kernels::sm87_a4w4_consumer_packed_offset(
                row, group, byte, physical_groups)] != 0U) {
          return false;
        }
      }
    }
    return scales[kernels::sm87_a4w4_factorized_lane_scale_offset(
                       row, 0U, 1U)] == one;
  };
  if (!row_is_zero(0U)) {
    std::cerr << "zero row did not publish zero codes/BF16 one\n";
    ok = false;
  }
  for (std::size_t row = kLogicalTokens; row < kDownLaunchTokens; ++row) {
    if (!row_is_zero(row)) {
      std::cerr << "Down padding row is not zero/one: row=" << row << '\n';
      ok = false;
      break;
    }
  }
  for (std::size_t row = kLogicalTokens; row < kGateLaunchTokens; ++row) {
    for (std::size_t tile = 0U; tile < kPartialTiles; ++tile) {
      if (partials[row * kPartialTiles + tile] != 0.0F) {
        std::cerr << "Gate padding maximum is nonzero: row=" << row
                  << " tile=" << tile << '\n';
        ok = false;
        row = kGateLaunchTokens;
        break;
      }
    }
  }
  (void)cap;
  return ok;
}

[[nodiscard]] bool verify_plane_boundary(const HostOutputs& host) {
  constexpr std::size_t kProbeRow = 1U;
  const std::uint16_t* const primary =
      host.candidate_primary.data() + kWordGuard;
  const std::uint16_t* const secondary =
      host.candidate_secondary.data() + kWordGuard;
  const std::uint8_t* const packed =
      host.candidate_packed.data() + kByteGuard;
  const std::uint16_t primary_last =
      primary[kProbeRow * kPrimaryStride + kPrimarySize - 1U];
  const std::uint16_t secondary_first =
      secondary[kProbeRow * kSecondaryStride];
  if (primary_last == 0U || secondary_first == 0U ||
      primary_last == secondary_first) {
    std::cerr << "K=12288 product boundary probes are not discriminating: "
              << "primary_last=0x" << std::hex << primary_last
              << " secondary_first=0x" << secondary_first << std::dec << '\n';
    return false;
  }

  const std::size_t groups = kIntermediateSize / 64U;
  const std::uint8_t primary_byte =
      packed[kernels::sm87_a4w4_consumer_packed_offset(
          kProbeRow, (kPrimarySize - 2U) / 64U,
          ((kPrimarySize - 2U) % 64U) / 2U, groups)];
  const std::uint8_t secondary_byte =
      packed[kernels::sm87_a4w4_consumer_packed_offset(
          kProbeRow, kPrimarySize / 64U, 0U, groups)];
  if (primary_byte == secondary_byte) {
    std::cerr << "K=12288 packed boundary probes are not discriminating: byte="
              << static_cast<unsigned int>(primary_byte) << '\n';
    return false;
  }
  std::cout << "K=12288 boundary PASS: primary_bf16=0x" << std::hex
            << primary_last << " secondary_bf16=0x" << secondary_first
            << " primary_code_byte=0x"
            << static_cast<unsigned int>(primary_byte)
            << " secondary_code_byte=0x"
            << static_cast<unsigned int>(secondary_byte) << std::dec << '\n';
  return true;
}

[[nodiscard]] bool verify_outputs(const Capacities& cap,
                                  const HostOutputs& host) {
  bool ok =
      guards_equal(host.baseline_primary, kWordGuard, kProductSentinel,
                   "baseline primary") &&
      guards_equal(host.baseline_secondary, kWordGuard, kProductSentinel,
                   "baseline secondary") &&
      guards_equal(host.candidate_primary, kWordGuard, kProductSentinel,
                   "candidate primary") &&
      guards_equal(host.candidate_secondary, kWordGuard, kProductSentinel,
                   "candidate secondary") &&
      guards_equal(host.baseline_packed, kByteGuard, kPackedSentinel,
                   "baseline packed") &&
      guards_equal(host.candidate_packed, kByteGuard, kPackedSentinel,
                   "candidate packed") &&
      guards_equal(host.baseline_scales, kWordGuard, kScaleSentinel,
                   "baseline scales") &&
      guards_equal(host.candidate_scales, kWordGuard, kScaleSentinel,
                   "candidate scales") &&
      guards_equal(host.partials, kFloatGuard, kPartialSentinel,
                   "tile maxima");
  ok = compare_payload(
           "Gate primary BF16", host.baseline_primary.data() + kWordGuard,
           host.candidate_primary.data() + kWordGuard,
           cap.primary_elements) &&
       ok;
  ok = compare_payload(
           "Gate secondary BF16",
           host.baseline_secondary.data() + kWordGuard,
           host.candidate_secondary.data() + kWordGuard,
           cap.secondary_elements) &&
       ok;
  ok = compare_payload(
           "Down packed A4", host.baseline_packed.data() + kByteGuard,
           host.candidate_packed.data() + kByteGuard, cap.packed_bytes) &&
       ok;
  ok = compare_payload(
           "Down BF16 scales", host.baseline_scales.data() + kWordGuard,
           host.candidate_scales.data() + kWordGuard,
           cap.scale_elements) &&
       ok;
  ok = verify_zero_row_and_padding(cap, host) && ok;
  ok = verify_plane_boundary(host) && ok;
  return ok;
}

}  // namespace

int main() {
  const auto plan = kernels::sm87_a4w4_gateup_r1_product_finalize_plan(
      kLogicalTokens, kDownLaunchTokens);
  if (!plan || plan.gate_launch_token_count != kGateLaunchTokens ||
      plan.partial_capacity_elements != kGateLaunchTokens * kPartialTiles ||
      plan.launch_ctas != kDownLaunchTokens) {
    std::cerr << "P1853/P1920/P2048 plan contract failed\n";
    return 1;
  }
  const int device_status = target_device_status();
  if (device_status != 0) {
    return device_status;
  }
  if (!query_resource_contract()) {
    return 1;
  }

  std::cout << "building deterministic real-shape synthetic payload\n";
  const HostInputs host_inputs = make_inputs();
  DeviceInputs device_inputs;
  if (!upload_inputs(host_inputs, &device_inputs)) {
    return 1;
  }
  const Capacities cap = capacities();
  DeviceOutputs device_outputs;
  if (!allocate_outputs(cap, &device_outputs)) {
    return 1;
  }
  Stream stream;
  if (!stream.create()) {
    std::cerr << "non-default stream creation failed\n";
    return 1;
  }
  if (!launch_paths(device_inputs, cap, &device_outputs, stream.get()) ||
      !cuda_ok(cudaStreamSynchronize(stream.get()),
               "non-default stream synchronize")) {
    return 1;
  }

  HostOutputs host_outputs;
  if (!download_outputs(cap, device_outputs, &host_outputs) ||
      !verify_outputs(cap, host_outputs)) {
    return 1;
  }
  std::cout << "PASS: P1853 Gate tile-max + P2048 product finalizer are "
               "byte-for-byte identical to baseline Gate + incumbent split "
               "quantizer, including zero/padding rows and the K=12288 plane "
               "boundary on a non-default stream\n";
  return 0;
}
