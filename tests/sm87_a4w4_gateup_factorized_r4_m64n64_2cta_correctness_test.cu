#include "q3x/kernels/sm87_a4w4_gateup_factorized_r4_m64n64_2cta.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

constexpr std::size_t kLogicalM = 1'853U;
constexpr std::size_t kLaunchM = 1'920U;
constexpr std::size_t kN = 17'408U;
constexpr std::size_t kK = 5'120U;
constexpr std::size_t kGroups = kK / 64U;
constexpr std::size_t kGroupsPerLane = kGroups / 4U;
constexpr std::uint16_t kGuard = 0x5555U;

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t elements) noexcept {
    return elements != 0U &&
           cudaMalloc(reinterpret_cast<void**>(&pointer_),
                      elements * sizeof(T)) == cudaSuccess;
  }
  [[nodiscard]] T* get() const noexcept { return pointer_; }

 private:
  T* pointer_{};
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

// 1: target, 0: clean unsupported-device skip, -1: CUDA failure.
[[nodiscard]] int target_status() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && count == 0)) {
    (void)cudaGetLastError();
    std::cout << "SKIP: CUDA target unavailable\n";
    return 0;
  }
  if (!cuda_ok(status, "cudaGetDeviceCount")) {
    return -1;
  }
  int device = -1;
  if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice")) {
    return -1;
  }
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return -1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the 16-SM SM87 target\n";
    return 0;
  }
  return 1;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] int a_even(const std::size_t group) noexcept {
  const int magnitude = static_cast<int>(1U + group % 7U);
  return (group & 1U) == 0U ? magnitude : -magnitude;
}

[[nodiscard]] int a_odd(const std::size_t group) noexcept {
  const int magnitude = static_cast<int>(1U + (3U * group + 1U) % 7U);
  return (group % 3U) == 0U ? -magnitude : magnitude;
}

[[nodiscard]] int up_even(const std::size_t n,
                          const std::size_t group) noexcept {
  return (n & 2U) == 0U ? a_even(group) : -a_even(group);
}

[[nodiscard]] int up_odd(const std::size_t n,
                         const std::size_t group) noexcept {
  return (n & 1U) == 0U ? a_odd(group) : -a_odd(group);
}

[[nodiscard]] float row_factor(const std::size_t m) noexcept {
  return static_cast<float>(1U << ((m / 64U + m % 3U) % 3U));
}

[[nodiscard]] float column_factor(const std::size_t n) noexcept {
  return static_cast<float>(1U << ((n / 64U + n % 5U) % 3U));
}

[[nodiscard]] std::uint16_t expected_value(const std::size_t m,
                                           const std::size_t n) noexcept {
  const std::size_t row_class = (m / 64U + m % 3U) % 3U;
  const std::size_t column_class = (n / 64U + n % 5U) % 3U;
  const std::size_t code_class = n % 4U;
  static std::uint16_t cache[3][3][4]{};
  static bool initialized[3][3][4]{};
  if (initialized[row_class][column_class][code_class]) {
    return cache[row_class][column_class][code_class];
  }
  constexpr std::array<float, 4> kAScale = {0.125F, 0.25F, 0.5F, 1.0F};
  constexpr std::array<float, 4> kGateScale = {1.0F, 2.0F, 4.0F, 8.0F};
  constexpr std::array<float, 4> kUpScale = {8.0F, 2.0F, 0.5F, 0.125F};
  float gate = 0.0F;
  float up = 0.0F;
  for (std::size_t lane = 0U; lane < 4U; ++lane) {
    std::int32_t gate_partial = 0;
    std::int32_t up_partial = 0;
    for (std::size_t in_lane = 0U; in_lane < kGroupsPerLane;
         ++in_lane) {
      const std::size_t group = lane * kGroupsPerLane + in_lane;
      gate_partial += 32 *
                      (a_even(group) * a_even(group) +
                       a_odd(group) * a_odd(group));
      up_partial += 32 *
                    (a_even(group) * up_even(n, group) +
                     a_odd(group) * up_odd(n, group));
    }
    const volatile float gate_scale =
        (kAScale[lane] * row_factor(m)) *
        (kGateScale[lane] * column_factor(n));
    const volatile float up_scale =
        (kAScale[lane] * row_factor(m)) *
        (kUpScale[lane] * column_factor(n));
    const volatile float gate_term =
        static_cast<float>(gate_partial) * gate_scale;
    const volatile float up_term = static_cast<float>(up_partial) * up_scale;
    gate = gate + gate_term;
    up = up + up_term;
  }
  // Every constructed gate is far above expf underflow, so the CUDA
  // epilogue's SiLU is exactly gate before the final IEEE float multiply.
  const volatile float product = gate * up;
  const std::uint16_t result = encode_bf16(product);
  cache[row_class][column_class][code_class] = result;
  initialized[row_class][column_class][code_class] = true;
  return result;
}

void fill_codes(std::vector<std::uint8_t>& packed_a,
                std::vector<std::uint8_t>& packed_gate,
                std::vector<std::uint8_t>& packed_up) {
  for (std::size_t m = 0U; m < kLaunchM; ++m) {
    for (std::size_t group = 0U; group < kGroups; ++group) {
      const std::uint8_t code = kernels::sm87_a4w4_pack_signed_pair(
          a_even(group), a_odd(group));
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        packed_a[kernels::sm87_a4w4_consumer_packed_offset(
            m, group, byte, kGroups)] = code;
      }
    }
  }
  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t group = 0U; group < kGroups; ++group) {
      const std::uint8_t gate_code = kernels::sm87_a4w4_pack_signed_pair(
          a_even(group), a_odd(group));
      const std::uint8_t up_code = kernels::sm87_a4w4_pack_signed_pair(
          up_even(n, group), up_odd(n, group));
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t offset =
            kernels::sm87_a4w4_consumer_packed_offset(
                n, group, byte, kGroups);
        packed_gate[offset] = gate_code;
        packed_up[offset] = up_code;
      }
    }
  }
}

void fill_scales(std::vector<std::uint16_t>& a_scales,
                 std::vector<std::uint16_t>& gate_scales,
                 std::vector<std::uint16_t>& up_scales) {
  constexpr std::array<float, 4> kAScale = {0.125F, 0.25F, 0.5F, 1.0F};
  constexpr std::array<float, 4> kGateScale = {1.0F, 2.0F, 4.0F, 8.0F};
  constexpr std::array<float, 4> kUpScale = {8.0F, 2.0F, 0.5F, 0.125F};
  for (std::size_t m = 0U; m < kLaunchM; ++m) {
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      a_scales[kernels::
          sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(m, lane)] =
          encode_bf16(kAScale[lane] * row_factor(m));
    }
  }
  for (std::size_t n = 0U; n < kN; ++n) {
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      const std::size_t offset = kernels::
          sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(n, lane);
      gate_scales[offset] =
          encode_bf16(kGateScale[lane] * column_factor(n));
      up_scales[offset] =
          encode_bf16(kUpScale[lane] * column_factor(n));
    }
  }
}

[[nodiscard]] bool reject(const int status, const char* const label) {
  if (status == static_cast<int>(cudaErrorInvalidValue)) {
    return true;
  }
  std::cerr << label << " did not fail closed: " << status << '\n';
  return false;
}

[[nodiscard]] bool run() {
  constexpr std::size_t a_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kLaunchM, kK);
  constexpr std::size_t b_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(kN, kK);
  constexpr std::size_t a_scale_count = kernels::
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_capacity_elements(
          kLaunchM);
  constexpr std::size_t b_scale_count = kernels::
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_capacity_elements(kN);
  constexpr std::size_t primary_count =
      kLaunchM * kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPrimaryStride;
  constexpr std::size_t secondary_count =
      kLaunchM * kernels::kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSecondaryStride;

  kernels::Sm87A4W4GateUpFactorizedR4M64N64TwoCtaResources resources{};
  if (!cuda_ok(static_cast<cudaError_t>(kernels::
                   query_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_resources_cuda(
                       &resources)),
               "Gate+Up R4 resource query") ||
      resources.local_bytes != 0U || resources.active_blocks_per_sm < 2 ||
      resources.resident_blocks < 32) {
    std::cerr << "Gate+Up R4 resource hard gate failed\n";
    return false;
  }

  std::vector<std::uint8_t> packed_a(a_bytes);
  std::vector<std::uint8_t> packed_gate(b_bytes);
  std::vector<std::uint8_t> packed_up(b_bytes);
  std::vector<std::uint16_t> a_scales(a_scale_count);
  std::vector<std::uint16_t> gate_scales(b_scale_count);
  std::vector<std::uint16_t> up_scales(b_scale_count);
  fill_codes(packed_a, packed_gate, packed_up);
  fill_scales(a_scales, gate_scales, up_scales);

  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint8_t> device_gate;
  DeviceBuffer<std::uint8_t> device_up;
  DeviceBuffer<std::uint16_t> device_a_scales;
  DeviceBuffer<std::uint16_t> device_gate_scales;
  DeviceBuffer<std::uint16_t> device_up_scales;
  DeviceBuffer<std::uint16_t> device_primary;
  DeviceBuffer<std::uint16_t> device_secondary;
  if (!device_a.allocate(a_bytes) || !device_gate.allocate(b_bytes) ||
      !device_up.allocate(b_bytes) ||
      !device_a_scales.allocate(a_scale_count) ||
      !device_gate_scales.allocate(b_scale_count) ||
      !device_up_scales.allocate(b_scale_count) ||
      !device_primary.allocate(primary_count) ||
      !device_secondary.allocate(secondary_count)) {
    std::cerr << "Gate+Up R4 release-shape allocation failed\n";
    return false;
  }

  cudaStream_t stream{};
  if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create nondefault stream")) {
    return false;
  }
  const auto destroy_stream = [&]() { (void)cudaStreamDestroy(stream); };
  const auto copy = [&](void* destination, const void* source,
                        const std::size_t bytes, const char* label) {
    return cuda_ok(cudaMemcpyAsync(destination, source, bytes,
                                   cudaMemcpyHostToDevice, stream),
                   label);
  };
  if (!copy(device_a.get(), packed_a.data(), a_bytes, "copy A") ||
      !copy(device_gate.get(), packed_gate.data(), b_bytes, "copy Gate") ||
      !copy(device_up.get(), packed_up.data(), b_bytes, "copy Up") ||
      !copy(device_a_scales.get(), a_scales.data(),
            a_scale_count * sizeof(std::uint16_t), "copy A scales") ||
      !copy(device_gate_scales.get(), gate_scales.data(),
            b_scale_count * sizeof(std::uint16_t), "copy Gate scales") ||
      !copy(device_up_scales.get(), up_scales.data(),
            b_scale_count * sizeof(std::uint16_t), "copy Up scales") ||
      !cuda_ok(cudaMemsetAsync(device_primary.get(), 0x55,
                              primary_count * sizeof(std::uint16_t), stream),
               "initialize primary") ||
      !cuda_ok(cudaMemsetAsync(device_secondary.get(), 0x55,
                              secondary_count * sizeof(std::uint16_t), stream),
               "initialize secondary") ||
      !cuda_ok(cudaStreamSynchronize(stream), "prepare release-shape data")) {
    destroy_stream();
    return false;
  }

  const auto launch = [&](const std::uint8_t* a, std::size_t a_capacity,
                          const std::uint16_t* a_scale,
                          std::size_t a_scale_capacity,
                          const std::uint8_t* gate,
                          std::size_t gate_capacity,
                          std::uint16_t* primary,
                          std::size_t primary_stride,
                          std::size_t primary_capacity,
                          std::uint16_t* secondary,
                          std::size_t secondary_stride,
                          std::size_t secondary_capacity,
                          std::size_t logical = kLogicalM,
                          std::size_t launch_m = kLaunchM,
                          std::size_t n = kN,
                          std::size_t k = kK) {
    return kernels::
        launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
            a, a_capacity, a_scale, a_scale_capacity, gate, gate_capacity,
            device_gate_scales.get(), b_scale_count, device_up.get(), b_bytes,
            device_up_scales.get(), b_scale_count, logical, launch_m, n, k,
            primary, primary_stride, primary_capacity, secondary,
            secondary_stride, secondary_capacity, stream);
  };

  const bool negatives =
      reject(launch(nullptr, a_bytes, device_a_scales.get(), a_scale_count,
                    device_gate.get(), b_bytes, device_primary.get(),
                    12'288U, primary_count, device_secondary.get(), 6'144U,
                    secondary_count),
             "null A") &&
      reject(launch(device_a.get(), a_bytes - 1U, device_a_scales.get(),
                    a_scale_count, device_gate.get(), b_bytes,
                    device_primary.get(), 12'288U, primary_count,
                    device_secondary.get(), 6'144U, secondary_count),
             "short A") &&
      reject(launch(device_a.get(), a_bytes,
                    reinterpret_cast<const std::uint16_t*>(
                        reinterpret_cast<const std::uint8_t*>(
                            device_a_scales.get()) + 2U),
                    a_scale_count, device_gate.get(), b_bytes,
                    device_primary.get(), 12'288U, primary_count,
                    device_secondary.get(), 6'144U, secondary_count),
             "misaligned A scale") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_gate.get(), b_bytes,
                    device_primary.get(), 12'287U, primary_count,
                    device_secondary.get(), 6'144U, secondary_count),
             "wrong primary stride") &&
      reject(launch(reinterpret_cast<const std::uint8_t*>(
                        device_primary.get()),
                    a_bytes, device_a_scales.get(), a_scale_count,
                    device_gate.get(), b_bytes, device_primary.get(),
                    12'288U, primary_count, device_secondary.get(), 6'144U,
                    secondary_count),
             "input/output alias") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_gate.get(), b_bytes,
                    device_primary.get(), 12'288U, primary_count,
                    device_secondary.get(), 6'144U, secondary_count, 1'792U),
             "logical token lower bound") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_gate.get(), b_bytes,
                    device_primary.get(), 12'288U, primary_count,
                    device_secondary.get(), 6'144U, secondary_count,
                    kLogicalM, 2'048U),
             "wrong launch token count") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_gate.get(), b_bytes,
                    device_primary.get(), 12'288U, primary_count,
                    device_secondary.get(), 6'144U, secondary_count,
                    kLogicalM, kLaunchM, kN + 64U),
             "wrong N") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_gate.get(), b_bytes,
                    device_primary.get(), 12'288U, primary_count,
                    device_secondary.get(), 6'144U, secondary_count,
                    kLogicalM, kLaunchM, kN, kK + 64U),
             "wrong K") &&
      reject(launch(device_a.get(), a_bytes, device_a_scales.get(),
                    a_scale_count, device_gate.get(), b_bytes,
                    device_primary.get(), 12'288U, primary_count,
                    device_primary.get(), 6'144U, secondary_count),
             "primary/secondary alias");
  const bool capacity_negatives =
      reject(kernels::
                 launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count - 1U, device_gate.get(), b_bytes,
                     device_gate_scales.get(), b_scale_count,
                     device_up.get(), b_bytes, device_up_scales.get(),
                     b_scale_count, kLogicalM, kLaunchM, kN, kK,
                     device_primary.get(), 12'288U, primary_count,
                     device_secondary.get(), 6'144U, secondary_count,
                     stream),
             "short A scale") &&
      reject(kernels::
                 launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_gate.get(), b_bytes - 1U,
                     device_gate_scales.get(), b_scale_count,
                     device_up.get(), b_bytes, device_up_scales.get(),
                     b_scale_count, kLogicalM, kLaunchM, kN, kK,
                     device_primary.get(), 12'288U, primary_count,
                     device_secondary.get(), 6'144U, secondary_count,
                     stream),
             "short Gate code") &&
      reject(kernels::
                 launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_gate.get(), b_bytes,
                     device_gate_scales.get(), b_scale_count - 1U,
                     device_up.get(), b_bytes, device_up_scales.get(),
                     b_scale_count, kLogicalM, kLaunchM, kN, kK,
                     device_primary.get(), 12'288U, primary_count,
                     device_secondary.get(), 6'144U, secondary_count,
                     stream),
             "short Gate scale") &&
      reject(kernels::
                 launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_gate.get(), b_bytes,
                     device_gate_scales.get(), b_scale_count,
                     device_up.get(), b_bytes - 1U,
                     device_up_scales.get(), b_scale_count, kLogicalM,
                     kLaunchM, kN, kK, device_primary.get(), 12'288U,
                     primary_count, device_secondary.get(), 6'144U,
                     secondary_count, stream),
             "short Up code") &&
      reject(kernels::
                 launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_gate.get(), b_bytes,
                     device_gate_scales.get(), b_scale_count,
                     device_up.get(), b_bytes, device_up_scales.get(),
                     b_scale_count - 1U, kLogicalM, kLaunchM, kN, kK,
                     device_primary.get(), 12'288U, primary_count,
                     device_secondary.get(), 6'144U, secondary_count,
                     stream),
             "short Up scale") &&
      reject(kernels::
                 launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_gate.get(), b_bytes,
                     device_gate_scales.get(), b_scale_count,
                     device_up.get(), b_bytes, device_up_scales.get(),
                     b_scale_count, kLogicalM, kLaunchM, kN, kK,
                     device_primary.get(), 12'288U, primary_count - 1U,
                     device_secondary.get(), 6'144U, secondary_count,
                     stream),
             "short primary output") &&
      reject(kernels::
                 launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_gate.get(), b_bytes,
                     device_gate_scales.get(), b_scale_count,
                     device_up.get(), b_bytes, device_up_scales.get(),
                     b_scale_count, kLogicalM, kLaunchM, kN, kK,
                     device_primary.get(), 12'288U, primary_count,
                     device_secondary.get(), 6'144U,
                     secondary_count - 1U, stream),
             "short secondary output") &&
      reject(kernels::
                 launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
                     device_a.get(), a_bytes, device_a_scales.get(),
                     a_scale_count, device_gate.get(), b_bytes,
                     device_gate_scales.get(), b_scale_count,
                     device_up.get(), b_bytes, device_up_scales.get(),
                     b_scale_count, kLogicalM, kLaunchM, kN, kK,
                     device_primary.get(), 12'288U, primary_count,
                     device_secondary.get(), 5'120U, secondary_count,
                     stream),
             "wrong secondary stride");
  if (!negatives || !capacity_negatives) {
    destroy_stream();
    return false;
  }

  const int launch_status = launch(
      device_a.get(), a_bytes, device_a_scales.get(), a_scale_count,
      device_gate.get(), b_bytes, device_primary.get(), 12'288U,
      primary_count, device_secondary.get(), 6'144U, secondary_count);
  if (!cuda_ok(static_cast<cudaError_t>(launch_status),
               "launch Gate+Up R4 release shape") ||
      !cuda_ok(cudaStreamSynchronize(stream),
               "synchronize Gate+Up R4 nondefault stream")) {
    destroy_stream();
    return false;
  }

  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
  if (!cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin Gate+Up R4 graph capture")) {
    destroy_stream();
    return false;
  }
  const int captured_launch_status = launch(
      device_a.get(), a_bytes, device_a_scales.get(), a_scale_count,
      device_gate.get(), b_bytes, device_primary.get(), 12'288U,
      primary_count, device_secondary.get(), 6'144U, secondary_count);
  if (!cuda_ok(static_cast<cudaError_t>(captured_launch_status),
               "capture Gate+Up R4 release shape") ||
      !cuda_ok(cudaStreamEndCapture(stream, &graph),
               "end Gate+Up R4 graph capture") ||
      !cuda_ok(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0U),
               "instantiate Gate+Up R4 graph") ||
      !cuda_ok(cudaGraphLaunch(graph_exec, stream),
               "launch Gate+Up R4 graph") ||
      !cuda_ok(cudaStreamSynchronize(stream),
               "synchronize Gate+Up R4 graph")) {
    if (graph_exec != nullptr) {
      (void)cudaGraphExecDestroy(graph_exec);
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    destroy_stream();
    return false;
  }
  (void)cudaGraphExecDestroy(graph_exec);
  (void)cudaGraphDestroy(graph);

  std::vector<std::uint16_t> primary(primary_count);
  std::vector<std::uint16_t> secondary(secondary_count);
  if (!cuda_ok(cudaMemcpy(primary.data(), device_primary.get(),
                          primary_count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy primary result") ||
      !cuda_ok(cudaMemcpy(secondary.data(), device_secondary.get(),
                          secondary_count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy secondary result")) {
    destroy_stream();
    return false;
  }
  destroy_stream();

  for (std::size_t m = 0U; m < kLaunchM; ++m) {
    for (std::size_t n = 0U; n < 12'288U; ++n) {
      const std::uint16_t expected =
          m < kLogicalM ? expected_value(m, n) : kGuard;
      const std::uint16_t actual = primary[m * 12'288U + n];
      if (actual != expected) {
        std::cerr << "primary mismatch m=" << m << " n=" << n
                  << " actual=0x" << std::hex << actual << " expected=0x"
                  << expected << std::dec << '\n';
        return false;
      }
    }
    for (std::size_t local_n = 0U; local_n < 6'144U; ++local_n) {
      const std::uint16_t expected =
          m < kLogicalM && local_n < 5'120U
              ? expected_value(m, 12'288U + local_n)
              : kGuard;
      const std::uint16_t actual = secondary[m * 6'144U + local_n];
      if (actual != expected) {
        std::cerr << "secondary mismatch m=" << m << " n=" << local_n
                  << " actual=0x" << std::hex << actual << " expected=0x"
                  << expected << std::dec << '\n';
        return false;
      }
    }
  }

  std::cout << "PASS: Gate+Up R4 M64N64 two-CTA release-shape correctness; "
               "P1853->P1920 tail, N12288 seam, padding, guards, nondefault "
               "stream, graph capture/replay, launch negatives, and resource "
               "gates\n";
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target <= 0) {
    return target == 0 ? 0 : 1;
  }
  return run() ? 0 : 1;
}
