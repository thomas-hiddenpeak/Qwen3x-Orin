#include "q3x/kernels/sm87_a4w4_factorized_lane_quantize.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

inline constexpr std::size_t kWordGuard = 32U;
inline constexpr std::size_t kFloatGuard = 16U;
inline constexpr std::size_t kByteGuard = 64U;
inline constexpr std::uint16_t kInputGuard = 0x7fc1U;
inline constexpr float kAlphaGuard = -91.75F;
inline constexpr std::uint8_t kPackedGuard = 0xa5U;
inline constexpr std::uint16_t kScaleGuard = 0xbeefU;

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

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] int nearest_even_code(const float value,
                                    const float clipped_maximum,
                                    const float stored_scale) noexcept {
  const float clipped =
      std::fmin(std::fmax(value, -clipped_maximum), clipped_maximum);
  const int rounded =
      static_cast<int>(std::nearbyint(clipped / stored_scale));
  return std::max(-7, std::min(7, rounded));
}

struct CaseSpec final {
  std::string name;
  std::size_t logical_tokens{};
  std::size_t launch_tokens{};
  std::size_t input_size{};
  std::size_t lane_count{};
  float clip_ratio{};
  bool split{};
  bool graph_capture{};
};

struct HostPayload final {
  std::vector<std::uint16_t> logical_input;
  std::vector<float> inverse_alpha;
};

[[nodiscard]] HostPayload make_payload(const CaseSpec& spec) {
  HostPayload result{
      std::vector<std::uint16_t>(spec.logical_tokens * spec.input_size),
      std::vector<float>(spec.input_size)};
  const std::size_t lane_input_size = spec.input_size / spec.lane_count;
  for (std::size_t k = 0U; k < spec.input_size; ++k) {
    result.inverse_alpha[k] =
        0.625F + 0.00625F * static_cast<float>((k * 37U + 11U) % 101U);
  }
  // Make the first values in every lane exact powers-of-two-friendly probes
  // for ties-to-even and signed saturation.
  for (std::size_t lane = 0U; lane < spec.lane_count; ++lane) {
    for (std::size_t inner = 0U; inner < 8U; ++inner) {
      result.inverse_alpha[lane * lane_input_size + inner] = 1.0F;
    }
  }

  for (std::size_t row = 0U; row < spec.logical_tokens; ++row) {
    for (std::size_t k = 0U; k < spec.input_size; ++k) {
      float value = 0.0F;
      if (row != 0U) {
        const int centered =
            static_cast<int>((row * 1'009U + k * 313U +
                              (row + 3U) * (k % 29U)) %
                             2'003U) -
            1'001;
        const float row_gain =
            0.77F + 0.013F * static_cast<float>((row + k / 64U) % 17U);
        value = static_cast<float>(centered) * 0.00317F * row_gain;
      }
      result.logical_input[row * spec.input_size + k] = encode_bf16(value);
    }
  }

  if (spec.logical_tokens > 1U) {
    constexpr float probes[8U] = {7.0F, 0.5F, 1.5F, 2.5F,
                                  -0.5F, -1.5F, -2.5F, -7.0F};
    for (std::size_t lane = 0U; lane < spec.lane_count; ++lane) {
      const std::size_t lane_begin = lane * lane_input_size;
      for (std::size_t inner = 0U; inner < lane_input_size; ++inner) {
        result.logical_input[spec.input_size + lane_begin + inner] =
            encode_bf16(0.0F);
      }
      for (std::size_t inner = 0U; inner < 8U; ++inner) {
        result.logical_input[spec.input_size + lane_begin + inner] =
            encode_bf16(probes[inner]);
      }
    }
  }
  return result;
}

struct Oracle final {
  std::vector<std::uint8_t> packed;
  std::vector<std::uint16_t> scales;
  std::size_t zero_lanes{};
  std::size_t saturated_codes{};
};

[[nodiscard]] Oracle make_oracle(const CaseSpec& spec,
                                 const HostPayload& payload) {
  const auto plan = kernels::sm87_a4w4_factorized_lane_quantize_plan(
      spec.logical_tokens, spec.launch_tokens, spec.input_size,
      spec.lane_count);
  Oracle result{std::vector<std::uint8_t>(plan.packed_capacity_bytes,
                                         0xcdU),
                std::vector<std::uint16_t>(plan.scale_capacity_elements,
                                           0xdeadU)};
  const std::size_t lane_input_size = plan.lane_input_size;
  for (std::size_t row = 0U; row < spec.launch_tokens; ++row) {
    for (std::size_t lane = 0U; lane < spec.lane_count; ++lane) {
      const bool valid_row = row < spec.logical_tokens;
      const std::size_t lane_begin = lane * lane_input_size;
      float maximum = 0.0F;
      if (valid_row) {
        for (std::size_t inner = 0U; inner < lane_input_size; ++inner) {
          const std::size_t k = lane_begin + inner;
          const float transformed =
              decode_bf16(payload.logical_input[row * spec.input_size + k]) *
              payload.inverse_alpha[k];
          maximum = std::fmax(maximum, std::fabs(transformed));
        }
      }
      const float clipped_maximum = maximum * spec.clip_ratio;
      std::uint16_t scale_bits = encode_bf16(
          maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      if (maximum == 0.0F) {
        ++result.zero_lanes;
      }
      result.scales[kernels::sm87_a4w4_factorized_lane_scale_offset(
          row, lane, spec.lane_count)] = scale_bits;

      for (std::size_t inner = 0U; inner < lane_input_size; inner += 2U) {
        const std::size_t even_k = lane_begin + inner;
        float even = 0.0F;
        float odd = 0.0F;
        if (valid_row) {
          even = decode_bf16(
                     payload.logical_input[row * spec.input_size + even_k]) *
                 payload.inverse_alpha[even_k];
          odd = decode_bf16(payload.logical_input[
                    row * spec.input_size + even_k + 1U]) *
                payload.inverse_alpha[even_k + 1U];
        }
        const int even_code =
            nearest_even_code(even, clipped_maximum, stored_scale);
        const int odd_code =
            nearest_even_code(odd, clipped_maximum, stored_scale);
        result.saturated_codes +=
            static_cast<std::size_t>(std::abs(even_code) == 7) +
            static_cast<std::size_t>(std::abs(odd_code) == 7);
        result.packed[kernels::sm87_a4w4_consumer_packed_offset(
            row, even_k / 64U, (even_k % 64U) / 2U,
            plan.physical_k64_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even_code, odd_code);
      }
    }
  }
  return result;
}

[[nodiscard]] bool all_equal(const std::uint8_t* const begin,
                             const std::uint8_t* const end,
                             const std::uint8_t value) {
  return std::all_of(begin, end,
                     [value](const std::uint8_t item) {
                       return item == value;
                     });
}

[[nodiscard]] bool all_equal(const std::uint16_t* const begin,
                             const std::uint16_t* const end,
                             const std::uint16_t value) {
  return std::all_of(begin, end,
                     [value](const std::uint16_t item) {
                       return item == value;
                     });
}

[[nodiscard]] bool launch_with_optional_graph(
    const CaseSpec& spec, const std::uint16_t* const primary,
    const std::size_t primary_stride, const std::size_t primary_capacity,
    const std::size_t primary_size, const std::uint16_t* const secondary,
    const std::size_t secondary_stride,
    const std::size_t secondary_capacity, const std::size_t secondary_size,
    const float* const inverse_alpha, std::uint8_t* const packed,
    const std::size_t packed_capacity, std::uint16_t* const scales,
    const std::size_t scale_capacity, cudaStream_t const stream) {
  const auto launch = [&]() {
    return spec.split
               ? kernels::
                     launch_sm87_a4w4_factorized_lane_quantize_bf16_split_cuda(
                         primary, primary_stride, primary_capacity,
                         primary_size, secondary, secondary_stride,
                         secondary_capacity, secondary_size, inverse_alpha,
                         spec.input_size, spec.logical_tokens,
                         spec.launch_tokens, spec.lane_count, spec.clip_ratio,
                         packed, packed_capacity, scales, scale_capacity,
                         stream)
               : kernels::launch_sm87_a4w4_factorized_lane_quantize_bf16_cuda(
                     primary, primary_stride, primary_capacity, inverse_alpha,
                     spec.input_size, spec.logical_tokens, spec.launch_tokens,
                     spec.input_size, spec.lane_count, spec.clip_ratio, packed,
                     packed_capacity, scales, scale_capacity, stream);
  };

  if (!spec.graph_capture) {
    return cuda_ok(static_cast<cudaError_t>(launch()),
                   spec.name + " launch") &&
           cuda_ok(cudaStreamSynchronize(stream), spec.name + " sync");
  }

  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool ok = cuda_ok(cudaStreamBeginCapture(
                        stream, cudaStreamCaptureModeThreadLocal),
                    spec.name + " begin graph capture");
  if (ok) {
    ok = cuda_ok(static_cast<cudaError_t>(launch()),
                 spec.name + " capture launch") &&
         cuda_ok(cudaStreamEndCapture(stream, &graph),
                 spec.name + " end graph capture") &&
         cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr,
                                      0U),
                 spec.name + " instantiate graph") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 spec.name + " graph replay 1") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 spec.name + " graph replay 2") &&
         cuda_ok(cudaStreamSynchronize(stream), spec.name + " graph sync");
  }
  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  return ok;
}

[[nodiscard]] bool run_case(const CaseSpec& spec) {
  const auto plan = kernels::sm87_a4w4_factorized_lane_quantize_plan(
      spec.logical_tokens, spec.launch_tokens, spec.input_size,
      spec.lane_count);
  if (!plan.valid()) {
    std::cerr << spec.name << ": invalid host plan\n";
    return false;
  }
  const HostPayload payload = make_payload(spec);
  const Oracle oracle = make_oracle(spec, payload);
  if (oracle.zero_lanes == 0U || oracle.saturated_codes == 0U) {
    std::cerr << spec.name << ": oracle missed zero/saturation coverage\n";
    return false;
  }

  const std::size_t primary_size =
      spec.split
          ? kernels::kSm87A4W4FactorizedLaneQuantizeDownPrimaryInput
          : spec.input_size;
  const std::size_t secondary_size =
      spec.split
          ? kernels::kSm87A4W4FactorizedLaneQuantizeDownSecondaryInput
          : 0U;
  const std::size_t primary_stride = primary_size + 7U;
  const std::size_t secondary_stride = secondary_size + 11U;
  const std::size_t primary_span =
      (spec.logical_tokens - 1U) * primary_stride + primary_size;
  const std::size_t secondary_span =
      spec.split
          ? (spec.logical_tokens - 1U) * secondary_stride + secondary_size
          : 0U;

  std::vector<std::uint16_t> host_primary(
      2U * kWordGuard + primary_span, kInputGuard);
  std::vector<std::uint16_t> host_secondary(
      spec.split ? 2U * kWordGuard + secondary_span : 1U, kInputGuard);
  std::uint16_t* const primary_payload =
      host_primary.data() + kWordGuard;
  std::uint16_t* const secondary_payload =
      spec.split ? host_secondary.data() + kWordGuard : nullptr;
  for (std::size_t row = 0U; row < spec.logical_tokens; ++row) {
    std::copy_n(payload.logical_input.data() + row * spec.input_size,
                primary_size,
                primary_payload + row * primary_stride);
    if (spec.split) {
      std::copy_n(payload.logical_input.data() + row * spec.input_size +
                      primary_size,
                  secondary_size,
                  secondary_payload + row * secondary_stride);
    }
  }
  const std::vector<std::uint16_t> original_primary = host_primary;
  const std::vector<std::uint16_t> original_secondary = host_secondary;

  std::vector<float> host_alpha(
      2U * kFloatGuard + spec.input_size, kAlphaGuard);
  std::copy(payload.inverse_alpha.begin(), payload.inverse_alpha.end(),
            host_alpha.begin() + kFloatGuard);
  const std::vector<float> original_alpha = host_alpha;
  std::vector<std::uint8_t> host_packed_initial(
      2U * kByteGuard + plan.packed_capacity_bytes, kPackedGuard);
  std::vector<std::uint16_t> host_scales_initial(
      2U * kWordGuard + plan.scale_capacity_elements, kScaleGuard);

  DeviceBuffer<std::uint16_t> device_primary;
  DeviceBuffer<std::uint16_t> device_secondary;
  DeviceBuffer<float> device_alpha;
  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint16_t> device_scales;
  if (!device_primary.allocate(host_primary.size()) ||
      (spec.split && !device_secondary.allocate(host_secondary.size())) ||
      !device_alpha.allocate(host_alpha.size()) ||
      !device_packed.allocate(host_packed_initial.size()) ||
      !device_scales.allocate(host_scales_initial.size())) {
    std::cerr << spec.name << ": device allocation failed\n";
    return false;
  }
  if (!cuda_ok(cudaMemcpy(device_primary.get(), host_primary.data(),
                          host_primary.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               spec.name + " copy primary") ||
      (spec.split &&
       !cuda_ok(cudaMemcpy(device_secondary.get(), host_secondary.data(),
                           host_secondary.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice),
                spec.name + " copy secondary")) ||
      !cuda_ok(cudaMemcpy(device_alpha.get(), host_alpha.data(),
                          host_alpha.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               spec.name + " copy inverse alpha") ||
      !cuda_ok(cudaMemcpy(device_packed.get(), host_packed_initial.data(),
                          host_packed_initial.size(), cudaMemcpyHostToDevice),
               spec.name + " initialize packed") ||
      !cuda_ok(cudaMemcpy(device_scales.get(), host_scales_initial.data(),
                          host_scales_initial.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               spec.name + " initialize scales")) {
    return false;
  }

  Stream stream;
  if (!stream.create()) {
    std::cerr << spec.name << ": nondefault stream creation failed\n";
    return false;
  }
  std::uint16_t* const device_primary_payload =
      device_primary.get() + kWordGuard;
  std::uint16_t* const device_secondary_payload =
      spec.split ? device_secondary.get() + kWordGuard : nullptr;
  float* const device_alpha_payload = device_alpha.get() + kFloatGuard;
  std::uint8_t* const device_packed_payload =
      device_packed.get() + kByteGuard;
  std::uint16_t* const device_scale_payload =
      device_scales.get() + kWordGuard;
  if (!launch_with_optional_graph(
          spec, device_primary_payload, primary_stride, primary_span,
          primary_size, device_secondary_payload, secondary_stride,
          secondary_span, secondary_size, device_alpha_payload,
          device_packed_payload, plan.packed_capacity_bytes,
          device_scale_payload, plan.scale_capacity_elements,
          stream.get())) {
    return false;
  }

  std::vector<std::uint16_t> actual_primary(host_primary.size());
  std::vector<std::uint16_t> actual_secondary(host_secondary.size());
  std::vector<float> actual_alpha(host_alpha.size());
  std::vector<std::uint8_t> actual_packed(host_packed_initial.size());
  std::vector<std::uint16_t> actual_scales(host_scales_initial.size());
  if (!cuda_ok(cudaMemcpy(actual_primary.data(), device_primary.get(),
                          actual_primary.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               spec.name + " copy primary back") ||
      (spec.split &&
       !cuda_ok(cudaMemcpy(actual_secondary.data(), device_secondary.get(),
                           actual_secondary.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost),
                spec.name + " copy secondary back")) ||
      !cuda_ok(cudaMemcpy(actual_alpha.data(), device_alpha.get(),
                          actual_alpha.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               spec.name + " copy alpha back") ||
      !cuda_ok(cudaMemcpy(actual_packed.data(), device_packed.get(),
                          actual_packed.size(), cudaMemcpyDeviceToHost),
               spec.name + " copy packed back") ||
      !cuda_ok(cudaMemcpy(actual_scales.data(), device_scales.get(),
                          actual_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               spec.name + " copy scales back")) {
    return false;
  }
  if (actual_primary != original_primary ||
      (spec.split && actual_secondary != original_secondary) ||
      actual_alpha != original_alpha) {
    std::cerr << spec.name << ": an input or inverse-alpha guard changed\n";
    return false;
  }
  if (!all_equal(actual_packed.data(),
                 actual_packed.data() + kByteGuard, kPackedGuard) ||
      !all_equal(actual_packed.data() + kByteGuard +
                     plan.packed_capacity_bytes,
                 actual_packed.data() + actual_packed.size(),
                 kPackedGuard) ||
      !all_equal(actual_scales.data(),
                 actual_scales.data() + kWordGuard, kScaleGuard) ||
      !all_equal(actual_scales.data() + kWordGuard +
                     plan.scale_capacity_elements,
                 actual_scales.data() + actual_scales.size(),
                 kScaleGuard)) {
    std::cerr << spec.name << ": output canary changed\n";
    return false;
  }

  const std::uint8_t* const packed = actual_packed.data() + kByteGuard;
  const std::uint16_t* const scales =
      actual_scales.data() + kWordGuard;
  std::size_t packed_mismatches = 0U;
  std::size_t scale_mismatches = 0U;
  for (std::size_t index = 0U; index < oracle.packed.size(); ++index) {
    if (packed[index] != oracle.packed[index]) {
      ++packed_mismatches;
      if (packed_mismatches <= 6U) {
        std::cerr << spec.name << " packed mismatch index=" << index
                  << " expected="
                  << static_cast<unsigned int>(oracle.packed[index])
                  << " actual=" << static_cast<unsigned int>(packed[index])
                  << '\n';
      }
    }
  }
  for (std::size_t index = 0U; index < oracle.scales.size(); ++index) {
    if (scales[index] != oracle.scales[index]) {
      ++scale_mismatches;
      if (scale_mismatches <= 6U) {
        std::cerr << spec.name << " scale mismatch index=" << index
                  << " expected=0x" << std::hex << oracle.scales[index]
                  << " actual=0x" << scales[index] << std::dec << '\n';
      }
    }
  }
  if (packed_mismatches != 0U || scale_mismatches != 0U) {
    std::cerr << spec.name << " FAIL packed_mismatches="
              << packed_mismatches
              << " scale_mismatches=" << scale_mismatches << '\n';
    return false;
  }
  std::cout << spec.name << " PASS logical_M=" << spec.logical_tokens
            << " launch_M=" << spec.launch_tokens << " K="
            << spec.input_size << " R=" << spec.lane_count
            << " split=" << spec.split
            << " graph=" << spec.graph_capture << '\n';
  return true;
}

struct ContiguousInvocation final {
  const std::uint16_t* input{};
  std::size_t stride{};
  std::size_t input_capacity{};
  const float* alpha{};
  std::size_t alpha_capacity{};
  std::size_t logical{};
  std::size_t launch{};
  std::size_t input_size{};
  std::size_t lanes{};
  float clip{};
  std::uint8_t* packed{};
  std::size_t packed_capacity{};
  std::uint16_t* scales{};
  std::size_t scale_capacity{};
  void* stream{};

  [[nodiscard]] int operator()() const noexcept {
    return kernels::launch_sm87_a4w4_factorized_lane_quantize_bf16_cuda(
        input, stride, input_capacity, alpha, alpha_capacity, logical, launch,
        input_size, lanes, clip, packed, packed_capacity, scales,
        scale_capacity, stream);
  }
};

struct SplitInvocation final {
  const std::uint16_t* primary{};
  std::size_t primary_stride{};
  std::size_t primary_capacity{};
  std::size_t primary_size{};
  const std::uint16_t* secondary{};
  std::size_t secondary_stride{};
  std::size_t secondary_capacity{};
  std::size_t secondary_size{};
  const float* alpha{};
  std::size_t alpha_capacity{};
  std::size_t logical{};
  std::size_t launch{};
  std::size_t lanes{};
  float clip{};
  std::uint8_t* packed{};
  std::size_t packed_capacity{};
  std::uint16_t* scales{};
  std::size_t scale_capacity{};
  void* stream{};

  [[nodiscard]] int operator()() const noexcept {
    return kernels::
        launch_sm87_a4w4_factorized_lane_quantize_bf16_split_cuda(
            primary, primary_stride, primary_capacity, primary_size,
            secondary, secondary_stride, secondary_capacity, secondary_size,
            alpha, alpha_capacity, logical, launch, lanes, clip, packed,
            packed_capacity, scales, scale_capacity, stream);
  }
};

[[nodiscard]] bool expect_invalid(const int status,
                                  const std::string& label) {
  if (status == static_cast<int>(cudaErrorInvalidValue)) {
    return true;
  }
  std::cerr << label << " did not fail closed, status=" << status << '\n';
  return false;
}

template <typename Invocation, typename Mutation>
[[nodiscard]] bool invalid_after_mutation(const Invocation& baseline,
                                          Mutation mutation,
                                          const std::string& label) {
  Invocation candidate = baseline;
  mutation(candidate);
  return expect_invalid(candidate(), label);
}

[[nodiscard]] bool run_invalid_tests() {
  constexpr std::size_t kLogical = 1U;
  constexpr std::size_t kLaunch = 64U;
  constexpr std::size_t kK =
      kernels::kSm87A4W4FactorizedLaneQuantizeGateInput;
  const auto plan = kernels::sm87_a4w4_factorized_lane_quantize_plan(
      kLogical, kLaunch, kK, 1U);
  const auto maximum_plan =
      kernels::sm87_a4w4_factorized_lane_quantize_plan(
          kLogical, kLaunch,
          kernels::kSm87A4W4FactorizedLaneQuantizeDownInput, 4U);
  DeviceBuffer<std::uint16_t> input;
  DeviceBuffer<float> alpha;
  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint16_t> scales;
  DeviceBuffer<std::uint16_t> primary;
  DeviceBuffer<std::uint16_t> secondary;
  if (!input.allocate(kK + 8U) ||
      !alpha.allocate(
          kernels::kSm87A4W4FactorizedLaneQuantizeDownInput + 8U) ||
      !packed.allocate(maximum_plan.packed_capacity_bytes + 16U) ||
      !scales.allocate(maximum_plan.scale_capacity_elements + 8U) ||
      !primary.allocate(
          kernels::kSm87A4W4FactorizedLaneQuantizeDownPrimaryInput + 8U) ||
      !secondary.allocate(
          kernels::kSm87A4W4FactorizedLaneQuantizeDownSecondaryInput + 8U)) {
    std::cerr << "invalid-test allocation failed\n";
    return false;
  }
  Stream stream;
  if (!stream.create()) {
    return false;
  }
  const ContiguousInvocation contiguous{
      input.get(), kK, kK, alpha.get(), kK, kLogical, kLaunch, kK, 1U,
      1.0F, packed.get(), plan.packed_capacity_bytes, scales.get(),
      plan.scale_capacity_elements, stream.get()};
  bool ok = true;
#define Q3X_INVALID_CONTIGUOUS(LABEL, BODY)                              \
  ok = invalid_after_mutation(                                           \
           contiguous, [](ContiguousInvocation& value) { BODY; }, LABEL) \
           &&                                                            \
       ok;
  Q3X_INVALID_CONTIGUOUS("null input", value.input = nullptr)
  Q3X_INVALID_CONTIGUOUS("null alpha", value.alpha = nullptr)
  Q3X_INVALID_CONTIGUOUS("null packed", value.packed = nullptr)
  Q3X_INVALID_CONTIGUOUS("null scales", value.scales = nullptr)
  Q3X_INVALID_CONTIGUOUS(
      "misaligned input",
      value.input = reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<const std::uint8_t*>(value.input) + 1U))
  Q3X_INVALID_CONTIGUOUS(
      "misaligned alpha",
      value.alpha = reinterpret_cast<const float*>(
          reinterpret_cast<const std::uint8_t*>(value.alpha) + 4U))
  Q3X_INVALID_CONTIGUOUS("misaligned packed", value.packed += 1U)
  Q3X_INVALID_CONTIGUOUS(
      "misaligned scales",
      value.scales = reinterpret_cast<std::uint16_t*>(
          reinterpret_cast<std::uint8_t*>(value.scales) + 1U))
  Q3X_INVALID_CONTIGUOUS("short input stride", value.stride = kK - 1U)
  Q3X_INVALID_CONTIGUOUS(
      "oversized input stride",
      value.stride =
          static_cast<std::size_t>(
              std::numeric_limits<unsigned int>::max()) +
          1U)
  Q3X_INVALID_CONTIGUOUS("short input capacity",
                         value.input_capacity = kK - 1U)
  Q3X_INVALID_CONTIGUOUS("short alpha capacity",
                         value.alpha_capacity = kK - 1U)
  Q3X_INVALID_CONTIGUOUS("short packed capacity",
                         value.packed_capacity -= 1U)
  Q3X_INVALID_CONTIGUOUS("short scale capacity",
                         value.scale_capacity -= 1U)
  Q3X_INVALID_CONTIGUOUS("zero logical tokens", value.logical = 0U)
  Q3X_INVALID_CONTIGUOUS("logical exceeds launch", value.logical = 65U)
  Q3X_INVALID_CONTIGUOUS("non-M64 launch", value.launch = 63U)
  Q3X_INVALID_CONTIGUOUS("oversized launch", value.launch = 4'160U)
  Q3X_INVALID_CONTIGUOUS("unsupported K", value.input_size = 6'208U)
  Q3X_INVALID_CONTIGUOUS("reserved R2", value.lanes = 2U)
  Q3X_INVALID_CONTIGUOUS("zero clip", value.clip = 0.0F)
  Q3X_INVALID_CONTIGUOUS("oversized clip", value.clip = 1.01F)
  Q3X_INVALID_CONTIGUOUS("NaN clip",
                         value.clip = std::numeric_limits<float>::quiet_NaN())
  Q3X_INVALID_CONTIGUOUS(
      "input-packed alias",
      value.packed = reinterpret_cast<std::uint8_t*>(
          const_cast<std::uint16_t*>(value.input)))
  Q3X_INVALID_CONTIGUOUS(
      "input-alpha alias",
      value.alpha = reinterpret_cast<const float*>(value.input))
  Q3X_INVALID_CONTIGUOUS(
      "packed-scale alias",
      value.scales = reinterpret_cast<std::uint16_t*>(value.packed))
#undef Q3X_INVALID_CONTIGUOUS

  constexpr std::size_t kPrimary =
      kernels::kSm87A4W4FactorizedLaneQuantizeDownPrimaryInput;
  constexpr std::size_t kSecondary =
      kernels::kSm87A4W4FactorizedLaneQuantizeDownSecondaryInput;
  const auto down_plan = maximum_plan;
  const SplitInvocation split{
      primary.get(), kPrimary, kPrimary, kPrimary,
      secondary.get(), kSecondary, kSecondary, kSecondary,
      alpha.get(), kernels::kSm87A4W4FactorizedLaneQuantizeDownInput,
      kLogical, kLaunch, 4U, 0.9F, packed.get(),
      down_plan.packed_capacity_bytes, scales.get(),
      down_plan.scale_capacity_elements, stream.get()};
#define Q3X_INVALID_SPLIT(LABEL, BODY)                                  \
  ok = invalid_after_mutation(                                           \
           split, [](SplitInvocation& value) { BODY; }, LABEL)           \
           &&                                                            \
       ok;
  Q3X_INVALID_SPLIT("split null primary", value.primary = nullptr)
  Q3X_INVALID_SPLIT("split null secondary", value.secondary = nullptr)
  Q3X_INVALID_SPLIT("split wrong primary size", value.primary_size -= 512U)
  Q3X_INVALID_SPLIT("split wrong secondary size",
                    value.secondary_size += 512U)
  Q3X_INVALID_SPLIT("split short primary stride",
                    value.primary_stride -= 1U)
  Q3X_INVALID_SPLIT(
      "split oversized secondary stride",
      value.secondary_stride =
          static_cast<std::size_t>(
              std::numeric_limits<unsigned int>::max()) +
          1U)
  Q3X_INVALID_SPLIT("split short secondary capacity",
                    value.secondary_capacity -= 1U)
  Q3X_INVALID_SPLIT("split plane alias", value.secondary = value.primary)
  Q3X_INVALID_SPLIT(
      "split output alias",
      value.packed = reinterpret_cast<std::uint8_t*>(
          const_cast<std::uint16_t*>(value.secondary)))
#undef Q3X_INVALID_SPLIT
  if (ok) {
    std::cout << "strict invalid-value matrix PASS\n";
  }
  return ok;
}

[[nodiscard]] bool host_plan_checks() {
  const auto gate = kernels::sm87_a4w4_factorized_lane_quantize_plan(
      1'853U, 1'920U, 5'120U, 1U);
  const auto down_r4 = kernels::sm87_a4w4_factorized_lane_quantize_plan(
      1'853U, 1'920U, 17'408U, 4U);
  const auto attention_o_r1 =
      kernels::sm87_a4w4_factorized_lane_quantize_plan(
          1'853U, 1'920U,
          kernels::kSm87A4W4FactorizedLaneQuantizeAttentionOInput, 1U);
  const auto down_r1 = kernels::sm87_a4w4_factorized_lane_quantize_plan(
      1'853U, 2'048U, 17'408U, 1U);
  const bool ok =
      gate.valid() && gate.launch_ctas == 1'920U &&
      gate.packed_capacity_bytes == 4'915'200U &&
      attention_o_r1.valid() &&
      attention_o_r1.launch_ctas == 1'920U &&
      attention_o_r1.packed_capacity_bytes == 5'898'240U &&
      down_r4.valid() &&
      down_r4.launch_ctas == 7'680U &&
      down_r4.scale_capacity_elements == 7'680U && down_r1.valid() &&
      down_r1.launch_ctas == 2'048U &&
      down_r1.packed_capacity_bytes == 17'825'792U;
  if (!ok) {
    std::cerr << "P1853/P1920/P2048 host plans failed\n";
  }
  return ok;
}

}  // namespace

int main() {
  if (std::fesetround(FE_TONEAREST) != 0) {
    std::cerr << "failed to select host nearest-even rounding\n";
    return 1;
  }
  if (!host_plan_checks()) {
    return 1;
  }
  const int device_status = target_device_status();
  if (device_status != 0) {
    return device_status;
  }
  kernels::Sm87A4W4FactorizedLaneQuantizeResources resources{};
  const int query_status =
      kernels::query_sm87_a4w4_factorized_lane_quantize_resources_cuda(
          &resources);
  if (!cuda_ok(static_cast<cudaError_t>(query_status),
               "factorized-lane resource query") ||
      resources.local_bytes != 0U || resources.registers_per_thread <= 0 ||
      resources.active_blocks_per_sm < 2 ||
      resources.maximum_threads_per_block < 256) {
    std::cerr << "resource gate failed: regs="
              << resources.registers_per_thread
              << " static=" << resources.static_shared_bytes
              << " dynamic=" << resources.dynamic_shared_bytes
              << " local=" << resources.local_bytes
              << " active=" << resources.active_blocks_per_sm << '\n';
    return 1;
  }
  std::cout << "resources PASS: regs=" << resources.registers_per_thread
            << " static=" << resources.static_shared_bytes
            << " dynamic=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active=" << resources.active_blocks_per_sm << '\n';

  const std::vector<CaseSpec> cases = {
      {"gate-r1-contiguous", 65U, 128U, 5'120U, 1U, 1.0F, false,
       false},
      {"gate-r4-contiguous", 63U, 64U, 5'120U, 4U, 0.91F, false,
       false},
      {"attention-o-r1-contiguous", 65U, 128U, 6'144U, 1U, 0.89F,
       false, false},
      {"down-r1-contiguous", 129U, 192U, 17'408U, 1U, 0.83F, false,
       false},
      {"down-r4-contiguous", 193U, 256U, 17'408U, 4U, 0.73F, false,
       false},
      {"down-r1-split", 7U, 64U, 17'408U, 1U, 1.0F, true, false},
      {"down-r4-split", 67U, 128U, 17'408U, 4U, 0.87F, true, false},
      {"gate-r4-graph", 3U, 64U, 5'120U, 4U, 1.0F, false, true},
      {"attention-o-r1-graph", 3U, 64U, 6'144U, 1U, 1.0F, false, true},
      {"down-r4-split-graph", 3U, 64U, 17'408U, 4U, 1.0F, true, true},
  };
  for (const auto& item : cases) {
    if (!run_case(item)) {
      return 1;
    }
  }
  if (!run_invalid_tests()) {
    return 1;
  }
  std::cout << "factorized-lane activation quantizer PASS\n";
  return 0;
}
