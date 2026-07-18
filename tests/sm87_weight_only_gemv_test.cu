#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "q3x/kernels/reference_gemv.h"
#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace q3x::kernels {

// Deliberately not part of the public kernel API. The implementation is
// linked into this test so performance comparisons can use identical buffers
// and one binary without weakening production shape dispatch.
[[nodiscard]] int launch_sm87_fp8_w8a16_gemv_bf16_scalar_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_small_m8_single_row_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_small_m8_row_pair_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] bool use_sm87_fp8_small_m_row_pair_test(
    std::size_t token_count, std::size_t rows) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_scalar_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] bool use_sm87_nvfp4_small_m_row_pair_test(
    std::size_t token_count, std::size_t rows) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m8_row_pair_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m8_scale_codebook_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m8_single_row_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels

namespace {

class TestContext {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool cuda_ok(const cudaError_t status,
                             const std::string& operation) {
    expect(status == cudaSuccess,
           operation + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return cudaErrorInvalidValue;
    }
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
};

class EventPair {
 public:
  EventPair() = default;
  EventPair(const EventPair&) = delete;
  EventPair& operator=(const EventPair&) = delete;

  ~EventPair() {
    if (stop_ != nullptr) {
      (void)cudaEventDestroy(stop_);
    }
    if (start_ != nullptr) {
      (void)cudaEventDestroy(start_);
    }
  }

  [[nodiscard]] bool create(TestContext& test) {
    bool ready = test.cuda_ok(cudaEventCreate(&start_), "create start event");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&stop_), "create stop event");
    return ready;
  }

  [[nodiscard]] cudaEvent_t start() const noexcept { return start_; }
  [[nodiscard]] cudaEvent_t stop() const noexcept { return stop_; }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t bits) noexcept {
  const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

[[nodiscard]] bool is_bf16_nan(const std::uint16_t bits) noexcept {
  return (bits & 0x7f80U) == 0x7f80U && (bits & 0x007fU) != 0U;
}

// Independent host decoders intentionally use arithmetic rather than the
// device kernel's bit construction.
[[nodiscard]] float decode_e4m3fn_host(const std::uint8_t bits) noexcept {
  const std::uint8_t magnitude = bits & 0x7fU;
  const int exponent = static_cast<int>((magnitude >> 3U) & 0x0fU);
  const int mantissa = static_cast<int>(magnitude & 0x07U);
  if (exponent == 0x0f && mantissa == 0x07) {
    return std::copysign(std::numeric_limits<float>::quiet_NaN(),
                         (bits & 0x80U) != 0U ? -1.0F : 1.0F);
  }
  const float value =
      exponent == 0
          ? std::ldexp(static_cast<float>(mantissa), -9)
          : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                       exponent - 7);
  return std::copysign(value, (bits & 0x80U) != 0U ? -1.0F : 1.0F);
}

[[nodiscard]] float decode_e2m1_host(const std::uint8_t nibble) noexcept {
  constexpr std::array<float, 16U> kValues{{
      0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
      -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
  }};
  return kValues[nibble & 0x0fU];
}

[[nodiscard]] std::vector<std::uint16_t> fp8_host_reference(
    const std::vector<std::uint8_t>& weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns) {
  std::vector<std::uint16_t> output(rows);
  for (std::size_t row = 0U; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0U; column < columns; ++column) {
      const double weight = static_cast<double>(
          decode_e4m3fn_host(weights[row * columns + column]));
      const double input =
          static_cast<double>(decode_bf16(activation[column]));
      sum += weight * static_cast<double>(weight_scale) * input;
    }
    output[row] = encode_bf16(static_cast<float>(sum));
  }
  return output;
}

[[nodiscard]] std::vector<std::uint16_t> nvfp4_host_reference(
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::uint8_t>& scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns) {
  const std::size_t packed_columns = columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  std::vector<std::uint16_t> output(rows);
  for (std::size_t row = 0U; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0U; column < columns; ++column) {
      const std::uint8_t byte =
          packed[row * packed_columns + column / 2U];
      const std::uint8_t nibble =
          (column & 1U) == 0U
              ? static_cast<std::uint8_t>(byte & 0x0fU)
              : static_cast<std::uint8_t>((byte >> 4U) & 0x0fU);
      const float block_scale = decode_e4m3fn_host(
          scales[row * scale_columns + column / 16U]);
      const double weight = static_cast<double>(decode_e2m1_host(nibble)) *
                            static_cast<double>(block_scale) *
                            static_cast<double>(weight_scale_2);
      sum += weight * static_cast<double>(decode_bf16(activation[column]));
    }
    output[row] = encode_bf16(static_cast<float>(sum));
  }
  return output;
}

void compare_bf16_outputs(TestContext& test,
                          const std::vector<std::uint16_t>& actual,
                          const std::vector<std::uint16_t>& expected,
                          const std::size_t columns,
                          const std::string& label) {
  test.expect(actual.size() == expected.size(), label + " output size");
  const std::size_t count = std::min(actual.size(), expected.size());
  for (std::size_t row = 0U; row < count; ++row) {
    if (is_bf16_nan(expected[row])) {
      // FMA/reduction is allowed to canonicalize the sign of a NaN. The CUDA
      // reference comparison below still checks the device-visible result;
      // this independent host oracle checks classification only.
      test.expect(is_bf16_nan(actual[row]),
                  label + " row " + std::to_string(row) +
                      " must preserve the expected NaN class");
      continue;
    }
    const float actual_value = decode_bf16(actual[row]);
    const float expected_value = decode_bf16(expected[row]);
    const float tolerance =
        2.0e-4F * std::sqrt(static_cast<float>(columns)) +
        1.0e-2F * std::max(1.0F, std::fabs(expected_value));
    if (!(std::isfinite(actual_value) &&
          std::fabs(actual_value - expected_value) <= tolerance)) {
      test.expect(false, label + " row " + std::to_string(row) +
                             ": expected " +
                             std::to_string(expected_value) + ", got " +
                             std::to_string(actual_value) + ", tolerance " +
                             std::to_string(tolerance));
    }
  }
}

void compare_cuda_reference_outputs(
    TestContext& test, const std::vector<std::uint16_t>& optimized,
    const std::vector<std::uint16_t>& reference, const std::size_t columns,
    const std::string& label) {
  test.expect(optimized.size() == reference.size(),
              label + " CUDA-reference output size");
  const std::size_t count = std::min(optimized.size(), reference.size());
  std::size_t bf16_mismatches = 0U;
  float maximum_absolute_error = 0.0F;
  float maximum_relative_error = 0.0F;
  for (std::size_t row = 0U; row < count; ++row) {
    if (optimized[row] != reference[row]) {
      ++bf16_mismatches;
    }
    const float optimized_value = decode_bf16(optimized[row]);
    const float reference_value = decode_bf16(reference[row]);
    if (is_bf16_nan(reference[row])) {
      test.expect(is_bf16_nan(optimized[row]),
                  label + " CUDA-reference row " + std::to_string(row) +
                      " NaN class mismatch");
      continue;
    }
    const float absolute_error =
        std::fabs(optimized_value - reference_value);
    const float relative_error =
        absolute_error / std::max(1.0e-6F, std::fabs(reference_value));
    maximum_absolute_error =
        std::max(maximum_absolute_error, absolute_error);
    maximum_relative_error =
        std::max(maximum_relative_error, relative_error);
    const float tolerance =
        2.0e-4F * std::sqrt(static_cast<float>(columns)) +
        1.0e-2F * std::max(1.0F, std::fabs(reference_value));
    test.expect(std::isfinite(optimized_value) &&
                    std::isfinite(reference_value) &&
                    absolute_error <= tolerance,
                label + " CUDA-reference row " + std::to_string(row) +
                    " exceeds tolerance");
  }
  std::cout << "DIFF: " << label << " optimized_vs_reference_bf16="
            << bf16_mismatches << '/' << count
            << " max_abs=" << maximum_absolute_error
            << " max_rel=" << maximum_relative_error << '\n';
}

[[nodiscard]] bool seed_stale_error(TestContext& test) {
  const cudaError_t status =
      cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
  test.expect(status == cudaErrorInvalidValue,
              "invalid CUDA copy seeds stale last-error");
  return status == cudaErrorInvalidValue;
}

void fill_activation(std::vector<std::uint16_t>& activation) {
  for (std::size_t index = 0U; index < activation.size(); ++index) {
    const int centered = static_cast<int>((index * 5U + 3U) % 31U) - 15;
    activation[index] =
        encode_bf16(static_cast<float>(centered) / 128.0F);
  }
}

void run_fp8_payload(TestContext& test, cudaStream_t stream,
                      const std::vector<std::uint8_t>& weights,
                      const float weight_scale,
                      const std::vector<std::uint16_t>& activation,
                      const std::size_t rows, const std::size_t columns,
                      const std::string& label,
                      const bool unaligned_weights = false,
                      const bool unaligned_activation = false,
                      const bool strict_bf16 = false,
                      const std::size_t token_count = 1U,
                      const bool use_small_m_api = false) {
  test.expect(token_count >= 1U && token_count <= 8U,
              label + " valid token count");
  test.expect(use_small_m_api || token_count == 1U,
              label + " multi-token payload uses the small-M API");
  test.expect(activation.size() == token_count * columns,
              label + " activation extent");
  if (token_count < 1U || token_count > 8U ||
      (!use_small_m_api && token_count != 1U) ||
      activation.size() != token_count * columns) {
    return;
  }
  const std::size_t output_count = token_count * rows;
  std::vector<std::uint16_t> expected(output_count);
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::vector<std::uint16_t> token_expected = fp8_host_reference(
        weights, weight_scale, activation.data() + token * columns, rows,
        columns);
    std::copy(token_expected.begin(), token_expected.end(),
              expected.begin() + token * rows);
  }
  std::vector<std::uint16_t> actual(output_count, 0U);
  std::vector<std::uint16_t> repeated(output_count, 0U);
  std::vector<std::uint16_t> reference(output_count, 0U);
  std::vector<std::uint16_t> baseline(output_count, 0U);

  DeviceBuffer<std::uint8_t> device_weights;
  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_output;
  DeviceBuffer<std::uint16_t> device_baseline;
  DeviceBuffer<float> device_reference_fp32;
  DeviceBuffer<std::uint16_t> device_reference_output;
  const std::size_t weight_offset = unaligned_weights ? 1U : 0U;
  const std::size_t activation_offset = unaligned_activation ? 1U : 0U;
  bool ready = test.cuda_ok(
      device_weights.allocate(weights.size() + weight_offset),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(
                       device_activation.allocate(activation.size() +
                                                  activation_offset),
                       label + " allocate activation");
  ready = ready && test.cuda_ok(device_output.allocate(output_count),
                                label + " allocate output");
  ready = ready && test.cuda_ok(device_reference_fp32.allocate(output_count),
                                label + " allocate reference FP32");
  ready = ready && test.cuda_ok(device_reference_output.allocate(output_count),
                                label + " allocate reference BF16");
  if (use_small_m_api) {
    ready = ready && test.cuda_ok(device_baseline.allocate(output_count),
                                  label + " allocate repeated-M1 baseline");
  }
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_weights.get() + weight_offset,
                                       weights.data(),
                                       weights.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_activation.get() +
                                           activation_offset,
                                       activation.data(),
                                       activation.size() * sizeof(activation[0]),
                                       cudaMemcpyHostToDevice, stream),
                       label + " copy activation");
  if (!ready) {
    return;
  }
  const std::uint8_t* const weights_device =
      device_weights.get() + weight_offset;
  const std::uint16_t* const activation_device =
      device_activation.get() + activation_offset;

  const auto launch_optimized = [&]() noexcept -> int {
    if (use_small_m_api) {
      return q3x::kernels::launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
          weights_device, weight_scale, activation_device, token_count, rows,
          columns, device_output.get(), static_cast<void*>(stream));
    }
    return q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
        weights_device, weight_scale, activation_device, rows, columns,
        device_output.get(), static_cast<void*>(stream));
  };

  (void)seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(launch_optimized()),
      label + " launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(actual.data(), device_output.get(),
                                       actual.size() * sizeof(actual[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy output");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_optimized()),
                       label + " repeat launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(repeated.data(), device_output.get(),
                                       repeated.size() * sizeof(repeated[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy repeated output");
  for (std::size_t token = 0U; token < token_count && ready; ++token) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(
            q3x::kernels::launch_fp8_gemv_reference_cuda(
                weights_device, weight_scale,
                activation_device + token * columns, rows, columns,
                device_reference_fp32.get() + token * rows,
                static_cast<void*>(stream))),
        label + " launch CUDA reference token " + std::to_string(token));
    if (use_small_m_api) {
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(q3x::kernels::
                               launch_sm87_fp8_w8a16_gemv_bf16_cuda(
                                   weights_device, weight_scale,
                                   activation_device + token * columns, rows,
                                   columns,
                                   device_baseline.get() + token * rows,
                                   static_cast<void*>(stream))),
                           label + " launch repeated-M1 baseline token " +
                               std::to_string(token));
    }
  }
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                               device_reference_fp32.get(), output_count,
                               device_reference_output.get(),
                               static_cast<void*>(stream))),
                       label + " convert CUDA reference");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(reference.data(),
                                       device_reference_output.get(),
                                       reference.size() * sizeof(reference[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy CUDA reference");
  if (use_small_m_api) {
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), device_baseline.get(),
                             baseline.size() * sizeof(baseline[0]),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy repeated-M1 baseline");
  }
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (ready) {
    compare_bf16_outputs(test, actual, expected, columns, label);
    compare_cuda_reference_outputs(test, actual, reference, columns, label);
    if (strict_bf16) {
      for (std::size_t index = 0U; index < output_count; ++index) {
        if (is_bf16_nan(expected[index])) {
          test.expect(is_bf16_nan(actual[index]),
                      label + " strict output " + std::to_string(index) +
                          " host NaN class");
          test.expect(is_bf16_nan(reference[index]),
                      label + " strict output " + std::to_string(index) +
                          " CUDA-reference NaN class");
        } else {
          test.expect(actual[index] == expected[index],
                      label + " strict output " + std::to_string(index) +
                          " must equal host BF16 bits");
          test.expect(actual[index] == reference[index],
                      label + " strict output " + std::to_string(index) +
                          " must equal CUDA-reference BF16 bits");
        }
      }
    }
    test.expect(actual == repeated, label + " is bitwise deterministic");
    if (use_small_m_api) {
      std::size_t mismatches = 0U;
      for (std::size_t index = 0U; index < output_count; ++index) {
        mismatches += actual[index] != baseline[index] ? 1U : 0U;
      }
      std::cout << "SMALL_M_DIFF: " << label
                << " optimized_vs_repeated_m1_bf16=" << mismatches << '/'
                << output_count << '\n';
      test.expect(actual == baseline,
                  label + " matches repeated production M1 bits");
    }
  }
}

void run_fp8_case(TestContext& test, cudaStream_t stream,
                  const std::size_t rows, const std::size_t columns,
                  const std::string& label,
                  const bool unaligned_weights = false,
                  const bool unaligned_activation = false,
                  const std::size_t token_count = 1U,
                  const bool use_small_m_api = false) {
  constexpr float kWeightScale = 1.0F / 128.0F;
  constexpr std::array<std::uint8_t, 13U> kFiniteCodes{{
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U, 0x3cU,
      0x40U, 0xb8U, 0x70U, 0x78U, 0x7eU, 0xfeU,
  }};
  std::vector<std::uint16_t> activation(token_count * columns);
  std::vector<std::uint8_t> weights(rows * columns);
  fill_activation(activation);
  for (std::size_t index = 0U; index < weights.size(); ++index) {
    weights[index] = kFiniteCodes[(index * 7U + 1U) % kFiniteCodes.size()];
  }
  run_fp8_payload(test, stream, weights, kWeightScale, activation, rows,
                  columns, label, unaligned_weights, unaligned_activation,
                  false, token_count, use_small_m_api);
}

void run_fp8_row_pair_direct_comparison(
    TestContext& test, cudaStream_t stream,
    const std::vector<std::uint8_t>& weights,
    const std::vector<std::uint16_t>& activations, const std::size_t rows,
    const std::size_t columns, const float weight_scale,
    const std::string& label) {
  constexpr std::size_t kTokens = 8U;
  test.expect(weights.size() == rows * columns,
              label + " weight extent");
  test.expect(activations.size() == kTokens * columns,
              label + " activation extent");
  if (weights.size() != rows * columns ||
      activations.size() != kTokens * columns) {
    return;
  }

  DeviceBuffer<std::uint8_t> device_weights;
  DeviceBuffer<std::uint16_t> device_activations;
  DeviceBuffer<std::uint16_t> device_baseline;
  DeviceBuffer<std::uint16_t> device_candidate;
  bool ready = test.cuda_ok(device_weights.allocate(weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(
                       device_activations.allocate(activations.size()),
                       label + " allocate activations");
  ready = ready && test.cuda_ok(device_baseline.allocate(kTokens * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(device_candidate.allocate(kTokens * rows),
                                label + " allocate row-pair output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_weights.get(), weights.data(),
                                       weights.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_activations.get(), activations.data(),
                           activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy activations");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m8_single_row_test_cuda(
              device_weights.get(), weight_scale, device_activations.get(),
              rows, columns, device_baseline.get(),
              static_cast<void*>(stream))),
      label + " launch single-row M8 baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_small_m8_row_pair_test_cuda(
                               device_weights.get(), weight_scale,
                               device_activations.get(), rows, columns,
                               device_candidate.get(),
                               static_cast<void*>(stream))),
                       label + " launch row-pair M8 candidate");
  std::vector<std::uint16_t> baseline(kTokens * rows);
  std::vector<std::uint16_t> candidate(kTokens * rows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), device_baseline.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy baseline output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           candidate.data(), device_candidate.get(),
                           candidate.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy row-pair output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize");
  if (!ready) {
    return;
  }

  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    mismatches += baseline[index] != candidate[index] ? 1U : 0U;
  }
  std::cout << "FP8_ROW_PAIR_DIFF: " << label
            << " candidate_vs_single_row_m8_bf16=" << mismatches << '/'
            << baseline.size() << '\n';
  test.expect(mismatches == 0U,
              label + " row-pair matches every single-row M8 BF16 bit");
}

void run_fp8_vector_codebook_case(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kColumns = 1'024U;
  constexpr std::size_t kBytePositions = 4U;
  constexpr std::size_t kRows = 256U * kBytePositions;
  constexpr std::array<float, kBytePositions> kDistinctActivations{{
      0.5F, 0.75F, 1.25F, 1.75F,
  }};
  std::vector<std::uint16_t> activation(kColumns, encode_bf16(1.0F));
  for (std::size_t position = 0U; position < kBytePositions; ++position) {
    activation[position] = encode_bf16(kDistinctActivations[position]);
  }
  std::vector<std::uint8_t> weights(kRows * kColumns, 0U);

  // Every E4M3FN encoding occupies every byte in the first uint32 load.
  // Distinct, exactly representable activations make a byte-to-K permutation
  // visible. The two reserved encodings additionally exercise signed NaNs.
  for (std::size_t code = 0U; code < 256U; ++code) {
    for (std::size_t position = 0U; position < kBytePositions; ++position) {
      const std::size_t row = code * kBytePositions + position;
      weights[row * kColumns + position] =
          static_cast<std::uint8_t>(code);
    }
  }
  run_fp8_payload(test, stream, weights, 1.0F, activation, kRows, kColumns,
                  "FP8 packed-x4 exhaustive codebook and byte positions",
                  false, false, true);

  constexpr std::size_t kTokens = 8U;
  constexpr std::array<float, kTokens> kTokenFactors{{
      0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 1.75F, 2.0F, 2.5F,
  }};
  std::vector<std::uint16_t> batched_activation(kTokens * kColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      batched_activation[token * kColumns + column] = encode_bf16(
          decode_bf16(activation[column]) * kTokenFactors[token]);
    }
  }
  run_fp8_payload(
      test, stream, weights, 1.0F, batched_activation, kRows, kColumns,
      "FP8 small-M8 packed-x4 exhaustive codebook and byte positions", false,
      false, true, kTokens, true);
  run_fp8_row_pair_direct_comparison(
      test, stream, weights, batched_activation, kRows, kColumns, 1.0F,
      "FP8 M8 row-pair exhaustive E4M3FN codebook and byte positions");
}

void run_fp8_row_pair_odd_rows_case(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kTokens = 8U;
  constexpr std::size_t kRows = 17U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr float kWeightScale = 1.0F / 128.0F;
  constexpr std::array<std::uint8_t, 13U> kFiniteCodes{{
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U, 0x3cU,
      0x40U, 0xb8U, 0x70U, 0x78U, 0x7eU, 0xfeU,
  }};
  std::vector<std::uint8_t> weights(kRows * kColumns);
  for (std::size_t row = 0U; row < kRows; ++row) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      weights[row * kColumns + column] = kFiniteCodes[
          (column * 11U + row * 7U + 3U) % kFiniteCodes.size()];
    }
  }
  std::vector<std::uint16_t> activations(kTokens * kColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const int centered =
          static_cast<int>((column * 13U + token * 17U + 5U) % 127U) - 63;
      activations[token * kColumns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
  }
  run_fp8_row_pair_direct_comparison(
      test, stream, weights, activations, kRows, kColumns, kWeightScale,
      "FP8 M8 row-pair odd N17 K5120 row-distinct finite codebook");
}

void run_nvfp4_payload(TestContext& test, cudaStream_t stream,
                       const std::vector<std::uint8_t>& packed,
                       const std::vector<std::uint8_t>& scales,
                       const float weight_scale_2,
                       const std::vector<std::uint16_t>& activation,
                       const std::size_t rows, const std::size_t columns,
                       const std::string& label,
                       const bool unaligned_packed = false,
                       const bool unaligned_activation = false,
                       const bool strict_bf16 = false,
                       const std::size_t token_count = 1U,
                       const bool use_small_m_api = false) {
  test.expect(token_count >= 1U && token_count <= 8U,
              label + " valid token count");
  test.expect(use_small_m_api || token_count == 1U,
              label + " multi-token payload uses the small-M API");
  test.expect(activation.size() == token_count * columns,
              label + " activation extent");
  if (token_count < 1U || token_count > 8U ||
      (!use_small_m_api && token_count != 1U) ||
      activation.size() != token_count * columns) {
    return;
  }
  const std::size_t output_count = token_count * rows;
  std::vector<std::uint16_t> expected(output_count);
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::vector<std::uint16_t> token_expected = nvfp4_host_reference(
        packed, scales, weight_scale_2,
        activation.data() + token * columns, rows, columns);
    std::copy(token_expected.begin(), token_expected.end(),
              expected.begin() + token * rows);
  }
  std::vector<std::uint16_t> actual(output_count, 0U);
  std::vector<std::uint16_t> repeated(output_count, 0U);
  std::vector<std::uint16_t> reference(output_count, 0U);
  std::vector<std::uint16_t> baseline(output_count, 0U);

  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_output;
  DeviceBuffer<std::uint16_t> device_baseline;
  DeviceBuffer<float> device_reference_fp32;
  DeviceBuffer<std::uint16_t> device_reference_output;
  const std::size_t packed_offset = unaligned_packed ? 1U : 0U;
  const std::size_t activation_offset = unaligned_activation ? 1U : 0U;
  bool ready = test.cuda_ok(
      device_packed.allocate(packed.size() + packed_offset),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate scales");
  ready = ready && test.cuda_ok(
                       device_activation.allocate(activation.size() +
                                                  activation_offset),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(device_output.allocate(output_count),
                                label + " allocate output");
  ready = ready && test.cuda_ok(device_reference_fp32.allocate(output_count),
                                label + " allocate reference FP32");
  ready = ready && test.cuda_ok(device_reference_output.allocate(output_count),
                                label + " allocate reference BF16");
  if (use_small_m_api) {
    ready = ready && test.cuda_ok(device_baseline.allocate(output_count),
                                  label + " allocate repeated-M1 baseline");
  }
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_packed.get() + packed_offset,
                                       packed.data(),
                                       packed.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_scales.get(), scales.data(),
                                       scales.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_activation.get() +
                                           activation_offset,
                                       activation.data(),
                                       activation.size() * sizeof(activation[0]),
                                       cudaMemcpyHostToDevice, stream),
                       label + " copy activation");
  if (!ready) {
    return;
  }
  const std::uint8_t* const packed_device =
      device_packed.get() + packed_offset;
  const std::uint16_t* const activation_device =
      device_activation.get() + activation_offset;
  const auto launch_optimized = [&]() noexcept -> int {
    if (use_small_m_api) {
      return q3x::kernels::launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
          packed_device, device_scales.get(), weight_scale_2,
          activation_device, token_count, rows, columns, device_output.get(),
          static_cast<void*>(stream));
    }
    return q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
        packed_device, device_scales.get(), weight_scale_2, activation_device,
        rows, columns, device_output.get(), static_cast<void*>(stream));
  };

  (void)seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(launch_optimized()),
      label + " launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(actual.data(), device_output.get(),
                                       actual.size() * sizeof(actual[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy output");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_optimized()),
                       label + " repeat launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(repeated.data(), device_output.get(),
                                       repeated.size() * sizeof(repeated[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy repeated output");
  for (std::size_t token = 0U; token < token_count && ready; ++token) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(
            q3x::kernels::launch_nvfp4_gemv_reference_cuda(
                packed_device, device_scales.get(), weight_scale_2,
                activation_device + token * columns, rows, columns,
                device_reference_fp32.get() + token * rows,
                static_cast<void*>(stream))),
        label + " launch CUDA reference token " + std::to_string(token));
    if (use_small_m_api) {
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(q3x::kernels::
                               launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
                                   packed_device, device_scales.get(),
                                   weight_scale_2,
                                   activation_device + token * columns, rows,
                                   columns,
                                   device_baseline.get() + token * rows,
                                   static_cast<void*>(stream))),
                           label + " launch repeated-M1 baseline token " +
                               std::to_string(token));
    }
  }
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                               device_reference_fp32.get(), output_count,
                               device_reference_output.get(),
                               static_cast<void*>(stream))),
                       label + " convert CUDA reference");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(reference.data(),
                                       device_reference_output.get(),
                                       reference.size() * sizeof(reference[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy CUDA reference");
  if (use_small_m_api) {
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), device_baseline.get(),
                             baseline.size() * sizeof(baseline[0]),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy repeated-M1 baseline");
  }
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (ready) {
    compare_bf16_outputs(test, actual, expected, columns, label);
    compare_cuda_reference_outputs(test, actual, reference, columns, label);
    if (strict_bf16) {
      for (std::size_t index = 0U; index < output_count; ++index) {
        if (is_bf16_nan(expected[index])) {
          test.expect(is_bf16_nan(actual[index]),
                      label + " strict output " + std::to_string(index) +
                          " host NaN class");
          test.expect(is_bf16_nan(reference[index]),
                      label + " strict output " + std::to_string(index) +
                          " CUDA-reference NaN class");
        } else {
          test.expect(actual[index] == expected[index],
                      label + " strict output " + std::to_string(index) +
                          " must equal host BF16 bits");
          test.expect(actual[index] == reference[index],
                      label + " strict output " + std::to_string(index) +
                          " must equal CUDA-reference BF16 bits");
        }
      }
    }
    test.expect(actual == repeated, label + " is bitwise deterministic");
    if (use_small_m_api) {
      std::size_t mismatches = 0U;
      for (std::size_t index = 0U; index < output_count; ++index) {
        mismatches += actual[index] != baseline[index] ? 1U : 0U;
      }
      std::cout << "SMALL_M_DIFF: " << label
                << " optimized_vs_repeated_m1_bf16=" << mismatches << '/'
                << output_count << '\n';
      test.expect(actual == baseline,
                  label + " matches repeated production M1 bits");
    }
  }
}

void run_nvfp4_case(TestContext& test, cudaStream_t stream,
                    const std::size_t rows, const std::size_t columns,
                    const std::string& label,
                    const bool unaligned_packed = false,
                    const bool unaligned_activation = false,
                    const std::size_t token_count = 1U,
                    const bool use_small_m_api = false) {
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 8U> kScaleCodes{{
      0x20U, 0x28U, 0x30U, 0x34U, 0x38U, 0x3cU, 0x40U, 0x44U,
  }};
  std::vector<std::uint16_t> activation(token_count * columns);
  std::vector<std::uint8_t> packed(rows * columns / 2U);
  std::vector<std::uint8_t> scales(rows * columns / 16U);
  fill_activation(activation);
  for (std::size_t index = 0U; index < packed.size(); ++index) {
    const std::uint8_t low = static_cast<std::uint8_t>((index * 3U) & 0x0fU);
    const std::uint8_t high =
        static_cast<std::uint8_t>((index * 11U + 5U) & 0x0fU);
    packed[index] = static_cast<std::uint8_t>(low | (high << 4U));
  }
  for (std::size_t index = 0U; index < scales.size(); ++index) {
    scales[index] = kScaleCodes[(index * 5U + 1U) % kScaleCodes.size()];
  }
  run_nvfp4_payload(test, stream, packed, scales, kWeightScale2, activation,
                    rows, columns, label, unaligned_packed,
                    unaligned_activation, false, token_count,
                    use_small_m_api);
}

void run_nvfp4_vector_codebook_case(TestContext& test,
                                    cudaStream_t stream) {
  constexpr std::size_t kColumns = 256U;
  constexpr std::size_t kCodebookRows = 16U * 8U;
  constexpr std::size_t kScaleRows = 16U * 2U;
  constexpr std::size_t kRows = kCodebookRows + kScaleRows;
  constexpr std::size_t kPackedColumns = kColumns / 2U;
  constexpr std::size_t kScaleColumns = kColumns / 16U;
  constexpr std::uint8_t kE2m1One = 0x02U;
  constexpr std::array<std::uint8_t, 16U> kDistinctScaleCodes{{
      0x20U, 0x24U, 0x28U, 0x2cU, 0x30U, 0x34U, 0x38U, 0x3cU,
      0x40U, 0x44U, 0x48U, 0x4cU, 0x50U, 0x54U, 0x58U, 0x5cU,
  }};
  constexpr std::array<float, 8U> kDistinctWordActivations{{
      0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 1.75F, 2.0F, 2.5F,
  }};

  std::vector<std::uint16_t> activation(kColumns, encode_bf16(1.0F));
  for (std::size_t index = 0U; index < kDistinctWordActivations.size();
       ++index) {
    activation[index] = encode_bf16(kDistinctWordActivations[index]);
  }
  std::vector<std::uint8_t> packed(kRows * kPackedColumns, 0U);
  std::vector<std::uint8_t> scales(kRows * kScaleColumns, 0x38U);

  // Every E2M1 code occupies each of the eight nibble positions in one
  // uint32 load. The corresponding activations are distinct, BF16-exact
  // values, so a byte/nibble-to-K permutation changes the result. All other
  // weights are +0, keeping the expected value independently auditable.
  for (std::size_t code = 0U; code < 16U; ++code) {
    for (std::size_t nibble_position = 0U; nibble_position < 8U;
         ++nibble_position) {
      const std::size_t row = code * 8U + nibble_position;
      const std::size_t byte = nibble_position / 2U;
      const unsigned int shift =
          static_cast<unsigned int>((nibble_position & 1U) * 4U);
      packed[row * kPackedColumns + byte] = static_cast<std::uint8_t>(
          static_cast<std::uint8_t>(code) << shift);
    }
  }

  // Each scale group is exercised once by its even-lane half and once by its
  // odd-lane half. Distinct scale codes catch an incorrect lane-pair source or
  // scale-column calculation.
  for (std::size_t group = 0U; group < 16U; ++group) {
    for (std::size_t half = 0U; half < 2U; ++half) {
      const std::size_t row = kCodebookRows + group * 2U + half;
      const std::size_t column = group * 16U + half * 8U;
      const std::size_t byte = column / 2U;
      packed[row * kPackedColumns + byte] = kE2m1One;
      scales[row * kScaleColumns + group] = kDistinctScaleCodes[group];
    }
  }

  run_nvfp4_payload(test, stream, packed, scales, 1.0F, activation, kRows,
                    kColumns, "NVFP4 vector codebook and lane-pair scales");

  constexpr std::size_t kTokens = 8U;
  constexpr std::array<float, kTokens> kTokenFactors{{
      0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 1.75F, 2.0F, 2.5F,
  }};
  std::vector<std::uint16_t> batched_activation(kTokens * kColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      batched_activation[token * kColumns + column] = encode_bf16(
          decode_bf16(activation[column]) * kTokenFactors[token]);
    }
  }
  run_nvfp4_payload(
      test, stream, packed, scales, 1.0F, batched_activation, kRows, kColumns,
      "NVFP4 small-M8 codebook and lane-pair scales", false, false, true,
      kTokens, true);
}

void run_nvfp4_row_pair_bitwise_case(TestContext& test,
                                     cudaStream_t stream,
                                     const std::size_t rows,
                                     const std::size_t columns,
                                     const std::string& label) {
  constexpr std::size_t kTokens = 8U;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 16U> kFiniteScaleCodes{{
      0x20U, 0x24U, 0x28U, 0x2cU, 0x30U, 0x34U, 0x38U, 0x3cU,
      0x40U, 0x44U, 0x48U, 0x4cU, 0x50U, 0x54U, 0x58U, 0x5cU,
  }};
  const std::size_t packed_columns = columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_count = rows * columns / 16U;
  std::vector<std::uint8_t> packed(packed_count);
  std::vector<std::uint8_t> scales(scale_count);
  std::vector<std::uint16_t> activations(kTokens * columns);

  // Every row covers all 256 low/high E2M1 pairs, but row-dependent offsets
  // ensure adjacent rows are not identical even when packed_columns is a
  // multiple of 256. This directly catches accidental row0 weight reuse.
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t packed_column = 0U; packed_column < packed_columns;
         ++packed_column) {
      const std::uint8_t low = static_cast<std::uint8_t>(
          (packed_column + row * 3U) & 0x0fU);
      const std::uint8_t high = static_cast<std::uint8_t>(
          ((packed_column >> 4U) + row * 5U) & 0x0fU);
      packed[row * packed_columns + packed_column] =
          static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t scale_column = 0U; scale_column < scale_columns;
         ++scale_column) {
      scales[row * scale_columns + scale_column] = kFiniteScaleCodes[
          (scale_column * 13U + row * 7U) % kFiniteScaleCodes.size()];
    }
  }
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered =
          static_cast<int>((column * 17U + token * 11U + 5U) % 127U) - 63;
      activations[token * columns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
  }

  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  DeviceBuffer<std::uint16_t> device_activations;
  DeviceBuffer<std::uint16_t> device_baseline;
  DeviceBuffer<std::uint16_t> device_row_pair;
  DeviceBuffer<std::uint16_t> device_scale_codebook;
  bool ready = test.cuda_ok(device_packed.allocate(packed.size()),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(
                       device_activations.allocate(activations.size()),
                       label + " allocate activations");
  ready = ready && test.cuda_ok(device_baseline.allocate(kTokens * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(device_row_pair.allocate(kTokens * rows),
                                label + " allocate row-pair output");
  ready = ready && test.cuda_ok(
                       device_scale_codebook.allocate(kTokens * rows),
                       label + " allocate scale-codebook output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_packed.get(), packed.data(),
                                       packed.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_scales.get(), scales.data(),
                                       scales.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy block scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_activations.get(), activations.data(),
                           activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy activations");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m8_single_row_test_cuda(
              device_packed.get(), device_scales.get(), kWeightScale2,
              device_activations.get(), rows, columns, device_baseline.get(),
              static_cast<void*>(stream))),
      label + " launch preserved single-row baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_small_m8_row_pair_test_cuda(
                               device_packed.get(), device_scales.get(),
                               kWeightScale2, device_activations.get(), rows,
                               columns, device_row_pair.get(),
                               static_cast<void*>(stream))),
                       label + " launch preserved row-pair baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_small_m8_scale_codebook_test_cuda(
                               device_packed.get(), device_scales.get(),
                               kWeightScale2, device_activations.get(), rows,
                               columns, device_scale_codebook.get(),
                               static_cast<void*>(stream))),
                       label + " launch production scale-codebook row-pair");
  std::vector<std::uint16_t> baseline(kTokens * rows);
  std::vector<std::uint16_t> row_pair(kTokens * rows);
  std::vector<std::uint16_t> scale_codebook(kTokens * rows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), device_baseline.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy baseline output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           row_pair.data(), device_row_pair.get(),
                           row_pair.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy row-pair output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           scale_codebook.data(), device_scale_codebook.get(),
                           scale_codebook.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy scale-codebook output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize");
  if (!ready) {
    return;
  }

  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    mismatches += baseline[index] != row_pair[index] ? 1U : 0U;
  }
  std::cout << "ROW_PAIR_DIFF: " << label
            << " preserved_row_pair_vs_single_row_m8_bf16=" << mismatches
            << '/'
            << baseline.size() << '\n';
  test.expect(mismatches == 0U,
              label + " preserved row-pair matches single-row BF16 bits");
  std::size_t scale_codebook_mismatches = 0U;
  for (std::size_t index = 0U; index < row_pair.size(); ++index) {
    scale_codebook_mismatches +=
        row_pair[index] != scale_codebook[index] ? 1U : 0U;
  }
  std::cout << "SCALE_CODEBOOK_DIFF: " << label
            << " production_vs_preserved_row_pair_bf16="
            << scale_codebook_mismatches << '/' << row_pair.size() << '\n';
  test.expect(scale_codebook_mismatches == 0U,
              label + " production scale codebook matches row-pair bits");
}

void run_nvfp4_scale_codebook_exhaustive_case(TestContext& test,
                                              cudaStream_t stream) {
  constexpr std::size_t kTokens = 8U;
  constexpr std::size_t kRows = 257U;
  constexpr std::size_t kColumns = 256U;
  constexpr std::size_t kPackedColumns = kColumns / 2U;
  constexpr std::size_t kScaleColumns = kColumns / 16U;
  constexpr float kWeightScale2 = 1.0F;
  const std::string label =
      "NVFP4 M8 shared E4M3FN exhaustive scale codebook odd rows";

  std::vector<std::uint8_t> packed(kRows * kPackedColumns, 0U);
  std::vector<std::uint8_t> scales(kRows * kScaleColumns, 0x38U);
  std::vector<std::uint16_t> activations(kTokens * kColumns);
  for (std::size_t row = 0U; row < kRows; ++row) {
    const std::size_t active_group = row % kScaleColumns;
    const std::size_t first_byte = active_group * 8U;
    for (std::size_t byte = 0U; byte < 8U; ++byte) {
      packed[row * kPackedColumns + first_byte + byte] =
          static_cast<std::uint8_t>(0x21U + ((row + byte) & 1U) * 0x12U);
    }
    // Rows 0..255 cover every E4M3FN code exactly once, including signed
    // zeros and both NaN encodings. Row 256 exercises the unpaired tail.
    scales[row * kScaleColumns + active_group] =
        static_cast<std::uint8_t>(row < 256U ? row : 0x38U);
  }
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const int centered =
          static_cast<int>((token * 29U + column * 7U + 3U) % 61U) - 30;
      activations[token * kColumns + column] =
          encode_bf16(static_cast<float>(centered) / 32.0F);
    }
  }

  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  DeviceBuffer<std::uint16_t> device_activations;
  DeviceBuffer<std::uint16_t> device_baseline;
  DeviceBuffer<std::uint16_t> device_candidate;
  bool ready = test.cuda_ok(device_packed.allocate(packed.size()),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(
                       device_activations.allocate(activations.size()),
                       label + " allocate activations");
  ready = ready && test.cuda_ok(device_baseline.allocate(kTokens * kRows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(device_candidate.allocate(kTokens * kRows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_packed.get(), packed.data(),
                                       packed.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_scales.get(), scales.data(),
                                       scales.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy block scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_activations.get(), activations.data(),
                           activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy activations");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m8_row_pair_test_cuda(
              device_packed.get(), device_scales.get(), kWeightScale2,
              device_activations.get(), kRows, kColumns,
              device_baseline.get(), static_cast<void*>(stream))),
                       label + " launch preserved row-pair baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_small_m8_scale_codebook_test_cuda(
                               device_packed.get(), device_scales.get(),
                               kWeightScale2, device_activations.get(), kRows,
                               kColumns, device_candidate.get(),
                               static_cast<void*>(stream))),
                       label + " launch production shared scale codebook");
  std::vector<std::uint16_t> baseline(kTokens * kRows);
  std::vector<std::uint16_t> candidate(kTokens * kRows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), device_baseline.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy baseline output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           candidate.data(), device_candidate.get(),
                           candidate.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy candidate output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize");
  if (!ready) {
    return;
  }

  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    mismatches += baseline[index] != candidate[index] ? 1U : 0U;
  }
  std::cout << "SCALE_CODEBOOK_DIFF: " << label
            << " production_vs_preserved_row_pair_bf16=" << mismatches << '/'
            << baseline.size() << '\n';
  test.expect(mismatches == 0U,
              label + " production matches every preserved BF16 bit");
  for (std::size_t token = 0U; token < kTokens; ++token) {
    test.expect(is_bf16_nan(baseline[token * kRows + 0x7fU]),
                label + " positive NaN scale remains NaN");
    test.expect(is_bf16_nan(baseline[token * kRows + 0xffU]),
                label + " negative NaN scale remains NaN");
  }
}

void run_small_m_production_k_comparison(TestContext& test,
                                         cudaStream_t stream,
                                         const std::size_t token_count) {
  constexpr std::size_t kRows = 3U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr float kFp8Scale = 1.0F / 128.0F;
  constexpr float kNvFp4Scale = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 13U> kFp8Codes{{
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U, 0x3cU,
      0x40U, 0xb8U, 0x70U, 0x78U, 0x7eU, 0xfeU,
  }};

  const std::string label = "small-M" + std::to_string(token_count);
  std::vector<std::uint16_t> activations(token_count * kColumns);
  for (std::size_t token = 0U; token < token_count; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const int centered =
          static_cast<int>((column * 5U + token * 7U + 3U) % 31U) - 15;
      activations[token * kColumns + column] =
          encode_bf16(static_cast<float>(centered) / 128.0F);
    }
  }

  DeviceBuffer<std::uint16_t> device_activations;
  DeviceBuffer<std::uint16_t> baseline;
  DeviceBuffer<std::uint16_t> batched;
  DeviceBuffer<float> reference_fp32;
  DeviceBuffer<std::uint16_t> reference_bf16;
  bool ready = test.cuda_ok(device_activations.allocate(activations.size()),
                            label + " allocate activations");
  ready = ready && test.cuda_ok(baseline.allocate(token_count * kRows),
                                label + " allocate baseline");
  ready = ready && test.cuda_ok(batched.allocate(token_count * kRows),
                                label + " allocate batched");
  ready = ready && test.cuda_ok(reference_fp32.allocate(token_count * kRows),
                                label + " allocate reference FP32");
  ready = ready && test.cuda_ok(reference_bf16.allocate(token_count * kRows),
                                label + " allocate reference BF16");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_activations.get(),
                                       activations.data(),
                                       activations.size() * sizeof(activations[0]),
                                       cudaMemcpyHostToDevice, stream),
                       label + " copy activations");
  if (!ready) {
    return;
  }

  std::vector<std::uint8_t> fp8_weights(kRows * kColumns);
  for (std::size_t index = 0U; index < fp8_weights.size(); ++index) {
    fp8_weights[index] =
        kFp8Codes[(index * 7U + 1U) % kFp8Codes.size()];
  }
  std::vector<std::uint16_t> fp8_expected(token_count * kRows);
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::vector<std::uint16_t> token_expected = fp8_host_reference(
        fp8_weights, kFp8Scale,
        activations.data() + token * kColumns, kRows, kColumns);
    std::copy(token_expected.begin(), token_expected.end(),
              fp8_expected.begin() + token * kRows);
  }
  DeviceBuffer<std::uint8_t> device_fp8_weights;
  ready = test.cuda_ok(device_fp8_weights.allocate(fp8_weights.size()),
                       label + " allocate FP8 weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_fp8_weights.get(),
                                       fp8_weights.data(), fp8_weights.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " copy FP8 weights");
  for (std::size_t token = 0U; token < token_count && ready; ++token) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(
            q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
                device_fp8_weights.get(), kFp8Scale,
                device_activations.get() + token * kColumns, kRows, kColumns,
                baseline.get() + token * kRows, static_cast<void*>(stream))),
        label + " repeated FP8 launch");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(
                             q3x::kernels::launch_fp8_gemv_reference_cuda(
                                 device_fp8_weights.get(), kFp8Scale,
                                 device_activations.get() + token * kColumns,
                                 kRows, kColumns,
                                 reference_fp32.get() + token * kRows,
                                 static_cast<void*>(stream))),
                         label + " FP8 CUDA reference launch");
  }
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                               reference_fp32.get(), token_count * kRows,
                               reference_bf16.get(),
                               static_cast<void*>(stream))),
                       label + " FP8 CUDA reference conversion");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
                               device_fp8_weights.get(), kFp8Scale,
                               device_activations.get(), token_count, kRows,
                               kColumns, batched.get(),
                               static_cast<void*>(stream))),
                       label + " batched FP8 launch");
  std::vector<std::uint16_t> fp8_baseline(token_count * kRows);
  std::vector<std::uint16_t> fp8_batched(token_count * kRows);
  std::vector<std::uint16_t> fp8_repeated(token_count * kRows);
  std::vector<std::uint16_t> fp8_reference(token_count * kRows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(fp8_baseline.data(), baseline.get(),
                                       fp8_baseline.size() * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy FP8 baseline");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(fp8_batched.data(), batched.get(),
                                       fp8_batched.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost, stream),
                       label + " copy FP8 batched");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
                               device_fp8_weights.get(), kFp8Scale,
                               device_activations.get(), token_count, kRows,
                               kColumns, batched.get(),
                               static_cast<void*>(stream))),
                       label + " repeat batched FP8 launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(fp8_repeated.data(), batched.get(),
                                       fp8_repeated.size() *
                                           sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy repeated FP8 batched");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(fp8_reference.data(),
                                       reference_bf16.get(),
                                       fp8_reference.size() *
                                           sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy FP8 CUDA reference");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize FP8");
  if (ready) {
    compare_bf16_outputs(test, fp8_batched, fp8_expected, kColumns,
                         label + " FP8 host reference");
    compare_cuda_reference_outputs(test, fp8_batched, fp8_reference,
                                   kColumns,
                                   label + " FP8 small-M CUDA reference");
    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < fp8_baseline.size(); ++index) {
      mismatches += fp8_baseline[index] != fp8_batched[index] ? 1U : 0U;
    }
    std::cout << "SMALL_M_DIFF: FP8 M" << token_count
              << " mixed production-K bf16="
              << mismatches << '/' << fp8_baseline.size() << '\n';
    test.expect(mismatches == 0U,
                label + " FP8 matches repeated production M1 bits");
    test.expect(fp8_batched == fp8_repeated,
                label + " FP8 batched result is bitwise deterministic");
  }

  const std::size_t packed_count = kRows * kColumns / 2U;
  const std::size_t scale_count = kRows * kColumns / 16U;
  std::vector<std::uint8_t> packed(packed_count);
  std::vector<std::uint8_t> scales(scale_count);
  for (std::size_t index = 0U; index < packed.size(); ++index) {
    packed[index] = static_cast<std::uint8_t>(
        ((index * 3U) & 0x0fU) | (((index * 11U + 5U) & 0x0fU) << 4U));
  }
  for (std::size_t index = 0U; index < scales.size(); ++index) {
    scales[index] = static_cast<std::uint8_t>(0x20U + (index % 8U) * 4U);
  }
  std::vector<std::uint16_t> nvfp4_expected(token_count * kRows);
  for (std::size_t token = 0U; token < token_count; ++token) {
    const std::vector<std::uint16_t> token_expected = nvfp4_host_reference(
        packed, scales, kNvFp4Scale,
        activations.data() + token * kColumns, kRows, kColumns);
    std::copy(token_expected.begin(), token_expected.end(),
              nvfp4_expected.begin() + token * kRows);
  }
  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  ready = test.cuda_ok(device_packed.allocate(packed.size()),
                       label + " allocate NVFP4 weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate NVFP4 scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_packed.get(), packed.data(),
                                       packed.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy NVFP4 weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_scales.get(), scales.data(),
                                       scales.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy NVFP4 scales");
  for (std::size_t token = 0U; token < token_count && ready; ++token) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(
            q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
                device_packed.get(), device_scales.get(), kNvFp4Scale,
                device_activations.get() + token * kColumns, kRows, kColumns,
                baseline.get() + token * kRows, static_cast<void*>(stream))),
        label + " repeated NVFP4 launch");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(
                             q3x::kernels::launch_nvfp4_gemv_reference_cuda(
                                 device_packed.get(), device_scales.get(),
                                 kNvFp4Scale,
                                 device_activations.get() + token * kColumns,
                                 kRows, kColumns,
                                 reference_fp32.get() + token * kRows,
                                 static_cast<void*>(stream))),
                         label + " NVFP4 CUDA reference launch");
  }
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                               reference_fp32.get(), token_count * kRows,
                               reference_bf16.get(),
                               static_cast<void*>(stream))),
                       label + " NVFP4 CUDA reference conversion");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
                               device_packed.get(), device_scales.get(),
                               kNvFp4Scale, device_activations.get(),
                               token_count, kRows, kColumns, batched.get(),
                               static_cast<void*>(stream))),
                       label + " batched NVFP4 launch");
  std::vector<std::uint16_t> nvfp4_baseline(token_count * kRows);
  std::vector<std::uint16_t> nvfp4_batched(token_count * kRows);
  std::vector<std::uint16_t> nvfp4_repeated(token_count * kRows);
  std::vector<std::uint16_t> nvfp4_reference(token_count * kRows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(nvfp4_baseline.data(), baseline.get(),
                                       nvfp4_baseline.size() *
                                           sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy NVFP4 baseline");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(nvfp4_batched.data(), batched.get(),
                                       nvfp4_batched.size() *
                                           sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost, stream),
                       label + " copy NVFP4 batched");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
                               device_packed.get(), device_scales.get(),
                               kNvFp4Scale, device_activations.get(),
                               token_count, kRows, kColumns, batched.get(),
                               static_cast<void*>(stream))),
                       label + " repeat batched NVFP4 launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(nvfp4_repeated.data(), batched.get(),
                                       nvfp4_repeated.size() *
                                           sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy repeated NVFP4 batched");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(nvfp4_reference.data(),
                                       reference_bf16.get(),
                                       nvfp4_reference.size() *
                                           sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy NVFP4 CUDA reference");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize NVFP4");
  if (ready) {
    compare_bf16_outputs(test, nvfp4_batched, nvfp4_expected, kColumns,
                         label + " NVFP4 host reference");
    compare_cuda_reference_outputs(test, nvfp4_batched, nvfp4_reference,
                                   kColumns,
                                   label + " NVFP4 small-M CUDA reference");
    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < nvfp4_baseline.size(); ++index) {
      mismatches += nvfp4_baseline[index] != nvfp4_batched[index] ? 1U : 0U;
    }
    std::cout << "SMALL_M_DIFF: NVFP4 M" << token_count
              << " mixed production-K bf16="
              << mismatches << '/' << nvfp4_baseline.size() << '\n';
    test.expect(mismatches == 0U,
                label + " NVFP4 matches repeated production M1 bits");
    test.expect(nvfp4_batched == nvfp4_repeated,
                label + " NVFP4 batched result is bitwise deterministic");
  }
}

void test_launch_validation(TestContext& test) {
  constexpr std::size_t kMaximum = std::numeric_limits<std::size_t>::max();
  auto* const byte_pointer = reinterpret_cast<const std::uint8_t*>(0x1000U);
  auto* const scale_pointer = reinterpret_cast<const std::uint8_t*>(0x4000U);
  auto* const activation =
      reinterpret_cast<const std::uint16_t*>(0x2000U);
  auto* const output = reinterpret_cast<std::uint16_t*>(0x3000U);
  auto* const wrapped_activation = reinterpret_cast<const std::uint16_t*>(
      std::numeric_limits<std::uintptr_t>::max() - 15U);
  auto* const wrapped_scale = reinterpret_cast<const std::uint8_t*>(
      std::numeric_limits<std::uintptr_t>::max());

  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
              nullptr, 1.0F, nullptr, 0U, 37U, nullptr)) == cudaSuccess,
      "FP8 empty shape is a no-op");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
              nullptr, 1.0F, nullptr, kMaximum, 0U, nullptr)) == cudaSuccess,
      "FP8 zero-K shape is a no-op before output-size validation");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
              nullptr, 1.0F, nullptr, 1U, 1U, nullptr)) ==
          cudaErrorInvalidValue,
      "FP8 rejects null non-empty pointers");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
              byte_pointer, std::numeric_limits<float>::quiet_NaN(), activation,
              1U, 1U, output)) == cudaErrorInvalidValue,
      "FP8 rejects NaN scale");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
              byte_pointer, -1.0F, activation, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 rejects negative scale");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
              byte_pointer, 1.0F, activation, kMaximum, 2U, output)) ==
          cudaErrorInvalidValue,
      "FP8 rejects dimension overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
              byte_pointer, 1.0F, activation, 1U, 16U,
              const_cast<std::uint16_t*>(activation))) ==
          cudaErrorInvalidValue,
      "FP8 rejects output/activation overlap");

  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
              nullptr, nullptr, 1.0F, nullptr, 0U, 16U, nullptr)) ==
          cudaSuccess,
      "NVFP4 empty aligned shape is a no-op");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
              nullptr, nullptr, 1.0F, nullptr, kMaximum, 0U, nullptr)) ==
          cudaSuccess,
      "NVFP4 zero-K shape is a no-op before output-size validation");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, 1U, 31U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 rejects non-group-aligned K");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
              nullptr, nullptr, 1.0F, nullptr, 1U, 16U, nullptr)) ==
          cudaErrorInvalidValue,
      "NVFP4 rejects null non-empty pointers");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
              byte_pointer, byte_pointer,
              std::numeric_limits<float>::infinity(), activation, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 rejects infinite scale");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, kMaximum, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 rejects dimension overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, 1U, 16U,
              const_cast<std::uint16_t*>(activation))) ==
          cudaErrorInvalidValue,
      "NVFP4 rejects output/activation overlap");

  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 0U, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 small-M rejects M=0");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 9U, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 small-M rejects M=9");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              nullptr, 1.0F, activation, 2U, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 small-M rejects null weights");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, nullptr, 2U, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 small-M rejects null activations");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 2U, 1U, 1U, nullptr)) ==
          cudaErrorInvalidValue,
      "FP8 small-M rejects null output");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, std::numeric_limits<float>::infinity(), activation,
              2U, 1U, 1U, output)) == cudaErrorInvalidValue,
      "FP8 small-M rejects non-finite scale");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, -1.0F, activation, 2U, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 small-M rejects negative scale");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 8U, 1U,
              kMaximum / 8U + 1U, output)) == cudaErrorInvalidValue,
      "FP8 small-M rejects M*K overflow");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 8U, kMaximum / 8U + 1U, 1U,
              output)) == cudaErrorInvalidValue,
      "FP8 small-M rejects M*N overflow");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 8U, 1U, kMaximum / 8U,
              output)) == cudaErrorInvalidValue,
      "FP8 small-M rejects activation byte-size overflow");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 8U, kMaximum / 8U, 1U,
              output)) == cudaErrorInvalidValue,
      "FP8 small-M rejects output byte-size overflow");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 2U, 1U, 16U,
              reinterpret_cast<std::uint16_t*>(0x2020U))) ==
          cudaErrorInvalidValue,
      "FP8 small-M rejects output overlap with a later activation token");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 2U, 1U, 16U,
              reinterpret_cast<std::uint16_t*>(0x1001U))) ==
          cudaErrorInvalidValue,
      "FP8 small-M rejects output overlap with weights");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              byte_pointer, 1.0F, wrapped_activation, 2U, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "FP8 small-M rejects a wrapping activation address range");

  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, 0U, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects M=0");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, 9U, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects M=9");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              nullptr, byte_pointer, 1.0F, activation, 2U, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects null packed weights");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, nullptr, 1.0F, activation, 2U, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects null block scales");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, nullptr, 2U, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects null activations");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, 2U, 1U, 16U,
              nullptr)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects null output");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, 2U, 1U, 31U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects non-group-aligned K");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer,
              std::numeric_limits<float>::quiet_NaN(), activation, 2U, 1U,
              16U, output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects non-finite scale");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer, -1.0F, activation, 2U, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects negative scale");
  constexpr std::size_t kLargestAlignedEighth =
      (kMaximum / 8U) & ~std::size_t{15U};
  constexpr std::size_t kFirstAlignedPastEighth =
      ((kMaximum / 8U) / 16U + 1U) * 16U;
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 8U, 1U,
              kFirstAlignedPastEighth, output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects M*K overflow");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, 8U, 1U,
              kLargestAlignedEighth, output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects activation byte-size overflow");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, 2U, 1U, 32U,
              reinterpret_cast<std::uint16_t*>(0x2040U))) ==
          cudaErrorInvalidValue,
      "NVFP4 small-M rejects output overlap with a later activation token");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, byte_pointer, 1.0F, activation, 2U, 1U, 16U,
              reinterpret_cast<std::uint16_t*>(0x1001U))) ==
          cudaErrorInvalidValue,
      "NVFP4 small-M rejects output overlap with packed weights");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 2U, 1U, 16U,
              reinterpret_cast<std::uint16_t*>(
                  const_cast<std::uint8_t*>(scale_pointer)))) ==
          cudaErrorInvalidValue,
      "NVFP4 small-M rejects output overlap with block scales");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              byte_pointer, wrapped_scale, 1.0F, activation, 2U, 2U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 small-M rejects a wrapping block-scale address range");

  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              nullptr, 1.0F, nullptr, 8U, 0U, kMaximum, nullptr)) ==
          cudaSuccess,
      "FP8 small-M zero rows ignores huge activation extent");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
              nullptr, 1.0F, nullptr, 8U, kMaximum, 0U, nullptr)) ==
          cudaSuccess,
      "FP8 small-M zero columns ignores huge output extent");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              nullptr, nullptr, 1.0F, nullptr, 8U, 0U,
              kMaximum - 15U, nullptr)) == cudaSuccess,
      "NVFP4 small-M zero rows ignores huge aligned activation extent");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              nullptr, nullptr, 1.0F, nullptr, 8U, kMaximum, 0U,
              nullptr)) == cudaSuccess,
      "NVFP4 small-M zero columns ignores huge output extent");

  test.expect(!q3x::kernels::use_sm87_nvfp4_small_m_row_pair_test(2U,
                                                                  17'408U),
              "NVFP4 M2 keeps the single-row production kernel");
  test.expect(!q3x::kernels::use_sm87_nvfp4_small_m_row_pair_test(8U, 1U),
              "NVFP4 M8 row-pair rejects one output row");
  test.expect(!q3x::kernels::use_sm87_nvfp4_small_m_row_pair_test(8U, 15U),
              "NVFP4 M8 row-pair rejects a partial row-pair block");
  test.expect(q3x::kernels::use_sm87_nvfp4_small_m_row_pair_test(8U, 16U),
              "NVFP4 M8 row-pair accepts one complete row-pair block");
  test.expect(q3x::kernels::use_sm87_nvfp4_small_m_row_pair_test(8U,
                                                                 17'408U),
              "NVFP4 M8 row-pair accepts production output rows");
  test.expect(!q3x::kernels::use_sm87_fp8_small_m_row_pair_test(2U, 10'240U),
              "FP8 M2 keeps the single-row production kernel");
  test.expect(!q3x::kernels::use_sm87_fp8_small_m_row_pair_test(8U, 1'023U),
              "FP8 M8 row-pair keeps a conservative small-row fallback");
  test.expect(q3x::kernels::use_sm87_fp8_small_m_row_pair_test(8U, 1'024U),
              "FP8 M8 row-pair accepts the smallest checkpoint projection");
  test.expect(q3x::kernels::use_sm87_fp8_small_m_row_pair_test(8U, 10'240U),
              "FP8 M8 row-pair accepts production QKV output rows");
}

[[nodiscard]] bool performance_enabled() noexcept {
  const char* value = std::getenv("Q3X_RUN_SM87_WEIGHT_ONLY_GEMV_PERF");
  if (value == nullptr) {
    value = std::getenv("Q3X_SM87_GEMV_PERF");
  }
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool small_m_performance_enabled() noexcept {
  const char* value = std::getenv("Q3X_RUN_SM87_SMALL_M_PERF");
  if (value == nullptr) {
    value = std::getenv("Q3X_RUN_SM87_SMALL_M8_PERF");
  }
  return performance_enabled() ||
         (value != nullptr && value[0] != '\0' &&
          !(value[0] == '0' && value[1] == '\0'));
}

[[nodiscard]] bool nvfp4_row_pair_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_ROW_PAIR_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_row_pair_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_ROW_PAIR_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

using NvFp4Launcher = int (*)(const std::uint8_t*, const std::uint8_t*, float,
                              const std::uint16_t*, std::size_t, std::size_t,
                              std::uint16_t*, void*) noexcept;
using Fp8Launcher = int (*)(const std::uint8_t*, float,
                            const std::uint16_t*, std::size_t, std::size_t,
                            std::uint16_t*, void*) noexcept;

[[nodiscard]] float measure_fp8_launcher(
    TestContext& test, cudaStream_t stream, Fp8Launcher launcher,
    const std::uint8_t* weights, const float weight_scale,
    const std::uint16_t* activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* output, const int iterations,
    const std::string& label) {
  EventPair events;
  bool ready = events.create(test);
  ready = ready && test.cuda_ok(cudaEventRecord(events.start(), stream),
                                label + " record start");
  for (int iteration = 0; iteration < iterations && ready; ++iteration) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(launcher(
            weights, weight_scale, activation, rows, columns, output,
            static_cast<void*>(stream))),
        label + " launch");
  }
  ready = ready && test.cuda_ok(cudaEventRecord(events.stop(), stream),
                                label + " record stop");
  ready = ready && test.cuda_ok(cudaEventSynchronize(events.stop()),
                                label + " event synchronize");
  float total_milliseconds = std::numeric_limits<float>::quiet_NaN();
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds,
                                            events.start(), events.stop()),
                       label + " elapsed time");
  if (!ready) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return total_milliseconds / static_cast<float>(iterations);
}

[[nodiscard]] float measure_nvfp4_launcher(
    TestContext& test, cudaStream_t stream, NvFp4Launcher launcher,
    const std::uint8_t* packed, const std::uint8_t* scales,
    const float weight_scale_2, const std::uint16_t* activation,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* output, const int iterations, const std::string& label) {
  EventPair events;
  bool ready = events.create(test);
  ready = ready && test.cuda_ok(cudaEventRecord(events.start(), stream),
                                label + " record start");
  for (int iteration = 0; iteration < iterations && ready; ++iteration) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(launcher(
            packed, scales, weight_scale_2, activation, rows, columns, output,
            static_cast<void*>(stream))),
        label + " launch");
  }
  ready = ready && test.cuda_ok(cudaEventRecord(events.stop(), stream),
                                label + " record stop");
  ready = ready && test.cuda_ok(cudaEventSynchronize(events.stop()),
                                label + " event synchronize");
  float total_milliseconds = std::numeric_limits<float>::quiet_NaN();
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds,
                                            events.start(), events.stop()),
                       label + " elapsed time");
  if (!ready) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return total_milliseconds / static_cast<float>(iterations);
}

template <typename LaunchTile>
[[nodiscard]] float measure_small_m_tile(
    TestContext& test, cudaStream_t stream, LaunchTile&& launch_tile,
    const int iterations, const std::string& label) {
  EventPair events;
  bool ready = events.create(test);
  ready = ready && test.cuda_ok(cudaEventRecord(events.start(), stream),
                                label + " record start");
  for (int iteration = 0; iteration < iterations && ready; ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_tile()),
                         label + " launch tile");
  }
  ready = ready && test.cuda_ok(cudaEventRecord(events.stop(), stream),
                                label + " record stop");
  ready = ready && test.cuda_ok(cudaEventSynchronize(events.stop()),
                                label + " event synchronize");
  float total_milliseconds = std::numeric_limits<float>::quiet_NaN();
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds,
                                            events.start(), events.stop()),
                       label + " elapsed time");
  if (!ready) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return total_milliseconds / static_cast<float>(iterations);
}

struct SmallMMeasurement {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float batched_milliseconds = std::numeric_limits<float>::quiet_NaN();
};

[[nodiscard]] SmallMMeasurement report_small_m_performance(
    TestContext& test, const std::string& label,
    const std::size_t token_count, const float required_speedup,
    const float baseline_first, const float batched_first,
    const float batched_second, const float baseline_second) {
  if (!(std::isfinite(baseline_first) && std::isfinite(baseline_second) &&
        std::isfinite(batched_first) && std::isfinite(batched_second))) {
    return {};
  }
  const float baseline_average =
      (baseline_first + baseline_second) * 0.5F;
  const float batched_average = (batched_first + batched_second) * 0.5F;
  const float speedup = baseline_average / batched_average;
  const bool gate_passed = speedup >= required_speedup;
  std::cout << "PERF_SMALL_M: " << label << " M=" << token_count
            << " baseline_mx_m1_ms=" << baseline_average
            << " batched_ms=" << batched_average
            << " speedup=" << speedup
            << " required_speedup=" << required_speedup
            << " gate=" << (gate_passed ? "PASS" : "FAIL") << '\n';
  test.expect(gate_passed, label + " small-M per-shape gate must pass");
  return {baseline_average, batched_average};
}

struct Fp8RowPairMeasurement {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
};

[[nodiscard]] Fp8RowPairMeasurement benchmark_fp8_row_pair_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 8U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 40;
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 12U> kFiniteCodes{{
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U,
      0x3cU, 0x40U, 0xb8U, 0x70U, 0x78U, 0xfeU,
  }};
  std::vector<std::uint8_t> host_weights(rows * columns);
  for (std::size_t index = 0U; index < host_weights.size(); ++index) {
    host_weights[index] =
        kFiniteCodes[(index * 7U + index / columns * 5U + 1U) %
                     kFiniteCodes.size()];
  }
  std::vector<std::uint16_t> host_activations(kTokens * columns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered =
          static_cast<int>((column * 13U + token * 17U + 3U) % 127U) - 63;
      host_activations[token * columns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
  }

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(weights.get(), host_weights.data(),
                                       host_weights.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize mixed finite weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize mixed activations");
  if (!ready) {
    return {};
  }

  const auto launch_baseline = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_fp8_w8a16_small_m8_single_row_test_cuda(
            weights.get(), kWeightScale, activations.get(), rows, columns,
            baseline_output.get(), static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() noexcept -> int {
    return q3x::kernels::launch_sm87_fp8_w8a16_small_m8_row_pair_test_cuda(
        weights.get(), kWeightScale, activations.get(), rows, columns,
        candidate_output.get(), static_cast<void*>(stream));
  };

  ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                       label + " correctness baseline launch");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate()),
                       label + " correctness candidate launch");
  std::vector<std::uint16_t> baseline(kTokens * rows);
  std::vector<std::uint16_t> candidate(kTokens * rows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), baseline_output.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy baseline output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           candidate.data(), candidate_output.get(),
                           candidate.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy candidate output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " correctness synchronize");
  if (!ready) {
    return {};
  }
  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    mismatches += baseline[index] != candidate[index] ? 1U : 0U;
  }
  const bool bitwise_equal = mismatches == 0U;
  std::cout << "FP8_ROW_PAIR_SHAPE_DIFF: " << label
            << " candidate_vs_single_row_m8_bf16=" << mismatches << '/'
            << baseline.size() << '\n';
  test.expect(bitwise_equal,
              label + " row-pair candidate matches every baseline BF16 bit");

  for (int iteration = 0; iteration < kWarmupIterations && ready;
       ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         label + " baseline warmup");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         label + " candidate warmup");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  if (!ready) {
    return {};
  }

  const float baseline_first = measure_small_m_tile(
      test, stream, launch_baseline, kMeasuredIterations,
      label + " baseline pass 1");
  const float candidate_first = measure_small_m_tile(
      test, stream, launch_candidate, kMeasuredIterations,
      label + " candidate pass 1");
  const float candidate_second = measure_small_m_tile(
      test, stream, launch_candidate, kMeasuredIterations,
      label + " candidate pass 2");
  const float baseline_second = measure_small_m_tile(
      test, stream, launch_baseline, kMeasuredIterations,
      label + " baseline pass 2");
  if (!(std::isfinite(baseline_first) &&
        std::isfinite(baseline_second) &&
        std::isfinite(candidate_first) &&
        std::isfinite(candidate_second))) {
    return {};
  }
  const float baseline_average =
      (baseline_first + baseline_second) * 0.5F;
  const float candidate_average =
      (candidate_first + candidate_second) * 0.5F;
  const float speedup = baseline_average / candidate_average;
  std::cout << "PERF_FP8_ROW_PAIR: " << label << " M=" << kTokens
            << " baseline_pass1_ms=" << baseline_first
            << " candidate_pass1_ms=" << candidate_first
            << " candidate_pass2_ms=" << candidate_second
            << " baseline_pass2_ms=" << baseline_second
            << " baseline_average_ms=" << baseline_average
            << " candidate_average_ms=" << candidate_average
            << " speedup=" << speedup
            << " uplift_percent=" << (speedup - 1.0F) * 100.0F << '\n';
  return {baseline_average, candidate_average, bitwise_equal};
}

void run_optional_fp8_row_pair_performance(TestContext& test,
                                            cudaStream_t stream) {
  if (!fp8_row_pair_performance_enabled()) {
    std::cout << "SKIP: FP8 M8 row-pair performance segment; set "
                 "Q3X_RUN_SM87_FP8_ROW_PAIR_PERF=1 to enable\n";
    return;
  }
  constexpr float kMinimumRequiredSpeedup = 1.03F;
  struct Shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t calls_per_prompt;
    const char* label;
  };
  constexpr std::array<Shape, 5U> kShapes{{
      {10'240U, 5'120U, 48U, "FP8 linear QKV 10240x5120"},
      {5'120U, 6'144U, 64U, "FP8 projection 5120x6144"},
      {6'144U, 5'120U, 48U, "FP8 projection 6144x5120"},
      {12'288U, 5'120U, 16U, "FP8 linear QKV 12288x5120"},
      {1'024U, 5'120U, 32U, "FP8 small projection 1024x5120"},
  }};
  std::array<Fp8RowPairMeasurement, kShapes.size()> measurements{};
  measurements[0] = benchmark_fp8_row_pair_shape(
      test, stream, kShapes[0].rows, kShapes[0].columns, kShapes[0].label);
  measurements[1] = benchmark_fp8_row_pair_shape(
      test, stream, kShapes[1].rows, kShapes[1].columns, kShapes[1].label);
  const auto passes_core_gate = [&](const std::size_t index) noexcept {
    const Fp8RowPairMeasurement& measurement = measurements[index];
    return measurement.bitwise_equal &&
           std::isfinite(measurement.baseline_milliseconds) &&
           std::isfinite(measurement.candidate_milliseconds) &&
           measurement.baseline_milliseconds /
                   measurement.candidate_milliseconds >=
               kMinimumRequiredSpeedup;
  };
  const bool core_gate = passes_core_gate(0U) && passes_core_gate(1U);
  test.expect(core_gate,
              "FP8 M8 row-pair both core shapes improve by at least 3%");
  if (!core_gate) {
    std::cout << "PERF_FP8_ROW_PAIR_CORE_GATE: gate=FAIL\n";
    return;
  }
  std::cout << "PERF_FP8_ROW_PAIR_CORE_GATE: gate=PASS\n";

  for (std::size_t index = 2U; index < kShapes.size(); ++index) {
    measurements[index] = benchmark_fp8_row_pair_shape(
        test, stream, kShapes[index].rows, kShapes[index].columns,
        kShapes[index].label);
  }
  double weighted_baseline = 0.0;
  double weighted_candidate = 0.0;
  bool all_correct = true;
  bool all_finite = true;
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    const Fp8RowPairMeasurement& measurement = measurements[index];
    all_correct = all_correct && measurement.bitwise_equal;
    all_finite = all_finite &&
                 std::isfinite(measurement.baseline_milliseconds) &&
                 std::isfinite(measurement.candidate_milliseconds);
    weighted_baseline += static_cast<double>(kShapes[index].calls_per_prompt) *
                         measurement.baseline_milliseconds;
    weighted_candidate +=
        static_cast<double>(kShapes[index].calls_per_prompt) *
        measurement.candidate_milliseconds;
  }
  const double weighted_speedup = weighted_baseline / weighted_candidate;
  const bool aggregate_gate =
      all_correct && all_finite && std::isfinite(weighted_speedup) &&
      weighted_speedup >= kMinimumRequiredSpeedup;
  std::cout << "PERF_FP8_ROW_PAIR_AGGREGATE: weighted_baseline_ms="
            << weighted_baseline
            << " weighted_candidate_ms=" << weighted_candidate
            << " speedup=" << weighted_speedup
            << " required_speedup=" << kMinimumRequiredSpeedup
            << " all_bitwise_equal=" << (all_correct ? "true" : "false")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "FP8 M8 row-pair five-shape weighted gate must pass");
}

[[nodiscard]] bool benchmark_nvfp4_row_pair_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 8U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 40;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr float kMinimumRequiredSpeedup = 1.03F;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_count = rows * columns / 16U;
  std::vector<std::uint16_t> host_activations(
      kTokens * columns, encode_bf16(1.0F));
  std::vector<std::uint8_t> host_scales(scale_count);
  const std::size_t scale_columns = columns / 16U;
  constexpr std::array<std::uint8_t, 3U> kSameBankScaleCodes{{
      0x20U, 0x40U, 0x60U,
  }};
  for (std::size_t index = 0U; index < host_scales.size(); ++index) {
    const std::size_t row = index / scale_columns;
    // Three distinct positive normal codes have the same low five bits, hence
    // occupy one shared bank. This makes each half-warp lookup three distinct
    // transactions and prevents a uniform-address broadcast from masking the
    // candidate's bank-conflict cost.
    host_scales[index] =
        kSameBankScaleCodes[(index + row * 3U) % kSameBankScaleCodes.size()];
  }
  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(scale_count),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(output.allocate(kTokens * rows),
                                label + " allocate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21, packed_count,
                                       stream),
                       label + " initialize packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(scales.get(), host_scales.data(),
                                       host_scales.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize mixed block scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return false;
  }

  const auto launch_baseline = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_small_m8_single_row_test_cuda(
            packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
            columns, output.get(), static_cast<void*>(stream));
  };
  const auto launch_row_pair = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_small_m8_row_pair_test_cuda(
            packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
            columns, output.get(), static_cast<void*>(stream));
  };
  const auto launch_scale_codebook = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_small_m8_scale_codebook_test_cuda(
            packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
            columns, output.get(), static_cast<void*>(stream));
  };
  for (int iteration = 0; iteration < kWarmupIterations && ready;
       ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         label + " baseline warmup");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_row_pair()),
                         label + " row-pair warmup");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_scale_codebook()),
                         label + " scale-codebook warmup");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  if (!ready) {
    return false;
  }

  // Mirrored B/R/R/B event order limits systematic temperature and clock
  // bias while retaining separate event intervals for each implementation.
  const float baseline_first = measure_small_m_tile(
      test, stream, launch_baseline, kMeasuredIterations,
      label + " baseline pass 1");
  const float row_pair_first = measure_small_m_tile(
      test, stream, launch_row_pair, kMeasuredIterations,
      label + " row-pair pass 1");
  const float row_pair_second = measure_small_m_tile(
      test, stream, launch_row_pair, kMeasuredIterations,
      label + " row-pair pass 2");
  const float baseline_second = measure_small_m_tile(
      test, stream, launch_baseline, kMeasuredIterations,
      label + " baseline pass 2");
  const float scale_baseline_first = measure_small_m_tile(
      test, stream, launch_row_pair, kMeasuredIterations,
      label + " scale baseline pass 1");
  const float scale_candidate_first = measure_small_m_tile(
      test, stream, launch_scale_codebook, kMeasuredIterations,
      label + " production scale-codebook pass 1");
  const float scale_candidate_second = measure_small_m_tile(
      test, stream, launch_scale_codebook, kMeasuredIterations,
      label + " production scale-codebook pass 2");
  const float scale_baseline_second = measure_small_m_tile(
      test, stream, launch_row_pair, kMeasuredIterations,
      label + " scale baseline pass 2");
  if (!(std::isfinite(baseline_first) &&
        std::isfinite(baseline_second) &&
        std::isfinite(row_pair_first) &&
        std::isfinite(row_pair_second) &&
        std::isfinite(scale_baseline_first) &&
        std::isfinite(scale_baseline_second) &&
        std::isfinite(scale_candidate_first) &&
        std::isfinite(scale_candidate_second))) {
    return false;
  }

  const float baseline_average =
      (baseline_first + baseline_second) * 0.5F;
  const float row_pair_average =
      (row_pair_first + row_pair_second) * 0.5F;
  const float speedup = baseline_average / row_pair_average;
  const float uplift_percent = (speedup - 1.0F) * 100.0F;
  const double encoded_gigabytes =
      static_cast<double>(packed_count + scale_count) / 1.0e9;
  const double baseline_gigabytes_per_second =
      encoded_gigabytes / (static_cast<double>(baseline_average) / 1.0e3);
  const double row_pair_gigabytes_per_second =
      encoded_gigabytes / (static_cast<double>(row_pair_average) / 1.0e3);
  const bool gate_passed = speedup >= kMinimumRequiredSpeedup;
  std::cout << "PERF_NVFP4_ROW_PAIR: " << label
            << " M=" << kTokens
            << " baseline_single_row_ms=" << baseline_average
            << " preserved_row_pair_ms=" << row_pair_average
            << " speedup=" << speedup
            << " uplift_percent=" << uplift_percent
            << " baseline_encoded_weight_GBps="
            << baseline_gigabytes_per_second
            << " row_pair_encoded_weight_GBps="
            << row_pair_gigabytes_per_second
            << " required_speedup=" << kMinimumRequiredSpeedup
            << " gate=" << (gate_passed ? "PASS" : "FAIL") << '\n';
  test.expect(gate_passed,
              label + " preserved row-pair must improve by at least 3%");
  const float scale_baseline_average =
      (scale_baseline_first + scale_baseline_second) * 0.5F;
  const float scale_candidate_average =
      (scale_candidate_first + scale_candidate_second) * 0.5F;
  const float scale_speedup =
      scale_baseline_average / scale_candidate_average;
  const float scale_uplift_percent = (scale_speedup - 1.0F) * 100.0F;
  const bool scale_gate_passed =
      scale_speedup >= kMinimumRequiredSpeedup;
  std::cout << "PERF_NVFP4_SCALE_CODEBOOK: " << label
            << " M=" << kTokens
            << " same_bank_scale_codes=true"
            << " baseline_row_pair_ms=" << scale_baseline_average
            << " production_shared_scale_codebook_ms="
            << scale_candidate_average
            << " speedup=" << scale_speedup
            << " uplift_percent=" << scale_uplift_percent
            << " required_speedup=" << kMinimumRequiredSpeedup
            << " gate=" << (scale_gate_passed ? "PASS" : "FAIL") << '\n';
  test.expect(scale_gate_passed,
              label + " production scale codebook must improve by at least 3%");
  return gate_passed && scale_gate_passed;
}

void run_optional_nvfp4_row_pair_performance(TestContext& test,
                                              cudaStream_t stream) {
  if (!nvfp4_row_pair_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M8 row-pair performance segment; set "
                 "Q3X_RUN_SM87_NVFP4_ROW_PAIR_PERF=1 to enable\n";
    return;
  }
  const bool gate_up = benchmark_nvfp4_row_pair_shape(
      test, stream, 17'408U, 5'120U,
      "NVFP4 MLP gate/up 17408x5120");
  const bool down = benchmark_nvfp4_row_pair_shape(
      test, stream, 5'120U, 17'408U,
      "NVFP4 MLP down 5120x17408");
  std::cout << "PERF_NVFP4_ROW_PAIR_AGGREGATE: "
               "row_pair_and_scale_codebook_both_shapes_over_3_percent="
            << (gate_up && down ? "PASS" : "FAIL") << '\n';
}

[[nodiscard]] SmallMMeasurement benchmark_fp8_small_m_shape(
    TestContext& test, cudaStream_t stream, const std::size_t token_count,
    const std::size_t rows, const std::size_t columns,
    const float required_speedup, const std::string& label,
    const bool mixed_weights = false) {
  constexpr int kWarmupIterations = 5;
  constexpr int kMeasuredIterations = 20;
  constexpr float kWeightScale = 1.0F / 64.0F;

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> output;
  std::vector<std::uint16_t> host_activations(
      token_count * columns, encode_bf16(1.0F));
  bool ready = test.cuda_ok(weights.allocate(rows * columns),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(output.allocate(token_count * rows),
                                label + " allocate output");
  if (mixed_weights) {
    constexpr std::array<std::uint8_t, 12U> kFiniteCodes{{
        0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U,
        0x3cU, 0x40U, 0xb8U, 0x70U, 0x78U, 0xfeU,
    }};
    std::vector<std::uint8_t> host_weights(rows * columns);
    for (std::size_t index = 0U; index < host_weights.size(); ++index) {
      host_weights[index] = kFiniteCodes[(index * 7U + 1U) %
                                         kFiniteCodes.size()];
    }
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(weights.get(), host_weights.data(),
                                         host_weights.size(),
                                         cudaMemcpyHostToDevice, stream),
                         label + " initialize mixed weights");
  } else {
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(weights.get(), 0x38, rows * columns,
                                         stream),
                         label + " initialize weights");
  }
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return {};
  }

  const auto launch_baseline = [&]() noexcept -> int {
    for (std::size_t token = 0U; token < token_count; ++token) {
      const int status = q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
          weights.get(), kWeightScale,
          activations.get() + token * columns, rows, columns,
          output.get() + token * rows, static_cast<void*>(stream));
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  };
  const auto launch_batched = [&]() noexcept -> int {
    return q3x::kernels::launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
        weights.get(), kWeightScale, activations.get(), token_count, rows,
        columns, output.get(), static_cast<void*>(stream));
  };
  for (int iteration = 0; iteration < kWarmupIterations && ready;
       ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         label + " baseline warmup");
    ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_batched()),
                                  label + " batched warmup");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  if (!ready) {
    return {};
  }

  const float baseline_first = measure_small_m_tile(
      test, stream, launch_baseline, kMeasuredIterations,
      label + " baseline pass 1");
  const float batched_first = measure_small_m_tile(
      test, stream, launch_batched, kMeasuredIterations,
      label + " batched pass 1");
  const float batched_second = measure_small_m_tile(
      test, stream, launch_batched, kMeasuredIterations,
      label + " batched pass 2");
  const float baseline_second = measure_small_m_tile(
      test, stream, launch_baseline, kMeasuredIterations,
      label + " baseline pass 2");
  return report_small_m_performance(
      test, label, token_count, required_speedup, baseline_first,
      batched_first, batched_second, baseline_second);
}

[[nodiscard]] SmallMMeasurement benchmark_nvfp4_small_m_shape(
    TestContext& test, cudaStream_t stream, const std::size_t token_count,
    const std::size_t rows, const std::size_t columns,
    const float required_speedup, const std::string& label) {
  constexpr int kWarmupIterations = 5;
  constexpr int kMeasuredIterations = 20;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_count = rows * columns / 16U;

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> output;
  std::vector<std::uint16_t> host_activations(
      token_count * columns, encode_bf16(1.0F));
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(scale_count),
                                label + " allocate scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(output.allocate(token_count * rows),
                                label + " allocate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21, packed_count,
                                       stream),
                       label + " initialize packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(scales.get(), 0x38, scale_count,
                                       stream),
                       label + " initialize scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return {};
  }

  const auto launch_baseline = [&]() noexcept -> int {
    for (std::size_t token = 0U; token < token_count; ++token) {
      const int status =
          q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
              packed.get(), scales.get(), kWeightScale2,
              activations.get() + token * columns, rows, columns,
              output.get() + token * rows, static_cast<void*>(stream));
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  };
  const auto launch_batched = [&]() noexcept -> int {
    return q3x::kernels::launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
        packed.get(), scales.get(), kWeightScale2, activations.get(),
        token_count, rows, columns, output.get(), static_cast<void*>(stream));
  };
  for (int iteration = 0; iteration < kWarmupIterations && ready;
       ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         label + " baseline warmup");
    ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_batched()),
                                  label + " batched warmup");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  if (!ready) {
    return {};
  }

  const float baseline_first = measure_small_m_tile(
      test, stream, launch_baseline, kMeasuredIterations,
      label + " baseline pass 1");
  const float batched_first = measure_small_m_tile(
      test, stream, launch_batched, kMeasuredIterations,
      label + " batched pass 1");
  const float batched_second = measure_small_m_tile(
      test, stream, launch_batched, kMeasuredIterations,
      label + " batched pass 2");
  const float baseline_second = measure_small_m_tile(
      test, stream, launch_baseline, kMeasuredIterations,
      label + " baseline pass 2");
  return report_small_m_performance(
      test, label, token_count, required_speedup, baseline_first,
      batched_first, batched_second, baseline_second);
}

void run_optional_small_m_performance(TestContext& test,
                                      cudaStream_t stream) {
  if (!small_m_performance_enabled()) {
    std::cout << "SKIP: small-M production-shape performance segment; set "
                 "Q3X_RUN_SM87_SMALL_M_PERF=1 to enable\n";
    return;
  }

  struct ProjectionShape {
    std::size_t rows;
    std::size_t columns;
    std::size_t calls_per_prompt;
    float required_speedup;
    const char* label;
  };
  constexpr std::array<ProjectionShape, 5U> kFp8Shapes{{
      {10'240U, 5'120U, 48U, 1.15F, "linear QKV 10240x5120"},
      {5'120U, 6'144U, 64U, 1.15F, "projection 5120x6144"},
      {6'144U, 5'120U, 48U, 1.15F, "projection 6144x5120"},
      {12'288U, 5'120U, 16U, 1.15F, "linear QKV 12288x5120"},
      {1'024U, 5'120U, 32U, 1.0F / 1.02F,
       "small projection 1024x5120"},
  }};
  constexpr std::array<ProjectionShape, 2U> kNvFp4Shapes{{
      {17'408U, 5'120U, 128U, 1.15F, "MLP gate/up 17408x5120"},
      {5'120U, 17'408U, 64U, 1.15F, "MLP down 5120x17408"},
  }};
  constexpr std::array<std::size_t, 3U> kTokenCounts{{2U, 4U, 8U}};
  // M1 and small-M kernels are optimized independently. Keep the M8 gate
  // below the historical ratio so codebook caching in the M1 denominator
  // cannot turn an absolute small-M improvement into a false regression.
  constexpr std::array<float, 3U> kAggregateRequired{{1.5F, 2.5F, 2.75F}};

  for (std::size_t token_index = 0U; token_index < kTokenCounts.size();
       ++token_index) {
    const std::size_t token_count = kTokenCounts[token_index];
    double weighted_baseline = 0.0;
    double weighted_batched = 0.0;
    bool measurements_finite = true;
    for (const ProjectionShape& shape : kFp8Shapes) {
      const SmallMMeasurement measurement = benchmark_fp8_small_m_shape(
          test, stream, token_count, shape.rows, shape.columns,
          shape.required_speedup,
          "FP8 " + std::string(shape.label));
      measurements_finite =
          measurements_finite &&
          std::isfinite(measurement.baseline_milliseconds) &&
          std::isfinite(measurement.batched_milliseconds);
      weighted_baseline += static_cast<double>(shape.calls_per_prompt) *
                           measurement.baseline_milliseconds;
      weighted_batched += static_cast<double>(shape.calls_per_prompt) *
                          measurement.batched_milliseconds;
    }
    for (const ProjectionShape& shape : kNvFp4Shapes) {
      const SmallMMeasurement measurement = benchmark_nvfp4_small_m_shape(
          test, stream, token_count, shape.rows, shape.columns,
          shape.required_speedup,
          "NVFP4 " + std::string(shape.label));
      measurements_finite =
          measurements_finite &&
          std::isfinite(measurement.baseline_milliseconds) &&
          std::isfinite(measurement.batched_milliseconds);
      weighted_baseline += static_cast<double>(shape.calls_per_prompt) *
                           measurement.baseline_milliseconds;
      weighted_batched += static_cast<double>(shape.calls_per_prompt) *
                          measurement.batched_milliseconds;
    }
    const double aggregate_speedup = weighted_baseline / weighted_batched;
    const bool aggregate_passed =
        measurements_finite && std::isfinite(aggregate_speedup) &&
        aggregate_speedup >= kAggregateRequired[token_index];
    std::cout << "PERF_SMALL_M_AGGREGATE: M=" << token_count
              << " weighted_baseline_ms=" << weighted_baseline
              << " weighted_batched_ms=" << weighted_batched
              << " speedup=" << aggregate_speedup
              << " required_speedup=" << kAggregateRequired[token_index]
              << " gate=" << (aggregate_passed ? "PASS" : "FAIL") << '\n';
    test.expect(aggregate_passed,
                "small-M realistic-call aggregate gate must pass for M=" +
                    std::to_string(token_count));
  }

  (void)benchmark_fp8_small_m_shape(
      test, stream, 8U, 10'240U, 5'120U, 1.15F,
      "FP8 linear QKV 10240x5120 mixed finite", true);
}

void benchmark_nvfp4_shape(TestContext& test, cudaStream_t stream,
                           const std::size_t rows,
                           const std::size_t columns,
                           const std::string& label) {
  constexpr int kWarmupIterations = 5;
  constexpr int kMeasuredIterations = 20;
  constexpr float kMinimumRequiredSpeedup = 1.15F;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_count = rows * columns / 16U;
  std::vector<std::uint16_t> activation(columns, encode_bf16(1.0F));
  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed");
  ready = ready &&
          test.cuda_ok(scales.allocate(scale_count), label + " allocate scales");
  ready = ready && test.cuda_ok(device_activation.allocate(columns),
                                label + " allocate activation");
  ready = ready &&
          test.cuda_ok(output.allocate(rows), label + " allocate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21, packed_count, stream),
                       label + " initialize packed");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(scales.get(), 0x38, scale_count, stream),
                       label + " initialize scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_activation.get(),
                                       activation.data(),
                                       activation.size() * sizeof(activation[0]),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  if (!ready) {
    return;
  }

  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(q3x::kernels::
                             launch_sm87_nvfp4_w4a16_gemv_bf16_scalar_test_cuda(
                                 packed.get(), scales.get(), kWeightScale2,
                                 device_activation.get(), rows, columns,
                                 output.get(), static_cast<void*>(stream))),
                         label + " scalar warmup launch");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(q3x::kernels::
                             launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
                                 packed.get(), scales.get(), kWeightScale2,
                                 device_activation.get(), rows, columns,
                                 output.get(), static_cast<void*>(stream))),
                         label + " vector warmup launch");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  if (!ready) {
    return;
  }

  // Measure in mirrored order to reduce systematic clock/temperature bias.
  const float scalar_first = measure_nvfp4_launcher(
      test, stream,
      q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_scalar_test_cuda,
      packed.get(), scales.get(), kWeightScale2, device_activation.get(), rows,
      columns, output.get(), kMeasuredIterations, label + " scalar pass 1");
  const float vector_first = measure_nvfp4_launcher(
      test, stream, q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda,
      packed.get(), scales.get(), kWeightScale2, device_activation.get(), rows,
      columns, output.get(), kMeasuredIterations, label + " vector pass 1");
  const float vector_second = measure_nvfp4_launcher(
      test, stream, q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda,
      packed.get(), scales.get(), kWeightScale2, device_activation.get(), rows,
      columns, output.get(), kMeasuredIterations, label + " vector pass 2");
  const float scalar_second = measure_nvfp4_launcher(
      test, stream,
      q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_scalar_test_cuda,
      packed.get(), scales.get(), kWeightScale2, device_activation.get(), rows,
      columns, output.get(), kMeasuredIterations, label + " scalar pass 2");
  if (!(std::isfinite(scalar_first) && std::isfinite(scalar_second) &&
        std::isfinite(vector_first) && std::isfinite(vector_second))) {
    return;
  }

  const float scalar_average = (scalar_first + scalar_second) * 0.5F;
  const float vector_average = (vector_first + vector_second) * 0.5F;
  const float speedup = scalar_average / vector_average;
  const double encoded_gigabytes =
      static_cast<double>(packed_count + scale_count) / 1.0e9;
  const double vector_gigabytes_per_second =
      encoded_gigabytes / (static_cast<double>(vector_average) / 1.0e3);
  const bool gate_passed = speedup >= kMinimumRequiredSpeedup;
  std::cout << "PERF: " << label << " scalar_average_ms=" << scalar_average
            << " vector_average_ms=" << vector_average
            << " speedup=" << speedup
            << " vector_encoded_weight_GBps="
            << vector_gigabytes_per_second
            << " required_speedup=" << kMinimumRequiredSpeedup
            << " gate=" << (gate_passed ? "PASS" : "FAIL") << '\n';
  test.expect(gate_passed, label + " vector path must be at least 15% faster");
}

void benchmark_fp8_shape(TestContext& test, cudaStream_t stream,
                         const std::size_t rows, const std::size_t columns,
                         const std::string& label,
                         const bool mixed_finite = false) {
  constexpr int kWarmupIterations = 5;
  constexpr int kMeasuredIterations = 20;
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 13U> kFiniteCodes{{
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U, 0x3cU,
      0x40U, 0xb8U, 0x70U, 0x78U, 0x7eU, 0xfeU,
  }};
  const std::size_t weight_count = rows * columns;
  std::vector<std::uint16_t> activation(columns, encode_bf16(1.0F));
  std::vector<std::uint8_t> host_weights;
  if (mixed_finite) {
    host_weights.resize(weight_count);
    for (std::size_t index = 0U; index < weight_count; ++index) {
      host_weights[index] =
          kFiniteCodes[(index * 7U + 1U) % kFiniteCodes.size()];
    }
  }
  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(weights.allocate(weight_count),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(device_activation.allocate(columns),
                                label + " allocate activation");
  ready = ready &&
          test.cuda_ok(output.allocate(rows), label + " allocate output");
  if (mixed_finite) {
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(weights.get(), host_weights.data(),
                                         host_weights.size(),
                                         cudaMemcpyHostToDevice, stream),
                         label + " initialize mixed weights");
  } else {
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(weights.get(), 0x38, weight_count,
                                         stream),
                         label + " initialize uniform weights");
  }
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_activation.get(),
                                       activation.data(),
                                       activation.size() * sizeof(activation[0]),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  if (!ready) {
    return;
  }

  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(
                             q3x::kernels::
                                 launch_sm87_fp8_w8a16_gemv_bf16_scalar_test_cuda(
                                     weights.get(), kWeightScale,
                                     device_activation.get(), rows, columns,
                                     output.get(), static_cast<void*>(stream))),
                         label + " scalar warmup launch");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(
                             q3x::kernels::
                                 launch_sm87_fp8_w8a16_gemv_bf16_cuda(
                                     weights.get(), kWeightScale,
                                     device_activation.get(), rows, columns,
                                     output.get(), static_cast<void*>(stream))),
                         label + " warmup launch");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  if (!ready) {
    return;
  }

  // Mirrored order reduces systematic clock and temperature bias.
  const float scalar_first = measure_fp8_launcher(
      test, stream,
      q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_scalar_test_cuda,
      weights.get(), kWeightScale, device_activation.get(), rows, columns,
      output.get(), kMeasuredIterations, label + " scalar pass 1");
  const float vector_first = measure_fp8_launcher(
      test, stream, q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda,
      weights.get(), kWeightScale, device_activation.get(), rows, columns,
      output.get(), kMeasuredIterations, label + " vector pass 1");
  const float vector_second = measure_fp8_launcher(
      test, stream, q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda,
      weights.get(), kWeightScale, device_activation.get(), rows, columns,
      output.get(), kMeasuredIterations, label + " vector pass 2");
  const float scalar_second = measure_fp8_launcher(
      test, stream,
      q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_scalar_test_cuda,
      weights.get(), kWeightScale, device_activation.get(), rows, columns,
      output.get(), kMeasuredIterations, label + " scalar pass 2");
  if (!(std::isfinite(scalar_first) && std::isfinite(scalar_second) &&
        std::isfinite(vector_first) && std::isfinite(vector_second))) {
    return;
  }

  const float scalar_average = (scalar_first + scalar_second) * 0.5F;
  const float vector_average = (vector_first + vector_second) * 0.5F;
  const float speedup = scalar_average / vector_average;
  const float minimum_required_speedup =
      rows >= 5'120U ? 1.15F : (1.0F / 1.02F);
  const double encoded_gigabytes =
      static_cast<double>(weight_count) / 1.0e9;
  const double gigabytes_per_second =
      encoded_gigabytes / (static_cast<double>(vector_average) / 1.0e3);
  const bool gate_passed = speedup >= minimum_required_speedup;
  std::cout << "PERF: " << label << " scalar_average_ms=" << scalar_average
            << " vector_average_ms=" << vector_average
            << " speedup=" << speedup
            << " vector_encoded_weight_GBps=" << gigabytes_per_second
            << " required_speedup=" << minimum_required_speedup
            << " gate=" << (gate_passed ? "PASS" : "FAIL") << '\n';
  test.expect(gate_passed,
              label + " packed-x4 performance gate must pass");
}

void run_optional_performance(TestContext& test, cudaStream_t stream) {
  if (!performance_enabled()) {
    std::cout << "SKIP: production-shape performance segment; set "
                 "Q3X_RUN_SM87_WEIGHT_ONLY_GEMV_PERF=1 to enable\n";
    return;
  }
  benchmark_fp8_shape(test, stream, 10'240U, 5'120U,
                      "FP8 linear QKV 10240x5120");
  benchmark_fp8_shape(test, stream, 5'120U, 6'144U,
                      "FP8 projection 5120x6144");
  benchmark_fp8_shape(test, stream, 6'144U, 5'120U,
                      "FP8 projection 6144x5120");
  benchmark_fp8_shape(test, stream, 12'288U, 5'120U,
                      "FP8 linear QKV 12288x5120");
  benchmark_fp8_shape(test, stream, 1'024U, 5'120U,
                      "FP8 small projection 1024x5120");
  benchmark_fp8_shape(test, stream, 10'240U, 5'120U,
                      "FP8 linear QKV 10240x5120 mixed finite", true);
  benchmark_nvfp4_shape(test, stream, 17'408U, 5'120U,
                        "NVFP4 MLP gate/up 17408x5120");
  benchmark_nvfp4_shape(test, stream, 5'120U, 17'408U,
                        "NVFP4 MLP down 5120x17408");
}

}  // namespace

int main() {
  TestContext test;
  test_launch_validation(test);

  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: SM87 weight-only GEMV test (no CUDA device)\n";
    (void)cudaGetLastError();
    return test.failures() == 0 ? 77 : 1;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "read CUDA device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: SM87 weight-only GEMV test requires sm_87, found sm_"
              << properties.major << properties.minor << '\n';
    return test.failures() == 0 ? 77 : 1;
  }
  std::cout << "SM87 weight-only GEMV device: " << properties.name << '\n';

  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create non-blocking stream")) {
    return 1;
  }

  for (std::size_t token_count = 2U; token_count <= 8U; ++token_count) {
    run_small_m_production_k_comparison(test, stream, token_count);
  }
  run_fp8_case(test, stream, 3U, 5'120U,
               "FP8 small-M1 production-K delegate", false, false, 1U,
               true);
  run_nvfp4_case(test, stream, 3U, 5'120U,
                 "NVFP4 small-M1 production-K delegate", false, false, 1U,
                 true);
  run_fp8_case(test, stream, 3U, 37U,
               "FP8 small-M3 scalar-K fallback", false, false, 3U, true);
  run_fp8_case(test, stream, 3U, 1'024U,
               "FP8 small-M3 unaligned-weight fallback", true, false, 3U,
               true);
  run_fp8_case(test, stream, 3U, 1'024U,
               "FP8 small-M3 unaligned-activation fallback", false, true,
               3U, true);
  run_fp8_case(test, stream, 3U, 6'144U,
               "FP8 small-M8 multi-loop K6144", false, false, 8U, true);
  run_fp8_case(test, stream, 3U, 17'408U,
               "FP8 small-M8 multi-loop K17408", false, false, 8U, true);
  run_nvfp4_case(test, stream, 3U, 48U,
                 "NVFP4 small-M3 scalar-K fallback", false, false, 3U,
                 true);
  run_nvfp4_case(test, stream, 3U, 256U,
                 "NVFP4 small-M3 unaligned-weight fallback", true, false,
                 3U, true);
  run_nvfp4_case(test, stream, 3U, 256U,
                 "NVFP4 small-M3 unaligned-activation fallback", false,
                 true, 3U, true);
  run_nvfp4_case(test, stream, 3U, 6'144U,
                 "NVFP4 small-M8 multi-loop K6144", false, false, 8U,
                 true);
  run_nvfp4_case(test, stream, 3U, 17'408U,
                 "NVFP4 small-M8 multi-loop K17408", false, false, 8U,
                 true);

  // Awkward rows and K tails exercise partial blocks/warps. Target-K cases
  // cover both fixed MLP reduction lengths without allocating full matrices.
  run_fp8_case(test, stream, 13U, 37U, "FP8 awkward 13x37");
  run_fp8_vector_codebook_case(test, stream);
  run_fp8_row_pair_odd_rows_case(test, stream);
  run_fp8_case(test, stream, 3U, 1'024U,
               "FP8 vector-shaped unaligned-weight scalar fallback", true,
               false);
  run_fp8_case(test, stream, 3U, 1'024U,
               "FP8 vector-shaped unaligned-activation scalar fallback", false,
               true);
  run_fp8_case(test, stream, 3U, 5'120U, "FP8 target-K 3x5120");
  run_fp8_case(test, stream, 3U, 6'144U, "FP8 target-K 3x6144");
  run_fp8_case(test, stream, 3U, 17'408U, "FP8 long-K 3x17408");
  run_nvfp4_case(test, stream, 13U, 48U, "NVFP4 awkward 13x48");
  run_nvfp4_vector_codebook_case(test, stream);
  run_nvfp4_scale_codebook_exhaustive_case(test, stream);
  run_nvfp4_row_pair_bitwise_case(
      test, stream, 17U, 5'120U,
      "NVFP4 M8 row-pair odd rows K5120 row-distinct full E2M1 codebook");
  run_nvfp4_row_pair_bitwise_case(
      test, stream, 19U, 17'408U,
      "NVFP4 M8 row-pair odd rows K17408 row-distinct full E2M1 codebook");
  run_nvfp4_case(test, stream, 3U, 256U,
                  "NVFP4 vector-shaped unaligned scalar fallback", true);
  run_nvfp4_case(test, stream, 3U, 5'120U, "NVFP4 target-K 3x5120");
  run_nvfp4_case(test, stream, 3U, 6'144U, "NVFP4 target-K 3x6144");
  run_nvfp4_case(test, stream, 3U, 17'408U,
                  "NVFP4 target-K 3x17408");
  run_optional_fp8_row_pair_performance(test, stream);
  run_optional_nvfp4_row_pair_performance(test, stream);
  run_optional_small_m_performance(test, stream);
  run_optional_performance(test, stream);

  (void)test.cuda_ok(cudaStreamDestroy(stream),
                     "destroy non-blocking stream");
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " SM87 weight-only GEMV assertion(s) failed\n";
    return 1;
  }
  std::cout << "SM87 weight-only GEMV tests passed\n";
  return 0;
}
