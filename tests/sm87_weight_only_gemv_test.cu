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

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_scalar_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
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
    const std::vector<std::uint16_t>& activation, const std::size_t rows,
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
    const std::vector<std::uint16_t>& activation, const std::size_t rows,
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
                      const bool strict_bf16 = false) {
  const std::vector<std::uint16_t> expected = fp8_host_reference(
      weights, weight_scale, activation, rows, columns);
  std::vector<std::uint16_t> actual(rows, 0U);
  std::vector<std::uint16_t> repeated(rows, 0U);
  std::vector<std::uint16_t> reference(rows, 0U);

  DeviceBuffer<std::uint8_t> device_weights;
  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_output;
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
  ready = ready && test.cuda_ok(device_output.allocate(rows),
                                label + " allocate output");
  ready = ready && test.cuda_ok(device_reference_fp32.allocate(rows),
                                label + " allocate reference FP32");
  ready = ready && test.cuda_ok(device_reference_output.allocate(rows),
                                label + " allocate reference BF16");
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

  (void)seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
              weights_device, weight_scale, activation_device, rows, columns,
              device_output.get(), static_cast<void*>(stream))),
      label + " launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(actual.data(), device_output.get(),
                                       actual.size() * sizeof(actual[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy output");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::kernels::
                               launch_sm87_fp8_w8a16_gemv_bf16_cuda(
                                   weights_device, weight_scale,
                                   activation_device, rows, columns,
                                   device_output.get(),
                                   static_cast<void*>(stream))),
                       label + " repeat launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(repeated.data(), device_output.get(),
                                       repeated.size() * sizeof(repeated[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy repeated output");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::kernels::launch_fp8_gemv_reference_cuda(
                               weights_device, weight_scale,
                               activation_device, rows, columns,
                               device_reference_fp32.get(),
                               static_cast<void*>(stream))),
                       label + " launch CUDA reference");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                               device_reference_fp32.get(), rows,
                               device_reference_output.get(),
                               static_cast<void*>(stream))),
                       label + " convert CUDA reference");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(reference.data(),
                                       device_reference_output.get(),
                                       reference.size() * sizeof(reference[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy CUDA reference");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (ready) {
    compare_bf16_outputs(test, actual, expected, columns, label);
    compare_cuda_reference_outputs(test, actual, reference, columns, label);
    if (strict_bf16) {
      for (std::size_t row = 0U; row < rows; ++row) {
        if (is_bf16_nan(expected[row])) {
          test.expect(is_bf16_nan(actual[row]),
                      label + " strict row " + std::to_string(row) +
                          " host NaN class");
          test.expect(is_bf16_nan(reference[row]),
                      label + " strict row " + std::to_string(row) +
                          " CUDA-reference NaN class");
        } else {
          test.expect(actual[row] == expected[row],
                      label + " strict row " + std::to_string(row) +
                          " must equal host BF16 bits");
          test.expect(actual[row] == reference[row],
                      label + " strict row " + std::to_string(row) +
                          " must equal CUDA-reference BF16 bits");
        }
      }
    }
    test.expect(actual == repeated, label + " is bitwise deterministic");
  }
}

void run_fp8_case(TestContext& test, cudaStream_t stream,
                  const std::size_t rows, const std::size_t columns,
                  const std::string& label,
                  const bool unaligned_weights = false,
                  const bool unaligned_activation = false) {
  constexpr float kWeightScale = 1.0F / 128.0F;
  constexpr std::array<std::uint8_t, 13U> kFiniteCodes{{
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U, 0x3cU,
      0x40U, 0xb8U, 0x70U, 0x78U, 0x7eU, 0xfeU,
  }};
  std::vector<std::uint16_t> activation(columns);
  std::vector<std::uint8_t> weights(rows * columns);
  fill_activation(activation);
  for (std::size_t index = 0U; index < weights.size(); ++index) {
    weights[index] = kFiniteCodes[(index * 7U + 1U) % kFiniteCodes.size()];
  }
  run_fp8_payload(test, stream, weights, kWeightScale, activation, rows,
                  columns, label, unaligned_weights, unaligned_activation);
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
}

void run_nvfp4_payload(TestContext& test, cudaStream_t stream,
                       const std::vector<std::uint8_t>& packed,
                       const std::vector<std::uint8_t>& scales,
                       const float weight_scale_2,
                       const std::vector<std::uint16_t>& activation,
                       const std::size_t rows, const std::size_t columns,
                       const std::string& label,
                       const bool unaligned_packed = false) {
  const std::vector<std::uint16_t> expected = nvfp4_host_reference(
      packed, scales, weight_scale_2, activation, rows, columns);
  std::vector<std::uint16_t> actual(rows, 0U);
  std::vector<std::uint16_t> repeated(rows, 0U);
  std::vector<std::uint16_t> reference(rows, 0U);

  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_output;
  DeviceBuffer<float> device_reference_fp32;
  DeviceBuffer<std::uint16_t> device_reference_output;
  const std::size_t packed_offset = unaligned_packed ? 1U : 0U;
  bool ready = test.cuda_ok(
      device_packed.allocate(packed.size() + packed_offset),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate scales");
  ready = ready && test.cuda_ok(device_activation.allocate(activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(device_output.allocate(rows),
                                label + " allocate output");
  ready = ready && test.cuda_ok(device_reference_fp32.allocate(rows),
                                label + " allocate reference FP32");
  ready = ready && test.cuda_ok(device_reference_output.allocate(rows),
                                label + " allocate reference BF16");
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
                       cudaMemcpyAsync(device_activation.get(),
                                       activation.data(),
                                       activation.size() * sizeof(activation[0]),
                                       cudaMemcpyHostToDevice, stream),
                       label + " copy activation");
  if (!ready) {
    return;
  }
  const std::uint8_t* const packed_device =
      device_packed.get() + packed_offset;

  (void)seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
              packed_device, device_scales.get(), weight_scale_2,
              device_activation.get(), rows, columns, device_output.get(),
              static_cast<void*>(stream))),
      label + " launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(actual.data(), device_output.get(),
                                       actual.size() * sizeof(actual[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy output");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::kernels::
                               launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
                                   packed_device, device_scales.get(),
                                   weight_scale_2, device_activation.get(), rows,
                                   columns, device_output.get(),
                                   static_cast<void*>(stream))),
                       label + " repeat launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(repeated.data(), device_output.get(),
                                       repeated.size() * sizeof(repeated[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy repeated output");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::kernels::launch_nvfp4_gemv_reference_cuda(
                               packed_device, device_scales.get(),
                               weight_scale_2, device_activation.get(), rows,
                               columns, device_reference_fp32.get(),
                               static_cast<void*>(stream))),
                       label + " launch CUDA reference");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                               device_reference_fp32.get(), rows,
                               device_reference_output.get(),
                               static_cast<void*>(stream))),
                       label + " convert CUDA reference");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(reference.data(),
                                       device_reference_output.get(),
                                       reference.size() * sizeof(reference[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy CUDA reference");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (ready) {
    compare_bf16_outputs(test, actual, expected, columns, label);
    compare_cuda_reference_outputs(test, actual, reference, columns, label);
    test.expect(actual == repeated, label + " is bitwise deterministic");
  }
}

void run_nvfp4_case(TestContext& test, cudaStream_t stream,
                    const std::size_t rows, const std::size_t columns,
                    const std::string& label,
                    const bool unaligned_packed = false) {
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 8U> kScaleCodes{{
      0x20U, 0x28U, 0x30U, 0x34U, 0x38U, 0x3cU, 0x40U, 0x44U,
  }};
  std::vector<std::uint16_t> activation(columns);
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
                    rows, columns, label, unaligned_packed);
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
}

void test_launch_validation(TestContext& test) {
  constexpr std::size_t kMaximum = std::numeric_limits<std::size_t>::max();
  auto* const byte_pointer = reinterpret_cast<const std::uint8_t*>(0x1000U);
  auto* const activation =
      reinterpret_cast<const std::uint16_t*>(0x2000U);
  auto* const output = reinterpret_cast<std::uint16_t*>(0x3000U);

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
}

[[nodiscard]] bool performance_enabled() noexcept {
  const char* value = std::getenv("Q3X_RUN_SM87_WEIGHT_ONLY_GEMV_PERF");
  if (value == nullptr) {
    value = std::getenv("Q3X_SM87_GEMV_PERF");
  }
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

  // Awkward rows and K tails exercise partial blocks/warps. Target-K cases
  // cover both fixed MLP reduction lengths without allocating full matrices.
  run_fp8_case(test, stream, 13U, 37U, "FP8 awkward 13x37");
  run_fp8_vector_codebook_case(test, stream);
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
  run_nvfp4_case(test, stream, 3U, 256U,
                  "NVFP4 vector-shaped unaligned scalar fallback", true);
  run_nvfp4_case(test, stream, 3U, 5'120U, "NVFP4 target-K 3x5120");
  run_nvfp4_case(test, stream, 3U, 6'144U, "NVFP4 target-K 3x6144");
  run_nvfp4_case(test, stream, 3U, 17'408U,
                  "NVFP4 target-K 3x17408");
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
