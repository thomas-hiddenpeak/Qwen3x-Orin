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

[[nodiscard]] int launch_sm87_fp8_w8a16_gemv_bf16_vector_uncapped_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_gemv_bf16_grid_cap_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_gemv_bf16_row_pair_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_m1_row_pair_grid_cap_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_m1_row_quad_grid_cap_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_m1_row_quad_swizzled_codebook_grid_cap_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_m1_kv_pair_row_quad_grid_cap_test_cuda(
    const std::uint8_t* key_weights, float key_weight_scale,
    const std::uint8_t* value_weights, float value_weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* key_output, std::uint16_t* value_output,
    std::size_t maximum_blocks, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
query_sm87_fp8_w8a16_m1_kv_pair_row_quad_resources_test_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] bool use_sm87_fp8_m1_row_pair_shape_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] std::size_t sm87_fp8_m1_row_quad_maximum_blocks_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] bool use_sm87_fp8_m1_persistent_rows_test(
    std::size_t rows) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_small_m8_single_row_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_small_m8_row_pair_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_small_m2_uncapped_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_small_m2_grid_cap_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_small_m2_row_pair_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_small_m2_row_pair_grid_cap_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_small_m2_row_quad_grid_cap_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] bool use_sm87_fp8_m2_row_pair_shape_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] std::size_t sm87_fp8_m2_row_quad_maximum_blocks_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] bool use_sm87_fp8_m2_persistent_rows_test(
    std::size_t rows) noexcept;

[[nodiscard]] bool use_sm87_fp8_small_m_row_pair_test(
    std::size_t token_count, std::size_t rows) noexcept;

[[nodiscard]] bool use_sm87_fp8_m8_fixed_shape_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] bool use_sm87_fp8_m16_wmma_fixed_shape_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] int launch_sm87_fp8_w8a16_small_m8_fixed_shape_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_small_m16_wmma_fixed_shape_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_small_m16_wmma_shared_ldm_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows, std::size_t columns,
    std::uint16_t* output, std::size_t shared_leading_dimension,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_scalar_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_vector_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_grid_cap_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_pair_grid_cap_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_grid_cap_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output, std::size_t maximum_blocks,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_row_quad_exact_shape_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gemv_bf16_down_dual_iteration_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
query_sm87_nvfp4_w4a16_m1_down_dual_iteration_resources_test_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] bool use_sm87_nvfp4_m1_scale_codebook_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] bool use_sm87_nvfp4_m1_row_quad_shape_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] bool use_sm87_nvfp4_m1_down_dual_iteration_shape_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] std::size_t
sm87_nvfp4_m1_persistent_maximum_blocks_test() noexcept;

[[nodiscard]] std::size_t
sm87_nvfp4_m1_row_pair_maximum_blocks_test() noexcept;

[[nodiscard]] std::size_t
sm87_nvfp4_m1_row_quad_maximum_blocks_test() noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m2_vector_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_grid_cap_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    std::size_t maximum_blocks, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_quad_grid_cap_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    std::size_t maximum_blocks, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] bool use_sm87_nvfp4_m2_scale_codebook_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] bool use_sm87_nvfp4_m2_row_quad_shape_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] std::size_t
sm87_nvfp4_m2_row_quad_maximum_blocks_test() noexcept;

[[nodiscard]] bool use_sm87_nvfp4_small_m_row_pair_test(
    std::size_t token_count, std::size_t rows) noexcept;

[[nodiscard]] bool use_sm87_nvfp4_m8_fixed_shape_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] bool use_sm87_nvfp4_m16_wmma_fixed_shape_test(
    std::size_t rows, std::size_t columns) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m16_wmma_fixed_shape_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_small_m16_wmma_k128_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

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

[[nodiscard]] bool is_bf16_finite(const std::uint16_t bits) noexcept {
  return (bits & 0x7f80U) != 0x7f80U;
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
                      const bool use_small_m_api = false,
                      const bool use_m16_api = false) {
  const bool valid_token_count =
      use_m16_api ? token_count == 16U
                  : token_count >= 1U && token_count <= 8U;
  test.expect(valid_token_count,
              label + " valid token count");
  test.expect(!(use_small_m_api && use_m16_api),
              label + " selects one multi-token API");
  test.expect(use_small_m_api || use_m16_api || token_count == 1U,
              label + " multi-token payload uses a tile API");
  test.expect(activation.size() == token_count * columns,
              label + " activation extent");
  if (!valid_token_count || (use_small_m_api && use_m16_api) ||
      (!use_small_m_api && !use_m16_api && token_count != 1U) ||
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
  if (use_small_m_api || use_m16_api) {
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
    if (use_m16_api) {
      return q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
          weights_device, weight_scale, activation_device, rows, columns,
          device_output.get(), static_cast<void*>(stream));
    }
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
    if (use_small_m_api || use_m16_api) {
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
  if (use_small_m_api || use_m16_api) {
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
    if (use_small_m_api || use_m16_api) {
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
                  const bool use_small_m_api = false,
                  const bool use_m16_api = false) {
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
                  false, token_count, use_small_m_api, use_m16_api);
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

void run_fp8_m2_row_pair_exhaustive_case(TestContext& test,
                                          cudaStream_t stream) {
  constexpr std::size_t kTokens = 2U;
  constexpr std::size_t kBytePositions = 4U;
  constexpr std::size_t kRows = 4'097U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::size_t kProductionGridCap = 2'048U;
  constexpr float kWeightScale = 1.0F / 128.0F;
  const std::string label =
      "FP8 M2 row-pair odd grid-stride N4097 K5120 full E4M3FN byte "
      "positions";

  // Zero background weights keep every production-K loop live without
  // allowing a large accumulated value to hide the contribution of a tiny
  // exhaustive code at final BF16 rounding. Place each E4M3FN encoding in
  // every byte of the first packed-x4 load.
  std::vector<std::uint8_t> host_weights(kRows * kColumns, 0U);
  for (std::size_t code = 0U; code < 256U; ++code) {
    for (std::size_t position = 0U; position < kBytePositions; ++position) {
      const std::size_t row = code * kBytePositions + position;
      host_weights[row * kColumns + position] =
          static_cast<std::uint8_t>(code);
    }
  }
  // With ceil(N/2)=2049 and a cap of 2048, block zero must execute a second
  // grid-stride iteration for this nonzero, unpaired final row.
  constexpr std::array<std::uint8_t, kBytePositions> kTailCodes{{
      0x38U, 0xb8U, 0x40U, 0xc0U,
  }};
  for (std::size_t position = 0U; position < kBytePositions; ++position) {
    host_weights[(kRows - 1U) * kColumns + position] = kTailCodes[position];
  }

  std::vector<std::uint16_t> host_activations(kTokens * kColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const int centered =
          static_cast<int>((column * 13U + token * 29U + 5U) % 127U) - 63;
      host_activations[token * kColumns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
    for (std::size_t position = 0U; position < kBytePositions; ++position) {
      const float distinct = static_cast<float>(
                                 (token + 1U) * (position + 2U)) /
                             8.0F;
      host_activations[token * kColumns + position] = encode_bf16(distinct);
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
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * kRows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * kRows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(weights.get(), host_weights.data(),
                                       host_weights.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " copy weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy activations");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m2_grid_cap_test_cuda(
              weights.get(), kWeightScale, activations.get(), kRows, kColumns,
              baseline_output.get(), kProductionGridCap,
              static_cast<void*>(stream))),
      label + " launch preserved cap2048 single-row baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_small_m2_row_pair_test_cuda(
                               weights.get(), kWeightScale, activations.get(),
                               kRows, kColumns, candidate_output.get(),
                               static_cast<void*>(stream))),
                       label + " launch direct row-pair");
  std::vector<std::uint16_t> baseline(kTokens * kRows);
  std::vector<std::uint16_t> candidate(kTokens * kRows);
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
                                label + " synchronize");
  if (!ready) {
    return;
  }

  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    mismatches += baseline[index] != candidate[index] ? 1U : 0U;
  }
  std::cout << "FP8_M2_ROW_PAIR_DIFF: " << label
            << " direct_row_pair_vs_preserved_cap2048_single_row_bf16="
            << mismatches << '/' << baseline.size() << '\n';
  test.expect(
      mismatches == 0U,
      label + " direct row-pair matches every preserved cap2048 BF16 bit");

  // Exercise the public dispatch on the smallest promoted shape as well as
  // the direct kernel entry. The allowlist assertions below independently
  // freeze all five promoted shapes and their near-miss fallbacks.
  constexpr std::size_t kPromotedRows = 1'024U;
  const std::size_t promoted_output_count = kTokens * kPromotedRows;
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m2_row_pair_test_cuda(
              weights.get(), kWeightScale, activations.get(), kPromotedRows,
              kColumns, candidate_output.get(), static_cast<void*>(stream))),
      label + " launch promoted-shape direct row-pair");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
                               weights.get(), kWeightScale, activations.get(),
                               kTokens, kPromotedRows, kColumns,
                               baseline_output.get(),
                               static_cast<void*>(stream))),
                       label + " launch promoted-shape public dispatch");
  baseline.resize(promoted_output_count);
  candidate.resize(promoted_output_count);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), baseline_output.get(),
                           promoted_output_count * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy promoted public output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           candidate.data(), candidate_output.get(),
                           promoted_output_count * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy promoted direct output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize promoted dispatch");
  if (!ready) {
    return;
  }
  mismatches = 0U;
  for (std::size_t index = 0U; index < promoted_output_count; ++index) {
    mismatches += baseline[index] != candidate[index] ? 1U : 0U;
  }
  std::cout << "FP8_M2_ROW_PAIR_PRODUCTION_DIFF: rows=" << kPromotedRows
            << " columns=" << kColumns
            << " public_vs_direct_row_pair_bf16=" << mismatches << '/'
            << promoted_output_count << '\n';
  test.expect(mismatches == 0U,
              label + " promoted public dispatch matches direct row-pair");
}

void run_fp8_m1_row_pair_exhaustive_case(TestContext& test,
                                          cudaStream_t stream) {
  constexpr std::size_t kBytePositions = 4U;
  constexpr std::size_t kRows = 4'097U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::size_t kProductionGridCap = 2'048U;
  constexpr float kWeightScale = 1.0F / 128.0F;
  const std::string label =
      "FP8 M1 row-pair odd grid-stride N4097 K5120 row-distinct full "
      "E4M3FN byte positions";

  // The decoded-zero background keeps every production-K loop live without
  // hiding tiny exhaustive codes at final BF16 rounding. Ordering code as the
  // inner dimension makes every paired exhaustive row use different weights.
  std::vector<std::uint8_t> host_weights(kRows * kColumns, 0U);
  for (std::size_t position = 0U; position < kBytePositions; ++position) {
    for (std::size_t code = 0U; code < 256U; ++code) {
      const std::size_t row = position * 256U + code;
      host_weights[row * kColumns + position] =
          static_cast<std::uint8_t>(code);
    }
  }
  // ceil(N/2)=2049 exceeds the frozen cap, so block zero must execute a
  // second row-pair grid-stride iteration for this nonzero odd tail.
  constexpr std::array<std::uint8_t, kBytePositions> kTailCodes{{
      0x38U, 0xb8U, 0x40U, 0xc0U,
  }};
  for (std::size_t position = 0U; position < kBytePositions; ++position) {
    host_weights[(kRows - 1U) * kColumns + position] = kTailCodes[position];
  }

  std::vector<std::uint16_t> host_activation(kColumns);
  for (std::size_t column = 0U; column < kColumns; ++column) {
    const int centered = static_cast<int>((column * 13U + 5U) % 127U) - 63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }
  for (std::size_t position = 0U; position < kBytePositions; ++position) {
    host_activation[position] =
        encode_bf16(static_cast<float>(position + 2U) / 8.0F);
  }

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  DeviceBuffer<std::uint16_t> swizzled_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activation.allocate(host_activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(kRows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kRows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(swizzled_output.allocate(kRows),
                                label + " allocate swizzled output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(weights.get(), host_weights.data(),
                                       host_weights.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " copy weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy activation");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_gemv_bf16_grid_cap_test_cuda(
              weights.get(), kWeightScale, activation.get(), kRows, kColumns,
              baseline_output.get(), kProductionGridCap,
              static_cast<void*>(stream))),
      label + " launch preserved production cap2048 baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_gemv_bf16_row_pair_test_cuda(
                               weights.get(), kWeightScale, activation.get(),
                               kRows, kColumns, candidate_output.get(),
                               static_cast<void*>(stream))),
                       label + " launch direct row-pair");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_m1_row_quad_swizzled_codebook_grid_cap_test_cuda(
                               weights.get(), kWeightScale, activation.get(),
                               kRows, kColumns, swizzled_output.get(), 64U,
                               static_cast<void*>(stream))),
                       label + " launch swizzled row-quad cap64");
  std::vector<std::uint16_t> baseline(kRows);
  std::vector<std::uint16_t> candidate(kRows);
  std::vector<std::uint16_t> swizzled(kRows);
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
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           swizzled.data(), swizzled_output.get(),
                           swizzled.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy swizzled output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize");
  if (!ready) {
    return;
  }

  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    mismatches += baseline[index] != candidate[index] ? 1U : 0U;
  }
  std::cout << "FP8_M1_ROW_PAIR_DIFF: " << label
            << " direct_row_pair_vs_preserved_cap2048_bf16=" << mismatches
            << '/' << baseline.size() << '\n';
  test.expect(mismatches == 0U,
              label +
                  " row-pair matches every preserved production BF16 bit");

  mismatches = 0U;
  for (std::size_t row = 0U; row < kRows; ++row) {
    mismatches += baseline[row] != swizzled[row] ? 1U : 0U;
  }
  std::cout << "FP8_M1_SWIZZLED_CODEBOOK_TAIL_DIFF: " << label
            << " swizzled_quad_cap=64"
            << " swizzled_vs_preserved_cap2048_bf16=" << mismatches << '/'
            << kRows << '\n';
  test.expect(mismatches == 0U,
              label +
                  " swizzled row-quad covers all codes/byte positions and "
                  "matches every preserved BF16 bit");

  constexpr std::size_t kPromotedRows = 1'024U;
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_gemv_bf16_row_pair_test_cuda(
              weights.get(), kWeightScale, activation.get(), kPromotedRows,
              kColumns, candidate_output.get(), static_cast<void*>(stream))),
      label + " launch promoted-shape direct row-pair");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_gemv_bf16_cuda(
                               weights.get(), kWeightScale, activation.get(),
                               kPromotedRows, kColumns, baseline_output.get(),
                               static_cast<void*>(stream))),
                       label + " launch promoted-shape public dispatch");
  baseline.resize(kPromotedRows);
  candidate.resize(kPromotedRows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), baseline_output.get(),
                           kPromotedRows * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy promoted public output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           candidate.data(), candidate_output.get(),
                           kPromotedRows * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy promoted direct output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize promoted dispatch");
  if (!ready) {
    return;
  }
  mismatches = 0U;
  for (std::size_t row = 0U; row < kPromotedRows; ++row) {
    mismatches += baseline[row] != candidate[row] ? 1U : 0U;
  }
  std::cout << "FP8_M1_ROW_PAIR_PRODUCTION_DIFF: rows=" << kPromotedRows
            << " columns=" << kColumns
            << " public_vs_direct_row_pair_bf16=" << mismatches << '/'
            << kPromotedRows << '\n';
  test.expect(mismatches == 0U,
              label + " promoted public dispatch matches direct row-pair");
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

void run_nvfp4_m1_scale_codebook_bitwise_case(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label,
    const bool use_public_candidate = false) {
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 16U> kFiniteScaleCodes{{
      0x20U, 0x24U, 0x28U, 0x2cU, 0x30U, 0x34U, 0x38U, 0x3cU,
      0x40U, 0x44U, 0x48U, 0x4cU, 0x50U, 0x54U, 0x58U, 0x5cU,
  }};
  const std::size_t packed_columns = columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  std::vector<std::uint8_t> packed(rows * packed_columns);
  std::vector<std::uint8_t> scales(rows * scale_columns);
  std::vector<std::uint16_t> activation(columns);

  // Every row traverses all 256 low/high E2M1 pairs. Row-dependent offsets
  // make odd tails and row addressing observable even when K is periodic.
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
    for (std::size_t scale_column = 0U; scale_column < scale_columns;
         ++scale_column) {
      scales[row * scale_columns + scale_column] = kFiniteScaleCodes[
          (scale_column * 13U + row * 7U) % kFiniteScaleCodes.size()];
    }
  }
  for (std::size_t column = 0U; column < columns; ++column) {
    const int centered =
        static_cast<int>((column * 17U + 5U) % 127U) - 63;
    activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_baseline;
  DeviceBuffer<std::uint16_t> device_candidate;
  bool ready = test.cuda_ok(device_packed.allocate(packed.size()),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(device_activation.allocate(activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(device_baseline.allocate(rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(device_candidate.allocate(rows),
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
                           device_activation.get(), activation.data(),
                           activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy activation");
  if (!ready) {
    return;
  }

  const auto launch_baseline = [&]() noexcept -> int {
    if (use_public_candidate) {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_test_cuda(
              device_packed.get(), device_scales.get(), kWeightScale2,
              device_activation.get(), rows, columns, device_baseline.get(),
              static_cast<void*>(stream));
    }
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_vector_test_cuda(
            device_packed.get(), device_scales.get(), kWeightScale2,
            device_activation.get(), rows, columns, device_baseline.get(),
            static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() noexcept -> int {
    if (use_public_candidate) {
      return q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
          device_packed.get(), device_scales.get(), kWeightScale2,
          device_activation.get(), rows, columns, device_candidate.get(),
          static_cast<void*>(stream));
    }
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_test_cuda(
            device_packed.get(), device_scales.get(), kWeightScale2,
            device_activation.get(), rows, columns, device_candidate.get(),
            static_cast<void*>(stream));
  };
  ready = test.cuda_ok(
      static_cast<cudaError_t>(launch_baseline()),
      label + (use_public_candidate
                   ? " launch uncapped M1 scale-codebook baseline"
                   : " launch preserved M1 vector baseline"));
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate()),
                       label + (use_public_candidate
                                    ? " launch public M1 production candidate"
                                    : " launch M1 shared scale candidate"));
  std::vector<std::uint16_t> baseline(rows);
  std::vector<std::uint16_t> candidate(rows);
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
  for (std::size_t row = 0U; row < rows; ++row) {
    mismatches += baseline[row] != candidate[row] ? 1U : 0U;
  }
  std::cout << "M1_SCALE_CODEBOOK_DIFF: " << label << ' '
            << (use_public_candidate
                    ? "public_vs_uncapped_scale_codebook_bf16="
                    : "candidate_vs_preserved_vector_bf16=")
            << mismatches << '/' << rows << '\n';
  test.expect(
      mismatches == 0U,
      label + (use_public_candidate
                   ? " public production matches every uncapped BF16 bit"
                   : " M1 candidate matches every baseline BF16 bit"));
}

void run_nvfp4_m1_scale_codebook_exhaustive_case(TestContext& test,
                                                  cudaStream_t stream) {
  constexpr std::size_t kRows = 257U;
  constexpr std::size_t kColumns = 256U;
  constexpr std::size_t kPackedColumns = kColumns / 2U;
  constexpr std::size_t kScaleColumns = kColumns / 16U;
  constexpr float kWeightScale2 = 1.0F;
  const std::string label =
      "NVFP4 M1 shared E4M3FN exhaustive scale codebook odd rows";

  std::vector<std::uint8_t> packed(kRows * kPackedColumns, 0U);
  std::vector<std::uint8_t> scales(kRows * kScaleColumns, 0x38U);
  std::vector<std::uint16_t> activation(kColumns);
  for (std::size_t row = 0U; row < kRows; ++row) {
    const std::size_t active_group = row % kScaleColumns;
    const std::size_t first_byte = active_group * 8U;
    for (std::size_t byte = 0U; byte < 8U; ++byte) {
      packed[row * kPackedColumns + first_byte + byte] =
          static_cast<std::uint8_t>(0x21U + ((row + byte) & 1U) * 0x12U);
    }
    // Rows 0..255 cover signed zeros, every finite value, and both NaNs.
    // Row 256 exercises the final partial block.
    scales[row * kScaleColumns + active_group] =
        static_cast<std::uint8_t>(row < 256U ? row : 0x38U);
  }
  for (std::size_t column = 0U; column < kColumns; ++column) {
    const int centered = static_cast<int>((column * 7U + 3U) % 61U) - 30;
    activation[column] =
        encode_bf16(static_cast<float>(centered) / 32.0F);
  }

  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_baseline;
  DeviceBuffer<std::uint16_t> device_candidate;
  bool ready = test.cuda_ok(device_packed.allocate(packed.size()),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(device_activation.allocate(activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(device_baseline.allocate(kRows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(device_candidate.allocate(kRows),
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
                           device_activation.get(), activation.data(),
                           activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy activation");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_vector_test_cuda(
              device_packed.get(), device_scales.get(), kWeightScale2,
              device_activation.get(), kRows, kColumns,
              device_baseline.get(), static_cast<void*>(stream))),
      label + " launch preserved M1 vector baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_test_cuda(
                               device_packed.get(), device_scales.get(),
                               kWeightScale2, device_activation.get(), kRows,
                               kColumns, device_candidate.get(),
                               static_cast<void*>(stream))),
                       label + " launch M1 shared scale candidate");
  std::vector<std::uint16_t> baseline(kRows);
  std::vector<std::uint16_t> candidate(kRows);
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
  for (std::size_t row = 0U; row < kRows; ++row) {
    mismatches += baseline[row] != candidate[row] ? 1U : 0U;
  }
  std::cout << "M1_SCALE_CODEBOOK_DIFF: " << label
            << " candidate_vs_preserved_vector_bf16=" << mismatches << '/'
            << kRows << '\n';
  test.expect(mismatches == 0U,
              label + " M1 candidate matches every baseline BF16 bit");
  test.expect(is_bf16_nan(baseline[0x7fU]),
              label + " positive NaN scale remains NaN");
  test.expect(is_bf16_nan(baseline[0xffU]),
              label + " negative NaN scale remains NaN");
}

void run_nvfp4_m1_row_pair_exhaustive_case(TestContext& test,
                                            cudaStream_t stream) {
  constexpr std::size_t kRows = 1'537U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::size_t kPackedColumns = kColumns / 2U;
  constexpr std::size_t kScaleColumns = kColumns / 16U;
  constexpr std::size_t kPackedBytePositions = 4U;
  constexpr std::size_t kE2M1PairCount = 256U;
  constexpr std::size_t kE2M1Rows =
      kPackedBytePositions * kE2M1PairCount;
  constexpr std::size_t kScaleRowOffset = kE2M1Rows;
  constexpr std::size_t kScaleCodeCount = 256U;
  constexpr std::size_t kFillerRowOffset =
      kScaleRowOffset + kScaleCodeCount;
  constexpr std::size_t kTailRow = kRows - 1U;
  constexpr std::size_t kPreservedBaselineGridCap = 96U;
  constexpr std::size_t kPreservedRowPairGridCap = 80U;
  constexpr std::size_t kProductionRowQuadGridCap = 64U;
  constexpr float kWeightScale2 = 1.0F;
  constexpr std::array<std::uint8_t, 16U> kFiniteScaleCodes{{
      0x20U, 0x24U, 0x28U, 0x2cU, 0x30U, 0x34U, 0x38U, 0x3cU,
      0x40U, 0x44U, 0x48U, 0x4cU, 0x50U, 0x54U, 0x58U, 0x5cU,
  }};
  const std::string label =
      "NVFP4 M1 row-pair/quad odd grid-stride N1537 K5120 full E2M1 "
      "pairs and E4M3FN scales";

  test.expect(
      q3x::kernels::sm87_nvfp4_m1_persistent_maximum_blocks_test() ==
          kPreservedBaselineGridCap,
      label + " frozen production baseline is cap96");
  test.expect(
      q3x::kernels::sm87_nvfp4_m1_row_pair_maximum_blocks_test() ==
          kPreservedRowPairGridCap,
      label + " frozen row-pair A/B cap is 80");
  test.expect(
      q3x::kernels::sm87_nvfp4_m1_row_quad_maximum_blocks_test() ==
          kProductionRowQuadGridCap,
      label + " frozen production row-quad cap is 64");

  // A finite decoded-zero background executes every production-K loop while
  // keeping one deliberately active packed byte or scale group observable.
  std::vector<std::uint8_t> packed(kRows * kPackedColumns, 0U);
  std::vector<std::uint8_t> scales(kRows * kScaleColumns, 0x38U);
  std::vector<std::uint16_t> activation(kColumns);
  for (std::size_t column = 0U; column < kColumns; ++column) {
    const int magnitude = 1 + static_cast<int>((column * 13U + 7U) % 31U);
    const int signed_magnitude =
        (column & 1U) == 0U ? magnitude : -magnitude;
    activation[column] =
        encode_bf16(static_cast<float>(signed_magnitude) / 64.0F);
  }

  // Cover all 256 low/high E2M1 pairs in each byte position of the vector
  // kernel's 32-bit packed-weight word. Consecutive rows therefore never
  // alias the same row payload.
  for (std::size_t byte_position = 0U;
       byte_position < kPackedBytePositions; ++byte_position) {
    for (std::size_t pair = 0U; pair < kE2M1PairCount; ++pair) {
      const std::size_t row = byte_position * kE2M1PairCount + pair;
      const std::uint8_t low = static_cast<std::uint8_t>(pair & 0x0fU);
      const std::uint8_t high =
          static_cast<std::uint8_t>((pair >> 4U) & 0x0fU);
      packed[row * kPackedColumns + byte_position] =
          static_cast<std::uint8_t>(low | (high << 4U));
    }
  }

  // The following 256 rows isolate every E4M3FN scale encoding, including
  // signed zeros, every finite value, and the positive/negative NaN classes.
  // A coprime group stride also exercises scale addressing across K.
  for (std::size_t code = 0U; code < kScaleCodeCount; ++code) {
    const std::size_t row = kScaleRowOffset + code;
    const std::size_t active_group = (code * 37U + 11U) % kScaleColumns;
    const std::size_t first_byte = active_group * 8U;
    for (std::size_t byte = 0U; byte < 8U; ++byte) {
      packed[row * kPackedColumns + first_byte + byte] =
          static_cast<std::uint8_t>(
              0x21U + (((code + byte) & 1U) != 0U ? 0x12U : 0U));
    }
    scales[row * kScaleColumns + active_group] =
        static_cast<std::uint8_t>(code);
  }

  // Fill the remainder of the first candidate grid wave with nonzero,
  // row-distinct finite payloads. With eight row-pair warps per block,
  // 80 blocks cover exactly 1280 rows, so every filler row and the final odd
  // row below must execute in a second grid-stride iteration.
  for (std::size_t row = kFillerRowOffset; row < kTailRow; ++row) {
    const std::size_t sequence = row - kFillerRowOffset;
    const std::size_t active_group =
        (sequence * 37U + 17U) % kScaleColumns;
    const std::size_t active_byte =
        active_group * 8U + (sequence % 8U);
    std::uint8_t low =
        static_cast<std::uint8_t>((sequence * 3U + 1U) & 0x0fU);
    std::uint8_t high =
        static_cast<std::uint8_t>((sequence * 5U + 2U) & 0x0fU);
    if (low == 0U && high == 0U) {
      high = 1U;
    }
    packed[row * kPackedColumns + active_byte] =
        static_cast<std::uint8_t>(low | (high << 4U));
    scales[row * kScaleColumns + active_group] =
        kFiniteScaleCodes[(sequence * 7U + 3U) %
                          kFiniteScaleCodes.size()];
  }
  constexpr std::size_t kTailGroup = kScaleColumns - 1U;
  packed[kTailRow * kPackedColumns + kTailGroup * 8U + 7U] = 0x6dU;
  scales[kTailRow * kScaleColumns + kTailGroup] = 0x58U;

  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_baseline;
  DeviceBuffer<std::uint16_t> device_candidate;
  DeviceBuffer<std::uint16_t> device_row_quad;
  bool ready = test.cuda_ok(device_packed.allocate(packed.size()),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(device_activation.allocate(activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(device_baseline.allocate(kRows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(device_candidate.allocate(kRows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(device_row_quad.allocate(kRows),
                                label + " allocate row-quad output");
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
                           device_activation.get(), activation.data(),
                           activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy activation");
  if (!ready) {
    return;
  }

  const auto launch_baseline = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_grid_cap_test_cuda(
            device_packed.get(), device_scales.get(), kWeightScale2,
            device_activation.get(), kRows, kColumns, device_baseline.get(),
            kPreservedBaselineGridCap, static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_pair_grid_cap_test_cuda(
            device_packed.get(), device_scales.get(), kWeightScale2,
            device_activation.get(), kRows, kColumns, device_candidate.get(),
            kPreservedRowPairGridCap, static_cast<void*>(stream));
  };
  const auto launch_row_quad = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_grid_cap_test_cuda(
            device_packed.get(), device_scales.get(), kWeightScale2,
            device_activation.get(), kRows, kColumns, device_row_quad.get(),
            kProductionRowQuadGridCap, static_cast<void*>(stream));
  };
  ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                       label + " launch preserved production cap96 baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate()),
                       label + " launch direct row-pair A/B cap80");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_row_quad()),
                       label + " launch tail-safe row-quad cap64");
  std::vector<std::uint16_t> baseline(kRows);
  std::vector<std::uint16_t> candidate(kRows);
  std::vector<std::uint16_t> row_quad(kRows);
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
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           row_quad.data(), device_row_quad.get(),
                           row_quad.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy row-quad output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize");
  if (!ready) {
    return;
  }

  std::size_t mismatches = 0U;
  std::size_t row_quad_mismatches = 0U;
  for (std::size_t row = 0U; row < kRows; ++row) {
    mismatches += baseline[row] != candidate[row] ? 1U : 0U;
    row_quad_mismatches += baseline[row] != row_quad[row] ? 1U : 0U;
  }
  std::cout << "NVFP4_M1_ROW_PAIR_DIFF: " << label
            << " direct_cap80_row_pair_vs_preserved_cap96_bf16="
            << mismatches << '/' << kRows << '\n';
  test.expect(mismatches == 0U,
              label + " row-pair matches every preserved baseline bit");
  std::cout << "NVFP4_M1_ROW_QUAD_TAIL_DIFF: " << label
            << " direct_cap64_row_quad_vs_preserved_cap96_bf16="
            << row_quad_mismatches << '/' << kRows << '\n';
  test.expect(row_quad_mismatches == 0U,
              label + " tail-safe row-quad matches every baseline bit");
  test.expect(is_bf16_nan(baseline[kScaleRowOffset + 0x7fU]) &&
                  is_bf16_nan(candidate[kScaleRowOffset + 0x7fU]) &&
                  is_bf16_nan(row_quad[kScaleRowOffset + 0x7fU]),
              label + " positive NaN scale remains in the BF16 NaN class");
  test.expect(is_bf16_nan(baseline[kScaleRowOffset + 0xffU]) &&
                  is_bf16_nan(candidate[kScaleRowOffset + 0xffU]) &&
                  is_bf16_nan(row_quad[kScaleRowOffset + 0xffU]),
              label + " negative NaN scale remains in the BF16 NaN class");
}

void run_nvfp4_m2_scale_codebook_payload(
    TestContext& test, cudaStream_t stream,
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::uint8_t>& scales,
    const std::vector<std::uint16_t>& activations,
    const float weight_scale_2, const std::size_t rows,
    const std::size_t columns, const std::string& label,
    const bool expect_nan_scale_rows = false) {
  constexpr std::size_t kTokens = 2U;
  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  DeviceBuffer<std::uint16_t> device_activations;
  DeviceBuffer<std::uint16_t> device_baseline;
  DeviceBuffer<std::uint16_t> device_candidate;
  DeviceBuffer<std::uint16_t> device_row_pair;
  DeviceBuffer<std::uint16_t> device_production;
  DeviceBuffer<std::uint16_t> device_repeated_m1;
  bool ready = test.cuda_ok(device_packed.allocate(packed.size()),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(
                       device_activations.allocate(activations.size()),
                       label + " allocate activations");
  ready = ready && test.cuda_ok(device_baseline.allocate(kTokens * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(device_candidate.allocate(kTokens * rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(device_row_pair.allocate(kTokens * rows),
                                label + " allocate row-pair output");
  ready = ready && test.cuda_ok(device_production.allocate(kTokens * rows),
                                label + " allocate production output");
  ready = ready && test.cuda_ok(device_repeated_m1.allocate(kTokens * rows),
                                label + " allocate repeated-M1 output");
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
          launch_sm87_nvfp4_w4a16_small_m2_vector_test_cuda(
              device_packed.get(), device_scales.get(), weight_scale_2,
              device_activations.get(), rows, columns, device_baseline.get(),
              static_cast<void*>(stream))),
      label + " launch preserved M2 vector baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda(
                               device_packed.get(), device_scales.get(),
                               weight_scale_2, device_activations.get(), rows,
                               columns, device_candidate.get(),
                               static_cast<void*>(stream))),
                       label + " launch M2 scale-codebook candidate");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_test_cuda(
                               device_packed.get(), device_scales.get(),
                               weight_scale_2, device_activations.get(), rows,
                               columns, device_row_pair.get(),
                               static_cast<void*>(stream))),
                       label + " launch M2 scale-codebook row-pair candidate");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
                               device_packed.get(), device_scales.get(),
                               weight_scale_2, device_activations.get(),
                               kTokens, rows, columns, device_production.get(),
                               static_cast<void*>(stream))),
                       label + " launch production M2 dispatch");
  for (std::size_t token = 0U; token < kTokens && ready; ++token) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(
            q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
                device_packed.get(), device_scales.get(), weight_scale_2,
                device_activations.get() + token * columns, rows, columns,
                device_repeated_m1.get() + token * rows,
                static_cast<void*>(stream))),
        label + " launch repeated M1 token " + std::to_string(token));
  }
  std::vector<std::uint16_t> baseline(kTokens * rows);
  std::vector<std::uint16_t> candidate(kTokens * rows);
  std::vector<std::uint16_t> row_pair(kTokens * rows);
  std::vector<std::uint16_t> production(kTokens * rows);
  std::vector<std::uint16_t> repeated_m1(kTokens * rows);
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
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           row_pair.data(), device_row_pair.get(),
                           row_pair.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy row-pair output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           production.data(), device_production.get(),
                           production.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy production output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           repeated_m1.data(), device_repeated_m1.get(),
                           repeated_m1.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy repeated-M1 output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " synchronize");
  if (!ready) {
    return;
  }

  std::size_t candidate_mismatches = 0U;
  std::size_t row_pair_mismatches = 0U;
  std::size_t production_mismatches = 0U;
  std::size_t repeated_m1_mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    candidate_mismatches += baseline[index] != candidate[index] ? 1U : 0U;
    row_pair_mismatches += baseline[index] != row_pair[index] ? 1U : 0U;
    production_mismatches += baseline[index] != production[index] ? 1U : 0U;
    repeated_m1_mismatches +=
        baseline[index] != repeated_m1[index] ? 1U : 0U;
  }
  std::cout << "M2_SCALE_CODEBOOK_DIFF: " << label
            << " candidate_vs_baseline_bf16=" << candidate_mismatches << '/'
            << baseline.size()
            << " row_pair_vs_baseline_bf16=" << row_pair_mismatches << '/'
            << baseline.size()
            << " production_vs_baseline_bf16=" << production_mismatches
            << '/' << baseline.size()
            << " repeated_m1_vs_baseline_bf16=" << repeated_m1_mismatches
            << '/' << baseline.size() << '\n';
  test.expect(candidate_mismatches == 0U,
              label + " M2 candidate matches every baseline BF16 bit");
  test.expect(row_pair_mismatches == 0U,
              label +
                  " M2 row-pair candidate matches every baseline BF16 bit");
  test.expect(production_mismatches == 0U,
              label + " production M2 matches every baseline BF16 bit");
  test.expect(repeated_m1_mismatches == 0U,
              label + " M2 baseline matches repeated M1 BF16 bits");
  if (expect_nan_scale_rows) {
    for (std::size_t token = 0U; token < kTokens; ++token) {
      test.expect(is_bf16_nan(baseline[token * rows + 0x7fU]),
                  label + " positive NaN scale remains NaN");
      test.expect(is_bf16_nan(baseline[token * rows + 0xffU]),
                  label + " negative NaN scale remains NaN");
      test.expect((baseline[token * rows] & 0x7fffU) == 0U,
                  label + " positive zero scale remains zero");
      test.expect((baseline[token * rows + 0x80U] & 0x7fffU) == 0U,
                  label + " negative zero scale remains zero");
    }
  }
}

void run_nvfp4_m2_scale_codebook_bitwise_case(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 2U;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 16U> kFiniteScaleCodes{{
      0x20U, 0x24U, 0x28U, 0x2cU, 0x30U, 0x34U, 0x38U, 0x3cU,
      0x40U, 0x44U, 0x48U, 0x4cU, 0x50U, 0x54U, 0x58U, 0x5cU,
  }};
  const std::size_t packed_columns = columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  std::vector<std::uint8_t> packed(rows * packed_columns);
  std::vector<std::uint8_t> scales(rows * scale_columns);
  std::vector<std::uint16_t> activations(kTokens * columns);
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
  run_nvfp4_m2_scale_codebook_payload(
      test, stream, packed, scales, activations, kWeightScale2, rows, columns,
      label);
}

void run_nvfp4_m2_scale_codebook_exhaustive_case(TestContext& test,
                                                  cudaStream_t stream) {
  constexpr std::size_t kTokens = 2U;
  constexpr std::size_t kRows = 257U;
  constexpr std::size_t kColumns = 256U;
  constexpr std::size_t kPackedColumns = kColumns / 2U;
  constexpr std::size_t kScaleColumns = kColumns / 16U;
  constexpr float kWeightScale2 = 1.0F;
  const std::string label =
      "NVFP4 M2 shared E4M3FN exhaustive scale codebook odd rows";
  std::vector<std::uint8_t> packed(kRows * kPackedColumns, 0U);
  std::vector<std::uint8_t> scales(kRows * kScaleColumns, 0x38U);
  std::vector<std::uint16_t> activations(kTokens * kColumns);
  std::array<bool, 16U> low_nibble_covered{};
  std::array<bool, 16U> high_nibble_covered{};
  // Every scale-code row contains all 16 E2M1 codes exactly once; row 256
  // repeats a finite scale while exercising the unpaired output-row tail.
  for (std::size_t row = 0U; row < kRows; ++row) {
    const std::size_t active_group = row % kScaleColumns;
    const std::size_t first_byte = active_group * 8U;
    for (std::size_t byte = 0U; byte < 8U; ++byte) {
      const std::uint8_t low = static_cast<std::uint8_t>(
          (row + byte) & 0x0fU);
      const std::uint8_t high = static_cast<std::uint8_t>(
          (row + byte + 8U) & 0x0fU);
      packed[row * kPackedColumns + first_byte + byte] =
          static_cast<std::uint8_t>(low | (high << 4U));
      low_nibble_covered[low] = true;
      high_nibble_covered[high] = true;
    }
    scales[row * kScaleColumns + active_group] =
        static_cast<std::uint8_t>(row < 256U ? row : 0x38U);
  }
  test.expect(std::all_of(low_nibble_covered.begin(),
                          low_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers every low-nibble E2M1 code");
  test.expect(std::all_of(high_nibble_covered.begin(),
                          high_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers every high-nibble E2M1 code");
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const int centered =
          static_cast<int>((token * 29U + column * 7U + 3U) % 61U) - 30;
      activations[token * kColumns + column] =
          encode_bf16(static_cast<float>(centered) / 32.0F);
    }
  }
  run_nvfp4_m2_scale_codebook_payload(
      test, stream, packed, scales, activations, kWeightScale2, kRows,
      kColumns, label, true);
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
              byte_pointer, 1.0F, activation, 16U, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 small-M public contract remains capped at M=8");
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
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              nullptr, 1.0F, nullptr, 0U, kMaximum, nullptr)) == cudaSuccess,
      "FP8 M16 zero rows is a no-op before tile-size validation");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              nullptr, 1.0F, nullptr, kMaximum, 0U, nullptr)) == cudaSuccess,
      "FP8 M16 zero columns is a no-op before tile-size validation");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              nullptr, 1.0F, activation, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects null weights for a non-empty shape");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, nullptr, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects null activations for a non-empty shape");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 1U, 1U, nullptr)) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects null output for a non-empty shape");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, std::numeric_limits<float>::infinity(),
              activation, 1U, 1U, output)) == cudaErrorInvalidValue,
      "FP8 M16 rejects a non-finite scale");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, -1.0F, activation, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects a negative scale");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, kMaximum, 2U, output)) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects rows*K overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 1U, kMaximum / 16U + 1U,
              output)) == cudaErrorInvalidValue,
      "FP8 M16 rejects 16*K overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, kMaximum / 16U + 1U, 1U,
              output)) == cudaErrorInvalidValue,
      "FP8 M16 rejects 16*N overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 1U, kMaximum / 16U,
              output)) == cudaErrorInvalidValue,
      "FP8 M16 rejects activation byte-size overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, kMaximum / 16U, 1U,
              output)) == cudaErrorInvalidValue,
      "FP8 M16 rejects output byte-size overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, wrapped_activation, 1U, 1U, output)) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects a wrapping activation range");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 1U, 1U,
              reinterpret_cast<std::uint16_t*>(
                  std::numeric_limits<std::uintptr_t>::max() - 15U))) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects a wrapping output range");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              byte_pointer, 1.0F, activation, 1U, 16U,
              reinterpret_cast<std::uint16_t*>(0x1008U))) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects output overlap with weights");

  constexpr std::uintptr_t kM16ActivationAddress = 0x2'0000'0000ULL;
  constexpr std::size_t kM16Rows = 10'240U;
  constexpr std::size_t kM16Columns = 5'120U;
  constexpr std::size_t kM16HalfActivationBytes =
      8U * kM16Columns * sizeof(std::uint16_t);
  constexpr std::size_t kM16HalfOutputBytes =
      8U * kM16Rows * sizeof(std::uint16_t);
  auto* const m16_weights =
      reinterpret_cast<const std::uint8_t*>(0x1'0000'0000ULL);
  auto* const m16_activations =
      reinterpret_cast<const std::uint16_t*>(kM16ActivationAddress);
  auto* const m16_output_first_overlaps_activation_second =
      reinterpret_cast<std::uint16_t*>(kM16ActivationAddress +
                                       kM16HalfActivationBytes);
  auto* const m16_output_second_overlaps_activation_first =
      reinterpret_cast<std::uint16_t*>(
          kM16ActivationAddress -
          (kM16HalfOutputBytes + kM16HalfActivationBytes));
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::
              launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
                  m16_weights, 1.0F, m16_activations, kM16Rows,
                  kM16Columns,
                  m16_output_first_overlaps_activation_second)) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects output[0:8]/activation[8:16] overlap");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::
              launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
                  m16_weights, 1.0F, m16_activations, kM16Rows,
                  kM16Columns,
                  m16_output_second_overlaps_activation_first)) ==
          cudaErrorInvalidValue,
      "FP8 M16 rejects output[8:16]/activation[0:8] overlap");

  constexpr std::size_t kLargestAlignedSixteenth =
      (kMaximum / 16U) & ~std::size_t{15U};
  constexpr std::size_t kFirstAlignedPastSixteenth =
      ((kMaximum / 16U) / 16U + 1U) * 16U;
  constexpr std::size_t kFirstOutputByteOverflowRows =
      kMaximum / 32U + 1U;
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              nullptr, nullptr, 1.0F, nullptr, 0U, 15U, nullptr)) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 validates K grouping before an empty-shape no-op");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              nullptr, nullptr, std::numeric_limits<float>::infinity(),
              nullptr, 0U, 16U, nullptr)) == cudaErrorInvalidValue,
      "NVFP4 M16 validates scale before an empty-shape no-op");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              nullptr, nullptr, 1.0F, nullptr, 0U, kMaximum - 15U,
              nullptr)) == cudaSuccess,
      "NVFP4 M16 zero rows is a no-op after legal K/scale validation");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              nullptr, nullptr, 1.0F, nullptr, kMaximum, 0U, nullptr)) ==
          cudaSuccess,
      "NVFP4 M16 zero columns is a no-op after legal K/scale validation");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              nullptr, scale_pointer, 1.0F, activation, 1U, 16U, output)) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects null packed weights for a non-empty shape");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, nullptr, 1.0F, activation, 1U, 16U, output)) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects null block scales for a non-empty shape");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, nullptr, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects null activations for a non-empty shape");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 1U, 16U,
              nullptr)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects null output for a non-empty shape");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 1U, 31U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects non-group-aligned K");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer,
              std::numeric_limits<float>::quiet_NaN(), activation, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects a non-finite scale");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, -1.0F, activation, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects a negative scale");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, kMaximum, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects rows*K overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 1U,
              kFirstAlignedPastSixteenth, output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects 16*K overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation,
              kMaximum / 16U + 1U, 16U, output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects a 16*N extent that also overflows N*K");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 1U,
              kLargestAlignedSixteenth, output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects activation byte-size overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation,
              kFirstOutputByteOverflowRows, 16U, output)) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects output byte-size overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              reinterpret_cast<const std::uint8_t*>(
                  std::numeric_limits<std::uintptr_t>::max() - 3U),
              scale_pointer, 1.0F, activation, 1U, 16U, output)) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects a wrapping packed-weight range");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, wrapped_scale, 1.0F, activation, 2U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects a wrapping block-scale range");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, wrapped_activation, 1U, 16U,
              output)) == cudaErrorInvalidValue,
      "NVFP4 M16 rejects a wrapping activation range");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 1U, 16U,
              reinterpret_cast<std::uint16_t*>(
                  std::numeric_limits<std::uintptr_t>::max() - 15U))) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects a wrapping output range");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 1U, 16U,
              reinterpret_cast<std::uint16_t*>(0x1004U))) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects output overlap with packed weights");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 2U, 16U,
              reinterpret_cast<std::uint16_t*>(
                  const_cast<std::uint8_t*>(scale_pointer)))) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects output overlap with block scales");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              byte_pointer, scale_pointer, 1.0F, activation, 1U, 16U,
              reinterpret_cast<std::uint16_t*>(0x2100U))) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects output overlap with the full activation tile");

  constexpr std::uintptr_t kNvFp4M16ActivationAddress =
      0x4'0000'0000ULL;
  constexpr std::size_t kNvFp4M16Rows = 17'408U;
  constexpr std::size_t kNvFp4M16Columns = 5'120U;
  constexpr std::size_t kNvFp4M16HalfActivationBytes =
      8U * kNvFp4M16Columns * sizeof(std::uint16_t);
  constexpr std::size_t kNvFp4M16HalfOutputBytes =
      8U * kNvFp4M16Rows * sizeof(std::uint16_t);
  auto* const nvfp4_m16_packed =
      reinterpret_cast<const std::uint8_t*>(0x3'0000'0000ULL);
  auto* const nvfp4_m16_scales =
      reinterpret_cast<const std::uint8_t*>(0x3'8000'0000ULL);
  auto* const nvfp4_m16_activations =
      reinterpret_cast<const std::uint16_t*>(kNvFp4M16ActivationAddress);
  auto* const nvfp4_m16_output_first_overlaps_activation_second =
      reinterpret_cast<std::uint16_t*>(
          kNvFp4M16ActivationAddress + kNvFp4M16HalfActivationBytes);
  auto* const nvfp4_m16_output_second_overlaps_activation_first =
      reinterpret_cast<std::uint16_t*>(
          kNvFp4M16ActivationAddress -
          (kNvFp4M16HalfOutputBytes + kNvFp4M16HalfActivationBytes));
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              nvfp4_m16_packed, nvfp4_m16_scales, 1.0F,
              nvfp4_m16_activations, kNvFp4M16Rows, kNvFp4M16Columns,
              nvfp4_m16_output_first_overlaps_activation_second)) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects output[0:8]/activation[8:16] overlap");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              nvfp4_m16_packed, nvfp4_m16_scales, 1.0F,
              nvfp4_m16_activations, kNvFp4M16Rows, kNvFp4M16Columns,
              nvfp4_m16_output_second_overlaps_activation_first)) ==
          cudaErrorInvalidValue,
      "NVFP4 M16 rejects output[8:16]/activation[0:8] overlap");

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
  test.expect(q3x::kernels::use_sm87_nvfp4_m8_fixed_shape_test(
                  17'408U, 5'120U),
              "NVFP4 M8 fixed-shape accepts gate/up");
  test.expect(q3x::kernels::use_sm87_nvfp4_m8_fixed_shape_test(
                  5'120U, 17'408U),
              "NVFP4 M8 fixed-shape accepts down");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m8_fixed_shape_test(
                  17'407U, 5'120U),
              "NVFP4 M8 fixed-shape rejects a near-miss row count");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m8_fixed_shape_test(
                  17'408U, 5'119U),
              "NVFP4 M8 fixed-shape rejects a near-miss K");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m8_fixed_shape_test(
                  248'320U, 5'120U),
              "NVFP4 M8 fixed-shape keeps lm_head on the generic path");
  test.expect(q3x::kernels::use_sm87_nvfp4_m16_wmma_fixed_shape_test(
                  17'408U, 5'120U),
              "NVFP4 M16 WMMA predicate accepts gate/up");
  test.expect(q3x::kernels::use_sm87_nvfp4_m16_wmma_fixed_shape_test(
                  5'120U, 17'408U),
              "NVFP4 M16 WMMA predicate accepts down");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m16_wmma_fixed_shape_test(
                  17'407U, 5'120U),
              "NVFP4 M16 WMMA predicate rejects a near-miss row count");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m16_wmma_fixed_shape_test(
                  17'408U, 5'104U),
              "NVFP4 M16 WMMA predicate rejects a near-miss K");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m16_wmma_fixed_shape_test(
                  248'320U, 5'120U),
              "NVFP4 M16 WMMA predicate keeps lm_head on two M8 tiles");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m16_wmma_fixed_shape_test_cuda(
              nullptr, nullptr, 1.0F, nullptr, 17'408U, 5'120U,
              nullptr)) == cudaErrorInvalidValue,
      "NVFP4 M16 WMMA test launcher rejects null non-empty buffers");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m1_scale_codebook_test(
                  7U, 5'120U),
              "NVFP4 M1 scale codebook rejects a partial warp block");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m1_scale_codebook_test(
                  8U, 1'024U),
              "NVFP4 M1 scale codebook keeps short-K vector baseline");
  test.expect(q3x::kernels::use_sm87_nvfp4_m1_scale_codebook_test(
                  8U, 5'120U),
              "NVFP4 M1 scale codebook accepts its minimum production gate");
  test.expect(q3x::kernels::use_sm87_nvfp4_m1_scale_codebook_test(
                  17'408U, 5'120U),
              "NVFP4 M1 scale codebook accepts gate/up shape");
  test.expect(q3x::kernels::use_sm87_nvfp4_m1_scale_codebook_test(
                  5'120U, 17'408U),
              "NVFP4 M1 scale codebook accepts down shape");
  test.expect(
      q3x::kernels::sm87_nvfp4_m1_persistent_maximum_blocks_test() == 96U,
      "NVFP4 M1 production persistent grid cap is frozen at 96 blocks");
  test.expect(q3x::kernels::use_sm87_nvfp4_m1_row_quad_shape_test(
                  17'408U, 5'120U),
              "NVFP4 M1 row-quad accepts gate/up shape");
  test.expect(q3x::kernels::use_sm87_nvfp4_m1_row_quad_shape_test(
                  5'120U, 17'408U),
              "NVFP4 M1 row-quad accepts down shape");
  test.expect(q3x::kernels::use_sm87_nvfp4_m1_row_quad_shape_test(
                  248'320U, 5'120U),
              "NVFP4 M1 row-quad accepts lm_head shape");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m1_row_quad_shape_test(
                  17'407U, 5'120U),
              "NVFP4 M1 row-quad rejects a near-miss row count");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m1_row_quad_shape_test(
                  17'408U, 5'104U),
              "NVFP4 M1 row-quad rejects a near-miss K");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m1_row_quad_shape_test(
                  5'120U, 5'120U),
              "NVFP4 M1 row-quad keeps unknown shapes on cap96");
  test.expect(
      q3x::kernels::use_sm87_nvfp4_m1_down_dual_iteration_shape_test(
          5'120U, 17'408U),
      "NVFP4 M1 down dual selector accepts only the production down shape");
  test.expect(
      !q3x::kernels::use_sm87_nvfp4_m1_down_dual_iteration_shape_test(
          17'408U, 5'120U),
      "NVFP4 M1 down dual selector rejects gate/up");
  test.expect(
      !q3x::kernels::use_sm87_nvfp4_m1_down_dual_iteration_shape_test(
          248'320U, 5'120U),
      "NVFP4 M1 down dual selector rejects lm_head");
  test.expect(
      !q3x::kernels::use_sm87_nvfp4_m1_down_dual_iteration_shape_test(
          5'119U, 17'408U),
      "NVFP4 M1 down dual selector rejects near-miss rows");
  test.expect(
      !q3x::kernels::use_sm87_nvfp4_m1_down_dual_iteration_shape_test(
          5'120U, 17'392U),
      "NVFP4 M1 down dual selector rejects near-miss columns");
  test.expect(
      q3x::kernels::sm87_nvfp4_m1_row_pair_maximum_blocks_test() == 80U,
      "NVFP4 M1 row-pair A/B baseline grid cap is frozen at 80 blocks");
  test.expect(
      q3x::kernels::sm87_nvfp4_m1_row_quad_maximum_blocks_test() == 64U,
      "NVFP4 M1 row-quad production grid cap is frozen at 64 blocks");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m2_scale_codebook_test(
                  7U, 5'120U),
              "NVFP4 M2 scale codebook rejects a partial warp block");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m2_scale_codebook_test(
                  8U, 1'024U),
              "NVFP4 M2 scale codebook keeps short-K vector baseline");
  test.expect(q3x::kernels::use_sm87_nvfp4_m2_scale_codebook_test(
                  8U, 5'120U),
              "NVFP4 M2 scale codebook accepts its minimum production gate");
  test.expect(q3x::kernels::use_sm87_nvfp4_m2_scale_codebook_test(
                  17'408U, 5'120U),
              "NVFP4 M2 scale codebook accepts gate/up shape");
  test.expect(q3x::kernels::use_sm87_nvfp4_m2_scale_codebook_test(
                  5'120U, 17'408U),
              "NVFP4 M2 scale codebook accepts down shape");
  test.expect(q3x::kernels::use_sm87_nvfp4_m2_row_quad_shape_test(
                  17'408U, 5'120U),
              "NVFP4 M2 row-quad accepts gate/up shape");
  test.expect(q3x::kernels::use_sm87_nvfp4_m2_row_quad_shape_test(
                  5'120U, 17'408U),
              "NVFP4 M2 row-quad accepts down shape");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m2_row_quad_shape_test(
                  17'407U, 5'120U),
              "NVFP4 M2 row-quad rejects a near-miss row count");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m2_row_quad_shape_test(
                  17'408U, 5'104U),
              "NVFP4 M2 row-quad rejects a near-miss K");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m2_row_quad_shape_test(
                  248'320U, 5'120U),
              "NVFP4 M2 row-quad keeps lm_head on the preserved path");
  test.expect(!q3x::kernels::use_sm87_nvfp4_m2_row_quad_shape_test(
                  5'120U, 5'120U),
              "NVFP4 M2 row-quad keeps unknown square shapes preserved");
  test.expect(
      q3x::kernels::sm87_nvfp4_m2_row_quad_maximum_blocks_test() == 64U,
      "NVFP4 M2 row-quad production grid cap is frozen at 64 blocks");
  test.expect(!q3x::kernels::use_sm87_fp8_small_m_row_pair_test(2U, 10'240U),
              "FP8 M2 uses its dedicated production-shape predicate");
  test.expect(q3x::kernels::use_sm87_fp8_m2_row_pair_shape_test(
                  10'240U, 5'120U),
              "FP8 M2 row-pair accepts 10240x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m2_row_pair_shape_test(
                  5'120U, 6'144U),
              "FP8 M2 row-pair accepts 5120x6144");
  test.expect(q3x::kernels::use_sm87_fp8_m2_row_pair_shape_test(
                  6'144U, 5'120U),
              "FP8 M2 row-pair accepts 6144x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m2_row_pair_shape_test(
                  12'288U, 5'120U),
              "FP8 M2 row-pair accepts 12288x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m2_row_pair_shape_test(
                  1'024U, 5'120U),
              "FP8 M2 row-pair accepts 1024x5120");
  test.expect(!q3x::kernels::use_sm87_fp8_m2_row_pair_shape_test(
                  5'120U, 5'120U),
              "FP8 M2 row-pair keeps unknown square shapes on cap2048");
  test.expect(!q3x::kernels::use_sm87_fp8_m2_row_pair_shape_test(
                  10'239U, 5'120U),
              "FP8 M2 row-pair rejects a near-miss row count");
  test.expect(!q3x::kernels::use_sm87_fp8_m2_row_pair_shape_test(
                  10'240U, 5'119U),
              "FP8 M2 row-pair rejects a near-miss K");
  test.expect(
      q3x::kernels::sm87_fp8_m2_row_quad_maximum_blocks_test(
          10'240U, 5'120U) == 1'536U,
      "FP8 M2 row-quad freezes 10240x5120 at cap1536");
  test.expect(
      q3x::kernels::sm87_fp8_m2_row_quad_maximum_blocks_test(
          5'120U, 6'144U) == 768U,
      "FP8 M2 row-quad freezes 5120x6144 at cap768");
  test.expect(
      q3x::kernels::sm87_fp8_m2_row_quad_maximum_blocks_test(
          6'144U, 5'120U) == 1'024U,
      "FP8 M2 row-quad freezes 6144x5120 at cap1024");
  test.expect(
      q3x::kernels::sm87_fp8_m2_row_quad_maximum_blocks_test(
          12'288U, 5'120U) == 2'048U,
      "FP8 M2 row-quad freezes 12288x5120 at cap2048");
  test.expect(
      q3x::kernels::sm87_fp8_m2_row_quad_maximum_blocks_test(
          1'024U, 5'120U) == 0U,
      "FP8 M2 row-quad preserves the small 1024x5120 row-pair");
  test.expect(
      q3x::kernels::sm87_fp8_m2_row_quad_maximum_blocks_test(
          5'120U, 5'120U) == 0U,
      "FP8 M2 row-quad preserves unknown shapes");
  test.expect(!q3x::kernels::use_sm87_fp8_small_m_row_pair_test(8U, 1'023U),
              "FP8 M8 row-pair keeps a conservative small-row fallback");
  test.expect(q3x::kernels::use_sm87_fp8_small_m_row_pair_test(8U, 1'024U),
              "FP8 M8 row-pair accepts the smallest checkpoint projection");
  test.expect(q3x::kernels::use_sm87_fp8_small_m_row_pair_test(8U, 10'240U),
              "FP8 M8 row-pair accepts production QKV output rows");
  test.expect(q3x::kernels::use_sm87_fp8_m8_fixed_shape_test(
                  10'240U, 5'120U),
              "FP8 M8 fixed-shape accepts 10240x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m8_fixed_shape_test(
                  5'120U, 6'144U),
              "FP8 M8 fixed-shape accepts 5120x6144");
  test.expect(q3x::kernels::use_sm87_fp8_m8_fixed_shape_test(
                  6'144U, 5'120U),
              "FP8 M8 fixed-shape accepts 6144x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m8_fixed_shape_test(
                  12'288U, 5'120U),
              "FP8 M8 fixed-shape accepts 12288x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m8_fixed_shape_test(
                  1'024U, 5'120U),
              "FP8 M8 fixed-shape accepts 1024x5120");
  test.expect(!q3x::kernels::use_sm87_fp8_m8_fixed_shape_test(
                  10'239U, 5'120U),
              "FP8 M8 fixed-shape rejects a near-miss row count");
  test.expect(!q3x::kernels::use_sm87_fp8_m8_fixed_shape_test(
                  10'240U, 5'119U),
              "FP8 M8 fixed-shape rejects a near-miss K");
  test.expect(!q3x::kernels::use_sm87_fp8_m8_fixed_shape_test(
                  5'120U, 5'120U),
              "FP8 M8 fixed-shape keeps unknown square shapes generic");
  test.expect(q3x::kernels::use_sm87_fp8_m16_wmma_fixed_shape_test(
                  10'240U, 5'120U),
              "FP8 M16 WMMA predicate accepts 10240x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m16_wmma_fixed_shape_test(
                  5'120U, 6'144U),
              "FP8 M16 WMMA predicate accepts padded-K 5120x6144");
  test.expect(q3x::kernels::use_sm87_fp8_m16_wmma_fixed_shape_test(
                  6'144U, 5'120U),
              "FP8 M16 WMMA predicate accepts 6144x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m16_wmma_fixed_shape_test(
                  12'288U, 5'120U),
              "FP8 M16 WMMA predicate accepts 12288x5120");
  test.expect(!q3x::kernels::use_sm87_fp8_m16_wmma_fixed_shape_test(
                  1'024U, 5'120U),
              "FP8 M16 WMMA predicate preserves 1024-row two-M8 fallback");
  test.expect(!q3x::kernels::use_sm87_fp8_m16_wmma_fixed_shape_test(
                  5'120U, 5'120U),
              "FP8 M16 WMMA predicate rejects unknown square shapes");
  test.expect(q3x::kernels::use_sm87_fp8_m1_row_pair_shape_test(
                  10'240U, 5'120U),
              "FP8 M1 row-pair accepts 10240x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m1_row_pair_shape_test(
                  5'120U, 6'144U),
              "FP8 M1 row-pair accepts 5120x6144");
  test.expect(q3x::kernels::use_sm87_fp8_m1_row_pair_shape_test(
                  6'144U, 5'120U),
              "FP8 M1 row-pair accepts 6144x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m1_row_pair_shape_test(
                  12'288U, 5'120U),
              "FP8 M1 row-pair accepts 12288x5120");
  test.expect(q3x::kernels::use_sm87_fp8_m1_row_pair_shape_test(
                  1'024U, 5'120U),
              "FP8 M1 row-pair accepts 1024x5120");
  test.expect(!q3x::kernels::use_sm87_fp8_m1_row_pair_shape_test(
                  5'120U, 5'120U),
              "FP8 M1 row-pair keeps unknown square shapes on cap2048");
  test.expect(!q3x::kernels::use_sm87_fp8_m1_row_pair_shape_test(
                  10'239U, 5'120U),
              "FP8 M1 row-pair rejects a near-miss row count");
  test.expect(!q3x::kernels::use_sm87_fp8_m1_row_pair_shape_test(
                  10'240U, 5'119U),
              "FP8 M1 row-pair rejects a near-miss K");
  test.expect(
      q3x::kernels::sm87_fp8_m1_row_quad_maximum_blocks_test(
          10'240U, 5'120U) == 1'536U,
      "FP8 M1 row-quad freezes 10240x5120 at cap1536");
  test.expect(
      q3x::kernels::sm87_fp8_m1_row_quad_maximum_blocks_test(
          5'120U, 6'144U) == 1'280U,
      "FP8 M1 row-quad freezes 5120x6144 at cap1280");
  test.expect(
      q3x::kernels::sm87_fp8_m1_row_quad_maximum_blocks_test(
          6'144U, 5'120U) == 768U,
      "FP8 M1 row-quad freezes 6144x5120 at cap768");
  test.expect(
      q3x::kernels::sm87_fp8_m1_row_quad_maximum_blocks_test(
          12'288U, 5'120U) == 2'048U,
      "FP8 M1 row-quad freezes 12288x5120 at cap2048");
  test.expect(
      q3x::kernels::sm87_fp8_m1_row_quad_maximum_blocks_test(
          1'024U, 5'120U) == 0U,
      "FP8 M1 row-quad preserves the small 1024x5120 row-pair");
  test.expect(
      q3x::kernels::sm87_fp8_m1_row_quad_maximum_blocks_test(
          5'120U, 5'120U) == 0U,
      "FP8 M1 row-quad preserves unknown shapes");
  test.expect(!q3x::kernels::use_sm87_fp8_m1_persistent_rows_test(1'023U),
              "FP8 M1 persistent rows keeps the small-row fallback");
  test.expect(q3x::kernels::use_sm87_fp8_m1_persistent_rows_test(1'024U),
              "FP8 M1 persistent rows accepts the smallest real projection");
  test.expect(q3x::kernels::use_sm87_fp8_m1_persistent_rows_test(65'537U),
              "FP8 M1 persistent rows accepts grid-stride large-row shapes");
  test.expect(!q3x::kernels::use_sm87_fp8_m2_persistent_rows_test(1'023U),
              "FP8 M2 persistent rows keeps the small-row fallback");
  test.expect(q3x::kernels::use_sm87_fp8_m2_persistent_rows_test(1'024U),
              "FP8 M2 persistent rows accepts the smallest real projection");
  test.expect(q3x::kernels::use_sm87_fp8_m2_persistent_rows_test(65'537U),
              "FP8 M2 persistent rows accepts grid-stride large-row shapes");

  auto* const exact_packed =
      reinterpret_cast<const std::uint8_t*>(0x1'0000'0000ULL);
  auto* const exact_scales =
      reinterpret_cast<const std::uint8_t*>(0x2'0000'0000ULL);
  auto* const exact_activation =
      reinterpret_cast<const std::uint16_t*>(0x3'0000'0000ULL);
  auto* const exact_output =
      reinterpret_cast<std::uint16_t*>(0x4'0000'0000ULL);
  const auto exact_shape_status = [&](const std::size_t rows,
                                      const std::size_t columns) noexcept {
    return static_cast<cudaError_t>(q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_row_quad_exact_shape_test_cuda(
            exact_packed, exact_scales, 1.0F, exact_activation, rows, columns,
            exact_output));
  };
  test.expect(exact_shape_status(17'407U, 5'120U) ==
                  cudaErrorInvalidValue,
              "NVFP4 M1 exact-shape launcher rejects a near-miss row count");
  test.expect(exact_shape_status(17'408U, 5'104U) ==
                  cudaErrorInvalidValue,
              "NVFP4 M1 exact-shape launcher rejects a near-miss K");
  test.expect(exact_shape_status(5'120U, 5'120U) == cudaErrorInvalidValue,
              "NVFP4 M1 exact-shape launcher rejects an unknown shape");
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

[[nodiscard]] bool nvfp4_m1_scale_codebook_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M1_SCALE_CODEBOOK_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m1_row_pair_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M1_ROW_PAIR_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m1_row_quad_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M1_ROW_QUAD_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m1_exact_shape_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M1_EXACT_SHAPE_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m1_down_dual_iteration_performance_enabled()
    noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M1_DOWN_DUAL_ITERATION_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m2_scale_codebook_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M2_SCALE_CODEBOOK_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m2_row_pair_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M2_ROW_PAIR_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m2_row_quad_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M2_ROW_QUAD_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m1_grid_cap_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M1_GRID_CAP_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m8_fixed_shape_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M8_FIXED_SHAPE_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m16_wmma_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M16_WMMA_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool nvfp4_m16_k128_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_NVFP4_M16_K128_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_row_pair_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_ROW_PAIR_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m8_fixed_shape_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M8_FIXED_SHAPE_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m16_wmma_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M16_WMMA_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m1_grid_cap_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M1_GRID_CAP_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m1_row_pair_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M1_ROW_PAIR_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m1_row_quad_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M1_ROW_QUAD_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m1_swizzled_codebook_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M1_SWIZZLED_CODEBOOK_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m1_kv_pair_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M1_KV_PAIR_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m2_grid_cap_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M2_GRID_CAP_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m2_row_pair_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M2_ROW_PAIR_PERF");
  return value != nullptr && value[0] != '\0' &&
         !(value[0] == '0' && value[1] == '\0');
}

[[nodiscard]] bool fp8_m2_row_quad_performance_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SM87_FP8_M2_ROW_QUAD_PERF");
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

constexpr std::array<std::size_t, 6U> kFp8M1GridCaps{{
    96U, 192U, 384U, 768U, 1'024U, 2'048U,
}};

struct Fp8M1GridCapMeasurements {
  std::array<float, kFp8M1GridCaps.size()> baseline_milliseconds{};
  std::array<float, kFp8M1GridCaps.size()> candidate_milliseconds{};
  std::array<bool, kFp8M1GridCaps.size()> bitwise_equal{};
};

[[nodiscard]] Fp8M1GridCapMeasurements benchmark_fp8_m1_grid_cap_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 40;
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 12U> kFiniteCodes{{
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U,
      0x3cU, 0x40U, 0xb8U, 0x70U, 0x78U, 0xfeU,
  }};
  Fp8M1GridCapMeasurements measurements;
  measurements.baseline_milliseconds.fill(
      std::numeric_limits<float>::quiet_NaN());
  measurements.candidate_milliseconds.fill(
      std::numeric_limits<float>::quiet_NaN());

  std::vector<std::uint8_t> host_weights(rows * columns);
  for (std::size_t index = 0U; index < host_weights.size(); ++index) {
    host_weights[index] =
        kFiniteCodes[(index * 7U + index / columns * 5U + 1U) %
                     kFiniteCodes.size()];
  }
  std::vector<std::uint16_t> host_activation(columns);
  for (std::size_t column = 0U; column < columns; ++column) {
    const int centered = static_cast<int>((column * 13U + 3U) % 127U) - 63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activation.allocate(host_activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(weights.get(), host_weights.data(),
                                       host_weights.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize mixed finite weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize mixed activation");
  if (!ready) {
    return measurements;
  }

  const auto launch_baseline = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_fp8_w8a16_gemv_bf16_vector_uncapped_test_cuda(
            weights.get(), kWeightScale, activation.get(), rows, columns,
            baseline_output.get(), static_cast<void*>(stream));
  };
  ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                       label + " correctness baseline launch");
  std::vector<std::uint16_t> baseline(rows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), baseline_output.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy correctness baseline");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " baseline synchronize");
  if (!ready) {
    return measurements;
  }

  for (std::size_t cap_index = 0U; cap_index < kFp8M1GridCaps.size();
       ++cap_index) {
    const std::size_t grid_cap = kFp8M1GridCaps[cap_index];
    const auto launch_candidate = [&]() noexcept -> int {
      return q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_grid_cap_test_cuda(
          weights.get(), kWeightScale, activation.get(), rows, columns,
          candidate_output.get(), grid_cap, static_cast<void*>(stream));
    };
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                         label + " correctness candidate launch cap=" +
                             std::to_string(grid_cap));
    std::vector<std::uint16_t> candidate(rows);
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy correctness candidate cap=" +
                             std::to_string(grid_cap));
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         label + " correctness synchronize cap=" +
                             std::to_string(grid_cap));
    if (!ready) {
      return measurements;
    }
    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < baseline.size(); ++index) {
      mismatches += baseline[index] != candidate[index] ? 1U : 0U;
    }
    measurements.bitwise_equal[cap_index] = mismatches == 0U;
    std::cout << "FP8_M1_GRID_CAP_DIFF: " << label
              << " grid_cap=" << grid_cap
              << " candidate_vs_uncapped_bf16=" << mismatches << '/'
              << baseline.size() << '\n';
    test.expect(mismatches == 0U,
                label + " cap=" + std::to_string(grid_cap) +
                    " matches every uncapped BF16 bit");

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           label + " candidate warmup cap=" +
                               std::to_string(grid_cap));
    }
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         label + " warmup synchronize cap=" +
                             std::to_string(grid_cap));
    if (!ready) {
      return measurements;
    }

    const float baseline_first = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        label + " baseline pass 1 cap=" + std::to_string(grid_cap));
    const float candidate_first = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        label + " candidate pass 1 cap=" + std::to_string(grid_cap));
    const float candidate_second = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        label + " candidate pass 2 cap=" + std::to_string(grid_cap));
    const float baseline_second = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        label + " baseline pass 2 cap=" + std::to_string(grid_cap));
    if (!(std::isfinite(baseline_first) &&
          std::isfinite(baseline_second) &&
          std::isfinite(candidate_first) &&
          std::isfinite(candidate_second))) {
      return measurements;
    }
    const float baseline_average =
        (baseline_first + baseline_second) * 0.5F;
    const float candidate_average =
        (candidate_first + candidate_second) * 0.5F;
    measurements.baseline_milliseconds[cap_index] = baseline_average;
    measurements.candidate_milliseconds[cap_index] = candidate_average;
    const float speedup = baseline_average / candidate_average;
    std::cout << "PERF_FP8_M1_GRID_CAP: " << label
              << " grid_cap=" << grid_cap
              << " baseline_pass1_ms=" << baseline_first
              << " candidate_pass1_ms=" << candidate_first
              << " candidate_pass2_ms=" << candidate_second
              << " baseline_pass2_ms=" << baseline_second
              << " baseline_average_ms=" << baseline_average
              << " candidate_average_ms=" << candidate_average
              << " speedup=" << speedup
              << " uplift_percent=" << (speedup - 1.0F) * 100.0F << '\n';
  }
  return measurements;
}

void run_fp8_m1_large_row_grid_stride_coverage(
    TestContext& test, cudaStream_t stream, const std::size_t grid_cap) {
  constexpr std::size_t kRows = 65'537U;
  constexpr std::size_t kColumns = 1'024U;
  constexpr float kWeightScale = 1.0F / 64.0F;
  const std::string label = "FP8 M1 grid-cap >65535-row coverage";
  std::vector<std::uint16_t> host_activation(
      kColumns, encode_bf16(1.0F / 256.0F));
  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(kRows * kColumns),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activation.allocate(kColumns),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(kRows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kRows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(weights.get(), 0x38, kRows * kColumns,
                                       stream),
                       label + " initialize weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_gemv_bf16_vector_uncapped_test_cuda(
              weights.get(), kWeightScale, activation.get(), kRows, kColumns,
              baseline_output.get(), static_cast<void*>(stream))),
      label + " launch uncapped baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_gemv_bf16_grid_cap_test_cuda(
                               weights.get(), kWeightScale, activation.get(),
                               kRows, kColumns, candidate_output.get(), grid_cap,
                               static_cast<void*>(stream))),
                       label + " launch capped candidate");
  std::vector<std::uint16_t> baseline(kRows);
  std::vector<std::uint16_t> candidate(kRows);
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
                                label + " synchronize");
  if (!ready) {
    return;
  }
  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    mismatches += baseline[index] != candidate[index] ? 1U : 0U;
  }
  std::cout << "FP8_M1_GRID_CAP_LARGE_ROWS_DIFF: grid_cap=" << grid_cap
            << " candidate_vs_uncapped_bf16=" << mismatches << '/'
            << baseline.size() << '\n';
  test.expect(mismatches == 0U,
              label + " covers every row with identical BF16 output");
}

void run_optional_fp8_m1_grid_cap_performance(TestContext& test,
                                               cudaStream_t stream) {
  if (!fp8_m1_grid_cap_performance_enabled()) {
    std::cout << "SKIP: FP8 M1 persistent-row grid-cap segment; set "
                 "Q3X_RUN_SM87_FP8_M1_GRID_CAP_PERF=1 to enable\n";
    return;
  }
  constexpr float kMinimumWeightedSpeedup = 1.03F;
  constexpr float kMinimumPerShapeSpeedup = 0.98F;
  struct Shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t calls_per_prompt;
    const char* label;
  };
  constexpr std::array<Shape, 6U> kShapes{{
      {10'240U, 5'120U, 48U, "FP8 M1 linear QKV 10240x5120"},
      {5'120U, 6'144U, 64U, "FP8 M1 projection 5120x6144"},
      {6'144U, 5'120U, 48U, "FP8 M1 projection 6144x5120"},
      {12'288U, 5'120U, 16U, "FP8 M1 linear QKV 12288x5120"},
      {1'024U, 5'120U, 32U, "FP8 M1 small projection 1024x5120"},
      {5'120U, 5'120U, 0U, "FP8 M1 square validation 5120x5120"},
  }};
  std::array<Fp8M1GridCapMeasurements, kShapes.size()> measurements{};
  measurements[0] = benchmark_fp8_m1_grid_cap_shape(
      test, stream, kShapes[0].rows, kShapes[0].columns, kShapes[0].label);
  measurements[1] = benchmark_fp8_m1_grid_cap_shape(
      test, stream, kShapes[1].rows, kShapes[1].columns, kShapes[1].label);

  std::size_t selected_cap_index = kFp8M1GridCaps.size();
  double selected_core_speedup = 0.0;
  for (std::size_t cap_index = 0U; cap_index < kFp8M1GridCaps.size();
       ++cap_index) {
    const float baseline0 =
        measurements[0].baseline_milliseconds[cap_index];
    const float baseline1 =
        measurements[1].baseline_milliseconds[cap_index];
    const float candidate0 =
        measurements[0].candidate_milliseconds[cap_index];
    const float candidate1 =
        measurements[1].candidate_milliseconds[cap_index];
    const bool finite = std::isfinite(baseline0) &&
                        std::isfinite(baseline1) &&
                        std::isfinite(candidate0) &&
                        std::isfinite(candidate1);
    const bool bitwise = measurements[0].bitwise_equal[cap_index] &&
                         measurements[1].bitwise_equal[cap_index];
    const double weighted_baseline = 48.0 * baseline0 + 64.0 * baseline1;
    const double weighted_candidate = 48.0 * candidate0 + 64.0 * candidate1;
    const double speedup = weighted_baseline / weighted_candidate;
    const bool no_regression = finite && candidate0 <= baseline0 / 0.98F &&
                               candidate1 <= baseline1 / 0.98F;
    std::cout << "PERF_FP8_M1_GRID_CAP_CORE: grid_cap="
              << kFp8M1GridCaps[cap_index]
              << " weighted_baseline_ms=" << weighted_baseline
              << " weighted_candidate_ms=" << weighted_candidate
              << " speedup=" << speedup
              << " bitwise=" << (bitwise ? "true" : "false")
              << " no_regression=" << (no_regression ? "true" : "false")
              << '\n';
    if (finite && bitwise && no_regression &&
        speedup > selected_core_speedup) {
      selected_core_speedup = speedup;
      selected_cap_index = cap_index;
    }
  }
  const bool selected = selected_cap_index < kFp8M1GridCaps.size();
  test.expect(selected, "FP8 M1 grid-cap sweep selects a valid core cap");
  if (!selected) {
    return;
  }
  const std::size_t sweep_cap = kFp8M1GridCaps[selected_cap_index];
  std::cout << "PERF_FP8_M1_GRID_CAP_SWEEP_BEST: grid_cap=" << sweep_cap
            << " core_speedup=" << selected_core_speedup << '\n';

  // The production cap was frozen after the first complete sweep. Retain the
  // sweep as a diagnostic, but gate the exact value compiled into dispatch so
  // run-to-run clock noise cannot silently validate a different cap.
  constexpr std::size_t kProductionCapIndex = kFp8M1GridCaps.size() - 1U;
  selected_cap_index = kProductionCapIndex;
  const std::size_t selected_cap = kFp8M1GridCaps[selected_cap_index];
  const double production_core_baseline =
      48.0 * measurements[0].baseline_milliseconds[selected_cap_index] +
      64.0 * measurements[1].baseline_milliseconds[selected_cap_index];
  const double production_core_candidate =
      48.0 * measurements[0].candidate_milliseconds[selected_cap_index] +
      64.0 * measurements[1].candidate_milliseconds[selected_cap_index];
  const double production_core_speedup =
      production_core_baseline / production_core_candidate;
  const bool production_core_gate =
      measurements[0].bitwise_equal[selected_cap_index] &&
      measurements[1].bitwise_equal[selected_cap_index] &&
      std::isfinite(production_core_speedup) &&
      production_core_speedup >= kMinimumWeightedSpeedup;
  std::cout << "PERF_FP8_M1_GRID_CAP_SELECTED: grid_cap=" << selected_cap
            << " production_core_speedup=" << production_core_speedup
            << " gate=" << (production_core_gate ? "PASS" : "FAIL") << '\n';
  test.expect(production_core_gate,
              "FP8 M1 production cap clears the frozen core gate");
  if (!production_core_gate) {
    return;
  }

  for (std::size_t shape_index = 2U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_fp8_m1_grid_cap_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }
  double weighted_baseline = 0.0;
  double weighted_candidate = 0.0;
  bool all_bitwise = true;
  bool all_finite = true;
  bool no_shape_regression = true;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    const float baseline =
        measurements[shape_index].baseline_milliseconds[selected_cap_index];
    const float candidate =
        measurements[shape_index].candidate_milliseconds[selected_cap_index];
    const bool finite = std::isfinite(baseline) && std::isfinite(candidate);
    const float speedup = baseline / candidate;
    all_finite = all_finite && finite;
    all_bitwise = all_bitwise &&
                  measurements[shape_index].bitwise_equal[selected_cap_index];
    no_shape_regression = no_shape_regression && finite &&
                          speedup >= kMinimumPerShapeSpeedup;
    weighted_baseline +=
        static_cast<double>(kShapes[shape_index].calls_per_prompt) * baseline;
    weighted_candidate +=
        static_cast<double>(kShapes[shape_index].calls_per_prompt) * candidate;
    std::cout << "PERF_FP8_M1_GRID_CAP_VALIDATION: "
              << kShapes[shape_index].label
              << " grid_cap=" << selected_cap
              << " baseline_ms=" << baseline
              << " candidate_ms=" << candidate
              << " speedup=" << speedup
              << " minimum_speedup=" << kMinimumPerShapeSpeedup
              << " gate="
              << (finite && speedup >= kMinimumPerShapeSpeedup ? "PASS"
                                                               : "FAIL")
              << '\n';
  }
  const double weighted_speedup = weighted_baseline / weighted_candidate;
  const bool aggregate_gate =
      all_finite && all_bitwise && no_shape_regression &&
      std::isfinite(weighted_speedup) &&
      weighted_speedup >= kMinimumWeightedSpeedup;
  std::cout << "PERF_FP8_M1_GRID_CAP_AGGREGATE: selected_cap="
            << selected_cap << " weighted_baseline_ms=" << weighted_baseline
            << " weighted_candidate_ms=" << weighted_candidate
            << " speedup=" << weighted_speedup
            << " required_speedup=" << kMinimumWeightedSpeedup
            << " all_bitwise=" << (all_bitwise ? "true" : "false")
            << " no_shape_regression="
            << (no_shape_regression ? "true" : "false")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "FP8 M1 selected grid cap clears every production gate");
  if (aggregate_gate) {
    run_fp8_m1_large_row_grid_stride_coverage(test, stream, selected_cap);
  }
}

constexpr std::array<std::size_t, 6U> kFp8M2GridCaps{{
    96U, 192U, 384U, 768U, 1'024U, 2'048U,
}};

struct Fp8M2GridCapMeasurements {
  std::array<float, kFp8M2GridCaps.size()> baseline_milliseconds{};
  std::array<float, kFp8M2GridCaps.size()> candidate_milliseconds{};
  std::array<bool, kFp8M2GridCaps.size()> bitwise_equal{};
};

[[nodiscard]] Fp8M2GridCapMeasurements benchmark_fp8_m2_grid_cap_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokenCount = 2U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr std::array<std::uint8_t, 12U> kFiniteCodes{{
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U,
      0x3cU, 0x40U, 0xb8U, 0x70U, 0x78U, 0xfeU,
  }};
  Fp8M2GridCapMeasurements measurements;
  measurements.baseline_milliseconds.fill(
      std::numeric_limits<float>::quiet_NaN());
  measurements.candidate_milliseconds.fill(
      std::numeric_limits<float>::quiet_NaN());

  std::vector<std::uint8_t> host_weights(rows * columns);
  for (std::size_t index = 0U; index < host_weights.size(); ++index) {
    host_weights[index] =
        kFiniteCodes[(index * 7U + index / columns * 5U + 1U) %
                     kFiniteCodes.size()];
  }
  std::vector<std::uint16_t> host_activations(kTokenCount * columns);
  for (std::size_t token = 0U; token < kTokenCount; ++token) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered = static_cast<int>(
                               (column * 13U + token * 29U + 3U) % 127U) -
                           63;
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
  ready = ready &&
          test.cuda_ok(activations.allocate(host_activations.size()),
                       label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokenCount * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokenCount * rows),
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
    return measurements;
  }

  const auto launch_baseline = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_fp8_w8a16_small_m2_uncapped_test_cuda(
            weights.get(), kWeightScale, activations.get(), rows, columns,
            baseline_output.get(), static_cast<void*>(stream));
  };
  ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                       label + " correctness baseline launch");
  std::vector<std::uint16_t> baseline(kTokenCount * rows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), baseline_output.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy correctness baseline");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " baseline synchronize");
  if (!ready) {
    return measurements;
  }

  for (std::size_t cap_index = 0U; cap_index < kFp8M2GridCaps.size();
       ++cap_index) {
    const std::size_t grid_cap = kFp8M2GridCaps[cap_index];
    const auto launch_candidate = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_fp8_w8a16_small_m2_grid_cap_test_cuda(
              weights.get(), kWeightScale, activations.get(), rows, columns,
              candidate_output.get(), grid_cap, static_cast<void*>(stream));
    };
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                         label + " correctness candidate launch cap=" +
                             std::to_string(grid_cap));
    std::vector<std::uint16_t> candidate(kTokenCount * rows);
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy correctness candidate cap=" +
                             std::to_string(grid_cap));
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         label + " correctness synchronize cap=" +
                             std::to_string(grid_cap));
    if (!ready) {
      return measurements;
    }
    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < baseline.size(); ++index) {
      mismatches += baseline[index] != candidate[index] ? 1U : 0U;
    }
    measurements.bitwise_equal[cap_index] = mismatches == 0U;
    std::cout << "FP8_M2_GRID_CAP_DIFF: " << label
              << " grid_cap=" << grid_cap
              << " candidate_vs_uncapped_bf16=" << mismatches << '/'
              << baseline.size() << '\n';
    test.expect(mismatches == 0U,
                label + " cap=" + std::to_string(grid_cap) +
                    " matches every uncapped M2 BF16 bit");

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           label + " candidate warmup cap=" +
                               std::to_string(grid_cap));
    }
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         label + " warmup synchronize cap=" +
                             std::to_string(grid_cap));
    if (!ready) {
      return measurements;
    }

    double baseline_total = 0.0;
    double candidate_total = 0.0;
    bool finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_suffix =
          " cap=" + std::to_string(grid_cap) +
          " round=" + std::to_string(round + 1);
      const float baseline_first = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          label + " baseline pass 1" + round_suffix);
      const float candidate_first = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          label + " candidate pass 1" + round_suffix);
      const float candidate_second = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          label + " candidate pass 2" + round_suffix);
      const float baseline_second = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          label + " baseline pass 2" + round_suffix);
      const bool round_finite =
          std::isfinite(baseline_first) &&
          std::isfinite(candidate_first) &&
          std::isfinite(candidate_second) &&
          std::isfinite(baseline_second);
      finite = finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_FP8_M2_GRID_CAP_ROUND: " << label
                << " grid_cap=" << grid_cap << " round=" << round + 1
                << " baseline_pass1_ms=" << baseline_first
                << " candidate_pass1_ms=" << candidate_first
                << " candidate_pass2_ms=" << candidate_second
                << " baseline_pass2_ms=" << baseline_second << '\n';
    }
    if (!finite) {
      return measurements;
    }
    constexpr double kPassesPerCandidate =
        2.0 * static_cast<double>(kMeasurementRounds);
    const float baseline_average =
        static_cast<float>(baseline_total / kPassesPerCandidate);
    const float candidate_average =
        static_cast<float>(candidate_total / kPassesPerCandidate);
    measurements.baseline_milliseconds[cap_index] = baseline_average;
    measurements.candidate_milliseconds[cap_index] = candidate_average;
    const float speedup = baseline_average / candidate_average;
    std::cout << "PERF_FP8_M2_GRID_CAP: " << label
              << " grid_cap=" << grid_cap
              << " baseline_average_ms=" << baseline_average
              << " candidate_average_ms=" << candidate_average
              << " speedup=" << speedup
              << " uplift_percent=" << (speedup - 1.0F) * 100.0F << '\n';
  }
  return measurements;
}

void run_fp8_m2_large_row_grid_stride_coverage(
    TestContext& test, cudaStream_t stream, const std::size_t grid_cap) {
  constexpr std::size_t kTokenCount = 2U;
  constexpr std::size_t kRows = 65'537U;
  constexpr std::size_t kColumns = 1'024U;
  constexpr float kWeightScale = 1.0F / 64.0F;
  const std::string label = "FP8 M2 grid-cap >65535-row coverage";
  std::vector<std::uint16_t> host_activations(
      kTokenCount * kColumns, encode_bf16(1.0F / 256.0F));
  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(kRows * kColumns),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready &&
          test.cuda_ok(baseline_output.allocate(kTokenCount * kRows),
                       label + " allocate baseline output");
  ready = ready &&
          test.cuda_ok(candidate_output.allocate(kTokenCount * kRows),
                       label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(weights.get(), 0x38, kRows * kColumns,
                                       stream),
                       label + " initialize weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          launch_sm87_fp8_w8a16_small_m2_uncapped_test_cuda(
              weights.get(), kWeightScale, activations.get(), kRows, kColumns,
              baseline_output.get(), static_cast<void*>(stream))),
      label + " launch uncapped baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::kernels::
                           launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
                               weights.get(), kWeightScale, activations.get(),
                               kTokenCount, kRows, kColumns,
                               candidate_output.get(),
                               static_cast<void*>(stream))),
                       label + " launch production candidate");
  std::vector<std::uint16_t> baseline(kTokenCount * kRows);
  std::vector<std::uint16_t> candidate(kTokenCount * kRows);
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
                                label + " synchronize");
  if (!ready) {
    return;
  }
  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    mismatches += baseline[index] != candidate[index] ? 1U : 0U;
  }
  std::cout << "FP8_M2_GRID_CAP_LARGE_ROWS_DIFF: grid_cap=" << grid_cap
            << " production_vs_uncapped_bf16=" << mismatches << '/'
            << baseline.size() << '\n';
  test.expect(mismatches == 0U,
              label + " covers every token row with identical BF16 output");
}

void run_optional_fp8_m2_grid_cap_performance(TestContext& test,
                                               cudaStream_t stream) {
  if (!fp8_m2_grid_cap_performance_enabled()) {
    std::cout << "SKIP: FP8 M2 persistent-row grid-cap segment; set "
                 "Q3X_RUN_SM87_FP8_M2_GRID_CAP_PERF=1 to enable\n";
    return;
  }
  constexpr float kMinimumWeightedSpeedup = 1.03F;
  constexpr float kMinimumPerShapeSpeedup = 0.98F;
  struct Shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t calls_per_prompt;
    const char* label;
  };
  constexpr std::array<Shape, 6U> kShapes{{
      {10'240U, 5'120U, 48U, "FP8 M2 linear QKV 10240x5120"},
      {5'120U, 6'144U, 64U, "FP8 M2 projection 5120x6144"},
      {6'144U, 5'120U, 48U, "FP8 M2 projection 6144x5120"},
      {12'288U, 5'120U, 16U, "FP8 M2 linear QKV 12288x5120"},
      {1'024U, 5'120U, 32U, "FP8 M2 small projection 1024x5120"},
      {5'120U, 5'120U, 0U, "FP8 M2 square validation 5120x5120"},
  }};
  std::array<Fp8M2GridCapMeasurements, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_fp8_m2_grid_cap_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }

  std::size_t selected_cap_index = kFp8M2GridCaps.size();
  double selected_weighted_speedup = 0.0;
  std::array<double, kFp8M2GridCaps.size()> weighted_speedups{};
  std::array<bool, kFp8M2GridCaps.size()> aggregate_gates{};
  for (std::size_t cap_index = 0U; cap_index < kFp8M2GridCaps.size();
       ++cap_index) {
    double weighted_baseline = 0.0;
    double weighted_candidate = 0.0;
    bool all_bitwise = true;
    bool all_finite = true;
    bool no_shape_regression = true;
    for (std::size_t shape_index = 0U; shape_index < kShapes.size();
         ++shape_index) {
      const float baseline =
          measurements[shape_index].baseline_milliseconds[cap_index];
      const float candidate =
          measurements[shape_index].candidate_milliseconds[cap_index];
      const bool finite = std::isfinite(baseline) && std::isfinite(candidate);
      const float speedup = baseline / candidate;
      all_finite = all_finite && finite;
      all_bitwise = all_bitwise &&
                    measurements[shape_index].bitwise_equal[cap_index];
      no_shape_regression = no_shape_regression && finite &&
                            speedup >= kMinimumPerShapeSpeedup;
      weighted_baseline +=
          static_cast<double>(kShapes[shape_index].calls_per_prompt) *
          baseline;
      weighted_candidate +=
          static_cast<double>(kShapes[shape_index].calls_per_prompt) *
          candidate;
      std::cout << "PERF_FP8_M2_GRID_CAP_VALIDATION: "
                << kShapes[shape_index].label
                << " grid_cap=" << kFp8M2GridCaps[cap_index]
                << " baseline_ms=" << baseline
                << " candidate_ms=" << candidate
                << " speedup=" << speedup
                << " minimum_speedup=" << kMinimumPerShapeSpeedup
                << " gate="
                << (finite && speedup >= kMinimumPerShapeSpeedup ? "PASS"
                                                                 : "FAIL")
                << '\n';
    }
    const double weighted_speedup = weighted_baseline / weighted_candidate;
    const bool aggregate_gate =
        all_finite && all_bitwise && no_shape_regression &&
        std::isfinite(weighted_speedup) &&
        weighted_speedup >= kMinimumWeightedSpeedup;
    weighted_speedups[cap_index] = weighted_speedup;
    aggregate_gates[cap_index] = aggregate_gate;
    std::cout << "PERF_FP8_M2_GRID_CAP_AGGREGATE: grid_cap="
              << kFp8M2GridCaps[cap_index]
              << " weighted_baseline_ms=" << weighted_baseline
              << " weighted_candidate_ms=" << weighted_candidate
              << " speedup=" << weighted_speedup
              << " required_speedup=" << kMinimumWeightedSpeedup
              << " all_bitwise=" << (all_bitwise ? "true" : "false")
              << " no_shape_regression="
              << (no_shape_regression ? "true" : "false")
              << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
    if (aggregate_gate && weighted_speedup > selected_weighted_speedup) {
      selected_cap_index = cap_index;
      selected_weighted_speedup = weighted_speedup;
    }
  }
  const bool sweep_selected = selected_cap_index < kFp8M2GridCaps.size();
  test.expect(sweep_selected,
              "FP8 M2 sweep selects a cap clearing every production gate");
  if (!sweep_selected) {
    std::cout << "PERF_FP8_M2_GRID_CAP_SWEEP_BEST: gate=FAIL\n";
    return;
  }
  std::cout << "PERF_FP8_M2_GRID_CAP_SWEEP_BEST: grid_cap="
            << kFp8M2GridCaps[selected_cap_index]
            << " weighted_speedup=" << selected_weighted_speedup << '\n';

  // Freeze and gate the exact cap compiled into production. The diagnostic
  // sweep remains useful, but clock noise must not validate another cap.
  constexpr std::size_t kProductionCapIndex = kFp8M2GridCaps.size() - 1U;
  selected_cap_index = kProductionCapIndex;
  const std::size_t selected_cap = kFp8M2GridCaps[selected_cap_index];
  const bool production_gate = aggregate_gates[selected_cap_index];
  std::cout << "PERF_FP8_M2_GRID_CAP_SELECTED: grid_cap=" << selected_cap
            << " weighted_speedup=" << weighted_speedups[selected_cap_index]
            << " gate=" << (production_gate ? "PASS" : "FAIL") << '\n';
  test.expect(production_gate,
              "FP8 M2 frozen production cap clears every production gate");
  if (!production_gate) {
    return;
  }
  run_fp8_m2_large_row_grid_stride_coverage(test, stream, selected_cap);
}

enum class Fp8M2CodeDistribution {
  kCheckpointLike,
  kSameBankStress,
};

[[nodiscard]] const char* fp8_m2_code_distribution_name(
    const Fp8M2CodeDistribution distribution) noexcept {
  return distribution == Fp8M2CodeDistribution::kCheckpointLike
             ? "checkpoint_like"
             : "same_bank_stress";
}

void fill_fp8_m2_code_distribution(
    std::vector<std::uint8_t>& weights, const std::size_t rows,
    const std::size_t columns, const Fp8M2CodeDistribution distribution) {
  constexpr std::array<std::uint8_t, 32U> kCheckpointLikeCodes{{
      0x00U, 0x00U, 0x80U, 0x00U, 0x10U, 0x90U, 0x18U, 0x98U,
      0x20U, 0xa0U, 0x28U, 0xa8U, 0x30U, 0xb0U, 0x34U, 0xb4U,
      0x38U, 0xb8U, 0x38U, 0xb8U, 0x39U, 0xb9U, 0x3aU, 0xbaU,
      0x3cU, 0xbcU, 0x40U, 0xc0U, 0x30U, 0x34U, 0xb0U, 0xb4U,
  }};
  constexpr std::array<std::uint8_t, 8U> kSameBankCodes{{
      0x18U, 0x38U, 0x58U, 0x78U,
      0x98U, 0xb8U, 0xd8U, 0xf8U,
  }};
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const std::size_t index = row * columns + column;
      if (distribution == Fp8M2CodeDistribution::kCheckpointLike) {
        weights[index] = kCheckpointLikeCodes[
            (column * 5U + row * 11U + (column >> 4U)) %
            kCheckpointLikeCodes.size()];
      } else {
        // Every code has identical low five bits, so simultaneous lookup
        // lanes contend for one shared-memory bank while decoding different
        // finite E4M3FN values and signs.
        weights[index] = kSameBankCodes[
            (column + row * 3U + (column >> 5U)) % kSameBankCodes.size()];
      }
    }
  }
}

struct Fp8KvPairKernelResources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

struct Fp8KvPairTiming {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
};

template <std::size_t Count>
[[nodiscard]] float median_fp8_kv_pair_timing(
    std::array<float, Count> values) {
  static_assert(Count != 0U);
  std::sort(values.begin(), values.end());
  if constexpr ((Count % 2U) == 0U) {
    return (values[Count / 2U - 1U] + values[Count / 2U]) * 0.5F;
  }
  return values[Count / 2U];
}

void run_fp8_m1_kv_pair_correctness_and_optional_performance(
    TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kRows = 1'024U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::size_t kGuardElements = 16U;
  constexpr float kKeyWeightScale = 1.0F / 64.0F;
  constexpr float kValueWeightScale = 1.0F / 96.0F;
  constexpr int kWarmupIterations = 8;
  constexpr int kMeasuredIterations = 80;
  constexpr int kMeasurementRounds = 3;
  constexpr std::uint16_t kKeyGuard = 0xa5a5U;
  constexpr std::uint16_t kValueGuard = 0x5a5aU;
  constexpr std::size_t kComparisonGridCap = 64U;
  constexpr std::size_t kSelectedGridCap = 128U;
  constexpr std::array<Fp8M2CodeDistribution, 2U> kDistributions{{
      Fp8M2CodeDistribution::kCheckpointLike,
      Fp8M2CodeDistribution::kSameBankStress,
  }};
  const std::string label = "FP8 M1 exact K/V pair 1024x5120";

  Fp8KvPairKernelResources row_quad_resources{};
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          query_sm87_fp8_w8a16_m1_kv_pair_row_quad_resources_test_cuda(
              &row_quad_resources.registers_per_thread,
              &row_quad_resources.static_shared_bytes,
              &row_quad_resources.local_bytes,
              &row_quad_resources.maximum_threads_per_block,
              &row_quad_resources.active_blocks_per_sm)),
      label + " query production cross-matrix row-quad resources");
  if (!ready) {
    return;
  }
  const auto report_resources = [&](const char* const candidate,
                                    const Fp8KvPairKernelResources& resources,
                                    const std::size_t outputs_per_cta) {
    std::cout << "PERF_FP8_M1_KV_PAIR_RESOURCES: candidate=" << candidate
              << " outputs_per_cta=" << outputs_per_cta
              << " registers_per_thread=" << resources.registers_per_thread
              << " static_shared_bytes=" << resources.static_shared_bytes
              << " local_bytes=" << resources.local_bytes
              << " maximum_threads_per_block="
              << resources.maximum_threads_per_block
              << " active_blocks_per_sm=" << resources.active_blocks_per_sm
              << '\n';
    test.expect(resources.registers_per_thread > 0 &&
                    resources.static_shared_bytes > 0U &&
                    resources.maximum_threads_per_block >= 256 &&
                    resources.active_blocks_per_sm > 0,
                label + " " + candidate + " reports usable resources");
  };
  report_resources("production_cross_matrix_row_quad_2K2V",
                   row_quad_resources, 4U);
  const bool selected_resource_gate =
      row_quad_resources.registers_per_thread <= 64 &&
      row_quad_resources.local_bytes == 0U &&
      row_quad_resources.active_blocks_per_sm >= 4;
  std::cout << "PERF_FP8_M1_KV_PAIR_RESOURCE_GATE: candidate="
               "production_cross_matrix_row_quad_2K2V"
            << " maximum_registers_per_thread=64 minimum_active_blocks_per_sm=4"
            << " require_zero_local_bytes=true gate="
            << (selected_resource_gate ? "PASS" : "FAIL") << '\n';
  test.expect(selected_resource_gate,
              label + " selected production kernel clears resource gate");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          query_sm87_fp8_w8a16_m1_kv_pair_row_quad_resources_test_cuda(
              nullptr, &row_quad_resources.static_shared_bytes,
              &row_quad_resources.local_bytes,
              &row_quad_resources.maximum_threads_per_block,
              &row_quad_resources.active_blocks_per_sm)) ==
          cudaErrorInvalidValue,
      label + " resource query rejects a null destination");

  std::vector<std::uint8_t> host_key_weights(kRows * kColumns);
  std::vector<std::uint8_t> host_value_weights(kRows * kColumns);
  std::vector<std::uint16_t> host_activation(kColumns);
  const auto fill_mixed_activation = [&]() {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const int centered =
          static_cast<int>((column * 17U + (column >> 3U) * 5U + 7U) %
                           127U) -
          63;
      host_activation[column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
  };
  fill_mixed_activation();

  DeviceBuffer<std::uint8_t> key_weights;
  DeviceBuffer<std::uint8_t> value_weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_key_output;
  DeviceBuffer<std::uint16_t> baseline_value_output;
  DeviceBuffer<std::uint16_t> candidate_key_storage;
  DeviceBuffer<std::uint16_t> candidate_value_storage;
  ready = test.cuda_ok(key_weights.allocate(host_key_weights.size()),
                       label + " allocate key weights");
  ready = ready && test.cuda_ok(value_weights.allocate(host_value_weights.size()),
                                label + " allocate value weights");
  ready = ready && test.cuda_ok(activation.allocate(host_activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_key_output.allocate(kRows),
                                label + " allocate baseline key output");
  ready = ready && test.cuda_ok(baseline_value_output.allocate(kRows),
                                label + " allocate baseline value output");
  ready = ready && test.cuda_ok(
                       candidate_key_storage.allocate(
                           kRows + 2U * kGuardElements),
                       label + " allocate guarded candidate key output");
  ready = ready && test.cuda_ok(
                       candidate_value_storage.allocate(
                           kRows + 2U * kGuardElements),
                       label + " allocate guarded candidate value output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  if (!ready) {
    return;
  }
  std::uint16_t* const candidate_key_output =
      candidate_key_storage.get() + kGuardElements;
  std::uint16_t* const candidate_value_output =
      candidate_value_storage.get() + kGuardElements;

  const auto row_quad_status =
      [&](const std::size_t rows, const std::size_t columns,
          const std::size_t cap, std::uint16_t* const key_output,
          std::uint16_t* const value_output) noexcept {
        return static_cast<cudaError_t>(q3x::kernels::
            launch_sm87_fp8_w8a16_m1_kv_pair_row_quad_grid_cap_test_cuda(
                key_weights.get(), kKeyWeightScale, value_weights.get(),
                kValueWeightScale, activation.get(), rows, columns,
                key_output, value_output, cap, static_cast<void*>(stream)));
      };
  const auto production_status =
      [&](const std::uint8_t* const first_weights,
          const float first_scale, const std::uint8_t* const second_weights,
          const float second_scale, const std::size_t rows,
          const std::size_t columns, std::uint16_t* const first_output,
          std::uint16_t* const second_output) noexcept {
        return static_cast<cudaError_t>(
            q3x::kernels::launch_sm87_fp8_w8a16_gemv_pair_bf16_cuda(
                first_weights, first_scale, second_weights, second_scale,
                activation.get(), rows, columns, first_output, second_output,
                static_cast<void*>(stream)));
      };
  test.expect(production_status(
                  key_weights.get(), kKeyWeightScale, value_weights.get(),
                  kValueWeightScale, kRows - 1U, kColumns,
                  candidate_key_output, candidate_value_output) ==
                  cudaErrorInvalidValue,
              label + " production pair rejects near-miss rows");
  test.expect(production_status(
                  key_weights.get(), kKeyWeightScale, value_weights.get(),
                  kValueWeightScale, kRows, kColumns - 1U,
                  candidate_key_output, candidate_value_output) ==
                  cudaErrorInvalidValue,
              label + " production pair rejects near-miss columns");
  test.expect(row_quad_status(kRows, kColumns, 0U, candidate_key_output,
                              candidate_value_output) ==
                  cudaErrorInvalidValue,
              label + " row-quad rejects zero grid cap");
  test.expect(row_quad_status(kRows, kColumns, kRows / 2U + 1U,
                              candidate_key_output,
                              candidate_value_output) ==
                  cudaErrorInvalidValue,
              label + " row-quad rejects more than 512 CTAs");
  test.expect(production_status(
                  key_weights.get(), kKeyWeightScale, value_weights.get(),
                  kValueWeightScale, kRows, kColumns, candidate_key_output,
                  candidate_key_output) ==
                  cudaErrorInvalidValue,
              label + " production pair rejects overlapping outputs");
  test.expect(
      production_status(key_weights.get(), -1.0F, value_weights.get(),
                        kValueWeightScale, kRows, kColumns,
                        candidate_key_output, candidate_value_output) ==
          cudaErrorInvalidValue,
      label + " production pair rejects a negative first scale");
  test.expect(
      production_status(
          key_weights.get(), kKeyWeightScale, value_weights.get(),
          kValueWeightScale, kRows, kColumns,
          reinterpret_cast<std::uint16_t*>(value_weights.get()),
          candidate_value_output) == cudaErrorInvalidValue,
      label + " production pair rejects first output overlapping second "
              "weights");
  test.expect(
      production_status(key_weights.get() + 1U, kKeyWeightScale,
                        value_weights.get(), kValueWeightScale, kRows,
                        kColumns, candidate_key_output,
                        candidate_value_output) == cudaErrorInvalidValue,
      label + " production pair rejects unaligned first weights");
  test.expect(
      production_status(nullptr, kKeyWeightScale, value_weights.get(),
                        kValueWeightScale, kRows, kColumns,
                        candidate_key_output, candidate_value_output) ==
          cudaErrorInvalidValue,
      label + " production pair rejects null first weights");

  const auto launch_baseline = [&]() noexcept -> int {
    const int key_status =
        q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
            key_weights.get(), kKeyWeightScale, activation.get(), kRows,
            kColumns, baseline_key_output.get(), static_cast<void*>(stream));
    if (key_status != static_cast<int>(cudaSuccess)) {
      return key_status;
    }
    return q3x::kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
        value_weights.get(), kValueWeightScale, activation.get(), kRows,
        kColumns, baseline_value_output.get(), static_cast<void*>(stream));
  };
  const auto launch_row_quad = [&](const std::size_t cap) noexcept -> int {
    return q3x::kernels::
        launch_sm87_fp8_w8a16_m1_kv_pair_row_quad_grid_cap_test_cuda(
            key_weights.get(), kKeyWeightScale, value_weights.get(),
            kValueWeightScale, activation.get(), kRows, kColumns,
            candidate_key_output, candidate_value_output, cap,
            static_cast<void*>(stream));
  };
  const auto launch_production_pair =
      [&](const std::size_t) noexcept -> int {
    return q3x::kernels::launch_sm87_fp8_w8a16_gemv_pair_bf16_cuda(
        key_weights.get(), kKeyWeightScale, value_weights.get(),
        kValueWeightScale, activation.get(), kRows, kColumns,
        candidate_key_output, candidate_value_output,
        static_cast<void*>(stream));
  };

  std::vector<std::uint16_t> baseline_key(kRows);
  std::vector<std::uint16_t> baseline_value(kRows);
  std::vector<std::uint16_t> candidate_key(kRows + 2U * kGuardElements);
  std::vector<std::uint16_t> candidate_value(kRows + 2U * kGuardElements);
  std::array<Fp8KvPairTiming, kDistributions.size()> row_quad_cap64{};
  std::array<Fp8KvPairTiming, kDistributions.size()> row_quad_cap128{};
  std::array<bool, kDistributions.size()> row_quad_cap128_correct{};
  bool full_code_production_correct = false;

  const auto check_candidate =
      [&](const auto& launch_candidate, const std::size_t cap,
          const std::string& candidate_label,
          const bool require_finite = true) {
        bool candidate_ready = test.cuda_ok(
            cudaMemsetAsync(candidate_key_storage.get(), 0xa5,
                            candidate_key.size() * sizeof(std::uint16_t),
                            stream),
            candidate_label + " poison guarded key output");
        candidate_ready = candidate_ready && test.cuda_ok(
            cudaMemsetAsync(candidate_value_storage.get(), 0x5a,
                            candidate_value.size() * sizeof(std::uint16_t),
                            stream),
            candidate_label + " poison guarded value output");
        candidate_ready = candidate_ready && test.cuda_ok(
            static_cast<cudaError_t>(launch_candidate(cap)),
            candidate_label + " correctness launch");
        candidate_ready = candidate_ready && test.cuda_ok(
            cudaMemcpyAsync(candidate_key.data(), candidate_key_storage.get(),
                            candidate_key.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost, stream),
            candidate_label + " copy guarded key output");
        candidate_ready = candidate_ready && test.cuda_ok(
            cudaMemcpyAsync(candidate_value.data(),
                            candidate_value_storage.get(),
                            candidate_value.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost, stream),
            candidate_label + " copy guarded value output");
        candidate_ready = candidate_ready && test.cuda_ok(
            cudaStreamSynchronize(stream),
            candidate_label + " correctness synchronize");
        if (!candidate_ready) {
          return false;
        }
        std::size_t key_mismatches = 0U;
        std::size_t value_mismatches = 0U;
        std::size_t nonfinite_outputs = 0U;
        for (std::size_t row = 0U; row < kRows; ++row) {
          key_mismatches +=
              baseline_key[row] != candidate_key[kGuardElements + row] ? 1U
                                                                        : 0U;
          value_mismatches +=
              baseline_value[row] !=
                      candidate_value[kGuardElements + row]
                  ? 1U
                  : 0U;
          nonfinite_outputs += !is_bf16_finite(baseline_key[row]) ? 1U : 0U;
          nonfinite_outputs += !is_bf16_finite(baseline_value[row]) ? 1U : 0U;
          nonfinite_outputs +=
              !is_bf16_finite(candidate_key[kGuardElements + row]) ? 1U : 0U;
          nonfinite_outputs +=
              !is_bf16_finite(candidate_value[kGuardElements + row]) ? 1U
                                                                      : 0U;
        }
        bool guards_intact = true;
        for (std::size_t guard = 0U; guard < kGuardElements; ++guard) {
          guards_intact =
              guards_intact && candidate_key[guard] == kKeyGuard &&
              candidate_key[kGuardElements + kRows + guard] == kKeyGuard &&
              candidate_value[guard] == kValueGuard &&
              candidate_value[kGuardElements + kRows + guard] == kValueGuard;
        }
        const bool bitwise =
            key_mismatches == 0U && value_mismatches == 0U;
        std::cout << "FP8_M1_KV_PAIR_DIFF: " << candidate_label
                  << " key_bf16_mismatches=" << key_mismatches << '/'
                  << kRows << " value_bf16_mismatches=" << value_mismatches
                  << '/' << kRows
                  << " output_guards=" << (guards_intact ? "intact" : "BAD")
                  << " nonfinite_outputs=" << nonfinite_outputs
                  << '\n';
        test.expect(bitwise,
                    candidate_label +
                        " matches both production row-pair BF16 outputs");
        test.expect(guards_intact,
                    candidate_label + " preserves both output guards");
        if (require_finite) {
          test.expect(nonfinite_outputs == 0U,
                      candidate_label + " keeps all four output views finite");
        }
        return bitwise && guards_intact &&
               (!require_finite || nonfinite_outputs == 0U);
      };

  const auto benchmark_candidate =
      [&](const auto& launch_candidate, const std::size_t cap,
          const std::string& candidate_label) {
        Fp8KvPairTiming timing{};
        bool timing_ready = true;
        for (int iteration = 0;
             iteration < kWarmupIterations && timing_ready; ++iteration) {
          timing_ready = test.cuda_ok(
              static_cast<cudaError_t>(launch_baseline()),
              candidate_label + " baseline warmup");
          timing_ready = timing_ready && test.cuda_ok(
              static_cast<cudaError_t>(launch_candidate(cap)),
              candidate_label + " candidate warmup");
        }
        timing_ready = timing_ready && test.cuda_ok(
            cudaStreamSynchronize(stream),
            candidate_label + " warmup synchronize");
        if (!timing_ready) {
          return timing;
        }
        constexpr std::size_t kTimedPasses =
            2U * static_cast<std::size_t>(kMeasurementRounds);
        std::array<float, kTimedPasses> baseline_passes{};
        std::array<float, kTimedPasses> candidate_passes{};
        bool all_finite = true;
        for (int round = 0; round < kMeasurementRounds; ++round) {
          const std::string round_label =
              candidate_label + " round=" + std::to_string(round + 1);
          const float baseline_first = measure_small_m_tile(
              test, stream, launch_baseline, kMeasuredIterations,
              round_label + " baseline pass 1");
          const float candidate_first = measure_small_m_tile(
              test, stream, [&]() noexcept { return launch_candidate(cap); },
              kMeasuredIterations, round_label + " candidate pass 1");
          const float candidate_second = measure_small_m_tile(
              test, stream, [&]() noexcept { return launch_candidate(cap); },
              kMeasuredIterations, round_label + " candidate pass 2");
          const float baseline_second = measure_small_m_tile(
              test, stream, launch_baseline, kMeasuredIterations,
              round_label + " baseline pass 2");
          baseline_passes[2U * static_cast<std::size_t>(round)] =
              baseline_first;
          baseline_passes[2U * static_cast<std::size_t>(round) + 1U] =
              baseline_second;
          candidate_passes[2U * static_cast<std::size_t>(round)] =
              candidate_first;
          candidate_passes[2U * static_cast<std::size_t>(round) + 1U] =
              candidate_second;
          const bool round_finite =
              std::isfinite(baseline_first) &&
              std::isfinite(candidate_first) &&
              std::isfinite(candidate_second) &&
              std::isfinite(baseline_second);
          all_finite = all_finite && round_finite;
          std::cout << "PERF_FP8_M1_KV_PAIR_ROUND: " << candidate_label
                    << " round=" << round + 1
                    << " baseline_pass1_ms=" << baseline_first
                    << " candidate_pass1_ms=" << candidate_first
                    << " candidate_pass2_ms=" << candidate_second
                    << " baseline_pass2_ms=" << baseline_second << '\n';
        }
        if (!all_finite) {
          return timing;
        }
        timing.baseline_milliseconds =
            median_fp8_kv_pair_timing(baseline_passes);
        timing.candidate_milliseconds =
            median_fp8_kv_pair_timing(candidate_passes);
        const float speedup = timing.baseline_milliseconds /
                              timing.candidate_milliseconds;
        constexpr double kEncodedWeightBytes =
            2.0 * static_cast<double>(kRows) *
            static_cast<double>(kColumns);
        const double encoded_weight_gigabytes_per_second =
            kEncodedWeightBytes /
            (static_cast<double>(timing.candidate_milliseconds) * 1.0e6);
        std::cout << "PERF_FP8_M1_KV_PAIR: " << candidate_label
                  << " baseline_two_production_row_pair_median_ms="
                  << timing.baseline_milliseconds
                  << " candidate_median_ms=" << timing.candidate_milliseconds
                  << " speedup=" << speedup
                  << " uplift_percent=" << (speedup - 1.0F) * 100.0F
                  << " candidate_encoded_weight_GBps="
                  << encoded_weight_gigabytes_per_second << '\n';
        test.expect(std::isfinite(speedup) &&
                        timing.baseline_milliseconds > 0.0F &&
                        timing.candidate_milliseconds > 0.0F,
                    candidate_label + " reports finite positive timing");
        return timing;
      };

  // One bounded target-shape fixture puts all 254 finite raw E4M3FN codes in
  // every byte position of the packed-x4 loads. The two NaN codes have a
  // separate classification/sign case below so they cannot make every dot
  // product NaN and erase the finite fixture's bitwise discrimination.
  std::array<std::array<bool, 256U>, 4U> full_key_code_coverage{};
  std::array<std::array<bool, 256U>, 4U> full_value_code_coverage{};
  const auto finite_code = [](const std::size_t index) noexcept {
    const std::size_t finite_index = index % 254U;
    return static_cast<std::uint8_t>(
        finite_index < 127U ? finite_index : finite_index + 1U);
  };
  for (std::size_t row = 0U; row < kRows; ++row) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const std::size_t byte_position = column & 3U;
      const std::uint8_t key_code =
          finite_code((column >> 2U) + row * 13U);
      host_key_weights[row * kColumns + column] = key_code;
      const std::uint8_t value_code =
          finite_code((column >> 2U) * 7U + row * 11U + 3U);
      host_value_weights[row * kColumns + column] = value_code;
      full_key_code_coverage[byte_position][key_code] = true;
      full_value_code_coverage[byte_position][value_code] = true;
    }
  }
  for (std::size_t byte_position = 0U; byte_position < 4U;
       ++byte_position) {
    const std::size_t key_codes = static_cast<std::size_t>(std::count(
        full_key_code_coverage[byte_position].begin(),
        full_key_code_coverage[byte_position].end(), true));
    const std::size_t value_codes = static_cast<std::size_t>(std::count(
        full_value_code_coverage[byte_position].begin(),
        full_value_code_coverage[byte_position].end(), true));
    test.expect(key_codes == 254U &&
                    !full_key_code_coverage[byte_position][0x7fU] &&
                    !full_key_code_coverage[byte_position][0xffU],
                label + " full-code key fixture covers byte position " +
                    std::to_string(byte_position));
    test.expect(value_codes == 254U &&
                    !full_value_code_coverage[byte_position][0x7fU] &&
                    !full_value_code_coverage[byte_position][0xffU],
                label + " full-code value fixture covers byte position " +
                    std::to_string(byte_position));
  }
  ready = test.cuda_ok(
      cudaMemcpyAsync(key_weights.get(), host_key_weights.data(),
                      host_key_weights.size(), cudaMemcpyHostToDevice, stream),
      label + " initialize full-code key weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           value_weights.get(), host_value_weights.data(),
                           host_value_weights.size(), cudaMemcpyHostToDevice,
                           stream),
                       label + " initialize full-code value weights");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_baseline()),
                       label + " launch full-code production baselines");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline_key.data(), baseline_key_output.get(),
                           baseline_key.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy full-code baseline key output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline_value.data(), baseline_value_output.get(),
                           baseline_value.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy full-code baseline value output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " full-code baseline synchronize");
  if (!ready) {
    return;
  }
  full_code_production_correct = check_candidate(
      launch_production_pair, kSelectedGridCap,
      label + " distribution=full_codebook_all_byte_positions "
              "candidate=production_cross_matrix_row_quad_2K2V cap=128");

  std::fill(host_key_weights.begin(), host_key_weights.end(), 0U);
  std::fill(host_value_weights.begin(), host_value_weights.end(), 0U);
  std::fill(host_activation.begin(), host_activation.end(),
            encode_bf16(0.0F));
  host_activation[0U] = encode_bf16(1.0F);
  host_key_weights[0U] = 0x7fU;
  host_key_weights[kColumns] = 0xffU;
  host_value_weights[0U] = 0xffU;
  host_value_weights[kColumns] = 0x7fU;
  ready = test.cuda_ok(
      cudaMemcpyAsync(key_weights.get(), host_key_weights.data(),
                      host_key_weights.size(), cudaMemcpyHostToDevice, stream),
      label + " initialize isolated NaN key weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           value_weights.get(), host_value_weights.data(),
                           host_value_weights.size(), cudaMemcpyHostToDevice,
                           stream),
                       label + " initialize isolated NaN value weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize isolated NaN activation");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_baseline()),
                       label + " launch isolated NaN production baselines");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline_key.data(), baseline_key_output.get(),
                           baseline_key.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy isolated NaN baseline key output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline_value.data(), baseline_value_output.get(),
                           baseline_value.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy isolated NaN baseline value output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " isolated NaN baseline synchronize");
  if (!ready) {
    return;
  }
  const bool nan_pair_bitwise_and_guards = check_candidate(
      launch_production_pair, kSelectedGridCap,
      label + " distribution=isolated_nan_classification "
              "candidate=production_cross_matrix_row_quad_2K2V cap=128",
      false);
  const auto expect_nan_class_and_sign =
      [&](const std::uint16_t candidate_bits,
          const std::uint16_t baseline_bits,
          const std::string& output_label) {
        test.expect(is_bf16_nan(candidate_bits) && is_bf16_nan(baseline_bits),
                    output_label + " remains in the BF16 NaN class");
        test.expect((candidate_bits & 0x8000U) ==
                        (baseline_bits & 0x8000U),
                    output_label +
                        " matches single-projection NaN output sign");
  };
  expect_nan_class_and_sign(candidate_key[kGuardElements], baseline_key[0U],
                            label + " key row0 raw-code-0x7f NaN");
  expect_nan_class_and_sign(candidate_key[kGuardElements + 1U],
                            baseline_key[1U],
                            label + " key row1 raw-code-0xff NaN");
  expect_nan_class_and_sign(candidate_value[kGuardElements],
                            baseline_value[0U],
                            label + " value row0 raw-code-0xff NaN");
  expect_nan_class_and_sign(candidate_value[kGuardElements + 1U],
                            baseline_value[1U],
                            label + " value row1 raw-code-0x7f NaN");
  std::size_t unexpected_nonfinite = 0U;
  for (std::size_t row = 2U; row < kRows; ++row) {
    unexpected_nonfinite +=
        !is_bf16_finite(candidate_key[kGuardElements + row]) ? 1U : 0U;
    unexpected_nonfinite +=
        !is_bf16_finite(candidate_value[kGuardElements + row]) ? 1U : 0U;
  }
  test.expect(nan_pair_bitwise_and_guards && unexpected_nonfinite == 0U,
              label + " isolates signed NaNs to the four classified rows");
  std::cout << "FP8_M1_KV_PAIR_NAN_CLASSIFICATION: classified_outputs=4"
            << " unexpected_nonfinite=" << unexpected_nonfinite
            << " bitwise_and_guards="
            << (nan_pair_bitwise_and_guards ? "true" : "false") << '\n';

  if (!fp8_m1_kv_pair_performance_enabled()) {
    std::cout << "SKIP: FP8 M1 K/V pair timing segment; set "
                 "Q3X_RUN_SM87_FP8_M1_KV_PAIR_PERF=1 to enable\n";
    return;
  }
  fill_mixed_activation();
  ready = test.cuda_ok(
      cudaMemcpyAsync(activation.get(), host_activation.data(),
                      host_activation.size() * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice, stream),
      label + " restore mixed activation for timing");
  if (!ready) {
    return;
  }

  for (std::size_t distribution_index = 0U;
       distribution_index < kDistributions.size(); ++distribution_index) {
    const Fp8M2CodeDistribution distribution =
        kDistributions[distribution_index];
    const std::string distribution_name =
        fp8_m2_code_distribution_name(distribution);
    fill_fp8_m2_code_distribution(host_key_weights, kRows, kColumns,
                                  distribution);
    for (std::size_t index = 0U; index < host_value_weights.size(); ++index) {
      host_value_weights[index] =
          host_key_weights[host_value_weights.size() - 1U - index];
    }
    ready = test.cuda_ok(
        cudaMemcpyAsync(key_weights.get(), host_key_weights.data(),
                        host_key_weights.size(), cudaMemcpyHostToDevice,
                        stream),
        label + " " + distribution_name + " initialize key weights");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             value_weights.get(), host_value_weights.data(),
                             host_value_weights.size(), cudaMemcpyHostToDevice,
                             stream),
                         label + " " + distribution_name +
                             " initialize value weights");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_baseline()),
                         label + " " + distribution_name +
                             " launch two production row-pair baselines");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline_key.data(), baseline_key_output.get(),
                             baseline_key.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " " + distribution_name +
                             " copy baseline key output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline_value.data(),
                             baseline_value_output.get(),
                             baseline_value.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " " + distribution_name +
                             " copy baseline value output");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         label + " " + distribution_name +
                             " baseline synchronize");
    if (!ready) {
      return;
    }

    const std::string comparison_label =
        label + " distribution=" + distribution_name +
        " candidate=cross_matrix_row_quad_2K2V cap=64";
    (void)check_candidate(launch_row_quad, kComparisonGridCap,
                          comparison_label);
    row_quad_cap64[distribution_index] = benchmark_candidate(
        launch_row_quad, kComparisonGridCap, comparison_label);
    const std::string production_label =
        label + " distribution=" + distribution_name +
        " candidate=production_cross_matrix_row_quad_2K2V cap=128";
    row_quad_cap128_correct[distribution_index] =
        check_candidate(launch_production_pair, kSelectedGridCap,
                        production_label);
    row_quad_cap128[distribution_index] = benchmark_candidate(
        launch_production_pair, kSelectedGridCap, production_label);
  }

  const auto report_cap =
      [&](const char* const candidate, const std::size_t cap,
          const std::array<Fp8KvPairTiming, 2U>& timings) {
        double baseline_total = 0.0;
        double candidate_total = 0.0;
        bool finite = true;
        for (const Fp8KvPairTiming& timing : timings) {
          finite = finite && std::isfinite(timing.baseline_milliseconds) &&
                   std::isfinite(timing.candidate_milliseconds);
          baseline_total += timing.baseline_milliseconds;
          candidate_total += timing.candidate_milliseconds;
        }
        const double speedup = baseline_total / candidate_total;
        std::cout << "PERF_FP8_M1_KV_PAIR_CAP_AGGREGATE: candidate="
                  << candidate << " cap=" << cap
                  << " distributions=checkpoint_like:"
                                   "same_bank_stress"
                  << " baseline_two_production_row_pair_ms=" << baseline_total
                  << " candidate_ms=" << candidate_total
                  << " speedup=" << speedup
                  << " uplift_percent=" << (speedup - 1.0) * 100.0
                  << " timing=" << (finite ? "finite" : "BAD") << '\n';
        test.expect(finite && std::isfinite(speedup),
                    label + " " + candidate + " cap" +
                        std::to_string(cap) +
                        " aggregate timing is finite");
      };
  report_cap("cross_matrix_row_quad_2K2V_comparison", kComparisonGridCap,
             row_quad_cap64);
  report_cap("cross_matrix_row_quad_2K2V_selected", kSelectedGridCap,
             row_quad_cap128);

  constexpr double kMinimumSelectedSpeedup = 1.10;
  bool selected_production_gate =
      selected_resource_gate && full_code_production_correct;
  for (std::size_t distribution_index = 0U;
       distribution_index < kDistributions.size(); ++distribution_index) {
    const Fp8KvPairTiming& timing = row_quad_cap128[distribution_index];
    const double speedup =
        static_cast<double>(timing.baseline_milliseconds) /
        static_cast<double>(timing.candidate_milliseconds);
    const bool cell_gate = row_quad_cap128_correct[distribution_index] &&
                           std::isfinite(speedup) &&
                           speedup >= kMinimumSelectedSpeedup;
    selected_production_gate = selected_production_gate && cell_gate;
    std::cout << "PERF_FP8_M1_KV_PAIR_SELECTED_VALIDATION: distribution="
              << fp8_m2_code_distribution_name(
                     kDistributions[distribution_index])
              << " candidate=production_cross_matrix_row_quad_2K2V cap=128"
              << " speedup=" << speedup
              << " required_speedup=" << kMinimumSelectedSpeedup
              << " bitwise_and_guards="
              << (row_quad_cap128_correct[distribution_index] ? "true"
                                                               : "false")
              << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
  }
  std::cout << "PERF_FP8_M1_KV_PAIR_SELECTED: candidate="
               "production_cross_matrix_row_quad_2K2V cap=128"
            << " full_code_bitwise_and_guards="
            << (full_code_production_correct ? "true" : "false")
            << " resource_gate="
            << (selected_resource_gate ? "PASS" : "FAIL")
            << " distribution_gates="
            << (selected_production_gate ? "PASS" : "FAIL")
            << " gate=" << (selected_production_gate ? "PASS" : "FAIL")
            << '\n';
  test.expect(selected_production_gate,
              label + " selected production pair clears exact/resource/"
                      "per-distribution 1.10x gates");
}

struct Fp8M2RowPairDistributionMeasurement {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
};

struct Fp8M2RowPairMeasurement {
  std::array<Fp8M2RowPairDistributionMeasurement, 2U> distributions{};
};

[[nodiscard]] Fp8M2RowPairMeasurement benchmark_fp8_m2_row_pair_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 2U;
  constexpr std::size_t kProductionGridCap = 2'048U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 40;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr std::array<Fp8M2CodeDistribution, 2U> kDistributions{{
      Fp8M2CodeDistribution::kCheckpointLike,
      Fp8M2CodeDistribution::kSameBankStress,
  }};

  std::vector<std::uint8_t> host_weights(rows * columns);
  std::vector<std::uint16_t> host_activations(kTokens * columns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered =
          static_cast<int>((column * 17U + token * 29U + 5U) % 127U) - 63;
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
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return {};
  }

  Fp8M2RowPairMeasurement measurement;
  for (std::size_t distribution_index = 0U;
       distribution_index < kDistributions.size(); ++distribution_index) {
    const Fp8M2CodeDistribution distribution =
        kDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + fp8_m2_code_distribution_name(distribution);
    fill_fp8_m2_code_distribution(host_weights, rows, columns, distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(weights.get(), host_weights.data(),
                        host_weights.size(), cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize weights");
    if (!ready) {
      return measurement;
    }

    const auto launch_baseline = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_fp8_w8a16_small_m2_grid_cap_test_cuda(
              weights.get(), kWeightScale, activations.get(), rows, columns,
              baseline_output.get(), kProductionGridCap,
              static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_fp8_w8a16_small_m2_row_pair_test_cuda(
              weights.get(), kWeightScale, activations.get(), rows, columns,
              candidate_output.get(), static_cast<void*>(stream));
    };

    ready = test.cuda_ok(
        static_cast<cudaError_t>(launch_baseline()),
        distribution_label +
            " correctness preserved cap2048 single-row baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         distribution_label + " correctness direct row-pair");
    std::vector<std::uint16_t> baseline(kTokens * rows);
    std::vector<std::uint16_t> candidate(kTokens * rows);
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy baseline output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy direct row-pair output");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " correctness synchronize");
    if (!ready) {
      return measurement;
    }

    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < baseline.size(); ++index) {
      mismatches += baseline[index] != candidate[index] ? 1U : 0U;
    }
    Fp8M2RowPairDistributionMeasurement& distribution_measurement =
        measurement.distributions[distribution_index];
    distribution_measurement.bitwise_equal = mismatches == 0U;
    test.expect(distribution_measurement.bitwise_equal,
                distribution_label +
                    " direct row-pair matches every preserved cap2048 BF16 "
                    "bit");

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           distribution_label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           distribution_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " warmup synchronize");
    if (!ready) {
      return measurement;
    }

    double baseline_total = 0.0;
    double candidate_total = 0.0;
    bool all_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          distribution_label + " round=" + std::to_string(round + 1);
      const float baseline_first = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 1");
      const float candidate_first = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 1");
      const float candidate_second = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 2");
      const float baseline_second = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 2");
      const bool round_finite =
          std::isfinite(baseline_first) &&
          std::isfinite(candidate_first) &&
          std::isfinite(candidate_second) &&
          std::isfinite(baseline_second);
      all_finite = all_finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_FP8_M2_ROW_PAIR_ROUND: " << label
                << " distribution="
                << fp8_m2_code_distribution_name(distribution)
                << " round=" << round + 1
                << " baseline_pass1_ms=" << baseline_first
                << " candidate_pass1_ms=" << candidate_first
                << " candidate_pass2_ms=" << candidate_second
                << " baseline_pass2_ms=" << baseline_second << '\n';
    }
    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    distribution_measurement.baseline_milliseconds =
        all_finite ? static_cast<float>(baseline_total / kTimedPasses)
                   : std::numeric_limits<float>::quiet_NaN();
    distribution_measurement.candidate_milliseconds =
        all_finite ? static_cast<float>(candidate_total / kTimedPasses)
                   : std::numeric_limits<float>::quiet_NaN();
    const float speedup = distribution_measurement.baseline_milliseconds /
                          distribution_measurement.candidate_milliseconds;
    std::cout << "PERF_FP8_M2_ROW_PAIR: " << label << " distribution="
              << fp8_m2_code_distribution_name(distribution)
              << " preserved_cap2048_single_row_ms="
              << distribution_measurement.baseline_milliseconds
              << " direct_row_pair_ms="
              << distribution_measurement.candidate_milliseconds
              << " speedup=" << speedup << " bitwise_mismatches="
              << mismatches << '/' << baseline.size() << '\n';
  }
  return measurement;
}

void run_optional_fp8_m2_row_pair_performance(TestContext& test,
                                               cudaStream_t stream) {
  if (!fp8_m2_row_pair_performance_enabled()) {
    std::cout << "SKIP: FP8 M2 row-pair performance segment; set "
                 "Q3X_RUN_SM87_FP8_M2_ROW_PAIR_PERF=1 to enable\n";
    return;
  }
  constexpr float kMinimumPerShapeDistributionSpeedup = 1.02F;
  constexpr float kMinimumCheckpointWeightedSpeedup = 1.05F;
  struct Shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t checkpoint_calls;
    const char* label;
  };
  constexpr std::array<Shape, 6U> kShapes{{
      {10'240U, 5'120U, 48U, "FP8 M2 row-pair QKV 10240x5120"},
      {5'120U, 6'144U, 64U, "FP8 M2 row-pair projection 5120x6144"},
      {6'144U, 5'120U, 48U, "FP8 M2 row-pair projection 6144x5120"},
      {12'288U, 5'120U, 16U, "FP8 M2 row-pair QKV 12288x5120"},
      {1'024U, 5'120U, 32U, "FP8 M2 row-pair small 1024x5120"},
      {5'120U, 5'120U, 0U, "FP8 M2 row-pair square 5120x5120"},
  }};
  constexpr std::array<Fp8M2CodeDistribution, 2U> kDistributions{{
      Fp8M2CodeDistribution::kCheckpointLike,
      Fp8M2CodeDistribution::kSameBankStress,
  }};
  std::array<Fp8M2RowPairMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_fp8_m2_row_pair_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }

  bool all_shape_distributions_pass = true;
  double checkpoint_weighted_baseline = 0.0;
  double checkpoint_weighted_candidate = 0.0;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    for (std::size_t distribution_index = 0U;
         distribution_index < kDistributions.size(); ++distribution_index) {
      const Fp8M2RowPairDistributionMeasurement& measurement =
          measurements[shape_index].distributions[distribution_index];
      const float speedup = measurement.baseline_milliseconds /
                            measurement.candidate_milliseconds;
      const bool finite =
          std::isfinite(measurement.baseline_milliseconds) &&
          std::isfinite(measurement.candidate_milliseconds) &&
          std::isfinite(speedup);
      const bool gate = measurement.bitwise_equal && finite &&
                        speedup >= kMinimumPerShapeDistributionSpeedup;
      all_shape_distributions_pass =
          all_shape_distributions_pass && gate;
      test.expect(
          gate,
          std::string(kShapes[shape_index].label) + " " +
              fp8_m2_code_distribution_name(
                  kDistributions[distribution_index]) +
              " clears the M2 row-pair performance gate");
      std::cout << "PERF_FP8_M2_ROW_PAIR_VALIDATION: "
                << kShapes[shape_index].label << " distribution="
                << fp8_m2_code_distribution_name(
                       kDistributions[distribution_index])
                << " preserved_cap2048_single_row_ms="
                << measurement.baseline_milliseconds
                << " direct_row_pair_ms="
                << measurement.candidate_milliseconds
                << " speedup=" << speedup
                << " required_speedup="
                << kMinimumPerShapeDistributionSpeedup << " bitwise="
                << (measurement.bitwise_equal ? "true" : "false")
                << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
    }
    const Fp8M2RowPairDistributionMeasurement& checkpoint =
        measurements[shape_index].distributions[0U];
    checkpoint_weighted_baseline +=
        static_cast<double>(kShapes[shape_index].checkpoint_calls) *
        checkpoint.baseline_milliseconds;
    checkpoint_weighted_candidate +=
        static_cast<double>(kShapes[shape_index].checkpoint_calls) *
        checkpoint.candidate_milliseconds;
  }
  const double checkpoint_weighted_speedup =
      checkpoint_weighted_baseline / checkpoint_weighted_candidate;
  const bool aggregate_gate =
      all_shape_distributions_pass &&
      std::isfinite(checkpoint_weighted_speedup) &&
      checkpoint_weighted_speedup >= kMinimumCheckpointWeightedSpeedup;
  std::cout << "PERF_FP8_M2_ROW_PAIR_AGGREGATE: "
            << "checkpoint_weighted_preserved_cap2048_single_row_ms="
            << checkpoint_weighted_baseline
            << " checkpoint_weighted_direct_row_pair_ms="
            << checkpoint_weighted_candidate
            << " speedup=" << checkpoint_weighted_speedup
            << " required_speedup=" << kMinimumCheckpointWeightedSpeedup
            << " profile_calls=48:64:48:16:32:0 all_shape_distributions="
            << (all_shape_distributions_pass ? "PASS" : "FAIL")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "FP8 M2 row-pair clears every production gate");
}

constexpr std::array<std::size_t, 6U> kFp8M2RowQuadGridCaps{{
    256U,
    512U,
    768U,
    1'024U,
    1'536U,
    2'048U,
}};

constexpr std::array<Fp8M2CodeDistribution, 2U>
    kFp8M2RowQuadDistributions{{
        Fp8M2CodeDistribution::kCheckpointLike,
        Fp8M2CodeDistribution::kSameBankStress,
    }};

struct Fp8M2RowQuadComparison {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
};

struct Fp8M2RowQuadDistributionMeasurement {
  std::array<Fp8M2RowQuadComparison, kFp8M2RowQuadGridCaps.size()>
      caps{};
};

struct Fp8M2RowQuadMeasurement {
  std::array<Fp8M2RowQuadDistributionMeasurement,
             kFp8M2RowQuadDistributions.size()>
      distributions{};
};

struct Fp8M2RowQuadShape {
  std::size_t rows;
  std::size_t columns;
  std::size_t profile_calls;
  std::size_t selected_quad_cap;
  const char* label;
};

void run_fp8_m2_row_quad_tail_correctness(TestContext& test,
                                           cudaStream_t stream) {
  constexpr std::size_t kTokens = 2U;
  constexpr std::size_t kBytePositions = 4U;
  constexpr std::size_t kColumns = 1'024U;
  constexpr std::array<std::size_t, 3U> kRows{{1'025U, 1'026U, 1'027U}};
  constexpr std::size_t kBaselineGridCap = 2'048U;
  constexpr float kWeightScale = 1.0F / 128.0F;
  const std::size_t max_rows = kRows.back();
  const std::string label =
      "FP8 M2 row-quad tails rows1025/1026/1027 full E4M3FN byte "
      "positions";

  // Each of the 256 raw E4M3FN encodings appears once in each byte of the
  // first packed-x4 load. A zero background isolates that contribution even
  // for the NaN encodings, while finite nonzero tail rows catch dropped work.
  std::vector<std::uint8_t> host_weights(max_rows * kColumns, 0U);
  std::array<std::array<bool, 256U>, kBytePositions> code_coverage{};
  for (std::size_t code = 0U; code < 256U; ++code) {
    for (std::size_t position = 0U; position < kBytePositions; ++position) {
      const std::size_t row = code * kBytePositions + position;
      host_weights[row * kColumns + position] =
          static_cast<std::uint8_t>(code);
      code_coverage[position][code] = true;
    }
  }
  constexpr std::array<std::uint8_t, 3U> kTailCodes{{
      0x38U,
      0xb8U,
      0x40U,
  }};
  for (std::size_t tail = 0U; tail < kTailCodes.size(); ++tail) {
    const std::size_t row = 1'024U + tail;
    for (std::size_t column = 0U; column < kColumns; ++column) {
      host_weights[row * kColumns + column] = kTailCodes[tail];
    }
  }
  for (std::size_t position = 0U; position < kBytePositions; ++position) {
    test.expect(
        std::all_of(code_coverage[position].begin(),
                    code_coverage[position].end(),
                    [](const bool covered) { return covered; }),
        label + " covers all 256 raw codes at byte position " +
            std::to_string(position));
  }

  std::vector<std::uint16_t> host_activations(kTokens * kColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const int centered = static_cast<int>(
                               (column * 13U + token * 29U + 5U) % 127U) -
                           63;
      host_activations[token * kColumns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
    for (std::size_t position = 0U; position < kBytePositions; ++position) {
      const float distinct =
          static_cast<float>((token + 1U) * (position + 2U)) / 8.0F;
      host_activations[token * kColumns + position] = encode_bf16(distinct);
    }
  }
  test.expect(host_activations[0U] != host_activations[kColumns],
              label + " uses distinct token activations");

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * max_rows),
                                label + " allocate row-pair output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * max_rows),
                                label + " allocate row-quad output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(weights.get(), host_weights.data(),
                                       host_weights.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize exhaustive weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize two token activations");
  if (!ready) {
    return;
  }

  std::vector<std::uint16_t> baseline(kTokens * max_rows);
  std::vector<std::uint16_t> candidate(kTokens * max_rows);
  for (const std::size_t rows : kRows) {
    const std::string row_label = label + " rows=" + std::to_string(rows);
    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::kernels::
            launch_sm87_fp8_w8a16_small_m2_row_pair_grid_cap_test_cuda(
                weights.get(), kWeightScale, activations.get(), rows,
                kColumns, baseline_output.get(), kBaselineGridCap,
                static_cast<void*>(stream))),
        row_label + " launch production row-pair cap2048 baseline");
    for (const std::size_t cap : kFp8M2RowQuadGridCaps) {
      if (!ready) {
        return;
      }
      const std::string cap_label =
          row_label + " row_quad_cap=" + std::to_string(cap);
      ready = test.cuda_ok(
          cudaMemsetAsync(candidate_output.get(), 0xa5,
                          kTokens * max_rows * sizeof(std::uint16_t), stream),
          cap_label + " poison row-quad output");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(q3x::kernels::
                               launch_sm87_fp8_w8a16_small_m2_row_quad_grid_cap_test_cuda(
                                   weights.get(), kWeightScale,
                                   activations.get(), rows, kColumns,
                                   candidate_output.get(), cap,
                                   static_cast<void*>(stream))),
                           cap_label + " launch tail-safe row-quad");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               baseline.data(), baseline_output.get(),
                               kTokens * rows * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-pair output");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               candidate.data(), candidate_output.get(),
                               kTokens * rows * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-quad output");
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " synchronize");
      if (!ready) {
        return;
      }
      std::size_t mismatches = 0U;
      for (std::size_t index = 0U; index < kTokens * rows; ++index) {
        mismatches += baseline[index] != candidate[index] ? 1U : 0U;
      }
      std::cout << "FP8_M2_ROW_QUAD_TAIL_DIFF: rows=" << rows
                << " columns=" << kColumns << " candidate_cap=" << cap
                << " mismatches=" << mismatches << '/' << kTokens * rows
                << '\n';
      test.expect(mismatches == 0U,
                  cap_label + " matches every row-pair BF16 bit");
    }
  }
}

[[nodiscard]] Fp8M2RowQuadMeasurement benchmark_fp8_m2_row_quad_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 2U;
  constexpr std::size_t kBaselineGridCap = 2'048U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale = 1.0F / 64.0F;

  std::vector<std::uint8_t> host_weights(rows * columns);
  std::vector<std::uint16_t> host_activations(kTokens * columns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered = static_cast<int>(
                               (column * 17U + token * 29U + 5U) % 127U) -
                           63;
      host_activations[token * columns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
  }
  test.expect(host_activations[0U] != host_activations[columns],
              label + " uses distinct token activations");

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * rows),
                                label + " allocate row-pair output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * rows),
                                label + " allocate row-quad output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize two token activations");
  Fp8M2RowQuadMeasurement measurement;
  if (!ready) {
    return measurement;
  }

  std::vector<std::uint16_t> baseline(kTokens * rows);
  std::vector<std::uint16_t> candidate(kTokens * rows);
  for (std::size_t distribution_index = 0U;
       distribution_index < kFp8M2RowQuadDistributions.size();
       ++distribution_index) {
    const Fp8M2CodeDistribution distribution =
        kFp8M2RowQuadDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + fp8_m2_code_distribution_name(distribution);
    fill_fp8_m2_code_distribution(host_weights, rows, columns, distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(weights.get(), host_weights.data(),
                        host_weights.size(), cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize weights");
    if (!ready) {
      return measurement;
    }

    for (std::size_t cap_index = 0U;
         cap_index < kFp8M2RowQuadGridCaps.size(); ++cap_index) {
      const std::size_t candidate_cap =
          kFp8M2RowQuadGridCaps[cap_index];
      const std::string cap_label =
          distribution_label + " row_quad_cap=" +
          std::to_string(candidate_cap);
      const auto launch_baseline = [&]() noexcept -> int {
        return q3x::kernels::
            launch_sm87_fp8_w8a16_small_m2_row_pair_grid_cap_test_cuda(
                weights.get(), kWeightScale, activations.get(), rows,
                columns, baseline_output.get(), kBaselineGridCap,
                static_cast<void*>(stream));
      };
      const auto launch_candidate = [&]() noexcept -> int {
        return q3x::kernels::
            launch_sm87_fp8_w8a16_small_m2_row_quad_grid_cap_test_cuda(
                weights.get(), kWeightScale, activations.get(), rows,
                columns, candidate_output.get(), candidate_cap,
                static_cast<void*>(stream));
      };

      ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_baseline()),
          cap_label + " correctness production row-pair cap2048 baseline");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           cap_label + " correctness row-quad candidate");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               baseline.data(), baseline_output.get(),
                               baseline.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-pair output");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               candidate.data(), candidate_output.get(),
                               candidate.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-quad output");
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " correctness synchronize");
      if (!ready) {
        return measurement;
      }

      std::size_t mismatches = 0U;
      for (std::size_t index = 0U; index < baseline.size(); ++index) {
        mismatches += baseline[index] != candidate[index] ? 1U : 0U;
      }
      Fp8M2RowQuadComparison& comparison =
          measurement.distributions[distribution_index].caps[cap_index];
      comparison.bitwise_equal = mismatches == 0U;
      std::cout << "FP8_M2_ROW_QUAD_DIFF: " << label << " distribution="
                << fp8_m2_code_distribution_name(distribution)
                << " baseline_pair_cap=" << kBaselineGridCap
                << " candidate_quad_cap=" << candidate_cap
                << " mismatches=" << mismatches << '/' << baseline.size()
                << '\n';
      test.expect(comparison.bitwise_equal,
                  cap_label + " matches every row-pair BF16 bit");

      for (int iteration = 0; iteration < kWarmupIterations && ready;
           ++iteration) {
        ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                             cap_label + " row-pair warmup");
        ready = ready && test.cuda_ok(
                             static_cast<cudaError_t>(launch_candidate()),
                             cap_label + " row-quad warmup");
      }
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " warmup synchronize");
      if (!ready) {
        return measurement;
      }

      double baseline_total = 0.0;
      double candidate_total = 0.0;
      bool all_finite = true;
      for (int round = 0; round < kMeasurementRounds; ++round) {
        const std::string round_label =
            cap_label + " round=" + std::to_string(round + 1);
        const float baseline_first = measure_small_m_tile(
            test, stream, launch_baseline, kMeasuredIterations,
            round_label + " baseline pass 1");
        const float candidate_first = measure_small_m_tile(
            test, stream, launch_candidate, kMeasuredIterations,
            round_label + " candidate pass 1");
        const float candidate_second = measure_small_m_tile(
            test, stream, launch_candidate, kMeasuredIterations,
            round_label + " candidate pass 2");
        const float baseline_second = measure_small_m_tile(
            test, stream, launch_baseline, kMeasuredIterations,
            round_label + " baseline pass 2");
        const bool round_finite =
            std::isfinite(baseline_first) &&
            std::isfinite(candidate_first) &&
            std::isfinite(candidate_second) &&
            std::isfinite(baseline_second);
        all_finite = all_finite && round_finite;
        if (round_finite) {
          baseline_total += baseline_first + baseline_second;
          candidate_total += candidate_first + candidate_second;
        }
        std::cout << "PERF_FP8_M2_ROW_QUAD_ROUND: " << label
                  << " distribution="
                  << fp8_m2_code_distribution_name(distribution)
                  << " baseline_pair_cap=" << kBaselineGridCap
                  << " candidate_quad_cap=" << candidate_cap
                  << " measured_iterations=" << kMeasuredIterations
                  << " round=" << round + 1
                  << " baseline_pass1_ms=" << baseline_first
                  << " candidate_pass1_ms=" << candidate_first
                  << " candidate_pass2_ms=" << candidate_second
                  << " baseline_pass2_ms=" << baseline_second << '\n';
      }
      constexpr double kTimedPasses =
          2.0 * static_cast<double>(kMeasurementRounds);
      comparison.baseline_milliseconds =
          all_finite
              ? static_cast<float>(baseline_total / kTimedPasses)
              : std::numeric_limits<float>::quiet_NaN();
      comparison.candidate_milliseconds =
          all_finite
              ? static_cast<float>(candidate_total / kTimedPasses)
              : std::numeric_limits<float>::quiet_NaN();
      const float speedup = comparison.baseline_milliseconds /
                            comparison.candidate_milliseconds;
      std::cout << "PERF_FP8_M2_ROW_QUAD_CELL: " << label
                << " distribution="
                << fp8_m2_code_distribution_name(distribution)
                << " baseline_pair_cap=" << kBaselineGridCap
                << " candidate_quad_cap=" << candidate_cap
                << " baseline_ms=" << comparison.baseline_milliseconds
                << " candidate_ms=" << comparison.candidate_milliseconds
                << " speedup=" << speedup << " bitwise="
                << (comparison.bitwise_equal ? "true" : "false") << '\n';
    }
  }
  return measurement;
}

void run_optional_fp8_m2_row_quad_performance(TestContext& test,
                                               cudaStream_t stream) {
  if (!fp8_m2_row_quad_performance_enabled()) {
    std::cout << "SKIP: FP8 M2 row-quad cap/correctness performance "
                 "segment; set Q3X_RUN_SM87_FP8_M2_ROW_QUAD_PERF=1 to "
                 "enable\n";
    return;
  }
  constexpr float kMinimumPerCellSpeedup = 1.02F;
  constexpr double kMinimumCheckpointWeightedSpeedup = 1.05;
  constexpr std::array<Fp8M2RowQuadShape, 5U> kShapes{{
      {10'240U, 5'120U, 48U, 1'536U,
       "FP8 M2 row-quad QKV 10240x5120"},
      {5'120U, 6'144U, 64U, 768U,
       "FP8 M2 row-quad projection 5120x6144"},
      {6'144U, 5'120U, 48U, 1'024U,
       "FP8 M2 row-quad projection 6144x5120"},
      {12'288U, 5'120U, 16U, 2'048U,
       "FP8 M2 row-quad QKV 12288x5120"},
      {1'024U, 5'120U, 32U, 0U,
       "FP8 M2 preserved row-pair small 1024x5120"},
  }};

  for (const Fp8M2RowQuadShape& shape : kShapes) {
    test.expect(
        q3x::kernels::sm87_fp8_m2_row_quad_maximum_blocks_test(
            shape.rows, shape.columns) == shape.selected_quad_cap,
        std::string(shape.label) +
            " performance cap matches production dispatch");
  }

  run_fp8_m2_row_quad_tail_correctness(test, stream);
  std::array<Fp8M2RowQuadMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_fp8_m2_row_quad_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }

  bool all_candidate_bits_equal = true;
  for (std::size_t cap_index = 0U;
       cap_index < kFp8M2RowQuadGridCaps.size(); ++cap_index) {
    double weighted_baseline = 0.0;
    double weighted_candidate = 0.0;
    for (std::size_t shape_index = 0U; shape_index < kShapes.size();
         ++shape_index) {
      for (std::size_t distribution_index = 0U;
           distribution_index < kFp8M2RowQuadDistributions.size();
           ++distribution_index) {
        all_candidate_bits_equal =
            all_candidate_bits_equal &&
            measurements[shape_index]
                .distributions[distribution_index]
                .caps[cap_index]
                .bitwise_equal;
      }
      const Fp8M2RowQuadComparison& checkpoint =
          measurements[shape_index].distributions[0U].caps[cap_index];
      weighted_baseline +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          checkpoint.baseline_milliseconds;
      weighted_candidate +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          checkpoint.candidate_milliseconds;
    }
    const double weighted_speedup =
        weighted_baseline / weighted_candidate;
    std::cout << "PERF_FP8_M2_ROW_QUAD_CAP: candidate_cap="
              << kFp8M2RowQuadGridCaps[cap_index]
              << " checkpoint_weighted_baseline_ms=" << weighted_baseline
              << " checkpoint_weighted_candidate_ms=" << weighted_candidate
              << " speedup=" << weighted_speedup << '\n';
  }

  bool all_selected_cells_pass = true;
  double selected_weighted_baseline = 0.0;
  double selected_weighted_candidate = 0.0;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    const Fp8M2RowQuadShape& shape = kShapes[shape_index];
    if (shape.selected_quad_cap == 0U) {
      // This small shape is intentionally left on the production row-pair.
      // Use one of its interleaved baseline timings in both totals so it is
      // represented in the real call mix without claiming row-quad uplift.
      const Fp8M2RowQuadComparison& checkpoint =
          measurements[shape_index].distributions[0U].caps[0U];
      const bool finite = std::isfinite(checkpoint.baseline_milliseconds) &&
                          checkpoint.baseline_milliseconds > 0.0F;
      all_selected_cells_pass = all_selected_cells_pass && finite;
      const double weighted =
          static_cast<double>(shape.profile_calls) *
          checkpoint.baseline_milliseconds;
      selected_weighted_baseline += weighted;
      selected_weighted_candidate += weighted;
      std::cout << "PERF_FP8_M2_ROW_QUAD_PRESERVED: " << shape.label
                << " selected=row_pair_cap2048 baseline_ms="
                << checkpoint.baseline_milliseconds
                << " profile_calls=" << shape.profile_calls
                << " gate=" << (finite ? "PASS" : "FAIL") << '\n';
      test.expect(finite,
                  std::string(shape.label) +
                      " has a finite preserved row-pair timing");
      continue;
    }
    const auto selected_cap_iterator =
        std::find(kFp8M2RowQuadGridCaps.begin(),
                  kFp8M2RowQuadGridCaps.end(), shape.selected_quad_cap);
    const bool selected_cap_measured =
        selected_cap_iterator != kFp8M2RowQuadGridCaps.end();
    test.expect(selected_cap_measured,
                std::string(shape.label) +
                    " selected row-quad cap is present in the sweep");
    if (!selected_cap_measured) {
      all_selected_cells_pass = false;
      continue;
    }
    const std::size_t selected_cap_index = static_cast<std::size_t>(
        selected_cap_iterator - kFp8M2RowQuadGridCaps.begin());
    for (std::size_t distribution_index = 0U;
         distribution_index < kFp8M2RowQuadDistributions.size();
         ++distribution_index) {
      const Fp8M2RowQuadComparison& selected =
          measurements[shape_index]
              .distributions[distribution_index]
              .caps[selected_cap_index];
      const float speedup = selected.baseline_milliseconds /
                            selected.candidate_milliseconds;
      const bool finite = std::isfinite(selected.baseline_milliseconds) &&
                          std::isfinite(selected.candidate_milliseconds) &&
                          std::isfinite(speedup) &&
                          selected.baseline_milliseconds > 0.0F &&
                          selected.candidate_milliseconds > 0.0F;
      const bool cell_gate = selected.bitwise_equal && finite &&
                             speedup >= kMinimumPerCellSpeedup;
      all_selected_cells_pass = all_selected_cells_pass && cell_gate;
      std::cout << "PERF_FP8_M2_ROW_QUAD_VALIDATION: "
                << shape.label << " distribution="
                << fp8_m2_code_distribution_name(
                       kFp8M2RowQuadDistributions[distribution_index])
                << " selected_cap=" << shape.selected_quad_cap
                << " baseline_ms=" << selected.baseline_milliseconds
                << " candidate_ms=" << selected.candidate_milliseconds
                << " speedup=" << speedup
                << " required_speedup=" << kMinimumPerCellSpeedup
                << " bitwise="
                << (selected.bitwise_equal ? "true" : "false")
                << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
      test.expect(cell_gate,
                  std::string(shape.label) + " " +
                      fp8_m2_code_distribution_name(
                          kFp8M2RowQuadDistributions[distribution_index]) +
                      " selected row-quad cap clears the cell gate");
    }
    const Fp8M2RowQuadComparison& checkpoint =
        measurements[shape_index]
            .distributions[0U]
            .caps[selected_cap_index];
    selected_weighted_baseline +=
        static_cast<double>(shape.profile_calls) *
        checkpoint.baseline_milliseconds;
    selected_weighted_candidate +=
        static_cast<double>(shape.profile_calls) *
        checkpoint.candidate_milliseconds;
  }

  const double selected_weighted_speedup =
      selected_weighted_baseline / selected_weighted_candidate;
  const bool aggregate_gate =
      all_candidate_bits_equal && all_selected_cells_pass &&
      std::isfinite(selected_weighted_speedup) &&
      selected_weighted_baseline > 0.0 &&
      selected_weighted_candidate > 0.0 &&
      selected_weighted_speedup >= kMinimumCheckpointWeightedSpeedup;
  std::cout << "PERF_FP8_M2_ROW_QUAD_SELECTED: baseline_pair_cap=2048"
            << " selected_caps=1536:768:1024:2048:pair"
            << " checkpoint_weighted_baseline_ms="
            << selected_weighted_baseline
            << " checkpoint_weighted_candidate_ms="
            << selected_weighted_candidate
            << " checkpoint_weighted_speedup=" << selected_weighted_speedup
            << " required_weighted_speedup="
            << kMinimumCheckpointWeightedSpeedup
            << " profile_calls=48:64:48:16:32 all_candidate_bitwise="
            << (all_candidate_bits_equal ? "true" : "false")
            << " all_selected_cells="
            << (all_selected_cells_pass ? "PASS" : "FAIL")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "FP8 M2 shape-selected row-quad mapping clears every gate");
}

constexpr std::array<std::size_t, 15U> kFp8M1RowQuadGridCaps{{
    48U,
    64U,
    80U,
    96U,
    128U,
    192U,
    256U,
    512U,
    768U,
    1'024U,
    1'280U,
    1'536U,
    2'048U,
    2'560U,
    3'072U,
}};

constexpr std::array<Fp8M2CodeDistribution, 2U>
    kFp8M1RowQuadDistributions{{
        Fp8M2CodeDistribution::kCheckpointLike,
        Fp8M2CodeDistribution::kSameBankStress,
    }};

struct Fp8M1RowQuadComparison {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
};

struct Fp8M1RowQuadDistributionMeasurement {
  std::array<Fp8M1RowQuadComparison, kFp8M1RowQuadGridCaps.size()> caps{};
};

struct Fp8M1RowQuadMeasurement {
  std::array<Fp8M1RowQuadDistributionMeasurement,
             kFp8M1RowQuadDistributions.size()>
      distributions{};
};

struct Fp8M1RowQuadShape {
  std::size_t rows;
  std::size_t columns;
  std::size_t profile_calls;
  std::size_t selected_quad_cap;
  const char* label;
};

void run_fp8_m1_row_quad_tail_correctness(TestContext& test,
                                           cudaStream_t stream) {
  constexpr std::size_t kBytePositions = 4U;
  constexpr std::size_t kColumns = 1'024U;
  constexpr std::array<std::size_t, 3U> kRows{{
      4'097U,
      4'098U,
      4'099U,
  }};
  constexpr std::size_t kBaselineGridCap = 2'048U;
  constexpr float kWeightScale = 1.0F / 128.0F;
  const std::size_t max_rows = kRows.back();
  const std::string label =
      "FP8 M1 row-quad tails rows4097/4098/4099 full E4M3FN byte "
      "positions";

  // Put every raw E4M3FN code in each byte of a packed-x4 load. The zero
  // background isolates that code, including both NaN encodings, while the
  // finite nonzero rows after 4096 make dropped tail work observable.
  std::vector<std::uint8_t> host_weights(max_rows * kColumns, 0U);
  std::array<std::array<bool, 256U>, kBytePositions> code_coverage{};
  for (std::size_t code = 0U; code < 256U; ++code) {
    for (std::size_t position = 0U; position < kBytePositions; ++position) {
      const std::size_t row = code * kBytePositions + position;
      host_weights[row * kColumns + position] =
          static_cast<std::uint8_t>(code);
      code_coverage[position][code] = true;
    }
  }
  constexpr std::array<std::uint8_t, 3U> kTailCodes{{
      0x38U,
      0xb8U,
      0x40U,
  }};
  for (std::size_t tail = 0U; tail < kTailCodes.size(); ++tail) {
    const std::size_t row = 4'096U + tail;
    for (std::size_t column = 0U; column < kColumns; ++column) {
      host_weights[row * kColumns + column] = kTailCodes[tail];
    }
  }
  for (std::size_t position = 0U; position < kBytePositions; ++position) {
    test.expect(
        std::all_of(code_coverage[position].begin(),
                    code_coverage[position].end(),
                    [](const bool covered) { return covered; }),
        label + " covers all 256 raw codes at byte position " +
            std::to_string(position));
  }

  std::vector<std::uint16_t> host_activation(kColumns);
  for (std::size_t column = 0U; column < kColumns; ++column) {
    const int centered =
        static_cast<int>((column * 13U + (column >> 3U) * 7U + 5U) % 127U) -
        63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }
  for (std::size_t position = 0U; position < kBytePositions; ++position) {
    host_activation[position] =
        encode_bf16(static_cast<float>(position + 2U) / 8.0F);
  }
  test.expect(host_activation[0U] != host_activation[1U] &&
                  host_activation[1U] != host_activation[2U] &&
                  host_activation[2U] != host_activation[3U],
              label + " uses distinct activation values by byte position");

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activation.allocate(host_activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(max_rows),
                                label + " allocate row-pair output");
  ready = ready && test.cuda_ok(candidate_output.allocate(max_rows),
                                label + " allocate row-quad output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(weights.get(), host_weights.data(),
                                       host_weights.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize exhaustive weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  if (!ready) {
    return;
  }

  std::vector<std::uint16_t> baseline(max_rows);
  std::vector<std::uint16_t> candidate(max_rows);
  for (const std::size_t rows : kRows) {
    const std::string row_label = label + " rows=" + std::to_string(rows);
    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::kernels::
            launch_sm87_fp8_w8a16_m1_row_pair_grid_cap_test_cuda(
                weights.get(), kWeightScale, activation.get(), rows,
                kColumns, baseline_output.get(), kBaselineGridCap,
                static_cast<void*>(stream))),
        row_label + " launch row-pair cap2048 baseline");
    for (const std::size_t cap : kFp8M1RowQuadGridCaps) {
      if (!ready) {
        return;
      }
      const std::string cap_label =
          row_label + " row_quad_cap=" + std::to_string(cap);
      ready = test.cuda_ok(
          cudaMemsetAsync(candidate_output.get(), 0xa5,
                          max_rows * sizeof(std::uint16_t), stream),
          cap_label + " poison row-quad output");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(q3x::kernels::
                               launch_sm87_fp8_w8a16_m1_row_quad_grid_cap_test_cuda(
                                   weights.get(), kWeightScale,
                                   activation.get(), rows, kColumns,
                                   candidate_output.get(), cap,
                                   static_cast<void*>(stream))),
                           cap_label + " launch tail-safe row-quad");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               baseline.data(), baseline_output.get(),
                               rows * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-pair output");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               candidate.data(), candidate_output.get(),
                               rows * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-quad output");
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " synchronize");
      if (!ready) {
        return;
      }
      std::size_t mismatches = 0U;
      for (std::size_t row = 0U; row < rows; ++row) {
        mismatches += baseline[row] != candidate[row] ? 1U : 0U;
      }
      std::cout << "FP8_M1_ROW_QUAD_TAIL_DIFF: rows=" << rows
                << " columns=" << kColumns << " baseline_pair_cap="
                << kBaselineGridCap << " candidate_quad_cap=" << cap
                << " mismatches=" << mismatches << '/' << rows << '\n';
      test.expect(mismatches == 0U,
                  cap_label + " matches every row-pair BF16 bit");
    }
  }
}

[[nodiscard]] Fp8M1RowQuadMeasurement benchmark_fp8_m1_row_quad_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kBaselineGridCap = 2'048U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale = 1.0F / 64.0F;

  std::vector<std::uint8_t> host_weights(rows * columns);
  std::vector<std::uint16_t> host_activation(columns);
  for (std::size_t column = 0U; column < columns; ++column) {
    const int centered = static_cast<int>((column * 17U + 5U) % 127U) - 63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activation.allocate(host_activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(rows),
                                label + " allocate row-pair output");
  ready = ready && test.cuda_ok(candidate_output.allocate(rows),
                                label + " allocate row-quad output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  Fp8M1RowQuadMeasurement measurement;
  if (!ready) {
    return measurement;
  }

  std::vector<std::uint16_t> baseline(rows);
  std::vector<std::uint16_t> candidate(rows);
  for (std::size_t distribution_index = 0U;
       distribution_index < kFp8M1RowQuadDistributions.size();
       ++distribution_index) {
    const Fp8M2CodeDistribution distribution =
        kFp8M1RowQuadDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + fp8_m2_code_distribution_name(distribution);
    fill_fp8_m2_code_distribution(host_weights, rows, columns, distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(weights.get(), host_weights.data(),
                        host_weights.size(), cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize weights");
    if (!ready) {
      return measurement;
    }

    for (std::size_t cap_index = 0U;
         cap_index < kFp8M1RowQuadGridCaps.size(); ++cap_index) {
      const std::size_t candidate_cap = kFp8M1RowQuadGridCaps[cap_index];
      const std::string cap_label =
          distribution_label + " row_quad_cap=" +
          std::to_string(candidate_cap);
      const auto launch_baseline = [&]() noexcept -> int {
        return q3x::kernels::
            launch_sm87_fp8_w8a16_m1_row_pair_grid_cap_test_cuda(
                weights.get(), kWeightScale, activation.get(), rows,
                columns, baseline_output.get(), kBaselineGridCap,
                static_cast<void*>(stream));
      };
      const auto launch_candidate = [&]() noexcept -> int {
        return q3x::kernels::
            launch_sm87_fp8_w8a16_m1_row_quad_grid_cap_test_cuda(
                weights.get(), kWeightScale, activation.get(), rows,
                columns, candidate_output.get(), candidate_cap,
                static_cast<void*>(stream));
      };

      ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_baseline()),
          cap_label + " correctness row-pair cap2048 baseline");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           cap_label + " correctness row-quad candidate");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               baseline.data(), baseline_output.get(),
                               baseline.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-pair output");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               candidate.data(), candidate_output.get(),
                               candidate.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-quad output");
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " correctness synchronize");
      if (!ready) {
        return measurement;
      }

      std::size_t mismatches = 0U;
      for (std::size_t row = 0U; row < rows; ++row) {
        mismatches += baseline[row] != candidate[row] ? 1U : 0U;
      }
      Fp8M1RowQuadComparison& comparison =
          measurement.distributions[distribution_index].caps[cap_index];
      comparison.bitwise_equal = mismatches == 0U;
      std::cout << "FP8_M1_ROW_QUAD_DIFF: " << label << " distribution="
                << fp8_m2_code_distribution_name(distribution)
                << " baseline_pair_cap=" << kBaselineGridCap
                << " candidate_quad_cap=" << candidate_cap
                << " mismatches=" << mismatches << '/' << rows << '\n';
      test.expect(comparison.bitwise_equal,
                  cap_label + " matches every row-pair BF16 bit");

      for (int iteration = 0; iteration < kWarmupIterations && ready;
           ++iteration) {
        ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                             cap_label + " row-pair warmup");
        ready = ready && test.cuda_ok(
                             static_cast<cudaError_t>(launch_candidate()),
                             cap_label + " row-quad warmup");
      }
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " warmup synchronize");
      if (!ready) {
        return measurement;
      }

      double baseline_total = 0.0;
      double candidate_total = 0.0;
      bool all_finite = true;
      for (int round = 0; round < kMeasurementRounds; ++round) {
        const std::string round_label =
            cap_label + " round=" + std::to_string(round + 1);
        // A/B/B/A keeps both launchers equally exposed to clock drift and
        // thermal movement within every measurement round.
        const float baseline_first = measure_small_m_tile(
            test, stream, launch_baseline, kMeasuredIterations,
            round_label + " baseline pass 1");
        const float candidate_first = measure_small_m_tile(
            test, stream, launch_candidate, kMeasuredIterations,
            round_label + " candidate pass 1");
        const float candidate_second = measure_small_m_tile(
            test, stream, launch_candidate, kMeasuredIterations,
            round_label + " candidate pass 2");
        const float baseline_second = measure_small_m_tile(
            test, stream, launch_baseline, kMeasuredIterations,
            round_label + " baseline pass 2");
        const bool round_finite =
            std::isfinite(baseline_first) &&
            std::isfinite(candidate_first) &&
            std::isfinite(candidate_second) &&
            std::isfinite(baseline_second);
        all_finite = all_finite && round_finite;
        if (round_finite) {
          baseline_total += baseline_first + baseline_second;
          candidate_total += candidate_first + candidate_second;
        }
        std::cout << "PERF_FP8_M1_ROW_QUAD_ROUND: " << label
                  << " distribution="
                  << fp8_m2_code_distribution_name(distribution)
                  << " baseline_pair_cap=" << kBaselineGridCap
                  << " candidate_quad_cap=" << candidate_cap
                  << " measured_iterations=" << kMeasuredIterations
                  << " round=" << round + 1
                  << " baseline_pass1_ms=" << baseline_first
                  << " candidate_pass1_ms=" << candidate_first
                  << " candidate_pass2_ms=" << candidate_second
                  << " baseline_pass2_ms=" << baseline_second << '\n';
      }
      constexpr double kTimedPasses =
          2.0 * static_cast<double>(kMeasurementRounds);
      comparison.baseline_milliseconds =
          all_finite
              ? static_cast<float>(baseline_total / kTimedPasses)
              : std::numeric_limits<float>::quiet_NaN();
      comparison.candidate_milliseconds =
          all_finite
              ? static_cast<float>(candidate_total / kTimedPasses)
              : std::numeric_limits<float>::quiet_NaN();
      const float speedup = comparison.baseline_milliseconds /
                            comparison.candidate_milliseconds;
      std::cout << "PERF_FP8_M1_ROW_QUAD_CELL: " << label
                << " distribution="
                << fp8_m2_code_distribution_name(distribution)
                << " baseline_pair_cap=" << kBaselineGridCap
                << " candidate_quad_cap=" << candidate_cap
                << " baseline_ms=" << comparison.baseline_milliseconds
                << " candidate_ms=" << comparison.candidate_milliseconds
                << " speedup=" << speedup << " bitwise="
                << (comparison.bitwise_equal ? "true" : "false") << '\n';
    }
  }
  return measurement;
}

void run_optional_fp8_m1_row_quad_performance(TestContext& test,
                                               cudaStream_t stream) {
  if (!fp8_m1_row_quad_performance_enabled()) {
    std::cout << "SKIP: FP8 M1 row-quad cap/correctness performance "
                 "segment; set Q3X_RUN_SM87_FP8_M1_ROW_QUAD_PERF=1 to "
                 "enable\n";
    return;
  }
  constexpr float kMinimumCheckpointShapeSpeedup = 1.005F;
  constexpr float kMinimumStressShapeSpeedup = 0.995F;
  constexpr double kMinimumCheckpointWeightedSpeedup = 1.018;
  constexpr double kMinimumStressWeightedSpeedup = 0.998;
  constexpr std::array<Fp8M1RowQuadShape, 5U> kShapes{{
      {10'240U, 5'120U, 48U, 1'536U,
       "FP8 M1 row-quad QKV 10240x5120"},
      {5'120U, 6'144U, 64U, 1'280U,
       "FP8 M1 row-quad projection 5120x6144"},
      {6'144U, 5'120U, 48U, 768U,
       "FP8 M1 row-quad projection 6144x5120"},
      {12'288U, 5'120U, 16U, 2'048U,
       "FP8 M1 row-quad QKV 12288x5120"},
      {1'024U, 5'120U, 32U, 0U,
       "FP8 M1 row-quad small projection 1024x5120"},
  }};

  for (const Fp8M1RowQuadShape& shape : kShapes) {
    test.expect(
        q3x::kernels::sm87_fp8_m1_row_quad_maximum_blocks_test(
            shape.rows, shape.columns) == shape.selected_quad_cap,
        std::string(shape.label) +
            " selected cap matches production dispatch");
  }

  run_fp8_m1_row_quad_tail_correctness(test, stream);
  std::array<Fp8M1RowQuadMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_fp8_m1_row_quad_shape(
        test, stream, kShapes[shape_index].rows,
        kShapes[shape_index].columns, kShapes[shape_index].label);
  }

  bool all_candidate_bits_equal = true;
  for (std::size_t cap_index = 0U;
       cap_index < kFp8M1RowQuadGridCaps.size(); ++cap_index) {
    for (std::size_t distribution_index = 0U;
         distribution_index < kFp8M1RowQuadDistributions.size();
         ++distribution_index) {
      double weighted_baseline = 0.0;
      double weighted_candidate = 0.0;
      bool cap_distribution_bits_equal = true;
      for (std::size_t shape_index = 0U; shape_index < kShapes.size();
           ++shape_index) {
        const Fp8M1RowQuadComparison& comparison =
            measurements[shape_index]
                .distributions[distribution_index]
                .caps[cap_index];
        cap_distribution_bits_equal =
            cap_distribution_bits_equal && comparison.bitwise_equal;
        weighted_baseline +=
            static_cast<double>(kShapes[shape_index].profile_calls) *
            comparison.baseline_milliseconds;
        weighted_candidate +=
            static_cast<double>(kShapes[shape_index].profile_calls) *
            comparison.candidate_milliseconds;
      }
      all_candidate_bits_equal =
          all_candidate_bits_equal && cap_distribution_bits_equal;
      const double weighted_speedup =
          weighted_baseline / weighted_candidate;
      std::cout << "PERF_FP8_M1_ROW_QUAD_WEIGHTED: distribution="
                << fp8_m2_code_distribution_name(
                       kFp8M1RowQuadDistributions[distribution_index])
                << " baseline_pair_cap=2048 candidate_quad_cap="
                << kFp8M1RowQuadGridCaps[cap_index]
                << " weighted_baseline_ms=" << weighted_baseline
                << " weighted_candidate_ms=" << weighted_candidate
                << " speedup=" << weighted_speedup
                << " profile_calls=48:64:48:16:32 bitwise="
                << (cap_distribution_bits_equal ? "true" : "false")
                << '\n';
    }
  }

  // Retain the complete sweep as a diagnostic around the frozen mapping.
  std::cout << "PERF_FP8_M1_ROW_QUAD_SWEEP: baseline_pair_cap=2048"
            << " candidate_caps=48:64:80:96:128:192:256:512:768:1024:"
               "1280:1536:2048:2560:3072"
            << " distributions=checkpoint_like:same_bank_stress"
            << " profile_calls=48:64:48:16:32 all_candidate_bitwise="
            << (all_candidate_bits_equal ? "true" : "false")
            << " gate=" << (all_candidate_bits_equal ? "PASS" : "FAIL")
            << '\n';
  test.expect(all_candidate_bits_equal,
              "FP8 M1 row-quad sweep matches every row-pair BF16 bit");

  std::array<double, kFp8M1RowQuadDistributions.size()>
      selected_weighted_baseline{};
  std::array<double, kFp8M1RowQuadDistributions.size()>
      selected_weighted_candidate{};
  bool all_selected_bitwise_finite = true;
  bool all_selected_cells_pass = true;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    const Fp8M1RowQuadShape& shape = kShapes[shape_index];
    if (shape.selected_quad_cap == 0U) {
      // This small shape remains on the production row-pair path. Include its
      // measured baseline in both sides of every weighted total.
      for (std::size_t distribution_index = 0U;
           distribution_index < kFp8M1RowQuadDistributions.size();
           ++distribution_index) {
        const Fp8M1RowQuadComparison& preserved =
            measurements[shape_index]
                .distributions[distribution_index]
                .caps[0U];
        const bool finite =
            std::isfinite(preserved.baseline_milliseconds) &&
            preserved.baseline_milliseconds > 0.0F;
        all_selected_bitwise_finite =
            all_selected_bitwise_finite && finite;
        all_selected_cells_pass = all_selected_cells_pass && finite;
        const double weighted =
            static_cast<double>(shape.profile_calls) *
            preserved.baseline_milliseconds;
        selected_weighted_baseline[distribution_index] += weighted;
        selected_weighted_candidate[distribution_index] += weighted;
        std::cout << "PERF_FP8_M1_ROW_QUAD_SELECTED_CELL: "
                  << shape.label << " distribution="
                  << fp8_m2_code_distribution_name(
                         kFp8M1RowQuadDistributions[distribution_index])
                  << " selected=row_pair_cap2048 baseline_ms="
                  << preserved.baseline_milliseconds
                  << " candidate_ms=" << preserved.baseline_milliseconds
                  << " speedup=1 bitwise=true finite="
                  << (finite ? "true" : "false")
                  << " profile_calls=" << shape.profile_calls
                  << " gate=" << (finite ? "PASS" : "FAIL") << '\n';
        test.expect(finite,
                    std::string(shape.label) + " " +
                        fp8_m2_code_distribution_name(
                            kFp8M1RowQuadDistributions[distribution_index]) +
                        " has a finite preserved row-pair timing");
      }
      continue;
    }

    const auto selected_cap_iterator =
        std::find(kFp8M1RowQuadGridCaps.begin(),
                  kFp8M1RowQuadGridCaps.end(), shape.selected_quad_cap);
    const bool selected_cap_measured =
        selected_cap_iterator != kFp8M1RowQuadGridCaps.end();
    test.expect(selected_cap_measured,
                std::string(shape.label) +
                    " selected row-quad cap is present in the sweep");
    if (!selected_cap_measured) {
      all_selected_bitwise_finite = false;
      all_selected_cells_pass = false;
      continue;
    }
    const std::size_t selected_cap_index = static_cast<std::size_t>(
        selected_cap_iterator - kFp8M1RowQuadGridCaps.begin());
    for (std::size_t distribution_index = 0U;
         distribution_index < kFp8M1RowQuadDistributions.size();
         ++distribution_index) {
      const Fp8M1RowQuadComparison& selected =
          measurements[shape_index]
              .distributions[distribution_index]
              .caps[selected_cap_index];
      const float speedup = selected.baseline_milliseconds /
                            selected.candidate_milliseconds;
      const bool finite =
          std::isfinite(selected.baseline_milliseconds) &&
          std::isfinite(selected.candidate_milliseconds) &&
          std::isfinite(speedup) &&
          selected.baseline_milliseconds > 0.0F &&
          selected.candidate_milliseconds > 0.0F;
      const float required_speedup =
          distribution_index == 0U ? kMinimumCheckpointShapeSpeedup
                                   : kMinimumStressShapeSpeedup;
      const bool bitwise_finite = selected.bitwise_equal && finite;
      const bool cell_gate = bitwise_finite && speedup >= required_speedup;
      all_selected_bitwise_finite =
          all_selected_bitwise_finite && bitwise_finite;
      all_selected_cells_pass = all_selected_cells_pass && cell_gate;
      selected_weighted_baseline[distribution_index] +=
          static_cast<double>(shape.profile_calls) *
          selected.baseline_milliseconds;
      selected_weighted_candidate[distribution_index] +=
          static_cast<double>(shape.profile_calls) *
          selected.candidate_milliseconds;
      std::cout << "PERF_FP8_M1_ROW_QUAD_SELECTED_CELL: " << shape.label
                << " distribution="
                << fp8_m2_code_distribution_name(
                       kFp8M1RowQuadDistributions[distribution_index])
                << " selected_quad_cap=" << shape.selected_quad_cap
                << " baseline_ms=" << selected.baseline_milliseconds
                << " candidate_ms=" << selected.candidate_milliseconds
                << " speedup=" << speedup
                << " required_speedup=" << required_speedup
                << " bitwise="
                << (selected.bitwise_equal ? "true" : "false")
                << " finite=" << (finite ? "true" : "false")
                << " profile_calls=" << shape.profile_calls
                << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
      test.expect(cell_gate,
                  std::string(shape.label) + " " +
                      fp8_m2_code_distribution_name(
                          kFp8M1RowQuadDistributions[distribution_index]) +
                      " selected row-quad cap clears its cell gate");
    }
  }

  bool all_weighted_gates_pass = true;
  std::array<double, kFp8M1RowQuadDistributions.size()>
      selected_weighted_speedups{};
  for (std::size_t distribution_index = 0U;
       distribution_index < kFp8M1RowQuadDistributions.size();
       ++distribution_index) {
    const double speedup =
        selected_weighted_baseline[distribution_index] /
        selected_weighted_candidate[distribution_index];
    selected_weighted_speedups[distribution_index] = speedup;
    const double required_speedup =
        distribution_index == 0U ? kMinimumCheckpointWeightedSpeedup
                                 : kMinimumStressWeightedSpeedup;
    const bool weighted_gate =
        std::isfinite(speedup) &&
        selected_weighted_baseline[distribution_index] > 0.0 &&
        selected_weighted_candidate[distribution_index] > 0.0 &&
        speedup >= required_speedup;
    all_weighted_gates_pass =
        all_weighted_gates_pass && weighted_gate;
    std::cout << "PERF_FP8_M1_ROW_QUAD_SELECTED_WEIGHTED: distribution="
              << fp8_m2_code_distribution_name(
                     kFp8M1RowQuadDistributions[distribution_index])
              << " selected_caps=1536:1280:768:2048:pair"
              << " weighted_baseline_ms="
              << selected_weighted_baseline[distribution_index]
              << " weighted_candidate_ms="
              << selected_weighted_candidate[distribution_index]
              << " speedup=" << speedup
              << " required_speedup=" << required_speedup
              << " profile_calls=48:64:48:16:32"
              << " gate=" << (weighted_gate ? "PASS" : "FAIL") << '\n';
    test.expect(weighted_gate,
                std::string("FP8 M1 selected row-quad ") +
                    fp8_m2_code_distribution_name(
                        kFp8M1RowQuadDistributions[distribution_index]) +
                    " weighted gate passes");
  }

  const bool selected_mapping_gate =
      all_selected_bitwise_finite && all_selected_cells_pass &&
      all_weighted_gates_pass;
  std::cout << "PERF_FP8_M1_ROW_QUAD_SELECTED: baseline_pair_cap=2048"
            << " selected_caps=1536:1280:768:2048:pair"
            << " checkpoint_weighted_speedup="
            << selected_weighted_speedups[0U]
            << " stress_weighted_speedup="
            << selected_weighted_speedups[1U]
            << " profile_calls=48:64:48:16:32"
            << " all_selected_bitwise_finite="
            << (all_selected_bitwise_finite ? "PASS" : "FAIL")
            << " all_selected_cells="
            << (all_selected_cells_pass ? "PASS" : "FAIL")
            << " all_weighted_gates="
            << (all_weighted_gates_pass ? "PASS" : "FAIL")
            << " gate=" << (selected_mapping_gate ? "PASS" : "FAIL")
            << '\n';
  test.expect(selected_mapping_gate,
              "FP8 M1 shape-selected row-quad mapping clears every gate");
}

struct Fp8M1SwizzledCodebookDistributionMeasurement {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
  bool output_finite = false;
  bool timing_finite = false;
};

struct Fp8M1SwizzledCodebookMeasurement {
  std::array<Fp8M1SwizzledCodebookDistributionMeasurement,
             kFp8M1RowQuadDistributions.size()>
      distributions{};
};

[[nodiscard]] Fp8M1SwizzledCodebookMeasurement
benchmark_fp8_m1_swizzled_codebook_shape(
    TestContext& test, cudaStream_t stream, const Fp8M1RowQuadShape& shape) {
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr std::array<float, 2U> kMinimumCellSpeedups{{
      1.50F,
      2.00F,
  }};

  std::vector<std::uint8_t> host_weights(shape.rows * shape.columns);
  std::vector<std::uint16_t> host_activation(shape.columns);
  for (std::size_t column = 0U; column < shape.columns; ++column) {
    const int centered = static_cast<int>((column * 17U + 5U) % 127U) - 63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            std::string(shape.label) + " allocate weights");
  ready = ready && test.cuda_ok(
                       activation.allocate(host_activation.size()),
                       std::string(shape.label) + " allocate activation");
  ready = ready && test.cuda_ok(
                       baseline_output.allocate(shape.rows),
                       std::string(shape.label) + " allocate baseline output");
  ready = ready && test.cuda_ok(
                       candidate_output.allocate(shape.rows),
                       std::string(shape.label) + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       std::string(shape.label) + " initialize activation");

  Fp8M1SwizzledCodebookMeasurement measurement;
  if (!ready) {
    return measurement;
  }

  std::vector<std::uint16_t> baseline(shape.rows);
  std::vector<std::uint16_t> candidate(shape.rows);
  for (std::size_t distribution_index = 0U;
       distribution_index < kFp8M1RowQuadDistributions.size();
       ++distribution_index) {
    const Fp8M2CodeDistribution distribution =
        kFp8M1RowQuadDistributions[distribution_index];
    const std::string label =
        std::string(shape.label) + " " +
        fp8_m2_code_distribution_name(distribution) + " cap=" +
        std::to_string(shape.selected_quad_cap);
    fill_fp8_m2_code_distribution(host_weights, shape.rows, shape.columns,
                                  distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(weights.get(), host_weights.data(),
                        host_weights.size(), cudaMemcpyHostToDevice, stream),
        label + " initialize weights");
    if (!ready) {
      return measurement;
    }

    const auto launch_baseline = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_fp8_w8a16_m1_row_quad_grid_cap_test_cuda(
              weights.get(), kWeightScale, activation.get(), shape.rows,
              shape.columns, baseline_output.get(), shape.selected_quad_cap,
              static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_fp8_w8a16_m1_row_quad_swizzled_codebook_grid_cap_test_cuda(
              weights.get(), kWeightScale, activation.get(), shape.rows,
              shape.columns, candidate_output.get(), shape.selected_quad_cap,
              static_cast<void*>(stream));
    };

    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         label + " launch current row-quad baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         label + " launch swizzled-codebook candidate");
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
      return measurement;
    }

    std::size_t mismatches = 0U;
    bool output_finite = true;
    for (std::size_t row = 0U; row < shape.rows; ++row) {
      mismatches += baseline[row] != candidate[row] ? 1U : 0U;
      output_finite = output_finite &&
                      std::isfinite(decode_bf16(baseline[row])) &&
                      std::isfinite(decode_bf16(candidate[row]));
    }
    auto& distribution_measurement =
        measurement.distributions[distribution_index];
    distribution_measurement.bitwise_equal = mismatches == 0U;
    distribution_measurement.output_finite = output_finite;
    std::cout << "FP8_M1_SWIZZLED_CODEBOOK_DIFF: " << shape.label
              << " distribution="
              << fp8_m2_code_distribution_name(distribution)
              << " baseline_quad_cap=" << shape.selected_quad_cap
              << " candidate_swizzled_cap=" << shape.selected_quad_cap
              << " mismatches=" << mismatches << '/' << shape.rows
              << " output_finite=" << (output_finite ? "true" : "false")
              << '\n';
    test.expect(distribution_measurement.bitwise_equal,
                label + " swizzled codebook matches every baseline BF16 bit");
    test.expect(output_finite, label + " outputs remain finite");

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
      return measurement;
    }

    double baseline_total = 0.0;
    double candidate_total = 0.0;
    bool timing_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          label + " round=" + std::to_string(round + 1);
      const float baseline_first = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 1");
      const float candidate_first = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 1");
      const float candidate_second = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 2");
      const float baseline_second = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 2");
      const bool round_finite =
          std::isfinite(baseline_first) && baseline_first > 0.0F &&
          std::isfinite(candidate_first) && candidate_first > 0.0F &&
          std::isfinite(candidate_second) && candidate_second > 0.0F &&
          std::isfinite(baseline_second) && baseline_second > 0.0F;
      timing_finite = timing_finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_FP8_M1_SWIZZLED_CODEBOOK_ROUND: " << shape.label
                << " distribution="
                << fp8_m2_code_distribution_name(distribution)
                << " cap=" << shape.selected_quad_cap
                << " measured_iterations=" << kMeasuredIterations
                << " round=" << round + 1
                << " baseline_pass1_ms=" << baseline_first
                << " candidate_pass1_ms=" << candidate_first
                << " candidate_pass2_ms=" << candidate_second
                << " baseline_pass2_ms=" << baseline_second << '\n';
    }

    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    distribution_measurement.baseline_milliseconds =
        timing_finite
            ? static_cast<float>(baseline_total / kTimedPasses)
            : std::numeric_limits<float>::quiet_NaN();
    distribution_measurement.candidate_milliseconds =
        timing_finite
            ? static_cast<float>(candidate_total / kTimedPasses)
            : std::numeric_limits<float>::quiet_NaN();
    distribution_measurement.timing_finite = timing_finite;
    const float speedup = distribution_measurement.baseline_milliseconds /
                          distribution_measurement.candidate_milliseconds;
    const float required_speedup =
        kMinimumCellSpeedups[distribution_index];
    const bool cell_gate = distribution_measurement.bitwise_equal &&
                           distribution_measurement.output_finite &&
                           distribution_measurement.timing_finite &&
                           std::isfinite(speedup) &&
                           speedup >= required_speedup;
    std::cout << "PERF_FP8_M1_SWIZZLED_CODEBOOK_CELL: " << shape.label
              << " distribution="
              << fp8_m2_code_distribution_name(distribution)
              << " baseline_quad_cap=" << shape.selected_quad_cap
              << " candidate_swizzled_cap=" << shape.selected_quad_cap
              << " baseline_ms="
              << distribution_measurement.baseline_milliseconds
              << " candidate_ms="
              << distribution_measurement.candidate_milliseconds
              << " speedup=" << speedup
              << " required_speedup=" << required_speedup
              << " bitwise="
              << (distribution_measurement.bitwise_equal ? "true" : "false")
              << " output_finite="
              << (distribution_measurement.output_finite ? "true" : "false")
              << " timing_finite="
              << (distribution_measurement.timing_finite ? "true" : "false")
              << " hard_gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
    test.expect(cell_gate,
                label + " clears bitwise/finite and speedup hard gate");
  }
  return measurement;
}

void run_optional_fp8_m1_swizzled_codebook_performance(
    TestContext& test, cudaStream_t stream) {
  if (!fp8_m1_swizzled_codebook_performance_enabled()) {
    std::cout
        << "SKIP: FP8 M1 swizzled-codebook performance segment; set "
           "Q3X_RUN_SM87_FP8_M1_SWIZZLED_CODEBOOK_PERF=1 to enable\n";
    return;
  }

  constexpr std::array<Fp8M1RowQuadShape, 4U> kShapes{{
      {10'240U, 5'120U, 48U, 1'536U,
       "FP8 M1 swizzled QKV 10240x5120"},
      {5'120U, 6'144U, 64U, 1'280U,
       "FP8 M1 swizzled projection 5120x6144"},
      {6'144U, 5'120U, 48U, 768U,
       "FP8 M1 swizzled projection 6144x5120"},
      {12'288U, 5'120U, 16U, 2'048U,
       "FP8 M1 swizzled QKV 12288x5120"},
  }};
  constexpr std::array<double, 2U> kMinimumWeightedSpeedups{{
      1.60,
      2.20,
  }};

  std::array<Fp8M1SwizzledCodebookMeasurement, kShapes.size()>
      measurements{};
  std::array<double, kFp8M1RowQuadDistributions.size()>
      weighted_speedups{};
  bool all_cells_pass = true;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    const Fp8M1RowQuadShape& shape = kShapes[shape_index];
    const bool frozen_cap =
        q3x::kernels::sm87_fp8_m1_row_quad_maximum_blocks_test(
            shape.rows, shape.columns) == shape.selected_quad_cap;
    test.expect(frozen_cap,
                std::string(shape.label) +
                    " uses the frozen production row-quad cap");
    all_cells_pass = all_cells_pass && frozen_cap;
    measurements[shape_index] =
        benchmark_fp8_m1_swizzled_codebook_shape(test, stream, shape);
  }

  for (std::size_t distribution_index = 0U;
       distribution_index < kFp8M1RowQuadDistributions.size();
       ++distribution_index) {
    double weighted_baseline = 0.0;
    double weighted_candidate = 0.0;
    bool distribution_gate = true;
    for (std::size_t shape_index = 0U; shape_index < kShapes.size();
         ++shape_index) {
      const auto& cell =
          measurements[shape_index].distributions[distribution_index];
      const double cell_speedup =
          static_cast<double>(cell.baseline_milliseconds) /
          static_cast<double>(cell.candidate_milliseconds);
      const double required_cell_speedup =
          distribution_index == 0U ? 1.50 : 2.00;
      const bool cell_gate = cell.bitwise_equal && cell.output_finite &&
                             cell.timing_finite &&
                             std::isfinite(cell_speedup) &&
                             cell_speedup >= required_cell_speedup;
      distribution_gate = distribution_gate && cell_gate;
      weighted_baseline +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          cell.baseline_milliseconds;
      weighted_candidate +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          cell.candidate_milliseconds;
    }
    const double weighted_speedup = weighted_baseline / weighted_candidate;
    weighted_speedups[distribution_index] = weighted_speedup;
    const double required_speedup =
        kMinimumWeightedSpeedups[distribution_index];
    const bool weighted_finite = std::isfinite(weighted_speedup) &&
                                 weighted_baseline > 0.0 &&
                                 weighted_candidate > 0.0;
    distribution_gate = distribution_gate && weighted_finite &&
                        weighted_speedup >= required_speedup;
    all_cells_pass = all_cells_pass && distribution_gate;
    std::cout << "PERF_FP8_M1_SWIZZLED_CODEBOOK_WEIGHTED: distribution="
              << fp8_m2_code_distribution_name(
                     kFp8M1RowQuadDistributions[distribution_index])
              << " frozen_caps=1536:1280:768:2048"
              << " weighted_baseline_ms=" << weighted_baseline
              << " weighted_candidate_ms=" << weighted_candidate
              << " speedup=" << weighted_speedup
              << " required_speedup=" << required_speedup
              << " profile_calls=48:64:48:16"
              << " hard_gate="
              << (distribution_gate ? "PASS" : "FAIL") << '\n';
    test.expect(distribution_gate,
                std::string("FP8 M1 swizzled ") +
                    fp8_m2_code_distribution_name(
                        kFp8M1RowQuadDistributions[distribution_index]) +
                    " weighted bitwise/finite and speedup gate passes");
  }

  std::cout << "PERF_FP8_M1_SWIZZLED_CODEBOOK_SELECTED:"
            << " baseline=current_row_quad candidate=swizzled_codebook"
            << " frozen_caps=1536:1280:768:2048"
            << " distributions=checkpoint_like:same_bank_stress"
            << " profile_calls=48:64:48:16"
            << " checkpoint_weighted_speedup=" << weighted_speedups[0U]
            << " stress_weighted_speedup=" << weighted_speedups[1U]
            << " hard_gate=" << (all_cells_pass ? "PASS" : "FAIL")
            << '\n';
  test.expect(all_cells_pass,
              "FP8 M1 swizzled codebook clears every correctness and "
              "performance gate");
}

struct Fp8M1RowPairDistributionMeasurement {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
};

struct Fp8M1RowPairMeasurement {
  std::array<Fp8M1RowPairDistributionMeasurement, 2U> distributions{};
};

[[nodiscard]] Fp8M1RowPairMeasurement benchmark_fp8_m1_row_pair_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kProductionGridCap = 2'048U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 40;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale = 1.0F / 64.0F;
  constexpr std::array<Fp8M2CodeDistribution, 2U> kDistributions{{
      Fp8M2CodeDistribution::kCheckpointLike,
      Fp8M2CodeDistribution::kSameBankStress,
  }};

  std::vector<std::uint8_t> host_weights(rows * columns);
  std::vector<std::uint16_t> host_activation(columns);
  for (std::size_t column = 0U; column < columns; ++column) {
    const int centered = static_cast<int>((column * 17U + 5U) % 127U) - 63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activation.allocate(host_activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  if (!ready) {
    return {};
  }

  Fp8M1RowPairMeasurement measurement;
  for (std::size_t distribution_index = 0U;
       distribution_index < kDistributions.size(); ++distribution_index) {
    const Fp8M2CodeDistribution distribution =
        kDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + fp8_m2_code_distribution_name(distribution);
    fill_fp8_m2_code_distribution(host_weights, rows, columns, distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(weights.get(), host_weights.data(),
                        host_weights.size(), cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize weights");
    if (!ready) {
      return measurement;
    }

    const auto launch_baseline = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_fp8_w8a16_gemv_bf16_grid_cap_test_cuda(
              weights.get(), kWeightScale, activation.get(), rows, columns,
              baseline_output.get(), kProductionGridCap,
              static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_fp8_w8a16_gemv_bf16_row_pair_test_cuda(
              weights.get(), kWeightScale, activation.get(), rows, columns,
              candidate_output.get(), static_cast<void*>(stream));
    };

    ready = test.cuda_ok(
        static_cast<cudaError_t>(launch_baseline()),
        distribution_label +
            " correctness preserved production cap2048 baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         distribution_label +
                             " correctness direct row-pair");
    std::vector<std::uint16_t> baseline(rows);
    std::vector<std::uint16_t> candidate(rows);
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy baseline output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy direct row-pair output");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " correctness synchronize");
    if (!ready) {
      return measurement;
    }

    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < baseline.size(); ++index) {
      mismatches += baseline[index] != candidate[index] ? 1U : 0U;
    }
    Fp8M1RowPairDistributionMeasurement& distribution_measurement =
        measurement.distributions[distribution_index];
    distribution_measurement.bitwise_equal = mismatches == 0U;
    test.expect(distribution_measurement.bitwise_equal,
                distribution_label +
                    " direct row-pair matches every preserved production "
                    "BF16 bit");

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           distribution_label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           distribution_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " warmup synchronize");
    if (!ready) {
      return measurement;
    }

    double baseline_total = 0.0;
    double candidate_total = 0.0;
    bool all_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          distribution_label + " round=" + std::to_string(round + 1);
      const float baseline_first = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 1");
      const float candidate_first = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 1");
      const float candidate_second = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 2");
      const float baseline_second = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 2");
      const bool round_finite =
          std::isfinite(baseline_first) &&
          std::isfinite(candidate_first) &&
          std::isfinite(candidate_second) &&
          std::isfinite(baseline_second);
      all_finite = all_finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_FP8_M1_ROW_PAIR_ROUND: " << label
                << " distribution="
                << fp8_m2_code_distribution_name(distribution)
                << " round=" << round + 1
                << " baseline_pass1_ms=" << baseline_first
                << " candidate_pass1_ms=" << candidate_first
                << " candidate_pass2_ms=" << candidate_second
                << " baseline_pass2_ms=" << baseline_second << '\n';
    }
    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    distribution_measurement.baseline_milliseconds =
        all_finite ? static_cast<float>(baseline_total / kTimedPasses)
                   : std::numeric_limits<float>::quiet_NaN();
    distribution_measurement.candidate_milliseconds =
        all_finite ? static_cast<float>(candidate_total / kTimedPasses)
                   : std::numeric_limits<float>::quiet_NaN();
    const float speedup = distribution_measurement.baseline_milliseconds /
                          distribution_measurement.candidate_milliseconds;
    std::cout << "PERF_FP8_M1_ROW_PAIR: " << label << " distribution="
              << fp8_m2_code_distribution_name(distribution)
              << " preserved_production_cap2048_ms="
              << distribution_measurement.baseline_milliseconds
              << " direct_row_pair_ms="
              << distribution_measurement.candidate_milliseconds
              << " speedup=" << speedup << " bitwise_mismatches="
              << mismatches << '/' << baseline.size() << '\n';
  }
  return measurement;
}

void run_optional_fp8_m1_row_pair_performance(TestContext& test,
                                               cudaStream_t stream) {
  if (!fp8_m1_row_pair_performance_enabled()) {
    std::cout << "SKIP: FP8 M1 row-pair performance segment; set "
                 "Q3X_RUN_SM87_FP8_M1_ROW_PAIR_PERF=1 to enable\n";
    return;
  }
  constexpr float kMinimumCheckpointShapeSpeedup = 1.02F;
  constexpr float kMinimumStressShapeSpeedup = 0.99F;
  constexpr float kMinimumCheckpointWeightedSpeedup = 1.03F;
  struct Shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t checkpoint_calls;
    const char* label;
  };
  constexpr std::array<Shape, 6U> kShapes{{
      {10'240U, 5'120U, 48U, "FP8 M1 row-pair QKV 10240x5120"},
      {5'120U, 6'144U, 64U, "FP8 M1 row-pair projection 5120x6144"},
      {6'144U, 5'120U, 48U, "FP8 M1 row-pair projection 6144x5120"},
      {12'288U, 5'120U, 16U, "FP8 M1 row-pair QKV 12288x5120"},
      {1'024U, 5'120U, 32U, "FP8 M1 row-pair small 1024x5120"},
      {5'120U, 5'120U, 0U, "FP8 M1 row-pair square 5120x5120"},
  }};
  constexpr std::array<Fp8M2CodeDistribution, 2U> kDistributions{{
      Fp8M2CodeDistribution::kCheckpointLike,
      Fp8M2CodeDistribution::kSameBankStress,
  }};
  std::array<Fp8M1RowPairMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_fp8_m1_row_pair_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }

  bool all_shape_distributions_pass = true;
  double checkpoint_weighted_baseline = 0.0;
  double checkpoint_weighted_candidate = 0.0;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    for (std::size_t distribution_index = 0U;
         distribution_index < kDistributions.size(); ++distribution_index) {
      const Fp8M1RowPairDistributionMeasurement& measurement =
          measurements[shape_index].distributions[distribution_index];
      const float speedup = measurement.baseline_milliseconds /
                            measurement.candidate_milliseconds;
      const bool finite =
          std::isfinite(measurement.baseline_milliseconds) &&
          std::isfinite(measurement.candidate_milliseconds) &&
          std::isfinite(speedup);
      const float required_speedup =
          kDistributions[distribution_index] ==
                  Fp8M2CodeDistribution::kCheckpointLike
              ? kMinimumCheckpointShapeSpeedup
              : kMinimumStressShapeSpeedup;
      const bool gate = measurement.bitwise_equal && finite &&
                        speedup >= required_speedup;
      all_shape_distributions_pass =
          all_shape_distributions_pass && gate;
      test.expect(
          gate,
          std::string(kShapes[shape_index].label) + " " +
              fp8_m2_code_distribution_name(
                  kDistributions[distribution_index]) +
              " clears the M1 row-pair performance gate");
      std::cout << "PERF_FP8_M1_ROW_PAIR_VALIDATION: "
                << kShapes[shape_index].label << " distribution="
                << fp8_m2_code_distribution_name(
                       kDistributions[distribution_index])
                << " preserved_production_cap2048_ms="
                << measurement.baseline_milliseconds
                << " direct_row_pair_ms="
                << measurement.candidate_milliseconds
                << " speedup=" << speedup
                << " required_speedup=" << required_speedup << " bitwise="
                << (measurement.bitwise_equal ? "true" : "false")
                << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
    }
    const Fp8M1RowPairDistributionMeasurement& checkpoint =
        measurements[shape_index].distributions[0U];
    checkpoint_weighted_baseline +=
        static_cast<double>(kShapes[shape_index].checkpoint_calls) *
        checkpoint.baseline_milliseconds;
    checkpoint_weighted_candidate +=
        static_cast<double>(kShapes[shape_index].checkpoint_calls) *
        checkpoint.candidate_milliseconds;
  }
  const double checkpoint_weighted_speedup =
      checkpoint_weighted_baseline / checkpoint_weighted_candidate;
  const bool aggregate_gate =
      all_shape_distributions_pass &&
      std::isfinite(checkpoint_weighted_speedup) &&
      checkpoint_weighted_speedup >= kMinimumCheckpointWeightedSpeedup;
  std::cout << "PERF_FP8_M1_ROW_PAIR_AGGREGATE: "
            << "checkpoint_weighted_preserved_production_cap2048_ms="
            << checkpoint_weighted_baseline
            << " checkpoint_weighted_direct_row_pair_ms="
            << checkpoint_weighted_candidate
            << " speedup=" << checkpoint_weighted_speedup
            << " required_speedup=" << kMinimumCheckpointWeightedSpeedup
            << " profile_calls=48:64:48:16:32:0 all_shape_distributions="
            << (all_shape_distributions_pass ? "PASS" : "FAIL")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "FP8 M1 row-pair clears every production gate");
}

struct Fp8M8FixedShapeMeasurement {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
};

[[nodiscard]] Fp8M8FixedShapeMeasurement
benchmark_fp8_m8_fixed_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 8U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
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
                       label + " initialize token-distinct activations");
  if (!ready) {
    return {};
  }

  const auto launch_baseline = [&]() noexcept -> int {
    return q3x::kernels::launch_sm87_fp8_w8a16_small_m8_row_pair_test_cuda(
        weights.get(), kWeightScale, activations.get(), rows, columns,
        baseline_output.get(), static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() noexcept -> int {
    return q3x::kernels::launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
        weights.get(), kWeightScale, activations.get(), kTokens, rows, columns,
        candidate_output.get(), static_cast<void*>(stream));
  };

  ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                       label + " correctness generic row-pair baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate()),
                       label + " correctness fixed-shape candidate");
  std::vector<std::uint16_t> baseline(kTokens * rows);
  std::vector<std::uint16_t> candidate(kTokens * rows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), baseline_output.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy generic row-pair output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           candidate.data(), candidate_output.get(),
                           candidate.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy fixed-shape output");
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
  std::cout << "FP8_M8_FIXED_SHAPE_DIFF: " << label
            << " candidate_vs_generic_row_pair_bf16=" << mismatches << '/'
            << baseline.size() << '\n';
  test.expect(bitwise_equal,
              label + " fixed shape matches every generic row-pair BF16 bit");

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

  double baseline_total = 0.0;
  double candidate_total = 0.0;
  bool finite = true;
  for (int round = 0; round < kMeasurementRounds; ++round) {
    const std::string round_suffix =
        " round=" + std::to_string(round + 1);
    const float baseline_first = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        label + " baseline pass 1" + round_suffix);
    const float candidate_first = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        label + " candidate pass 1" + round_suffix);
    const float candidate_second = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        label + " candidate pass 2" + round_suffix);
    const float baseline_second = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        label + " baseline pass 2" + round_suffix);
    const bool round_finite =
        std::isfinite(baseline_first) && std::isfinite(candidate_first) &&
        std::isfinite(candidate_second) && std::isfinite(baseline_second);
    finite = finite && round_finite;
    if (round_finite) {
      baseline_total += baseline_first + baseline_second;
      candidate_total += candidate_first + candidate_second;
    }
    std::cout << "PERF_FP8_M8_FIXED_SHAPE_ROUND: " << label
              << " round=" << round + 1
              << " baseline_pass1_ms=" << baseline_first
              << " candidate_pass1_ms=" << candidate_first
              << " candidate_pass2_ms=" << candidate_second
              << " baseline_pass2_ms=" << baseline_second << '\n';
  }
  if (!finite) {
    return {};
  }
  constexpr double kTimedPasses =
      2.0 * static_cast<double>(kMeasurementRounds);
  const float baseline_average =
      static_cast<float>(baseline_total / kTimedPasses);
  const float candidate_average =
      static_cast<float>(candidate_total / kTimedPasses);
  const float speedup = baseline_average / candidate_average;
  std::cout << "PERF_FP8_M8_FIXED_SHAPE: " << label
            << " baseline_average_ms=" << baseline_average
            << " candidate_average_ms=" << candidate_average
            << " speedup=" << speedup
            << " uplift_percent=" << (speedup - 1.0F) * 100.0F
            << " bitwise_mismatches=" << mismatches << '/' << baseline.size()
            << '\n';
  return {baseline_average, candidate_average, bitwise_equal};
}

void run_optional_fp8_m8_fixed_shape_performance(TestContext& test,
                                                  cudaStream_t stream) {
  if (!fp8_m8_fixed_shape_performance_enabled()) {
    std::cout << "SKIP: FP8 M8 fixed-shape performance segment; set "
                 "Q3X_RUN_SM87_FP8_M8_FIXED_SHAPE_PERF=1 to enable\n";
    return;
  }
  constexpr float kMinimumWeightedSpeedup = 1.03F;
  constexpr float kMinimumPerShapeSpeedup = 0.98F;
  struct Shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t calls_per_prompt;
    const char* label;
  };
  constexpr std::array<Shape, 5U> kShapes{{
      {10'240U, 5'120U, 48U, "FP8 M8 fixed 10240x5120"},
      {5'120U, 6'144U, 64U, "FP8 M8 fixed 5120x6144"},
      {6'144U, 5'120U, 48U, "FP8 M8 fixed 6144x5120"},
      {12'288U, 5'120U, 16U, "FP8 M8 fixed 12288x5120"},
      {1'024U, 5'120U, 32U, "FP8 M8 fixed 1024x5120"},
  }};
  std::array<Fp8M8FixedShapeMeasurement, kShapes.size()> measurements{};
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    measurements[index] = benchmark_fp8_m8_fixed_shape(
        test, stream, kShapes[index].rows, kShapes[index].columns,
        kShapes[index].label);
  }

  double weighted_baseline = 0.0;
  double weighted_candidate = 0.0;
  bool all_bitwise = true;
  bool all_finite = true;
  bool no_shape_regression = true;
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    const Fp8M8FixedShapeMeasurement& measurement = measurements[index];
    const bool finite =
        std::isfinite(measurement.baseline_milliseconds) &&
        std::isfinite(measurement.candidate_milliseconds);
    const float speedup =
        measurement.baseline_milliseconds / measurement.candidate_milliseconds;
    all_bitwise = all_bitwise && measurement.bitwise_equal;
    all_finite = all_finite && finite;
    no_shape_regression = no_shape_regression && finite &&
                          speedup >= kMinimumPerShapeSpeedup;
    weighted_baseline +=
        static_cast<double>(kShapes[index].calls_per_prompt) *
        measurement.baseline_milliseconds;
    weighted_candidate +=
        static_cast<double>(kShapes[index].calls_per_prompt) *
        measurement.candidate_milliseconds;
    std::cout << "PERF_FP8_M8_FIXED_SHAPE_VALIDATION: "
              << kShapes[index].label << " baseline_ms="
              << measurement.baseline_milliseconds << " candidate_ms="
              << measurement.candidate_milliseconds << " speedup=" << speedup
              << " minimum_speedup=" << kMinimumPerShapeSpeedup
              << " gate="
              << (finite && speedup >= kMinimumPerShapeSpeedup ? "PASS"
                                                               : "FAIL")
              << '\n';
  }
  const double weighted_speedup = weighted_baseline / weighted_candidate;
  const bool aggregate_gate =
      all_bitwise && all_finite && no_shape_regression &&
      std::isfinite(weighted_speedup) &&
      weighted_speedup >= kMinimumWeightedSpeedup;
  std::cout << "PERF_FP8_M8_FIXED_SHAPE_AGGREGATE: weighted_baseline_ms="
            << weighted_baseline
            << " weighted_candidate_ms=" << weighted_candidate
            << " speedup=" << weighted_speedup
            << " required_speedup=" << kMinimumWeightedSpeedup
            << " all_bitwise=" << (all_bitwise ? "true" : "false")
            << " no_shape_regression="
            << (no_shape_regression ? "true" : "false")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "FP8 M8 fixed shapes clear every production gate");
}

struct NvFp4M16WmmaMeasurement {
  double baseline_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double candidate_milliseconds = std::numeric_limits<double>::quiet_NaN();
  bool deterministic = false;
  bool within_tolerance = false;
};

struct NvFp4M16OutputCheck {
  bool deterministic = false;
  bool within_tolerance = false;
  std::size_t reference_nan_count = 0U;
};

[[nodiscard]] NvFp4M16OutputCheck check_nvfp4_m16_outputs(
    TestContext& test, const std::vector<std::uint16_t>& candidate,
    const std::vector<std::uint16_t>& baseline,
    const std::vector<std::uint16_t>& replay, const std::size_t columns,
    const std::size_t expected_reference_nan_count,
    const std::string& label) {
  test.expect(candidate.size() == baseline.size(),
              label + " candidate/baseline output size");
  test.expect(candidate.size() == replay.size(),
              label + " candidate/replay output size");
  const std::size_t count =
      std::min(candidate.size(), std::min(baseline.size(), replay.size()));
  std::size_t bf16_mismatches = 0U;
  std::size_t deterministic_mismatches = 0U;
  std::size_t reference_nan_count = 0U;
  std::size_t nan_class_mismatches = 0U;
  float maximum_absolute_error = 0.0F;
  float maximum_relative_error = 0.0F;
  bool within_tolerance = true;
  for (std::size_t index = 0U; index < count; ++index) {
    bf16_mismatches += candidate[index] != baseline[index] ? 1U : 0U;
    deterministic_mismatches += candidate[index] != replay[index] ? 1U : 0U;
    if (is_bf16_nan(baseline[index])) {
      ++reference_nan_count;
      const bool candidate_is_nan = is_bf16_nan(candidate[index]);
      nan_class_mismatches += candidate_is_nan ? 0U : 1U;
      within_tolerance = within_tolerance && candidate_is_nan;
      continue;
    }
    const float baseline_value = decode_bf16(baseline[index]);
    const float candidate_value = decode_bf16(candidate[index]);
    const float absolute_error =
        std::fabs(candidate_value - baseline_value);
    const float relative_error =
        absolute_error / std::max(1.0e-6F, std::fabs(baseline_value));
    maximum_absolute_error =
        std::max(maximum_absolute_error, absolute_error);
    maximum_relative_error =
        std::max(maximum_relative_error, relative_error);
    const float tolerance =
        2.0e-4F * std::sqrt(static_cast<float>(columns)) +
        1.0e-2F * std::max(1.0F, std::fabs(baseline_value));
    within_tolerance = within_tolerance &&
                       std::isfinite(candidate_value) &&
                       std::isfinite(baseline_value) &&
                       absolute_error <= tolerance;
  }
  const bool deterministic = deterministic_mismatches == 0U;
  const bool nan_coverage =
      reference_nan_count == expected_reference_nan_count &&
      nan_class_mismatches == 0U;
  std::cout << "NVFP4_M16_WMMA_DIFF: " << label
            << " candidate_vs_two_m8_bf16=" << bf16_mismatches << '/'
            << count << " deterministic_mismatches="
            << deterministic_mismatches << '/' << count
            << " reference_nan_count=" << reference_nan_count
            << " nan_class_mismatches=" << nan_class_mismatches
            << " max_abs=" << maximum_absolute_error
            << " max_rel=" << maximum_relative_error
            << " tolerance_gate=" << (within_tolerance ? "PASS" : "FAIL")
            << '\n';
  test.expect(deterministic, label + " candidate replay is bitwise stable");
  test.expect(nan_coverage, label + " has the required NaN-class coverage");
  test.expect(within_tolerance,
              label + " clears the existing numerical tolerance");
  return {deterministic, within_tolerance && nan_coverage,
          reference_nan_count};
}

[[nodiscard]] NvFp4M16WmmaMeasurement benchmark_nvfp4_m16_wmma_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 16U;
  constexpr std::size_t kHalfTokens = 8U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  // Deliberately not a power-of-two scale. Together with the odd activation
  // denominators below this prevents an accidentally exact-only fixture.
  constexpr float kWeightScale2 = 1.0F / 37.0F;
  constexpr std::array<std::uint8_t, 16U> kFiniteScaleCodes{{
      0x00U, 0x80U, 0x08U, 0x88U, 0x18U, 0x98U, 0x20U, 0xa0U,
      0x28U, 0xa8U, 0x30U, 0xb0U, 0x38U, 0xb8U, 0x40U, 0xc0U,
  }};
  const std::size_t packed_columns = columns / 2U;
  const std::size_t scale_columns = columns / 16U;

  std::vector<std::uint8_t> host_packed(rows * packed_columns);
  std::array<bool, 16U> low_nibble_covered{};
  std::array<bool, 16U> high_nibble_covered{};
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t packed_column = 0U; packed_column < packed_columns;
         ++packed_column) {
      const std::uint8_t low = static_cast<std::uint8_t>(
          (packed_column + row * 3U + (packed_column >> 3U)) & 0x0fU);
      const std::uint8_t high = static_cast<std::uint8_t>(
          (packed_column * 5U + row * 7U + (packed_column >> 3U) * 3U + 1U) &
          0x0fU);
      host_packed[row * packed_columns + packed_column] =
          static_cast<std::uint8_t>(low | (high << 4U));
      low_nibble_covered[low] = true;
      high_nibble_covered[high] = true;
    }
  }
  test.expect(std::all_of(low_nibble_covered.begin(),
                          low_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers all E2M1 low-nibble codes");
  test.expect(std::all_of(high_nibble_covered.begin(),
                          high_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers all E2M1 high-nibble codes");

  std::vector<std::uint8_t> host_scales(rows * scale_columns);
  std::array<bool, kFiniteScaleCodes.size()> scale_codes_covered{};
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t group = 0U; group < scale_columns; ++group) {
      const std::size_t code_index =
          (row * 5U + group * 7U + (group >> 3U)) & 0x0fU;
      host_scales[row * scale_columns + group] =
          kFiniteScaleCodes[code_index];
      scale_codes_covered[code_index] = true;
    }
  }
  test.expect(std::all_of(scale_codes_covered.begin(),
                          scale_codes_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers every finite scale fixture code");

  std::vector<std::uint16_t> host_activations(kTokens * columns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered =
          static_cast<int>((column * 17U + token * 19U + 5U) % 31U) - 15;
      const int odd_denominator =
          3 + 2 * static_cast<int>((column * 7U + token * 3U) % 11U);
      const int exponent =
          static_cast<int>((column * 5U + token * 7U) % 5U) - 3;
      host_activations[token * columns + column] = encode_bf16(std::ldexp(
          static_cast<float>(centered) /
              static_cast<float>(odd_denominator),
          exponent));
    }
  }

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  DeviceBuffer<std::uint16_t> replay_output;
  bool ready = test.cuda_ok(packed.allocate(host_packed.size()),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(host_scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(replay_output.allocate(kTokens * rows),
                                label + " allocate replay output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(packed.get(), host_packed.data(),
                                       host_packed.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize packed E2M1 weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(scales.get(), host_scales.data(),
                                       host_scales.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize grouped scale codes");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize nonpower activations");
  if (!ready) {
    return {};
  }

  const auto launch_baseline = [&]() noexcept -> int {
    int status =
        q3x::kernels::launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
            packed.get(), scales.get(), kWeightScale2, activations.get(),
            kHalfTokens, rows, columns, baseline_output.get(),
            static_cast<void*>(stream));
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    return q3x::kernels::launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
        packed.get(), scales.get(), kWeightScale2,
        activations.get() + kHalfTokens * columns, kHalfTokens, rows, columns,
        baseline_output.get() + kHalfTokens * rows,
        static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() noexcept -> int {
    return q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
        packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
        columns, candidate_output.get(), static_cast<void*>(stream));
  };
  const auto launch_replay = [&]() noexcept -> int {
    return q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
        packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
        columns, replay_output.get(), static_cast<void*>(stream));
  };

  ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                       label + " correctness two-M8 baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate()),
                       label + " correctness M16 WMMA candidate");
  ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_replay()),
                                label + " deterministic replay");
  std::vector<std::uint16_t> baseline(kTokens * rows);
  std::vector<std::uint16_t> candidate(kTokens * rows);
  std::vector<std::uint16_t> replay(kTokens * rows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), baseline_output.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy two-M8 baseline");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           candidate.data(), candidate_output.get(),
                           candidate.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy M16 candidate");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           replay.data(), replay_output.get(),
                           replay.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy M16 replay");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " correctness synchronize");
  if (!ready) {
    return {};
  }
  const NvFp4M16OutputCheck finite_check = check_nvfp4_m16_outputs(
      test, candidate, baseline, replay, columns, 0U,
      label + " full E2M1/group-scale fixture");

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

  double baseline_total = 0.0;
  double candidate_total = 0.0;
  bool finite_timing = true;
  for (int round = 0; round < kMeasurementRounds; ++round) {
    const std::string suffix = " round=" + std::to_string(round + 1);
    const float baseline_first = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        label + " baseline pass 1" + suffix);
    const float candidate_first = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        label + " candidate pass 1" + suffix);
    const float candidate_second = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        label + " candidate pass 2" + suffix);
    const float baseline_second = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        label + " baseline pass 2" + suffix);
    const bool round_finite =
        std::isfinite(baseline_first) && std::isfinite(candidate_first) &&
        std::isfinite(candidate_second) && std::isfinite(baseline_second);
    finite_timing = finite_timing && round_finite;
    if (round_finite) {
      baseline_total += baseline_first + baseline_second;
      candidate_total += candidate_first + candidate_second;
    }
    std::cout << "PERF_NVFP4_M16_WMMA_ROUND: " << label
              << " round=" << round + 1
              << " baseline_pass1_ms=" << baseline_first
              << " candidate_pass1_ms=" << candidate_first
              << " candidate_pass2_ms=" << candidate_second
              << " baseline_pass2_ms=" << baseline_second << '\n';
  }
  constexpr double kTimedPasses =
      2.0 * static_cast<double>(kMeasurementRounds);
  const double baseline_average = baseline_total / kTimedPasses;
  const double candidate_average = candidate_total / kTimedPasses;
  std::cout << "PERF_NVFP4_M16_WMMA: " << label
            << " baseline_two_m8_ms=" << baseline_average
            << " candidate_m16_ms=" << candidate_average
            << " speedup=" << baseline_average / candidate_average << '\n';

  // Reuse the timed buffers for an exhaustive 256 scale-code x 16 E2M1
  // matrix. Each of the first 4096 rows has exactly one selected nibble and
  // one selected 16-value scale group. Critical K positions cross nibble,
  // scale-group, vector-load, and K64 staging boundaries. All other values
  // are zero, which also checks the required 0 * NaN propagation inside the
  // two reserved E4M3FN scale groups.
  constexpr std::size_t kScaleCodes = 256U;
  constexpr std::size_t kNibbles = 16U;
  constexpr std::size_t kExhaustiveRows = kScaleCodes * kNibbles;
  constexpr std::size_t kNanCombinations = 2U * kNibbles;
  constexpr std::array<std::size_t, 12U> kTargetColumns{{
      0U, 1U, 15U, 16U, 31U, 32U, 47U, 48U, 62U, 63U, 64U, 65U,
  }};
  std::fill(host_packed.begin(), host_packed.end(), 0U);
  std::fill(host_scales.begin(), host_scales.end(), 0U);
  std::array<bool, kTargetColumns.size()> target_columns_covered{};
  for (std::size_t scale_code = 0U; scale_code < kScaleCodes;
       ++scale_code) {
    for (std::size_t nibble = 0U; nibble < kNibbles; ++nibble) {
      const std::size_t row = scale_code * kNibbles + nibble;
      const std::size_t target_index = row % kTargetColumns.size();
      const std::size_t column = kTargetColumns[target_index];
      std::uint8_t& packed_byte =
          host_packed[row * packed_columns + column / 2U];
      if ((column & 1U) == 0U) {
        packed_byte = static_cast<std::uint8_t>(nibble);
      } else {
        packed_byte = static_cast<std::uint8_t>(nibble << 4U);
      }
      host_scales[row * scale_columns + column / 16U] =
          static_cast<std::uint8_t>(scale_code);
      target_columns_covered[target_index] = true;
    }
  }
  test.expect(rows >= kExhaustiveRows,
              label + " has room for all scale x nibble rows");
  test.expect(std::all_of(target_columns_covered.begin(),
                          target_columns_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers every K boundary fixture");
  ready = test.cuda_ok(
      cudaMemcpyAsync(packed.get(), host_packed.data(), host_packed.size(),
                      cudaMemcpyHostToDevice, stream),
      label + " initialize exhaustive scale/nibble weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(scales.get(), host_scales.data(),
                                       host_scales.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize exhaustive scale codes");
  ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                                label + " exhaustive two-M8 baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate()),
                       label + " exhaustive M16 candidate");
  ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_replay()),
                                label + " exhaustive deterministic replay");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), baseline_output.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy exhaustive baseline");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           candidate.data(), candidate_output.get(),
                           candidate.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy exhaustive candidate");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           replay.data(), replay_output.get(),
                           replay.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy exhaustive replay");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " exhaustive synchronize");
  if (!ready) {
    return {};
  }
  constexpr std::size_t kExpectedNanOutputs = kNanCombinations * kTokens;
  const NvFp4M16OutputCheck nan_check = check_nvfp4_m16_outputs(
      test, candidate, baseline, replay, columns, kExpectedNanOutputs,
      label + " exhaustive scale x E2M1 fixture");
  std::size_t positive_nan_row_count = 0U;
  std::size_t negative_nan_row_count = 0U;
  bool token_boundary_covered = true;
  constexpr std::size_t kPositiveNanFirstRow = 0x7fU * kNibbles;
  constexpr std::size_t kNegativeNanFirstRow = 0xffU * kNibbles;
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t nibble = 0U; nibble < kNibbles; ++nibble) {
      positive_nan_row_count +=
          is_bf16_nan(
              baseline[token * rows + kPositiveNanFirstRow + nibble])
              ? 1U
              : 0U;
      negative_nan_row_count +=
          is_bf16_nan(
              baseline[token * rows + kNegativeNanFirstRow + nibble])
              ? 1U
              : 0U;
    }
  }
  for (const std::size_t token : {7U, 8U}) {
    for (const std::size_t first_row : {kPositiveNanFirstRow,
                                        kNegativeNanFirstRow}) {
      for (std::size_t nibble = 0U; nibble < kNibbles; ++nibble) {
        const std::size_t row = first_row + nibble;
        token_boundary_covered =
            token_boundary_covered &&
            is_bf16_nan(baseline[token * rows + row]) &&
            is_bf16_nan(candidate[token * rows + row]);
      }
    }
  }
  test.expect(positive_nan_row_count == kNibbles * kTokens,
              label + " propagates all 0x7f x E2M1 combinations");
  test.expect(negative_nan_row_count == kNibbles * kTokens,
              label + " propagates all 0xff x E2M1 combinations");
  test.expect(token_boundary_covered,
              label + " preserves NaN class across token 7/8");

  const bool deterministic =
      finite_check.deterministic && nan_check.deterministic;
  const bool within_tolerance =
      finite_check.within_tolerance && nan_check.within_tolerance;
  return {finite_timing ? baseline_average
                        : std::numeric_limits<double>::quiet_NaN(),
          finite_timing ? candidate_average
                        : std::numeric_limits<double>::quiet_NaN(),
          deterministic, within_tolerance};
}

void run_optional_nvfp4_m16_wmma_performance(TestContext& test,
                                              cudaStream_t stream) {
  if (!nvfp4_m16_wmma_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M16 WMMA performance segment; set "
                 "Q3X_RUN_SM87_NVFP4_M16_WMMA_PERF=1 to enable\n";
    return;
  }

  constexpr double kMinimumPerShapeSpeedup = 1.10;
  constexpr double kMinimumWeightedSpeedup = 1.15;
  struct Shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t calls_per_prompt;
    const char* label;
  };
  constexpr std::array<Shape, 2U> kShapes{{
      {17'408U, 5'120U, 256U, "NVFP4 M16 WMMA 17408x5120"},
      {5'120U, 17'408U, 128U, "NVFP4 M16 WMMA 5120x17408"},
  }};
  std::array<NvFp4M16WmmaMeasurement, kShapes.size()> measurements{};
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    measurements[index] = benchmark_nvfp4_m16_wmma_shape(
        test, stream, kShapes[index].rows, kShapes[index].columns,
        kShapes[index].label);
  }

  double weighted_baseline = 0.0;
  double weighted_candidate = 0.0;
  bool all_shapes_pass = true;
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    const NvFp4M16WmmaMeasurement& measurement = measurements[index];
    const double speedup = measurement.baseline_milliseconds /
                           measurement.candidate_milliseconds;
    const bool finite =
        std::isfinite(measurement.baseline_milliseconds) &&
        std::isfinite(measurement.candidate_milliseconds) &&
        std::isfinite(speedup);
    const bool shape_gate =
        finite && measurement.deterministic && measurement.within_tolerance &&
        speedup >= kMinimumPerShapeSpeedup;
    all_shapes_pass = all_shapes_pass && shape_gate;
    test.expect(shape_gate,
                std::string(kShapes[index].label) +
                    " clears correctness and per-shape performance gates");
    weighted_baseline +=
        static_cast<double>(kShapes[index].calls_per_prompt) *
        measurement.baseline_milliseconds;
    weighted_candidate +=
        static_cast<double>(kShapes[index].calls_per_prompt) *
        measurement.candidate_milliseconds;
    std::cout << "PERF_NVFP4_M16_WMMA_VALIDATION: "
              << kShapes[index].label << " baseline_two_m8_ms="
              << measurement.baseline_milliseconds
              << " candidate_m16_ms=" << measurement.candidate_milliseconds
              << " speedup=" << speedup
              << " minimum_speedup=" << kMinimumPerShapeSpeedup
              << " gate=" << (shape_gate ? "PASS" : "FAIL") << '\n';
  }
  const double weighted_speedup = weighted_baseline / weighted_candidate;
  const bool aggregate_gate =
      all_shapes_pass && std::isfinite(weighted_speedup) &&
      weighted_speedup >= kMinimumWeightedSpeedup;
  std::cout << "PERF_NVFP4_M16_WMMA_AGGREGATE: "
            << "weighted_baseline_two_m8_ms=" << weighted_baseline
            << " weighted_candidate_m16_ms=" << weighted_candidate
            << " speedup=" << weighted_speedup
            << " required_speedup=" << kMinimumWeightedSpeedup
            << " profile_calls=256:128"
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "NVFP4 M16 WMMA clears the weighted production gate");
}

struct Fp8M16WmmaMeasurement {
  double baseline_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double candidate_milliseconds = std::numeric_limits<double>::quiet_NaN();
  bool deterministic = false;
  bool within_tolerance = false;
};

[[nodiscard]] Fp8M16WmmaMeasurement benchmark_fp8_m16_wmma_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 16U;
  constexpr std::size_t kHalfTokens = 8U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
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
  DeviceBuffer<std::uint16_t> replay_output;
  bool ready = test.cuda_ok(weights.allocate(host_weights.size()),
                            label + " allocate weights");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(replay_output.allocate(kTokens * rows),
                                label + " allocate replay output");
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
                       label + " initialize token-distinct activations");
  if (!ready) {
    return {};
  }

  const auto launch_baseline = [&]() noexcept -> int {
    int status = q3x::kernels::launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
        weights.get(), kWeightScale, activations.get(), kHalfTokens, rows,
        columns, baseline_output.get(), static_cast<void*>(stream));
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    status = q3x::kernels::launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
        weights.get(), kWeightScale,
        activations.get() + kHalfTokens * columns, kHalfTokens, rows,
        columns, baseline_output.get() + kHalfTokens * rows,
        static_cast<void*>(stream));
    return status;
  };
  const auto launch_candidate = [&]() noexcept -> int {
    if (q3x::kernels::use_sm87_fp8_m16_wmma_fixed_shape_test(rows,
                                                              columns)) {
      return q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
          weights.get(), kWeightScale, activations.get(), rows, columns,
          candidate_output.get(), static_cast<void*>(stream));
    }
    return q3x::kernels::
        launch_sm87_fp8_w8a16_small_m16_wmma_fixed_shape_test_cuda(
            weights.get(), kWeightScale, activations.get(), rows, columns,
            candidate_output.get(), static_cast<void*>(stream));
  };
  const auto launch_replay = [&]() noexcept -> int {
    if (q3x::kernels::use_sm87_fp8_m16_wmma_fixed_shape_test(rows,
                                                              columns)) {
      return q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
          weights.get(), kWeightScale, activations.get(), rows, columns,
          replay_output.get(), static_cast<void*>(stream));
    }
    return q3x::kernels::
        launch_sm87_fp8_w8a16_small_m16_wmma_fixed_shape_test_cuda(
            weights.get(), kWeightScale, activations.get(), rows, columns,
            replay_output.get(), static_cast<void*>(stream));
  };

  ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                       label + " correctness two-M8 baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate()),
                       label + " correctness WMMA candidate");
  ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_replay()),
                                label + " deterministic replay");
  std::vector<std::uint16_t> baseline(kTokens * rows);
  std::vector<std::uint16_t> candidate(kTokens * rows);
  std::vector<std::uint16_t> replay(kTokens * rows);
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
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           replay.data(), replay_output.get(),
                           replay.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy replay output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " correctness synchronize");
  if (!ready) {
    return {};
  }

  std::size_t bf16_mismatches = 0U;
  std::size_t deterministic_mismatches = 0U;
  float maximum_absolute_error = 0.0F;
  float maximum_relative_error = 0.0F;
  bool within_tolerance = true;
  for (std::size_t index = 0U; index < candidate.size(); ++index) {
    bf16_mismatches += candidate[index] != baseline[index] ? 1U : 0U;
    deterministic_mismatches += candidate[index] != replay[index] ? 1U : 0U;
    if (is_bf16_nan(baseline[index])) {
      within_tolerance =
          within_tolerance && is_bf16_nan(candidate[index]);
      continue;
    }
    const float baseline_value = decode_bf16(baseline[index]);
    const float candidate_value = decode_bf16(candidate[index]);
    const float absolute_error =
        std::fabs(candidate_value - baseline_value);
    const float relative_error =
        absolute_error / std::max(1.0e-6F, std::fabs(baseline_value));
    maximum_absolute_error =
        std::max(maximum_absolute_error, absolute_error);
    maximum_relative_error =
        std::max(maximum_relative_error, relative_error);
    const float tolerance =
        2.0e-4F * std::sqrt(static_cast<float>(columns)) +
        1.0e-2F * std::max(1.0F, std::fabs(baseline_value));
    within_tolerance = within_tolerance &&
                       std::isfinite(candidate_value) &&
                       std::isfinite(baseline_value) &&
                       absolute_error <= tolerance;
  }
  bool deterministic = deterministic_mismatches == 0U;
  std::cout << "FP8_M16_WMMA_DIFF: " << label
            << " candidate_vs_two_m8_bf16="
            << bf16_mismatches << '/' << candidate.size()
            << " deterministic_mismatches=" << deterministic_mismatches
            << '/' << candidate.size()
            << " max_abs=" << maximum_absolute_error
            << " max_rel=" << maximum_relative_error
            << " tolerance_gate=" << (within_tolerance ? "PASS" : "FAIL")
            << '\n';
  test.expect(deterministic,
              label + " reproduces every M16 BF16 output bitwise");
  test.expect(within_tolerance,
              label + " clears the existing FP8 numerical tolerance");
  compare_cuda_reference_outputs(test, candidate, baseline, columns,
                                 label + " candidate vs two M8");

  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              weights.get(), kWeightScale, activations.get(), rows, columns,
              replay_output.get(), static_cast<void*>(stream))),
      label + " public M16 selection launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           replay.data(), replay_output.get(),
                           replay.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy public M16 selection output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " public M16 selection synchronize");
  if (!ready) {
    return {};
  }
  const bool public_uses_wmma =
      q3x::kernels::use_sm87_fp8_m16_wmma_fixed_shape_test(rows, columns);
  const std::vector<std::uint16_t>& public_expected =
      public_uses_wmma ? candidate : baseline;
  std::size_t public_selection_mismatches = 0U;
  for (std::size_t index = 0U; index < replay.size(); ++index) {
    public_selection_mismatches +=
        replay[index] != public_expected[index] ? 1U : 0U;
  }
  const bool public_selection_matches = public_selection_mismatches == 0U;
  std::cout << "FP8_M16_PUBLIC_SELECTION_DIFF: " << label
            << " selected=" << (public_uses_wmma ? "wmma_m16" : "two_m8")
            << " mismatches=" << public_selection_mismatches << '/'
            << replay.size() << '\n';
  test.expect(public_selection_matches,
              label + " public M16 selects the required production path");
  deterministic = deterministic && public_selection_matches;
  within_tolerance = within_tolerance && public_selection_matches;

  // The /256 timing fixture often makes distinct FP32 reduction trees land
  // on identical BF16 values. Exercise the primary shape once more with
  // odd, non-power-of-two divisors and values spanning twelve exponents.
  if (rows == 10'240U && columns == 5'120U) {
    std::vector<std::uint16_t> nonpower_activations(kTokens * columns);
    for (std::size_t token = 0U; token < kTokens; ++token) {
      for (std::size_t column = 0U; column < columns; ++column) {
        const int centered =
            static_cast<int>((column * 37U + token * 19U + 11U) % 61U) - 30;
        const int odd_denominator =
            3 + 2 * static_cast<int>((column * 7U + token * 5U) % 15U);
        const int exponent =
            static_cast<int>((column * 11U + token * 3U) % 12U) - 6;
        const float value = std::ldexp(
            static_cast<float>(centered) /
                static_cast<float>(odd_denominator),
            exponent);
        nonpower_activations[token * columns + column] = encode_bf16(value);
      }
    }
    ready = test.cuda_ok(
        cudaMemcpyAsync(
            activations.get(), nonpower_activations.data(),
            nonpower_activations.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice, stream),
        label + " initialize nonpower cross-exponent activations");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_baseline()),
                         label + " nonpower two-M8 baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         label + " nonpower WMMA candidate");
    ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_replay()),
                                  label + " nonpower deterministic replay");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy nonpower baseline output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy nonpower candidate output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             replay.data(), replay_output.get(),
                             replay.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy nonpower replay output");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  label + " nonpower synchronize");
    if (!ready) {
      return {};
    }

    std::size_t nonpower_bf16_mismatches = 0U;
    std::size_t nonpower_deterministic_mismatches = 0U;
    float nonpower_maximum_absolute_error = 0.0F;
    float nonpower_maximum_relative_error = 0.0F;
    bool nonpower_within_tolerance = true;
    for (std::size_t index = 0U; index < candidate.size(); ++index) {
      nonpower_bf16_mismatches +=
          candidate[index] != baseline[index] ? 1U : 0U;
      nonpower_deterministic_mismatches +=
          candidate[index] != replay[index] ? 1U : 0U;
      if (is_bf16_nan(baseline[index])) {
        nonpower_within_tolerance =
            nonpower_within_tolerance && is_bf16_nan(candidate[index]);
        continue;
      }
      const float baseline_value = decode_bf16(baseline[index]);
      const float candidate_value = decode_bf16(candidate[index]);
      const float absolute_error =
          std::fabs(candidate_value - baseline_value);
      const float relative_error =
          absolute_error / std::max(1.0e-6F, std::fabs(baseline_value));
      nonpower_maximum_absolute_error =
          std::max(nonpower_maximum_absolute_error, absolute_error);
      nonpower_maximum_relative_error =
          std::max(nonpower_maximum_relative_error, relative_error);
      const float tolerance =
          2.0e-4F * std::sqrt(static_cast<float>(columns)) +
          1.0e-2F * std::max(1.0F, std::fabs(baseline_value));
      nonpower_within_tolerance =
          nonpower_within_tolerance && std::isfinite(candidate_value) &&
          std::isfinite(baseline_value) && absolute_error <= tolerance;
    }
    const bool nonpower_deterministic =
        nonpower_deterministic_mismatches == 0U;
    std::cout << "FP8_M16_WMMA_NONPOWER_DIFF: " << label
              << " candidate_vs_two_m8_bf16=" << nonpower_bf16_mismatches
              << '/' << candidate.size()
              << " deterministic_mismatches="
              << nonpower_deterministic_mismatches << '/' << candidate.size()
              << " max_abs=" << nonpower_maximum_absolute_error
              << " max_rel=" << nonpower_maximum_relative_error
              << " tolerance_gate="
              << (nonpower_within_tolerance ? "PASS" : "FAIL") << '\n';
    test.expect(nonpower_deterministic,
                label + " nonpower activation replay is deterministic");
    test.expect(nonpower_within_tolerance,
                label + " nonpower activations clear numerical tolerance");
    compare_cuda_reference_outputs(
        test, candidate, baseline, columns,
        label + " nonpower candidate vs two M8");
    deterministic = deterministic && nonpower_deterministic;
    within_tolerance = within_tolerance && nonpower_within_tolerance;

    ready = test.cuda_ok(
        cudaMemcpyAsync(
            activations.get(), host_activations.data(),
            host_activations.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice, stream),
        label + " restore timing activations");
  }

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

  double baseline_total = 0.0;
  double candidate_total = 0.0;
  bool finite = true;
  for (int round = 0; round < kMeasurementRounds; ++round) {
    const std::string suffix = " round=" + std::to_string(round + 1);
    const float baseline_first = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        label + " baseline pass 1" + suffix);
    const float candidate_first = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        label + " candidate pass 1" + suffix);
    const float candidate_second = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        label + " candidate pass 2" + suffix);
    const float baseline_second = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        label + " baseline pass 2" + suffix);
    const bool round_finite =
        std::isfinite(baseline_first) && std::isfinite(candidate_first) &&
        std::isfinite(candidate_second) && std::isfinite(baseline_second);
    finite = finite && round_finite;
    if (round_finite) {
      baseline_total += baseline_first + baseline_second;
      candidate_total += candidate_first + candidate_second;
    }
    std::cout << "PERF_FP8_M16_WMMA_ROUND: " << label
              << " round=" << round + 1
              << " baseline_pass1_ms=" << baseline_first
              << " candidate_pass1_ms=" << candidate_first
              << " candidate_pass2_ms=" << candidate_second
              << " baseline_pass2_ms=" << baseline_second << '\n';
  }
  constexpr double kTimedPasses =
      2.0 * static_cast<double>(kMeasurementRounds);
  const double baseline_average = baseline_total / kTimedPasses;
  const double candidate_average = candidate_total / kTimedPasses;
  const double speedup = baseline_average / candidate_average;
  std::cout << "PERF_FP8_M16_WMMA: " << label
            << " baseline_two_m8_ms="
            << baseline_average << " candidate_m16_ms=" << candidate_average
            << " speedup=" << speedup << '\n';

  if (columns == 5'120U && rows != 1'024U) {
    const auto launch_ldm64 = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_fp8_w8a16_small_m16_wmma_shared_ldm_test_cuda(
              weights.get(), kWeightScale, activations.get(), rows, columns,
              candidate_output.get(), 64U, static_cast<void*>(stream));
    };
    const auto launch_ldm72 = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_fp8_w8a16_small_m16_wmma_shared_ldm_test_cuda(
              weights.get(), kWeightScale, activations.get(), rows, columns,
              replay_output.get(), 72U, static_cast<void*>(stream));
    };

    ready = test.cuda_ok(static_cast<cudaError_t>(launch_ldm64()),
                         label + " ldm64 correctness launch");
    ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_ldm72()),
                                  label + " ldm72 correctness launch");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy ldm64 output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             replay.data(), replay_output.get(),
                             replay.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy ldm72 output");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  label + " ldm correctness synchronize");
    if (!ready) {
      return {};
    }
    std::size_t ldm_mismatches = 0U;
    for (std::size_t index = 0U; index < candidate.size(); ++index) {
      ldm_mismatches += candidate[index] != replay[index] ? 1U : 0U;
    }
    const bool ldm_bitwise_equal = ldm_mismatches == 0U;
    std::cout << "FP8_M16_WMMA_LDM_DIFF: " << label
              << " ldm64_vs_ldm72_bf16=" << ldm_mismatches << '/'
              << candidate.size() << '\n';
    test.expect(ldm_bitwise_equal,
                label + " ldm64 and ldm72 are bitwise identical");
    deterministic = deterministic && ldm_bitwise_equal;
    within_tolerance = within_tolerance && ldm_bitwise_equal;

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_ldm64()),
                           label + " ldm64 warmup");
      ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_ldm72()),
                                    label + " ldm72 warmup");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  label + " ldm warmup synchronize");
    if (!ready) {
      return {};
    }

    double ldm64_total = 0.0;
    double ldm72_total = 0.0;
    bool ldm_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string suffix = " round=" + std::to_string(round + 1);
      const float ldm64_first = measure_small_m_tile(
          test, stream, launch_ldm64, kMeasuredIterations,
          label + " ldm64 pass 1" + suffix);
      const float ldm72_first = measure_small_m_tile(
          test, stream, launch_ldm72, kMeasuredIterations,
          label + " ldm72 pass 1" + suffix);
      const float ldm72_second = measure_small_m_tile(
          test, stream, launch_ldm72, kMeasuredIterations,
          label + " ldm72 pass 2" + suffix);
      const float ldm64_second = measure_small_m_tile(
          test, stream, launch_ldm64, kMeasuredIterations,
          label + " ldm64 pass 2" + suffix);
      const bool round_finite =
          std::isfinite(ldm64_first) && std::isfinite(ldm72_first) &&
          std::isfinite(ldm72_second) && std::isfinite(ldm64_second);
      ldm_finite = ldm_finite && round_finite;
      if (round_finite) {
        ldm64_total += ldm64_first + ldm64_second;
        ldm72_total += ldm72_first + ldm72_second;
      }
      std::cout << "PERF_FP8_M16_WMMA_LDM_ROUND: " << label
                << " round=" << round + 1
                << " ldm64_pass1_ms=" << ldm64_first
                << " ldm72_pass1_ms=" << ldm72_first
                << " ldm72_pass2_ms=" << ldm72_second
                << " ldm64_pass2_ms=" << ldm64_second << '\n';
    }
    const double ldm64_average = ldm64_total / kTimedPasses;
    const double ldm72_average = ldm72_total / kTimedPasses;
    const double ldm72_speedup = ldm64_average / ldm72_average;
    finite = finite && ldm_finite && std::isfinite(ldm72_speedup);
    std::cout << "PERF_FP8_M16_WMMA_LDM: " << label
              << " ldm64_ms=" << ldm64_average
              << " ldm72_ms=" << ldm72_average
              << " ldm72_speedup=" << ldm72_speedup << '\n';
    test.expect(ldm_finite && std::isfinite(ldm72_speedup),
                label + " ldm64/ldm72 mirrored timing is finite");
  }

  // Performance is already frozen above. Reuse the primary-shape buffers to
  // cover every E4M3FN code in every byte position, including both reserved
  // NaN encodings, without contaminating the timing fixture.
  if (rows == 10'240U && columns == 5'120U) {
    std::fill(host_weights.begin(), host_weights.end(), 0U);
    constexpr std::size_t kCodes = 256U;
    constexpr std::size_t kBytePositions = 4U;
    for (std::size_t code = 0U; code < kCodes; ++code) {
      for (std::size_t byte_position = 0U;
           byte_position < kBytePositions; ++byte_position) {
        const std::size_t row = code * kBytePositions + byte_position;
        const std::size_t packed_word =
            (code * 29U + byte_position * 31U) % (columns / 4U);
        const std::size_t column = packed_word * 4U + byte_position;
        host_weights[row * columns + column] =
            static_cast<std::uint8_t>(code);
      }
    }
    std::vector<std::uint16_t> exhaustive_activations(kTokens * columns);
    for (std::size_t token = 0U; token < kTokens; ++token) {
      for (std::size_t column = 0U; column < columns; ++column) {
        const int magnitude =
            1 + static_cast<int>((column * 13U + token * 7U) % 29U);
        const int numerator =
            ((column + token) & 1U) == 0U ? magnitude : -magnitude;
        const int odd_denominator =
            3 + 2 * static_cast<int>((column * 5U + token * 3U) % 11U);
        const int exponent =
            static_cast<int>((column * 3U + token * 5U) % 8U) - 4;
        exhaustive_activations[token * columns + column] = encode_bf16(
            std::ldexp(static_cast<float>(numerator) /
                           static_cast<float>(odd_denominator),
                       exponent));
      }
    }
    ready = test.cuda_ok(
        cudaMemcpyAsync(weights.get(), host_weights.data(),
                        host_weights.size(), cudaMemcpyHostToDevice, stream),
        label + " initialize exhaustive FP8 byte-position weights");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             activations.get(), exhaustive_activations.data(),
                             exhaustive_activations.size() *
                                 sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice, stream),
                         label + " initialize exhaustive activations");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_baseline()),
                         label + " exhaustive two-M8 baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         label + " exhaustive WMMA candidate");
    ready = ready && test.cuda_ok(static_cast<cudaError_t>(launch_replay()),
                                  label + " exhaustive deterministic replay");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy exhaustive baseline output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy exhaustive candidate output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             replay.data(), replay_output.get(),
                             replay.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         label + " copy exhaustive replay output");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  label + " exhaustive synchronize");
    if (!ready) {
      return {};
    }

    std::size_t exhaustive_bf16_mismatches = 0U;
    std::size_t exhaustive_deterministic_mismatches = 0U;
    std::size_t reference_nan_count = 0U;
    std::size_t nan_class_mismatches = 0U;
    float exhaustive_maximum_absolute_error = 0.0F;
    float exhaustive_maximum_relative_error = 0.0F;
    bool exhaustive_within_tolerance = true;
    for (std::size_t index = 0U; index < candidate.size(); ++index) {
      exhaustive_bf16_mismatches +=
          candidate[index] != baseline[index] ? 1U : 0U;
      exhaustive_deterministic_mismatches +=
          candidate[index] != replay[index] ? 1U : 0U;
      if (is_bf16_nan(baseline[index])) {
        ++reference_nan_count;
        const bool candidate_is_nan = is_bf16_nan(candidate[index]);
        nan_class_mismatches += candidate_is_nan ? 0U : 1U;
        exhaustive_within_tolerance =
            exhaustive_within_tolerance && candidate_is_nan;
        continue;
      }
      const float baseline_value = decode_bf16(baseline[index]);
      const float candidate_value = decode_bf16(candidate[index]);
      const float absolute_error =
          std::fabs(candidate_value - baseline_value);
      const float relative_error =
          absolute_error / std::max(1.0e-6F, std::fabs(baseline_value));
      exhaustive_maximum_absolute_error =
          std::max(exhaustive_maximum_absolute_error, absolute_error);
      exhaustive_maximum_relative_error =
          std::max(exhaustive_maximum_relative_error, relative_error);
      const float tolerance =
          2.0e-4F * std::sqrt(static_cast<float>(columns)) +
          1.0e-2F * std::max(1.0F, std::fabs(baseline_value));
      exhaustive_within_tolerance =
          exhaustive_within_tolerance && std::isfinite(candidate_value) &&
          std::isfinite(baseline_value) && absolute_error <= tolerance;
    }
    constexpr std::size_t kExpectedNanOutputs =
        2U * kBytePositions * kTokens;
    const bool exhaustive_deterministic =
        exhaustive_deterministic_mismatches == 0U;
    const bool nan_coverage = reference_nan_count == kExpectedNanOutputs &&
                              nan_class_mismatches == 0U;
    std::cout << "FP8_M16_WMMA_EXHAUSTIVE_DIFF: " << label
              << " candidate_vs_two_m8_bf16="
              << exhaustive_bf16_mismatches << '/' << candidate.size()
              << " deterministic_mismatches="
              << exhaustive_deterministic_mismatches << '/'
              << candidate.size() << " reference_nan_count="
              << reference_nan_count
              << " nan_class_mismatches=" << nan_class_mismatches
              << " max_abs=" << exhaustive_maximum_absolute_error
              << " max_rel=" << exhaustive_maximum_relative_error
              << " tolerance_gate="
              << (exhaustive_within_tolerance ? "PASS" : "FAIL") << '\n';
    test.expect(exhaustive_deterministic,
                label + " exhaustive replay is deterministic");
    test.expect(nan_coverage,
                label + " covers both NaN codes in all byte positions");
    test.expect(exhaustive_within_tolerance,
                label + " exhaustive codes clear numerical tolerance");
    compare_cuda_reference_outputs(
        test, candidate, baseline, columns,
        label + " exhaustive candidate vs two M8");
    deterministic = deterministic && exhaustive_deterministic;
    within_tolerance = within_tolerance && exhaustive_within_tolerance &&
                       nan_coverage;
  }

  return {finite ? baseline_average
                 : std::numeric_limits<double>::quiet_NaN(),
          finite ? candidate_average
                 : std::numeric_limits<double>::quiet_NaN(),
          deterministic, within_tolerance};
}

void run_optional_fp8_m16_wmma_performance(TestContext& test,
                                            cudaStream_t stream) {
  if (!fp8_m16_wmma_performance_enabled()) {
    std::cout << "SKIP: FP8 M16 WMMA performance segment; set "
                 "Q3X_RUN_SM87_FP8_M16_WMMA_PERF=1 to enable\n";
    return;
  }

  struct Shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t calls_per_prompt;
    double minimum_speedup;
    bool allow_fallback;
    const char* label;
  };
  constexpr std::array<Shape, 5U> kShapes{{
      {10'240U, 5'120U, 96U, 1.15, false,
       "FP8 M16 WMMA 10240x5120"},
      {5'120U, 6'144U, 128U, 1.05, false,
       "FP8 M16 WMMA 5120x6144"},
      {6'144U, 5'120U, 96U, 1.05, false,
       "FP8 M16 WMMA 6144x5120"},
      {12'288U, 5'120U, 32U, 1.05, false,
       "FP8 M16 WMMA 12288x5120"},
      {1'024U, 5'120U, 64U, 1.05, true,
       "FP8 M16 WMMA 1024x5120"},
  }};
  std::array<Fp8M16WmmaMeasurement, kShapes.size()> measurements{};
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    measurements[index] = benchmark_fp8_m16_wmma_shape(
        test, stream, kShapes[index].rows, kShapes[index].columns,
        kShapes[index].label);
  }

  double weighted_baseline = 0.0;
  double weighted_raw_candidate = 0.0;
  double weighted_selected_candidate = 0.0;
  bool required_shapes_pass = true;
  for (std::size_t index = 0U; index < kShapes.size(); ++index) {
    const Fp8M16WmmaMeasurement& measurement = measurements[index];
    const double speedup = measurement.baseline_milliseconds /
                           measurement.candidate_milliseconds;
    const bool finite =
        std::isfinite(measurement.baseline_milliseconds) &&
        std::isfinite(measurement.candidate_milliseconds) &&
        std::isfinite(speedup);
    const bool raw_performance_gate =
        finite && speedup >= kShapes[index].minimum_speedup;
    const bool selected_fallback =
        kShapes[index].allow_fallback && finite && !raw_performance_gate;
    const bool shape_gate =
        measurement.deterministic && measurement.within_tolerance &&
        (raw_performance_gate || selected_fallback);
    required_shapes_pass = required_shapes_pass && shape_gate;
    test.expect(shape_gate,
                std::string(kShapes[index].label) +
                    " clears correctness and selected performance policy");

    weighted_baseline +=
        static_cast<double>(kShapes[index].calls_per_prompt) *
        measurement.baseline_milliseconds;
    weighted_raw_candidate +=
        static_cast<double>(kShapes[index].calls_per_prompt) *
        measurement.candidate_milliseconds;
    weighted_selected_candidate +=
        static_cast<double>(kShapes[index].calls_per_prompt) *
        (selected_fallback ? measurement.baseline_milliseconds
                           : measurement.candidate_milliseconds);
    std::cout << "PERF_FP8_M16_WMMA_VALIDATION: " << kShapes[index].label
              << " baseline_two_m8_ms="
              << measurement.baseline_milliseconds
              << " candidate_m16_ms=" << measurement.candidate_milliseconds
              << " speedup=" << speedup
              << " minimum_speedup=" << kShapes[index].minimum_speedup
              << " selected="
              << (selected_fallback ? "two_m8_fallback" : "wmma_m16")
              << " gate=" << (shape_gate ? "PASS" : "FAIL") << '\n';
  }
  const double raw_weighted_speedup =
      weighted_baseline / weighted_raw_candidate;
  const double selected_weighted_speedup =
      weighted_baseline / weighted_selected_candidate;
  constexpr double kMinimumSelectedWeightedSpeedup = 1.05;
  const bool aggregate_gate =
      required_shapes_pass && std::isfinite(selected_weighted_speedup) &&
      selected_weighted_speedup >= kMinimumSelectedWeightedSpeedup;
  std::cout << "PERF_FP8_M16_WMMA_AGGREGATE: weighted_baseline_two_m8_ms="
            << weighted_baseline
            << " weighted_raw_candidate_ms=" << weighted_raw_candidate
            << " raw_speedup=" << raw_weighted_speedup
            << " weighted_selected_candidate_ms="
            << weighted_selected_candidate
            << " selected_speedup=" << selected_weighted_speedup
            << " required_selected_speedup="
            << kMinimumSelectedWeightedSpeedup
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL")
            << '\n';
  test.expect(aggregate_gate,
              "FP8 M16 WMMA selected shapes clear weighted speedup gate");
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

enum class NvFp4M1ScaleDistribution {
  kCheckpointLike,
  kSameBankStress,
};

constexpr std::array<std::uint8_t, 32U> kNvFp4M1CheckpointLikeScaleCodes{{
    0x4eU, 0x50U, 0x52U, 0x54U, 0x55U, 0x56U, 0x57U, 0x58U,
    0x58U, 0x58U, 0x59U, 0x59U, 0x59U, 0x5aU, 0x5aU, 0x5bU,
    0x5bU, 0x5cU, 0x5cU, 0x5dU, 0x5dU, 0x5eU, 0x5fU, 0x60U,
    0x60U, 0x61U, 0x62U, 0x63U, 0x64U, 0x65U, 0x66U, 0x67U,
}};

constexpr std::array<std::uint8_t, 3U> kNvFp4M1SameBankScaleCodes{{
    0x20U,
    0x40U,
    0x60U,
}};

// The exact-shape kernel advances one warp's row quad by 64 blocks * 8 warps
// * 4 rows = 2048 rows. A prime fixture period prevents an omitted or stale
// rolling-offset update from aliasing the same packed/scaled input rows.
constexpr std::size_t kNvFp4M1ExactShapeFixturePeriod = 251U;
constexpr std::size_t kNvFp4M1ExactShapeRowStride = 2'048U;
constexpr std::size_t kNvFp4M1DownDualFixturePeriod = 257U;

[[nodiscard]] const char* nvfp4_m1_scale_distribution_name(
    const NvFp4M1ScaleDistribution distribution) noexcept {
  return distribution == NvFp4M1ScaleDistribution::kCheckpointLike
             ? "checkpoint_like"
             : "same_bank_stress";
}

void fill_nvfp4_m1_scale_distribution(
    std::vector<std::uint8_t>& scales, const std::size_t scale_columns,
    const NvFp4M1ScaleDistribution distribution) {
  for (std::size_t index = 0U; index < scales.size(); ++index) {
    const std::size_t row = index / scale_columns;
    const std::size_t scale_column = index - row * scale_columns;
    if (distribution == NvFp4M1ScaleDistribution::kCheckpointLike) {
      scales[index] = kNvFp4M1CheckpointLikeScaleCodes[
          (scale_column * 5U + row * 11U + (scale_column >> 3U)) %
          kNvFp4M1CheckpointLikeScaleCodes.size()];
    } else {
      // All three codes share their low five bits and therefore map to the
      // same shared-memory bank while retaining different decoded values.
      scales[index] =
          kNvFp4M1SameBankScaleCodes[
              (scale_column + row * 3U) %
              kNvFp4M1SameBankScaleCodes.size()];
    }
  }
}

void fill_nvfp4_m1_exact_shape_scale_distribution(
    std::vector<std::uint8_t>& scales, const std::size_t scale_columns,
    const NvFp4M1ScaleDistribution distribution) {
  for (std::size_t index = 0U; index < scales.size(); ++index) {
    const std::size_t row = index / scale_columns;
    const std::size_t scale_column = index - row * scale_columns;
    const std::size_t row_phase =
        row % kNvFp4M1ExactShapeFixturePeriod;
    if (distribution == NvFp4M1ScaleDistribution::kCheckpointLike) {
      scales[index] = kNvFp4M1CheckpointLikeScaleCodes[
          (scale_column * 5U + row_phase * 11U + (scale_column >> 3U)) %
          kNvFp4M1CheckpointLikeScaleCodes.size()];
    } else {
      // Preserve the same-bank stress while varying the decoded value across
      // rows, including rows separated by the production 2048-row stride.
      scales[index] = kNvFp4M1SameBankScaleCodes[
          (scale_column + row_phase) % kNvFp4M1SameBankScaleCodes.size()];
    }
  }
}

void fill_nvfp4_m1_down_dual_scale_distribution(
    std::vector<std::uint8_t>& scales, const std::size_t scale_columns,
    const NvFp4M1ScaleDistribution distribution) {
  for (std::size_t index = 0U; index < scales.size(); ++index) {
    const std::size_t row = index / scale_columns;
    const std::size_t scale_column = index - row * scale_columns;
    const std::size_t row_phase =
        row % kNvFp4M1DownDualFixturePeriod;
    if (distribution == NvFp4M1ScaleDistribution::kCheckpointLike) {
      scales[index] = kNvFp4M1CheckpointLikeScaleCodes[
          (scale_column * 5U + row_phase * 11U +
           (scale_column >> 3U)) %
          kNvFp4M1CheckpointLikeScaleCodes.size()];
    } else {
      scales[index] = kNvFp4M1SameBankScaleCodes[
          (scale_column + row_phase) %
          kNvFp4M1SameBankScaleCodes.size()];
    }
  }
}

constexpr std::array<NvFp4M1ScaleDistribution, 2U>
    kNvFp4M16K128ScaleDistributions{{
        NvFp4M1ScaleDistribution::kCheckpointLike,
        NvFp4M1ScaleDistribution::kSameBankStress,
    }};

struct NvFp4M16K128DistributionMeasurement {
  float current_k64_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_k128_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
  bool output_finite = false;
  bool production_bitwise_equal = false;
  bool production_output_finite = false;
};

struct NvFp4M16K128ShapeMeasurement {
  std::array<NvFp4M16K128DistributionMeasurement,
             kNvFp4M16K128ScaleDistributions.size()>
      distributions{};
};

struct NvFp4M16K128Shape {
  std::size_t rows;
  std::size_t columns;
  std::size_t profile_calls;
  const char* label;
};

[[nodiscard]] NvFp4M16K128ShapeMeasurement
benchmark_nvfp4_m16_k128_shape(TestContext& test, cudaStream_t stream,
                                const NvFp4M16K128Shape& shape) {
  constexpr std::size_t kTokens = 16U;
  const std::size_t kRows = shape.rows;
  const std::size_t kColumns = shape.columns;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr float kMinimumCellSpeedup = 1.02F;
  constexpr std::uint8_t kCurrentOutputSentinel = 0xa5U;
  constexpr std::uint8_t kCandidateOutputSentinel = 0x5aU;
  constexpr std::uint8_t kProductionOutputSentinel = 0x3cU;
  const std::string label = shape.label;
  const std::size_t packed_columns = kColumns / 2U;
  const std::size_t scale_columns = kColumns / 16U;

  std::vector<std::uint8_t> host_packed(kRows * packed_columns);
  std::array<bool, 16U> low_nibble_covered{};
  std::array<bool, 16U> high_nibble_covered{};
  for (std::size_t row = 0U; row < kRows; ++row) {
    for (std::size_t packed_column = 0U; packed_column < packed_columns;
         ++packed_column) {
      const std::uint8_t low = static_cast<std::uint8_t>(
          (packed_column + row * 3U + (packed_column >> 3U)) & 0x0fU);
      const std::uint8_t high = static_cast<std::uint8_t>(
          (packed_column * 5U + row * 7U +
           (packed_column >> 3U) * 3U + 1U) &
          0x0fU);
      host_packed[row * packed_columns + packed_column] =
          static_cast<std::uint8_t>(low | (high << 4U));
      low_nibble_covered[low] = true;
      high_nibble_covered[high] = true;
    }
  }
  test.expect(std::all_of(low_nibble_covered.begin(),
                          low_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers every E2M1 low-nibble code");
  test.expect(std::all_of(high_nibble_covered.begin(),
                          high_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers every E2M1 high-nibble code");

  std::vector<std::uint8_t> host_scales(kRows * scale_columns);
  std::vector<std::uint16_t> host_activations(kTokens * kColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const int centered =
          static_cast<int>((column * 17U + token * 19U + 5U) % 127U) - 63;
      host_activations[token * kColumns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
  }

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> current_k64_output;
  DeviceBuffer<std::uint16_t> candidate_k128_output;
  bool ready = test.cuda_ok(packed.allocate(host_packed.size()),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(host_scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready &&
          test.cuda_ok(current_k64_output.allocate(kTokens * kRows),
                       label + " allocate current K64 output");
  ready = ready &&
          test.cuda_ok(candidate_k128_output.allocate(kTokens * kRows),
                       label + " allocate candidate K128 output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(packed.get(), host_packed.data(),
                                       host_packed.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize mixed all-nibble weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return {};
  }

  std::vector<std::uint16_t> current_k64(kTokens * kRows);
  std::vector<std::uint16_t> candidate_k128(kTokens * kRows);
  NvFp4M16K128ShapeMeasurement shape_measurement{};
  for (std::size_t distribution_index = 0U;
       distribution_index < kNvFp4M16K128ScaleDistributions.size();
       ++distribution_index) {
    const NvFp4M1ScaleDistribution distribution =
        kNvFp4M16K128ScaleDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_scale_distribution(host_scales, scale_columns,
                                     distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize scales");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(
                             current_k64_output.get(), kCurrentOutputSentinel,
                             kTokens * kRows * sizeof(std::uint16_t), stream),
                         distribution_label +
                             " initialize current K64 output sentinel");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(
                             candidate_k128_output.get(),
                             kCandidateOutputSentinel,
                             kTokens * kRows * sizeof(std::uint16_t), stream),
                         distribution_label +
                             " initialize candidate K128 output sentinel");
    if (!ready) {
      return shape_measurement;
    }

    const auto launch_current_k64 = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m16_wmma_fixed_shape_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activations.get(),
              kRows, kColumns, current_k64_output.get(),
              static_cast<void*>(stream));
    };
    const auto launch_candidate_k128 = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m16_wmma_k128_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activations.get(),
              kRows, kColumns, candidate_k128_output.get(),
              static_cast<void*>(stream));
    };
    const auto launch_production = [&]() noexcept -> int {
      return q3x::kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
          packed.get(), scales.get(), kWeightScale2, activations.get(), kRows,
          kColumns, candidate_k128_output.get(), static_cast<void*>(stream));
    };

    ready = test.cuda_ok(static_cast<cudaError_t>(launch_current_k64()),
                         distribution_label + " correctness current K64");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate_k128()),
                         distribution_label + " correctness K128/LD136");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             current_k64.data(), current_k64_output.get(),
                             current_k64.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy current K64");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate_k128.data(),
                             candidate_k128_output.get(),
                             candidate_k128.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy candidate K128");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " correctness synchronize");
    if (!ready) {
      return shape_measurement;
    }

    std::size_t mismatches = 0U;
    bool output_finite = true;
    for (std::size_t index = 0U; index < current_k64.size(); ++index) {
      mismatches += current_k64[index] != candidate_k128[index] ? 1U : 0U;
      output_finite = output_finite &&
                      std::isfinite(decode_bf16(current_k64[index])) &&
                      std::isfinite(decode_bf16(candidate_k128[index]));
    }
    NvFp4M16K128DistributionMeasurement& measurement =
        shape_measurement.distributions[distribution_index];
    measurement.bitwise_equal = mismatches == 0U;
    measurement.output_finite = output_finite;
    std::cout << "NVFP4_M16_K128_DIFF: " << label << " distribution="
              << nvfp4_m1_scale_distribution_name(distribution)
              << " current_k64_vs_k128_ld136_mismatches=" << mismatches << '/'
              << current_k64.size()
              << " output_finite=" << (output_finite ? "true" : "false")
              << '\n';
    test.expect(measurement.bitwise_equal,
                distribution_label +
                    " K128/LD136 is bitwise equal to current K64");
    test.expect(output_finite,
                distribution_label + " K64 and K128 outputs remain finite");

    ready = test.cuda_ok(
        cudaMemsetAsync(candidate_k128_output.get(),
                        kProductionOutputSentinel,
                        kTokens * kRows * sizeof(std::uint16_t), stream),
        distribution_label + " poison production output");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_production()),
                         distribution_label + " launch production M16");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate_k128.data(),
                             candidate_k128_output.get(),
                             candidate_k128.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy production M16 output");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label +
                             " production correctness synchronize");
    if (!ready) {
      return shape_measurement;
    }

    std::size_t production_mismatches = 0U;
    bool production_output_finite = true;
    for (std::size_t index = 0U; index < current_k64.size(); ++index) {
      production_mismatches +=
          current_k64[index] != candidate_k128[index] ? 1U : 0U;
      production_output_finite =
          production_output_finite &&
          std::isfinite(decode_bf16(current_k64[index])) &&
          std::isfinite(decode_bf16(candidate_k128[index]));
    }
    measurement.production_bitwise_equal = production_mismatches == 0U;
    measurement.production_output_finite = production_output_finite;
    std::cout << "NVFP4_M16_K128_PRODUCTION_DIFF: " << label
              << " distribution="
              << nvfp4_m1_scale_distribution_name(distribution)
              << " current_k64_vs_public_m16_mismatches="
              << production_mismatches << '/' << current_k64.size()
              << " output_finite="
              << (production_output_finite ? "true" : "false") << '\n';
    test.expect(measurement.production_bitwise_equal,
                distribution_label +
                    " public production M16 is bitwise equal to current K64");
    test.expect(production_output_finite,
                distribution_label +
                    " public production M16 output remains finite");

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_current_k64()),
                           distribution_label + " current K64 warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate_k128()),
                           distribution_label + " K128/LD136 warmup");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  distribution_label + " warmup synchronize");
    if (!ready) {
      return shape_measurement;
    }

    double current_k64_total = 0.0;
    double candidate_k128_total = 0.0;
    bool finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string suffix =
          distribution_label + " round=" + std::to_string(round + 1);
      const float current_first = measure_small_m_tile(
          test, stream, launch_current_k64, kMeasuredIterations,
          suffix + " current K64 pass 1");
      const float candidate_first = measure_small_m_tile(
          test, stream, launch_candidate_k128, kMeasuredIterations,
          suffix + " candidate K128 pass 1");
      const float candidate_second = measure_small_m_tile(
          test, stream, launch_candidate_k128, kMeasuredIterations,
          suffix + " candidate K128 pass 2");
      const float current_second = measure_small_m_tile(
          test, stream, launch_current_k64, kMeasuredIterations,
          suffix + " current K64 pass 2");
      const bool round_finite =
          std::isfinite(current_first) && std::isfinite(candidate_first) &&
          std::isfinite(candidate_second) && std::isfinite(current_second);
      finite = finite && round_finite;
      if (round_finite) {
        current_k64_total += current_first + current_second;
        candidate_k128_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_NVFP4_M16_K128_ROUND: " << label
                << " distribution="
                << nvfp4_m1_scale_distribution_name(distribution)
                << " round=" << round + 1
                << " current_k64_pass1_ms=" << current_first
                << " candidate_k128_pass1_ms=" << candidate_first
                << " candidate_k128_pass2_ms=" << candidate_second
                << " current_k64_pass2_ms=" << current_second << '\n';
    }
    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    measurement.current_k64_milliseconds =
        finite ? static_cast<float>(current_k64_total / kTimedPasses)
               : std::numeric_limits<float>::quiet_NaN();
    measurement.candidate_k128_milliseconds =
        finite ? static_cast<float>(candidate_k128_total / kTimedPasses)
               : std::numeric_limits<float>::quiet_NaN();
    const float speedup = measurement.current_k64_milliseconds /
                          measurement.candidate_k128_milliseconds;
    const bool valid =
        finite && std::isfinite(measurement.current_k64_milliseconds) &&
        std::isfinite(measurement.candidate_k128_milliseconds) &&
        std::isfinite(speedup) &&
        measurement.current_k64_milliseconds > 0.0F &&
        measurement.candidate_k128_milliseconds > 0.0F;
    const bool gate = measurement.bitwise_equal && measurement.output_finite &&
                      measurement.production_bitwise_equal &&
                      measurement.production_output_finite && valid &&
                      speedup >= kMinimumCellSpeedup;
    test.expect(gate, distribution_label + " clears K128/LD136 gate");
    std::cout << "PERF_NVFP4_M16_K128_VALIDATION: " << label
              << " distribution="
              << nvfp4_m1_scale_distribution_name(distribution)
              << " current_k64_ms=" << measurement.current_k64_milliseconds
              << " candidate_k128_ms="
              << measurement.candidate_k128_milliseconds
              << " speedup=" << speedup
              << " required_speedup=" << kMinimumCellSpeedup
              << " bitwise="
              << (measurement.bitwise_equal ? "true" : "false")
              << " output_finite="
              << (measurement.output_finite ? "true" : "false")
              << " production_bitwise="
              << (measurement.production_bitwise_equal ? "true" : "false")
              << " production_output_finite="
              << (measurement.production_output_finite ? "true" : "false")
              << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  }
  return shape_measurement;
}

void run_optional_nvfp4_m16_k128_performance(TestContext& test,
                                               cudaStream_t stream) {
  if (!nvfp4_m16_k128_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M16 K128/LD136 performance segment; set "
                 "Q3X_RUN_SM87_NVFP4_M16_K128_PERF=1 to enable\n";
    return;
  }

  constexpr float kMinimumCellSpeedup = 1.02F;
  constexpr double kMinimumWeightedSpeedup = 1.05;
  constexpr std::array<NvFp4M16K128Shape, 2U> kShapes{{
      {17'408U, 5'120U, 128U,
       "NVFP4 M16 gate/up 17408x5120 K128/LD136"},
      {5'120U, 17'408U, 64U,
       "NVFP4 M16 down 5120x17408 K128/LD136"},
  }};

  std::array<NvFp4M16K128ShapeMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] =
        benchmark_nvfp4_m16_k128_shape(test, stream, kShapes[shape_index]);
  }

  double weighted_current_k64 = 0.0;
  double weighted_candidate_k128 = 0.0;
  bool all_cells_pass = true;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    double shape_current_k64 = 0.0;
    double shape_candidate_k128 = 0.0;
    bool shape_cells_pass = true;
    for (std::size_t distribution_index = 0U;
         distribution_index < kNvFp4M16K128ScaleDistributions.size();
         ++distribution_index) {
      const NvFp4M16K128DistributionMeasurement& cell =
          measurements[shape_index].distributions[distribution_index];
      const double speedup =
          static_cast<double>(cell.current_k64_milliseconds) /
          static_cast<double>(cell.candidate_k128_milliseconds);
      const bool finite =
          std::isfinite(cell.current_k64_milliseconds) &&
          std::isfinite(cell.candidate_k128_milliseconds) &&
          cell.current_k64_milliseconds > 0.0F &&
          cell.candidate_k128_milliseconds > 0.0F &&
          std::isfinite(speedup);
      const bool cell_gate =
          cell.bitwise_equal && cell.output_finite &&
          cell.production_bitwise_equal && cell.production_output_finite &&
          finite && speedup >= kMinimumCellSpeedup;
      shape_cells_pass = shape_cells_pass && cell_gate;
      all_cells_pass = all_cells_pass && cell_gate;
      shape_current_k64 += cell.current_k64_milliseconds;
      shape_candidate_k128 += cell.candidate_k128_milliseconds;
      weighted_current_k64 +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          cell.current_k64_milliseconds;
      weighted_candidate_k128 +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          cell.candidate_k128_milliseconds;
    }
    const double shape_speedup = shape_current_k64 / shape_candidate_k128;
    std::cout << "PERF_NVFP4_M16_K128_SHAPE_AGGREGATE: "
              << kShapes[shape_index].label
              << " current_k64_ms=" << shape_current_k64
              << " candidate_k128_ms=" << shape_candidate_k128
              << " speedup=" << shape_speedup
              << " profile_calls=" << kShapes[shape_index].profile_calls
              << " all_cells=" << (shape_cells_pass ? "PASS" : "FAIL")
              << '\n';
  }

  const double weighted_speedup =
      weighted_current_k64 / weighted_candidate_k128;
  const bool aggregate_gate =
      all_cells_pass && std::isfinite(weighted_speedup) &&
      weighted_current_k64 > 0.0 && weighted_candidate_k128 > 0.0 &&
      weighted_speedup >= kMinimumWeightedSpeedup;
  std::cout << "PERF_NVFP4_M16_K128_AGGREGATE: weighted_current_k64_ms="
            << weighted_current_k64
            << " weighted_candidate_k128_ms=" << weighted_candidate_k128
            << " speedup=" << weighted_speedup
            << " required_speedup=" << kMinimumWeightedSpeedup
            << " profile_calls=128:64"
            << " distributions=checkpoint_like:same_bank_stress"
            << " all_cells=" << (all_cells_pass ? "PASS" : "FAIL")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "NVFP4 M16 gate/up and down K128/LD136 clear the weighted "
              "production gate");
}

constexpr std::array<std::size_t, 6U> kNvFp4GridCaps{{
    96U, 192U, 384U, 768U, 1'024U, 2'048U,
}};

constexpr std::array<NvFp4M1ScaleDistribution, 2U>
    kNvFp4GridCapScaleDistributions{{
        NvFp4M1ScaleDistribution::kCheckpointLike,
        NvFp4M1ScaleDistribution::kSameBankStress,
    }};

struct NvFp4M1GridCapMeasurements {
  std::array<std::array<float, kNvFp4GridCaps.size()>,
             kNvFp4GridCapScaleDistributions.size()>
      baseline_milliseconds{};
  std::array<std::array<float, kNvFp4GridCaps.size()>,
             kNvFp4GridCapScaleDistributions.size()>
      candidate_milliseconds{};
  std::array<std::array<bool, kNvFp4GridCaps.size()>,
             kNvFp4GridCapScaleDistributions.size()>
      bitwise_equal{};
};

struct NvFp4M1GridCapShape {
  std::size_t rows;
  std::size_t columns;
  std::size_t profile_calls;
  const char* label;
};

[[nodiscard]] NvFp4M1GridCapMeasurements benchmark_nvfp4_m1_grid_cap_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  const int warmup_iterations = rows >= 100'000U ? 4 : 10;
  const int measured_iterations = rows >= 100'000U ? 10 : 24;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  const std::size_t scale_count = rows * scale_columns;

  NvFp4M1GridCapMeasurements measurements{};
  for (auto& distribution : measurements.baseline_milliseconds) {
    distribution.fill(std::numeric_limits<float>::quiet_NaN());
  }
  for (auto& distribution : measurements.candidate_milliseconds) {
    distribution.fill(std::numeric_limits<float>::quiet_NaN());
  }

  std::vector<std::uint8_t> host_scales(scale_count);
  std::vector<std::uint16_t> host_activation(columns);
  for (std::size_t column = 0U; column < columns; ++column) {
    const int centered =
        static_cast<int>((column * 13U + 3U) % 127U) - 63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(scale_count),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activation.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(
                       baseline_output.allocate(rows),
                       label + " allocate baseline output");
  ready = ready && test.cuda_ok(
                       candidate_output.allocate(rows),
                       label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21, packed_count,
                                       stream),
                       label + " initialize packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize mixed activations");
  if (!ready) {
    return measurements;
  }

  for (std::size_t distribution_index = 0U;
       distribution_index < kNvFp4GridCapScaleDistributions.size();
       ++distribution_index) {
    const NvFp4M1ScaleDistribution distribution =
        kNvFp4GridCapScaleDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_scale_distribution(host_scales, scale_columns,
                                     distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    if (!ready) {
      return measurements;
    }

    const auto launch_baseline = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activations.get(),
              rows, columns, baseline_output.get(),
              static_cast<void*>(stream));
    };

    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         distribution_label +
                             " correctness uncapped baseline");
    std::vector<std::uint16_t> baseline(rows);
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy uncapped baseline");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " baseline synchronize");
    if (!ready) {
      return measurements;
    }

    for (std::size_t cap_index = 0U; cap_index < kNvFp4GridCaps.size();
         ++cap_index) {
      const std::size_t grid_cap = kNvFp4GridCaps[cap_index];
      const auto launch_candidate = [&]() noexcept -> int {
        return q3x::kernels::
            launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_grid_cap_test_cuda(
                packed.get(), scales.get(), kWeightScale2,
                activations.get(), rows, columns, candidate_output.get(),
                grid_cap, static_cast<void*>(stream));
      };

      ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_candidate()),
          distribution_label + " correctness capped candidate cap=" +
              std::to_string(grid_cap));
      std::vector<std::uint16_t> candidate(rows);
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               candidate.data(), candidate_output.get(),
                               candidate.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           distribution_label + " copy candidate cap=" +
                               std::to_string(grid_cap));
      ready = ready && test.cuda_ok(
                           cudaStreamSynchronize(stream),
                           distribution_label + " correctness synchronize cap=" +
                               std::to_string(grid_cap));
      if (!ready) {
        return measurements;
      }
      std::size_t mismatches = 0U;
      for (std::size_t index = 0U; index < baseline.size(); ++index) {
        mismatches += baseline[index] != candidate[index] ? 1U : 0U;
      }
      measurements.bitwise_equal[distribution_index][cap_index] =
          mismatches == 0U;
      std::cout << "NVFP4_M1_GRID_CAP_DIFF: " << label << " distribution="
                << nvfp4_m1_scale_distribution_name(distribution)
                << " grid_cap=" << grid_cap
                << " candidate_vs_uncapped_bf16=" << mismatches << '/'
                << baseline.size() << '\n';
      test.expect(mismatches == 0U,
                  distribution_label + " cap=" + std::to_string(grid_cap) +
                      " matches every uncapped BF16 bit");

      for (int iteration = 0; iteration < warmup_iterations && ready;
           ++iteration) {
        ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                             distribution_label + " baseline warmup");
        ready = ready && test.cuda_ok(
                             static_cast<cudaError_t>(launch_candidate()),
                             distribution_label + " candidate warmup cap=" +
                                 std::to_string(grid_cap));
      }
      ready = ready && test.cuda_ok(
                           cudaStreamSynchronize(stream),
                           distribution_label + " warmup synchronize cap=" +
                               std::to_string(grid_cap));
      if (!ready) {
        return measurements;
      }

      double baseline_total = 0.0;
      double candidate_total = 0.0;
      bool finite = true;
      for (int round = 0; round < kMeasurementRounds; ++round) {
        const std::string suffix =
            " cap=" + std::to_string(grid_cap) +
            " round=" + std::to_string(round + 1);
        const float baseline_first = measure_small_m_tile(
            test, stream, launch_baseline, measured_iterations,
            distribution_label + " baseline pass 1" + suffix);
        const float candidate_first = measure_small_m_tile(
            test, stream, launch_candidate, measured_iterations,
            distribution_label + " candidate pass 1" + suffix);
        const float candidate_second = measure_small_m_tile(
            test, stream, launch_candidate, measured_iterations,
            distribution_label + " candidate pass 2" + suffix);
        const float baseline_second = measure_small_m_tile(
            test, stream, launch_baseline, measured_iterations,
            distribution_label + " baseline pass 2" + suffix);
        const bool round_finite =
            std::isfinite(baseline_first) &&
            std::isfinite(candidate_first) &&
            std::isfinite(candidate_second) &&
            std::isfinite(baseline_second);
        finite = finite && round_finite;
        if (round_finite) {
          baseline_total += baseline_first + baseline_second;
          candidate_total += candidate_first + candidate_second;
        }
        std::cout << "PERF_NVFP4_M1_GRID_CAP_ROUND: " << label
                  << " distribution="
                  << nvfp4_m1_scale_distribution_name(distribution)
                  << " grid_cap=" << grid_cap << " round=" << round + 1
                  << " baseline_pass1_ms=" << baseline_first
                  << " candidate_pass1_ms=" << candidate_first
                  << " candidate_pass2_ms=" << candidate_second
                  << " baseline_pass2_ms=" << baseline_second << '\n';
      }
      if (!finite) {
        return measurements;
      }
      constexpr double kTimedPasses =
          2.0 * static_cast<double>(kMeasurementRounds);
      const float baseline_average =
          static_cast<float>(baseline_total / kTimedPasses);
      const float candidate_average =
          static_cast<float>(candidate_total / kTimedPasses);
      measurements.baseline_milliseconds[distribution_index][cap_index] =
          baseline_average;
      measurements.candidate_milliseconds[distribution_index][cap_index] =
          candidate_average;
      const float speedup = baseline_average / candidate_average;
      std::cout << "PERF_NVFP4_M1_GRID_CAP: " << label << " distribution="
                << nvfp4_m1_scale_distribution_name(distribution)
                << " grid_cap=" << grid_cap
                << " baseline_average_ms=" << baseline_average
                << " candidate_average_ms=" << candidate_average
                << " speedup=" << speedup
                << " uplift_percent=" << (speedup - 1.0F) * 100.0F
                << '\n';
    }
  }
  return measurements;
}

template <std::size_t ShapeCount>
[[nodiscard]] std::size_t select_nvfp4_m1_grid_cap(
    TestContext& test,
    const std::array<NvFp4M1GridCapShape, ShapeCount>& shapes,
    const std::array<NvFp4M1GridCapMeasurements, ShapeCount>& measurements,
    const std::size_t frozen_cap_index) {
  constexpr float kMinimumWeightedSpeedup = 1.03F;
  constexpr float kMinimumPerShapeSpeedup = 0.98F;
  std::array<bool, kNvFp4GridCaps.size()> aggregate_gates{};
  std::array<double, kNvFp4GridCaps.size()> weighted_speedups{};
  std::size_t selected_cap_index = kNvFp4GridCaps.size();
  double selected_weighted_speedup = 0.0;
  for (std::size_t cap_index = 0U; cap_index < kNvFp4GridCaps.size();
       ++cap_index) {
    double weighted_baseline = 0.0;
    double weighted_candidate = 0.0;
    bool all_finite = true;
    bool all_bitwise = true;
    bool no_shape_regression = true;
    for (std::size_t shape_index = 0U; shape_index < ShapeCount;
         ++shape_index) {
      for (std::size_t distribution_index = 0U;
           distribution_index < kNvFp4GridCapScaleDistributions.size();
           ++distribution_index) {
        const float baseline = measurements[shape_index]
                                   .baseline_milliseconds[distribution_index]
                                                         [cap_index];
        const float candidate = measurements[shape_index]
                                    .candidate_milliseconds[distribution_index]
                                                          [cap_index];
        const bool finite =
            std::isfinite(baseline) && std::isfinite(candidate);
        const float speedup = baseline / candidate;
        all_finite = all_finite && finite;
        all_bitwise = all_bitwise &&
                      measurements[shape_index]
                          .bitwise_equal[distribution_index][cap_index];
        no_shape_regression = no_shape_regression && finite &&
                              speedup >= kMinimumPerShapeSpeedup;
        if (distribution_index == 0U) {
          weighted_baseline +=
              static_cast<double>(shapes[shape_index].profile_calls) *
              baseline;
          weighted_candidate +=
              static_cast<double>(shapes[shape_index].profile_calls) *
              candidate;
        }
        std::cout << "PERF_NVFP4_M1_GRID_CAP_VALIDATION: "
                  << shapes[shape_index].label
                  << " distribution="
                  << nvfp4_m1_scale_distribution_name(
                         kNvFp4GridCapScaleDistributions[distribution_index])
                  << " grid_cap=" << kNvFp4GridCaps[cap_index]
                  << " baseline_ms=" << baseline
                  << " candidate_ms=" << candidate
                  << " speedup=" << speedup
                  << " minimum_speedup=" << kMinimumPerShapeSpeedup
                  << " gate="
                  << (finite && speedup >= kMinimumPerShapeSpeedup ? "PASS"
                                                                    : "FAIL")
                  << '\n';
      }
    }
    const double weighted_speedup = weighted_baseline / weighted_candidate;
    const bool aggregate_gate =
        all_finite && all_bitwise && no_shape_regression &&
        std::isfinite(weighted_speedup) &&
        weighted_speedup >= kMinimumWeightedSpeedup;
    aggregate_gates[cap_index] = aggregate_gate;
    weighted_speedups[cap_index] = weighted_speedup;
    std::cout << "PERF_NVFP4_M1_GRID_CAP_AGGREGATE: grid_cap="
              << kNvFp4GridCaps[cap_index]
              << " checkpoint_weighted_baseline_ms=" << weighted_baseline
              << " checkpoint_weighted_candidate_ms=" << weighted_candidate
              << " speedup=" << weighted_speedup
              << " required_speedup=" << kMinimumWeightedSpeedup
              << " all_bitwise=" << (all_bitwise ? "true" : "false")
              << " no_shape_regression="
              << (no_shape_regression ? "true" : "false")
              << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
    if (aggregate_gate && weighted_speedup > selected_weighted_speedup) {
      selected_cap_index = cap_index;
      selected_weighted_speedup = weighted_speedup;
    }
  }
  if (frozen_cap_index < kNvFp4GridCaps.size()) {
    const bool frozen_gate = aggregate_gates[frozen_cap_index];
    selected_cap_index =
        frozen_gate ? frozen_cap_index : kNvFp4GridCaps.size();
    selected_weighted_speedup = weighted_speedups[frozen_cap_index];
    test.expect(frozen_gate,
                "NVFP4 M1 frozen grid cap=" +
                    std::to_string(kNvFp4GridCaps[frozen_cap_index]) +
                    " passes every production gate");
    std::cout << "PERF_NVFP4_M1_GRID_CAP_SELECTED: grid_cap="
              << kNvFp4GridCaps[frozen_cap_index]
              << " weighted_speedup=" << selected_weighted_speedup
              << " gate=" << (frozen_gate ? "PASS" : "FAIL") << '\n';
    return selected_cap_index;
  }
  return kNvFp4GridCaps.size();
}

void run_nvfp4_m1_large_row_grid_stride_coverage(
    TestContext& test, cudaStream_t stream, const std::size_t grid_cap) {
  constexpr std::size_t kRows = 65'537U;
  constexpr std::size_t kColumns = 256U;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  const std::string label = "NVFP4 M1 grid-cap >65535-row coverage";
  constexpr std::size_t kPackedColumns = kColumns / 2U;
  constexpr std::size_t kScaleColumns = kColumns / 16U;
  std::vector<std::uint8_t> host_scales(kRows * kScaleColumns);
  fill_nvfp4_m1_scale_distribution(
      host_scales, kScaleColumns,
      NvFp4M1ScaleDistribution::kCheckpointLike);
  std::vector<std::uint16_t> host_activation(
      kColumns, encode_bf16(1.0F / 256.0F));

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(kRows * kPackedColumns),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(host_scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activation.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(
                       baseline_output.allocate(kRows),
                       label + " allocate baseline output");
  ready = ready && test.cuda_ok(
                       candidate_output.allocate(kRows),
                       label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21,
                                       kRows * kPackedColumns, stream),
                       label + " initialize packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(scales.get(), host_scales.data(),
                                       host_scales.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize block scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return;
  }

  const auto launch_baseline = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_test_cuda(
            packed.get(), scales.get(), kWeightScale2, activations.get(),
            kRows, kColumns, baseline_output.get(),
            static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_grid_cap_test_cuda(
            packed.get(), scales.get(), kWeightScale2, activations.get(),
            kRows, kColumns, candidate_output.get(), grid_cap,
            static_cast<void*>(stream));
  };
  ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                       label + " launch uncapped baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate()),
                       label + " launch capped candidate");
  std::vector<std::uint16_t> baseline(kRows);
  std::vector<std::uint16_t> candidate(kRows);
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
                                label + " synchronize");
  if (!ready) {
    return;
  }
  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < baseline.size(); ++index) {
    mismatches += baseline[index] != candidate[index] ? 1U : 0U;
  }
  std::cout << "NVFP4_M1_GRID_CAP_LARGE_ROWS_DIFF: grid_cap=" << grid_cap
            << " candidate_vs_uncapped_bf16=" << mismatches << '/'
            << baseline.size() << '\n';
  test.expect(mismatches == 0U,
              label + " covers every row with identical BF16 output");
}

void run_optional_nvfp4_m1_grid_cap_performance(TestContext& test,
                                                 cudaStream_t stream) {
  if (!nvfp4_m1_grid_cap_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M1 persistent-row grid-cap segment; set "
                 "Q3X_RUN_SM87_NVFP4_M1_GRID_CAP_PERF=1 to enable\n";
    return;
  }
  constexpr std::array<NvFp4M1GridCapShape, 3U> kShapes{{
      {17'408U, 5'120U, 256U, "NVFP4 M1 MLP gate/up 17408x5120"},
      {5'120U, 17'408U, 128U, "NVFP4 M1 MLP down 5120x17408"},
      {248'320U, 5'120U, 2U, "NVFP4 M1 lm_head 248320x5120"},
  }};
  std::array<NvFp4M1GridCapMeasurements, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_nvfp4_m1_grid_cap_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }
  const std::size_t production_cap =
      q3x::kernels::sm87_nvfp4_m1_persistent_maximum_blocks_test();
  std::size_t production_cap_index = kNvFp4GridCaps.size();
  for (std::size_t cap_index = 0U; cap_index < kNvFp4GridCaps.size();
       ++cap_index) {
    if (kNvFp4GridCaps[cap_index] == production_cap) {
      production_cap_index = cap_index;
      break;
    }
  }
  const bool production_cap_is_measured =
      production_cap_index < kNvFp4GridCaps.size();
  test.expect(production_cap_is_measured,
              "NVFP4 M1 production grid cap is present in the sweep");
  if (!production_cap_is_measured) {
    return;
  }
  const std::size_t selected_cap_index =
      select_nvfp4_m1_grid_cap(test, kShapes, measurements,
                              production_cap_index);
  if (selected_cap_index < kNvFp4GridCaps.size()) {
    run_nvfp4_m1_large_row_grid_stride_coverage(
        test, stream, kNvFp4GridCaps[selected_cap_index]);
  }
}

constexpr std::array<std::size_t, 3U> kNvFp4M1RowPairGridCaps{{
    64U,
    80U,
    96U,
}};

constexpr std::array<NvFp4M1ScaleDistribution, 2U>
    kNvFp4M1RowPairScaleDistributions{{
        NvFp4M1ScaleDistribution::kCheckpointLike,
        NvFp4M1ScaleDistribution::kSameBankStress,
    }};

struct NvFp4M1RowPairDistributionMeasurement {
  std::array<float, kNvFp4M1RowPairGridCaps.size()>
      baseline_milliseconds{};
  std::array<float, kNvFp4M1RowPairGridCaps.size()>
      candidate_milliseconds{};
  std::array<bool, kNvFp4M1RowPairGridCaps.size()> bitwise_equal{};
};

struct NvFp4M1RowPairMeasurement {
  std::array<NvFp4M1RowPairDistributionMeasurement,
             kNvFp4M1RowPairScaleDistributions.size()>
      distributions{};
};

struct NvFp4M1RowPairShape {
  std::size_t rows;
  std::size_t columns;
  std::size_t profile_calls;
  const char* label;
};

[[nodiscard]] NvFp4M1RowPairMeasurement
benchmark_nvfp4_m1_row_pair_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kProductionGridCap = 96U;
  constexpr int kWarmupIterations = 10;
  constexpr int kDefaultMeasuredIterations = 40;
  constexpr int kLmHeadMeasuredIterations = 10;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  const int measured_iterations =
      rows >= 100'000U ? kLmHeadMeasuredIterations
                       : kDefaultMeasuredIterations;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  std::vector<std::uint8_t> host_scales(rows * scale_columns);
  std::vector<std::uint16_t> host_activation(columns);
  for (std::size_t column = 0U; column < columns; ++column) {
    const int centered =
        static_cast<int>((column * 17U + 5U) % 127U) - 63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  NvFp4M1RowPairMeasurement measurement;
  for (NvFp4M1RowPairDistributionMeasurement& distribution :
       measurement.distributions) {
    distribution.baseline_milliseconds.fill(
        std::numeric_limits<float>::quiet_NaN());
    distribution.candidate_milliseconds.fill(
        std::numeric_limits<float>::quiet_NaN());
  }

  test.expect(
      q3x::kernels::sm87_nvfp4_m1_persistent_maximum_blocks_test() ==
          kProductionGridCap,
      label + " performance baseline is the frozen production cap96");

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(host_scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activation.allocate(host_activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21, packed_count,
                                       stream),
                       label + " initialize packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize mixed activation");
  if (!ready) {
    return measurement;
  }

  std::vector<std::uint16_t> baseline(rows);
  std::vector<std::uint16_t> candidate(rows);
  for (std::size_t distribution_index = 0U;
       distribution_index < kNvFp4M1RowPairScaleDistributions.size();
       ++distribution_index) {
    const NvFp4M1ScaleDistribution distribution =
        kNvFp4M1RowPairScaleDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_scale_distribution(host_scales, scale_columns,
                                     distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    if (!ready) {
      return measurement;
    }

    for (std::size_t cap_index = 0U;
         cap_index < kNvFp4M1RowPairGridCaps.size(); ++cap_index) {
      const std::size_t candidate_grid_cap =
          kNvFp4M1RowPairGridCaps[cap_index];
      const std::string cap_label =
          distribution_label + " candidate_cap=" +
          std::to_string(candidate_grid_cap);
      const auto launch_baseline = [&]() noexcept -> int {
        return q3x::kernels::
            launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_grid_cap_test_cuda(
                packed.get(), scales.get(), kWeightScale2, activation.get(),
                rows, columns, baseline_output.get(), kProductionGridCap,
                static_cast<void*>(stream));
      };
      const auto launch_candidate = [&]() noexcept -> int {
        return q3x::kernels::
            launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_pair_grid_cap_test_cuda(
                packed.get(), scales.get(), kWeightScale2, activation.get(),
                rows, columns, candidate_output.get(), candidate_grid_cap,
                static_cast<void*>(stream));
      };
      const auto launch_public = [&]() noexcept -> int {
        return q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
            packed.get(), scales.get(), kWeightScale2, activation.get(), rows,
            columns, baseline_output.get(), static_cast<void*>(stream));
      };

      ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_baseline()),
          cap_label + " correctness preserved production cap96 baseline");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           cap_label + " correctness direct row-pair");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               baseline.data(), baseline_output.get(),
                               baseline.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy baseline output");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               candidate.data(), candidate_output.get(),
                               candidate.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy candidate output");
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " correctness synchronize");
      if (!ready) {
        return measurement;
      }

      std::size_t mismatches = 0U;
      for (std::size_t row = 0U; row < rows; ++row) {
        mismatches += baseline[row] != candidate[row] ? 1U : 0U;
      }
      NvFp4M1RowPairDistributionMeasurement& distribution_measurement =
          measurement.distributions[distribution_index];
      distribution_measurement.bitwise_equal[cap_index] = mismatches == 0U;
      std::cout << "NVFP4_M1_ROW_PAIR_PERF_DIFF: " << label
                << " distribution="
                << nvfp4_m1_scale_distribution_name(distribution)
                << " baseline_cap=" << kProductionGridCap
                << " candidate_cap=" << candidate_grid_cap
                << " mismatches=" << mismatches << '/' << rows << '\n';
      test.expect(
          distribution_measurement.bitwise_equal[cap_index],
          cap_label +
              " row-pair matches every preserved production BF16 bit");

      const std::size_t preserved_row_pair_cap =
          q3x::kernels::sm87_nvfp4_m1_row_pair_maximum_blocks_test();
      if (candidate_grid_cap == preserved_row_pair_cap) {
        ready = test.cuda_ok(
            static_cast<cudaError_t>(launch_public()),
            cap_label + " launch public production dispatch");
        ready = ready && test.cuda_ok(
                             cudaMemcpyAsync(
                                 baseline.data(), baseline_output.get(),
                                 baseline.size() * sizeof(std::uint16_t),
                                 cudaMemcpyDeviceToHost, stream),
                             cap_label + " copy public production output");
        ready = ready && test.cuda_ok(
                             cudaStreamSynchronize(stream),
                             cap_label + " public production synchronize");
        if (!ready) {
          return measurement;
        }
        std::size_t public_mismatches = 0U;
        for (std::size_t row = 0U; row < rows; ++row) {
          public_mismatches += baseline[row] != candidate[row] ? 1U : 0U;
        }
        std::cout << "NVFP4_M1_ROW_PAIR_PUBLIC_COMPAT_DIFF: " << label
                  << " distribution="
                  << nvfp4_m1_scale_distribution_name(distribution)
                  << " public_vs_preserved_pair_cap="
                  << preserved_row_pair_cap
                  << " mismatches=" << public_mismatches << '/' << rows
                  << '\n';
        test.expect(
            public_mismatches == 0U,
            cap_label +
                " public row-quad remains bitwise-compatible with row-pair");
      }

      for (int iteration = 0; iteration < kWarmupIterations && ready;
           ++iteration) {
        ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                             cap_label + " baseline warmup");
        ready = ready && test.cuda_ok(
                             static_cast<cudaError_t>(launch_candidate()),
                             cap_label + " candidate warmup");
      }
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " warmup synchronize");
      if (!ready) {
        return measurement;
      }

      double baseline_total = 0.0;
      double candidate_total = 0.0;
      bool all_finite = true;
      for (int round = 0; round < kMeasurementRounds; ++round) {
        const std::string round_label =
            cap_label + " round=" + std::to_string(round + 1);
        const float baseline_first = measure_small_m_tile(
            test, stream, launch_baseline, measured_iterations,
            round_label + " baseline pass 1");
        const float candidate_first = measure_small_m_tile(
            test, stream, launch_candidate, measured_iterations,
            round_label + " candidate pass 1");
        const float candidate_second = measure_small_m_tile(
            test, stream, launch_candidate, measured_iterations,
            round_label + " candidate pass 2");
        const float baseline_second = measure_small_m_tile(
            test, stream, launch_baseline, measured_iterations,
            round_label + " baseline pass 2");
        const bool round_finite =
            std::isfinite(baseline_first) &&
            std::isfinite(candidate_first) &&
            std::isfinite(candidate_second) &&
            std::isfinite(baseline_second);
        all_finite = all_finite && round_finite;
        if (round_finite) {
          baseline_total += baseline_first + baseline_second;
          candidate_total += candidate_first + candidate_second;
        }
        std::cout << "PERF_NVFP4_M1_ROW_PAIR_ROUND: " << label
                  << " distribution="
                  << nvfp4_m1_scale_distribution_name(distribution)
                  << " baseline_cap=" << kProductionGridCap
                  << " candidate_cap=" << candidate_grid_cap
                  << " measured_iterations=" << measured_iterations
                  << " round=" << round + 1
                  << " baseline_pass1_ms=" << baseline_first
                  << " candidate_pass1_ms=" << candidate_first
                  << " candidate_pass2_ms=" << candidate_second
                  << " baseline_pass2_ms=" << baseline_second << '\n';
      }
      constexpr double kTimedPasses =
          2.0 * static_cast<double>(kMeasurementRounds);
      distribution_measurement.baseline_milliseconds[cap_index] =
          all_finite
              ? static_cast<float>(baseline_total / kTimedPasses)
              : std::numeric_limits<float>::quiet_NaN();
      distribution_measurement.candidate_milliseconds[cap_index] =
          all_finite
              ? static_cast<float>(candidate_total / kTimedPasses)
              : std::numeric_limits<float>::quiet_NaN();
      const float speedup =
          distribution_measurement.baseline_milliseconds[cap_index] /
          distribution_measurement.candidate_milliseconds[cap_index];
      std::cout << "PERF_NVFP4_M1_ROW_PAIR: " << label
                << " distribution="
                << nvfp4_m1_scale_distribution_name(distribution)
                << " preserved_production_cap96_ms="
                << distribution_measurement
                       .baseline_milliseconds[cap_index]
                << " candidate_cap=" << candidate_grid_cap
                << " direct_row_pair_ms="
                << distribution_measurement
                       .candidate_milliseconds[cap_index]
                << " speedup=" << speedup << " bitwise_mismatches="
                << mismatches << '/' << rows << '\n';
    }
  }
  return measurement;
}

void run_optional_nvfp4_m1_row_pair_performance(TestContext& test,
                                                 cudaStream_t stream) {
  if (!nvfp4_m1_row_pair_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M1 scale-codebook row-pair performance "
                 "segment; set Q3X_RUN_SM87_NVFP4_M1_ROW_PAIR_PERF=1 to "
                 "enable\n";
    return;
  }
  constexpr float kMinimumCheckpointShapeSpeedup = 1.02F;
  constexpr float kMinimumStressShapeSpeedup = 0.99F;
  constexpr float kMinimumCheckpointWeightedSpeedup = 1.05F;
  constexpr std::array<NvFp4M1RowPairShape, 3U> kShapes{{
      {17'408U, 5'120U, 128U, "NVFP4 M1 row-pair 17408x5120"},
      {5'120U, 17'408U, 64U, "NVFP4 M1 row-pair 5120x17408"},
      {248'320U, 5'120U, 1U, "NVFP4 M1 row-pair lm_head 248320x5120"},
  }};

  std::array<NvFp4M1RowPairMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_nvfp4_m1_row_pair_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }

  std::size_t selected_cap_index = kNvFp4M1RowPairGridCaps.size();
  double selected_weighted_speedup = 0.0;
  std::array<double, kNvFp4M1RowPairGridCaps.size()> weighted_speedups{};
  std::array<bool, kNvFp4M1RowPairGridCaps.size()> aggregate_gates{};
  for (std::size_t cap_index = 0U;
       cap_index < kNvFp4M1RowPairGridCaps.size(); ++cap_index) {
    bool all_shape_distributions_pass = true;
    double checkpoint_weighted_baseline = 0.0;
    double checkpoint_weighted_candidate = 0.0;
    for (std::size_t shape_index = 0U; shape_index < kShapes.size();
         ++shape_index) {
      for (std::size_t distribution_index = 0U;
           distribution_index < kNvFp4M1RowPairScaleDistributions.size();
           ++distribution_index) {
        const NvFp4M1RowPairDistributionMeasurement& measurement =
            measurements[shape_index].distributions[distribution_index];
        const float baseline =
            measurement.baseline_milliseconds[cap_index];
        const float candidate =
            measurement.candidate_milliseconds[cap_index];
        const float speedup = baseline / candidate;
        const bool finite = std::isfinite(baseline) &&
                            std::isfinite(candidate) &&
                            std::isfinite(speedup) && baseline > 0.0F &&
                            candidate > 0.0F;
        const bool checkpoint_distribution =
            kNvFp4M1RowPairScaleDistributions[distribution_index] ==
            NvFp4M1ScaleDistribution::kCheckpointLike;
        const float required_speedup =
            checkpoint_distribution
                ? kMinimumCheckpointShapeSpeedup
                : kMinimumStressShapeSpeedup;
        const bool gate = measurement.bitwise_equal[cap_index] && finite &&
                          speedup >= required_speedup;
        all_shape_distributions_pass =
            all_shape_distributions_pass && gate;
        std::cout << "PERF_NVFP4_M1_ROW_PAIR_VALIDATION: "
                  << kShapes[shape_index].label << " distribution="
                  << nvfp4_m1_scale_distribution_name(
                         kNvFp4M1RowPairScaleDistributions
                             [distribution_index])
                  << " baseline_cap=96 candidate_cap="
                  << kNvFp4M1RowPairGridCaps[cap_index]
                  << " baseline_ms=" << baseline
                  << " candidate_ms=" << candidate
                  << " speedup=" << speedup
                  << " required_speedup=" << required_speedup
                  << " bitwise="
                  << (measurement.bitwise_equal[cap_index] ? "true"
                                                           : "false")
                  << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
      }
      const NvFp4M1RowPairDistributionMeasurement& checkpoint =
          measurements[shape_index].distributions[0U];
      checkpoint_weighted_baseline +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          checkpoint.baseline_milliseconds[cap_index];
      checkpoint_weighted_candidate +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          checkpoint.candidate_milliseconds[cap_index];
    }

    const double checkpoint_weighted_speedup =
        checkpoint_weighted_baseline / checkpoint_weighted_candidate;
    const bool aggregate_gate =
        all_shape_distributions_pass &&
        std::isfinite(checkpoint_weighted_speedup) &&
        checkpoint_weighted_baseline > 0.0 &&
        checkpoint_weighted_candidate > 0.0 &&
        checkpoint_weighted_speedup >=
            kMinimumCheckpointWeightedSpeedup;
    weighted_speedups[cap_index] = checkpoint_weighted_speedup;
    aggregate_gates[cap_index] = aggregate_gate;
    std::cout << "PERF_NVFP4_M1_ROW_PAIR_AGGREGATE: baseline_cap=96"
              << " candidate_cap="
              << kNvFp4M1RowPairGridCaps[cap_index]
              << " checkpoint_weighted_baseline_ms="
              << checkpoint_weighted_baseline
              << " checkpoint_weighted_candidate_ms="
              << checkpoint_weighted_candidate
              << " speedup=" << checkpoint_weighted_speedup
              << " required_speedup="
              << kMinimumCheckpointWeightedSpeedup
              << " profile_calls=128:64:1 all_shape_distributions="
              << (all_shape_distributions_pass ? "PASS" : "FAIL")
              << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
    if (aggregate_gate &&
        checkpoint_weighted_speedup > selected_weighted_speedup) {
      selected_cap_index = cap_index;
      selected_weighted_speedup = checkpoint_weighted_speedup;
    }
  }

  const bool selected =
      selected_cap_index < kNvFp4M1RowPairGridCaps.size();
  test.expect(selected,
              "NVFP4 M1 row-pair sweep selects a cap clearing every gate");
  if (!selected) {
    std::cout << "PERF_NVFP4_M1_ROW_PAIR_SELECTED: gate=FAIL\n";
    return;
  }
  std::cout << "PERF_NVFP4_M1_ROW_PAIR_SELECTED: candidate_cap="
            << kNvFp4M1RowPairGridCaps[selected_cap_index]
            << " checkpoint_weighted_speedup="
            << selected_weighted_speedup << " gate=PASS\n";

  const std::size_t preserved_cap =
      q3x::kernels::sm87_nvfp4_m1_row_pair_maximum_blocks_test();
  std::size_t preserved_cap_index = kNvFp4M1RowPairGridCaps.size();
  for (std::size_t cap_index = 0U;
       cap_index < kNvFp4M1RowPairGridCaps.size(); ++cap_index) {
    if (kNvFp4M1RowPairGridCaps[cap_index] == preserved_cap) {
      preserved_cap_index = cap_index;
      break;
    }
  }
  const bool preserved_cap_measured =
      preserved_cap_index < kNvFp4M1RowPairGridCaps.size();
  test.expect(preserved_cap_measured,
              "NVFP4 M1 preserved row-pair cap is present in the sweep");
  if (!preserved_cap_measured) {
    return;
  }
  const bool preserved_gate = aggregate_gates[preserved_cap_index];
  std::cout << "PERF_NVFP4_M1_ROW_PAIR_PRESERVED: candidate_cap="
            << preserved_cap << " checkpoint_weighted_speedup="
            << weighted_speedups[preserved_cap_index]
            << " gate=" << (preserved_gate ? "PASS" : "FAIL") << '\n';
  test.expect(preserved_gate,
              "NVFP4 M1 frozen row-pair A/B cap clears every gate");
}

constexpr std::array<std::size_t, 3U> kNvFp4M1RowQuadGridCaps{{
    48U,
    64U,
    80U,
}};

constexpr std::array<NvFp4M1ScaleDistribution, 2U>
    kNvFp4M1RowQuadScaleDistributions{{
        NvFp4M1ScaleDistribution::kCheckpointLike,
        NvFp4M1ScaleDistribution::kSameBankStress,
    }};

struct NvFp4M1RowQuadDistributionMeasurement {
  std::array<float, kNvFp4M1RowQuadGridCaps.size()>
      baseline_milliseconds{};
  std::array<float, kNvFp4M1RowQuadGridCaps.size()>
      candidate_milliseconds{};
  std::array<bool, kNvFp4M1RowQuadGridCaps.size()> bitwise_equal{};
};

struct NvFp4M1RowQuadMeasurement {
  std::array<NvFp4M1RowQuadDistributionMeasurement,
             kNvFp4M1RowQuadScaleDistributions.size()>
      distributions{};
};

struct NvFp4M1RowQuadShape {
  std::size_t rows;
  std::size_t columns;
  std::size_t profile_calls;
  const char* label;
};

[[nodiscard]] NvFp4M1RowQuadMeasurement
benchmark_nvfp4_m1_row_quad_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kBaselineGridCap = 80U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::size_t kPackedPatternBytes = 256U;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  std::vector<std::uint8_t> host_scales(rows * scale_columns);
  std::vector<std::uint16_t> host_activation(columns);
  for (std::size_t column = 0U; column < columns; ++column) {
    const int centered =
        static_cast<int>((column * 17U + 5U) % 127U) - 63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  std::array<std::uint8_t, kPackedPatternBytes> host_packed_pattern{};
  std::array<bool, 16U> low_nibble_covered{};
  std::array<bool, 16U> high_nibble_covered{};
  for (std::size_t index = 0U; index < host_packed_pattern.size(); ++index) {
    const std::uint8_t low = static_cast<std::uint8_t>(
        (index * 3U + (index >> 3U) + 1U) & 0x0fU);
    const std::uint8_t high = static_cast<std::uint8_t>(
        (index * 5U + (index >> 4U) * 7U + 3U) & 0x0fU);
    host_packed_pattern[index] =
        static_cast<std::uint8_t>(low | (high << 4U));
    low_nibble_covered[low] = true;
    high_nibble_covered[high] = true;
  }
  test.expect(std::all_of(low_nibble_covered.begin(),
                          low_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers every E2M1 low-nibble code");
  test.expect(std::all_of(high_nibble_covered.begin(),
                          high_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " covers every E2M1 high-nibble code");
  test.expect((columns / 2U) % host_packed_pattern.size() == 0U,
              label + " repeats the full-nibble pattern within every row");

  NvFp4M1RowQuadMeasurement measurement;
  for (NvFp4M1RowQuadDistributionMeasurement& distribution :
       measurement.distributions) {
    distribution.baseline_milliseconds.fill(
        std::numeric_limits<float>::quiet_NaN());
    distribution.candidate_milliseconds.fill(
        std::numeric_limits<float>::quiet_NaN());
  }

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(host_scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activation.allocate(host_activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(rows),
                                label + " allocate row-pair output");
  ready = ready && test.cuda_ok(candidate_output.allocate(rows),
                                label + " allocate row-quad output");
  const std::size_t initial_packed_bytes =
      std::min(packed_count, host_packed_pattern.size());
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           packed.get(), host_packed_pattern.data(),
                           initial_packed_bytes, cudaMemcpyHostToDevice,
                           stream),
                       label + " initialize mixed all-nibble seed");
  for (std::size_t initialized = initial_packed_bytes;
       ready && initialized < packed_count;) {
    const std::size_t copy_bytes =
        std::min(initialized, packed_count - initialized);
    ready = test.cuda_ok(
        cudaMemcpyAsync(packed.get() + initialized, packed.get(), copy_bytes,
                        cudaMemcpyDeviceToDevice, stream),
        label + " expand mixed all-nibble weights");
    initialized += copy_bytes;
  }
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize mixed activation");
  if (!ready) {
    return measurement;
  }

  std::vector<std::uint16_t> baseline(rows);
  std::vector<std::uint16_t> candidate(rows);
  for (std::size_t distribution_index = 0U;
       distribution_index < kNvFp4M1RowQuadScaleDistributions.size();
       ++distribution_index) {
    const NvFp4M1ScaleDistribution distribution =
        kNvFp4M1RowQuadScaleDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_scale_distribution(host_scales, scale_columns,
                                     distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    if (!ready) {
      return measurement;
    }

    for (std::size_t cap_index = 0U;
         cap_index < kNvFp4M1RowQuadGridCaps.size(); ++cap_index) {
      const std::size_t candidate_grid_cap =
          kNvFp4M1RowQuadGridCaps[cap_index];
      const std::string cap_label =
          distribution_label + " candidate_cap=" +
          std::to_string(candidate_grid_cap);
      const auto launch_baseline = [&]() noexcept -> int {
        return q3x::kernels::
            launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_pair_grid_cap_test_cuda(
                packed.get(), scales.get(), kWeightScale2, activation.get(),
                rows, columns, baseline_output.get(), kBaselineGridCap,
                static_cast<void*>(stream));
      };
      const auto launch_candidate = [&]() noexcept -> int {
        return q3x::kernels::
            launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_grid_cap_test_cuda(
                packed.get(), scales.get(), kWeightScale2, activation.get(),
                rows, columns, candidate_output.get(), candidate_grid_cap,
                static_cast<void*>(stream));
      };

      ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_baseline()),
          cap_label + " correctness direct row-pair cap80 baseline");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           cap_label + " correctness direct row-quad");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               baseline.data(), baseline_output.get(),
                               baseline.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-pair output");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               candidate.data(), candidate_output.get(),
                               candidate.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy row-quad output");
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " correctness synchronize");
      if (!ready) {
        return measurement;
      }

      std::size_t mismatches = 0U;
      for (std::size_t row = 0U; row < rows; ++row) {
        mismatches += baseline[row] != candidate[row] ? 1U : 0U;
      }
      NvFp4M1RowQuadDistributionMeasurement& distribution_measurement =
          measurement.distributions[distribution_index];
      distribution_measurement.bitwise_equal[cap_index] = mismatches == 0U;
      std::cout << "NVFP4_M1_ROW_QUAD_DIFF: " << label
                << " distribution="
                << nvfp4_m1_scale_distribution_name(distribution)
                << " baseline_pair_cap=" << kBaselineGridCap
                << " candidate_quad_cap=" << candidate_grid_cap
                << " mismatches=" << mismatches << '/' << rows << '\n';
      test.expect(distribution_measurement.bitwise_equal[cap_index],
                  cap_label +
                      " row-quad matches every row-pair BF16 bit");

      for (int iteration = 0; iteration < kWarmupIterations && ready;
           ++iteration) {
        ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                             cap_label + " row-pair warmup");
        ready = ready && test.cuda_ok(
                             static_cast<cudaError_t>(launch_candidate()),
                             cap_label + " row-quad warmup");
      }
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " warmup synchronize");
      if (!ready) {
        return measurement;
      }

      double baseline_total = 0.0;
      double candidate_total = 0.0;
      bool all_finite = true;
      for (int round = 0; round < kMeasurementRounds; ++round) {
        const std::string round_label =
            cap_label + " round=" + std::to_string(round + 1);
        const float baseline_first = measure_small_m_tile(
            test, stream, launch_baseline, kMeasuredIterations,
            round_label + " baseline pass 1");
        const float candidate_first = measure_small_m_tile(
            test, stream, launch_candidate, kMeasuredIterations,
            round_label + " candidate pass 1");
        const float candidate_second = measure_small_m_tile(
            test, stream, launch_candidate, kMeasuredIterations,
            round_label + " candidate pass 2");
        const float baseline_second = measure_small_m_tile(
            test, stream, launch_baseline, kMeasuredIterations,
            round_label + " baseline pass 2");
        const bool round_finite =
            std::isfinite(baseline_first) &&
            std::isfinite(candidate_first) &&
            std::isfinite(candidate_second) &&
            std::isfinite(baseline_second);
        all_finite = all_finite && round_finite;
        if (round_finite) {
          baseline_total += baseline_first + baseline_second;
          candidate_total += candidate_first + candidate_second;
        }
        std::cout << "PERF_NVFP4_M1_ROW_QUAD_ROUND: " << label
                  << " distribution="
                  << nvfp4_m1_scale_distribution_name(distribution)
                  << " baseline_pair_cap=" << kBaselineGridCap
                  << " candidate_quad_cap=" << candidate_grid_cap
                  << " measured_iterations=" << kMeasuredIterations
                  << " round=" << round + 1
                  << " baseline_pass1_ms=" << baseline_first
                  << " candidate_pass1_ms=" << candidate_first
                  << " candidate_pass2_ms=" << candidate_second
                  << " baseline_pass2_ms=" << baseline_second << '\n';
      }
      constexpr double kTimedPasses =
          2.0 * static_cast<double>(kMeasurementRounds);
      distribution_measurement.baseline_milliseconds[cap_index] =
          all_finite
              ? static_cast<float>(baseline_total / kTimedPasses)
              : std::numeric_limits<float>::quiet_NaN();
      distribution_measurement.candidate_milliseconds[cap_index] =
          all_finite
              ? static_cast<float>(candidate_total / kTimedPasses)
              : std::numeric_limits<float>::quiet_NaN();
      const float speedup =
          distribution_measurement.baseline_milliseconds[cap_index] /
          distribution_measurement.candidate_milliseconds[cap_index];
      std::cout << "PERF_NVFP4_M1_ROW_QUAD: " << label
                << " distribution="
                << nvfp4_m1_scale_distribution_name(distribution)
                << " direct_row_pair_cap80_ms="
                << distribution_measurement
                       .baseline_milliseconds[cap_index]
                << " candidate_quad_cap=" << candidate_grid_cap
                << " direct_row_quad_ms="
                << distribution_measurement
                       .candidate_milliseconds[cap_index]
                << " speedup=" << speedup << " bitwise_mismatches="
                << mismatches << '/' << rows << '\n';
    }
  }
  return measurement;
}

void run_optional_nvfp4_m1_row_quad_performance(TestContext& test,
                                                 cudaStream_t stream) {
  if (!nvfp4_m1_row_quad_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M1 scale-codebook row-quad performance "
                 "segment; set Q3X_RUN_SM87_NVFP4_M1_ROW_QUAD_PERF=1 to "
                 "enable\n";
    return;
  }
  const std::size_t kSelectedGridCap =
      q3x::kernels::sm87_nvfp4_m1_row_quad_maximum_blocks_test();
  constexpr float kMinimumSelectedSpeedup = 1.02F;
  constexpr std::array<NvFp4M1RowQuadShape, 3U> kShapes{{
      {17'408U, 5'120U, 256U, "NVFP4 M1 row-quad 17408x5120"},
      {5'120U, 17'408U, 128U, "NVFP4 M1 row-quad 5120x17408"},
      {248'320U, 5'120U, 2U,
       "NVFP4 M1 row-quad lm_head 248320x5120"},
  }};

  std::array<NvFp4M1RowQuadMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_nvfp4_m1_row_quad_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }

  std::size_t selected_cap_index = kNvFp4M1RowQuadGridCaps.size();
  for (std::size_t cap_index = 0U;
       cap_index < kNvFp4M1RowQuadGridCaps.size(); ++cap_index) {
    if (kNvFp4M1RowQuadGridCaps[cap_index] == kSelectedGridCap) {
      selected_cap_index = cap_index;
      break;
    }
  }
  const bool selected_cap_present =
      selected_cap_index < kNvFp4M1RowQuadGridCaps.size();
  test.expect(selected_cap_present,
              "NVFP4 M1 row-quad diagnostics include selected cap64");
  if (!selected_cap_present) {
    std::cout << "PERF_NVFP4_M1_ROW_QUAD_SELECTED: "
                 "scope=all_production_shapes "
                 "candidate_quad_cap=64 gate=FAIL reason=cap_not_measured\n";
    return;
  }

  for (std::size_t cap_index = 0U;
       cap_index < kNvFp4M1RowQuadGridCaps.size(); ++cap_index) {
    double checkpoint_weighted_baseline = 0.0;
    double checkpoint_weighted_candidate = 0.0;
    for (std::size_t shape_index = 0U; shape_index < kShapes.size();
         ++shape_index) {
      for (std::size_t distribution_index = 0U;
           distribution_index < kNvFp4M1RowQuadScaleDistributions.size();
           ++distribution_index) {
        const NvFp4M1RowQuadDistributionMeasurement& measurement =
            measurements[shape_index].distributions[distribution_index];
        const float baseline =
            measurement.baseline_milliseconds[cap_index];
        const float candidate =
            measurement.candidate_milliseconds[cap_index];
        const float speedup = baseline / candidate;
        const bool finite = std::isfinite(baseline) &&
                            std::isfinite(candidate) &&
                            std::isfinite(speedup) && baseline > 0.0F &&
                            candidate > 0.0F;
        const bool selected_cell = cap_index == selected_cap_index;
        const bool selected_cell_gate =
            measurement.bitwise_equal[cap_index] && finite &&
            speedup >= kMinimumSelectedSpeedup;
        std::cout << "PERF_NVFP4_M1_ROW_QUAD_DIAGNOSTIC: "
                  << kShapes[shape_index].label << " distribution="
                  << nvfp4_m1_scale_distribution_name(
                         kNvFp4M1RowQuadScaleDistributions
                             [distribution_index])
                  << " baseline_pair_cap=80 candidate_quad_cap="
                  << kNvFp4M1RowQuadGridCaps[cap_index]
                  << " baseline_ms=" << baseline
                  << " candidate_ms=" << candidate
                  << " speedup=" << speedup
                  << " bitwise="
                  << (measurement.bitwise_equal[cap_index] ? "true"
                                                           : "false")
                  << " scope="
                  << (selected_cell ? "selected_all_shapes" : "diagnostic")
                  << " performance_gate="
                  << (selected_cell
                          ? (selected_cell_gate ? "PASS" : "FAIL")
                          : "N/A")
                  << '\n';
      }

      const NvFp4M1RowQuadDistributionMeasurement& checkpoint =
          measurements[shape_index].distributions[0U];
      const float checkpoint_baseline =
          checkpoint.baseline_milliseconds[cap_index];
      const float checkpoint_candidate =
          checkpoint.candidate_milliseconds[cap_index];
      checkpoint_weighted_baseline +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          checkpoint_baseline;
      checkpoint_weighted_candidate +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          checkpoint_candidate;
    }

    const double checkpoint_weighted_speedup =
        checkpoint_weighted_baseline / checkpoint_weighted_candidate;
    std::cout
        << "PERF_NVFP4_M1_ROW_QUAD_WEIGHTED_DIAGNOSTIC: baseline_pair_cap=80"
              << " candidate_quad_cap="
              << kNvFp4M1RowQuadGridCaps[cap_index]
              << " checkpoint_weighted_baseline_ms="
              << checkpoint_weighted_baseline
              << " checkpoint_weighted_candidate_ms="
              << checkpoint_weighted_candidate
              << " speedup=" << checkpoint_weighted_speedup
              << " profile_calls=256:128:2 scope=diagnostic\n";
  }

  bool selected_gate = true;
  bool selected_all_bitwise = true;
  float selected_minimum_speedup = std::numeric_limits<float>::infinity();
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    for (std::size_t distribution_index = 0U;
         distribution_index < kNvFp4M1RowQuadScaleDistributions.size();
         ++distribution_index) {
      const NvFp4M1RowQuadDistributionMeasurement& measurement =
          measurements[shape_index].distributions[distribution_index];
      const float baseline =
          measurement.baseline_milliseconds[selected_cap_index];
      const float candidate =
          measurement.candidate_milliseconds[selected_cap_index];
      const float speedup = baseline / candidate;
      const bool bitwise = measurement.bitwise_equal[selected_cap_index];
      const bool cell_gate =
          bitwise && std::isfinite(baseline) && std::isfinite(candidate) &&
          std::isfinite(speedup) && baseline > 0.0F && candidate > 0.0F &&
          speedup >= kMinimumSelectedSpeedup;
      selected_all_bitwise = selected_all_bitwise && bitwise;
      selected_minimum_speedup =
          std::min(selected_minimum_speedup, speedup);
      selected_gate = selected_gate && cell_gate;
    }
  }
  test.expect(
      selected_gate,
      "NVFP4 M1 row-quad cap64 clears all production-shape distribution "
      "gates");
  std::cout
      << "PERF_NVFP4_M1_ROW_QUAD_SELECTED: scope=all_production_shapes"
      << " baseline_pair_cap=80 candidate_quad_cap=" << kSelectedGridCap
            << " minimum_cell_speedup=" << selected_minimum_speedup
            << " required_each=" << kMinimumSelectedSpeedup
            << " all_bitwise="
            << (selected_all_bitwise ? "true" : "false")
            << " gate=" << (selected_gate ? "PASS" : "FAIL") << '\n';
}

struct NvFp4M1ExactShapeDistributionMeasurement {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
  bool output_finite = false;
};

struct NvFp4M1ExactShapeMeasurement {
  std::array<NvFp4M1ExactShapeDistributionMeasurement,
             kNvFp4M1RowQuadScaleDistributions.size()>
      distributions{};
};

[[nodiscard]] NvFp4M1ExactShapeMeasurement
benchmark_nvfp4_m1_exact_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::size_t profile_calls,
    const std::string& label) {
  constexpr std::size_t kBaselineGridCap = 64U;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr float kMinimumCellSpeedup = 1.015F;
  constexpr std::uint8_t kBaselineOutputSentinel = 0xa5U;
  constexpr std::uint8_t kCandidateOutputSentinel = 0x5aU;
  const int warmup_iterations = rows >= 100'000U ? 4 : 10;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t packed_columns = columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  std::vector<std::uint8_t> host_scales(rows * scale_columns);
  std::vector<std::uint16_t> host_activation(columns);
  for (std::size_t column = 0U; column < columns; ++column) {
    const int centered =
        static_cast<int>((column * 17U + 5U) % 127U) - 63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  std::array<std::uint8_t, kNvFp4M1ExactShapeFixturePeriod>
      host_packed_pattern{};
  std::array<bool, 16U> low_nibble_covered{};
  std::array<bool, 16U> high_nibble_covered{};
  for (std::size_t index = 0U; index < host_packed_pattern.size(); ++index) {
    const std::uint8_t low = static_cast<std::uint8_t>(
        (index * 3U + (index >> 3U) + 1U) & 0x0fU);
    const std::uint8_t high = static_cast<std::uint8_t>(
        (index * 5U + (index >> 4U) * 7U + 3U) & 0x0fU);
    host_packed_pattern[index] =
        static_cast<std::uint8_t>(low | (high << 4U));
    low_nibble_covered[low] = true;
    high_nibble_covered[high] = true;
  }
  test.expect(std::all_of(low_nibble_covered.begin(),
                          low_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " exact fixture covers every E2M1 low-nibble code");
  test.expect(std::all_of(high_nibble_covered.begin(),
                          high_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " exact fixture covers every E2M1 high-nibble code");
  test.expect(
      (kNvFp4M1ExactShapeRowStride %
       kNvFp4M1ExactShapeFixturePeriod) != 0U,
      label + " exact scale fixture does not alias the 2048-row stride");
  test.expect(
      ((kNvFp4M1ExactShapeRowStride %
        kNvFp4M1ExactShapeFixturePeriod) *
       (packed_columns % kNvFp4M1ExactShapeFixturePeriod)) %
              kNvFp4M1ExactShapeFixturePeriod !=
          0U,
      label + " exact packed fixture does not alias the 2048-row stride");

  NvFp4M1ExactShapeMeasurement measurement{};
  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(host_scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activation.allocate(host_activation.size()),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(rows),
                                label + " allocate candidate output");
  const std::size_t initial_packed_bytes =
      std::min(packed_count, host_packed_pattern.size());
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           packed.get(), host_packed_pattern.data(),
                           initial_packed_bytes, cudaMemcpyHostToDevice,
                           stream),
                       label + " initialize mixed all-nibble packed seed");
  for (std::size_t initialized = initial_packed_bytes;
       ready && initialized < packed_count;) {
    const std::size_t copy_bytes =
        std::min(initialized, packed_count - initialized);
    ready = test.cuda_ok(
        cudaMemcpyAsync(packed.get() + initialized, packed.get(), copy_bytes,
                        cudaMemcpyDeviceToDevice, stream),
        label + " expand prime-period packed fixture");
    initialized += copy_bytes;
  }
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  if (!ready) {
    return measurement;
  }

  std::vector<std::uint16_t> baseline(rows);
  std::vector<std::uint16_t> candidate(rows);
  for (std::size_t distribution_index = 0U;
       distribution_index < kNvFp4M1RowQuadScaleDistributions.size();
       ++distribution_index) {
    const NvFp4M1ScaleDistribution distribution =
        kNvFp4M1RowQuadScaleDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_exact_shape_scale_distribution(
        host_scales, scale_columns, distribution);
    bool scale_rows_differ_across_stride = false;
    for (std::size_t scale_column = 0U;
         scale_column < scale_columns &&
         !scale_rows_differ_across_stride;
         ++scale_column) {
      scale_rows_differ_across_stride =
          host_scales[scale_column] !=
          host_scales[kNvFp4M1ExactShapeRowStride * scale_columns +
                      scale_column];
    }
    test.expect(scale_rows_differ_across_stride,
                distribution_label +
                    " scales differ across the 2048-row stride");
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(
                             baseline_output.get(), kBaselineOutputSentinel,
                             rows * sizeof(std::uint16_t), stream),
                         distribution_label +
                             " initialize baseline output sentinel");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(
                             candidate_output.get(), kCandidateOutputSentinel,
                             rows * sizeof(std::uint16_t), stream),
                         distribution_label +
                             " initialize candidate output sentinel");
    if (!ready) {
      return measurement;
    }

    const auto launch_baseline = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_grid_cap_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activation.get(),
              rows, columns, baseline_output.get(), kBaselineGridCap,
              static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_row_quad_exact_shape_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activation.get(),
              rows, columns, candidate_output.get(),
              static_cast<void*>(stream));
    };

    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         distribution_label + " correctness baseline cap64");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         distribution_label + " correctness exact shape");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy baseline output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy candidate output");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " correctness synchronize");
    if (!ready) {
      return measurement;
    }

    std::size_t mismatches = 0U;
    bool output_finite = true;
    for (std::size_t row = 0U; row < rows; ++row) {
      mismatches += baseline[row] != candidate[row] ? 1U : 0U;
      output_finite = output_finite &&
                      std::isfinite(decode_bf16(baseline[row])) &&
                      std::isfinite(decode_bf16(candidate[row]));
    }
    NvFp4M1ExactShapeDistributionMeasurement& cell =
        measurement.distributions[distribution_index];
    cell.bitwise_equal = mismatches == 0U;
    cell.output_finite = output_finite;

    for (int iteration = 0; iteration < warmup_iterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           distribution_label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           distribution_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  distribution_label + " warmup synchronize");
    if (!ready) {
      return measurement;
    }

    double baseline_total = 0.0;
    double candidate_total = 0.0;
    bool timings_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          distribution_label + " round=" + std::to_string(round + 1);
      const float baseline_first = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 1");
      const float candidate_first = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 1");
      const float candidate_second = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 2");
      const float baseline_second = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 2");
      const bool round_finite =
          std::isfinite(baseline_first) && std::isfinite(candidate_first) &&
          std::isfinite(candidate_second) && std::isfinite(baseline_second);
      timings_finite = timings_finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_NVFP4_M1_EXACT_SHAPE_ROUND: " << label
                << " distribution="
                << nvfp4_m1_scale_distribution_name(distribution)
                << " profile_calls=" << profile_calls
                << " measured_iterations=" << kMeasuredIterations
                << " round=" << round + 1
                << " baseline_pass1_ms=" << baseline_first
                << " candidate_pass1_ms=" << candidate_first
                << " candidate_pass2_ms=" << candidate_second
                << " baseline_pass2_ms=" << baseline_second << '\n';
    }
    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    if (timings_finite) {
      cell.baseline_milliseconds =
          static_cast<float>(baseline_total / kTimedPasses);
      cell.candidate_milliseconds =
          static_cast<float>(candidate_total / kTimedPasses);
    }
    const float speedup =
        cell.baseline_milliseconds / cell.candidate_milliseconds;
    const bool finite = timings_finite && output_finite &&
                        std::isfinite(speedup) &&
                        cell.baseline_milliseconds > 0.0F &&
                        cell.candidate_milliseconds > 0.0F;
    const bool cell_gate = cell.bitwise_equal && finite &&
                           speedup >= kMinimumCellSpeedup;
    test.expect(cell_gate,
                distribution_label +
                    " exact-shape correctness, finite, and speedup gate");
    std::cout << "PERF_NVFP4_M1_EXACT_SHAPE: " << label
              << " distribution="
              << nvfp4_m1_scale_distribution_name(distribution)
              << " profile_calls=" << profile_calls
              << " baseline_row_quad_cap64_ms="
              << cell.baseline_milliseconds
              << " candidate_exact_shape_ms="
              << cell.candidate_milliseconds << " speedup=" << speedup
              << " required_speedup=" << kMinimumCellSpeedup
              << " bitwise_mismatches=" << mismatches << '/' << rows
              << " output_finite=" << (output_finite ? "true" : "false")
              << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
  }
  return measurement;
}

void run_optional_nvfp4_m1_exact_shape_performance(TestContext& test,
                                                    cudaStream_t stream) {
  if (!nvfp4_m1_exact_shape_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M1 exact-shape performance segment; set "
                 "Q3X_RUN_SM87_NVFP4_M1_EXACT_SHAPE_PERF=1 to enable\n";
    return;
  }
  constexpr std::array<NvFp4M1RowQuadShape, 3U> kShapes{{
      {17'408U, 5'120U, 128U, "NVFP4 M1 exact gate/up 17408x5120"},
      {5'120U, 17'408U, 64U, "NVFP4 M1 exact down 5120x17408"},
      {248'320U, 5'120U, 1U, "NVFP4 M1 exact lm_head 248320x5120"},
  }};
  constexpr std::array<double, 2U> kMinimumWeightedSpeedups{{
      1.025,
      1.020,
  }};
  std::array<NvFp4M1ExactShapeMeasurement, kShapes.size()> measurements{};
  std::array<double, kNvFp4M1RowQuadScaleDistributions.size()>
      weighted_speedups{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    const NvFp4M1RowQuadShape& shape = kShapes[shape_index];
    measurements[shape_index] = benchmark_nvfp4_m1_exact_shape(
        test, stream, shape.rows, shape.columns, shape.profile_calls,
        shape.label);
  }

  bool all_gates = true;
  for (std::size_t distribution_index = 0U;
       distribution_index < kNvFp4M1RowQuadScaleDistributions.size();
       ++distribution_index) {
    double weighted_baseline = 0.0;
    double weighted_candidate = 0.0;
    bool distribution_gate = true;
    for (std::size_t shape_index = 0U; shape_index < kShapes.size();
         ++shape_index) {
      const NvFp4M1ExactShapeDistributionMeasurement& cell =
          measurements[shape_index].distributions[distribution_index];
      const bool finite = std::isfinite(cell.baseline_milliseconds) &&
                          std::isfinite(cell.candidate_milliseconds) &&
                          cell.baseline_milliseconds > 0.0F &&
                          cell.candidate_milliseconds > 0.0F;
      const double cell_speedup =
          static_cast<double>(cell.baseline_milliseconds) /
          static_cast<double>(cell.candidate_milliseconds);
      distribution_gate = distribution_gate && cell.bitwise_equal &&
                          cell.output_finite && finite &&
                          std::isfinite(cell_speedup) &&
                          cell_speedup >= 1.015;
      weighted_baseline +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          cell.baseline_milliseconds;
      weighted_candidate +=
          static_cast<double>(kShapes[shape_index].profile_calls) *
          cell.candidate_milliseconds;
    }
    const double weighted_speedup = weighted_baseline / weighted_candidate;
    weighted_speedups[distribution_index] = weighted_speedup;
    const double required_speedup =
        kMinimumWeightedSpeedups[distribution_index];
    distribution_gate = distribution_gate &&
                        std::isfinite(weighted_speedup) &&
                        weighted_baseline > 0.0 && weighted_candidate > 0.0 &&
                        weighted_speedup >= required_speedup;
    all_gates = all_gates && distribution_gate;
    std::cout << "PERF_NVFP4_M1_EXACT_SHAPE_WEIGHTED: distribution="
              << nvfp4_m1_scale_distribution_name(
                     kNvFp4M1RowQuadScaleDistributions[distribution_index])
              << " baseline_row_quad_cap64_ms=" << weighted_baseline
              << " candidate_exact_shape_ms=" << weighted_candidate
              << " speedup=" << weighted_speedup
              << " required_speedup=" << required_speedup
              << " profile_calls=128:64:1"
              << " gate=" << (distribution_gate ? "PASS" : "FAIL")
              << '\n';
  }
  test.expect(all_gates,
              "NVFP4 M1 exact-shape clears correctness and performance "
              "gates");
  std::cout << "PERF_NVFP4_M1_EXACT_SHAPE_SELECTED: baseline="
               "current_row_quad_cap64 candidate=exact_shape"
            << " distributions=checkpoint_like:same_bank_stress"
            << " profile_calls=128:64:1 checkpoint_weighted_speedup="
            << weighted_speedups[0U] << " stress_weighted_speedup="
            << weighted_speedups[1U]
            << " gate=" << (all_gates ? "PASS" : "FAIL") << '\n';
}

struct NvFp4M1DownDualKernelResources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

void run_nvfp4_m1_down_dual_fixture(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const bool run_performance,
    const std::string& label) {
  constexpr std::size_t kGuardElements = 16U;
  constexpr std::size_t kGridCap = 64U;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::uint16_t kBaselineGuard = 0xa5a5U;
  constexpr std::uint16_t kCandidateGuard = 0x5a5aU;
  constexpr std::uint16_t kProductionGuard = 0x3c3cU;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  constexpr float kMinimumSpeedup = 1.025F;
  constexpr std::array<NvFp4M1ScaleDistribution, 2U> kDistributions{{
      NvFp4M1ScaleDistribution::kCheckpointLike,
      NvFp4M1ScaleDistribution::kSameBankStress,
  }};

  const std::size_t packed_columns = columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  const std::size_t packed_count = rows * packed_columns;
  const std::size_t scale_count = rows * scale_columns;
  std::vector<std::uint8_t> host_scales(scale_count);
  std::vector<std::uint16_t> host_activation(columns);
  for (std::size_t column = 0U; column < columns; ++column) {
    const int centered =
        static_cast<int>((column * 19U + (column >> 3U) * 7U + 11U) %
                         127U) -
        63;
    host_activation[column] =
        encode_bf16(static_cast<float>(centered) / 256.0F);
  }

  std::array<std::uint8_t, kNvFp4M1DownDualFixturePeriod>
      host_packed_pattern{};
  std::array<bool, 16U> low_nibble_covered{};
  std::array<bool, 16U> high_nibble_covered{};
  for (std::size_t index = 0U; index < host_packed_pattern.size(); ++index) {
    const std::uint8_t low = static_cast<std::uint8_t>(
        (index * 5U + (index >> 2U) * 3U + 1U) & 0x0fU);
    const std::uint8_t high = static_cast<std::uint8_t>(
        (index * 7U + (index >> 3U) * 5U + 9U) & 0x0fU);
    host_packed_pattern[index] =
        static_cast<std::uint8_t>(low | (high << 4U));
    low_nibble_covered[low] = true;
    high_nibble_covered[high] = true;
  }
  test.expect(std::all_of(low_nibble_covered.begin(),
                          low_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " prime fixture covers every E2M1 low nibble");
  test.expect(std::all_of(high_nibble_covered.begin(),
                          high_nibble_covered.end(),
                          [](const bool covered) { return covered; }),
              label + " prime fixture covers every E2M1 high nibble");
  test.expect(kNvFp4M1DownDualFixturePeriod !=
                  kNvFp4M1ExactShapeFixturePeriod &&
                  (kNvFp4M1ExactShapeRowStride %
                   kNvFp4M1DownDualFixturePeriod) != 0U,
              label + " uses a distinct prime grid-stride fixture");

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_storage;
  DeviceBuffer<std::uint16_t> candidate_storage;
  DeviceBuffer<std::uint16_t> production_storage;
  DeviceBuffer<std::uint8_t> unaligned_packed_storage;
  DeviceBuffer<std::uint16_t> unaligned_activation_storage;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(scale_count),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activation.allocate(columns),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(
                       baseline_storage.allocate(rows + 2U * kGuardElements),
                       label + " allocate guarded baseline output");
  ready = ready && test.cuda_ok(
                       candidate_storage.allocate(rows + 2U * kGuardElements),
                       label + " allocate guarded candidate output");
  if (run_performance) {
    ready = ready && test.cuda_ok(
                         production_storage.allocate(
                             rows + 2U * kGuardElements),
                         label + " allocate guarded public output");
    ready = ready && test.cuda_ok(
                         unaligned_packed_storage.allocate(packed_count + 1U),
                         label + " allocate packed+1 fallback fixture");
    ready = ready && test.cuda_ok(
                         unaligned_activation_storage.allocate(columns + 2U),
                         label + " allocate activation+2 fallback fixture");
  }
  const std::size_t seed_bytes =
      std::min(packed_count, host_packed_pattern.size());
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           packed.get(), host_packed_pattern.data(),
                           seed_bytes, cudaMemcpyHostToDevice, stream),
                       label + " initialize prime packed seed");
  for (std::size_t initialized = seed_bytes;
       ready && initialized < packed_count;) {
    const std::size_t copy_bytes =
        std::min(initialized, packed_count - initialized);
    ready = test.cuda_ok(
        cudaMemcpyAsync(packed.get() + initialized, packed.get(), copy_bytes,
                        cudaMemcpyDeviceToDevice, stream),
        label + " expand prime packed fixture");
    initialized += copy_bytes;
  }
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           columns * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  if (run_performance) {
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             unaligned_packed_storage.get() + 1U,
                             packed.get(), packed_count,
                             cudaMemcpyDeviceToDevice, stream),
                         label + " initialize packed+1 fallback fixture");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             unaligned_activation_storage.get() + 2U,
                             activation.get(),
                             columns * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToDevice, stream),
                         label + " initialize activation+2 fallback fixture");
  }
  if (!ready) {
    return;
  }

  std::uint16_t* const baseline_output =
      baseline_storage.get() + kGuardElements;
  std::uint16_t* const candidate_output =
      candidate_storage.get() + kGuardElements;
  std::uint16_t* const production_output =
      run_performance ? production_storage.get() + kGuardElements : nullptr;
  const auto candidate_status =
      [&](const std::uint8_t* const candidate_packed,
          const std::uint8_t* const candidate_scales,
          const std::uint16_t* const candidate_activation,
          const std::size_t candidate_rows,
          const std::size_t candidate_columns,
          std::uint16_t* const candidate_result) noexcept {
        return static_cast<cudaError_t>(q3x::kernels::
            launch_sm87_nvfp4_w4a16_gemv_bf16_down_dual_iteration_test_cuda(
                candidate_packed, candidate_scales, kWeightScale2,
                candidate_activation, candidate_rows, candidate_columns,
                candidate_result, static_cast<void*>(stream)));
      };

  if (!run_performance) {
    ready = test.cuda_ok(
        cudaMemsetAsync(candidate_storage.get(), 0x5a,
                        (rows + 2U * kGuardElements) *
                            sizeof(std::uint16_t),
                        stream),
        label + " poison validation output");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  label + " validation pre-synchronize");
    if (!ready) {
      return;
    }
    test.expect(candidate_status(nullptr, scales.get(), activation.get(),
                                 rows, columns, candidate_output) ==
                    cudaErrorInvalidValue,
                label + " null input fails before enqueue");
    test.expect(candidate_status(packed.get(), scales.get(), activation.get(),
                                 rows - 1U, columns, candidate_output) ==
                    cudaErrorInvalidValue,
                label + " near-miss rows fail before enqueue");
    test.expect(candidate_status(packed.get(), scales.get(), activation.get(),
                                 rows, columns - 16U, candidate_output) ==
                    cudaErrorInvalidValue,
                label + " near-miss columns fail before enqueue");
    test.expect(candidate_status(packed.get() + 1U, scales.get(),
                                 activation.get(), rows, columns,
                                 candidate_output) == cudaErrorInvalidValue,
                label + " unaligned packed input fails before enqueue");
    test.expect(candidate_status(packed.get(), scales.get(),
                                 activation.get() + 1U, rows, columns,
                                 candidate_output) == cudaErrorInvalidValue,
                label + " unaligned activation fails before enqueue");
    test.expect(
        candidate_status(
            packed.get(), scales.get(), activation.get(), rows, columns,
            reinterpret_cast<std::uint16_t*>(packed.get())) ==
            cudaErrorInvalidValue,
        label + " aliased output fails before enqueue");
    test.expect(
        candidate_status(
            packed.get(), scales.get(), activation.get(), rows, columns,
            reinterpret_cast<std::uint16_t*>(
                reinterpret_cast<std::uint8_t*>(candidate_output) + 1U)) ==
            cudaErrorInvalidValue,
        label + " unaligned output fails before enqueue");

    auto* const fake_packed =
        reinterpret_cast<const std::uint8_t*>(0x1'0000'0000ULL);
    auto* const fake_scales =
        reinterpret_cast<const std::uint8_t*>(0x2'0000'0000ULL);
    auto* const fake_activation =
        reinterpret_cast<const std::uint16_t*>(0x3'0000'0000ULL);
    auto* const fake_output =
        reinterpret_cast<std::uint16_t*>(0x4'0000'0000ULL);
    test.expect(candidate_status(fake_packed, fake_scales, fake_activation,
                                 17'408U, 5'120U, fake_output) ==
                    cudaErrorInvalidValue,
                label + " gate/up shape has no candidate instance");
    test.expect(candidate_status(fake_packed, fake_scales, fake_activation,
                                 248'320U, 5'120U, fake_output) ==
                    cudaErrorInvalidValue,
                label + " lm-head shape has no candidate instance");
    test.expect(cudaStreamQuery(stream) == cudaSuccess,
                label + " rejected calls leave the stream empty");
    std::vector<std::uint16_t> validation_output(
        rows + 2U * kGuardElements);
    ready = test.cuda_ok(
        cudaMemcpyAsync(validation_output.data(), candidate_storage.get(),
                        validation_output.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost, stream),
        label + " copy validation poison");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  label + " validation synchronize");
    const bool validation_untouched =
        ready && std::all_of(validation_output.begin(),
                             validation_output.end(),
                             [](const std::uint16_t value) {
                               return value == kCandidateGuard;
                             });
    test.expect(validation_untouched,
                label + " rejected calls do not touch output storage");
    if (!ready) {
      return;
    }
  }

  const auto launch_baseline = [&]() noexcept -> int {
    if (run_performance) {
      // Keep the pre-promotion exact row-quad kernel as a same-cubin baseline;
      // the public ABI now selects the dual-iteration candidate for down.
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_row_quad_exact_shape_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activation.get(),
              rows, columns, baseline_output, static_cast<void*>(stream));
    }
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_grid_cap_test_cuda(
            packed.get(), scales.get(), kWeightScale2, activation.get(), rows,
            columns, baseline_output, kGridCap,
            static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_down_dual_iteration_test_cuda(
            packed.get(), scales.get(), kWeightScale2, activation.get(), rows,
            columns, candidate_output, static_cast<void*>(stream));
  };
  const auto launch_public =
      [&](const std::uint8_t* const public_packed,
          const std::uint16_t* const public_activation,
          std::uint16_t* const public_output) noexcept -> int {
    return q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
        public_packed, scales.get(), kWeightScale2, public_activation, rows,
        columns, public_output, static_cast<void*>(stream));
  };
  const auto launch_scalar =
      [&](const std::uint8_t* const scalar_packed,
          const std::uint16_t* const scalar_activation,
          std::uint16_t* const scalar_output) noexcept -> int {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_gemv_bf16_scalar_test_cuda(
            scalar_packed, scales.get(), kWeightScale2, scalar_activation,
            rows, columns, scalar_output, static_cast<void*>(stream));
  };

  std::vector<std::uint16_t> baseline(rows + 2U * kGuardElements);
  std::vector<std::uint16_t> candidate(rows + 2U * kGuardElements);
  std::vector<std::uint16_t> production(
      run_performance ? rows + 2U * kGuardElements : 0U);
  bool all_distribution_gates = true;
  bool all_production_dispatch_gates = true;
  for (const NvFp4M1ScaleDistribution distribution : kDistributions) {
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_down_dual_scale_distribution(
        host_scales, scale_columns, distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), scale_count,
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(
                             baseline_storage.get(), 0xa5,
                             baseline.size() * sizeof(std::uint16_t), stream),
                         distribution_label + " poison baseline guards");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(
                             candidate_storage.get(), 0x5a,
                             candidate.size() * sizeof(std::uint16_t), stream),
                         distribution_label + " poison candidate guards");
    if (run_performance) {
      ready = ready && test.cuda_ok(
                           cudaMemsetAsync(
                               production_storage.get(), 0x3c,
                               production.size() * sizeof(std::uint16_t),
                               stream),
                           distribution_label + " poison public guards");
    }
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_baseline()),
                         distribution_label + " correctness baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         distribution_label + " correctness candidate");
    if (run_performance) {
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_public(
                               packed.get(), activation.get(),
                               production_output)),
                           distribution_label + " correctness public");
    }
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_storage.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy guarded baseline");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_storage.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy guarded candidate");
    if (run_performance) {
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               production.data(), production_storage.get(),
                               production.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           distribution_label + " copy guarded public");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  distribution_label + " synchronize");
    if (!ready) {
      return;
    }

    std::size_t mismatches = 0U;
    std::size_t public_mismatches = 0U;
    std::size_t nonfinite_outputs = 0U;
    for (std::size_t row = 0U; row < rows; ++row) {
      mismatches +=
          baseline[kGuardElements + row] !=
                  candidate[kGuardElements + row]
              ? 1U
              : 0U;
      nonfinite_outputs +=
          !is_bf16_finite(baseline[kGuardElements + row]) ? 1U : 0U;
      nonfinite_outputs +=
          !is_bf16_finite(candidate[kGuardElements + row]) ? 1U : 0U;
      if (run_performance) {
        public_mismatches +=
            production[kGuardElements + row] !=
                    candidate[kGuardElements + row]
                ? 1U
                : 0U;
        nonfinite_outputs +=
            !is_bf16_finite(production[kGuardElements + row]) ? 1U : 0U;
      }
    }
    bool guards_intact = true;
    for (std::size_t guard = 0U; guard < kGuardElements; ++guard) {
      guards_intact =
          guards_intact && baseline[guard] == kBaselineGuard &&
          baseline[kGuardElements + rows + guard] == kBaselineGuard &&
          candidate[guard] == kCandidateGuard &&
          candidate[kGuardElements + rows + guard] == kCandidateGuard;
      if (run_performance) {
        guards_intact =
            guards_intact && production[guard] == kProductionGuard &&
            production[kGuardElements + rows + guard] == kProductionGuard;
      }
    }
    const bool correctness_gate =
        mismatches == 0U && public_mismatches == 0U &&
        nonfinite_outputs == 0U && guards_intact;
    test.expect(correctness_gate,
                distribution_label +
                    " is bitwise, finite, and preserves both canaries");
    std::cout << "NVFP4_M1_DOWN_DUAL_DIFF: " << distribution_label
              << " rows=" << rows << " columns=" << columns
              << " bf16_mismatches=" << mismatches << '/' << rows
              << " public_vs_direct_bf16_mismatches=" << public_mismatches
              << '/' << (run_performance ? rows : 0U)
              << " output_guards=" << (guards_intact ? "intact" : "BAD")
              << " nonfinite_outputs=" << nonfinite_outputs << '\n';

    if (!run_performance) {
      all_distribution_gates = all_distribution_gates && correctness_gate;
      continue;
    }

    if (distribution == NvFp4M1ScaleDistribution::kCheckpointLike) {
      cudaGraph_t graph = nullptr;
      bool graph_ready = test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          distribution_label + " begin public graph capture");
      if (graph_ready) {
        graph_ready = test.cuda_ok(
            static_cast<cudaError_t>(launch_public(
                packed.get(), activation.get(), production_output)),
            distribution_label + " capture public launch");
        graph_ready =
            test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         distribution_label + " end public graph capture") &&
            graph_ready;
      }
      std::size_t graph_node_count = 0U;
      if (graph_ready) {
        graph_ready = test.cuda_ok(
            cudaGraphGetNodes(graph, nullptr, &graph_node_count),
            distribution_label + " query public graph nodes");
      }
      const bool graph_gate = graph_ready && graph_node_count == 1U;
      test.expect(graph_gate,
                  distribution_label +
                      " public aligned down capture contains one kernel node");
      all_production_dispatch_gates =
          all_production_dispatch_gates && graph_gate;
      if (graph != nullptr) {
        (void)test.cuda_ok(cudaGraphDestroy(graph),
                           distribution_label + " destroy public graph");
      }
      std::cout << "NVFP4_M1_DOWN_DUAL_GRAPH: nodes=" << graph_node_count
                << " required=1 gate="
                << (graph_ready && graph_node_count == 1U ? "PASS" : "FAIL")
                << '\n';

      const auto check_fallback =
          [&](const std::uint8_t* const fallback_packed,
              const std::uint16_t* const fallback_activation,
              const std::string& fallback_label) {
        bool fallback_ready = test.cuda_ok(
            cudaMemsetAsync(
                baseline_storage.get(), 0xa5,
                baseline.size() * sizeof(std::uint16_t), stream),
            fallback_label + " poison scalar-reference guards");
        fallback_ready = fallback_ready && test.cuda_ok(
            cudaMemsetAsync(
                production_storage.get(), 0x3c,
                production.size() * sizeof(std::uint16_t), stream),
            fallback_label + " poison public-fallback guards");
        fallback_ready = fallback_ready && test.cuda_ok(
            static_cast<cudaError_t>(launch_scalar(
                fallback_packed, fallback_activation, baseline_output)),
            fallback_label + " direct scalar reference");
        fallback_ready = fallback_ready && test.cuda_ok(
            static_cast<cudaError_t>(launch_public(
                fallback_packed, fallback_activation, production_output)),
            fallback_label + " public fallback");
        fallback_ready = fallback_ready && test.cuda_ok(
            cudaMemcpyAsync(
                baseline.data(), baseline_storage.get(),
                baseline.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost, stream),
            fallback_label + " copy scalar reference");
        fallback_ready = fallback_ready && test.cuda_ok(
            cudaMemcpyAsync(
                production.data(), production_storage.get(),
                production.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost, stream),
            fallback_label + " copy public fallback");
        fallback_ready = fallback_ready && test.cuda_ok(
            cudaStreamSynchronize(stream), fallback_label + " synchronize");
        if (!fallback_ready) {
          return false;
        }
        std::size_t fallback_mismatches = 0U;
        std::size_t fallback_nonfinite = 0U;
        for (std::size_t row = 0U; row < rows; ++row) {
          const std::uint16_t reference_bits =
              baseline[kGuardElements + row];
          const std::uint16_t public_bits =
              production[kGuardElements + row];
          fallback_mismatches += reference_bits != public_bits ? 1U : 0U;
          fallback_nonfinite += !is_bf16_finite(reference_bits) ? 1U : 0U;
          fallback_nonfinite += !is_bf16_finite(public_bits) ? 1U : 0U;
        }
        bool fallback_guards = true;
        for (std::size_t guard = 0U; guard < kGuardElements; ++guard) {
          fallback_guards =
              fallback_guards && baseline[guard] == kBaselineGuard &&
              baseline[kGuardElements + rows + guard] == kBaselineGuard &&
              production[guard] == kProductionGuard &&
              production[kGuardElements + rows + guard] == kProductionGuard;
        }
        const bool fallback_gate =
            fallback_mismatches == 0U && fallback_nonfinite == 0U &&
            fallback_guards;
        test.expect(fallback_gate,
                    fallback_label + " matches the direct scalar fallback");
        std::cout << "NVFP4_M1_DOWN_DUAL_FALLBACK_DIFF: "
                  << fallback_label
                  << " public_vs_scalar_bf16_mismatches="
                  << fallback_mismatches << '/' << rows
                  << " output_guards="
                  << (fallback_guards ? "intact" : "BAD")
                  << " nonfinite_outputs=" << fallback_nonfinite << '\n';
        return fallback_gate;
      };
      const bool packed_fallback = check_fallback(
          unaligned_packed_storage.get() + 1U, activation.get(),
          distribution_label + " packed+1");
      const bool activation_fallback = check_fallback(
          packed.get(), unaligned_activation_storage.get() + 2U,
          distribution_label + " activation+2");
      test.expect(packed_fallback && activation_fallback,
                  distribution_label +
                      " preserves both production unaligned fallbacks");
      all_production_dispatch_gates =
          all_production_dispatch_gates && packed_fallback &&
          activation_fallback;
    }

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           distribution_label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           distribution_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  distribution_label + " warmup synchronize");
    if (!ready) {
      return;
    }

    constexpr std::size_t kTimedPasses =
        2U * static_cast<std::size_t>(kMeasurementRounds);
    std::array<float, kTimedPasses> baseline_passes{};
    std::array<float, kTimedPasses> candidate_passes{};
    bool timings_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          distribution_label + " round=" + std::to_string(round + 1);
      const float baseline_first = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 1");
      const float candidate_first = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 1");
      const float candidate_second = measure_small_m_tile(
          test, stream, launch_candidate, kMeasuredIterations,
          round_label + " candidate pass 2");
      const float baseline_second = measure_small_m_tile(
          test, stream, launch_baseline, kMeasuredIterations,
          round_label + " baseline pass 2");
      baseline_passes[2U * static_cast<std::size_t>(round)] =
          baseline_first;
      baseline_passes[2U * static_cast<std::size_t>(round) + 1U] =
          baseline_second;
      candidate_passes[2U * static_cast<std::size_t>(round)] =
          candidate_first;
      candidate_passes[2U * static_cast<std::size_t>(round) + 1U] =
          candidate_second;
      const bool round_finite =
          std::isfinite(baseline_first) &&
          std::isfinite(candidate_first) &&
          std::isfinite(candidate_second) &&
          std::isfinite(baseline_second);
      timings_finite = timings_finite && round_finite;
      std::cout << "PERF_NVFP4_M1_DOWN_DUAL_ROUND: "
                << distribution_label << " round=" << round + 1
                << " baseline_pass1_ms=" << baseline_first
                << " candidate_pass1_ms=" << candidate_first
                << " candidate_pass2_ms=" << candidate_second
                << " baseline_pass2_ms=" << baseline_second << '\n';
    }
    const float baseline_milliseconds =
        timings_finite
            ? median_fp8_kv_pair_timing(baseline_passes)
            : std::numeric_limits<float>::quiet_NaN();
    const float candidate_milliseconds =
        timings_finite
            ? median_fp8_kv_pair_timing(candidate_passes)
            : std::numeric_limits<float>::quiet_NaN();
    const float speedup = baseline_milliseconds / candidate_milliseconds;
    const bool performance_gate =
        correctness_gate && timings_finite && std::isfinite(speedup) &&
        baseline_milliseconds > 0.0F && candidate_milliseconds > 0.0F &&
        speedup >= kMinimumSpeedup;
    test.expect(performance_gate,
                distribution_label + " clears down-only 1.025 speedup gate");
    all_distribution_gates =
        all_distribution_gates && performance_gate;
    std::cout << "PERF_NVFP4_M1_DOWN_DUAL: " << distribution_label
              << " baseline_preserved_exact_row_quad_median_ms="
              << baseline_milliseconds
              << " candidate_dual_iteration_median_ms="
              << candidate_milliseconds << " speedup=" << speedup
              << " required_speedup=" << kMinimumSpeedup
              << " gate=" << (performance_gate ? "PASS" : "FAIL") << '\n';
  }

  if (!run_performance) {
    test.expect(all_distribution_gates,
                label + " clears both lightweight correctness gates");
    return;
  }

  // Isolate both E4M3FN NaN encodings in each half of the dual iteration:
  // rows 0/1 use scale column 0 and activation columns 0..15, while rows 2/3
  // use scale column 16 and activation columns 256..271. All later rows stay
  // finite and each classified result retains the preserved baseline class
  // and sign.
  std::fill(host_scales.begin(), host_scales.end(), std::uint8_t{0U});
  host_scales[0U] = 0x7fU;
  host_scales[scale_columns] = 0xffU;
  host_scales[2U * scale_columns + 16U] = 0x7fU;
  host_scales[3U * scale_columns + 16U] = 0xffU;
  std::fill(host_activation.begin(), host_activation.end(),
            encode_bf16(0.0F));
  std::fill_n(host_activation.begin(), 16U, encode_bf16(1.0F));
  std::fill_n(host_activation.begin() + 256U, 16U, encode_bf16(1.0F));
  ready = test.cuda_ok(cudaMemsetAsync(packed.get(), 0x11, packed_count,
                                        stream),
                       label + " initialize isolated-NaN packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(scales.get(), host_scales.data(),
                                       scale_count, cudaMemcpyHostToDevice,
                                       stream),
                       label + " initialize isolated-NaN scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           columns * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize isolated-NaN activation");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(
                           baseline_storage.get(), 0xa5,
                           baseline.size() * sizeof(std::uint16_t), stream),
                       label + " poison isolated-NaN baseline guards");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(
                           candidate_storage.get(), 0x5a,
                           candidate.size() * sizeof(std::uint16_t), stream),
                       label + " poison isolated-NaN candidate guards");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(
                           production_storage.get(), 0x3c,
                           production.size() * sizeof(std::uint16_t), stream),
                       label + " poison isolated-NaN public guards");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_baseline()),
                       label + " isolated-NaN baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_candidate()),
                       label + " isolated-NaN candidate");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(launch_public(
                           packed.get(), activation.get(),
                           production_output)),
                       label + " isolated-NaN public");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           baseline.data(), baseline_storage.get(),
                           baseline.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy isolated-NaN baseline");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           candidate.data(), candidate_storage.get(),
                           candidate.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy isolated-NaN candidate");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           production.data(), production_storage.get(),
                           production.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy isolated-NaN public");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " isolated-NaN synchronize");
  if (!ready) {
    return;
  }
  bool nan_class_and_sign = true;
  std::size_t nan_bitwise_mismatches = 0U;
  std::size_t nan_public_mismatches = 0U;
  for (std::size_t row = 0U; row < 4U; ++row) {
    const std::uint16_t baseline_bits = baseline[kGuardElements + row];
    const std::uint16_t candidate_bits = candidate[kGuardElements + row];
    const std::uint16_t production_bits =
        production[kGuardElements + row];
    nan_bitwise_mismatches += baseline_bits != candidate_bits ? 1U : 0U;
    nan_public_mismatches += production_bits != candidate_bits ? 1U : 0U;
    nan_class_and_sign =
        nan_class_and_sign && is_bf16_nan(baseline_bits) &&
        is_bf16_nan(candidate_bits) && is_bf16_nan(production_bits) &&
        ((baseline_bits ^ candidate_bits) & 0x8000U) == 0U &&
        ((baseline_bits ^ production_bits) & 0x8000U) == 0U;
  }
  std::size_t unexpected_nonfinite = 0U;
  std::size_t finite_mismatches = 0U;
  std::size_t finite_public_mismatches = 0U;
  for (std::size_t row = 4U; row < rows; ++row) {
    const std::uint16_t baseline_bits = baseline[kGuardElements + row];
    const std::uint16_t candidate_bits = candidate[kGuardElements + row];
    const std::uint16_t production_bits =
        production[kGuardElements + row];
    unexpected_nonfinite += !is_bf16_finite(baseline_bits) ? 1U : 0U;
    unexpected_nonfinite += !is_bf16_finite(candidate_bits) ? 1U : 0U;
    unexpected_nonfinite += !is_bf16_finite(production_bits) ? 1U : 0U;
    finite_mismatches += baseline_bits != candidate_bits ? 1U : 0U;
    finite_public_mismatches +=
        production_bits != candidate_bits ? 1U : 0U;
  }
  bool nan_guards_intact = true;
  for (std::size_t guard = 0U; guard < kGuardElements; ++guard) {
    nan_guards_intact =
        nan_guards_intact && baseline[guard] == kBaselineGuard &&
        baseline[kGuardElements + rows + guard] == kBaselineGuard &&
        candidate[guard] == kCandidateGuard &&
        candidate[kGuardElements + rows + guard] == kCandidateGuard &&
        production[guard] == kProductionGuard &&
        production[kGuardElements + rows + guard] == kProductionGuard;
  }
  const bool nan_gate = nan_bitwise_mismatches == 0U &&
                        nan_class_and_sign && unexpected_nonfinite == 0U &&
                        finite_mismatches == 0U &&
                        finite_public_mismatches == 0U &&
                        nan_public_mismatches == 0U && nan_guards_intact;
  test.expect(nan_gate,
              label +
                  " classifies isolated 0x7f/0xff NaNs and preserves guards");
  test.expect(all_distribution_gates,
              label + " clears both full down-only distribution gates");
  test.expect(all_production_dispatch_gates,
              label + " clears graph and unaligned production gates");
  std::cout << "NVFP4_M1_DOWN_DUAL_NAN: classified_outputs=4"
            << " nan_bitwise_mismatches=" << nan_bitwise_mismatches << "/4"
            << " public_vs_direct_nan_mismatches=" << nan_public_mismatches
            << "/4"
            << " class_and_sign="
            << (nan_class_and_sign ? "true" : "false")
            << " finite_mismatches=" << finite_mismatches << '/'
            << (rows - 4U)
            << " public_vs_direct_finite_mismatches="
            << finite_public_mismatches << '/' << (rows - 4U)
            << " unexpected_nonfinite=" << unexpected_nonfinite
            << " output_guards="
            << (nan_guards_intact ? "intact" : "BAD")
            << " gate=" << (nan_gate ? "PASS" : "FAIL") << '\n';
  const bool overall_gate = all_distribution_gates &&
                            all_production_dispatch_gates && nan_gate;
  std::cout << "PERF_NVFP4_M1_DOWN_DUAL_SELECTED: baseline="
               "preserved_exact_row_quad_5120x17408"
            << " candidate=production_down_dual_iteration"
            << " distributions=checkpoint_like:same_bank_stress"
            << " required_each=" << kMinimumSpeedup
            << " all_correct=" << (overall_gate ? "true" : "false")
            << " gate=" << (overall_gate ? "PASS" : "FAIL")
            << '\n';
}

void run_nvfp4_m1_down_dual_iteration_probe(TestContext& test,
                                             cudaStream_t stream) {
  NvFp4M1DownDualKernelResources resources{};
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::
          query_sm87_nvfp4_w4a16_m1_down_dual_iteration_resources_test_cuda(
              &resources.registers_per_thread,
              &resources.static_shared_bytes, &resources.local_bytes,
              &resources.maximum_threads_per_block,
              &resources.active_blocks_per_sm)),
      "query NVFP4 M1 down dual-iteration resources");
  test.expect(
      static_cast<cudaError_t>(q3x::kernels::
          query_sm87_nvfp4_w4a16_m1_down_dual_iteration_resources_test_cuda(
              nullptr, &resources.static_shared_bytes,
              &resources.local_bytes,
              &resources.maximum_threads_per_block,
              &resources.active_blocks_per_sm)) == cudaErrorInvalidValue,
      "NVFP4 M1 down dual resource query rejects null destination");
  const bool resource_gate =
      ready && resources.registers_per_thread <= 64 &&
      resources.static_shared_bytes <= 1'088U &&
      resources.local_bytes == 0U &&
      resources.maximum_threads_per_block >= 256 &&
      resources.active_blocks_per_sm >= 4;
  test.expect(resource_gate,
              "NVFP4 M1 down dual clears 64r/1088B/0local/active4 gate");
  std::cout << "PERF_NVFP4_M1_DOWN_DUAL_RESOURCES: registers_per_thread="
            << resources.registers_per_thread
            << " static_shared_bytes=" << resources.static_shared_bytes
            << " local_bytes=" << resources.local_bytes
            << " maximum_threads_per_block="
            << resources.maximum_threads_per_block
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " gate=" << (resource_gate ? "PASS" : "FAIL") << '\n';
  if (!ready) {
    return;
  }

  run_nvfp4_m1_down_dual_fixture(
      test, stream, 2'048U, 512U, false,
      "NVFP4 M1 down dual lightweight 2048x512");
  if (!nvfp4_m1_down_dual_iteration_performance_enabled()) {
    std::cout
        << "SKIP: NVFP4 M1 down dual-iteration full segment; set "
           "Q3X_RUN_SM87_NVFP4_M1_DOWN_DUAL_ITERATION_PERF=1 to enable\n";
    return;
  }
  run_nvfp4_m1_down_dual_fixture(
      test, stream, 5'120U, 17'408U, true,
      "NVFP4 M1 down dual exact 5120x17408");
}

[[nodiscard]] bool benchmark_nvfp4_m1_scale_codebook_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const float required_speedup,
    const std::string& label) {
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  const int warmup_iterations = rows >= 100'000U ? 4 : 10;
  const int measured_iterations = rows >= 100'000U ? 10 : 40;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  const std::size_t scale_count = rows * scale_columns;
  std::vector<std::uint8_t> host_scales(scale_count);
  std::vector<std::uint16_t> host_activation(
      columns, encode_bf16(1.0F));

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(scale_count),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activation.allocate(columns),
                                label + " allocate activation");
  ready = ready && test.cuda_ok(baseline_output.allocate(rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21, packed_count,
                                       stream),
                       label + " initialize packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation.get(), host_activation.data(),
                           host_activation.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activation");
  if (!ready) {
    return false;
  }

  constexpr std::array<NvFp4M1ScaleDistribution, 2U> kDistributions{{
      NvFp4M1ScaleDistribution::kCheckpointLike,
      NvFp4M1ScaleDistribution::kSameBankStress,
  }};
  bool all_passed = true;
  for (const NvFp4M1ScaleDistribution distribution : kDistributions) {
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_scale_distribution(host_scales, scale_columns,
                                     distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    if (!ready) {
      return false;
    }

    const auto launch_baseline = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_vector_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activation.get(),
              rows, columns, baseline_output.get(),
              static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activation.get(),
              rows, columns, candidate_output.get(),
              static_cast<void*>(stream));
    };

    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         distribution_label + " correctness baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         distribution_label + " correctness candidate");
    std::vector<std::uint16_t> baseline(rows);
    std::vector<std::uint16_t> candidate(rows);
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy baseline output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy candidate output");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " correctness synchronize");
    if (!ready) {
      return false;
    }
    std::size_t mismatches = 0U;
    for (std::size_t row = 0U; row < rows; ++row) {
      mismatches += baseline[row] != candidate[row] ? 1U : 0U;
    }
    const bool bitwise_equal = mismatches == 0U;
    test.expect(bitwise_equal,
                distribution_label +
                    " candidate matches every baseline BF16 bit");

    for (int iteration = 0; iteration < warmup_iterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           distribution_label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           distribution_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " warmup synchronize");
    if (!ready) {
      return false;
    }

    const float baseline_first = measure_small_m_tile(
        test, stream, launch_baseline, measured_iterations,
        distribution_label + " baseline pass 1");
    const float candidate_first = measure_small_m_tile(
        test, stream, launch_candidate, measured_iterations,
        distribution_label + " candidate pass 1");
    const float candidate_second = measure_small_m_tile(
        test, stream, launch_candidate, measured_iterations,
        distribution_label + " candidate pass 2");
    const float baseline_second = measure_small_m_tile(
        test, stream, launch_baseline, measured_iterations,
        distribution_label + " baseline pass 2");
    const bool finite =
        std::isfinite(baseline_first) && std::isfinite(candidate_first) &&
        std::isfinite(candidate_second) && std::isfinite(baseline_second);
    const float baseline_average =
        finite ? (baseline_first + baseline_second) * 0.5F
               : std::numeric_limits<float>::quiet_NaN();
    const float candidate_average =
        finite ? (candidate_first + candidate_second) * 0.5F
               : std::numeric_limits<float>::quiet_NaN();
    const float speedup = baseline_average / candidate_average;
    const bool gate_passed = bitwise_equal && finite &&
                             std::isfinite(speedup) &&
                             speedup >= required_speedup;
    std::cout << "PERF_NVFP4_M1_SCALE_CODEBOOK: " << label
              << " distribution="
              << nvfp4_m1_scale_distribution_name(distribution)
              << " baseline_pass1_ms=" << baseline_first
              << " candidate_pass1_ms=" << candidate_first
              << " candidate_pass2_ms=" << candidate_second
              << " baseline_pass2_ms=" << baseline_second
              << " baseline_average_ms=" << baseline_average
              << " candidate_average_ms=" << candidate_average
              << " speedup=" << speedup
              << " uplift_percent=" << (speedup - 1.0F) * 100.0F
              << " bitwise_mismatches=" << mismatches << '/' << rows
              << " required_speedup=" << required_speedup
              << " gate=" << (gate_passed ? "PASS" : "FAIL") << '\n';
    test.expect(gate_passed,
                distribution_label + " clears its performance gate");
    all_passed = all_passed && gate_passed;
  }
  return all_passed;
}

void run_optional_nvfp4_m1_scale_codebook_performance(
    TestContext& test, cudaStream_t stream) {
  if (!nvfp4_m1_scale_codebook_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M1 scale-codebook performance segment; set "
                 "Q3X_RUN_SM87_NVFP4_M1_SCALE_CODEBOOK_PERF=1 to enable\n";
    return;
  }
  constexpr float kMlpRequiredSpeedup = 1.03F;
  constexpr float kLmHeadRequiredSpeedup = 1.0F;
  const bool gate_up = benchmark_nvfp4_m1_scale_codebook_shape(
      test, stream, 17'408U, 5'120U, kMlpRequiredSpeedup,
      "NVFP4 M1 MLP gate/up 17408x5120");
  const bool down = benchmark_nvfp4_m1_scale_codebook_shape(
      test, stream, 5'120U, 17'408U, kMlpRequiredSpeedup,
      "NVFP4 M1 MLP down 5120x17408");
  const bool lm_head = benchmark_nvfp4_m1_scale_codebook_shape(
      test, stream, 248'320U, 5'120U, kLmHeadRequiredSpeedup,
      "NVFP4 M1 lm_head 248320x5120");
  const bool aggregate = gate_up && down && lm_head;
  std::cout << "PERF_NVFP4_M1_SCALE_CODEBOOK_AGGREGATE: "
               "all_distributions_and_shapes="
            << (aggregate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate,
              "NVFP4 M1 scale codebook clears all shape/distribution gates");
}

[[nodiscard]] bool benchmark_nvfp4_m2_scale_codebook_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 2U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 40;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr float kRequiredSpeedup = 1.03F;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  const std::size_t scale_count = rows * scale_columns;
  std::vector<std::uint8_t> host_scales(scale_count);
  std::vector<std::uint16_t> host_activations(
      kTokens * columns, encode_bf16(1.0F));

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(scale_count),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21, packed_count,
                                       stream),
                       label + " initialize packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return false;
  }

  constexpr std::array<NvFp4M1ScaleDistribution, 2U> kDistributions{{
      NvFp4M1ScaleDistribution::kCheckpointLike,
      NvFp4M1ScaleDistribution::kSameBankStress,
  }};
  bool all_passed = true;
  for (const NvFp4M1ScaleDistribution distribution : kDistributions) {
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_scale_distribution(host_scales, scale_columns,
                                     distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    if (!ready) {
      return false;
    }

    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::kernels::
            launch_sm87_nvfp4_w4a16_small_m2_vector_test_cuda(
                packed.get(), scales.get(), kWeightScale2,
                activations.get(), rows, columns, baseline_output.get(),
                static_cast<void*>(stream))),
        distribution_label + " correctness baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(q3x::kernels::
                             launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda(
                                 packed.get(), scales.get(), kWeightScale2,
                                 activations.get(), rows, columns,
                                 candidate_output.get(),
                                 static_cast<void*>(stream))),
                         distribution_label + " correctness candidate");
    std::vector<std::uint16_t> baseline(kTokens * rows);
    std::vector<std::uint16_t> candidate(kTokens * rows);
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy baseline output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy candidate output");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " correctness synchronize");
    if (!ready) {
      return false;
    }
    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < baseline.size(); ++index) {
      mismatches += baseline[index] != candidate[index] ? 1U : 0U;
    }
    const bool bitwise_equal = mismatches == 0U;
    test.expect(bitwise_equal,
                distribution_label +
                    " candidate matches every baseline BF16 bit");

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(
          static_cast<cudaError_t>(q3x::kernels::
              launch_sm87_nvfp4_w4a16_small_m2_vector_test_cuda(
                  packed.get(), scales.get(), kWeightScale2,
                  activations.get(), rows, columns, baseline_output.get(),
                  static_cast<void*>(stream))),
          distribution_label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(q3x::kernels::
                               launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda(
                                   packed.get(), scales.get(), kWeightScale2,
                                   activations.get(), rows, columns,
                                   candidate_output.get(),
                                   static_cast<void*>(stream))),
                           distribution_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " warmup synchronize");
    if (!ready) {
      return false;
    }

    const float baseline_first = measure_nvfp4_launcher(
        test, stream,
        q3x::kernels::launch_sm87_nvfp4_w4a16_small_m2_vector_test_cuda,
        packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
        columns, baseline_output.get(), kMeasuredIterations,
        distribution_label + " baseline pass 1");
    const float candidate_first = measure_nvfp4_launcher(
        test, stream,
        q3x::kernels::
            launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda,
        packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
        columns, candidate_output.get(), kMeasuredIterations,
        distribution_label + " candidate pass 1");
    const float candidate_second = measure_nvfp4_launcher(
        test, stream,
        q3x::kernels::
            launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda,
        packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
        columns, candidate_output.get(), kMeasuredIterations,
        distribution_label + " candidate pass 2");
    const float baseline_second = measure_nvfp4_launcher(
        test, stream,
        q3x::kernels::launch_sm87_nvfp4_w4a16_small_m2_vector_test_cuda,
        packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
        columns, baseline_output.get(), kMeasuredIterations,
        distribution_label + " baseline pass 2");
    const bool finite =
        std::isfinite(baseline_first) && std::isfinite(candidate_first) &&
        std::isfinite(candidate_second) && std::isfinite(baseline_second);
    const float baseline_average =
        finite ? (baseline_first + baseline_second) * 0.5F
               : std::numeric_limits<float>::quiet_NaN();
    const float candidate_average =
        finite ? (candidate_first + candidate_second) * 0.5F
               : std::numeric_limits<float>::quiet_NaN();
    const float speedup = baseline_average / candidate_average;
    const bool gate_passed = bitwise_equal && finite &&
                             std::isfinite(speedup) &&
                             speedup > kRequiredSpeedup;
    std::cout << "PERF_NVFP4_M2_SCALE_CODEBOOK: " << label
              << " distribution="
              << nvfp4_m1_scale_distribution_name(distribution)
              << " baseline_pass1_ms=" << baseline_first
              << " candidate_pass1_ms=" << candidate_first
              << " candidate_pass2_ms=" << candidate_second
              << " baseline_pass2_ms=" << baseline_second
              << " baseline_average_ms=" << baseline_average
              << " candidate_average_ms=" << candidate_average
              << " speedup=" << speedup
              << " uplift_percent=" << (speedup - 1.0F) * 100.0F
              << " bitwise_mismatches=" << mismatches << '/'
              << baseline.size() << " required_speedup=" << kRequiredSpeedup
              << " gate=" << (gate_passed ? "PASS" : "FAIL") << '\n';
    test.expect(gate_passed,
                distribution_label + " clears the M2 performance gate");
    all_passed = all_passed && gate_passed;
  }
  return all_passed;
}

void run_optional_nvfp4_m2_scale_codebook_performance(
    TestContext& test, cudaStream_t stream) {
  if (!nvfp4_m2_scale_codebook_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M2 scale-codebook performance segment; set "
                 "Q3X_RUN_SM87_NVFP4_M2_SCALE_CODEBOOK_PERF=1 to enable\n";
    return;
  }
  const bool gate_up = benchmark_nvfp4_m2_scale_codebook_shape(
      test, stream, 17'408U, 5'120U,
      "NVFP4 M2 MLP gate/up 17408x5120");
  const bool down = benchmark_nvfp4_m2_scale_codebook_shape(
      test, stream, 5'120U, 17'408U,
      "NVFP4 M2 MLP down 5120x17408");
  const bool aggregate = gate_up && down;
  std::cout << "PERF_NVFP4_M2_SCALE_CODEBOOK_AGGREGATE: "
               "all_distributions_and_shapes="
            << (aggregate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate,
              "NVFP4 M2 scale codebook clears all shape/distribution gates");
}

struct NvFp4M2RowPairDistributionMeasurement {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
};

struct NvFp4M2RowPairMeasurement {
  std::array<NvFp4M2RowPairDistributionMeasurement, 2U> distributions{};
};

[[nodiscard]] NvFp4M2RowPairMeasurement
benchmark_nvfp4_m2_row_pair_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 2U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 40;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::array<NvFp4M1ScaleDistribution, 2U> kDistributions{{
      NvFp4M1ScaleDistribution::kCheckpointLike,
      NvFp4M1ScaleDistribution::kSameBankStress,
  }};
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  std::vector<std::uint8_t> host_scales(rows * scale_columns);
  std::vector<std::uint16_t> host_activations(kTokens * columns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered =
          static_cast<int>((column * 17U + token * 29U + 5U) % 127U) - 63;
      host_activations[token * columns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
  }

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(host_scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21, packed_count,
                                       stream),
                       label + " initialize packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return {};
  }

  NvFp4M2RowPairMeasurement measurement;
  for (std::size_t distribution_index = 0U;
       distribution_index < kDistributions.size(); ++distribution_index) {
    const NvFp4M1ScaleDistribution distribution =
        kDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_scale_distribution(host_scales, scale_columns,
                                     distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    if (!ready) {
      return measurement;
    }

    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::kernels::
            launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda(
                packed.get(), scales.get(), kWeightScale2, activations.get(),
                rows, columns, baseline_output.get(),
                static_cast<void*>(stream))),
        distribution_label + " correctness preserved single-row baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(q3x::kernels::
                             launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_test_cuda(
                                 packed.get(), scales.get(), kWeightScale2,
                                 activations.get(), rows, columns,
                                 candidate_output.get(),
                                 static_cast<void*>(stream))),
                         distribution_label +
                             " correctness row-pair candidate");
    std::vector<std::uint16_t> baseline(kTokens * rows);
    std::vector<std::uint16_t> candidate(kTokens * rows);
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy baseline output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy candidate output");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " correctness synchronize");
    if (!ready) {
      return measurement;
    }
    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < baseline.size(); ++index) {
      mismatches += baseline[index] != candidate[index] ? 1U : 0U;
    }
    const bool bitwise_equal = mismatches == 0U;
    measurement.distributions[distribution_index].bitwise_equal =
        bitwise_equal;
    test.expect(bitwise_equal,
                distribution_label +
                    " row-pair candidate matches every production BF16 bit");

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(
          static_cast<cudaError_t>(q3x::kernels::
              launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda(
                  packed.get(), scales.get(), kWeightScale2,
                  activations.get(), rows, columns, baseline_output.get(),
                  static_cast<void*>(stream))),
          distribution_label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(q3x::kernels::
                               launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_test_cuda(
                                   packed.get(), scales.get(), kWeightScale2,
                                   activations.get(), rows, columns,
                                   candidate_output.get(),
                                   static_cast<void*>(stream))),
                           distribution_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " warmup synchronize");
    if (!ready) {
      return measurement;
    }

    double baseline_total = 0.0;
    double candidate_total = 0.0;
    bool all_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          distribution_label + " round=" + std::to_string(round + 1);
      const float baseline_first = measure_nvfp4_launcher(
          test, stream,
          q3x::kernels::
              launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda,
          packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
          columns, baseline_output.get(), kMeasuredIterations,
          round_label + " baseline pass 1");
      const float candidate_first = measure_nvfp4_launcher(
          test, stream,
          q3x::kernels::
              launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_test_cuda,
          packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
          columns, candidate_output.get(), kMeasuredIterations,
          round_label + " candidate pass 1");
      const float candidate_second = measure_nvfp4_launcher(
          test, stream,
          q3x::kernels::
              launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_test_cuda,
          packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
          columns, candidate_output.get(), kMeasuredIterations,
          round_label + " candidate pass 2");
      const float baseline_second = measure_nvfp4_launcher(
          test, stream,
          q3x::kernels::
              launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda,
          packed.get(), scales.get(), kWeightScale2, activations.get(), rows,
          columns, baseline_output.get(), kMeasuredIterations,
          round_label + " baseline pass 2");
      const bool round_finite =
          std::isfinite(baseline_first) &&
          std::isfinite(candidate_first) &&
          std::isfinite(candidate_second) &&
          std::isfinite(baseline_second);
      all_finite = all_finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_NVFP4_M2_ROW_PAIR_ROUND: " << label
                << " distribution="
                << nvfp4_m1_scale_distribution_name(distribution)
                << " round=" << round + 1
                << " baseline_pass1_ms=" << baseline_first
                << " candidate_pass1_ms=" << candidate_first
                << " candidate_pass2_ms=" << candidate_second
                << " baseline_pass2_ms=" << baseline_second << '\n';
    }
    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    NvFp4M2RowPairDistributionMeasurement& distribution_measurement =
        measurement.distributions[distribution_index];
    distribution_measurement.baseline_milliseconds =
        all_finite
            ? static_cast<float>(baseline_total / kTimedPasses)
            : std::numeric_limits<float>::quiet_NaN();
    distribution_measurement.candidate_milliseconds =
        all_finite
            ? static_cast<float>(candidate_total / kTimedPasses)
            : std::numeric_limits<float>::quiet_NaN();
    const float speedup = distribution_measurement.baseline_milliseconds /
                          distribution_measurement.candidate_milliseconds;
    std::cout << "PERF_NVFP4_M2_ROW_PAIR: " << label << " distribution="
              << nvfp4_m1_scale_distribution_name(distribution)
              << " baseline_scale_codebook_ms="
              << distribution_measurement.baseline_milliseconds
              << " candidate_row_pair_ms="
              << distribution_measurement.candidate_milliseconds
              << " speedup=" << speedup << " bitwise_mismatches="
              << mismatches << '/' << baseline.size() << '\n';
  }
  return measurement;
}

void run_optional_nvfp4_m2_row_pair_performance(TestContext& test,
                                                 cudaStream_t stream) {
  if (!nvfp4_m2_row_pair_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M2 scale-codebook row-pair performance "
                 "segment; set Q3X_RUN_SM87_NVFP4_M2_ROW_PAIR_PERF=1 to "
                 "enable\n";
    return;
  }
  constexpr float kMinimumPerShapeDistributionSpeedup = 1.02F;
  constexpr float kMinimumCheckpointWeightedSpeedup = 1.05F;
  struct Shape {
    std::size_t rows;
    std::size_t columns;
    std::size_t checkpoint_calls;
    const char* label;
  };
  constexpr std::array<Shape, 2U> kShapes{{
      {17'408U, 5'120U, 128U, "NVFP4 M2 row-pair 17408x5120"},
      {5'120U, 17'408U, 64U, "NVFP4 M2 row-pair 5120x17408"},
  }};
  constexpr std::array<NvFp4M1ScaleDistribution, 2U> kDistributions{{
      NvFp4M1ScaleDistribution::kCheckpointLike,
      NvFp4M1ScaleDistribution::kSameBankStress,
  }};
  std::array<NvFp4M2RowPairMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_nvfp4_m2_row_pair_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }

  bool all_shape_distributions_pass = true;
  double checkpoint_weighted_baseline = 0.0;
  double checkpoint_weighted_candidate = 0.0;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    for (std::size_t distribution_index = 0U;
         distribution_index < kDistributions.size(); ++distribution_index) {
      const NvFp4M2RowPairDistributionMeasurement& measurement =
          measurements[shape_index].distributions[distribution_index];
      const float speedup = measurement.baseline_milliseconds /
                            measurement.candidate_milliseconds;
      const bool finite =
          std::isfinite(measurement.baseline_milliseconds) &&
          std::isfinite(measurement.candidate_milliseconds) &&
          std::isfinite(speedup);
      const bool gate = measurement.bitwise_equal && finite &&
                        speedup >= kMinimumPerShapeDistributionSpeedup;
      all_shape_distributions_pass =
          all_shape_distributions_pass && gate;
      test.expect(
          gate,
          std::string(kShapes[shape_index].label) + " " +
              nvfp4_m1_scale_distribution_name(
                  kDistributions[distribution_index]) +
              " clears the M2 row-pair performance gate");
      std::cout << "PERF_NVFP4_M2_ROW_PAIR_VALIDATION: "
                << kShapes[shape_index].label << " distribution="
                << nvfp4_m1_scale_distribution_name(
                       kDistributions[distribution_index])
                << " baseline_scale_codebook_ms="
                << measurement.baseline_milliseconds
                << " candidate_row_pair_ms="
                << measurement.candidate_milliseconds
                << " speedup=" << speedup
                << " required_speedup="
                << kMinimumPerShapeDistributionSpeedup
                << " bitwise="
                << (measurement.bitwise_equal ? "true" : "false")
                << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
    }
    const NvFp4M2RowPairDistributionMeasurement& checkpoint =
        measurements[shape_index].distributions[0U];
    checkpoint_weighted_baseline +=
        static_cast<double>(kShapes[shape_index].checkpoint_calls) *
        checkpoint.baseline_milliseconds;
    checkpoint_weighted_candidate +=
        static_cast<double>(kShapes[shape_index].checkpoint_calls) *
        checkpoint.candidate_milliseconds;
  }
  const double checkpoint_weighted_speedup =
      checkpoint_weighted_baseline / checkpoint_weighted_candidate;
  const bool aggregate_gate =
      all_shape_distributions_pass &&
      std::isfinite(checkpoint_weighted_speedup) &&
      checkpoint_weighted_speedup >= kMinimumCheckpointWeightedSpeedup;
  std::cout << "PERF_NVFP4_M2_ROW_PAIR_AGGREGATE: "
            << "checkpoint_weighted_baseline_ms="
            << checkpoint_weighted_baseline
            << " checkpoint_weighted_candidate_ms="
            << checkpoint_weighted_candidate
            << " speedup=" << checkpoint_weighted_speedup
            << " required_speedup=" << kMinimumCheckpointWeightedSpeedup
            << " profile_calls=128:64 all_shape_distributions="
            << (all_shape_distributions_pass ? "PASS" : "FAIL")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "NVFP4 M2 row-pair clears every production gate");
}

constexpr std::array<std::size_t, 3U> kNvFp4M2RowQuadPairGridCaps{{
    64U,
    80U,
    96U,
}};

constexpr std::array<std::size_t, 3U> kNvFp4M2RowQuadGridCaps{{
    48U,
    64U,
    80U,
}};

constexpr std::size_t kNvFp4M2NaturalGridCap = 65'535U;

constexpr std::array<NvFp4M1ScaleDistribution, 2U>
    kNvFp4M2RowQuadScaleDistributions{{
        NvFp4M1ScaleDistribution::kCheckpointLike,
        NvFp4M1ScaleDistribution::kSameBankStress,
    }};

struct NvFp4M2RowQuadComparison {
  float natural_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool bitwise_equal = false;
};

struct NvFp4M2RowQuadDistributionMeasurement {
  std::array<NvFp4M2RowQuadComparison,
             kNvFp4M2RowQuadPairGridCaps.size()>
      row_pair{};
  std::array<NvFp4M2RowQuadComparison, kNvFp4M2RowQuadGridCaps.size()>
      row_quad{};
};

struct NvFp4M2RowQuadMeasurement {
  std::array<NvFp4M2RowQuadDistributionMeasurement,
             kNvFp4M2RowQuadScaleDistributions.size()>
      distributions{};
};

struct NvFp4M2RowQuadShape {
  std::size_t rows;
  std::size_t columns;
  std::size_t checkpoint_calls;
  const char* label;
};

void initialize_nvfp4_m2_row_quad_mixed_weights(
    TestContext& test, cudaStream_t stream, DeviceBuffer<std::uint8_t>& packed,
    const std::size_t packed_count, const std::string& label,
    bool& ready) {
  constexpr std::size_t kPatternBytes = 256U;
  std::array<std::uint8_t, kPatternBytes> pattern{};
  std::array<bool, 16U> low_nibble_covered{};
  std::array<bool, 16U> high_nibble_covered{};
  for (std::size_t index = 0U; index < pattern.size(); ++index) {
    const std::uint8_t low = static_cast<std::uint8_t>(
        (index * 3U + (index >> 3U) + 1U) & 0x0fU);
    const std::uint8_t high = static_cast<std::uint8_t>(
        (index * 5U + (index >> 4U) * 7U + 3U) & 0x0fU);
    pattern[index] = static_cast<std::uint8_t>(low | (high << 4U));
    low_nibble_covered[low] = true;
    high_nibble_covered[high] = true;
  }
  const bool all_low_codes =
      std::all_of(low_nibble_covered.begin(), low_nibble_covered.end(),
                  [](const bool covered) { return covered; });
  const bool all_high_codes =
      std::all_of(high_nibble_covered.begin(), high_nibble_covered.end(),
                  [](const bool covered) { return covered; });
  test.expect(all_low_codes, label + " covers every E2M1 low-nibble code");
  test.expect(all_high_codes,
              label + " covers every E2M1 high-nibble code");
  if (!ready || packed_count == 0U) {
    return;
  }
  const std::size_t seed_bytes = std::min(packed_count, pattern.size());
  ready = test.cuda_ok(
      cudaMemcpyAsync(packed.get(), pattern.data(), seed_bytes,
                      cudaMemcpyHostToDevice, stream),
      label + " initialize mixed all-nibble weight seed");
  for (std::size_t initialized = seed_bytes;
       ready && initialized < packed_count;) {
    const std::size_t copy_bytes =
        std::min(initialized, packed_count - initialized);
    ready = test.cuda_ok(
        cudaMemcpyAsync(packed.get() + initialized, packed.get(), copy_bytes,
                        cudaMemcpyDeviceToDevice, stream),
        label + " expand mixed all-nibble weights");
    initialized += copy_bytes;
  }
}

void run_nvfp4_m2_row_quad_tail_correctness(TestContext& test,
                                             cudaStream_t stream) {
  constexpr std::size_t kTokens = 2U;
  constexpr std::size_t kColumns = 256U;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr std::array<std::size_t, 3U> kRows{{257U, 258U, 259U}};
  const std::size_t max_rows = kRows.back();
  const std::size_t packed_columns = kColumns / 2U;
  const std::size_t scale_columns = kColumns / 16U;
  const std::size_t packed_count = max_rows * packed_columns;
  std::vector<std::uint8_t> host_scales(max_rows * scale_columns);
  fill_nvfp4_m1_scale_distribution(
      host_scales, scale_columns,
      NvFp4M1ScaleDistribution::kCheckpointLike);
  std::vector<std::uint16_t> host_activations(kTokens * kColumns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const int centered = static_cast<int>(
                               (column * 19U + token * 37U + 11U) % 127U) -
                           63;
      host_activations[token * kColumns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
  }
  test.expect(host_activations[0U] != host_activations[kColumns],
              "NVFP4 M2 row-quad tail uses distinct token activations");

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  const std::string label = "NVFP4 M2 row-quad tails rows257/258/259";
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(host_scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * max_rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * max_rows),
                                label + " allocate candidate output");
  initialize_nvfp4_m2_row_quad_mixed_weights(
      test, stream, packed, packed_count, label, ready);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(scales.get(), host_scales.data(),
                                       host_scales.size(),
                                       cudaMemcpyHostToDevice, stream),
                       label + " initialize block scales");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize two token activations");
  if (!ready) {
    return;
  }

  std::vector<std::uint16_t> baseline(kTokens * max_rows);
  std::vector<std::uint16_t> candidate(kTokens * max_rows);
  for (const std::size_t rows : kRows) {
    const std::string row_label = label + " rows=" + std::to_string(rows);
    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::kernels::
            launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda(
                packed.get(), scales.get(), kWeightScale2, activations.get(),
                rows, kColumns, baseline_output.get(),
                static_cast<void*>(stream))),
        row_label + " launch single-row baseline");
    for (const std::size_t cap : kNvFp4M2RowQuadGridCaps) {
      if (!ready) {
        return;
      }
      const std::string cap_label =
          row_label + " quad_cap=" + std::to_string(cap);
      ready = test.cuda_ok(
          cudaMemsetAsync(candidate_output.get(), 0xa5,
                          kTokens * max_rows * sizeof(std::uint16_t), stream),
          cap_label + " poison candidate output");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(q3x::kernels::
                               launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_quad_grid_cap_test_cuda(
                                   packed.get(), scales.get(), kWeightScale2,
                                   activations.get(), rows, kColumns,
                                   candidate_output.get(), cap,
                                   static_cast<void*>(stream))),
                           cap_label + " launch tail-safe row-quad");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               baseline.data(), baseline_output.get(),
                               kTokens * rows * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy baseline output");
      ready = ready && test.cuda_ok(
                           cudaMemcpyAsync(
                               candidate.data(), candidate_output.get(),
                               kTokens * rows * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
                           cap_label + " copy candidate output");
      ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                    cap_label + " synchronize");
      if (!ready) {
        return;
      }
      std::size_t mismatches = 0U;
      for (std::size_t index = 0U; index < kTokens * rows; ++index) {
        mismatches += baseline[index] != candidate[index] ? 1U : 0U;
      }
      std::cout << "NVFP4_M2_ROW_QUAD_TAIL_DIFF: rows=" << rows
                << " columns=" << kColumns << " quad_cap=" << cap
                << " mismatches=" << mismatches << '/' << kTokens * rows
                << '\n';
      test.expect(mismatches == 0U,
                  cap_label + " matches every single-row baseline BF16 bit");
    }
  }
}

[[nodiscard]] NvFp4M2RowQuadMeasurement
benchmark_nvfp4_m2_row_quad_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 2U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 24;
  constexpr int kMeasurementRounds = 3;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  std::vector<std::uint8_t> host_scales(rows * scale_columns);
  std::vector<std::uint16_t> host_activations(kTokens * columns);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered = static_cast<int>(
                               (column * 17U + token * 29U + 5U) % 127U) -
                           63;
      host_activations[token * columns + column] =
          encode_bf16(static_cast<float>(centered) / 256.0F);
    }
  }
  test.expect(host_activations[0U] != host_activations[columns],
              label + " uses distinct token activations");

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> natural_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(host_scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(natural_output.allocate(kTokens * rows),
                                label + " allocate production output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * rows),
                                label + " allocate candidate output");
  initialize_nvfp4_m2_row_quad_mixed_weights(
      test, stream, packed, packed_count, label, ready);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize two token activations");
  NvFp4M2RowQuadMeasurement measurement;
  if (!ready) {
    return measurement;
  }

  std::vector<std::uint16_t> natural(kTokens * rows);
  std::vector<std::uint16_t> candidate(kTokens * rows);
  for (std::size_t distribution_index = 0U;
       distribution_index < kNvFp4M2RowQuadScaleDistributions.size();
       ++distribution_index) {
    const NvFp4M1ScaleDistribution distribution =
        kNvFp4M2RowQuadScaleDistributions[distribution_index];
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_scale_distribution(host_scales, scale_columns,
                                     distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    if (!ready) {
      return measurement;
    }

    const auto launch_natural = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_grid_cap_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activations.get(),
              rows, columns, natural_output.get(),
              kNvFp4M2NaturalGridCap, static_cast<void*>(stream));
    };
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_natural()),
                         distribution_label +
                             " correctness production natural");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             natural.data(), natural_output.get(),
                             natural.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy production output");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  distribution_label +
                                      " production synchronize");
    if (!ready) {
      return measurement;
    }

    const auto benchmark_candidate =
        [&](const bool row_quad, const std::size_t cap,
            NvFp4M2RowQuadComparison& comparison) {
          const char* const kind = row_quad ? "row_quad" : "row_pair";
          const std::string cap_label =
              distribution_label + " " + kind + "_cap=" +
              std::to_string(cap);
          const auto launch_candidate = [&]() noexcept -> int {
            if (row_quad) {
              return q3x::kernels::
                  launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_quad_grid_cap_test_cuda(
                      packed.get(), scales.get(), kWeightScale2,
                      activations.get(), rows, columns, candidate_output.get(),
                      cap, static_cast<void*>(stream));
            }
            return q3x::kernels::
                launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_grid_cap_test_cuda(
                    packed.get(), scales.get(), kWeightScale2,
                    activations.get(), rows, columns, candidate_output.get(),
                    cap, static_cast<void*>(stream));
          };

          ready = test.cuda_ok(
              static_cast<cudaError_t>(launch_candidate()),
              cap_label + " correctness candidate");
          ready = ready && test.cuda_ok(
                               cudaMemcpyAsync(
                                   candidate.data(), candidate_output.get(),
                                   candidate.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream),
                               cap_label + " copy candidate output");
          ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                        cap_label +
                                            " correctness synchronize");
          if (!ready) {
            return;
          }
          std::size_t mismatches = 0U;
          for (std::size_t index = 0U; index < natural.size(); ++index) {
            mismatches += natural[index] != candidate[index] ? 1U : 0U;
          }
          comparison.bitwise_equal = mismatches == 0U;
          std::cout << "NVFP4_M2_ROW_QUAD_DIFF: " << label
                    << " distribution="
                    << nvfp4_m1_scale_distribution_name(distribution)
                    << " candidate_kind=" << kind << " candidate_cap=" << cap
                    << " mismatches=" << mismatches << '/' << natural.size()
                    << '\n';
          test.expect(comparison.bitwise_equal,
                      cap_label +
                          " matches every production natural BF16 bit");

          for (int iteration = 0; iteration < kWarmupIterations && ready;
               ++iteration) {
            ready = test.cuda_ok(static_cast<cudaError_t>(launch_natural()),
                                 cap_label + " natural warmup");
            ready = ready && test.cuda_ok(
                                 static_cast<cudaError_t>(launch_candidate()),
                                 cap_label + " candidate warmup");
          }
          ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                        cap_label + " warmup synchronize");
          if (!ready) {
            return;
          }

          double natural_total = 0.0;
          double candidate_total = 0.0;
          bool all_finite = true;
          for (int round = 0; round < kMeasurementRounds; ++round) {
            const std::string round_label =
                cap_label + " round=" + std::to_string(round + 1);
            const float natural_first = measure_small_m_tile(
                test, stream, launch_natural, kMeasuredIterations,
                round_label + " natural pass 1");
            const float candidate_first = measure_small_m_tile(
                test, stream, launch_candidate, kMeasuredIterations,
                round_label + " candidate pass 1");
            const float candidate_second = measure_small_m_tile(
                test, stream, launch_candidate, kMeasuredIterations,
                round_label + " candidate pass 2");
            const float natural_second = measure_small_m_tile(
                test, stream, launch_natural, kMeasuredIterations,
                round_label + " natural pass 2");
            const bool round_finite =
                std::isfinite(natural_first) &&
                std::isfinite(candidate_first) &&
                std::isfinite(candidate_second) &&
                std::isfinite(natural_second);
            all_finite = all_finite && round_finite;
            if (round_finite) {
              natural_total += natural_first + natural_second;
              candidate_total += candidate_first + candidate_second;
            }
            std::cout << "PERF_NVFP4_M2_ROW_QUAD_ROUND: " << label
                      << " distribution="
                      << nvfp4_m1_scale_distribution_name(distribution)
                      << " candidate_kind=" << kind
                      << " candidate_cap=" << cap
                      << " measured_iterations=" << kMeasuredIterations
                      << " round=" << round + 1
                      << " natural_pass1_ms=" << natural_first
                      << " candidate_pass1_ms=" << candidate_first
                      << " candidate_pass2_ms=" << candidate_second
                      << " natural_pass2_ms=" << natural_second << '\n';
          }
          constexpr double kTimedPasses =
              2.0 * static_cast<double>(kMeasurementRounds);
          comparison.natural_milliseconds =
              all_finite
                  ? static_cast<float>(natural_total / kTimedPasses)
                  : std::numeric_limits<float>::quiet_NaN();
          comparison.candidate_milliseconds =
              all_finite
                  ? static_cast<float>(candidate_total / kTimedPasses)
                  : std::numeric_limits<float>::quiet_NaN();
          const float speedup = comparison.natural_milliseconds /
                                comparison.candidate_milliseconds;
          std::cout << "PERF_NVFP4_M2_ROW_QUAD_CELL: " << label
                    << " distribution="
                    << nvfp4_m1_scale_distribution_name(distribution)
                    << " candidate_kind=" << kind << " candidate_cap=" << cap
                    << " production_natural_ms="
                    << comparison.natural_milliseconds
                    << " candidate_ms=" << comparison.candidate_milliseconds
                    << " speedup=" << speedup
                    << " bitwise="
                    << (comparison.bitwise_equal ? "true" : "false")
                    << '\n';
        };

    NvFp4M2RowQuadDistributionMeasurement& distribution_measurement =
        measurement.distributions[distribution_index];
    for (std::size_t cap_index = 0U;
         cap_index < kNvFp4M2RowQuadPairGridCaps.size(); ++cap_index) {
      benchmark_candidate(false, kNvFp4M2RowQuadPairGridCaps[cap_index],
                          distribution_measurement.row_pair[cap_index]);
      if (!ready) {
        return measurement;
      }
    }
    for (std::size_t cap_index = 0U;
         cap_index < kNvFp4M2RowQuadGridCaps.size(); ++cap_index) {
      benchmark_candidate(true, kNvFp4M2RowQuadGridCaps[cap_index],
                          distribution_measurement.row_quad[cap_index]);
      if (!ready) {
        return measurement;
      }
    }
  }
  return measurement;
}

void run_optional_nvfp4_m2_row_quad_performance(TestContext& test,
                                                 cudaStream_t stream) {
  if (!nvfp4_m2_row_quad_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M2 row-quad cap/correctness performance "
                 "segment; set Q3X_RUN_SM87_NVFP4_M2_ROW_QUAD_PERF=1 to "
                 "enable\n";
    return;
  }
  constexpr float kMinimumPerCellSpeedup = 1.02F;
  constexpr double kMinimumCheckpointWeightedSpeedup = 1.05;
  constexpr std::array<NvFp4M2RowQuadShape, 2U> kShapes{{
      {17'408U, 5'120U, 128U, "NVFP4 M2 row-quad gate/up 17408x5120"},
      {5'120U, 17'408U, 64U, "NVFP4 M2 row-quad down 5120x17408"},
  }};

  run_nvfp4_m2_row_quad_tail_correctness(test, stream);
  std::array<NvFp4M2RowQuadMeasurement, kShapes.size()> measurements{};
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    measurements[shape_index] = benchmark_nvfp4_m2_row_quad_shape(
        test, stream, kShapes[shape_index].rows, kShapes[shape_index].columns,
        kShapes[shape_index].label);
  }

  const auto checkpoint_weighted_speedup =
      [&](const bool row_quad, const std::size_t cap_index) {
        double natural_total = 0.0;
        double candidate_total = 0.0;
        for (std::size_t shape_index = 0U; shape_index < kShapes.size();
             ++shape_index) {
          const NvFp4M2RowQuadDistributionMeasurement& checkpoint =
              measurements[shape_index].distributions[0U];
          const NvFp4M2RowQuadComparison& comparison =
              row_quad ? checkpoint.row_quad[cap_index]
                       : checkpoint.row_pair[cap_index];
          natural_total +=
              static_cast<double>(kShapes[shape_index].checkpoint_calls) *
              comparison.natural_milliseconds;
          candidate_total +=
              static_cast<double>(kShapes[shape_index].checkpoint_calls) *
              comparison.candidate_milliseconds;
        }
        return natural_total / candidate_total;
      };

  std::size_t best_pair_index = 0U;
  double best_pair_weighted_speedup = -std::numeric_limits<double>::infinity();
  for (std::size_t cap_index = 0U;
       cap_index < kNvFp4M2RowQuadPairGridCaps.size(); ++cap_index) {
    const double speedup = checkpoint_weighted_speedup(false, cap_index);
    std::cout << "PERF_NVFP4_M2_ROW_QUAD_PAIR_CAP: cap="
              << kNvFp4M2RowQuadPairGridCaps[cap_index]
              << " checkpoint_weighted_vs_natural=" << speedup << '\n';
    if (std::isfinite(speedup) && speedup > best_pair_weighted_speedup) {
      best_pair_weighted_speedup = speedup;
      best_pair_index = cap_index;
    }
  }

  std::size_t best_quad_index = 0U;
  double best_quad_weighted_speedup = -std::numeric_limits<double>::infinity();
  for (std::size_t cap_index = 0U;
       cap_index < kNvFp4M2RowQuadGridCaps.size(); ++cap_index) {
    const double speedup = checkpoint_weighted_speedup(true, cap_index);
    std::cout << "PERF_NVFP4_M2_ROW_QUAD_CAP: cap="
              << kNvFp4M2RowQuadGridCaps[cap_index]
              << " checkpoint_weighted_vs_natural=" << speedup << '\n';
    if (std::isfinite(speedup) && speedup > best_quad_weighted_speedup) {
      best_quad_weighted_speedup = speedup;
      best_quad_index = cap_index;
    }
  }

  const std::size_t production_quad_cap =
      q3x::kernels::sm87_nvfp4_m2_row_quad_maximum_blocks_test();
  test.expect(production_quad_cap == 64U,
              "NVFP4 M2 selected production row-quad cap is frozen at 64");
  const auto production_quad_iterator =
      std::find(kNvFp4M2RowQuadGridCaps.begin(),
                kNvFp4M2RowQuadGridCaps.end(), production_quad_cap);
  const bool production_cap_measured =
      production_quad_iterator != kNvFp4M2RowQuadGridCaps.end();
  test.expect(production_cap_measured,
              "NVFP4 M2 production row-quad cap is present in the sweep");
  if (!production_cap_measured) {
    return;
  }
  const std::size_t production_quad_index = static_cast<std::size_t>(
      production_quad_iterator - kNvFp4M2RowQuadGridCaps.begin());
  const bool production_matches_dynamic_winner =
      production_quad_index == best_quad_index;
  test.expect(production_matches_dynamic_winner,
              "NVFP4 M2 production cap64 remains the measured row-quad "
              "winner");

  bool all_candidate_bits_equal = true;
  bool all_winner_cells_pass = true;
  double checkpoint_natural_total = 0.0;
  double checkpoint_pair_total = 0.0;
  double checkpoint_quad_total = 0.0;
  for (std::size_t shape_index = 0U; shape_index < kShapes.size();
       ++shape_index) {
    for (std::size_t distribution_index = 0U;
         distribution_index < kNvFp4M2RowQuadScaleDistributions.size();
         ++distribution_index) {
      const NvFp4M2RowQuadDistributionMeasurement& distribution =
          measurements[shape_index].distributions[distribution_index];
      for (const NvFp4M2RowQuadComparison& comparison :
           distribution.row_pair) {
        all_candidate_bits_equal =
            all_candidate_bits_equal && comparison.bitwise_equal;
      }
      for (const NvFp4M2RowQuadComparison& comparison :
           distribution.row_quad) {
        all_candidate_bits_equal =
            all_candidate_bits_equal && comparison.bitwise_equal;
      }

      const NvFp4M2RowQuadComparison& best_pair =
          distribution.row_pair[best_pair_index];
      const NvFp4M2RowQuadComparison& winner =
          distribution.row_quad[production_quad_index];
      const NvFp4M2RowQuadComparison& quad64 =
          distribution.row_quad[production_quad_index];
      const double pair_vs_natural =
          static_cast<double>(best_pair.natural_milliseconds) /
          best_pair.candidate_milliseconds;
      const double winner_vs_natural =
          static_cast<double>(winner.natural_milliseconds) /
          winner.candidate_milliseconds;
      const double winner_vs_pair =
          static_cast<double>(best_pair.candidate_milliseconds) /
          winner.candidate_milliseconds;
      const double quad64_vs_natural =
          static_cast<double>(quad64.natural_milliseconds) /
          quad64.candidate_milliseconds;
      const double quad64_vs_pair =
          static_cast<double>(best_pair.candidate_milliseconds) /
          quad64.candidate_milliseconds;
      const bool finite = std::isfinite(pair_vs_natural) &&
                          std::isfinite(winner_vs_natural) &&
                          std::isfinite(winner_vs_pair);
      const bool cell_gate =
          winner.bitwise_equal && best_pair.bitwise_equal && finite &&
          winner_vs_natural >= kMinimumPerCellSpeedup &&
          winner_vs_pair >= kMinimumPerCellSpeedup;
      all_winner_cells_pass = all_winner_cells_pass && cell_gate;
      test.expect(
          cell_gate,
          std::string(kShapes[shape_index].label) + " " +
              nvfp4_m1_scale_distribution_name(
                  kNvFp4M2RowQuadScaleDistributions[distribution_index]) +
              " winning row-quad cap beats natural and best row-pair");
      std::cout << "PERF_NVFP4_M2_ROW_QUAD_VALIDATION: "
                << kShapes[shape_index].label << " distribution="
                << nvfp4_m1_scale_distribution_name(
                       kNvFp4M2RowQuadScaleDistributions[distribution_index])
                << " best_pair_cap="
                << kNvFp4M2RowQuadPairGridCaps[best_pair_index]
                << " dynamic_best_quad_cap="
                << kNvFp4M2RowQuadGridCaps[best_quad_index]
                << " production_quad_cap=" << production_quad_cap
                << " pair_vs_natural=" << pair_vs_natural
                << " winner_vs_natural=" << winner_vs_natural
                << " winner_vs_pair=" << winner_vs_pair
                << " quad64_vs_natural=" << quad64_vs_natural
                << " quad64_vs_pair=" << quad64_vs_pair
                << " required_speedup=" << kMinimumPerCellSpeedup
                << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
    }

    const NvFp4M2RowQuadDistributionMeasurement& checkpoint =
        measurements[shape_index].distributions[0U];
    const NvFp4M2RowQuadComparison& best_pair =
        checkpoint.row_pair[best_pair_index];
    const NvFp4M2RowQuadComparison& winner =
        checkpoint.row_quad[production_quad_index];
    const double calls =
        static_cast<double>(kShapes[shape_index].checkpoint_calls);
    checkpoint_natural_total += calls * winner.natural_milliseconds;
    checkpoint_pair_total += calls * best_pair.candidate_milliseconds;
    checkpoint_quad_total += calls * winner.candidate_milliseconds;
  }

  const double checkpoint_winner_vs_natural =
      checkpoint_natural_total / checkpoint_quad_total;
  const double checkpoint_winner_vs_pair =
      checkpoint_pair_total / checkpoint_quad_total;
  const bool aggregate_gate =
      production_matches_dynamic_winner && all_candidate_bits_equal &&
      all_winner_cells_pass &&
      std::isfinite(checkpoint_winner_vs_natural) &&
      std::isfinite(checkpoint_winner_vs_pair) &&
      checkpoint_winner_vs_natural >= kMinimumCheckpointWeightedSpeedup &&
      checkpoint_winner_vs_pair >= kMinimumCheckpointWeightedSpeedup;
  std::cout << "PERF_NVFP4_M2_ROW_QUAD_SELECTED: best_pair_cap="
            << kNvFp4M2RowQuadPairGridCaps[best_pair_index]
            << " dynamic_best_quad_cap="
            << kNvFp4M2RowQuadGridCaps[best_quad_index]
            << " production_quad_cap=" << production_quad_cap
            << " production_matches_dynamic_winner="
            << (production_matches_dynamic_winner ? "true" : "false")
            << " best_pair_weighted_vs_natural="
            << best_pair_weighted_speedup
            << " best_quad_weighted_vs_natural="
            << best_quad_weighted_speedup
            << " winner_checkpoint_weighted_vs_natural="
            << checkpoint_winner_vs_natural
            << " winner_checkpoint_weighted_vs_pair="
            << checkpoint_winner_vs_pair
            << " required_weighted_speedup="
            << kMinimumCheckpointWeightedSpeedup
            << " profile_calls=128:64 all_candidate_bitwise="
            << (all_candidate_bits_equal ? "true" : "false")
            << " all_cells="
            << (all_winner_cells_pass ? "PASS" : "FAIL")
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "NVFP4 M2 production cap64 clears every production gate");
}

struct NvFp4M8FixedShapeMeasurement {
  float baseline_milliseconds = std::numeric_limits<float>::quiet_NaN();
  float candidate_milliseconds = std::numeric_limits<float>::quiet_NaN();
  bool passed = false;
};

[[nodiscard]] NvFp4M8FixedShapeMeasurement
benchmark_nvfp4_m8_fixed_shape(
    TestContext& test, cudaStream_t stream, const std::size_t rows,
    const std::size_t columns, const std::string& label) {
  constexpr std::size_t kTokens = 8U;
  constexpr int kWarmupIterations = 10;
  constexpr int kMeasuredIterations = 40;
  constexpr float kWeightScale2 = 1.0F / 64.0F;
  constexpr float kRequiredSpeedup = 1.03F;
  const std::size_t packed_count = rows * columns / 2U;
  const std::size_t scale_columns = columns / 16U;
  const std::size_t scale_count = rows * scale_columns;
  std::vector<std::uint8_t> host_scales(scale_count);
  std::vector<std::uint16_t> host_activations(
      kTokens * columns, encode_bf16(1.0F));
  std::vector<std::uint16_t> baseline(kTokens * rows);
  std::vector<std::uint16_t> candidate(kTokens * rows);

  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<std::uint8_t> scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> baseline_output;
  DeviceBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(packed.allocate(packed_count),
                            label + " allocate packed weights");
  ready = ready && test.cuda_ok(scales.allocate(scale_count),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(activations.allocate(host_activations.size()),
                                label + " allocate activations");
  ready = ready && test.cuda_ok(baseline_output.allocate(kTokens * rows),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kTokens * rows),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(packed.get(), 0x21, packed_count,
                                       stream),
                       label + " initialize packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.get(), host_activations.data(),
                           host_activations.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize activations");
  if (!ready) {
    return {};
  }

  constexpr std::array<NvFp4M1ScaleDistribution, 2U> kDistributions{{
      NvFp4M1ScaleDistribution::kCheckpointLike,
      NvFp4M1ScaleDistribution::kSameBankStress,
  }};
  double baseline_sum = 0.0;
  double candidate_sum = 0.0;
  bool all_cells_passed = true;
  for (const NvFp4M1ScaleDistribution distribution : kDistributions) {
    const std::string distribution_label =
        label + " " + nvfp4_m1_scale_distribution_name(distribution);
    fill_nvfp4_m1_scale_distribution(host_scales, scale_columns,
                                     distribution);
    ready = test.cuda_ok(
        cudaMemcpyAsync(scales.get(), host_scales.data(), host_scales.size(),
                        cudaMemcpyHostToDevice, stream),
        distribution_label + " initialize block scales");
    if (!ready) {
      return {};
    }

    const auto launch_baseline = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m8_scale_codebook_test_cuda(
              packed.get(), scales.get(), kWeightScale2, activations.get(),
              rows, columns, baseline_output.get(),
              static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() noexcept -> int {
      return q3x::kernels::
          launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
              packed.get(), scales.get(), kWeightScale2, activations.get(),
              kTokens, rows, columns, candidate_output.get(),
              static_cast<void*>(stream));
    };

    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         distribution_label + " correctness baseline");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
                         distribution_label + " correctness candidate");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             baseline.data(), baseline_output.get(),
                             baseline.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy baseline output");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             candidate.data(), candidate_output.get(),
                             candidate.size() * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream),
                         distribution_label + " copy candidate output");
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " correctness synchronize");
    if (!ready) {
      return {};
    }
    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < baseline.size(); ++index) {
      mismatches += baseline[index] != candidate[index] ? 1U : 0U;
    }
    const bool bitwise_equal = mismatches == 0U;
    test.expect(bitwise_equal,
                distribution_label +
                    " fixed shape matches every production BF16 bit");

    for (int iteration = 0; iteration < kWarmupIterations && ready;
         ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           distribution_label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(launch_candidate()),
                           distribution_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(
                         cudaStreamSynchronize(stream),
                         distribution_label + " warmup synchronize");
    if (!ready) {
      return {};
    }

    const float baseline_first = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        distribution_label + " baseline pass 1");
    const float candidate_first = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        distribution_label + " candidate pass 1");
    const float candidate_second = measure_small_m_tile(
        test, stream, launch_candidate, kMeasuredIterations,
        distribution_label + " candidate pass 2");
    const float baseline_second = measure_small_m_tile(
        test, stream, launch_baseline, kMeasuredIterations,
        distribution_label + " baseline pass 2");
    const bool finite =
        std::isfinite(baseline_first) && std::isfinite(candidate_first) &&
        std::isfinite(candidate_second) && std::isfinite(baseline_second);
    const float baseline_average =
        finite ? (baseline_first + baseline_second) * 0.5F
               : std::numeric_limits<float>::quiet_NaN();
    const float candidate_average =
        finite ? (candidate_first + candidate_second) * 0.5F
               : std::numeric_limits<float>::quiet_NaN();
    const float speedup = baseline_average / candidate_average;
    const bool cell_gate = bitwise_equal && finite &&
                           std::isfinite(speedup) &&
                           speedup >= kRequiredSpeedup;
    std::cout << "PERF_NVFP4_M8_FIXED_SHAPE: " << label
              << " distribution="
              << nvfp4_m1_scale_distribution_name(distribution)
              << " baseline_pass1_ms=" << baseline_first
              << " candidate_pass1_ms=" << candidate_first
              << " candidate_pass2_ms=" << candidate_second
              << " baseline_pass2_ms=" << baseline_second
              << " baseline_average_ms=" << baseline_average
              << " candidate_average_ms=" << candidate_average
              << " speedup=" << speedup
              << " uplift_percent=" << (speedup - 1.0F) * 100.0F
              << " bitwise_mismatches=" << mismatches << '/'
              << baseline.size() << " required_speedup=" << kRequiredSpeedup
              << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
    test.expect(cell_gate,
                distribution_label +
                    " clears the fixed-shape performance gate");
    if (finite) {
      baseline_sum += static_cast<double>(baseline_average);
      candidate_sum += static_cast<double>(candidate_average);
    }
    all_cells_passed = all_cells_passed && cell_gate;
  }

  const float baseline_average = static_cast<float>(
      baseline_sum / static_cast<double>(kDistributions.size()));
  const float candidate_average = static_cast<float>(
      candidate_sum / static_cast<double>(kDistributions.size()));
  const float shape_speedup = baseline_average / candidate_average;
  const bool shape_gate =
      all_cells_passed && std::isfinite(shape_speedup) &&
      shape_speedup >= kRequiredSpeedup;
  std::cout << "PERF_NVFP4_M8_FIXED_SHAPE_SHAPE: " << label
            << " baseline_average_ms=" << baseline_average
            << " candidate_average_ms=" << candidate_average
            << " speedup=" << shape_speedup
            << " uplift_percent=" << (shape_speedup - 1.0F) * 100.0F
            << " all_distribution_cells_passed="
            << (all_cells_passed ? "true" : "false")
            << " required_speedup=" << kRequiredSpeedup
            << " gate=" << (shape_gate ? "PASS" : "FAIL") << '\n';
  test.expect(shape_gate,
              label + " clears its aggregate fixed-shape performance gate");
  return {baseline_average, candidate_average, shape_gate};
}

void run_optional_nvfp4_m8_fixed_shape_performance(
    TestContext& test, cudaStream_t stream) {
  if (!nvfp4_m8_fixed_shape_performance_enabled()) {
    std::cout << "SKIP: NVFP4 M8 fixed-shape performance segment; set "
                 "Q3X_RUN_SM87_NVFP4_M8_FIXED_SHAPE_PERF=1 to enable\n";
    return;
  }
  constexpr double kGateUpCallsPerPrompt = 256.0;
  constexpr double kDownCallsPerPrompt = 128.0;
  constexpr double kRequiredSpeedup = 1.03;
  const NvFp4M8FixedShapeMeasurement gate_up =
      benchmark_nvfp4_m8_fixed_shape(
          test, stream, 17'408U, 5'120U,
          "NVFP4 M8 MLP gate/up 17408x5120");
  const NvFp4M8FixedShapeMeasurement down =
      benchmark_nvfp4_m8_fixed_shape(
          test, stream, 5'120U, 17'408U,
          "NVFP4 M8 MLP down 5120x17408");
  const double weighted_baseline =
      kGateUpCallsPerPrompt * gate_up.baseline_milliseconds +
      kDownCallsPerPrompt * down.baseline_milliseconds;
  const double weighted_candidate =
      kGateUpCallsPerPrompt * gate_up.candidate_milliseconds +
      kDownCallsPerPrompt * down.candidate_milliseconds;
  const double weighted_speedup = weighted_baseline / weighted_candidate;
  const bool aggregate_gate =
      gate_up.passed && down.passed &&
      std::isfinite(weighted_speedup) &&
      weighted_speedup >= kRequiredSpeedup;
  std::cout << "PERF_NVFP4_M8_FIXED_SHAPE_AGGREGATE: "
            << "weighted_baseline_ms=" << weighted_baseline
            << " weighted_candidate_ms=" << weighted_candidate
            << " speedup=" << weighted_speedup
            << " uplift_percent=" << (weighted_speedup - 1.0) * 100.0
            << " required_speedup=" << kRequiredSpeedup
            << " gate=" << (aggregate_gate ? "PASS" : "FAIL") << '\n';
  test.expect(aggregate_gate,
              "NVFP4 M8 fixed shape clears both shapes and weighted gate");
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
  run_fp8_case(test, stream, 13U, 37U,
               "FP8 public M16 two-M8 scalar-K fallback", false, false, 16U,
               false, true);
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
  run_fp8_m2_row_pair_exhaustive_case(test, stream);
  run_fp8_m1_row_pair_exhaustive_case(test, stream);
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
  run_nvfp4_m1_scale_codebook_exhaustive_case(test, stream);
  run_nvfp4_m1_row_pair_exhaustive_case(test, stream);
  run_nvfp4_m1_scale_codebook_bitwise_case(
      test, stream, 17U, 5'120U,
      "NVFP4 M1 scale codebook odd rows K5120 full E2M1 codebook");
  run_nvfp4_m1_scale_codebook_bitwise_case(
      test, stream, 19U, 17'408U,
      "NVFP4 M1 scale codebook odd rows K17408 full E2M1 codebook");
  run_nvfp4_m1_scale_codebook_bitwise_case(
      test, stream, 769U, 5'120U,
      "NVFP4 M1 public persistent grid-stride rows769 K5120", true);
  run_nvfp4_m2_scale_codebook_exhaustive_case(test, stream);
  run_nvfp4_m2_scale_codebook_bitwise_case(
      test, stream, 17U, 5'120U,
      "NVFP4 M2 scale codebook odd rows K5120 full E2M1 codebook");
  run_nvfp4_m2_scale_codebook_bitwise_case(
      test, stream, 19U, 17'408U,
      "NVFP4 M2 scale codebook odd rows K17408 full E2M1 codebook");
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
  run_optional_fp8_m1_grid_cap_performance(test, stream);
  run_optional_fp8_m1_row_pair_performance(test, stream);
  run_optional_fp8_m1_row_quad_performance(test, stream);
  run_optional_fp8_m1_swizzled_codebook_performance(test, stream);
  run_fp8_m1_kv_pair_correctness_and_optional_performance(test, stream);
  run_optional_fp8_m2_grid_cap_performance(test, stream);
  run_optional_fp8_m2_row_pair_performance(test, stream);
  run_optional_fp8_m2_row_quad_performance(test, stream);
  run_optional_nvfp4_m1_grid_cap_performance(test, stream);
  run_optional_nvfp4_m1_row_pair_performance(test, stream);
  run_optional_nvfp4_m1_row_quad_performance(test, stream);
  run_nvfp4_m1_down_dual_iteration_probe(test, stream);
  run_optional_nvfp4_m1_exact_shape_performance(test, stream);
  run_optional_fp8_m8_fixed_shape_performance(test, stream);
  run_optional_fp8_m16_wmma_performance(test, stream);
  run_optional_nvfp4_m16_wmma_performance(test, stream);
  run_optional_nvfp4_m16_k128_performance(test, stream);
  run_optional_fp8_row_pair_performance(test, stream);
  run_optional_nvfp4_m1_scale_codebook_performance(test, stream);
  run_optional_nvfp4_m2_scale_codebook_performance(test, stream);
  run_optional_nvfp4_m2_row_pair_performance(test, stream);
  run_optional_nvfp4_m2_row_quad_performance(test, stream);
  run_optional_nvfp4_m8_fixed_shape_performance(test, stream);
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
