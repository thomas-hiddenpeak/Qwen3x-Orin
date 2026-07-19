#include "q3x/kernels/reference_gemv.h"

#include "q3x/quantization/nvfp4.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr std::size_t kMaximumBlocks = 65535U;
constexpr std::size_t kMaximumPairTokens = 16U;
constexpr std::size_t kMaximumGridX =
    static_cast<std::size_t>(std::numeric_limits<int>::max());

[[nodiscard]] bool multiply_overflows(const std::size_t left,
                                      const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
}

[[nodiscard]] bool element_count_overflows(const std::size_t rows,
                                           const std::size_t columns) noexcept {
  return multiply_overflows(rows, columns);
}

[[nodiscard]] bool is_empty(const std::size_t rows,
                            const std::size_t columns) noexcept {
  return rows == 0U || columns == 0U;
}

__device__ __forceinline__ float decode_bf16_device(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_device(
    const float value) {
  unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ float decode_e4m3fn_device(
    const std::uint8_t bits) {
  const std::uint8_t magnitude = bits & 0x7fU;
  const int exponent = static_cast<int>((magnitude >> 3U) & 0x0fU);
  const int mantissa = static_cast<int>(magnitude & 0x07U);
  if (exponent == 0x0f && mantissa == 0x07) {
    return copysignf(nanf(""), (bits & 0x80U) != 0U ? -1.0F : 1.0F);
  }
  const float value =
      exponent == 0
          ? ldexpf(static_cast<float>(mantissa), -9)
          : ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
  return copysignf(value, (bits & 0x80U) != 0U ? -1.0F : 1.0F);
}

__device__ __forceinline__ float decode_e2m1_device(
    const std::uint8_t nibble) {
  constexpr float values[16] = {
      0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
      -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
  };
  return values[nibble & 0x0fU];
}

struct Bf16WeightAccessor {
  const std::uint16_t* weights;
  std::size_t columns;

  __device__ __forceinline__ float at(const std::size_t row,
                                      const std::size_t column) const {
    return decode_bf16_device(weights[row * columns + column]);
  }
};

struct Fp8WeightAccessor {
  const std::uint8_t* weights;
  float weight_scale;
  std::size_t columns;

  __device__ __forceinline__ float at(const std::size_t row,
                                      const std::size_t column) const {
    return decode_e4m3fn_device(weights[row * columns + column]) *
           weight_scale;
  }
};

struct Fp8StaticWeightAccessor {
  const std::uint8_t* weights;
  float input_scale;
  float inverse_input_scale;
  std::size_t columns;

  __device__ __forceinline__ float at(const std::size_t row,
                                      const std::size_t column) const {
    return decode_e4m3fn_device(weights[row * columns + column]);
  }

  __device__ __forceinline__ float activation(
      const std::uint16_t bits) const {
    const float scaled = fmaxf(
        -448.0F,
        fminf(decode_bf16_device(bits) * inverse_input_scale, 448.0F));
    const __nv_fp8_e4m3 quantized(scaled);
    return decode_e4m3fn_device(quantized.__x);
  }
};

struct NvFp4WeightAccessor {
  const std::uint8_t* packed_weights;
  const std::uint8_t* block_scales;
  float weight_scale_2;
  std::size_t packed_row_stride;
  std::size_t scale_row_stride;

  __device__ __forceinline__ float at(const std::size_t row,
                                      const std::size_t column) const {
    const std::uint8_t packed =
        packed_weights[row * packed_row_stride + column / 2U];
    const std::uint8_t nibble =
        (column & 1U) != 0U
            ? static_cast<std::uint8_t>((packed >> 4U) & 0x0fU)
            : static_cast<std::uint8_t>(packed & 0x0fU);
    const std::uint8_t scale =
        block_scales[row * scale_row_stride +
                     column / q3x::quantization::kNvFp4GroupSize];
    return decode_e2m1_device(nibble) * decode_e4m3fn_device(scale) *
           weight_scale_2;
  }
};

template <typename WeightAccessor>
__global__ void gemv_kernel(const WeightAccessor weights,
                            const std::uint16_t* const activation,
                            const std::size_t rows,
                            const std::size_t columns,
                            float* const output) {
  __shared__ float partial[kThreads];

  for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < rows;
       row += static_cast<std::size_t>(gridDim.x)) {
    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
      sum = fmaf(weights.at(row, column), decode_bf16_device(activation[column]),
                 sum);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0U) {
      output[row] = partial[0];
    }
    __syncthreads();
  }
}

__global__ void bf16_gemv_pair_tile_bf16_kernel(
    const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const first_output,
    std::uint16_t* const second_output) {
  __shared__ float partial[kThreads];

  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  const std::size_t token = static_cast<std::size_t>(blockIdx.y);
  const bool second_projection = blockIdx.z != 0U;
  const std::uint16_t* const weights =
      second_projection ? second_weights : first_weights;
  std::uint16_t* const output =
      second_projection ? second_output : first_output;
  const std::uint16_t* const token_input = input + token * columns;

  float sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    sum = fmaf(decode_bf16_device(weights[row * columns + column]),
               decode_bf16_device(token_input[column]), sum);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    output[token * rows + row] = encode_bf16_device(partial[0]);
  }
}

__global__ void fp8_static_gemv_kernel(
    const Fp8StaticWeightAccessor weights,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    const float output_scale,
    float* const output) {
  __shared__ float partial[kThreads];

  for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < rows;
       row += static_cast<std::size_t>(gridDim.x)) {
    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
      sum = fmaf(weights.at(row, column),
                 weights.activation(activation[column]), sum);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0U) {
      output[row] = partial[0] * output_scale;
    }
    __syncthreads();
  }
}

template <typename WeightAccessor>
[[nodiscard]] int launch_gemv(const WeightAccessor& weights,
                              const std::uint16_t* const activation,
                              const std::size_t rows,
                              const std::size_t columns,
                              float* const output,
                              void* const cuda_stream) noexcept {
  const std::size_t wanted_blocks =
      rows < kMaximumBlocks ? rows : kMaximumBlocks;
  const auto blocks = static_cast<unsigned int>(wanted_blocks);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);

  // CUDA stores last-error per host thread. Discard an error owned by an
  // earlier runtime call so this API reports only its own launch boundary.
  (void)cudaGetLastError();
  gemv_kernel<<<blocks, kThreads, 0U, stream>>>(weights, activation, rows,
                                                columns, output);
  return static_cast<int>(cudaGetLastError());
}

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

[[nodiscard]] bool ranges_overlap(const void* const first,
                                  const std::size_t first_bytes,
                                  const void* const second,
                                  const std::size_t second_bytes) noexcept {
  if (byte_range_overflows(first, first_bytes) ||
      byte_range_overflows(second, second_bytes)) {
    return true;
  }
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

}  // namespace

int launch_bf16_gemv_reference_cuda(
    const std::uint16_t* const weights,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    float* const output,
    void* const cuda_stream) noexcept {
  if (element_count_overflows(rows, columns)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (is_empty(rows, columns)) {
    return static_cast<int>(cudaSuccess);
  }
  if (weights == nullptr || activation == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return launch_gemv(Bf16WeightAccessor{weights, columns}, activation, rows,
                     columns, output, cuda_stream);
}

int launch_bf16_gemv_pair_tile_bf16_cuda(
    const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input,
    const std::size_t token_count,
    const std::size_t rows,
    const std::size_t columns,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output,
    void* const cuda_stream) noexcept {
  if (token_count == 0U || token_count > kMaximumPairTokens ||
      multiply_overflows(rows, columns)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (is_empty(rows, columns)) {
    return static_cast<int>(cudaSuccess);
  }
  if (multiply_overflows(token_count, columns) ||
      multiply_overflows(token_count, rows) || rows > kMaximumGridX) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t weight_elements = rows * columns;
  const std::size_t input_elements = token_count * columns;
  const std::size_t output_elements = token_count * rows;
  if (multiply_overflows(weight_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(input_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(output_elements, sizeof(std::uint16_t)) ||
      first_weights == nullptr || second_weights == nullptr ||
      input == nullptr || first_output == nullptr ||
      second_output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t weight_bytes = weight_elements * sizeof(std::uint16_t);
  const std::size_t input_bytes = input_elements * sizeof(std::uint16_t);
  const std::size_t output_bytes = output_elements * sizeof(std::uint16_t);
  if (ranges_overlap(first_output, output_bytes, first_weights,
                     weight_bytes) ||
      ranges_overlap(first_output, output_bytes, second_weights,
                     weight_bytes) ||
      ranges_overlap(first_output, output_bytes, input, input_bytes) ||
      ranges_overlap(second_output, output_bytes, first_weights,
                     weight_bytes) ||
      ranges_overlap(second_output, output_bytes, second_weights,
                     weight_bytes) ||
      ranges_overlap(second_output, output_bytes, input, input_bytes) ||
      ranges_overlap(first_output, output_bytes, second_output,
                     output_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const dim3 blocks(static_cast<unsigned int>(rows),
                    static_cast<unsigned int>(token_count), 2U);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  bf16_gemv_pair_tile_bf16_kernel<<<blocks, kThreads, 0U, stream>>>(
      first_weights, second_weights, input, rows, columns, first_output,
      second_output);
  return static_cast<int>(cudaGetLastError());
}

int launch_fp8_gemv_reference_cuda(
    const std::uint8_t* const weights,
    const float weight_scale,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    float* const output,
    void* const cuda_stream) noexcept {
  if (!std::isfinite(weight_scale) || weight_scale < 0.0F ||
      element_count_overflows(rows, columns)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (is_empty(rows, columns)) {
    return static_cast<int>(cudaSuccess);
  }
  if (weights == nullptr || activation == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return launch_gemv(Fp8WeightAccessor{weights, weight_scale, columns},
                     activation, rows, columns, output, cuda_stream);
}

int launch_fp8_static_gemv_reference_cuda(
    const std::uint8_t* const weights,
    const float weight_scale,
    const float input_scale,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    float* const output,
    void* const cuda_stream) noexcept {
  if (!std::isfinite(weight_scale) || weight_scale < 0.0F ||
      !std::isfinite(input_scale) || input_scale <= 0.0F ||
      element_count_overflows(rows, columns)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (is_empty(rows, columns)) {
    return static_cast<int>(cudaSuccess);
  }
  if (weights == nullptr || activation == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t wanted_blocks =
      rows < kMaximumBlocks ? rows : kMaximumBlocks;
  const auto blocks = static_cast<unsigned int>(wanted_blocks);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  const Fp8StaticWeightAccessor accessor{
      weights, input_scale, 1.0F / input_scale, columns};
  (void)cudaGetLastError();
  fp8_static_gemv_kernel<<<blocks, kThreads, 0U, stream>>>(
      accessor, activation, rows, columns, weight_scale * input_scale,
      output);
  return static_cast<int>(cudaGetLastError());
}

int launch_nvfp4_gemv_reference_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    const float weight_scale_2,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    float* const output,
    void* const cuda_stream) noexcept {
  if (columns % q3x::quantization::kNvFp4GroupSize != 0U ||
      !std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F ||
      element_count_overflows(rows, columns)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (is_empty(rows, columns)) {
    return static_cast<int>(cudaSuccess);
  }
  if (packed_weights == nullptr || block_scales == nullptr ||
      activation == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const NvFp4WeightAccessor accessor{
      packed_weights, block_scales, weight_scale_2,
      columns / q3x::quantization::kNvFp4ValuesPerByte,
      columns / q3x::quantization::kNvFp4GroupSize};
  return launch_gemv(accessor, activation, rows, columns, output, cuda_stream);
}

}  // namespace q3x::kernels
