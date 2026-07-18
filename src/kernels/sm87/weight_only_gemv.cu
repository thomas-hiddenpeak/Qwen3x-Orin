#include "q3x/kernels/sm87_weight_only_gemv.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarpsPerBlock = 8U;
constexpr unsigned int kThreads = kWarpSize * kWarpsPerBlock;
constexpr std::size_t kMaximumBlocks = 65'535U;
constexpr std::size_t kMaximumSmallMTokens = 8U;
constexpr std::size_t kFp8EncodedValueCount = 256U;
constexpr std::size_t kFp8VectorValuesPerLane = 4U;
constexpr std::size_t kFp8VectorColumnsPerBlock =
    kThreads * kFp8VectorValuesPerLane;
constexpr std::size_t kNvFp4GroupSize = 16U;
constexpr std::size_t kNvFp4ValuesPerByte = 2U;
constexpr std::size_t kNvFp4PackedValuesPerScale =
    kNvFp4GroupSize / kNvFp4ValuesPerByte;
constexpr std::size_t kNvFp4VectorPackedBytesPerLane = 4U;
constexpr std::size_t kNvFp4VectorValuesPerLane =
    kNvFp4VectorPackedBytesPerLane * kNvFp4ValuesPerByte;
constexpr std::size_t kNvFp4VectorColumnsPerWarp =
    kWarpSize * kNvFp4VectorValuesPerLane;

[[nodiscard]] bool multiply_overflows(const std::size_t left,
                                      const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
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
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

[[nodiscard]] unsigned int block_count_for_rows(
    const std::size_t rows) noexcept {
  const std::size_t wanted =
      rows / kWarpsPerBlock + (rows % kWarpsPerBlock != 0U ? 1U : 0U);
  return static_cast<unsigned int>(wanted < kMaximumBlocks ? wanted
                                                           : kMaximumBlocks);
}

[[nodiscard]] unsigned int block_count_for_single_row(
    const std::size_t rows) noexcept {
  return static_cast<unsigned int>(rows < kMaximumBlocks ? rows
                                                         : kMaximumBlocks);
}

__device__ __forceinline__ float decode_bf16(const std::uint16_t bits) {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_rne(const float value) {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

// Constructing the exactly representable FP32 value directly avoids the
// ldexpf-heavy scalar decoder used by the correctness reference on SM87.
__device__ __forceinline__ float decode_e4m3fn(const std::uint8_t bits) {
  const unsigned int sign =
      static_cast<unsigned int>(bits & 0x80U) << 24U;
  const unsigned int magnitude = static_cast<unsigned int>(bits & 0x7fU);
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;

  if (magnitude == 0x7fU) {
    return __uint_as_float(sign | 0x7fc0'0000U);
  }
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return __uint_as_float(sign);
    }
    const unsigned int leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const unsigned int fp32_exponent = 118U + leading;
    const unsigned int fp32_mantissa =
        (mantissa - (1U << leading)) << (23U - leading);
    return __uint_as_float(sign | (fp32_exponent << 23U) | fp32_mantissa);
  }
  return __uint_as_float(sign | ((120U + exponent) << 23U) |
                         (mantissa << 20U));
}

__device__ __forceinline__ float decode_e2m1(const std::uint8_t nibble) {
  const unsigned int sign =
      static_cast<unsigned int>(nibble & 0x08U) << 28U;
  const unsigned int magnitude = static_cast<unsigned int>(nibble & 0x07U);
  if (magnitude == 0U) {
    return __uint_as_float(sign);
  }
  const unsigned int exponent = 126U + (magnitude >> 1U);
  const unsigned int mantissa =
      ((magnitude & 1U) != 0U && magnitude != 1U) ? 0x0040'0000U : 0U;
  return __uint_as_float(sign | (exponent << 23U) | mantissa);
}

__device__ __forceinline__ float warp_sum(float value) {
  constexpr unsigned int kMask = 0xffff'ffffU;
  value += __shfl_down_sync(kMask, value, 16U);
  value += __shfl_down_sync(kMask, value, 8U);
  value += __shfl_down_sync(kMask, value, 4U);
  value += __shfl_down_sync(kMask, value, 2U);
  value += __shfl_down_sync(kMask, value, 1U);
  return value;
}

__global__ __launch_bounds__(kThreads) void
fp8_w8a16_gemv_bf16_scalar_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  __shared__ float warp_sums[kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < rows;
       row += static_cast<std::size_t>(gridDim.x)) {
    const std::uint8_t* const row_weights = weights + row * columns;
    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
      sum = fmaf(decode_e4m3fn(row_weights[column]),
                 decode_bf16(activation[column]), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0U) {
      warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0U) {
      float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
      block_sum = warp_sum(block_sum) * weight_scale;
      if (lane == 0U) {
        output[row] = encode_bf16_rne(block_sum);
      }
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreads) void
fp8_w8a16_gemv_bf16_vector_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  // The E4M3FN code space is only 256 floats. Decode it once per output-row
  // block, then replace every inner-loop integer decode with a shared lookup.
  decoded_weights[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  __syncthreads();
  for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < rows;
       row += static_cast<std::size_t>(gridDim.x)) {
    const std::uint8_t* const row_weights = weights + row * columns;
    float accumulators[4]{0.0F, 0.0F, 0.0F, 0.0F};

    // A 32-bit weight load and a 64-bit BF16 activation load supply four
    // adjacent K values. All production K values dispatched here are exact
    // multiples of the 1024 columns covered by one 256-thread iteration.
    for (std::size_t first_column =
             static_cast<std::size_t>(threadIdx.x) *
             kFp8VectorValuesPerLane;
         first_column < columns;
         first_column += kFp8VectorColumnsPerBlock) {
      const std::uint32_t packed_weights =
          *reinterpret_cast<const std::uint32_t*>(row_weights + first_column);
      const std::uint64_t packed_activation =
          *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
        const std::uint8_t encoded_weight = static_cast<std::uint8_t>(
            (packed_weights >> (value * 8U)) & 0xffU);
        const std::uint16_t encoded_activation =
            static_cast<std::uint16_t>(
                (packed_activation >> (value * 16U)) & 0xffffU);
        accumulators[value] =
            fmaf(decoded_weights[encoded_weight],
                 decode_bf16(encoded_activation), accumulators[value]);
      }
    }

    float sum = (accumulators[0] + accumulators[1]) +
                (accumulators[2] + accumulators[3]);
    sum = warp_sum(sum);
    if (lane == 0U) {
      warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0U) {
      float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
      block_sum = warp_sum(block_sum) * weight_scale;
      if (lane == 0U) {
        output[row] = encode_bf16_rne(block_sum);
      }
    }
    __syncthreads();
  }
}

template <std::size_t TokenCount>
__global__ __launch_bounds__(kThreads) void
fp8_w8a16_small_m_gemm_bf16_vector_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  static_assert(TokenCount >= 2U && TokenCount <= kMaximumSmallMTokens);
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[TokenCount][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  // All tokens share the same row weights, including the decoded codebook.
  decoded_weights[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  __syncthreads();

  for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < rows;
       row += static_cast<std::size_t>(gridDim.x)) {
    const std::uint8_t* const row_weights = weights + row * columns;
    float accumulators[TokenCount]{};

    // One packed weight word is decoded once and reused across the complete
    // token tile. Token-major activation rows remain 64-bit aligned because
    // the dispatcher requires both the base and K stride to be aligned.
    for (std::size_t first_column =
             static_cast<std::size_t>(threadIdx.x) *
             kFp8VectorValuesPerLane;
         first_column < columns;
         first_column += kFp8VectorColumnsPerBlock) {
      const std::uint32_t packed_weights =
          *reinterpret_cast<const std::uint32_t*>(row_weights + first_column);
      std::uint64_t packed_activations[TokenCount];
#pragma unroll
      for (unsigned int token = 0U; token < TokenCount; ++token) {
        packed_activations[token] =
            *reinterpret_cast<const std::uint64_t*>(
                activations + static_cast<std::size_t>(token) * columns +
                first_column);
      }
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
        const std::uint8_t encoded_weight = static_cast<std::uint8_t>(
            (packed_weights >> (value * 8U)) & 0xffU);
        const float decoded_weight = decoded_weights[encoded_weight];
#pragma unroll
        for (unsigned int token = 0U; token < TokenCount; ++token) {
          const std::uint16_t encoded_activation =
              static_cast<std::uint16_t>(
                  (packed_activations[token] >> (value * 16U)) & 0xffffU);
          accumulators[token] =
              fmaf(decoded_weight, decode_bf16(encoded_activation),
                   accumulators[token]);
        }
      }
    }

#pragma unroll
    for (unsigned int token = 0U; token < TokenCount; ++token) {
      const float sum = warp_sum(accumulators[token]);
      if (lane == 0U) {
        warp_sums[token][warp] = sum;
      }
    }
    __syncthreads();
    if (warp == 0U) {
#pragma unroll
      for (unsigned int token = 0U; token < TokenCount; ++token) {
        float block_sum =
            lane < kWarpsPerBlock ? warp_sums[token][lane] : 0.0F;
        block_sum = warp_sum(block_sum) * weight_scale;
        if (lane == 0U) {
          output[static_cast<std::size_t>(token) * rows + row] =
              encode_bf16_rne(block_sum);
        }
      }
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreads) void
nvfp4_w4a16_gemv_bf16_scalar_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::size_t packed_columns = columns / kNvFp4ValuesPerByte;
  const std::size_t scale_columns = columns / kNvFp4GroupSize;
  const std::size_t first_row =
      static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::size_t row_stride =
      static_cast<std::size_t>(gridDim.x) * kWarpsPerBlock;

  for (std::size_t row = first_row; row < rows; row += row_stride) {
    const std::uint8_t* const row_weights =
        packed_weights + row * packed_columns;
    const std::uint8_t* const row_scales =
        block_scales + row * scale_columns;
    float sum = 0.0F;

    // Eight adjacent lanes consume one group of eight packed bytes (16 E2M1
    // values). The group leader decodes the E4M3 scale once and broadcasts it;
    // every packed byte is fetched once and supplies both of its nibbles.
    for (std::size_t packed_column = lane; packed_column < packed_columns;
         packed_column += kWarpSize) {
      const unsigned int active = __activemask();
      float block_scale = 0.0F;
      if ((lane & (kNvFp4PackedValuesPerScale - 1U)) == 0U) {
        block_scale = decode_e4m3fn(
            row_scales[packed_column / kNvFp4PackedValuesPerScale]);
      }
      const int scale_source = static_cast<int>(
          lane & ~(kNvFp4PackedValuesPerScale - 1U));
      block_scale = __shfl_sync(active, block_scale, scale_source);

      const std::uint8_t packed = row_weights[packed_column];
      const std::size_t column =
          packed_column * kNvFp4ValuesPerByte;
      const float low = decode_e2m1(packed & 0x0fU) * block_scale;
      const float high = decode_e2m1(packed >> 4U) * block_scale;
      sum = fmaf(low, decode_bf16(activation[column]), sum);
      sum = fmaf(high, decode_bf16(activation[column + 1U]), sum);
    }
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row] = encode_bf16_rne(sum);
    }
  }
}

__global__ __launch_bounds__(kThreads) void
nvfp4_w4a16_gemv_bf16_vector_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::size_t packed_columns = columns / kNvFp4ValuesPerByte;
  const std::size_t scale_columns = columns / kNvFp4GroupSize;
  const std::size_t first_row =
      static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::size_t row_stride =
      static_cast<std::size_t>(gridDim.x) * kWarpsPerBlock;

  for (std::size_t row = first_row; row < rows; row += row_stride) {
    const std::uint8_t* const row_weights =
        packed_weights + row * packed_columns;
    const std::uint8_t* const row_scales =
        block_scales + row * scale_columns;
    float accumulators[4]{0.0F, 0.0F, 0.0F, 0.0F};

    // Each lane consumes four adjacent packed bytes (eight E2M1 values).
    // Adjacent lanes therefore cover the two halves of one 16-value scale
    // group. The even lane decodes that group's scale and broadcasts it to
    // its odd partner. One warp advances by 256 logical columns per loop.
    for (std::size_t packed_column =
             static_cast<std::size_t>(lane) *
             kNvFp4VectorPackedBytesPerLane;
         packed_column < packed_columns;
         packed_column += kNvFp4VectorColumnsPerWarp /
                          kNvFp4ValuesPerByte) {
      float block_scale = 0.0F;
      if ((lane & 1U) == 0U) {
        block_scale = decode_e4m3fn(
            row_scales[packed_column / kNvFp4PackedValuesPerScale]);
      }
      block_scale = __shfl_sync(0xffff'ffffU, block_scale,
                                static_cast<int>(lane & ~1U));

      const std::uint32_t packed =
          *reinterpret_cast<const std::uint32_t*>(row_weights +
                                                  packed_column);
      const std::size_t first_column =
          packed_column * kNvFp4ValuesPerByte;
#pragma unroll
      for (unsigned int value = 0U;
           value < kNvFp4VectorValuesPerLane; ++value) {
        const std::uint8_t nibble = static_cast<std::uint8_t>(
            (packed >> (value * 4U)) & 0x0fU);
        const float scaled_weight = decode_e2m1(nibble) * block_scale;
        accumulators[value & 3U] =
            fmaf(scaled_weight, decode_bf16(activation[first_column + value]),
                 accumulators[value & 3U]);
      }
    }

    float sum = (accumulators[0] + accumulators[1]) +
                (accumulators[2] + accumulators[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row] = encode_bf16_rne(sum);
    }
  }
}

template <std::size_t TokenCount>
__global__ __launch_bounds__(kThreads) void
nvfp4_w4a16_small_m_gemm_bf16_vector_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  static_assert(TokenCount >= 2U && TokenCount <= kMaximumSmallMTokens);
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::size_t packed_columns = columns / kNvFp4ValuesPerByte;
  const std::size_t scale_columns = columns / kNvFp4GroupSize;
  const std::size_t first_row =
      static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const std::size_t row_stride =
      static_cast<std::size_t>(gridDim.x) * kWarpsPerBlock;

  for (std::size_t row = first_row; row < rows; row += row_stride) {
    const std::uint8_t* const row_weights =
        packed_weights + row * packed_columns;
    const std::uint8_t* const row_scales =
        block_scales + row * scale_columns;
    float accumulators[TokenCount]{};

    for (std::size_t packed_column =
             static_cast<std::size_t>(lane) *
             kNvFp4VectorPackedBytesPerLane;
         packed_column < packed_columns;
         packed_column += kNvFp4VectorColumnsPerWarp /
                          kNvFp4ValuesPerByte) {
      float block_scale = 0.0F;
      if ((lane & 1U) == 0U) {
        block_scale = decode_e4m3fn(
            row_scales[packed_column / kNvFp4PackedValuesPerScale]);
      }
      block_scale = __shfl_sync(0xffff'ffffU, block_scale,
                                static_cast<int>(lane & ~1U));
      const std::uint32_t packed =
          *reinterpret_cast<const std::uint32_t*>(row_weights +
                                                  packed_column);
      const std::size_t first_column =
          packed_column * kNvFp4ValuesPerByte;

      // Reuse the same registers for the low and high four-value activation
      // words so M=8 does not retain sixteen 64-bit values at once.
#pragma unroll
      for (unsigned int half = 0U; half < 2U; ++half) {
        std::uint64_t packed_activations[TokenCount];
#pragma unroll
        for (unsigned int token = 0U; token < TokenCount; ++token) {
          packed_activations[token] =
              *reinterpret_cast<const std::uint64_t*>(
                  activations + static_cast<std::size_t>(token) * columns +
                  first_column + half * 4U);
        }
#pragma unroll
        for (unsigned int value = 0U; value < 4U; ++value) {
          const unsigned int packed_value = half * 4U + value;
          const std::uint8_t nibble = static_cast<std::uint8_t>(
              (packed >> (packed_value * 4U)) & 0x0fU);
          const float scaled_weight = decode_e2m1(nibble) * block_scale;
#pragma unroll
          for (unsigned int token = 0U; token < TokenCount; ++token) {
            const std::uint16_t encoded_activation =
                static_cast<std::uint16_t>(
                    (packed_activations[token] >> (value * 16U)) & 0xffffU);
            accumulators[token] =
                fmaf(scaled_weight, decode_bf16(encoded_activation),
                     accumulators[token]);
          }
        }
      }
    }

#pragma unroll
    for (unsigned int token = 0U; token < TokenCount; ++token) {
      const float sum = warp_sum(accumulators[token]) * weight_scale_2;
      if (lane == 0U) {
        output[static_cast<std::size_t>(token) * rows + row] =
            encode_bf16_rne(sum);
      }
    }
  }
}

[[nodiscard]] int invalid_value() noexcept {
  return static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] int validate_fp8_launch(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) noexcept {
  if (!std::isfinite(weight_scale) || weight_scale < 0.0F ||
      multiply_overflows(rows, columns)) {
    return invalid_value();
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (multiply_overflows(columns, sizeof(std::uint16_t)) ||
      multiply_overflows(rows, sizeof(std::uint16_t))) {
    return invalid_value();
  }
  if (weights == nullptr || activation == nullptr || output == nullptr) {
    return invalid_value();
  }

  const std::size_t weight_bytes = rows * columns;
  const std::size_t activation_bytes = columns * sizeof(std::uint16_t);
  const std::size_t output_bytes = rows * sizeof(std::uint16_t);
  if (ranges_overlap(output, output_bytes, weights, weight_bytes) ||
      ranges_overlap(output, output_bytes, activation, activation_bytes)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

void launch_fp8_scalar_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  const unsigned int blocks = block_count_for_single_row(rows);
  fp8_w8a16_gemv_bf16_scalar_kernel<<<blocks, kThreads, 0U, stream>>>(
      weights, weight_scale, activation, rows, columns, output);
}

void launch_fp8_vector_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  const unsigned int blocks = block_count_for_single_row(rows);
  fp8_w8a16_gemv_bf16_vector_kernel<<<blocks, kThreads, 0U, stream>>>(
      weights, weight_scale, activation, rows, columns, output);
}

[[nodiscard]] int validate_nvfp4_launch(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) noexcept {
  if ((columns % kNvFp4GroupSize) != 0U ||
      !std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F ||
      multiply_overflows(rows, columns)) {
    return invalid_value();
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (multiply_overflows(columns, sizeof(std::uint16_t)) ||
      multiply_overflows(rows, sizeof(std::uint16_t))) {
    return invalid_value();
  }
  if (packed_weights == nullptr || block_scales == nullptr ||
      activation == nullptr || output == nullptr) {
    return invalid_value();
  }

  const std::size_t packed_bytes = rows * (columns / kNvFp4ValuesPerByte);
  const std::size_t scale_bytes = rows * (columns / kNvFp4GroupSize);
  const std::size_t activation_bytes = columns * sizeof(std::uint16_t);
  const std::size_t output_bytes = rows * sizeof(std::uint16_t);
  if (ranges_overlap(output, output_bytes, packed_weights, packed_bytes) ||
      ranges_overlap(output, output_bytes, block_scales, scale_bytes) ||
      ranges_overlap(output, output_bytes, activation, activation_bytes)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

void launch_nvfp4_scalar_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  const unsigned int blocks = block_count_for_rows(rows);
  nvfp4_w4a16_gemv_bf16_scalar_kernel<<<blocks, kThreads, 0U, stream>>>(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
}

void launch_nvfp4_vector_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  const unsigned int blocks = block_count_for_rows(rows);
  nvfp4_w4a16_gemv_bf16_vector_kernel<<<blocks, kThreads, 0U, stream>>>(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
}

[[nodiscard]] int validate_fp8_small_m_launch(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t token_count,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const output) noexcept {
  if (token_count == 0U || token_count > kMaximumSmallMTokens ||
      !std::isfinite(weight_scale) || weight_scale < 0.0F ||
      multiply_overflows(rows, columns)) {
    return invalid_value();
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (multiply_overflows(token_count, columns) ||
      multiply_overflows(token_count, rows)) {
    return invalid_value();
  }
  const std::size_t activation_elements = token_count * columns;
  const std::size_t output_elements = token_count * rows;
  if (multiply_overflows(activation_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(output_elements, sizeof(std::uint16_t))) {
    return invalid_value();
  }
  if (weights == nullptr || activations == nullptr || output == nullptr) {
    return invalid_value();
  }

  const std::size_t weight_bytes = rows * columns;
  const std::size_t activation_bytes =
      activation_elements * sizeof(std::uint16_t);
  const std::size_t output_bytes = output_elements * sizeof(std::uint16_t);
  if (ranges_overlap(output, output_bytes, weights, weight_bytes) ||
      ranges_overlap(output, output_bytes, activations, activation_bytes)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int validate_nvfp4_small_m_launch(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t token_count,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const output) noexcept {
  if (token_count == 0U || token_count > kMaximumSmallMTokens ||
      (columns % kNvFp4GroupSize) != 0U ||
      !std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F ||
      multiply_overflows(rows, columns)) {
    return invalid_value();
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (multiply_overflows(token_count, columns) ||
      multiply_overflows(token_count, rows)) {
    return invalid_value();
  }
  const std::size_t activation_elements = token_count * columns;
  const std::size_t output_elements = token_count * rows;
  if (multiply_overflows(activation_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(output_elements, sizeof(std::uint16_t))) {
    return invalid_value();
  }
  if (packed_weights == nullptr || block_scales == nullptr ||
      activations == nullptr || output == nullptr) {
    return invalid_value();
  }

  const std::size_t packed_bytes = rows * (columns / kNvFp4ValuesPerByte);
  const std::size_t scale_bytes = rows * (columns / kNvFp4GroupSize);
  const std::size_t activation_bytes =
      activation_elements * sizeof(std::uint16_t);
  const std::size_t output_bytes = output_elements * sizeof(std::uint16_t);
  if (ranges_overlap(output, output_bytes, packed_weights, packed_bytes) ||
      ranges_overlap(output, output_bytes, block_scales, scale_bytes) ||
      ranges_overlap(output, output_bytes, activations, activation_bytes)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

template <std::size_t TokenCount>
void launch_fp8_small_m_vector_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  const unsigned int blocks = block_count_for_single_row(rows);
  fp8_w8a16_small_m_gemm_bf16_vector_kernel<TokenCount>
      <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activations,
                                        rows, columns, output);
}

template <std::size_t TokenCount>
void launch_nvfp4_small_m_vector_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  const unsigned int blocks = block_count_for_rows(rows);
  nvfp4_w4a16_small_m_gemm_bf16_vector_kernel<TokenCount>
      <<<blocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, rows,
          columns, output);
}

}  // namespace

int launch_sm87_fp8_w8a16_gemv_bf16_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_fp8_launch(
      weights, weight_scale, activation, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  const bool vector_shape =
      (columns % kFp8VectorColumnsPerBlock) == 0U &&
      (reinterpret_cast<std::uintptr_t>(weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U;
  if (vector_shape) {
    launch_fp8_vector_unchecked(weights, weight_scale, activation, rows,
                                columns, output, stream);
  } else {
    launch_fp8_scalar_unchecked(weights, weight_scale, activation, rows,
                                columns, output, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

// Test-only scalar baseline for mirrored CUDA-event A/B measurements. It is
// intentionally omitted from the public header.
int launch_sm87_fp8_w8a16_gemv_bf16_scalar_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_fp8_launch(
      weights, weight_scale, activation, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_scalar_unchecked(weights, weight_scale, activation, rows,
                              columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  const bool vector_shape =
      (columns % kNvFp4VectorColumnsPerWarp) == 0U &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U;
  if (vector_shape) {
    launch_nvfp4_vector_unchecked(packed_weights, block_scales,
                                  weight_scale_2, activation, rows, columns,
                                  output, stream);
  } else {
    launch_nvfp4_scalar_unchecked(packed_weights, block_scales,
                                  weight_scale_2, activation, rows, columns,
                                  output, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

// Test-only entry point: intentionally omitted from the public header. It
// allows the CUDA-event gate to compare the production vector path with the
// exact scalar kernel in one binary and on identical device buffers.
int launch_sm87_nvfp4_w4a16_gemv_bf16_scalar_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_scalar_unchecked(packed_weights, block_scales, weight_scale_2,
                                activation, rows, columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t token_count,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_small_m_launch(
      weights, weight_scale, activations, token_count, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  if (token_count == 1U) {
    return launch_sm87_fp8_w8a16_gemv_bf16_cuda(
        weights, weight_scale, activations, rows, columns, output,
        cuda_stream);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const bool vector_shape =
      (columns % kFp8VectorColumnsPerBlock) == 0U &&
      (reinterpret_cast<std::uintptr_t>(weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (vector_shape) {
    (void)cudaGetLastError();
    switch (token_count) {
      case 2U:
        launch_fp8_small_m_vector_unchecked<2U>(
            weights, weight_scale, activations, rows, columns, output, stream);
        break;
      case 3U:
        launch_fp8_small_m_vector_unchecked<3U>(
            weights, weight_scale, activations, rows, columns, output, stream);
        break;
      case 4U:
        launch_fp8_small_m_vector_unchecked<4U>(
            weights, weight_scale, activations, rows, columns, output, stream);
        break;
      case 5U:
        launch_fp8_small_m_vector_unchecked<5U>(
            weights, weight_scale, activations, rows, columns, output, stream);
        break;
      case 6U:
        launch_fp8_small_m_vector_unchecked<6U>(
            weights, weight_scale, activations, rows, columns, output, stream);
        break;
      case 7U:
        launch_fp8_small_m_vector_unchecked<7U>(
            weights, weight_scale, activations, rows, columns, output, stream);
        break;
      case 8U:
        launch_fp8_small_m_vector_unchecked<8U>(
            weights, weight_scale, activations, rows, columns, output, stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }

  for (std::size_t token = 0U; token < token_count; ++token) {
    const int status = launch_sm87_fp8_w8a16_gemv_bf16_cuda(
        weights, weight_scale, activations + token * columns, rows, columns,
        output + token * rows, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t token_count,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_small_m_launch(
      packed_weights, block_scales, weight_scale_2, activations, token_count,
      rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  if (token_count == 1U) {
    return launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
        packed_weights, block_scales, weight_scale_2, activations, rows,
        columns, output, cuda_stream);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const bool vector_shape =
      (columns % kNvFp4VectorColumnsPerWarp) == 0U &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (vector_shape) {
    (void)cudaGetLastError();
    switch (token_count) {
      case 2U:
        launch_nvfp4_small_m_vector_unchecked<2U>(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, stream);
        break;
      case 3U:
        launch_nvfp4_small_m_vector_unchecked<3U>(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, stream);
        break;
      case 4U:
        launch_nvfp4_small_m_vector_unchecked<4U>(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, stream);
        break;
      case 5U:
        launch_nvfp4_small_m_vector_unchecked<5U>(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, stream);
        break;
      case 6U:
        launch_nvfp4_small_m_vector_unchecked<6U>(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, stream);
        break;
      case 7U:
        launch_nvfp4_small_m_vector_unchecked<7U>(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, stream);
        break;
      case 8U:
        launch_nvfp4_small_m_vector_unchecked<8U>(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }

  for (std::size_t token = 0U; token < token_count; ++token) {
    const int status = launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
        packed_weights, block_scales, weight_scale_2,
        activations + token * columns, rows, columns, output + token * rows,
        cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
