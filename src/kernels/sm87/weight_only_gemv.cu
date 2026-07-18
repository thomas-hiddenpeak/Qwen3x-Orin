#include "q3x/kernels/sm87_weight_only_gemv.h"

#include <cuda_runtime.h>
#include <mma.h>

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
constexpr std::size_t kFp8M1PersistentMinimumRows = 1'024U;
constexpr unsigned int kFp8M1PersistentMaximumBlocks = 2'048U;
constexpr std::size_t kFp8M2PersistentMinimumRows = 1'024U;
constexpr unsigned int kFp8M2PersistentMaximumBlocks = 2'048U;
constexpr std::size_t kFp8RowPairMinimumRows = 1'024U;
constexpr std::size_t kNvFp4RowPairMinimumRows = kWarpsPerBlock * 2U;
constexpr std::size_t kNvFp4M1ScaleCodebookMinimumRows = kWarpsPerBlock;
constexpr std::size_t kNvFp4M1ScaleCodebookMinimumColumns = 5'120U;
constexpr std::size_t kNvFp4M2ScaleCodebookMinimumRows = kWarpsPerBlock;
constexpr std::size_t kNvFp4M2ScaleCodebookMinimumColumns = 5'120U;
constexpr std::size_t kFp8EncodedValueCount = 256U;
constexpr std::size_t kFp8VectorValuesPerLane = 4U;
constexpr std::size_t kFp8VectorColumnsPerBlock =
    kThreads * kFp8VectorValuesPerLane;
constexpr std::size_t kNvFp4GroupSize = 16U;
constexpr std::size_t kNvFp4ValuesPerByte = 2U;
constexpr std::size_t kNvFp4EncodedValueCount = 16U;
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

[[nodiscard]] constexpr bool use_fp8_m1_persistent_rows(
    const std::size_t rows) noexcept {
  return rows >= kFp8M1PersistentMinimumRows;
}

[[nodiscard]] constexpr bool use_fp8_m2_persistent_rows(
    const std::size_t rows) noexcept {
  return rows >= kFp8M2PersistentMinimumRows;
}

[[nodiscard]] constexpr bool use_nvfp4_small_m_row_pair(
    const std::size_t token_count, const std::size_t rows) noexcept {
  // Avoid doing a complete second-row FMA stream for partial blocks on tiny
  // M=8 matrices. No other token count has passed the production gate.
  return token_count == 8U && rows >= kNvFp4RowPairMinimumRows;
}

[[nodiscard]] constexpr bool use_nvfp4_m8_fixed_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  return (rows == 17'408U && columns == 5'120U) ||
         (rows == 5'120U && columns == 17'408U);
}

[[nodiscard]] constexpr bool use_nvfp4_m16_wmma_fixed_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  return (rows == 17'408U && columns == 5'120U) ||
         (rows == 5'120U && columns == 17'408U);
}

[[nodiscard]] constexpr bool use_nvfp4_m1_scale_codebook(
    const std::size_t rows, const std::size_t columns) noexcept {
  return rows >= kNvFp4M1ScaleCodebookMinimumRows &&
         columns >= kNvFp4M1ScaleCodebookMinimumColumns;
}

[[nodiscard]] constexpr bool use_nvfp4_m2_scale_codebook(
    const std::size_t rows, const std::size_t columns) noexcept {
  return rows >= kNvFp4M2ScaleCodebookMinimumRows &&
         columns >= kNvFp4M2ScaleCodebookMinimumColumns;
}

[[nodiscard]] constexpr bool use_fp8_small_m_row_pair(
    const std::size_t token_count, const std::size_t rows) noexcept {
  // All five checkpoint-bound FP8 projections have at least 1024 output rows.
  // Tiny and synthetic matrices retain the lower-register single-row kernel.
  return token_count == 8U && rows >= kFp8RowPairMinimumRows;
}

[[nodiscard]] constexpr bool use_fp8_m8_fixed_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  return (rows == 10'240U && columns == 5'120U) ||
         (rows == 5'120U && columns == 6'144U) ||
         (rows == 6'144U && columns == 5'120U) ||
         (rows == 12'288U && columns == 5'120U) ||
         (rows == 1'024U && columns == 5'120U);
}

[[nodiscard]] constexpr bool use_fp8_m16_wmma_fixed_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  // The 1024-row projection is intentionally absent: its measured WMMA path
  // regresses versus two production M8 launches and must retain that fallback.
  return (rows == 10'240U && columns == 5'120U) ||
         (rows == 5'120U && columns == 6'144U) ||
         (rows == 6'144U && columns == 5'120U) ||
         (rows == 12'288U && columns == 5'120U);
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
  const unsigned int nonzero_mask =
      0U - static_cast<unsigned int>(magnitude != 0U);
  const unsigned int mantissa =
      ((magnitude & 1U) &
       static_cast<unsigned int>(magnitude > 1U))
      << 22U;
  const unsigned int finite_bits =
      ((126U + (magnitude >> 1U)) << 23U) | mantissa;
  return __uint_as_float(sign | (finite_bits & nonzero_mask));
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

// Production M=8 output-row pair path. A complete 256-thread block keeps
// the single-row kernel's K partition and reduction tree for each row, while
// adjacent rows share every packed BF16 activation load and decode. The token
// loop is transposed after predecoding both packed FP8 words: this shortens the
// lifetime of packed activation registers without changing the K update order
// of any individual (row, token, thread) accumulator.
__global__ __launch_bounds__(kThreads, 5) void
fp8_w8a16_small_m8_gemm_bf16_row_pair_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 8U;
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[2U][kTokenCount][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  decoded_weights[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  __syncthreads();

  const std::size_t first_row = 2U * static_cast<std::size_t>(blockIdx.x);
  const std::size_t row_stride =
      2U * static_cast<std::size_t>(gridDim.x);
  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row1 = row0 + 1U;
    const bool has_row1 = row1 < rows;
    const std::uint8_t* const row0_weights = weights + row0 * columns;
    const std::uint8_t* const row1_weights =
        has_row1 ? weights + row1 * columns : row0_weights;
    float accumulators0[kTokenCount]{};
    float accumulators1[kTokenCount]{};

    for (std::size_t first_column =
             static_cast<std::size_t>(threadIdx.x) *
             kFp8VectorValuesPerLane;
         first_column < columns;
         first_column += kFp8VectorColumnsPerBlock) {
      const std::uint32_t packed_weights0 =
          *reinterpret_cast<const std::uint32_t*>(row0_weights +
                                                  first_column);
      const std::uint32_t packed_weights1 =
          *reinterpret_cast<const std::uint32_t*>(row1_weights +
                                                  first_column);
      float decoded_weights0[kFp8VectorValuesPerLane];
      float decoded_weights1[kFp8VectorValuesPerLane];
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
        const unsigned int shift = value * 8U;
        const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
            (packed_weights0 >> shift) & 0xffU);
        const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
            (packed_weights1 >> shift) & 0xffU);
        decoded_weights0[value] = decoded_weights[encoded_weight0];
        decoded_weights1[value] = decoded_weights[encoded_weight1];
      }

#pragma unroll
      for (unsigned int token = 0U; token < kTokenCount; ++token) {
        const std::uint64_t packed_activation =
            *reinterpret_cast<const std::uint64_t*>(
                activations + static_cast<std::size_t>(token) * columns +
                first_column);
#pragma unroll
        for (unsigned int value = 0U; value < kFp8VectorValuesPerLane;
             ++value) {
          const std::uint16_t encoded_activation =
              static_cast<std::uint16_t>(
                  (packed_activation >> (value * 16U)) & 0xffffU);
          const float decoded_activation = decode_bf16(encoded_activation);
          accumulators0[token] =
              fmaf(decoded_weights0[value], decoded_activation,
                   accumulators0[token]);
          accumulators1[token] =
              fmaf(decoded_weights1[value], decoded_activation,
                   accumulators1[token]);
        }
      }
    }

#pragma unroll
    for (unsigned int token = 0U; token < kTokenCount; ++token) {
      const float sum0 = warp_sum(accumulators0[token]);
      const float sum1 = warp_sum(accumulators1[token]);
      if (lane == 0U) {
        warp_sums[0U][token][warp] = sum0;
        warp_sums[1U][token][warp] = sum1;
      }
    }
    __syncthreads();
    if (warp == 0U) {
#pragma unroll
      for (unsigned int token = 0U; token < kTokenCount; ++token) {
        float block_sum0 =
            lane < kWarpsPerBlock ? warp_sums[0U][token][lane] : 0.0F;
        float block_sum1 =
            lane < kWarpsPerBlock ? warp_sums[1U][token][lane] : 0.0F;
        block_sum0 = warp_sum(block_sum0) * weight_scale;
        block_sum1 = warp_sum(block_sum1) * weight_scale;
        if (lane == 0U) {
          output[static_cast<std::size_t>(token) * rows + row0] =
              encode_bf16_rne(block_sum0);
          if (has_row1) {
            output[static_cast<std::size_t>(token) * rows + row1] =
                encode_bf16_rne(block_sum1);
          }
        }
      }
    }
    __syncthreads();
  }
}

// Production-shape specializations. Every specialization launches one block
// per valid row pair, so the generic grid-stride loop and odd-tail branch
// disappear while the packed loads, FMA nesting, and two-level reduction
// remain byte-for-byte ordered like the generic row-pair kernel.
template <std::size_t kRows, std::size_t kColumns>
__global__ __launch_bounds__(kThreads, 5) void
fp8_w8a16_small_m8_gemm_bf16_fixed_shape_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 8U;
  constexpr unsigned int kColumnStep =
      static_cast<unsigned int>(kFp8VectorColumnsPerBlock);
  static_assert(kRows % 2U == 0U);
  static_assert(kColumns % kFp8VectorColumnsPerBlock == 0U);

  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[2U][kTokenCount][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  decoded_weights[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  __syncthreads();

  const unsigned int row0 = 2U * blockIdx.x;
  const unsigned int row1 = row0 + 1U;
  const std::uint8_t* const row0_weights =
      weights + static_cast<std::size_t>(row0) * kColumns;
  const std::uint8_t* const row1_weights =
      weights + static_cast<std::size_t>(row1) * kColumns;
  float accumulators0[kTokenCount]{};
  float accumulators1[kTokenCount]{};

  for (unsigned int first_column =
           threadIdx.x * kFp8VectorValuesPerLane;
       first_column < kColumns; first_column += kColumnStep) {
    const std::uint32_t packed_weights0 =
        *reinterpret_cast<const std::uint32_t*>(row0_weights +
                                                first_column);
    const std::uint32_t packed_weights1 =
        *reinterpret_cast<const std::uint32_t*>(row1_weights +
                                                first_column);
    float decoded_weights0[kFp8VectorValuesPerLane];
    float decoded_weights1[kFp8VectorValuesPerLane];
#pragma unroll
    for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
      const unsigned int shift = value * 8U;
      const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
          (packed_weights0 >> shift) & 0xffU);
      const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
          (packed_weights1 >> shift) & 0xffU);
      decoded_weights0[value] = decoded_weights[encoded_weight0];
      decoded_weights1[value] = decoded_weights[encoded_weight1];
    }

#pragma unroll
    for (unsigned int token = 0U; token < kTokenCount; ++token) {
      const std::uint64_t packed_activation =
          *reinterpret_cast<const std::uint64_t*>(
              activations + static_cast<std::size_t>(token) * kColumns +
              first_column);
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane;
           ++value) {
        const std::uint16_t encoded_activation =
            static_cast<std::uint16_t>(
                (packed_activation >> (value * 16U)) & 0xffffU);
        const float decoded_activation = decode_bf16(encoded_activation);
        accumulators0[token] =
            fmaf(decoded_weights0[value], decoded_activation,
                 accumulators0[token]);
        accumulators1[token] =
            fmaf(decoded_weights1[value], decoded_activation,
                 accumulators1[token]);
      }
    }
  }

#pragma unroll
  for (unsigned int token = 0U; token < kTokenCount; ++token) {
    const float sum0 = warp_sum(accumulators0[token]);
    const float sum1 = warp_sum(accumulators1[token]);
    if (lane == 0U) {
      warp_sums[0U][token][warp] = sum0;
      warp_sums[1U][token][warp] = sum1;
    }
  }
  __syncthreads();
  if (warp == 0U) {
#pragma unroll
    for (unsigned int token = 0U; token < kTokenCount; ++token) {
      float block_sum0 =
          lane < kWarpsPerBlock ? warp_sums[0U][token][lane] : 0.0F;
      float block_sum1 =
          lane < kWarpsPerBlock ? warp_sums[1U][token][lane] : 0.0F;
      block_sum0 = warp_sum(block_sum0) * weight_scale;
      block_sum1 = warp_sum(block_sum1) * weight_scale;
      if (lane == 0U) {
        output[static_cast<std::size_t>(token) * kRows + row0] =
            encode_bf16_rne(block_sum0);
        output[static_cast<std::size_t>(token) * kRows + row1] =
            encode_bf16_rne(block_sum1);
      }
    }
  }
}

__device__ __forceinline__ void decode_fp8x4_to_bf16x4(
    const std::uint32_t packed, const std::uint16_t* const decoded_weights,
    std::uint32_t* const output) {
  const std::uint16_t value0 =
      decoded_weights[static_cast<std::uint8_t>(packed & 0xffU)];
  const std::uint16_t value1 = decoded_weights[static_cast<std::uint8_t>(
      (packed >> 8U) & 0xffU)];
  const std::uint16_t value2 = decoded_weights[static_cast<std::uint8_t>(
      (packed >> 16U) & 0xffU)];
  const std::uint16_t value3 = decoded_weights[static_cast<std::uint8_t>(
      (packed >> 24U) & 0xffU)];
  output[0] = static_cast<std::uint32_t>(value0) |
              (static_cast<std::uint32_t>(value1) << 16U);
  output[1] = static_cast<std::uint32_t>(value2) |
              (static_cast<std::uint32_t>(value3) << 16U);
}

// Fixed-M16 production kernel. The checkpoint's row-major W[N,K]
// storage is exactly the column-major B[K,N] operand expected by BF16 WMMA,
// so a block can decode 128 rows on the fly without an offline transpose.
// B dies after the final MMA and shares its allocation with the 8 KiB FP32
// output tile. Production shapes use ldm=72 (20.75 KiB total) to remove shared
// bank conflicts; only the test-only raw 1024-row characterization retains
// ldm=64 (18.5 KiB).
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 64U>
__global__ __launch_bounds__(kThreads, 5) void
fp8_w8a16_small_m16_gemm_bf16_wmma_fixed_shape_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 16U;
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kOutputColumnsPerWarp = 16U;
  constexpr unsigned int kColumnsPerStage = 64U;
  constexpr unsigned int kBf16ValuesPerActivationWord = 4U;
  constexpr unsigned int kActivationWordsPerToken =
      kColumnsPerStage / kBf16ValuesPerActivationWord;
  constexpr unsigned int kSharedActivationWordsPerToken =
      kSharedLeadingDimension / kBf16ValuesPerActivationWord;
  constexpr unsigned int kWeightVectorsPerRow =
      kColumnsPerStage / sizeof(uint4);
  constexpr unsigned int kBf16ValuesPerWeightWord = 2U;
  constexpr unsigned int kSharedWeightWordsPerRow =
      kSharedLeadingDimension / kBf16ValuesPerWeightWord;
  constexpr unsigned int kSharedWeightWordCount =
      kOutputColumnsPerBlock * kSharedWeightWordsPerRow;
  constexpr unsigned int kSharedOutputCount =
      kTokenCount * kOutputColumnsPerBlock;
  static_assert(kRows % kOutputColumnsPerBlock == 0U);
  static_assert(kColumns % kColumnsPerStage == 0U);
  static_assert(kWeightVectorsPerRow == 4U);
  static_assert(kSharedLeadingDimension >= kColumnsPerStage);
  static_assert(kSharedLeadingDimension % 8U == 0U);

  union __align__(32) BOrCStorage {
    std::uint32_t weights[kSharedWeightWordCount];
    float output[kSharedOutputCount];
  };
  __shared__ std::uint16_t decoded_weights[kFp8EncodedValueCount];
  __shared__ __align__(32) std::uint64_t
      shared_activations[kTokenCount * kSharedActivationWordsPerToken];
  __shared__ BOrCStorage b_or_c;

  namespace wmma = nvcuda::wmma;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  decoded_weights[thread] = encode_bf16_rne(
      decode_e4m3fn(static_cast<std::uint8_t>(thread)));
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0F);
  const unsigned int first_output_column =
      blockIdx.x * kOutputColumnsPerBlock;

#pragma unroll 1
  for (unsigned int first_k = 0U; first_k < kColumns;
       first_k += kColumnsPerStage) {
    const unsigned int token = thread / kActivationWordsPerToken;
    const unsigned int activation_word =
        thread % kActivationWordsPerToken;
    shared_activations[token * kSharedActivationWordsPerToken +
                       activation_word] =
        *reinterpret_cast<const std::uint64_t*>(
            activations + static_cast<std::size_t>(token) * kColumns +
            first_k +
            activation_word * kBf16ValuesPerActivationWord);

#pragma unroll
    for (unsigned int pass = 0U; pass < 2U; ++pass) {
      const unsigned int vector_index = thread + pass * kThreads;
      const unsigned int local_output_column =
          vector_index / kWeightVectorsPerRow;
      const unsigned int vector_in_row =
          vector_index % kWeightVectorsPerRow;
      const uint4 packed = *reinterpret_cast<const uint4*>(
          weights +
          static_cast<std::size_t>(first_output_column +
                                   local_output_column) *
              kColumns +
          first_k + vector_in_row * sizeof(uint4));
      std::uint32_t* const decoded =
          b_or_c.weights +
          local_output_column * kSharedWeightWordsPerRow +
          vector_in_row * (sizeof(uint4) / sizeof(std::uint16_t));
      decode_fp8x4_to_bf16x4(packed.x, decoded_weights, decoded);
      decode_fp8x4_to_bf16x4(packed.y, decoded_weights, decoded + 2U);
      decode_fp8x4_to_bf16x4(packed.z, decoded_weights, decoded + 4U);
      decode_fp8x4_to_bf16x4(packed.w, decoded_weights, decoded + 6U);
    }
    __syncthreads();

    const auto* const shared_a =
        reinterpret_cast<const __nv_bfloat16*>(shared_activations);
    const auto* const shared_b =
        reinterpret_cast<const __nv_bfloat16*>(b_or_c.weights);
#pragma unroll 1
    for (unsigned int inner_k = 0U; inner_k < kColumnsPerStage;
         inner_k += 16U) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          activation_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::col_major>
          weight_fragment;
      wmma::load_matrix_sync(activation_fragment, shared_a + inner_k,
                             kSharedLeadingDimension);
      wmma::load_matrix_sync(
          weight_fragment,
          shared_b +
              warp * kOutputColumnsPerWarp * kSharedLeadingDimension +
              inner_k,
          kSharedLeadingDimension);
      wmma::mma_sync(accumulator, activation_fragment, weight_fragment,
                     accumulator);
    }
    // This barrier protects both the next-stage decode and the B-to-C union
    // lifetime transition after the final stage.
    __syncthreads();
  }

  wmma::store_matrix_sync(
      b_or_c.output + warp * kOutputColumnsPerWarp, accumulator,
      kOutputColumnsPerBlock, wmma::mem_row_major);
  __syncthreads();

#pragma unroll
  for (unsigned int index = thread; index < kSharedOutputCount;
       index += kThreads) {
    const unsigned int token = index / kOutputColumnsPerBlock;
    const unsigned int local_output_column =
        index % kOutputColumnsPerBlock;
    output[static_cast<std::size_t>(token) * kRows + first_output_column +
           local_output_column] =
        encode_bf16_rne(b_or_c.output[index] * weight_scale);
  }
}

__device__ __forceinline__ void decode_nvfp4x8_to_bf16x8(
    const std::uint32_t packed,
    const std::uint16_t* const decoded_products,
    std::uint32_t* const output) {
#pragma unroll
  for (unsigned int byte_index = 0U; byte_index < sizeof(packed);
       ++byte_index) {
    const std::uint8_t encoded = static_cast<std::uint8_t>(
        (packed >> (byte_index * 8U)) & 0xffU);
    const std::uint16_t low = decoded_products[encoded & 0x0fU];
    const std::uint16_t high = decoded_products[encoded >> 4U];
    output[byte_index] = static_cast<std::uint32_t>(low) |
                         (static_cast<std::uint32_t>(high) << 16U);
  }
}

// Production C16 kernel for the two checkpoint-bound NVFP4 MLP projections.
// Each block decodes an N=128, K=64 tile directly from canonical row-major
// packed E2M1 plus row-major E4M3FN group scales. The decoded W[N,K] tile is
// the column-major B[K,N] operand consumed by BF16 WMMA, so no offline
// transpose or expanded checkpoint representation is required.
//
// Padding the shared leading dimension from 64 to 72 rotates adjacent weight
// rows by four shared-memory banks. A complete BF16 product table removes the
// per-stage E2M1-by-E4M3 multiply without changing finite values: that product
// has at most six significant binary digits and is therefore BF16-exact.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
__global__ __launch_bounds__(kThreads, 5) void
nvfp4_w4a16_small_m16_gemm_bf16_wmma_fixed_shape_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 16U;
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kOutputColumnsPerWarp = 16U;
  constexpr unsigned int kColumnsPerStage = 64U;
  constexpr unsigned int kBf16ValuesPerActivationWord = 4U;
  constexpr unsigned int kActivationWordsPerToken =
      kColumnsPerStage / kBf16ValuesPerActivationWord;
  constexpr unsigned int kSharedActivationWordsPerToken =
      kSharedLeadingDimension / kBf16ValuesPerActivationWord;
  constexpr unsigned int kPackedColumns =
      static_cast<unsigned int>(kColumns / kNvFp4ValuesPerByte);
  constexpr unsigned int kScaleColumns =
      static_cast<unsigned int>(kColumns / kNvFp4GroupSize);
  constexpr unsigned int kValuesPerWeightVector =
      sizeof(uint4) * kNvFp4ValuesPerByte;
  constexpr unsigned int kWeightVectorsPerRow =
      kColumnsPerStage / kValuesPerWeightVector;
  constexpr unsigned int kBf16ValuesPerWeightWord = 2U;
  constexpr unsigned int kSharedWeightWordsPerRow =
      kSharedLeadingDimension / kBf16ValuesPerWeightWord;
  constexpr unsigned int kSharedWeightWordCount =
      kOutputColumnsPerBlock * kSharedWeightWordsPerRow;
  constexpr unsigned int kSharedOutputCount =
      kTokenCount * kOutputColumnsPerBlock;
  constexpr unsigned int kProductWordCount =
      kFp8EncodedValueCount * kNvFp4EncodedValueCount /
      kBf16ValuesPerWeightWord;
  constexpr unsigned int kProductInitializationPasses =
      kProductWordCount / kThreads;
  static_assert(kRows % kOutputColumnsPerBlock == 0U);
  static_assert(kColumns % kColumnsPerStage == 0U);
  static_assert(kWeightVectorsPerRow == 2U);
  static_assert(kSharedLeadingDimension >= kColumnsPerStage);
  static_assert(kSharedLeadingDimension % 8U == 0U);
  static_assert(kProductWordCount == 2'048U);
  static_assert(kProductInitializationPasses == 8U);

  union __align__(32) BOrCStorage {
    std::uint32_t weights[kSharedWeightWordCount];
    float output[kSharedOutputCount];
  };
  __shared__ __align__(32)
      std::uint32_t decoded_product_words[kProductWordCount];
  __shared__ __align__(32) std::uint64_t
      shared_activations[kTokenCount * kSharedActivationWordsPerToken];
  __shared__ BOrCStorage b_or_c;

  namespace wmma = nvcuda::wmma;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;

  // Each pass writes one contiguous 1 KiB stripe. A thread keeps one adjacent
  // E2M1 pair while scale codes advance by 32, so every uint32 store contains
  // the low/high nibble products used for one canonical packed byte.
  const unsigned int product_pair = thread & 7U;
  const float decoded_low =
      decode_e2m1(static_cast<std::uint8_t>(product_pair * 2U));
  const float decoded_high =
      decode_e2m1(static_cast<std::uint8_t>(product_pair * 2U + 1U));
#pragma unroll 1
  for (unsigned int pass = 0U; pass < kProductInitializationPasses; ++pass) {
    const unsigned int scale_code = (thread >> 3U) + pass * 32U;
    const float decoded_scale =
        decode_e4m3fn(static_cast<std::uint8_t>(scale_code));
    const std::uint16_t low =
        encode_bf16_rne(decoded_low * decoded_scale);
    const std::uint16_t high =
        encode_bf16_rne(decoded_high * decoded_scale);
    decoded_product_words[thread + pass * kThreads] =
        static_cast<std::uint32_t>(low) |
        (static_cast<std::uint32_t>(high) << 16U);
  }
  __syncthreads();

  const auto* const decoded_products =
      reinterpret_cast<const std::uint16_t*>(decoded_product_words);
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0F);
  const unsigned int first_output_column =
      blockIdx.x * kOutputColumnsPerBlock;
  const unsigned int local_output_column = thread / kWeightVectorsPerRow;
  const unsigned int vector_in_row = thread % kWeightVectorsPerRow;
  const unsigned int output_column =
      first_output_column + local_output_column;

#pragma unroll 1
  for (unsigned int first_k = 0U; first_k < kColumns;
       first_k += kColumnsPerStage) {
    const unsigned int token = thread / kActivationWordsPerToken;
    const unsigned int activation_word =
        thread % kActivationWordsPerToken;
    shared_activations[token * kSharedActivationWordsPerToken +
                       activation_word] =
        *reinterpret_cast<const std::uint64_t*>(
            activations + static_cast<std::size_t>(token) * kColumns +
            first_k +
            activation_word * kBf16ValuesPerActivationWord);

    const unsigned int first_vector_column =
        first_k + vector_in_row * kValuesPerWeightVector;
    const uint4 packed = *reinterpret_cast<const uint4*>(
        packed_weights +
        static_cast<std::size_t>(output_column) * kPackedColumns +
        first_vector_column / kNvFp4ValuesPerByte);
    const std::uint16_t encoded_scales =
        *reinterpret_cast<const std::uint16_t*>(
            block_scales +
            static_cast<std::size_t>(output_column) * kScaleColumns +
            first_vector_column / kNvFp4GroupSize);
    const std::uint8_t scale0 =
        static_cast<std::uint8_t>(encoded_scales & 0xffU);
    const std::uint8_t scale1 =
        static_cast<std::uint8_t>(encoded_scales >> 8U);
    const std::uint16_t* const products0 =
        decoded_products +
        static_cast<unsigned int>(scale0) * kNvFp4EncodedValueCount;
    const std::uint16_t* const products1 =
        decoded_products +
        static_cast<unsigned int>(scale1) * kNvFp4EncodedValueCount;
    std::uint32_t* const decoded =
        b_or_c.weights +
        local_output_column * kSharedWeightWordsPerRow +
        vector_in_row *
            (kValuesPerWeightVector / kBf16ValuesPerWeightWord);
    decode_nvfp4x8_to_bf16x8(packed.x, products0, decoded);
    decode_nvfp4x8_to_bf16x8(packed.y, products0, decoded + 4U);
    decode_nvfp4x8_to_bf16x8(packed.z, products1, decoded + 8U);
    decode_nvfp4x8_to_bf16x8(packed.w, products1, decoded + 12U);
    __syncthreads();

    const auto* const shared_a =
        reinterpret_cast<const __nv_bfloat16*>(shared_activations);
    const auto* const shared_b =
        reinterpret_cast<const __nv_bfloat16*>(b_or_c.weights);
#pragma unroll 1
    for (unsigned int inner_k = 0U; inner_k < kColumnsPerStage;
         inner_k += 16U) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          activation_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::col_major>
          weight_fragment;
      wmma::load_matrix_sync(activation_fragment, shared_a + inner_k,
                             kSharedLeadingDimension);
      wmma::load_matrix_sync(
          weight_fragment,
          shared_b +
              warp * kOutputColumnsPerWarp * kSharedLeadingDimension +
              inner_k,
          kSharedLeadingDimension);
      wmma::mma_sync(accumulator, activation_fragment, weight_fragment,
                     accumulator);
    }
    __syncthreads();
  }

  wmma::store_matrix_sync(
      b_or_c.output + warp * kOutputColumnsPerWarp, accumulator,
      kOutputColumnsPerBlock, wmma::mem_row_major);
  __syncthreads();

#pragma unroll
  for (unsigned int index = thread; index < kSharedOutputCount;
       index += kThreads) {
    const unsigned int token = index / kOutputColumnsPerBlock;
    const unsigned int local_column = index % kOutputColumnsPerBlock;
    output[static_cast<std::size_t>(token) * kRows + first_output_column +
           local_column] =
        encode_bf16_rne(b_or_c.output[index] * weight_scale_2);
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
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();
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
      // Two aligned 64-bit loads replace eight scalar BF16 loads while the
      // half/value traversal keeps each accumulator's K order unchanged.
#pragma unroll
      for (unsigned int half = 0U; half < 2U; ++half) {
        const std::uint64_t packed_activation =
            *reinterpret_cast<const std::uint64_t*>(
                activation + first_column + half * 4U);
#pragma unroll
        for (unsigned int value = 0U; value < 4U; ++value) {
          const unsigned int packed_value = half * 4U + value;
          const std::uint8_t nibble = static_cast<std::uint8_t>(
              (packed >> (packed_value * 4U)) & 0x0fU);
          const std::uint16_t encoded_activation =
              static_cast<std::uint16_t>(
                  (packed_activation >> (value * 16U)) & 0xffffU);
          const float scaled_weight = decoded_weights[nibble] * block_scale;
          accumulators[value] =
              fmaf(scaled_weight, decode_bf16(encoded_activation),
                   accumulators[value]);
        }
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

// M=1 vector scale-codebook path. The 256 threads cooperatively decode the
// E4M3FN codebook while the first 16 also initialize the E2M1 codebook, so
// both tables are made visible by the vector kernel's existing barrier.
__global__ __launch_bounds__(kThreads) void
nvfp4_w4a16_gemv_bf16_scale_codebook_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  decoded_scales[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();
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

    for (std::size_t packed_column =
             static_cast<std::size_t>(lane) *
             kNvFp4VectorPackedBytesPerLane;
         packed_column < packed_columns;
         packed_column += kNvFp4VectorColumnsPerWarp /
                          kNvFp4ValuesPerByte) {
      float block_scale = 0.0F;
      if ((lane & 1U) == 0U) {
        block_scale = decoded_scales[
            row_scales[packed_column / kNvFp4PackedValuesPerScale]];
      }
      block_scale = __shfl_sync(0xffff'ffffU, block_scale,
                                static_cast<int>(lane & ~1U));

      const std::uint32_t packed =
          *reinterpret_cast<const std::uint32_t*>(row_weights +
                                                  packed_column);
      const std::size_t first_column =
          packed_column * kNvFp4ValuesPerByte;
#pragma unroll
      for (unsigned int half = 0U; half < 2U; ++half) {
        const std::uint64_t packed_activation =
            *reinterpret_cast<const std::uint64_t*>(
                activation + first_column + half * 4U);
#pragma unroll
        for (unsigned int value = 0U; value < 4U; ++value) {
          const unsigned int packed_value = half * 4U + value;
          const std::uint8_t nibble = static_cast<std::uint8_t>(
              (packed >> (packed_value * 4U)) & 0x0fU);
          const std::uint16_t encoded_activation =
              static_cast<std::uint16_t>(
                  (packed_activation >> (value * 16U)) & 0xffffU);
          const float scaled_weight = decoded_weights[nibble] * block_scale;
          accumulators[value] =
              fmaf(scaled_weight, decode_bf16(encoded_activation),
                   accumulators[value]);
        }
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
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();
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
          const float scaled_weight = decoded_weights[nibble] * block_scale;
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

// M=2 scale-codebook path. All 256 threads initialize one E4M3FN entry
// while the first 16 initialize E2M1, reusing the baseline kernel's barrier.
// Activation loads, per-token FMA order, and reductions remain unchanged.
__global__ __launch_bounds__(kThreads) void
nvfp4_w4a16_small_m2_gemm_bf16_scale_codebook_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  constexpr std::size_t kTokenCount = 2U;
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  decoded_scales[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();
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
    float accumulators[kTokenCount]{};

    for (std::size_t packed_column =
             static_cast<std::size_t>(lane) *
             kNvFp4VectorPackedBytesPerLane;
         packed_column < packed_columns;
         packed_column += kNvFp4VectorColumnsPerWarp /
                          kNvFp4ValuesPerByte) {
      float block_scale = 0.0F;
      if ((lane & 1U) == 0U) {
        block_scale = decoded_scales[
            row_scales[packed_column / kNvFp4PackedValuesPerScale]];
      }
      block_scale = __shfl_sync(0xffff'ffffU, block_scale,
                                static_cast<int>(lane & ~1U));
      const std::uint32_t packed =
          *reinterpret_cast<const std::uint32_t*>(row_weights +
                                                  packed_column);
      const std::size_t first_column =
          packed_column * kNvFp4ValuesPerByte;

#pragma unroll
      for (unsigned int half = 0U; half < 2U; ++half) {
        std::uint64_t packed_activations[kTokenCount];
#pragma unroll
        for (unsigned int token = 0U; token < kTokenCount; ++token) {
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
          const float scaled_weight = decoded_weights[nibble] * block_scale;
#pragma unroll
          for (unsigned int token = 0U; token < kTokenCount; ++token) {
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
    for (unsigned int token = 0U; token < kTokenCount; ++token) {
      const float sum = warp_sum(accumulators[token]) * weight_scale_2;
      if (lane == 0U) {
        output[static_cast<std::size_t>(token) * rows + row] =
            encode_bf16_rne(sum);
      }
    }
  }
}

// Preserved M=8 row-pair baseline without the E4M3FN scale codebook. A warp
// evaluates two adjacent output rows so
// token-major BF16 activation words are loaded and decoded once, then reused
// with the independent packed weights and block scales for both rows. The
// four-block launch bound keeps the paired accumulator tile spill-free on
// SM87. This remains available to mirrored performance tests.
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_small_m8_gemm_bf16_row_pair_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 8U;
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

  const std::size_t packed_columns = columns / kNvFp4ValuesPerByte;
  const std::size_t scale_columns = columns / kNvFp4GroupSize;
  const std::size_t first_row =
      2U * (static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp);
  const std::size_t row_stride =
      static_cast<std::size_t>(gridDim.x) * kWarpsPerBlock * 2U;

  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row1 = row0 + 1U;
    const bool has_row1 = row1 < rows;
    const std::uint8_t* const row0_weights =
        packed_weights + row0 * packed_columns;
    const std::uint8_t* const row0_scales =
        block_scales + row0 * scale_columns;
    const std::uint8_t* const row1_weights =
        has_row1 ? packed_weights + row1 * packed_columns : row0_weights;
    const std::uint8_t* const row1_scales =
        has_row1 ? block_scales + row1 * scale_columns : row0_scales;
    float accumulators0[kTokenCount]{};
    float accumulators1[kTokenCount]{};

    for (std::size_t packed_column =
             static_cast<std::size_t>(lane) *
             kNvFp4VectorPackedBytesPerLane;
         packed_column < packed_columns;
         packed_column += kNvFp4VectorColumnsPerWarp /
                          kNvFp4ValuesPerByte) {
      float block_scale0 = 0.0F;
      float block_scale1 = 0.0F;
      if ((lane & 1U) == 0U) {
        const std::size_t scale_column =
            packed_column / kNvFp4PackedValuesPerScale;
        block_scale0 = decode_e4m3fn(row0_scales[scale_column]);
        if (has_row1) {
          block_scale1 = decode_e4m3fn(row1_scales[scale_column]);
        }
      }
      const int scale_source = static_cast<int>(lane & ~1U);
      block_scale0 =
          __shfl_sync(0xffff'ffffU, block_scale0, scale_source);
      block_scale1 =
          __shfl_sync(0xffff'ffffU, block_scale1, scale_source);

      const std::uint32_t packed0 =
          *reinterpret_cast<const std::uint32_t*>(row0_weights +
                                                  packed_column);
      const std::uint32_t packed1 =
          has_row1 ? *reinterpret_cast<const std::uint32_t*>(
                         row1_weights + packed_column)
                   : 0U;
      const std::size_t first_column =
          packed_column * kNvFp4ValuesPerByte;

#pragma unroll
      for (unsigned int half = 0U; half < 2U; ++half) {
        std::uint64_t packed_activations[kTokenCount];
#pragma unroll
        for (unsigned int token = 0U; token < kTokenCount; ++token) {
          packed_activations[token] =
              *reinterpret_cast<const std::uint64_t*>(
                  activations + static_cast<std::size_t>(token) * columns +
                  first_column + half * 4U);
        }
#pragma unroll
        for (unsigned int value = 0U; value < 4U; ++value) {
          const unsigned int packed_value = half * 4U + value;
          const unsigned int shift = packed_value * 4U;
          const std::uint8_t nibble0 = static_cast<std::uint8_t>(
              (packed0 >> shift) & 0x0fU);
          const std::uint8_t nibble1 = static_cast<std::uint8_t>(
              (packed1 >> shift) & 0x0fU);
          const float scaled_weight0 =
              decoded_weights[nibble0] * block_scale0;
          const float scaled_weight1 =
              decoded_weights[nibble1] * block_scale1;
#pragma unroll
          for (unsigned int token = 0U; token < kTokenCount; ++token) {
            const std::uint16_t encoded_activation =
                static_cast<std::uint16_t>(
                    (packed_activations[token] >> (value * 16U)) & 0xffffU);
            const float decoded_activation = decode_bf16(encoded_activation);
            accumulators0[token] =
                fmaf(scaled_weight0, decoded_activation,
                     accumulators0[token]);
            accumulators1[token] =
                fmaf(scaled_weight1, decoded_activation,
                     accumulators1[token]);
          }
        }
      }
    }

#pragma unroll
    for (unsigned int token = 0U; token < kTokenCount; ++token) {
      const float sum0 = warp_sum(accumulators0[token]) * weight_scale_2;
      const float sum1 = warp_sum(accumulators1[token]) * weight_scale_2;
      if (lane == 0U) {
        output[static_cast<std::size_t>(token) * rows + row0] =
            encode_bf16_rne(sum0);
        if (has_row1) {
          output[static_cast<std::size_t>(token) * rows + row1] =
              encode_bf16_rne(sum1);
        }
      }
    }
  }
}

// Production M=8 row-pair path. In addition to the existing 16-entry
// E2M1 table, each block decodes the complete E4M3FN scale code space once.
// Inner-loop row scale decoding then becomes a shared lookup. Both fixed MLP
// shapes cleared the mixed-scale mirrored performance gate.
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_small_m8_gemm_bf16_scale_codebook_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 8U;
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  decoded_scales[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  __syncthreads();

  const std::size_t packed_columns = columns / kNvFp4ValuesPerByte;
  const std::size_t scale_columns = columns / kNvFp4GroupSize;
  const std::size_t first_row =
      2U * (static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp);
  const std::size_t row_stride =
      static_cast<std::size_t>(gridDim.x) * kWarpsPerBlock * 2U;

  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row1 = row0 + 1U;
    const bool has_row1 = row1 < rows;
    const std::uint8_t* const row0_weights =
        packed_weights + row0 * packed_columns;
    const std::uint8_t* const row0_scales =
        block_scales + row0 * scale_columns;
    const std::uint8_t* const row1_weights =
        has_row1 ? packed_weights + row1 * packed_columns : row0_weights;
    const std::uint8_t* const row1_scales =
        has_row1 ? block_scales + row1 * scale_columns : row0_scales;
    float accumulators0[kTokenCount]{};
    float accumulators1[kTokenCount]{};

    for (std::size_t packed_column =
             static_cast<std::size_t>(lane) *
             kNvFp4VectorPackedBytesPerLane;
         packed_column < packed_columns;
         packed_column += kNvFp4VectorColumnsPerWarp /
                          kNvFp4ValuesPerByte) {
      float block_scale0 = 0.0F;
      float block_scale1 = 0.0F;
      if ((lane & 1U) == 0U) {
        const std::size_t scale_column =
            packed_column / kNvFp4PackedValuesPerScale;
        block_scale0 = decoded_scales[row0_scales[scale_column]];
        if (has_row1) {
          block_scale1 = decoded_scales[row1_scales[scale_column]];
        }
      }
      const int scale_source = static_cast<int>(lane & ~1U);
      block_scale0 =
          __shfl_sync(0xffff'ffffU, block_scale0, scale_source);
      block_scale1 =
          __shfl_sync(0xffff'ffffU, block_scale1, scale_source);

      const std::uint32_t packed0 =
          *reinterpret_cast<const std::uint32_t*>(row0_weights +
                                                  packed_column);
      const std::uint32_t packed1 =
          has_row1 ? *reinterpret_cast<const std::uint32_t*>(
                         row1_weights + packed_column)
                   : 0U;
      const std::size_t first_column =
          packed_column * kNvFp4ValuesPerByte;

#pragma unroll
      for (unsigned int half = 0U; half < 2U; ++half) {
        std::uint64_t packed_activations[kTokenCount];
#pragma unroll
        for (unsigned int token = 0U; token < kTokenCount; ++token) {
          packed_activations[token] =
              *reinterpret_cast<const std::uint64_t*>(
                  activations + static_cast<std::size_t>(token) * columns +
                  first_column + half * 4U);
        }
#pragma unroll
        for (unsigned int value = 0U; value < 4U; ++value) {
          const unsigned int packed_value = half * 4U + value;
          const unsigned int shift = packed_value * 4U;
          const std::uint8_t nibble0 = static_cast<std::uint8_t>(
              (packed0 >> shift) & 0x0fU);
          const std::uint8_t nibble1 = static_cast<std::uint8_t>(
              (packed1 >> shift) & 0x0fU);
          const float scaled_weight0 =
              decoded_weights[nibble0] * block_scale0;
          const float scaled_weight1 =
              decoded_weights[nibble1] * block_scale1;
#pragma unroll
          for (unsigned int token = 0U; token < kTokenCount; ++token) {
            const std::uint16_t encoded_activation =
                static_cast<std::uint16_t>(
                    (packed_activations[token] >> (value * 16U)) & 0xffffU);
            const float decoded_activation = decode_bf16(encoded_activation);
            accumulators0[token] =
                fmaf(scaled_weight0, decoded_activation,
                     accumulators0[token]);
            accumulators1[token] =
                fmaf(scaled_weight1, decoded_activation,
                     accumulators1[token]);
          }
        }
      }
    }

#pragma unroll
    for (unsigned int token = 0U; token < kTokenCount; ++token) {
      const float sum0 = warp_sum(accumulators0[token]) * weight_scale_2;
      const float sum1 = warp_sum(accumulators1[token]) * weight_scale_2;
      if (lane == 0U) {
        output[static_cast<std::size_t>(token) * rows + row0] =
            encode_bf16_rne(sum0);
        if (has_row1) {
          output[static_cast<std::size_t>(token) * rows + row1] =
              encode_bf16_rne(sum1);
        }
      }
    }
  }
}

// Production-shape specializations. Rows and columns are compile-time
// constants, both real shapes contain a whole number of row-pair blocks, and
// therefore every launched warp owns exactly one valid row pair.
template <std::size_t kRows, std::size_t kColumns>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_small_m8_gemm_bf16_fixed_shape_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 8U;
  constexpr unsigned int kPackedColumns =
      static_cast<unsigned int>(kColumns / kNvFp4ValuesPerByte);
  constexpr unsigned int kScaleColumns =
      static_cast<unsigned int>(kColumns / kNvFp4GroupSize);
  constexpr unsigned int kPackedColumnStep =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  static_assert(kRows % (kWarpsPerBlock * 2U) == 0U);
  static_assert(kColumns % kNvFp4VectorColumnsPerWarp == 0U);

  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  decoded_scales[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  __syncthreads();

  const std::size_t row0 =
      2U * (static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp);
  const std::size_t row1 = row0 + 1U;
  const std::uint8_t* const row0_weights =
      packed_weights + row0 * kPackedColumns;
  const std::uint8_t* const row0_scales =
      block_scales + row0 * kScaleColumns;
  const std::uint8_t* const row1_weights =
      packed_weights + row1 * kPackedColumns;
  const std::uint8_t* const row1_scales =
      block_scales + row1 * kScaleColumns;
  float accumulators0[kTokenCount]{};
  float accumulators1[kTokenCount]{};

  for (unsigned int packed_column =
           lane * kNvFp4VectorPackedBytesPerLane;
       packed_column < kPackedColumns;
       packed_column += kPackedColumnStep) {
    float block_scale0 = 0.0F;
    float block_scale1 = 0.0F;
    if ((lane & 1U) == 0U) {
      const unsigned int scale_column =
          packed_column / kNvFp4PackedValuesPerScale;
      block_scale0 = decoded_scales[row0_scales[scale_column]];
      block_scale1 = decoded_scales[row1_scales[scale_column]];
    }
    const int scale_source = static_cast<int>(lane & ~1U);
    block_scale0 =
        __shfl_sync(0xffff'ffffU, block_scale0, scale_source);
    block_scale1 =
        __shfl_sync(0xffff'ffffU, block_scale1, scale_source);

    const std::uint32_t packed0 =
        *reinterpret_cast<const std::uint32_t*>(row0_weights +
                                                packed_column);
    const std::uint32_t packed1 =
        *reinterpret_cast<const std::uint32_t*>(row1_weights +
                                                packed_column);
    const unsigned int first_column =
        packed_column * kNvFp4ValuesPerByte;

#pragma unroll
    for (unsigned int half = 0U; half < 2U; ++half) {
      std::uint64_t packed_activations[kTokenCount];
#pragma unroll
      for (unsigned int token = 0U; token < kTokenCount; ++token) {
        packed_activations[token] =
            *reinterpret_cast<const std::uint64_t*>(
                activations + static_cast<std::size_t>(token) * kColumns +
                first_column + half * 4U);
      }
#pragma unroll
      for (unsigned int value = 0U; value < 4U; ++value) {
        const unsigned int packed_value = half * 4U + value;
        const unsigned int shift = packed_value * 4U;
        const std::uint8_t nibble0 = static_cast<std::uint8_t>(
            (packed0 >> shift) & 0x0fU);
        const std::uint8_t nibble1 = static_cast<std::uint8_t>(
            (packed1 >> shift) & 0x0fU);
        const float scaled_weight0 =
            decoded_weights[nibble0] * block_scale0;
        const float scaled_weight1 =
            decoded_weights[nibble1] * block_scale1;
#pragma unroll
        for (unsigned int token = 0U; token < kTokenCount; ++token) {
          const std::uint16_t encoded_activation =
              static_cast<std::uint16_t>(
                  (packed_activations[token] >> (value * 16U)) & 0xffffU);
          const float decoded_activation = decode_bf16(encoded_activation);
          accumulators0[token] =
              fmaf(scaled_weight0, decoded_activation, accumulators0[token]);
          accumulators1[token] =
              fmaf(scaled_weight1, decoded_activation, accumulators1[token]);
        }
      }
    }
  }

#pragma unroll
  for (unsigned int token = 0U; token < kTokenCount; ++token) {
    const float sum0 = warp_sum(accumulators0[token]) * weight_scale_2;
    const float sum1 = warp_sum(accumulators1[token]) * weight_scale_2;
    if (lane == 0U) {
      output[static_cast<std::size_t>(token) * kRows + row0] =
          encode_bf16_rne(sum0);
      output[static_cast<std::size_t>(token) * kRows + row1] =
          encode_bf16_rne(sum1);
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

void launch_fp8_vector_grid_cap_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  const unsigned int uncapped = block_count_for_single_row(rows);
  const unsigned int blocks =
      uncapped < maximum_blocks ? uncapped : maximum_blocks;
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

void launch_nvfp4_scale_codebook_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  const unsigned int blocks = block_count_for_rows(rows);
  nvfp4_w4a16_gemv_bf16_scale_codebook_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activation, rows,
          columns, output);
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

[[nodiscard]] int validate_fp8_m16_launch(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) noexcept {
  constexpr std::size_t kTokenCount = 16U;
  if (!std::isfinite(weight_scale) || weight_scale < 0.0F ||
      multiply_overflows(rows, columns)) {
    return invalid_value();
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (multiply_overflows(kTokenCount, columns) ||
      multiply_overflows(kTokenCount, rows)) {
    return invalid_value();
  }
  const std::size_t activation_elements = kTokenCount * columns;
  const std::size_t output_elements = kTokenCount * rows;
  if (multiply_overflows(activation_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(output_elements, sizeof(std::uint16_t)) ||
      weights == nullptr || activations == nullptr || output == nullptr) {
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

[[nodiscard]] int validate_nvfp4_m16_launch(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) noexcept {
  constexpr std::size_t kTokenCount = 16U;
  if ((columns % kNvFp4GroupSize) != 0U ||
      !std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F ||
      multiply_overflows(rows, columns)) {
    return invalid_value();
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (multiply_overflows(kTokenCount, columns) ||
      multiply_overflows(kTokenCount, rows)) {
    return invalid_value();
  }
  const std::size_t activation_elements = kTokenCount * columns;
  const std::size_t output_elements = kTokenCount * rows;
  if (multiply_overflows(activation_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(output_elements, sizeof(std::uint16_t)) ||
      packed_weights == nullptr || block_scales == nullptr ||
      activations == nullptr || output == nullptr) {
    return invalid_value();
  }

  const std::size_t packed_bytes =
      rows * (columns / kNvFp4ValuesPerByte);
  const std::size_t scale_bytes = rows * (columns / kNvFp4GroupSize);
  const std::size_t activation_bytes =
      activation_elements * sizeof(std::uint16_t);
  const std::size_t output_bytes =
      output_elements * sizeof(std::uint16_t);
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
void launch_fp8_small_m_vector_grid_cap_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  const unsigned int uncapped = block_count_for_single_row(rows);
  const unsigned int blocks =
      uncapped < maximum_blocks ? uncapped : maximum_blocks;
  fp8_w8a16_small_m_gemm_bf16_vector_kernel<TokenCount>
      <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activations,
                                        rows, columns, output);
}

void launch_fp8_small_m8_row_pair_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = 2U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < kMaximumBlocks ? wanted : kMaximumBlocks);
  fp8_w8a16_small_m8_gemm_bf16_row_pair_kernel
      <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activations,
                                        rows, columns, output);
}

template <std::size_t kRows, std::size_t kColumns>
void launch_fp8_small_m8_fixed_shape_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / 2U);
  fp8_w8a16_small_m8_gemm_bf16_fixed_shape_kernel<kRows, kColumns>
      <<<kBlocks, kThreads, 0U, stream>>>(weights, weight_scale, activations,
                                         output);
}

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 64U>
void launch_fp8_small_m16_wmma_fixed_shape_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  fp8_w8a16_small_m16_gemm_bf16_wmma_fixed_shape_kernel<
      kRows, kColumns, kSharedLeadingDimension>
      <<<kBlocks, kThreads, 0U, stream>>>(weights, weight_scale, activations,
                                         output);
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

void launch_nvfp4_small_m2_scale_codebook_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  const unsigned int blocks = block_count_for_rows(rows);
  nvfp4_w4a16_small_m2_gemm_bf16_scale_codebook_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, rows,
          columns, output);
}

void launch_nvfp4_small_m8_row_pair_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = kWarpsPerBlock * 2U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < kMaximumBlocks ? wanted : kMaximumBlocks);
  nvfp4_w4a16_small_m8_gemm_bf16_row_pair_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, rows,
          columns, output);
}

void launch_nvfp4_small_m8_scale_codebook_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = kWarpsPerBlock * 2U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < kMaximumBlocks ? wanted : kMaximumBlocks);
  nvfp4_w4a16_small_m8_gemm_bf16_scale_codebook_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, rows,
          columns, output);
}

template <std::size_t kRows, std::size_t kColumns>
void launch_nvfp4_small_m8_fixed_shape_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / (kWarpsPerBlock * 2U));
  nvfp4_w4a16_small_m8_gemm_bf16_fixed_shape_kernel<kRows, kColumns>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output);
}

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
void launch_nvfp4_small_m16_wmma_fixed_shape_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m16_gemm_bf16_wmma_fixed_shape_kernel<
      kRows, kColumns, kSharedLeadingDimension>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output);
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
    if (use_fp8_m1_persistent_rows(rows)) {
      launch_fp8_vector_grid_cap_unchecked(
          weights, weight_scale, activation, rows, columns, output,
          kFp8M1PersistentMaximumBlocks, stream);
    } else {
      launch_fp8_vector_unchecked(weights, weight_scale, activation, rows,
                                  columns, output, stream);
    }
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

[[nodiscard]] bool use_sm87_fp8_m1_persistent_rows_test(
    const std::size_t rows) noexcept {
  return use_fp8_m1_persistent_rows(rows);
}

// Test-only uncapped vector baseline retained even if production later adopts
// persistent row blocks. It is intentionally omitted from the public header.
int launch_sm87_fp8_w8a16_gemv_bf16_vector_uncapped_test_cuda(
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
  const bool vector_shape =
      (columns % kFp8VectorColumnsPerBlock) == 0U &&
      (reinterpret_cast<std::uintptr_t>(weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_vector_unchecked(weights, weight_scale, activation, rows, columns,
                              output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_gemv_bf16_grid_cap_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_launch(
      weights, weight_scale, activation, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  if (maximum_blocks == 0U || maximum_blocks > kMaximumBlocks) {
    return invalid_value();
  }
  const bool vector_shape =
      (columns % kFp8VectorColumnsPerBlock) == 0U &&
      (reinterpret_cast<std::uintptr_t>(weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_vector_grid_cap_unchecked(
      weights, weight_scale, activation, rows, columns, output,
      static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

// Test-only query sharing the exact predicate used by production dispatch.
[[nodiscard]] bool use_sm87_fp8_small_m_row_pair_test(
    const std::size_t token_count, const std::size_t rows) noexcept {
  return use_fp8_small_m_row_pair(token_count, rows);
}

[[nodiscard]] bool use_sm87_fp8_m8_fixed_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_fp8_m8_fixed_shape(rows, columns);
}

[[nodiscard]] bool use_sm87_fp8_m16_wmma_fixed_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_fp8_m16_wmma_fixed_shape(rows, columns);
}

// Test-only direct entry points keep the preserved M=8 implementation and the
// row-pair candidate in the same cubin for bitwise and mirrored event A/B.
int launch_sm87_fp8_w8a16_small_m8_single_row_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 8U;
  const int validation = validate_fp8_small_m_launch(
      weights, weight_scale, activations, kTokenCount, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const bool vector_shape =
      (columns % kFp8VectorColumnsPerBlock) == 0U &&
      (reinterpret_cast<std::uintptr_t>(weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_small_m_vector_unchecked<kTokenCount>(
      weights, weight_scale, activations, rows, columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_small_m8_row_pair_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 8U;
  const int validation = validate_fp8_small_m_launch(
      weights, weight_scale, activations, kTokenCount, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const bool vector_shape =
      (columns % kFp8VectorColumnsPerBlock) == 0U &&
      (reinterpret_cast<std::uintptr_t>(weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_small_m8_row_pair_unchecked(weights, weight_scale, activations,
                                         rows, columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_small_m8_fixed_shape_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 8U;
  const int validation = validate_fp8_small_m_launch(
      weights, weight_scale, activations, kTokenCount, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const bool vector_shape =
      (columns % kFp8VectorColumnsPerBlock) == 0U &&
      (reinterpret_cast<std::uintptr_t>(weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape || !use_fp8_m8_fixed_shape(rows, columns)) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  if (rows == 10'240U) {
    launch_fp8_small_m8_fixed_shape_unchecked<10'240U, 5'120U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 5'120U) {
    launch_fp8_small_m8_fixed_shape_unchecked<5'120U, 6'144U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 6'144U) {
    launch_fp8_small_m8_fixed_shape_unchecked<6'144U, 5'120U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 12'288U) {
    launch_fp8_small_m8_fixed_shape_unchecked<12'288U, 5'120U>(
        weights, weight_scale, activations, output, stream);
  } else {
    launch_fp8_small_m8_fixed_shape_unchecked<1'024U, 5'120U>(
        weights, weight_scale, activations, output, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

// Test-only raw C16 entry retains the rejected 1024-row WMMA characterization.
// Production uses the separate public M16 wrapper and its two-M8 fallback;
// the existing variable small-M API remains capped at eight tokens.
int launch_sm87_fp8_w8a16_small_m16_wmma_fixed_shape_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!use_fp8_m8_fixed_shape(rows, columns)) {
    return invalid_value();
  }
  const int validation = validate_fp8_m16_launch(
      weights, weight_scale, activations, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) ||
      (reinterpret_cast<std::uintptr_t>(weights) % alignof(uint4)) != 0U ||
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) != 0U) {
    return validation != static_cast<int>(cudaSuccess) ? validation
                                                        : invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  if (rows == 10'240U) {
    launch_fp8_small_m16_wmma_fixed_shape_unchecked<10'240U, 5'120U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 5'120U) {
    launch_fp8_small_m16_wmma_fixed_shape_unchecked<5'120U, 6'144U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 6'144U) {
    launch_fp8_small_m16_wmma_fixed_shape_unchecked<6'144U, 5'120U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 12'288U) {
    launch_fp8_small_m16_wmma_fixed_shape_unchecked<12'288U, 5'120U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else {
    launch_fp8_small_m16_wmma_fixed_shape_unchecked<1'024U, 5'120U>(
        weights, weight_scale, activations, output, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

// Test-only same-cubin entry used to isolate shared-memory leading-dimension
// effects on the three promoted K=5120 projections. Keeping both variants in
// one cubin avoids comparing builds with different layout or link decisions.
int launch_sm87_fp8_w8a16_small_m16_wmma_shared_ldm_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const std::size_t shared_leading_dimension,
    void* const cuda_stream) noexcept {
  if (columns != 5'120U ||
      !use_fp8_m16_wmma_fixed_shape(rows, columns) ||
      (shared_leading_dimension != 64U &&
       shared_leading_dimension != 72U)) {
    return invalid_value();
  }
  const int validation = validate_fp8_m16_launch(
      weights, weight_scale, activations, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) ||
      (reinterpret_cast<std::uintptr_t>(weights) % alignof(uint4)) != 0U ||
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) != 0U) {
    return validation != static_cast<int>(cudaSuccess) ? validation
                                                        : invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  if (shared_leading_dimension == 64U) {
    if (rows == 10'240U) {
      launch_fp8_small_m16_wmma_fixed_shape_unchecked<10'240U, 5'120U, 64U>(
          weights, weight_scale, activations, output, stream);
    } else if (rows == 6'144U) {
      launch_fp8_small_m16_wmma_fixed_shape_unchecked<6'144U, 5'120U, 64U>(
          weights, weight_scale, activations, output, stream);
    } else {
      launch_fp8_small_m16_wmma_fixed_shape_unchecked<12'288U, 5'120U, 64U>(
          weights, weight_scale, activations, output, stream);
    }
  } else if (rows == 10'240U) {
    launch_fp8_small_m16_wmma_fixed_shape_unchecked<10'240U, 5'120U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 6'144U) {
    launch_fp8_small_m16_wmma_fixed_shape_unchecked<6'144U, 5'120U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else {
    launch_fp8_small_m16_wmma_fixed_shape_unchecked<12'288U, 5'120U, 72U>(
        weights, weight_scale, activations, output, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

// Test-only preserved uncapped M=2 launcher and same-cubin grid-cap candidate.
// These stay outside the public header so a sweep cannot alter production
// dispatch before every correctness and performance gate has passed.
int launch_sm87_fp8_w8a16_small_m2_uncapped_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 2U;
  const int validation = validate_fp8_small_m_launch(
      weights, weight_scale, activations, kTokenCount, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const bool vector_shape =
      (columns % kFp8VectorColumnsPerBlock) == 0U &&
      (reinterpret_cast<std::uintptr_t>(weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_small_m_vector_unchecked<kTokenCount>(
      weights, weight_scale, activations, rows, columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_small_m2_grid_cap_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 2U;
  const int validation = validate_fp8_small_m_launch(
      weights, weight_scale, activations, kTokenCount, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  if (maximum_blocks == 0U || maximum_blocks > kMaximumBlocks) {
    return invalid_value();
  }
  const bool vector_shape =
      (columns % kFp8VectorColumnsPerBlock) == 0U &&
      (reinterpret_cast<std::uintptr_t>(weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_small_m_vector_grid_cap_unchecked<kTokenCount>(
      weights, weight_scale, activations, rows, columns, output,
      static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

[[nodiscard]] bool use_sm87_fp8_m2_persistent_rows_test(
    const std::size_t rows) noexcept {
  return use_fp8_m2_persistent_rows(rows);
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
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U;
  if (vector_shape && use_nvfp4_m1_scale_codebook(rows, columns)) {
    launch_nvfp4_scale_codebook_unchecked(
        packed_weights, block_scales, weight_scale_2, activation, rows,
        columns, output, stream);
  } else if (vector_shape) {
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

// Test-only preserved M=1 vector baseline and shared-scale candidate. Both
// reject non-vector shapes so mirrored CUDA-event comparisons cannot silently
// measure different dispatch fallbacks.
int launch_sm87_nvfp4_w4a16_gemv_bf16_vector_test_cuda(
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
  const bool vector_shape =
      (columns % kNvFp4VectorColumnsPerWarp) == 0U &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_vector_unchecked(packed_weights, block_scales, weight_scale_2,
                                activation, rows, columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_test_cuda(
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
  const bool vector_shape =
      (columns % kNvFp4VectorColumnsPerWarp) == 0U &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_scale_codebook_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

// Test-only query sharing the exact M=1 production gate.
[[nodiscard]] bool use_sm87_nvfp4_m1_scale_codebook_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m1_scale_codebook(rows, columns);
}

// Test-only direct M=2 entries preserve the current vector baseline and keep
// the scale-codebook candidate in the same cubin for bitwise and event A/B.
int launch_sm87_nvfp4_w4a16_small_m2_vector_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 2U;
  const int validation = validate_nvfp4_small_m_launch(
      packed_weights, block_scales, weight_scale_2, activations, kTokenCount,
      rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const bool vector_shape =
      (columns % kNvFp4VectorColumnsPerWarp) == 0U &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_small_m_vector_unchecked<kTokenCount>(
      packed_weights, block_scales, weight_scale_2, activations, rows,
      columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 2U;
  const int validation = validate_nvfp4_small_m_launch(
      packed_weights, block_scales, weight_scale_2, activations, kTokenCount,
      rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const bool vector_shape =
      (columns % kNvFp4VectorColumnsPerWarp) == 0U &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_small_m2_scale_codebook_unchecked(
      packed_weights, block_scales, weight_scale_2, activations, rows,
      columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

[[nodiscard]] bool use_sm87_nvfp4_m2_scale_codebook_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m2_scale_codebook(rows, columns);
}

// Test-only query sharing the exact predicate used by production dispatch.
[[nodiscard]] bool use_sm87_nvfp4_small_m_row_pair_test(
    const std::size_t token_count, const std::size_t rows) noexcept {
  return use_nvfp4_small_m_row_pair(token_count, rows);
}

[[nodiscard]] bool use_sm87_nvfp4_m8_fixed_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m8_fixed_shape(rows, columns);
}

[[nodiscard]] bool use_sm87_nvfp4_m16_wmma_fixed_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m16_wmma_fixed_shape(rows, columns);
}

// Test-only raw exact-shape C16 entry for architecture-level validation. The
// public M16 launcher below owns production eligibility and safe fallback.
int launch_sm87_nvfp4_w4a16_small_m16_wmma_fixed_shape_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns)) {
    return invalid_value();
  }
  const int validation = validate_nvfp4_m16_launch(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) % alignof(uint4)) ==
          0U &&
      (reinterpret_cast<std::uintptr_t>(block_scales) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!aligned) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  if (rows == 17'408U) {
    launch_nvfp4_small_m16_wmma_fixed_shape_unchecked<17'408U, 5'120U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  } else {
    launch_nvfp4_small_m16_wmma_fixed_shape_unchecked<5'120U, 17'408U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  }
  return static_cast<int>(cudaGetLastError());
}

// Test-only direct entry for the preserved row-pair baseline without the
// shared E4M3FN scale codebook.
int launch_sm87_nvfp4_w4a16_small_m8_row_pair_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 8U;
  const int validation = validate_nvfp4_small_m_launch(
      packed_weights, block_scales, weight_scale_2, activations, kTokenCount,
      rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const bool vector_shape =
      (columns % kNvFp4VectorColumnsPerWarp) == 0U &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_small_m8_row_pair_unchecked(
      packed_weights, block_scales, weight_scale_2, activations, rows,
      columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

// Test-only direct entry for the production shared E4M3FN scale codebook.
int launch_sm87_nvfp4_w4a16_small_m8_scale_codebook_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 8U;
  const int validation = validate_nvfp4_small_m_launch(
      packed_weights, block_scales, weight_scale_2, activations, kTokenCount,
      rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const bool vector_shape =
      (columns % kNvFp4VectorColumnsPerWarp) == 0U &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_small_m8_scale_codebook_unchecked(
      packed_weights, block_scales, weight_scale_2, activations, rows,
      columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

// Test-only preserved single-row M=8 baseline. Production dispatch uses the
// scale-codebook row-pair kernel after both fixed MLP shapes cleared the gate.
int launch_sm87_nvfp4_w4a16_small_m8_single_row_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 8U;
  const int validation = validate_nvfp4_small_m_launch(
      packed_weights, block_scales, weight_scale_2, activations, kTokenCount,
      rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  const bool vector_shape =
      (columns % kNvFp4VectorColumnsPerWarp) == 0U &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_small_m_vector_unchecked<kTokenCount>(
      packed_weights, block_scales, weight_scale_2, activations, rows,
      columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kHalfTokens = 8U;
  const int validation = validate_fp8_m16_launch(
      weights, weight_scale, activations, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }

  const bool wmma_shape =
      use_fp8_m16_wmma_fixed_shape(rows, columns) &&
      (reinterpret_cast<std::uintptr_t>(weights) % alignof(uint4)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (wmma_shape) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    if (rows == 10'240U) {
      launch_fp8_small_m16_wmma_fixed_shape_unchecked<10'240U, 5'120U, 72U>(
          weights, weight_scale, activations, output, stream);
    } else if (rows == 5'120U) {
      launch_fp8_small_m16_wmma_fixed_shape_unchecked<5'120U, 6'144U, 72U>(
          weights, weight_scale, activations, output, stream);
    } else if (rows == 6'144U) {
      launch_fp8_small_m16_wmma_fixed_shape_unchecked<6'144U, 5'120U, 72U>(
          weights, weight_scale, activations, output, stream);
    } else {
      launch_fp8_small_m16_wmma_fixed_shape_unchecked<12'288U, 5'120U, 72U>(
          weights, weight_scale, activations, output, stream);
    }
    return static_cast<int>(cudaGetLastError());
  }

  int status = launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
      weights, weight_scale, activations, kHalfTokens, rows, columns, output,
      cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
      weights, weight_scale, activations + kHalfTokens * columns, kHalfTokens,
      rows, columns, output + kHalfTokens * rows, cuda_stream);
  return status;
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
        if (use_fp8_m2_persistent_rows(rows)) {
          launch_fp8_small_m_vector_grid_cap_unchecked<2U>(
              weights, weight_scale, activations, rows, columns, output,
              kFp8M2PersistentMaximumBlocks, stream);
        } else {
          launch_fp8_small_m_vector_unchecked<2U>(
              weights, weight_scale, activations, rows, columns, output,
              stream);
        }
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
        if (use_fp8_m8_fixed_shape(rows, columns)) {
          if (rows == 10'240U) {
            launch_fp8_small_m8_fixed_shape_unchecked<10'240U, 5'120U>(
                weights, weight_scale, activations, output, stream);
          } else if (rows == 5'120U) {
            launch_fp8_small_m8_fixed_shape_unchecked<5'120U, 6'144U>(
                weights, weight_scale, activations, output, stream);
          } else if (rows == 6'144U) {
            launch_fp8_small_m8_fixed_shape_unchecked<6'144U, 5'120U>(
                weights, weight_scale, activations, output, stream);
          } else if (rows == 12'288U) {
            launch_fp8_small_m8_fixed_shape_unchecked<12'288U, 5'120U>(
                weights, weight_scale, activations, output, stream);
          } else {
            launch_fp8_small_m8_fixed_shape_unchecked<1'024U, 5'120U>(
                weights, weight_scale, activations, output, stream);
          }
        } else if (use_fp8_small_m_row_pair(token_count, rows)) {
          launch_fp8_small_m8_row_pair_unchecked(
              weights, weight_scale, activations, rows, columns, output,
              stream);
        } else {
          launch_fp8_small_m_vector_unchecked<8U>(
              weights, weight_scale, activations, rows, columns, output,
              stream);
        }
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

int launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kHalfTokens = 8U;
  const int validation = validate_nvfp4_m16_launch(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }

  const bool wmma_shape =
      use_nvfp4_m16_wmma_fixed_shape(rows, columns) &&
      (reinterpret_cast<std::uintptr_t>(packed_weights) % alignof(uint4)) ==
          0U &&
      (reinterpret_cast<std::uintptr_t>(block_scales) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) == 0U;
  if (wmma_shape) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    if (rows == 17'408U) {
      launch_nvfp4_small_m16_wmma_fixed_shape_unchecked<17'408U, 5'120U>(
          packed_weights, block_scales, weight_scale_2, activations, output,
          stream);
    } else {
      launch_nvfp4_small_m16_wmma_fixed_shape_unchecked<5'120U, 17'408U>(
          packed_weights, block_scales, weight_scale_2, activations, output,
          stream);
    }
    return static_cast<int>(cudaGetLastError());
  }

  int status = launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
      packed_weights, block_scales, weight_scale_2, activations, kHalfTokens,
      rows, columns, output, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
      packed_weights, block_scales, weight_scale_2,
      activations + kHalfTokens * columns, kHalfTokens, rows, columns,
      output + kHalfTokens * rows, cuda_stream);
  return status;
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
        if (use_nvfp4_m2_scale_codebook(rows, columns)) {
          launch_nvfp4_small_m2_scale_codebook_unchecked(
              packed_weights, block_scales, weight_scale_2, activations, rows,
              columns, output, stream);
        } else {
          launch_nvfp4_small_m_vector_unchecked<2U>(
              packed_weights, block_scales, weight_scale_2, activations, rows,
              columns, output, stream);
        }
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
        if (use_nvfp4_m8_fixed_shape(rows, columns)) {
          if (rows == 17'408U) {
            launch_nvfp4_small_m8_fixed_shape_unchecked<17'408U, 5'120U>(
                packed_weights, block_scales, weight_scale_2, activations,
                output, stream);
          } else {
            launch_nvfp4_small_m8_fixed_shape_unchecked<5'120U, 17'408U>(
                packed_weights, block_scales, weight_scale_2, activations,
                output, stream);
          }
        } else if (use_nvfp4_small_m_row_pair(token_count, rows)) {
          launch_nvfp4_small_m8_scale_codebook_unchecked(
              packed_weights, block_scales, weight_scale_2, activations, rows,
              columns, output, stream);
        } else {
          launch_nvfp4_small_m_vector_unchecked<8U>(
              packed_weights, block_scales, weight_scale_2, activations, rows,
              columns, output, stream);
        }
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
