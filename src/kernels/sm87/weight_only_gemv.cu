#include "q3x/kernels/sm87_weight_only_gemv.h"

#include "projection_route_registry.h"

#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

namespace registry = sm87_detail;

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarpsPerBlock = 8U;
constexpr unsigned int kThreads = kWarpSize * kWarpsPerBlock;
constexpr std::size_t kMaximumBlocks = 65'535U;
constexpr std::size_t kMaximumSmallMTokens = 8U;
constexpr std::size_t kFp8M1PersistentMinimumRows =
    registry::kFp8PersistentMinimumRows;
constexpr unsigned int kFp8M1PersistentMaximumBlocks =
    registry::kFp8PersistentMaximumBlocks;
constexpr std::size_t kFp8M2PersistentMinimumRows =
    registry::kFp8PersistentMinimumRows;
constexpr unsigned int kFp8M2PersistentMaximumBlocks =
    registry::kFp8PersistentMaximumBlocks;
constexpr std::size_t kFp8RowPairMinimumRows =
    registry::kFp8PersistentMinimumRows;
constexpr std::size_t kNvFp4RowPairMinimumRows =
    registry::kNvFp4M8ScaleCodebookMinimumRows;
constexpr std::size_t kNvFp4M1ScaleCodebookMinimumRows =
    registry::kNvFp4M1ScaleCodebookMinimumRows;
constexpr std::size_t kNvFp4M1ScaleCodebookMinimumColumns =
    registry::kNvFp4M1ScaleCodebookMinimumColumns;
constexpr unsigned int kNvFp4M1PersistentMaximumBlocks =
    registry::kNvFp4M1ScaleCodebookMaximumBlocks;
constexpr unsigned int kNvFp4M1RowPairMaximumBlocks = 80U;
constexpr unsigned int kNvFp4M1RowQuadMaximumBlocks = 64U;
constexpr std::size_t kNvFp4M2ScaleCodebookMinimumRows =
    registry::kNvFp4M1ScaleCodebookMinimumRows;
constexpr std::size_t kNvFp4M2ScaleCodebookMinimumColumns =
    registry::kNvFp4M1ScaleCodebookMinimumColumns;
constexpr unsigned int kNvFp4M2RowQuadMaximumBlocks =
    registry::kNvFp4M2RowQuadMaximumBlocks;
constexpr std::size_t kFp8EncodedValueCount = 256U;
constexpr std::size_t kFp8VectorValuesPerLane = 4U;
constexpr std::size_t kFp8VectorColumnsPerBlock =
    kThreads * kFp8VectorValuesPerLane;
constexpr std::size_t kFp8KvPairRows = 1'024U;
constexpr std::size_t kFp8KvPairColumns = 5'120U;
constexpr unsigned int kFp8KvPairSelectedMaximumBlocks = 128U;
constexpr std::size_t kFp8FullAttentionQRows = 12'288U;
constexpr unsigned int kFp8FullAttentionBlocks = 2'048U;
constexpr unsigned int kFp8FullAttentionKvFirstBlock = 1'024U;
constexpr unsigned int kFp8FullAttentionKvBlocks = 512U;
constexpr std::size_t kFp8QkvRows = 10'240U;
constexpr std::size_t kFp8ZRows = 6'144U;
constexpr std::size_t kFp8QkvZColumns = 5'120U;
constexpr unsigned int kFp8QkvRowQuads =
    static_cast<unsigned int>(kFp8QkvRows / 4U);
constexpr unsigned int kFp8ZRowQuads =
    static_cast<unsigned int>(kFp8ZRows / 4U);
constexpr unsigned int kFp8QkvZMaximumTestBlocks =
    kFp8QkvRowQuads + kFp8ZRowQuads;
constexpr unsigned int kFp8QkvZProductionBlocks = 1'536U;
constexpr unsigned int kFp8QkvZMaximumZBlocks = 768U;
constexpr std::size_t kNvFp4GroupSize = registry::kNvFp4GroupSize;
constexpr std::size_t kNvFp4ValuesPerByte = 2U;
constexpr std::size_t kNvFp4EncodedValueCount = 16U;
constexpr std::size_t kNvFp4PackedValuesPerScale =
    kNvFp4GroupSize / kNvFp4ValuesPerByte;
constexpr std::size_t kNvFp4VectorPackedBytesPerLane = 4U;
constexpr std::size_t kNvFp4VectorValuesPerLane =
    kNvFp4VectorPackedBytesPerLane * kNvFp4ValuesPerByte;
constexpr std::size_t kNvFp4VectorColumnsPerWarp =
    kWarpSize * kNvFp4VectorValuesPerLane;

static_assert(kFp8VectorColumnsPerBlock == registry::kFp8VectorColumns);
static_assert(kNvFp4VectorColumnsPerWarp == registry::kNvFp4VectorColumns);

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

template <std::size_t Alignment>
[[nodiscard]] bool pointer_is_aligned(const void* const pointer) noexcept {
  return (reinterpret_cast<std::uintptr_t>(pointer) % Alignment) == 0U;
}

[[nodiscard]] registry::ProjectionQuery make_fp8_projection_query(
    const std::size_t token_count, const std::uint8_t* const weights,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns) noexcept {
  return registry::ProjectionQuery{
      registry::WeightEncoding::kFp8,
      token_count,
      rows,
      columns,
      pointer_is_aligned<alignof(std::uint32_t)>(weights),
      pointer_is_aligned<alignof(uint4)>(weights),
      pointer_is_aligned<alignof(std::uint64_t)>(activations),
      true};
}

[[nodiscard]] registry::ProjectionQuery make_nvfp4_projection_query(
    const std::size_t token_count,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns) noexcept {
  return registry::ProjectionQuery{
      registry::WeightEncoding::kNvFp4,
      token_count,
      rows,
      columns,
      pointer_is_aligned<alignof(std::uint32_t)>(packed_weights),
      pointer_is_aligned<alignof(uint4)>(packed_weights),
      pointer_is_aligned<alignof(std::uint64_t)>(activations),
      pointer_is_aligned<alignof(std::uint16_t)>(block_scales)};
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

[[nodiscard]] constexpr bool use_fp8_m1_row_pair_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  // The row-pair path cleared checkpoint-like and shared-bank-stress gates on
  // these five checkpoint-bound projections. Unknown shapes retain the
  // cap-2048 single-row implementation.
  return registry::is_fp8_checkpoint_shape(
      registry::classify_projection_shape(registry::WeightEncoding::kFp8,
                                          rows, columns));
}

[[nodiscard]] constexpr unsigned int fp8_m1_row_quad_maximum_blocks(
    const std::size_t rows, const std::size_t columns) noexcept {
  return registry::fp8_m1_row_quad_maximum_blocks(
      registry::classify_projection_shape(registry::WeightEncoding::kFp8,
                                          rows, columns));
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
  return registry::is_nvfp4_mlp_shape(
      registry::classify_projection_shape(registry::WeightEncoding::kNvFp4,
                                          rows, columns));
}

[[nodiscard]] constexpr bool use_nvfp4_m16_wmma_fixed_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m8_fixed_shape(rows, columns);
}

[[nodiscard]] constexpr bool use_nvfp4_m1_scale_codebook(
    const std::size_t rows, const std::size_t columns) noexcept {
  return rows >= kNvFp4M1ScaleCodebookMinimumRows &&
         columns >= kNvFp4M1ScaleCodebookMinimumColumns;
}

[[nodiscard]] constexpr bool use_nvfp4_m1_row_quad_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  // These are the complete M=1 NVFP4 shape families observed in the pinned
  // checkpoint profile. Each cleared checkpoint-like and same-bank-stress
  // row-quad gates at the 64-block residency-matched cap.
  return registry::is_nvfp4_checkpoint_shape(
      registry::classify_projection_shape(registry::WeightEncoding::kNvFp4,
                                          rows, columns));
}

[[nodiscard]] constexpr bool use_nvfp4_m1_down_activation_staged_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  // Stage the exact down projection's 34-KiB activation once per CTA. All
  // near-misses and remaining shapes keep their independently gated paths.
  return registry::classify_projection_shape(
             registry::WeightEncoding::kNvFp4, rows, columns) ==
         registry::ProjectionShape::kNvFp4_5120x17408;
}

[[nodiscard]] constexpr bool use_nvfp4_m1_gate_up_activation_staged_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  // Stage the exact gate/up projection's 10-KiB activation once per CTA.
  // Keep lm-head, down-projection, near-misses, and all other shapes on their
  // independently gated production paths.
  return registry::classify_projection_shape(
             registry::WeightEncoding::kNvFp4, rows, columns) ==
         registry::ProjectionShape::kNvFp4_17408x5120;
}

[[nodiscard]] constexpr bool use_nvfp4_m1_lm_head_activation_staged_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  // Stage the exact vocabulary projection's 10-KiB activation once per CTA.
  // Keep it separate from gate/up and all near-miss shapes.
  return registry::classify_projection_shape(
             registry::WeightEncoding::kNvFp4, rows, columns) ==
         registry::ProjectionShape::kNvFp4_248320x5120;
}

[[nodiscard]] constexpr bool use_nvfp4_m2_scale_codebook(
    const std::size_t rows, const std::size_t columns) noexcept {
  return rows >= kNvFp4M2ScaleCodebookMinimumRows &&
         columns >= kNvFp4M2ScaleCodebookMinimumColumns;
}

[[nodiscard]] constexpr bool use_nvfp4_m2_row_quad_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m8_fixed_shape(rows, columns);
}

[[nodiscard]] constexpr bool use_fp8_small_m_row_pair(
    const std::size_t token_count, const std::size_t rows) noexcept {
  // All five checkpoint-bound FP8 projections have at least 1024 output rows.
  // Tiny and synthetic matrices retain the lower-register single-row kernel.
  return token_count == 8U && rows >= kFp8RowPairMinimumRows;
}

[[nodiscard]] constexpr bool use_fp8_m8_fixed_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  return registry::is_fp8_checkpoint_shape(
      registry::classify_projection_shape(registry::WeightEncoding::kFp8,
                                          rows, columns));
}

[[nodiscard]] constexpr bool use_fp8_m2_row_pair_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  // The M=2 candidate cleared both checkpoint-like and shared-bank-stress
  // gates on these five checkpoint-bound projections. Unknown shapes retain
  // the lower-risk cap-2048 single-row implementation.
  return use_fp8_m8_fixed_shape(rows, columns);
}

[[nodiscard]] constexpr unsigned int fp8_m2_row_quad_maximum_blocks(
    const std::size_t rows, const std::size_t columns) noexcept {
  return registry::fp8_m2_row_quad_maximum_blocks(
      registry::classify_projection_shape(registry::WeightEncoding::kFp8,
                                          rows, columns));
}

[[nodiscard]] constexpr bool use_fp8_m16_wmma_fixed_shape(
    const std::size_t rows, const std::size_t columns) noexcept {
  // The 1024-row projection is intentionally absent: its measured WMMA path
  // regresses versus two production M8 launches and must retain that fallback.
  return registry::is_fp8_m16_wmma_shape(
      registry::classify_projection_shape(registry::WeightEncoding::kFp8,
                                          rows, columns));
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

// Production M=1 output-row pair path. Each thread retains the original
// vector kernel's four value-position accumulator chains for both rows, while
// adjacent rows share each packed BF16 activation load and decode.
__global__ __launch_bounds__(kThreads, 5) void
fp8_w8a16_gemv_bf16_row_pair_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[2U][kWarpsPerBlock];
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
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};

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
      const std::uint64_t packed_activation =
          *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
        const unsigned int weight_shift = value * 8U;
        const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
            (packed_weights0 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
            (packed_weights1 >> weight_shift) & 0xffU);
        const std::uint16_t encoded_activation =
            static_cast<std::uint16_t>(
                (packed_activation >> (value * 16U)) & 0xffffU);
        const float decoded_activation = decode_bf16(encoded_activation);
        accumulators0[value] =
            fmaf(decoded_weights[encoded_weight0], decoded_activation,
                 accumulators0[value]);
        accumulators1[value] =
            fmaf(decoded_weights[encoded_weight1], decoded_activation,
                 accumulators1[value]);
      }
    }

    float sum0 = (accumulators0[0] + accumulators0[1]) +
                 (accumulators0[2] + accumulators0[3]);
    float sum1 = (accumulators1[0] + accumulators1[1]) +
                 (accumulators1[2] + accumulators1[3]);
    sum0 = warp_sum(sum0);
    sum1 = warp_sum(sum1);
    if (lane == 0U) {
      warp_sums[0U][warp] = sum0;
      warp_sums[1U][warp] = sum1;
    }
    __syncthreads();
    if (warp == 0U) {
      float block_sum0 =
          lane < kWarpsPerBlock ? warp_sums[0U][lane] : 0.0F;
      float block_sum1 =
          lane < kWarpsPerBlock ? warp_sums[1U][lane] : 0.0F;
      block_sum0 = warp_sum(block_sum0) * weight_scale;
      block_sum1 = warp_sum(block_sum1) * weight_scale;
      if (lane == 0U) {
        output[row0] = encode_bf16_rne(block_sum0);
        if (has_row1) {
          output[row1] = encode_bf16_rne(block_sum1);
        }
      }
    }
    __syncthreads();
  }
}

// Test-only preserved baseline for the production M=1 row-quad path. It keeps
// the original linear FP8 codebook layout so the bank-swizzled production
// kernel can be measured against the exact prior implementation in one cubin.
template <bool CompleteRowQuads>
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_row_quad_unswizzled_baseline_test_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  decoded_weights[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  __syncthreads();

  const std::size_t first_row = 4U * static_cast<std::size_t>(blockIdx.x);
  const std::size_t row_stride =
      4U * static_cast<std::size_t>(gridDim.x);
  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row1 = row0 + 1U;
    const std::size_t row2 = row0 + 2U;
    const std::size_t row3 = row0 + 3U;
    const bool has_row1 = CompleteRowQuads || row1 < rows;
    const bool has_row2 = CompleteRowQuads || row2 < rows;
    const bool has_row3 = CompleteRowQuads || row3 < rows;
    const std::uint8_t* const row0_weights = weights + row0 * columns;
    const std::uint8_t* const row1_weights =
        has_row1 ? weights + row1 * columns : row0_weights;
    const std::uint8_t* const row2_weights =
        has_row2 ? weights + row2 * columns : row0_weights;
    const std::uint8_t* const row3_weights =
        has_row3 ? weights + row3 * columns : row0_weights;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

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
      const std::uint32_t packed_weights2 =
          *reinterpret_cast<const std::uint32_t*>(row2_weights +
                                                  first_column);
      const std::uint32_t packed_weights3 =
          *reinterpret_cast<const std::uint32_t*>(row3_weights +
                                                  first_column);
      const std::uint64_t packed_activation =
          *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
        const unsigned int weight_shift = value * 8U;
        const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
            (packed_weights0 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
            (packed_weights1 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight2 = static_cast<std::uint8_t>(
            (packed_weights2 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight3 = static_cast<std::uint8_t>(
            (packed_weights3 >> weight_shift) & 0xffU);
        const std::uint16_t encoded_activation =
            static_cast<std::uint16_t>(
                (packed_activation >> (value * 16U)) & 0xffffU);
        const float decoded_activation = decode_bf16(encoded_activation);
        accumulators0[value] =
            fmaf(decoded_weights[encoded_weight0], decoded_activation,
                 accumulators0[value]);
        accumulators1[value] =
            fmaf(decoded_weights[encoded_weight1], decoded_activation,
                 accumulators1[value]);
        accumulators2[value] =
            fmaf(decoded_weights[encoded_weight2], decoded_activation,
                 accumulators2[value]);
        accumulators3[value] =
            fmaf(decoded_weights[encoded_weight3], decoded_activation,
                 accumulators3[value]);
      }
    }

    float sum0 = (accumulators0[0] + accumulators0[1]) +
                 (accumulators0[2] + accumulators0[3]);
    float sum1 = (accumulators1[0] + accumulators1[1]) +
                 (accumulators1[2] + accumulators1[3]);
    float sum2 = (accumulators2[0] + accumulators2[1]) +
                 (accumulators2[2] + accumulators2[3]);
    float sum3 = (accumulators3[0] + accumulators3[1]) +
                 (accumulators3[2] + accumulators3[3]);
    sum0 = warp_sum(sum0);
    sum1 = warp_sum(sum1);
    sum2 = warp_sum(sum2);
    sum3 = warp_sum(sum3);
    if (lane == 0U) {
      warp_sums[0U][warp] = sum0;
      warp_sums[1U][warp] = sum1;
      warp_sums[2U][warp] = sum2;
      warp_sums[3U][warp] = sum3;
    }
    __syncthreads();
    if (warp == 0U) {
      float block_sum0 =
          lane < kWarpsPerBlock ? warp_sums[0U][lane] : 0.0F;
      float block_sum1 =
          lane < kWarpsPerBlock ? warp_sums[1U][lane] : 0.0F;
      float block_sum2 =
          lane < kWarpsPerBlock ? warp_sums[2U][lane] : 0.0F;
      float block_sum3 =
          lane < kWarpsPerBlock ? warp_sums[3U][lane] : 0.0F;
      block_sum0 = warp_sum(block_sum0) * weight_scale;
      block_sum1 = warp_sum(block_sum1) * weight_scale;
      block_sum2 = warp_sum(block_sum2) * weight_scale;
      block_sum3 = warp_sum(block_sum3) * weight_scale;
      if (lane == 0U) {
        output[row0] = encode_bf16_rne(block_sum0);
        if constexpr (CompleteRowQuads) {
          output[row1] = encode_bf16_rne(block_sum1);
          output[row2] = encode_bf16_rne(block_sum2);
          output[row3] = encode_bf16_rne(block_sum3);
        } else {
          if (has_row1) {
            output[row1] = encode_bf16_rne(block_sum1);
          }
          if (has_row2) {
            output[row2] = encode_bf16_rne(block_sum2);
          }
          if (has_row3) {
            output[row3] = encode_bf16_rne(block_sum3);
          }
        }
      }
    }
    __syncthreads();
  }
}

// Production M=1 output-row quad path. Four adjacent rows share each packed
// activation load and decode. The eight high-code groups are XOR-folded into
// the shared-memory bank bits so codes that differ by 0x20 do not serialize on
// one bank. The bijection changes neither decoded FP32 bits nor FMA/reduction
// ordering; the incomplete specialization retains exhaustive tail coverage.
[[nodiscard]] __device__ __forceinline__ unsigned int
fp8_swizzled_codebook_slot(const std::uint8_t code) {
  const unsigned int value = static_cast<unsigned int>(code);
  return value ^ (value >> 5U);
}

template <bool CompleteRowQuads>
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_row_quad_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  const std::size_t first_row = 4U * static_cast<std::size_t>(blockIdx.x);
  const std::size_t row_stride =
      4U * static_cast<std::size_t>(gridDim.x);
  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row1 = row0 + 1U;
    const std::size_t row2 = row0 + 2U;
    const std::size_t row3 = row0 + 3U;
    const bool has_row1 = CompleteRowQuads || row1 < rows;
    const bool has_row2 = CompleteRowQuads || row2 < rows;
    const bool has_row3 = CompleteRowQuads || row3 < rows;
    const std::uint8_t* const row0_weights = weights + row0 * columns;
    const std::uint8_t* const row1_weights =
        has_row1 ? weights + row1 * columns : row0_weights;
    const std::uint8_t* const row2_weights =
        has_row2 ? weights + row2 * columns : row0_weights;
    const std::uint8_t* const row3_weights =
        has_row3 ? weights + row3 * columns : row0_weights;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

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
      const std::uint32_t packed_weights2 =
          *reinterpret_cast<const std::uint32_t*>(row2_weights +
                                                  first_column);
      const std::uint32_t packed_weights3 =
          *reinterpret_cast<const std::uint32_t*>(row3_weights +
                                                  first_column);
      // Fold bits 7:5 into bits 2:0 independently in all four bytes. The
      // mask removes cross-byte bits introduced by the packed right shift, so
      // each resulting byte is exactly code ^ (code >> 5).
      constexpr std::uint32_t kFp8PackedHighBits = 0x0707'0707U;
      const std::uint32_t swizzled_weights0 =
          packed_weights0 ^ ((packed_weights0 >> 5U) & kFp8PackedHighBits);
      const std::uint32_t swizzled_weights1 =
          packed_weights1 ^ ((packed_weights1 >> 5U) & kFp8PackedHighBits);
      const std::uint32_t swizzled_weights2 =
          packed_weights2 ^ ((packed_weights2 >> 5U) & kFp8PackedHighBits);
      const std::uint32_t swizzled_weights3 =
          packed_weights3 ^ ((packed_weights3 >> 5U) & kFp8PackedHighBits);
      const std::uint64_t packed_activation =
          *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
        const unsigned int weight_shift = value * 8U;
        const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
            (swizzled_weights0 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
            (swizzled_weights1 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight2 = static_cast<std::uint8_t>(
            (swizzled_weights2 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight3 = static_cast<std::uint8_t>(
            (swizzled_weights3 >> weight_shift) & 0xffU);
        const std::uint16_t encoded_activation =
            static_cast<std::uint16_t>(
                (packed_activation >> (value * 16U)) & 0xffffU);
        const float decoded_activation = decode_bf16(encoded_activation);
        accumulators0[value] =
            fmaf(decoded_weights[encoded_weight0], decoded_activation,
                 accumulators0[value]);
        accumulators1[value] =
            fmaf(decoded_weights[encoded_weight1], decoded_activation,
                 accumulators1[value]);
        accumulators2[value] =
            fmaf(decoded_weights[encoded_weight2], decoded_activation,
                 accumulators2[value]);
        accumulators3[value] =
            fmaf(decoded_weights[encoded_weight3], decoded_activation,
                 accumulators3[value]);
      }
    }

    float sum0 = (accumulators0[0] + accumulators0[1]) +
                 (accumulators0[2] + accumulators0[3]);
    float sum1 = (accumulators1[0] + accumulators1[1]) +
                 (accumulators1[2] + accumulators1[3]);
    float sum2 = (accumulators2[0] + accumulators2[1]) +
                 (accumulators2[2] + accumulators2[3]);
    float sum3 = (accumulators3[0] + accumulators3[1]) +
                 (accumulators3[2] + accumulators3[3]);
    sum0 = warp_sum(sum0);
    sum1 = warp_sum(sum1);
    sum2 = warp_sum(sum2);
    sum3 = warp_sum(sum3);
    if (lane == 0U) {
      warp_sums[0U][warp] = sum0;
      warp_sums[1U][warp] = sum1;
      warp_sums[2U][warp] = sum2;
      warp_sums[3U][warp] = sum3;
    }
    __syncthreads();
    if (warp == 0U) {
      float block_sum0 =
          lane < kWarpsPerBlock ? warp_sums[0U][lane] : 0.0F;
      float block_sum1 =
          lane < kWarpsPerBlock ? warp_sums[1U][lane] : 0.0F;
      float block_sum2 =
          lane < kWarpsPerBlock ? warp_sums[2U][lane] : 0.0F;
      float block_sum3 =
          lane < kWarpsPerBlock ? warp_sums[3U][lane] : 0.0F;
      block_sum0 = warp_sum(block_sum0) * weight_scale;
      block_sum1 = warp_sum(block_sum1) * weight_scale;
      block_sum2 = warp_sum(block_sum2) * weight_scale;
      block_sum3 = warp_sum(block_sum3) * weight_scale;
      if (lane == 0U) {
        output[row0] = encode_bf16_rne(block_sum0);
        if constexpr (CompleteRowQuads) {
          output[row1] = encode_bf16_rne(block_sum1);
          output[row2] = encode_bf16_rne(block_sum2);
          output[row3] = encode_bf16_rne(block_sum3);
        } else {
          if (has_row1) {
            output[row1] = encode_bf16_rne(block_sum1);
          }
          if (has_row2) {
            output[row2] = encode_bf16_rne(block_sum2);
          }
          if (has_row3) {
            output[row3] = encode_bf16_rne(block_sum3);
          }
        }
      }
    }
    __syncthreads();
  }
}

// Complete-row-quad body with a trailing block barrier. Full-attention Q still
// uses it, and the QKV/Z test-only predecessor retains it as the exact A/B
// baseline for the promoted no-tail-barrier production body below.
__device__ __forceinline__ void fp8_w8a16_gemv_bf16_complete_row_quad_body(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t columns,
    const unsigned int row0, std::uint16_t* const output,
    const float* const decoded_weights,
    float (*const warp_sums)[kWarpsPerBlock], const unsigned int lane,
    const unsigned int warp) {
  const std::uint8_t* const row0_weights =
      weights + static_cast<std::size_t>(row0) * columns;
  const std::uint8_t* const row1_weights = row0_weights + columns;
  const std::uint8_t* const row2_weights = row1_weights + columns;
  const std::uint8_t* const row3_weights = row2_weights + columns;
  float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

  for (std::size_t first_column =
           static_cast<std::size_t>(threadIdx.x) * kFp8VectorValuesPerLane;
       first_column < columns;
       first_column += kFp8VectorColumnsPerBlock) {
    const std::uint32_t packed_weights0 =
        *reinterpret_cast<const std::uint32_t*>(row0_weights + first_column);
    const std::uint32_t packed_weights1 =
        *reinterpret_cast<const std::uint32_t*>(row1_weights + first_column);
    const std::uint32_t packed_weights2 =
        *reinterpret_cast<const std::uint32_t*>(row2_weights + first_column);
    const std::uint32_t packed_weights3 =
        *reinterpret_cast<const std::uint32_t*>(row3_weights + first_column);
    constexpr std::uint32_t kFp8PackedHighBits = 0x0707'0707U;
    const std::uint32_t swizzled_weights0 =
        packed_weights0 ^ ((packed_weights0 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_weights1 =
        packed_weights1 ^ ((packed_weights1 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_weights2 =
        packed_weights2 ^ ((packed_weights2 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_weights3 =
        packed_weights3 ^ ((packed_weights3 >> 5U) & kFp8PackedHighBits);
    const std::uint64_t packed_activation =
        *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
    for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
      const unsigned int weight_shift = value * 8U;
      const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
          (swizzled_weights0 >> weight_shift) & 0xffU);
      const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
          (swizzled_weights1 >> weight_shift) & 0xffU);
      const std::uint8_t encoded_weight2 = static_cast<std::uint8_t>(
          (swizzled_weights2 >> weight_shift) & 0xffU);
      const std::uint8_t encoded_weight3 = static_cast<std::uint8_t>(
          (swizzled_weights3 >> weight_shift) & 0xffU);
      const std::uint16_t encoded_activation = static_cast<std::uint16_t>(
          (packed_activation >> (value * 16U)) & 0xffffU);
      const float decoded_activation = decode_bf16(encoded_activation);
      accumulators0[value] =
          fmaf(decoded_weights[encoded_weight0], decoded_activation,
               accumulators0[value]);
      accumulators1[value] =
          fmaf(decoded_weights[encoded_weight1], decoded_activation,
               accumulators1[value]);
      accumulators2[value] =
          fmaf(decoded_weights[encoded_weight2], decoded_activation,
               accumulators2[value]);
      accumulators3[value] =
          fmaf(decoded_weights[encoded_weight3], decoded_activation,
               accumulators3[value]);
    }
  }

  float sum0 = (accumulators0[0] + accumulators0[1]) +
               (accumulators0[2] + accumulators0[3]);
  float sum1 = (accumulators1[0] + accumulators1[1]) +
               (accumulators1[2] + accumulators1[3]);
  float sum2 = (accumulators2[0] + accumulators2[1]) +
               (accumulators2[2] + accumulators2[3]);
  float sum3 = (accumulators3[0] + accumulators3[1]) +
               (accumulators3[2] + accumulators3[3]);
  sum0 = warp_sum(sum0);
  sum1 = warp_sum(sum1);
  sum2 = warp_sum(sum2);
  sum3 = warp_sum(sum3);
  if (lane == 0U) {
    warp_sums[0U][warp] = sum0;
    warp_sums[1U][warp] = sum1;
    warp_sums[2U][warp] = sum2;
    warp_sums[3U][warp] = sum3;
  }
  __syncthreads();
  if (warp == 0U) {
    float block_sum0 = lane < kWarpsPerBlock ? warp_sums[0U][lane] : 0.0F;
    float block_sum1 = lane < kWarpsPerBlock ? warp_sums[1U][lane] : 0.0F;
    float block_sum2 = lane < kWarpsPerBlock ? warp_sums[2U][lane] : 0.0F;
    float block_sum3 = lane < kWarpsPerBlock ? warp_sums[3U][lane] : 0.0F;
    block_sum0 = warp_sum(block_sum0) * weight_scale;
    block_sum1 = warp_sum(block_sum1) * weight_scale;
    block_sum2 = warp_sum(block_sum2) * weight_scale;
    block_sum3 = warp_sum(block_sum3) * weight_scale;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(block_sum0);
      output[row0 + 1U] = encode_bf16_rne(block_sum1);
      output[row0 + 2U] = encode_bf16_rne(block_sum2);
      output[row0 + 3U] = encode_bf16_rne(block_sum3);
    }
  }
  __syncthreads();
}

// Test-only A/B predecessor for the production QKV/Z fusion. It retains the
// original single reduction-scratch slot and the tail block barrier after
// every row quad so same-binary tests can measure the promoted implementation
// against the exact predecessor.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_qkv_z_two_phase_tail_barrier_test_kernel(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  const unsigned int z_blocks =
      gridDim.x < kFp8QkvZMaximumZBlocks ? gridDim.x
                                        : kFp8QkvZMaximumZBlocks;
  // Keeping the phase loop rolled leaves one copy of the sizeable four-row
  // body in SASS while retaining the two distinct grid-stride topologies.
#pragma unroll 1
  for (unsigned int phase = 0U; phase < 2U; ++phase) {
    const bool is_qkv = phase == 0U;
    const unsigned int phase_blocks = is_qkv ? gridDim.x : z_blocks;
    const unsigned int phase_row_quads =
        is_qkv ? kFp8QkvRowQuads : kFp8ZRowQuads;
    if (blockIdx.x >= phase_blocks) {
      continue;
    }
    const std::uint8_t* const phase_weights =
        is_qkv ? qkv_weights : z_weights;
    const float phase_weight_scale =
        is_qkv ? qkv_weight_scale : z_weight_scale;
    std::uint16_t* const phase_output = is_qkv ? qkv_output : z_output;
    for (unsigned int row_quad = blockIdx.x;
         row_quad < phase_row_quads; row_quad += phase_blocks) {
      fp8_w8a16_gemv_bf16_complete_row_quad_body(
          phase_weights, phase_weight_scale, activation, columns,
          4U * row_quad, phase_output, decoded_weights, warp_sums, lane, warp);
    }
  }
}

// Production reduction-scratch pipeline body. It is intentionally an exact
// arithmetic clone of fp8_w8a16_gemv_bf16_complete_row_quad_body, except that
// it omits the final block barrier. The production kernel below gives adjacent
// logical bodies disjoint scratch slots, while the retained producer barrier
// still makes every warp's partial sums visible to warp 0.
__device__ __forceinline__ void
fp8_w8a16_gemv_bf16_complete_row_quad_no_tail_barrier_body(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t columns,
    const unsigned int row0, std::uint16_t* const output,
    const float* const decoded_weights,
    float (*const warp_sums)[kWarpsPerBlock], const unsigned int lane,
    const unsigned int warp) {
  const std::uint8_t* const row0_weights =
      weights + static_cast<std::size_t>(row0) * columns;
  const std::uint8_t* const row1_weights = row0_weights + columns;
  const std::uint8_t* const row2_weights = row1_weights + columns;
  const std::uint8_t* const row3_weights = row2_weights + columns;
  float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

  for (std::size_t first_column =
           static_cast<std::size_t>(threadIdx.x) * kFp8VectorValuesPerLane;
       first_column < columns;
       first_column += kFp8VectorColumnsPerBlock) {
    const std::uint32_t packed_weights0 =
        *reinterpret_cast<const std::uint32_t*>(row0_weights + first_column);
    const std::uint32_t packed_weights1 =
        *reinterpret_cast<const std::uint32_t*>(row1_weights + first_column);
    const std::uint32_t packed_weights2 =
        *reinterpret_cast<const std::uint32_t*>(row2_weights + first_column);
    const std::uint32_t packed_weights3 =
        *reinterpret_cast<const std::uint32_t*>(row3_weights + first_column);
    constexpr std::uint32_t kFp8PackedHighBits = 0x0707'0707U;
    const std::uint32_t swizzled_weights0 =
        packed_weights0 ^ ((packed_weights0 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_weights1 =
        packed_weights1 ^ ((packed_weights1 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_weights2 =
        packed_weights2 ^ ((packed_weights2 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_weights3 =
        packed_weights3 ^ ((packed_weights3 >> 5U) & kFp8PackedHighBits);
    const std::uint64_t packed_activation =
        *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
    for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
      const unsigned int weight_shift = value * 8U;
      const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
          (swizzled_weights0 >> weight_shift) & 0xffU);
      const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
          (swizzled_weights1 >> weight_shift) & 0xffU);
      const std::uint8_t encoded_weight2 = static_cast<std::uint8_t>(
          (swizzled_weights2 >> weight_shift) & 0xffU);
      const std::uint8_t encoded_weight3 = static_cast<std::uint8_t>(
          (swizzled_weights3 >> weight_shift) & 0xffU);
      const std::uint16_t encoded_activation = static_cast<std::uint16_t>(
          (packed_activation >> (value * 16U)) & 0xffffU);
      const float decoded_activation = decode_bf16(encoded_activation);
      accumulators0[value] =
          fmaf(decoded_weights[encoded_weight0], decoded_activation,
               accumulators0[value]);
      accumulators1[value] =
          fmaf(decoded_weights[encoded_weight1], decoded_activation,
               accumulators1[value]);
      accumulators2[value] =
          fmaf(decoded_weights[encoded_weight2], decoded_activation,
               accumulators2[value]);
      accumulators3[value] =
          fmaf(decoded_weights[encoded_weight3], decoded_activation,
               accumulators3[value]);
    }
  }

  float sum0 = (accumulators0[0] + accumulators0[1]) +
               (accumulators0[2] + accumulators0[3]);
  float sum1 = (accumulators1[0] + accumulators1[1]) +
               (accumulators1[2] + accumulators1[3]);
  float sum2 = (accumulators2[0] + accumulators2[1]) +
               (accumulators2[2] + accumulators2[3]);
  float sum3 = (accumulators3[0] + accumulators3[1]) +
               (accumulators3[2] + accumulators3[3]);
  sum0 = warp_sum(sum0);
  sum1 = warp_sum(sum1);
  sum2 = warp_sum(sum2);
  sum3 = warp_sum(sum3);
  if (lane == 0U) {
    warp_sums[0U][warp] = sum0;
    warp_sums[1U][warp] = sum1;
    warp_sums[2U][warp] = sum2;
    warp_sums[3U][warp] = sum3;
  }
  __syncthreads();
  if (warp == 0U) {
    float block_sum0 = lane < kWarpsPerBlock ? warp_sums[0U][lane] : 0.0F;
    float block_sum1 = lane < kWarpsPerBlock ? warp_sums[1U][lane] : 0.0F;
    float block_sum2 = lane < kWarpsPerBlock ? warp_sums[2U][lane] : 0.0F;
    float block_sum3 = lane < kWarpsPerBlock ? warp_sums[3U][lane] : 0.0F;
    block_sum0 = warp_sum(block_sum0) * weight_scale;
    block_sum1 = warp_sum(block_sum1) * weight_scale;
    block_sum2 = warp_sum(block_sum2) * weight_scale;
    block_sum3 = warp_sum(block_sum3) * weight_scale;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(block_sum0);
      output[row0 + 1U] = encode_bf16_rne(block_sum1);
      output[row0 + 2U] = encode_bf16_rne(block_sum2);
      output[row0 + 3U] = encode_bf16_rne(block_sum3);
    }
  }
}

// Production topology-preserving QKV/Z fusion. Body i+1 writes the opposite
// reduction slot while warp 0 consumes body i. The retained barrier in body
// i+1 cannot release until warp 0 has completed body i, so body i+2 cannot
// reuse body i's slot early. scratch_slot deliberately spans both ordered
// phases. The fixed 1,536-CTA public launch retains the established 1,536/768
// QKV/Z grid-stride topologies while sharing codebook setup between phases.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_qkv_z_two_phase_reduction_scratch_ping_pong_kernel(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[2U][4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  const unsigned int z_blocks =
      gridDim.x < kFp8QkvZMaximumZBlocks ? gridDim.x
                                        : kFp8QkvZMaximumZBlocks;
  unsigned int scratch_slot = 0U;
#pragma unroll 1
  for (unsigned int phase = 0U; phase < 2U; ++phase) {
    const bool is_qkv = phase == 0U;
    const unsigned int phase_blocks = is_qkv ? gridDim.x : z_blocks;
    const unsigned int phase_row_quads =
        is_qkv ? kFp8QkvRowQuads : kFp8ZRowQuads;
    if (blockIdx.x >= phase_blocks) {
      continue;
    }
    const std::uint8_t* const phase_weights =
        is_qkv ? qkv_weights : z_weights;
    const float phase_weight_scale =
        is_qkv ? qkv_weight_scale : z_weight_scale;
    std::uint16_t* const phase_output = is_qkv ? qkv_output : z_output;
    for (unsigned int row_quad = blockIdx.x;
         row_quad < phase_row_quads; row_quad += phase_blocks) {
      fp8_w8a16_gemv_bf16_complete_row_quad_no_tail_barrier_body(
          phase_weights, phase_weight_scale, activation, columns,
          4U * row_quad, phase_output, decoded_weights,
          warp_sums[scratch_slot], lane, warp);
      scratch_slot ^= 1U;
    }
  }
}

// Exact-shape cross-matrix row-quad kernel. A CTA computes two adjacent rows
// from both matrices, so one packed activation load/decode feeds all four FP8
// weight streams. At the selected persistent cap this also amortizes codebook
// setup across multiple row quads. The individual accumulator and reduction
// chains retain the single-projection BF16 result exactly.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_projection_pair_row_quad_kernel(
    const std::uint8_t* const key_weights,
    const float key_weight_scale,
    const std::uint8_t* const value_weights,
    const float value_weight_scale,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    std::uint16_t* const key_output,
    std::uint16_t* const value_output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  const std::size_t first_row = 2U * static_cast<std::size_t>(blockIdx.x);
  const std::size_t row_stride = 2U * static_cast<std::size_t>(gridDim.x);
  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row1 = row0 + 1U;
    const std::uint8_t* const key_row0_weights =
        key_weights + row0 * columns;
    const std::uint8_t* const key_row1_weights =
        key_weights + row1 * columns;
    const std::uint8_t* const value_row0_weights =
        value_weights + row0 * columns;
    const std::uint8_t* const value_row1_weights =
        value_weights + row1 * columns;
    float key_accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float key_accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float value_accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float value_accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};

    for (std::size_t first_column =
             static_cast<std::size_t>(threadIdx.x) *
             kFp8VectorValuesPerLane;
         first_column < columns;
         first_column += kFp8VectorColumnsPerBlock) {
      const std::uint32_t packed_key_weights0 =
          *reinterpret_cast<const std::uint32_t*>(key_row0_weights +
                                                  first_column);
      const std::uint32_t packed_key_weights1 =
          *reinterpret_cast<const std::uint32_t*>(key_row1_weights +
                                                  first_column);
      const std::uint32_t packed_value_weights0 =
          *reinterpret_cast<const std::uint32_t*>(value_row0_weights +
                                                  first_column);
      const std::uint32_t packed_value_weights1 =
          *reinterpret_cast<const std::uint32_t*>(value_row1_weights +
                                                  first_column);
      constexpr std::uint32_t kFp8PackedHighBits = 0x0707'0707U;
      const std::uint32_t swizzled_key_weights0 =
          packed_key_weights0 ^
          ((packed_key_weights0 >> 5U) & kFp8PackedHighBits);
      const std::uint32_t swizzled_key_weights1 =
          packed_key_weights1 ^
          ((packed_key_weights1 >> 5U) & kFp8PackedHighBits);
      const std::uint32_t swizzled_value_weights0 =
          packed_value_weights0 ^
          ((packed_value_weights0 >> 5U) & kFp8PackedHighBits);
      const std::uint32_t swizzled_value_weights1 =
          packed_value_weights1 ^
          ((packed_value_weights1 >> 5U) & kFp8PackedHighBits);
      const std::uint64_t packed_activation =
          *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane;
           ++value) {
        const unsigned int weight_shift = value * 8U;
        const std::uint8_t encoded_key_weight0 =
            static_cast<std::uint8_t>(
                (swizzled_key_weights0 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_key_weight1 =
            static_cast<std::uint8_t>(
                (swizzled_key_weights1 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_value_weight0 =
            static_cast<std::uint8_t>(
                (swizzled_value_weights0 >> weight_shift) & 0xffU);
        const std::uint8_t encoded_value_weight1 =
            static_cast<std::uint8_t>(
                (swizzled_value_weights1 >> weight_shift) & 0xffU);
        const std::uint16_t encoded_activation =
            static_cast<std::uint16_t>(
                (packed_activation >> (value * 16U)) & 0xffffU);
        const float decoded_activation = decode_bf16(encoded_activation);
        key_accumulators0[value] =
            fmaf(decoded_weights[encoded_key_weight0], decoded_activation,
                 key_accumulators0[value]);
        key_accumulators1[value] =
            fmaf(decoded_weights[encoded_key_weight1], decoded_activation,
                 key_accumulators1[value]);
        value_accumulators0[value] =
            fmaf(decoded_weights[encoded_value_weight0], decoded_activation,
                 value_accumulators0[value]);
        value_accumulators1[value] =
            fmaf(decoded_weights[encoded_value_weight1], decoded_activation,
                 value_accumulators1[value]);
      }
    }

    float key_sum0 = (key_accumulators0[0] + key_accumulators0[1]) +
                     (key_accumulators0[2] + key_accumulators0[3]);
    float key_sum1 = (key_accumulators1[0] + key_accumulators1[1]) +
                     (key_accumulators1[2] + key_accumulators1[3]);
    float value_sum0 =
        (value_accumulators0[0] + value_accumulators0[1]) +
        (value_accumulators0[2] + value_accumulators0[3]);
    float value_sum1 =
        (value_accumulators1[0] + value_accumulators1[1]) +
        (value_accumulators1[2] + value_accumulators1[3]);
    key_sum0 = warp_sum(key_sum0);
    key_sum1 = warp_sum(key_sum1);
    value_sum0 = warp_sum(value_sum0);
    value_sum1 = warp_sum(value_sum1);
    if (lane == 0U) {
      warp_sums[0U][warp] = key_sum0;
      warp_sums[1U][warp] = key_sum1;
      warp_sums[2U][warp] = value_sum0;
      warp_sums[3U][warp] = value_sum1;
    }
    __syncthreads();
    if (warp == 0U) {
      float key_block_sum0 =
          lane < kWarpsPerBlock ? warp_sums[0U][lane] : 0.0F;
      float key_block_sum1 =
          lane < kWarpsPerBlock ? warp_sums[1U][lane] : 0.0F;
      float value_block_sum0 =
          lane < kWarpsPerBlock ? warp_sums[2U][lane] : 0.0F;
      float value_block_sum1 =
          lane < kWarpsPerBlock ? warp_sums[3U][lane] : 0.0F;
      key_block_sum0 = warp_sum(key_block_sum0) * key_weight_scale;
      key_block_sum1 = warp_sum(key_block_sum1) * key_weight_scale;
      value_block_sum0 = warp_sum(value_block_sum0) * value_weight_scale;
      value_block_sum1 = warp_sum(value_block_sum1) * value_weight_scale;
      if (lane == 0U) {
        key_output[row0] = encode_bf16_rne(key_block_sum0);
        key_output[row1] = encode_bf16_rne(key_block_sum1);
        value_output[row0] = encode_bf16_rne(value_block_sum0);
        value_output[row1] = encode_bf16_rne(value_block_sum1);
      }
    }
    __syncthreads();
  }
}

// One exact K/V row-pair task, kept separate from the production persistent
// kernel so the full-attention fusion can preserve that baseline verbatim.
// The arithmetic and reduction sequence intentionally mirrors the body above.
__device__ __forceinline__ void
fp8_w8a16_gemv_bf16_complete_projection_pair_row_pair_body(
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation, const std::size_t columns,
    const std::size_t row0, std::uint16_t* const key_output,
    std::uint16_t* const value_output, const float* const decoded_weights,
    float (*const warp_sums)[kWarpsPerBlock], const unsigned int lane,
    const unsigned int warp) {
  const std::size_t row1 = row0 + 1U;
  const std::uint8_t* const key_row0_weights =
      key_weights + row0 * columns;
  const std::uint8_t* const key_row1_weights =
      key_weights + row1 * columns;
  const std::uint8_t* const value_row0_weights =
      value_weights + row0 * columns;
  const std::uint8_t* const value_row1_weights =
      value_weights + row1 * columns;
  float key_accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float key_accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float value_accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float value_accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};

  for (std::size_t first_column =
           static_cast<std::size_t>(threadIdx.x) *
           kFp8VectorValuesPerLane;
       first_column < columns;
       first_column += kFp8VectorColumnsPerBlock) {
    const std::uint32_t packed_key_weights0 =
        *reinterpret_cast<const std::uint32_t*>(key_row0_weights +
                                                first_column);
    const std::uint32_t packed_key_weights1 =
        *reinterpret_cast<const std::uint32_t*>(key_row1_weights +
                                                first_column);
    const std::uint32_t packed_value_weights0 =
        *reinterpret_cast<const std::uint32_t*>(value_row0_weights +
                                                first_column);
    const std::uint32_t packed_value_weights1 =
        *reinterpret_cast<const std::uint32_t*>(value_row1_weights +
                                                first_column);
    constexpr std::uint32_t kFp8PackedHighBits = 0x0707'0707U;
    const std::uint32_t swizzled_key_weights0 =
        packed_key_weights0 ^
        ((packed_key_weights0 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_key_weights1 =
        packed_key_weights1 ^
        ((packed_key_weights1 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_value_weights0 =
        packed_value_weights0 ^
        ((packed_value_weights0 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_value_weights1 =
        packed_value_weights1 ^
        ((packed_value_weights1 >> 5U) & kFp8PackedHighBits);
    const std::uint64_t packed_activation =
        *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
    for (unsigned int value = 0U; value < kFp8VectorValuesPerLane;
         ++value) {
      const unsigned int weight_shift = value * 8U;
      const std::uint8_t encoded_key_weight0 = static_cast<std::uint8_t>(
          (swizzled_key_weights0 >> weight_shift) & 0xffU);
      const std::uint8_t encoded_key_weight1 = static_cast<std::uint8_t>(
          (swizzled_key_weights1 >> weight_shift) & 0xffU);
      const std::uint8_t encoded_value_weight0 = static_cast<std::uint8_t>(
          (swizzled_value_weights0 >> weight_shift) & 0xffU);
      const std::uint8_t encoded_value_weight1 = static_cast<std::uint8_t>(
          (swizzled_value_weights1 >> weight_shift) & 0xffU);
      const std::uint16_t encoded_activation = static_cast<std::uint16_t>(
          (packed_activation >> (value * 16U)) & 0xffffU);
      const float decoded_activation = decode_bf16(encoded_activation);
      key_accumulators0[value] =
          fmaf(decoded_weights[encoded_key_weight0], decoded_activation,
               key_accumulators0[value]);
      key_accumulators1[value] =
          fmaf(decoded_weights[encoded_key_weight1], decoded_activation,
               key_accumulators1[value]);
      value_accumulators0[value] =
          fmaf(decoded_weights[encoded_value_weight0], decoded_activation,
               value_accumulators0[value]);
      value_accumulators1[value] =
          fmaf(decoded_weights[encoded_value_weight1], decoded_activation,
               value_accumulators1[value]);
    }
  }

  float key_sum0 = (key_accumulators0[0] + key_accumulators0[1]) +
                   (key_accumulators0[2] + key_accumulators0[3]);
  float key_sum1 = (key_accumulators1[0] + key_accumulators1[1]) +
                   (key_accumulators1[2] + key_accumulators1[3]);
  float value_sum0 =
      (value_accumulators0[0] + value_accumulators0[1]) +
      (value_accumulators0[2] + value_accumulators0[3]);
  float value_sum1 =
      (value_accumulators1[0] + value_accumulators1[1]) +
      (value_accumulators1[2] + value_accumulators1[3]);
  key_sum0 = warp_sum(key_sum0);
  key_sum1 = warp_sum(key_sum1);
  value_sum0 = warp_sum(value_sum0);
  value_sum1 = warp_sum(value_sum1);
  if (lane == 0U) {
    warp_sums[0U][warp] = key_sum0;
    warp_sums[1U][warp] = key_sum1;
    warp_sums[2U][warp] = value_sum0;
    warp_sums[3U][warp] = value_sum1;
  }
  __syncthreads();
  if (warp == 0U) {
    float key_block_sum0 =
        lane < kWarpsPerBlock ? warp_sums[0U][lane] : 0.0F;
    float key_block_sum1 =
        lane < kWarpsPerBlock ? warp_sums[1U][lane] : 0.0F;
    float value_block_sum0 =
        lane < kWarpsPerBlock ? warp_sums[2U][lane] : 0.0F;
    float value_block_sum1 =
        lane < kWarpsPerBlock ? warp_sums[3U][lane] : 0.0F;
    key_block_sum0 = warp_sum(key_block_sum0) * key_weight_scale;
    key_block_sum1 = warp_sum(key_block_sum1) * key_weight_scale;
    value_block_sum0 = warp_sum(value_block_sum0) * value_weight_scale;
    value_block_sum1 = warp_sum(value_block_sum1) * value_weight_scale;
    if (lane == 0U) {
      key_output[row0] = encode_bf16_rne(key_block_sum0);
      key_output[row1] = encode_bf16_rne(key_block_sum1);
      value_output[row0] = encode_bf16_rne(value_block_sum0);
      value_output[row1] = encode_bf16_rne(value_block_sum1);
    }
  }
  __syncthreads();
}

// Exact full-attention fusion. The Q phase exactly retains the
// production 2,048-CTA row-quad order for [12288, 5120]. Blocks 1024..1535
// then consume one of the 512 K/V row-pair tasks, reusing the codebook and
// collapsing the existing Q + K/V launch chain into one kernel.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_q_kv_two_phase_row_group_kernel(
    const std::uint8_t* const q_weights, const float q_weight_scale,
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation,
    std::uint16_t* const q_output, std::uint16_t* const key_output,
    std::uint16_t* const value_output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  constexpr unsigned int kQRowQuads =
      static_cast<unsigned int>(kFp8FullAttentionQRows / 4U);
  for (unsigned int row_quad = blockIdx.x; row_quad < kQRowQuads;
       row_quad += kFp8FullAttentionBlocks) {
    fp8_w8a16_gemv_bf16_complete_row_quad_body(
        q_weights, q_weight_scale, activation, kFp8KvPairColumns,
        4U * row_quad, q_output, decoded_weights, warp_sums, lane, warp);
  }

  if (blockIdx.x >= kFp8FullAttentionKvFirstBlock &&
      blockIdx.x <
          kFp8FullAttentionKvFirstBlock + kFp8FullAttentionKvBlocks) {
    const std::size_t row0 =
        2U * static_cast<std::size_t>(blockIdx.x -
                                     kFp8FullAttentionKvFirstBlock);
    fp8_w8a16_gemv_bf16_complete_projection_pair_row_pair_body(
        key_weights, key_weight_scale, value_weights, value_weight_scale,
        activation, kFp8KvPairColumns, row0, key_output, value_output,
        decoded_weights, warp_sums, lane, warp);
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

// Production M=2 output-row pair path. Adjacent rows share packed BF16
// activation loads and decodes while retaining independent FP8 weights,
// accumulators, and reduction trees.
__global__ __launch_bounds__(kThreads, 5) void
fp8_w8a16_small_m2_gemm_bf16_row_pair_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 2U;
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

// Production M=2 output-row quad path. Four adjacent output rows share each
// packed BF16 activation load and decode, while every (row, token) accumulator
// retains the row-pair baseline's K update order and reduction tree.
// CompleteRowQuads removes the tail predicates for the selected checkpoint
// shapes, all of which have row counts divisible by four.
template <bool CompleteRowQuads>
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_small_m2_gemm_bf16_row_quad_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 2U;
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[4U][kTokenCount][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  decoded_weights[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  __syncthreads();

  const std::size_t first_row = 4U * static_cast<std::size_t>(blockIdx.x);
  const std::size_t row_stride =
      4U * static_cast<std::size_t>(gridDim.x);
  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row1 = row0 + 1U;
    const std::size_t row2 = row0 + 2U;
    const std::size_t row3 = row0 + 3U;
    const bool has_row1 = CompleteRowQuads || row1 < rows;
    const bool has_row2 = CompleteRowQuads || row2 < rows;
    const bool has_row3 = CompleteRowQuads || row3 < rows;
    const std::uint8_t* const row0_weights = weights + row0 * columns;
    const std::uint8_t* const row1_weights =
        has_row1 ? weights + row1 * columns : row0_weights;
    const std::uint8_t* const row2_weights =
        has_row2 ? weights + row2 * columns : row0_weights;
    const std::uint8_t* const row3_weights =
        has_row3 ? weights + row3 * columns : row0_weights;

    float accumulator00 = 0.0F;
    float accumulator01 = 0.0F;
    float accumulator10 = 0.0F;
    float accumulator11 = 0.0F;
    float accumulator20 = 0.0F;
    float accumulator21 = 0.0F;
    float accumulator30 = 0.0F;
    float accumulator31 = 0.0F;

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
      const std::uint32_t packed_weights2 =
          *reinterpret_cast<const std::uint32_t*>(row2_weights +
                                                  first_column);
      const std::uint32_t packed_weights3 =
          *reinterpret_cast<const std::uint32_t*>(row3_weights +
                                                  first_column);
      const std::uint64_t packed_activation0 =
          *reinterpret_cast<const std::uint64_t*>(activations +
                                                  first_column);
      const std::uint64_t packed_activation1 =
          *reinterpret_cast<const std::uint64_t*>(activations + columns +
                                                  first_column);

#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane;
           ++value) {
        const unsigned int weight_shift = value * 8U;
        const unsigned int activation_shift = value * 16U;
        const float decoded_weight0 = decoded_weights[static_cast<std::uint8_t>(
            (packed_weights0 >> weight_shift) & 0xffU)];
        const float decoded_weight1 = decoded_weights[static_cast<std::uint8_t>(
            (packed_weights1 >> weight_shift) & 0xffU)];
        const float decoded_weight2 = decoded_weights[static_cast<std::uint8_t>(
            (packed_weights2 >> weight_shift) & 0xffU)];
        const float decoded_weight3 = decoded_weights[static_cast<std::uint8_t>(
            (packed_weights3 >> weight_shift) & 0xffU)];
        const float decoded_activation0 = decode_bf16(
            static_cast<std::uint16_t>(
                (packed_activation0 >> activation_shift) & 0xffffU));
        const float decoded_activation1 = decode_bf16(
            static_cast<std::uint16_t>(
                (packed_activation1 >> activation_shift) & 0xffffU));
        accumulator00 =
            fmaf(decoded_weight0, decoded_activation0, accumulator00);
        accumulator01 =
            fmaf(decoded_weight0, decoded_activation1, accumulator01);
        accumulator10 =
            fmaf(decoded_weight1, decoded_activation0, accumulator10);
        accumulator11 =
            fmaf(decoded_weight1, decoded_activation1, accumulator11);
        accumulator20 =
            fmaf(decoded_weight2, decoded_activation0, accumulator20);
        accumulator21 =
            fmaf(decoded_weight2, decoded_activation1, accumulator21);
        accumulator30 =
            fmaf(decoded_weight3, decoded_activation0, accumulator30);
        accumulator31 =
            fmaf(decoded_weight3, decoded_activation1, accumulator31);
      }
    }

    const float sum00 = warp_sum(accumulator00);
    const float sum01 = warp_sum(accumulator01);
    const float sum10 = warp_sum(accumulator10);
    const float sum11 = warp_sum(accumulator11);
    const float sum20 = warp_sum(accumulator20);
    const float sum21 = warp_sum(accumulator21);
    const float sum30 = warp_sum(accumulator30);
    const float sum31 = warp_sum(accumulator31);
    if (lane == 0U) {
      warp_sums[0U][0U][warp] = sum00;
      warp_sums[0U][1U][warp] = sum01;
      warp_sums[1U][0U][warp] = sum10;
      warp_sums[1U][1U][warp] = sum11;
      warp_sums[2U][0U][warp] = sum20;
      warp_sums[2U][1U][warp] = sum21;
      warp_sums[3U][0U][warp] = sum30;
      warp_sums[3U][1U][warp] = sum31;
    }
    __syncthreads();

    if (warp == 0U) {
      float block_sum00 =
          lane < kWarpsPerBlock ? warp_sums[0U][0U][lane] : 0.0F;
      float block_sum01 =
          lane < kWarpsPerBlock ? warp_sums[0U][1U][lane] : 0.0F;
      float block_sum10 =
          lane < kWarpsPerBlock ? warp_sums[1U][0U][lane] : 0.0F;
      float block_sum11 =
          lane < kWarpsPerBlock ? warp_sums[1U][1U][lane] : 0.0F;
      float block_sum20 =
          lane < kWarpsPerBlock ? warp_sums[2U][0U][lane] : 0.0F;
      float block_sum21 =
          lane < kWarpsPerBlock ? warp_sums[2U][1U][lane] : 0.0F;
      float block_sum30 =
          lane < kWarpsPerBlock ? warp_sums[3U][0U][lane] : 0.0F;
      float block_sum31 =
          lane < kWarpsPerBlock ? warp_sums[3U][1U][lane] : 0.0F;
      block_sum00 = warp_sum(block_sum00) * weight_scale;
      block_sum01 = warp_sum(block_sum01) * weight_scale;
      block_sum10 = warp_sum(block_sum10) * weight_scale;
      block_sum11 = warp_sum(block_sum11) * weight_scale;
      block_sum20 = warp_sum(block_sum20) * weight_scale;
      block_sum21 = warp_sum(block_sum21) * weight_scale;
      block_sum30 = warp_sum(block_sum30) * weight_scale;
      block_sum31 = warp_sum(block_sum31) * weight_scale;
      if (lane == 0U) {
        output[row0] = encode_bf16_rne(block_sum00);
        output[rows + row0] = encode_bf16_rne(block_sum01);
        if constexpr (CompleteRowQuads) {
          output[row1] = encode_bf16_rne(block_sum10);
          output[rows + row1] = encode_bf16_rne(block_sum11);
          output[row2] = encode_bf16_rne(block_sum20);
          output[rows + row2] = encode_bf16_rne(block_sum21);
          output[row3] = encode_bf16_rne(block_sum30);
          output[rows + row3] = encode_bf16_rne(block_sum31);
        } else {
          if (has_row1) {
            output[row1] = encode_bf16_rne(block_sum10);
            output[rows + row1] = encode_bf16_rne(block_sum11);
          }
          if (has_row2) {
            output[row2] = encode_bf16_rne(block_sum20);
            output[rows + row2] = encode_bf16_rne(block_sum21);
          }
          if (has_row3) {
            output[row3] = encode_bf16_rne(block_sum30);
            output[rows + row3] = encode_bf16_rne(block_sum31);
          }
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

// Fixed-M32 production kernel. It keeps the production M16 shared-memory
// footprint by retaining one 16-token A/C panel: each K-stage decodes B once,
// consumes A[0:16], then overwrites only A and consumes A[16:32]. The two
// accumulator dependency chains preserve the exact K/MMA order of two
// production M16 launches.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
__global__ __launch_bounds__(kThreads, 5) void
fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, std::uint16_t* const output) {
  constexpr unsigned int kPanelTokenCount = 16U;
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
      kPanelTokenCount * kOutputColumnsPerBlock;
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
      shared_activations[kPanelTokenCount *
                         kSharedActivationWordsPerToken];
  __shared__ BOrCStorage b_or_c;

  namespace wmma = nvcuda::wmma;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;
  decoded_weights[thread] = encode_bf16_rne(
      decode_e4m3fn(static_cast<std::uint8_t>(thread)));
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
  wmma::fill_fragment(accumulator0, 0.0F);
  wmma::fill_fragment(accumulator1, 0.0F);
  const unsigned int first_output_column =
      blockIdx.x * kOutputColumnsPerBlock;

#pragma unroll 1
  for (unsigned int first_k = 0U; first_k < kColumns;
       first_k += kColumnsPerStage) {
    const unsigned int token = thread / kActivationWordsPerToken;
    const unsigned int activation_word =
        thread % kActivationWordsPerToken;
    const std::size_t activation_column =
        first_k + activation_word * kBf16ValuesPerActivationWord;
    shared_activations[token * kSharedActivationWordsPerToken +
                       activation_word] =
        *reinterpret_cast<const std::uint64_t*>(
            activations + static_cast<std::size_t>(token) * kColumns +
            activation_column);

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
      wmma::mma_sync(accumulator0, activation_fragment, weight_fragment,
                     accumulator0);
    }
    // All A0 fragment loads must retire before the panel is overwritten.
    __syncthreads();

    shared_activations[token * kSharedActivationWordsPerToken +
                       activation_word] =
        *reinterpret_cast<const std::uint64_t*>(
            activations +
            static_cast<std::size_t>(token + kPanelTokenCount) * kColumns +
            activation_column);
    __syncthreads();

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
      wmma::mma_sync(accumulator1, activation_fragment, weight_fragment,
                     accumulator1);
    }
    // Protect both shared panels before the next B decode/A0 load and the
    // final B-to-C union transition.
    __syncthreads();
  }

  wmma::store_matrix_sync(
      b_or_c.output + warp * kOutputColumnsPerWarp, accumulator0,
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
  // The first output panel must be fully consumed before C is reused.
  __syncthreads();

  wmma::store_matrix_sync(
      b_or_c.output + warp * kOutputColumnsPerWarp, accumulator1,
      kOutputColumnsPerBlock, wmma::mem_row_major);
  __syncthreads();
#pragma unroll
  for (unsigned int index = thread; index < kSharedOutputCount;
       index += kThreads) {
    const unsigned int token = index / kOutputColumnsPerBlock;
    const unsigned int local_output_column =
        index % kOutputColumnsPerBlock;
    output[static_cast<std::size_t>(token + kPanelTokenCount) * kRows +
           first_output_column + local_output_column] =
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

__device__ __forceinline__ std::uint32_t multiply_bf16x2_bits(
    const std::uint32_t value_bits, const std::uint16_t scale_bits) {
  const __nv_bfloat162_raw value_raw{
      static_cast<std::uint16_t>(value_bits),
      static_cast<std::uint16_t>(value_bits >> 16U)};
  const __nv_bfloat162_raw scale_raw{scale_bits, scale_bits};
  const __nv_bfloat162 values(value_raw);
  const __nv_bfloat162 scales(scale_raw);
  const __nv_bfloat162_raw result_raw =
      static_cast<__nv_bfloat162_raw>(__hmul2_rn(values, scales));
  return static_cast<std::uint32_t>(result_raw.x) |
         (static_cast<std::uint32_t>(result_raw.y) << 16U);
}

__device__ __forceinline__ void decode_nvfp4x8_to_bf16x8_factorized(
    const std::uint32_t packed,
    const std::uint32_t* const decoded_e2m1_byte_pairs,
    const std::uint16_t decoded_scale, std::uint32_t* const output) {
#pragma unroll
  for (unsigned int byte_index = 0U; byte_index < sizeof(packed);
       ++byte_index) {
    const std::uint8_t encoded = static_cast<std::uint8_t>(
        (packed >> (byte_index * 8U)) & 0xffU);
    output[byte_index] = multiply_bf16x2_bits(
        decoded_e2m1_byte_pairs[encoded], decoded_scale);
  }
}

__device__ __forceinline__ uint4
decode_nvfp4x8_to_bf16x8_factorized_vector(
    const std::uint32_t packed,
    const std::uint32_t* const decoded_e2m1_byte_pairs,
    const std::uint16_t decoded_scale) {
  uint4 output{};
  output.x = multiply_bf16x2_bits(
      decoded_e2m1_byte_pairs[static_cast<std::uint8_t>(packed)],
      decoded_scale);
  output.y = multiply_bf16x2_bits(
      decoded_e2m1_byte_pairs[
          static_cast<std::uint8_t>(packed >> 8U)],
      decoded_scale);
  output.z = multiply_bf16x2_bits(
      decoded_e2m1_byte_pairs[
          static_cast<std::uint8_t>(packed >> 16U)],
      decoded_scale);
  output.w = multiply_bf16x2_bits(
      decoded_e2m1_byte_pairs[
          static_cast<std::uint8_t>(packed >> 24U)],
      decoded_scale);
  return output;
}

// Test-only SM80+ raw-weight staging primitives. The candidate using these
// helpers keeps one 16-byte cell per thread in shared memory. Each thread
// waits for its own prior async copy, consumes that cell into registers, and
// only then starts overwriting the same cell with the next K64 stage.
__device__ __forceinline__ void cp_async_cg_shared_global_16(
    void* const shared_destination, const void* const global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(global_source)
               : "memory");
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      *reinterpret_cast<const uint4*>(global_source);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

template <bool kFactorized>
struct NvFp4M32ProductLookupStorage;

template <>
struct alignas(32) NvFp4M32ProductLookupStorage<false> {
  std::uint32_t product_words[kFp8EncodedValueCount *
                              kNvFp4EncodedValueCount / 2U];
};

template <>
struct alignas(32) NvFp4M32ProductLookupStorage<true> {
  std::uint32_t e2m1_byte_pairs[kFp8EncodedValueCount];
  std::uint16_t scale_values[kFp8EncodedValueCount];
};

static_assert(sizeof(NvFp4M32ProductLookupStorage<false>) == 8'192U);
static_assert(sizeof(NvFp4M32ProductLookupStorage<true>) == 1'536U);

// Test-only exhaustive semantic gate for the factorized M32 lookup. Every
// thread compares one E4M3FN scale code and one packed pair of E2M1 values.
__global__ void nvfp4_factorized_product_lookup_exhaustive_kernel(
    std::uint32_t* const factorized, std::uint32_t* const reference) {
  constexpr unsigned int kPackedByteCount = 256U;
  constexpr unsigned int kCombinationCount =
      kFp8EncodedValueCount * kPackedByteCount;
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kCombinationCount) {
    return;
  }
  const std::uint8_t scale_code =
      static_cast<std::uint8_t>(index / kPackedByteCount);
  const std::uint8_t packed =
      static_cast<std::uint8_t>(index % kPackedByteCount);
  const std::uint8_t low_code = packed & 0x0fU;
  const std::uint8_t high_code = packed >> 4U;
  const std::uint16_t decoded_low = encode_bf16_rne(decode_e2m1(low_code));
  const std::uint16_t decoded_high =
      encode_bf16_rne(decode_e2m1(high_code));
  const std::uint16_t decoded_scale =
      encode_bf16_rne(decode_e4m3fn(scale_code));
  const std::uint32_t value_pair =
      static_cast<std::uint32_t>(decoded_low) |
      (static_cast<std::uint32_t>(decoded_high) << 16U);
  factorized[index] = multiply_bf16x2_bits(value_pair, decoded_scale);

  const float scale = decode_e4m3fn(scale_code);
  const std::uint16_t reference_low =
      encode_bf16_rne(decode_e2m1(low_code) * scale);
  const std::uint16_t reference_high =
      encode_bf16_rne(decode_e2m1(high_code) * scale);
  reference[index] = static_cast<std::uint32_t>(reference_low) |
                     (static_cast<std::uint32_t>(reference_high) << 16U);
}

// Preserved K64 C16 baseline for the two checkpoint-bound NVFP4 MLP
// projections. Production uses the K128 exact-shape specialization below;
// this implementation remains test-addressable for bitwise and performance
// comparisons.
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

// Preserved fixed-M32 baseline for the two checkpoint-bound NVFP4 MLP
// projections. Production uses the scale-window specialization below. Two
// A[16,K64] panels remain resident in shared memory while
// the decoded B[K64,N128] fragment is loaded once and reused by both independent
// 16-token accumulator chains. B and the complete C[32,N128] tile share one
// static union; the resulting footprint is 8,192 + 4,608 + 18,432 = 31,232
// bytes and requires no dynamic shared memory.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
__global__ __launch_bounds__(kThreads, 5) void
nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 32U;
  constexpr unsigned int kPanelTokenCount = 16U;
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kOutputColumnsPerWarp = 16U;
  constexpr unsigned int kColumnsPerStage = 64U;
  constexpr unsigned int kBf16ValuesPerActivationWord = 4U;
  constexpr unsigned int kActivationWordsPerToken =
      kColumnsPerStage / kBf16ValuesPerActivationWord;
  constexpr unsigned int kSharedActivationWordsPerToken =
      kSharedLeadingDimension / kBf16ValuesPerActivationWord;
  constexpr unsigned int kActivationWordCount =
      kTokenCount * kActivationWordsPerToken;
  constexpr unsigned int kActivationLoadPasses =
      kActivationWordCount / kThreads;
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
  static_assert((kRows == 5'120U && kColumns == 17'408U) ||
                (kRows == 17'408U && kColumns == 5'120U));
  static_assert(kRows % kOutputColumnsPerBlock == 0U);
  static_assert(kColumns % kColumnsPerStage == 0U);
  static_assert(kActivationWordCount == 512U);
  static_assert(kActivationLoadPasses == 2U);
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
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
  wmma::fill_fragment(accumulator0, 0.0F);
  wmma::fill_fragment(accumulator1, 0.0F);
  const unsigned int first_output_column =
      blockIdx.x * kOutputColumnsPerBlock;
  const unsigned int local_output_column = thread / kWeightVectorsPerRow;
  const unsigned int vector_in_row = thread % kWeightVectorsPerRow;
  const unsigned int output_column =
      first_output_column + local_output_column;

#pragma unroll 1
  for (unsigned int first_k = 0U; first_k < kColumns;
       first_k += kColumnsPerStage) {
#pragma unroll
    for (unsigned int load_pass = 0U;
         load_pass < kActivationLoadPasses; ++load_pass) {
      const unsigned int activation_index =
          thread + load_pass * kThreads;
      const unsigned int token =
          activation_index / kActivationWordsPerToken;
      const unsigned int activation_word =
          activation_index % kActivationWordsPerToken;
      shared_activations[token * kSharedActivationWordsPerToken +
                         activation_word] =
          *reinterpret_cast<const std::uint64_t*>(
              activations + static_cast<std::size_t>(token) * kColumns +
              first_k +
              activation_word * kBf16ValuesPerActivationWord);
    }

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
      wmma::load_matrix_sync(
          weight_fragment,
          shared_b +
              warp * kOutputColumnsPerWarp * kSharedLeadingDimension +
              inner_k,
          kSharedLeadingDimension);
      wmma::load_matrix_sync(activation_fragment, shared_a + inner_k,
                             kSharedLeadingDimension);
      wmma::mma_sync(accumulator0, activation_fragment, weight_fragment,
                     accumulator0);
      wmma::load_matrix_sync(
          activation_fragment,
          shared_a + kPanelTokenCount * kSharedLeadingDimension + inner_k,
          kSharedLeadingDimension);
      wmma::mma_sync(accumulator1, activation_fragment, weight_fragment,
                     accumulator1);
    }
    __syncthreads();
  }

  wmma::store_matrix_sync(
      b_or_c.output + warp * kOutputColumnsPerWarp, accumulator0,
      kOutputColumnsPerBlock, wmma::mem_row_major);
  wmma::store_matrix_sync(
      b_or_c.output + kPanelTokenCount * kOutputColumnsPerBlock +
          warp * kOutputColumnsPerWarp,
      accumulator1, kOutputColumnsPerBlock, wmma::mem_row_major);
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

// K256 scale-window family of the fixed-M32 K64 kernel. Production also
// factorizes the signed-product lookup into packed E2M1 pairs and E4M3 scales;
// the full product-table specialization remains test-addressable as the
// previous-production baseline.
// Eight adjacent U16 scale words for each output row are cooperatively loaded
// into the otherwise-unused [64,72) padding of the K64 shared B tile. Four
// consecutive K64 stages then reuse that window, turning the strided per-row
// global scale loads into 16-byte row segments without changing decoded B or
// the WMMA accumulation order.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U,
          bool kFactorizedProductLookup = false,
          bool kVectorizedDecodedStore = false,
          unsigned int kValidTokenCount = 32U>
__global__ __launch_bounds__(kThreads, 5) void
nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 32U;
  constexpr unsigned int kPanelTokenCount = 16U;
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kOutputColumnsPerWarp = 16U;
  constexpr unsigned int kColumnsPerStage = 64U;
  constexpr unsigned int kColumnsPerScaleWindow = 256U;
  constexpr unsigned int kStagesPerScaleWindow =
      kColumnsPerScaleWindow / kColumnsPerStage;
  constexpr unsigned int kScaleWordsPerOutputRow =
      kColumnsPerScaleWindow / kNvFp4GroupSize;
  constexpr unsigned int kBf16ValuesPerActivationWord = 4U;
  constexpr unsigned int kActivationWordsPerToken =
      kColumnsPerStage / kBf16ValuesPerActivationWord;
  constexpr unsigned int kSharedActivationWordsPerToken =
      kSharedLeadingDimension / kBf16ValuesPerActivationWord;
  constexpr unsigned int kActivationWordCount =
      kTokenCount * kActivationWordsPerToken;
  constexpr unsigned int kActivationLoadPasses =
      kActivationWordCount / kThreads;
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
  constexpr unsigned int kScalePaddingFirstWord =
      kColumnsPerStage / kBf16ValuesPerWeightWord;
  static_assert((kRows == 5'120U && kColumns == 17'408U) ||
                (kRows == 17'408U && kColumns == 5'120U));
  static_assert(kRows % kOutputColumnsPerBlock == 0U);
  static_assert(kColumns % kColumnsPerScaleWindow == 0U);
  static_assert(kActivationWordCount == 512U);
  static_assert(kActivationLoadPasses == 2U);
  static_assert(kWeightVectorsPerRow == 2U);
  static_assert(kSharedLeadingDimension == 72U);
  static_assert(kStagesPerScaleWindow == 4U);
  static_assert(kScaleWordsPerOutputRow == 16U);
  static_assert(kScaleWordsPerOutputRow / 2U ==
                kSharedLeadingDimension - kColumnsPerStage);
  static_assert(kProductWordCount == 2'048U);
  static_assert(kProductInitializationPasses == 8U);
  static_assert(!kVectorizedDecodedStore || kFactorizedProductLookup);
  static_assert(kValidTokenCount == 18U || kValidTokenCount == kTokenCount);
  static_assert((kSharedWeightWordsPerRow * sizeof(std::uint32_t)) %
                    alignof(uint4) ==
                0U);
  static_assert(((kValuesPerWeightVector / kBf16ValuesPerWeightWord) *
                 sizeof(std::uint32_t)) %
                    alignof(uint4) ==
                0U);

  union __align__(32) BOrCStorage {
    std::uint32_t weights[kSharedWeightWordCount];
    float output[kSharedOutputCount];
  };
  __shared__ NvFp4M32ProductLookupStorage<kFactorizedProductLookup>
      product_lookup;
  __shared__ __align__(32) std::uint64_t
      shared_activations[kTokenCount * kSharedActivationWordsPerToken];
  __shared__ BOrCStorage b_or_c;

  namespace wmma = nvcuda::wmma;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;

  if constexpr (kFactorizedProductLookup) {
    const std::uint8_t encoded = static_cast<std::uint8_t>(thread);
    const std::uint16_t low =
        encode_bf16_rne(decode_e2m1(encoded & 0x0fU));
    const std::uint16_t high =
        encode_bf16_rne(decode_e2m1(encoded >> 4U));
    product_lookup.e2m1_byte_pairs[thread] =
        static_cast<std::uint32_t>(low) |
        (static_cast<std::uint32_t>(high) << 16U);
    product_lookup.scale_values[thread] =
        encode_bf16_rne(decode_e4m3fn(encoded));
  } else {
    const unsigned int product_pair = thread & 7U;
    const float decoded_low =
        decode_e2m1(static_cast<std::uint8_t>(product_pair * 2U));
    const float decoded_high =
        decode_e2m1(static_cast<std::uint8_t>(product_pair * 2U + 1U));
#pragma unroll 1
    for (unsigned int pass = 0U; pass < kProductInitializationPasses;
         ++pass) {
      const unsigned int scale_code = (thread >> 3U) + pass * 32U;
      const float decoded_scale =
          decode_e4m3fn(static_cast<std::uint8_t>(scale_code));
      const std::uint16_t low =
          encode_bf16_rne(decoded_low * decoded_scale);
      const std::uint16_t high =
          encode_bf16_rne(decoded_high * decoded_scale);
      product_lookup.product_words[thread + pass * kThreads] =
          static_cast<std::uint32_t>(low) |
          (static_cast<std::uint32_t>(high) << 16U);
    }
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
  wmma::fill_fragment(accumulator0, 0.0F);
  wmma::fill_fragment(accumulator1, 0.0F);
  const unsigned int first_output_column =
      blockIdx.x * kOutputColumnsPerBlock;
  const unsigned int local_output_column = thread / kWeightVectorsPerRow;
  const unsigned int vector_in_row = thread % kWeightVectorsPerRow;
  const unsigned int output_column =
      first_output_column + local_output_column;

#pragma unroll 1
  for (unsigned int first_window_k = 0U; first_window_k < kColumns;
       first_window_k += kColumnsPerScaleWindow) {
#pragma unroll
    for (unsigned int pass = 0U; pass < 4U; ++pass) {
      const unsigned int scale_index = thread + pass * kThreads;
      const unsigned int scale_row =
          scale_index / (kScaleWordsPerOutputRow / 2U);
      const unsigned int scale_word =
          scale_index % (kScaleWordsPerOutputRow / 2U);
      const std::uint16_t raw_scale =
          *reinterpret_cast<const std::uint16_t*>(
              block_scales +
              static_cast<std::size_t>(first_output_column + scale_row) *
                  kScaleColumns +
              first_window_k / kNvFp4GroupSize + scale_word * 2U);
      const unsigned int partner_scale = __shfl_down_sync(
          0xffffffffU, static_cast<unsigned int>(raw_scale), 1U);
      if ((scale_word & 1U) == 0U) {
        b_or_c.weights[scale_row * kSharedWeightWordsPerRow +
                       kScalePaddingFirstWord + scale_word / 2U] =
            static_cast<std::uint32_t>(raw_scale) |
            (static_cast<std::uint32_t>(partner_scale) << 16U);
      }
    }
    __syncthreads();

#pragma unroll 1
    for (unsigned int stage = 0U; stage < kStagesPerScaleWindow; ++stage) {
      const unsigned int first_k =
          first_window_k + stage * kColumnsPerStage;
#pragma unroll
      for (unsigned int load_pass = 0U;
           load_pass < kActivationLoadPasses; ++load_pass) {
        const unsigned int activation_index =
            thread + load_pass * kThreads;
        const unsigned int token =
            activation_index / kActivationWordsPerToken;
        const unsigned int activation_word =
            activation_index % kActivationWordsPerToken;
        if constexpr (kValidTokenCount == kTokenCount) {
          shared_activations[token * kSharedActivationWordsPerToken +
                             activation_word] =
              *reinterpret_cast<const std::uint64_t*>(
                  activations + static_cast<std::size_t>(token) * kColumns +
                  first_k +
                  activation_word * kBf16ValuesPerActivationWord);
        } else {
          shared_activations[token * kSharedActivationWordsPerToken +
                             activation_word] =
              token < kValidTokenCount
                  ? *reinterpret_cast<const std::uint64_t*>(
                        activations +
                        static_cast<std::size_t>(token) * kColumns + first_k +
                        activation_word * kBf16ValuesPerActivationWord)
                  : std::uint64_t{0U};
        }
      }

      const unsigned int first_vector_column =
          first_k + vector_in_row * kValuesPerWeightVector;
      const uint4 packed = *reinterpret_cast<const uint4*>(
          packed_weights +
          static_cast<std::size_t>(output_column) * kPackedColumns +
          first_vector_column / kNvFp4ValuesPerByte);
      const std::uint32_t staged_scale_pairs =
          b_or_c.weights[local_output_column * kSharedWeightWordsPerRow +
                         kScalePaddingFirstWord + stage];
      const std::uint16_t encoded_scales = static_cast<std::uint16_t>(
          staged_scale_pairs >> (vector_in_row * 16U));
      const std::uint8_t scale0 =
          static_cast<std::uint8_t>(encoded_scales & 0xffU);
      const std::uint8_t scale1 =
          static_cast<std::uint8_t>(encoded_scales >> 8U);
      std::uint32_t* const decoded =
          b_or_c.weights +
          local_output_column * kSharedWeightWordsPerRow +
          vector_in_row *
              (kValuesPerWeightVector / kBf16ValuesPerWeightWord);
      if constexpr (kFactorizedProductLookup) {
        const std::uint16_t decoded_scale0 =
            product_lookup.scale_values[scale0];
        const std::uint16_t decoded_scale1 =
            product_lookup.scale_values[scale1];
        if constexpr (kVectorizedDecodedStore) {
          auto* const decoded_vectors = reinterpret_cast<uint4*>(decoded);
          decoded_vectors[0] = decode_nvfp4x8_to_bf16x8_factorized_vector(
              packed.x, product_lookup.e2m1_byte_pairs, decoded_scale0);
          decoded_vectors[1] = decode_nvfp4x8_to_bf16x8_factorized_vector(
              packed.y, product_lookup.e2m1_byte_pairs, decoded_scale0);
          decoded_vectors[2] = decode_nvfp4x8_to_bf16x8_factorized_vector(
              packed.z, product_lookup.e2m1_byte_pairs, decoded_scale1);
          decoded_vectors[3] = decode_nvfp4x8_to_bf16x8_factorized_vector(
              packed.w, product_lookup.e2m1_byte_pairs, decoded_scale1);
        } else {
          decode_nvfp4x8_to_bf16x8_factorized(
              packed.x, product_lookup.e2m1_byte_pairs, decoded_scale0,
              decoded);
          decode_nvfp4x8_to_bf16x8_factorized(
              packed.y, product_lookup.e2m1_byte_pairs, decoded_scale0,
              decoded + 4U);
          decode_nvfp4x8_to_bf16x8_factorized(
              packed.z, product_lookup.e2m1_byte_pairs, decoded_scale1,
              decoded + 8U);
          decode_nvfp4x8_to_bf16x8_factorized(
              packed.w, product_lookup.e2m1_byte_pairs, decoded_scale1,
              decoded + 12U);
        }
      } else {
        const auto* const decoded_products =
            reinterpret_cast<const std::uint16_t*>(
                product_lookup.product_words);
        const std::uint16_t* const products0 =
            decoded_products +
            static_cast<unsigned int>(scale0) * kNvFp4EncodedValueCount;
        const std::uint16_t* const products1 =
            decoded_products +
            static_cast<unsigned int>(scale1) * kNvFp4EncodedValueCount;
        decode_nvfp4x8_to_bf16x8(packed.x, products0, decoded);
        decode_nvfp4x8_to_bf16x8(packed.y, products0, decoded + 4U);
        decode_nvfp4x8_to_bf16x8(packed.z, products1, decoded + 8U);
        decode_nvfp4x8_to_bf16x8(packed.w, products1, decoded + 12U);
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
        wmma::load_matrix_sync(
            weight_fragment,
            shared_b +
                warp * kOutputColumnsPerWarp * kSharedLeadingDimension +
                inner_k,
            kSharedLeadingDimension);
        wmma::load_matrix_sync(activation_fragment, shared_a + inner_k,
                               kSharedLeadingDimension);
        wmma::mma_sync(accumulator0, activation_fragment, weight_fragment,
                       accumulator0);
        wmma::load_matrix_sync(
            activation_fragment,
            shared_a + kPanelTokenCount * kSharedLeadingDimension + inner_k,
            kSharedLeadingDimension);
        wmma::mma_sync(accumulator1, activation_fragment, weight_fragment,
                       accumulator1);
      }
      __syncthreads();
    }
  }

  wmma::store_matrix_sync(
      b_or_c.output + warp * kOutputColumnsPerWarp, accumulator0,
      kOutputColumnsPerBlock, wmma::mem_row_major);
  wmma::store_matrix_sync(
      b_or_c.output + kPanelTokenCount * kOutputColumnsPerBlock +
          warp * kOutputColumnsPerWarp,
      accumulator1, kOutputColumnsPerBlock, wmma::mem_row_major);
  __syncthreads();

#pragma unroll
  for (unsigned int index = thread; index < kSharedOutputCount;
       index += kThreads) {
    const unsigned int token = index / kOutputColumnsPerBlock;
    const unsigned int local_column = index % kOutputColumnsPerBlock;
    if constexpr (kValidTokenCount == kTokenCount) {
      output[static_cast<std::size_t>(token) * kRows + first_output_column +
             local_column] =
          encode_bf16_rne(b_or_c.output[index] * weight_scale_2);
    } else if (token < kValidTokenCount) {
      output[static_cast<std::size_t>(token) * kRows + first_output_column +
             local_column] =
          encode_bf16_rne(b_or_c.output[index] * weight_scale_2);
    }
  }
}

// Production runtime-valid-count form of the factorized/vector-store M32
// design. The first 16-token panel is always valid; the second panel reads
// only rows below valid_token_count and zero-fills the rest. The epilogue
// likewise writes only the exact caller-owned [valid_token_count, rows] span.
// Keeping this as a separate kernel leaves the validated fixed-M18 and M32
// production specializations and their SASS unchanged while serving one
// cubin instance per checkpoint shape for M=17 and M=19..31.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
__global__ __launch_bounds__(kThreads, 5) void
nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    const unsigned int valid_token_count) {
  constexpr unsigned int kTokenCount = 32U;
  constexpr unsigned int kPanelTokenCount = 16U;
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kOutputColumnsPerWarp = 16U;
  constexpr unsigned int kColumnsPerStage = 64U;
  constexpr unsigned int kColumnsPerScaleWindow = 256U;
  constexpr unsigned int kStagesPerScaleWindow =
      kColumnsPerScaleWindow / kColumnsPerStage;
  constexpr unsigned int kScaleWordsPerOutputRow =
      kColumnsPerScaleWindow / kNvFp4GroupSize;
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
  constexpr unsigned int kScalePaddingFirstWord =
      kColumnsPerStage / kBf16ValuesPerWeightWord;
  static_assert((kRows == 5'120U && kColumns == 17'408U) ||
                (kRows == 17'408U && kColumns == 5'120U));
  static_assert(kRows % kOutputColumnsPerBlock == 0U);
  static_assert(kColumns % kColumnsPerScaleWindow == 0U);
  static_assert(kActivationWordsPerToken * kTokenCount == 512U);
  static_assert(kWeightVectorsPerRow == 2U);
  static_assert(kSharedLeadingDimension == 72U);
  static_assert(kStagesPerScaleWindow == 4U);
  static_assert(kScaleWordsPerOutputRow == 16U);
  static_assert(kScaleWordsPerOutputRow / 2U ==
                kSharedLeadingDimension - kColumnsPerStage);
  static_assert((kSharedWeightWordsPerRow * sizeof(std::uint32_t)) %
                    alignof(uint4) ==
                0U);
  static_assert(((kValuesPerWeightVector / kBf16ValuesPerWeightWord) *
                 sizeof(std::uint32_t)) %
                    alignof(uint4) ==
                0U);

  union __align__(32) BOrCStorage {
    std::uint32_t weights[kSharedWeightWordCount];
    float output[kSharedOutputCount];
  };
  __shared__ NvFp4M32ProductLookupStorage<true> product_lookup;
  __shared__ __align__(32) std::uint64_t
      shared_activations[kTokenCount * kSharedActivationWordsPerToken];
  __shared__ BOrCStorage b_or_c;

  namespace wmma = nvcuda::wmma;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;

  const std::uint8_t encoded = static_cast<std::uint8_t>(thread);
  const std::uint16_t low =
      encode_bf16_rne(decode_e2m1(encoded & 0x0fU));
  const std::uint16_t high =
      encode_bf16_rne(decode_e2m1(encoded >> 4U));
  product_lookup.e2m1_byte_pairs[thread] =
      static_cast<std::uint32_t>(low) |
      (static_cast<std::uint32_t>(high) << 16U);
  product_lookup.scale_values[thread] =
      encode_bf16_rne(decode_e4m3fn(encoded));
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
  wmma::fill_fragment(accumulator0, 0.0F);
  wmma::fill_fragment(accumulator1, 0.0F);
  const unsigned int first_output_column =
      blockIdx.x * kOutputColumnsPerBlock;
  const unsigned int local_output_column = thread / kWeightVectorsPerRow;
  const unsigned int vector_in_row = thread % kWeightVectorsPerRow;
  const unsigned int output_column =
      first_output_column + local_output_column;

#pragma unroll 1
  for (unsigned int first_window_k = 0U; first_window_k < kColumns;
       first_window_k += kColumnsPerScaleWindow) {
#pragma unroll
    for (unsigned int pass = 0U; pass < 4U; ++pass) {
      const unsigned int scale_index = thread + pass * kThreads;
      const unsigned int scale_row =
          scale_index / (kScaleWordsPerOutputRow / 2U);
      const unsigned int scale_word =
          scale_index % (kScaleWordsPerOutputRow / 2U);
      const std::uint16_t raw_scale =
          *reinterpret_cast<const std::uint16_t*>(
              block_scales +
              static_cast<std::size_t>(first_output_column + scale_row) *
                  kScaleColumns +
              first_window_k / kNvFp4GroupSize + scale_word * 2U);
      const unsigned int partner_scale = __shfl_down_sync(
          0xffffffffU, static_cast<unsigned int>(raw_scale), 1U);
      if ((scale_word & 1U) == 0U) {
        b_or_c.weights[scale_row * kSharedWeightWordsPerRow +
                       kScalePaddingFirstWord + scale_word / 2U] =
            static_cast<std::uint32_t>(raw_scale) |
            (static_cast<std::uint32_t>(partner_scale) << 16U);
      }
    }
    __syncthreads();

#pragma unroll 1
    for (unsigned int stage = 0U; stage < kStagesPerScaleWindow; ++stage) {
      const unsigned int first_k =
          first_window_k + stage * kColumnsPerStage;

      const unsigned int first_token = thread / kActivationWordsPerToken;
      const unsigned int first_activation_word =
          thread % kActivationWordsPerToken;
      shared_activations[first_token * kSharedActivationWordsPerToken +
                         first_activation_word] =
          *reinterpret_cast<const std::uint64_t*>(
              activations + static_cast<std::size_t>(first_token) * kColumns +
              first_k +
              first_activation_word * kBf16ValuesPerActivationWord);

      const unsigned int second_activation_index = thread + kThreads;
      const unsigned int second_token =
          second_activation_index / kActivationWordsPerToken;
      const unsigned int second_activation_word =
          second_activation_index % kActivationWordsPerToken;
      shared_activations[second_token * kSharedActivationWordsPerToken +
                         second_activation_word] =
          second_token < valid_token_count
              ? *reinterpret_cast<const std::uint64_t*>(
                    activations +
                    static_cast<std::size_t>(second_token) * kColumns +
                    first_k + second_activation_word *
                                  kBf16ValuesPerActivationWord)
              : std::uint64_t{0U};

      const unsigned int first_vector_column =
          first_k + vector_in_row * kValuesPerWeightVector;
      const uint4 packed = *reinterpret_cast<const uint4*>(
          packed_weights +
          static_cast<std::size_t>(output_column) * kPackedColumns +
          first_vector_column / kNvFp4ValuesPerByte);
      const std::uint32_t staged_scale_pairs =
          b_or_c.weights[local_output_column * kSharedWeightWordsPerRow +
                         kScalePaddingFirstWord + stage];
      const std::uint16_t encoded_scales = static_cast<std::uint16_t>(
          staged_scale_pairs >> (vector_in_row * 16U));
      const std::uint8_t scale0 =
          static_cast<std::uint8_t>(encoded_scales & 0xffU);
      const std::uint8_t scale1 =
          static_cast<std::uint8_t>(encoded_scales >> 8U);
      const std::uint16_t decoded_scale0 =
          product_lookup.scale_values[scale0];
      const std::uint16_t decoded_scale1 =
          product_lookup.scale_values[scale1];
      std::uint32_t* const decoded =
          b_or_c.weights +
          local_output_column * kSharedWeightWordsPerRow +
          vector_in_row *
              (kValuesPerWeightVector / kBf16ValuesPerWeightWord);
      auto* const decoded_vectors = reinterpret_cast<uint4*>(decoded);
      decoded_vectors[0] = decode_nvfp4x8_to_bf16x8_factorized_vector(
          packed.x, product_lookup.e2m1_byte_pairs, decoded_scale0);
      decoded_vectors[1] = decode_nvfp4x8_to_bf16x8_factorized_vector(
          packed.y, product_lookup.e2m1_byte_pairs, decoded_scale0);
      decoded_vectors[2] = decode_nvfp4x8_to_bf16x8_factorized_vector(
          packed.z, product_lookup.e2m1_byte_pairs, decoded_scale1);
      decoded_vectors[3] = decode_nvfp4x8_to_bf16x8_factorized_vector(
          packed.w, product_lookup.e2m1_byte_pairs, decoded_scale1);
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
        wmma::load_matrix_sync(
            weight_fragment,
            shared_b +
                warp * kOutputColumnsPerWarp * kSharedLeadingDimension +
                inner_k,
            kSharedLeadingDimension);
        wmma::load_matrix_sync(activation_fragment, shared_a + inner_k,
                               kSharedLeadingDimension);
        wmma::mma_sync(accumulator0, activation_fragment, weight_fragment,
                       accumulator0);
        wmma::load_matrix_sync(
            activation_fragment,
            shared_a + kPanelTokenCount * kSharedLeadingDimension + inner_k,
            kSharedLeadingDimension);
        wmma::mma_sync(accumulator1, activation_fragment, weight_fragment,
                       accumulator1);
      }
      __syncthreads();
    }
  }

  wmma::store_matrix_sync(
      b_or_c.output + warp * kOutputColumnsPerWarp, accumulator0,
      kOutputColumnsPerBlock, wmma::mem_row_major);
  wmma::store_matrix_sync(
      b_or_c.output + kPanelTokenCount * kOutputColumnsPerBlock +
          warp * kOutputColumnsPerWarp,
      accumulator1, kOutputColumnsPerBlock, wmma::mem_row_major);
  __syncthreads();

#pragma unroll
  for (unsigned int index = thread; index < kSharedOutputCount;
       index += kThreads) {
    const unsigned int token = index / kOutputColumnsPerBlock;
    const unsigned int local_column = index % kOutputColumnsPerBlock;
    if (token < valid_token_count) {
      output[static_cast<std::size_t>(token) * kRows + first_output_column +
             local_column] =
          encode_bf16_rne(b_or_c.output[index] * weight_scale_2);
    }
  }
}

// Test-only single-slot raw-weight cp.async candidate. A K64 x N128 packed
// weight tile is exactly 4 KiB: every thread owns one aligned uint4 cell.
// After waiting for the current stage, the cell is copied into registers
// before the next stage starts overwriting the same shared-memory address.
// This provides register/shared logical double buffering without a second
// shared slot or an additional CTA-wide barrier.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
__global__ __launch_bounds__(kThreads, 5) void
nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_raw_weight_cp_async_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    const unsigned int valid_token_count) {
  constexpr unsigned int kTokenCount = 32U;
  constexpr unsigned int kPanelTokenCount = 16U;
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kOutputColumnsPerWarp = 16U;
  constexpr unsigned int kColumnsPerStage = 64U;
  constexpr unsigned int kColumnsPerScaleWindow = 256U;
  constexpr unsigned int kStagesPerScaleWindow =
      kColumnsPerScaleWindow / kColumnsPerStage;
  constexpr unsigned int kScaleWordsPerOutputRow =
      kColumnsPerScaleWindow / kNvFp4GroupSize;
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
  constexpr unsigned int kScalePaddingFirstWord =
      kColumnsPerStage / kBf16ValuesPerWeightWord;
  static_assert((kRows == 5'120U && kColumns == 17'408U) ||
                (kRows == 17'408U && kColumns == 5'120U));
  static_assert(kRows % kOutputColumnsPerBlock == 0U);
  static_assert(kColumns % kColumnsPerScaleWindow == 0U);
  static_assert(kActivationWordsPerToken * kTokenCount == 512U);
  static_assert(kWeightVectorsPerRow == 2U);
  static_assert(kSharedLeadingDimension == 72U);
  static_assert(kStagesPerScaleWindow == 4U);
  static_assert(kScaleWordsPerOutputRow == 16U);
  static_assert(kScaleWordsPerOutputRow / 2U ==
                kSharedLeadingDimension - kColumnsPerStage);
  static_assert(kThreads * sizeof(uint4) == 4'096U);
  static_assert((kSharedWeightWordsPerRow * sizeof(std::uint32_t)) %
                    alignof(uint4) ==
                0U);
  static_assert(((kValuesPerWeightVector / kBf16ValuesPerWeightWord) *
                 sizeof(std::uint32_t)) %
                    alignof(uint4) ==
                0U);

  union __align__(32) BOrCStorage {
    std::uint32_t weights[kSharedWeightWordCount];
    float output[kSharedOutputCount];
  };
  __shared__ NvFp4M32ProductLookupStorage<true> product_lookup;
  __shared__ __align__(32) std::uint64_t
      shared_activations[kTokenCount * kSharedActivationWordsPerToken];
  __shared__ BOrCStorage b_or_c;
  __shared__ __align__(16) uint4 shared_raw_weights[kThreads];

  namespace wmma = nvcuda::wmma;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;

  const std::uint8_t encoded = static_cast<std::uint8_t>(thread);
  const std::uint16_t low =
      encode_bf16_rne(decode_e2m1(encoded & 0x0fU));
  const std::uint16_t high =
      encode_bf16_rne(decode_e2m1(encoded >> 4U));
  product_lookup.e2m1_byte_pairs[thread] =
      static_cast<std::uint32_t>(low) |
      (static_cast<std::uint32_t>(high) << 16U);
  product_lookup.scale_values[thread] =
      encode_bf16_rne(decode_e4m3fn(encoded));
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
  wmma::fill_fragment(accumulator0, 0.0F);
  wmma::fill_fragment(accumulator1, 0.0F);
  const unsigned int first_output_column =
      blockIdx.x * kOutputColumnsPerBlock;
  const unsigned int local_output_column = thread / kWeightVectorsPerRow;
  const unsigned int vector_in_row = thread % kWeightVectorsPerRow;
  const unsigned int output_column =
      first_output_column + local_output_column;

  const unsigned int initial_vector_column =
      vector_in_row * kValuesPerWeightVector;
  cp_async_cg_shared_global_16(
      shared_raw_weights + thread,
      packed_weights +
          static_cast<std::size_t>(output_column) * kPackedColumns +
          initial_vector_column / kNvFp4ValuesPerByte);
  cp_async_commit_group();

#pragma unroll 1
  for (unsigned int first_window_k = 0U; first_window_k < kColumns;
       first_window_k += kColumnsPerScaleWindow) {
#pragma unroll
    for (unsigned int pass = 0U; pass < 4U; ++pass) {
      const unsigned int scale_index = thread + pass * kThreads;
      const unsigned int scale_row =
          scale_index / (kScaleWordsPerOutputRow / 2U);
      const unsigned int scale_word =
          scale_index % (kScaleWordsPerOutputRow / 2U);
      const std::uint16_t raw_scale =
          *reinterpret_cast<const std::uint16_t*>(
              block_scales +
              static_cast<std::size_t>(first_output_column + scale_row) *
                  kScaleColumns +
              first_window_k / kNvFp4GroupSize + scale_word * 2U);
      const unsigned int partner_scale = __shfl_down_sync(
          0xffffffffU, static_cast<unsigned int>(raw_scale), 1U);
      if ((scale_word & 1U) == 0U) {
        b_or_c.weights[scale_row * kSharedWeightWordsPerRow +
                       kScalePaddingFirstWord + scale_word / 2U] =
            static_cast<std::uint32_t>(raw_scale) |
            (static_cast<std::uint32_t>(partner_scale) << 16U);
      }
    }
    __syncthreads();

#pragma unroll 1
    for (unsigned int stage = 0U; stage < kStagesPerScaleWindow; ++stage) {
      const unsigned int first_k =
          first_window_k + stage * kColumnsPerStage;

      cp_async_wait_group_0();
      const uint4 packed = shared_raw_weights[thread];

      const unsigned int first_token = thread / kActivationWordsPerToken;
      const unsigned int first_activation_word =
          thread % kActivationWordsPerToken;
      shared_activations[first_token * kSharedActivationWordsPerToken +
                         first_activation_word] =
          *reinterpret_cast<const std::uint64_t*>(
              activations + static_cast<std::size_t>(first_token) * kColumns +
              first_k +
              first_activation_word * kBf16ValuesPerActivationWord);

      const unsigned int second_activation_index = thread + kThreads;
      const unsigned int second_token =
          second_activation_index / kActivationWordsPerToken;
      const unsigned int second_activation_word =
          second_activation_index % kActivationWordsPerToken;
      shared_activations[second_token * kSharedActivationWordsPerToken +
                         second_activation_word] =
          second_token < valid_token_count
              ? *reinterpret_cast<const std::uint64_t*>(
                    activations +
                    static_cast<std::size_t>(second_token) * kColumns +
                    first_k + second_activation_word *
                                  kBf16ValuesPerActivationWord)
              : std::uint64_t{0U};

      const std::uint32_t staged_scale_pairs =
          b_or_c.weights[local_output_column * kSharedWeightWordsPerRow +
                         kScalePaddingFirstWord + stage];
      const std::uint16_t encoded_scales = static_cast<std::uint16_t>(
          staged_scale_pairs >> (vector_in_row * 16U));
      const std::uint8_t scale0 =
          static_cast<std::uint8_t>(encoded_scales & 0xffU);
      const std::uint8_t scale1 =
          static_cast<std::uint8_t>(encoded_scales >> 8U);
      const std::uint16_t decoded_scale0 =
          product_lookup.scale_values[scale0];
      const std::uint16_t decoded_scale1 =
          product_lookup.scale_values[scale1];
      std::uint32_t* const decoded =
          b_or_c.weights +
          local_output_column * kSharedWeightWordsPerRow +
          vector_in_row *
              (kValuesPerWeightVector / kBf16ValuesPerWeightWord);
      auto* const decoded_vectors = reinterpret_cast<uint4*>(decoded);
      decoded_vectors[0] = decode_nvfp4x8_to_bf16x8_factorized_vector(
          packed.x, product_lookup.e2m1_byte_pairs, decoded_scale0);
      decoded_vectors[1] = decode_nvfp4x8_to_bf16x8_factorized_vector(
          packed.y, product_lookup.e2m1_byte_pairs, decoded_scale0);
      decoded_vectors[2] = decode_nvfp4x8_to_bf16x8_factorized_vector(
          packed.z, product_lookup.e2m1_byte_pairs, decoded_scale1);
      decoded_vectors[3] = decode_nvfp4x8_to_bf16x8_factorized_vector(
          packed.w, product_lookup.e2m1_byte_pairs, decoded_scale1);

      // All four packed words have logically been consumed before reusing the
      // single raw slot. The SASS acceptance gate requires the raw LDS.128 to
      // precede this next LDGSTS; decoded shared-B stores may be scheduled
      // later because they no longer read the raw slot.
      const unsigned int next_first_k = first_k + kColumnsPerStage;
      if (next_first_k < kColumns) {
        const unsigned int next_vector_column =
            next_first_k + vector_in_row * kValuesPerWeightVector;
        cp_async_cg_shared_global_16(
            shared_raw_weights + thread,
            packed_weights +
                static_cast<std::size_t>(output_column) * kPackedColumns +
                next_vector_column / kNvFp4ValuesPerByte);
        cp_async_commit_group();
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
        wmma::load_matrix_sync(
            weight_fragment,
            shared_b +
                warp * kOutputColumnsPerWarp * kSharedLeadingDimension +
                inner_k,
            kSharedLeadingDimension);
        wmma::load_matrix_sync(activation_fragment, shared_a + inner_k,
                               kSharedLeadingDimension);
        wmma::mma_sync(accumulator0, activation_fragment, weight_fragment,
                       accumulator0);
        wmma::load_matrix_sync(
            activation_fragment,
            shared_a + kPanelTokenCount * kSharedLeadingDimension + inner_k,
            kSharedLeadingDimension);
        wmma::mma_sync(accumulator1, activation_fragment, weight_fragment,
                       accumulator1);
      }
      __syncthreads();
    }
  }

  wmma::store_matrix_sync(
      b_or_c.output + warp * kOutputColumnsPerWarp, accumulator0,
      kOutputColumnsPerBlock, wmma::mem_row_major);
  wmma::store_matrix_sync(
      b_or_c.output + kPanelTokenCount * kOutputColumnsPerBlock +
          warp * kOutputColumnsPerWarp,
      accumulator1, kOutputColumnsPerBlock, wmma::mem_row_major);
  __syncthreads();

#pragma unroll
  for (unsigned int index = thread; index < kSharedOutputCount;
       index += kThreads) {
    const unsigned int token = index / kOutputColumnsPerBlock;
    const unsigned int local_column = index % kOutputColumnsPerBlock;
    if (token < valid_token_count) {
      output[static_cast<std::size_t>(token) * kRows + first_output_column +
             local_column] =
          encode_bf16_rne(b_or_c.output[index] * weight_scale_2);
    }
  }
}

// Production exact-shape specialization that doubles the K stage from 64 to
// 128 for both checkpoint-bound NVFP4 MLP projections.
// Two cooperative load passes fill A[16,128] and B[128,128] with a padded
// shared leading dimension of 136. The eight K16 WMMA operations retain the
// exact order of two consecutive production K64 stages; the stride-16 product
// table and the N=128 CTA/warp mapping are otherwise unchanged.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 136U>
__global__ __launch_bounds__(kThreads, 3) void
nvfp4_w4a16_small_m16_gemm_bf16_wmma_k128_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 16U;
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kOutputColumnsPerWarp = 16U;
  constexpr unsigned int kColumnsPerStage = 128U;
  constexpr unsigned int kBf16ValuesPerActivationWord = 4U;
  constexpr unsigned int kActivationWordsPerToken =
      kColumnsPerStage / kBf16ValuesPerActivationWord;
  constexpr unsigned int kSharedActivationWordsPerToken =
      kSharedLeadingDimension / kBf16ValuesPerActivationWord;
  constexpr unsigned int kActivationWordCount =
      kTokenCount * kActivationWordsPerToken;
  constexpr unsigned int kActivationLoadPasses =
      kActivationWordCount / kThreads;
  constexpr unsigned int kPackedColumns =
      static_cast<unsigned int>(kColumns / kNvFp4ValuesPerByte);
  constexpr unsigned int kScaleColumns =
      static_cast<unsigned int>(kColumns / kNvFp4GroupSize);
  constexpr unsigned int kValuesPerWeightVector =
      sizeof(uint4) * kNvFp4ValuesPerByte;
  constexpr unsigned int kWeightVectorsPerRow =
      kColumnsPerStage / kValuesPerWeightVector;
  constexpr unsigned int kWeightLoadPasses = 2U;
  constexpr unsigned int kWeightLoaderThreadsPerRow =
      kWeightVectorsPerRow / kWeightLoadPasses;
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
  static_assert((kRows == 5'120U && kColumns == 17'408U) ||
                (kRows == 17'408U && kColumns == 5'120U));
  static_assert(kRows % kOutputColumnsPerBlock == 0U);
  static_assert(kColumns % kColumnsPerStage == 0U);
  static_assert(kActivationWordCount == 512U);
  static_assert(kActivationLoadPasses == 2U);
  static_assert(kWeightVectorsPerRow == 4U);
  static_assert(kWeightLoadPasses == 2U);
  static_assert(kWeightLoaderThreadsPerRow == 2U);
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
  const unsigned int local_output_column =
      thread / kWeightLoaderThreadsPerRow;
  const unsigned int vector_in_row_lane =
      thread % kWeightLoaderThreadsPerRow;
  const unsigned int output_column =
      first_output_column + local_output_column;

#pragma unroll 1
  for (unsigned int first_k = 0U; first_k < kColumns;
       first_k += kColumnsPerStage) {
#pragma unroll
    for (unsigned int load_pass = 0U;
         load_pass < kActivationLoadPasses; ++load_pass) {
      const unsigned int activation_index =
          thread + load_pass * kThreads;
      const unsigned int token =
          activation_index / kActivationWordsPerToken;
      const unsigned int activation_word =
          activation_index % kActivationWordsPerToken;
      shared_activations[token * kSharedActivationWordsPerToken +
                         activation_word] =
          *reinterpret_cast<const std::uint64_t*>(
              activations + static_cast<std::size_t>(token) * kColumns +
              first_k +
              activation_word * kBf16ValuesPerActivationWord);
    }

#pragma unroll
    for (unsigned int load_pass = 0U; load_pass < kWeightLoadPasses;
         ++load_pass) {
      const unsigned int vector_in_row =
          vector_in_row_lane +
          load_pass * kWeightLoaderThreadsPerRow;
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
          static_cast<unsigned int>(scale0) *
              kNvFp4EncodedValueCount;
      const std::uint16_t* const products1 =
          decoded_products +
          static_cast<unsigned int>(scale1) *
              kNvFp4EncodedValueCount;
      std::uint32_t* const decoded =
          b_or_c.weights +
          local_output_column * kSharedWeightWordsPerRow +
          vector_in_row *
              (kValuesPerWeightVector / kBf16ValuesPerWeightWord);
      decode_nvfp4x8_to_bf16x8(packed.x, products0, decoded);
      decode_nvfp4x8_to_bf16x8(packed.y, products0, decoded + 4U);
      decode_nvfp4x8_to_bf16x8(packed.z, products1, decoded + 8U);
      decode_nvfp4x8_to_bf16x8(packed.w, products1, decoded + 12U);
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

// Test-only fixed-M32 candidate B. It preserves the production K128/LD136
// shared-memory footprint by keeping a single 16-token A panel resident. B is
// decoded once per K128 stage, A is overwritten between the two independent
// accumulator chains, and B is reused for the second panel. The complete
// C[32,N128] tile remains smaller than B, so static shared memory stays at
// 47,360 bytes and no dynamic shared memory is required.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 136U>
__global__ __launch_bounds__(kThreads, 3) void
nvfp4_w4a16_small_m32_gemm_bf16_wmma_k128_single_a_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 32U;
  constexpr unsigned int kPanelTokenCount = 16U;
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kOutputColumnsPerWarp = 16U;
  constexpr unsigned int kColumnsPerStage = 128U;
  constexpr unsigned int kBf16ValuesPerActivationWord = 4U;
  constexpr unsigned int kActivationWordsPerToken =
      kColumnsPerStage / kBf16ValuesPerActivationWord;
  constexpr unsigned int kSharedActivationWordsPerToken =
      kSharedLeadingDimension / kBf16ValuesPerActivationWord;
  constexpr unsigned int kActivationWordCount =
      kPanelTokenCount * kActivationWordsPerToken;
  constexpr unsigned int kActivationLoadPasses =
      kActivationWordCount / kThreads;
  constexpr unsigned int kPackedColumns =
      static_cast<unsigned int>(kColumns / kNvFp4ValuesPerByte);
  constexpr unsigned int kScaleColumns =
      static_cast<unsigned int>(kColumns / kNvFp4GroupSize);
  constexpr unsigned int kValuesPerWeightVector =
      sizeof(uint4) * kNvFp4ValuesPerByte;
  constexpr unsigned int kWeightVectorsPerRow =
      kColumnsPerStage / kValuesPerWeightVector;
  constexpr unsigned int kWeightLoadPasses = 2U;
  constexpr unsigned int kWeightLoaderThreadsPerRow =
      kWeightVectorsPerRow / kWeightLoadPasses;
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
  static_assert((kRows == 5'120U && kColumns == 17'408U) ||
                (kRows == 17'408U && kColumns == 5'120U));
  static_assert(kRows % kOutputColumnsPerBlock == 0U);
  static_assert(kColumns % kColumnsPerStage == 0U);
  static_assert(kActivationWordCount == 512U);
  static_assert(kActivationLoadPasses == 2U);
  static_assert(kWeightVectorsPerRow == 4U);
  static_assert(kWeightLoadPasses == 2U);
  static_assert(kWeightLoaderThreadsPerRow == 2U);
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
      shared_activations[kPanelTokenCount *
                         kSharedActivationWordsPerToken];
  __shared__ BOrCStorage b_or_c;

  namespace wmma = nvcuda::wmma;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;

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
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
  wmma::fill_fragment(accumulator0, 0.0F);
  wmma::fill_fragment(accumulator1, 0.0F);
  const unsigned int first_output_column =
      blockIdx.x * kOutputColumnsPerBlock;
  const unsigned int local_output_column =
      thread / kWeightLoaderThreadsPerRow;
  const unsigned int vector_in_row_lane =
      thread % kWeightLoaderThreadsPerRow;
  const unsigned int output_column =
      first_output_column + local_output_column;

#pragma unroll 1
  for (unsigned int first_k = 0U; first_k < kColumns;
       first_k += kColumnsPerStage) {
#pragma unroll
    for (unsigned int load_pass = 0U;
         load_pass < kActivationLoadPasses; ++load_pass) {
      const unsigned int activation_index =
          thread + load_pass * kThreads;
      const unsigned int token =
          activation_index / kActivationWordsPerToken;
      const unsigned int activation_word =
          activation_index % kActivationWordsPerToken;
      shared_activations[token * kSharedActivationWordsPerToken +
                         activation_word] =
          *reinterpret_cast<const std::uint64_t*>(
              activations + static_cast<std::size_t>(token) * kColumns +
              first_k +
              activation_word * kBf16ValuesPerActivationWord);
    }

#pragma unroll
    for (unsigned int load_pass = 0U; load_pass < kWeightLoadPasses;
         ++load_pass) {
      const unsigned int vector_in_row =
          vector_in_row_lane +
          load_pass * kWeightLoaderThreadsPerRow;
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
      wmma::mma_sync(accumulator0, activation_fragment, weight_fragment,
                     accumulator0);
    }
    // Every warp must finish reading panel 0 before shared A is overwritten.
    __syncthreads();

#pragma unroll
    for (unsigned int load_pass = 0U;
         load_pass < kActivationLoadPasses; ++load_pass) {
      const unsigned int activation_index =
          thread + load_pass * kThreads;
      const unsigned int token =
          activation_index / kActivationWordsPerToken;
      const unsigned int activation_word =
          activation_index % kActivationWordsPerToken;
      shared_activations[token * kSharedActivationWordsPerToken +
                         activation_word] =
          *reinterpret_cast<const std::uint64_t*>(
              activations +
              static_cast<std::size_t>(token + kPanelTokenCount) * kColumns +
              first_k +
              activation_word * kBf16ValuesPerActivationWord);
    }
    __syncthreads();

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
      wmma::mma_sync(accumulator1, activation_fragment, weight_fragment,
                     accumulator1);
    }
    __syncthreads();
  }

  wmma::store_matrix_sync(
      b_or_c.output + warp * kOutputColumnsPerWarp, accumulator0,
      kOutputColumnsPerBlock, wmma::mem_row_major);
  wmma::store_matrix_sync(
      b_or_c.output + kPanelTokenCount * kOutputColumnsPerBlock +
          warp * kOutputColumnsPerWarp,
      accumulator1, kOutputColumnsPerBlock, wmma::mem_row_major);
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

// M=1 row-pair A/B baseline. A warp evaluates two adjacent output
// rows so both rows share each BF16 activation load and decode, while their
// packed weights, group scales, four accumulation chains, and reductions stay
// independent. Keeping the four chains per row preserves the production
// kernel's exact floating-point operation order.
__global__ __launch_bounds__(kThreads, 5) void
nvfp4_w4a16_gemv_bf16_scale_codebook_row_pair_kernel(
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
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};

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
        const std::uint64_t packed_activation =
            *reinterpret_cast<const std::uint64_t*>(
                activation + first_column + half * 4U);
#pragma unroll
        for (unsigned int value = 0U; value < 4U; ++value) {
          const unsigned int packed_value = half * 4U + value;
          const unsigned int shift = packed_value * 4U;
          const std::uint8_t nibble0 = static_cast<std::uint8_t>(
              (packed0 >> shift) & 0x0fU);
          const std::uint8_t nibble1 = static_cast<std::uint8_t>(
              (packed1 >> shift) & 0x0fU);
          const std::uint16_t encoded_activation =
              static_cast<std::uint16_t>(
                  (packed_activation >> (value * 16U)) & 0xffffU);
          const float decoded_activation =
              decode_bf16(encoded_activation);
          const float scaled_weight0 =
              decoded_weights[nibble0] * block_scale0;
          accumulators0[value] =
              fmaf(scaled_weight0, decoded_activation,
                   accumulators0[value]);
          const float scaled_weight1 =
              decoded_weights[nibble1] * block_scale1;
          accumulators1[value] =
              fmaf(scaled_weight1, decoded_activation,
                   accumulators1[value]);
        }
      }
    }

    float sum0 = (accumulators0[0] + accumulators0[1]) +
                 (accumulators0[2] + accumulators0[3]);
    float sum1 = (accumulators1[0] + accumulators1[1]) +
                 (accumulators1[2] + accumulators1[3]);
    sum0 = warp_sum(sum0) * weight_scale_2;
    sum1 = warp_sum(sum1) * weight_scale_2;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(sum0);
      if (has_row1) {
        output[row1] = encode_bf16_rne(sum1);
      }
    }
  }
}

// Preserved test-only M=1 row-quad baseline. Four adjacent output rows share
// every BF16 activation load and decode while retaining independent packed
// weights, scales, four-way accumulation chains, and reductions. The direct
// capped test ABI keeps this generic implementation for production A/B.
template <bool CompleteRowQuads>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_kernel(
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
      4U * (static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp);
  const std::size_t row_stride =
      static_cast<std::size_t>(gridDim.x) * kWarpsPerBlock * 4U;

  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row1 = row0 + 1U;
    const std::size_t row2 = row0 + 2U;
    const std::size_t row3 = row0 + 3U;
    const bool has_row1 = CompleteRowQuads || row1 < rows;
    const bool has_row2 = CompleteRowQuads || row2 < rows;
    const bool has_row3 = CompleteRowQuads || row3 < rows;
    const std::uint8_t* const row0_weights =
        packed_weights + row0 * packed_columns;
    const std::uint8_t* const row0_scales =
        block_scales + row0 * scale_columns;
    const std::uint8_t* const row1_weights =
        has_row1 ? packed_weights + row1 * packed_columns : row0_weights;
    const std::uint8_t* const row1_scales =
        has_row1 ? block_scales + row1 * scale_columns : row0_scales;
    const std::uint8_t* const row2_weights =
        has_row2 ? packed_weights + row2 * packed_columns : row0_weights;
    const std::uint8_t* const row2_scales =
        has_row2 ? block_scales + row2 * scale_columns : row0_scales;
    const std::uint8_t* const row3_weights =
        has_row3 ? packed_weights + row3 * packed_columns : row0_weights;
    const std::uint8_t* const row3_scales =
        has_row3 ? block_scales + row3 * scale_columns : row0_scales;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

    for (std::size_t packed_column =
             static_cast<std::size_t>(lane) *
             kNvFp4VectorPackedBytesPerLane;
         packed_column < packed_columns;
         packed_column += kNvFp4VectorColumnsPerWarp /
                          kNvFp4ValuesPerByte) {
      float block_scale0 = 0.0F;
      float block_scale1 = 0.0F;
      float block_scale2 = 0.0F;
      float block_scale3 = 0.0F;
      if ((lane & 1U) == 0U) {
        const std::size_t scale_column =
            packed_column / kNvFp4PackedValuesPerScale;
        block_scale0 = decoded_scales[row0_scales[scale_column]];
        if (has_row1) {
          block_scale1 = decoded_scales[row1_scales[scale_column]];
        }
        if (has_row2) {
          block_scale2 = decoded_scales[row2_scales[scale_column]];
        }
        if (has_row3) {
          block_scale3 = decoded_scales[row3_scales[scale_column]];
        }
      }
      const int scale_source = static_cast<int>(lane & ~1U);
      block_scale0 =
          __shfl_sync(0xffff'ffffU, block_scale0, scale_source);
      block_scale1 =
          __shfl_sync(0xffff'ffffU, block_scale1, scale_source);
      block_scale2 =
          __shfl_sync(0xffff'ffffU, block_scale2, scale_source);
      block_scale3 =
          __shfl_sync(0xffff'ffffU, block_scale3, scale_source);

      const std::uint32_t packed0 =
          *reinterpret_cast<const std::uint32_t*>(row0_weights +
                                                  packed_column);
      const std::uint32_t packed1 =
          has_row1 ? *reinterpret_cast<const std::uint32_t*>(
                         row1_weights + packed_column)
                   : 0U;
      const std::uint32_t packed2 =
          has_row2 ? *reinterpret_cast<const std::uint32_t*>(
                         row2_weights + packed_column)
                   : 0U;
      const std::uint32_t packed3 =
          has_row3 ? *reinterpret_cast<const std::uint32_t*>(
                         row3_weights + packed_column)
                   : 0U;
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
          const unsigned int shift = packed_value * 4U;
          const std::uint16_t encoded_activation =
              static_cast<std::uint16_t>(
                  (packed_activation >> (value * 16U)) & 0xffffU);
          const float decoded_activation =
              decode_bf16(encoded_activation);
          accumulators0[value] =
              fmaf(decoded_weights[(packed0 >> shift) & 0x0fU] *
                       block_scale0,
                   decoded_activation, accumulators0[value]);
          accumulators1[value] =
              fmaf(decoded_weights[(packed1 >> shift) & 0x0fU] *
                       block_scale1,
                   decoded_activation, accumulators1[value]);
          accumulators2[value] =
              fmaf(decoded_weights[(packed2 >> shift) & 0x0fU] *
                       block_scale2,
                   decoded_activation, accumulators2[value]);
          accumulators3[value] =
              fmaf(decoded_weights[(packed3 >> shift) & 0x0fU] *
                       block_scale3,
                   decoded_activation, accumulators3[value]);
        }
      }
    }

    float sum = (accumulators0[0] + accumulators0[1]) +
                (accumulators0[2] + accumulators0[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(sum);
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U && has_row1) {
      output[row1] = encode_bf16_rne(sum);
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U && has_row2) {
      output[row2] = encode_bf16_rne(sum);
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U && has_row3) {
      output[row3] = encode_bf16_rne(sum);
    }
  }
}

// Production M=1 row-quad specialization for the three checkpoint shapes.
// Keeping the complete-quad shape facts and rolling byte strides at compile
// time removes per-row dynamic address construction without changing the
// arithmetic order.
template <std::size_t Rows, std::size_t Columns>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_exact_shape_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output) {
  static_assert((Rows % 4U) == 0U);
  static_assert((Columns % kNvFp4VectorColumnsPerWarp) == 0U);
  static_assert(
      (Rows - 1U) * (Columns / kNvFp4ValuesPerByte) +
              Columns / kNvFp4ValuesPerByte <=
          std::numeric_limits<std::uint32_t>::max());
  static_assert((Rows - 1U) * (Columns / kNvFp4GroupSize) +
                    Columns / kNvFp4GroupSize <=
                std::numeric_limits<std::uint32_t>::max());

  constexpr std::uint32_t kPackedColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4ValuesPerByte);
  constexpr std::uint32_t kScaleColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4GroupSize);
  constexpr std::uint32_t kRowStride =
      kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;
  static_assert(Rows >= kRowStride);
  static_assert((Rows + kRowStride) * kPackedColumns <=
                std::numeric_limits<std::uint32_t>::max());
  static_assert((Rows + kRowStride) * kScaleColumns <=
                std::numeric_limits<std::uint32_t>::max());
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

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < Rows;
       row0 += kRowStride) {
    const std::uint32_t row1 = row0 + 1U;
    const std::uint32_t row2 = row0 + 2U;
    const std::uint32_t row3 = row0 + 3U;
    const auto row0_weights = packed_weights + packed_row_offset;
    const auto row0_scales = block_scales + scale_row_offset;
    const auto row1_weights = row0_weights + kPackedColumns;
    const auto row1_scales = row0_scales + kScaleColumns;
    const auto row2_weights = row1_weights + kPackedColumns;
    const auto row2_scales = row1_scales + kScaleColumns;
    const auto row3_weights = row2_weights + kPackedColumns;
    const auto row3_scales = row2_scales + kScaleColumns;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

#pragma unroll 1
    for (std::uint32_t packed_column =
             lane * kNvFp4VectorPackedBytesPerLane;
         packed_column < kPackedColumns;
         packed_column += kNvFp4VectorColumnsPerWarp /
                          kNvFp4ValuesPerByte) {
      float block_scale0 = 0.0F;
      float block_scale1 = 0.0F;
      float block_scale2 = 0.0F;
      float block_scale3 = 0.0F;
      if ((lane & 1U) == 0U) {
        const std::uint32_t scale_column =
            packed_column / kNvFp4PackedValuesPerScale;
        block_scale0 = decoded_scales[row0_scales[scale_column]];
        block_scale1 = decoded_scales[row1_scales[scale_column]];
        block_scale2 = decoded_scales[row2_scales[scale_column]];
        block_scale3 = decoded_scales[row3_scales[scale_column]];
      }
      const int scale_source = static_cast<int>(lane & ~1U);
      block_scale0 =
          __shfl_sync(0xffff'ffffU, block_scale0, scale_source);
      block_scale1 =
          __shfl_sync(0xffff'ffffU, block_scale1, scale_source);
      block_scale2 =
          __shfl_sync(0xffff'ffffU, block_scale2, scale_source);
      block_scale3 =
          __shfl_sync(0xffff'ffffU, block_scale3, scale_source);

      const std::uint32_t packed0 =
          *reinterpret_cast<const std::uint32_t*>(row0_weights +
                                                  packed_column);
      const std::uint32_t packed1 =
          *reinterpret_cast<const std::uint32_t*>(row1_weights +
                                                  packed_column);
      const std::uint32_t packed2 =
          *reinterpret_cast<const std::uint32_t*>(row2_weights +
                                                  packed_column);
      const std::uint32_t packed3 =
          *reinterpret_cast<const std::uint32_t*>(row3_weights +
                                                  packed_column);
      const std::uint32_t first_column =
          packed_column * kNvFp4ValuesPerByte;

#pragma unroll
      for (unsigned int half = 0U; half < 2U; ++half) {
        const std::uint64_t packed_activation =
            *reinterpret_cast<const std::uint64_t*>(
                activation + first_column + half * 4U);
#pragma unroll
        for (unsigned int value = 0U; value < 4U; ++value) {
          const unsigned int packed_value = half * 4U + value;
          const unsigned int shift = packed_value * 4U;
          const std::uint16_t encoded_activation =
              static_cast<std::uint16_t>(
                  (packed_activation >> (value * 16U)) & 0xffffU);
          const float decoded_activation =
              decode_bf16(encoded_activation);
          accumulators0[value] =
              fmaf(decoded_weights[(packed0 >> shift) & 0x0fU] *
                       block_scale0,
                   decoded_activation, accumulators0[value]);
          accumulators1[value] =
              fmaf(decoded_weights[(packed1 >> shift) & 0x0fU] *
                       block_scale1,
                   decoded_activation, accumulators1[value]);
          accumulators2[value] =
              fmaf(decoded_weights[(packed2 >> shift) & 0x0fU] *
                       block_scale2,
                   decoded_activation, accumulators2[value]);
          accumulators3[value] =
              fmaf(decoded_weights[(packed3 >> shift) & 0x0fU] *
                       block_scale3,
                   decoded_activation, accumulators3[value]);
        }
      }
    }

    float sum = (accumulators0[0] + accumulators0[1]) +
                (accumulators0[2] + accumulators0[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(sum);
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row1] = encode_bf16_rne(sum);
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row2] = encode_bf16_rne(sum);
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row3] = encode_bf16_rne(sum);
    }
    packed_row_offset += kPackedRowStride;
    scale_row_offset += kScaleRowStride;
  }
}

// Preserved test-only down baseline. It advances two adjacent packed-x8 slices
// while retaining the original indexed raw-scale broadcasts and accumulator
// order. Production uses the adjacent-lane XOR specialization below.
template <std::size_t Rows, std::size_t Columns>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_down_dual_iteration_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output) {
  static_assert((Rows % 4U) == 0U);
  static_assert((Columns % (2U * kNvFp4VectorColumnsPerWarp)) == 0U);

  constexpr std::uint32_t kPackedColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4ValuesPerByte);
  constexpr std::uint32_t kScaleColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4GroupSize);
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4GroupSize;
  constexpr std::uint32_t kRowStride =
      kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;
  static_assert(Rows >= kRowStride);
  static_assert((Rows + kRowStride) * kPackedColumns <=
                std::numeric_limits<std::uint32_t>::max());
  static_assert((Rows + kRowStride) * kScaleColumns <=
                std::numeric_limits<std::uint32_t>::max());

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

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < Rows;
       row0 += kRowStride) {
    const std::uint32_t row1 = row0 + 1U;
    const std::uint32_t row2 = row0 + 2U;
    const std::uint32_t row3 = row0 + 3U;
    const auto row0_weights = packed_weights + packed_row_offset;
    const auto row0_scales = block_scales + scale_row_offset;
    const auto row1_weights = row0_weights + kPackedColumns;
    const auto row1_scales = row0_scales + kScaleColumns;
    const auto row2_weights = row1_weights + kPackedColumns;
    const auto row2_scales = row1_scales + kScaleColumns;
    const auto row3_weights = row2_weights + kPackedColumns;
    const auto row3_scales = row2_scales + kScaleColumns;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

#pragma unroll 1
    for (std::uint32_t packed_column =
             lane * kNvFp4VectorPackedBytesPerLane;
         packed_column < kPackedColumns;
         packed_column += 2U * kPackedIterationStride) {
      // Even lanes fetch the four row codes for the first packed-x8 slice;
      // odd lanes fetch the corresponding codes for the adjacent slice. One
      // packed u32 per lane is then broadcast twice to its lane pair.
      const std::uint32_t scale_column =
          packed_column / kNvFp4PackedValuesPerScale +
          (lane & 1U) * kScaleIterationStride;
      const std::uint32_t local_raw_scale_codes =
          static_cast<std::uint32_t>(row0_scales[scale_column]) |
          (static_cast<std::uint32_t>(row1_scales[scale_column]) << 8U) |
          (static_cast<std::uint32_t>(row2_scales[scale_column]) << 16U) |
          (static_cast<std::uint32_t>(row3_scales[scale_column]) << 24U);

#pragma unroll
      for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
        const int scale_source =
            static_cast<int>((lane & ~1U) + iteration);
        const std::uint32_t raw_scale_codes = __shfl_sync(
            0xffff'ffffU, local_raw_scale_codes, scale_source);
        const float block_scale0 =
            decoded_scales[raw_scale_codes & 0xffU];
        const float block_scale1 =
            decoded_scales[(raw_scale_codes >> 8U) & 0xffU];
        const float block_scale2 =
            decoded_scales[(raw_scale_codes >> 16U) & 0xffU];
        const float block_scale3 =
            decoded_scales[(raw_scale_codes >> 24U) & 0xffU];
        const std::uint32_t iteration_packed_column =
            packed_column + iteration * kPackedIterationStride;
        const std::uint32_t packed0 =
            *reinterpret_cast<const std::uint32_t*>(
                row0_weights + iteration_packed_column);
        const std::uint32_t packed1 =
            *reinterpret_cast<const std::uint32_t*>(
                row1_weights + iteration_packed_column);
        const std::uint32_t packed2 =
            *reinterpret_cast<const std::uint32_t*>(
                row2_weights + iteration_packed_column);
        const std::uint32_t packed3 =
            *reinterpret_cast<const std::uint32_t*>(
                row3_weights + iteration_packed_column);
        const std::uint32_t first_column =
            iteration_packed_column * kNvFp4ValuesPerByte;

#pragma unroll
        for (unsigned int half = 0U; half < 2U; ++half) {
          const std::uint64_t packed_activation =
              *reinterpret_cast<const std::uint64_t*>(
                  activation + first_column + half * 4U);
#pragma unroll
          for (unsigned int value = 0U; value < 4U; ++value) {
            const unsigned int packed_value = half * 4U + value;
            const unsigned int shift = packed_value * 4U;
            const std::uint16_t encoded_activation =
                static_cast<std::uint16_t>(
                    (packed_activation >> (value * 16U)) & 0xffffU);
            const float decoded_activation =
                decode_bf16(encoded_activation);
            accumulators0[value] =
                fmaf(decoded_weights[(packed0 >> shift) & 0x0fU] *
                         block_scale0,
                     decoded_activation, accumulators0[value]);
            accumulators1[value] =
                fmaf(decoded_weights[(packed1 >> shift) & 0x0fU] *
                         block_scale1,
                     decoded_activation, accumulators1[value]);
            accumulators2[value] =
                fmaf(decoded_weights[(packed2 >> shift) & 0x0fU] *
                         block_scale2,
                     decoded_activation, accumulators2[value]);
            accumulators3[value] =
                fmaf(decoded_weights[(packed3 >> shift) & 0x0fU] *
                         block_scale3,
                     decoded_activation, accumulators3[value]);
          }
        }
      }
    }

    float sum = (accumulators0[0] + accumulators0[1]) +
                (accumulators0[2] + accumulators0[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(sum);
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row1] = encode_bf16_rne(sum);
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row2] = encode_bf16_rne(sum);
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row3] = encode_bf16_rne(sum);
    }
    packed_row_offset += kPackedRowStride;
    scale_row_offset += kScaleRowStride;
  }
}

// Preserved adjacent-lane specialization for same-binary M=1 baselines. It
// preserves the exact row-quad accumulator order while replacing indexed
// raw-scale broadcasts with one XOR exchange per two packed-x8 phases. Even
// lanes own phase 0, odd lanes own phase 1, and every lane sorts local/partner
// payloads before consuming phase 0 then phase 1.
template <std::size_t Rows, std::size_t Columns>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_xor_dual_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output) {
  static_assert((Rows % 4U) == 0U);
  static_assert((Columns % (2U * kNvFp4VectorColumnsPerWarp)) == 0U);

  constexpr std::uint32_t kPackedColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4ValuesPerByte);
  constexpr std::uint32_t kScaleColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4GroupSize);
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4GroupSize;
  constexpr std::uint32_t kRowStride =
      kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;
  static_assert(Rows >= kRowStride);
  static_assert((Rows + kRowStride) * kPackedColumns <=
                std::numeric_limits<std::uint32_t>::max());
  static_assert((Rows + kRowStride) * kScaleColumns <=
                std::numeric_limits<std::uint32_t>::max());

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

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < Rows;
       row0 += kRowStride) {
    const std::uint32_t row1 = row0 + 1U;
    const std::uint32_t row2 = row0 + 2U;
    const std::uint32_t row3 = row0 + 3U;
    const auto row0_weights = packed_weights + packed_row_offset;
    const auto row0_scales = block_scales + scale_row_offset;
    const auto row1_weights = row0_weights + kPackedColumns;
    const auto row1_scales = row0_scales + kScaleColumns;
    const auto row2_weights = row1_weights + kPackedColumns;
    const auto row2_scales = row1_scales + kScaleColumns;
    const auto row3_weights = row2_weights + kPackedColumns;
    const auto row3_scales = row2_scales + kScaleColumns;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

#pragma unroll 1
    for (std::uint32_t packed_column =
             lane * kNvFp4VectorPackedBytesPerLane;
         packed_column < kPackedColumns;
         packed_column += 2U * kPackedIterationStride) {
      const std::uint32_t scale_column =
          packed_column / kNvFp4PackedValuesPerScale +
          (lane & 1U) * kScaleIterationStride;
      std::uint32_t local_raw_scale_codes =
          static_cast<std::uint32_t>(row0_scales[scale_column]) |
          (static_cast<std::uint32_t>(row1_scales[scale_column]) << 8U) |
          (static_cast<std::uint32_t>(row2_scales[scale_column]) << 16U) |
          (static_cast<std::uint32_t>(row3_scales[scale_column]) << 24U);
      const std::uint32_t partner_raw_scale_codes = __shfl_xor_sync(
          0xffff'ffffU, local_raw_scale_codes, 1);
      const std::uint32_t odd_lane_mask = 0U - (lane & 1U);
      const std::uint32_t phase0_raw_scale_codes =
          (local_raw_scale_codes & ~odd_lane_mask) |
          (partner_raw_scale_codes & odd_lane_mask);
      local_raw_scale_codes ^=
          partner_raw_scale_codes ^ phase0_raw_scale_codes;

#pragma unroll
      for (unsigned int phase = 0U; phase < 2U; ++phase) {
        const std::uint32_t raw_scale_codes =
            phase == 0U ? phase0_raw_scale_codes : local_raw_scale_codes;
        const float block_scale0 =
            decoded_scales[raw_scale_codes & 0xffU];
        const float block_scale1 =
            decoded_scales[(raw_scale_codes >> 8U) & 0xffU];
        const float block_scale2 =
            decoded_scales[(raw_scale_codes >> 16U) & 0xffU];
        const float block_scale3 =
            decoded_scales[(raw_scale_codes >> 24U) & 0xffU];
        const std::uint32_t phase_packed_column =
            packed_column + phase * kPackedIterationStride;
        const std::uint32_t packed0 =
            *reinterpret_cast<const std::uint32_t*>(
                row0_weights + phase_packed_column);
        const std::uint32_t packed1 =
            *reinterpret_cast<const std::uint32_t*>(
                row1_weights + phase_packed_column);
        const std::uint32_t packed2 =
            *reinterpret_cast<const std::uint32_t*>(
                row2_weights + phase_packed_column);
        const std::uint32_t packed3 =
            *reinterpret_cast<const std::uint32_t*>(
                row3_weights + phase_packed_column);
        const std::uint32_t first_column =
            phase_packed_column * kNvFp4ValuesPerByte;

#pragma unroll
        for (unsigned int half = 0U; half < 2U; ++half) {
          const std::uint64_t packed_activation =
              *reinterpret_cast<const std::uint64_t*>(
                  activation + first_column + half * 4U);
#pragma unroll
          for (unsigned int value = 0U; value < 4U; ++value) {
            const unsigned int packed_value = half * 4U + value;
            const unsigned int shift = packed_value * 4U;
            const std::uint16_t encoded_activation =
                static_cast<std::uint16_t>(
                    (packed_activation >> (value * 16U)) & 0xffffU);
            const float decoded_activation =
                decode_bf16(encoded_activation);
            accumulators0[value] =
                fmaf(decoded_weights[(packed0 >> shift) & 0x0fU] *
                         block_scale0,
                     decoded_activation, accumulators0[value]);
            accumulators1[value] =
                fmaf(decoded_weights[(packed1 >> shift) & 0x0fU] *
                         block_scale1,
                     decoded_activation, accumulators1[value]);
            accumulators2[value] =
                fmaf(decoded_weights[(packed2 >> shift) & 0x0fU] *
                         block_scale2,
                     decoded_activation, accumulators2[value]);
            accumulators3[value] =
                fmaf(decoded_weights[(packed3 >> shift) & 0x0fU] *
                         block_scale3,
                     decoded_activation, accumulators3[value]);
          }
        }
      }
    }

    float sum = (accumulators0[0] + accumulators0[1]) +
                (accumulators0[2] + accumulators0[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(sum);
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row1] = encode_bf16_rne(sum);
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row2] = encode_bf16_rne(sum);
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row3] = encode_bf16_rne(sum);
    }
    packed_row_offset += kPackedRowStride;
    scale_row_offset += kScaleRowStride;
  }
}

// Activation-staged production specialization shared by exact down, gate/up,
// and lm-head paths. The activation is staged once per CTA and then reused by
// every grid-stride row quad. The direct XOR-dual kernel above remains
// available as the same-binary baseline.
template <std::size_t Rows, std::size_t Columns>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output) {
  static_assert((Rows % 4U) == 0U);
  static_assert((Columns % (2U * kNvFp4VectorColumnsPerWarp)) == 0U);
  static_assert((Columns % 4U) == 0U);

  constexpr std::uint32_t kPackedColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4ValuesPerByte);
  constexpr std::uint32_t kScaleColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4GroupSize);
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4GroupSize;
  constexpr std::uint32_t kActivationVectorCount =
      static_cast<std::uint32_t>(Columns / 8U);
  constexpr std::uint32_t kActivationWordCount =
      static_cast<std::uint32_t>(Columns / 4U);
  constexpr std::uint32_t kRowStride =
      kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;
  static_assert(Rows >= kRowStride);
  static_assert((Rows + kRowStride) * kPackedColumns <=
                std::numeric_limits<std::uint32_t>::max());
  static_assert((Rows + kRowStride) * kScaleColumns <=
                std::numeric_limits<std::uint32_t>::max());

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const auto activation_words =
      reinterpret_cast<const std::uint64_t*>(activation);
  auto staged_activation_words =
      reinterpret_cast<std::uint64_t*>(staged_activation);
  for (std::uint32_t word = threadIdx.x; word < kActivationWordCount;
       word += kThreads) {
    staged_activation_words[word] = activation_words[word];
  }
  decoded_scales[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < Rows;
       row0 += kRowStride) {
    const std::uint32_t row1 = row0 + 1U;
    const std::uint32_t row2 = row0 + 2U;
    const std::uint32_t row3 = row0 + 3U;
    const auto row0_weights = packed_weights + packed_row_offset;
    const auto row0_scales = block_scales + scale_row_offset;
    const auto row1_weights = row0_weights + kPackedColumns;
    const auto row1_scales = row0_scales + kScaleColumns;
    const auto row2_weights = row1_weights + kPackedColumns;
    const auto row2_scales = row1_scales + kScaleColumns;
    const auto row3_weights = row2_weights + kPackedColumns;
    const auto row3_scales = row2_scales + kScaleColumns;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

#pragma unroll 1
    for (std::uint32_t packed_column =
             lane * kNvFp4VectorPackedBytesPerLane;
         packed_column < kPackedColumns;
         packed_column += 2U * kPackedIterationStride) {
      const std::uint32_t scale_column =
          packed_column / kNvFp4PackedValuesPerScale +
          (lane & 1U) * kScaleIterationStride;
      std::uint32_t local_raw_scale_codes =
          static_cast<std::uint32_t>(row0_scales[scale_column]) |
          (static_cast<std::uint32_t>(row1_scales[scale_column]) << 8U) |
          (static_cast<std::uint32_t>(row2_scales[scale_column]) << 16U) |
          (static_cast<std::uint32_t>(row3_scales[scale_column]) << 24U);
      const std::uint32_t partner_raw_scale_codes = __shfl_xor_sync(
          0xffff'ffffU, local_raw_scale_codes, 1);
      const std::uint32_t odd_lane_mask = 0U - (lane & 1U);
      const std::uint32_t phase0_raw_scale_codes =
          (local_raw_scale_codes & ~odd_lane_mask) |
          (partner_raw_scale_codes & odd_lane_mask);
      local_raw_scale_codes ^=
          partner_raw_scale_codes ^ phase0_raw_scale_codes;

#pragma unroll
      for (unsigned int phase = 0U; phase < 2U; ++phase) {
        const std::uint32_t raw_scale_codes =
            phase == 0U ? phase0_raw_scale_codes : local_raw_scale_codes;
        const float block_scale0 =
            decoded_scales[raw_scale_codes & 0xffU];
        const float block_scale1 =
            decoded_scales[(raw_scale_codes >> 8U) & 0xffU];
        const float block_scale2 =
            decoded_scales[(raw_scale_codes >> 16U) & 0xffU];
        const float block_scale3 =
            decoded_scales[(raw_scale_codes >> 24U) & 0xffU];
        const std::uint32_t phase_packed_column =
            packed_column + phase * kPackedIterationStride;
        const std::uint32_t packed0 =
            *reinterpret_cast<const std::uint32_t*>(
                row0_weights + phase_packed_column);
        const std::uint32_t packed1 =
            *reinterpret_cast<const std::uint32_t*>(
                row1_weights + phase_packed_column);
        const std::uint32_t packed2 =
            *reinterpret_cast<const std::uint32_t*>(
                row2_weights + phase_packed_column);
        const std::uint32_t packed3 =
            *reinterpret_cast<const std::uint32_t*>(
                row3_weights + phase_packed_column);
        const std::uint32_t first_column =
            phase_packed_column * kNvFp4ValuesPerByte;
        const ulonglong2 packed_activations =
            staged_activation[first_column / 8U];

#pragma unroll
        for (unsigned int half = 0U; half < 2U; ++half) {
          const std::uint64_t packed_activation =
              half == 0U ? packed_activations.x : packed_activations.y;
#pragma unroll
          for (unsigned int value = 0U; value < 4U; ++value) {
            const unsigned int packed_value = half * 4U + value;
            const unsigned int shift = packed_value * 4U;
            const std::uint16_t encoded_activation =
                static_cast<std::uint16_t>(
                    (packed_activation >> (value * 16U)) & 0xffffU);
            const float decoded_activation =
                decode_bf16(encoded_activation);
            accumulators0[value] =
                fmaf(decoded_weights[(packed0 >> shift) & 0x0fU] *
                         block_scale0,
                     decoded_activation, accumulators0[value]);
            accumulators1[value] =
                fmaf(decoded_weights[(packed1 >> shift) & 0x0fU] *
                         block_scale1,
                     decoded_activation, accumulators1[value]);
            accumulators2[value] =
                fmaf(decoded_weights[(packed2 >> shift) & 0x0fU] *
                         block_scale2,
                     decoded_activation, accumulators2[value]);
            accumulators3[value] =
                fmaf(decoded_weights[(packed3 >> shift) & 0x0fU] *
                         block_scale3,
                     decoded_activation, accumulators3[value]);
          }
        }
      }
    }

    float sum = (accumulators0[0] + accumulators0[1]) +
                (accumulators0[2] + accumulators0[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(sum);
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row1] = encode_bf16_rne(sum);
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row2] = encode_bf16_rne(sum);
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row3] = encode_bf16_rne(sum);
    }
    packed_row_offset += kPackedRowStride;
    scale_row_offset += kScaleRowStride;
  }
}

// One rolled phase of the gate/up pair kernel. Keeping the complete
// accumulator lifetime inside this helper lets the second phase reuse the
// first phase's registers while both phases share the CTA-staged activation
// and decode codebooks.
template <std::size_t Rows, std::size_t Columns>
__device__ __forceinline__ void
nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_phase(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    std::uint16_t* const output, const ulonglong2* const staged_activation,
    const float* const decoded_weights, const float* const decoded_scales,
    const unsigned int lane, const unsigned int warp) {
  constexpr std::uint32_t kPackedColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4ValuesPerByte);
  constexpr std::uint32_t kScaleColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4GroupSize);
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4GroupSize;
  constexpr std::uint32_t kRowStride =
      kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < Rows;
       row0 += kRowStride) {
    const std::uint32_t row1 = row0 + 1U;
    const std::uint32_t row2 = row0 + 2U;
    const std::uint32_t row3 = row0 + 3U;
    const auto row0_weights = packed_weights + packed_row_offset;
    const auto row0_scales = block_scales + scale_row_offset;
    const auto row1_weights = row0_weights + kPackedColumns;
    const auto row1_scales = row0_scales + kScaleColumns;
    const auto row2_weights = row1_weights + kPackedColumns;
    const auto row2_scales = row1_scales + kScaleColumns;
    const auto row3_weights = row2_weights + kPackedColumns;
    const auto row3_scales = row2_scales + kScaleColumns;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

#pragma unroll 1
    for (std::uint32_t packed_column =
             lane * kNvFp4VectorPackedBytesPerLane;
         packed_column < kPackedColumns;
         packed_column += 2U * kPackedIterationStride) {
      const std::uint32_t scale_column =
          packed_column / kNvFp4PackedValuesPerScale +
          (lane & 1U) * kScaleIterationStride;
      std::uint32_t local_raw_scale_codes =
          static_cast<std::uint32_t>(row0_scales[scale_column]) |
          (static_cast<std::uint32_t>(row1_scales[scale_column]) << 8U) |
          (static_cast<std::uint32_t>(row2_scales[scale_column]) << 16U) |
          (static_cast<std::uint32_t>(row3_scales[scale_column]) << 24U);
      const std::uint32_t partner_raw_scale_codes = __shfl_xor_sync(
          0xffff'ffffU, local_raw_scale_codes, 1);
      const std::uint32_t odd_lane_mask = 0U - (lane & 1U);
      const std::uint32_t phase0_raw_scale_codes =
          (local_raw_scale_codes & ~odd_lane_mask) |
          (partner_raw_scale_codes & odd_lane_mask);
      local_raw_scale_codes ^=
          partner_raw_scale_codes ^ phase0_raw_scale_codes;

#pragma unroll
      for (unsigned int phase = 0U; phase < 2U; ++phase) {
        const std::uint32_t raw_scale_codes =
            phase == 0U ? phase0_raw_scale_codes : local_raw_scale_codes;
        const float block_scale0 =
            decoded_scales[raw_scale_codes & 0xffU];
        const float block_scale1 =
            decoded_scales[(raw_scale_codes >> 8U) & 0xffU];
        const float block_scale2 =
            decoded_scales[(raw_scale_codes >> 16U) & 0xffU];
        const float block_scale3 =
            decoded_scales[(raw_scale_codes >> 24U) & 0xffU];
        const std::uint32_t phase_packed_column =
            packed_column + phase * kPackedIterationStride;
        const std::uint32_t packed0 =
            *reinterpret_cast<const std::uint32_t*>(
                row0_weights + phase_packed_column);
        const std::uint32_t packed1 =
            *reinterpret_cast<const std::uint32_t*>(
                row1_weights + phase_packed_column);
        const std::uint32_t packed2 =
            *reinterpret_cast<const std::uint32_t*>(
                row2_weights + phase_packed_column);
        const std::uint32_t packed3 =
            *reinterpret_cast<const std::uint32_t*>(
                row3_weights + phase_packed_column);
        const std::uint32_t first_column =
            phase_packed_column * kNvFp4ValuesPerByte;
        const ulonglong2 packed_activations =
            staged_activation[first_column / 8U];

#pragma unroll
        for (unsigned int half = 0U; half < 2U; ++half) {
          const std::uint64_t packed_activation =
              half == 0U ? packed_activations.x : packed_activations.y;
#pragma unroll
          for (unsigned int value = 0U; value < 4U; ++value) {
            const unsigned int packed_value = half * 4U + value;
            const unsigned int shift = packed_value * 4U;
            const std::uint16_t encoded_activation =
                static_cast<std::uint16_t>(
                    (packed_activation >> (value * 16U)) & 0xffffU);
            const float decoded_activation =
                decode_bf16(encoded_activation);
            accumulators0[value] =
                fmaf(decoded_weights[(packed0 >> shift) & 0x0fU] *
                         block_scale0,
                     decoded_activation, accumulators0[value]);
            accumulators1[value] =
                fmaf(decoded_weights[(packed1 >> shift) & 0x0fU] *
                         block_scale1,
                     decoded_activation, accumulators1[value]);
            accumulators2[value] =
                fmaf(decoded_weights[(packed2 >> shift) & 0x0fU] *
                         block_scale2,
                     decoded_activation, accumulators2[value]);
            accumulators3[value] =
                fmaf(decoded_weights[(packed3 >> shift) & 0x0fU] *
                         block_scale3,
                     decoded_activation, accumulators3[value]);
          }
        }
      }
    }

    float sum = (accumulators0[0] + accumulators0[1]) +
                (accumulators0[2] + accumulators0[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(sum);
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row1] = encode_bf16_rne(sum);
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row2] = encode_bf16_rne(sum);
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[row3] = encode_bf16_rne(sum);
    }
    packed_row_offset += kPackedRowStride;
    scale_row_offset += kScaleRowStride;
  }
}

// Exact production down projection that reuses the activation-staged phase,
// preserves the raw BF16 output boundary, and folds the following residual
// add and centered RMSNorm into one cooperative launch. All 64 CTAs repeat
// the exact reduction after the grid barrier; only the first 20 CTAs publish
// disjoint 256-element normalized slices.
template <std::size_t Rows, std::size_t Columns>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_down_residual_norm_activation_staged_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation,
    std::uint16_t* const raw_down_output,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output) {
  static_assert(Rows == 5'120U && Columns == 17'408U);
  constexpr std::uint32_t kActivationVectorCount =
      static_cast<std::uint32_t>(Columns / 8U);
  constexpr std::uint32_t kActivationWordCount =
      static_cast<std::uint32_t>(Columns / 4U);
  constexpr std::uint32_t kRowStride =
      kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kWarpsPerBlock * 4U;
  constexpr std::uint32_t kNormalizedBlocks = Rows / kThreads;

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const auto activation_words =
      reinterpret_cast<const std::uint64_t*>(activation);
  auto staged_activation_words =
      reinterpret_cast<std::uint64_t*>(staged_activation);
  for (std::uint32_t word = threadIdx.x; word < kActivationWordCount;
       word += kThreads) {
    staged_activation_words[word] = activation_words[word];
  }
  decoded_scales[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

  nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_phase<Rows, Columns>(
      packed_weights, block_scales, weight_scale_2, raw_down_output,
      staged_activation, decoded_weights, decoded_scales, lane, warp);

  // Lane zero publishes each raw projection value. Redistribute only rows
  // owned by this CTA after all eight warps finish and reproduce the separate
  // residual kernel's BF16 rounding boundary.
  __syncthreads();
  for (std::uint32_t local_row = threadIdx.x;; local_row += kThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= Rows) {
      break;
    }
    residual_output[row] = encode_bf16_rne(
        decode_bf16(residual_left[row]) + decode_bf16(raw_down_output[row]));
  }

  cooperative_groups::this_grid().sync();
  float sum = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < Rows;
       index += kThreads) {
    const float value = decode_bf16(residual_output[index]);
    sum = fmaf(value, value, sum);
  }
  // The FP8 scale codebook is dead after the projection phase. Reuse it for
  // the 256-thread reduction to retain the 35,904-byte shared footprint.
  decoded_scales[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      decoded_scales[threadIdx.x] += decoded_scales[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(decoded_scales[0] / static_cast<float>(Rows) + epsilon);
  if (blockIdx.x < kNormalizedBlocks) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(blockIdx.x) * kThreads + threadIdx.x;
    const float gamma = decode_bf16(norm_weight[index]) + 1.0F;
    normalized_output[index] = encode_bf16_rne(
        decode_bf16(residual_output[index]) * inverse_rms * gamma);
  }
}

// Stage activation and decode codebooks once, then run gate and up as rolled
// phases so only one phase's accumulators are live. The false instance is the
// preserved test-only pair candidate; the true instance is the production
// kernel and adds a CTA-parallel SiLU multiply after both BF16 stores.
template <std::size_t Rows, std::size_t Columns, bool FuseSilu>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_kernel(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const activation,
    std::uint16_t* const gate_output, std::uint16_t* const up_output) {
  static_assert((Rows % 4U) == 0U);
  static_assert((Columns % (2U * kNvFp4VectorColumnsPerWarp)) == 0U);
  static_assert((Columns % 4U) == 0U);
  constexpr std::uint32_t kActivationVectorCount =
      static_cast<std::uint32_t>(Columns / 8U);
  constexpr std::uint32_t kActivationWordCount =
      static_cast<std::uint32_t>(Columns / 4U);
  constexpr std::uint32_t kRowStride =
      kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U;
  constexpr std::uint32_t kPackedColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4ValuesPerByte);
  constexpr std::uint32_t kScaleColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4GroupSize);
  static_assert(Rows >= kRowStride);
  static_assert((Rows + kRowStride) * kPackedColumns <=
                std::numeric_limits<std::uint32_t>::max());
  static_assert((Rows + kRowStride) * kScaleColumns <=
                std::numeric_limits<std::uint32_t>::max());

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const auto activation_words =
      reinterpret_cast<const std::uint64_t*>(activation);
  auto staged_activation_words =
      reinterpret_cast<std::uint64_t*>(staged_activation);
  for (std::uint32_t word = threadIdx.x; word < kActivationWordCount;
       word += kThreads) {
    staged_activation_words[word] = activation_words[word];
  }
  decoded_scales[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

#pragma unroll 1
  for (unsigned int pair_phase = 0U; pair_phase < 2U; ++pair_phase) {
    nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_phase<Rows, Columns>(
        pair_phase == 0U ? gate_packed_weights : up_packed_weights,
        pair_phase == 0U ? gate_block_scales : up_block_scales,
        pair_phase == 0U ? gate_weight_scale_2 : up_weight_scale_2,
        pair_phase == 0U ? gate_output : up_output, staged_activation,
        decoded_weights, decoded_scales, lane, warp);
  }

  if constexpr (FuseSilu) {
    // The projection helper writes only from lane zero, so synchronize the
    // complete CTA before redistributing its rows across all threads. Each CTA
    // owns 32 contiguous rows per row-stride round. local_row 0..255 maps the
    // first eight rounds; local_row 256..287 maps the exact-shape tail.
    __syncthreads();
    constexpr std::uint32_t kRowsPerCtaPerStride = kWarpsPerBlock * 4U;
    for (std::uint32_t local_row = threadIdx.x;; local_row += kThreads) {
      const std::uint32_t row =
          static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
          (local_row / kRowsPerCtaPerStride) * kRowStride +
          local_row % kRowsPerCtaPerStride;
      if (row >= Rows) {
        break;
      }
      const float gate = decode_bf16(gate_output[row]);
      const float up = decode_bf16(up_output[row]);
      gate_output[row] =
          encode_bf16_rne(gate / (1.0F + expf(-gate)) * up);
    }
  }
}

// Exact-order fusion of the post-attention residual/RMSNorm
// with the production gate/up/SiLU kernel. Every CTA repeats the same
// 256-thread RMS reduction used by residual_add_centered_rms_norm_5120_kernel
// and materializes the normalized BF16 activation directly in shared memory.
// CTA zero alone publishes the residual output. Repeating the small reduction
// avoids a grid-wide barrier and the intermediate normalized global buffer;
// the much larger gate/up projection then follows the established 64-CTA
// topology and arithmetic order.
template <std::size_t Columns, bool UseWarpTailReduction>
__device__ __forceinline__ void
stage_residual_centered_rms_norm_bf16(
    const std::uint16_t* const left, const std::uint16_t* const right,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const staged_activation,
    float* const norm_partial_or_decoded_scales) {
  float sum = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < Columns;
       index += kThreads) {
    const std::uint16_t residual = encode_bf16_rne(
        decode_bf16(left[index]) + decode_bf16(right[index]));
    staged_activation[index] = residual;
    if (blockIdx.x == 0U) {
      residual_output[index] = residual;
    }
    const float value = decode_bf16(residual);
    sum = fmaf(value, value, sum);
  }
  norm_partial_or_decoded_scales[threadIdx.x] = sum;
  __syncthreads();
  if constexpr (UseWarpTailReduction) {
    // Retain the baseline tree through the 256->128->64->32 shared-memory
    // levels. Once only warp zero remains, shuffle-down performs the exact
    // same 16/8/4/2/1 pairings without a block barrier after each warp-local
    // level; one final barrier publishes lane zero's result to every warp.
    for (unsigned int stride = kThreads / 2U; stride >= kWarpSize;
         stride >>= 1U) {
      if (threadIdx.x < stride) {
        norm_partial_or_decoded_scales[threadIdx.x] +=
            norm_partial_or_decoded_scales[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x < kWarpSize) {
      float warp_partial = norm_partial_or_decoded_scales[threadIdx.x];
#pragma unroll
      for (unsigned int stride = kWarpSize / 2U; stride != 0U;
           stride >>= 1U) {
        warp_partial +=
            __shfl_down_sync(0xffff'ffffU, warp_partial, stride);
      }
      if (threadIdx.x == 0U) {
        norm_partial_or_decoded_scales[0] = warp_partial;
      }
    }
    __syncthreads();
  } else {
    // Preserved test-only baseline used for same-binary bitwise and timing
    // comparisons with the warp-tail production reduction above.
    for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
      if (threadIdx.x < stride) {
        norm_partial_or_decoded_scales[threadIdx.x] +=
            norm_partial_or_decoded_scales[threadIdx.x + stride];
      }
      __syncthreads();
    }
  }
  const float inverse_rms =
      rsqrtf(norm_partial_or_decoded_scales[0] /
                 static_cast<float>(Columns) +
             epsilon);
  for (std::uint32_t index = threadIdx.x; index < Columns;
       index += kThreads) {
    const float gamma = decode_bf16(norm_weight[index]) + 1.0F;
    staged_activation[index] = encode_bf16_rne(
        decode_bf16(staged_activation[index]) * inverse_rms * gamma);
  }
  __syncthreads();
}

template <std::size_t Rows, std::size_t Columns,
          bool UseWarpTailReduction = true>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_activation_staged_kernel(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output, std::uint16_t* const up_output) {
  static_assert((Rows % 4U) == 0U);
  static_assert((Columns % (2U * kNvFp4VectorColumnsPerWarp)) == 0U);
  static_assert((Columns % 8U) == 0U);
  constexpr std::uint32_t kActivationVectorCount =
      static_cast<std::uint32_t>(Columns / 8U);
  constexpr std::uint32_t kRowStride =
      kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U;
  constexpr std::uint32_t kPackedColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4ValuesPerByte);
  constexpr std::uint32_t kScaleColumns =
      static_cast<std::uint32_t>(Columns / kNvFp4GroupSize);
  static_assert(Rows >= kRowStride);
  static_assert((Rows + kRowStride) * kPackedColumns <=
                std::numeric_limits<std::uint32_t>::max());
  static_assert((Rows + kRowStride) * kScaleColumns <=
                std::numeric_limits<std::uint32_t>::max());

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  // The RMS reduction and FP8 scale codebook are phase-disjoint and both use
  // exactly one float per thread, so sharing this storage preserves the
  // production kernel's 11,328-byte shared-memory footprint.
  __shared__ float norm_partial_or_decoded_scales[kFp8EncodedValueCount];
  auto staged_activation_bf16 =
      reinterpret_cast<std::uint16_t*>(staged_activation);
  stage_residual_centered_rms_norm_bf16<Columns, UseWarpTailReduction>(
      residual_left, residual_right, norm_weight, epsilon, residual_output,
      staged_activation_bf16, norm_partial_or_decoded_scales);

  norm_partial_or_decoded_scales[threadIdx.x] =
      decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
#pragma unroll 1
  for (unsigned int pair_phase = 0U; pair_phase < 2U; ++pair_phase) {
    nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_phase<Rows, Columns>(
        pair_phase == 0U ? gate_packed_weights : up_packed_weights,
        pair_phase == 0U ? gate_block_scales : up_block_scales,
        pair_phase == 0U ? gate_weight_scale_2 : up_weight_scale_2,
        pair_phase == 0U ? gate_output : up_output, staged_activation,
        decoded_weights, norm_partial_or_decoded_scales, lane, warp);
  }

  __syncthreads();
  constexpr std::uint32_t kRowsPerCtaPerStride = kWarpsPerBlock * 4U;
  for (std::uint32_t local_row = threadIdx.x;; local_row += kThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= Rows) {
      break;
    }
    const float gate = decode_bf16(gate_output[row]);
    const float up = decode_bf16(up_output[row]);
    gate_output[row] =
        encode_bf16_rne(gate / (1.0F + expf(-gate)) * up);
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

// Production M=2 row-pair path. Two adjacent output rows share each
// token-major activation load and decode while retaining independent packed
// weights, group scales, accumulators, and per-row reduction order. The
// complete E4M3FN scale codebook remains shared by the whole block.
__global__ __launch_bounds__(kThreads, 5) void
nvfp4_w4a16_small_m2_gemm_bf16_scale_codebook_row_pair_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  constexpr unsigned int kTokenCount = 2U;
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
            const float decoded_activation =
                decode_bf16(encoded_activation);
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

// Production M=2 row-quad path. Four adjacent output rows share each
// token-major activation load and decode while retaining independent
// per-row/per-token FMA streams and warp reductions. The complete-row
// specialization removes all tail predicates from checkpoint shapes, and the
// four-block launch bound keeps the explicit eight-accumulator tile spill-free
// on SM87.
template <bool CompleteRowQuads>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_small_m2_gemm_bf16_scale_codebook_row_quad_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
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
      4U * (static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp);
  const std::size_t row_stride =
      static_cast<std::size_t>(gridDim.x) * kWarpsPerBlock * 4U;

  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row1 = row0 + 1U;
    const std::size_t row2 = row0 + 2U;
    const std::size_t row3 = row0 + 3U;
    const bool has_row1 = CompleteRowQuads || row1 < rows;
    const bool has_row2 = CompleteRowQuads || row2 < rows;
    const bool has_row3 = CompleteRowQuads || row3 < rows;
    const std::uint8_t* const row0_weights =
        packed_weights + row0 * packed_columns;
    const std::uint8_t* const row0_scales =
        block_scales + row0 * scale_columns;
    const std::uint8_t* const row1_weights =
        has_row1 ? packed_weights + row1 * packed_columns : row0_weights;
    const std::uint8_t* const row1_scales =
        has_row1 ? block_scales + row1 * scale_columns : row0_scales;
    const std::uint8_t* const row2_weights =
        has_row2 ? packed_weights + row2 * packed_columns : row0_weights;
    const std::uint8_t* const row2_scales =
        has_row2 ? block_scales + row2 * scale_columns : row0_scales;
    const std::uint8_t* const row3_weights =
        has_row3 ? packed_weights + row3 * packed_columns : row0_weights;
    const std::uint8_t* const row3_scales =
        has_row3 ? block_scales + row3 * scale_columns : row0_scales;

    float accumulator00 = 0.0F;
    float accumulator01 = 0.0F;
    float accumulator10 = 0.0F;
    float accumulator11 = 0.0F;
    float accumulator20 = 0.0F;
    float accumulator21 = 0.0F;
    float accumulator30 = 0.0F;
    float accumulator31 = 0.0F;

    for (std::size_t packed_column =
             static_cast<std::size_t>(lane) *
             kNvFp4VectorPackedBytesPerLane;
         packed_column < packed_columns;
         packed_column += kNvFp4VectorColumnsPerWarp /
                          kNvFp4ValuesPerByte) {
      float block_scale0 = 0.0F;
      float block_scale1 = 0.0F;
      float block_scale2 = 0.0F;
      float block_scale3 = 0.0F;
      if ((lane & 1U) == 0U) {
        const std::size_t scale_column =
            packed_column / kNvFp4PackedValuesPerScale;
        block_scale0 = decoded_scales[row0_scales[scale_column]];
        if (has_row1) {
          block_scale1 = decoded_scales[row1_scales[scale_column]];
        }
        if (has_row2) {
          block_scale2 = decoded_scales[row2_scales[scale_column]];
        }
        if (has_row3) {
          block_scale3 = decoded_scales[row3_scales[scale_column]];
        }
      }
      const int scale_source = static_cast<int>(lane & ~1U);
      block_scale0 =
          __shfl_sync(0xffff'ffffU, block_scale0, scale_source);
      block_scale1 =
          __shfl_sync(0xffff'ffffU, block_scale1, scale_source);
      block_scale2 =
          __shfl_sync(0xffff'ffffU, block_scale2, scale_source);
      block_scale3 =
          __shfl_sync(0xffff'ffffU, block_scale3, scale_source);

      const std::uint32_t packed0 =
          *reinterpret_cast<const std::uint32_t*>(row0_weights +
                                                  packed_column);
      const std::uint32_t packed1 =
          has_row1 ? *reinterpret_cast<const std::uint32_t*>(
                         row1_weights + packed_column)
                   : 0U;
      const std::uint32_t packed2 =
          has_row2 ? *reinterpret_cast<const std::uint32_t*>(
                         row2_weights + packed_column)
                   : 0U;
      const std::uint32_t packed3 =
          has_row3 ? *reinterpret_cast<const std::uint32_t*>(
                         row3_weights + packed_column)
                   : 0U;
      const std::size_t first_column =
          packed_column * kNvFp4ValuesPerByte;

#pragma unroll
      for (unsigned int half = 0U; half < 2U; ++half) {
        const std::uint64_t packed_activation0 =
            *reinterpret_cast<const std::uint64_t*>(
                activations + first_column + half * 4U);
        const std::uint64_t packed_activation1 =
            *reinterpret_cast<const std::uint64_t*>(
                activations + columns + first_column + half * 4U);
#pragma unroll
        for (unsigned int value = 0U; value < 4U; ++value) {
          const unsigned int packed_value = half * 4U + value;
          const unsigned int shift = packed_value * 4U;
          const float decoded_activation0 = decode_bf16(
              static_cast<std::uint16_t>(
                  (packed_activation0 >> (value * 16U)) & 0xffffU));
          const float decoded_activation1 = decode_bf16(
              static_cast<std::uint16_t>(
                  (packed_activation1 >> (value * 16U)) & 0xffffU));

          float scaled_weight =
              decoded_weights[(packed0 >> shift) & 0x0fU] * block_scale0;
          accumulator00 =
              fmaf(scaled_weight, decoded_activation0, accumulator00);
          accumulator01 =
              fmaf(scaled_weight, decoded_activation1, accumulator01);
          scaled_weight =
              decoded_weights[(packed1 >> shift) & 0x0fU] * block_scale1;
          accumulator10 =
              fmaf(scaled_weight, decoded_activation0, accumulator10);
          accumulator11 =
              fmaf(scaled_weight, decoded_activation1, accumulator11);
          scaled_weight =
              decoded_weights[(packed2 >> shift) & 0x0fU] * block_scale2;
          accumulator20 =
              fmaf(scaled_weight, decoded_activation0, accumulator20);
          accumulator21 =
              fmaf(scaled_weight, decoded_activation1, accumulator21);
          scaled_weight =
              decoded_weights[(packed3 >> shift) & 0x0fU] * block_scale3;
          accumulator30 =
              fmaf(scaled_weight, decoded_activation0, accumulator30);
          accumulator31 =
              fmaf(scaled_weight, decoded_activation1, accumulator31);
        }
      }
    }

    float sum = warp_sum(accumulator00) * weight_scale_2;
    if (lane == 0U) {
      output[row0] = encode_bf16_rne(sum);
    }
    sum = warp_sum(accumulator10) * weight_scale_2;
    if (lane == 0U && has_row1) {
      output[row1] = encode_bf16_rne(sum);
    }
    sum = warp_sum(accumulator20) * weight_scale_2;
    if (lane == 0U && has_row2) {
      output[row2] = encode_bf16_rne(sum);
    }
    sum = warp_sum(accumulator30) * weight_scale_2;
    if (lane == 0U && has_row3) {
      output[row3] = encode_bf16_rne(sum);
    }

    sum = warp_sum(accumulator01) * weight_scale_2;
    if (lane == 0U) {
      output[rows + row0] = encode_bf16_rne(sum);
    }
    sum = warp_sum(accumulator11) * weight_scale_2;
    if (lane == 0U && has_row1) {
      output[rows + row1] = encode_bf16_rne(sum);
    }
    sum = warp_sum(accumulator21) * weight_scale_2;
    if (lane == 0U && has_row2) {
      output[rows + row2] = encode_bf16_rne(sum);
    }
    sum = warp_sum(accumulator31) * weight_scale_2;
    if (lane == 0U && has_row3) {
      output[rows + row3] = encode_bf16_rne(sum);
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

[[nodiscard]] int validate_fp8_pair_launch(
    const std::uint8_t* const first_weights,
    const float first_weight_scale,
    const std::uint8_t* const second_weights,
    const float second_weight_scale,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output,
    const std::size_t maximum_blocks) noexcept {
  // The fused kernel is defined only for the checkpoint K/V shape. A CTA owns
  // two rows from each matrix, so a launch may not request more CTAs than row
  // pairs even through the test-only cap override.
  if (rows != kFp8KvPairRows || columns != kFp8KvPairColumns ||
      maximum_blocks == 0U || maximum_blocks > rows / 2U) {
    return invalid_value();
  }
  const int first_validation = validate_fp8_launch(
      first_weights, first_weight_scale, activation, rows, columns,
      first_output);
  if (first_validation != static_cast<int>(cudaSuccess)) {
    return first_validation;
  }
  const int second_validation = validate_fp8_launch(
      second_weights, second_weight_scale, activation, rows, columns,
      second_output);
  if (second_validation != static_cast<int>(cudaSuccess)) {
    return second_validation;
  }

  const std::size_t weight_bytes = rows * columns;
  const std::size_t output_bytes = rows * sizeof(std::uint16_t);
  if (ranges_overlap(first_output, output_bytes, second_weights,
                     weight_bytes) ||
      ranges_overlap(second_output, output_bytes, first_weights,
                     weight_bytes) ||
      ranges_overlap(first_output, output_bytes, second_output,
                     output_bytes)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int validate_fp8_qkv_z_launch(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    const std::size_t maximum_blocks) noexcept {
  if (qkv_rows != kFp8QkvRows || z_rows != kFp8ZRows ||
      columns != kFp8QkvZColumns || maximum_blocks == 0U ||
      maximum_blocks > kFp8QkvZMaximumTestBlocks) {
    return invalid_value();
  }
  const int qkv_validation = validate_fp8_launch(
      qkv_weights, qkv_weight_scale, activation, qkv_rows, columns,
      qkv_output);
  if (qkv_validation != static_cast<int>(cudaSuccess)) {
    return qkv_validation;
  }
  const int z_validation = validate_fp8_launch(
      z_weights, z_weight_scale, activation, z_rows, columns, z_output);
  if (z_validation != static_cast<int>(cudaSuccess)) {
    return z_validation;
  }

  const std::size_t qkv_weight_bytes = kFp8QkvRows * kFp8QkvZColumns;
  const std::size_t z_weight_bytes = kFp8ZRows * kFp8QkvZColumns;
  const std::size_t qkv_output_bytes =
      kFp8QkvRows * sizeof(std::uint16_t);
  const std::size_t z_output_bytes = kFp8ZRows * sizeof(std::uint16_t);
  if (ranges_overlap(qkv_weights, qkv_weight_bytes, z_weights,
                     z_weight_bytes) ||
      ranges_overlap(qkv_output, qkv_output_bytes, z_weights,
                     z_weight_bytes) ||
      ranges_overlap(z_output, z_output_bytes, qkv_weights,
                     qkv_weight_bytes) ||
      ranges_overlap(qkv_output, qkv_output_bytes, z_output,
                     z_output_bytes)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool fp8_qkv_z_launch_is_aligned(
    const std::uint8_t* const qkv_weights,
    const std::uint8_t* const z_weights,
    const std::uint16_t* const activation,
    const std::uint16_t* const qkv_output,
    const std::uint16_t* const z_output) noexcept {
  return (reinterpret_cast<std::uintptr_t>(qkv_weights) %
          alignof(std::uint32_t)) == 0U &&
         (reinterpret_cast<std::uintptr_t>(z_weights) %
          alignof(std::uint32_t)) == 0U &&
         (reinterpret_cast<std::uintptr_t>(activation) %
          alignof(std::uint64_t)) == 0U &&
         (reinterpret_cast<std::uintptr_t>(qkv_output) %
          alignof(std::uint16_t)) == 0U &&
         (reinterpret_cast<std::uintptr_t>(z_output) %
          alignof(std::uint16_t)) == 0U;
}

void launch_fp8_qkv_z_two_phase_unchecked(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    const unsigned int blocks, cudaStream_t const stream) noexcept {
  fp8_w8a16_gemv_bf16_qkv_z_two_phase_reduction_scratch_ping_pong_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          qkv_weights, qkv_weight_scale, z_weights, z_weight_scale,
          activation, columns, qkv_output, z_output);
}

void launch_fp8_qkv_z_tail_barrier_test_unchecked(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    const unsigned int blocks, cudaStream_t const stream) noexcept {
  fp8_w8a16_gemv_bf16_qkv_z_two_phase_tail_barrier_test_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          qkv_weights, qkv_weight_scale, z_weights, z_weight_scale,
          activation, columns, qkv_output, z_output);
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

void launch_fp8_m1_row_pair_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = 2U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < kFp8M1PersistentMaximumBlocks
          ? wanted
          : kFp8M1PersistentMaximumBlocks);
  fp8_w8a16_gemv_bf16_row_pair_kernel
      <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activation,
                                        rows, columns, output);
}

// Common capped launcher used by production plus the direct row-pair and
// swizzled row-quad test entries. RowsPerBlock selects only output grouping;
// the preserved unswizzled baseline intentionally bypasses this helper.
template <std::size_t RowsPerBlock>
void launch_fp8_m1_output_row_group_grid_cap_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  static_assert(RowsPerBlock == 2U || RowsPerBlock == 4U);
  const std::size_t wanted =
      rows / RowsPerBlock + (rows % RowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < maximum_blocks ? wanted : maximum_blocks);
  if constexpr (RowsPerBlock == 2U) {
    fp8_w8a16_gemv_bf16_row_pair_kernel
        <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activation,
                                          rows, columns, output);
  } else if ((rows % RowsPerBlock) == 0U) {
    fp8_w8a16_gemv_bf16_row_quad_kernel<true>
        <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activation,
                                          rows, columns, output);
  } else {
    fp8_w8a16_gemv_bf16_row_quad_kernel<false>
        <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activation,
                                          rows, columns, output);
  }
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

[[nodiscard]] int validate_nvfp4_gate_up_pair_launch(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const gate_output,
    std::uint16_t* const up_output) noexcept {
  int validation = validate_nvfp4_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2, activation,
      rows, columns, gate_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  validation = validate_nvfp4_launch(
      up_packed_weights, up_block_scales, up_weight_scale_2, activation, rows,
      columns, up_output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }

  const std::size_t packed_bytes =
      rows * (columns / kNvFp4ValuesPerByte);
  const std::size_t scale_bytes = rows * (columns / kNvFp4GroupSize);
  const std::size_t output_bytes = rows * sizeof(std::uint16_t);
  if (ranges_overlap(gate_output, output_bytes, up_output, output_bytes) ||
      ranges_overlap(gate_output, output_bytes, up_packed_weights,
                     packed_bytes) ||
      ranges_overlap(gate_output, output_bytes, up_block_scales,
                     scale_bytes) ||
      ranges_overlap(up_output, output_bytes, gate_packed_weights,
                     packed_bytes) ||
      ranges_overlap(up_output, output_bytes, gate_block_scales,
                     scale_bytes)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int validate_nvfp4_residual_norm_gate_up_silu_launch(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_output) noexcept {
  const int pair_validation = validate_nvfp4_gate_up_pair_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_right,
      rows, columns, gate_output, up_output);
  if (pair_validation != static_cast<int>(cudaSuccess)) {
    return pair_validation;
  }
  if (!std::isfinite(epsilon) || epsilon <= 0.0F ||
      residual_left == nullptr || norm_weight == nullptr ||
      residual_output == nullptr ||
      multiply_overflows(columns, sizeof(std::uint16_t)) ||
      byte_range_overflows(residual_left,
                           columns * sizeof(std::uint16_t)) ||
      byte_range_overflows(norm_weight,
                           columns * sizeof(std::uint16_t)) ||
      byte_range_overflows(residual_output,
                           columns * sizeof(std::uint16_t))) {
    return invalid_value();
  }

  const std::size_t input_bytes = columns * sizeof(std::uint16_t);
  const std::size_t projection_output_bytes = rows * sizeof(std::uint16_t);
  const std::size_t packed_bytes =
      rows * (columns / kNvFp4ValuesPerByte);
  const std::size_t scale_bytes = rows * (columns / kNvFp4GroupSize);
  if (ranges_overlap(residual_output, input_bytes, residual_left,
                     input_bytes) ||
      ranges_overlap(residual_output, input_bytes, residual_right,
                     input_bytes) ||
      ranges_overlap(residual_output, input_bytes, norm_weight,
                     input_bytes) ||
      ranges_overlap(residual_output, input_bytes, gate_packed_weights,
                     packed_bytes) ||
      ranges_overlap(residual_output, input_bytes, gate_block_scales,
                     scale_bytes) ||
      ranges_overlap(residual_output, input_bytes, up_packed_weights,
                     packed_bytes) ||
      ranges_overlap(residual_output, input_bytes, up_block_scales,
                     scale_bytes) ||
      ranges_overlap(residual_output, input_bytes, gate_output,
                     projection_output_bytes) ||
      ranges_overlap(residual_output, input_bytes, up_output,
                     projection_output_bytes) ||
      ranges_overlap(gate_output, projection_output_bytes, residual_left,
                     input_bytes) ||
      ranges_overlap(gate_output, projection_output_bytes, norm_weight,
                     input_bytes) ||
      ranges_overlap(up_output, projection_output_bytes, residual_left,
                     input_bytes) ||
      ranges_overlap(up_output, projection_output_bytes, norm_weight,
                     input_bytes)) {
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

void launch_nvfp4_scale_codebook_grid_cap_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  const unsigned int uncapped = block_count_for_rows(rows);
  const unsigned int blocks =
      uncapped < maximum_blocks ? uncapped : maximum_blocks;
  nvfp4_w4a16_gemv_bf16_scale_codebook_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activation, rows,
          columns, output);
}

void launch_nvfp4_scale_codebook_row_pair_grid_cap_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = kWarpsPerBlock * 2U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < maximum_blocks ? wanted : maximum_blocks);
  nvfp4_w4a16_gemv_bf16_scale_codebook_row_pair_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activation, rows,
          columns, output);
}

void launch_nvfp4_scale_codebook_row_quad_grid_cap_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = kWarpsPerBlock * 4U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < maximum_blocks ? wanted : maximum_blocks);
  if ((rows % 4U) == 0U) {
    nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_kernel<true>
        <<<blocks, kThreads, 0U, stream>>>(
            packed_weights, block_scales, weight_scale_2, activation, rows,
            columns, output);
  } else {
    nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_kernel<false>
        <<<blocks, kThreads, 0U, stream>>>(
            packed_weights, block_scales, weight_scale_2, activation, rows,
            columns, output);
  }
}

template <std::size_t Rows, std::size_t Columns>
void launch_nvfp4_scale_codebook_row_quad_exact_shape_instance_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_exact_shape_kernel<
      Rows, Columns><<<kNvFp4M1RowQuadMaximumBlocks, kThreads, 0U, stream>>>(
      packed_weights, block_scales, weight_scale_2, activation, output);
}

void launch_nvfp4_scale_codebook_row_quad_exact_shape_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  if (rows == 17'408U && columns == 5'120U) {
    launch_nvfp4_scale_codebook_row_quad_exact_shape_instance_unchecked<
        17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                         activation, output, stream);
  } else if (rows == 5'120U && columns == 17'408U) {
    launch_nvfp4_scale_codebook_row_quad_exact_shape_instance_unchecked<
        5'120U, 17'408U>(packed_weights, block_scales, weight_scale_2,
                         activation, output, stream);
  } else {
    launch_nvfp4_scale_codebook_row_quad_exact_shape_instance_unchecked<
        248'320U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                          activation, output, stream);
  }
}

template <std::size_t Rows, std::size_t Columns>
void launch_nvfp4_down_indexed_dual_baseline_instance_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_down_dual_iteration_kernel<
      Rows, Columns><<<kNvFp4M1RowQuadMaximumBlocks, kThreads, 0U, stream>>>(
      packed_weights, block_scales, weight_scale_2, activation, output);
}

void launch_nvfp4_down_indexed_dual_baseline_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  launch_nvfp4_down_indexed_dual_baseline_instance_unchecked<5'120U, 17'408U>(
      packed_weights, block_scales, weight_scale_2, activation, output,
      stream);
}

void launch_nvfp4_down_indexed_dual_baseline_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  if (rows == 5'120U && columns == 17'408U) {
    launch_nvfp4_down_indexed_dual_baseline_unchecked(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  } else {
    // The direct ABI admits only this bounded correctness fixture in addition
    // to the production down shape, so no gate/up or lm-head instance exists.
    launch_nvfp4_down_indexed_dual_baseline_instance_unchecked<2'048U, 512U>(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  }
}

template <std::size_t Rows, std::size_t Columns>
void launch_nvfp4_k5120_xor_dual_instance_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_xor_dual_kernel<
      Rows, Columns><<<kNvFp4M1RowQuadMaximumBlocks, kThreads, 0U, stream>>>(
      packed_weights, block_scales, weight_scale_2, activation, output);
}

void launch_nvfp4_down_xor_dual_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  launch_nvfp4_k5120_xor_dual_instance_unchecked<5'120U, 17'408U>(
      packed_weights, block_scales, weight_scale_2, activation, output, stream);
}

// Direct test ABI for the preserved down XOR baseline plus its bounded
// correctness fixture.
void launch_nvfp4_down_xor_dual_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  if (rows == 5'120U && columns == 17'408U) {
    launch_nvfp4_down_xor_dual_unchecked(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  } else {
    // The direct ABI admits only this bounded correctness fixture in addition
    // to the production down shape.
    launch_nvfp4_k5120_xor_dual_instance_unchecked<2'048U, 512U>(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  }
}

// Preserved direct-activation same-binary baseline for the production staged
// gate/up path.
void launch_nvfp4_gate_up_xor_dual_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  launch_nvfp4_k5120_xor_dual_instance_unchecked<17'408U, 5'120U>(
      packed_weights, block_scales, weight_scale_2, activation, output,
      stream);
}

void launch_nvfp4_gate_up_xor_dual_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  if (rows == 17'408U && columns == 5'120U) {
    launch_nvfp4_gate_up_xor_dual_unchecked(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  } else {
    // The direct ABI admits only this bounded correctness fixture in addition
    // to gate/up, so no down-projection or lm-head instance exists.
    launch_nvfp4_k5120_xor_dual_instance_unchecked<2'048U, 512U>(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  }
}

// Preserved direct-activation test baseline for same-binary comparisons with
// the production staged lm-head path.
void launch_nvfp4_lm_head_xor_dual_baseline_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  launch_nvfp4_k5120_xor_dual_instance_unchecked<248'320U, 5'120U>(
      packed_weights, block_scales, weight_scale_2, activation, output,
      stream);
}

void launch_nvfp4_lm_head_xor_dual_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  if (rows == 248'320U && columns == 5'120U) {
    launch_nvfp4_lm_head_xor_dual_baseline_unchecked(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  } else {
    // The direct ABI admits only this bounded correctness fixture in addition
    // to lm-head, so no gate/up or down-projection instance exists.
    launch_nvfp4_k5120_xor_dual_instance_unchecked<2'048U, 512U>(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  }
}

template <std::size_t Rows, std::size_t Columns>
void launch_nvfp4_k5120_activation_staged_instance_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel<
      Rows, Columns><<<kNvFp4M1RowQuadMaximumBlocks, kThreads, 0U, stream>>>(
      packed_weights, block_scales, weight_scale_2, activation, output);
}

void launch_nvfp4_lm_head_activation_staged_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  launch_nvfp4_k5120_activation_staged_instance_unchecked<248'320U, 5'120U>(
      packed_weights, block_scales, weight_scale_2, activation, output,
      stream);
}

void launch_nvfp4_gate_up_activation_staged_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  launch_nvfp4_k5120_activation_staged_instance_unchecked<17'408U, 5'120U>(
      packed_weights, block_scales, weight_scale_2, activation, output,
      stream);
}

// Production long-K down specialization. The direct-activation XOR kernel
// remains available above as the same-binary benchmark baseline.
void launch_nvfp4_down_activation_staged_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  launch_nvfp4_k5120_activation_staged_instance_unchecked<5'120U, 17'408U>(
      packed_weights, block_scales, weight_scale_2, activation, output,
      stream);
}

[[nodiscard]] cudaError_t
launch_nvfp4_down_residual_norm_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const raw_down_output,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output,
    cudaStream_t const stream) noexcept {
  const std::uint8_t* packed_argument = packed_weights;
  const std::uint8_t* scales_argument = block_scales;
  float scale_argument = weight_scale_2;
  const std::uint16_t* activation_argument = activation;
  std::uint16_t* raw_argument = raw_down_output;
  const std::uint16_t* residual_left_argument = residual_left;
  const std::uint16_t* norm_weight_argument = norm_weight;
  float epsilon_argument = epsilon;
  std::uint16_t* residual_argument = residual_output;
  std::uint16_t* normalized_argument = normalized_output;
  void* arguments[] = {
      &packed_argument,       &scales_argument,       &scale_argument,
      &activation_argument,   &raw_argument,          &residual_left_argument,
      &norm_weight_argument,  &epsilon_argument,      &residual_argument,
      &normalized_argument,
  };
  return cudaLaunchCooperativeKernel(
      nvfp4_w4a16_down_residual_norm_activation_staged_kernel<
          5'120U, 17'408U>,
      dim3{kNvFp4M1RowQuadMaximumBlocks}, dim3{kThreads}, arguments, 0U,
      stream);
}

void launch_nvfp4_down_activation_staged_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  if (rows == 5'120U && columns == 17'408U) {
    launch_nvfp4_down_activation_staged_unchecked(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  } else {
    launch_nvfp4_k5120_activation_staged_instance_unchecked<2'048U, 512U>(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  }
}

// Direct test ABI for the production gate/up staged specialization plus its
// bounded correctness fixture.
void launch_nvfp4_gate_up_activation_staged_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  if (rows == 17'408U && columns == 5'120U) {
    launch_nvfp4_gate_up_activation_staged_unchecked(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  } else {
    launch_nvfp4_k5120_activation_staged_instance_unchecked<2'048U, 512U>(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  }
}

template <std::size_t Rows, std::size_t Columns, bool FuseSilu>
void launch_nvfp4_gate_up_pair_activation_staged_instance_unchecked(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const activation,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    cudaStream_t const stream) noexcept {
  nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_kernel<
      Rows, Columns, FuseSilu>
      <<<kNvFp4M1RowQuadMaximumBlocks, kThreads, 0U, stream>>>(
          gate_packed_weights, gate_block_scales, gate_weight_scale_2,
          up_packed_weights, up_block_scales, up_weight_scale_2, activation,
          gate_output, up_output);
}

void launch_nvfp4_gate_up_pair_activation_staged_test_unchecked(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const gate_output,
    std::uint16_t* const up_output, cudaStream_t const stream) noexcept {
  if (rows == 17'408U && columns == 5'120U) {
    launch_nvfp4_gate_up_pair_activation_staged_instance_unchecked<
        17'408U, 5'120U, false>(
        gate_packed_weights, gate_block_scales, gate_weight_scale_2,
        up_packed_weights, up_block_scales, up_weight_scale_2, activation,
        gate_output, up_output, stream);
  } else {
    launch_nvfp4_gate_up_pair_activation_staged_instance_unchecked<
        2'048U, 512U, false>(
        gate_packed_weights, gate_block_scales, gate_weight_scale_2,
        up_packed_weights, up_block_scales, up_weight_scale_2, activation,
        gate_output, up_output, stream);
  }
}

void launch_nvfp4_gate_up_silu_activation_staged_test_unchecked(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const gate_output,
    std::uint16_t* const up_output, cudaStream_t const stream) noexcept {
  if (rows == 17'408U && columns == 5'120U) {
    launch_nvfp4_gate_up_pair_activation_staged_instance_unchecked<
        17'408U, 5'120U, true>(
        gate_packed_weights, gate_block_scales, gate_weight_scale_2,
        up_packed_weights, up_block_scales, up_weight_scale_2, activation,
        gate_output, up_output, stream);
  } else {
    launch_nvfp4_gate_up_pair_activation_staged_instance_unchecked<
        2'048U, 512U, true>(
        gate_packed_weights, gate_block_scales, gate_weight_scale_2,
        up_packed_weights, up_block_scales, up_weight_scale_2, activation,
        gate_output, up_output, stream);
  }
}

template <std::size_t Rows, std::size_t Columns,
          bool UseWarpTailReduction = true>
void launch_nvfp4_residual_norm_gate_up_silu_instance_unchecked(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    cudaStream_t const stream) noexcept {
  nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_activation_staged_kernel<
      Rows, Columns, UseWarpTailReduction>
      <<<kNvFp4M1RowQuadMaximumBlocks, kThreads, 0U, stream>>>(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, residual_output, gate_output,
      up_output);
}

void launch_nvfp4_residual_norm_gate_up_silu_shared_tree_test_unchecked(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    cudaStream_t const stream) noexcept {
  if (rows == 17'408U && columns == 5'120U) {
    launch_nvfp4_residual_norm_gate_up_silu_instance_unchecked<17'408U,
                                                                 5'120U,
                                                                 false>(
        gate_packed_weights, gate_block_scales, gate_weight_scale_2,
        up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
        residual_right, norm_weight, epsilon, residual_output, gate_output,
        up_output, stream);
  } else {
    launch_nvfp4_residual_norm_gate_up_silu_instance_unchecked<2'048U, 512U,
                                                                 false>(
        gate_packed_weights, gate_block_scales, gate_weight_scale_2,
        up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
        residual_right, norm_weight, epsilon, residual_output, gate_output,
        up_output, stream);
  }
}

void launch_nvfp4_residual_norm_gate_up_silu_test_unchecked(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    cudaStream_t const stream) noexcept {
  if (rows == 17'408U && columns == 5'120U) {
    launch_nvfp4_residual_norm_gate_up_silu_instance_unchecked<17'408U,
                                                                 5'120U>(
        gate_packed_weights, gate_block_scales, gate_weight_scale_2,
        up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
        residual_right, norm_weight, epsilon, residual_output, gate_output,
        up_output, stream);
  } else {
    launch_nvfp4_residual_norm_gate_up_silu_instance_unchecked<2'048U, 512U>(
        gate_packed_weights, gate_block_scales, gate_weight_scale_2,
        up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
        residual_right, norm_weight, epsilon, residual_output, gate_output,
        up_output, stream);
  }
}

void launch_nvfp4_lm_head_activation_staged_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  if (rows == 248'320U && columns == 5'120U) {
    launch_nvfp4_lm_head_activation_staged_unchecked(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  } else {
    launch_nvfp4_k5120_activation_staged_instance_unchecked<2'048U, 512U>(
        packed_weights, block_scales, weight_scale_2, activation, output,
        stream);
  }
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

[[nodiscard]] int validate_fp8_m32_launch(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) noexcept {
  constexpr std::size_t kTokenCount = 32U;
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

[[nodiscard]] int validate_nvfp4_m18_launch(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) noexcept {
  constexpr std::size_t kTokenCount = 18U;
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
      activations == nullptr || output == nullptr ||
      !pointer_is_aligned<alignof(std::uint16_t)>(activations) ||
      !pointer_is_aligned<alignof(std::uint16_t)>(output)) {
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

[[nodiscard]] int validate_nvfp4_m17_m31_launch(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t valid_token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) noexcept {
  if (valid_token_count < 17U || valid_token_count > 31U ||
      (columns % kNvFp4GroupSize) != 0U ||
      !std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F ||
      multiply_overflows(rows, columns)) {
    return invalid_value();
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (multiply_overflows(valid_token_count, columns) ||
      multiply_overflows(valid_token_count, rows)) {
    return invalid_value();
  }
  const std::size_t activation_elements = valid_token_count * columns;
  const std::size_t output_elements = valid_token_count * rows;
  if (multiply_overflows(activation_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(output_elements, sizeof(std::uint16_t)) ||
      packed_weights == nullptr || block_scales == nullptr ||
      activations == nullptr || output == nullptr ||
      !pointer_is_aligned<alignof(std::uint16_t)>(activations) ||
      !pointer_is_aligned<alignof(std::uint16_t)>(output)) {
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

[[nodiscard]] int validate_nvfp4_m32_launch(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) noexcept {
  constexpr std::size_t kTokenCount = 32U;
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

void launch_fp8_small_m2_row_pair_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = 2U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < kFp8M2PersistentMaximumBlocks
          ? wanted
          : kFp8M2PersistentMaximumBlocks);
  fp8_w8a16_small_m2_gemm_bf16_row_pair_kernel
      <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activations,
                                        rows, columns, output);
}

void launch_fp8_small_m2_row_pair_grid_cap_test_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = 2U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < maximum_blocks ? wanted : maximum_blocks);
  fp8_w8a16_small_m2_gemm_bf16_row_pair_kernel
      <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activations,
                                        rows, columns, output);
}

void launch_fp8_small_m2_row_quad_grid_cap_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = 4U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < maximum_blocks ? wanted : maximum_blocks);
  if ((rows % kRowsPerBlock) == 0U) {
    fp8_w8a16_small_m2_gemm_bf16_row_quad_kernel<true>
        <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activations,
                                          rows, columns, output);
  } else {
    fp8_w8a16_small_m2_gemm_bf16_row_quad_kernel<false>
        <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activations,
                                          rows, columns, output);
  }
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

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
void launch_fp8_small_m32_wmma_fixed_shape_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel<
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

void launch_nvfp4_small_m2_scale_codebook_single_row_test_unchecked(
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

void launch_nvfp4_small_m2_scale_codebook_row_pair_grid_cap_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = kWarpsPerBlock * 2U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < maximum_blocks ? wanted : maximum_blocks);
  nvfp4_w4a16_small_m2_gemm_bf16_scale_codebook_row_pair_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, rows,
          columns, output);
}

void launch_nvfp4_small_m2_scale_codebook_row_quad_grid_cap_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const unsigned int maximum_blocks,
    cudaStream_t const stream) noexcept {
  constexpr std::size_t kRowsPerBlock = kWarpsPerBlock * 4U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < maximum_blocks ? wanted : maximum_blocks);
  if ((rows % 4U) == 0U) {
    nvfp4_w4a16_small_m2_gemm_bf16_scale_codebook_row_quad_kernel<true>
        <<<blocks, kThreads, 0U, stream>>>(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output);
  } else {
    nvfp4_w4a16_small_m2_gemm_bf16_scale_codebook_row_quad_kernel<false>
        <<<blocks, kThreads, 0U, stream>>>(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output);
  }
}

void launch_nvfp4_small_m2_scale_codebook_unchecked(
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
  nvfp4_w4a16_small_m2_gemm_bf16_scale_codebook_row_pair_kernel
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

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
void launch_nvfp4_small_m32_wmma_k64_dual_a_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_kernel<
      kRows, kColumns, kSharedLeadingDimension>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output);
}

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
void launch_nvfp4_small_m32_wmma_k64_dual_a_scale_window_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
      kRows, kColumns, kSharedLeadingDimension>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output);
}

// Preserved scalar-store factorized specialization. Production uses the
// vector-store specialization below; this remains the same-cubin baseline.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
void launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_lookup_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
      kRows, kColumns, kSharedLeadingDimension, true>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output);
}

// Production factorized specialization that explicitly groups each four
// decoded BF16x2 words into one aligned shared-memory uint4 store.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U,
          unsigned int kValidTokenCount = 32U>
void launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_vector_store_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
      kRows, kColumns, kSharedLeadingDimension, true, true, kValidTokenCount>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output);
}

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
void launch_nvfp4_small_m17_m31_wmma_k64_dual_a_runtime_mask_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const unsigned int valid_token_count, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_kernel<
      kRows, kColumns, kSharedLeadingDimension>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output,
          valid_token_count);
}

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
void launch_nvfp4_small_m17_m31_wmma_k64_dual_a_runtime_mask_raw_weight_cp_async_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const unsigned int valid_token_count, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_raw_weight_cp_async_kernel<
      kRows, kColumns, kSharedLeadingDimension>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output,
          valid_token_count);
}

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 136U>
void launch_nvfp4_small_m16_wmma_k128_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m16_gemm_bf16_wmma_k128_kernel<
      kRows, kColumns, kSharedLeadingDimension>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output);
}

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 136U>
void launch_nvfp4_small_m32_wmma_k128_single_a_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m32_gemm_bf16_wmma_k128_single_a_kernel<
      kRows, kColumns, kSharedLeadingDimension>
      <<<kBlocks, kThreads, 0U, stream>>>(
          packed_weights, block_scales, weight_scale_2, activations, output);
}

}  // namespace

int launch_sm87_nvfp4_factorized_product_lookup_exhaustive_test_cuda(
    std::uint32_t* const factorized, std::uint32_t* const reference,
    void* const cuda_stream) noexcept {
  if (factorized == nullptr || reference == nullptr ||
      factorized == reference) {
    return invalid_value();
  }
  constexpr unsigned int kBlocks = 256U;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  nvfp4_factorized_product_lookup_exhaustive_kernel
      <<<kBlocks, kThreads, 0U, stream>>>(factorized, reference);
  return static_cast<int>(cudaGetLastError());
}

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
  const registry::ProjectionPlan plan = registry::select_projection_plan(
      make_fp8_projection_query(1U, weights, activation, rows, columns));
  switch (plan.route) {
    case registry::ProjectionRoute::kFp8M1RowQuad:
      launch_fp8_m1_output_row_group_grid_cap_unchecked<4U>(
          weights, weight_scale, activation, rows, columns, output,
          plan.maximum_blocks, stream);
      break;
    case registry::ProjectionRoute::kFp8M1RowPair:
      launch_fp8_m1_row_pair_unchecked(
          weights, weight_scale, activation, rows, columns, output, stream);
      break;
    case registry::ProjectionRoute::kFp8M1VectorGridCap:
      launch_fp8_vector_grid_cap_unchecked(
          weights, weight_scale, activation, rows, columns, output,
          plan.maximum_blocks, stream);
      break;
    case registry::ProjectionRoute::kFp8M1Vector:
      launch_fp8_vector_unchecked(weights, weight_scale, activation, rows,
                                  columns, output, stream);
      break;
    case registry::ProjectionRoute::kFp8M1Scalar:
      launch_fp8_scalar_unchecked(weights, weight_scale, activation, rows,
                                  columns, output, stream);
      break;
    default:
      return invalid_value();
  }
  return static_cast<int>(cudaGetLastError());
}

// Production exact-shape FP8 QKV/Z projection. The fixed launch retains the
// independently gated 1,536-CTA QKV and 768-CTA Z row-quad topologies while
// sharing launch overhead and decoded-codebook setup between both phases. Its
// double-buffered reduction scratch removes the predecessor's tail barrier.
int launch_sm87_fp8_w8a16_gemv_qkv_z_bf16_cuda(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    void* const cuda_stream) noexcept {
  const int validation = validate_fp8_qkv_z_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, activation,
      qkv_rows, z_rows, columns, qkv_output, z_output,
      kFp8QkvZProductionBlocks);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_qkv_z_launch_is_aligned(qkv_weights, z_weights, activation,
                                   qkv_output, z_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_qkv_z_two_phase_unchecked(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, activation,
      columns, qkv_output, z_output, kFp8QkvZProductionBlocks, stream);
  return static_cast<int>(cudaGetLastError());
}

// Production exact-shape FP8 K/V projection. One launch shares activation
// decode and codebook setup across both matrices while preserving each
// single-projection BF16 result bit-for-bit.
int launch_sm87_fp8_w8a16_gemv_pair_bf16_cuda(
    const std::uint8_t* const first_weights,
    const float first_weight_scale,
    const std::uint8_t* const second_weights,
    const float second_weight_scale,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output,
    void* const cuda_stream) noexcept {
  const int validation = validate_fp8_pair_launch(
      first_weights, first_weight_scale, second_weights, second_weight_scale,
      activation, rows, columns, first_output, second_output,
      kFp8KvPairSelectedMaximumBlocks);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool vector_shape =
      (reinterpret_cast<std::uintptr_t>(first_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(second_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(first_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(second_output) %
       alignof(std::uint16_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_projection_pair_row_quad_kernel
      <<<kFp8KvPairSelectedMaximumBlocks, kThreads, 0U, stream>>>(
          first_weights, first_weight_scale, second_weights,
          second_weight_scale, activation, rows, columns, first_output,
          second_output);
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

int launch_sm87_fp8_w8a16_gemv_bf16_row_pair_test_cuda(
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
  launch_fp8_m1_row_pair_unchecked(
      weights, weight_scale, activation, rows, columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_row_pair_grid_cap_test_cuda(
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
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_m1_output_row_group_grid_cap_unchecked<2U>(
      weights, weight_scale, activation, rows, columns, output,
      static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_row_quad_grid_cap_test_cuda(
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
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }
  constexpr std::size_t kRowsPerBlock = 4U;
  const std::size_t wanted =
      rows / kRowsPerBlock + (rows % kRowsPerBlock != 0U ? 1U : 0U);
  const unsigned int blocks = static_cast<unsigned int>(
      wanted < maximum_blocks ? wanted : maximum_blocks);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  if ((rows % kRowsPerBlock) == 0U) {
    fp8_w8a16_gemv_bf16_row_quad_unswizzled_baseline_test_kernel<true>
        <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activation,
                                          rows, columns, output);
  } else {
    fp8_w8a16_gemv_bf16_row_quad_unswizzled_baseline_test_kernel<false>
        <<<blocks, kThreads, 0U, stream>>>(weights, weight_scale, activation,
                                          rows, columns, output);
  }
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_row_quad_swizzled_codebook_grid_cap_test_cuda(
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
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_m1_output_row_group_grid_cap_unchecked<4U>(
      weights, weight_scale, activation, rows, columns, output,
      static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_qkv_z_two_phase_fused_grid_cap_test_cuda(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_qkv_z_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, activation,
      qkv_rows, z_rows, columns, qkv_output, z_output, maximum_blocks);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_qkv_z_launch_is_aligned(qkv_weights, z_weights, activation,
                                   qkv_output, z_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_qkv_z_two_phase_unchecked(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, activation,
      columns, qkv_output, z_output,
      static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_qkv_z_reduction_scratch_ping_pong_grid_cap_test_cuda(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_qkv_z_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, activation,
      qkv_rows, z_rows, columns, qkv_output, z_output, maximum_blocks);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_qkv_z_launch_is_aligned(qkv_weights, z_weights, activation,
                                   qkv_output, z_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_qkv_z_two_phase_unchecked(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, activation,
      columns, qkv_output, z_output,
      static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_qkv_z_tail_barrier_grid_cap_test_cuda(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_qkv_z_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, activation,
      qkv_rows, z_rows, columns, qkv_output, z_output, maximum_blocks);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_qkv_z_launch_is_aligned(qkv_weights, z_weights, activation,
                                   qkv_output, z_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_qkv_z_tail_barrier_test_unchecked(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, activation,
      columns, qkv_output, z_output,
      static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_qkv_z_two_phase_fused_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      fp8_w8a16_gemv_bf16_qkv_z_two_phase_reduction_scratch_ping_pong_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_qkv_z_two_phase_reduction_scratch_ping_pong_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int query_sm87_fp8_w8a16_m1_qkv_z_reduction_scratch_ping_pong_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      fp8_w8a16_gemv_bf16_qkv_z_two_phase_reduction_scratch_ping_pong_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_qkv_z_two_phase_reduction_scratch_ping_pong_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_fp8_w8a16_gemv_q_kv_bf16_cuda(
    const std::uint8_t* const q_weights, const float q_weight_scale,
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation, const std::size_t q_rows,
    const std::size_t kv_rows, const std::size_t columns,
    std::uint16_t* const q_output,
    std::uint16_t* const key_output, std::uint16_t* const value_output,
    void* const cuda_stream) noexcept {
  if (q_rows != kFp8FullAttentionQRows || kv_rows != kFp8KvPairRows ||
      columns != kFp8KvPairColumns) {
    return invalid_value();
  }
  const int q_validation = validate_fp8_launch(
      q_weights, q_weight_scale, activation, kFp8FullAttentionQRows,
      kFp8KvPairColumns, q_output);
  if (q_validation != static_cast<int>(cudaSuccess)) {
    return q_validation;
  }
  const int key_validation = validate_fp8_launch(
      key_weights, key_weight_scale, activation, kFp8KvPairRows,
      kFp8KvPairColumns, key_output);
  if (key_validation != static_cast<int>(cudaSuccess)) {
    return key_validation;
  }
  const int value_validation = validate_fp8_launch(
      value_weights, value_weight_scale, activation, kFp8KvPairRows,
      kFp8KvPairColumns, value_output);
  if (value_validation != static_cast<int>(cudaSuccess)) {
    return value_validation;
  }
  const bool vector_shape =
      (reinterpret_cast<std::uintptr_t>(q_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(key_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(value_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(q_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(key_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(value_output) %
       alignof(std::uint16_t)) == 0U;
  const std::size_t q_output_bytes =
      kFp8FullAttentionQRows * sizeof(std::uint16_t);
  const std::size_t kv_output_bytes =
      kFp8KvPairRows * sizeof(std::uint16_t);
  const std::size_t q_weight_bytes =
      kFp8FullAttentionQRows * kFp8KvPairColumns;
  const std::size_t kv_weight_bytes =
      kFp8KvPairRows * kFp8KvPairColumns;
  if (!vector_shape ||
      ranges_overlap(q_output, q_output_bytes, key_output,
                     kv_output_bytes) ||
      ranges_overlap(q_output, q_output_bytes, value_output,
                     kv_output_bytes) ||
      ranges_overlap(key_output, kv_output_bytes, value_output,
                     kv_output_bytes) ||
      ranges_overlap(q_output, q_output_bytes, key_weights,
                     kv_weight_bytes) ||
      ranges_overlap(q_output, q_output_bytes, value_weights,
                     kv_weight_bytes) ||
      ranges_overlap(key_output, kv_output_bytes, q_weights,
                     q_weight_bytes) ||
      ranges_overlap(key_output, kv_output_bytes, value_weights,
                     kv_weight_bytes) ||
      ranges_overlap(value_output, kv_output_bytes, q_weights,
                     q_weight_bytes) ||
      ranges_overlap(value_output, kv_output_bytes, key_weights,
                     kv_weight_bytes)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_q_kv_two_phase_row_group_kernel
      <<<kFp8FullAttentionBlocks, kThreads, 0U, stream>>>(
          q_weights, q_weight_scale, key_weights, key_weight_scale,
          value_weights, value_weight_scale, activation, q_output,
          key_output, value_output);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_q_kv_two_phase_fused_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, fp8_w8a16_gemv_bf16_q_kv_two_phase_row_group_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_q_kv_two_phase_row_group_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_fp8_w8a16_m1_kv_pair_row_quad_grid_cap_test_cuda(
    const std::uint8_t* const key_weights,
    const float key_weight_scale,
    const std::uint8_t* const value_weights,
    const float value_weight_scale,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    std::uint16_t* const key_output,
    std::uint16_t* const value_output,
    const std::size_t maximum_blocks,
    void* const cuda_stream) noexcept {
  const int validation = validate_fp8_pair_launch(
      key_weights, key_weight_scale, value_weights, value_weight_scale,
      activation, rows, columns, key_output, value_output, maximum_blocks);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool vector_shape =
      (reinterpret_cast<std::uintptr_t>(key_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(value_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(key_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(value_output) %
       alignof(std::uint16_t)) == 0U;
  if (!vector_shape) {
    return invalid_value();
  }

  const unsigned int blocks = static_cast<unsigned int>(maximum_blocks);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_projection_pair_row_quad_kernel
      <<<blocks, kThreads, 0U, stream>>>(
          key_weights, key_weight_scale, value_weights, value_weight_scale,
          activation, rows, columns, key_output, value_output);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_kv_pair_row_quad_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      fp8_w8a16_gemv_bf16_projection_pair_row_quad_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_projection_pair_row_quad_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool use_sm87_fp8_m1_row_pair_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_fp8_m1_row_pair_shape(rows, columns);
}

[[nodiscard]] std::size_t sm87_fp8_m1_row_quad_maximum_blocks_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return fp8_m1_row_quad_maximum_blocks(rows, columns);
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

// Test-only fixed-M32 candidate. It is intentionally restricted to the four
// exact production FP8 WMMA shapes and is not reachable from public dispatch.
int launch_sm87_fp8_w8a16_small_m32_wmma_fixed_shape_test_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_fp8_m32_launch(
      weights, weight_scale, activations, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!use_fp8_m16_wmma_fixed_shape(rows, columns) ||
      (reinterpret_cast<std::uintptr_t>(weights) % alignof(uint4)) != 0U ||
      (reinterpret_cast<std::uintptr_t>(activations) %
       alignof(std::uint64_t)) != 0U) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  if (rows == 10'240U) {
    launch_fp8_small_m32_wmma_fixed_shape_unchecked<10'240U, 5'120U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 5'120U) {
    launch_fp8_small_m32_wmma_fixed_shape_unchecked<5'120U, 6'144U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 6'144U) {
    launch_fp8_small_m32_wmma_fixed_shape_unchecked<6'144U, 5'120U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else {
    launch_fp8_small_m32_wmma_fixed_shape_unchecked<12'288U, 5'120U, 72U>(
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

int launch_sm87_fp8_w8a16_small_m2_row_pair_test_cuda(
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
  launch_fp8_small_m2_row_pair_unchecked(
      weights, weight_scale, activations, rows, columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

// Test-only capped launchers keep the exact production row-pair baseline and
// the row-quad candidate in the same cubin for occupancy-matched A/B sweeps.
int launch_sm87_fp8_w8a16_small_m2_row_pair_grid_cap_test_cuda(
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
  launch_fp8_small_m2_row_pair_grid_cap_test_unchecked(
      weights, weight_scale, activations, rows, columns, output,
      static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_small_m2_row_quad_grid_cap_test_cuda(
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
  launch_fp8_small_m2_row_quad_grid_cap_unchecked(
      weights, weight_scale, activations, rows, columns, output,
      static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

[[nodiscard]] bool use_sm87_fp8_m2_row_pair_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_fp8_m2_row_pair_shape(rows, columns);
}

[[nodiscard]] std::size_t sm87_fp8_m2_row_quad_maximum_blocks_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return fp8_m2_row_quad_maximum_blocks(rows, columns);
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
  const registry::ProjectionPlan plan = registry::select_projection_plan(
      make_nvfp4_projection_query(1U, packed_weights, block_scales,
                                  activation, rows, columns));
  switch (plan.route) {
    case registry::ProjectionRoute::kNvFp4M1DownActivationStaged:
      launch_nvfp4_down_activation_staged_unchecked(
          packed_weights, block_scales, weight_scale_2, activation, output,
          stream);
      break;
    case registry::ProjectionRoute::kNvFp4M1GateUpActivationStaged:
      launch_nvfp4_gate_up_activation_staged_unchecked(
          packed_weights, block_scales, weight_scale_2, activation, output,
          stream);
      break;
    case registry::ProjectionRoute::kNvFp4M1LmHeadActivationStaged:
      launch_nvfp4_lm_head_activation_staged_unchecked(
          packed_weights, block_scales, weight_scale_2, activation, output,
          stream);
      break;
    case registry::ProjectionRoute::kNvFp4M1ScaleCodebook:
      launch_nvfp4_scale_codebook_grid_cap_unchecked(
          packed_weights, block_scales, weight_scale_2, activation, rows,
          columns, output, plan.maximum_blocks, stream);
      break;
    case registry::ProjectionRoute::kNvFp4M1Vector:
      launch_nvfp4_vector_unchecked(packed_weights, block_scales,
                                    weight_scale_2, activation, rows, columns,
                                    output, stream);
      break;
    case registry::ProjectionRoute::kNvFp4M1Scalar:
      launch_nvfp4_scalar_unchecked(packed_weights, block_scales,
                                    weight_scale_2, activation, rows, columns,
                                    output, stream);
      break;
    default:
      return invalid_value();
  }
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_gemv_gate_up_silu_bf16_cuda(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const gate_output,
    std::uint16_t* const up_output, void* const cuda_stream) noexcept {
  if (rows != 17'408U || columns != 5'120U) {
    return invalid_value();
  }
  const int validation = validate_nvfp4_gate_up_pair_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, activation, rows,
      columns, gate_output, up_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(gate_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(gate_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_output) %
       alignof(std::uint16_t)) == 0U;
  if (!aligned) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_gate_up_pair_activation_staged_instance_unchecked<
      17'408U, 5'120U, true>(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, activation,
      gate_output, up_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_bf16_cuda(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    void* const cuda_stream) noexcept {
  if (rows != 17'408U || columns != 5'120U) {
    return invalid_value();
  }
  const int validation = validate_nvfp4_residual_norm_gate_up_silu_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, rows, columns, residual_output,
      gate_output, up_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(gate_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_left) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_right) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(norm_weight) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(gate_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_output) %
       alignof(std::uint16_t)) == 0U;
  if (!aligned) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_residual_norm_gate_up_silu_instance_unchecked<17'408U,
                                                               5'120U>(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, residual_output, gate_output,
      up_output, stream);
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

// Test-only capped launch of the exact production M=1 kernel. The existing
// entry above intentionally remains an uncapped same-cubin baseline after the
// production cap promotion so the gate can detect any cap-specific regression.
int launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_grid_cap_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  if (maximum_blocks == 0U || maximum_blocks > kMaximumBlocks) {
    return invalid_value();
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
  launch_nvfp4_scale_codebook_grid_cap_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

// Test-only capped launch of the preserved M=1 row-pair A/B baseline.
int launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_pair_grid_cap_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  if (maximum_blocks == 0U || maximum_blocks > kMaximumBlocks) {
    return invalid_value();
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
  launch_nvfp4_scale_codebook_row_pair_grid_cap_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_grid_cap_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  if (maximum_blocks == 0U || maximum_blocks > kMaximumBlocks) {
    return invalid_value();
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
  launch_nvfp4_scale_codebook_row_quad_grid_cap_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_row_quad_exact_shape_test_cuda(
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
  if (!vector_shape || !use_nvfp4_m1_row_quad_shape(rows, columns)) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_scale_codebook_row_quad_exact_shape_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_down_dual_iteration_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 5'120U && columns == 17'408U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  // Preserved indexed-shuffle baseline for same-binary A/B validation.
  launch_nvfp4_down_indexed_dual_baseline_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_down_dual_iteration_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_down_dual_iteration_kernel<
          5'120U, 17'408U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_down_dual_iteration_kernel<
          5'120U, 17'408U>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_down_xor_dual_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 5'120U && columns == 17'408U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_down_xor_dual_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_down_xor_dual_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_xor_dual_kernel<
          5'120U, 17'408U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_xor_dual_kernel<
          5'120U, 17'408U>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_down_activation_staged_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 5'120U && columns == 17'408U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_down_activation_staged_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_down_activation_staged_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel<
          5'120U, 17'408U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel<
          5'120U, 17'408U>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_down_residual_norm_bf16_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const raw_down_output,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kRows = 5'120U;
  constexpr std::size_t kColumns = 17'408U;
  constexpr std::size_t kPackedBytes = kRows * kColumns / 2U;
  constexpr std::size_t kScaleBytes = kRows * kColumns / 16U;
  constexpr std::size_t kActivationBytes =
      kColumns * sizeof(std::uint16_t);
  constexpr std::size_t kOutputBytes = kRows * sizeof(std::uint16_t);
  if (rows != kRows || columns != kColumns) {
    return invalid_value();
  }
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, kRows,
      kColumns, raw_down_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!std::isfinite(epsilon) || epsilon <= 0.0F ||
      residual_left == nullptr || norm_weight == nullptr ||
      residual_output == nullptr || normalized_output == nullptr ||
      byte_range_overflows(residual_left, kOutputBytes) ||
      byte_range_overflows(norm_weight, kOutputBytes) ||
      byte_range_overflows(residual_output, kOutputBytes) ||
      byte_range_overflows(normalized_output, kOutputBytes)) {
    return invalid_value();
  }
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_left) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(norm_weight) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(raw_down_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(normalized_output) %
       alignof(std::uint16_t)) == 0U;
  if (!aligned ||
      ranges_overlap(raw_down_output, kOutputBytes, residual_left,
                     kOutputBytes) ||
      ranges_overlap(raw_down_output, kOutputBytes, norm_weight,
                     kOutputBytes) ||
      ranges_overlap(raw_down_output, kOutputBytes, residual_output,
                     kOutputBytes) ||
      ranges_overlap(raw_down_output, kOutputBytes, normalized_output,
                     kOutputBytes) ||
      ranges_overlap(residual_output, kOutputBytes, residual_left,
                     kOutputBytes) ||
      ranges_overlap(residual_output, kOutputBytes, norm_weight,
                     kOutputBytes) ||
      ranges_overlap(residual_output, kOutputBytes, normalized_output,
                     kOutputBytes) ||
      ranges_overlap(normalized_output, kOutputBytes, residual_left,
                     kOutputBytes) ||
      ranges_overlap(normalized_output, kOutputBytes, norm_weight,
                     kOutputBytes) ||
      ranges_overlap(residual_output, kOutputBytes, packed_weights,
                     kPackedBytes) ||
      ranges_overlap(residual_output, kOutputBytes, block_scales,
                     kScaleBytes) ||
      ranges_overlap(residual_output, kOutputBytes, activation,
                     kActivationBytes) ||
      ranges_overlap(normalized_output, kOutputBytes, packed_weights,
                     kPackedBytes) ||
      ranges_overlap(normalized_output, kOutputBytes, block_scales,
                     kScaleBytes) ||
      ranges_overlap(normalized_output, kOutputBytes, activation,
                     kActivationBytes)) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  const cudaError_t launch_status =
      launch_nvfp4_down_residual_norm_unchecked(
          packed_weights, block_scales, weight_scale_2, activation,
          residual_left, norm_weight, epsilon, raw_down_output,
          residual_output, normalized_output, stream);
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_down_residual_norm_resources_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_down_residual_norm_activation_staged_kernel<
          5'120U, 17'408U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_down_residual_norm_activation_staged_kernel<
          5'120U, 17'408U>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_k5120_xor_dual_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 17'408U && columns == 5'120U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_gate_up_xor_dual_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_k5120_xor_dual_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_xor_dual_kernel<
          17'408U, 5'120U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_xor_dual_kernel<
          17'408U, 5'120U>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_lm_head_xor_dual_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 248'320U && columns == 5'120U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_lm_head_xor_dual_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_lm_head_xor_dual_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_xor_dual_kernel<
          248'320U, 5'120U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_xor_dual_kernel<
          248'320U, 5'120U>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_lm_head_activation_staged_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 248'320U && columns == 5'120U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_lm_head_activation_staged_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_lm_head_activation_staged_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel<
          248'320U, 5'120U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel<
          248'320U, 5'120U>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_gate_up_activation_staged_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_launch(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 17'408U && columns == 5'120U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_gate_up_activation_staged_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_gate_up_activation_staged_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel<
          17'408U, 5'120U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel<
          17'408U, 5'120U>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_test_cuda(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const gate_output,
    std::uint16_t* const up_output, void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_gate_up_pair_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, activation, rows,
      columns, gate_output, up_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 17'408U && columns == 5'120U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(gate_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(gate_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_gate_up_pair_activation_staged_test_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, activation, rows,
      columns, gate_output, up_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_gate_up_pair_activation_staged_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_kernel<17'408U,
                                                                    5'120U,
                                                                    false>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_kernel<17'408U,
                                                                    5'120U,
                                                                    false>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_gemv_bf16_gate_up_silu_activation_staged_test_cuda(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const gate_output,
    std::uint16_t* const up_output, void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_gate_up_pair_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, activation, rows,
      columns, gate_output, up_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 17'408U && columns == 5'120U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(gate_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(activation) %
       alignof(std::uint64_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(gate_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_gate_up_silu_activation_staged_test_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, activation, rows,
      columns, gate_output, up_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_gate_up_silu_activation_staged_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_kernel<17'408U,
                                                                    5'120U,
                                                                    true>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_kernel<17'408U,
                                                                    5'120U,
                                                                    true>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_test_cuda(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_residual_norm_gate_up_silu_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, rows, columns, residual_output,
      gate_output, up_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 17'408U && columns == 5'120U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(gate_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_left) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_right) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(norm_weight) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(gate_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_residual_norm_gate_up_silu_test_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, rows, columns, residual_output,
      gate_output, up_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_shared_tree_test_cuda(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_residual_norm_gate_up_silu_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, rows, columns, residual_output,
      gate_output, up_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape =
      (rows == 17'408U && columns == 5'120U) ||
      (rows == 2'048U && columns == 512U);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(gate_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_left) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_right) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(norm_weight) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(residual_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(gate_output) %
       alignof(std::uint16_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_output) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_residual_norm_gate_up_silu_shared_tree_test_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, rows, columns, residual_output,
      gate_output, up_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_residual_norm_gate_up_silu_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_activation_staged_kernel<
          17'408U, 5'120U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_activation_staged_kernel<
          17'408U, 5'120U>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int query_sm87_nvfp4_w4a16_m1_residual_norm_gate_up_silu_shared_tree_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_activation_staged_kernel<
          17'408U, 5'120U, false>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_activation_staged_kernel<
          17'408U, 5'120U, false>,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

// Test-only query sharing the exact M=1 production gate.
[[nodiscard]] bool use_sm87_nvfp4_m1_scale_codebook_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m1_scale_codebook(rows, columns);
}

[[nodiscard]] bool use_sm87_nvfp4_m1_row_quad_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m1_row_quad_shape(rows, columns);
}

[[nodiscard]] bool use_sm87_nvfp4_m1_down_activation_staged_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m1_down_activation_staged_shape(rows, columns);
}

[[nodiscard]] bool use_sm87_nvfp4_m1_gate_up_activation_staged_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m1_gate_up_activation_staged_shape(rows, columns);
}

[[nodiscard]] bool use_sm87_nvfp4_m1_lm_head_activation_staged_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m1_lm_head_activation_staged_shape(rows, columns);
}

// Test-only queries for the generic persistent cap, preserved row-pair A/B
// cap, and production row-quad cap.
[[nodiscard]] std::size_t
sm87_nvfp4_m1_persistent_maximum_blocks_test() noexcept {
  return kNvFp4M1PersistentMaximumBlocks;
}

[[nodiscard]] std::size_t
sm87_nvfp4_m1_row_pair_maximum_blocks_test() noexcept {
  return kNvFp4M1RowPairMaximumBlocks;
}

[[nodiscard]] std::size_t
sm87_nvfp4_m1_row_quad_maximum_blocks_test() noexcept {
  return kNvFp4M1RowQuadMaximumBlocks;
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
  launch_nvfp4_small_m2_scale_codebook_single_row_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activations, rows,
      columns, output, stream);
  return static_cast<int>(cudaGetLastError());
}

// Test-only direct entry for the exact M=2 row-pair kernel used by production.
// The entry above remains the preserved single-row same-cubin baseline.
int launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_test_cuda(
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

// Test-only capped launchers keep the exact production row-pair kernel and
// the row-quad candidate in one cubin for mirrored natural/occupancy-cap A/B.
int launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_pair_grid_cap_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 2U;
  const int validation = validate_nvfp4_small_m_launch(
      packed_weights, block_scales, weight_scale_2, activations, kTokenCount,
      rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  if (maximum_blocks == 0U || maximum_blocks > kMaximumBlocks) {
    return invalid_value();
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
  launch_nvfp4_small_m2_scale_codebook_row_pair_grid_cap_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activations, rows,
      columns, output, static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_small_m2_scale_codebook_row_quad_grid_cap_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const std::size_t maximum_blocks, void* const cuda_stream) noexcept {
  constexpr std::size_t kTokenCount = 2U;
  const int validation = validate_nvfp4_small_m_launch(
      packed_weights, block_scales, weight_scale_2, activations, kTokenCount,
      rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }
  if (maximum_blocks == 0U || maximum_blocks > kMaximumBlocks) {
    return invalid_value();
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
  launch_nvfp4_small_m2_scale_codebook_row_quad_grid_cap_unchecked(
      packed_weights, block_scales, weight_scale_2, activations, rows,
      columns, output, static_cast<unsigned int>(maximum_blocks), stream);
  return static_cast<int>(cudaGetLastError());
}

[[nodiscard]] bool use_sm87_nvfp4_m2_scale_codebook_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m2_scale_codebook(rows, columns);
}

[[nodiscard]] bool use_sm87_nvfp4_m2_row_quad_shape_test(
    const std::size_t rows, const std::size_t columns) noexcept {
  return use_nvfp4_m2_row_quad_shape(rows, columns);
}

[[nodiscard]] std::size_t
sm87_nvfp4_m2_row_quad_maximum_blocks_test() noexcept {
  return kNvFp4M2RowQuadMaximumBlocks;
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

// Test-only direct entry for the K128/LD136 exact-shape candidates.
int launch_sm87_nvfp4_w4a16_small_m16_wmma_k128_test_cuda(
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
    launch_nvfp4_small_m16_wmma_k128_unchecked<17'408U, 5'120U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  } else {
    launch_nvfp4_small_m16_wmma_k128_unchecked<5'120U, 17'408U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  }
  return static_cast<int>(cudaGetLastError());
}

// Test-only direct entry for the preserved pre-scale-window K64/LD72
// fixed-M32 baseline.
int launch_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_m32_launch(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns)) {
    return invalid_value();
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
    launch_nvfp4_small_m32_wmma_k64_dual_a_unchecked<17'408U, 5'120U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  } else {
    launch_nvfp4_small_m32_wmma_k64_dual_a_unchecked<5'120U, 17'408U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 17'408U) {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_kernel<17'408U,
                                                                    5'120U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_kernel<17'408U,
                                                                      5'120U>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_kernel<5'120U,
                                                                   17'408U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_kernel<5'120U,
                                                                     17'408U>,
          static_cast<int>(kThreads), 0U);
    }
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

// Test-only direct entry for the preserved pre-factorization K256 scale-window
// baseline. This keeps the public ABI free of experiment controls.
int launch_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_scale_window_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_m32_launch(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns)) {
    return invalid_value();
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
    launch_nvfp4_small_m32_wmma_k64_dual_a_scale_window_unchecked<17'408U,
                                                                    5'120U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  } else {
    launch_nvfp4_small_m32_wmma_k64_dual_a_scale_window_unchecked<5'120U,
                                                                   17'408U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_scale_window_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 17'408U) {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
            17'408U, 5'120U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              17'408U, 5'120U>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
            5'120U, 17'408U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              5'120U, 17'408U>,
          static_cast<int>(kThreads), 0U);
    }
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_factorized_lookup_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_m32_launch(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns)) {
    return invalid_value();
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
    launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_lookup_unchecked<
        17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                         activations, output, stream);
  } else {
    launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_lookup_unchecked<
        5'120U, 17'408U>(packed_weights, block_scales, weight_scale_2,
                         activations, output, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_factorized_lookup_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 17'408U) {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
            17'408U, 5'120U, 72U, true>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              17'408U, 5'120U, 72U, true>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
            5'120U, 17'408U, 72U, true>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              5'120U, 17'408U, 72U, true>,
          static_cast<int>(kThreads), 0U);
    }
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_factorized_vector_store_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_m32_launch(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns)) {
    return invalid_value();
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
    launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_vector_store_unchecked<
        17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                         activations, output, stream);
  } else {
    launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_vector_store_unchecked<
        5'120U, 17'408U>(packed_weights, block_scales, weight_scale_2,
                         activations, output, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_factorized_vector_store_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 17'408U) {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
            17'408U, 5'120U, 72U, true, true>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              17'408U, 5'120U, 72U, true, true>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
            5'120U, 17'408U, 72U, true, true>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              5'120U, 17'408U, 72U, true, true>,
          static_cast<int>(kThreads), 0U);
    }
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

// Test-only resource gate for the exact production M18 specialization.
int query_sm87_nvfp4_w4a16_small_m18_masked_m32_wmma_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 17'408U) {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
            17'408U, 5'120U, 72U, true, true, 18U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              17'408U, 5'120U, 72U, true, true, 18U>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
            5'120U, 17'408U, 72U, true, true, 18U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              5'120U, 17'408U, 72U, true, true, 18U>,
          static_cast<int>(kThreads), 0U);
    }
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

// Test-only direct entry for one runtime-valid masked-M32 kernel. Production
// dispatch remains on the fixed M18/M32 specializations and existing
// decompositions while this candidate is evaluated.
int launch_sm87_nvfp4_w4a16_small_m17_m31_runtime_masked_m32_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t valid_token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_m17_m31_launch(
      packed_weights, block_scales, weight_scale_2, activations,
      valid_token_count, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns)) {
    return invalid_value();
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
    launch_nvfp4_small_m17_m31_wmma_k64_dual_a_runtime_mask_unchecked<
        17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                         activations,
                         static_cast<unsigned int>(valid_token_count), output,
                         stream);
  } else {
    launch_nvfp4_small_m17_m31_wmma_k64_dual_a_runtime_mask_unchecked<
        5'120U, 17'408U>(packed_weights, block_scales, weight_scale_2,
                         activations,
                         static_cast<unsigned int>(valid_token_count), output,
                         stream);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_small_m17_m31_runtime_masked_m32_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 17'408U) {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_kernel<
            17'408U, 5'120U, 72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_kernel<
              17'408U, 5'120U, 72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_kernel<
            5'120U, 17'408U, 72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_kernel<
              5'120U, 17'408U, 72U>,
          static_cast<int>(kThreads), 0U);
    }
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

// Test-only direct entry for the single-4-KiB raw-weight cp.async candidate.
// Production dispatch, the public header, and the projection registry remain
// intentionally unchanged.
int launch_sm87_nvfp4_w4a16_small_m17_m31_runtime_masked_m32_raw_weight_cp_async_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t valid_token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_m17_m31_launch(
      packed_weights, block_scales, weight_scale_2, activations,
      valid_token_count, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns)) {
    return invalid_value();
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
    launch_nvfp4_small_m17_m31_wmma_k64_dual_a_runtime_mask_raw_weight_cp_async_unchecked<
        17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                         activations,
                         static_cast<unsigned int>(valid_token_count), output,
                         stream);
  } else {
    launch_nvfp4_small_m17_m31_wmma_k64_dual_a_runtime_mask_raw_weight_cp_async_unchecked<
        5'120U, 17'408U>(packed_weights, block_scales, weight_scale_2,
                         activations,
                         static_cast<unsigned int>(valid_token_count), output,
                         stream);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_small_m17_m31_runtime_masked_m32_raw_weight_cp_async_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 17'408U) {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_raw_weight_cp_async_kernel<
            17'408U, 5'120U, 72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_raw_weight_cp_async_kernel<
              17'408U, 5'120U, 72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_raw_weight_cp_async_kernel<
            5'120U, 17'408U, 72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m17_m31_gemm_bf16_wmma_k64_dual_a_runtime_mask_raw_weight_cp_async_kernel<
              5'120U, 17'408U, 72U>,
          static_cast<int>(kThreads), 0U);
    }
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

// Test-only direct entry for the K128/LD136 fixed-M32 single-resident-A-panel
// candidate. Production dispatch and the public ABI remain unchanged.
int launch_sm87_nvfp4_w4a16_small_m32_wmma_k128_single_a_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_m32_launch(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns)) {
    return invalid_value();
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
    launch_nvfp4_small_m32_wmma_k128_single_a_unchecked<17'408U, 5'120U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  } else {
    launch_nvfp4_small_m32_wmma_k128_single_a_unchecked<5'120U, 17'408U>(
        packed_weights, block_scales, weight_scale_2, activations, output,
        stream);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_small_m32_wmma_k128_single_a_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_nvfp4_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 17'408U) {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k128_single_a_kernel<17'408U,
                                                                         5'120U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k128_single_a_kernel<17'408U,
                                                                           5'120U>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k128_single_a_kernel<5'120U,
                                                                        17'408U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k128_single_a_kernel<5'120U,
                                                                          17'408U>,
          static_cast<int>(kThreads), 0U);
    }
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
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

  const registry::ProjectionPlan plan = registry::select_projection_plan(
      make_fp8_projection_query(16U, weights, activations, rows, columns));
  if (plan.route == registry::ProjectionRoute::kFp8M16Wmma) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    switch (plan.shape) {
      case registry::ProjectionShape::kFp8_10240x5120:
        launch_fp8_small_m16_wmma_fixed_shape_unchecked<10'240U, 5'120U,
                                                        72U>(
            weights, weight_scale, activations, output, stream);
        break;
      case registry::ProjectionShape::kFp8_5120x6144:
        launch_fp8_small_m16_wmma_fixed_shape_unchecked<5'120U, 6'144U, 72U>(
            weights, weight_scale, activations, output, stream);
        break;
      case registry::ProjectionShape::kFp8_6144x5120:
        launch_fp8_small_m16_wmma_fixed_shape_unchecked<6'144U, 5'120U, 72U>(
            weights, weight_scale, activations, output, stream);
        break;
      case registry::ProjectionShape::kFp8_12288x5120:
        launch_fp8_small_m16_wmma_fixed_shape_unchecked<12'288U, 5'120U,
                                                        72U>(
            weights, weight_scale, activations, output, stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }
  if (plan.route != registry::ProjectionRoute::kSplitM16IntoM8) {
    return invalid_value();
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

int launch_sm87_fp8_w8a16_m32_gemm_bf16_cuda(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kHalfTokens = 16U;
  const int validation = validate_fp8_m32_launch(
      weights, weight_scale, activations, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }

  const registry::ProjectionPlan plan = registry::select_projection_plan(
      make_fp8_projection_query(32U, weights, activations, rows, columns));
  if (plan.route == registry::ProjectionRoute::kFp8M32Wmma) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    switch (plan.shape) {
      case registry::ProjectionShape::kFp8_10240x5120:
        launch_fp8_small_m32_wmma_fixed_shape_unchecked<10'240U, 5'120U,
                                                        72U>(
            weights, weight_scale, activations, output, stream);
        break;
      case registry::ProjectionShape::kFp8_5120x6144:
        launch_fp8_small_m32_wmma_fixed_shape_unchecked<5'120U, 6'144U, 72U>(
            weights, weight_scale, activations, output, stream);
        break;
      case registry::ProjectionShape::kFp8_6144x5120:
        launch_fp8_small_m32_wmma_fixed_shape_unchecked<6'144U, 5'120U, 72U>(
            weights, weight_scale, activations, output, stream);
        break;
      case registry::ProjectionShape::kFp8_12288x5120:
        launch_fp8_small_m32_wmma_fixed_shape_unchecked<12'288U, 5'120U,
                                                        72U>(
            weights, weight_scale, activations, output, stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }
  if (plan.route != registry::ProjectionRoute::kSplitM32IntoM16) {
    return invalid_value();
  }

  int status = launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
      weights, weight_scale, activations, rows, columns, output, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  return launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
      weights, weight_scale, activations + kHalfTokens * columns, rows,
      columns, output + kHalfTokens * rows, cuda_stream);
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

  const registry::ProjectionPlan plan = registry::select_projection_plan(
      make_fp8_projection_query(token_count, weights, activations, rows,
                                columns));
  if (plan.route != registry::ProjectionRoute::kSerialM1) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    switch (plan.route) {
      case registry::ProjectionRoute::kFp8M2RowQuad:
        launch_fp8_small_m2_row_quad_grid_cap_unchecked(
            weights, weight_scale, activations, rows, columns, output,
            plan.maximum_blocks, stream);
        break;
      case registry::ProjectionRoute::kFp8M2RowPair:
        launch_fp8_small_m2_row_pair_unchecked(
            weights, weight_scale, activations, rows, columns, output, stream);
        break;
      case registry::ProjectionRoute::kFp8M2VectorGridCap:
        launch_fp8_small_m_vector_grid_cap_unchecked<2U>(
            weights, weight_scale, activations, rows, columns, output,
            plan.maximum_blocks, stream);
        break;
      case registry::ProjectionRoute::kFp8SmallMVector:
        switch (token_count) {
          case 2U:
            launch_fp8_small_m_vector_unchecked<2U>(
                weights, weight_scale, activations, rows, columns, output,
                stream);
            break;
          case 3U:
            launch_fp8_small_m_vector_unchecked<3U>(
                weights, weight_scale, activations, rows, columns, output,
                stream);
            break;
          case 4U:
            launch_fp8_small_m_vector_unchecked<4U>(
                weights, weight_scale, activations, rows, columns, output,
                stream);
            break;
          case 5U:
            launch_fp8_small_m_vector_unchecked<5U>(
                weights, weight_scale, activations, rows, columns, output,
                stream);
            break;
          case 6U:
            launch_fp8_small_m_vector_unchecked<6U>(
                weights, weight_scale, activations, rows, columns, output,
                stream);
            break;
          case 7U:
            launch_fp8_small_m_vector_unchecked<7U>(
                weights, weight_scale, activations, rows, columns, output,
                stream);
            break;
          case 8U:
            launch_fp8_small_m_vector_unchecked<8U>(
                weights, weight_scale, activations, rows, columns, output,
                stream);
            break;
          default:
            return invalid_value();
        }
        break;
      case registry::ProjectionRoute::kFp8M8Fixed:
        switch (plan.shape) {
          case registry::ProjectionShape::kFp8_10240x5120:
            launch_fp8_small_m8_fixed_shape_unchecked<10'240U, 5'120U>(
                weights, weight_scale, activations, output, stream);
            break;
          case registry::ProjectionShape::kFp8_5120x6144:
            launch_fp8_small_m8_fixed_shape_unchecked<5'120U, 6'144U>(
                weights, weight_scale, activations, output, stream);
            break;
          case registry::ProjectionShape::kFp8_6144x5120:
            launch_fp8_small_m8_fixed_shape_unchecked<6'144U, 5'120U>(
                weights, weight_scale, activations, output, stream);
            break;
          case registry::ProjectionShape::kFp8_12288x5120:
            launch_fp8_small_m8_fixed_shape_unchecked<12'288U, 5'120U>(
                weights, weight_scale, activations, output, stream);
            break;
          case registry::ProjectionShape::kFp8_1024x5120:
            launch_fp8_small_m8_fixed_shape_unchecked<1'024U, 5'120U>(
                weights, weight_scale, activations, output, stream);
            break;
          default:
            return invalid_value();
        }
        break;
      case registry::ProjectionRoute::kFp8M8RowPair:
        launch_fp8_small_m8_row_pair_unchecked(
            weights, weight_scale, activations, rows, columns, output, stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }

  for (std::size_t token = 0U; token < plan.launch_count; ++token) {
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

  const registry::ProjectionPlan plan = registry::select_projection_plan(
      make_nvfp4_projection_query(16U, packed_weights, block_scales,
                                  activations, rows, columns));
  if (plan.route == registry::ProjectionRoute::kNvFp4M16Wmma) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    switch (plan.shape) {
      case registry::ProjectionShape::kNvFp4_17408x5120:
        launch_nvfp4_small_m16_wmma_k128_unchecked<17'408U, 5'120U>(
            packed_weights, block_scales, weight_scale_2, activations, output,
            stream);
        break;
      case registry::ProjectionShape::kNvFp4_5120x17408:
        launch_nvfp4_small_m16_wmma_k128_unchecked<5'120U, 17'408U>(
            packed_weights, block_scales, weight_scale_2, activations, output,
            stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }
  if (plan.route != registry::ProjectionRoute::kSplitM16IntoM8) {
    return invalid_value();
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

int launch_sm87_nvfp4_w4a16_m18_gemm_bf16_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kPrefixTokens = 16U;
  constexpr std::size_t kTailTokens = 2U;
  const int validation = validate_nvfp4_m18_launch(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }

  const registry::ProjectionPlan plan = registry::select_projection_plan(
      make_nvfp4_projection_query(18U, packed_weights, block_scales,
                                  activations, rows, columns));
  if (plan.route == registry::ProjectionRoute::kNvFp4M18MaskedM32Wmma) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    switch (plan.shape) {
      case registry::ProjectionShape::kNvFp4_17408x5120:
        launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_vector_store_unchecked<
            17'408U, 5'120U, 72U, 18U>(
            packed_weights, block_scales, weight_scale_2, activations, output,
            stream);
        break;
      case registry::ProjectionShape::kNvFp4_5120x17408:
        launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_vector_store_unchecked<
            5'120U, 17'408U, 72U, 18U>(
            packed_weights, block_scales, weight_scale_2, activations, output,
            stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }
  if (plan.route != registry::ProjectionRoute::kSplitM18IntoM16M2) {
    return invalid_value();
  }

  int status = launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  return launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
      packed_weights, block_scales, weight_scale_2,
      activations + kPrefixTokens * columns, kTailTokens, rows, columns,
      output + kPrefixTokens * rows, cuda_stream);
}

int launch_sm87_nvfp4_w4a16_m17_m31_gemm_bf16_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kPrefixTokens = 16U;
  constexpr std::size_t kMaximumTailTokens = 8U;
  if (token_count == 18U) {
    return invalid_value();
  }
  const int validation = validate_nvfp4_m17_m31_launch(
      packed_weights, block_scales, weight_scale_2, activations, token_count,
      rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }

  const registry::ProjectionPlan plan = registry::select_projection_plan(
      make_nvfp4_projection_query(token_count, packed_weights, block_scales,
                                  activations, rows, columns));
  if (plan.route ==
      registry::ProjectionRoute::kNvFp4M17M31RuntimeMaskedM32Wmma) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    switch (plan.shape) {
      case registry::ProjectionShape::kNvFp4_17408x5120:
        launch_nvfp4_small_m17_m31_wmma_k64_dual_a_runtime_mask_unchecked<
            17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                             activations,
                             static_cast<unsigned int>(token_count), output,
                             stream);
        break;
      case registry::ProjectionShape::kNvFp4_5120x17408:
        launch_nvfp4_small_m17_m31_wmma_k64_dual_a_runtime_mask_unchecked<
            5'120U, 17'408U>(packed_weights, block_scales, weight_scale_2,
                             activations,
                             static_cast<unsigned int>(token_count), output,
                             stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }
  if (plan.route !=
      registry::ProjectionRoute::kSplitM17M31IntoM16AndSmallM) {
    return invalid_value();
  }

  int status = launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  std::size_t token_offset = kPrefixTokens;
  while (token_offset < token_count) {
    const std::size_t launch_tokens =
        std::min(kMaximumTailTokens, token_count - token_offset);
    status = launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
        packed_weights, block_scales, weight_scale_2,
        activations + token_offset * columns, launch_tokens, rows, columns,
        output + token_offset * rows, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    token_offset += launch_tokens;
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_w4a16_m32_gemm_bf16_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kHalfTokens = 16U;
  const int validation = validate_nvfp4_m32_launch(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U) {
    return validation;
  }

  const registry::ProjectionPlan plan = registry::select_projection_plan(
      make_nvfp4_projection_query(32U, packed_weights, block_scales,
                                  activations, rows, columns));
  if (plan.route == registry::ProjectionRoute::kNvFp4M32Wmma) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    switch (plan.shape) {
      case registry::ProjectionShape::kNvFp4_17408x5120:
        launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_vector_store_unchecked<
            17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                             activations, output, stream);
        break;
      case registry::ProjectionShape::kNvFp4_5120x17408:
        launch_nvfp4_small_m32_wmma_k64_dual_a_factorized_vector_store_unchecked<
            5'120U, 17'408U>(packed_weights, block_scales, weight_scale_2,
                             activations, output, stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }
  if (plan.route != registry::ProjectionRoute::kSplitM32IntoM16) {
    return invalid_value();
  }

  int status = launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
      packed_weights, block_scales, weight_scale_2, activations, rows, columns,
      output, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  return launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
      packed_weights, block_scales, weight_scale_2,
      activations + kHalfTokens * columns, rows, columns,
      output + kHalfTokens * rows, cuda_stream);
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

  const registry::ProjectionPlan nvfp4_plan =
      registry::select_projection_plan(make_nvfp4_projection_query(
          token_count, packed_weights, block_scales, activations, rows,
          columns));
  if (nvfp4_plan.route != registry::ProjectionRoute::kSerialM1) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    (void)cudaGetLastError();
    switch (nvfp4_plan.route) {
      case registry::ProjectionRoute::kNvFp4M2RowQuad:
        launch_nvfp4_small_m2_scale_codebook_row_quad_grid_cap_unchecked(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, nvfp4_plan.maximum_blocks, stream);
        break;
      case registry::ProjectionRoute::kNvFp4M2ScaleCodebook:
        launch_nvfp4_small_m2_scale_codebook_unchecked(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, stream);
        break;
      case registry::ProjectionRoute::kNvFp4SmallMVector:
        switch (token_count) {
          case 2U:
            launch_nvfp4_small_m_vector_unchecked<2U>(
                packed_weights, block_scales, weight_scale_2, activations,
                rows, columns, output, stream);
            break;
          case 3U:
            launch_nvfp4_small_m_vector_unchecked<3U>(
                packed_weights, block_scales, weight_scale_2, activations,
                rows, columns, output, stream);
            break;
          case 4U:
            launch_nvfp4_small_m_vector_unchecked<4U>(
                packed_weights, block_scales, weight_scale_2, activations,
                rows, columns, output, stream);
            break;
          case 5U:
            launch_nvfp4_small_m_vector_unchecked<5U>(
                packed_weights, block_scales, weight_scale_2, activations,
                rows, columns, output, stream);
            break;
          case 6U:
            launch_nvfp4_small_m_vector_unchecked<6U>(
                packed_weights, block_scales, weight_scale_2, activations,
                rows, columns, output, stream);
            break;
          case 7U:
            launch_nvfp4_small_m_vector_unchecked<7U>(
                packed_weights, block_scales, weight_scale_2, activations,
                rows, columns, output, stream);
            break;
          case 8U:
            launch_nvfp4_small_m_vector_unchecked<8U>(
                packed_weights, block_scales, weight_scale_2, activations,
                rows, columns, output, stream);
            break;
          default:
            return invalid_value();
        }
        break;
      case registry::ProjectionRoute::kNvFp4M8Fixed:
        switch (nvfp4_plan.shape) {
          case registry::ProjectionShape::kNvFp4_17408x5120:
            launch_nvfp4_small_m8_fixed_shape_unchecked<17'408U, 5'120U>(
                packed_weights, block_scales, weight_scale_2, activations,
                output, stream);
            break;
          case registry::ProjectionShape::kNvFp4_5120x17408:
            launch_nvfp4_small_m8_fixed_shape_unchecked<5'120U, 17'408U>(
                packed_weights, block_scales, weight_scale_2, activations,
                output, stream);
            break;
          default:
            return invalid_value();
        }
        break;
      case registry::ProjectionRoute::kNvFp4M8ScaleCodebook:
        launch_nvfp4_small_m8_scale_codebook_unchecked(
            packed_weights, block_scales, weight_scale_2, activations, rows,
            columns, output, stream);
        break;
      default:
        return invalid_value();
    }
    return static_cast<int>(cudaGetLastError());
  }

  for (std::size_t token = 0U; token < nvfp4_plan.launch_count; ++token) {
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
