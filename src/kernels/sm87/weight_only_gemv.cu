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
constexpr std::size_t kFp8FullAttentionPackedColumns =
    kFp8KvPairColumns / kFp8VectorValuesPerLane;
constexpr unsigned int kFp8FullAttentionQRowQuads =
    static_cast<unsigned int>(kFp8FullAttentionQRows / 4U);
constexpr unsigned int kFp8FullAttentionKvRowPairs =
    static_cast<unsigned int>(kFp8KvPairRows / 2U);
constexpr unsigned int kFp8FullAttentionAosoa4PackBlocks =
    kFp8FullAttentionQRowQuads + kFp8FullAttentionKvRowPairs;
constexpr unsigned int kFp8FullAttentionCta512Threads = 2U * kThreads;
constexpr unsigned int kFp8FullAttentionCta512Blocks =
    kFp8FullAttentionBlocks / 2U;
constexpr std::size_t kFp8QkvRows = 10'240U;
constexpr std::size_t kFp8ZRows = 6'144U;
constexpr std::size_t kFp8QkvZColumns = 5'120U;
constexpr std::size_t kFp8OutputProjectionRows = 5'120U;
constexpr std::size_t kFp8OutputProjectionColumns = 6'144U;
constexpr unsigned int kFp8OutputProjectionRowQuads =
    static_cast<unsigned int>(kFp8OutputProjectionRows / 4U);
constexpr std::size_t kFp8OutputProjectionPackedColumns =
    kFp8OutputProjectionColumns / kFp8VectorValuesPerLane;
constexpr unsigned int kFp8OutputProjectionAosoa4Blocks = 1'024U;
constexpr unsigned int kFp8QkvRowQuads =
    static_cast<unsigned int>(kFp8QkvRows / 4U);
constexpr unsigned int kFp8ZRowQuads =
    static_cast<unsigned int>(kFp8ZRows / 4U);
constexpr unsigned int kFp8QkvZMaximumTestBlocks =
    kFp8QkvRowQuads + kFp8ZRowQuads;
constexpr unsigned int kFp8QkvZProductionBlocks = 1'536U;
constexpr unsigned int kFp8QkvZMaximumZBlocks = 768U;
constexpr std::size_t kLinearAttentionAbRows = 48U;
constexpr unsigned int kFp8QkvZAbFirstTailBlock = 1'024U;
constexpr std::size_t kLinearAttentionConvHistoryWidth = 3U;
constexpr std::size_t kLinearAttentionConvKernelWidth = 4U;
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
static_assert((kFp8OutputProjectionRows % 4U) == 0U);
static_assert((kFp8OutputProjectionColumns % kFp8VectorValuesPerLane) == 0U);
static_assert((kFp8FullAttentionQRows % 4U) == 0U);
static_assert((kFp8KvPairRows % 2U) == 0U);
static_assert((kFp8KvPairColumns % kFp8VectorValuesPerLane) == 0U);
static_assert(kFp8FullAttentionQRowQuads ==
              kFp8FullAttentionBlocks +
                  kFp8FullAttentionKvFirstBlock);
static_assert(kFp8FullAttentionKvRowPairs == kFp8FullAttentionKvBlocks);
static_assert(kFp8FullAttentionCta512Threads == 512U);
static_assert((kFp8FullAttentionBlocks % 2U) == 0U);
static_assert((kFp8FullAttentionKvFirstBlock % 2U) == 0U);
static_assert((kFp8FullAttentionKvBlocks % 2U) == 0U);

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

[[nodiscard]] __device__ __forceinline__ std::uint32_t
fp8_swizzle_packed_codes(const std::uint32_t packed) {
  constexpr std::uint32_t kFp8PackedHighBits = 0x0707'0707U;
  return packed ^ ((packed >> 5U) & kFp8PackedHighBits);
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

// Exact [5120,6144] production/test body. The sidecar groups four adjacent
// rows for each packed four-column word as uint4{x,y,z,w}. Every byte is
// already transformed with code ^ (code >> 5), so one aligned 128-bit load
// replaces four scalar row loads and the canonical-layout runtime swizzle.
// The historical internal name is retained so the selected test SASS symbol
// remains stable while both public production and direct test launchers share
// this one kernel image.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_row_quad_aosoa4_preswizzled_test_kernel(
    const uint4* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  const std::size_t packed_columns = columns / kFp8VectorValuesPerLane;
  const std::size_t first_row = 4U * static_cast<std::size_t>(blockIdx.x);
  const std::size_t row_stride =
      4U * static_cast<std::size_t>(gridDim.x);
  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row_quad = row0 / 4U;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

    for (std::size_t first_column =
             static_cast<std::size_t>(threadIdx.x) *
             kFp8VectorValuesPerLane;
         first_column < columns;
         first_column += kFp8VectorColumnsPerBlock) {
      const std::size_t packed_column =
          first_column / kFp8VectorValuesPerLane;
      const uint4 packed_weights =
          weights[row_quad * packed_columns + packed_column];
      const std::uint64_t packed_activation =
          *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
        const unsigned int weight_shift = value * 8U;
        const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
            (packed_weights.x >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
            (packed_weights.y >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight2 = static_cast<std::uint8_t>(
            (packed_weights.z >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight3 = static_cast<std::uint8_t>(
            (packed_weights.w >> weight_shift) & 0xffU);
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
        output[row0 + 1U] = encode_bf16_rne(block_sum1);
        output[row0 + 2U] = encode_bf16_rne(block_sum2);
        output[row0 + 3U] = encode_bf16_rne(block_sum3);
      }
    }
    __syncthreads();
  }
}

// Test-only production-topology twin for the bounded Decode output-
// projection cache-policy screen. Only the AoSoA4 sidecar's aligned uint4
// load is evict-first; activation loads retain the production compiler-
// default cache policy.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_row_quad_aosoa4_preswizzled_cs_test_kernel(
    const uint4* const weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  const std::size_t packed_columns = columns / kFp8VectorValuesPerLane;
  const std::size_t first_row = 4U * static_cast<std::size_t>(blockIdx.x);
  const std::size_t row_stride =
      4U * static_cast<std::size_t>(gridDim.x);
  for (std::size_t row0 = first_row; row0 < rows; row0 += row_stride) {
    const std::size_t row_quad = row0 / 4U;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

    for (std::size_t first_column =
             static_cast<std::size_t>(threadIdx.x) *
             kFp8VectorValuesPerLane;
         first_column < columns;
         first_column += kFp8VectorColumnsPerBlock) {
      const std::size_t packed_column =
          first_column / kFp8VectorValuesPerLane;
      const uint4 packed_weights = __ldcs(
          weights + row_quad * packed_columns + packed_column);
      const std::uint64_t packed_activation =
          *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane; ++value) {
        const unsigned int weight_shift = value * 8U;
        const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
            (packed_weights.x >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
            (packed_weights.y >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight2 = static_cast<std::uint8_t>(
            (packed_weights.z >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight3 = static_cast<std::uint8_t>(
            (packed_weights.w >> weight_shift) & 0xffU);
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
        output[row0 + 1U] = encode_bf16_rne(block_sum1);
        output[row0 + 2U] = encode_bf16_rne(block_sum2);
        output[row0 + 3U] = encode_bf16_rne(block_sum3);
      }
    }
    __syncthreads();
  }
}

// Test-only Decode chain candidate. The output projection's raw BF16 value is
// still rounded independently before the residual add, but the dead raw
// publication is replaced by the runner-visible rounded residual. Keeping a
// separate kernel image leaves the production ABI and its selected SASS
// untouched until the complete two-launch chain clears the checkpoint gate.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_row_quad_aosoa4_residual_epilogue_test_kernel(
    const uint4* const weights, const float weight_scale,
    const std::uint16_t* const activation,
    const std::uint16_t* const residual_left,
    std::uint16_t* const residual_output) {
  constexpr std::size_t kRows = kFp8OutputProjectionRows;
  constexpr std::size_t kColumns = kFp8OutputProjectionColumns;
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  constexpr std::size_t kPackedColumns =
      kColumns / kFp8VectorValuesPerLane;
  const std::size_t first_row = 4U * static_cast<std::size_t>(blockIdx.x);
  const std::size_t row_stride =
      4U * static_cast<std::size_t>(gridDim.x);
  for (std::size_t row0 = first_row; row0 < kRows; row0 += row_stride) {
    const std::size_t row_quad = row0 / 4U;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

    for (std::size_t first_column =
             static_cast<std::size_t>(threadIdx.x) *
             kFp8VectorValuesPerLane;
         first_column < kColumns;
         first_column += kFp8VectorColumnsPerBlock) {
      const std::size_t packed_column =
          first_column / kFp8VectorValuesPerLane;
      const uint4 packed_weights =
          weights[row_quad * kPackedColumns + packed_column];
      const std::uint64_t packed_activation =
          *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
      for (unsigned int value = 0U; value < kFp8VectorValuesPerLane;
           ++value) {
        const unsigned int weight_shift = value * 8U;
        const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
            (packed_weights.x >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
            (packed_weights.y >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight2 = static_cast<std::uint8_t>(
            (packed_weights.z >> weight_shift) & 0xffU);
        const std::uint8_t encoded_weight3 = static_cast<std::uint8_t>(
            (packed_weights.w >> weight_shift) & 0xffU);
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
        std::uint16_t raw = encode_bf16_rne(block_sum0);
        residual_output[row0] = encode_bf16_rne(
            decode_bf16(residual_left[row0]) + decode_bf16(raw));
        raw = encode_bf16_rne(block_sum1);
        residual_output[row0 + 1U] = encode_bf16_rne(
            decode_bf16(residual_left[row0 + 1U]) + decode_bf16(raw));
        raw = encode_bf16_rne(block_sum2);
        residual_output[row0 + 2U] = encode_bf16_rne(
            decode_bf16(residual_left[row0 + 2U]) + decode_bf16(raw));
        raw = encode_bf16_rne(block_sum3);
        residual_output[row0 + 3U] = encode_bf16_rne(
            decode_bf16(residual_left[row0 + 3U]) + decode_bf16(raw));
      }
    }
    __syncthreads();
  }
}

// Out-of-place builder for the exact output-projection sidecar. Adjacent
// packed words remain coalesced within each source row; the aligned uint4
// store interleaves the four rows without changing the total byte extent.
__global__ __launch_bounds__(kThreads) void
fp8_w8a16_output_projection_aosoa4_pack_kernel(
    const std::uint8_t* const canonical_weights,
    uint4* const sidecar_weights) {
  const std::size_t row_quad = static_cast<std::size_t>(blockIdx.x);
  const std::uint8_t* const row0_weights =
      canonical_weights +
      (4U * row_quad) * kFp8OutputProjectionColumns;
  const auto* const row0_words =
      reinterpret_cast<const std::uint32_t*>(row0_weights);
  const auto* const row1_words =
      row0_words + kFp8OutputProjectionPackedColumns;
  const auto* const row2_words =
      row1_words + kFp8OutputProjectionPackedColumns;
  const auto* const row3_words =
      row2_words + kFp8OutputProjectionPackedColumns;
  uint4* const output =
      sidecar_weights + row_quad * kFp8OutputProjectionPackedColumns;

  for (std::size_t packed_column = static_cast<std::size_t>(threadIdx.x);
       packed_column < kFp8OutputProjectionPackedColumns;
       packed_column += kThreads) {
    const std::uint32_t packed0 =
        fp8_swizzle_packed_codes(row0_words[packed_column]);
    const std::uint32_t packed1 =
        fp8_swizzle_packed_codes(row1_words[packed_column]);
    const std::uint32_t packed2 =
        fp8_swizzle_packed_codes(row2_words[packed_column]);
    const std::uint32_t packed3 =
        fp8_swizzle_packed_codes(row3_words[packed_column]);
    output[packed_column] = uint4{packed0, packed1, packed2, packed3};
  }
}

// Test-only out-of-place builder for the exact full-attention sidecars. Q
// groups four adjacent rows at each packed column. The K/V sidecar groups two
// adjacent K rows followed by the corresponding two V rows. Every packed word
// is codebook-preswizzled once here, leaving the measured GEMV with one
// aligned uint4 load and no runtime code^(code>>5) transform.
__global__ __launch_bounds__(kThreads) void
fp8_w8a16_q_kv_aosoa4_preswizzled_pack_test_kernel(
    const std::uint8_t* const canonical_q_weights,
    const std::uint8_t* const canonical_key_weights,
    const std::uint8_t* const canonical_value_weights,
    uint4* const q_sidecar_weights, uint4* const kv_sidecar_weights) {
  const unsigned int group = blockIdx.x;
  if (group < kFp8FullAttentionQRowQuads) {
    const auto* const row0_words =
        reinterpret_cast<const std::uint32_t*>(
            canonical_q_weights +
            static_cast<std::size_t>(4U * group) * kFp8KvPairColumns);
    const auto* const row1_words =
        row0_words + kFp8FullAttentionPackedColumns;
    const auto* const row2_words =
        row1_words + kFp8FullAttentionPackedColumns;
    const auto* const row3_words =
        row2_words + kFp8FullAttentionPackedColumns;
    uint4* const output =
        q_sidecar_weights +
        static_cast<std::size_t>(group) *
            kFp8FullAttentionPackedColumns;
    for (std::size_t packed_column =
             static_cast<std::size_t>(threadIdx.x);
         packed_column < kFp8FullAttentionPackedColumns;
         packed_column += kThreads) {
      output[packed_column] = uint4{
          fp8_swizzle_packed_codes(row0_words[packed_column]),
          fp8_swizzle_packed_codes(row1_words[packed_column]),
          fp8_swizzle_packed_codes(row2_words[packed_column]),
          fp8_swizzle_packed_codes(row3_words[packed_column])};
    }
    return;
  }

  const unsigned int row_pair = group - kFp8FullAttentionQRowQuads;
  const auto* const key_row0_words =
      reinterpret_cast<const std::uint32_t*>(
          canonical_key_weights +
          static_cast<std::size_t>(2U * row_pair) * kFp8KvPairColumns);
  const auto* const key_row1_words =
      key_row0_words + kFp8FullAttentionPackedColumns;
  const auto* const value_row0_words =
      reinterpret_cast<const std::uint32_t*>(
          canonical_value_weights +
          static_cast<std::size_t>(2U * row_pair) * kFp8KvPairColumns);
  const auto* const value_row1_words =
      value_row0_words + kFp8FullAttentionPackedColumns;
  uint4* const output =
      kv_sidecar_weights +
      static_cast<std::size_t>(row_pair) *
          kFp8FullAttentionPackedColumns;
  for (std::size_t packed_column =
           static_cast<std::size_t>(threadIdx.x);
       packed_column < kFp8FullAttentionPackedColumns;
       packed_column += kThreads) {
    output[packed_column] = uint4{
        fp8_swizzle_packed_codes(key_row0_words[packed_column]),
        fp8_swizzle_packed_codes(key_row1_words[packed_column]),
        fp8_swizzle_packed_codes(value_row0_words[packed_column]),
        fp8_swizzle_packed_codes(value_row1_words[packed_column])};
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

// Test-only exact-K=5120 arithmetic twin of the production QKV/Z row-quad
// body.  The canonical FP8 layout and compiler-default global-load policy are
// unchanged.  Each lane visits exactly five K stages.  While the current
// stage is resident in registers, the four U32 words for the next stage are
// issued before the current stage's first FFMA and are not consumed until the
// following stage.  The candidate explicitly requests a rolled loop; pinned-
// build SASS must confirm that ptxas retained it instead of forming a full-K
// preload.  Results characterize the combined rolled/register-lookahead
// schedule.
__device__ __forceinline__ void
fp8_w8a16_gemv_bf16_complete_row_quad_no_tail_barrier_register_lookahead_test_body(
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

  std::size_t first_column =
      static_cast<std::size_t>(threadIdx.x) * kFp8VectorValuesPerLane;
  std::uint32_t current_weights0 = *reinterpret_cast<const std::uint32_t*>(
      row0_weights + first_column);
  std::uint32_t current_weights1 = *reinterpret_cast<const std::uint32_t*>(
      row1_weights + first_column);
  std::uint32_t current_weights2 = *reinterpret_cast<const std::uint32_t*>(
      row2_weights + first_column);
  std::uint32_t current_weights3 = *reinterpret_cast<const std::uint32_t*>(
      row3_weights + first_column);
  constexpr std::uint32_t kFp8PackedHighBits = 0x0707'0707U;

#pragma unroll 1
  for (; first_column < columns;
       first_column += kFp8VectorColumnsPerBlock) {
    const std::uint32_t swizzled_weights0 =
        current_weights0 ^
        ((current_weights0 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_weights1 =
        current_weights1 ^
        ((current_weights1 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_weights2 =
        current_weights2 ^
        ((current_weights2 >> 5U) & kFp8PackedHighBits);
    const std::uint32_t swizzled_weights3 =
        current_weights3 ^
        ((current_weights3 >> 5U) & kFp8PackedHighBits);

    std::uint32_t next_weights0 = 0U;
    std::uint32_t next_weights1 = 0U;
    std::uint32_t next_weights2 = 0U;
    std::uint32_t next_weights3 = 0U;
    const std::size_t next_column =
        first_column + kFp8VectorColumnsPerBlock;
    if (next_column < columns) {
      next_weights0 = *reinterpret_cast<const std::uint32_t*>(
          row0_weights + next_column);
      next_weights1 = *reinterpret_cast<const std::uint32_t*>(
          row1_weights + next_column);
      next_weights2 = *reinterpret_cast<const std::uint32_t*>(
          row2_weights + next_column);
      next_weights3 = *reinterpret_cast<const std::uint32_t*>(
          row3_weights + next_column);
    }

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

    current_weights0 = next_weights0;
    current_weights1 = next_weights1;
    current_weights2 = next_weights2;
    current_weights3 = next_weights3;
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

// Exact arithmetic QKV/Z row-quad body selected by production and retained by
// the test ABI under the historical symbol name. Only the four one-pass
// canonical FP8 weight words use the evict-first streaming cache operator.
// Activation loads retain the default policy, and the scalar per-tensor weight
// scale remains a kernel argument rather than a global-memory scale tensor.
__device__ __forceinline__ void
fp8_w8a16_gemv_bf16_complete_row_quad_no_tail_barrier_cs_test_body(
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
    const std::uint32_t packed_weights0 = __ldcs(
        reinterpret_cast<const unsigned int*>(row0_weights + first_column));
    const std::uint32_t packed_weights1 = __ldcs(
        reinterpret_cast<const unsigned int*>(row1_weights + first_column));
    const std::uint32_t packed_weights2 = __ldcs(
        reinterpret_cast<const unsigned int*>(row2_weights + first_column));
    const std::uint32_t packed_weights3 = __ldcs(
        reinterpret_cast<const unsigned int*>(row3_weights + first_column));
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

template <bool ScratchWillBeReused>
__device__ __forceinline__ void bf16_exact_shared_tree_reduce(
    const float sum, float* const partial, std::uint16_t* const output,
    const unsigned int row) {
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride > 1U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    partial[0U] += partial[1U];
    output[row] = encode_bf16_rne(partial[0U]);
  }
  if constexpr (ScratchWillBeReused) {
    __syncthreads();
  }
}

// Test-only exact-M1 composite candidate. Twenty-four light tail CTAs each
// accumulate two adjacent rows from A and B together so all four projections
// share one activation load. The original 256-thread shared reductions then
// execute in A-row0/A-row1/B-row0/B-row1 order and remain bitwise exact.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_kernel(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output) {
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

  if (blockIdx.x < kFp8QkvZAbFirstTailBlock ||
      blockIdx.x >=
          kFp8QkvZAbFirstTailBlock + kLinearAttentionAbRows / 2U) {
    return;
  }
  const unsigned int row0 =
      2U * (blockIdx.x - kFp8QkvZAbFirstTailBlock);
  const unsigned int row1 = row0 + 1U;
  __syncthreads();
  const std::uint16_t* const a_row0_weights =
      a_weights + static_cast<std::size_t>(row0) * columns;
  const std::uint16_t* const a_row1_weights = a_row0_weights + columns;
  const std::uint16_t* const b_row0_weights =
      b_weights + static_cast<std::size_t>(row0) * columns;
  const std::uint16_t* const b_row1_weights = b_row0_weights + columns;
  float a_sum0 = 0.0F;
  float a_sum1 = 0.0F;
  float b_sum0 = 0.0F;
  float b_sum1 = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns;
       column += kThreads) {
    const float decoded_activation = decode_bf16(activation[column]);
    a_sum0 = fmaf(decode_bf16(a_row0_weights[column]), decoded_activation,
                  a_sum0);
    a_sum1 = fmaf(decode_bf16(a_row1_weights[column]), decoded_activation,
                  a_sum1);
    b_sum0 = fmaf(decode_bf16(b_row0_weights[column]), decoded_activation,
                  b_sum0);
    b_sum1 = fmaf(decode_bf16(b_row1_weights[column]), decoded_activation,
                  b_sum1);
  }

  bf16_exact_shared_tree_reduce<true>(a_sum0, decoded_weights, a_output,
                                      row0);
  bf16_exact_shared_tree_reduce<true>(a_sum1, decoded_weights, a_output,
                                      row1);
  bf16_exact_shared_tree_reduce<true>(b_sum0, decoded_weights, b_output,
                                      row0);
  bf16_exact_shared_tree_reduce<false>(b_sum1, decoded_weights, b_output,
                                       row1);
}

// Test-only production-topology twin for the exact K=5120 one-stage direct-
// LDG register-lookahead screen.  QKV/Z use the bounded body above; launch
// geometry, phase order, scratch ping-pong, and the BF16 A/B tail are exact
// copies of production.  The public kernel and dispatch remain untouched.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_register_lookahead_test_kernel(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output) {
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
      fp8_w8a16_gemv_bf16_complete_row_quad_no_tail_barrier_register_lookahead_test_body(
          phase_weights, phase_weight_scale, activation, columns,
          4U * row_quad, phase_output, decoded_weights,
          warp_sums[scratch_slot], lane, warp);
      scratch_slot ^= 1U;
    }
  }

  if (blockIdx.x < kFp8QkvZAbFirstTailBlock ||
      blockIdx.x >=
          kFp8QkvZAbFirstTailBlock + kLinearAttentionAbRows / 2U) {
    return;
  }
  const unsigned int row0 =
      2U * (blockIdx.x - kFp8QkvZAbFirstTailBlock);
  const unsigned int row1 = row0 + 1U;
  __syncthreads();
  const std::uint16_t* const a_row0_weights =
      a_weights + static_cast<std::size_t>(row0) * columns;
  const std::uint16_t* const a_row1_weights = a_row0_weights + columns;
  const std::uint16_t* const b_row0_weights =
      b_weights + static_cast<std::size_t>(row0) * columns;
  const std::uint16_t* const b_row1_weights = b_row0_weights + columns;
  float a_sum0 = 0.0F;
  float a_sum1 = 0.0F;
  float b_sum0 = 0.0F;
  float b_sum1 = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns;
       column += kThreads) {
    const float decoded_activation = decode_bf16(activation[column]);
    a_sum0 = fmaf(decode_bf16(a_row0_weights[column]), decoded_activation,
                  a_sum0);
    a_sum1 = fmaf(decode_bf16(a_row1_weights[column]), decoded_activation,
                  a_sum1);
    b_sum0 = fmaf(decode_bf16(b_row0_weights[column]), decoded_activation,
                  b_sum0);
    b_sum1 = fmaf(decode_bf16(b_row1_weights[column]), decoded_activation,
                  b_sum1);
  }

  bf16_exact_shared_tree_reduce<true>(a_sum0, decoded_weights, a_output,
                                      row0);
  bf16_exact_shared_tree_reduce<true>(a_sum1, decoded_weights, a_output,
                                      row1);
  bf16_exact_shared_tree_reduce<true>(b_sum0, decoded_weights, b_output,
                                      row0);
  bf16_exact_shared_tree_reduce<false>(b_sum1, decoded_weights, b_output,
                                       row1);
}

// Test-only exact-M1 Decode-chain candidate. It is a distinct copy of the
// production QKV/Z/A/B composite so the public dispatch remains untouched.
// The CTA that owns each QKV row quad also consumes the four rounded BF16
// projection values, applies the width-4 causal convolution and SiLU in the
// reference operation order, and publishes the convolved values in place.
// Warp 0 is the sole projection publisher; one warp barrier makes lane 0's
// four stores visible before lanes 0..3 independently update their channels.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_causal_conv_epilogue_test_kernel(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history) {
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

      if (is_qkv && warp == 0U) {
        __syncwarp();
        if (lane < 4U) {
          const std::size_t channel =
              4U * static_cast<std::size_t>(row_quad) + lane;
          const std::size_t history_offset =
              channel * kLinearAttentionConvHistoryWidth;
          const std::size_t weight_offset =
              channel * kLinearAttentionConvKernelWidth;
          const std::uint16_t current_bits = qkv_output[channel];
          float convolution = 0.0F;
          convolution =
              fmaf(decode_bf16(history[history_offset]),
                   decode_bf16(conv_weight[weight_offset]), convolution);
          convolution =
              fmaf(decode_bf16(history[history_offset + 1U]),
                   decode_bf16(conv_weight[weight_offset + 1U]),
                   convolution);
          convolution =
              fmaf(decode_bf16(history[history_offset + 2U]),
                   decode_bf16(conv_weight[weight_offset + 2U]),
                   convolution);
          convolution =
              fmaf(decode_bf16(current_bits),
                   decode_bf16(conv_weight[weight_offset + 3U]),
                   convolution);
          qkv_output[channel] = encode_bf16_rne(
              convolution / (1.0F + expf(-convolution)));
          history[history_offset] = history[history_offset + 1U];
          history[history_offset + 1U] = history[history_offset + 2U];
          history[history_offset + 2U] = current_bits;
        }
      }
    }
  }

  if (blockIdx.x < kFp8QkvZAbFirstTailBlock ||
      blockIdx.x >=
          kFp8QkvZAbFirstTailBlock + kLinearAttentionAbRows / 2U) {
    return;
  }
  const unsigned int row0 =
      2U * (blockIdx.x - kFp8QkvZAbFirstTailBlock);
  const unsigned int row1 = row0 + 1U;
  __syncthreads();
  const std::uint16_t* const a_row0_weights =
      a_weights + static_cast<std::size_t>(row0) * columns;
  const std::uint16_t* const a_row1_weights = a_row0_weights + columns;
  const std::uint16_t* const b_row0_weights =
      b_weights + static_cast<std::size_t>(row0) * columns;
  const std::uint16_t* const b_row1_weights = b_row0_weights + columns;
  float a_sum0 = 0.0F;
  float a_sum1 = 0.0F;
  float b_sum0 = 0.0F;
  float b_sum1 = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns;
       column += kThreads) {
    const float decoded_activation = decode_bf16(activation[column]);
    a_sum0 = fmaf(decode_bf16(a_row0_weights[column]), decoded_activation,
                  a_sum0);
    a_sum1 = fmaf(decode_bf16(a_row1_weights[column]), decoded_activation,
                  a_sum1);
    b_sum0 = fmaf(decode_bf16(b_row0_weights[column]), decoded_activation,
                  b_sum0);
    b_sum1 = fmaf(decode_bf16(b_row1_weights[column]), decoded_activation,
                  b_sum1);
  }

  bf16_exact_shared_tree_reduce<true>(a_sum0, decoded_weights, a_output,
                                      row0);
  bf16_exact_shared_tree_reduce<true>(a_sum1, decoded_weights, a_output,
                                      row1);
  bf16_exact_shared_tree_reduce<true>(b_sum0, decoded_weights, b_output,
                                      row0);
  bf16_exact_shared_tree_reduce<false>(b_sum1, decoded_weights, b_output,
                                       row1);
}

// Production-selected topology, also exposed through the retained test ABI
// for Function-identity and rollback screens. QKV/Z use evict-first streaming
// loads for canonical FP8 weight words. Activation and BF16 A/B weight loads
// retain the compiler-default cache policy.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_cs_test_kernel(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output) {
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
      fp8_w8a16_gemv_bf16_complete_row_quad_no_tail_barrier_cs_test_body(
          phase_weights, phase_weight_scale, activation, columns,
          4U * row_quad, phase_output, decoded_weights,
          warp_sums[scratch_slot], lane, warp);
      scratch_slot ^= 1U;
    }
  }

  if (blockIdx.x < kFp8QkvZAbFirstTailBlock ||
      blockIdx.x >=
          kFp8QkvZAbFirstTailBlock + kLinearAttentionAbRows / 2U) {
    return;
  }
  const unsigned int row0 =
      2U * (blockIdx.x - kFp8QkvZAbFirstTailBlock);
  const unsigned int row1 = row0 + 1U;
  __syncthreads();
  const std::uint16_t* const a_row0_weights =
      a_weights + static_cast<std::size_t>(row0) * columns;
  const std::uint16_t* const a_row1_weights = a_row0_weights + columns;
  const std::uint16_t* const b_row0_weights =
      b_weights + static_cast<std::size_t>(row0) * columns;
  const std::uint16_t* const b_row1_weights = b_row0_weights + columns;
  float a_sum0 = 0.0F;
  float a_sum1 = 0.0F;
  float b_sum0 = 0.0F;
  float b_sum1 = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns;
       column += kThreads) {
    const float decoded_activation = decode_bf16(activation[column]);
    a_sum0 = fmaf(decode_bf16(a_row0_weights[column]), decoded_activation,
                  a_sum0);
    a_sum1 = fmaf(decode_bf16(a_row1_weights[column]), decoded_activation,
                  a_sum1);
    b_sum0 = fmaf(decode_bf16(b_row0_weights[column]), decoded_activation,
                  b_sum0);
    b_sum1 = fmaf(decode_bf16(b_row1_weights[column]), decoded_activation,
                  b_sum1);
  }

  bf16_exact_shared_tree_reduce<true>(a_sum0, decoded_weights, a_output,
                                      row0);
  bf16_exact_shared_tree_reduce<true>(a_sum1, decoded_weights, a_output,
                                      row1);
  bf16_exact_shared_tree_reduce<true>(b_sum0, decoded_weights, b_output,
                                      row0);
  bf16_exact_shared_tree_reduce<false>(b_sum1, decoded_weights, b_output,
                                       row1);
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
// TailBarrier=true is the production predecessor. TailBarrier=false is used
// only by the double-slot kernel's final K/V body; that scratch slot is never
// reused before CTA exit, so the trailing barrier is unnecessary.
template <bool TailBarrier>
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
  if constexpr (TailBarrier) {
    __syncthreads();
  }
}

// Test-only AoSoA4/preswizzled twin of the two production arithmetic bodies.
// ProjectionPair=false is one Q row quad; ProjectionPair=true is one K/V row
// pair. The accumulator updates, warp/block reductions, scaling, and BF16-RNE
// publication order are unchanged. Only four canonical U32 loads plus their
// runtime swizzles become one aligned uint4 sidecar load.
template <bool ProjectionPair, bool DualHalfCta = false>
__device__ __forceinline__ void
fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_no_tail_test_body(
    const uint4* const sidecar_weights, const float first_weight_scale,
    const float second_weight_scale,
    const std::uint16_t* const activation,
    const std::size_t sidecar_group, const std::size_t row0,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output,
    const float* const decoded_weights,
    float (*const warp_sums)[kWarpsPerBlock]) {
  float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
  float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

  const unsigned int logical_tid =
      DualHalfCta ? threadIdx.x & (kThreads - 1U) : threadIdx.x;
  for (std::size_t first_column =
           static_cast<std::size_t>(logical_tid) *
           kFp8VectorValuesPerLane;
       first_column < kFp8KvPairColumns;
       first_column += kFp8VectorColumnsPerBlock) {
    const std::size_t packed_column =
        first_column / kFp8VectorValuesPerLane;
    const uint4 packed_weights =
        sidecar_weights[sidecar_group * kFp8FullAttentionPackedColumns +
                        packed_column];
    const std::uint64_t packed_activation =
        *reinterpret_cast<const std::uint64_t*>(activation + first_column);
#pragma unroll
    for (unsigned int value = 0U; value < kFp8VectorValuesPerLane;
         ++value) {
      const unsigned int weight_shift = value * 8U;
      const std::uint8_t encoded_weight0 = static_cast<std::uint8_t>(
          (packed_weights.x >> weight_shift) & 0xffU);
      const std::uint8_t encoded_weight1 = static_cast<std::uint8_t>(
          (packed_weights.y >> weight_shift) & 0xffU);
      const std::uint8_t encoded_weight2 = static_cast<std::uint8_t>(
          (packed_weights.z >> weight_shift) & 0xffU);
      const std::uint8_t encoded_weight3 = static_cast<std::uint8_t>(
          (packed_weights.w >> weight_shift) & 0xffU);
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
  const unsigned int warp_sum_lane =
      threadIdx.x & (kWarpSize - 1U);
  if (warp_sum_lane == 0U) {
    unsigned int warp_sum_warp = threadIdx.x / kWarpSize;
    if constexpr (DualHalfCta) {
      warp_sum_warp &= kWarpsPerBlock - 1U;
    }
    warp_sums[0U][warp_sum_warp] = sum0;
    warp_sums[1U][warp_sum_warp] = sum1;
    warp_sums[2U][warp_sum_warp] = sum2;
    warp_sums[3U][warp_sum_warp] = sum3;
  }
  __syncthreads();
  unsigned int block_sum_warp = threadIdx.x / kWarpSize;
  if constexpr (DualHalfCta) {
    block_sum_warp &= kWarpsPerBlock - 1U;
  }
  if (block_sum_warp == 0U) {
    const unsigned int block_sum_lane =
        threadIdx.x & (kWarpSize - 1U);
    float block_sum0 =
        block_sum_lane < kWarpsPerBlock
            ? warp_sums[0U][block_sum_lane]
            : 0.0F;
    float block_sum1 =
        block_sum_lane < kWarpsPerBlock
            ? warp_sums[1U][block_sum_lane]
            : 0.0F;
    float block_sum2 =
        block_sum_lane < kWarpsPerBlock
            ? warp_sums[2U][block_sum_lane]
            : 0.0F;
    float block_sum3 =
        block_sum_lane < kWarpsPerBlock
            ? warp_sums[3U][block_sum_lane]
            : 0.0F;
    if constexpr (ProjectionPair) {
      block_sum0 = warp_sum(block_sum0) * first_weight_scale;
      block_sum1 = warp_sum(block_sum1) * first_weight_scale;
      block_sum2 = warp_sum(block_sum2) * second_weight_scale;
      block_sum3 = warp_sum(block_sum3) * second_weight_scale;
      if (block_sum_lane == 0U) {
        first_output[row0] = encode_bf16_rne(block_sum0);
        first_output[row0 + 1U] = encode_bf16_rne(block_sum1);
        second_output[row0] = encode_bf16_rne(block_sum2);
        second_output[row0 + 1U] = encode_bf16_rne(block_sum3);
      }
    } else {
      block_sum0 = warp_sum(block_sum0) * first_weight_scale;
      block_sum1 = warp_sum(block_sum1) * first_weight_scale;
      block_sum2 = warp_sum(block_sum2) * first_weight_scale;
      block_sum3 = warp_sum(block_sum3) * first_weight_scale;
      if (block_sum_lane == 0U) {
        first_output[row0] = encode_bf16_rne(block_sum0);
        first_output[row0 + 1U] = encode_bf16_rne(block_sum1);
        first_output[row0 + 2U] = encode_bf16_rne(block_sum2);
        first_output[row0 + 3U] = encode_bf16_rne(block_sum3);
      }
    }
  }
}

// Test-only predecessor for the production full-attention fusion. It retains
// the original single reduction-scratch slot and the tail block barrier after
// every Q and K/V body so same-binary tests can measure the promoted kernel
// against the exact prior implementation.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_q_kv_two_phase_tail_barrier_test_kernel(
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
    fp8_w8a16_gemv_bf16_complete_projection_pair_row_pair_body<true>(
        key_weights, key_weight_scale, value_weights, value_weight_scale,
        activation, kFp8KvPairColumns, row0, key_output, value_output,
        decoded_weights, warp_sums, lane, warp);
  }
}

// Production topology-preserving full-attention Q + K/V fusion. Consecutive Q
// row quads, plus the ordered Q-to-K/V handoff in blocks 1024..1535, alternate
// reduction-scratch slots. The producer barrier in body i+1 cannot release
// until warp 0 has completed body i. Every CTA executes at most two bodies,
// so neither slot is reused before kernel completion.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_q_kv_two_phase_reduction_scratch_ping_pong_kernel(
    const std::uint8_t* const q_weights, const float q_weight_scale,
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation,
    std::uint16_t* const q_output, std::uint16_t* const key_output,
    std::uint16_t* const value_output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[2U][4U][kWarpsPerBlock];
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  constexpr unsigned int kQRowQuads =
      static_cast<unsigned int>(kFp8FullAttentionQRows / 4U);
  unsigned int scratch_slot = 0U;
  for (unsigned int row_quad = blockIdx.x; row_quad < kQRowQuads;
       row_quad += kFp8FullAttentionBlocks) {
    fp8_w8a16_gemv_bf16_complete_row_quad_no_tail_barrier_body(
        q_weights, q_weight_scale, activation, kFp8KvPairColumns,
        4U * row_quad, q_output, decoded_weights, warp_sums[scratch_slot],
        lane, warp);
    scratch_slot ^= 1U;
  }

  if (blockIdx.x >= kFp8FullAttentionKvFirstBlock &&
      blockIdx.x <
          kFp8FullAttentionKvFirstBlock + kFp8FullAttentionKvBlocks) {
    const std::size_t row0 =
        2U * static_cast<std::size_t>(blockIdx.x -
                                     kFp8FullAttentionKvFirstBlock);
    fp8_w8a16_gemv_bf16_complete_projection_pair_row_pair_body<false>(
        key_weights, key_weight_scale, value_weights, value_weight_scale,
        activation, kFp8KvPairColumns, row0, key_output, value_output,
        decoded_weights, warp_sums[scratch_slot], lane, warp);
  }
}

// Test-only sidecar candidate for the production full-attention fusion. The
// 2,048x256 topology, two reduction-scratch slots, logical-body order, and
// Q-to-K/V handoff are identical to production. Q and combined K/V sidecars
// supply the already-preswizzled four-word group with one uint4 load.
__global__ __launch_bounds__(kThreads, 4) void
fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_test_kernel(
    const uint4* const q_sidecar_weights, const float q_weight_scale,
    const uint4* const kv_sidecar_weights, const float key_weight_scale,
    const float value_weight_scale,
    const std::uint16_t* const activation,
    std::uint16_t* const q_output, std::uint16_t* const key_output,
    std::uint16_t* const value_output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[2U][4U][kWarpsPerBlock];
  const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
  decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  __syncthreads();

  unsigned int scratch_slot = 0U;
  for (unsigned int row_quad = blockIdx.x;
       row_quad < kFp8FullAttentionQRowQuads;
       row_quad += kFp8FullAttentionBlocks) {
    fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_no_tail_test_body<false>(
        q_sidecar_weights, q_weight_scale, q_weight_scale, activation,
        row_quad, 4U * static_cast<std::size_t>(row_quad), q_output,
        q_output, decoded_weights, warp_sums[scratch_slot]);
    scratch_slot ^= 1U;
  }

  if (blockIdx.x >= kFp8FullAttentionKvFirstBlock &&
      blockIdx.x <
          kFp8FullAttentionKvFirstBlock + kFp8FullAttentionKvBlocks) {
    const unsigned int row_pair =
        blockIdx.x - kFp8FullAttentionKvFirstBlock;
    const std::size_t row0 = 2U * static_cast<std::size_t>(row_pair);
    fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_no_tail_test_body<true>(
        kv_sidecar_weights, key_weight_scale, value_weight_scale,
        activation, row_pair, row0, key_output, value_output,
        decoded_weights, warp_sums[scratch_slot]);
  }
}

// Test-only 512-thread coarsening of the sidecar candidate. Each aligned
// 256-thread half is one logical worker from the 2,048-worker production map.
// The even partition boundaries guarantee that both halves execute the same
// number of logical bodies, so every body-wide barrier remains CTA-safe.
__global__ __launch_bounds__(kFp8FullAttentionCta512Threads, 2) void
fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_cta512_test_kernel(
    const uint4* const q_sidecar_weights, const float q_weight_scale,
    const uint4* const kv_sidecar_weights, const float key_weight_scale,
    const float value_weight_scale,
    const std::uint16_t* const activation,
    std::uint16_t* const q_output, std::uint16_t* const key_output,
    std::uint16_t* const value_output) {
  __shared__ float decoded_weights[kFp8EncodedValueCount];
  __shared__ float warp_sums[2U][2U][4U][kWarpsPerBlock];
  if (threadIdx.x < kThreads) {
    const std::uint8_t code = static_cast<std::uint8_t>(threadIdx.x);
    decoded_weights[fp8_swizzled_codebook_slot(code)] = decode_e4m3fn(code);
  }
  __syncthreads();

  const unsigned int half = threadIdx.x / kThreads;
  const unsigned int logical_worker = 2U * blockIdx.x + half;
  unsigned int scratch_slot = 0U;
  for (unsigned int row_quad = logical_worker;
       row_quad < kFp8FullAttentionQRowQuads;
       row_quad += kFp8FullAttentionBlocks) {
    fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_no_tail_test_body<
        false, true>(q_sidecar_weights, q_weight_scale, q_weight_scale,
                     activation, row_quad,
                     4U * static_cast<std::size_t>(row_quad), q_output,
                     q_output, decoded_weights,
                     warp_sums[half][scratch_slot]);
    scratch_slot ^= 1U;
  }

  if (logical_worker >= kFp8FullAttentionKvFirstBlock &&
      logical_worker <
          kFp8FullAttentionKvFirstBlock + kFp8FullAttentionKvBlocks) {
    const unsigned int row_pair =
        logical_worker - kFp8FullAttentionKvFirstBlock;
    const std::size_t row0 = 2U * static_cast<std::size_t>(row_pair);
    fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_no_tail_test_body<true,
                                                                  true>(
        kv_sidecar_weights, key_weight_scale, value_weight_scale,
        activation, row_pair, row0, key_output, value_output,
        decoded_weights, warp_sums[half][scratch_slot]);
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

// Preserved fixed-M32 single-resident-A predecessor. The test-only direct
// entry and resource query keep it available for same-cubin regression and
// performance comparisons after production moves to dual-resident A.
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

// Fixed-M32 production dual-resident-A kernel. Both 16-token activation panels
// remain resident for the full K64 stage, allowing each decoded B fragment to
// feed two independent accumulator chains without changing either chain's
// K/MMA order. The complete C[32,N128] tile shares storage with B.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
__global__ __launch_bounds__(kThreads, 5) void
fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel(
    const std::uint8_t* const weights, const float weight_scale,
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
    shared_activations[(token + kPanelTokenCount) *
                           kSharedActivationWordsPerToken +
                       activation_word] =
        *reinterpret_cast<const std::uint64_t*>(
            activations +
            static_cast<std::size_t>(token + kPanelTokenCount) * kColumns +
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
    // Protect both resident A panels and decoded B before the next stage, and
    // protect the final B-to-C union lifetime transition.
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

// Construct eight exact E2M1 BF16 values without a shared lookup. PRMT treats
// the low 16 bits of its selector as four byte selectors. Values 0..7 select
// one byte from the two lookup immediates; values 8..15 replicate the sign bit
// of the selected source byte. Two four-nibble halves therefore need only
// byte permutations and bitwise composition, including signed zero (code 8).
__device__ __forceinline__ uint4 decode_e2m1x8_to_bf16x8_prmt(
    const std::uint32_t packed) {
  constexpr std::uint32_t kLowBytes0To3 = 0xc080'0000U;
  constexpr std::uint32_t kLowBytes4To7 = 0xc080'4000U;
  constexpr std::uint32_t kHighBytes0To3 = 0x3f3f'3f00U;
  constexpr std::uint32_t kHighBytes4To7 = 0x4040'4040U;
  constexpr std::uint32_t kMagnitudeSelectorMask = 0x7777U;
  constexpr std::uint32_t kSignByteMask = 0x8080'8080U;
  constexpr std::uint32_t kFirstFourSignSelector = 0xd9c8U;
  constexpr std::uint32_t kSecondFourSignSelector = 0xfbeaU;
  constexpr std::uint32_t kFirstPairInterleave = 0x5140U;
  constexpr std::uint32_t kSecondPairInterleave = 0x7362U;

  const std::uint32_t shifted_signs = packed << 4U;
  const std::uint32_t first_selector =
      packed & kMagnitudeSelectorMask;
  const std::uint32_t second_selector =
      (packed >> 16U) & kMagnitudeSelectorMask;
  const std::uint32_t first_low = __byte_perm(
      kLowBytes0To3, kLowBytes4To7, first_selector);
  const std::uint32_t second_low = __byte_perm(
      kLowBytes0To3, kLowBytes4To7, second_selector);
  const std::uint32_t first_signs =
      __byte_perm(shifted_signs, packed, kFirstFourSignSelector) &
      kSignByteMask;
  const std::uint32_t second_signs =
      __byte_perm(shifted_signs, packed, kSecondFourSignSelector) &
      kSignByteMask;
  const std::uint32_t first_high =
      __byte_perm(kHighBytes0To3, kHighBytes4To7, first_selector) |
      first_signs;
  const std::uint32_t second_high =
      __byte_perm(kHighBytes0To3, kHighBytes4To7, second_selector) |
      second_signs;

  uint4 result{};
  result.x = __byte_perm(first_low, first_high, kFirstPairInterleave);
  result.y = __byte_perm(first_low, first_high, kSecondPairInterleave);
  result.z = __byte_perm(second_low, second_high, kFirstPairInterleave);
  result.w = __byte_perm(second_low, second_high, kSecondPairInterleave);
  return result;
}

__device__ __forceinline__ uint4
decode_nvfp4x8_to_bf16x8_table_free_vector(
    const std::uint32_t packed, const std::uint16_t decoded_scale) {
  uint4 result = decode_e2m1x8_to_bf16x8_prmt(packed);
  result.x = multiply_bf16x2_bits(result.x, decoded_scale);
  result.y = multiply_bf16x2_bits(result.y, decoded_scale);
  result.z = multiply_bf16x2_bits(result.z, decoded_scale);
  result.w = multiply_bf16x2_bits(result.w, decoded_scale);
  return result;
}

// Exhaustive semantic gate for the table-free pair constructor. Each thread
// covers one E4M3 scale code and one packed byte, including signed zero and
// the E4M3FN NaN encodings, while retaining the production BF16x2 multiply.
__global__ void nvfp4_table_free_e2m1_exhaustive_kernel(
    std::uint32_t* const candidate, std::uint32_t* const reference) {
  constexpr unsigned int kPackedByteCount = 256U;
  constexpr unsigned int kPackedBytePositions = 4U;
  constexpr unsigned int kCombinationCount =
      kFp8EncodedValueCount * kPackedByteCount * kPackedBytePositions;
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kCombinationCount) {
    return;
  }
  const unsigned int pair_index = index / kPackedBytePositions;
  const unsigned int byte_position = index % kPackedBytePositions;
  const std::uint8_t scale_code =
      static_cast<std::uint8_t>(pair_index / kPackedByteCount);
  const std::uint8_t packed =
      static_cast<std::uint8_t>(pair_index % kPackedByteCount);
  const std::uint16_t decoded_scale =
      encode_bf16_rne(decode_e4m3fn(scale_code));
  const uint4 decoded = decode_e2m1x8_to_bf16x8_prmt(
      static_cast<std::uint32_t>(packed) << (byte_position * 8U));
  const std::uint32_t decoded_pair =
      byte_position == 0U   ? decoded.x
      : byte_position == 1U ? decoded.y
      : byte_position == 2U ? decoded.z
                            : decoded.w;
  candidate[index] = multiply_bf16x2_bits(decoded_pair, decoded_scale);

  const std::uint16_t low =
      encode_bf16_rne(decode_e2m1(packed & 0x0fU));
  const std::uint16_t high =
      encode_bf16_rne(decode_e2m1(packed >> 4U));
  const std::uint32_t reference_pair =
      static_cast<std::uint32_t>(low) |
      (static_cast<std::uint32_t>(high) << 16U);
  reference[index] = multiply_bf16x2_bits(reference_pair, decoded_scale);
}

// SM80+ raw-weight staging primitives. The exact-M32 gate/up production path
// and the retained M17-M31 test probe keep one 16-byte cell per thread in
// shared memory. Each thread waits for its prior async copy, consumes that
// cell into registers, and only then overwrites it with the next K64 stage.
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

template <bool kFactorized, bool kTableFreeE2M1 = false>
struct NvFp4M32ProductLookupStorage;

template <>
struct alignas(32) NvFp4M32ProductLookupStorage<false, false> {
  std::uint32_t product_words[kFp8EncodedValueCount *
                              kNvFp4EncodedValueCount / 2U];
};

template <>
struct alignas(32) NvFp4M32ProductLookupStorage<true, false> {
  std::uint32_t e2m1_byte_pairs[kFp8EncodedValueCount];
  std::uint16_t scale_values[kFp8EncodedValueCount];
};

// Table-free production specialization. The E4M3 scale codebook remains in
// shared memory while exact E2M1 BF16 pairs are constructed with PRMT in the
// stage loop, removing the conflict-prone 1 KiB pair table.
template <>
struct alignas(32) NvFp4M32ProductLookupStorage<true, true> {
  std::uint16_t scale_values[kFp8EncodedValueCount];
};

static_assert(sizeof(NvFp4M32ProductLookupStorage<false, false>) == 8'192U);
static_assert(sizeof(NvFp4M32ProductLookupStorage<true, false>) == 1'536U);
static_assert(sizeof(NvFp4M32ProductLookupStorage<true, true>) == 512U);

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

// K256 scale-window family of the fixed-M32 K64 kernel. It supports the full
// product table, the preserved factorized E2M1-pair baseline, and the exact
// table-free E2M1 production specialization while retaining E4M3 scales.
// Eight adjacent U16 scale words for each output row are cooperatively loaded
// into the otherwise-unused [64,72) padding of the K64 shared B tile. Four
// consecutive K64 stages then reuse that window, turning the strided per-row
// global scale loads into 16-byte row segments without changing decoded B or
// the WMMA accumulation order.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U,
          bool kFactorizedProductLookup = false,
          bool kVectorizedDecodedStore = false,
          unsigned int kValidTokenCount = 32U,
          bool kTableFreeE2M1 = false>
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
  static_assert(!kTableFreeE2M1 ||
                (kFactorizedProductLookup && kVectorizedDecodedStore));
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
  __shared__ NvFp4M32ProductLookupStorage<kFactorizedProductLookup,
                                          kTableFreeE2M1>
      product_lookup;
  __shared__ __align__(32) std::uint64_t
      shared_activations[kTokenCount * kSharedActivationWordsPerToken];
  __shared__ BOrCStorage b_or_c;

  namespace wmma = nvcuda::wmma;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;

  if constexpr (kFactorizedProductLookup) {
    const std::uint8_t encoded = static_cast<std::uint8_t>(thread);
    if constexpr (!kTableFreeE2M1) {
      const std::uint16_t low =
          encode_bf16_rne(decode_e2m1(encoded & 0x0fU));
      const std::uint16_t high =
          encode_bf16_rne(decode_e2m1(encoded >> 4U));
      product_lookup.e2m1_byte_pairs[thread] =
          static_cast<std::uint32_t>(low) |
          (static_cast<std::uint32_t>(high) << 16U);
    }
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
          if constexpr (kTableFreeE2M1) {
            decoded_vectors[0] =
                decode_nvfp4x8_to_bf16x8_table_free_vector(packed.x,
                                                            decoded_scale0);
            decoded_vectors[1] =
                decode_nvfp4x8_to_bf16x8_table_free_vector(packed.y,
                                                            decoded_scale0);
            decoded_vectors[2] =
                decode_nvfp4x8_to_bf16x8_table_free_vector(packed.z,
                                                            decoded_scale1);
            decoded_vectors[3] =
                decode_nvfp4x8_to_bf16x8_table_free_vector(packed.w,
                                                            decoded_scale1);
          } else {
            decoded_vectors[0] = decode_nvfp4x8_to_bf16x8_factorized_vector(
                packed.x, product_lookup.e2m1_byte_pairs, decoded_scale0);
            decoded_vectors[1] = decode_nvfp4x8_to_bf16x8_factorized_vector(
                packed.y, product_lookup.e2m1_byte_pairs, decoded_scale0);
            decoded_vectors[2] = decode_nvfp4x8_to_bf16x8_factorized_vector(
                packed.z, product_lookup.e2m1_byte_pairs, decoded_scale1);
            decoded_vectors[3] = decode_nvfp4x8_to_bf16x8_factorized_vector(
                packed.w, product_lookup.e2m1_byte_pairs, decoded_scale1);
          }
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
// Keeping this as a separate kernel isolates the validated fixed-M18 and
// exact-M32 production specializations while serving one cubin instance per
// checkpoint shape for M=17 and M=19..31.
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

// Production exact-M32 gate/up specialization. This carries the promoted
// K256 scale window and table-free E2M1 decode unchanged, while replacing each
// thread's synchronous 16-byte packed-weight load with one single-slot
// cp.async cell. The slot is exactly 4 KiB per CTA; there is no second raw
// weight slot and no extra A or decoded-B buffer.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
__global__ __launch_bounds__(kThreads, 5) void
nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_table_free_raw_weight_cp_async_kernel(
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
  static_assert(kRows == 17'408U && kColumns == 5'120U);
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
  __shared__ NvFp4M32ProductLookupStorage<true, true> product_lookup;
  __shared__ __align__(32) std::uint64_t
      shared_activations[kTokenCount * kSharedActivationWordsPerToken];
  __shared__ BOrCStorage b_or_c;
  __shared__ __align__(16) uint4 shared_raw_weights[kThreads];

  namespace wmma = nvcuda::wmma;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / kWarpSize;

  const std::uint8_t encoded = static_cast<std::uint8_t>(thread);
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
          *reinterpret_cast<const std::uint64_t*>(
              activations +
              static_cast<std::size_t>(second_token) * kColumns + first_k +
              second_activation_word * kBf16ValuesPerActivationWord);

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
      decoded_vectors[0] = decode_nvfp4x8_to_bf16x8_table_free_vector(
          packed.x, decoded_scale0);
      decoded_vectors[1] = decode_nvfp4x8_to_bf16x8_table_free_vector(
          packed.y, decoded_scale0);
      decoded_vectors[2] = decode_nvfp4x8_to_bf16x8_table_free_vector(
          packed.z, decoded_scale1);
      decoded_vectors[3] = decode_nvfp4x8_to_bf16x8_table_free_vector(
          packed.w, decoded_scale1);

      // Move the following packed tile only after the current uint4 is owned
      // in registers. The existing CTA barrier still protects decoded B/A;
      // it also leaves this asynchronous transfer in flight during WMMA.
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
    output[static_cast<std::size_t>(token) * kRows + first_output_column +
           local_column] =
        encode_bf16_rne(b_or_c.output[index] * weight_scale_2);
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

// Exact-shape 512-thread counterpart of the predecessor gate/up phase. The
// 32-CTA x 16-warp topology preserves the kernel's 512 global
// warps and exact 2,048-row stride while halving the number of CTAs that
// repeat residual/RMSNorm setup.
template <bool CtaLocalOutput = false>
__device__ __forceinline__ void
nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_coarsened_512_phase(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    std::uint16_t* const output, const ulonglong2* const staged_activation,
    const float* const decoded_weights, const float* const decoded_scales,
    const unsigned int lane, const unsigned int warp) {
  constexpr std::uint32_t kRows = 17'408U;
  constexpr std::uint32_t kColumns = 5'120U;
  constexpr std::uint32_t kBlocks = 32U;
  constexpr std::uint32_t kWarps = 16U;
  constexpr std::uint32_t kPackedColumns =
      kColumns / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleColumns = kColumns / kNvFp4GroupSize;
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4GroupSize;
  constexpr std::uint32_t kRowStride = kBlocks * kWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kWarps * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;
  static_assert(kBlocks * kWarps ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock);

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarps + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
  std::uint32_t local_row0 = warp * 4U;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < kRows;
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
      output[CtaLocalOutput ? local_row0 : row0] = encode_bf16_rne(sum);
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[CtaLocalOutput ? local_row0 + 1U : row1] =
          encode_bf16_rne(sum);
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[CtaLocalOutput ? local_row0 + 2U : row2] =
          encode_bf16_rne(sum);
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[CtaLocalOutput ? local_row0 + 3U : row3] =
          encode_bf16_rne(sum);
    }
    packed_row_offset += kPackedRowStride;
    scale_row_offset += kScaleRowStride;
    local_row0 += kRowsPerCtaPerStride;
  }
}

enum class NvFp4TestCachePolicy : unsigned int {
  kCacheGlobal,
  kStreaming,
};

template <NvFp4TestCachePolicy Policy>
__device__ __forceinline__ std::uint8_t nvfp4_test_cache_load_u8(
    const std::uint8_t* const address) {
  if constexpr (Policy == NvFp4TestCachePolicy::kCacheGlobal) {
    return __ldcg(address);
  }
  return __ldcs(address);
}

template <NvFp4TestCachePolicy Policy>
__device__ __forceinline__ std::uint32_t nvfp4_test_cache_load_u32(
    const std::uint8_t* const address) {
  const auto word = reinterpret_cast<const unsigned int*>(address);
  if constexpr (Policy == NvFp4TestCachePolicy::kCacheGlobal) {
    return __ldcg(word);
  }
  return __ldcs(word);
}

// Test-only cache-policy twin of the exact M1 lm-head production kernel.
// Keep the production template byte-for-byte isolated: this body changes only
// the one-pass packed-weight and block-scale loads selected by Policy. The
// activation copy deliberately retains the compiler-default cache policy.
template <std::size_t Rows, std::size_t Columns,
          NvFp4TestCachePolicy Policy>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_gemv_bf16_lm_head_activation_staged_cache_policy_test_kernel(
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
          static_cast<std::uint32_t>(
              nvfp4_test_cache_load_u8<Policy>(row0_scales + scale_column)) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<Policy>(row1_scales + scale_column))
           << 8U) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<Policy>(row2_scales + scale_column))
           << 16U) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<Policy>(row3_scales + scale_column))
           << 24U);
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
        const std::uint32_t packed0 = nvfp4_test_cache_load_u32<Policy>(
            row0_weights + phase_packed_column);
        const std::uint32_t packed1 = nvfp4_test_cache_load_u32<Policy>(
            row1_weights + phase_packed_column);
        const std::uint32_t packed2 = nvfp4_test_cache_load_u32<Policy>(
            row2_weights + phase_packed_column);
        const std::uint32_t packed3 = nvfp4_test_cache_load_u32<Policy>(
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

// Test-only Decode scale sidecar. One K=512 tile contains 32 E4M3 scale
// codes for each of four adjacent rows. The 128 codes are stored in the
// exact adjacent-lane-pair consumption order as 128 packed six-bit deltas:
//   [lane-pair][phase][row-within-quad].
// That is 96 bytes, or exactly 24 aligned words, per row-quad/K512 tile.
// The first 24 lanes load one word each, so the warp requests three aligned
// 32-byte sectors instead of four sectors of canonical U8 scale codes. The
// tensor-specific base reconstructs the original E4M3 byte before the
// existing shared codebook lookup; projection arithmetic is unchanged.
constexpr unsigned int kNvFp4Scale6BitsPerCode = 6U;
constexpr unsigned int kNvFp4Scale6CodesPerLanePair = 8U;
constexpr unsigned int kNvFp4Scale6CodesPerRowQuadTile = 128U;
constexpr unsigned int kNvFp4Scale6WordsPerRowQuadTile = 24U;
constexpr unsigned int kNvFp4Scale6ColumnsPerTile = 512U;
constexpr unsigned int kNvFp4Scale6MaximumBase = 192U;
static_assert(kNvFp4Scale6CodesPerRowQuadTile * kNvFp4Scale6BitsPerCode ==
              kNvFp4Scale6WordsPerRowQuadTile * 32U);

__device__ __forceinline__ std::uint32_t
nvfp4_scale6_raw_scale_codes_for_lane(
    const std::uint8_t* const scale6_sidecar,
    const unsigned int scale_base, const std::uint32_t row0,
    const std::uint32_t packed_column, const std::uint32_t packed_columns,
    const unsigned int lane) {
  constexpr unsigned int kFullWarpMask = 0xffff'ffffU;
  constexpr std::uint32_t kPackedColumnsPerScale6Tile =
      kNvFp4Scale6ColumnsPerTile / kNvFp4ValuesPerByte;
  const std::uint32_t tiles_per_row_quad =
      packed_columns / kPackedColumnsPerScale6Tile;
  const std::uint32_t tile_in_row =
      packed_column / kPackedColumnsPerScale6Tile;
  const std::uint32_t tile_index =
      (row0 / 4U) * tiles_per_row_quad + tile_in_row;
  const auto tile_words = reinterpret_cast<const unsigned int*>(
      scale6_sidecar +
      static_cast<std::size_t>(tile_index) *
          kNvFp4Scale6WordsPerRowQuadTile * sizeof(std::uint32_t));

  std::uint32_t resident_word = 0U;
  if (lane < kNvFp4Scale6WordsPerRowQuadTile) {
    resident_word = __ldcs(tile_words + lane);
  }

  const unsigned int lane_pair = lane >> 1U;
  const unsigned int phase = lane & 1U;
  const unsigned int first_code =
      lane_pair * kNvFp4Scale6CodesPerLanePair + phase * 4U;
  const unsigned int first_bit = first_code * kNvFp4Scale6BitsPerCode;
  const unsigned int first_word = first_bit >> 5U;
  const unsigned int shift = first_bit & 31U;
  const std::uint32_t low_word =
      __shfl_sync(kFullWarpMask, resident_word, first_word);
  const std::uint32_t high_word =
      __shfl_sync(kFullWarpMask, resident_word, first_word + 1U);
  std::uint32_t packed_deltas = low_word >> shift;
  if (shift > 8U) {
    packed_deltas |= high_word << (32U - shift);
  }
  packed_deltas &= 0x00ff'ffffU;

  const std::uint32_t delta_bytes =
      (packed_deltas & 0x3fU) |
      (((packed_deltas >> 6U) & 0x3fU) << 8U) |
      (((packed_deltas >> 12U) & 0x3fU) << 16U) |
      (((packed_deltas >> 18U) & 0x3fU) << 24U);
  return delta_bytes + scale_base * 0x0101'0101U;
}

template <std::uint32_t Rows, std::uint32_t Columns,
          bool CtaLocalOutput>
__device__ __forceinline__ void
nvfp4_w4a16_scale6_activation_staged_phase(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const scale6_sidecar,
    const unsigned int scale_base, const float weight_scale_2,
    std::uint16_t* const output, const ulonglong2* const staged_activation,
    const float* const decoded_weights, const float* const decoded_scales,
    const unsigned int lane, const unsigned int warp) {
  constexpr std::uint32_t kBlocks = 32U;
  constexpr std::uint32_t kWarps = 16U;
  constexpr std::uint32_t kPackedColumns =
      Columns / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kRowStride = kBlocks * kWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kWarps * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  static_assert((Rows % 4U) == 0U);
  static_assert((Columns % kNvFp4Scale6ColumnsPerTile) == 0U);
  static_assert(kBlocks * kWarps ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock);

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarps + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t local_row0 = warp * 4U;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < Rows;
       row0 += kRowStride) {
    const auto row0_weights = packed_weights + packed_row_offset;
    const auto row1_weights = row0_weights + kPackedColumns;
    const auto row2_weights = row1_weights + kPackedColumns;
    const auto row3_weights = row2_weights + kPackedColumns;
    float accumulators0[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators1[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators2[4]{0.0F, 0.0F, 0.0F, 0.0F};
    float accumulators3[4]{0.0F, 0.0F, 0.0F, 0.0F};

#pragma unroll 1
    for (std::uint32_t packed_column =
             lane * kNvFp4VectorPackedBytesPerLane;
         packed_column < kPackedColumns;
         packed_column += 2U * kPackedIterationStride) {
      std::uint32_t local_raw_scale_codes =
          nvfp4_scale6_raw_scale_codes_for_lane(
              scale6_sidecar, scale_base, row0, packed_column,
              kPackedColumns, lane);
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
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
                row0_weights + phase_packed_column);
        const std::uint32_t packed1 =
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
                row1_weights + phase_packed_column);
        const std::uint32_t packed2 =
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
                row2_weights + phase_packed_column);
        const std::uint32_t packed3 =
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
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
      output[CtaLocalOutput ? local_row0 : row0] = encode_bf16_rne(sum);
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[CtaLocalOutput ? local_row0 + 1U : row0 + 1U] =
          encode_bf16_rne(sum);
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[CtaLocalOutput ? local_row0 + 2U : row0 + 2U] =
          encode_bf16_rne(sum);
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[CtaLocalOutput ? local_row0 + 3U : row0 + 3U] =
          encode_bf16_rne(sum);
    }
    packed_row_offset += kPackedRowStride;
    local_row0 += kRowsPerCtaPerStride;
  }
}

// Test-only exact arithmetic twin of the production gate/up phase. Packed
// weights and block scales are one-pass streams much larger than L1, so the
// candidate changes only those global-load cache operators. Keeping a
// separate helper leaves every instruction-selection input on the production
// path untouched until one policy clears its hard gate.
template <NvFp4TestCachePolicy Policy>
__device__ __forceinline__ void
nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_coarsened_512_cache_policy_phase(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    std::uint16_t* const output, const ulonglong2* const staged_activation,
    const float* const decoded_weights, const float* const decoded_scales,
    const unsigned int lane, const unsigned int warp) {
  constexpr std::uint32_t kRows = 17'408U;
  constexpr std::uint32_t kColumns = 5'120U;
  constexpr std::uint32_t kBlocks = 32U;
  constexpr std::uint32_t kWarps = 16U;
  constexpr std::uint32_t kPackedColumns =
      kColumns / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleColumns = kColumns / kNvFp4GroupSize;
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4GroupSize;
  constexpr std::uint32_t kRowStride = kBlocks * kWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kWarps * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;
  static_assert(kBlocks * kWarps ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock);

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarps + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
  std::uint32_t local_row0 = warp * 4U;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < kRows;
       row0 += kRowStride) {
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
      std::uint32_t local_raw_scale_codes = static_cast<std::uint32_t>(
                                                nvfp4_test_cache_load_u8<Policy>(
                                                    row0_scales + scale_column)) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<Policy>(
                   row1_scales + scale_column))
           << 8U) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<Policy>(
                   row2_scales + scale_column))
           << 16U) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<Policy>(
                   row3_scales + scale_column))
           << 24U);
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
        const std::uint32_t packed0 = nvfp4_test_cache_load_u32<Policy>(
            row0_weights + phase_packed_column);
        const std::uint32_t packed1 = nvfp4_test_cache_load_u32<Policy>(
            row1_weights + phase_packed_column);
        const std::uint32_t packed2 = nvfp4_test_cache_load_u32<Policy>(
            row2_weights + phase_packed_column);
        const std::uint32_t packed3 = nvfp4_test_cache_load_u32<Policy>(
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
      output[local_row0] = encode_bf16_rne(sum);
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[local_row0 + 1U] = encode_bf16_rne(sum);
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[local_row0 + 2U] = encode_bf16_rne(sum);
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      output[local_row0 + 3U] = encode_bf16_rne(sum);
    }
    packed_row_offset += kPackedRowStride;
    scale_row_offset += kScaleRowStride;
    local_row0 += kRowsPerCtaPerStride;
  }
}

// Production down-projection phase. The physical grouping is 32 CTAs x 16
// warps while retaining
// the same 512 global projection warps, 2,048-row stride, per-row FMA order,
// warp reduction, scale multiply, and BF16 publication boundary.
__device__ __forceinline__ void
nvfp4_w4a16_down_activation_staged_phase(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    std::uint16_t* const output, const ulonglong2* const staged_activation,
    const float* const decoded_weights, const float* const decoded_scales,
    const unsigned int lane, const unsigned int warp) {
  constexpr std::uint32_t kRows = 5'120U;
  constexpr std::uint32_t kColumns = 17'408U;
  constexpr std::uint32_t kBlocks = 32U;
  constexpr std::uint32_t kWarps = 16U;
  constexpr std::uint32_t kPackedColumns =
      kColumns / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleColumns = kColumns / kNvFp4GroupSize;
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4GroupSize;
  constexpr std::uint32_t kRowStride = kBlocks * kWarps * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;
  static_assert(kBlocks * kWarps ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock);

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarps + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < kRows;
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

// Test-only exact arithmetic twin of the production down phase. Only the
// one-pass packed weights and block scales use the streaming cache operator;
// the production helper remains a separate, untouched instruction-selection
// input.
__device__ __forceinline__ void
nvfp4_w4a16_down_activation_staged_cs_test_phase(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    std::uint16_t* const output, const ulonglong2* const staged_activation,
    const float* const decoded_weights, const float* const decoded_scales,
    const unsigned int lane, const unsigned int warp) {
  constexpr std::uint32_t kRows = 5'120U;
  constexpr std::uint32_t kColumns = 17'408U;
  constexpr std::uint32_t kBlocks = 32U;
  constexpr std::uint32_t kWarps = 16U;
  constexpr std::uint32_t kPackedColumns =
      kColumns / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleColumns = kColumns / kNvFp4GroupSize;
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4GroupSize;
  constexpr std::uint32_t kRowStride = kBlocks * kWarps * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;
  static_assert(kBlocks * kWarps ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock);

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarps + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < kRows;
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
      std::uint32_t local_raw_scale_codes = static_cast<std::uint32_t>(
                                                nvfp4_test_cache_load_u8<
                                                    NvFp4TestCachePolicy::
                                                        kStreaming>(
                                                    row0_scales + scale_column)) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<NvFp4TestCachePolicy::kStreaming>(
                   row1_scales + scale_column))
           << 8U) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<NvFp4TestCachePolicy::kStreaming>(
                   row2_scales + scale_column))
           << 16U) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<NvFp4TestCachePolicy::kStreaming>(
                   row3_scales + scale_column))
           << 24U);
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
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
                row0_weights + phase_packed_column);
        const std::uint32_t packed1 =
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
                row1_weights + phase_packed_column);
        const std::uint32_t packed2 =
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
                row2_weights + phase_packed_column);
        const std::uint32_t packed3 =
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
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

// Test-only runner candidate for the exact M1 Decode down chain. This is a
// source-isolated twin of the selected streaming projection phase, except
// lane zero keeps the independently rounded raw BF16 value in a register and
// immediately applies the existing left-then-raw residual operation. The
// runner's raw workspace is intentionally not referenced by this phase.
__device__ __forceinline__ void
nvfp4_w4a16_down_activation_staged_dead_raw_inline_residual_test_phase(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const residual_left,
    std::uint16_t* const residual_output,
    const ulonglong2* const staged_activation,
    const float* const decoded_weights, const float* const decoded_scales,
    const unsigned int lane, const unsigned int warp) {
  constexpr std::uint32_t kRows = 5'120U;
  constexpr std::uint32_t kColumns = 17'408U;
  constexpr std::uint32_t kBlocks = 32U;
  constexpr std::uint32_t kWarps = 16U;
  constexpr std::uint32_t kPackedColumns =
      kColumns / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleColumns = kColumns / kNvFp4GroupSize;
  constexpr std::uint32_t kPackedIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4ValuesPerByte;
  constexpr std::uint32_t kScaleIterationStride =
      kNvFp4VectorColumnsPerWarp / kNvFp4GroupSize;
  constexpr std::uint32_t kRowStride = kBlocks * kWarps * 4U;
  constexpr std::uint32_t kPackedRowStride =
      kRowStride * kPackedColumns;
  constexpr std::uint32_t kScaleRowStride =
      kRowStride * kScaleColumns;
  static_assert(kBlocks * kWarps ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock);

  const std::uint32_t first_row =
      4U * (static_cast<std::uint32_t>(blockIdx.x) * kWarps + warp);
  std::uint32_t packed_row_offset = first_row * kPackedColumns;
  std::uint32_t scale_row_offset = first_row * kScaleColumns;
#pragma unroll 1
  for (std::uint32_t row0 = first_row; row0 < kRows;
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
      std::uint32_t local_raw_scale_codes = static_cast<std::uint32_t>(
                                                nvfp4_test_cache_load_u8<
                                                    NvFp4TestCachePolicy::
                                                        kStreaming>(
                                                    row0_scales + scale_column)) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<NvFp4TestCachePolicy::kStreaming>(
                   row1_scales + scale_column))
           << 8U) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<NvFp4TestCachePolicy::kStreaming>(
                   row2_scales + scale_column))
           << 16U) |
          (static_cast<std::uint32_t>(
               nvfp4_test_cache_load_u8<NvFp4TestCachePolicy::kStreaming>(
                   row3_scales + scale_column))
           << 24U);
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
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
                row0_weights + phase_packed_column);
        const std::uint32_t packed1 =
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
                row1_weights + phase_packed_column);
        const std::uint32_t packed2 =
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
                row2_weights + phase_packed_column);
        const std::uint32_t packed3 =
            nvfp4_test_cache_load_u32<NvFp4TestCachePolicy::kStreaming>(
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
      const std::uint16_t raw_bits = encode_bf16_rne(sum);
      residual_output[row0] = encode_bf16_rne(
          decode_bf16(residual_left[row0]) + decode_bf16(raw_bits));
    }
    sum = (accumulators1[0] + accumulators1[1]) +
          (accumulators1[2] + accumulators1[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      const std::uint16_t raw_bits = encode_bf16_rne(sum);
      residual_output[row1] = encode_bf16_rne(
          decode_bf16(residual_left[row1]) + decode_bf16(raw_bits));
    }
    sum = (accumulators2[0] + accumulators2[1]) +
          (accumulators2[2] + accumulators2[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      const std::uint16_t raw_bits = encode_bf16_rne(sum);
      residual_output[row2] = encode_bf16_rne(
          decode_bf16(residual_left[row2]) + decode_bf16(raw_bits));
    }
    sum = (accumulators3[0] + accumulators3[1]) +
          (accumulators3[2] + accumulators3[3]);
    sum = warp_sum(sum) * weight_scale_2;
    if (lane == 0U) {
      const std::uint16_t raw_bits = encode_bf16_rne(sum);
      residual_output[row3] = encode_bf16_rne(
          decode_bf16(residual_left[row3]) + decode_bf16(raw_bits));
    }
    packed_row_offset += kPackedRowStride;
    scale_row_offset += kScaleRowStride;
  }
}

// Test-only 64x256 predecessor retained for direct production A/B checks.
// It preserves the historical raw BF16 boundary and centered RMSNorm tree.
template <std::size_t Rows, std::size_t Columns>
__global__ __launch_bounds__(kThreads, 4) void
nvfp4_w4a16_down_residual_norm_activation_staged_predecessor_test_kernel(
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

// Exact production 32x512 down/residual/norm kernel. Projection remains 512
// warps in the same global-warp order, and all threads redistribute residual
// rows. After the grid barrier, the low 256 threads preserve the predecessor's
// RMS accumulation/reduction order; the high half skips norm arithmetic while
// still participating in every CTA barrier.
__global__ __launch_bounds__(512, 2) void
nvfp4_w4a16_down_residual_norm_activation_staged_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation,
    std::uint16_t* const raw_down_output,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output) {
  constexpr std::uint32_t kRows = 5'120U;
  constexpr std::uint32_t kColumns = 17'408U;
  constexpr unsigned int kCoarsenedThreads = 512U;
  constexpr unsigned int kNormThreads = 256U;
  constexpr unsigned int kCoarsenedWarps = 16U;
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr std::uint32_t kActivationVectorCount = kColumns / 8U;
  constexpr std::uint32_t kActivationWordCount = kColumns / 4U;
  constexpr std::uint32_t kRowStride =
      kCoarsenedBlocks * kCoarsenedWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kCoarsenedWarps * 4U;
  constexpr std::uint32_t kNormalizedBlocks = kRows / kNormThreads;
  static_assert(kCoarsenedThreads == kCoarsenedWarps * kWarpSize);
  static_assert(kRowStride ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U);

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid & (kWarpSize - 1U);
  const unsigned int warp = tid / kWarpSize;
  const auto activation_words =
      reinterpret_cast<const std::uint64_t*>(activation);
  auto staged_activation_words =
      reinterpret_cast<std::uint64_t*>(staged_activation);
  for (std::uint32_t word = tid; word < kActivationWordCount;
       word += kCoarsenedThreads) {
    staged_activation_words[word] = activation_words[word];
  }
  if (tid < kFp8EncodedValueCount) {
    decoded_scales[tid] =
        decode_e4m3fn(static_cast<std::uint8_t>(tid));
  }
  if (tid < kNvFp4EncodedValueCount) {
    decoded_weights[tid] =
        decode_e2m1(static_cast<std::uint8_t>(tid));
  }
  __syncthreads();

  nvfp4_w4a16_down_activation_staged_phase(
      packed_weights, block_scales, weight_scale_2, raw_down_output,
      staged_activation, decoded_weights, decoded_scales, lane, warp);

  __syncthreads();
  for (std::uint32_t local_row = tid;;
       local_row += kCoarsenedThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= kRows) {
      break;
    }
    residual_output[row] = encode_bf16_rne(
        decode_bf16(residual_left[row]) + decode_bf16(raw_down_output[row]));
  }

  cooperative_groups::this_grid().sync();
  if (tid < kNormThreads) {
    float sum = 0.0F;
    for (std::uint32_t index = tid; index < kRows;
         index += kNormThreads) {
      const float value = decode_bf16(residual_output[index]);
      sum = fmaf(value, value, sum);
    }
    decoded_scales[tid] = sum;
  }
  __syncthreads();
  for (unsigned int stride = kNormThreads / 2U; stride != 0U;
       stride >>= 1U) {
    if (tid < stride) {
      decoded_scales[tid] += decoded_scales[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(decoded_scales[0] / static_cast<float>(kRows) + epsilon);
  if (blockIdx.x < kNormalizedBlocks && tid < kNormThreads) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(blockIdx.x) * kNormThreads + tid;
    const float gamma = decode_bf16(norm_weight[index]) + 1.0F;
    normalized_output[index] = encode_bf16_rne(
        decode_bf16(residual_output[index]) * inverse_rms * gamma);
  }
}

// Test-only cooperative twin of the exact production down/residual/norm
// kernel. Topology, staging, arithmetic, barriers, and publication boundaries
// remain identical; only its projection phase selects streaming packed/scale
// loads.
__global__ __launch_bounds__(512, 2) void
nvfp4_w4a16_down_residual_norm_activation_staged_cs_test_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation,
    std::uint16_t* const raw_down_output,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output) {
  constexpr std::uint32_t kRows = 5'120U;
  constexpr std::uint32_t kColumns = 17'408U;
  constexpr unsigned int kCoarsenedThreads = 512U;
  constexpr unsigned int kNormThreads = 256U;
  constexpr unsigned int kCoarsenedWarps = 16U;
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr std::uint32_t kActivationVectorCount = kColumns / 8U;
  constexpr std::uint32_t kActivationWordCount = kColumns / 4U;
  constexpr std::uint32_t kRowStride =
      kCoarsenedBlocks * kCoarsenedWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kCoarsenedWarps * 4U;
  constexpr std::uint32_t kNormalizedBlocks = kRows / kNormThreads;
  static_assert(kCoarsenedThreads == kCoarsenedWarps * kWarpSize);
  static_assert(kRowStride ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U);

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid & (kWarpSize - 1U);
  const unsigned int warp = tid / kWarpSize;
  const auto activation_words =
      reinterpret_cast<const std::uint64_t*>(activation);
  auto staged_activation_words =
      reinterpret_cast<std::uint64_t*>(staged_activation);
  for (std::uint32_t word = tid; word < kActivationWordCount;
       word += kCoarsenedThreads) {
    staged_activation_words[word] = activation_words[word];
  }
  if (tid < kFp8EncodedValueCount) {
    decoded_scales[tid] =
        decode_e4m3fn(static_cast<std::uint8_t>(tid));
  }
  if (tid < kNvFp4EncodedValueCount) {
    decoded_weights[tid] =
        decode_e2m1(static_cast<std::uint8_t>(tid));
  }
  __syncthreads();

  nvfp4_w4a16_down_activation_staged_cs_test_phase(
      packed_weights, block_scales, weight_scale_2, raw_down_output,
      staged_activation, decoded_weights, decoded_scales, lane, warp);

  __syncthreads();
  for (std::uint32_t local_row = tid;;
       local_row += kCoarsenedThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= kRows) {
      break;
    }
    residual_output[row] = encode_bf16_rne(
        decode_bf16(residual_left[row]) + decode_bf16(raw_down_output[row]));
  }

  cooperative_groups::this_grid().sync();
  if (tid < kNormThreads) {
    float sum = 0.0F;
    for (std::uint32_t index = tid; index < kRows;
         index += kNormThreads) {
      const float value = decode_bf16(residual_output[index]);
      sum = fmaf(value, value, sum);
    }
    decoded_scales[tid] = sum;
  }
  __syncthreads();
  for (unsigned int stride = kNormThreads / 2U; stride != 0U;
       stride >>= 1U) {
    if (tid < stride) {
      decoded_scales[tid] += decoded_scales[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(decoded_scales[0] / static_cast<float>(kRows) + epsilon);
  if (blockIdx.x < kNormalizedBlocks && tid < kNormThreads) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(blockIdx.x) * kNormThreads + tid;
    const float gamma = decode_bf16(norm_weight[index]) + 1.0F;
    normalized_output[index] = encode_bf16_rne(
        decode_bf16(residual_output[index]) * inverse_rms * gamma);
  }
}

// Test-only runner contract candidate. The raw workspace parameter is kept
// solely so validation/capture can exercise the existing runner signature;
// the kernel never reads or writes that workspace. Projection lane zero
// performs the raw BF16 boundary and residual publication in place, after
// which the selected grid synchronization and centered-RMSNorm tree are
// unchanged.
__global__ __launch_bounds__(512, 2) void
nvfp4_w4a16_down_residual_norm_dead_raw_inline_residual_test_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation,
    std::uint16_t* const raw_down_output,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output) {
  constexpr std::uint32_t kRows = 5'120U;
  constexpr std::uint32_t kColumns = 17'408U;
  constexpr unsigned int kCoarsenedThreads = 512U;
  constexpr unsigned int kNormThreads = 256U;
  constexpr unsigned int kCoarsenedWarps = 16U;
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr std::uint32_t kActivationVectorCount = kColumns / 8U;
  constexpr std::uint32_t kActivationWordCount = kColumns / 4U;
  constexpr std::uint32_t kRowStride =
      kCoarsenedBlocks * kCoarsenedWarps * 4U;
  constexpr std::uint32_t kNormalizedBlocks = kRows / kNormThreads;
  static_assert(kCoarsenedThreads == kCoarsenedWarps * kWarpSize);
  static_assert(kRowStride ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U);

  (void)raw_down_output;
  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid & (kWarpSize - 1U);
  const unsigned int warp = tid / kWarpSize;
  const auto activation_words =
      reinterpret_cast<const std::uint64_t*>(activation);
  auto staged_activation_words =
      reinterpret_cast<std::uint64_t*>(staged_activation);
  for (std::uint32_t word = tid; word < kActivationWordCount;
       word += kCoarsenedThreads) {
    staged_activation_words[word] = activation_words[word];
  }
  if (tid < kFp8EncodedValueCount) {
    decoded_scales[tid] =
        decode_e4m3fn(static_cast<std::uint8_t>(tid));
  }
  if (tid < kNvFp4EncodedValueCount) {
    decoded_weights[tid] =
        decode_e2m1(static_cast<std::uint8_t>(tid));
  }
  __syncthreads();

  nvfp4_w4a16_down_activation_staged_dead_raw_inline_residual_test_phase(
      packed_weights, block_scales, weight_scale_2, residual_left,
      residual_output, staged_activation, decoded_weights, decoded_scales,
      lane, warp);

  cooperative_groups::this_grid().sync();
  if (tid < kNormThreads) {
    float sum = 0.0F;
    for (std::uint32_t index = tid; index < kRows;
         index += kNormThreads) {
      const float value = decode_bf16(residual_output[index]);
      sum = fmaf(value, value, sum);
    }
    decoded_scales[tid] = sum;
  }
  __syncthreads();
  for (unsigned int stride = kNormThreads / 2U; stride != 0U;
       stride >>= 1U) {
    if (tid < stride) {
      decoded_scales[tid] += decoded_scales[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(decoded_scales[0] / static_cast<float>(kRows) + epsilon);
  if (blockIdx.x < kNormalizedBlocks && tid < kNormThreads) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(blockIdx.x) * kNormThreads + tid;
    const float gamma = decode_bf16(norm_weight[index]) + 1.0F;
    normalized_output[index] = encode_bf16_rne(
        decode_bf16(residual_output[index]) * inverse_rms * gamma);
  }
}

// Production-selected cooperative scale6 Function, also exposed through the
// retained test ABI under its historical symbol name. Canonical packed weights
// keep the streaming load policy; only block-scale addressing and exact byte
// reconstruction differ.
__global__ __launch_bounds__(512, 2) void
nvfp4_w4a16_down_residual_norm_activation_staged_scale6_test_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const scale6_sidecar,
    const unsigned int scale_base, const float weight_scale_2,
    const std::uint16_t* const activation,
    std::uint16_t* const raw_down_output,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output) {
  constexpr std::uint32_t kRows = 5'120U;
  constexpr std::uint32_t kColumns = 17'408U;
  constexpr unsigned int kCoarsenedThreads = 512U;
  constexpr unsigned int kNormThreads = 256U;
  constexpr unsigned int kCoarsenedWarps = 16U;
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr std::uint32_t kActivationVectorCount = kColumns / 8U;
  constexpr std::uint32_t kActivationWordCount = kColumns / 4U;
  constexpr std::uint32_t kRowStride =
      kCoarsenedBlocks * kCoarsenedWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kCoarsenedWarps * 4U;
  constexpr std::uint32_t kNormalizedBlocks = kRows / kNormThreads;
  static_assert(kCoarsenedThreads == kCoarsenedWarps * kWarpSize);
  static_assert(kRowStride ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U);

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float decoded_scales[kFp8EncodedValueCount];
  const unsigned int tid = threadIdx.x;
  const unsigned int lane = tid & (kWarpSize - 1U);
  const unsigned int warp = tid / kWarpSize;
  const auto activation_words =
      reinterpret_cast<const std::uint64_t*>(activation);
  auto staged_activation_words =
      reinterpret_cast<std::uint64_t*>(staged_activation);
  for (std::uint32_t word = tid; word < kActivationWordCount;
       word += kCoarsenedThreads) {
    staged_activation_words[word] = activation_words[word];
  }
  if (tid < kFp8EncodedValueCount) {
    decoded_scales[tid] =
        decode_e4m3fn(static_cast<std::uint8_t>(tid));
  }
  if (tid < kNvFp4EncodedValueCount) {
    decoded_weights[tid] =
        decode_e2m1(static_cast<std::uint8_t>(tid));
  }
  __syncthreads();

  nvfp4_w4a16_scale6_activation_staged_phase<kRows, kColumns, false>(
      packed_weights, scale6_sidecar, scale_base, weight_scale_2,
      raw_down_output, staged_activation, decoded_weights, decoded_scales,
      lane, warp);

  __syncthreads();
  for (std::uint32_t local_row = tid;;
       local_row += kCoarsenedThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= kRows) {
      break;
    }
    residual_output[row] = encode_bf16_rne(
        decode_bf16(residual_left[row]) + decode_bf16(raw_down_output[row]));
  }

  cooperative_groups::this_grid().sync();
  if (tid < kNormThreads) {
    float sum = 0.0F;
    for (std::uint32_t index = tid; index < kRows;
         index += kNormThreads) {
      const float value = decode_bf16(residual_output[index]);
      sum = fmaf(value, value, sum);
    }
    decoded_scales[tid] = sum;
  }
  __syncthreads();
  for (unsigned int stride = kNormThreads / 2U; stride != 0U;
       stride >>= 1U) {
    if (tid < stride) {
      decoded_scales[tid] += decoded_scales[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(decoded_scales[0] / static_cast<float>(kRows) + epsilon);
  if (blockIdx.x < kNormalizedBlocks && tid < kNormThreads) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(blockIdx.x) * kNormThreads + tid;
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

// Exact-order 64x256 predecessor fusion of the post-attention residual/RMSNorm
// with gate/up/SiLU. Every CTA repeats the same
// 256-thread RMS reduction used by residual_add_centered_rms_norm_5120_kernel
// and materializes the normalized BF16 activation directly in shared memory.
// CTA zero alone publishes the residual output. Repeating the small reduction
// avoids a grid-wide barrier and the intermediate normalized global buffer;
// the much larger gate/up projection then follows the retained 64-CTA topology
// and arithmetic order.
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
  // predecessor kernel's 11,328-byte shared-memory footprint.
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

// Preserve the exact predecessor 256-thread residual/RMSNorm arithmetic while
// letting a 512-thread CTA participate safely in every block barrier. Threads
// 256..511 remain inactive during this phase; the low half keeps the original
// i += 256 accumulation order and 256-element reduction tree.
__device__ __forceinline__ void
stage_residual_centered_rms_norm_bf16_coarsened_512(
    const std::uint16_t* const left, const std::uint16_t* const right,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const staged_activation,
    float* const norm_partial_or_decoded_scales) {
  constexpr std::uint32_t kColumns = 5'120U;
  constexpr unsigned int kNormThreads = 256U;
  const unsigned int tid = threadIdx.x;
  if (tid < kNormThreads) {
    float sum = 0.0F;
    for (std::uint32_t index = tid; index < kColumns;
         index += kNormThreads) {
      const std::uint16_t residual = encode_bf16_rne(
          decode_bf16(left[index]) + decode_bf16(right[index]));
      staged_activation[index] = residual;
      if (blockIdx.x == 0U) {
        residual_output[index] = residual;
      }
      const float value = decode_bf16(residual);
      sum = fmaf(value, value, sum);
    }
    norm_partial_or_decoded_scales[tid] = sum;
  }
  __syncthreads();

  for (unsigned int stride = kNormThreads / 2U; stride >= kWarpSize;
       stride >>= 1U) {
    if (tid < stride) {
      norm_partial_or_decoded_scales[tid] +=
          norm_partial_or_decoded_scales[tid + stride];
    }
    __syncthreads();
  }
  if (tid < kWarpSize) {
    float warp_partial = norm_partial_or_decoded_scales[tid];
#pragma unroll
    for (unsigned int stride = kWarpSize / 2U; stride != 0U;
         stride >>= 1U) {
      warp_partial +=
          __shfl_down_sync(0xffff'ffffU, warp_partial, stride);
    }
    if (tid == 0U) {
      norm_partial_or_decoded_scales[0] = warp_partial;
    }
  }
  __syncthreads();

  if (tid < kNormThreads) {
    const float inverse_rms =
        rsqrtf(norm_partial_or_decoded_scales[0] /
                   static_cast<float>(kColumns) +
               epsilon);
    for (std::uint32_t index = tid; index < kColumns;
         index += kNormThreads) {
      const float gamma = decode_bf16(norm_weight[index]) + 1.0F;
      staged_activation[index] = encode_bf16_rne(
          decode_bf16(staged_activation[index]) * inverse_rms * gamma);
    }
  }
  __syncthreads();
}

// Test-only companion for a Decode chain whose producer has already
// published the independently rounded residual. It preserves the production
// 256-thread accumulation order, reduction tree, centered-gamma arithmetic,
// and BF16 normalized-activation boundary while removing only the duplicated
// left/right load, add, RNE, and CTA-zero residual publication.
__device__ __forceinline__ void
stage_prerounded_residual_centered_rms_norm_bf16_coarsened_512(
    const std::uint16_t* const residual,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const staged_activation,
    float* const norm_partial_or_decoded_scales) {
  constexpr std::uint32_t kColumns = 5'120U;
  constexpr unsigned int kNormThreads = 256U;
  const unsigned int tid = threadIdx.x;
  if (tid < kNormThreads) {
    float sum = 0.0F;
    for (std::uint32_t index = tid; index < kColumns;
         index += kNormThreads) {
      const std::uint16_t value_bits = residual[index];
      staged_activation[index] = value_bits;
      const float value = decode_bf16(value_bits);
      sum = fmaf(value, value, sum);
    }
    norm_partial_or_decoded_scales[tid] = sum;
  }
  __syncthreads();

  for (unsigned int stride = kNormThreads / 2U; stride >= kWarpSize;
       stride >>= 1U) {
    if (tid < stride) {
      norm_partial_or_decoded_scales[tid] +=
          norm_partial_or_decoded_scales[tid + stride];
    }
    __syncthreads();
  }
  if (tid < kWarpSize) {
    float warp_partial = norm_partial_or_decoded_scales[tid];
#pragma unroll
    for (unsigned int stride = kWarpSize / 2U; stride != 0U;
         stride >>= 1U) {
      warp_partial +=
          __shfl_down_sync(0xffff'ffffU, warp_partial, stride);
    }
    if (tid == 0U) {
      norm_partial_or_decoded_scales[0] = warp_partial;
    }
  }
  __syncthreads();

  if (tid < kNormThreads) {
    const float inverse_rms =
        rsqrtf(norm_partial_or_decoded_scales[0] /
                   static_cast<float>(kColumns) +
               epsilon);
    for (std::uint32_t index = tid; index < kColumns;
         index += kNormThreads) {
      const float gamma = decode_bf16(norm_weight[index]) + 1.0F;
      staged_activation[index] = encode_bf16_rne(
          decode_bf16(staged_activation[index]) * inverse_rms * gamma);
    }
  }
  __syncthreads();
}

// Production CTA-coarsened exact-shape kernel. It changes only the physical
// grouping of the same 512 projection warps: 32 CTAs x 16 warps instead of
// the predecessor's 64 CTAs x 8 warps.
__global__ __launch_bounds__(512, 2) void
nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_coarsened_512_kernel(
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
  constexpr unsigned int kCoarsenedThreads = 512U;
  constexpr unsigned int kCoarsenedWarps = 16U;
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr std::uint32_t kRows = 17'408U;
  constexpr std::uint32_t kColumns = 5'120U;
  constexpr std::uint32_t kActivationVectorCount = kColumns / 8U;
  constexpr std::uint32_t kRowStride =
      kCoarsenedBlocks * kCoarsenedWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kCoarsenedWarps * 4U;
  static_assert(kCoarsenedThreads == kCoarsenedWarps * kWarpSize);
  static_assert(kRowStride ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U);

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float norm_partial_or_decoded_scales[kFp8EncodedValueCount];
  auto staged_activation_bf16 =
      reinterpret_cast<std::uint16_t*>(staged_activation);
  stage_residual_centered_rms_norm_bf16_coarsened_512(
      residual_left, residual_right, norm_weight, epsilon, residual_output,
      staged_activation_bf16, norm_partial_or_decoded_scales);

  if (threadIdx.x < kFp8EncodedValueCount) {
    norm_partial_or_decoded_scales[threadIdx.x] =
        decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  }
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
#pragma unroll 1
  for (unsigned int pair_phase = 0U; pair_phase < 2U; ++pair_phase) {
    nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_coarsened_512_phase(
        pair_phase == 0U ? gate_packed_weights : up_packed_weights,
        pair_phase == 0U ? gate_block_scales : up_block_scales,
        pair_phase == 0U ? gate_weight_scale_2 : up_weight_scale_2,
        pair_phase == 0U ? gate_output : up_output, staged_activation,
        decoded_weights, norm_partial_or_decoded_scales, lane, warp);
  }

  __syncthreads();
  for (std::uint32_t local_row = threadIdx.x;;
       local_row += kCoarsenedThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= kRows) {
      break;
    }
    const float gate = decode_bf16(gate_output[row]);
    const float up = decode_bf16(up_output[row]);
    gate_output[row] =
        encode_bf16_rne(gate / (1.0F + expf(-gate)) * up);
  }
}

// Decode-runner-only contract. The reference runner does not consume
// the independently rounded up projection after this boundary, so keep both
// rounded gate/up intermediates CTA-local and publish only the final
// SiLU(gate)*up BF16 vector. The 576-entry arrays cover nine 64-row strides
// for logical blocks 0..15; logical blocks 16..31 use only eight strides.
// The final 512-thread mapping is identical to production's balanced CTA
// epilogue and therefore preserves its SFU parallelism and arithmetic order.
__global__ __launch_bounds__(512, 2) void
nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_kernel(
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
    std::uint16_t* const gate_output) {
  constexpr unsigned int kCoarsenedThreads = 512U;
  constexpr unsigned int kCoarsenedWarps = 16U;
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr std::uint32_t kRows = 17'408U;
  constexpr std::uint32_t kColumns = 5'120U;
  constexpr std::uint32_t kActivationVectorCount = kColumns / 8U;
  constexpr std::uint32_t kRowStride =
      kCoarsenedBlocks * kCoarsenedWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kCoarsenedWarps * 4U;
  constexpr std::uint32_t kMaximumRowsPerCta = 576U;
  static_assert(kCoarsenedThreads == kCoarsenedWarps * kWarpSize);
  static_assert(kRowStride ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U);

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float norm_partial_or_decoded_scales[kFp8EncodedValueCount];
  __shared__ std::uint16_t staged_gate[kMaximumRowsPerCta];
  __shared__ std::uint16_t staged_up[kMaximumRowsPerCta];
  auto staged_activation_bf16 =
      reinterpret_cast<std::uint16_t*>(staged_activation);
  stage_residual_centered_rms_norm_bf16_coarsened_512(
      residual_left, residual_right, norm_weight, epsilon, residual_output,
      staged_activation_bf16, norm_partial_or_decoded_scales);

  if (threadIdx.x < kFp8EncodedValueCount) {
    norm_partial_or_decoded_scales[threadIdx.x] =
        decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  }
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
#pragma unroll 1
  for (unsigned int pair_phase = 0U; pair_phase < 2U; ++pair_phase) {
    nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_coarsened_512_phase<
        true>(pair_phase == 0U ? gate_packed_weights : up_packed_weights,
              pair_phase == 0U ? gate_block_scales : up_block_scales,
              pair_phase == 0U ? gate_weight_scale_2 : up_weight_scale_2,
              pair_phase == 0U ? staged_gate : staged_up,
              staged_activation, decoded_weights,
              norm_partial_or_decoded_scales, lane, warp);
  }

  __syncthreads();
  for (std::uint32_t local_row = threadIdx.x;;
       local_row += kCoarsenedThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= kRows) {
      break;
    }
    const float gate = decode_bf16(staged_gate[local_row]);
    const float up = decode_bf16(staged_up[local_row]);
    gate_output[row] =
        encode_bf16_rne(gate / (1.0F + expf(-gate)) * up);
  }
}

// Test-only production twin for packed-weight/block-scale cache-policy
// screens. The residual/RMSNorm setup, two rounded projection phases,
// CTA-local dead-up staging, balanced SiLU epilogue, grid, and shared layout
// are copied from production; only the phase helper's global load operator
// differs.
template <NvFp4TestCachePolicy Policy>
__global__ __launch_bounds__(512, 2) void
nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_cache_policy_test_kernel(
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
    std::uint16_t* const gate_output) {
  constexpr unsigned int kCoarsenedThreads = 512U;
  constexpr unsigned int kCoarsenedWarps = 16U;
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr std::uint32_t kRows = 17'408U;
  constexpr std::uint32_t kColumns = 5'120U;
  constexpr std::uint32_t kActivationVectorCount = kColumns / 8U;
  constexpr std::uint32_t kRowStride =
      kCoarsenedBlocks * kCoarsenedWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kCoarsenedWarps * 4U;
  constexpr std::uint32_t kMaximumRowsPerCta = 576U;
  static_assert(kCoarsenedThreads == kCoarsenedWarps * kWarpSize);
  static_assert(kRowStride ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U);

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float norm_partial_or_decoded_scales[kFp8EncodedValueCount];
  __shared__ std::uint16_t staged_gate[kMaximumRowsPerCta];
  __shared__ std::uint16_t staged_up[kMaximumRowsPerCta];
  auto staged_activation_bf16 =
      reinterpret_cast<std::uint16_t*>(staged_activation);
  stage_residual_centered_rms_norm_bf16_coarsened_512(
      residual_left, residual_right, norm_weight, epsilon, residual_output,
      staged_activation_bf16, norm_partial_or_decoded_scales);

  if (threadIdx.x < kFp8EncodedValueCount) {
    norm_partial_or_decoded_scales[threadIdx.x] =
        decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  }
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
#pragma unroll 1
  for (unsigned int pair_phase = 0U; pair_phase < 2U; ++pair_phase) {
    nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_coarsened_512_cache_policy_phase<
        Policy>(
        pair_phase == 0U ? gate_packed_weights : up_packed_weights,
        pair_phase == 0U ? gate_block_scales : up_block_scales,
        pair_phase == 0U ? gate_weight_scale_2 : up_weight_scale_2,
        pair_phase == 0U ? staged_gate : staged_up, staged_activation,
        decoded_weights, norm_partial_or_decoded_scales, lane, warp);
  }

  __syncthreads();
  for (std::uint32_t local_row = threadIdx.x;;
       local_row += kCoarsenedThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= kRows) {
      break;
    }
    const float gate = decode_bf16(staged_gate[local_row]);
    const float up = decode_bf16(staged_up[local_row]);
    gate_output[row] =
        encode_bf16_rne(gate / (1.0F + expf(-gate)) * up);
  }
}

// Test-only dead-up twin using independent six-bit scale sidecars for gate
// and up. The residual/RMSNorm setup, CTA-local rounded intermediates, SiLU
// epilogue, topology, and publication contract match the selected Decode
// kernel; canonical packed-weight loads remain streaming.
__global__ __launch_bounds__(512, 2) void
nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_scale6_test_kernel(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_scale6_sidecar,
    const unsigned int gate_scale_base,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_scale6_sidecar,
    const unsigned int up_scale_base, const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output) {
  constexpr unsigned int kCoarsenedThreads = 512U;
  constexpr unsigned int kCoarsenedWarps = 16U;
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr std::uint32_t kRows = 17'408U;
  constexpr std::uint32_t kColumns = 5'120U;
  constexpr std::uint32_t kActivationVectorCount = kColumns / 8U;
  constexpr std::uint32_t kRowStride =
      kCoarsenedBlocks * kCoarsenedWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kCoarsenedWarps * 4U;
  constexpr std::uint32_t kMaximumRowsPerCta = 576U;
  static_assert(kCoarsenedThreads == kCoarsenedWarps * kWarpSize);
  static_assert(kRowStride ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U);

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float norm_partial_or_decoded_scales[kFp8EncodedValueCount];
  __shared__ std::uint16_t staged_gate[kMaximumRowsPerCta];
  __shared__ std::uint16_t staged_up[kMaximumRowsPerCta];
  auto staged_activation_bf16 =
      reinterpret_cast<std::uint16_t*>(staged_activation);
  stage_residual_centered_rms_norm_bf16_coarsened_512(
      residual_left, residual_right, norm_weight, epsilon, residual_output,
      staged_activation_bf16, norm_partial_or_decoded_scales);

  if (threadIdx.x < kFp8EncodedValueCount) {
    norm_partial_or_decoded_scales[threadIdx.x] =
        decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  }
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
#pragma unroll 1
  for (unsigned int pair_phase = 0U; pair_phase < 2U; ++pair_phase) {
    nvfp4_w4a16_scale6_activation_staged_phase<kRows, kColumns, true>(
        pair_phase == 0U ? gate_packed_weights : up_packed_weights,
        pair_phase == 0U ? gate_scale6_sidecar : up_scale6_sidecar,
        pair_phase == 0U ? gate_scale_base : up_scale_base,
        pair_phase == 0U ? gate_weight_scale_2 : up_weight_scale_2,
        pair_phase == 0U ? staged_gate : staged_up, staged_activation,
        decoded_weights, norm_partial_or_decoded_scales, lane, warp);
  }

  __syncthreads();
  for (std::uint32_t local_row = threadIdx.x;;
       local_row += kCoarsenedThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= kRows) {
      break;
    }
    const float gate = decode_bf16(staged_gate[local_row]);
    const float up = decode_bf16(staged_up[local_row]);
    gate_output[row] =
        encode_bf16_rne(gate / (1.0F + expf(-gate)) * up);
  }
}

// Test-only second half of the Decode residual-chain candidate. The producer
// has already materialized the visible BF16 residual, so this kernel consumes
// it directly and otherwise keeps the selected dead-up projection topology,
// local BF16 gate/up boundaries, and balanced SiLU epilogue unchanged.
__global__ __launch_bounds__(512, 2) void
nvfp4_w4a16_gemv_bf16_prerounded_residual_norm_gate_up_silu_dead_up_shared_pair_test_kernel(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const gate_output) {
  constexpr unsigned int kCoarsenedThreads = 512U;
  constexpr unsigned int kCoarsenedWarps = 16U;
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr std::uint32_t kRows = 17'408U;
  constexpr std::uint32_t kColumns = 5'120U;
  constexpr std::uint32_t kActivationVectorCount = kColumns / 8U;
  constexpr std::uint32_t kRowStride =
      kCoarsenedBlocks * kCoarsenedWarps * 4U;
  constexpr std::uint32_t kRowsPerCtaPerStride = kCoarsenedWarps * 4U;
  constexpr std::uint32_t kMaximumRowsPerCta = 576U;
  static_assert(kCoarsenedThreads == kCoarsenedWarps * kWarpSize);
  static_assert(kRowStride ==
                kNvFp4M1RowQuadMaximumBlocks * kWarpsPerBlock * 4U);

  __shared__ ulonglong2 staged_activation[kActivationVectorCount];
  __shared__ float decoded_weights[kNvFp4EncodedValueCount];
  __shared__ float norm_partial_or_decoded_scales[kFp8EncodedValueCount];
  __shared__ std::uint16_t staged_gate[kMaximumRowsPerCta];
  __shared__ std::uint16_t staged_up[kMaximumRowsPerCta];
  auto staged_activation_bf16 =
      reinterpret_cast<std::uint16_t*>(staged_activation);
  stage_prerounded_residual_centered_rms_norm_bf16_coarsened_512(
      residual, norm_weight, epsilon, staged_activation_bf16,
      norm_partial_or_decoded_scales);

  if (threadIdx.x < kFp8EncodedValueCount) {
    norm_partial_or_decoded_scales[threadIdx.x] =
        decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x));
  }
  if (threadIdx.x < kNvFp4EncodedValueCount) {
    decoded_weights[threadIdx.x] =
        decode_e2m1(static_cast<std::uint8_t>(threadIdx.x));
  }
  __syncthreads();

  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
#pragma unroll 1
  for (unsigned int pair_phase = 0U; pair_phase < 2U; ++pair_phase) {
    nvfp4_w4a16_gemv_bf16_gate_up_pair_activation_staged_coarsened_512_phase<
        true>(pair_phase == 0U ? gate_packed_weights : up_packed_weights,
              pair_phase == 0U ? gate_block_scales : up_block_scales,
              pair_phase == 0U ? gate_weight_scale_2 : up_weight_scale_2,
              pair_phase == 0U ? staged_gate : staged_up,
              staged_activation, decoded_weights,
              norm_partial_or_decoded_scales, lane, warp);
  }

  __syncthreads();
  for (std::uint32_t local_row = threadIdx.x;;
       local_row += kCoarsenedThreads) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(blockIdx.x) * kRowsPerCtaPerStride +
        (local_row / kRowsPerCtaPerStride) * kRowStride +
        local_row % kRowsPerCtaPerStride;
    if (row >= kRows) {
      break;
    }
    const float gate = decode_bf16(staged_gate[local_row]);
    const float up = decode_bf16(staged_up[local_row]);
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

[[nodiscard]] int validate_fp8_qkv_z_bf16_ab_launch(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t ab_rows,
    const std::size_t columns, std::uint16_t* const qkv_output,
    std::uint16_t* const z_output, std::uint16_t* const a_output,
    std::uint16_t* const b_output) noexcept {
  const int qkv_z_validation = validate_fp8_qkv_z_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, activation,
      qkv_rows, z_rows, columns, qkv_output, z_output,
      kFp8QkvZProductionBlocks);
  if (qkv_z_validation != static_cast<int>(cudaSuccess)) {
    return qkv_z_validation;
  }
  if (ab_rows != kLinearAttentionAbRows || a_weights == nullptr ||
      b_weights == nullptr || a_output == nullptr || b_output == nullptr) {
    return invalid_value();
  }

  constexpr std::size_t kQkvWeightBytes =
      kFp8QkvRows * kFp8QkvZColumns;
  constexpr std::size_t kZWeightBytes = kFp8ZRows * kFp8QkvZColumns;
  constexpr std::size_t kAbWeightBytes =
      kLinearAttentionAbRows * kFp8QkvZColumns * sizeof(std::uint16_t);
  constexpr std::size_t kActivationBytes =
      kFp8QkvZColumns * sizeof(std::uint16_t);
  constexpr std::size_t kQkvOutputBytes =
      kFp8QkvRows * sizeof(std::uint16_t);
  constexpr std::size_t kZOutputBytes = kFp8ZRows * sizeof(std::uint16_t);
  constexpr std::size_t kAbOutputBytes =
      kLinearAttentionAbRows * sizeof(std::uint16_t);
  const void* const inputs[]{qkv_weights, z_weights, a_weights, b_weights,
                             activation};
  constexpr std::size_t input_bytes[]{kQkvWeightBytes, kZWeightBytes,
                                       kAbWeightBytes, kAbWeightBytes,
                                       kActivationBytes};
  void* const outputs[]{qkv_output, z_output, a_output, b_output};
  constexpr std::size_t output_bytes[]{kQkvOutputBytes, kZOutputBytes,
                                        kAbOutputBytes, kAbOutputBytes};
  for (std::size_t output_index = 0U; output_index < 4U; ++output_index) {
    if (byte_range_overflows(outputs[output_index],
                             output_bytes[output_index])) {
      return invalid_value();
    }
    for (std::size_t input_index = 0U; input_index < 5U; ++input_index) {
      if (ranges_overlap(outputs[output_index], output_bytes[output_index],
                         inputs[input_index], input_bytes[input_index])) {
        return invalid_value();
      }
    }
    for (std::size_t other_output = output_index + 1U; other_output < 4U;
         ++other_output) {
      if (ranges_overlap(outputs[output_index], output_bytes[output_index],
                         outputs[other_output],
                         output_bytes[other_output])) {
        return invalid_value();
      }
    }
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool fp8_qkv_z_bf16_ab_launch_is_aligned(
    const std::uint8_t* const qkv_weights,
    const std::uint8_t* const z_weights,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation,
    const std::uint16_t* const qkv_output,
    const std::uint16_t* const z_output,
    const std::uint16_t* const a_output,
    const std::uint16_t* const b_output) noexcept {
  return fp8_qkv_z_launch_is_aligned(qkv_weights, z_weights, activation,
                                     qkv_output, z_output) &&
         pointer_is_aligned<alignof(std::uint16_t)>(a_weights) &&
         pointer_is_aligned<alignof(std::uint16_t)>(b_weights) &&
         pointer_is_aligned<alignof(std::uint16_t)>(a_output) &&
         pointer_is_aligned<alignof(std::uint16_t)>(b_output);
}

[[nodiscard]] int
validate_fp8_qkv_z_bf16_ab_causal_conv_epilogue_launch(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t ab_rows,
    const std::size_t columns, std::uint16_t* const qkv_output,
    std::uint16_t* const z_output, std::uint16_t* const a_output,
    std::uint16_t* const b_output,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history) noexcept {
  const int composite_validation = validate_fp8_qkv_z_bf16_ab_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, qkv_rows, z_rows, ab_rows, columns, qkv_output,
      z_output, a_output, b_output);
  if (composite_validation != static_cast<int>(cudaSuccess)) {
    return composite_validation;
  }
  if (conv_weight == nullptr || history == nullptr) {
    return invalid_value();
  }

  constexpr std::size_t kQkvWeightBytes =
      kFp8QkvRows * kFp8QkvZColumns;
  constexpr std::size_t kZWeightBytes =
      kFp8ZRows * kFp8QkvZColumns;
  constexpr std::size_t kAbWeightBytes =
      kLinearAttentionAbRows * kFp8QkvZColumns * sizeof(std::uint16_t);
  constexpr std::size_t kActivationBytes =
      kFp8QkvZColumns * sizeof(std::uint16_t);
  constexpr std::size_t kQkvOutputBytes =
      kFp8QkvRows * sizeof(std::uint16_t);
  constexpr std::size_t kZOutputBytes =
      kFp8ZRows * sizeof(std::uint16_t);
  constexpr std::size_t kAbOutputBytes =
      kLinearAttentionAbRows * sizeof(std::uint16_t);
  constexpr std::size_t kConvWeightBytes =
      kFp8QkvRows * kLinearAttentionConvKernelWidth *
      sizeof(std::uint16_t);
  constexpr std::size_t kHistoryBytes =
      kFp8QkvRows * kLinearAttentionConvHistoryWidth *
      sizeof(std::uint16_t);
  if (byte_range_overflows(conv_weight, kConvWeightBytes) ||
      byte_range_overflows(history, kHistoryBytes) ||
      ranges_overlap(conv_weight, kConvWeightBytes, history,
                     kHistoryBytes)) {
    return invalid_value();
  }

  const void* const existing_inputs[]{qkv_weights, z_weights, a_weights,
                                      b_weights, activation};
  constexpr std::size_t existing_input_bytes[]{
      kQkvWeightBytes, kZWeightBytes, kAbWeightBytes, kAbWeightBytes,
      kActivationBytes};
  const void* const existing_outputs[]{qkv_output, z_output, a_output,
                                       b_output};
  constexpr std::size_t existing_output_bytes[]{
      kQkvOutputBytes, kZOutputBytes, kAbOutputBytes, kAbOutputBytes};
  for (std::size_t index = 0U; index < 5U; ++index) {
    if (ranges_overlap(conv_weight, kConvWeightBytes,
                       existing_inputs[index], existing_input_bytes[index]) ||
        ranges_overlap(history, kHistoryBytes, existing_inputs[index],
                       existing_input_bytes[index])) {
      return invalid_value();
    }
  }
  for (std::size_t index = 0U; index < 4U; ++index) {
    if (ranges_overlap(conv_weight, kConvWeightBytes,
                       existing_outputs[index],
                       existing_output_bytes[index]) ||
        ranges_overlap(history, kHistoryBytes, existing_outputs[index],
                       existing_output_bytes[index])) {
      return invalid_value();
    }
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool
fp8_qkv_z_bf16_ab_causal_conv_epilogue_launch_is_aligned(
    const std::uint8_t* const qkv_weights,
    const std::uint8_t* const z_weights,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation,
    const std::uint16_t* const qkv_output,
    const std::uint16_t* const z_output,
    const std::uint16_t* const a_output,
    const std::uint16_t* const b_output,
    const std::uint16_t* const conv_weight,
    const std::uint16_t* const history) noexcept {
  return fp8_qkv_z_bf16_ab_launch_is_aligned(
             qkv_weights, z_weights, a_weights, b_weights, activation,
             qkv_output, z_output, a_output, b_output) &&
         pointer_is_aligned<alignof(std::uint16_t)>(conv_weight) &&
         pointer_is_aligned<alignof(std::uint16_t)>(history);
}

[[nodiscard]] int validate_fp8_q_kv_launch(
    const std::uint8_t* const q_weights, const float q_weight_scale,
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation, const std::size_t q_rows,
    const std::size_t kv_rows, const std::size_t columns,
    std::uint16_t* const q_output, std::uint16_t* const key_output,
    std::uint16_t* const value_output) noexcept {
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

  const std::size_t q_output_bytes =
      kFp8FullAttentionQRows * sizeof(std::uint16_t);
  const std::size_t kv_output_bytes =
      kFp8KvPairRows * sizeof(std::uint16_t);
  const std::size_t q_weight_bytes =
      kFp8FullAttentionQRows * kFp8KvPairColumns;
  const std::size_t kv_weight_bytes =
      kFp8KvPairRows * kFp8KvPairColumns;
  if (ranges_overlap(q_output, q_output_bytes, key_output,
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
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool fp8_q_kv_launch_is_aligned(
    const std::uint8_t* const q_weights,
    const std::uint8_t* const key_weights,
    const std::uint8_t* const value_weights,
    const std::uint16_t* const activation,
    const std::uint16_t* const q_output,
    const std::uint16_t* const key_output,
    const std::uint16_t* const value_output) noexcept {
  return pointer_is_aligned<alignof(std::uint32_t)>(q_weights) &&
         pointer_is_aligned<alignof(std::uint32_t)>(key_weights) &&
         pointer_is_aligned<alignof(std::uint32_t)>(value_weights) &&
         pointer_is_aligned<alignof(std::uint64_t)>(activation) &&
         pointer_is_aligned<alignof(std::uint16_t)>(q_output) &&
         pointer_is_aligned<alignof(std::uint16_t)>(key_output) &&
         pointer_is_aligned<alignof(std::uint16_t)>(value_output);
}

[[nodiscard]] int validate_fp8_q_kv_aosoa4_preswizzled_test_launch(
    const std::uint8_t* const q_sidecar,
    const float q_weight_scale,
    const std::uint8_t* const kv_sidecar,
    const float key_weight_scale,
    const float value_weight_scale,
    const std::uint16_t* const activation,
    const std::size_t q_rows,
    const std::size_t kv_rows,
    const std::size_t columns,
    std::uint16_t* const q_output,
    std::uint16_t* const key_output,
    std::uint16_t* const value_output) noexcept {
  if (q_rows != kFp8FullAttentionQRows || kv_rows != kFp8KvPairRows ||
      columns != kFp8KvPairColumns ||
      !std::isfinite(q_weight_scale) || q_weight_scale < 0.0F ||
      !std::isfinite(key_weight_scale) || key_weight_scale < 0.0F ||
      !std::isfinite(value_weight_scale) || value_weight_scale < 0.0F ||
      q_sidecar == nullptr || kv_sidecar == nullptr ||
      activation == nullptr || q_output == nullptr ||
      key_output == nullptr || value_output == nullptr) {
    return invalid_value();
  }

  constexpr std::size_t kQSidecarBytes =
      kFp8FullAttentionQRows * kFp8KvPairColumns;
  constexpr std::size_t kKvSidecarBytes =
      2U * kFp8KvPairRows * kFp8KvPairColumns;
  constexpr std::size_t kActivationBytes =
      kFp8KvPairColumns * sizeof(std::uint16_t);
  constexpr std::size_t kQOutputBytes =
      kFp8FullAttentionQRows * sizeof(std::uint16_t);
  constexpr std::size_t kKvOutputBytes =
      kFp8KvPairRows * sizeof(std::uint16_t);
  const void* const inputs[]{q_sidecar, kv_sidecar, activation};
  constexpr std::size_t input_bytes[]{kQSidecarBytes, kKvSidecarBytes,
                                       kActivationBytes};
  void* const outputs[]{q_output, key_output, value_output};
  constexpr std::size_t output_bytes[]{kQOutputBytes, kKvOutputBytes,
                                        kKvOutputBytes};
  for (std::size_t first = 0U; first < 3U; ++first) {
    for (std::size_t second = first + 1U; second < 3U; ++second) {
      if (ranges_overlap(inputs[first], input_bytes[first], inputs[second],
                         input_bytes[second])) {
        return invalid_value();
      }
    }
  }
  for (std::size_t output_index = 0U; output_index < 3U; ++output_index) {
    for (std::size_t input_index = 0U; input_index < 3U; ++input_index) {
      if (ranges_overlap(outputs[output_index], output_bytes[output_index],
                         inputs[input_index], input_bytes[input_index])) {
        return invalid_value();
      }
    }
    for (std::size_t other_output = output_index + 1U;
         other_output < 3U; ++other_output) {
      if (ranges_overlap(outputs[output_index], output_bytes[output_index],
                         outputs[other_output],
                         output_bytes[other_output])) {
        return invalid_value();
      }
    }
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int validate_fp8_q_kv_aosoa4_pack_test_launch(
    const std::uint8_t* const canonical_q,
    const std::uint8_t* const canonical_key,
    const std::uint8_t* const canonical_value,
    const std::size_t q_rows,
    const std::size_t kv_rows,
    const std::size_t columns,
    std::uint8_t* const q_sidecar,
    std::uint8_t* const kv_sidecar) noexcept {
  if (q_rows != kFp8FullAttentionQRows || kv_rows != kFp8KvPairRows ||
      columns != kFp8KvPairColumns || canonical_q == nullptr ||
      canonical_key == nullptr || canonical_value == nullptr ||
      q_sidecar == nullptr || kv_sidecar == nullptr) {
    return invalid_value();
  }
  constexpr std::size_t kQBytes =
      kFp8FullAttentionQRows * kFp8KvPairColumns;
  constexpr std::size_t kKvBytes =
      kFp8KvPairRows * kFp8KvPairColumns;
  constexpr std::size_t kCombinedKvBytes = 2U * kKvBytes;
  const void* const inputs[]{canonical_q, canonical_key, canonical_value};
  constexpr std::size_t input_bytes[]{kQBytes, kKvBytes, kKvBytes};
  void* const outputs[]{q_sidecar, kv_sidecar};
  constexpr std::size_t output_bytes[]{kQBytes, kCombinedKvBytes};
  for (std::size_t output_index = 0U; output_index < 2U; ++output_index) {
    for (std::size_t input_index = 0U; input_index < 3U; ++input_index) {
      if (ranges_overlap(outputs[output_index], output_bytes[output_index],
                         inputs[input_index], input_bytes[input_index])) {
        return invalid_value();
      }
    }
  }
  if (ranges_overlap(q_sidecar, kQBytes, kv_sidecar, kCombinedKvBytes)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
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

void launch_fp8_qkv_z_bf16_ab_pair_tail_composite_unchecked(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output,
    cudaStream_t const stream) noexcept {
  fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_kernel
      <<<kFp8QkvZProductionBlocks, kThreads, 0U, stream>>>(
          qkv_weights, qkv_weight_scale, z_weights, z_weight_scale,
          a_weights, b_weights, activation, columns, qkv_output, z_output,
          a_output, b_output);
}

void
launch_fp8_qkv_z_bf16_ab_pair_tail_composite_register_lookahead_test_unchecked(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output,
    cudaStream_t const stream) noexcept {
  fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_register_lookahead_test_kernel
      <<<kFp8QkvZProductionBlocks, kThreads, 0U, stream>>>(
          qkv_weights, qkv_weight_scale, z_weights, z_weight_scale,
          a_weights, b_weights, activation, columns, qkv_output, z_output,
          a_output, b_output);
}

void
launch_fp8_qkv_z_bf16_ab_pair_tail_composite_causal_conv_epilogue_test_unchecked(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history, cudaStream_t const stream) noexcept {
  fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_causal_conv_epilogue_test_kernel
      <<<kFp8QkvZProductionBlocks, kThreads, 0U, stream>>>(
          qkv_weights, qkv_weight_scale, z_weights, z_weight_scale,
          a_weights, b_weights, activation, columns, qkv_output, z_output,
          a_output, b_output, conv_weight, history);
}

void launch_fp8_qkv_z_bf16_ab_pair_tail_composite_cs_test_unchecked(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t columns,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output,
    cudaStream_t const stream) noexcept {
  fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_cs_test_kernel
      <<<kFp8QkvZProductionBlocks, kThreads, 0U, stream>>>(
          qkv_weights, qkv_weight_scale, z_weights, z_weight_scale,
          a_weights, b_weights, activation, columns, qkv_output, z_output,
          a_output, b_output);
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

void launch_fp8_q_kv_reduction_scratch_ping_pong_test_unchecked(
    const std::uint8_t* const q_weights, const float q_weight_scale,
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation, std::uint16_t* const q_output,
    std::uint16_t* const key_output, std::uint16_t* const value_output,
    cudaStream_t const stream) noexcept {
  fp8_w8a16_gemv_bf16_q_kv_two_phase_reduction_scratch_ping_pong_kernel
      <<<kFp8FullAttentionBlocks, kThreads, 0U, stream>>>(
          q_weights, q_weight_scale, key_weights, key_weight_scale,
          value_weights, value_weight_scale, activation, q_output,
          key_output, value_output);
}

void launch_fp8_q_kv_tail_barrier_test_unchecked(
    const std::uint8_t* const q_weights, const float q_weight_scale,
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation, std::uint16_t* const q_output,
    std::uint16_t* const key_output, std::uint16_t* const value_output,
    cudaStream_t const stream) noexcept {
  fp8_w8a16_gemv_bf16_q_kv_two_phase_tail_barrier_test_kernel
      <<<kFp8FullAttentionBlocks, kThreads, 0U, stream>>>(
          q_weights, q_weight_scale, key_weights, key_weight_scale,
          value_weights, value_weight_scale, activation, q_output,
          key_output, value_output);
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

[[nodiscard]] bool nvfp4_scale6_sidecar_size(
    const std::size_t rows, const std::size_t columns,
    std::size_t* const bytes) noexcept {
  if (bytes == nullptr || (rows % 4U) != 0U ||
      (columns % kNvFp4Scale6ColumnsPerTile) != 0U) {
    return false;
  }
  const std::size_t row_quads = rows / 4U;
  const std::size_t tiles_per_row =
      columns / kNvFp4Scale6ColumnsPerTile;
  if (multiply_overflows(row_quads, tiles_per_row)) {
    return false;
  }
  const std::size_t tiles = row_quads * tiles_per_row;
  constexpr std::size_t kBytesPerTile =
      kNvFp4Scale6WordsPerRowQuadTile * sizeof(std::uint32_t);
  if (multiply_overflows(tiles, kBytesPerTile)) {
    return false;
  }
  *bytes = tiles * kBytesPerTile;
  return true;
}

[[nodiscard]] int validate_nvfp4_scale6_launch(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const scale6_sidecar,
    const unsigned int scale_base, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output) noexcept {
  if (!std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F ||
      scale_base > kNvFp4Scale6MaximumBase ||
      multiply_overflows(rows, columns)) {
    return invalid_value();
  }
  std::size_t scale6_bytes = 0U;
  if (!nvfp4_scale6_sidecar_size(rows, columns, &scale6_bytes)) {
    return invalid_value();
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (packed_weights == nullptr || scale6_sidecar == nullptr ||
      activation == nullptr || output == nullptr ||
      multiply_overflows(columns, sizeof(std::uint16_t)) ||
      multiply_overflows(rows, sizeof(std::uint16_t))) {
    return invalid_value();
  }

  const std::size_t packed_bytes =
      rows * (columns / kNvFp4ValuesPerByte);
  const std::size_t activation_bytes = columns * sizeof(std::uint16_t);
  const std::size_t output_bytes = rows * sizeof(std::uint16_t);
  if (ranges_overlap(output, output_bytes, packed_weights, packed_bytes) ||
      ranges_overlap(output, output_bytes, scale6_sidecar, scale6_bytes) ||
      ranges_overlap(output, output_bytes, activation, activation_bytes)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int
validate_nvfp4_scale6_residual_norm_gate_up_silu_launch(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_scale6_sidecar,
    const unsigned int gate_scale_base,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_scale6_sidecar,
    const unsigned int up_scale_base, const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_workspace) noexcept {
  int validation = validate_nvfp4_scale6_launch(
      gate_packed_weights, gate_scale6_sidecar, gate_scale_base,
      gate_weight_scale_2, residual_right, rows, columns, gate_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  validation = validate_nvfp4_scale6_launch(
      up_packed_weights, up_scale6_sidecar, up_scale_base,
      up_weight_scale_2, residual_right, rows, columns, up_workspace);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0U ||
      columns == 0U || !std::isfinite(epsilon) || epsilon <= 0.0F ||
      residual_left == nullptr || norm_weight == nullptr ||
      residual_output == nullptr ||
      multiply_overflows(columns, sizeof(std::uint16_t))) {
    return invalid_value();
  }

  std::size_t scale6_bytes = 0U;
  if (!nvfp4_scale6_sidecar_size(rows, columns, &scale6_bytes)) {
    return invalid_value();
  }
  const std::size_t input_bytes = columns * sizeof(std::uint16_t);
  const std::size_t projection_output_bytes =
      rows * sizeof(std::uint16_t);
  const std::size_t packed_bytes =
      rows * (columns / kNvFp4ValuesPerByte);
  if (byte_range_overflows(residual_left, input_bytes) ||
      byte_range_overflows(norm_weight, input_bytes) ||
      byte_range_overflows(residual_output, input_bytes) ||
      ranges_overlap(gate_output, projection_output_bytes, up_workspace,
                     projection_output_bytes) ||
      ranges_overlap(gate_output, projection_output_bytes,
                     up_packed_weights, packed_bytes) ||
      ranges_overlap(gate_output, projection_output_bytes,
                     up_scale6_sidecar, scale6_bytes) ||
      ranges_overlap(up_workspace, projection_output_bytes,
                     gate_packed_weights, packed_bytes) ||
      ranges_overlap(up_workspace, projection_output_bytes,
                     gate_scale6_sidecar, scale6_bytes) ||
      ranges_overlap(residual_output, input_bytes, residual_left,
                     input_bytes) ||
      ranges_overlap(residual_output, input_bytes, residual_right,
                     input_bytes) ||
      ranges_overlap(residual_output, input_bytes, norm_weight,
                     input_bytes) ||
      ranges_overlap(residual_output, input_bytes, gate_packed_weights,
                     packed_bytes) ||
      ranges_overlap(residual_output, input_bytes, gate_scale6_sidecar,
                     scale6_bytes) ||
      ranges_overlap(residual_output, input_bytes, up_packed_weights,
                     packed_bytes) ||
      ranges_overlap(residual_output, input_bytes, up_scale6_sidecar,
                     scale6_bytes) ||
      ranges_overlap(residual_output, input_bytes, gate_output,
                     projection_output_bytes) ||
      ranges_overlap(residual_output, input_bytes, up_workspace,
                     projection_output_bytes) ||
      ranges_overlap(gate_output, projection_output_bytes, residual_left,
                     input_bytes) ||
      ranges_overlap(gate_output, projection_output_bytes, norm_weight,
                     input_bytes) ||
      ranges_overlap(up_workspace, projection_output_bytes, residual_left,
                     input_bytes) ||
      ranges_overlap(up_workspace, projection_output_bytes, norm_weight,
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

template <std::size_t Rows, std::size_t Columns>
void launch_nvfp4_lm_head_activation_staged_cs_test_instance_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  nvfp4_w4a16_gemv_bf16_lm_head_activation_staged_cache_policy_test_kernel<
      Rows, Columns, NvFp4TestCachePolicy::kStreaming>
      <<<kNvFp4M1RowQuadMaximumBlocks, kThreads, 0U, stream>>>(
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
launch_nvfp4_down_residual_norm_predecessor_test_unchecked(
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
      nvfp4_w4a16_down_residual_norm_activation_staged_predecessor_test_kernel<
          5'120U, 17'408U>,
      dim3{kNvFp4M1RowQuadMaximumBlocks}, dim3{kThreads}, arguments, 0U,
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
      nvfp4_w4a16_down_residual_norm_activation_staged_kernel,
      dim3{32U}, dim3{512U}, arguments, 0U, stream);
}

[[nodiscard]] cudaError_t
launch_nvfp4_down_residual_norm_cs_test_unchecked(
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
      nvfp4_w4a16_down_residual_norm_activation_staged_cs_test_kernel,
      dim3{32U}, dim3{512U}, arguments, 0U, stream);
}

[[nodiscard]] cudaError_t
launch_nvfp4_down_residual_norm_dead_raw_inline_residual_test_unchecked(
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
      nvfp4_w4a16_down_residual_norm_dead_raw_inline_residual_test_kernel,
      dim3{32U}, dim3{512U}, arguments, 0U, stream);
}

[[nodiscard]] cudaError_t
launch_nvfp4_down_residual_norm_scale6_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const scale6_sidecar,
    const unsigned int scale_base, const float weight_scale_2,
    const std::uint16_t* const activation,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const raw_down_output,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output,
    cudaStream_t const stream) noexcept {
  const std::uint8_t* packed_argument = packed_weights;
  const std::uint8_t* scale6_argument = scale6_sidecar;
  unsigned int base_argument = scale_base;
  float scale_argument = weight_scale_2;
  const std::uint16_t* activation_argument = activation;
  std::uint16_t* raw_argument = raw_down_output;
  const std::uint16_t* residual_left_argument = residual_left;
  const std::uint16_t* norm_weight_argument = norm_weight;
  float epsilon_argument = epsilon;
  std::uint16_t* residual_argument = residual_output;
  std::uint16_t* normalized_argument = normalized_output;
  void* arguments[] = {
      &packed_argument,      &scale6_argument,       &base_argument,
      &scale_argument,       &activation_argument,   &raw_argument,
      &residual_left_argument, &norm_weight_argument, &epsilon_argument,
      &residual_argument,    &normalized_argument,
  };
  return cudaLaunchCooperativeKernel(
      nvfp4_w4a16_down_residual_norm_activation_staged_scale6_test_kernel,
      dim3{32U}, dim3{512U}, arguments, 0U, stream);
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

void launch_nvfp4_residual_norm_gate_up_silu_cta_coarsened_512_unchecked(
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
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr unsigned int kCoarsenedThreads = 512U;
  nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_coarsened_512_kernel
      <<<kCoarsenedBlocks, kCoarsenedThreads, 0U, stream>>>(
          gate_packed_weights, gate_block_scales, gate_weight_scale_2,
          up_packed_weights, up_block_scales, up_weight_scale_2,
          residual_left, residual_right, norm_weight, epsilon,
          residual_output, gate_output, up_output);
}

void launch_nvfp4_residual_norm_gate_up_silu_dead_up_shared_pair_unchecked(
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
    std::uint16_t* const gate_output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr unsigned int kCoarsenedThreads = 512U;
  nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_kernel
      <<<kCoarsenedBlocks, kCoarsenedThreads, 0U, stream>>>(
          gate_packed_weights, gate_block_scales, gate_weight_scale_2,
          up_packed_weights, up_block_scales, up_weight_scale_2,
          residual_left, residual_right, norm_weight, epsilon,
          residual_output, gate_output);
}

void launch_nvfp4_residual_norm_gate_up_silu_dead_up_shared_pair_cg_test_unchecked(
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
    std::uint16_t* const gate_output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr unsigned int kCoarsenedThreads = 512U;
  nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_cache_policy_test_kernel<
      NvFp4TestCachePolicy::kCacheGlobal>
      <<<kCoarsenedBlocks, kCoarsenedThreads, 0U, stream>>>(
          gate_packed_weights, gate_block_scales, gate_weight_scale_2,
          up_packed_weights, up_block_scales, up_weight_scale_2,
          residual_left, residual_right, norm_weight, epsilon,
          residual_output, gate_output);
}

void launch_nvfp4_residual_norm_gate_up_silu_dead_up_shared_pair_cs_test_unchecked(
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
    std::uint16_t* const gate_output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr unsigned int kCoarsenedThreads = 512U;
  nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_cache_policy_test_kernel<
      NvFp4TestCachePolicy::kStreaming>
      <<<kCoarsenedBlocks, kCoarsenedThreads, 0U, stream>>>(
          gate_packed_weights, gate_block_scales, gate_weight_scale_2,
          up_packed_weights, up_block_scales, up_weight_scale_2,
          residual_left, residual_right, norm_weight, epsilon,
          residual_output, gate_output);
}

void launch_nvfp4_residual_norm_gate_up_silu_dead_up_scale6_test_unchecked(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_scale6_sidecar,
    const unsigned int gate_scale_base,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_scale6_sidecar,
    const unsigned int up_scale_base, const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kCoarsenedBlocks = 32U;
  constexpr unsigned int kCoarsenedThreads = 512U;
  nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_scale6_test_kernel
      <<<kCoarsenedBlocks, kCoarsenedThreads, 0U, stream>>>(
          gate_packed_weights, gate_scale6_sidecar, gate_scale_base,
          gate_weight_scale_2, up_packed_weights, up_scale6_sidecar,
          up_scale_base, up_weight_scale_2, residual_left, residual_right,
          norm_weight, epsilon, residual_output, gate_output);
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

void launch_nvfp4_lm_head_activation_staged_cs_test_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  if (rows == 248'320U && columns == 5'120U) {
    launch_nvfp4_lm_head_activation_staged_cs_test_instance_unchecked<
        248'320U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                          activation, output, stream);
  } else {
    launch_nvfp4_lm_head_activation_staged_cs_test_instance_unchecked<2'048U,
                                                                      512U>(
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

template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
void launch_fp8_small_m32_wmma_dual_resident_a_unchecked(
    const std::uint8_t* const weights, const float weight_scale,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel<
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

// Preserved scalar-store factorized specialization. The prior-production
// vector-store specialization below is the direct baseline for table-free
// E2M1; this scalar form remains test-addressable for the earlier store gate.
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

// Preserved prior-production factorized specialization that explicitly
// groups each four decoded BF16x2 words into one aligned shared-memory uint4
// store. It remains the same-cubin performance and exactness baseline.
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

// Exact M32 production specialization that retains the E4M3 scale codebook
// and WMMA sequence but constructs E2M1 pairs with PRMT instead of loading the
// shared pair table. Its direct launcher is also exposed to the test binary.
template <std::size_t kRows, std::size_t kColumns,
          unsigned int kSharedLeadingDimension = 72U>
void launch_nvfp4_small_m32_wmma_k64_dual_a_table_free_e2m1_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
      kRows, kColumns, kSharedLeadingDimension, true, true, 32U, true>
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
          unsigned int kSharedLeadingDimension = 72U>
void launch_nvfp4_small_m32_wmma_k64_dual_a_table_free_raw_weight_cp_async_unchecked(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, std::uint16_t* const output,
    cudaStream_t const stream) noexcept {
  constexpr unsigned int kOutputColumnsPerBlock = 128U;
  constexpr unsigned int kBlocks =
      static_cast<unsigned int>(kRows / kOutputColumnsPerBlock);
  nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_table_free_raw_weight_cp_async_kernel<
      kRows, kColumns, kSharedLeadingDimension>
      <<<kBlocks, kThreads, 0U, stream>>>(packed_weights, block_scales,
                                         weight_scale_2, activations, output);
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

int launch_sm87_nvfp4_table_free_e2m1_exhaustive_test_cuda(
    std::uint32_t* const candidate, std::uint32_t* const reference,
    void* const cuda_stream) noexcept {
  if (candidate == nullptr || reference == nullptr ||
      candidate == reference) {
    return invalid_value();
  }
  constexpr unsigned int kBlocks = 1'024U;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  nvfp4_table_free_e2m1_exhaustive_kernel<<<kBlocks, kThreads, 0U, stream>>>(
      candidate, reference);
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

int launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_bf16_cuda(
    const std::uint8_t* const sidecar_weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (rows != kFp8OutputProjectionRows ||
      columns != kFp8OutputProjectionColumns) {
    return invalid_value();
  }
  const int validation = validate_fp8_launch(
      sidecar_weights, weight_scale, activation, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  constexpr std::size_t kSidecarBytes =
      kFp8OutputProjectionRows * kFp8OutputProjectionColumns;
  constexpr std::size_t kActivationBytes =
      kFp8OutputProjectionColumns * sizeof(std::uint16_t);
  const bool aligned =
      pointer_is_aligned<alignof(uint4)>(sidecar_weights) &&
      pointer_is_aligned<alignof(std::uint64_t)>(activation) &&
      pointer_is_aligned<alignof(std::uint16_t)>(output);
  if (!aligned ||
      ranges_overlap(sidecar_weights, kSidecarBytes,
                     activation, kActivationBytes)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_row_quad_aosoa4_preswizzled_test_kernel
      <<<kFp8OutputProjectionAosoa4Blocks, kThreads, 0U, stream>>>(
          reinterpret_cast<const uint4*>(sidecar_weights), weight_scale,
          activation, rows, columns, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_cs_test_cuda(
    const std::uint8_t* const sidecar_weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (rows != kFp8OutputProjectionRows ||
      columns != kFp8OutputProjectionColumns) {
    return invalid_value();
  }
  const int validation = validate_fp8_launch(
      sidecar_weights, weight_scale, activation, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  constexpr std::size_t kSidecarBytes =
      kFp8OutputProjectionRows * kFp8OutputProjectionColumns;
  constexpr std::size_t kActivationBytes =
      kFp8OutputProjectionColumns * sizeof(std::uint16_t);
  const bool aligned =
      pointer_is_aligned<alignof(uint4)>(sidecar_weights) &&
      pointer_is_aligned<alignof(std::uint64_t)>(activation) &&
      pointer_is_aligned<alignof(std::uint16_t)>(output);
  if (!aligned ||
      ranges_overlap(sidecar_weights, kSidecarBytes,
                     activation, kActivationBytes)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_row_quad_aosoa4_preswizzled_cs_test_kernel
      <<<kFp8OutputProjectionAosoa4Blocks, kThreads, 0U, stream>>>(
          reinterpret_cast<const uint4*>(sidecar_weights), weight_scale,
          activation, rows, columns, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_pack_cuda(
    const std::uint8_t* const canonical_weights,
    std::uint8_t* const sidecar_weights, const std::size_t rows,
    const std::size_t columns, void* const cuda_stream) noexcept {
  if (rows != kFp8OutputProjectionRows ||
      columns != kFp8OutputProjectionColumns ||
      canonical_weights == nullptr || sidecar_weights == nullptr) {
    return invalid_value();
  }
  constexpr std::size_t kWeightBytes =
      kFp8OutputProjectionRows * kFp8OutputProjectionColumns;
  const bool aligned =
      pointer_is_aligned<alignof(std::uint32_t)>(canonical_weights) &&
      pointer_is_aligned<alignof(uint4)>(sidecar_weights);
  if (!aligned ||
      ranges_overlap(canonical_weights, kWeightBytes,
                     sidecar_weights, kWeightBytes)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_output_projection_aosoa4_pack_kernel
      <<<kFp8OutputProjectionRowQuads, kThreads, 0U, stream>>>(
          canonical_weights, reinterpret_cast<uint4*>(sidecar_weights));
  return static_cast<int>(cudaGetLastError());
}

// Production exact-M1 linear-attention input projection. The established
// QKV/Z work is unchanged; 24 otherwise-light tail CTAs also compute two
// adjacent BF16 A/B rows while sharing each activation load. Canonical FP8
// QKV/Z weight words use the selected evict-first streaming policy while all
// four tensors retain their predecessor arithmetic and publication points.
int launch_sm87_fp8_w8a16_gemv_qkv_z_bf16_ab_pair_cuda(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t ab_rows,
    const std::size_t columns, std::uint16_t* const qkv_output,
    std::uint16_t* const z_output, std::uint16_t* const a_output,
    std::uint16_t* const b_output, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_qkv_z_bf16_ab_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, qkv_rows, z_rows, ab_rows, columns, qkv_output,
      z_output, a_output, b_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_qkv_z_bf16_ab_launch_is_aligned(
          qkv_weights, z_weights, a_weights, b_weights, activation,
          qkv_output, z_output, a_output, b_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_qkv_z_bf16_ab_pair_tail_composite_cs_test_unchecked(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, columns, qkv_output, z_output, a_output,
      b_output, stream);
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

int launch_sm87_fp8_w8a16_m1_row_quad_aosoa4_preswizzled_test_cuda(
    const std::uint8_t* const aosoa4_preswizzled_weights,
    const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  return launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_bf16_cuda(
      aosoa4_preswizzled_weights, weight_scale, activation, rows, columns,
      output, cuda_stream);
}

int launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_resident_grid64_test_cuda(
    const std::uint8_t* const sidecar_weights, const float weight_scale,
    const std::uint16_t* const activation, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (rows != kFp8OutputProjectionRows ||
      columns != kFp8OutputProjectionColumns) {
    return invalid_value();
  }
  const int validation = validate_fp8_launch(
      sidecar_weights, weight_scale, activation, rows, columns, output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  constexpr std::size_t kSidecarBytes =
      kFp8OutputProjectionRows * kFp8OutputProjectionColumns;
  constexpr std::size_t kActivationBytes =
      kFp8OutputProjectionColumns * sizeof(std::uint16_t);
  const bool aligned =
      pointer_is_aligned<alignof(uint4)>(sidecar_weights) &&
      pointer_is_aligned<alignof(std::uint64_t)>(activation) &&
      pointer_is_aligned<alignof(std::uint16_t)>(output);
  if (!aligned ||
      ranges_overlap(sidecar_weights, kSidecarBytes,
                     activation, kActivationBytes)) {
    return invalid_value();
  }

  constexpr unsigned int kResidentGridBlocks = 64U;
  static_assert((kFp8OutputProjectionRowQuads % kResidentGridBlocks) == 0U);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_row_quad_aosoa4_preswizzled_test_kernel
      <<<kResidentGridBlocks, kThreads, 0U, stream>>>(
          reinterpret_cast<const uint4*>(sidecar_weights), weight_scale,
          activation, rows, columns, output);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_residual_epilogue_test_cuda(
    const std::uint8_t* const aosoa4_preswizzled_weights,
    const float weight_scale, const std::uint16_t* const activation,
    const std::uint16_t* const residual_left, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const residual_output,
    void* const cuda_stream) noexcept {
  if (rows != kFp8OutputProjectionRows ||
      columns != kFp8OutputProjectionColumns || residual_left == nullptr) {
    return invalid_value();
  }
  const int validation = validate_fp8_launch(
      aosoa4_preswizzled_weights, weight_scale, activation, rows, columns,
      residual_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  constexpr std::size_t kWeightBytes =
      kFp8OutputProjectionRows * kFp8OutputProjectionColumns;
  constexpr std::size_t kActivationBytes =
      kFp8OutputProjectionColumns * sizeof(std::uint16_t);
  constexpr std::size_t kResidualBytes =
      kFp8OutputProjectionRows * sizeof(std::uint16_t);
  const bool aligned =
      pointer_is_aligned<alignof(uint4)>(aosoa4_preswizzled_weights) &&
      pointer_is_aligned<alignof(std::uint64_t)>(activation) &&
      pointer_is_aligned<alignof(std::uint16_t)>(residual_left) &&
      pointer_is_aligned<alignof(std::uint16_t)>(residual_output);
  if (!aligned ||
      byte_range_overflows(residual_left, kResidualBytes) ||
      ranges_overlap(residual_output, kResidualBytes, residual_left,
                     kResidualBytes) ||
      ranges_overlap(residual_left, kResidualBytes,
                     aosoa4_preswizzled_weights, kWeightBytes) ||
      ranges_overlap(residual_left, kResidualBytes, activation,
                     kActivationBytes)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_row_quad_aosoa4_residual_epilogue_test_kernel
      <<<kFp8OutputProjectionAosoa4Blocks, kThreads, 0U, stream>>>(
          reinterpret_cast<const uint4*>(aosoa4_preswizzled_weights),
          weight_scale, activation, residual_left, residual_output);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_row_quad_aosoa4_preswizzled_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_row_quad_aosoa4_preswizzled_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_row_quad_aosoa4_preswizzled_test_kernel,
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

int query_sm87_fp8_w8a16_m1_output_projection_aosoa4_resident_grid64_resources_test_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  return query_sm87_fp8_w8a16_m1_row_quad_aosoa4_preswizzled_resources_test_cuda(
      registers_per_thread, static_shared_bytes, local_bytes,
      maximum_threads_per_block, active_blocks_per_sm);
}

int query_sm87_fp8_w8a16_m1_output_projection_aosoa4_cs_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_row_quad_aosoa4_preswizzled_cs_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_row_quad_aosoa4_preswizzled_cs_test_kernel,
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

int query_sm87_fp8_w8a16_m1_output_projection_aosoa4_residual_epilogue_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_row_quad_aosoa4_residual_epilogue_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_row_quad_aosoa4_residual_epilogue_test_kernel,
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

int launch_sm87_fp8_w8a16_m1_qkv_z_bf16_ab_pair_tail_composite_test_cuda(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t ab_rows,
    const std::size_t columns, std::uint16_t* const qkv_output,
    std::uint16_t* const z_output, std::uint16_t* const a_output,
    std::uint16_t* const b_output, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_qkv_z_bf16_ab_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, qkv_rows, z_rows, ab_rows, columns, qkv_output,
      z_output, a_output, b_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_qkv_z_bf16_ab_launch_is_aligned(
          qkv_weights, z_weights, a_weights, b_weights, activation,
          qkv_output, z_output, a_output, b_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_qkv_z_bf16_ab_pair_tail_composite_unchecked(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, columns, qkv_output, z_output, a_output,
      b_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_qkv_z_bf16_ab_pair_tail_composite_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_kernel,
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

int launch_sm87_fp8_w8a16_m1_qkv_z_bf16_ab_pair_tail_composite_register_lookahead_test_cuda(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t ab_rows,
    const std::size_t columns, std::uint16_t* const qkv_output,
    std::uint16_t* const z_output, std::uint16_t* const a_output,
    std::uint16_t* const b_output, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_qkv_z_bf16_ab_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, qkv_rows, z_rows, ab_rows, columns, qkv_output,
      z_output, a_output, b_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_qkv_z_bf16_ab_launch_is_aligned(
          qkv_weights, z_weights, a_weights, b_weights, activation,
          qkv_output, z_output, a_output, b_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_qkv_z_bf16_ab_pair_tail_composite_register_lookahead_test_unchecked(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, columns, qkv_output, z_output, a_output,
      b_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_qkv_z_bf16_ab_pair_tail_composite_register_lookahead_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_register_lookahead_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_register_lookahead_test_kernel,
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

int launch_sm87_fp8_w8a16_m1_qkv_z_bf16_ab_pair_tail_composite_causal_conv_epilogue_test_cuda(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t ab_rows,
    const std::size_t columns, std::uint16_t* const qkv_output,
    std::uint16_t* const z_output, std::uint16_t* const a_output,
    std::uint16_t* const b_output,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history, void* const cuda_stream) noexcept {
  const int validation =
      validate_fp8_qkv_z_bf16_ab_causal_conv_epilogue_launch(
          qkv_weights, qkv_weight_scale, z_weights, z_weight_scale,
          a_weights, b_weights, activation, qkv_rows, z_rows, ab_rows,
          columns, qkv_output, z_output, a_output, b_output, conv_weight,
          history);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_qkv_z_bf16_ab_causal_conv_epilogue_launch_is_aligned(
          qkv_weights, z_weights, a_weights, b_weights, activation,
          qkv_output, z_output, a_output, b_output, conv_weight, history)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_qkv_z_bf16_ab_pair_tail_composite_causal_conv_epilogue_test_unchecked(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, columns, qkv_output, z_output, a_output,
      b_output, conv_weight, history, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_qkv_z_bf16_ab_pair_tail_composite_causal_conv_epilogue_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_causal_conv_epilogue_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_causal_conv_epilogue_test_kernel,
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

int launch_sm87_fp8_w8a16_m1_qkv_z_bf16_ab_pair_tail_composite_cs_test_cuda(
    const std::uint8_t* const qkv_weights, const float qkv_weight_scale,
    const std::uint8_t* const z_weights, const float z_weight_scale,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const activation, const std::size_t qkv_rows,
    const std::size_t z_rows, const std::size_t ab_rows,
    const std::size_t columns, std::uint16_t* const qkv_output,
    std::uint16_t* const z_output, std::uint16_t* const a_output,
    std::uint16_t* const b_output, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_qkv_z_bf16_ab_launch(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, qkv_rows, z_rows, ab_rows, columns, qkv_output,
      z_output, a_output, b_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_qkv_z_bf16_ab_launch_is_aligned(
          qkv_weights, z_weights, a_weights, b_weights, activation,
          qkv_output, z_output, a_output, b_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_qkv_z_bf16_ab_pair_tail_composite_cs_test_unchecked(
      qkv_weights, qkv_weight_scale, z_weights, z_weight_scale, a_weights,
      b_weights, activation, columns, qkv_output, z_output, a_output,
      b_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_qkv_z_bf16_ab_pair_tail_composite_cs_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_cs_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_qkv_z_bf16_ab_pair_tail_composite_cs_test_kernel,
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

// Production exact-shape full-attention Q + K/V projection. The fixed launch
// keeps the established 2,048-CTA topology while alternating two reduction
// scratch slots across each CTA's at-most-two ordered logical bodies.
int launch_sm87_fp8_w8a16_gemv_q_kv_bf16_cuda(
    const std::uint8_t* const q_weights, const float q_weight_scale,
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation, const std::size_t q_rows,
    const std::size_t kv_rows, const std::size_t columns,
    std::uint16_t* const q_output,
    std::uint16_t* const key_output, std::uint16_t* const value_output,
    void* const cuda_stream) noexcept {
  const int validation = validate_fp8_q_kv_launch(
      q_weights, q_weight_scale, key_weights, key_weight_scale,
      value_weights, value_weight_scale, activation, q_rows, kv_rows,
      columns, q_output, key_output, value_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_q_kv_launch_is_aligned(q_weights, key_weights, value_weights,
                                  activation, q_output, key_output,
                                  value_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_q_kv_two_phase_reduction_scratch_ping_pong_kernel
      <<<kFp8FullAttentionBlocks, kThreads, 0U, stream>>>(
          q_weights, q_weight_scale, key_weights, key_weight_scale,
          value_weights, value_weight_scale, activation, q_output,
          key_output, value_output);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_q_kv_aosoa4_preswizzled_test_cuda(
    const std::uint8_t* const q_sidecar, const float q_weight_scale,
    const std::uint8_t* const kv_sidecar, const float key_weight_scale,
    const float value_weight_scale,
    const std::uint16_t* const activation, const std::size_t q_rows,
    const std::size_t kv_rows, const std::size_t columns,
    std::uint16_t* const q_output, std::uint16_t* const key_output,
    std::uint16_t* const value_output, void* const cuda_stream) noexcept {
  const int validation =
      validate_fp8_q_kv_aosoa4_preswizzled_test_launch(
          q_sidecar, q_weight_scale, kv_sidecar, key_weight_scale,
          value_weight_scale, activation, q_rows, kv_rows, columns,
          q_output, key_output, value_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool aligned =
      pointer_is_aligned<alignof(uint4)>(q_sidecar) &&
      pointer_is_aligned<alignof(uint4)>(kv_sidecar) &&
      pointer_is_aligned<alignof(std::uint64_t)>(activation) &&
      pointer_is_aligned<alignof(std::uint16_t)>(q_output) &&
      pointer_is_aligned<alignof(std::uint16_t)>(key_output) &&
      pointer_is_aligned<alignof(std::uint16_t)>(value_output);
  if (!aligned) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_test_kernel
      <<<kFp8FullAttentionBlocks, kThreads, 0U, stream>>>(
          reinterpret_cast<const uint4*>(q_sidecar), q_weight_scale,
          reinterpret_cast<const uint4*>(kv_sidecar), key_weight_scale,
          value_weight_scale, activation, q_output, key_output,
          value_output);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_q_kv_aosoa4_preswizzled_cta512_test_cuda(
    const std::uint8_t* const q_sidecar, const float q_weight_scale,
    const std::uint8_t* const kv_sidecar, const float key_weight_scale,
    const float value_weight_scale,
    const std::uint16_t* const activation, const std::size_t q_rows,
    const std::size_t kv_rows, const std::size_t columns,
    std::uint16_t* const q_output, std::uint16_t* const key_output,
    std::uint16_t* const value_output, void* const cuda_stream) noexcept {
  const int validation =
      validate_fp8_q_kv_aosoa4_preswizzled_test_launch(
          q_sidecar, q_weight_scale, kv_sidecar, key_weight_scale,
          value_weight_scale, activation, q_rows, kv_rows, columns,
          q_output, key_output, value_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool aligned =
      pointer_is_aligned<alignof(uint4)>(q_sidecar) &&
      pointer_is_aligned<alignof(uint4)>(kv_sidecar) &&
      pointer_is_aligned<alignof(std::uint64_t)>(activation) &&
      pointer_is_aligned<alignof(std::uint16_t)>(q_output) &&
      pointer_is_aligned<alignof(std::uint16_t)>(key_output) &&
      pointer_is_aligned<alignof(std::uint16_t)>(value_output);
  if (!aligned) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_cta512_test_kernel
      <<<kFp8FullAttentionCta512Blocks,
         kFp8FullAttentionCta512Threads, 0U, stream>>>(
          reinterpret_cast<const uint4*>(q_sidecar), q_weight_scale,
          reinterpret_cast<const uint4*>(kv_sidecar), key_weight_scale,
          value_weight_scale, activation, q_output, key_output,
          value_output);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_q_kv_aosoa4_preswizzled_pack_test_cuda(
    const std::uint8_t* const canonical_q,
    const std::uint8_t* const canonical_key,
    const std::uint8_t* const canonical_value,
    const std::size_t q_rows, const std::size_t kv_rows,
    const std::size_t columns, std::uint8_t* const q_sidecar,
    std::uint8_t* const kv_sidecar, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_q_kv_aosoa4_pack_test_launch(
      canonical_q, canonical_key, canonical_value, q_rows, kv_rows,
      columns, q_sidecar, kv_sidecar);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool aligned =
      pointer_is_aligned<alignof(std::uint32_t)>(canonical_q) &&
      pointer_is_aligned<alignof(std::uint32_t)>(canonical_key) &&
      pointer_is_aligned<alignof(std::uint32_t)>(canonical_value) &&
      pointer_is_aligned<alignof(uint4)>(q_sidecar) &&
      pointer_is_aligned<alignof(uint4)>(kv_sidecar);
  if (!aligned) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  fp8_w8a16_q_kv_aosoa4_preswizzled_pack_test_kernel
      <<<kFp8FullAttentionAosoa4PackBlocks, kThreads, 0U, stream>>>(
          canonical_q, canonical_key, canonical_value,
          reinterpret_cast<uint4*>(q_sidecar),
          reinterpret_cast<uint4*>(kv_sidecar));
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_q_kv_aosoa4_preswizzled_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_test_kernel,
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

int query_sm87_fp8_w8a16_m1_q_kv_aosoa4_preswizzled_cta512_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_cta512_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_q_kv_aosoa4_preswizzled_cta512_test_kernel,
      static_cast<int>(kFp8FullAttentionCta512Threads), 0U);
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
      &attributes,
      fp8_w8a16_gemv_bf16_q_kv_two_phase_reduction_scratch_ping_pong_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_q_kv_two_phase_reduction_scratch_ping_pong_kernel,
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

int launch_sm87_fp8_w8a16_m1_q_kv_reduction_scratch_ping_pong_test_cuda(
    const std::uint8_t* const q_weights, const float q_weight_scale,
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation, const std::size_t q_rows,
    const std::size_t kv_rows, const std::size_t columns,
    std::uint16_t* const q_output, std::uint16_t* const key_output,
    std::uint16_t* const value_output, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_q_kv_launch(
      q_weights, q_weight_scale, key_weights, key_weight_scale,
      value_weights, value_weight_scale, activation, q_rows, kv_rows,
      columns, q_output, key_output, value_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_q_kv_launch_is_aligned(q_weights, key_weights, value_weights,
                                  activation, q_output, key_output,
                                  value_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_q_kv_reduction_scratch_ping_pong_test_unchecked(
      q_weights, q_weight_scale, key_weights, key_weight_scale,
      value_weights, value_weight_scale, activation, q_output, key_output,
      value_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_fp8_w8a16_m1_q_kv_tail_barrier_test_cuda(
    const std::uint8_t* const q_weights, const float q_weight_scale,
    const std::uint8_t* const key_weights, const float key_weight_scale,
    const std::uint8_t* const value_weights, const float value_weight_scale,
    const std::uint16_t* const activation, const std::size_t q_rows,
    const std::size_t kv_rows, const std::size_t columns,
    std::uint16_t* const q_output, std::uint16_t* const key_output,
    std::uint16_t* const value_output, void* const cuda_stream) noexcept {
  const int validation = validate_fp8_q_kv_launch(
      q_weights, q_weight_scale, key_weights, key_weight_scale,
      value_weights, value_weight_scale, activation, q_rows, kv_rows,
      columns, q_output, key_output, value_output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!fp8_q_kv_launch_is_aligned(q_weights, key_weights, value_weights,
                                  activation, q_output, key_output,
                                  value_output)) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_fp8_q_kv_tail_barrier_test_unchecked(
      q_weights, q_weight_scale, key_weights, key_weight_scale,
      value_weights, value_weight_scale, activation, q_output, key_output,
      value_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_m1_q_kv_tail_barrier_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_q_kv_two_phase_tail_barrier_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_q_kv_two_phase_tail_barrier_test_kernel,
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

int query_sm87_fp8_w8a16_m1_q_kv_reduction_scratch_ping_pong_resources_test_cuda(
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
      fp8_w8a16_gemv_bf16_q_kv_two_phase_reduction_scratch_ping_pong_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      fp8_w8a16_gemv_bf16_q_kv_two_phase_reduction_scratch_ping_pong_kernel,
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

// Test-only fixed-M32 single-resident-A predecessor. It is intentionally
// restricted to the four exact production FP8 WMMA shapes and is not
// reachable from public dispatch.
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

// Test-only direct replay entry for the production dual-resident-A M32
// kernel. It remains separate from public dispatch and accepts only the four
// promoted FP8 checkpoint shapes.
int launch_sm87_fp8_w8a16_small_m32_wmma_dual_resident_a_test_cuda(
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
    launch_fp8_small_m32_wmma_dual_resident_a_unchecked<10'240U, 5'120U,
                                                        72U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 5'120U) {
    launch_fp8_small_m32_wmma_dual_resident_a_unchecked<5'120U, 6'144U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 6'144U) {
    launch_fp8_small_m32_wmma_dual_resident_a_unchecked<6'144U, 5'120U, 72U>(
        weights, weight_scale, activations, output, stream);
  } else if (rows == 12'288U) {
    launch_fp8_small_m32_wmma_dual_resident_a_unchecked<12'288U, 5'120U,
                                                        72U>(
        weights, weight_scale, activations, output, stream);
  } else {
    return invalid_value();
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_fp8_w8a16_small_m32_wmma_fixed_shape_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_fp8_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 10'240U) {
    status = cudaFuncGetAttributes(
        &attributes,
        fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel<10'240U,
                                                                  5'120U,
                                                                  72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel<10'240U,
                                                                    5'120U,
                                                                    72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else if (rows == 5'120U) {
    status = cudaFuncGetAttributes(
        &attributes,
        fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel<5'120U, 6'144U,
                                                                  72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel<5'120U,
                                                                    6'144U,
                                                                    72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else if (rows == 6'144U) {
    status = cudaFuncGetAttributes(
        &attributes,
        fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel<6'144U, 5'120U,
                                                                  72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel<6'144U,
                                                                    5'120U,
                                                                    72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else if (rows == 12'288U) {
    status = cudaFuncGetAttributes(
        &attributes,
        fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel<12'288U,
                                                                  5'120U,
                                                                  72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          fp8_w8a16_small_m32_gemm_bf16_wmma_fixed_shape_kernel<12'288U,
                                                                    5'120U,
                                                                    72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    return invalid_value();
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

int query_sm87_fp8_w8a16_small_m32_wmma_dual_resident_a_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (!use_fp8_m16_wmma_fixed_shape(rows, columns) ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaSuccess;
  int active_blocks = 0;
  if (rows == 10'240U) {
    status = cudaFuncGetAttributes(
        &attributes,
        fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel<
            10'240U, 5'120U, 72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel<
              10'240U, 5'120U, 72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else if (rows == 5'120U) {
    status = cudaFuncGetAttributes(
        &attributes,
        fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel<
            5'120U, 6'144U, 72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel<
              5'120U, 6'144U, 72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else if (rows == 6'144U) {
    status = cudaFuncGetAttributes(
        &attributes,
        fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel<
            6'144U, 5'120U, 72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel<
              6'144U, 5'120U, 72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else if (rows == 12'288U) {
    status = cudaFuncGetAttributes(
        &attributes,
        fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel<
            12'288U, 5'120U, 72U>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          fp8_w8a16_small_m32_gemm_bf16_wmma_dual_resident_a_kernel<
              12'288U, 5'120U, 72U>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    return invalid_value();
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
  launch_nvfp4_residual_norm_gate_up_silu_cta_coarsened_512_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, residual_output, gate_output,
      up_output, stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_dead_up_bf16_cuda(
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
    std::uint16_t* const gate_output, std::uint16_t* const up_workspace,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_residual_norm_gate_up_silu_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, rows, columns, residual_output,
      gate_output, up_workspace);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape = rows == 17'408U && columns == 5'120U;
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
      (reinterpret_cast<std::uintptr_t>(up_workspace) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_residual_norm_gate_up_silu_dead_up_shared_pair_cs_test_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, residual_output, gate_output,
      stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_dead_up_default_test_cuda(
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
    std::uint16_t* const gate_output, std::uint16_t* const up_workspace,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_residual_norm_gate_up_silu_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, rows, columns, residual_output,
      gate_output, up_workspace);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape = rows == 17'408U && columns == 5'120U;
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
      (reinterpret_cast<std::uintptr_t>(up_workspace) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_residual_norm_gate_up_silu_dead_up_shared_pair_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, residual_output, gate_output,
      stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_dead_up_cg_test_cuda(
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
    std::uint16_t* const gate_output, std::uint16_t* const up_workspace,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_residual_norm_gate_up_silu_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, rows, columns, residual_output,
      gate_output, up_workspace);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape = rows == 17'408U && columns == 5'120U;
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
      (reinterpret_cast<std::uintptr_t>(up_workspace) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_residual_norm_gate_up_silu_dead_up_shared_pair_cg_test_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, residual_output, gate_output,
      stream);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_dead_up_cs_test_cuda(
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
    std::uint16_t* const gate_output, std::uint16_t* const up_workspace,
    void* const cuda_stream) noexcept {
  const int validation = validate_nvfp4_residual_norm_gate_up_silu_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, rows, columns, residual_output,
      gate_output, up_workspace);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool supported_shape = rows == 17'408U && columns == 5'120U;
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
      (reinterpret_cast<std::uintptr_t>(up_workspace) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_residual_norm_gate_up_silu_dead_up_shared_pair_cs_test_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, residual_output, gate_output,
      stream);
  return static_cast<int>(cudaGetLastError());
}

// Test-only six-bit block-scale sidecar entry point. Production, default
// rollback, and streaming-cache ABIs remain unchanged and never select it.
int launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_dead_up_scale6_test_cuda(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_scale6_sidecar,
    const unsigned int gate_scale_base,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_scale6_sidecar,
    const unsigned int up_scale_base, const float up_weight_scale_2,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const residual_right,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_workspace,
    void* const cuda_stream) noexcept {
  const int validation =
      validate_nvfp4_scale6_residual_norm_gate_up_silu_launch(
          gate_packed_weights, gate_scale6_sidecar, gate_scale_base,
          gate_weight_scale_2, up_packed_weights, up_scale6_sidecar,
          up_scale_base, up_weight_scale_2, residual_left, residual_right,
          norm_weight, epsilon, rows, columns, residual_output, gate_output,
          up_workspace);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  constexpr std::size_t kRows = 17'408U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::uintptr_t kSidecarAlignment = 32U;
  const bool supported_shape = rows == kRows && columns == kColumns;
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(gate_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(gate_scale6_sidecar) %
       kSidecarAlignment) == 0U &&
      (reinterpret_cast<std::uintptr_t>(up_scale6_sidecar) %
       kSidecarAlignment) == 0U &&
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
      (reinterpret_cast<std::uintptr_t>(up_workspace) %
       alignof(std::uint16_t)) == 0U;
  if (!supported_shape || !aligned) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_nvfp4_residual_norm_gate_up_silu_dead_up_scale6_test_unchecked(
      gate_packed_weights, gate_scale6_sidecar, gate_scale_base,
      gate_weight_scale_2, up_packed_weights, up_scale6_sidecar,
      up_scale_base, up_weight_scale_2, residual_left, residual_right,
      norm_weight, epsilon, residual_output, gate_output, stream);
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
      launch_nvfp4_down_residual_norm_cs_test_unchecked(
          packed_weights, block_scales, weight_scale_2, activation,
          residual_left, norm_weight, epsilon, raw_down_output,
          residual_output, normalized_output, stream);
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_down_residual_norm_default_test_cuda(
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

int launch_sm87_nvfp4_w4a16_down_residual_norm_cs_test_cuda(
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
      launch_nvfp4_down_residual_norm_cs_test_unchecked(
          packed_weights, block_scales, weight_scale_2, activation,
          residual_left, norm_weight, epsilon, raw_down_output,
          residual_output, normalized_output, stream);
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_down_residual_norm_dead_raw_inline_residual_test_cuda(
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
      launch_nvfp4_down_residual_norm_dead_raw_inline_residual_test_unchecked(
          packed_weights, block_scales, weight_scale_2, activation,
          residual_left, norm_weight, epsilon, raw_down_output,
          residual_output, normalized_output, stream);
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_down_residual_norm_scale6_test_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const scale6_sidecar,
    const unsigned int scale_base, const float weight_scale_2,
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
  constexpr std::size_t kActivationBytes =
      kColumns * sizeof(std::uint16_t);
  constexpr std::size_t kOutputBytes = kRows * sizeof(std::uint16_t);
  constexpr std::size_t kScale6Bytes =
      (kRows / 4U) * (kColumns / kNvFp4Scale6ColumnsPerTile) *
      kNvFp4Scale6WordsPerRowQuadTile * sizeof(std::uint32_t);
  if (rows != kRows || columns != kColumns) {
    return invalid_value();
  }
  const int validation = validate_nvfp4_scale6_launch(
      packed_weights, scale6_sidecar, scale_base, weight_scale_2,
      activation, rows, columns, raw_down_output);
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
  constexpr std::uintptr_t kSidecarAlignment = 32U;
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(packed_weights) %
       alignof(std::uint32_t)) == 0U &&
      (reinterpret_cast<std::uintptr_t>(scale6_sidecar) %
       kSidecarAlignment) == 0U &&
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
      ranges_overlap(residual_output, kOutputBytes, scale6_sidecar,
                     kScale6Bytes) ||
      ranges_overlap(residual_output, kOutputBytes, activation,
                     kActivationBytes) ||
      ranges_overlap(normalized_output, kOutputBytes, packed_weights,
                     kPackedBytes) ||
      ranges_overlap(normalized_output, kOutputBytes, scale6_sidecar,
                     kScale6Bytes) ||
      ranges_overlap(normalized_output, kOutputBytes, activation,
                     kActivationBytes)) {
    return invalid_value();
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  const cudaError_t launch_status =
      launch_nvfp4_down_residual_norm_scale6_test_unchecked(
          packed_weights, scale6_sidecar, scale_base, weight_scale_2,
          activation, residual_left, norm_weight, epsilon, raw_down_output,
          residual_output, normalized_output, stream);
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_w4a16_down_residual_norm_scale6_bf16_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const scale6_sidecar,
    const unsigned int scale_base, const float weight_scale_2,
    const std::uint16_t* const activation,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const raw_down_output,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output,
    void* const cuda_stream) noexcept {
  // Keep the retained test ABI and the production route on the exact same
  // validation and cooperative-kernel Function.
  return launch_sm87_nvfp4_w4a16_down_residual_norm_scale6_test_cuda(
      packed_weights, scale6_sidecar, scale_base, weight_scale_2, activation,
      residual_left, norm_weight, epsilon, rows, columns, raw_down_output,
      residual_output, normalized_output, cuda_stream);
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
      nvfp4_w4a16_down_residual_norm_activation_staged_cs_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_down_residual_norm_activation_staged_cs_test_kernel, 512,
      0U);
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

int query_sm87_nvfp4_w4a16_m1_down_residual_norm_default_resources_test_cuda(
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
      &attributes, nvfp4_w4a16_down_residual_norm_activation_staged_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, nvfp4_w4a16_down_residual_norm_activation_staged_kernel,
      512, 0U);
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

int query_sm87_nvfp4_w4a16_m1_down_residual_norm_cs_resources_test_cuda(
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
      nvfp4_w4a16_down_residual_norm_activation_staged_cs_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_down_residual_norm_activation_staged_cs_test_kernel, 512,
      0U);
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

int query_sm87_nvfp4_w4a16_m1_down_residual_norm_dead_raw_inline_residual_resources_test_cuda(
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
      nvfp4_w4a16_down_residual_norm_dead_raw_inline_residual_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_down_residual_norm_dead_raw_inline_residual_test_kernel,
      512, 0U);
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

int query_sm87_nvfp4_w4a16_m1_down_residual_norm_scale6_resources_test_cuda(
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
      nvfp4_w4a16_down_residual_norm_activation_staged_scale6_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_down_residual_norm_activation_staged_scale6_test_kernel,
      512, 0U);
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

int query_sm87_nvfp4_w4a16_m1_down_residual_norm_scale6_resources_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  return
      query_sm87_nvfp4_w4a16_m1_down_residual_norm_scale6_resources_test_cuda(
          registers_per_thread, static_shared_bytes, local_bytes,
          maximum_threads_per_block, active_blocks_per_sm);
}

int launch_sm87_nvfp4_w4a16_down_residual_norm_predecessor_test_cuda(
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
      launch_nvfp4_down_residual_norm_predecessor_test_unchecked(
          packed_weights, block_scales, weight_scale_2, activation,
          residual_left, norm_weight, epsilon, raw_down_output,
          residual_output, normalized_output, stream);
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_down_residual_norm_predecessor_resources_test_cuda(
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
      nvfp4_w4a16_down_residual_norm_activation_staged_predecessor_test_kernel<
          5'120U, 17'408U>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_down_residual_norm_activation_staged_predecessor_test_kernel<
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

int launch_sm87_nvfp4_w4a16_gemv_bf16_lm_head_activation_staged_cs_test_cuda(
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
  launch_nvfp4_lm_head_activation_staged_cs_test_unchecked(
      packed_weights, block_scales, weight_scale_2, activation, rows, columns,
      output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_m1_lm_head_activation_staged_cs_resources_test_cuda(
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
      nvfp4_w4a16_gemv_bf16_lm_head_activation_staged_cache_policy_test_kernel<
          248'320U, 5'120U, NvFp4TestCachePolicy::kStreaming>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_lm_head_activation_staged_cache_policy_test_kernel<
          248'320U, 5'120U, NvFp4TestCachePolicy::kStreaming>,
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

int launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_coarsened_512_test_cuda(
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
  const bool supported_shape = rows == 17'408U && columns == 5'120U;
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
  launch_nvfp4_residual_norm_gate_up_silu_cta_coarsened_512_unchecked(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual_left,
      residual_right, norm_weight, epsilon, residual_output, gate_output,
      up_output, stream);
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

int launch_sm87_nvfp4_w4a16_prerounded_residual_norm_gate_up_silu_dead_up_test_cuda(
    const std::uint8_t* const gate_packed_weights,
    const std::uint8_t* const gate_block_scales,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_packed_weights,
    const std::uint8_t* const up_block_scales,
    const float up_weight_scale_2,
    const std::uint16_t* const residual,
    const std::uint16_t* const norm_weight, const float epsilon,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_workspace,
    void* const cuda_stream) noexcept {
  const int pair_validation = validate_nvfp4_gate_up_pair_launch(
      gate_packed_weights, gate_block_scales, gate_weight_scale_2,
      up_packed_weights, up_block_scales, up_weight_scale_2, residual, rows,
      columns, gate_output, up_workspace);
  if (pair_validation != static_cast<int>(cudaSuccess)) {
    return pair_validation;
  }
  constexpr std::size_t kInputBytes =
      kFp8OutputProjectionRows * sizeof(std::uint16_t);
  constexpr std::size_t kProjectionOutputBytes =
      17'408U * sizeof(std::uint16_t);
  if (rows != 17'408U || columns != kFp8OutputProjectionRows ||
      norm_weight == nullptr || !std::isfinite(epsilon) || epsilon <= 0.0F ||
      byte_range_overflows(norm_weight, kInputBytes) ||
      ranges_overlap(gate_output, kProjectionOutputBytes, norm_weight,
                     kInputBytes) ||
      ranges_overlap(up_workspace, kProjectionOutputBytes, norm_weight,
                     kInputBytes)) {
    return invalid_value();
  }
  const bool aligned =
      pointer_is_aligned<alignof(std::uint32_t)>(gate_packed_weights) &&
      pointer_is_aligned<alignof(std::uint32_t)>(up_packed_weights) &&
      pointer_is_aligned<alignof(std::uint16_t)>(residual) &&
      pointer_is_aligned<alignof(std::uint16_t)>(norm_weight) &&
      pointer_is_aligned<alignof(std::uint16_t)>(gate_output) &&
      pointer_is_aligned<alignof(std::uint16_t)>(up_workspace);
  if (!aligned) {
    return invalid_value();
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  nvfp4_w4a16_gemv_bf16_prerounded_residual_norm_gate_up_silu_dead_up_shared_pair_test_kernel
      <<<32U, 512U, 0U, stream>>>(
          gate_packed_weights, gate_block_scales, gate_weight_scale_2,
          up_packed_weights, up_block_scales, up_weight_scale_2, residual,
          norm_weight, epsilon, gate_output);
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

int query_sm87_nvfp4_w4a16_m1_residual_norm_gate_up_silu_coarsened_512_resources_test_cuda(
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
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_coarsened_512_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_coarsened_512_kernel,
      512, 0U);
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

int query_sm87_nvfp4_w4a16_m1_residual_norm_gate_up_silu_dead_up_shared_pair_resources_test_cuda(
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
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_cache_policy_test_kernel<
          NvFp4TestCachePolicy::kStreaming>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_cache_policy_test_kernel<
          NvFp4TestCachePolicy::kStreaming>,
      512, 0U);
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

int query_sm87_nvfp4_w4a16_m1_residual_norm_gate_up_silu_dead_up_default_resources_test_cuda(
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
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_kernel,
      512, 0U);
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

int query_sm87_nvfp4_w4a16_m1_residual_norm_gate_up_silu_dead_up_cg_resources_test_cuda(
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
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_cache_policy_test_kernel<
          NvFp4TestCachePolicy::kCacheGlobal>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_cache_policy_test_kernel<
          NvFp4TestCachePolicy::kCacheGlobal>,
      512, 0U);
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

int query_sm87_nvfp4_w4a16_m1_residual_norm_gate_up_silu_dead_up_cs_resources_test_cuda(
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
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_cache_policy_test_kernel<
          NvFp4TestCachePolicy::kStreaming>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_shared_pair_cache_policy_test_kernel<
          NvFp4TestCachePolicy::kStreaming>,
      512, 0U);
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

int query_sm87_nvfp4_w4a16_m1_residual_norm_gate_up_silu_dead_up_scale6_resources_test_cuda(
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
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_scale6_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_residual_norm_gate_up_silu_dead_up_scale6_test_kernel,
      512, 0U);
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

int query_sm87_nvfp4_w4a16_m1_prerounded_residual_norm_gate_up_silu_dead_up_resources_test_cuda(
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
      nvfp4_w4a16_gemv_bf16_prerounded_residual_norm_gate_up_silu_dead_up_shared_pair_test_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      nvfp4_w4a16_gemv_bf16_prerounded_residual_norm_gate_up_silu_dead_up_shared_pair_test_kernel,
      512, 0U);
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

int launch_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_table_free_e2m1_test_cuda(
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
    launch_nvfp4_small_m32_wmma_k64_dual_a_table_free_e2m1_unchecked<
        17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                         activations, output, stream);
  } else {
    launch_nvfp4_small_m32_wmma_k64_dual_a_table_free_e2m1_unchecked<
        5'120U, 17'408U>(packed_weights, block_scales, weight_scale_2,
                         activations, output, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_table_free_e2m1_resources_test_cuda(
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
            17'408U, 5'120U, 72U, true, true, 32U, true>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              17'408U, 5'120U, 72U, true, true, 32U, true>,
          static_cast<int>(kThreads), 0U);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
            5'120U, 17'408U, 72U, true, true, 32U, true>);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_scale_window_kernel<
              5'120U, 17'408U, 72U, true, true, 32U, true>,
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

// Test-only direct entry for the production exact-M32 gate/up table-free,
// single-slot packed-weight cp.async specialization. Production reaches the
// same kernel through its private unchecked launcher; the public ABI does not
// expose this diagnostic entry.
int launch_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_table_free_raw_weight_cp_async_test_cuda(
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
  if (rows != 17'408U || columns != 5'120U) {
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
  launch_nvfp4_small_m32_wmma_k64_dual_a_table_free_raw_weight_cp_async_unchecked<
      17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                       activations, output, stream);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_w4a16_small_m32_wmma_k64_dual_a_table_free_raw_weight_cp_async_resources_test_cuda(
    const std::size_t rows, const std::size_t columns,
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes, int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (rows != 17'408U || columns != 5'120U ||
      registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_table_free_raw_weight_cp_async_kernel<
          17'408U, 5'120U, 72U>);
  int active_blocks = 0;
  if (status == cudaSuccess) {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks,
        nvfp4_w4a16_small_m32_gemm_bf16_wmma_k64_dual_a_table_free_raw_weight_cp_async_kernel<
            17'408U, 5'120U, 72U>,
        static_cast<int>(kThreads), 0U);
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
        launch_fp8_small_m32_wmma_dual_resident_a_unchecked<10'240U, 5'120U,
                                                            72U>(
            weights, weight_scale, activations, output, stream);
        break;
      case registry::ProjectionShape::kFp8_5120x6144:
        launch_fp8_small_m32_wmma_dual_resident_a_unchecked<5'120U, 6'144U,
                                                            72U>(
            weights, weight_scale, activations, output, stream);
        break;
      case registry::ProjectionShape::kFp8_6144x5120:
        launch_fp8_small_m32_wmma_dual_resident_a_unchecked<6'144U, 5'120U,
                                                            72U>(
            weights, weight_scale, activations, output, stream);
        break;
      case registry::ProjectionShape::kFp8_12288x5120:
        launch_fp8_small_m32_wmma_dual_resident_a_unchecked<12'288U, 5'120U,
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
        launch_nvfp4_small_m32_wmma_k64_dual_a_table_free_raw_weight_cp_async_unchecked<
            17'408U, 5'120U>(packed_weights, block_scales, weight_scale_2,
                             activations, output, stream);
        break;
      case registry::ProjectionShape::kNvFp4_5120x17408:
        launch_nvfp4_small_m32_wmma_k64_dual_a_table_free_e2m1_unchecked<
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
