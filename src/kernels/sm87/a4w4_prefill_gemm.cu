#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr std::size_t kPackedK64Bytes =
    kSm87A4W4PrefillTileK / 2U;
inline constexpr std::size_t kStageABytes =
    kSm87A4W4PrefillTileM * kPackedK64Bytes;
inline constexpr std::size_t kStageBBytes =
    kSm87A4W4PrefillTileN * kPackedK64Bytes;
inline constexpr std::size_t kLargeMStageABytes =
    kSm87A4W4PrefillLargeMTileM * kPackedK64Bytes;
inline constexpr std::size_t kLargeMStageBBytes =
    kSm87A4W4PrefillLargeMTileN * kPackedK64Bytes;
inline constexpr std::size_t kWideStageABytes =
    kSm87A4W4PrefillWideTileM * kPackedK64Bytes;
inline constexpr std::size_t kWideStageBBytes =
    kSm87A4W4PrefillWideTileN * kPackedK64Bytes;
inline constexpr int kRequiredSmCount = 16;

struct alignas(16) Sm87A4W4PipelineStage final {
  std::uint8_t a[kStageABytes];
  std::uint8_t b[kStageBBytes];
  std::uint16_t a_scales[kSm87A4W4PrefillTileM];
  std::uint16_t b_scales[kSm87A4W4PrefillTileN];
};

struct alignas(16) Sm87A4W4M64N64PipelineStage final {
  std::uint8_t a[kLargeMStageABytes];
  std::uint8_t b[kLargeMStageBBytes];
  std::uint16_t a_scales[kSm87A4W4PrefillLargeMTileM];
  std::uint16_t b_scales[kSm87A4W4PrefillLargeMTileN];
};

struct alignas(16) Sm87A4W4M64N256K64Stage final {
  std::uint8_t a[kWideStageABytes];
  std::uint8_t b[kWideStageBBytes];
  std::uint16_t a_scales[kSm87A4W4PrefillWideTileM];
  std::uint16_t b_scales[kSm87A4W4PrefillWideTileN];
};

// One asynchronous copy group owns two consecutive K64 groups.  The kernel
// still consumes the two halves independently and in ascending group order.
struct alignas(16) Sm87A4W4M64N256K128Stage final {
  Sm87A4W4M64N256K64Stage group[2U];
};

static_assert(kStageABytes == 1'024U);
static_assert(kStageBBytes == 4'096U);
static_assert(sizeof(Sm87A4W4PipelineStage) == 5'440U);
static_assert(kLargeMStageABytes == 2'048U);
static_assert(kLargeMStageBBytes == 2'048U);
static_assert(sizeof(Sm87A4W4M64N64PipelineStage) == 4'352U);
static_assert(kWideStageABytes == 2'048U);
static_assert(kWideStageBBytes == 8'192U);
static_assert(sizeof(Sm87A4W4M64N256K64Stage) == 10'880U);
static_assert(sizeof(Sm87A4W4M64N256K128Stage) == 21'760U);

[[nodiscard]] constexpr bool aligned(const void* const pointer,
                                     const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] constexpr bool product_fits(const std::size_t first,
                                          const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr bool consumer_capacity_fits(
    const std::size_t outer_count,
    const std::size_t k64_group_count) noexcept {
  const std::size_t outer_blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return outer_blocks != 0U && k64_group_count != 0U &&
         product_fits(outer_blocks, k64_group_count) &&
         product_fits(outer_blocks * k64_group_count,
                      kSm87A4W4ConsumerOuterBlock) &&
         product_fits(outer_blocks * k64_group_count *
                          kSm87A4W4ConsumerOuterBlock,
                      kSm87A4W4ConsumerPackedKBlockBytes);
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

extern "C" __global__ __launch_bounds__(kSm87A4W4PrefillThreads)
void q3x_sm87_a4_quantize_bf16_kernel(
    const std::uint16_t* const input_bf16,
    const std::size_t input_row_stride_elements,
    const std::size_t k64_group_count,
    const std::size_t group_count,
    const float clip_ratio,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_k64_scales_bf16) {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const std::size_t group_ordinal =
      static_cast<std::size_t>(blockIdx.x) * kSm87A4W4PrefillWarps + warp;
  if (group_ordinal >= group_count) {
    return;
  }
  const std::size_t row = group_ordinal / k64_group_count;
  const std::size_t group = group_ordinal - row * k64_group_count;
  const std::size_t input_offset =
      row * input_row_stride_elements + group * kSm87A4W4PrefillTileK;
  float even = decode_bf16(input_bf16[input_offset + 2U * lane]);
  float odd = decode_bf16(input_bf16[input_offset + 2U * lane + 1U]);
  float maximum = fmaxf(fabsf(even), fabsf(odd));
#pragma unroll
  for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
    maximum = fmaxf(maximum,
                    __shfl_down_sync(0xffffffffU, maximum, delta));
  }
  maximum = __shfl_sync(0xffffffffU, maximum, 0U);
  const float clipped_maximum = maximum * clip_ratio;
  std::uint16_t scale_bits =
      encode_bf16(maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
  float stored_scale = decode_bf16(scale_bits);
  if (maximum != 0.0F && stored_scale == 0.0F) {
    scale_bits = 1U;
    stored_scale = decode_bf16(scale_bits);
  }
  even = fminf(fmaxf(even, -clipped_maximum), clipped_maximum);
  odd = fminf(fmaxf(odd, -clipped_maximum), clipped_maximum);
  const int even_rounded = stored_scale == 0.0F
                               ? 0
                               : __float2int_rn(even / stored_scale);
  const int odd_rounded = stored_scale == 0.0F
                              ? 0
                              : __float2int_rn(odd / stored_scale);
  const int even_code = even_rounded < -7 ? -7 :
                        (even_rounded > 7 ? 7 : even_rounded);
  const int odd_code = odd_rounded < -7 ? -7 :
                       (odd_rounded > 7 ? 7 : odd_rounded);
  packed_a[sm87_a4w4_consumer_packed_offset(
      row, group, lane, k64_group_count)] =
      sm87_a4w4_pack_signed_pair(even_code, odd_code);
  if (lane == 0U) {
    a_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
        row, group, k64_group_count)] =
        scale_bits;
  }
}

__device__ __forceinline__ void cp_async_16(
    void* const destination, const void* const source,
    const unsigned int source_bytes) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(shared_address), "l"(source), "r"(source_bytes)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_commit() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(PendingGroups)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void prefetch_stage(
    Sm87A4W4PipelineStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t token_count,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t k64_group,
    const std::size_t k64_group_count) noexcept {
  constexpr std::size_t kVectorCount =
      (kStageABytes + kStageBBytes) / 16U;
  for (std::size_t vector = threadIdx.x; vector < kVectorCount;
       vector += blockDim.x) {
    if (vector < kStageABytes / 16U) {
      const std::size_t row = vector / 2U;
      const std::size_t row_vector = vector % 2U;
      const std::size_t global_row = m_tile_start + row;
      const bool valid = global_row < token_count;
      const std::uint8_t* const source =
          valid ? packed_a + sm87_a4w4_consumer_packed_offset(
                                 global_row, k64_group,
                                 row_vector * 16U, k64_group_count)
                : packed_a;
      cp_async_16(
          stage.a + sm87_a4w4_swizzled_k64_byte_offset(
                        row, row_vector * 16U),
          source, valid ? 16U : 0U);
    } else {
      const std::size_t b_vector = vector - kStageABytes / 16U;
      const std::size_t row = b_vector / 2U;
      const std::size_t row_vector = b_vector % 2U;
      const std::size_t global_row = n_tile_start + row;
      const std::uint8_t* const source =
          packed_b + sm87_a4w4_consumer_packed_offset(
                         global_row, k64_group, row_vector * 16U,
                         k64_group_count);
      cp_async_16(
          stage.b + sm87_a4w4_swizzled_k64_byte_offset(
                        row, row_vector * 16U),
          source, 16U);
    }
  }
  if (threadIdx.x < kSm87A4W4PrefillTileM) {
    const std::size_t global_row = m_tile_start + threadIdx.x;
    stage.a_scales[threadIdx.x] =
        global_row < token_count
            ? a_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                  global_row, k64_group, k64_group_count)]
            : static_cast<std::uint16_t>(0U);
  }
  if (threadIdx.x < kSm87A4W4PrefillTileN) {
    const std::size_t global_row = n_tile_start + threadIdx.x;
    stage.b_scales[threadIdx.x] =
        b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_row, k64_group, k64_group_count)];
  }
  cp_async_commit();
}

__device__ __forceinline__ void prefetch_m64n64_stage(
    Sm87A4W4M64N64PipelineStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t k64_group,
    const std::size_t k64_group_count) noexcept {
  // M64N64 contains exactly 256 aligned 16-byte vectors, so every thread
  // contributes one cp.async and neither operand monopolizes a second pass.
  constexpr std::size_t kVectorCount =
      (kLargeMStageABytes + kLargeMStageBBytes) / 16U;
  static_assert(kVectorCount == kSm87A4W4PrefillThreads);
  const std::size_t vector = threadIdx.x;
  if (vector < kLargeMStageABytes / 16U) {
    const std::size_t row = vector / 2U;
    const std::size_t row_vector = vector % 2U;
    const std::size_t global_row = m_tile_start + row;
    const std::uint8_t* const source =
        packed_a + sm87_a4w4_consumer_packed_offset(
                       global_row, k64_group, row_vector * 16U,
                       k64_group_count);
    cp_async_16(
        stage.a + sm87_a4w4_swizzled_k64_byte_offset(
                      row, row_vector * 16U),
        source, 16U);
  } else {
    const std::size_t b_vector = vector - kLargeMStageABytes / 16U;
    const std::size_t row = b_vector / 2U;
    const std::size_t row_vector = b_vector % 2U;
    const std::size_t global_row = n_tile_start + row;
    const std::uint8_t* const source =
        packed_b + sm87_a4w4_consumer_packed_offset(
                       global_row, k64_group, row_vector * 16U,
                       k64_group_count);
    cp_async_16(
        stage.b + sm87_a4w4_swizzled_k64_byte_offset(
                      row, row_vector * 16U),
        source, 16U);
  }
  if (threadIdx.x < kSm87A4W4PrefillLargeMTileM) {
    const std::size_t global_m = m_tile_start + threadIdx.x;
    const std::size_t global_n = n_tile_start + threadIdx.x;
    stage.a_scales[threadIdx.x] =
        a_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_m, k64_group, k64_group_count)];
    stage.b_scales[threadIdx.x] =
        b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_n, k64_group, k64_group_count)];
  }
  cp_async_commit();
}

__device__ __forceinline__ void prefetch_m64n256_k128_stage(
    Sm87A4W4M64N256K128Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t first_k64_group,
    const std::size_t k64_group_count) noexcept {
  constexpr std::size_t kVectorsPerK64 =
      (kWideStageABytes + kWideStageBBytes) / 16U;
  constexpr std::size_t kVectorCount = 2U * kVectorsPerK64;
  static_assert(kVectorsPerK64 == 640U);
  static_assert(kVectorCount == 5U * kSm87A4W4PrefillThreads);

  // Exactly five 16-byte copies per thread fill the complete K128 logical
  // buffer.  An odd final K64 group is zero-filled but never consumed.
  for (std::size_t vector = threadIdx.x; vector < kVectorCount;
       vector += blockDim.x) {
    const std::size_t group_half = vector / kVectorsPerK64;
    const std::size_t group_vector =
        vector - group_half * kVectorsPerK64;
    const std::size_t k64_group = first_k64_group + group_half;
    const bool valid_group = k64_group < k64_group_count;
    Sm87A4W4M64N256K64Stage& group_stage = stage.group[group_half];
    if (group_vector < kWideStageABytes / 16U) {
      const std::size_t row = group_vector / 2U;
      const std::size_t row_vector = group_vector % 2U;
      const std::size_t global_row = m_tile_start + row;
      const std::uint8_t* const source =
          valid_group
              ? packed_a + sm87_a4w4_consumer_packed_offset(
                               global_row, k64_group, row_vector * 16U,
                               k64_group_count)
              : packed_a;
      cp_async_16(
          group_stage.a + sm87_a4w4_swizzled_k64_byte_offset(
                              row, row_vector * 16U),
          source, valid_group ? 16U : 0U);
    } else {
      const std::size_t b_vector =
          group_vector - kWideStageABytes / 16U;
      const std::size_t row = b_vector / 2U;
      const std::size_t row_vector = b_vector % 2U;
      const std::size_t global_row = n_tile_start + row;
      const std::uint8_t* const source =
          valid_group
              ? packed_b + sm87_a4w4_consumer_packed_offset(
                               global_row, k64_group, row_vector * 16U,
                               k64_group_count)
              : packed_b;
      cp_async_16(
          group_stage.b + sm87_a4w4_swizzled_k64_byte_offset(
                              row, row_vector * 16U),
          source, valid_group ? 16U : 0U);
    }
  }

#pragma unroll
  for (unsigned int group_half = 0U; group_half < 2U; ++group_half) {
    const std::size_t k64_group = first_k64_group + group_half;
    const bool valid_group = k64_group < k64_group_count;
    Sm87A4W4M64N256K64Stage& group_stage = stage.group[group_half];
    if (threadIdx.x < kSm87A4W4PrefillWideTileM) {
      const std::size_t global_m = m_tile_start + threadIdx.x;
      group_stage.a_scales[threadIdx.x] =
          valid_group
              ? a_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                    global_m, k64_group, k64_group_count)]
              : static_cast<std::uint16_t>(0U);
    }
    const std::size_t global_n = n_tile_start + threadIdx.x;
    group_stage.b_scales[threadIdx.x] =
        valid_group
            ? b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                  global_n, k64_group, k64_group_count)]
            : static_cast<std::uint16_t>(0U);
  }
  cp_async_commit();
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4PrefillThreads,
                      kSm87A4W4PrefillCtasPerSm)
void q3x_sm87_a4w4_prefill_gemm_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t k64_group_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t m_tile_count,
    const std::size_t work_tile_count) {
  __shared__ Sm87A4W4PipelineStage
      pipeline[kSm87A4W4PrefillPipelineStages];

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const std::size_t warp_m = warp % 2U;
  const std::size_t warp_n = warp / 2U;

  for (std::size_t work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    // N-major ordering groups every M tile of the same B tile contiguously.
    // At P512, each B N128 tile is reused across sixteen M32 tiles through L2,
    // while each A half-block is revisited only once per two N64 B blocks.
    const std::size_t n_tile = work_tile / m_tile_count;
    const std::size_t m_tile = work_tile - n_tile * m_tile_count;
    const std::size_t m_tile_start = m_tile * kSm87A4W4PrefillTileM;
    const std::size_t n_tile_start = n_tile * kSm87A4W4PrefillTileN;

    float accumulators[4U][4U]{};

    prefetch_stage(pipeline[0U], packed_a, a_k64_scales_bf16,
                   packed_b, b_k64_scales_bf16, token_count, m_tile_start,
                   n_tile_start, 0U, k64_group_count);
    if (k64_group_count > 1U) {
      prefetch_stage(pipeline[1U], packed_a, a_k64_scales_bf16,
                     packed_b, b_k64_scales_bf16, token_count, m_tile_start,
                     n_tile_start, 1U, k64_group_count);
    }

    for (std::size_t group = 0U; group < k64_group_count; ++group) {
      if (group + 2U < k64_group_count) {
        prefetch_stage(
            pipeline[(group + 2U) % kSm87A4W4PrefillPipelineStages],
            packed_a, a_k64_scales_bf16, packed_b,
            b_k64_scales_bf16, token_count, m_tile_start,
            n_tile_start, group + 2U, k64_group_count);
      }
      if (group + 1U == k64_group_count) {
        cp_async_wait<0>();
      } else {
        cp_async_wait<1>();
      }
      __syncthreads();

      const Sm87A4W4PipelineStage& stage =
          pipeline[group % kSm87A4W4PrefillPipelineStages];
      const std::uint8_t* const warp_a =
          stage.a + warp_m * 16U * kPackedK64Bytes;
      const Sm87A4W4AFragment a =
          sm87_a4w4_load_a_fragment_swizzled_shared(warp_a, lane);

#pragma unroll
      for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
        const std::size_t fragment_n = warp_n * 32U + fragment * 8U;
        const std::uint8_t* const warp_b =
            stage.b + fragment_n * kPackedK64Bytes;
        const Sm87A4W4BFragment b =
            sm87_a4w4_load_b_fragment_swizzled_shared(warp_b, lane);
        Sm87A4W4Accumulator partial{};
        sm87_a4w4_mma_m16n8k64(partial, a, b);
        const std::int32_t integer_partial[4U] = {
            partial.x0, partial.x1, partial.x2, partial.x3};

#pragma unroll
        for (unsigned int output = 0U; output < 4U; ++output) {
          const Sm87A4W4AccumulatorCoordinate coordinate =
              sm87_a4w4_accumulator_coordinate(lane, output);
          const std::size_t local_m = warp_m * 16U + coordinate.m;
          const std::size_t local_n = fragment_n + coordinate.n;
          const float a_scale = decode_bf16(stage.a_scales[local_m]);
          const float b_scale = decode_bf16(stage.b_scales[local_n]);
          accumulators[fragment][output] +=
              static_cast<float>(integer_partial[output]) * a_scale *
              b_scale;
        }
      }
      // All warps must finish reading this stage before the next iteration
      // can recycle it for group+3.
      __syncthreads();
    }

#pragma unroll
    for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        const Sm87A4W4AccumulatorCoordinate coordinate =
            sm87_a4w4_accumulator_coordinate(lane, output);
        const std::size_t global_m =
            m_tile_start + warp_m * 16U + coordinate.m;
        const std::size_t global_n =
            n_tile_start + warp_n * 32U + fragment * 8U + coordinate.n;
        if (global_m < token_count && global_n < output_size) {
          output_bf16[global_m * output_row_stride_elements + global_n] =
              encode_bf16(accumulators[fragment][output]);
        }
      }
    }
    __syncthreads();
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4PrefillThreads,
                      kSm87A4W4PrefillCtasPerSm)
void q3x_sm87_a4w4_prefill_gemm_m64n64_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t k64_group_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t m_tile_count,
    const std::size_t work_tile_count) {
  __shared__ Sm87A4W4M64N64PipelineStage
      pipeline[kSm87A4W4PrefillPipelineStages];

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const std::size_t warp_m = warp / 2U;
  const std::size_t warp_n = warp % 2U;

  for (std::size_t work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    // N-major ordering keeps all M64 consumers of one B N64 block adjacent.
    // Compared with M32N128 this halves B bytes per output while retaining
    // 4,096 outputs, 32 MMA instructions/K64, and eight warps per CTA.
    const std::size_t n_tile = work_tile / m_tile_count;
    const std::size_t m_tile = work_tile - n_tile * m_tile_count;
    const std::size_t m_tile_start =
        m_tile * kSm87A4W4PrefillLargeMTileM;
    const std::size_t n_tile_start =
        n_tile * kSm87A4W4PrefillLargeMTileN;

    float accumulators[4U][4U]{};

    prefetch_m64n64_stage(
        pipeline[0U], packed_a, a_k64_scales_bf16, packed_b,
        b_k64_scales_bf16, m_tile_start, n_tile_start, 0U,
        k64_group_count);
    if (k64_group_count > 1U) {
      prefetch_m64n64_stage(
          pipeline[1U], packed_a, a_k64_scales_bf16, packed_b,
          b_k64_scales_bf16, m_tile_start, n_tile_start, 1U,
          k64_group_count);
    }

    for (std::size_t group = 0U; group < k64_group_count; ++group) {
      if (group + 2U < k64_group_count) {
        prefetch_m64n64_stage(
            pipeline[(group + 2U) % kSm87A4W4PrefillPipelineStages],
            packed_a, a_k64_scales_bf16, packed_b,
            b_k64_scales_bf16, m_tile_start, n_tile_start, group + 2U,
            k64_group_count);
      }
      if (group + 1U == k64_group_count) {
        cp_async_wait<0>();
      } else {
        cp_async_wait<1>();
      }
      __syncthreads();

      const Sm87A4W4M64N64PipelineStage& stage =
          pipeline[group % kSm87A4W4PrefillPipelineStages];
      const std::uint8_t* const warp_a =
          stage.a + warp_m * 16U * kPackedK64Bytes;
      const Sm87A4W4AFragment a =
          sm87_a4w4_load_a_fragment_swizzled_shared(warp_a, lane);

#pragma unroll
      for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
        const std::size_t fragment_n = warp_n * 32U + fragment * 8U;
        const std::uint8_t* const warp_b =
            stage.b + fragment_n * kPackedK64Bytes;
        const Sm87A4W4BFragment b =
            sm87_a4w4_load_b_fragment_swizzled_shared(warp_b, lane);
        Sm87A4W4Accumulator partial{};
        sm87_a4w4_mma_m16n8k64(partial, a, b);
        const std::int32_t integer_partial[4U] = {
            partial.x0, partial.x1, partial.x2, partial.x3};

#pragma unroll
        for (unsigned int output = 0U; output < 4U; ++output) {
          const Sm87A4W4AccumulatorCoordinate coordinate =
              sm87_a4w4_accumulator_coordinate(lane, output);
          const std::size_t local_m = warp_m * 16U + coordinate.m;
          const std::size_t local_n = fragment_n + coordinate.n;
          const float a_scale = decode_bf16(stage.a_scales[local_m]);
          const float b_scale = decode_bf16(stage.b_scales[local_n]);
          accumulators[fragment][output] +=
              static_cast<float>(integer_partial[output]) * a_scale *
              b_scale;
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        const Sm87A4W4AccumulatorCoordinate coordinate =
            sm87_a4w4_accumulator_coordinate(lane, output);
        const std::size_t global_m =
            m_tile_start + warp_m * 16U + coordinate.m;
        const std::size_t global_n =
            n_tile_start + warp_n * 32U + fragment * 8U + coordinate.n;
        output_bf16[global_m * output_row_stride_elements + global_n] =
            encode_bf16(accumulators[fragment][output]);
      }
    }
    __syncthreads();
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4PrefillThreads,
                      kSm87A4W4PrefillCtasPerSm)
void q3x_sm87_a4w4_prefill_gemm_m64n256_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k64_scales_bf16,
    const unsigned int k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int n_tile_count) {
  __shared__ Sm87A4W4M64N256K128Stage
      pipeline[kSm87A4W4PrefillWidePipelineStages];

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  const unsigned int pair_count = (k64_group_count + 1U) / 2U;

  for (unsigned int n_tile = 0U; n_tile < n_tile_count; ++n_tile) {
    for (unsigned int m_tile = blockIdx.x; m_tile < m_tile_count;
         m_tile += gridDim.x) {
      // N-major ordering keeps every M64 consumer of an N256 block adjacent.
      // Every persistent CTA completes its M stripe before moving to the next
      // N block, avoiding integer division in the register-heavy inner kernel.
      // Each warp owns M32N64 and accumulates 64 FP32 outputs/thread.
      const unsigned int m_tile_start =
          m_tile * kSm87A4W4PrefillWideTileM;
      const unsigned int n_tile_start =
          n_tile * kSm87A4W4PrefillWideTileN;

      float accumulators[2U][8U][4U]{};

      prefetch_m64n256_k128_stage(
          pipeline[0U], packed_a, a_k64_scales_bf16, packed_b,
          b_k64_scales_bf16, m_tile_start, n_tile_start, 0U,
          k64_group_count);
      cp_async_wait<0>();
      __syncthreads();

      for (unsigned int pair = 0U; pair < pair_count; ++pair) {
        const bool has_next_pair = pair + 1U < pair_count;
        if (has_next_pair) {
          prefetch_m64n256_k128_stage(
              pipeline[(pair + 1U) % kSm87A4W4PrefillWidePipelineStages],
              packed_a, a_k64_scales_bf16, packed_b, b_k64_scales_bf16,
              m_tile_start, n_tile_start, 2U * (pair + 1U),
              k64_group_count);
        }

        const Sm87A4W4M64N256K128Stage& stage =
            pipeline[pair % kSm87A4W4PrefillWidePipelineStages];
#pragma unroll
        for (unsigned int group_half = 0U; group_half < 2U; ++group_half) {
          if (2U * pair + group_half >= k64_group_count) {
            continue;
          }
          const Sm87A4W4M64N256K64Stage& group_stage =
              stage.group[group_half];
#pragma unroll
          for (unsigned int fragment_m = 0U; fragment_m < 2U;
               ++fragment_m) {
            const unsigned int fragment_m_start =
                warp_m * 32U + fragment_m * 16U;
            const std::uint8_t* const warp_a =
                group_stage.a + fragment_m_start * kPackedK64Bytes;
            const Sm87A4W4AFragment a =
                sm87_a4w4_load_a_fragment_swizzled_shared(warp_a, lane);

#pragma unroll
            for (unsigned int fragment_n = 0U; fragment_n < 8U;
                 ++fragment_n) {
              const unsigned int fragment_n_start =
                  warp_n * 64U + fragment_n * 8U;
              const std::uint8_t* const warp_b =
                  group_stage.b + fragment_n_start * kPackedK64Bytes;
              const Sm87A4W4BFragment b =
                  sm87_a4w4_load_b_fragment_swizzled_shared(warp_b, lane);
              Sm87A4W4Accumulator partial{};
              sm87_a4w4_mma_m16n8k64(partial, a, b);
              const std::int32_t integer_partial[4U] = {
                  partial.x0, partial.x1, partial.x2, partial.x3};

#pragma unroll
              for (unsigned int output = 0U; output < 4U; ++output) {
                const Sm87A4W4AccumulatorCoordinate coordinate =
                    sm87_a4w4_accumulator_coordinate(lane, output);
                const unsigned int local_m =
                    fragment_m_start + coordinate.m;
                const unsigned int local_n =
                    fragment_n_start + coordinate.n;
                const float a_scale =
                    decode_bf16(group_stage.a_scales[local_m]);
                const float b_scale =
                    decode_bf16(group_stage.b_scales[local_n]);
                accumulators[fragment_m][fragment_n][output] +=
                    static_cast<float>(integer_partial[output]) * a_scale *
                    b_scale;
              }
            }
          }
        }

        if (has_next_pair) {
          cp_async_wait<0>();
        }
        // This makes the freshly copied alternate slot visible and prevents a
        // fast warp from recycling the current slot before every warp is done.
        __syncthreads();
      }

#pragma unroll
      for (unsigned int fragment_m = 0U; fragment_m < 2U; ++fragment_m) {
#pragma unroll
        for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
#pragma unroll
          for (unsigned int output = 0U; output < 4U; ++output) {
            const Sm87A4W4AccumulatorCoordinate coordinate =
                sm87_a4w4_accumulator_coordinate(lane, output);
            const unsigned int global_m =
                m_tile_start + warp_m * 32U + fragment_m * 16U + coordinate.m;
            const unsigned int global_n =
                n_tile_start + warp_n * 64U + fragment_n * 8U + coordinate.n;
            output_bf16[global_m * output_row_stride_elements + global_n] =
                encode_bf16(accumulators[fragment_m][fragment_n][output]);
          }
        }
      }
      __syncthreads();
    }
  }
}

[[nodiscard]] int validate_sm87(
    cudaDeviceProp* const properties = nullptr) noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp local{};
  status = cudaGetDeviceProperties(&local, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (local.major != kSm87A4W4RequiredComputeMajor ||
      local.minor != kSm87A4W4RequiredComputeMinor ||
      local.multiProcessorCount != kRequiredSmCount) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int launch_sm87_a4_quantize_bf16_cuda(
    const std::uint16_t* const input_bf16,
    const std::size_t input_row_stride_elements,
    const std::size_t token_count,
    const std::size_t input_size,
    const float clip_ratio,
    std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const a_k64_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  if (token_count == 0U || input_size == 0U ||
      input_size % kSm87A4W4PrefillTileK != 0U ||
      !(clip_ratio > 0.0F && clip_ratio <= 1.0F) ||
      !aligned(input_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k64_scales_bf16, alignof(std::uint16_t)) ||
      input_row_stride_elements < input_size) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t k64_groups = input_size / kSm87A4W4PrefillTileK;
  if (!consumer_capacity_fits(token_count, k64_groups) ||
      !product_fits(token_count, k64_groups)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_packed_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_scale_elements =
      sm87_a4w4_consumer_scale_capacity_elements(token_count, input_size);
  if (packed_a_capacity_bytes < required_packed_bytes ||
      a_scale_capacity_elements < required_scale_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_sm87();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const std::size_t group_count = token_count * k64_groups;
  const std::size_t blocks =
      sm87_a4w4_ceil_div(group_count, kSm87A4W4PrefillWarps);
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4_quantize_bf16_kernel<<<
      static_cast<unsigned int>(blocks),
      static_cast<unsigned int>(kSm87A4W4PrefillThreads), 0U, stream>>>(
      input_bf16, input_row_stride_elements, k64_groups, group_count,
      clip_ratio, packed_a, a_k64_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

int query_sm87_a4w4_prefill_gemm_resources_cuda(
    Sm87A4W4PrefillGemmResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4PrefillGemmResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_prefill_gemm_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_prefill_gemm_kernel,
      static_cast<int>(kSm87A4W4PrefillThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = attributes.maxDynamicSharedSizeBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  return static_cast<int>(cudaSuccess);
}

int query_sm87_a4w4_prefill_gemm_m64n64_resources_cuda(
    Sm87A4W4PrefillGemmResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4PrefillGemmResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_prefill_gemm_m64n64_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_prefill_gemm_m64n64_kernel,
      static_cast<int>(kSm87A4W4PrefillThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = attributes.maxDynamicSharedSizeBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4PrefillCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4PrefillThreads) ||
      resources->static_shared_bytes !=
          sizeof(Sm87A4W4M64N64PipelineStage) *
              kSm87A4W4PrefillPipelineStages) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int query_sm87_a4w4_prefill_gemm_m64n256_resources_cuda(
    Sm87A4W4PrefillGemmResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4PrefillGemmResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_prefill_gemm_m64n256_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_prefill_gemm_m64n256_kernel,
      static_cast<int>(kSm87A4W4PrefillThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = attributes.maxDynamicSharedSizeBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4PrefillCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4PrefillThreads) ||
      resources->static_shared_bytes !=
          sizeof(Sm87A4W4M64N256K128Stage) *
              kSm87A4W4PrefillWidePipelineStages) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_prefill_gemm_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k64_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4PrefillGemmPlan plan =
      sm87_a4w4_prefill_gemm_plan(token_count, output_size, input_size);
  if (plan.launch_ctas == 0U || !aligned(packed_a, 16U) ||
      !aligned(packed_b, 16U) ||
      !aligned(a_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(b_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(output_bf16, alignof(std::uint16_t)) ||
      !product_fits(plan.k64_groups, kPackedK64Bytes) ||
      !consumer_capacity_fits(token_count, plan.k64_groups) ||
      !consumer_capacity_fits(output_size, plan.k64_groups) ||
      output_row_stride_elements < output_size) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_a_scale_elements =
      sm87_a4w4_consumer_scale_capacity_elements(token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(output_size, input_size);
  const std::size_t required_b_scale_elements =
      sm87_a4w4_consumer_scale_capacity_elements(output_size, input_size);
  if (packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scale_elements ||
      packed_b_capacity_bytes < required_b_bytes ||
      b_scale_capacity_elements < required_b_scale_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device_status = validate_sm87();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const Sm87A4W4PrefillGemmPlan wide_plan =
      sm87_a4w4_prefill_gemm_m64n256_plan(
          token_count, output_size, input_size);
  const bool use_m64n256 = wide_plan.launch_ctas != 0U;
  if (use_m64n256 &&
      (wide_plan.k64_groups >
           std::numeric_limits<unsigned int>::max() ||
       output_row_stride_elements >
           std::numeric_limits<unsigned int>::max() ||
       wide_plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
       wide_plan.n_tiles > std::numeric_limits<unsigned int>::max())) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const bool use_large_m =
      !use_m64n256 &&
      sm87_a4w4_prefill_uses_large_m_candidate(token_count);
  Sm87A4W4PrefillGemmResources resources{};
  const int resource_status =
      use_m64n256
          ? query_sm87_a4w4_prefill_gemm_m64n256_resources_cuda(&resources)
          : (use_large_m
                 ? query_sm87_a4w4_prefill_gemm_m64n64_resources_cuda(
                       &resources)
                 : query_sm87_a4w4_prefill_gemm_resources_cuda(&resources));
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  if (resources.active_blocks_per_sm <
          static_cast<int>(kSm87A4W4PrefillCtasPerSm) ||
      resources.local_bytes != 0U) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (use_m64n256) {
    q3x_sm87_a4w4_prefill_gemm_m64n256_kernel<<<
        static_cast<unsigned int>(wide_plan.launch_ctas),
        static_cast<unsigned int>(kSm87A4W4PrefillThreads), 0U, stream>>>(
        packed_a, a_k64_scales_bf16, packed_b, b_k64_scales_bf16,
        static_cast<unsigned int>(wide_plan.k64_groups), output_bf16,
        static_cast<unsigned int>(output_row_stride_elements),
        static_cast<unsigned int>(wide_plan.m_tiles),
        static_cast<unsigned int>(wide_plan.n_tiles));
    return static_cast<int>(cudaPeekAtLastError());
  }
  if (use_large_m) {
    const std::size_t m64_tiles =
        token_count / kSm87A4W4PrefillLargeMTileM;
    q3x_sm87_a4w4_prefill_gemm_m64n64_kernel<<<
        static_cast<unsigned int>(plan.launch_ctas),
        static_cast<unsigned int>(kSm87A4W4PrefillThreads), 0U, stream>>>(
        packed_a, a_k64_scales_bf16, packed_b, b_k64_scales_bf16,
        plan.k64_groups, output_bf16, output_row_stride_elements, m64_tiles,
        plan.work_tiles);
    return static_cast<int>(cudaPeekAtLastError());
  }
  q3x_sm87_a4w4_prefill_gemm_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(kSm87A4W4PrefillThreads), 0U, stream>>>(
      packed_a, a_k64_scales_bf16, packed_b, b_k64_scales_bf16,
      token_count, output_size, plan.k64_groups, output_bf16,
      output_row_stride_elements, plan.m_tiles, plan.work_tiles);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
