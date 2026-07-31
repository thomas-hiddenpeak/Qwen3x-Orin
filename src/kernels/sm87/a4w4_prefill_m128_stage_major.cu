#include "q3x/kernels/sm87_a4w4_prefill_m128_stage_major.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr std::size_t kPackedK64Bytes = 32U;
inline constexpr std::size_t kPhysicalK64PerK128 = 2U;
inline constexpr std::size_t kStageABytes =
    kSm87A4W4M128StageMajorTileM * kPackedK64Bytes;
inline constexpr std::size_t kStageBBytes =
    kSm87A4W4M128StageMajorTileN * kPackedK64Bytes;
inline constexpr int kRequiredSmCount = 16;

struct alignas(16) Sm87A4W4M128StageMajorStage final {
  std::uint8_t a[kPhysicalK64PerK128][kStageABytes];
  std::uint8_t b[kPhysicalK64PerK128][kStageBBytes];
  std::uint16_t a_scales[kSm87A4W4M128StageMajorTileM];
  std::uint16_t b_scales[kSm87A4W4M128StageMajorTileN];
};

struct alignas(16) Sm87A4W4M128StageMajorPipeline final {
  Sm87A4W4M128StageMajorStage
      slot[kSm87A4W4M128StageMajorPipelineSlots];
};

struct alignas(16) Sm87A4W4M128StageMajorPairedStage final {
  std::uint8_t a[kPhysicalK64PerK128][kStageABytes];
  std::uint8_t gate_b[kPhysicalK64PerK128][kStageABytes];
  std::uint8_t up_b[kPhysicalK64PerK128][kStageABytes];
  std::uint16_t a_scales[kSm87A4W4M128StageMajorTileM];
  std::uint16_t gate_b_scales[kSm87A4W4M128StageMajorPairedTileN];
  std::uint16_t up_b_scales[kSm87A4W4M128StageMajorPairedTileN];
};

struct alignas(16) Sm87A4W4M128StageMajorPairedPipeline final {
  Sm87A4W4M128StageMajorPairedStage
      slot[kSm87A4W4M128StageMajorPipelineSlots];
};

union alignas(16) Sm87A4W4M128StageMajorPairedShared final {
  Sm87A4W4M128StageMajorPairedPipeline pipeline;
  float product[kSm87A4W4M128StageMajorTileM]
               [kSm87A4W4M128StageMajorPairedTileN];
};

static_assert(kStageABytes == 4'096U);
static_assert(kStageBBytes == 8'192U);
static_assert(sizeof(Sm87A4W4M128StageMajorStage) ==
              kSm87A4W4M128StageMajorStageBytes);
static_assert(sizeof(Sm87A4W4M128StageMajorPipeline) ==
              kSm87A4W4M128StageMajorSharedBytes);
static_assert(sizeof(Sm87A4W4M128StageMajorPairedStage) ==
              kSm87A4W4M128StageMajorStageBytes);
static_assert(sizeof(Sm87A4W4M128StageMajorPairedPipeline) ==
              kSm87A4W4M128StageMajorSharedBytes);
static_assert(sizeof(Sm87A4W4M128StageMajorPairedShared) ==
              kSm87A4W4M128StageMajorPairedSharedBytes);

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

[[nodiscard]] constexpr bool required_consumer_bytes(
    const std::size_t outer_count,
    const std::size_t physical_k64_group_count,
    std::size_t* const bytes) noexcept {
  if (bytes == nullptr || outer_count == 0U ||
      physical_k64_group_count == 0U) {
    return false;
  }
  const std::size_t outer_blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  constexpr std::size_t elements_per_block_group =
      kSm87A4W4ConsumerOuterBlock *
      kSm87A4W4ConsumerPackedKBlockBytes;
  if (!product_fits(outer_blocks, physical_k64_group_count)) {
    return false;
  }
  const std::size_t block_groups =
      outer_blocks * physical_k64_group_count;
  if (!product_fits(block_groups, elements_per_block_group)) {
    return false;
  }
  *bytes = block_groups * elements_per_block_group;
  return true;
}

[[nodiscard]] constexpr bool required_k128_scales(
    const std::size_t outer_count,
    const std::size_t k128_group_count,
    std::size_t* const elements) noexcept {
  if (elements == nullptr || outer_count == 0U || k128_group_count == 0U) {
    return false;
  }
  const std::size_t outer_blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  if (!product_fits(outer_blocks, k128_group_count)) {
    return false;
  }
  const std::size_t block_groups = outer_blocks * k128_group_count;
  if (!product_fits(block_groups, kSm87A4W4ConsumerOuterBlock)) {
    return false;
  }
  *elements = block_groups * kSm87A4W4ConsumerOuterBlock;
  return true;
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

[[nodiscard]] __device__ __forceinline__ float silu_product(
    const float gate, const float up) noexcept {
  if (gate >= 0.0F) {
    return (gate / (1.0F + expf(-gate))) * up;
  }
  const float exponential = expf(gate);
  return (gate * exponential / (1.0F + exponential)) * up;
}

__device__ __forceinline__ void cp_async_16(
    void* const destination,
    const void* const source) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(source)
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

template <unsigned int Remaining>
__device__ __forceinline__ void cp_async_wait() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(Remaining)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void prefetch_stage(
    Sm87A4W4M128StageMajorStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k128_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr unsigned int kVectorsPerPhysicalK64 =
      static_cast<unsigned int>((kStageABytes + kStageBBytes) / 16U);
  constexpr unsigned int kVectorCount =
      static_cast<unsigned int>(kPhysicalK64PerK128) *
      kVectorsPerPhysicalK64;
  static_assert(kVectorsPerPhysicalK64 == 768U);
  static_assert(kVectorCount ==
                6U * kSm87A4W4M128StageMajorThreads);

  // Six vector loads per thread fill one complete logical K128 slot.  The
  // slot is stage-major, so every consumer switches one base pointer when
  // advancing the pipeline instead of independently rotating A/B planes.
#pragma unroll
  for (unsigned int iteration = 0U; iteration < 6U; ++iteration) {
    const unsigned int vector =
        static_cast<unsigned int>(threadIdx.x) +
        iteration * static_cast<unsigned int>(blockDim.x);
    const unsigned int half = vector / kVectorsPerPhysicalK64;
    const unsigned int group_vector =
        vector - half * kVectorsPerPhysicalK64;
    const unsigned int physical_group =
        k128_group * static_cast<unsigned int>(kPhysicalK64PerK128) + half;
    if (group_vector < kStageABytes / 16U) {
      const unsigned int row = group_vector / 2U;
      const unsigned int row_vector = group_vector % 2U;
      const std::uint8_t* const source =
          packed_a + sm87_a4w4_consumer_packed_offset(
                         static_cast<std::size_t>(m_tile_start) + row,
                         physical_group, row_vector * 16U,
                         physical_k64_group_count);
      cp_async_16(
          stage.a[half] + sm87_a4w4_swizzled_k64_byte_offset(
                              row, row_vector * 16U),
          source);
    } else {
      const unsigned int b_vector =
          group_vector - static_cast<unsigned int>(kStageABytes / 16U);
      const unsigned int row = b_vector / 2U;
      const unsigned int row_vector = b_vector % 2U;
      const std::uint8_t* const source =
          packed_b + sm87_a4w4_consumer_packed_offset(
                         static_cast<std::size_t>(n_tile_start) + row,
                         physical_group, row_vector * 16U,
                         physical_k64_group_count);
      cp_async_16(
          stage.b[half] + sm87_a4w4_swizzled_k64_byte_offset(
                              row, row_vector * 16U),
          source);
    }
  }

  if (threadIdx.x < kSm87A4W4M128StageMajorTileM) {
    stage.a_scales[threadIdx.x] =
        a_k128_scales_bf16[sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(m_tile_start) + threadIdx.x,
            k128_group, k128_group_count)];
  }
  stage.b_scales[threadIdx.x] =
      b_k128_scales_bf16[sm87_a4w4_consumer_k128_scale_offset(
          static_cast<std::size_t>(n_tile_start) + threadIdx.x,
          k128_group, k128_group_count)];
  cp_async_commit();
}

__device__ __forceinline__ void prefetch_paired_stage(
    Sm87A4W4M128StageMajorPairedStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr unsigned int kOperandVectors =
      static_cast<unsigned int>(kStageABytes / 16U);
  constexpr unsigned int kVectorsPerPhysicalK64 = 3U * kOperandVectors;
  constexpr unsigned int kVectorCount =
      static_cast<unsigned int>(kPhysicalK64PerK128) *
      kVectorsPerPhysicalK64;
  static_assert(kOperandVectors == 256U);
  static_assert(kVectorsPerPhysicalK64 == 768U);
  static_assert(kVectorCount ==
                6U * kSm87A4W4M128StageMajorThreads);

#pragma unroll
  for (unsigned int iteration = 0U; iteration < 6U; ++iteration) {
    const unsigned int vector =
        static_cast<unsigned int>(threadIdx.x) +
        iteration * static_cast<unsigned int>(blockDim.x);
    const unsigned int half = vector / kVectorsPerPhysicalK64;
    const unsigned int group_vector =
        vector - half * kVectorsPerPhysicalK64;
    const unsigned int operand = group_vector / kOperandVectors;
    const unsigned int operand_vector =
        group_vector - operand * kOperandVectors;
    const unsigned int row = operand_vector / 2U;
    const unsigned int row_vector = operand_vector % 2U;
    const unsigned int physical_group =
        k128_group * static_cast<unsigned int>(kPhysicalK64PerK128) + half;
    if (operand == 0U) {
      cp_async_16(
          stage.a[half] + sm87_a4w4_swizzled_k64_byte_offset(
                              row, row_vector * 16U),
          packed_a + sm87_a4w4_consumer_packed_offset(
                         static_cast<std::size_t>(m_tile_start) + row,
                         physical_group, row_vector * 16U,
                         physical_k64_group_count));
    } else if (operand == 1U) {
      cp_async_16(
          stage.gate_b[half] + sm87_a4w4_swizzled_k64_byte_offset(
                                   row, row_vector * 16U),
          packed_gate_b + sm87_a4w4_consumer_packed_offset(
                                static_cast<std::size_t>(n_tile_start) + row,
                                physical_group, row_vector * 16U,
                                physical_k64_group_count));
    } else {
      cp_async_16(
          stage.up_b[half] + sm87_a4w4_swizzled_k64_byte_offset(
                                 row, row_vector * 16U),
          packed_up_b + sm87_a4w4_consumer_packed_offset(
                              static_cast<std::size_t>(n_tile_start) + row,
                              physical_group, row_vector * 16U,
                              physical_k64_group_count));
    }
  }

  if (threadIdx.x < kSm87A4W4M128StageMajorTileM) {
    stage.a_scales[threadIdx.x] =
        a_k128_scales_bf16[sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(m_tile_start) + threadIdx.x,
            k128_group, k128_group_count)];
  }
  if (threadIdx.x < kSm87A4W4M128StageMajorPairedTileN) {
    const std::size_t global_n =
        static_cast<std::size_t>(n_tile_start) + threadIdx.x;
    stage.gate_b_scales[threadIdx.x] =
        gate_b_k128_scales_bf16[
            sm87_a4w4_consumer_k128_scale_offset(
                global_n, k128_group, k128_group_count)];
    stage.up_b_scales[threadIdx.x] =
        up_b_k128_scales_bf16[
            sm87_a4w4_consumer_k128_scale_offset(
                global_n, k128_group, k128_group_count)];
  }
  cp_async_commit();
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4M128StageMajorThreads,
                      kSm87A4W4M128StageMajorCtasPerSm)
void q3x_sm87_a4w4_prefill_m128n256_k128_stage_major_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k128_scales_bf16,
    const unsigned int k128_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& pipeline =
      *reinterpret_cast<Sm87A4W4M128StageMajorPipeline*>(dynamic_shared);

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;

  for (unsigned int work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const unsigned int n_tile = work_tile / m_tile_count;
    const unsigned int m_tile = work_tile - n_tile * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4M128StageMajorTileM;
    const unsigned int n_tile_start =
        n_tile * kSm87A4W4M128StageMajorTileN;
    const unsigned int fragment_m_start = warp * kSm87A4W4MmaM;

    // One warp owns M16N256: 32 native N8 fragments and 128 FP32 outputs per
    // lane.  This is the deliberate resource exchange versus M64N256: B is
    // staged once for twice as many rows, while occupancy is reduced to one
    // CTA/SM and the output accumulator lifetime doubles.
    float accumulators[32U][4U]{};

    prefetch_stage(
        pipeline.slot[0U], packed_a, a_k128_scales_bf16, packed_b,
        b_k128_scales_bf16, m_tile_start, n_tile_start, 0U,
        physical_k64_group_count, k128_group_count);
    cp_async_wait<0U>();
    __syncthreads();

    for (unsigned int group = 0U; group < k128_group_count; ++group) {
      const bool has_next = group + 1U < k128_group_count;
      if (has_next) {
        prefetch_stage(
            pipeline.slot[(group + 1U) %
                          kSm87A4W4M128StageMajorPipelineSlots],
            packed_a, a_k128_scales_bf16, packed_b,
            b_k128_scales_bf16, m_tile_start, n_tile_start, group + 1U,
            physical_k64_group_count, k128_group_count);
      }

      const Sm87A4W4M128StageMajorStage& stage =
          pipeline.slot[group % kSm87A4W4M128StageMajorPipelineSlots];
      const Sm87A4W4AFragment a0 =
          sm87_a4w4_load_a_fragment_swizzled_shared(
              stage.a[0U] + fragment_m_start * kPackedK64Bytes, lane);
      const Sm87A4W4AFragment a1 =
          sm87_a4w4_load_a_fragment_swizzled_shared(
              stage.a[1U] + fragment_m_start * kPackedK64Bytes, lane);
      const unsigned int local_m_low = fragment_m_start + lane / 4U;
      const unsigned int local_m_high = local_m_low + 8U;
      const float a_scale_low = decode_bf16(stage.a_scales[local_m_low]);
      const float a_scale_high = decode_bf16(stage.a_scales[local_m_high]);

#pragma unroll
      for (unsigned int fragment_n = 0U; fragment_n < 32U; ++fragment_n) {
        const unsigned int fragment_n_start = fragment_n * 8U;
        Sm87A4W4Accumulator partial{};
        const Sm87A4W4BFragment b0 =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.b[0U] + fragment_n_start * kPackedK64Bytes, lane);
        sm87_a4w4_mma_m16n8k64(partial, a0, b0);
        const Sm87A4W4BFragment b1 =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.b[1U] + fragment_n_start * kPackedK64Bytes, lane);
        // The second K64 instruction consumes the first one's S32 result as
        // C.  Conversion and scaling occur only after this ordered pair.
        sm87_a4w4_mma_m16n8k64(partial, a1, b1);

        const unsigned int local_n_even =
            fragment_n_start + 2U * (lane % 4U);
        const unsigned int local_n_odd = local_n_even + 1U;
        const float b_scale_even =
            decode_bf16(stage.b_scales[local_n_even]);
        const float b_scale_odd =
            decode_bf16(stage.b_scales[local_n_odd]);
        const float scale_low_even = a_scale_low * b_scale_even;
        const float scale_low_odd = a_scale_low * b_scale_odd;
        const float scale_high_even = a_scale_high * b_scale_even;
        const float scale_high_odd = a_scale_high * b_scale_odd;
        accumulators[fragment_n][0U] +=
            static_cast<float>(partial.x0) * scale_low_even;
        accumulators[fragment_n][1U] +=
            static_cast<float>(partial.x1) * scale_low_odd;
        accumulators[fragment_n][2U] +=
            static_cast<float>(partial.x2) * scale_high_even;
        accumulators[fragment_n][3U] +=
            static_cast<float>(partial.x3) * scale_high_odd;
      }

      if (has_next) {
        cp_async_wait<0U>();
      }
      // Publish the next slot and keep every warp off the current slot until
      // all eight consumers have completed their M16N256 strip.
      __syncthreads();
    }

#pragma unroll
    for (unsigned int fragment_n = 0U; fragment_n < 32U; ++fragment_n) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        const Sm87A4W4AccumulatorCoordinate coordinate =
            sm87_a4w4_accumulator_coordinate(lane, output);
        const unsigned int global_m =
            m_tile_start + fragment_m_start + coordinate.m;
        const unsigned int global_n =
            n_tile_start + fragment_n * 8U + coordinate.n;
        output_bf16[static_cast<std::size_t>(global_m) *
                        output_row_stride_elements +
                    global_n] = encode_bf16(accumulators[fragment_n][output]);
      }
    }
    __syncthreads();
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4M128StageMajorThreads,
                      kSm87A4W4M128StageMajorCtasPerSm)
void q3x_sm87_a4w4_prefill_m128n128_k128_stage_major_paired_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const unsigned int k128_group_count,
    const unsigned int physical_k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k128_scales_bf16,
    const unsigned int output_k128_group_count,
    const unsigned int output_physical_k64_group_count,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<Sm87A4W4M128StageMajorPairedShared*>(
          dynamic_shared);

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int fragment_m_start = warp * kSm87A4W4MmaM;

  for (unsigned int work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const unsigned int n_tile = work_tile / m_tile_count;
    const unsigned int m_tile = work_tile - n_tile * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4M128StageMajorTileM;
    const unsigned int n_tile_start =
        n_tile * kSm87A4W4M128StageMajorPairedTileN;

    // Gate and Up each own 64 FP32 results/lane.  Together they retain the
    // same 128-result register cardinality as the generic M128N256 kernel,
    // while the shared A code planes feed both projection streams.
    float gate_accumulators[16U][4U]{};
    float up_accumulators[16U][4U]{};

    prefetch_paired_stage(
        shared.pipeline.slot[0U], packed_a, a_k128_scales_bf16,
        packed_gate_b, gate_b_k128_scales_bf16, packed_up_b,
        up_b_k128_scales_bf16, m_tile_start, n_tile_start, 0U,
        physical_k64_group_count, k128_group_count);
    cp_async_wait<0U>();
    __syncthreads();

    for (unsigned int group = 0U; group < k128_group_count; ++group) {
      const bool has_next = group + 1U < k128_group_count;
      if (has_next) {
        prefetch_paired_stage(
            shared.pipeline.slot[(group + 1U) %
                                 kSm87A4W4M128StageMajorPipelineSlots],
            packed_a, a_k128_scales_bf16, packed_gate_b,
            gate_b_k128_scales_bf16, packed_up_b,
            up_b_k128_scales_bf16, m_tile_start, n_tile_start,
            group + 1U, physical_k64_group_count, k128_group_count);
      }

      const Sm87A4W4M128StageMajorPairedStage& stage =
          shared.pipeline.slot[
              group % kSm87A4W4M128StageMajorPipelineSlots];
      const Sm87A4W4AFragment a0 =
          sm87_a4w4_load_a_fragment_swizzled_shared(
              stage.a[0U] + fragment_m_start * kPackedK64Bytes, lane);
      const Sm87A4W4AFragment a1 =
          sm87_a4w4_load_a_fragment_swizzled_shared(
              stage.a[1U] + fragment_m_start * kPackedK64Bytes, lane);
      const unsigned int local_m_low = fragment_m_start + lane / 4U;
      const unsigned int local_m_high = local_m_low + 8U;
      const float a_scale_low = decode_bf16(stage.a_scales[local_m_low]);
      const float a_scale_high = decode_bf16(stage.a_scales[local_m_high]);

#pragma unroll
      for (unsigned int fragment_n = 0U; fragment_n < 16U; ++fragment_n) {
        const unsigned int fragment_n_start = fragment_n * 8U;
        Sm87A4W4Accumulator gate_partial{};
        Sm87A4W4Accumulator up_partial{};

        const Sm87A4W4BFragment gate_b0 =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.gate_b[0U] + fragment_n_start * kPackedK64Bytes,
                lane);
        const Sm87A4W4BFragment up_b0 =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.up_b[0U] + fragment_n_start * kPackedK64Bytes,
                lane);
        sm87_a4w4_mma_m16n8k64(gate_partial, a0, gate_b0);
        sm87_a4w4_mma_m16n8k64(up_partial, a0, up_b0);

        const Sm87A4W4BFragment gate_b1 =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.gate_b[1U] + fragment_n_start * kPackedK64Bytes,
                lane);
        const Sm87A4W4BFragment up_b1 =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.up_b[1U] + fragment_n_start * kPackedK64Bytes,
                lane);
        sm87_a4w4_mma_m16n8k64(gate_partial, a1, gate_b1);
        sm87_a4w4_mma_m16n8k64(up_partial, a1, up_b1);

        const unsigned int local_n_even =
            fragment_n_start + 2U * (lane % 4U);
        const unsigned int local_n_odd = local_n_even + 1U;
        const float gate_scale_even =
            decode_bf16(stage.gate_b_scales[local_n_even]);
        const float gate_scale_odd =
            decode_bf16(stage.gate_b_scales[local_n_odd]);
        const float up_scale_even =
            decode_bf16(stage.up_b_scales[local_n_even]);
        const float up_scale_odd =
            decode_bf16(stage.up_b_scales[local_n_odd]);
        gate_accumulators[fragment_n][0U] +=
            static_cast<float>(gate_partial.x0) *
            (a_scale_low * gate_scale_even);
        gate_accumulators[fragment_n][1U] +=
            static_cast<float>(gate_partial.x1) *
            (a_scale_low * gate_scale_odd);
        gate_accumulators[fragment_n][2U] +=
            static_cast<float>(gate_partial.x2) *
            (a_scale_high * gate_scale_even);
        gate_accumulators[fragment_n][3U] +=
            static_cast<float>(gate_partial.x3) *
            (a_scale_high * gate_scale_odd);
        up_accumulators[fragment_n][0U] +=
            static_cast<float>(up_partial.x0) *
            (a_scale_low * up_scale_even);
        up_accumulators[fragment_n][1U] +=
            static_cast<float>(up_partial.x1) *
            (a_scale_low * up_scale_odd);
        up_accumulators[fragment_n][2U] +=
            static_cast<float>(up_partial.x2) *
            (a_scale_high * up_scale_even);
        up_accumulators[fragment_n][3U] +=
            static_cast<float>(up_partial.x3) *
            (a_scale_high * up_scale_odd);
      }

      if (has_next) {
        cp_async_wait<0U>();
      }
      __syncthreads();
    }

    // The K128 pipeline is dead.  Reuse its allocation as the complete FP32
    // product tile so all 128 values of each output scale group participate
    // in one maximum before either physical K64 block is emitted.
#pragma unroll
    for (unsigned int fragment_n = 0U; fragment_n < 16U; ++fragment_n) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        const Sm87A4W4AccumulatorCoordinate coordinate =
            sm87_a4w4_accumulator_coordinate(lane, output);
        const unsigned int local_m = fragment_m_start + coordinate.m;
        const unsigned int local_n = fragment_n * 8U + coordinate.n;
        shared.product[local_m][local_n] = silu_product(
            gate_accumulators[fragment_n][output],
            up_accumulators[fragment_n][output]);
      }
    }
    __syncthreads();

#pragma unroll
    for (unsigned int row_iteration = 0U; row_iteration < 16U;
         ++row_iteration) {
      const unsigned int local_m = warp + row_iteration * 8U;
      const unsigned int global_m = m_tile_start + local_m;
      float value0 = shared.product[local_m][2U * lane];
      float value1 = shared.product[local_m][2U * lane + 1U];
      float value2 = shared.product[local_m][64U + 2U * lane];
      float value3 = shared.product[local_m][64U + 2U * lane + 1U];
      float maximum = fmaxf(
          fmaxf(fabsf(value0), fabsf(value1)),
          fmaxf(fabsf(value2), fabsf(value3)));
#pragma unroll
      for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
        maximum = fmaxf(
            maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
      }
      maximum = __shfl_sync(0xffffffffU, maximum, 0U);
      const float clipped_maximum = maximum * output_clip_ratio;
      std::uint16_t scale_bits = encode_bf16(
          maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      value0 = fminf(fmaxf(value0, -clipped_maximum), clipped_maximum);
      value1 = fminf(fmaxf(value1, -clipped_maximum), clipped_maximum);
      value2 = fminf(fmaxf(value2, -clipped_maximum), clipped_maximum);
      value3 = fminf(fmaxf(value3, -clipped_maximum), clipped_maximum);
      int code0 = stored_scale == 0.0F
                      ? 0
                      : __float2int_rn(value0 / stored_scale);
      int code1 = stored_scale == 0.0F
                      ? 0
                      : __float2int_rn(value1 / stored_scale);
      int code2 = stored_scale == 0.0F
                      ? 0
                      : __float2int_rn(value2 / stored_scale);
      int code3 = stored_scale == 0.0F
                      ? 0
                      : __float2int_rn(value3 / stored_scale);
      code0 = code0 < -7 ? -7 : (code0 > 7 ? 7 : code0);
      code1 = code1 < -7 ? -7 : (code1 > 7 ? 7 : code1);
      code2 = code2 < -7 ? -7 : (code2 > 7 ? 7 : code2);
      code3 = code3 < -7 ? -7 : (code3 > 7 ? 7 : code3);

      const unsigned int first_physical_group =
          n_tile_start / kSm87A4W4ConsumerKBlock;
      packed_output[sm87_a4w4_consumer_packed_offset(
          global_m, first_physical_group, lane,
          output_physical_k64_group_count)] =
          sm87_a4w4_pack_signed_pair(code0, code1);
      packed_output[sm87_a4w4_consumer_packed_offset(
          global_m, first_physical_group + 1U, lane,
          output_physical_k64_group_count)] =
          sm87_a4w4_pack_signed_pair(code2, code3);
      if (lane == 0U) {
        output_k128_scales_bf16[
            sm87_a4w4_consumer_k128_scale_offset(
                global_m, n_tile, output_k128_group_count)] = scale_bits;
      }
    }
    __syncthreads();
  }
}

[[nodiscard]] int validate_target(
    cudaDeviceProp* const properties = nullptr,
    const std::size_t required_shared_bytes =
        kSm87A4W4M128StageMajorSharedBytes) noexcept {
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
      local.multiProcessorCount != kRequiredSmCount ||
      local.sharedMemPerBlockOptin < required_shared_bytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int admit_dynamic_shared() noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      q3x_sm87_a4w4_prefill_m128n256_k128_stage_major_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4M128StageMajorSharedBytes)));
}

[[nodiscard]] int admit_paired_dynamic_shared() noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      q3x_sm87_a4w4_prefill_m128n128_k128_stage_major_paired_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4M128StageMajorPairedSharedBytes)));
}

}  // namespace

int query_sm87_a4w4_m128_stage_major_resources_cuda(
    Sm87A4W4M128StageMajorResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4M128StageMajorResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_target(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const int shared_status = admit_dynamic_shared();
  if (shared_status != static_cast<int>(cudaSuccess)) {
    return shared_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_prefill_m128n256_k128_stage_major_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_prefill_m128n256_k128_stage_major_kernel,
      static_cast<int>(kSm87A4W4M128StageMajorThreads),
      kSm87A4W4M128StageMajorSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kSm87A4W4M128StageMajorSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(kSm87A4W4M128StageMajorMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4M128StageMajorCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4M128StageMajorThreads)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int query_sm87_a4w4_m128_stage_major_paired_resources_cuda(
    Sm87A4W4M128StageMajorResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4M128StageMajorResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_target(
      &properties, kSm87A4W4M128StageMajorPairedSharedBytes);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const int shared_status = admit_paired_dynamic_shared();
  if (shared_status != static_cast<int>(cudaSuccess)) {
    return shared_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_prefill_m128n128_k128_stage_major_paired_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_prefill_m128n128_k128_stage_major_paired_kernel,
      static_cast<int>(kSm87A4W4M128StageMajorThreads),
      kSm87A4W4M128StageMajorPairedSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4M128StageMajorPairedSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(kSm87A4W4M128StageMajorMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4M128StageMajorCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4M128StageMajorThreads)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_m128_stage_major_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k128_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4M128StageMajorPlan plan =
      sm87_a4w4_m128_stage_major_plan(token_count, output_size, input_size);
  if (plan.launch_ctas == 0U || !aligned(packed_a, 16U) ||
      !aligned(a_k128_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_b, 16U) ||
      !aligned(b_k128_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(output_bf16, alignof(std::uint16_t)) ||
      output_row_stride_elements < output_size ||
      !product_fits(token_count, output_row_stride_elements) ||
      plan.k128_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups > std::numeric_limits<unsigned int>::max() ||
      output_row_stride_elements > std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.launch_ctas > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  std::size_t required_a_bytes = 0U;
  std::size_t required_a_scales = 0U;
  std::size_t required_b_bytes = 0U;
  std::size_t required_b_scales = 0U;
  if (!required_consumer_bytes(token_count, plan.physical_k64_groups,
                               &required_a_bytes) ||
      !required_k128_scales(token_count, plan.k128_groups,
                            &required_a_scales) ||
      !required_consumer_bytes(output_size, plan.physical_k64_groups,
                               &required_b_bytes) ||
      !required_k128_scales(output_size, plan.k128_groups,
                            &required_b_scales) ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      packed_b_capacity_bytes < required_b_bytes ||
      b_scale_capacity_elements < required_b_scales) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int device_status = validate_target();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  Sm87A4W4M128StageMajorResources resources{};
  const int resource_status =
      query_sm87_a4w4_m128_stage_major_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_prefill_m128n256_k128_stage_major_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(kSm87A4W4M128StageMajorThreads),
      kSm87A4W4M128StageMajorSharedBytes, stream>>>(
      packed_a, a_k128_scales_bf16, packed_b, b_k128_scales_bf16,
      static_cast<unsigned int>(plan.k128_groups),
      static_cast<unsigned int>(plan.physical_k64_groups), output_bf16,
      static_cast<unsigned int>(output_row_stride_elements),
      static_cast<unsigned int>(plan.m_tiles),
      static_cast<unsigned int>(plan.work_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_a4w4_m128_stage_major_paired_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k128_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4M128StageMajorPairedPlan plan =
      sm87_a4w4_m128_stage_major_paired_plan(
          token_count, intermediate_size, input_size);
  if (plan.launch_ctas == 0U ||
      !(output_clip_ratio > 0.0F && output_clip_ratio <= 1.0F) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k128_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k128_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k128_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_output, 16U) ||
      !aligned(output_k128_scales_bf16, alignof(std::uint16_t)) ||
      plan.k128_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups > std::numeric_limits<unsigned int>::max() ||
      plan.output_physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.n_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.launch_ctas > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  std::size_t required_a_bytes = 0U;
  std::size_t required_a_scales = 0U;
  std::size_t required_b_bytes = 0U;
  std::size_t required_b_scales = 0U;
  std::size_t required_output_bytes = 0U;
  std::size_t required_output_scales = 0U;
  if (!required_consumer_bytes(token_count, plan.physical_k64_groups,
                               &required_a_bytes) ||
      !required_k128_scales(token_count, plan.k128_groups,
                            &required_a_scales) ||
      !required_consumer_bytes(intermediate_size,
                               plan.physical_k64_groups,
                               &required_b_bytes) ||
      !required_k128_scales(intermediate_size, plan.k128_groups,
                            &required_b_scales) ||
      !required_consumer_bytes(token_count,
                               plan.output_physical_k64_groups,
                               &required_output_bytes) ||
      !required_k128_scales(token_count, plan.n_tiles,
                            &required_output_scales) ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      gate_b_scale_capacity_elements < required_b_scales ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      up_b_scale_capacity_elements < required_b_scales ||
      packed_output_capacity_bytes < required_output_bytes ||
      output_scale_capacity_elements < required_output_scales) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int device_status = validate_target(
      nullptr, kSm87A4W4M128StageMajorPairedSharedBytes);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  Sm87A4W4M128StageMajorResources resources{};
  const int resource_status =
      query_sm87_a4w4_m128_stage_major_paired_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_prefill_m128n128_k128_stage_major_paired_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(kSm87A4W4M128StageMajorThreads),
      kSm87A4W4M128StageMajorPairedSharedBytes, stream>>>(
      packed_a, a_k128_scales_bf16, packed_gate_b,
      gate_b_k128_scales_bf16, packed_up_b, up_b_k128_scales_bf16,
      static_cast<unsigned int>(plan.k128_groups),
      static_cast<unsigned int>(plan.physical_k64_groups),
      output_clip_ratio, packed_output, output_k128_scales_bf16,
      static_cast<unsigned int>(plan.n_tiles),
      static_cast<unsigned int>(plan.output_physical_k64_groups),
      static_cast<unsigned int>(plan.m_tiles),
      static_cast<unsigned int>(plan.work_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
