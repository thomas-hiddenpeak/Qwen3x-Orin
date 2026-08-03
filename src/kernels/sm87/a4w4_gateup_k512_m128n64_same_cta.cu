#include "q3x/kernels/sm87_a4w4_gateup_k512_m128n64_same_cta.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(
        kSm87A4W4GateUpK512M128N64SameCtaPackedK64Bytes);

struct alignas(16) SameCtaStage final {
  std::uint8_t a[kSm87A4W4GateUpK512M128N64SameCtaK64PerStage]
                [kSm87A4W4GateUpK512M128N64SameCtaTileM *
                 kSm87A4W4GateUpK512M128N64SameCtaPackedK64Bytes];
  std::uint8_t b[kSm87A4W4GateUpK512M128N64SameCtaK64PerStage]
                [kSm87A4W4GateUpK512M128N64SameCtaTileN *
                 kSm87A4W4GateUpK512M128N64SameCtaPackedK64Bytes];
};

struct alignas(16) SameCtaScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpK512M128N64SameCtaTileM];
  std::uint16_t b[kSm87A4W4GateUpK512M128N64SameCtaTileN];
};

struct alignas(16) SameCtaPipeline final {
  SameCtaStage stage[kSm87A4W4GateUpK512M128N64SameCtaStages];
  SameCtaScaleSlot
      scale[kSm87A4W4GateUpK512M128N64SameCtaScaleSlots];
};

struct alignas(16) SameCtaShared final {
  SameCtaPipeline pipeline;
  // Component-major ownership keeps every warp's handoff naturally banked.
  // The Up phase has byte-identical MMA ownership and reads the exact FP32
  // component written by its Gate phase without a coordinate permutation.
  float gate[kSm87A4W4GateUpK512M128N64SameCtaWarps]
            [2U][4U][4U][32U];
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(SameCtaStage) ==
              kSm87A4W4GateUpK512M128N64SameCtaStageBytes);
static_assert(sizeof(SameCtaScaleSlot) ==
              kSm87A4W4GateUpK512M128N64SameCtaScaleSlotBytes);
static_assert(sizeof(SameCtaPipeline) ==
              kSm87A4W4GateUpK512M128N64SameCtaPipelineBytes);
static_assert(sizeof(SameCtaShared) ==
              kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes);
static_assert(sizeof(Float4) == 16U);

[[nodiscard]] constexpr bool aligned(const void* const pointer,
                                     const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] bool byte_ranges_overlap(
    const void* const first, const std::size_t first_bytes,
    const void* const second, const std::size_t second_bytes) noexcept {
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  constexpr std::uintptr_t maximum =
      std::numeric_limits<std::uintptr_t>::max();
  if (first_bytes > maximum - first_begin ||
      second_bytes > maximum - second_begin) {
    return true;
  }
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
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
    void* const destination, const void* const source) noexcept {
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

// One K256 stage has 1,024 A vectors and 512 B vectors.  Across 256 threads,
// every lane issues four A plus two B cp.async vectors.  Gate and Up invoke
// this routine independently, so no projection is kept live in registers.
__device__ __forceinline__ void issue_k256_codes(
    SameCtaStage& stage, const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int physical_k256_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512M128N64SameCtaTileM *
          kPackedK64Bytes / 16U);
  constexpr unsigned int kBVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512M128N64SameCtaTileN *
          kPackedK64Bytes / 16U);
  constexpr unsigned int kAVectors =
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512M128N64SameCtaK64PerStage) *
      kAVectorsPerPlane;
  constexpr unsigned int kBVectors =
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512M128N64SameCtaK64PerStage) *
      kBVectorsPerPlane;
  static_assert(kAVectors ==
                4U * kSm87A4W4GateUpK512M128N64SameCtaThreads);
  static_assert(kBVectors ==
                2U * kSm87A4W4GateUpK512M128N64SameCtaThreads);

#pragma unroll
  for (unsigned int iteration = 0U; iteration < 4U; ++iteration) {
    const unsigned int vector =
        threadIdx.x +
        iteration * kSm87A4W4GateUpK512M128N64SameCtaThreads;
    const unsigned int plane = vector / kAVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kAVectorsPerPlane;
    const unsigned int row = vector_in_plane / 2U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int physical_k64 =
        physical_k256_group *
            static_cast<unsigned int>(
                kSm87A4W4GateUpK512M128N64SameCtaK64PerStage) +
        plane;
    const unsigned int byte_in_row = 16U * row_vector;
    cp_async_16(
        stage.a[plane] +
            sm87_a4w4_swizzled_k64_byte_offset(row, byte_in_row),
        packed_a + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64, byte_in_row,
                       physical_k64_group_count));
  }

#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector =
        threadIdx.x +
        iteration * kSm87A4W4GateUpK512M128N64SameCtaThreads;
    const unsigned int plane = vector / kBVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kBVectorsPerPlane;
    const unsigned int row = vector_in_plane / 2U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int physical_k64 =
        physical_k256_group *
            static_cast<unsigned int>(
                kSm87A4W4GateUpK512M128N64SameCtaK64PerStage) +
        plane;
    const unsigned int byte_in_row = 16U * row_vector;
    cp_async_16(
        stage.b[plane] +
            sm87_a4w4_swizzled_k64_byte_offset(row, byte_in_row),
        packed_b + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(absolute_n_tile_start) +
                           row,
                       physical_k64, byte_in_row,
                       physical_k64_group_count));
  }
}

__device__ __forceinline__ void issue_k512_scales(
    SameCtaScaleSlot& slot,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  constexpr unsigned int kRowsPerVector = 8U;
  constexpr unsigned int kAVectors =
      kSm87A4W4GateUpK512M128N64SameCtaTileM / kRowsPerVector;
  constexpr unsigned int kBVectors =
      kSm87A4W4GateUpK512M128N64SameCtaTileN / kRowsPerVector;
  static_assert(kAVectors == 16U && kBVectors == 8U);
  if (threadIdx.x < kAVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    cp_async_16(
        slot.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_k512_m128n64_same_cta_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x < kBVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    cp_async_16(
        slot.b + first_row,
        b_k512_scales_bf16 +
            sm87_a4w4_gateup_k512_m128n64_same_cta_scale_offset(
                static_cast<std::size_t>(absolute_n_tile_start) +
                    first_row,
                k512_group, k512_group_count));
  }
}

__device__ __forceinline__ void issue_physical_k256(
    SameCtaPipeline& pipeline,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int physical_k256_group,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  const unsigned int stage_slot = physical_k256_group & 1U;
  issue_k256_codes(pipeline.stage[stage_slot], packed_a, packed_b,
                   m_tile_start, absolute_n_tile_start,
                   physical_k256_group, physical_k64_group_count);
  if ((physical_k256_group & 1U) == 0U) {
    const unsigned int group = physical_k256_group / 2U;
    issue_k512_scales(
        pipeline.scale[group & 1U], a_k512_scales_bf16,
        b_k512_scales_bf16, m_tile_start, absolute_n_tile_start,
        group, k512_group_count);
  }
  cp_async_commit();
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4AFragment
load_a_ldmatrix_x4(const std::uint8_t* const shared_a,
                   const unsigned int lane) noexcept {
  const unsigned int matrix = lane >> 3U;
  const unsigned int logical_row =
      (lane & 7U) + ((matrix & 1U) << 3U);
  const unsigned int logical_byte = (matrix >> 1U) * 16U;
  const auto* const source =
      shared_a + sm87_a4w4_swizzled_k64_byte_offset(
                     logical_row, logical_byte);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  Sm87A4W4AFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1),
        "=r"(fragment.x2), "=r"(fragment.x3)
      : "r"(shared_address)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return fragment;
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4BFragment
load_b_ldmatrix_x2(const std::uint8_t* const shared_b,
                   const unsigned int lane) noexcept {
  const unsigned int provider = lane & 15U;
  const unsigned int logical_row = provider & 7U;
  const unsigned int logical_byte = (provider >> 3U) * 16U;
  const auto* const source =
      shared_b + sm87_a4w4_swizzled_k64_byte_offset(
                     logical_row, logical_byte);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  Sm87A4W4BFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
      : "=r"(fragment.x0), "=r"(fragment.x1)
      : "r"(shared_address)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return fragment;
}

__device__ __forceinline__ void clear_partials(
    Sm87A4W4Accumulator (&partials)[2U][4U]) noexcept {
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
#pragma unroll
    for (unsigned int n_fragment = 0U; n_fragment < 4U;
         ++n_fragment) {
      partials[m_panel][n_fragment] = Sm87A4W4Accumulator{};
    }
  }
}

__device__ __forceinline__ void accumulate_k256_stage(
    const SameCtaStage& stage,
    Sm87A4W4Accumulator (&partials)[2U][4U]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m_start = (warp >> 1U) * 32U;
  const unsigned int local_n_start = (warp & 1U) * 32U;

#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4GateUpK512M128N64SameCtaK64PerStage;
       ++plane) {
    Sm87A4W4AFragment a[2U];
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
      a[m_panel] = load_a_ldmatrix_x4(
          stage.a[plane] +
              (local_m_start + m_panel * 16U) * kPackedK64Bytes,
          lane);
    }
#pragma unroll
    for (unsigned int n_fragment = 0U; n_fragment < 4U;
         ++n_fragment) {
      const Sm87A4W4BFragment b = load_b_ldmatrix_x2(
          stage.b[plane] +
              (local_n_start + n_fragment * 8U) * kPackedK64Bytes,
          lane);
#pragma unroll
      for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
        sm87_a4w4_mma_m16n8k64(partials[m_panel][n_fragment],
                               a[m_panel], b);
      }
    }
  }
}

__device__ __forceinline__ void apply_k512_group(
    Float4 (&accumulators)[2U][4U],
    const Sm87A4W4Accumulator (&partials)[2U][4U],
    const SameCtaScaleSlot& scale) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m_start = (warp >> 1U) * 32U;
  const unsigned int local_n_start = (warp & 1U) * 32U;
  constexpr unsigned int kMask = 0xffff'ffffU;

  const float a_owned = decode_bf16(scale.a[local_m_start + lane]);
  const float b_owned = decode_bf16(scale.b[local_n_start + lane]);
  const unsigned int m_low = lane >> 2U;
  const unsigned int m_high = m_low + 8U;
  const unsigned int n_even = 2U * (lane & 3U);
  const unsigned int n_odd = n_even + 1U;

#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
    const unsigned int m_owner = m_panel * 16U;
    const float a0 = __shfl_sync(kMask, a_owned, m_owner + m_low);
    const float a1 = __shfl_sync(kMask, a_owned, m_owner + m_high);
#pragma unroll
    for (unsigned int n_fragment = 0U; n_fragment < 4U;
         ++n_fragment) {
      const unsigned int n_owner = n_fragment * 8U;
      const float b0 = __shfl_sync(kMask, b_owned, n_owner + n_even);
      const float b1 = __shfl_sync(kMask, b_owned, n_owner + n_odd);
      const float scale00 = __fmul_rn(a0, b0);
      const float scale01 = __fmul_rn(a0, b1);
      const float scale10 = __fmul_rn(a1, b0);
      const float scale11 = __fmul_rn(a1, b1);
      const Sm87A4W4Accumulator& partial =
          partials[m_panel][n_fragment];
      Float4& output = accumulators[m_panel][n_fragment];
      output.x0 = __fmaf_rn(static_cast<float>(partial.x0), scale00,
                            output.x0);
      output.x1 = __fmaf_rn(static_cast<float>(partial.x1), scale01,
                            output.x1);
      output.x2 = __fmaf_rn(static_cast<float>(partial.x2), scale10,
                            output.x2);
      output.x3 = __fmaf_rn(static_cast<float>(partial.x3), scale11,
                            output.x3);
    }
  }
}

// The two-stage ring deliberately matches the K512 boundary: stage zero is
// the even K256 half and stage one the odd half.  After consuming stage zero,
// the next even half may be prefetched while the current odd half completes.
__device__ __forceinline__ void compute_projection(
    SameCtaPipeline& pipeline,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    Float4 (&accumulators)[2U][4U]) noexcept {
  issue_physical_k256(
      pipeline, packed_a, a_k512_scales_bf16, packed_b,
      b_k512_scales_bf16, m_tile_start, absolute_n_tile_start, 0U,
      k512_group_count, physical_k64_group_count);
  issue_physical_k256(
      pipeline, packed_a, a_k512_scales_bf16, packed_b,
      b_k512_scales_bf16, m_tile_start, absolute_n_tile_start, 1U,
      k512_group_count, physical_k64_group_count);

  for (unsigned int group = 0U; group < k512_group_count; ++group) {
    Sm87A4W4Accumulator partials[2U][4U];
    clear_partials(partials);

    cp_async_wait<1U>();
    __syncthreads();
    accumulate_k256_stage(pipeline.stage[0U], partials);
    // No producer may recycle stage zero before all warp LDS operations
    // retire.  This handoff also makes current shared-scale reads complete.
    __syncthreads();

    const unsigned int next_group = group + 1U;
    if (next_group < k512_group_count) {
      issue_physical_k256(
          pipeline, packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16, m_tile_start, absolute_n_tile_start,
          2U * next_group, k512_group_count,
          physical_k64_group_count);
      cp_async_wait<1U>();
    } else {
      cp_async_wait<0U>();
    }
    __syncthreads();
    accumulate_k256_stage(pipeline.stage[1U], partials);
    apply_k512_group(accumulators, partials,
                     pipeline.scale[group & 1U]);
    // Release stage one and the current scale slot before either can be
    // reused for the next K512 group.
    __syncthreads();

    if (next_group < k512_group_count) {
      issue_physical_k256(
          pipeline, packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16, m_tile_start, absolute_n_tile_start,
          2U * next_group + 1U, k512_group_count,
          physical_k64_group_count);
    }
  }
}

[[nodiscard]] __device__ __forceinline__ float float4_value(
    const Float4& value, const unsigned int coordinate) noexcept {
  switch (coordinate) {
    case 0U:
      return value.x0;
    case 1U:
      return value.x1;
    case 2U:
      return value.x2;
    default:
      return value.x3;
  }
}

__device__ __forceinline__ void store_gate_plane(
    SameCtaShared& shared,
    const Float4 (&accumulators)[2U][4U]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
#pragma unroll
    for (unsigned int n_fragment = 0U; n_fragment < 4U;
         ++n_fragment) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        shared.gate[warp][m_panel][n_fragment][output][lane] =
            float4_value(accumulators[m_panel][n_fragment], output);
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void store_product(
    const SameCtaShared& shared,
    const Float4 (&up_accumulators)[2U][4U],
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int output_n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m_start = (warp >> 1U) * 32U;
  const unsigned int local_n_start = (warp & 1U) * 32U;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
#pragma unroll
    for (unsigned int n_fragment = 0U; n_fragment < 4U;
         ++n_fragment) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        const Sm87A4W4AccumulatorCoordinate coordinate =
            sm87_a4w4_accumulator_coordinate(lane, output);
        const unsigned int local_m =
            local_m_start + m_panel * 16U + coordinate.m;
        const unsigned int local_n =
            local_n_start + n_fragment * 8U + coordinate.n;
        const unsigned int global_m = m_tile_start + local_m;
        if (global_m < logical_token_count) {
          const float gate =
              shared.gate[warp][m_panel][n_fragment][output][lane];
          const float up =
              float4_value(up_accumulators[m_panel][n_fragment],
                           output);
          output_bf16[
              static_cast<std::size_t>(global_m) *
                      output_row_stride_elements +
                  output_n_tile_start + local_n] =
              encode_bf16(silu_product(gate, up));
        }
      }
    }
  }
  // All Gate reads retire before a persistent CTA reuses this plane.
  __syncthreads();
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpK512M128N64SameCtaThreads,
                      kSm87A4W4GateUpK512M128N64SameCtaCtasPerSm)
void q3x_sm87_a4w4_gateup_k512_m128n64_same_cta_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile_count,
    const unsigned int absolute_n_start,
    const unsigned int n_tile_count,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<SameCtaShared*>(dynamic_shared);

  // N-major persistent work keeps a complete CTA wave on adjacent M owners
  // of the same N64 weight tile.  The 32-CTA grid exposes exactly two CTA
  // residency slots per SM on the pinned 16-SM target.
  const unsigned int work_cells = m_tile_count * n_tile_count;
  for (unsigned int cell = blockIdx.x; cell < work_cells;
       cell += gridDim.x) {
    const unsigned int n_tile = cell / m_tile_count;
    const unsigned int m_tile = cell - n_tile * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4GateUpK512M128N64SameCtaTileM;
    const unsigned int absolute_n_tile_start =
        absolute_n_start +
        n_tile * kSm87A4W4GateUpK512M128N64SameCtaTileN;
    const unsigned int output_n_tile_start =
        n_tile * kSm87A4W4GateUpK512M128N64SameCtaTileN;

    Float4 gate_accumulators[2U][4U]{};
    compute_projection(
        shared.pipeline, packed_a, a_k512_scales_bf16,
        packed_gate_b, gate_b_k512_scales_bf16, m_tile_start,
        absolute_n_tile_start, k512_group_count,
        physical_k64_group_count, gate_accumulators);
    store_gate_plane(shared, gate_accumulators);

    Float4 up_accumulators[2U][4U]{};
    compute_projection(
        shared.pipeline, packed_a, a_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, m_tile_start,
        absolute_n_tile_start, k512_group_count,
        physical_k64_group_count, up_accumulators);
    store_product(shared, up_accumulators, logical_token_count,
                  m_tile_start, output_n_tile_start, output_bf16,
                  output_row_stride_elements);
  }
}

namespace {

[[nodiscard]] int validate_sm87(
    cudaDeviceProp* const output_properties = nullptr) noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (properties.major != kSm87A4W4RequiredComputeMajor ||
      properties.minor != kSm87A4W4RequiredComputeMinor ||
      properties.multiProcessorCount != static_cast<int>(kRequiredSmCount)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_k512_m128n64_same_cta_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes));
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool require_model_shape,
    void* const cuda_stream) noexcept {
  const auto plan = sm87_a4w4_gateup_k512_m128n64_same_cta_plan(
      logical_token_count, launch_token_count, intermediate_size,
      input_size, n_start, n_count);
  const std::size_t model_output_stride =
      n_start == 0U
          ? kSm87A4W4GateUpK512M128N64SameCtaPrimaryWidth
          : kSm87A4W4GateUpK512M128N64SameCtaSecondaryRowStride;
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      (require_model_shape &&
       (!sm87_a4w4_gateup_k512_m128n64_same_cta_is_model_plan(plan) ||
        output_row_stride_elements != model_output_stride)) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k512_scales_bf16, 16U) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k512_scales_bf16, 16U) ||
      !aligned(output_bf16, 16U) ||
      output_row_stride_elements < n_count ||
      output_row_stride_elements % 2U != 0U ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.n_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_cells > std::numeric_limits<unsigned int>::max() ||
      plan.k512_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      logical_token_count > std::numeric_limits<unsigned int>::max() ||
      launch_token_count > std::numeric_limits<unsigned int>::max() ||
      intermediate_size > std::numeric_limits<unsigned int>::max() ||
      input_size > std::numeric_limits<unsigned int>::max() ||
      n_start > std::numeric_limits<unsigned int>::max() ||
      n_count > std::numeric_limits<unsigned int>::max() ||
      output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_gateup_k512_m128n64_same_cta_product_fits(
          logical_token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(launch_token_count,
                                               input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(intermediate_size,
                                               input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_gateup_k512_m128n64_same_cta_scale_capacity(
          launch_token_count, input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_gateup_k512_m128n64_same_cta_scale_capacity(
          intermediate_size, input_size);
  const std::size_t required_output_elements =
      logical_token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      gate_b_scale_capacity_elements < required_b_scales ||
      up_b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements ||
      !sm87_a4w4_gateup_k512_m128n64_same_cta_product_fits(
          required_a_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_gateup_k512_m128n64_same_cta_product_fits(
          required_b_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_gateup_k512_m128n64_same_cta_product_fits(
          required_output_elements, sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t required_output_bytes =
      required_output_elements * sizeof(std::uint16_t);
  if (byte_ranges_overlap(output_bf16, required_output_bytes, packed_a,
                          required_a_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          a_k512_scales_bf16,
                          required_a_scale_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          packed_gate_b, required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          gate_b_k512_scales_bf16,
                          required_b_scale_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          packed_up_b, required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          up_b_k512_scales_bf16,
                          required_b_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int target_status = validate_sm87();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaError_t shared_status = configure_dynamic_shared();
  if (shared_status != cudaSuccess) {
    return static_cast<int>(shared_status);
  }
  const unsigned int planned = static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      planned < maximum_launch_ctas ? planned : maximum_launch_ctas;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_gateup_k512_m128n64_same_cta_kernel
      <<<launch_ctas,
         static_cast<unsigned int>(
             kSm87A4W4GateUpK512M128N64SameCtaThreads),
         kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes,
         stream>>>(
          packed_a, a_k512_scales_bf16, packed_gate_b,
          gate_b_k512_scales_bf16, packed_up_b,
          up_b_k512_scales_bf16,
          static_cast<unsigned int>(logical_token_count),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(n_start),
          static_cast<unsigned int>(plan.n_tiles),
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          output_bf16,
          static_cast<unsigned int>(output_row_stride_elements));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_k512_m128n64_same_cta_resources_cuda(
    Sm87A4W4GateUpK512M128N64SameCtaResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpK512M128N64SameCtaResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_sm87(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  cudaError_t status = configure_dynamic_shared();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_gateup_k512_m128n64_same_cta_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_k512_m128n64_same_cta_kernel,
      static_cast<int>(kSm87A4W4GateUpK512M128N64SameCtaThreads),
      kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes;
  resources->configured_dynamic_shared_limit_bytes =
      static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes);
  resources->device_optin_shared_limit_bytes =
      static_cast<std::size_t>(properties.sharedMemPerBlockOptin);
  resources->device_shared_per_sm_bytes =
      static_cast<std::size_t>(properties.sharedMemPerMultiprocessor);
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4GateUpK512M128N64SameCtaMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes ||
      resources->device_shared_per_sm_bytes <
          kSm87A4W4GateUpK512M128N64SameCtaCtasPerSm *
              kSm87A4W4GateUpK512M128N64SameCtaDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpK512M128N64SameCtaThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(
              kSm87A4W4GateUpK512M128N64SameCtaCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_k512_m128n64_same_cta_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_k512_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_k512_scales_bf16,
      up_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size, n_start,
      n_count, output_bf16, output_row_stride_elements,
      output_capacity_elements,
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512M128N64SameCtaPersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_gateup_k512_m128n64_same_cta_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_k512_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_k512_scales_bf16,
      up_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size, n_start,
      n_count, output_bf16, output_row_stride_elements,
      output_capacity_elements, maximum_launch_ctas, false,
      cuda_stream);
}

}  // namespace q3x::kernels
