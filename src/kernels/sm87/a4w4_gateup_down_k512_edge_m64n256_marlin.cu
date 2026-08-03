#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m64n256_marlin.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kPackedK64Bytes = 32U;
std::atomic<bool> g_marlin_resources_ready{false};

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

struct alignas(16) M64N256MarlinStage final {
  std::uint8_t a[kSm87A4W4GateUpDownEdgeM64N256MarlinK64PerStage]
                [kSm87A4W4GateUpDownEdgeM64N256MarlinTileM *
                 kPackedK64Bytes];
  std::uint8_t gate[
      kSm87A4W4GateUpDownEdgeM64N256MarlinK64PerStage]
                   [kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN *
                    kPackedK64Bytes];
  std::uint8_t up[kSm87A4W4GateUpDownEdgeM64N256MarlinK64PerStage]
                 [kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN *
                  kPackedK64Bytes];
};

struct alignas(16) M64N256MarlinScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpDownEdgeM64N256MarlinTileM];
  std::uint16_t gate[
      kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN];
  std::uint16_t up[
      kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN];
};

struct alignas(16) M64N256MarlinPipeline final {
  M64N256MarlinStage
      stage[kSm87A4W4GateUpDownEdgeM64N256MarlinStages];
  M64N256MarlinScaleSlot
      scale[kSm87A4W4GateUpDownEdgeM64N256MarlinScaleSlots];
};

union alignas(16) M64N256MarlinEdgeHalf final {
  std::uint16_t product[
      kSm87A4W4GateUpDownEdgeM64N256MarlinTileM]
      [kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN];
  std::int32_t scratch[2U]
                      [kSm87A4W4GateUpDownEdgeM64N256MarlinWarps]
                      [kSm87A4W4GateUpDownEdgeM64N256MarlinM16PanelsPerWarp]
                      [32U][4U];
};

struct alignas(16) M64N256MarlinEdge final {
  M64N256MarlinEdgeHalf
      half[kSm87A4W4GateUpDownEdgeM64N256MarlinCellsPerEdge];
};

struct alignas(16) M64N256MarlinAuxScratch final {
  std::int32_t value[2U]
                    [kSm87A4W4GateUpDownEdgeM64N256MarlinWarps]
                    [kSm87A4W4GateUpDownEdgeM64N256MarlinM16PanelsPerWarp]
                    [32U][4U];
};

struct alignas(16) M64N256MarlinShared final {
  M64N256MarlinPipeline pipeline;
  M64N256MarlinEdge edge;
  M64N256MarlinAuxScratch aux_scratch;
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(M64N256MarlinStage) ==
              kSm87A4W4GateUpDownEdgeM64N256MarlinStageBytes);
static_assert(sizeof(M64N256MarlinScaleSlot) ==
              kSm87A4W4GateUpDownEdgeM64N256MarlinScaleSlotBytes);
static_assert(sizeof(M64N256MarlinPipeline) ==
              kSm87A4W4GateUpDownEdgeM64N256MarlinPipelineBytes);
static_assert(sizeof(M64N256MarlinEdgeHalf) * 2U ==
              kSm87A4W4GateUpDownEdgeM64N256MarlinEdgePlaneBytes);
static_assert(sizeof(M64N256MarlinEdge) ==
              kSm87A4W4GateUpDownEdgeM64N256MarlinEdgePlaneBytes);
static_assert(sizeof(M64N256MarlinAuxScratch) ==
              kSm87A4W4GateUpDownEdgeM64N256MarlinAuxScratchBytes);
static_assert(sizeof(M64N256MarlinShared) ==
              kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes);
static_assert(sizeof(Float4) == 16U);

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

// Every stage is already in consumer order: B row [32*warp+8*n8+r]
// is the exact M64N32 owner record used by that warp.  The canonical XOR
// byte swizzle then makes both A and B legal LDS.128/ldmatrix operands.
__device__ __forceinline__ void issue_k64_codes(
    M64N256MarlinStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m64_start,
    const unsigned int absolute_n256_start,
    const unsigned int physical_k64,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectors =
      kSm87A4W4GateUpDownEdgeM64N256MarlinAStageBytes / 16U;
  constexpr unsigned int kBVectors =
      kSm87A4W4GateUpDownEdgeM64N256MarlinBStageBytes / 16U;
  static_assert(kAVectors == 128U);
  static_assert(kBVectors == 512U);

  if (threadIdx.x < kAVectors) {
    const unsigned int a_row = threadIdx.x >> 1U;
    const unsigned int a_half = threadIdx.x & 1U;
    cp_async_16(
        stage.a[0U] +
            sm87_a4w4_swizzled_k64_byte_offset(a_row, 16U * a_half),
        packed_a + sm87_a4w4_gateup_down_edge_packed_offset(
                       static_cast<std::size_t>(m64_start) + a_row,
                       physical_k64, 16U * a_half,
                       physical_k64_group_count));
  }

#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector = threadIdx.x + 256U * iteration;
    const unsigned int row = vector >> 1U;
    const unsigned int half = vector & 1U;
    const std::size_t source_offset =
        sm87_a4w4_gateup_down_edge_packed_offset(
            static_cast<std::size_t>(absolute_n256_start) + row,
            physical_k64, 16U * half, physical_k64_group_count);
    const std::size_t destination_offset =
        sm87_a4w4_swizzled_k64_byte_offset(row, 16U * half);
    cp_async_16(stage.gate[0U] + destination_offset,
                packed_gate_b + source_offset);
    cp_async_16(stage.up[0U] + destination_offset,
                packed_up_b + source_offset);
  }
}

__device__ __forceinline__ void issue_k512_scales(
    M64N256MarlinScaleSlot& slot,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m64_start,
    const unsigned int absolute_n256_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  constexpr unsigned int kRowsPerVector = 8U;
  constexpr unsigned int kAVectors =
      kSm87A4W4GateUpDownEdgeM64N256MarlinTileM / kRowsPerVector;
  constexpr unsigned int kBVectors =
      kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN /
      kRowsPerVector;
  static_assert(kAVectors == 8U);
  static_assert(kBVectors == 32U);

  if (threadIdx.x < kAVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    cp_async_16(
        slot.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(m64_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x < kBVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    const std::size_t source_offset =
        sm87_a4w4_gateup_down_edge_scale_offset(
            static_cast<std::size_t>(absolute_n256_start) + first_row,
            k512_group, k512_group_count);
    cp_async_16(slot.gate + first_row,
                gate_b_k512_scales_bf16 + source_offset);
    cp_async_16(slot.up + first_row,
                up_b_k512_scales_bf16 + source_offset);
  }
}

__device__ __forceinline__ void issue_flat_k64(
    M64N256MarlinPipeline& pipeline,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m64_start,
    const unsigned int absolute_n256_start,
    const unsigned int flat_k64,
    const unsigned int k512_group_count) noexcept {
  const unsigned int stage =
      flat_k64 % kSm87A4W4GateUpDownEdgeM64N256MarlinStages;
  const unsigned int group = flat_k64 >> 3U;
  const unsigned int phase = flat_k64 & 7U;
  issue_k64_codes(
      pipeline.stage[stage], packed_a, packed_gate_b, packed_up_b,
      m64_start, absolute_n256_start, flat_k64,
      k512_group_count *
          kSm87A4W4GateUpDownEdgeM64N256MarlinK64PerScale);
  if (phase == 0U) {
    issue_k512_scales(
        pipeline.scale[group & 1U], a_k512_scales_bf16,
        gate_b_k512_scales_bf16, up_b_k512_scales_bf16,
        m64_start, absolute_n256_start, group, k512_group_count);
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
    Sm87A4W4Accumulator (&gate_partial)[4U][2U],
    Sm87A4W4Accumulator (&up_partial)[4U][2U]) noexcept {
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < 2U; ++n8) {
      gate_partial[panel][n8] = Sm87A4W4Accumulator{};
      up_partial[panel][n8] = Sm87A4W4Accumulator{};
    }
  }
}

__device__ __forceinline__ void clear_scratch(
    M64N256MarlinEdgeHalf& half,
    M64N256MarlinAuxScratch& aux) noexcept {
  constexpr unsigned int kHalfWords =
      sizeof(half.scratch) / sizeof(std::int32_t);
  auto* const words = &half.scratch[0U][0U][0U][0U][0U];
#pragma unroll 1
  for (unsigned int index = threadIdx.x; index < kHalfWords;
       index += blockDim.x) {
    words[index] = 0;
  }
  constexpr unsigned int kAuxWords =
      sizeof(aux.value) / sizeof(std::int32_t);
  auto* const aux_words = &aux.value[0U][0U][0U][0U][0U];
#pragma unroll 1
  for (unsigned int index = threadIdx.x; index < kAuxWords;
       index += blockDim.x) {
    aux_words[index] = 0;
  }
  __syncthreads();
}

template <unsigned int Projection, unsigned int N8>
__device__ __forceinline__ void accumulate_shared_n8(
    const M64N256MarlinStage& stage,
    std::int32_t (&scratch)
        [kSm87A4W4GateUpDownEdgeM64N256MarlinWarps]
        [kSm87A4W4GateUpDownEdgeM64N256MarlinM16PanelsPerWarp]
        [32U][4U]) noexcept {
  static_assert(Projection < 2U);
  static_assert(N8 < 4U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp_n32 = threadIdx.x >> 5U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    Sm87A4W4Accumulator partial{};
    const unsigned int shared_n8 =
        warp_n32 * 32U + N8 * 8U;
    const std::uint8_t* const shared_b =
        Projection == 0U ? stage.gate[0U] : stage.up[0U];
    const Sm87A4W4BFragment b = load_b_ldmatrix_x2(
        shared_b + shared_n8 * kPackedK64Bytes, lane);
    const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
        stage.a[0U] + panel * 16U * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(partial, a, b);
    std::int32_t* const destination =
        scratch[warp_n32][panel][lane];
    destination[0U] += partial.x0;
    destination[1U] += partial.x1;
    destination[2U] += partial.x2;
    destination[3U] += partial.x3;
  }
}

// The first three N8 fragments remain in registers.  The fourth is folded
// into the current half-edge as exact S32 scratch, one projection at a time,
// keeping the live temporary set to sixteen registers.
__device__ __forceinline__ void accumulate_k64_stage(
    const M64N256MarlinStage& stage,
    Sm87A4W4Accumulator (&gate_partial)[4U][2U],
    Sm87A4W4Accumulator (&up_partial)[4U][2U],
    M64N256MarlinEdgeHalf& half,
    M64N256MarlinAuxScratch& aux) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp_n32 = threadIdx.x >> 5U;
#pragma unroll
  for (unsigned int n8 = 0U; n8 < 2U; ++n8) {
    const unsigned int shared_n8 = warp_n32 * 32U + n8 * 8U;
    const Sm87A4W4BFragment gate = load_b_ldmatrix_x2(
        stage.gate[0U] + shared_n8 * kPackedK64Bytes, lane);
    const Sm87A4W4BFragment up = load_b_ldmatrix_x2(
        stage.up[0U] + shared_n8 * kPackedK64Bytes, lane);
#pragma unroll
    for (unsigned int panel = 0U; panel < 4U; ++panel) {
      const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
          stage.a[0U] + panel * 16U * kPackedK64Bytes, lane);
      sm87_a4w4_mma_m16n8k64(gate_partial[panel][n8], a, gate);
      sm87_a4w4_mma_m16n8k64(up_partial[panel][n8], a, up);
    }
  }
  accumulate_shared_n8<0U, 2U>(stage, aux.value[0U]);
  accumulate_shared_n8<1U, 2U>(stage, aux.value[1U]);
  accumulate_shared_n8<0U, 3U>(stage, half.scratch[0U]);
  accumulate_shared_n8<1U, 3U>(stage, half.scratch[1U]);
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4Accumulator
load_scratch_accumulator(const M64N256MarlinEdgeHalf& half,
                         const unsigned int projection,
                         const unsigned int warp,
                         const unsigned int panel,
                         const unsigned int lane) noexcept {
  const std::int32_t* const source =
      half.scratch[projection][warp][panel][lane];
  return {source[0U], source[1U], source[2U], source[3U]};
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4Accumulator
load_aux_accumulator(const M64N256MarlinAuxScratch& aux,
                     const unsigned int projection,
                     const unsigned int warp,
                     const unsigned int panel,
                     const unsigned int lane) noexcept {
  const std::int32_t* const source =
      aux.value[projection][warp][panel][lane];
  return {source[0U], source[1U], source[2U], source[3U]};
}

__device__ __forceinline__ void fold_scaled_projection(
    Float4& destination, const Sm87A4W4Accumulator& partial,
    const float a0, const float a1, const float b0,
    const float b1) noexcept {
  destination.x0 = __fmaf_rn(
      static_cast<float>(partial.x0), __fmul_rn(a0, b0),
      destination.x0);
  destination.x1 = __fmaf_rn(
      static_cast<float>(partial.x1), __fmul_rn(a0, b1),
      destination.x1);
  destination.x2 = __fmaf_rn(
      static_cast<float>(partial.x2), __fmul_rn(a1, b0),
      destination.x2);
  destination.x3 = __fmaf_rn(
      static_cast<float>(partial.x3), __fmul_rn(a1, b1),
      destination.x3);
}

// All eight K64 stages first close one exact S32 K512 partial.  Only then is
// the BF16 scale product rounded and one FP32 FMA issued.  Both Gate and Up
// long accumulators stay warp-owned for the complete input K.
__device__ __forceinline__ void apply_k512_group(
    Float4 (&gate_accumulator)[4U][4U],
    Float4 (&up_accumulator)[4U][4U],
    const Sm87A4W4Accumulator (&gate_partial)[4U][2U],
    const Sm87A4W4Accumulator (&up_partial)[4U][2U],
    const M64N256MarlinEdgeHalf& half,
    const M64N256MarlinAuxScratch& aux,
    const M64N256MarlinScaleSlot& scale) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp_n32 = threadIdx.x >> 5U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);

#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    const unsigned int row0 = panel * 16U + coordinate0.m;
    const unsigned int row1 = panel * 16U + coordinate2.m;
    const float a0 = decode_bf16(scale.a[row0]);
    const float a1 = decode_bf16(scale.a[row1]);
#pragma unroll
    for (unsigned int n8 = 0U; n8 < 2U; ++n8) {
      const unsigned int column0 =
          warp_n32 * 32U + n8 * 8U + coordinate0.n;
      const unsigned int column1 = column0 + 1U;
      fold_scaled_projection(
          gate_accumulator[panel][n8], gate_partial[panel][n8], a0,
          a1, decode_bf16(scale.gate[column0]),
          decode_bf16(scale.gate[column1]));
      fold_scaled_projection(
          up_accumulator[panel][n8], up_partial[panel][n8], a0, a1,
          decode_bf16(scale.up[column0]),
          decode_bf16(scale.up[column1]));
    }

    constexpr unsigned int kAuxN8 = 2U;
    const unsigned int aux_column0 =
        warp_n32 * 32U + kAuxN8 * 8U + coordinate0.n;
    const unsigned int aux_column1 = aux_column0 + 1U;
    {
      const Sm87A4W4Accumulator partial =
          load_aux_accumulator(aux, 0U, warp_n32, panel, lane);
      fold_scaled_projection(
          gate_accumulator[panel][kAuxN8], partial, a0, a1,
          decode_bf16(scale.gate[aux_column0]),
          decode_bf16(scale.gate[aux_column1]));
    }
    {
      const Sm87A4W4Accumulator partial =
          load_aux_accumulator(aux, 1U, warp_n32, panel, lane);
      fold_scaled_projection(
          up_accumulator[panel][kAuxN8], partial, a0, a1,
          decode_bf16(scale.up[aux_column0]),
          decode_bf16(scale.up[aux_column1]));
    }

    constexpr unsigned int kScratchN8 = 3U;
    const unsigned int scratch_column0 =
        warp_n32 * 32U + kScratchN8 * 8U + coordinate0.n;
    const unsigned int scratch_column1 = scratch_column0 + 1U;
    {
      const Sm87A4W4Accumulator partial =
          load_scratch_accumulator(half, 0U, warp_n32, panel, lane);
      fold_scaled_projection(
          gate_accumulator[panel][kScratchN8], partial, a0, a1,
          decode_bf16(scale.gate[scratch_column0]),
          decode_bf16(scale.gate[scratch_column1]));
    }
    {
      const Sm87A4W4Accumulator partial =
          load_scratch_accumulator(half, 1U, warp_n32, panel, lane);
      fold_scaled_projection(
          up_accumulator[panel][kScratchN8], partial, a0, a1,
          decode_bf16(scale.up[scratch_column0]),
          decode_bf16(scale.up[scratch_column1]));
    }
  }
}

__device__ __forceinline__ void store_bf16_product(
    const Float4 (&gate_accumulator)[4U][4U],
    const Float4 (&up_accumulator)[4U][4U],
    const unsigned int global_m64_start,
    const unsigned int logical_token_count,
    M64N256MarlinEdgeHalf& half) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp_n32 = threadIdx.x >> 5U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    const unsigned int row0 = panel * 16U + coordinate0.m;
    const unsigned int row1 = panel * 16U + coordinate2.m;
    const bool valid0 = global_m64_start + row0 < logical_token_count;
    const bool valid1 = global_m64_start + row1 < logical_token_count;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < 4U; ++n8) {
      const unsigned int column0 =
          warp_n32 * 32U + n8 * 8U + coordinate0.n;
      const unsigned int column1 = column0 + 1U;
      const Float4& gate = gate_accumulator[panel][n8];
      const Float4& up = up_accumulator[panel][n8];
      half.product[row0][column0] =
          valid0
              ? encode_bf16(silu_product(gate.x0, up.x0))
              : 0U;
      half.product[row0][column1] =
          valid0
              ? encode_bf16(silu_product(gate.x1, up.x1))
              : 0U;
      half.product[row1][column0] =
          valid1
              ? encode_bf16(silu_product(gate.x2, up.x2))
              : 0U;
      half.product[row1][column1] =
          valid1
              ? encode_bf16(silu_product(gate.x3, up.x3))
              : 0U;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void compute_cell(
    M64N256MarlinShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m64_start,
    const unsigned int absolute_n256_start,
    const unsigned int cell,
    const unsigned int input_k512_group_count) noexcept {
  Float4 gate_accumulator[4U][4U]{};
  Float4 up_accumulator[4U][4U]{};
  M64N256MarlinEdgeHalf& half = shared.edge.half[cell];
  const unsigned int total_k64 = input_k512_group_count * 8U;

  issue_flat_k64(
      shared.pipeline, packed_a, a_k512_scales_bf16,
      packed_gate_b, gate_b_k512_scales_bf16, packed_up_b,
      up_b_k512_scales_bf16, m64_start, absolute_n256_start, 0U,
      input_k512_group_count);
  issue_flat_k64(
      shared.pipeline, packed_a, a_k512_scales_bf16,
      packed_gate_b, gate_b_k512_scales_bf16, packed_up_b,
      up_b_k512_scales_bf16, m64_start, absolute_n256_start, 1U,
      input_k512_group_count);
  issue_flat_k64(
      shared.pipeline, packed_a, a_k512_scales_bf16,
      packed_gate_b, gate_b_k512_scales_bf16, packed_up_b,
      up_b_k512_scales_bf16, m64_start, absolute_n256_start, 2U,
      input_k512_group_count);

#pragma unroll 1
  for (unsigned int group = 0U; group < input_k512_group_count;
       ++group) {
    clear_scratch(half, shared.aux_scratch);
    Sm87A4W4Accumulator gate_partial[4U][2U];
    Sm87A4W4Accumulator up_partial[4U][2U];
    clear_partials(gate_partial, up_partial);

#pragma unroll
    for (unsigned int phase = 0U; phase < 8U; ++phase) {
      const unsigned int flat_k64 = group * 8U + phase;
      if (flat_k64 + 2U < total_k64) {
        cp_async_wait<2U>();
      } else if (flat_k64 + 1U < total_k64) {
        cp_async_wait<1U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();

      accumulate_k64_stage(
          shared.pipeline.stage[
              flat_k64 %
              kSm87A4W4GateUpDownEdgeM64N256MarlinStages],
          gate_partial,
          up_partial, half, shared.aux_scratch);

      // Complete-reader handoff before the three-stage ring recycles this
      // exact K64 slot.
      __syncthreads();
      if (flat_k64 + 3U < total_k64) {
        issue_flat_k64(
            shared.pipeline, packed_a, a_k512_scales_bf16,
            packed_gate_b, gate_b_k512_scales_bf16, packed_up_b,
            up_b_k512_scales_bf16, m64_start, absolute_n256_start,
            flat_k64 + 3U, input_k512_group_count);
      }
    }

    apply_k512_group(
        gate_accumulator, up_accumulator, gate_partial, up_partial,
        half, shared.aux_scratch,
        shared.pipeline.scale[group & 1U]);
    // Scratch belongs to arbitrary lanes during the next cooperative clear.
    __syncthreads();
  }

  store_bf16_product(gate_accumulator, up_accumulator, m64_start,
                     logical_token_count, half);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t load_edge_value(
    const M64N256MarlinEdge& edge, const unsigned int row,
    const unsigned int column) noexcept {
  const unsigned int half =
      column /
      kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN;
  const unsigned int local_column =
      column -
      half * kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN;
  return edge.half[half].product[row][local_column];
}

__device__ __forceinline__ void quantize_edge(
    const M64N256MarlinEdge& edge,
    const unsigned int m64_start,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int lane = threadIdx.x & 31U;

#pragma unroll 1
  for (unsigned int row_iteration = 0U; row_iteration < 8U;
       ++row_iteration) {
    const unsigned int local_row = warp + row_iteration * 8U;
    const unsigned int global_row = m64_start + local_row;
    float maximum = 0.0F;
#pragma unroll
    for (unsigned int pair = 0U; pair < 8U; ++pair) {
      const unsigned int even_column = 16U * lane + 2U * pair;
      maximum = fmaxf(
          maximum,
          fabsf(decode_bf16(load_edge_value(
              edge, local_row, even_column))));
      maximum = fmaxf(
          maximum,
          fabsf(decode_bf16(load_edge_value(
              edge, local_row, even_column + 1U))));
    }
#pragma unroll
    for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
      maximum = fmaxf(
          maximum, __shfl_down_sync(0xffff'ffffU, maximum, delta));
    }
    maximum = __shfl_sync(0xffff'ffffU, maximum, 0U);
    const float clipped_maximum = maximum * output_clip_ratio;
    std::uint16_t scale_bits = encode_bf16(
        maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
    float stored_scale = decode_bf16(scale_bits);
    if (maximum != 0.0F && stored_scale == 0.0F) {
      scale_bits = 1U;
      stored_scale = decode_bf16(scale_bits);
    }

    const unsigned int physical_group = edge_group * 8U + lane / 4U;
    const unsigned int first_byte = 8U * (lane & 3U);
#pragma unroll
    for (unsigned int pair = 0U; pair < 8U; ++pair) {
      const unsigned int even_column = 16U * lane + 2U * pair;
      const float even_value = decode_bf16(
          load_edge_value(edge, local_row, even_column));
      const float odd_value = decode_bf16(
          load_edge_value(edge, local_row, even_column + 1U));
      const float even = fminf(
          fmaxf(even_value, -clipped_maximum), clipped_maximum);
      const float odd = fminf(
          fmaxf(odd_value, -clipped_maximum), clipped_maximum);
      const int even_rounded = stored_scale == 0.0F
                                   ? 0
                                   : __float2int_rn(even / stored_scale);
      const int odd_rounded = stored_scale == 0.0F
                                  ? 0
                                  : __float2int_rn(odd / stored_scale);
      const int even_code = even_rounded < -7
                                ? -7
                                : (even_rounded > 7 ? 7 : even_rounded);
      const int odd_code = odd_rounded < -7
                               ? -7
                               : (odd_rounded > 7 ? 7 : odd_rounded);
      packed_output[sm87_a4w4_gateup_down_edge_packed_offset(
          global_row, physical_group, first_byte + pair,
          edge_group_count * 8U)] =
          sm87_a4w4_pack_signed_pair(even_code, odd_code);
    }
    if (lane == 0U) {
      output_k512_scales_bf16[
          sm87_a4w4_gateup_down_edge_scale_offset(
              global_row, edge_group, edge_group_count)] = scale_bits;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void compute_edge(
    M64N256MarlinShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m64_tile,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int m64_start =
      m64_tile * kSm87A4W4GateUpDownEdgeM64N256MarlinTileM;
  const unsigned int edge_n_start =
      edge_group * kSm87A4W4GateUpDownEdgeM64N256MarlinTileN;

#pragma unroll 1
  for (unsigned int cell = 0U; cell < 2U; ++cell) {
    compute_cell(
        shared, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, logical_token_count, m64_start,
        edge_n_start +
            cell *
                kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN,
        cell, input_k512_group_count);
  }
  quantize_edge(shared.edge, m64_start, edge_group, edge_group_count,
                output_clip_ratio, packed_output,
                output_k512_scales_bf16);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(
        kSm87A4W4GateUpDownEdgeM64N256MarlinThreads,
        kSm87A4W4GateUpDownEdgeM64N256MarlinCtasPerSm)
void q3x_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m64_tile_count,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) {
  if (gridDim.x !=
          kSm87A4W4GateUpDownEdgeM64N256MarlinPersistentCtas ||
      m64_tile_count == 0U || edge_group_count == 0U ||
      input_k512_group_count == 0U) {
    return;
  }

  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<M64N256MarlinShared*>(dynamic_shared);

  // All resident CTAs traverse the same N512 edge while owning disjoint M64
  // rows.  This is the fixed 16-CTA B-cache phase used by the incumbent
  // persistent schedule; short-M tails simply leave excess CTAs idle.
#pragma unroll 1
  for (unsigned int m64_tile = blockIdx.x; m64_tile < m64_tile_count;
       m64_tile += gridDim.x) {
#pragma unroll 1
    for (unsigned int edge_group = 0U;
         edge_group < edge_group_count; ++edge_group) {
      compute_edge(
          shared, packed_a, a_k512_scales_bf16, packed_gate_b,
          gate_b_k512_scales_bf16, packed_up_b,
          up_b_k512_scales_bf16, logical_token_count, m64_tile,
          edge_group, edge_group_count, input_k512_group_count,
          output_clip_ratio, packed_output,
          output_k512_scales_bf16);
    }
  }
}

namespace {

[[nodiscard]] int validate_target(
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
      properties.multiProcessorCount != kRequiredSmCount ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes ||
      properties.sharedMemPerMultiprocessor <
          kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes));
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
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    const bool require_model_shape,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpDownEdgeM64N256MarlinPlan plan =
      require_model_shape
          ? sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size)
          : sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_test_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size);
  if (plan.launch_ctas !=
          kSm87A4W4GateUpDownEdgeM64N256MarlinPersistentCtas ||
      !(output_clip_ratio > 0.0F && output_clip_ratio <= 1.0F) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k512_scales_bf16, 16U) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k512_scales_bf16, 16U) ||
      !aligned(packed_output, 16U) ||
      !aligned(output_k512_scales_bf16, 16U) ||
      plan.logical_token_count > std::numeric_limits<unsigned int>::max() ||
      plan.m64_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.edge_groups > std::numeric_limits<unsigned int>::max() ||
      plan.input_k512_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.input_physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.output_physical_k64_groups >
          std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          launch_token_count, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          launch_token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          intermediate_size, input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          intermediate_size, input_size);
  const std::size_t required_output_bytes =
      sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          launch_token_count, intermediate_size);
  const std::size_t required_output_scales =
      sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          launch_token_count, intermediate_size);
  if (required_a_bytes == 0U || required_a_scales == 0U ||
      required_b_bytes == 0U || required_b_scales == 0U ||
      required_output_bytes == 0U || required_output_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      gate_b_scale_capacity_elements < required_b_scales ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      up_b_scale_capacity_elements < required_b_scales ||
      packed_output_capacity_bytes < required_output_bytes ||
      output_scale_capacity_elements < required_output_scales ||
      !sm87_a4w4_gateup_down_edge_product_fits(
          required_a_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_gateup_down_edge_product_fits(
          required_b_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_gateup_down_edge_product_fits(
          required_output_scales, sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t required_output_scale_bytes =
      required_output_scales * sizeof(std::uint16_t);
  const auto output_overlaps = [&](const void* const input,
                                   const std::size_t input_bytes) noexcept {
    return byte_ranges_overlap(packed_output, required_output_bytes,
                               input, input_bytes) ||
           byte_ranges_overlap(output_k512_scales_bf16,
                               required_output_scale_bytes,
                               input, input_bytes);
  };
  if (output_overlaps(packed_a, required_a_bytes) ||
      output_overlaps(a_k512_scales_bf16, required_a_scale_bytes) ||
      output_overlaps(packed_gate_b, required_b_bytes) ||
      output_overlaps(gate_b_k512_scales_bf16, required_b_scale_bytes) ||
      output_overlaps(packed_up_b, required_b_bytes) ||
      output_overlaps(up_b_k512_scales_bf16, required_b_scale_bytes) ||
      byte_ranges_overlap(packed_output, required_output_bytes,
                          output_k512_scales_bf16,
                          required_output_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  if (stream != nullptr) {
    const cudaError_t status =
        cudaStreamIsCapturing(stream, &capture_status);
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
  }
  if (capture_status != cudaStreamCaptureStatusNone) {
    if (!g_marlin_resources_ready.load(std::memory_order_acquire)) {
      return static_cast<int>(cudaErrorNotReady);
    }
  } else {
    Sm87A4W4GateUpDownEdgeM64N256MarlinResources resources{};
    const int resource_status =
        query_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_resources_cuda(
            &resources);
    if (resource_status != static_cast<int>(cudaSuccess)) {
      return resource_status;
    }
  }

  (void)cudaGetLastError();
  q3x_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_kernel<<<
      static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeM64N256MarlinPersistentCtas),
      static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeM64N256MarlinThreads),
      kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes,
      stream>>>(
      packed_a, a_k512_scales_bf16, packed_gate_b,
      gate_b_k512_scales_bf16, packed_up_b,
      up_b_k512_scales_bf16,
      static_cast<unsigned int>(plan.logical_token_count),
      static_cast<unsigned int>(plan.m64_tiles),
      static_cast<unsigned int>(plan.edge_groups),
      static_cast<unsigned int>(plan.input_k512_groups),
      output_clip_ratio, packed_output, output_k512_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_resources_cuda(
    Sm87A4W4GateUpDownEdgeM64N256MarlinResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  g_marlin_resources_ready.store(false, std::memory_order_release);
  *resources = Sm87A4W4GateUpDownEdgeM64N256MarlinResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_target(&properties);
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
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_kernel,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM64N256MarlinThreads),
      kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes;
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
              kSm87A4W4GateUpDownEdgeM64N256MarlinMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes ||
      resources->device_shared_per_sm_bytes <
          kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM64N256MarlinThreads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM64N256MarlinCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  g_marlin_resources_ready.store(true, std::memory_order_release);
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_cuda(
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
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_k512_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_k512_scales_bf16,
      up_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size,
      output_clip_ratio, packed_output, packed_output_capacity_bytes,
      output_k512_scales_bf16, output_scale_capacity_elements,
      true, cuda_stream);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_test_cuda(
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
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_k512_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_k512_scales_bf16,
      up_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size,
      output_clip_ratio, packed_output, packed_output_capacity_bytes,
      output_k512_scales_bf16, output_scale_capacity_elements,
      false, cuda_stream);
}

}  // namespace q3x::kernels
