#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m32n512_owner.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(
        kSm87A4W4GateUpDownEdgeM32N512OwnerPackedK64Bytes);

struct alignas(16) M32N512OwnerStage final {
  std::uint8_t a[kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage]
                [kSm87A4W4GateUpDownEdgeM32N512OwnerTileM *
                 kPackedK64Bytes];
  std::uint8_t gate[kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage]
                   [kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN *
                    kPackedK64Bytes];
  std::uint8_t up[kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage]
                 [kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN *
                  kPackedK64Bytes];
};

struct alignas(16) M32N512OwnerScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpDownEdgeM32N512OwnerTileM];
  std::uint16_t gate[
      kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN];
  std::uint16_t up[
      kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN];
};

struct alignas(16) M32N512OwnerPipeline final {
  M32N512OwnerStage
      stage[kSm87A4W4GateUpDownEdgeM32N512OwnerStages];
  M32N512OwnerScaleSlot
      scale[kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlots];
};

// Logical order is [row][pair-within-lane][quantizer-lane].  This is the
// production edge swizzle with only the row extent reduced from M64 to M32.
struct alignas(16) M32N512OwnerEdgePlane final {
  std::uint32_t pair[kSm87A4W4GateUpDownEdgeM32N512OwnerTileM][8U][32U];
};

struct alignas(16) M32N512OwnerShared final {
  M32N512OwnerPipeline pipeline;
  M32N512OwnerEdgePlane edge;
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(M32N512OwnerStage) ==
              kSm87A4W4GateUpDownEdgeM32N512OwnerStageBytes);
static_assert(sizeof(M32N512OwnerScaleSlot) ==
              kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlotBytes);
static_assert(sizeof(M32N512OwnerPipeline) ==
              kSm87A4W4GateUpDownEdgeM32N512OwnerPipelineBytes);
static_assert(sizeof(M32N512OwnerEdgePlane) ==
              kSm87A4W4GateUpDownEdgeM32N512OwnerEdgePlaneBytes);
static_assert(sizeof(M32N512OwnerShared) ==
              kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes);
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

[[nodiscard]] __device__ __forceinline__ unsigned int edge_swizzled_lane(
    const unsigned int row, const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return quantizer_lane ^ ((4U * row + pair_iteration) & 31U);
}

__device__ __forceinline__ void store_edge_pair(
    M32N512OwnerEdgePlane& edge, const unsigned int row,
    const unsigned int logical_even_column,
    const std::uint16_t even_bits, const std::uint16_t odd_bits) noexcept {
  const unsigned int logical_pair = logical_even_column / 2U;
  const unsigned int pair_iteration = logical_pair & 7U;
  const unsigned int quantizer_lane = logical_pair / 8U;
  const unsigned int physical_lane =
      edge_swizzled_lane(row, pair_iteration, quantizer_lane);
  edge.pair[row][pair_iteration][physical_lane] =
      static_cast<std::uint32_t>(even_bits) |
      (static_cast<std::uint32_t>(odd_bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t load_edge_pair(
    const M32N512OwnerEdgePlane& edge, const unsigned int row,
    const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return edge.pair[row][pair_iteration][
      edge_swizzled_lane(row, pair_iteration, quantizer_lane)];
}

// One complete stage is exactly 10,240 bytes.  The first 128 threads issue
// one A vector, while every thread issues one Gate/Up B-vector pair.  Four
// independently committed K128 stages form the exact K512 group.
__device__ __forceinline__ void issue_k128_codes(
    M32N512OwnerStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m32_start,
    const unsigned int absolute_n64_start,
    const unsigned int physical_k128_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      kSm87A4W4GateUpDownEdgeM32N512OwnerTileM *
      kPackedK64Bytes / 16U;
  constexpr unsigned int kBVectorsPerPlane =
      kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN *
      kPackedK64Bytes / 16U;
  constexpr unsigned int kAVectors =
      kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage *
      kAVectorsPerPlane;
  constexpr unsigned int kBVectors =
      kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage *
      kBVectorsPerPlane;
  static_assert(kAVectors * 2U ==
                kSm87A4W4GateUpDownEdgeM32N512OwnerThreads);
  static_assert(kBVectors ==
                kSm87A4W4GateUpDownEdgeM32N512OwnerThreads);

  if (threadIdx.x < kAVectors) {
    const unsigned int a_vector = threadIdx.x;
    const unsigned int a_plane = a_vector / kAVectorsPerPlane;
    const unsigned int a_vector_in_plane =
        a_vector - a_plane * kAVectorsPerPlane;
    const unsigned int a_row = a_vector_in_plane / 2U;
    const unsigned int a_half = a_vector_in_plane & 1U;
    const unsigned int a_physical_k64 =
        physical_k128_group *
            kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage +
        a_plane;
    cp_async_16(
        stage.a[a_plane] +
            sm87_a4w4_swizzled_k64_byte_offset(a_row, 16U * a_half),
        packed_a + sm87_a4w4_gateup_down_edge_packed_offset(
                       static_cast<std::size_t>(m32_start) + a_row,
                       a_physical_k64, 16U * a_half,
                       physical_k64_group_count));
  }

  const unsigned int b_vector = threadIdx.x;
  const unsigned int b_plane = b_vector / kBVectorsPerPlane;
  const unsigned int b_vector_in_plane =
      b_vector - b_plane * kBVectorsPerPlane;
  const unsigned int b_row = b_vector_in_plane / 2U;
  const unsigned int b_half = b_vector_in_plane & 1U;
  const unsigned int b_physical_k64 =
      physical_k128_group *
          kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage +
      b_plane;
  const std::size_t source_offset =
      sm87_a4w4_gateup_down_edge_packed_offset(
          static_cast<std::size_t>(absolute_n64_start) + b_row,
          b_physical_k64, 16U * b_half,
          physical_k64_group_count);
  const std::size_t destination_offset =
      sm87_a4w4_swizzled_k64_byte_offset(b_row, 16U * b_half);
  cp_async_16(stage.gate[b_plane] + destination_offset,
              packed_gate_b + source_offset);
  cp_async_16(stage.up[b_plane] + destination_offset,
              packed_up_b + source_offset);
}

__device__ __forceinline__ void issue_k512_scales(
    M32N512OwnerScaleSlot& slot,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m32_start,
    const unsigned int absolute_n64_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  constexpr unsigned int kRowsPerVector = 8U;
  constexpr unsigned int kAVectors =
      kSm87A4W4GateUpDownEdgeM32N512OwnerTileM / kRowsPerVector;
  constexpr unsigned int kBVectors =
      kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN / kRowsPerVector;
  static_assert(kAVectors == 4U);
  static_assert(kBVectors == 8U);

  if (threadIdx.x < kAVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    cp_async_16(
        slot.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(m32_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x < kBVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    const std::size_t source_offset =
        sm87_a4w4_gateup_down_edge_scale_offset(
            static_cast<std::size_t>(absolute_n64_start) + first_row,
            k512_group, k512_group_count);
    cp_async_16(slot.gate + first_row,
                gate_b_k512_scales_bf16 + source_offset);
    cp_async_16(slot.up + first_row,
                up_b_k512_scales_bf16 + source_offset);
  }
}

__device__ __forceinline__ void issue_k128_stage(
    M32N512OwnerStage& stage,
    M32N512OwnerScaleSlot& scale,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m32_start,
    const unsigned int absolute_n64_start,
    const unsigned int group,
    const unsigned int phase,
    const unsigned int group_count,
    const unsigned int physical_group_count) noexcept {
  issue_k128_codes(stage, packed_a, packed_gate_b, packed_up_b,
                   m32_start, absolute_n64_start, 4U * group + phase,
                   physical_group_count);
  if (phase == 0U) {
    issue_k512_scales(scale, a_k512_scales_bf16,
                      gate_b_k512_scales_bf16,
                      up_b_k512_scales_bf16, m32_start,
                      absolute_n64_start, group, group_count);
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

// Warp w owns N[8*w:8*w+8] and two temporal M16 panels.  Each Gate/Up B
// fragment is loaded once and reused by both panels, keeping the live
// partial set at sixteen registers/thread for both projections together.
__device__ __forceinline__ void accumulate_k128_stage(
    const M32N512OwnerStage& stage,
    Sm87A4W4Accumulator (&gate_partial)[2U],
    Sm87A4W4Accumulator (&up_partial)[2U]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp_n8 = threadIdx.x >> 5U;
  const unsigned int n8_start = warp_n8 * 8U;

#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage;
       ++plane) {
    const Sm87A4W4BFragment gate = load_b_ldmatrix_x2(
        stage.gate[plane] + n8_start * kPackedK64Bytes, lane);
    const Sm87A4W4BFragment up = load_b_ldmatrix_x2(
        stage.up[plane] + n8_start * kPackedK64Bytes, lane);
#pragma unroll
    for (unsigned int panel = 0U;
         panel <
             kSm87A4W4GateUpDownEdgeM32N512OwnerM16PanelsPerWarp;
         ++panel) {
      const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
          stage.a[plane] + panel * 16U * kPackedK64Bytes, lane);
      sm87_a4w4_mma_m16n8k64(gate_partial[panel], a, gate);
      sm87_a4w4_mma_m16n8k64(up_partial[panel], a, up);
    }
  }
}

// The scale/FMA boundary is deliberately identical to production: the eight
// physical K64 contributions first form one S32 K512 partial.  Only then is
// one BF16 scale product rounded with __fmul_rn and accumulated with one
// __fmaf_rn.  The caller invokes this in increasing K512-group order.
__device__ __forceinline__ void apply_k512_group(
    Float4 (&gate_accumulator)[2U],
    Float4 (&up_accumulator)[2U],
    const Sm87A4W4Accumulator (&gate_partial)[2U],
    const Sm87A4W4Accumulator (&up_partial)[2U],
    const M32N512OwnerScaleSlot& scale) noexcept {
  constexpr unsigned int kMask = 0xffff'ffffU;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp_n8 = threadIdx.x >> 5U;
  const unsigned int row_in_half = lane >> 2U;
  const unsigned int n_even = 2U * (lane & 3U);
  const unsigned int n_odd = n_even + 1U;
  const float a_owned = decode_bf16(scale.a[lane]);
  const float gate_owned =
      lane < 8U
          ? decode_bf16(scale.gate[warp_n8 * 8U + lane])
          : 0.0F;
  const float up_owned =
      lane < 8U
          ? decode_bf16(scale.up[warp_n8 * 8U + lane])
          : 0.0F;
  const float gate0 = __shfl_sync(kMask, gate_owned, n_even);
  const float gate1 = __shfl_sync(kMask, gate_owned, n_odd);
  const float up0 = __shfl_sync(kMask, up_owned, n_even);
  const float up1 = __shfl_sync(kMask, up_owned, n_odd);

#pragma unroll
  for (unsigned int panel = 0U;
       panel < kSm87A4W4GateUpDownEdgeM32N512OwnerM16PanelsPerWarp;
       ++panel) {
    const unsigned int row0 = panel * 16U + row_in_half;
    const float a0 = __shfl_sync(kMask, a_owned, row0);
    const float a1 = __shfl_sync(kMask, a_owned, row0 + 8U);
    const float gate00 = __fmul_rn(a0, gate0);
    const float gate01 = __fmul_rn(a0, gate1);
    const float gate10 = __fmul_rn(a1, gate0);
    const float gate11 = __fmul_rn(a1, gate1);
    const Sm87A4W4Accumulator& gate = gate_partial[panel];
    Float4& gate_output = gate_accumulator[panel];
    gate_output.x0 = __fmaf_rn(static_cast<float>(gate.x0), gate00,
                               gate_output.x0);
    gate_output.x1 = __fmaf_rn(static_cast<float>(gate.x1), gate01,
                               gate_output.x1);
    gate_output.x2 = __fmaf_rn(static_cast<float>(gate.x2), gate10,
                               gate_output.x2);
    gate_output.x3 = __fmaf_rn(static_cast<float>(gate.x3), gate11,
                               gate_output.x3);

    const float up00 = __fmul_rn(a0, up0);
    const float up01 = __fmul_rn(a0, up1);
    const float up10 = __fmul_rn(a1, up0);
    const float up11 = __fmul_rn(a1, up1);
    const Sm87A4W4Accumulator& up = up_partial[panel];
    Float4& up_output = up_accumulator[panel];
    up_output.x0 = __fmaf_rn(static_cast<float>(up.x0), up00,
                             up_output.x0);
    up_output.x1 = __fmaf_rn(static_cast<float>(up.x1), up01,
                             up_output.x1);
    up_output.x2 = __fmaf_rn(static_cast<float>(up.x2), up10,
                             up_output.x2);
    up_output.x3 = __fmaf_rn(static_cast<float>(up.x3), up11,
                             up_output.x3);
  }
}

__device__ __forceinline__ void store_bf16_product_n64(
    M32N512OwnerEdgePlane& edge,
    const Float4 (&gate_accumulator)[2U],
    const Float4 (&up_accumulator)[2U],
    const unsigned int global_m32_start,
    const unsigned int logical_token_count,
    const unsigned int cell_in_edge) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp_n8 = threadIdx.x >> 5U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const unsigned int fragment_n =
      cell_in_edge *
          kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN +
      warp_n8 * 8U + coordinate0.n;

#pragma unroll
  for (unsigned int panel = 0U;
       panel < kSm87A4W4GateUpDownEdgeM32N512OwnerM16PanelsPerWarp;
       ++panel) {
    const unsigned int row0 = panel * 16U + coordinate0.m;
    const unsigned int row1 = panel * 16U + coordinate2.m;
    const bool valid0 = global_m32_start + row0 < logical_token_count;
    const bool valid1 = global_m32_start + row1 < logical_token_count;
    const Float4& gate = gate_accumulator[panel];
    const Float4& up = up_accumulator[panel];
    const std::uint16_t bits00 =
        valid0 ? encode_bf16(silu_product(gate.x0, up.x0)) : 0U;
    const std::uint16_t bits01 =
        valid0 ? encode_bf16(silu_product(gate.x1, up.x1)) : 0U;
    const std::uint16_t bits10 =
        valid1 ? encode_bf16(silu_product(gate.x2, up.x2)) : 0U;
    const std::uint16_t bits11 =
        valid1 ? encode_bf16(silu_product(gate.x3, up.x3)) : 0U;
    store_edge_pair(edge, row0, fragment_n, bits00, bits01);
    store_edge_pair(edge, row1, fragment_n, bits10, bits11);
  }
}

__device__ __forceinline__ void compute_n64_cell(
    M32N512OwnerShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m32_start,
    const unsigned int absolute_n64_start,
    const unsigned int cell_in_edge,
    const unsigned int input_k512_group_count) noexcept {
  Float4 gate_accumulator[2U]{};
  Float4 up_accumulator[2U]{};
  const unsigned int physical_group_count =
      input_k512_group_count * 8U;

  // Prime all four K128 phases as independent async groups.  wait<3>
  // exposes phase zero while leaving three groups eligible to overlap its
  // compute.
#pragma unroll
  for (unsigned int phase = 0U;
       phase < kSm87A4W4GateUpDownEdgeM32N512OwnerStages; ++phase) {
    issue_k128_stage(
        shared.pipeline.stage[phase], shared.pipeline.scale[0U],
        packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, m32_start, absolute_n64_start, 0U,
        phase, input_k512_group_count, physical_group_count);
  }
  cp_async_wait<3U>();
  __syncthreads();

#pragma unroll 1
  for (unsigned int group = 0U; group < input_k512_group_count;
       ++group) {
    Sm87A4W4Accumulator gate_partial[2U]{};
    Sm87A4W4Accumulator up_partial[2U]{};
    const unsigned int next_group = group + 1U;
#pragma unroll
    for (unsigned int phase = 0U;
         phase < kSm87A4W4GateUpDownEdgeM32N512OwnerStages; ++phase) {
      accumulate_k128_stage(shared.pipeline.stage[phase], gate_partial,
                            up_partial);
      if (phase + 1U ==
          kSm87A4W4GateUpDownEdgeM32N512OwnerStages) {
        apply_k512_group(
            gate_accumulator, up_accumulator, gate_partial, up_partial,
            shared.pipeline.scale[group & 1U]);
      }

      if (next_group < input_k512_group_count) {
        // The ring normally retains three outstanding groups.  Waiting to
        // two makes the next phase visible; the barrier simultaneously
        // releases the current slot before it is refilled for group+1.
        cp_async_wait<2U>();
        __syncthreads();
        issue_k128_stage(
            shared.pipeline.stage[phase],
            shared.pipeline.scale[next_group & 1U], packed_a,
            a_k512_scales_bf16, packed_gate_b,
            gate_b_k512_scales_bf16, packed_up_b,
            up_b_k512_scales_bf16, m32_start, absolute_n64_start,
            next_group, phase, input_k512_group_count,
            physical_group_count);
      } else if (phase == 0U) {
        // No refills keep the outstanding count at three in the final
        // group.  Drain once after phase zero, then consume the remaining
        // immutable stages without redundant CTA barriers.
        cp_async_wait<0U>();
        __syncthreads();
      }
    }
  }

  store_bf16_product_n64(shared.edge, gate_accumulator, up_accumulator,
                         m32_start, logical_token_count, cell_in_edge);

  // Reader release for all four pipeline stages and complete product
  // publication before the next N64 cell recycles the ring.
  __syncthreads();
}

__device__ __forceinline__ void quantize_edge(
    const M32N512OwnerEdgePlane& edge,
    const unsigned int m32_start,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int lane = threadIdx.x & 31U;

#pragma unroll 1
  for (unsigned int row_iteration = 0U; row_iteration < 4U;
       ++row_iteration) {
    const unsigned int local_row =
        warp + row_iteration *
                   kSm87A4W4GateUpDownEdgeM32N512OwnerWarps;
    const unsigned int global_row = m32_start + local_row;
    float maximum = 0.0F;
#pragma unroll 1
    for (unsigned int pair = 0U; pair < 8U; ++pair) {
      const std::uint32_t word =
          load_edge_pair(edge, local_row, pair, lane);
      maximum = fmaxf(
          maximum,
          fabsf(decode_bf16(static_cast<std::uint16_t>(word))));
      maximum = fmaxf(
          maximum,
          fabsf(decode_bf16(static_cast<std::uint16_t>(word >> 16U))));
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
#pragma unroll 1
    for (unsigned int pair = 0U; pair < 8U; ++pair) {
      const std::uint32_t word =
          load_edge_pair(edge, local_row, pair, lane);
      const float even_value =
          decode_bf16(static_cast<std::uint16_t>(word));
      const float odd_value =
          decode_bf16(static_cast<std::uint16_t>(word >> 16U));
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
    M32N512OwnerShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m32_tile,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int m32_start =
      m32_tile * kSm87A4W4GateUpDownEdgeM32N512OwnerTileM;
  const unsigned int edge_n_start =
      edge_group * kSm87A4W4GateUpDownEdgeM32N512OwnerTileN;

#pragma unroll 1
  for (unsigned int cell = 0U;
       cell < kSm87A4W4GateUpDownEdgeM32N512OwnerCellsPerEdge;
       ++cell) {
    compute_n64_cell(
        shared, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, logical_token_count, m32_start,
        edge_n_start +
            cell *
                kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN,
        cell, input_k512_group_count);
  }
  quantize_edge(shared.edge, m32_start, edge_group, edge_group_count,
                output_clip_ratio, packed_output,
                output_k512_scales_bf16);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpDownEdgeM32N512OwnerThreads,
                      kSm87A4W4GateUpDownEdgeM32N512OwnerCtasPerSm)
void q3x_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resource_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m64_pair_count,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) {
  if (gridDim.x < 2U || (gridDim.x & 1U) != 0U ||
      m64_pair_count == 0U || edge_group_count == 0U ||
      input_k512_group_count == 0U) {
    return;
  }

  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<M32N512OwnerShared*>(dynamic_shared);
  const unsigned int team = blockIdx.x >> 1U;
  const unsigned int m32_lane = blockIdx.x & 1U;
  const unsigned int team_count = gridDim.x >> 1U;
  const unsigned int base_waves = m64_pair_count / team_count;
  const unsigned int base_pairs = base_waves * team_count;
  const unsigned int residual_pairs = m64_pair_count - base_pairs;

  // Base waves keep every logical CTA pair on adjacent M32 rows while all
  // teams traverse the same N512 stream.  Pairing is a cache hint only; no
  // correctness condition depends on physical block-to-SM placement.
#pragma unroll 1
  for (unsigned int wave = 0U; wave < base_waves; ++wave) {
    const unsigned int m64_pair = wave * team_count + team;
    const unsigned int m32_tile = 2U * m64_pair + m32_lane;
#pragma unroll 1
    for (unsigned int edge_group = 0U;
         edge_group < edge_group_count; ++edge_group) {
      compute_edge(
          shared, packed_a, a_k512_scales_bf16, packed_gate_b,
          gate_b_k512_scales_bf16, packed_up_b,
          up_b_k512_scales_bf16, logical_token_count, m32_tile,
          edge_group, edge_group_count, input_k512_group_count,
          output_clip_ratio, packed_output,
          output_k512_scales_bf16);
    }
  }

  if (residual_pairs != 0U) {
    const unsigned int residual_cells =
        residual_pairs * edge_group_count;
#pragma unroll 1
    for (unsigned int iteration = 0U;; ++iteration) {
      const unsigned int ordinal = team + iteration * team_count;
      if (ordinal >= residual_cells) {
        break;
      }
      const unsigned int edge_group = ordinal / residual_pairs;
      const unsigned int residual_pair =
          ordinal - edge_group * residual_pairs;
      const unsigned int m64_pair = base_pairs + residual_pair;
      const unsigned int m32_tile = 2U * m64_pair + m32_lane;
      compute_edge(
          shared, packed_a, a_k512_scales_bf16, packed_gate_b,
          gate_b_k512_scales_bf16, packed_up_b,
          up_b_k512_scales_bf16, logical_token_count, m32_tile,
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
          kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes ||
      properties.sharedMemPerMultiprocessor <
          kSm87A4W4GateUpDownEdgeM32N512OwnerCtasPerSm *
              kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resource_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes));
}

}  // namespace

int query_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resources_cuda(
    Sm87A4W4GateUpDownEdgeM32N512OwnerResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpDownEdgeM32N512OwnerResources{};
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
      q3x_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resource_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resource_kernel,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM32N512OwnerThreads),
      kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes;
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
              kSm87A4W4GateUpDownEdgeM32N512OwnerMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes ||
      resources->device_shared_per_sm_bytes <
          kSm87A4W4GateUpDownEdgeM32N512OwnerCtasPerSm *
              kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM32N512OwnerThreads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM32N512OwnerCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
