#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kPackedK64Bytes = 32U;
inline constexpr int kRequiredSmCount = 16;
std::atomic<bool> g_paired_ldmatrix_resources_ready{false};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

struct alignas(16) PairedBFragment final {
  Sm87A4W4BFragment gate;
  Sm87A4W4BFragment up;
};

struct alignas(16) PairedLdmatrixStage final {
  std::uint8_t
      a[kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixK64PerStage]
       [kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixTileM *
        kPackedK64Bytes];
  PairedBFragment
      paired_b[kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixK64PerStage]
              [kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixN8Warps]
              [32U];
};

struct alignas(16) PairedLdmatrixScaleSlot final {
  std::uint16_t
      a[kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixTileM];
  std::uint16_t
      paired_b[kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixComputeTileN]
              [2U];
};

struct alignas(16) PairedLdmatrixPipeline final {
  PairedLdmatrixStage
      stage[kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixStages];
  PairedLdmatrixScaleSlot
      scale[kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixStages];
};

struct alignas(16) PairedLdmatrixEdgePlane final {
  std::uint32_t pair[64U][8U][32U];
};

struct alignas(16) PairedLdmatrixShared final {
  PairedLdmatrixPipeline pipeline;
  PairedLdmatrixEdgePlane edge;
};

static_assert(sizeof(Float4) == 16U);
static_assert(sizeof(PairedBFragment) == 16U);
static_assert(sizeof(PairedLdmatrixStage) ==
              kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixStageBytes);
static_assert(sizeof(PairedLdmatrixScaleSlot) ==
              kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixScaleSlotBytes);
static_assert(sizeof(PairedLdmatrixPipeline) ==
              kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPipelineBytes);
static_assert(sizeof(PairedLdmatrixEdgePlane) ==
              kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixEdgePlaneBytes);
static_assert(sizeof(PairedLdmatrixShared) ==
              kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixDynamicSharedBytes);

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
    PairedLdmatrixEdgePlane& edge, const unsigned int row,
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
    const PairedLdmatrixEdgePlane& edge, const unsigned int row,
    const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return edge.pair[row][pair_iteration][
      edge_swizzled_lane(row, pair_iteration, quantizer_lane)];
}

// Each of 512 threads issues two A vectors and two paired-B records.  The B
// record is one already-authenticated Gate/Up v2 fragment and is shared by
// the upper/lower M64 warp pair.
__device__ __forceinline__ void issue_k256_codes(
    PairedLdmatrixStage& destination,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const paired_b_codes,
    const unsigned int m_tile_start,
    const unsigned int absolute_n64_start,
    const unsigned int physical_k256_group,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixTileM *
      kPackedK64Bytes / 16U;
  constexpr unsigned int kVectors =
      kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixK64PerStage *
      kAVectorsPerPlane;
  static_assert(kVectors ==
                2U *
                    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixThreads);
#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector =
        threadIdx.x +
        iteration *
            kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixThreads;
    const unsigned int plane = vector / kAVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kAVectorsPerPlane;
    const unsigned int row = vector_in_plane / 2U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int physical_k64 =
        physical_k256_group *
            kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixK64PerStage +
        plane;
    cp_async_16(
        destination.a[plane] +
            sm87_a4w4_swizzled_k64_byte_offset(
                row, 16U * row_vector),
        packed_a + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64, 16U * row_vector,
                       physical_k64_group_count));

    const unsigned int n8 = vector_in_plane / 32U;
    const unsigned int lane = vector_in_plane & 31U;
    const unsigned int k512_group = physical_k256_group / 2U;
    const unsigned int k64_in_group =
        (physical_k256_group & 1U) *
            kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixK64PerStage +
        plane;
    const std::size_t b_offset =
        sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
            static_cast<std::size_t>(absolute_n64_start) + n8 * 8U,
            k512_group, k64_in_group, lane, k512_group_count);
    cp_async_16(&destination.paired_b[plane][n8][lane],
                paired_b_codes + b_offset);
  }
}

__device__ __forceinline__ void issue_scales(
    PairedLdmatrixScaleSlot& destination,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n64_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        destination.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x >= 16U && threadIdx.x < 32U) {
    const unsigned int first_row = 4U * (threadIdx.x - 16U);
    cp_async_16(
        destination.paired_b[first_row],
        paired_b_scales_bf16 +
            sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
                static_cast<std::size_t>(absolute_n64_start) + first_row,
                k512_group, k512_group_count));
  }
}

__device__ __forceinline__ void issue_even_k256_and_scales(
    PairedLdmatrixStage& stage,
    PairedLdmatrixScaleSlot& scale,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const paired_b_codes,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n64_start,
    const unsigned int group,
    const unsigned int group_count,
    const unsigned int physical_group_count) noexcept {
  issue_k256_codes(stage, packed_a, paired_b_codes, m_tile_start,
                   absolute_n64_start, 2U * group, group_count,
                   physical_group_count);
  issue_scales(scale, a_k512_scales_bf16, paired_b_scales_bf16,
               m_tile_start, absolute_n64_start, group, group_count);
  cp_async_commit();
}

__device__ __forceinline__ void issue_odd_k256(
    PairedLdmatrixStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const paired_b_codes,
    const unsigned int m_tile_start,
    const unsigned int absolute_n64_start,
    const unsigned int group,
    const unsigned int group_count,
    const unsigned int physical_group_count) noexcept {
  issue_k256_codes(stage, packed_a, paired_b_codes, m_tile_start,
                   absolute_n64_start, 2U * group + 1U, group_count,
                   physical_group_count);
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

[[nodiscard]] __device__ __forceinline__ PairedBFragment
load_paired_b_shared(const PairedBFragment* const pointer) noexcept {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
  PairedBFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.gate.x0), "=r"(fragment.gate.x1),
        "=r"(fragment.up.x0), "=r"(fragment.up.x1)
      : "r"(shared_address)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return fragment;
}

__device__ __forceinline__ void accumulate_k256_stage(
    const PairedLdmatrixStage& stage,
    const unsigned int warp_n8,
    const unsigned int m_half,
    Sm87A4W4Accumulator (&gate_partial)[4U],
    Sm87A4W4Accumulator (&up_partial)[4U]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixK64PerStage;
       ++plane) {
    const PairedBFragment b =
        load_paired_b_shared(&stage.paired_b[plane][warp_n8][lane]);
#pragma unroll
    for (unsigned int panel = 0U; panel < 4U; ++panel) {
      const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
          stage.a[plane] +
              (m_half * 64U + panel * 16U) * kPackedK64Bytes,
          lane);
      sm87_a4w4_mma_m16n8k64(gate_partial[panel], a, b.gate);
      sm87_a4w4_mma_m16n8k64(up_partial[panel], a, b.up);
    }
  }
}

__device__ __forceinline__ void apply_k512_group(
    Float4 (&gate_accumulator)[4U],
    Float4 (&up_accumulator)[4U],
    const Sm87A4W4Accumulator (&gate_partial)[4U],
    const Sm87A4W4Accumulator (&up_partial)[4U],
    const PairedLdmatrixScaleSlot& scale,
    const unsigned int warp_n8,
    const unsigned int m_half) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int lane_column = lane & 3U;
  const unsigned int first_n = warp_n8 * 8U + lane_column * 2U;
  const float gate_scale0 = decode_bf16(scale.paired_b[first_n][0U]);
  const float up_scale0 = decode_bf16(scale.paired_b[first_n][1U]);
  const float gate_scale1 = decode_bf16(scale.paired_b[first_n + 1U][0U]);
  const float up_scale1 = decode_bf16(scale.paired_b[first_n + 1U][1U]);
  const unsigned int row_in_fragment = lane >> 2U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    const unsigned int first_m =
        m_half * 64U + panel * 16U + row_in_fragment;
    const float a_scale0 = decode_bf16(scale.a[first_m]);
    const float a_scale1 = decode_bf16(scale.a[first_m + 8U]);
    const float gate00 = __fmul_rn(a_scale0, gate_scale0);
    const float gate01 = __fmul_rn(a_scale0, gate_scale1);
    const float gate10 = __fmul_rn(a_scale1, gate_scale0);
    const float gate11 = __fmul_rn(a_scale1, gate_scale1);
    const float up00 = __fmul_rn(a_scale0, up_scale0);
    const float up01 = __fmul_rn(a_scale0, up_scale1);
    const float up10 = __fmul_rn(a_scale1, up_scale0);
    const float up11 = __fmul_rn(a_scale1, up_scale1);
    gate_accumulator[panel].x0 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x0), gate00,
        gate_accumulator[panel].x0);
    gate_accumulator[panel].x1 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x1), gate01,
        gate_accumulator[panel].x1);
    gate_accumulator[panel].x2 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x2), gate10,
        gate_accumulator[panel].x2);
    gate_accumulator[panel].x3 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x3), gate11,
        gate_accumulator[panel].x3);
    up_accumulator[panel].x0 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x0), up00,
        up_accumulator[panel].x0);
    up_accumulator[panel].x1 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x1), up01,
        up_accumulator[panel].x1);
    up_accumulator[panel].x2 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x2), up10,
        up_accumulator[panel].x2);
    up_accumulator[panel].x3 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x3), up11,
        up_accumulator[panel].x3);
  }
}

__device__ __forceinline__ void store_bf16_product_n64(
    PairedLdmatrixEdgePlane& shared_half,
    PairedLdmatrixEdgePlane& scratch_half,
    const Float4 (&gate_accumulator)[4U],
    const Float4 (&up_accumulator)[4U],
    const unsigned int global_m_start,
    const unsigned int logical_token_count,
    const unsigned int cell_in_edge,
    const unsigned int warp_n8,
    const unsigned int m_half) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const unsigned int fragment_n =
      cell_in_edge *
          kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixComputeTileN +
      warp_n8 * 8U + coordinate0.n;
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    const unsigned int row0 =
        m_half * 64U + panel * 16U + coordinate0.m;
    const unsigned int row1 =
        m_half * 64U + panel * 16U + coordinate2.m;
    const bool valid0 = global_m_start + row0 < logical_token_count;
    const bool valid1 = global_m_start + row1 < logical_token_count;
    const std::uint16_t bits00 =
        valid0 ? encode_bf16(silu_product(gate_accumulator[panel].x0,
                                          up_accumulator[panel].x0))
               : 0U;
    const std::uint16_t bits01 =
        valid0 ? encode_bf16(silu_product(gate_accumulator[panel].x1,
                                          up_accumulator[panel].x1))
               : 0U;
    const std::uint16_t bits10 =
        valid1 ? encode_bf16(silu_product(gate_accumulator[panel].x2,
                                          up_accumulator[panel].x2))
               : 0U;
    const std::uint16_t bits11 =
        valid1 ? encode_bf16(silu_product(gate_accumulator[panel].x3,
                                          up_accumulator[panel].x3))
               : 0U;
    PairedLdmatrixEdgePlane& edge0 =
        row0 < 64U ? shared_half : scratch_half;
    PairedLdmatrixEdgePlane& edge1 =
        row1 < 64U ? shared_half : scratch_half;
    store_edge_pair(edge0, row0 & 63U, fragment_n, bits00, bits01);
    store_edge_pair(edge1, row1 & 63U, fragment_n, bits10, bits11);
  }
}

__device__ __forceinline__ void compute_n64_cell(
    PairedLdmatrixShared& shared,
    PairedLdmatrixEdgePlane& scratch_half,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const paired_b_codes,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int absolute_n64_start,
    const unsigned int cell_in_edge,
    const unsigned int input_k512_group_count) noexcept {
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_n8 = warp & 7U;
  const unsigned int m_half = warp >> 3U;
  Float4 gate_accumulator[4U]{};
  Float4 up_accumulator[4U]{};

  issue_even_k256_and_scales(
      shared.pipeline.stage[0U], shared.pipeline.scale[0U], packed_a,
      a_k512_scales_bf16, paired_b_codes, paired_b_scales_bf16,
      m_tile_start, absolute_n64_start, 0U, input_k512_group_count,
      input_k512_group_count * 8U);
  issue_odd_k256(
      shared.pipeline.stage[1U], packed_a, paired_b_codes, m_tile_start,
      absolute_n64_start, 0U, input_k512_group_count,
      input_k512_group_count * 8U);
  cp_async_wait<1U>();
  __syncthreads();

  for (unsigned int group = 0U; group < input_k512_group_count;
       ++group) {
    Sm87A4W4Accumulator gate_partial[4U]{};
    Sm87A4W4Accumulator up_partial[4U]{};
    accumulate_k256_stage(shared.pipeline.stage[0U], warp_n8, m_half,
                          gate_partial, up_partial);

    // One handoff makes odd visible and releases even.  The next even stage
    // is published while odd computes.
    cp_async_wait<0U>();
    __syncthreads();
    const unsigned int next_group = group + 1U;
    if (next_group < input_k512_group_count) {
      issue_even_k256_and_scales(
          shared.pipeline.stage[0U],
          shared.pipeline.scale[next_group & 1U], packed_a,
          a_k512_scales_bf16, paired_b_codes,
          paired_b_scales_bf16, m_tile_start, absolute_n64_start,
          next_group, input_k512_group_count,
          input_k512_group_count * 8U);
    }
    accumulate_k256_stage(shared.pipeline.stage[1U], warp_n8, m_half,
                          gate_partial, up_partial);
    apply_k512_group(gate_accumulator, up_accumulator, gate_partial,
                     up_partial, shared.pipeline.scale[group & 1U],
                     warp_n8, m_half);
    if (next_group < input_k512_group_count) {
      cp_async_wait<0U>();
      __syncthreads();
      issue_odd_k256(
          shared.pipeline.stage[1U], packed_a, paired_b_codes,
          m_tile_start, absolute_n64_start, next_group,
          input_k512_group_count, input_k512_group_count * 8U);
    }
  }
  store_bf16_product_n64(
      shared.edge, scratch_half, gate_accumulator, up_accumulator,
      m_tile_start, logical_token_count, cell_in_edge, warp_n8, m_half);
  __syncthreads();
}

__device__ __forceinline__ void quantize_m64_half(
    const PairedLdmatrixEdgePlane& edge,
    const unsigned int global_m_start,
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
    const unsigned int local_row = warp + row_iteration * 16U;
    const unsigned int global_row = global_m_start + local_row;
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

__device__ __forceinline__ void refill_shared_edge_from_scratch(
    PairedLdmatrixEdgePlane& shared_edge,
    const PairedLdmatrixEdgePlane& scratch_edge) noexcept {
  constexpr unsigned int kVectors =
      kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixEdgePlaneBytes /
      sizeof(uint4);
  auto* const destination = reinterpret_cast<uint4*>(&shared_edge);
  const auto* const source = reinterpret_cast<const uint4*>(&scratch_edge);
  for (unsigned int vector = threadIdx.x; vector < kVectors;
       vector += blockDim.x) {
    destination[vector] = source[vector];
  }
  __syncthreads();
}

__device__ __forceinline__ void compute_edge_cell(
    PairedLdmatrixShared& shared,
    PairedLdmatrixEdgePlane& scratch_half,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const paired_b_codes,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int m_tile_start =
      m_tile * kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixTileM;
  const unsigned int edge_n_start =
      edge_group * kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixTileN;
#pragma unroll 1
  for (unsigned int cell = 0U;
       cell <
       kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixCellsPerEdge;
       ++cell) {
    compute_n64_cell(
        shared, scratch_half, packed_a, a_k512_scales_bf16,
        paired_b_codes, paired_b_scales_bf16, logical_token_count,
        m_tile_start,
        edge_n_start +
            cell *
                kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixComputeTileN,
        cell, input_k512_group_count);
  }
  quantize_m64_half(shared.edge, m_tile_start, edge_group,
                    edge_group_count, output_clip_ratio, packed_output,
                    output_k512_scales_bf16);
  refill_shared_edge_from_scratch(shared.edge, scratch_half);
  quantize_m64_half(shared.edge, m_tile_start + 64U, edge_group,
                    edge_group_count, output_clip_ratio, packed_output,
                    output_k512_scales_bf16);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(
        kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixThreads,
        kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixCtasPerSm)
void q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const paired_b_codes,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile_count,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const cta_scratch,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared = *reinterpret_cast<PairedLdmatrixShared*>(dynamic_shared);
  auto& scratch_half = *reinterpret_cast<PairedLdmatrixEdgePlane*>(
      cta_scratch +
      static_cast<std::size_t>(blockIdx.x) *
          kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixScratchBytesPerCta);

  const unsigned int base_waves = m_tile_count / gridDim.x;
  const unsigned int base_m_tiles = base_waves * gridDim.x;
  const unsigned int residual_m_tiles = m_tile_count - base_m_tiles;
  const unsigned int base_iterations = base_waves * edge_group_count;
  const unsigned int residual_cells =
      residual_m_tiles * edge_group_count;
  for (unsigned int iteration = 0U;; ++iteration) {
    unsigned int m_tile = 0U;
    unsigned int edge_group = 0U;
    if (iteration < base_iterations) {
      const unsigned int wave = iteration / edge_group_count;
      edge_group = iteration - wave * edge_group_count;
      m_tile = wave * gridDim.x + blockIdx.x;
    } else {
      const unsigned int residual_iteration =
          iteration - base_iterations;
      const unsigned int ordinal =
          blockIdx.x + residual_iteration * gridDim.x;
      if (ordinal >= residual_cells) {
        break;
      }
      edge_group = ordinal / residual_m_tiles;
      m_tile = base_m_tiles + ordinal - edge_group * residual_m_tiles;
    }
    compute_edge_cell(
        shared, scratch_half, packed_a, a_k512_scales_bf16,
        paired_b_codes, paired_b_scales_bf16, logical_token_count,
        m_tile, edge_group, edge_group_count, input_k512_group_count,
        output_clip_ratio, packed_output, output_k512_scales_bf16);
  }
}

namespace {

[[nodiscard]] constexpr bool aligned(
    const void* const pointer, const std::size_t alignment) noexcept {
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
  return first_begin < second_begin + second_bytes &&
         second_begin < first_begin + first_bytes;
}

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
          kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixDynamicSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const paired_b_codes,
    const std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* const paired_b_scales_bf16,
    const std::size_t paired_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const cta_scratch,
    const std::size_t cta_scratch_capacity_bytes,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool require_model_shape,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPlan plan =
      require_model_shape
          ? sm87_a4w4_gateup_down_edge_m128n512_paired_ldmatrix_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size)
          : sm87_a4w4_gateup_down_edge_m128n512_paired_ldmatrix_test_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size, maximum_launch_ctas);
  if (plan.launch_ctas == 0U ||
      !(output_clip_ratio > 0.0F && output_clip_ratio <= 1.0F) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(paired_b_codes, 16U) ||
      !aligned(paired_b_scales_bf16, 16U) ||
      !aligned(cta_scratch, 16U) || !aligned(packed_output, 16U) ||
      !aligned(output_k512_scales_bf16, 16U) ||
      cta_scratch_capacity_bytes < plan.required_scratch_bytes ||
      plan.logical_token_count > std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.edge_groups > std::numeric_limits<unsigned int>::max() ||
      plan.input_k512_groups >
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
      sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
          intermediate_size, input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
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
      paired_b_code_capacity_bytes < required_b_bytes ||
      paired_b_scale_capacity_elements < required_b_scales ||
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
                                   const std::size_t bytes) noexcept {
    return byte_ranges_overlap(packed_output, required_output_bytes,
                               input, bytes) ||
           byte_ranges_overlap(output_k512_scales_bf16,
                               required_output_scale_bytes,
                               input, bytes);
  };
  const auto scratch_overlaps = [&](const void* const input,
                                    const std::size_t bytes) noexcept {
    return byte_ranges_overlap(cta_scratch, plan.required_scratch_bytes,
                               input, bytes);
  };
  if (output_overlaps(packed_a, required_a_bytes) ||
      output_overlaps(a_k512_scales_bf16, required_a_scale_bytes) ||
      output_overlaps(paired_b_codes, required_b_bytes) ||
      output_overlaps(paired_b_scales_bf16, required_b_scale_bytes) ||
      byte_ranges_overlap(packed_output, required_output_bytes,
                          output_k512_scales_bf16,
                          required_output_scale_bytes) ||
      scratch_overlaps(packed_a, required_a_bytes) ||
      scratch_overlaps(a_k512_scales_bf16, required_a_scale_bytes) ||
      scratch_overlaps(paired_b_codes, required_b_bytes) ||
      scratch_overlaps(paired_b_scales_bf16, required_b_scale_bytes) ||
      scratch_overlaps(packed_output, required_output_bytes) ||
      scratch_overlaps(output_k512_scales_bf16,
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
    if (!g_paired_ldmatrix_resources_ready.load(
            std::memory_order_acquire)) {
      return static_cast<int>(cudaErrorNotReady);
    }
  } else {
    Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixResources resources{};
    const int status =
        query_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_resources_cuda(
            &resources);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }

  (void)cudaGetLastError();
  q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixThreads),
      kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixDynamicSharedBytes,
      stream>>>(
      packed_a, a_k512_scales_bf16, paired_b_codes,
      paired_b_scales_bf16,
      static_cast<unsigned int>(plan.logical_token_count),
      static_cast<unsigned int>(plan.m_tiles),
      static_cast<unsigned int>(plan.edge_groups),
      static_cast<unsigned int>(plan.input_k512_groups),
      output_clip_ratio, cta_scratch, packed_output,
      output_k512_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_resources_cuda(
    Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixResources* const resources)
    noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  cudaError_t status = configure_kernel();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_kernel,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixThreads),
      kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixDynamicSharedBytes;
  resources->configured_dynamic_shared_limit_bytes =
      attributes.maxDynamicSharedSizeBytes;
  resources->device_optin_shared_limit_bytes =
      properties.sharedMemPerBlockOptin;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread > 128 ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes != 132'096U ||
      resources->configured_dynamic_shared_limit_bytes < 132'096U ||
      resources->device_optin_shared_limit_bytes < 132'096U ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block < 512 ||
      resources->active_blocks_per_sm != 1) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  g_paired_ldmatrix_resources_ready.store(true,
                                          std::memory_order_release);
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const paired_b_codes,
    const std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* const paired_b_scales_bf16,
    const std::size_t paired_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const cta_scratch,
    const std::size_t cta_scratch_capacity_bytes,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, paired_b_codes,
      paired_b_code_capacity_bytes, paired_b_scales_bf16,
      paired_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size,
      output_clip_ratio, cta_scratch, cta_scratch_capacity_bytes,
      packed_output, packed_output_capacity_bytes,
      output_k512_scales_bf16, output_scale_capacity_elements, 16U,
      true, cuda_stream);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_test_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const paired_b_codes,
    const std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* const paired_b_scales_bf16,
    const std::size_t paired_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const cta_scratch,
    const std::size_t cta_scratch_capacity_bytes,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, paired_b_codes,
      paired_b_code_capacity_bytes, paired_b_scales_bf16,
      paired_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size,
      output_clip_ratio, cta_scratch, cta_scratch_capacity_bytes,
      packed_output, packed_output_capacity_bytes,
      output_k512_scales_bf16, output_scale_capacity_elements,
      maximum_launch_ctas, false, cuda_stream);
}

}  // namespace q3x::kernels
