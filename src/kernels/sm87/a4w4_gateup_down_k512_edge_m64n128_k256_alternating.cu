#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(kSm87A4W4GateUpDownEdgePackedK64Bytes);

struct alignas(16) Sm87A4W4GateUpDownEdgeStage final {
  std::uint8_t a[kSm87A4W4GateUpDownEdgeK64PerCopy]
                [kSm87A4W4GateUpDownEdgeTileM *
                 kSm87A4W4GateUpDownEdgePackedK64Bytes];
  std::uint8_t gate[kSm87A4W4GateUpDownEdgeK64PerCopy]
                   [kSm87A4W4GateUpDownEdgeComputeTileN *
                    kSm87A4W4GateUpDownEdgePackedK64Bytes];
  std::uint8_t up[kSm87A4W4GateUpDownEdgeK64PerCopy]
                 [kSm87A4W4GateUpDownEdgeComputeTileN *
                  kSm87A4W4GateUpDownEdgePackedK64Bytes];
};

struct alignas(16) Sm87A4W4GateUpDownEdgeScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpDownEdgeTileM];
  std::uint16_t gate[kSm87A4W4GateUpDownEdgeComputeTileN];
  std::uint16_t up[kSm87A4W4GateUpDownEdgeComputeTileN];
};

struct alignas(16) Sm87A4W4GateUpDownEdgePipeline final {
  Sm87A4W4GateUpDownEdgeStage
      stage[kSm87A4W4GateUpDownEdgeStages];
  Sm87A4W4GateUpDownEdgeScaleSlot
      scale[kSm87A4W4GateUpDownEdgeScaleSlots];
};

// Logical edge order is [row][quantizer-pair-iteration][quantizer-lane].
// The final coordinate is XOR-swizzled by row and pair.  A quantizer warp
// therefore reads 32 distinct banks for a fixed row/pair, while one MMA
// fragment's eight rows x four adjacent pairs also map to 32 distinct banks.
struct alignas(16) Sm87A4W4GateUpDownEdgePlane final {
  std::uint32_t pair[kSm87A4W4GateUpDownEdgeTileM][8U][32U];
};

struct alignas(16) Sm87A4W4GateUpDownEdgeShared final {
  Sm87A4W4GateUpDownEdgePipeline pipeline;
  Sm87A4W4GateUpDownEdgePlane edge;
};

struct alignas(16) Sm87A4W4GateUpDownEdgeFloat4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(Sm87A4W4GateUpDownEdgeStage) ==
              kSm87A4W4GateUpDownEdgeStageBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgeScaleSlot) ==
              kSm87A4W4GateUpDownEdgeScaleSlotBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgePipeline) ==
              kSm87A4W4GateUpDownEdgePipelineBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgePlane) ==
              kSm87A4W4GateUpDownEdgePlaneBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgeShared) ==
              kSm87A4W4GateUpDownEdgeDynamicSharedBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgeFloat4) == 16U);

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
    Sm87A4W4GateUpDownEdgePlane& edge, const unsigned int row,
    const unsigned int logical_even_column,
    const std::uint16_t even_bits, const std::uint16_t odd_bits) noexcept {
  const unsigned int logical_pair = logical_even_column / 2U;
  const unsigned int pair_iteration = logical_pair % 8U;
  const unsigned int quantizer_lane = logical_pair / 8U;
  const unsigned int physical_lane =
      edge_swizzled_lane(row, pair_iteration, quantizer_lane);
  edge.pair[row][pair_iteration][physical_lane] =
      static_cast<std::uint32_t>(even_bits) |
      (static_cast<std::uint32_t>(odd_bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t load_edge_pair(
    const Sm87A4W4GateUpDownEdgePlane& edge, const unsigned int row,
    const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  const unsigned int physical_lane =
      edge_swizzled_lane(row, pair_iteration, quantizer_lane);
  return edge.pair[row][pair_iteration][physical_lane];
}

// One K256 stage is four independently swizzled K64 code planes.  Every
// thread issues one A vector and two vectors for each of Gate and Up.  Gate
// and Up retain the authenticated canonical v1 weight presentation.
__device__ __forceinline__ void issue_k256_codes(
    Sm87A4W4GateUpDownEdgeStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int physical_k256_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeTileM * kPackedK64Bytes / 16U);
  constexpr unsigned int kBVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeComputeTileN * kPackedK64Bytes / 16U);
  constexpr unsigned int kAVectors =
      static_cast<unsigned int>(kSm87A4W4GateUpDownEdgeK64PerCopy) *
      kAVectorsPerPlane;
  constexpr unsigned int kBVectors =
      static_cast<unsigned int>(kSm87A4W4GateUpDownEdgeK64PerCopy) *
      kBVectorsPerPlane;
  static_assert(kAVectors == kSm87A4W4GateUpDownEdgeThreads);
  static_assert(kBVectors == 2U * kSm87A4W4GateUpDownEdgeThreads);

  const unsigned int a_vector = threadIdx.x;
  const unsigned int a_plane = a_vector / kAVectorsPerPlane;
  const unsigned int a_vector_in_plane =
      a_vector - a_plane * kAVectorsPerPlane;
  const unsigned int a_row = a_vector_in_plane / 2U;
  const unsigned int a_row_vector = a_vector_in_plane % 2U;
  const unsigned int a_physical_k64 =
      physical_k256_group *
          static_cast<unsigned int>(kSm87A4W4GateUpDownEdgeK64PerCopy) +
      a_plane;
  cp_async_16(
      stage.a[a_plane] + sm87_a4w4_swizzled_k64_byte_offset(
                               a_row, 16U * a_row_vector),
      packed_a + sm87_a4w4_consumer_packed_offset(
                     static_cast<std::size_t>(m_tile_start) + a_row,
                     a_physical_k64, 16U * a_row_vector,
                     physical_k64_group_count));

#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int b_vector =
        threadIdx.x +
        iteration * kSm87A4W4GateUpDownEdgeThreads;
    const unsigned int b_plane = b_vector / kBVectorsPerPlane;
    const unsigned int b_vector_in_plane =
        b_vector - b_plane * kBVectorsPerPlane;
    const unsigned int b_row = b_vector_in_plane / 2U;
    const unsigned int b_row_vector = b_vector_in_plane % 2U;
    const unsigned int b_physical_k64 =
        physical_k256_group *
            static_cast<unsigned int>(kSm87A4W4GateUpDownEdgeK64PerCopy) +
        b_plane;
    const std::size_t source_offset =
        sm87_a4w4_consumer_packed_offset(
            static_cast<std::size_t>(absolute_n_tile_start) + b_row,
            b_physical_k64, 16U * b_row_vector,
            physical_k64_group_count);
    const std::size_t destination_offset =
        sm87_a4w4_swizzled_k64_byte_offset(
            b_row, 16U * b_row_vector);
    cp_async_16(stage.gate[b_plane] + destination_offset,
                packed_gate_b + source_offset);
    cp_async_16(stage.up[b_plane] + destination_offset,
                packed_up_b + source_offset);
  }
}

__device__ __forceinline__ void issue_k512_scales(
    Sm87A4W4GateUpDownEdgeScaleSlot& slot,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  constexpr unsigned int kRowsPerVector = 8U;
  constexpr unsigned int kAVectors =
      kSm87A4W4GateUpDownEdgeTileM / kRowsPerVector;
  constexpr unsigned int kBVectors =
      kSm87A4W4GateUpDownEdgeComputeTileN / kRowsPerVector;
  static_assert(kAVectors == 8U);
  static_assert(kBVectors == 16U);

  if (threadIdx.x < kAVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    cp_async_16(
        slot.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x < kBVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    const std::size_t source_offset =
        sm87_a4w4_gateup_down_edge_scale_offset(
            static_cast<std::size_t>(absolute_n_tile_start) + first_row,
            k512_group, k512_group_count);
    cp_async_16(slot.gate + first_row,
                gate_b_k512_scales_bf16 + source_offset);
    cp_async_16(slot.up + first_row,
                up_b_k512_scales_bf16 + source_offset);
  }
}

__device__ __forceinline__ void issue_even_k256_and_scales(
    Sm87A4W4GateUpDownEdgeStage& stage,
    Sm87A4W4GateUpDownEdgeScaleSlot& scale,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  issue_k256_codes(stage, packed_a, packed_gate_b, packed_up_b,
                   m_tile_start, absolute_n_tile_start,
                   2U * k512_group, physical_k64_group_count);
  issue_k512_scales(scale, a_k512_scales_bf16,
                    gate_b_k512_scales_bf16,
                    up_b_k512_scales_bf16, m_tile_start,
                    absolute_n_tile_start, k512_group,
                    k512_group_count);
  cp_async_commit();
}

__device__ __forceinline__ void issue_odd_k256(
    Sm87A4W4GateUpDownEdgeStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int physical_k64_group_count) noexcept {
  issue_k256_codes(stage, packed_a, packed_gate_b, packed_up_b,
                   m_tile_start, absolute_n_tile_start,
                   2U * k512_group + 1U,
                   physical_k64_group_count);
  cp_async_commit();
}

// Each warp owns one M16N32 tile and accumulates both projections.  This
// preserves the production cell's eight IMMA operations per output fragment
// and K512 S32-before-scale numerical boundary while removing the Gate/Up
// cross-warp shared exchange.
__device__ __forceinline__ void accumulate_k256_stage(
    const Sm87A4W4GateUpDownEdgeStage& stage,
    Sm87A4W4Accumulator (&gate_partials)[1U][4U],
    Sm87A4W4Accumulator (&up_partials)[1U][4U]) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp / kSm87A4W4GateUpDownEdgeWarpColumns;
  const unsigned int warp_n =
      warp % kSm87A4W4GateUpDownEdgeWarpColumns;
  const unsigned int local_m_start =
      warp_m * kSm87A4W4GateUpDownEdgeWarpTileM;
  const unsigned int local_n_start =
      warp_n * kSm87A4W4GateUpDownEdgeWarpTileN;

#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4GateUpDownEdgeK64PerCopy; ++plane) {
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 1U; ++m_panel) {
      const Sm87A4W4AFragment a_fragment =
          sm87_a4w4_load_a_fragment_swizzled_shared(
              stage.a[plane] +
                  (local_m_start + m_panel * 16U) * kPackedK64Bytes,
              lane);
#pragma unroll
      for (unsigned int n_fragment = 0U; n_fragment < 4U;
           ++n_fragment) {
        const unsigned int fragment_n =
            local_n_start + n_fragment * 8U;
        const Sm87A4W4BFragment gate_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.gate[plane] + fragment_n * kPackedK64Bytes,
                lane);
        const Sm87A4W4BFragment up_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.up[plane] + fragment_n * kPackedK64Bytes,
                lane);
        sm87_a4w4_mma_m16n8k64(
            gate_partials[m_panel][n_fragment], a_fragment,
            gate_fragment);
        sm87_a4w4_mma_m16n8k64(
            up_partials[m_panel][n_fragment], a_fragment, up_fragment);
      }
    }
  }
}

__device__ __forceinline__ void apply_k512_group(
    Sm87A4W4GateUpDownEdgeFloat4 (&gate_accumulators)[1U][4U],
    Sm87A4W4GateUpDownEdgeFloat4 (&up_accumulators)[1U][4U],
    const Sm87A4W4Accumulator (&gate_partials)[1U][4U],
    const Sm87A4W4Accumulator (&up_partials)[1U][4U],
    const Sm87A4W4GateUpDownEdgeScaleSlot& scale) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp / kSm87A4W4GateUpDownEdgeWarpColumns;
  const unsigned int warp_n =
      warp % kSm87A4W4GateUpDownEdgeWarpColumns;
  const unsigned int local_m_start =
      warp_m * kSm87A4W4GateUpDownEdgeWarpTileM;
  const unsigned int local_n_start =
      warp_n * kSm87A4W4GateUpDownEdgeWarpTileN;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);

#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < 1U; ++m_panel) {
    const unsigned int fragment_m = local_m_start + m_panel * 16U;
    const float a_scale0 =
        decode_bf16(scale.a[fragment_m + coordinate0.m]);
    const float a_scale1 =
        decode_bf16(scale.a[fragment_m + coordinate2.m]);
#pragma unroll
    for (unsigned int n_fragment = 0U; n_fragment < 4U;
         ++n_fragment) {
      const unsigned int fragment_n =
          local_n_start + n_fragment * 8U;
      const unsigned int local_n0 = fragment_n + coordinate0.n;
      const unsigned int local_n1 = fragment_n + coordinate1.n;
      const float gate_scale0 = decode_bf16(scale.gate[local_n0]);
      const float gate_scale1 = decode_bf16(scale.gate[local_n1]);
      const float up_scale0 = decode_bf16(scale.up[local_n0]);
      const float up_scale1 = decode_bf16(scale.up[local_n1]);
      const float gate00 = __fmul_rn(a_scale0, gate_scale0);
      const float gate01 = __fmul_rn(a_scale0, gate_scale1);
      const float gate10 = __fmul_rn(a_scale1, gate_scale0);
      const float gate11 = __fmul_rn(a_scale1, gate_scale1);
      const float up00 = __fmul_rn(a_scale0, up_scale0);
      const float up01 = __fmul_rn(a_scale0, up_scale1);
      const float up10 = __fmul_rn(a_scale1, up_scale0);
      const float up11 = __fmul_rn(a_scale1, up_scale1);
      const Sm87A4W4Accumulator& gate =
          gate_partials[m_panel][n_fragment];
      const Sm87A4W4Accumulator& up =
          up_partials[m_panel][n_fragment];
      Sm87A4W4GateUpDownEdgeFloat4& gate_output =
          gate_accumulators[m_panel][n_fragment];
      Sm87A4W4GateUpDownEdgeFloat4& up_output =
          up_accumulators[m_panel][n_fragment];
      gate_output.x0 = __fmaf_rn(static_cast<float>(gate.x0), gate00,
                                 gate_output.x0);
      gate_output.x1 = __fmaf_rn(static_cast<float>(gate.x1), gate01,
                                 gate_output.x1);
      gate_output.x2 = __fmaf_rn(static_cast<float>(gate.x2), gate10,
                                 gate_output.x2);
      gate_output.x3 = __fmaf_rn(static_cast<float>(gate.x3), gate11,
                                 gate_output.x3);
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
}

__device__ __forceinline__ void store_bf16_product_cell(
    Sm87A4W4GateUpDownEdgePlane& edge,
    const Sm87A4W4GateUpDownEdgeFloat4 (&gate_accumulators)[1U][4U],
    const Sm87A4W4GateUpDownEdgeFloat4 (&up_accumulators)[1U][4U],
    const unsigned int global_m_start,
    const unsigned int logical_token_count,
    const unsigned int cell_in_edge) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp / kSm87A4W4GateUpDownEdgeWarpColumns;
  const unsigned int warp_n =
      warp % kSm87A4W4GateUpDownEdgeWarpColumns;
  const unsigned int local_m_start =
      warp_m * kSm87A4W4GateUpDownEdgeWarpTileM;
  const unsigned int local_n_start =
      cell_in_edge * kSm87A4W4GateUpDownEdgeComputeTileN +
      warp_n * kSm87A4W4GateUpDownEdgeWarpTileN;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);

#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < 1U; ++m_panel) {
    const unsigned int panel_m = local_m_start + m_panel * 16U;
#pragma unroll
    for (unsigned int n_fragment = 0U; n_fragment < 4U;
         ++n_fragment) {
      const unsigned int fragment_n =
          local_n_start + n_fragment * 8U;
      const Sm87A4W4GateUpDownEdgeFloat4& gate =
          gate_accumulators[m_panel][n_fragment];
      const Sm87A4W4GateUpDownEdgeFloat4& up =
          up_accumulators[m_panel][n_fragment];
      const unsigned int row0 = panel_m + coordinate0.m;
      const unsigned int row1 = panel_m + coordinate2.m;
      const bool valid0 = global_m_start + row0 < logical_token_count;
      const bool valid1 = global_m_start + row1 < logical_token_count;
      const std::uint16_t bits00 =
          valid0 ? encode_bf16(silu_product(gate.x0, up.x0)) : 0U;
      const std::uint16_t bits01 =
          valid0 ? encode_bf16(silu_product(gate.x1, up.x1)) : 0U;
      const std::uint16_t bits10 =
          valid1 ? encode_bf16(silu_product(gate.x2, up.x2)) : 0U;
      const std::uint16_t bits11 =
          valid1 ? encode_bf16(silu_product(gate.x3, up.x3)) : 0U;
      store_edge_pair(edge, row0, fragment_n + coordinate0.n,
                      bits00, bits01);
      store_edge_pair(edge, row1, fragment_n + coordinate0.n,
                      bits10, bits11);
    }
  }
}

__device__ __forceinline__ void compute_n128_cell(
    Sm87A4W4GateUpDownEdgeShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int cell_in_edge,
    const unsigned int input_k512_group_count) noexcept {
  Sm87A4W4GateUpDownEdgeFloat4 gate_accumulators[1U][4U]{};
  Sm87A4W4GateUpDownEdgeFloat4 up_accumulators[1U][4U]{};

  // Publish the first even and odd K256 stages as two ordered async groups.
  // wait_group<1> makes the complete even stage visible while odd remains
  // eligible to overlap its compute.
  issue_even_k256_and_scales(
      shared.pipeline.stage[0U], shared.pipeline.scale[0U], packed_a,
      a_k512_scales_bf16, packed_gate_b, gate_b_k512_scales_bf16,
      packed_up_b, up_b_k512_scales_bf16, m_tile_start,
      absolute_n_tile_start, 0U, input_k512_group_count,
      input_k512_group_count * 8U);
  issue_odd_k256(
      shared.pipeline.stage[1U], packed_a, packed_gate_b, packed_up_b,
      m_tile_start, absolute_n_tile_start, 0U,
      input_k512_group_count * 8U);
  cp_async_wait<1U>();
  __syncthreads();

  for (unsigned int group = 0U; group < input_k512_group_count;
       ++group) {
    // Gate and Up retain the incumbent paired M16N32 register ownership.
    // Both K256 halves accumulate into the same S32 fragments; scale is
    // applied exactly once after all eight physical K64 planes.
    Sm87A4W4Accumulator gate_partials[1U][4U]{};
    Sm87A4W4Accumulator up_partials[1U][4U]{};
    accumulate_k256_stage(shared.pipeline.stage[0U], gate_partials,
                          up_partials);

    // Odd becomes globally visible and every warp releases even at one
    // barrier.  Stage 0 can then be recycled while odd compute proceeds.
    cp_async_wait<0U>();
    __syncthreads();
    const unsigned int next_group = group + 1U;
    if (next_group < input_k512_group_count) {
      issue_even_k256_and_scales(
          shared.pipeline.stage[0U],
          shared.pipeline.scale[
              next_group % kSm87A4W4GateUpDownEdgeScaleSlots],
          packed_a, a_k512_scales_bf16, packed_gate_b,
          gate_b_k512_scales_bf16, packed_up_b,
          up_b_k512_scales_bf16, m_tile_start, absolute_n_tile_start,
          next_group, input_k512_group_count,
          input_k512_group_count * 8U);
    }

    accumulate_k256_stage(shared.pipeline.stage[1U], gate_partials,
                          up_partials);
    apply_k512_group(
        gate_accumulators, up_accumulators, gate_partials, up_partials,
        shared.pipeline.scale[
            group % kSm87A4W4GateUpDownEdgeScaleSlots]);

    if (next_group < input_k512_group_count) {
      // The next even stage and scale slot become visible while all warps
      // release odd.  The next odd publication then overlaps even compute.
      cp_async_wait<0U>();
      __syncthreads();
      issue_odd_k256(
          shared.pipeline.stage[1U], packed_a, packed_gate_b,
          packed_up_b, m_tile_start, absolute_n_tile_start,
          next_group, input_k512_group_count * 8U);
    }
  }

  // Edge writes do not overlap either pipeline stage.  The product barrier
  // therefore also serves as the final odd-stage reader-release barrier:
  // startup(1) + even-ready(10) + inter-group odd-finish(9) + product(1)
  // = 21 barriers per N128 cell at model K5120.
  store_bf16_product_cell(
      shared.edge, gate_accumulators, up_accumulators, m_tile_start,
      logical_token_count, cell_in_edge);
  __syncthreads();
}

__device__ __forceinline__ void quantize_edge_cell(
    const Sm87A4W4GateUpDownEdgePlane& edge,
    const unsigned int m_tile_start,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int output_physical_k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;

#pragma unroll 1
  for (unsigned int row_iteration = 0U; row_iteration < 4U;
       ++row_iteration) {
    const unsigned int local_row =
        warp + row_iteration * kSm87A4W4GateUpDownEdgeWarps;
    const unsigned int global_row = m_tile_start + local_row;
    float maximum = 0.0F;
#pragma unroll 1
    for (unsigned int pair = 0U; pair < 8U; ++pair) {
      const std::uint32_t word =
          load_edge_pair(edge, local_row, pair, lane);
      const float even =
          decode_bf16(static_cast<std::uint16_t>(word));
      const float odd =
          decode_bf16(static_cast<std::uint16_t>(word >> 16U));
      maximum = fmaxf(maximum, fabsf(even));
      maximum = fmaxf(maximum, fabsf(odd));
    }

#pragma unroll
    for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
      maximum = fmaxf(
          maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0U);
    const float clipped_maximum = maximum * output_clip_ratio;
    std::uint16_t scale_bits =
        encode_bf16(maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
    float stored_scale = decode_bf16(scale_bits);
    if (maximum != 0.0F && stored_scale == 0.0F) {
      scale_bits = 1U;
      stored_scale = decode_bf16(scale_bits);
    }

    const unsigned int physical_group = edge_group * 8U + lane / 4U;
    const unsigned int first_byte = 8U * (lane % 4U);
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
          output_physical_k64_group_count)] =
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

__device__ __forceinline__ void compute_edge_cell(
    Sm87A4W4GateUpDownEdgeShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int m_tile_start =
      m_tile * kSm87A4W4GateUpDownEdgeTileM;
  const unsigned int edge_n_start =
      edge_group * kSm87A4W4GateUpDownEdgeScaleK;
  for (unsigned int cell = 0U;
       cell < kSm87A4W4GateUpDownEdgeCellsPerScale; ++cell) {
    compute_n128_cell(
        shared, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, logical_token_count, m_tile_start,
        edge_n_start + cell * kSm87A4W4GateUpDownEdgeComputeTileN,
        cell, input_k512_group_count);
  }
  quantize_edge_cell(
      shared.edge, m_tile_start, edge_group, edge_group_count,
      edge_group_count * 8U, output_clip_ratio, packed_output,
      output_k512_scales_bf16);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpDownEdgeThreads,
                      kSm87A4W4GateUpDownEdgeCtasPerSm)
void q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile_count,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared = *reinterpret_cast<Sm87A4W4GateUpDownEdgeShared*>(
      dynamic_shared);

  const unsigned int base_waves = m_tile_count / gridDim.x;
  const unsigned int base_m_tiles = base_waves * gridDim.x;
  const unsigned int residual_m_tiles = m_tile_count - base_m_tiles;
  const unsigned int base_iterations = base_waves * edge_group_count;
  const unsigned int residual_edge_cells =
      residual_m_tiles * edge_group_count;

  // A single compute call site keeps scheduling state out of the already
  // register-dense MMA core.  The first region is the fixed-M base wave.  The
  // second region distributes a partial final M wave as complete N-major
  // (M64,K512) cells across all CTAs.
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
      if (ordinal >= residual_edge_cells) {
        break;
      }
      edge_group = ordinal / residual_m_tiles;
      m_tile = base_m_tiles +
               ordinal - edge_group * residual_m_tiles;
    }
    compute_edge_cell(
        shared, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, logical_token_count, m_tile,
        edge_group, edge_group_count, input_k512_group_count,
        output_clip_ratio, packed_output, output_k512_scales_bf16);
  }
}

namespace {

[[nodiscard]] int validate_alternating_target(
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
          kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_alternating_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes));
}

}  // namespace

int query_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_resources_cuda(
    Sm87A4W4GateUpDownEdgeM64N128K256AlternatingResources* const resources)
    noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources =
      Sm87A4W4GateUpDownEdgeM64N128K256AlternatingResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_alternating_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  cudaError_t status = configure_alternating_dynamic_shared();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_kernel,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM64N128K256AlternatingThreads),
      kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes;
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
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM64N128K256AlternatingMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM64N128K256AlternatingThreads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM64N128K256AlternatingCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
