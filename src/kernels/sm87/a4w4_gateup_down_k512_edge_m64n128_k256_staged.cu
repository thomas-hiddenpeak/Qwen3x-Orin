#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m64n128_k256_staged.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kPackedK64Bytes = 32U;

struct alignas(16) K256StagedStage final {
  std::uint8_t a
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedK64PerCopy]
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM *
       kPackedK64Bytes];
  std::uint8_t gate
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedK64PerCopy]
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN *
       kPackedK64Bytes];
  std::uint8_t up
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedK64PerCopy]
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN *
       kPackedK64Bytes];
};

struct alignas(16) K256StagedScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM];
  std::uint16_t gate[kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN];
  std::uint16_t up[kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN];
};

struct alignas(16) K256StagedPipeline final {
  K256StagedStage
      stage[kSm87A4W4GateUpDownEdgeM64N128K256StagedStages];
  K256StagedScaleSlot
      scale[kSm87A4W4GateUpDownEdgeM64N128K256StagedScaleSlots];
};

// Component-major, lane-minor exchange.  For any fragment component one Gate
// warp publishes 32 consecutive words, and the matching Up warp reads the
// same conflict-free bank permutation after the seam barrier.
struct alignas(16) K256StagedGateExchange final {
  float value
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedProjectionWarps]
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp]
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp]
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedComponents]
      [32U];
};

union alignas(16) K256StagedWork final {
  K256StagedPipeline pipeline;
  K256StagedGateExchange gate_exchange;
};

struct alignas(16) K256StagedEdgePlane final {
  std::uint32_t pair
      [kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM][8U][32U];
};

struct alignas(16) K256StagedShared final {
  K256StagedWork work;
  K256StagedEdgePlane edge;
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

struct BPair final {
  Sm87A4W4BFragment n0;
  Sm87A4W4BFragment n1;
};

struct K256StagedKernelParams final {
  const std::uint8_t* packed_a;
  const std::uint16_t* a_k512_scales_bf16;
  const std::uint8_t* packed_gate_b;
  const std::uint16_t* gate_b_k512_scales_bf16;
  const std::uint8_t* packed_up_b;
  const std::uint16_t* up_b_k512_scales_bf16;
  std::uint8_t* packed_output;
  std::uint16_t* output_k512_scales_bf16;
  unsigned int logical_token_count;
  unsigned int m_tile_count;
  unsigned int edge_group_count;
  unsigned int k512_group_count;
  float output_clip_ratio;
};

static_assert(sizeof(K256StagedStage) ==
              kSm87A4W4GateUpDownEdgeM64N128K256StagedStageBytes);
static_assert(sizeof(K256StagedScaleSlot) ==
              kSm87A4W4GateUpDownEdgeM64N128K256StagedScaleSlotBytes);
static_assert(sizeof(K256StagedPipeline) ==
              kSm87A4W4GateUpDownEdgeM64N128K256StagedPipelineBytes);
static_assert(sizeof(K256StagedGateExchange) ==
              kSm87A4W4GateUpDownEdgeM64N128K256StagedGateExchangeBytes);
static_assert(sizeof(K256StagedWork) ==
              kSm87A4W4GateUpDownEdgeM64N128K256StagedPipelineBytes);
static_assert(sizeof(K256StagedEdgePlane) ==
              kSm87A4W4GateUpDownEdgeM64N128K256StagedEdgePlaneBytes);
static_assert(sizeof(K256StagedShared) ==
              kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes);
static_assert(sizeof(Float4) == 16U);
static_assert(sizeof(BPair) == 16U);

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

__device__ __forceinline__ void cp_async_16_cg(
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
    K256StagedEdgePlane& edge, const unsigned int row,
    const unsigned int logical_even_column,
    const std::uint16_t even_bits,
    const std::uint16_t odd_bits) noexcept {
  const unsigned int logical_pair = logical_even_column / 2U;
  const unsigned int pair_iteration = logical_pair & 7U;
  const unsigned int quantizer_lane = logical_pair / 8U;
  const unsigned int physical_lane = edge_swizzled_lane(
      row, pair_iteration, quantizer_lane);
  edge.pair[row][pair_iteration][physical_lane] =
      static_cast<std::uint32_t>(even_bits) |
      (static_cast<std::uint32_t>(odd_bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t load_edge_pair(
    const K256StagedEdgePlane& edge, const unsigned int row,
    const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return edge.pair[row][pair_iteration][edge_swizzled_lane(
      row, pair_iteration, quantizer_lane)];
}

// Five aligned vectors per thread publish one complete A+GateB+UpB K256
// stage.  No consumer sees a partial stage.
__device__ __forceinline__ void issue_k256_codes(
    K256StagedStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m64_start,
    const unsigned int absolute_n128_start,
    const unsigned int physical_k256_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM *
      kPackedK64Bytes / 16U;
  constexpr unsigned int kBVectorsPerPlane =
      kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN *
      kPackedK64Bytes / 16U;
  constexpr unsigned int kAVectors =
      kSm87A4W4GateUpDownEdgeM64N128K256StagedK64PerCopy *
      kAVectorsPerPlane;
  constexpr unsigned int kBVectors =
      kSm87A4W4GateUpDownEdgeM64N128K256StagedK64PerCopy *
      kBVectorsPerPlane;
  static_assert(kAVectors ==
                kSm87A4W4GateUpDownEdgeM64N128K256StagedThreads);
  static_assert(kBVectors ==
                2U * kSm87A4W4GateUpDownEdgeM64N128K256StagedThreads);

  const unsigned int a_vector = threadIdx.x;
  const unsigned int a_plane = a_vector / kAVectorsPerPlane;
  const unsigned int a_in_plane = a_vector - a_plane * kAVectorsPerPlane;
  const unsigned int a_row = a_in_plane / 2U;
  const unsigned int a_byte = 16U * (a_in_plane & 1U);
  const unsigned int a_k64 =
      physical_k256_group *
          kSm87A4W4GateUpDownEdgeM64N128K256StagedK64PerCopy +
      a_plane;
  cp_async_16_cg(
      stage.a[a_plane] +
          sm87_a4w4_swizzled_k64_byte_offset(a_row, a_byte),
      packed_a + sm87_a4w4_gateup_down_edge_packed_offset(
                     static_cast<std::size_t>(m64_start) + a_row,
                     a_k64, a_byte, physical_k64_group_count));

#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int b_vector =
        threadIdx.x +
        iteration *
            kSm87A4W4GateUpDownEdgeM64N128K256StagedThreads;
    const unsigned int b_plane = b_vector / kBVectorsPerPlane;
    const unsigned int b_in_plane =
        b_vector - b_plane * kBVectorsPerPlane;
    const unsigned int b_row = b_in_plane / 2U;
    const unsigned int b_byte = 16U * (b_in_plane & 1U);
    const unsigned int b_k64 =
        physical_k256_group *
            kSm87A4W4GateUpDownEdgeM64N128K256StagedK64PerCopy +
        b_plane;
    const std::size_t source_offset =
        sm87_a4w4_gateup_down_edge_packed_offset(
            static_cast<std::size_t>(absolute_n128_start) + b_row,
            b_k64, b_byte, physical_k64_group_count);
    const std::size_t destination_offset =
        sm87_a4w4_swizzled_k64_byte_offset(b_row, b_byte);
    cp_async_16_cg(stage.gate[b_plane] + destination_offset,
                   packed_gate_b + source_offset);
    cp_async_16_cg(stage.up[b_plane] + destination_offset,
                   packed_up_b + source_offset);
  }
}

__device__ __forceinline__ void issue_k512_scales(
    K256StagedScaleSlot& slot,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m64_start,
    const unsigned int absolute_n128_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  if (threadIdx.x < 8U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16_cg(
        slot.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(m64_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    const std::size_t source_offset =
        sm87_a4w4_gateup_down_edge_scale_offset(
            static_cast<std::size_t>(absolute_n128_start) + first_row,
            k512_group, k512_group_count);
    cp_async_16_cg(slot.gate + first_row,
                   gate_b_k512_scales_bf16 + source_offset);
    cp_async_16_cg(slot.up + first_row,
                   up_b_k512_scales_bf16 + source_offset);
  }
}

__device__ __forceinline__ void issue_stage_with_scales(
    K256StagedStage& stage,
    K256StagedScaleSlot& scale,
    const K256StagedKernelParams& params,
    const unsigned int m64_start,
    const unsigned int absolute_n128_start,
    const unsigned int k512_group,
    const unsigned int physical_k256_group) noexcept {
  issue_k256_codes(
      stage, params.packed_a, params.packed_gate_b, params.packed_up_b,
      m64_start, absolute_n128_start, physical_k256_group,
      params.k512_group_count * 8U);
  issue_k512_scales(
      scale, params.a_k512_scales_bf16,
      params.gate_b_k512_scales_bf16,
      params.up_b_k512_scales_bf16, m64_start, absolute_n128_start,
      k512_group, params.k512_group_count);
  cp_async_commit();
}

__device__ __forceinline__ void issue_stage_codes_only(
    K256StagedStage& stage,
    const K256StagedKernelParams& params,
    const unsigned int m64_start,
    const unsigned int absolute_n128_start,
    const unsigned int physical_k256_group) noexcept {
  issue_k256_codes(
      stage, params.packed_a, params.packed_gate_b, params.packed_up_b,
      m64_start, absolute_n128_start, physical_k256_group,
      params.k512_group_count * 8U);
  cp_async_commit();
}

[[nodiscard]] __device__ __forceinline__ BPair load_b_pair(
    const std::uint8_t* const projection_b_plane,
    const unsigned int crew,
    const unsigned int lane) noexcept {
  const unsigned int n_start = 16U * crew;
  return {sm87_a4w4_load_b_fragment_swizzled_shared(
              projection_b_plane + n_start * kPackedK64Bytes, lane),
          sm87_a4w4_load_b_fragment_swizzled_shared(
              projection_b_plane + (n_start + 8U) * kPackedK64Bytes,
              lane)};
}

__device__ __forceinline__ void accumulate_plane_b_outer(
    const K256StagedStage& stage,
    const unsigned int plane,
    const BPair& b,
    Sm87A4W4Accumulator
        (&partials)
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp;
       ++m16) {
    const Sm87A4W4AFragment a =
        sm87_a4w4_load_a_fragment_swizzled_shared(
            stage.a[plane] + 16U * m16 * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(partials[m16][0U], a, b.n0);
    sm87_a4w4_mma_m16n8k64(partials[m16][1U], a, b.n1);
  }
}

// Two B pairs remain in registers while each is reused across all four M16
// panels.  The explicit ping-pong order prevents an M-inner reload of B.
__device__ __forceinline__ void accumulate_k256_stage(
    const K256StagedStage& stage,
    const unsigned int projection,
    Sm87A4W4Accumulator
        (&partials)
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 7U;
  const std::uint8_t* const plane0 =
      projection == 0U ? stage.gate[0U] : stage.up[0U];
  const std::uint8_t* const plane1 =
      projection == 0U ? stage.gate[1U] : stage.up[1U];
  const std::uint8_t* const plane2 =
      projection == 0U ? stage.gate[2U] : stage.up[2U];
  const std::uint8_t* const plane3 =
      projection == 0U ? stage.gate[3U] : stage.up[3U];
  BPair b0 = load_b_pair(plane0, crew, lane);
  BPair b1 = load_b_pair(plane1, crew, lane);
  accumulate_plane_b_outer(stage, 0U, b0, partials);
  b0 = load_b_pair(plane2, crew, lane);
  accumulate_plane_b_outer(stage, 1U, b1, partials);
  b1 = load_b_pair(plane3, crew, lane);
  accumulate_plane_b_outer(stage, 2U, b0, partials);
  accumulate_plane_b_outer(stage, 3U, b1, partials);
}

__device__ __forceinline__ void apply_k512_group(
    Float4
        (&accumulators)
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp],
    const Sm87A4W4Accumulator
        (&partials)
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp],
    const K256StagedScaleSlot& scale,
    const unsigned int projection) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 7U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const std::uint16_t* const b_scale =
      projection == 0U ? scale.gate : scale.up;
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp;
       ++m16) {
    const unsigned int base_m = 16U * m16;
    const float a_scale0 =
        decode_bf16(scale.a[base_m + coordinate0.m]);
    const float a_scale1 =
        decode_bf16(scale.a[base_m + coordinate2.m]);
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp;
         ++n8) {
      const unsigned int base_n = 16U * crew + 8U * n8;
      const float b_scale0 =
          decode_bf16(b_scale[base_n + coordinate0.n]);
      const float b_scale1 =
          decode_bf16(b_scale[base_n + coordinate1.n]);
      const Sm87A4W4Accumulator& partial = partials[m16][n8];
      Float4& output = accumulators[m16][n8];
      output.x0 = __fmaf_rn(
          static_cast<float>(partial.x0),
          __fmul_rn(a_scale0, b_scale0), output.x0);
      output.x1 = __fmaf_rn(
          static_cast<float>(partial.x1),
          __fmul_rn(a_scale0, b_scale1), output.x1);
      output.x2 = __fmaf_rn(
          static_cast<float>(partial.x2),
          __fmul_rn(a_scale1, b_scale0), output.x2);
      output.x3 = __fmaf_rn(
          static_cast<float>(partial.x3),
          __fmul_rn(a_scale1, b_scale1), output.x3);
    }
  }
}

__device__ __forceinline__ void publish_gate(
    K256StagedGateExchange& exchange,
    const Float4
        (&accumulators)
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 7U;
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp;
       ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp;
         ++n8) {
      const Float4& value = accumulators[m16][n8];
      exchange.value[crew][m16][n8][0U][lane] = value.x0;
      exchange.value[crew][m16][n8][1U][lane] = value.x1;
      exchange.value[crew][m16][n8][2U][lane] = value.x2;
      exchange.value[crew][m16][n8][3U][lane] = value.x3;
    }
  }
}

__device__ __forceinline__ void consume_up_and_store_edge(
    const K256StagedGateExchange& exchange,
    K256StagedEdgePlane& edge,
    const Float4
        (&accumulators)
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp],
    const unsigned int global_m64_start,
    const unsigned int logical_token_count,
    const unsigned int cell_in_edge) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 7U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp;
       ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp;
         ++n8) {
      const unsigned int row0 = 16U * m16 + coordinate0.m;
      const unsigned int row1 = 16U * m16 + coordinate2.m;
      const unsigned int base_n = 16U * crew + 8U * n8;
      const unsigned int column = base_n + coordinate0.n;
      const bool valid0 =
          global_m64_start + row0 < logical_token_count;
      const bool valid1 =
          global_m64_start + row1 < logical_token_count;
      const Float4& up = accumulators[m16][n8];
      const float gate0 = exchange.value[crew][m16][n8][0U][lane];
      const float gate1 = exchange.value[crew][m16][n8][1U][lane];
      const float gate2 = exchange.value[crew][m16][n8][2U][lane];
      const float gate3 = exchange.value[crew][m16][n8][3U][lane];
      const std::uint16_t bits00 =
          valid0 ? encode_bf16(silu_product(gate0, up.x0)) : 0U;
      const std::uint16_t bits01 =
          valid0 ? encode_bf16(silu_product(gate1, up.x1)) : 0U;
      const std::uint16_t bits10 =
          valid1 ? encode_bf16(silu_product(gate2, up.x2)) : 0U;
      const std::uint16_t bits11 =
          valid1 ? encode_bf16(silu_product(gate3, up.x3)) : 0U;
      const unsigned int edge_column =
          cell_in_edge *
              kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN +
          column;
      store_edge_pair(edge, row0, edge_column, bits00, bits01);
      store_edge_pair(edge, row1, edge_column, bits10, bits11);
    }
  }
}

__device__ __forceinline__ void compute_n128_cell(
    K256StagedShared& shared,
    const K256StagedKernelParams& params,
    const unsigned int m64_start,
    const unsigned int absolute_n128_start,
    const unsigned int cell_in_edge) noexcept {
  const unsigned int projection = threadIdx.x >> 8U;
  Float4
      accumulators
          [kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp]
          [kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp]{};

  issue_stage_with_scales(
      shared.work.pipeline.stage[0U], shared.work.pipeline.scale[0U],
      params, m64_start, absolute_n128_start, 0U, 0U);
  issue_stage_codes_only(
      shared.work.pipeline.stage[1U], params, m64_start,
      absolute_n128_start, 1U);
  cp_async_wait<1U>();
  __syncthreads();

  for (unsigned int group = 0U; group < params.k512_group_count;
       ++group) {
    Sm87A4W4Accumulator
        partials
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp]{};
    accumulate_k256_stage(
        shared.work.pipeline.stage[0U], projection, partials);

    // Whole-stage publication: odd is globally complete and every consumer
    // has released even before stage 0 is recycled for the next group.
    cp_async_wait<0U>();
    __syncthreads();
    const unsigned int next_group = group + 1U;
    if (next_group < params.k512_group_count) {
      issue_stage_with_scales(
          shared.work.pipeline.stage[0U],
          shared.work.pipeline.scale[
              next_group %
              kSm87A4W4GateUpDownEdgeM64N128K256StagedScaleSlots],
          params, m64_start, absolute_n128_start, next_group,
          2U * next_group);
    }

    accumulate_k256_stage(
        shared.work.pipeline.stage[1U], projection, partials);
    apply_k512_group(
        accumulators, partials,
        shared.work.pipeline.scale[
            group %
            kSm87A4W4GateUpDownEdgeM64N128K256StagedScaleSlots],
        projection);

    if (next_group < params.k512_group_count) {
      cp_async_wait<0U>();
      __syncthreads();
      issue_stage_codes_only(
          shared.work.pipeline.stage[1U], params, m64_start,
          absolute_n128_start, 2U * next_group + 1U);
    } else {
      // Final pipeline release; the next stores may legally reuse the same
      // allocation as the Gate exchange.
      __syncthreads();
    }
  }

  if (projection == 0U) {
    publish_gate(shared.work.gate_exchange, accumulators);
  }
  __syncthreads();
  if (projection != 0U) {
    consume_up_and_store_edge(
        shared.work.gate_exchange, shared.edge, accumulators,
        m64_start, params.logical_token_count, cell_in_edge);
  }
  __syncthreads();
}

__device__ __forceinline__ void quantize_edge_cell(
    const K256StagedEdgePlane& edge,
    const unsigned int m64_start,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
#pragma unroll 1
  for (unsigned int row_iteration = 0U; row_iteration < 4U;
       ++row_iteration) {
    const unsigned int local_row =
        warp + row_iteration *
                   kSm87A4W4GateUpDownEdgeM64N128K256StagedWarps;
    const unsigned int global_row = m64_start + local_row;
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
          fabsf(decode_bf16(
              static_cast<std::uint16_t>(word >> 16U))));
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
      const float odd_value = decode_bf16(
          static_cast<std::uint16_t>(word >> 16U));
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

__device__ __forceinline__ void compute_edge_cell(
    K256StagedShared& shared,
    const K256StagedKernelParams& params,
    const unsigned int m_tile,
    const unsigned int edge_group) noexcept {
  const unsigned int m64_start =
      m_tile * kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM;
  const unsigned int edge_n_start =
      edge_group * kSm87A4W4GateUpDownEdgeM64N128K256StagedEdgeK;
  for (unsigned int cell = 0U;
       cell < kSm87A4W4GateUpDownEdgeM64N128K256StagedCellsPerEdge;
       ++cell) {
    compute_n128_cell(
        shared, params, m64_start,
        edge_n_start +
            cell * kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN,
        cell);
  }
  quantize_edge_cell(
      shared.edge, m64_start, edge_group, params.edge_group_count,
      params.output_clip_ratio, params.packed_output,
      params.output_k512_scales_bf16);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(
        kSm87A4W4GateUpDownEdgeM64N128K256StagedThreads,
        kSm87A4W4GateUpDownEdgeM64N128K256StagedCtasPerSm)
void q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_staged_kernel(
    const K256StagedKernelParams params) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared = *reinterpret_cast<K256StagedShared*>(dynamic_shared);

  const unsigned int base_waves = params.m_tile_count / gridDim.x;
  const unsigned int base_m_tiles = base_waves * gridDim.x;
  const unsigned int residual_m_tiles =
      params.m_tile_count - base_m_tiles;
  const unsigned int base_iterations =
      base_waves * params.edge_group_count;
  const unsigned int residual_edge_cells =
      residual_m_tiles * params.edge_group_count;
  for (unsigned int iteration = 0U;; ++iteration) {
    unsigned int m_tile = 0U;
    unsigned int edge_group = 0U;
    if (iteration < base_iterations) {
      const unsigned int wave = iteration / params.edge_group_count;
      edge_group = iteration - wave * params.edge_group_count;
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
    compute_edge_cell(shared, params, m_tile, edge_group);
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
      properties.multiProcessorCount != 16 ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_staged_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes));
}

}  // namespace

int query_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_staged_resources_cuda(
    Sm87A4W4GateUpDownEdgeM64N128K256StagedResources* const resources)
    noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpDownEdgeM64N128K256StagedResources{};
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
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_staged_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_staged_kernel,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM64N128K256StagedThreads),
      kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes;
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
              kSm87A4W4GateUpDownEdgeM64N128K256StagedMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM64N128K256StagedThreads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM64N128K256StagedCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
