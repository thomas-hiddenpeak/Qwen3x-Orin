#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m32n128_pair_shared_accum.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

namespace cg = cooperative_groups;

inline constexpr unsigned int kPackedK64Bytes = 32U;

struct alignas(16) SharedAccumAStage final {
  std::uint8_t plane
      [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumK64PerCopy]
      [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileM *
       kPackedK64Bytes];
};

// For a fixed component, the 32 lanes of one warp address 32 consecutive
// words.  Gate and Up crews therefore issue conflict-free warp LDS/STS even
// though the logical output coordinates are fragment-scattered.
struct alignas(16) SharedAccumPlanes final {
  float value
      [2U]
      [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumProjectionWarps]
      [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp]
      [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp]
      [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumComponents]
      [32U];
};

struct alignas(16) SharedAccumEdgePlane final {
  std::uint32_t pair
      [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileM]
      [8U][32U];
};

struct alignas(16) SharedAccumShared final {
  SharedAccumAStage
      a_ring[kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumStages];
  SharedAccumPlanes accum;
  SharedAccumEdgePlane edge;
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

// The final Up value starts only after all four S32 fields have been consumed.
// A union makes that lifetime boundary explicit to ptxas: there is never a
// second 32-register final-Up allocation next to the K512 partials.
union FragmentState final {
  Sm87A4W4Accumulator partial;
  Float4 final_up;

  __device__ __forceinline__ FragmentState() noexcept {}
};

struct alignas(16) SharedAccumWorkspace final {
  unsigned int ticket
      [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumSmCount];
  unsigned int error;
};

struct SharedAccumComputeParams final {
  const std::uint8_t* packed_a;
  const std::uint16_t* a_k512_scales_bf16;
  const std::uint8_t* packed_gate_b;
  const std::uint16_t* gate_b_k512_scales_bf16;
  const std::uint8_t* packed_up_b;
  const std::uint16_t* up_b_k512_scales_bf16;
  std::uint8_t* packed_output;
  std::uint16_t* output_k512_scales_bf16;
  SharedAccumWorkspace* workspace;
  unsigned int logical_token_count;
  unsigned int pair_m_tile_count;
  unsigned int edge_group_count;
  unsigned int k512_group_count;
  unsigned int physical_k64_group_count;
  unsigned int work_edge_cell_count;
  float output_clip_ratio;
};

struct SharedAccumKernelParams final {
  SharedAccumComputeParams compute;
  SharedAccumWorkspace* workspace;
};

static_assert(sizeof(SharedAccumAStage) ==
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumAStageBytes);
static_assert(sizeof(SharedAccumPlanes) ==
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumPlanesBytes);
static_assert(sizeof(SharedAccumEdgePlane) ==
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumEdgePlaneBytes);
static_assert(sizeof(SharedAccumShared) ==
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes);
static_assert(sizeof(Float4) == 16U);
static_assert(sizeof(FragmentState) == 16U);
static_assert(sizeof(SharedAccumWorkspace) ==
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumWorkspaceBytes);

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

[[nodiscard]] __device__ __forceinline__ std::uint32_t
load_global_ca_u32(const std::uint8_t* const pointer) noexcept {
  std::uint32_t result = 0U;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("ld.global.ca.u32 %0, [%1];"
               : "=r"(result)
               : "l"(pointer)
               : "memory");
#else
  asm volatile("trap;");
#endif
  return result;
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t
load_global_ca_u16(const std::uint16_t* const pointer) noexcept {
  std::uint16_t result = 0U;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("ld.global.ca.u16 %0, [%1];"
               : "=h"(result)
               : "l"(pointer)
               : "memory");
#else
  asm volatile("trap;");
#endif
  return result;
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t
load_global_cg_u16(const std::uint16_t* const pointer) noexcept {
  std::uint16_t result = 0U;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("ld.global.cg.u16 %0, [%1];"
               : "=h"(result)
               : "l"(pointer)
               : "memory");
#else
  asm volatile("trap;");
#endif
  return result;
}

[[nodiscard]] __device__ __forceinline__ unsigned int sm_id() noexcept {
  unsigned int result = 0U;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
  asm volatile("mov.u32 %0, %%smid;" : "=r"(result));
#else
  asm volatile("trap;");
#endif
  return result;
}

[[nodiscard]] __device__ __forceinline__ unsigned int edge_swizzled_lane(
    const unsigned int row, const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return quantizer_lane ^ ((4U * row + pair_iteration) & 31U);
}

__device__ __forceinline__ void store_edge_pair(
    SharedAccumEdgePlane& edge, const unsigned int row,
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
    const SharedAccumEdgePlane& edge, const unsigned int row,
    const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return edge.pair[row][pair_iteration][edge_swizzled_lane(
      row, pair_iteration, quantizer_lane)];
}

// M32*K256 contains exactly 256 aligned sixteen-byte vectors, so every CTA
// thread contributes one cp.async.cg and one commit to each ring slot.
__device__ __forceinline__ void issue_a_k256_stage(
    SharedAccumAStage& destination,
    const std::uint8_t* const packed_a,
    const unsigned int m32_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileM *
      kPackedK64Bytes / 16U;
  static_assert(
      kVectorsPerPlane *
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumK64PerCopy ==
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumThreads);
  const unsigned int vector = threadIdx.x;
  const unsigned int plane = vector / kVectorsPerPlane;
  const unsigned int vector_in_plane = vector - plane * kVectorsPerPlane;
  const unsigned int row = vector_in_plane / 2U;
  const unsigned int byte = 16U * (vector_in_plane & 1U);
  const unsigned int physical_k64 =
      physical_k256_stage *
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumK64PerCopy +
      plane;
  cp_async_16_cg(
      destination.plane[plane] +
          sm87_a4w4_swizzled_k64_byte_offset(row, byte),
      packed_a + sm87_a4w4_gateup_down_edge_packed_offset(
                     static_cast<std::size_t>(m32_start) + row,
                     physical_k64, byte, physical_k64_group_count));
  cp_async_commit();
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4BFragment
load_v1_b_fragment_ca(
    const std::uint8_t* const packed_b,
    const unsigned int absolute_n8_start,
    const unsigned int physical_k64,
    const unsigned int lane,
    const unsigned int physical_k64_group_count) noexcept {
  const unsigned int n = absolute_n8_start + lane / 4U;
  const unsigned int byte0 = 4U * (lane & 3U);
  const unsigned int byte1 = 16U + byte0;
  return {load_global_ca_u32(
              packed_b + sm87_a4w4_gateup_down_edge_packed_offset(
                             n, physical_k64, byte0,
                             physical_k64_group_count)),
          load_global_ca_u32(
              packed_b + sm87_a4w4_gateup_down_edge_packed_offset(
                             n, physical_k64, byte1,
                             physical_k64_group_count))};
}

__device__ __forceinline__ void clear_partials(
    FragmentState
        (&state)
            [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp])
    noexcept {
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp;
       ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp;
         ++n8) {
      state[m16][n8].partial = Sm87A4W4Accumulator{};
    }
  }
}

__device__ __forceinline__ void accumulate_k256_stage(
    const SharedAccumAStage& stage,
    const std::uint8_t* const projection_b,
    const unsigned int absolute_n128_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count,
    FragmentState
        (&state)
            [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp])
    noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 3U;
  const unsigned int crew_n_start =
      absolute_n128_start + 32U * crew;
#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumK64PerCopy;
       ++plane) {
    Sm87A4W4BFragment
        b[kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp];
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp;
         ++n8) {
      b[n8] = load_v1_b_fragment_ca(
          projection_b, crew_n_start + 8U * n8,
          physical_k256_stage *
                  kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumK64PerCopy +
              plane,
          lane, physical_k64_group_count);
    }
#pragma unroll
    for (unsigned int m16 = 0U;
         m16 < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp;
         ++m16) {
      const Sm87A4W4AFragment a =
          sm87_a4w4_load_a_fragment_swizzled_shared(
              stage.plane[plane] + 16U * m16 * kPackedK64Bytes,
              lane);
#pragma unroll
      for (unsigned int n8 = 0U;
           n8 < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp;
           ++n8) {
        sm87_a4w4_mma_m16n8k64(state[m16][n8].partial, a, b[n8]);
      }
    }
  }
}

template <unsigned int Mode>
__device__ __forceinline__ void apply_k512_group(
    SharedAccumPlanes& planes,
    FragmentState
        (&state)
            [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp],
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const projection_b_k512_scales_bf16,
    const unsigned int projection,
    const unsigned int m32_start,
    const unsigned int absolute_n128_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  static_assert(Mode <= 2U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 3U;
  const unsigned int lane_column = lane & 3U;
  const unsigned int row_in_fragment = lane >> 2U;
  const unsigned int scale_source_lane = lane & ~3U;
  const unsigned int crew_n_start =
      absolute_n128_start + 32U * crew;

#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp;
       ++m16) {
    unsigned int packed_a_scale = 0U;
    if (lane_column == 0U) {
      const unsigned int first_m =
          m32_start + 16U * m16 + row_in_fragment;
      const std::uint16_t scale0 = load_global_cg_u16(
          a_k512_scales_bf16 +
          sm87_a4w4_gateup_down_edge_scale_offset(
              first_m, k512_group, k512_group_count));
      const std::uint16_t scale1 = load_global_cg_u16(
          a_k512_scales_bf16 +
          sm87_a4w4_gateup_down_edge_scale_offset(
              first_m + 8U, k512_group, k512_group_count));
      packed_a_scale = static_cast<unsigned int>(scale0) |
                       (static_cast<unsigned int>(scale1) << 16U);
    }
    packed_a_scale = __shfl_sync(
        0xffff'ffffU, packed_a_scale, scale_source_lane);
    const float a_scale0 =
        decode_bf16(static_cast<std::uint16_t>(packed_a_scale));
    const float a_scale1 = decode_bf16(
        static_cast<std::uint16_t>(packed_a_scale >> 16U));

#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp;
         ++n8) {
      unsigned int packed_b_scale = 0U;
      if (lane < 4U) {
        const unsigned int first_n =
            crew_n_start + 8U * n8 + 2U * lane;
        const std::uint16_t scale0 = load_global_ca_u16(
            projection_b_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                first_n, k512_group, k512_group_count));
        const std::uint16_t scale1 = load_global_ca_u16(
            projection_b_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                first_n + 1U, k512_group, k512_group_count));
        packed_b_scale = static_cast<unsigned int>(scale0) |
                         (static_cast<unsigned int>(scale1) << 16U);
      }
      packed_b_scale = __shfl_sync(
          0xffff'ffffU, packed_b_scale, lane_column);
      const float b_scale0 =
          decode_bf16(static_cast<std::uint16_t>(packed_b_scale));
      const float b_scale1 = decode_bf16(
          static_cast<std::uint16_t>(packed_b_scale >> 16U));

      Float4 output{};
      if constexpr (Mode != 0U) {
        output.x0 = planes.value[projection][crew][m16][n8][0U][lane];
        output.x1 = planes.value[projection][crew][m16][n8][1U][lane];
        output.x2 = planes.value[projection][crew][m16][n8][2U][lane];
        output.x3 = planes.value[projection][crew][m16][n8][3U][lane];
      }
      const Sm87A4W4Accumulator partial = state[m16][n8].partial;
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

      if constexpr (Mode == 2U) {
        if (projection != 0U) {
          state[m16][n8].final_up = output;
        } else {
          planes.value[0U][crew][m16][n8][0U][lane] = output.x0;
          planes.value[0U][crew][m16][n8][1U][lane] = output.x1;
          planes.value[0U][crew][m16][n8][2U][lane] = output.x2;
          planes.value[0U][crew][m16][n8][3U][lane] = output.x3;
        }
      } else {
        planes.value[projection][crew][m16][n8][0U][lane] = output.x0;
        planes.value[projection][crew][m16][n8][1U][lane] = output.x1;
        planes.value[projection][crew][m16][n8][2U][lane] = output.x2;
        planes.value[projection][crew][m16][n8][3U][lane] = output.x3;
      }
    }
  }
}

__device__ __forceinline__ void emit_final_up(
    const SharedAccumPlanes& planes,
    SharedAccumEdgePlane& edge,
    const FragmentState
        (&state)
            [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp],
    const unsigned int global_m32_start,
    const unsigned int logical_token_count,
    const unsigned int cell_in_edge) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 3U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp;
       ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp;
         ++n8) {
      const unsigned int row0 = 16U * m16 + coordinate0.m;
      const unsigned int row1 = 16U * m16 + coordinate2.m;
      const unsigned int local_n = 32U * crew + 8U * n8;
      const unsigned int column = local_n + coordinate0.n;
      const bool valid0 =
          global_m32_start + row0 < logical_token_count;
      const bool valid1 =
          global_m32_start + row1 < logical_token_count;
      const Float4& up = state[m16][n8].final_up;
      const float gate0 = planes.value[0U][crew][m16][n8][0U][lane];
      const float gate1 = planes.value[0U][crew][m16][n8][1U][lane];
      const float gate2 = planes.value[0U][crew][m16][n8][2U][lane];
      const float gate3 = planes.value[0U][crew][m16][n8][3U][lane];
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
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileN +
          column;
      store_edge_pair(edge, row0, edge_column, bits00, bits01);
      store_edge_pair(edge, row1, edge_column, bits10, bits11);
    }
  }
}

__device__ __forceinline__ void compute_n128_cell(
    SharedAccumShared& shared,
    const SharedAccumComputeParams& params,
    const unsigned int m32_start,
    const unsigned int absolute_n128_start,
    const unsigned int cell_in_edge) noexcept {
  const unsigned int projection = threadIdx.x >> 7U;
  const std::uint8_t* const projection_b =
      projection == 0U ? params.packed_gate_b : params.packed_up_b;
  const std::uint16_t* const projection_b_scales =
      projection == 0U ? params.gate_b_k512_scales_bf16
                       : params.up_b_k512_scales_bf16;
  FragmentState
      state[kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp]
           [kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp];
  clear_partials(state);

  const unsigned int physical_stage_count =
      2U * params.k512_group_count;
  const unsigned int initial_stage_count =
      physical_stage_count <
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumStages
          ? physical_stage_count
          : static_cast<unsigned int>(
                kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumStages);
  for (unsigned int stage = 0U; stage < initial_stage_count; ++stage) {
    issue_a_k256_stage(shared.a_ring[stage], params.packed_a,
                       m32_start, stage,
                       params.physical_k64_group_count);
  }
  if (initial_stage_count == 3U) {
    cp_async_wait<2U>();
  } else if (initial_stage_count == 2U) {
    cp_async_wait<1U>();
  } else {
    cp_async_wait<0U>();
  }
  __syncthreads();

  for (unsigned int physical_stage = 0U;
       physical_stage < physical_stage_count; ++physical_stage) {
    const unsigned int group = physical_stage / 2U;
    accumulate_k256_stage(
        shared.a_ring[
            physical_stage %
            kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumStages],
        projection_b, absolute_n128_start, physical_stage,
        params.physical_k64_group_count, state);
    if ((physical_stage & 1U) != 0U) {
      const bool final_group = group + 1U == params.k512_group_count;
      if (final_group) {
        apply_k512_group<2U>(
            shared.accum, state, params.a_k512_scales_bf16,
            projection_b_scales, projection, m32_start,
            absolute_n128_start, group, params.k512_group_count);
      } else if (group == 0U) {
        apply_k512_group<0U>(
            shared.accum, state, params.a_k512_scales_bf16,
            projection_b_scales, projection, m32_start,
            absolute_n128_start, group, params.k512_group_count);
        clear_partials(state);
      } else {
        apply_k512_group<1U>(
            shared.accum, state, params.a_k512_scales_bf16,
            projection_b_scales, projection, m32_start,
            absolute_n128_start, group, params.k512_group_count);
        clear_partials(state);
      }
    }

    // One transition barrier per K256 stage.  It simultaneously proves all
    // warps are done with the released slot and publishes the already-issued
    // next slot.  Therefore the A pipeline contributes two barriers/K512;
    // the final seam barrier below is separately accounted.
    const unsigned int remaining =
        physical_stage_count - physical_stage - 1U;
    if (remaining != 0U) {
      if (remaining >= 2U) {
        cp_async_wait<1U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();
      const unsigned int future_stage =
          physical_stage +
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumStages;
      if (future_stage < physical_stage_count) {
        issue_a_k256_stage(
            shared.a_ring[
                future_stage %
                kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumStages],
            params.packed_a, m32_start, future_stage,
            params.physical_k64_group_count);
      }
    }
  }

  // Gate final shared writes become visible before Up consumes them.  Up's
  // final values still occupy the former S32 partial registers and are never
  // round-tripped through the Up shared plane for the final group.
  __syncthreads();
  if (projection != 0U) {
    emit_final_up(shared.accum, shared.edge, state, m32_start,
                  params.logical_token_count, cell_in_edge);
  }
  __syncthreads();
}

__device__ __forceinline__ void quantize_edge_cell(
    const SharedAccumEdgePlane& edge,
    const unsigned int m32_start,
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
                   kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumWarps;
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
    SharedAccumShared& shared,
    const SharedAccumComputeParams& params,
    const unsigned int pair_m_tile,
    const unsigned int pair_rank,
    const unsigned int edge_group) noexcept {
  const unsigned int m32_start =
      pair_m_tile *
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumPairTileM +
      pair_rank *
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileM;
  const unsigned int edge_n_start =
      edge_group *
      kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumEdgeK;
  for (unsigned int cell = 0U;
       cell < kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCellsPerEdge;
       ++cell) {
    compute_n128_cell(
        shared, params, m32_start,
        edge_n_start +
            cell *
                kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileN,
        cell);
  }
  quantize_edge_cell(
      shared.edge, m32_start, edge_group, params.edge_group_count,
      params.output_clip_ratio, params.packed_output,
      params.output_k512_scales_bf16);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumThreads,
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCtasPerSm)
void q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_shared_accum_kernel(
    const SharedAccumKernelParams params) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared = *reinterpret_cast<SharedAccumShared*>(dynamic_shared);
  unsigned int slot = 0U;
  unsigned int rank = 0U;
  {
    // The pinned target is admitted only after a one-time probe establishes
    // physical SM IDs 0..15.  This same-launch ticket check then proves that
    // all IDs are present exactly twice; unlike a separate setup launch, it
    // cannot become stale when CUDA places the compute grid.
    cg::grid_group grid = cg::this_grid();
    if (blockIdx.x <
            kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumSmCount &&
        threadIdx.x == 0U) {
      params.workspace->ticket[blockIdx.x] = 0U;
    }
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      params.workspace->error =
          gridDim.x ==
                  kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumPersistentCtas
              ? 0U
              : 1U;
    }
    grid.sync();

    slot = sm_id();
    if (threadIdx.x == 0U) {
      unsigned int observed_rank =
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCtasPerSm;
      if (slot <
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumSmCount) {
        observed_rank =
            atomicAdd(params.workspace->ticket + slot, 1U);
      }
      shared.edge.pair[0U][0U][0U] = observed_rank;
      if (slot >=
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumSmCount ||
          observed_rank >=
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCtasPerSm) {
        atomicExch(&params.workspace->error, 1U);
      }
    }
    __syncthreads();
    rank = shared.edge.pair[0U][0U][0U];
    grid.sync();
    if (blockIdx.x <
            kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumSmCount &&
        threadIdx.x == 0U &&
        params.workspace->ticket[blockIdx.x] !=
            kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCtasPerSm) {
      atomicExch(&params.workspace->error, 1U);
    }
    grid.sync();
    if (params.workspace->error != 0U) {
      return;
    }
  }

  // Both CTAs on one physical SM follow the same static ordinal sequence and
  // differ only by their disjoint M32 half.  No cell-level synchronization,
  // queue atomic, or host synchronization is required.
  const SharedAccumComputeParams& compute = params.compute;
  for (unsigned int ordinal = slot;
       ordinal < compute.work_edge_cell_count;
       ordinal +=
           kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumSmCount) {
    const unsigned int edge_group =
        ordinal / compute.pair_m_tile_count;
    const unsigned int pair_m_tile =
        ordinal - edge_group * compute.pair_m_tile_count;
    if (edge_group >= compute.edge_group_count) {
      return;
    }
    compute_edge_cell(shared, compute, pair_m_tile, rank, edge_group);
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
      properties.multiProcessorCount !=
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumSmCount ||
      properties.cooperativeLaunch == 0 ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_shared_accum_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes));
}

}  // namespace

int query_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_shared_accum_resources_cuda(
    Sm87A4W4GateUpDownEdgeM32N128PairSharedAccumResources* const resources)
    noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources =
      Sm87A4W4GateUpDownEdgeM32N128PairSharedAccumResources{};
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
      q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_shared_accum_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_shared_accum_kernel,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumThreads),
      kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes;
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
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumThreads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
