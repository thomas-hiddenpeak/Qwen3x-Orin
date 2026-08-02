#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m32n128_pair.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

namespace cg = cooperative_groups;

inline constexpr unsigned int kPackedK64Bytes = 32U;

struct alignas(16) M32N128PairAStage final {
  std::uint8_t plane
      [kSm87A4W4GateUpDownEdgeM32N128PairK64PerCopy]
      [kSm87A4W4GateUpDownEdgeM32N128PairTileM *
       kPackedK64Bytes];
};

union alignas(16) M32N128PairWork final {
  M32N128PairAStage
      stage[kSm87A4W4GateUpDownEdgeM32N128PairStages];
  float gate
      [kSm87A4W4GateUpDownEdgeM32N128PairTileM]
      [kSm87A4W4GateUpDownEdgeM32N128PairTileN];
};

// Same logical [row][pair-iteration][quantizer-lane] plane and XOR mapping
// as the production M64 edge.  Only the row extent is halved.
struct alignas(16) M32N128PairEdgePlane final {
  std::uint32_t pair
      [kSm87A4W4GateUpDownEdgeM32N128PairTileM][8U][32U];
};

struct alignas(16) M32N128PairShared final {
  M32N128PairWork work;
  M32N128PairEdgePlane edge;
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

// Admission-only global coordination storage.  It is allocated afresh by
// each launcher invocation; no process-global ticket state is reused.
struct alignas(16) M32N128PairWorkspace final {
  unsigned int ticket[kSm87A4W4GateUpDownEdgeM32N128PairSmCount];
  unsigned int rank_mask[kSm87A4W4GateUpDownEdgeM32N128PairSmCount];
  unsigned int error;
};

struct M32N128PairKernelParams final {
  const std::uint8_t* packed_a;
  const std::uint16_t* a_k512_scales_bf16;
  const std::uint8_t* packed_gate_b;
  const std::uint16_t* gate_b_k512_scales_bf16;
  const std::uint8_t* packed_up_b;
  const std::uint16_t* up_b_k512_scales_bf16;
  std::uint8_t* packed_output;
  std::uint16_t* output_k512_scales_bf16;
  M32N128PairWorkspace* workspace;
  unsigned int logical_token_count;
  unsigned int pair_m_tile_count;
  unsigned int edge_group_count;
  unsigned int k512_group_count;
  unsigned int physical_k64_group_count;
  unsigned int work_edge_cell_count;
  float output_clip_ratio;
};

static_assert(sizeof(M32N128PairAStage) ==
              kSm87A4W4GateUpDownEdgeM32N128PairAStageBytes);
static_assert(sizeof(M32N128PairWork) ==
              kSm87A4W4GateUpDownEdgeM32N128PairGateExchangeBytes);
static_assert(sizeof(M32N128PairEdgePlane) ==
              kSm87A4W4GateUpDownEdgeM32N128PairEdgePlaneBytes);
static_assert(sizeof(M32N128PairShared) ==
              kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes);
static_assert(sizeof(Float4) == 16U);
static_assert(sizeof(M32N128PairWorkspace) ==
              kSm87A4W4GateUpDownEdgeM32N128PairWorkspaceBytes);

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

[[nodiscard]] __device__ __forceinline__ unsigned int
edge_swizzled_lane(
    const unsigned int row, const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return quantizer_lane ^ ((4U * row + pair_iteration) & 31U);
}

__device__ __forceinline__ void store_edge_pair(
    M32N128PairEdgePlane& edge, const unsigned int row,
    const unsigned int logical_even_column,
    const std::uint16_t even_bits,
    const std::uint16_t odd_bits) noexcept {
  const unsigned int logical_pair = logical_even_column / 2U;
  const unsigned int pair_iteration = logical_pair % 8U;
  const unsigned int quantizer_lane = logical_pair / 8U;
  const unsigned int physical_lane = edge_swizzled_lane(
      row, pair_iteration, quantizer_lane);
  edge.pair[row][pair_iteration][physical_lane] =
      static_cast<std::uint32_t>(even_bits) |
      (static_cast<std::uint32_t>(odd_bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
load_edge_pair(
    const M32N128PairEdgePlane& edge, const unsigned int row,
    const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  const unsigned int physical_lane = edge_swizzled_lane(
      row, pair_iteration, quantizer_lane);
  return edge.pair[row][pair_iteration][physical_lane];
}

// A is the only staged GEMM operand.  M32*K256 has exactly 256 aligned
// sixteen-byte vectors, so each CTA thread issues one cp.async.cg.
__device__ __forceinline__ void issue_a_k256_stage(
    M32N128PairAStage& destination,
    const std::uint8_t* const packed_a,
    const unsigned int m32_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      kSm87A4W4GateUpDownEdgeM32N128PairTileM *
      kPackedK64Bytes / 16U;
  static_assert(
      kVectorsPerPlane *
              kSm87A4W4GateUpDownEdgeM32N128PairK64PerCopy ==
          kSm87A4W4GateUpDownEdgeM32N128PairThreads);
  const unsigned int vector = threadIdx.x;
  const unsigned int plane = vector / kVectorsPerPlane;
  const unsigned int vector_in_plane =
      vector - plane * kVectorsPerPlane;
  const unsigned int row = vector_in_plane / 2U;
  const unsigned int byte = 16U * (vector_in_plane & 1U);
  const unsigned int physical_k64 =
      physical_k256_stage *
          kSm87A4W4GateUpDownEdgeM32N128PairK64PerCopy +
      plane;
  cp_async_16_cg(
      destination.plane[plane] +
          sm87_a4w4_swizzled_k64_byte_offset(row, byte),
      packed_a + sm87_a4w4_gateup_down_edge_packed_offset(
                     static_cast<std::size_t>(m32_start) + row,
                     physical_k64, byte,
                     physical_k64_group_count));
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
  return {
      load_global_ca_u32(
          packed_b + sm87_a4w4_gateup_down_edge_packed_offset(
                         n, physical_k64, byte0,
                         physical_k64_group_count)),
      load_global_ca_u32(
          packed_b + sm87_a4w4_gateup_down_edge_packed_offset(
                         n, physical_k64, byte1,
                         physical_k64_group_count))};
}

__device__ __forceinline__ void clear_partials(
    Sm87A4W4Accumulator
        (&partials)
            [kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp]) noexcept {
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp; ++n8) {
      partials[m16][n8] = Sm87A4W4Accumulator{};
    }
  }
}

__device__ __forceinline__ void accumulate_k256_stage(
    const M32N128PairAStage& stage,
    const std::uint8_t* const projection_b,
    const unsigned int absolute_n128_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count,
    Sm87A4W4Accumulator
        (&partials)
            [kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 3U;
  const unsigned int crew_n_start =
      absolute_n128_start +
      crew * kSm87A4W4GateUpDownEdgeM32N128PairWarpTileN;
#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4GateUpDownEdgeM32N128PairK64PerCopy;
       ++plane) {
    Sm87A4W4BFragment
        b[kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp];
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp; ++n8) {
      b[n8] = load_v1_b_fragment_ca(
          projection_b, crew_n_start + 8U * n8,
          physical_k256_stage *
                  kSm87A4W4GateUpDownEdgeM32N128PairK64PerCopy +
              plane,
          lane, physical_k64_group_count);
    }
#pragma unroll
    for (unsigned int m16 = 0U;
         m16 < kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp; ++m16) {
      const Sm87A4W4AFragment a =
          sm87_a4w4_load_a_fragment_swizzled_shared(
              stage.plane[plane] + 16U * m16 * kPackedK64Bytes,
              lane);
#pragma unroll
      for (unsigned int n8 = 0U;
           n8 < kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp; ++n8) {
        sm87_a4w4_mma_m16n8k64(partials[m16][n8], a, b[n8]);
      }
    }
  }
}

__device__ __forceinline__ void apply_k512_group(
    Float4
        (&accumulators)
            [kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp],
    const Sm87A4W4Accumulator
        (&partials)
            [kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp],
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const projection_b_k512_scales_bf16,
    const unsigned int m32_start,
    const unsigned int absolute_n128_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 3U;
  const unsigned int lane_column = lane & 3U;
  const unsigned int row_in_fragment = lane >> 2U;
  const unsigned int scale_source_lane = lane & ~3U;
  const unsigned int crew_n_start =
      absolute_n128_start +
      crew * kSm87A4W4GateUpDownEdgeM32N128PairWarpTileN;

#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp; ++m16) {
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
    const float a_scale0 = decode_bf16(
        static_cast<std::uint16_t>(packed_a_scale));
    const float a_scale1 = decode_bf16(
        static_cast<std::uint16_t>(packed_a_scale >> 16U));

#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp; ++n8) {
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
      const float b_scale0 = decode_bf16(
          static_cast<std::uint16_t>(packed_b_scale));
      const float b_scale1 = decode_bf16(
          static_cast<std::uint16_t>(packed_b_scale >> 16U));
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
    float (&exchange)
        [kSm87A4W4GateUpDownEdgeM32N128PairTileM]
        [kSm87A4W4GateUpDownEdgeM32N128PairTileN],
    const Float4
        (&accumulators)
            [kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int crew = (threadIdx.x >> 5U) & 3U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const Sm87A4W4AccumulatorCoordinate coordinate3 =
      sm87_a4w4_accumulator_coordinate(lane, 3U);
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp; ++n8) {
      const unsigned int base_m = 16U * m16;
      const unsigned int base_n =
          crew * kSm87A4W4GateUpDownEdgeM32N128PairWarpTileN +
          8U * n8;
      const Float4& value = accumulators[m16][n8];
      exchange[base_m + coordinate0.m][base_n + coordinate0.n] =
          value.x0;
      exchange[base_m + coordinate1.m][base_n + coordinate1.n] =
          value.x1;
      exchange[base_m + coordinate2.m][base_n + coordinate2.n] =
          value.x2;
      exchange[base_m + coordinate3.m][base_n + coordinate3.n] =
          value.x3;
    }
  }
}

__device__ __forceinline__ void consume_up_and_store_edge(
    const float (&exchange)
        [kSm87A4W4GateUpDownEdgeM32N128PairTileM]
        [kSm87A4W4GateUpDownEdgeM32N128PairTileN],
    M32N128PairEdgePlane& edge,
    const Float4
        (&accumulators)
            [kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp]
            [kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp],
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
       m16 < kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp; ++n8) {
      const unsigned int base_m = 16U * m16;
      const unsigned int base_n =
          crew * kSm87A4W4GateUpDownEdgeM32N128PairWarpTileN +
          8U * n8;
      const unsigned int row0 = base_m + coordinate0.m;
      const unsigned int row1 = base_m + coordinate2.m;
      const unsigned int column = base_n + coordinate0.n;
      const bool valid0 =
          global_m32_start + row0 < logical_token_count;
      const bool valid1 =
          global_m32_start + row1 < logical_token_count;
      const Float4& up = accumulators[m16][n8];
      const std::uint16_t bits00 =
          valid0 ? encode_bf16(silu_product(
                       exchange[row0][column], up.x0))
                 : 0U;
      const std::uint16_t bits01 =
          valid0 ? encode_bf16(silu_product(
                       exchange[row0][column + 1U], up.x1))
                 : 0U;
      const std::uint16_t bits10 =
          valid1 ? encode_bf16(silu_product(
                       exchange[row1][column], up.x2))
                 : 0U;
      const std::uint16_t bits11 =
          valid1 ? encode_bf16(silu_product(
                       exchange[row1][column + 1U], up.x3))
                 : 0U;
      const unsigned int edge_column =
          cell_in_edge *
              kSm87A4W4GateUpDownEdgeM32N128PairTileN +
          column;
      store_edge_pair(edge, row0, edge_column, bits00, bits01);
      store_edge_pair(edge, row1, edge_column, bits10, bits11);
    }
  }
}

__device__ __forceinline__ void compute_n128_cell(
    M32N128PairShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m32_start,
    const unsigned int absolute_n128_start,
    const unsigned int cell_in_edge,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  const unsigned int projection = threadIdx.x >> 7U;
  const std::uint8_t* const projection_b =
      projection == 0U ? packed_gate_b : packed_up_b;
  const std::uint16_t* const projection_b_scales =
      projection == 0U ? gate_b_k512_scales_bf16
                       : up_b_k512_scales_bf16;
  Float4
      accumulators
          [kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp]
          [kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp]{};
  Sm87A4W4Accumulator
      partials[kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp]
              [kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp]{};

  const unsigned int physical_stage_count = 2U * k512_group_count;
  const unsigned int initial_stage_count =
      physical_stage_count <
              kSm87A4W4GateUpDownEdgeM32N128PairStages
          ? physical_stage_count
          : static_cast<unsigned int>(
                kSm87A4W4GateUpDownEdgeM32N128PairStages);
  for (unsigned int stage = 0U; stage < initial_stage_count; ++stage) {
    issue_a_k256_stage(shared.work.stage[stage], packed_a,
                       m32_start, stage,
                       physical_k64_group_count);
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
        shared.work.stage[
            physical_stage %
            kSm87A4W4GateUpDownEdgeM32N128PairStages],
        projection_b, absolute_n128_start, physical_stage,
        physical_k64_group_count, partials);
    if ((physical_stage & 1U) != 0U) {
      apply_k512_group(
          accumulators, partials, a_k512_scales_bf16,
          projection_b_scales, m32_start, absolute_n128_start,
          group, k512_group_count);
      if (group + 1U < k512_group_count) {
        clear_partials(partials);
      }
    }

    // There is exactly one CTA barrier per K256 stage (two per K512 group).
    // Waiting for the already-issued next stage and synchronizing the CTA
    // simultaneously (a) publishes that stage to every consuming warp and
    // (b) proves every warp is finished with the current ring slot.  Only
    // after that combined transition is the just-released slot refilled with
    // stage p+3; its visibility is covered by a later transition barrier.
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
          kSm87A4W4GateUpDownEdgeM32N128PairStages;
      if (future_stage < physical_stage_count) {
        issue_a_k256_stage(
            shared.work.stage[
                future_stage %
                kSm87A4W4GateUpDownEdgeM32N128PairStages],
            packed_a, m32_start, future_stage,
            physical_k64_group_count);
      }
    }
  }

  // The A ring is dead here; the same 16 KiB work allocation becomes the
  // FP32 Gate exchange.  Projection crews remain concurrent through the
  // entire K sweep and synchronize only at this semantic seam.
  if (projection == 0U) {
    publish_gate(shared.work.gate, accumulators);
  }
  __syncthreads();
  if (projection != 0U) {
    consume_up_and_store_edge(
        shared.work.gate, shared.edge, accumulators, m32_start,
        logical_token_count, cell_in_edge);
  }
  __syncthreads();
}

__device__ __forceinline__ void quantize_edge_cell(
    const M32N128PairEdgePlane& edge,
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
                   kSm87A4W4GateUpDownEdgeM32N128PairWarps;
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
          maximum,
          __shfl_down_sync(0xffff'ffffU, maximum, delta));
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
                                : (even_rounded > 7 ? 7
                                                    : even_rounded);
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
              global_row, edge_group, edge_group_count)] =
          scale_bits;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void compute_edge_cell(
    M32N128PairShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int pair_m_tile,
    const unsigned int pair_rank,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int m32_start =
      pair_m_tile *
          kSm87A4W4GateUpDownEdgeM32N128PairPairTileM +
      pair_rank * kSm87A4W4GateUpDownEdgeM32N128PairTileM;
  const unsigned int edge_n_start =
      edge_group * kSm87A4W4GateUpDownEdgeM32N128PairEdgeK;
  for (unsigned int cell = 0U;
       cell < kSm87A4W4GateUpDownEdgeM32N128PairCellsPerEdge;
       ++cell) {
    compute_n128_cell(
        shared, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, logical_token_count, m32_start,
        edge_n_start +
            cell * kSm87A4W4GateUpDownEdgeM32N128PairTileN,
        cell, k512_group_count, physical_k64_group_count);
  }
  quantize_edge_cell(
      shared.edge, m32_start, edge_group, edge_group_count,
      output_clip_ratio, packed_output, output_k512_scales_bf16);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(
        kSm87A4W4GateUpDownEdgeM32N128PairThreads,
        kSm87A4W4GateUpDownEdgeM32N128PairCtasPerSm)
void q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_kernel(
    const M32N128PairKernelParams params) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<M32N128PairShared*>(dynamic_shared);
  cg::grid_group grid = cg::this_grid();

  // Cooperative launch guarantees all 32 CTAs are resident together.  The
  // first half of the grid initializes request-local state, then %smid
  // tickets establish the two ranks on each of the 16 SMs.  Both grid-wide
  // checks are uniform, so a malformed placement exits without a spinner or
  // a divergent grid barrier.
  if (blockIdx.x < kSm87A4W4GateUpDownEdgeM32N128PairSmCount &&
      threadIdx.x == 0U) {
    params.workspace->ticket[blockIdx.x] = 0U;
    params.workspace->rank_mask[blockIdx.x] = 0U;
  }
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    params.workspace->error = 0U;
  }
  grid.sync();

  const unsigned int sm = sm_id();
  if (threadIdx.x == 0U) {
    unsigned int pair_rank = 2U;
    if (sm < kSm87A4W4GateUpDownEdgeM32N128PairSmCount) {
      pair_rank = atomicAdd(params.workspace->ticket + sm, 1U);
      if (pair_rank < 2U) {
        atomicOr(params.workspace->rank_mask + sm, 1U << pair_rank);
      }
    }
    shared.edge.pair[0U][0U][0U] = pair_rank;
    if (sm >= kSm87A4W4GateUpDownEdgeM32N128PairSmCount ||
        pair_rank >= 2U) {
      atomicExch(&params.workspace->error, 1U);
    }
  }
  __syncthreads();
  const unsigned int pair_rank = shared.edge.pair[0U][0U][0U];
  grid.sync();
  if (blockIdx.x < kSm87A4W4GateUpDownEdgeM32N128PairSmCount &&
      threadIdx.x == 0U &&
      params.workspace->rank_mask[blockIdx.x] != 3U) {
    atomicExch(&params.workspace->error, 1U);
  }
  grid.sync();
  if (params.workspace->error != 0U) {
    return;
  }

  // With exactly two resident CTAs per SM, the static SM-major sequence is
  // a complete, collision-free cover.  Both ranks derive the same ordinal
  // and differ only in their output-disjoint M32 half; no per-cell spin or
  // cross-CTA dependency remains.
  for (unsigned int ordinal = sm;
       ordinal < params.work_edge_cell_count;
       ordinal +=
           kSm87A4W4GateUpDownEdgeM32N128PairSmCount) {
    const unsigned int edge_group =
        ordinal / params.pair_m_tile_count;
    const unsigned int pair_m_tile =
        ordinal - edge_group * params.pair_m_tile_count;
    if (edge_group >= params.edge_group_count) {
      return;
    }
    compute_edge_cell(
        shared, params.packed_a, params.a_k512_scales_bf16,
        params.packed_gate_b, params.gate_b_k512_scales_bf16,
        params.packed_up_b, params.up_b_k512_scales_bf16,
        params.logical_token_count, pair_m_tile, pair_rank,
        edge_group, params.edge_group_count, params.k512_group_count,
        params.physical_k64_group_count, params.output_clip_ratio,
        params.packed_output, params.output_k512_scales_bf16);
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
          kSm87A4W4GateUpDownEdgeM32N128PairSmCount ||
      properties.cooperativeLaunch == 0 ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes));
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
    void* const cooperative_workspace,
    const std::size_t cooperative_workspace_capacity_bytes,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpDownEdgeM32N128PairPlan plan =
      require_model_shape
          ? sm87_a4w4_gateup_down_edge_m32n128_pair_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size)
          : sm87_a4w4_gateup_down_edge_m32n128_pair_test_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size);
  if (plan.work_edge_cells == 0U ||
      plan.launch_ctas !=
          kSm87A4W4GateUpDownEdgeM32N128PairPersistentCtas ||
      !(output_clip_ratio > 0.0F && output_clip_ratio <= 1.0F) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k512_scales_bf16, 16U) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k512_scales_bf16, 16U) ||
      !aligned(packed_output, 16U) ||
      !aligned(output_k512_scales_bf16, 16U) ||
      !aligned(cooperative_workspace,
               kSm87A4W4GateUpDownEdgeM32N128PairWorkspaceAlignment) ||
      cooperative_workspace_capacity_bytes <
          kSm87A4W4GateUpDownEdgeM32N128PairWorkspaceBytes ||
      plan.logical_token_count >
          std::numeric_limits<unsigned int>::max() ||
      plan.pair_m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.edge_groups > std::numeric_limits<unsigned int>::max() ||
      plan.input_k512_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.input_physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.work_edge_cells >
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
      output_scale_capacity_elements < required_output_scales) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t required_output_scale_bytes =
      required_output_scales * sizeof(std::uint16_t);
  const auto output_overlaps = [=](
                                   const void* const input,
                                   const std::size_t input_bytes) noexcept {
    return byte_ranges_overlap(
               packed_output, required_output_bytes, input,
               input_bytes) ||
           byte_ranges_overlap(
               output_k512_scales_bf16,
               required_output_scale_bytes, input, input_bytes);
  };
  if (output_overlaps(packed_a, required_a_bytes) ||
      output_overlaps(a_k512_scales_bf16,
                      required_a_scale_bytes) ||
      output_overlaps(packed_gate_b, required_b_bytes) ||
      output_overlaps(gate_b_k512_scales_bf16,
                      required_b_scale_bytes) ||
      output_overlaps(packed_up_b, required_b_bytes) ||
      output_overlaps(up_b_k512_scales_bf16,
                      required_b_scale_bytes) ||
      byte_ranges_overlap(
          packed_output, required_output_bytes,
          output_k512_scales_bf16, required_output_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  auto* const workspace =
      reinterpret_cast<M32N128PairWorkspace*>(cooperative_workspace);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  M32N128PairKernelParams params{
      packed_a,
      a_k512_scales_bf16,
      packed_gate_b,
      gate_b_k512_scales_bf16,
      packed_up_b,
      up_b_k512_scales_bf16,
      packed_output,
      output_k512_scales_bf16,
      workspace,
      static_cast<unsigned int>(plan.logical_token_count),
      static_cast<unsigned int>(plan.pair_m_tiles),
      static_cast<unsigned int>(plan.edge_groups),
      static_cast<unsigned int>(plan.input_k512_groups),
      static_cast<unsigned int>(plan.input_physical_k64_groups),
      static_cast<unsigned int>(plan.work_edge_cells),
      output_clip_ratio};
  void* arguments[] = {&params};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_kernel),
      dim3(static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeM32N128PairPersistentCtas)),
      dim3(static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeM32N128PairThreads)),
      arguments,
      kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes, stream);
  return static_cast<int>(status);
}

}  // namespace

int query_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_resources_cuda(
    Sm87A4W4GateUpDownEdgeM32N128PairResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpDownEdgeM32N128PairResources{};
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
      q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_kernel,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM32N128PairThreads),
      kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes;
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
              kSm87A4W4GateUpDownEdgeM32N128PairMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM32N128PairThreads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4GateUpDownEdgeM32N128PairCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_cuda(
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
    void* const cooperative_workspace,
    const std::size_t cooperative_workspace_capacity_bytes,
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
      output_k512_scales_bf16, output_scale_capacity_elements, true,
      cooperative_workspace, cooperative_workspace_capacity_bytes,
      cuda_stream);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_test_cuda(
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
  Sm87A4W4GateUpDownEdgeM32N128PairResources resources{};
  const int resource_status =
      query_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  void* workspace = nullptr;
  cudaError_t status = cudaMalloc(
      &workspace, kSm87A4W4GateUpDownEdgeM32N128PairWorkspaceBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const int launch_status = launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_k512_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_k512_scales_bf16,
      up_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size,
      output_clip_ratio, packed_output, packed_output_capacity_bytes,
      output_k512_scales_bf16, output_scale_capacity_elements,
      false, workspace,
      kSm87A4W4GateUpDownEdgeM32N128PairWorkspaceBytes,
      cuda_stream);
  if (launch_status == static_cast<int>(cudaSuccess)) {
    status = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(cuda_stream));
  } else {
    status = static_cast<cudaError_t>(launch_status);
  }
  const cudaError_t free_status = cudaFree(workspace);
  return static_cast<int>(status != cudaSuccess ? status : free_status);
}

}  // namespace q3x::kernels
