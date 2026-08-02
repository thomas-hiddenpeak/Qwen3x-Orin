#include "q3x/kernels/sm87_a4w4_gateup_k512_m64n128_register_pipeline.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes = 32U;
std::atomic<bool> g_register_pipeline_resources_ready{false};

struct alignas(16) RegisterPipelineAStage final {
  std::uint8_t plane
      [kSm87A4W4GateUpK512M64N128RegisterPipelineK64PerGroup]
      [kSm87A4W4GateUpK512M64N128RegisterPipelineTileM *
       kPackedK64Bytes];
};

struct alignas(16) RegisterPipelineScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpK512M64N128RegisterPipelineTileM];
  std::uint16_t paired_b
      [kSm87A4W4GateUpK512M64N128RegisterPipelineComputeTileN][2U];
};

struct alignas(16) RegisterPipelineSlot final {
  RegisterPipelineAStage a;
  RegisterPipelineScaleSlot scale;
};

struct alignas(16) RegisterPipelineSlots final {
  RegisterPipelineSlot
      slot[kSm87A4W4GateUpK512M64N128RegisterPipelineAStages];
};

struct alignas(16) RegisterPipelineGateExchange final {
  float value
      [kSm87A4W4GateUpK512M64N128RegisterPipelineProjectionWarps]
      [kSm87A4W4GateUpK512M64N128RegisterPipelineFragmentsPerWarp]
      [kSm87A4W4GateUpK512M64N128RegisterPipelinePanelsPerWarp]
      [4U][32U];
};

union alignas(16) RegisterPipelineWork final {
  RegisterPipelineSlots pipeline;
  RegisterPipelineGateExchange gate;
};

struct alignas(16) RegisterPipelineEdge final {
  std::uint32_t pair
      [kSm87A4W4GateUpK512M64N128RegisterPipelineTileM][8U][32U];
};

struct alignas(16) RegisterPipelineShared final {
  RegisterPipelineWork work;
  RegisterPipelineEdge edge;
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(RegisterPipelineAStage) ==
              kSm87A4W4GateUpK512M64N128RegisterPipelineAStageBytes);
static_assert(sizeof(RegisterPipelineScaleSlot) ==
              kSm87A4W4GateUpK512M64N128RegisterPipelineScaleSlotBytes);
static_assert(sizeof(RegisterPipelineSlot) ==
              kSm87A4W4GateUpK512M64N128RegisterPipelineSlotBytes);
static_assert(sizeof(RegisterPipelineSlots) ==
              kSm87A4W4GateUpK512M64N128RegisterPipelinePipelineBytes);
static_assert(sizeof(RegisterPipelineGateExchange) ==
              kSm87A4W4GateUpK512M64N128RegisterPipelineGateExchangeBytes);
static_assert(sizeof(RegisterPipelineWork) ==
              kSm87A4W4GateUpK512M64N128RegisterPipelinePipelineBytes);
static_assert(sizeof(RegisterPipelineEdge) ==
              kSm87A4W4GateUpK512M64N128RegisterPipelineEdgeBytes);
static_assert(sizeof(RegisterPipelineShared) ==
              kSm87A4W4GateUpK512M64N128RegisterPipelineDynamicSharedBytes);

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

[[nodiscard]] __device__ __forceinline__ unsigned int edge_swizzled_lane(
    const unsigned int row, const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return quantizer_lane ^ ((4U * row + pair_iteration) & 31U);
}

__device__ __forceinline__ void store_edge_pair(
    RegisterPipelineEdge& edge, const unsigned int row,
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
    const RegisterPipelineEdge& edge, const unsigned int row,
    const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return edge.pair[row][pair_iteration]
                  [edge_swizzled_lane(row, pair_iteration,
                                      quantizer_lane)];
}

__device__ __forceinline__ void issue_a_codes(
    RegisterPipelineAStage& stage,
    const std::uint8_t* const packed_a,
    const unsigned int m_tile_start,
    const unsigned int k512_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      kSm87A4W4GateUpK512M64N128RegisterPipelineTileM *
      kPackedK64Bytes / 16U;
  constexpr unsigned int kVectors =
      kSm87A4W4GateUpK512M64N128RegisterPipelineK64PerGroup *
      kVectorsPerPlane;
  static_assert(kVectors == 2U *
                                kSm87A4W4GateUpK512M64N128RegisterPipelineThreads);
#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector =
        threadIdx.x +
        iteration *
            kSm87A4W4GateUpK512M64N128RegisterPipelineThreads;
    const unsigned int plane = vector / kVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kVectorsPerPlane;
    const unsigned int row = vector_in_plane / 2U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int physical_k64 = 8U * k512_group + plane;
    const unsigned int byte_in_row = 16U * row_vector;
    cp_async_16(
        stage.plane[plane] +
            sm87_a4w4_swizzled_k64_byte_offset(row, byte_in_row),
        packed_a + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64, byte_in_row,
                       physical_k64_group_count));
  }
}

__device__ __forceinline__ void issue_scales(
    RegisterPipelineScaleSlot& slot,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  if (threadIdx.x < 8U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        slot.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x < 32U) {
    const unsigned int n64_half = threadIdx.x >> 4U;
    const unsigned int vector_in_half = threadIdx.x & 15U;
    const unsigned int first_row =
        n64_half * 64U + vector_in_half * 4U;
    cp_async_16(
        &slot.paired_b[first_row][0U],
        paired_b_scales_bf16 +
            sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
                static_cast<std::size_t>(absolute_n_tile_start) +
                    first_row,
                k512_group, k512_group_count));
  }
}

__device__ __forceinline__ void issue_k512_group(
    RegisterPipelineSlot& slot,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  issue_a_codes(slot.a, packed_a, m_tile_start, k512_group,
                physical_k64_group_count);
  issue_scales(slot.scale, a_k512_scales_bf16,
               paired_b_scales_bf16, m_tile_start,
               absolute_n_tile_start, k512_group,
               k512_group_count);
  cp_async_commit();
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4BFragment
load_projection_b_fragment_ca(const std::uint8_t* const pointer) noexcept {
  Sm87A4W4BFragment result{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];"
               : "=r"(result.x0), "=r"(result.x1)
               : "l"(pointer)
               : "memory");
#else
  asm volatile("trap;");
#endif
  return result;
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

__device__ __forceinline__ void accumulate_k512_plane_outer(
    const RegisterPipelineAStage& stage,
    const std::uint8_t* const projection_major_v3_b_codes,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count,
    Sm87A4W4Accumulator (&partials)[2U][4U]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int projection = warp >> 3U;
  const unsigned int projection_warp = warp & 7U;
  const unsigned int fragment_n =
      absolute_n_tile_start + projection_warp * 16U;

#pragma unroll 1
  for (unsigned int plane = 0U; plane < 8U; ++plane) {
    const Sm87A4W4BFragment b0 = load_projection_b_fragment_ca(
        projection_major_v3_b_codes +
        sm87_a4w4_gateup_k512_projection_major_code_lane_offset(
            fragment_n, k512_group, plane, projection, lane,
            k512_group_count));
    const Sm87A4W4BFragment b1 = load_projection_b_fragment_ca(
        projection_major_v3_b_codes +
        sm87_a4w4_gateup_k512_projection_major_code_lane_offset(
            fragment_n + 8U, k512_group, plane, projection, lane,
            k512_group_count));
#pragma unroll
    for (unsigned int panel = 0U; panel < 4U; ++panel) {
      const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
          stage.plane[plane] + panel * 16U * kPackedK64Bytes,
          lane);
      sm87_a4w4_mma_m16n8k64(partials[0U][panel], a, b0);
      sm87_a4w4_mma_m16n8k64(partials[1U][panel], a, b1);
    }
  }
}

template <unsigned int Fragment>
__device__ __forceinline__ void apply_k512_fragment(
    Float4 (&accumulator)[2U][4U],
    const Sm87A4W4Accumulator (&partials)[2U][4U],
    const RegisterPipelineScaleSlot& scale) noexcept {
  static_assert(Fragment < 2U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int projection = warp >> 3U;
  const unsigned int projection_warp = warp & 7U;
  constexpr unsigned int kMask = 0xffff'ffffU;
  const float a_owned_low = decode_bf16(scale.a[lane]);
  const float a_owned_high = decode_bf16(scale.a[32U + lane]);
  unsigned int paired_scale = 0U;
  if (lane < 8U) {
    const unsigned int local_n =
        projection_warp * 16U + Fragment * 8U + lane;
    paired_scale =
        static_cast<unsigned int>(scale.paired_b[local_n][0U]) |
        (static_cast<unsigned int>(scale.paired_b[local_n][1U]) <<
         16U);
  }
  const unsigned int shift = projection * 16U;
  const unsigned int n_even = 2U * (lane & 3U);
  const unsigned int n_odd = n_even + 1U;
  const float b0 = decode_bf16(static_cast<std::uint16_t>(
      (__shfl_sync(kMask, paired_scale, n_even) >> shift) & 0xffffU));
  const float b1 = decode_bf16(static_cast<std::uint16_t>(
      (__shfl_sync(kMask, paired_scale, n_odd) >> shift) & 0xffffU));
  const unsigned int m_low = lane >> 2U;
  const unsigned int m_high = m_low + 8U;

#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    const float a_owned = panel < 2U ? a_owned_low : a_owned_high;
    const unsigned int owner = (panel & 1U) * 16U;
    const float a0 = __shfl_sync(kMask, a_owned, owner + m_low);
    const float a1 = __shfl_sync(kMask, a_owned, owner + m_high);
    const float scale00 = __fmul_rn(a0, b0);
    const float scale01 = __fmul_rn(a0, b1);
    const float scale10 = __fmul_rn(a1, b0);
    const float scale11 = __fmul_rn(a1, b1);
    const Sm87A4W4Accumulator& partial = partials[Fragment][panel];
    Float4& output = accumulator[Fragment][panel];
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

__device__ __forceinline__ void exchange_gate_and_store_bf16_edge(
    RegisterPipelineShared& shared,
    const Float4 (&accumulator)[2U][4U],
    const unsigned int logical_token_count,
    const unsigned int global_m_start,
    const unsigned int cell_in_edge) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int projection = warp >> 3U;
  const unsigned int projection_warp = warp & 7U;

  if (projection == 0U) {
#pragma unroll
    for (unsigned int fragment = 0U; fragment < 2U; ++fragment) {
#pragma unroll
      for (unsigned int panel = 0U; panel < 4U; ++panel) {
        const Float4& value = accumulator[fragment][panel];
        shared.work.gate.value[projection_warp][fragment][panel][0U]
                              [lane] = value.x0;
        shared.work.gate.value[projection_warp][fragment][panel][1U]
                              [lane] = value.x1;
        shared.work.gate.value[projection_warp][fragment][panel][2U]
                              [lane] = value.x2;
        shared.work.gate.value[projection_warp][fragment][panel][3U]
                              [lane] = value.x3;
      }
    }
  }
  __syncthreads();

  if (projection != 0U) {
    const Sm87A4W4AccumulatorCoordinate coordinate0 =
        sm87_a4w4_accumulator_coordinate(lane, 0U);
    const Sm87A4W4AccumulatorCoordinate coordinate2 =
        sm87_a4w4_accumulator_coordinate(lane, 2U);
#pragma unroll
    for (unsigned int fragment = 0U; fragment < 2U; ++fragment) {
#pragma unroll
      for (unsigned int panel = 0U; panel < 4U; ++panel) {
        const unsigned int row0 = panel * 16U + coordinate0.m;
        const unsigned int row1 = panel * 16U + coordinate2.m;
        const unsigned int column =
            cell_in_edge * 128U + projection_warp * 16U +
            fragment * 8U + coordinate0.n;
        const Float4& up = accumulator[fragment][panel];
        const bool valid0 = global_m_start + row0 < logical_token_count;
        const bool valid1 = global_m_start + row1 < logical_token_count;
        const std::uint16_t bits00 =
            valid0
                ? encode_bf16(silu_product(
                      shared.work.gate.value[projection_warp][fragment]
                                            [panel][0U][lane],
                      up.x0))
                : 0U;
        const std::uint16_t bits01 =
            valid0
                ? encode_bf16(silu_product(
                      shared.work.gate.value[projection_warp][fragment]
                                            [panel][1U][lane],
                      up.x1))
                : 0U;
        const std::uint16_t bits10 =
            valid1
                ? encode_bf16(silu_product(
                      shared.work.gate.value[projection_warp][fragment]
                                            [panel][2U][lane],
                      up.x2))
                : 0U;
        const std::uint16_t bits11 =
            valid1
                ? encode_bf16(silu_product(
                      shared.work.gate.value[projection_warp][fragment]
                                            [panel][3U][lane],
                      up.x3))
                : 0U;
        store_edge_pair(shared.edge, row0, column, bits00, bits01);
        store_edge_pair(shared.edge, row1, column, bits10, bits11);
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void compute_n128_cell(
    RegisterPipelineShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const projection_major_v3_b_codes,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int cell_in_edge,
    const unsigned int input_k512_group_count) noexcept {
  Float4 accumulator[2U][4U]{};

  issue_k512_group(shared.work.pipeline.slot[0U], packed_a,
                   a_k512_scales_bf16, paired_b_scales_bf16,
                   m_tile_start, absolute_n_tile_start, 0U,
                   input_k512_group_count,
                   input_k512_group_count * 8U);
  if (input_k512_group_count > 1U) {
    issue_k512_group(shared.work.pipeline.slot[1U], packed_a,
                     a_k512_scales_bf16, paired_b_scales_bf16,
                     m_tile_start, absolute_n_tile_start, 1U,
                     input_k512_group_count,
                     input_k512_group_count * 8U);
    cp_async_wait<1U>();
  } else {
    cp_async_wait<0U>();
  }
  __syncthreads();

  for (unsigned int group = 0U; group < input_k512_group_count; ++group) {
    RegisterPipelineSlot& slot =
        shared.work.pipeline.slot[group & 1U];
    Sm87A4W4Accumulator partials[2U][4U]{};
    accumulate_k512_plane_outer(
        slot.a, projection_major_v3_b_codes,
        absolute_n_tile_start, group, input_k512_group_count,
        partials);
    apply_k512_fragment<0U>(accumulator, partials, slot.scale);
    apply_k512_fragment<1U>(accumulator, partials, slot.scale);

    __syncthreads();
    const unsigned int replacement = group + 2U;
    if (replacement < input_k512_group_count) {
      issue_k512_group(
          slot, packed_a, a_k512_scales_bf16,
          paired_b_scales_bf16, m_tile_start, absolute_n_tile_start,
          replacement, input_k512_group_count,
          input_k512_group_count * 8U);
    }
    if (group + 1U < input_k512_group_count) {
      if (replacement < input_k512_group_count) {
        cp_async_wait<1U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();
    }
  }

  exchange_gate_and_store_bf16_edge(
      shared, accumulator, logical_token_count, m_tile_start,
      cell_in_edge);
}

__device__ __forceinline__ void quantize_edge_cell(
    const RegisterPipelineEdge& edge,
    const unsigned int m_tile_start,
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
    const unsigned int local_row = warp + row_iteration * 16U;
    const unsigned int global_row = m_tile_start + local_row;
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
    RegisterPipelineShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const projection_major_v3_b_codes,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int m_tile_start = m_tile * 64U;
  const unsigned int edge_n_start = edge_group * 512U;
#pragma unroll 1
  for (unsigned int cell = 0U; cell < 4U; ++cell) {
    compute_n128_cell(
        shared, packed_a, a_k512_scales_bf16,
        projection_major_v3_b_codes, paired_b_scales_bf16,
        logical_token_count, m_tile_start,
        edge_n_start + cell * 128U, cell,
        input_k512_group_count);
  }
  quantize_edge_cell(
      shared.edge, m_tile_start, edge_group, edge_group_count,
      output_clip_ratio, packed_output, output_k512_scales_bf16);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(
        kSm87A4W4GateUpK512M64N128RegisterPipelineThreads,
        kSm87A4W4GateUpK512M64N128RegisterPipelineCtasPerSm)
void q3x_sm87_a4w4_gateup_k512_m64n128_register_pipeline_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const projection_major_v3_b_codes,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile_count,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<RegisterPipelineShared*>(dynamic_shared);

  // Sweep N-major so all resident CTAs consume the same projection-major B
  // region together.  Besides preserving cross-CTA L2 reuse, this avoids
  // carrying a residual-wave divisor through the register-dense MMA call.
  const unsigned int work_edge_cells = m_tile_count * edge_group_count;
  for (unsigned int ordinal = blockIdx.x; ordinal < work_edge_cells;
       ordinal += gridDim.x) {
    const unsigned int edge_group = ordinal / m_tile_count;
    const unsigned int m_tile = ordinal - edge_group * m_tile_count;
    compute_edge_cell(
        shared, packed_a, a_k512_scales_bf16,
        projection_major_v3_b_codes, paired_b_scales_bf16,
        logical_token_count, m_tile, edge_group, edge_group_count,
        input_k512_group_count, output_clip_ratio, packed_output,
        output_k512_scales_bf16);
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
      q3x_sm87_a4w4_gateup_k512_m64n128_register_pipeline_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpK512M64N128RegisterPipelineDynamicSharedBytes));
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const projection_major_v3_b_codes,
    const std::size_t projection_major_v3_b_code_capacity_bytes,
    const std::uint16_t* const paired_b_scales_bf16,
    const std::size_t paired_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool require_model_shape,
    void* const cuda_stream) noexcept {
  const auto plan =
      require_model_shape
          ? sm87_a4w4_gateup_down_edge_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size)
          : sm87_a4w4_gateup_down_edge_test_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size, maximum_launch_ctas);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(projection_major_v3_b_codes, 16U) ||
      !aligned(paired_b_scales_bf16, 16U) ||
      !aligned(packed_output, 16U) ||
      !aligned(output_k512_scales_bf16, 16U) ||
      !std::isfinite(output_clip_ratio) || output_clip_ratio <= 0.0F ||
      output_clip_ratio > 1.0F ||
      logical_token_count > std::numeric_limits<unsigned int>::max() ||
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
      sm87_a4w4_gateup_k512_projection_major_code_capacity_bytes(
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
      projection_major_v3_b_code_capacity_bytes < required_b_bytes ||
      paired_b_scale_capacity_elements < required_b_scales ||
      packed_output_capacity_bytes < required_output_bytes ||
      output_scale_capacity_elements < required_output_scales) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t output_scale_bytes =
      required_output_scales * sizeof(std::uint16_t);
  const auto output_overlaps =
      [&](const void* const pointer, const std::size_t bytes) noexcept {
        return byte_ranges_overlap(packed_output, required_output_bytes,
                                   pointer, bytes) ||
               byte_ranges_overlap(output_k512_scales_bf16,
                                   output_scale_bytes, pointer, bytes);
      };
  if (output_overlaps(packed_a, required_a_bytes) ||
      output_overlaps(a_k512_scales_bf16, a_scale_bytes) ||
      output_overlaps(projection_major_v3_b_codes, required_b_bytes) ||
      output_overlaps(paired_b_scales_bf16, b_scale_bytes) ||
      byte_ranges_overlap(packed_output, required_output_bytes,
                          output_k512_scales_bf16,
                          output_scale_bytes)) {
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
    if (!g_register_pipeline_resources_ready.load(
            std::memory_order_acquire)) {
      return static_cast<int>(cudaErrorNotReady);
    }
  } else if (!g_register_pipeline_resources_ready.load(
                 std::memory_order_acquire)) {
    Sm87A4W4GateUpK512M64N128RegisterPipelineResources resources{};
    const int status =
        query_sm87_a4w4_gateup_k512_m64n128_register_pipeline_resources_cuda(
            &resources);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }

  (void)cudaGetLastError();
  q3x_sm87_a4w4_gateup_k512_m64n128_register_pipeline_kernel
      <<<static_cast<unsigned int>(plan.launch_ctas),
         static_cast<unsigned int>(
             kSm87A4W4GateUpK512M64N128RegisterPipelineThreads),
         kSm87A4W4GateUpK512M64N128RegisterPipelineDynamicSharedBytes,
         stream>>>(
          packed_a, a_k512_scales_bf16,
          projection_major_v3_b_codes, paired_b_scales_bf16,
          static_cast<unsigned int>(logical_token_count),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.edge_groups),
          static_cast<unsigned int>(plan.input_k512_groups),
          output_clip_ratio, packed_output, output_k512_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_k512_m64n128_register_pipeline_resources_cuda(
    Sm87A4W4GateUpK512M64N128RegisterPipelineResources* const resources)
    noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpK512M64N128RegisterPipelineResources{};
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
      q3x_sm87_a4w4_gateup_k512_m64n128_register_pipeline_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_k512_m64n128_register_pipeline_kernel,
      static_cast<int>(
          kSm87A4W4GateUpK512M64N128RegisterPipelineThreads),
      kSm87A4W4GateUpK512M64N128RegisterPipelineDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpK512M64N128RegisterPipelineDynamicSharedBytes;
  resources->configured_dynamic_shared_limit_bytes =
      static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes);
  resources->device_optin_shared_limit_bytes =
      static_cast<std::size_t>(properties.sharedMemPerBlockOptin);
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread > 128 ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes != 99'584U ||
      resources->configured_dynamic_shared_limit_bytes < 99'584U ||
      resources->device_optin_shared_limit_bytes < 99'584U ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block < 512 ||
      resources->active_blocks_per_sm < 1) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  g_register_pipeline_resources_ready.store(true,
                                             std::memory_order_release);
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_k512_m64n128_register_pipeline_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const projection_major_v3_b_codes,
    const std::size_t projection_major_v3_b_code_capacity_bytes,
    const std::uint16_t* const paired_b_scales_bf16,
    const std::size_t paired_b_scale_capacity_elements,
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
      a_scale_capacity_elements, projection_major_v3_b_codes,
      projection_major_v3_b_code_capacity_bytes,
      paired_b_scales_bf16, paired_b_scale_capacity_elements,
      logical_token_count, launch_token_count, intermediate_size,
      input_size, output_clip_ratio, packed_output,
      packed_output_capacity_bytes, output_k512_scales_bf16,
      output_scale_capacity_elements,
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512M64N128RegisterPipelinePersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_gateup_k512_m64n128_register_pipeline_test_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const projection_major_v3_b_codes,
    const std::size_t projection_major_v3_b_code_capacity_bytes,
    const std::uint16_t* const paired_b_scales_bf16,
    const std::size_t paired_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, projection_major_v3_b_codes,
      projection_major_v3_b_code_capacity_bytes,
      paired_b_scales_bf16, paired_b_scale_capacity_elements,
      logical_token_count, launch_token_count, intermediate_size,
      input_size, output_clip_ratio, packed_output,
      packed_output_capacity_bytes, output_k512_scales_bf16,
      output_scale_capacity_elements, maximum_launch_ctas, false,
      cuda_stream);
}

}  // namespace q3x::kernels
