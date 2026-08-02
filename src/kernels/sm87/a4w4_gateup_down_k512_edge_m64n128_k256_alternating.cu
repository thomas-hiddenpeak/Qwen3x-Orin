#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(kSm87A4W4GateUpDownEdgePackedK64Bytes);
std::atomic<bool> g_alternating_resources_ready{false};
std::atomic<bool> g_ldmatrix_pairfeed_resources_ready{false};

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

// Consumer-paired shared feed.  The staged byte layout is already the exact
// x4/x2 ldmatrix mapping proven by the production Down pair-ring kernel, so
// this changes neither the authenticated v1 payload nor the cp.async landing
// geometry.  One x4 load supplies the complete M16 A fragment; independent
// x2 loads supply the matching Gate and Up N8 fragments.
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

__device__ __forceinline__ void accumulate_k256_stage_ldmatrix_pairfeed(
    const Sm87A4W4GateUpDownEdgeStage& stage,
    Sm87A4W4Accumulator (&gate_partials)[1U][4U],
    Sm87A4W4Accumulator (&up_partials)[1U][4U]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m_start =
      (warp / kSm87A4W4GateUpDownEdgeWarpColumns) *
      kSm87A4W4GateUpDownEdgeWarpTileM;
  const unsigned int local_n_start =
      (warp % kSm87A4W4GateUpDownEdgeWarpColumns) *
      kSm87A4W4GateUpDownEdgeWarpTileN;

#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4GateUpDownEdgeK64PerCopy; ++plane) {
    const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
        stage.a[plane] + local_m_start * kPackedK64Bytes, lane);
#pragma unroll
    for (unsigned int n_fragment = 0U; n_fragment < 4U;
         ++n_fragment) {
      const unsigned int fragment_n =
          local_n_start + n_fragment * 8U;
      const Sm87A4W4BFragment gate = load_b_ldmatrix_x2(
          stage.gate[plane] + fragment_n * kPackedK64Bytes, lane);
      const Sm87A4W4BFragment up = load_b_ldmatrix_x2(
          stage.up[plane] + fragment_n * kPackedK64Bytes, lane);
      sm87_a4w4_mma_m16n8k64(
          gate_partials[0U][n_fragment], a, gate);
      sm87_a4w4_mma_m16n8k64(
          up_partials[0U][n_fragment], a, up);
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

__device__ __forceinline__ void apply_k512_group_pairfeed(
    Sm87A4W4GateUpDownEdgeFloat4 (&gate_accumulators)[1U][4U],
    Sm87A4W4GateUpDownEdgeFloat4 (&up_accumulators)[1U][4U],
    const Sm87A4W4Accumulator (&gate_partials)[1U][4U],
    const Sm87A4W4Accumulator (&up_partials)[1U][4U],
    const Sm87A4W4GateUpDownEdgeScaleSlot& scale) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m_start =
      (warp / kSm87A4W4GateUpDownEdgeWarpColumns) *
      kSm87A4W4GateUpDownEdgeWarpTileM;
  const unsigned int local_n_start =
      (warp % kSm87A4W4GateUpDownEdgeWarpColumns) *
      kSm87A4W4GateUpDownEdgeWarpTileN;
  constexpr unsigned int kMask = 0xffff'ffffU;

  // Each lane owns one scale from each operand.  Warp shuffles replace the
  // repeated scalar shared loads for all four N8 fragments.
  const float a_owned =
      decode_bf16(scale.a[local_m_start + (lane & 15U)]);
  const float gate_owned =
      decode_bf16(scale.gate[local_n_start + lane]);
  const float up_owned =
      decode_bf16(scale.up[local_n_start + lane]);
  const unsigned int m_low = lane >> 2U;
  const unsigned int m_high = m_low + 8U;
  const unsigned int n_even = 2U * (lane & 3U);
  const unsigned int n_odd = n_even + 1U;
  const float a0 = __shfl_sync(kMask, a_owned, m_low);
  const float a1 = __shfl_sync(kMask, a_owned, m_high);

#pragma unroll
  for (unsigned int n_fragment = 0U; n_fragment < 4U;
       ++n_fragment) {
    const unsigned int owner = n_fragment * 8U;
    const float gate0 =
        __shfl_sync(kMask, gate_owned, owner + n_even);
    const float gate1 =
        __shfl_sync(kMask, gate_owned, owner + n_odd);
    const float up0 = __shfl_sync(kMask, up_owned, owner + n_even);
    const float up1 = __shfl_sync(kMask, up_owned, owner + n_odd);
    const float gate00 = __fmul_rn(a0, gate0);
    const float gate01 = __fmul_rn(a0, gate1);
    const float gate10 = __fmul_rn(a1, gate0);
    const float gate11 = __fmul_rn(a1, gate1);
    const float up00 = __fmul_rn(a0, up0);
    const float up01 = __fmul_rn(a0, up1);
    const float up10 = __fmul_rn(a1, up0);
    const float up11 = __fmul_rn(a1, up1);
    const Sm87A4W4Accumulator& gate = gate_partials[0U][n_fragment];
    const Sm87A4W4Accumulator& up = up_partials[0U][n_fragment];
    Sm87A4W4GateUpDownEdgeFloat4& gate_output =
        gate_accumulators[0U][n_fragment];
    Sm87A4W4GateUpDownEdgeFloat4& up_output =
        up_accumulators[0U][n_fragment];
    gate_output.x0 = __fmaf_rn(
        static_cast<float>(gate.x0), gate00, gate_output.x0);
    gate_output.x1 = __fmaf_rn(
        static_cast<float>(gate.x1), gate01, gate_output.x1);
    gate_output.x2 = __fmaf_rn(
        static_cast<float>(gate.x2), gate10, gate_output.x2);
    gate_output.x3 = __fmaf_rn(
        static_cast<float>(gate.x3), gate11, gate_output.x3);
    up_output.x0 = __fmaf_rn(
        static_cast<float>(up.x0), up00, up_output.x0);
    up_output.x1 = __fmaf_rn(
        static_cast<float>(up.x1), up01, up_output.x1);
    up_output.x2 = __fmaf_rn(
        static_cast<float>(up.x2), up10, up_output.x2);
    up_output.x3 = __fmaf_rn(
        static_cast<float>(up.x3), up11, up_output.x3);
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

template <bool UseLdmatrixPairfeed>
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
    if constexpr (UseLdmatrixPairfeed) {
      accumulate_k256_stage_ldmatrix_pairfeed(
          shared.pipeline.stage[0U], gate_partials, up_partials);
    } else {
      accumulate_k256_stage(shared.pipeline.stage[0U], gate_partials,
                            up_partials);
    }

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

    if constexpr (UseLdmatrixPairfeed) {
      accumulate_k256_stage_ldmatrix_pairfeed(
          shared.pipeline.stage[1U], gate_partials, up_partials);
      apply_k512_group_pairfeed(
          gate_accumulators, up_accumulators, gate_partials, up_partials,
          shared.pipeline.scale[
              group % kSm87A4W4GateUpDownEdgeScaleSlots]);
    } else {
      accumulate_k256_stage(shared.pipeline.stage[1U], gate_partials,
                            up_partials);
      apply_k512_group(
          gate_accumulators, up_accumulators, gate_partials, up_partials,
          shared.pipeline.scale[
              group % kSm87A4W4GateUpDownEdgeScaleSlots]);
    }

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

template <bool UseLdmatrixPairfeed>
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
    compute_n128_cell<UseLdmatrixPairfeed>(
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
    compute_edge_cell<false>(
        shared, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, logical_token_count, m_tile,
        edge_group, edge_group_count, input_k512_group_count,
        output_clip_ratio, packed_output, output_k512_scales_bf16);
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpDownEdgeThreads,
                      kSm87A4W4GateUpDownEdgeCtasPerSm)
void q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_kernel(
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
    compute_edge_cell<true>(
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

[[nodiscard]] cudaError_t configure_ldmatrix_pairfeed_dynamic_shared()
    noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes));
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
    const unsigned int maximum_launch_ctas,
    const bool require_model_shape,
    const bool use_ldmatrix_pairfeed,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpDownEdgePlan plan =
      require_model_shape
          ? sm87_a4w4_gateup_down_edge_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size)
          : sm87_a4w4_gateup_down_edge_test_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size, maximum_launch_ctas);
  if (plan.launch_ctas == 0U ||
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
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.edge_groups > std::numeric_limits<unsigned int>::max() ||
      plan.input_k512_groups > std::numeric_limits<unsigned int>::max() ||
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
    const cudaError_t status = cudaStreamIsCapturing(stream, &capture_status);
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
  }
  if (capture_status != cudaStreamCaptureStatusNone) {
    const bool resources_ready =
        use_ldmatrix_pairfeed
            ? g_ldmatrix_pairfeed_resources_ready.load(
                  std::memory_order_acquire)
            : g_alternating_resources_ready.load(
                  std::memory_order_acquire);
    if (!resources_ready) {
      return static_cast<int>(cudaErrorNotReady);
    }
  } else {
    Sm87A4W4GateUpDownEdgeM64N128K256AlternatingResources resources{};
    const int resource_status = use_ldmatrix_pairfeed
        ? query_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_resources_cuda(
              &resources)
        : query_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_resources_cuda(
              &resources);
    if (resource_status != static_cast<int>(cudaSuccess)) {
      return resource_status;
    }
  }

  (void)cudaGetLastError();
  if (use_ldmatrix_pairfeed) {
    q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_kernel<<<
        static_cast<unsigned int>(plan.launch_ctas),
        static_cast<unsigned int>(
            kSm87A4W4GateUpDownEdgeM64N128K256AlternatingThreads),
        kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes,
        stream>>>(
        packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b, up_b_k512_scales_bf16,
        static_cast<unsigned int>(plan.logical_token_count),
        static_cast<unsigned int>(plan.m_tiles),
        static_cast<unsigned int>(plan.edge_groups),
        static_cast<unsigned int>(plan.input_k512_groups),
        output_clip_ratio, packed_output, output_k512_scales_bf16);
  } else {
    q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_kernel<<<
        static_cast<unsigned int>(plan.launch_ctas),
        static_cast<unsigned int>(
            kSm87A4W4GateUpDownEdgeM64N128K256AlternatingThreads),
        kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes,
        stream>>>(
        packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b, up_b_k512_scales_bf16,
        static_cast<unsigned int>(plan.logical_token_count),
        static_cast<unsigned int>(plan.m_tiles),
        static_cast<unsigned int>(plan.edge_groups),
        static_cast<unsigned int>(plan.input_k512_groups),
        output_clip_ratio, packed_output, output_k512_scales_bf16);
  }
  return static_cast<int>(cudaPeekAtLastError());
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
  g_alternating_resources_ready.store(true, std::memory_order_release);
  return static_cast<int>(cudaSuccess);
}

int query_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_resources_cuda(
    Sm87A4W4GateUpDownEdgeM64N128K256LdmatrixPairfeedResources* const
        resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources =
      Sm87A4W4GateUpDownEdgeM64N128K256LdmatrixPairfeedResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_alternating_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  cudaError_t status = configure_ldmatrix_pairfeed_dynamic_shared();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_kernel,
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
  g_ldmatrix_pairfeed_resources_ready.store(true,
                                             std::memory_order_release);
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_cuda(
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
      static_cast<unsigned int>(kSm87A4W4GateUpDownEdgePersistentCtas),
      true, false, cuda_stream);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_test_cuda(
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
    const unsigned int maximum_launch_ctas,
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
      maximum_launch_ctas, false, false, cuda_stream);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_cuda(
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
      static_cast<unsigned int>(kSm87A4W4GateUpDownEdgePersistentCtas),
      true, true, cuda_stream);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_ldmatrix_pairfeed_test_cuda(
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
    const unsigned int maximum_launch_ctas,
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
      maximum_launch_ctas, false, true, cuda_stream);
}

}  // namespace q3x::kernels
