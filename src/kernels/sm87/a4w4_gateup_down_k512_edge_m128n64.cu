#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n64.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(
        kSm87A4W4GateUpDownEdgeM128N64PackedK64Bytes);

struct alignas(16) Sm87A4W4GateUpDownEdgeM128N64Stage final {
  std::uint8_t a[kSm87A4W4GateUpDownEdgeM128N64TileM *
                 kSm87A4W4GateUpDownEdgeM128N64PackedK64Bytes];
  std::uint8_t gate[kSm87A4W4GateUpDownEdgeM128N64TileN *
                    kSm87A4W4GateUpDownEdgeM128N64PackedK64Bytes];
  std::uint8_t up[kSm87A4W4GateUpDownEdgeM128N64TileN *
                  kSm87A4W4GateUpDownEdgeM128N64PackedK64Bytes];
};

struct alignas(16) Sm87A4W4GateUpDownEdgeM128N64ScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpDownEdgeM128N64TileM];
  std::uint16_t gate[kSm87A4W4GateUpDownEdgeM128N64TileN];
  std::uint16_t up[kSm87A4W4GateUpDownEdgeM128N64TileN];
};

struct alignas(16) Sm87A4W4GateUpDownEdgeM128N64Pipeline final {
  Sm87A4W4GateUpDownEdgeM128N64Stage
      stage[kSm87A4W4GateUpDownEdgeM128N64Stages];
  Sm87A4W4GateUpDownEdgeM128N64ScaleSlot
      scale[kSm87A4W4GateUpDownEdgeM128N64ScaleSlots];
};

// Logical order is [row][pair iteration][quantizer lane].  The XOR swizzle
// makes a fixed row/pair a 32-bank read and preserves the retained edge's
// exact product ordering while expanding the plane to M128.
struct alignas(16) Sm87A4W4GateUpDownEdgeM128N64Plane final {
  std::uint32_t pair[kSm87A4W4GateUpDownEdgeM128N64TileM][8U][32U];
};

struct alignas(16) Sm87A4W4GateUpDownEdgeM128N64Shared final {
  Sm87A4W4GateUpDownEdgeM128N64Pipeline pipeline;
  Sm87A4W4GateUpDownEdgeM128N64Plane edge;
};

struct alignas(16) Sm87A4W4GateUpDownEdgeM128N64Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

// Every long-lived and transient fragment has a compile-time name.  This is
// deliberate: runtime-indexed fragment arrays are lowered to thread-local
// memory by nvcc for kernels at this register density.
struct Sm87A4W4GateUpDownEdgeM128N64Outputs final {
  Sm87A4W4GateUpDownEdgeM128N64Float4 gate0{};
  Sm87A4W4GateUpDownEdgeM128N64Float4 gate1{};
  Sm87A4W4GateUpDownEdgeM128N64Float4 gate2{};
  Sm87A4W4GateUpDownEdgeM128N64Float4 gate3{};
  Sm87A4W4GateUpDownEdgeM128N64Float4 up0{};
  Sm87A4W4GateUpDownEdgeM128N64Float4 up1{};
  Sm87A4W4GateUpDownEdgeM128N64Float4 up2{};
  Sm87A4W4GateUpDownEdgeM128N64Float4 up3{};
};

struct Sm87A4W4GateUpDownEdgeM128N64Partials final {
  Sm87A4W4Accumulator gate0{};
  Sm87A4W4Accumulator gate1{};
  Sm87A4W4Accumulator gate2{};
  Sm87A4W4Accumulator gate3{};
  Sm87A4W4Accumulator up0{};
  Sm87A4W4Accumulator up1{};
  Sm87A4W4Accumulator up2{};
  Sm87A4W4Accumulator up3{};
};

static_assert(sizeof(Sm87A4W4GateUpDownEdgeM128N64Stage) ==
              kSm87A4W4GateUpDownEdgeM128N64StageBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgeM128N64ScaleSlot) ==
              kSm87A4W4GateUpDownEdgeM128N64ScaleSlotBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgeM128N64Pipeline) ==
              kSm87A4W4GateUpDownEdgeM128N64PipelineBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgeM128N64Plane) ==
              kSm87A4W4GateUpDownEdgeM128N64PlaneBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgeM128N64Shared) ==
              kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes);
static_assert(sizeof(Sm87A4W4GateUpDownEdgeM128N64Outputs) == 128U);
static_assert(sizeof(Sm87A4W4GateUpDownEdgeM128N64Partials) == 128U);

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
    Sm87A4W4GateUpDownEdgeM128N64Plane& edge,
    const unsigned int row, const unsigned int logical_even_column,
    const std::uint16_t even_bits,
    const std::uint16_t odd_bits) noexcept {
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
    const Sm87A4W4GateUpDownEdgeM128N64Plane& edge,
    const unsigned int row, const unsigned int pair_iteration,
    const unsigned int quantizer_lane) noexcept {
  return edge.pair[row][pair_iteration]
                  [edge_swizzled_lane(row, pair_iteration,
                                      quantizer_lane)];
}

// One K64 stage is exactly 512 aligned vectors: 256 A vectors followed by
// 128 Gate vectors and 128 Up vectors.  Every CTA thread therefore issues one
// code copy per stage; only the first warp additionally transfers scales.
__device__ __forceinline__ void issue_k64_codes(
    Sm87A4W4GateUpDownEdgeM128N64Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int physical_k64_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectors =
      static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeM128N64AStageBytes / 16U);
  constexpr unsigned int kBVectors =
      static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeM128N64BStageBytes / 16U);
  static_assert(kAVectors + 2U * kBVectors ==
                kSm87A4W4GateUpDownEdgeM128N64Threads);

  const unsigned int linear = threadIdx.x;
  const unsigned int operand =
      linear < kAVectors ? 0U : (linear < kAVectors + kBVectors ? 1U : 2U);
  const unsigned int operand_vector =
      operand == 0U
          ? linear
          : (operand == 1U ? linear - kAVectors
                           : linear - kAVectors - kBVectors);
  const unsigned int row = operand_vector / 2U;
  const unsigned int row_vector = operand_vector & 1U;
  const unsigned int outer_start =
      operand == 0U ? m_tile_start : absolute_n_tile_start;
  std::uint8_t* const destination =
      operand == 0U ? stage.a : (operand == 1U ? stage.gate : stage.up);
  const std::uint8_t* const source =
      operand == 0U ? packed_a
                    : (operand == 1U ? packed_gate_b : packed_up_b);
  cp_async_16(
      destination +
          sm87_a4w4_swizzled_k64_byte_offset(row, 16U * row_vector),
      source + sm87_a4w4_consumer_packed_offset(
                   static_cast<std::size_t>(outer_start) + row,
                   physical_k64_group, 16U * row_vector,
                   physical_k64_group_count));
}

__device__ __forceinline__ void issue_k512_scales(
    Sm87A4W4GateUpDownEdgeM128N64ScaleSlot& slot,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  constexpr unsigned int kRowsPerVector = 8U;
  constexpr unsigned int kAVectors =
      kSm87A4W4GateUpDownEdgeM128N64TileM / kRowsPerVector;
  constexpr unsigned int kBVectors =
      kSm87A4W4GateUpDownEdgeM128N64TileN / kRowsPerVector;
  if (threadIdx.x < kAVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    cp_async_16(
        slot.a + first_row,
        a_k512_scales_bf16 + sm87_a4w4_gateup_down_edge_scale_offset(
                                  static_cast<std::size_t>(m_tile_start) +
                                      first_row,
                                  k512_group, k512_group_count));
  } else if (threadIdx.x < kAVectors + kBVectors) {
    const unsigned int first_row =
        kRowsPerVector * (threadIdx.x - kAVectors);
    cp_async_16(
        slot.gate + first_row,
        gate_b_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(absolute_n_tile_start) + first_row,
                k512_group, k512_group_count));
  } else if (threadIdx.x < kAVectors + 2U * kBVectors) {
    const unsigned int first_row =
        kRowsPerVector * (threadIdx.x - kAVectors - kBVectors);
    cp_async_16(
        slot.up + first_row,
        up_b_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(absolute_n_tile_start) + first_row,
                k512_group, k512_group_count));
  }
}

__device__ __forceinline__ void issue_k64(
    Sm87A4W4GateUpDownEdgeM128N64Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int physical_k64_group,
    const unsigned int physical_k64_group_count) noexcept {
  issue_k64_codes(stage, packed_a, packed_gate_b, packed_up_b,
                  m_tile_start, absolute_n_tile_start,
                  physical_k64_group, physical_k64_group_count);
  cp_async_commit();
}

__device__ __forceinline__ void issue_k64_and_scales(
    Sm87A4W4GateUpDownEdgeM128N64Stage& stage,
    Sm87A4W4GateUpDownEdgeM128N64ScaleSlot& scale,
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
  issue_k64_codes(stage, packed_a, packed_gate_b, packed_up_b,
                  m_tile_start, absolute_n_tile_start,
                  8U * k512_group, physical_k64_group_count);
  issue_k512_scales(scale, a_k512_scales_bf16,
                    gate_b_k512_scales_bf16,
                    up_b_k512_scales_bf16, m_tile_start,
                    absolute_n_tile_start, k512_group,
                    k512_group_count);
  cp_async_commit();
}

template <unsigned int FragmentN>
__device__ __forceinline__ void accumulate_fragment(
    const Sm87A4W4GateUpDownEdgeM128N64Stage& stage,
    const Sm87A4W4AFragment& a_fragment,
    Sm87A4W4Accumulator& gate_partial,
    Sm87A4W4Accumulator& up_partial,
    const unsigned int local_n_start,
    const unsigned int lane) noexcept {
  static_assert(FragmentN < 4U);
  constexpr unsigned int kFragmentOffset = 8U * FragmentN;
  const Sm87A4W4BFragment gate_fragment =
      sm87_a4w4_load_b_fragment_swizzled_shared(
          stage.gate +
              (local_n_start + kFragmentOffset) * kPackedK64Bytes,
          lane);
  const Sm87A4W4BFragment up_fragment =
      sm87_a4w4_load_b_fragment_swizzled_shared(
          stage.up +
              (local_n_start + kFragmentOffset) * kPackedK64Bytes,
          lane);
  sm87_a4w4_mma_m16n8k64(gate_partial, a_fragment, gate_fragment);
  sm87_a4w4_mma_m16n8k64(up_partial, a_fragment, up_fragment);
}

__device__ __forceinline__ void accumulate_k64_stage(
    const Sm87A4W4GateUpDownEdgeM128N64Stage& stage,
    Sm87A4W4GateUpDownEdgeM128N64Partials& partials) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp / kSm87A4W4GateUpDownEdgeM128N64WarpColumns;
  const unsigned int warp_n =
      warp % kSm87A4W4GateUpDownEdgeM128N64WarpColumns;
  const unsigned int local_m_start =
      warp_m * kSm87A4W4GateUpDownEdgeM128N64WarpTileM;
  const unsigned int local_n_start =
      warp_n * kSm87A4W4GateUpDownEdgeM128N64WarpTileN;
  const Sm87A4W4AFragment a_fragment =
      sm87_a4w4_load_a_fragment_swizzled_shared(
          stage.a + local_m_start * kPackedK64Bytes, lane);
  accumulate_fragment<0U>(stage, a_fragment, partials.gate0,
                          partials.up0, local_n_start, lane);
  accumulate_fragment<1U>(stage, a_fragment, partials.gate1,
                          partials.up1, local_n_start, lane);
  accumulate_fragment<2U>(stage, a_fragment, partials.gate2,
                          partials.up2, local_n_start, lane);
  accumulate_fragment<3U>(stage, a_fragment, partials.gate3,
                          partials.up3, local_n_start, lane);
}

template <unsigned int Remaining>
__device__ __forceinline__ void wait_and_accumulate(
    const Sm87A4W4GateUpDownEdgeM128N64Stage& stage,
    Sm87A4W4GateUpDownEdgeM128N64Partials& partials) noexcept {
  cp_async_wait<Remaining>();
  __syncthreads();
  accumulate_k64_stage(stage, partials);
  __syncthreads();
}

__device__ __forceinline__ void update_output(
    Sm87A4W4GateUpDownEdgeM128N64Float4& output,
    const Sm87A4W4Accumulator& partial, const float scale00,
    const float scale01, const float scale10,
    const float scale11) noexcept {
  output.x0 =
      __fmaf_rn(static_cast<float>(partial.x0), scale00, output.x0);
  output.x1 =
      __fmaf_rn(static_cast<float>(partial.x1), scale01, output.x1);
  output.x2 =
      __fmaf_rn(static_cast<float>(partial.x2), scale10, output.x2);
  output.x3 =
      __fmaf_rn(static_cast<float>(partial.x3), scale11, output.x3);
}

template <unsigned int FragmentN>
__device__ __forceinline__ void apply_fragment(
    Sm87A4W4GateUpDownEdgeM128N64Float4& gate_output,
    Sm87A4W4GateUpDownEdgeM128N64Float4& up_output,
    const Sm87A4W4Accumulator& gate_partial,
    const Sm87A4W4Accumulator& up_partial,
    const Sm87A4W4GateUpDownEdgeM128N64ScaleSlot& scale,
    const unsigned int local_m_start,
    const unsigned int local_n_start,
    const Sm87A4W4AccumulatorCoordinate coordinate0,
    const Sm87A4W4AccumulatorCoordinate coordinate1,
    const Sm87A4W4AccumulatorCoordinate coordinate2) noexcept {
  static_assert(FragmentN < 4U);
  constexpr unsigned int kFragmentOffset = 8U * FragmentN;
  const float a0 = decode_bf16(scale.a[local_m_start + coordinate0.m]);
  const float a1 = decode_bf16(scale.a[local_m_start + coordinate2.m]);
  const unsigned int n0 =
      local_n_start + kFragmentOffset + coordinate0.n;
  const unsigned int n1 =
      local_n_start + kFragmentOffset + coordinate1.n;
  const float gate0 = decode_bf16(scale.gate[n0]);
  const float gate1 = decode_bf16(scale.gate[n1]);
  const float up0 = decode_bf16(scale.up[n0]);
  const float up1 = decode_bf16(scale.up[n1]);
  update_output(gate_output, gate_partial, __fmul_rn(a0, gate0),
                __fmul_rn(a0, gate1), __fmul_rn(a1, gate0),
                __fmul_rn(a1, gate1));
  update_output(up_output, up_partial, __fmul_rn(a0, up0),
                __fmul_rn(a0, up1), __fmul_rn(a1, up0),
                __fmul_rn(a1, up1));
}

__device__ __forceinline__ void apply_k512_group(
    Sm87A4W4GateUpDownEdgeM128N64Outputs& outputs,
    const Sm87A4W4GateUpDownEdgeM128N64Partials& partials,
    const Sm87A4W4GateUpDownEdgeM128N64ScaleSlot& scale) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp / kSm87A4W4GateUpDownEdgeM128N64WarpColumns;
  const unsigned int warp_n =
      warp % kSm87A4W4GateUpDownEdgeM128N64WarpColumns;
  const unsigned int local_m_start =
      warp_m * kSm87A4W4GateUpDownEdgeM128N64WarpTileM;
  const unsigned int local_n_start =
      warp_n * kSm87A4W4GateUpDownEdgeM128N64WarpTileN;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  apply_fragment<0U>(outputs.gate0, outputs.up0, partials.gate0,
                     partials.up0, scale, local_m_start, local_n_start,
                     coordinate0, coordinate1, coordinate2);
  apply_fragment<1U>(outputs.gate1, outputs.up1, partials.gate1,
                     partials.up1, scale, local_m_start, local_n_start,
                     coordinate0, coordinate1, coordinate2);
  apply_fragment<2U>(outputs.gate2, outputs.up2, partials.gate2,
                     partials.up2, scale, local_m_start, local_n_start,
                     coordinate0, coordinate1, coordinate2);
  apply_fragment<3U>(outputs.gate3, outputs.up3, partials.gate3,
                     partials.up3, scale, local_m_start, local_n_start,
                     coordinate0, coordinate1, coordinate2);
}

template <unsigned int FragmentN>
__device__ __forceinline__ void store_product_fragment(
    Sm87A4W4GateUpDownEdgeM128N64Plane& edge,
    const Sm87A4W4GateUpDownEdgeM128N64Float4& gate,
    const Sm87A4W4GateUpDownEdgeM128N64Float4& up,
    const unsigned int global_m_start,
    const unsigned int logical_token_count,
    const unsigned int strip_in_edge,
    const unsigned int local_m_start,
    const unsigned int local_n_start,
    const Sm87A4W4AccumulatorCoordinate coordinate0,
    const Sm87A4W4AccumulatorCoordinate coordinate2) noexcept {
  static_assert(FragmentN < 4U);
  constexpr unsigned int kFragmentOffset = 8U * FragmentN;
  const unsigned int row0 = local_m_start + coordinate0.m;
  const unsigned int row1 = local_m_start + coordinate2.m;
  const unsigned int column =
      strip_in_edge * kSm87A4W4GateUpDownEdgeM128N64TileN +
      local_n_start + kFragmentOffset + coordinate0.n;
  const bool valid0 = global_m_start + row0 < logical_token_count;
  const bool valid1 = global_m_start + row1 < logical_token_count;
  store_edge_pair(edge, row0, column,
                  valid0 ? encode_bf16(silu_product(gate.x0, up.x0)) : 0U,
                  valid0 ? encode_bf16(silu_product(gate.x1, up.x1)) : 0U);
  store_edge_pair(edge, row1, column,
                  valid1 ? encode_bf16(silu_product(gate.x2, up.x2)) : 0U,
                  valid1 ? encode_bf16(silu_product(gate.x3, up.x3)) : 0U);
}

__device__ __forceinline__ void store_bf16_product_strip(
    Sm87A4W4GateUpDownEdgeM128N64Plane& edge,
    const Sm87A4W4GateUpDownEdgeM128N64Outputs& outputs,
    const unsigned int global_m_start,
    const unsigned int logical_token_count,
    const unsigned int strip_in_edge) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp / kSm87A4W4GateUpDownEdgeM128N64WarpColumns;
  const unsigned int warp_n =
      warp % kSm87A4W4GateUpDownEdgeM128N64WarpColumns;
  const unsigned int local_m_start =
      warp_m * kSm87A4W4GateUpDownEdgeM128N64WarpTileM;
  const unsigned int local_n_start =
      warp_n * kSm87A4W4GateUpDownEdgeM128N64WarpTileN;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  store_product_fragment<0U>(edge, outputs.gate0, outputs.up0,
                             global_m_start, logical_token_count,
                             strip_in_edge, local_m_start, local_n_start,
                             coordinate0, coordinate2);
  store_product_fragment<1U>(edge, outputs.gate1, outputs.up1,
                             global_m_start, logical_token_count,
                             strip_in_edge, local_m_start, local_n_start,
                             coordinate0, coordinate2);
  store_product_fragment<2U>(edge, outputs.gate2, outputs.up2,
                             global_m_start, logical_token_count,
                             strip_in_edge, local_m_start, local_n_start,
                             coordinate0, coordinate2);
  store_product_fragment<3U>(edge, outputs.gate3, outputs.up3,
                             global_m_start, logical_token_count,
                             strip_in_edge, local_m_start, local_n_start,
                             coordinate0, coordinate2);
  __syncthreads();
}

__device__ __forceinline__ void compute_n64_strip(
    Sm87A4W4GateUpDownEdgeM128N64Shared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int strip_in_edge,
    const unsigned int input_k512_group_count) noexcept {
  Sm87A4W4GateUpDownEdgeM128N64Outputs outputs{};
  const unsigned int physical_k64_group_count =
      input_k512_group_count * 8U;

  issue_k64_and_scales(
      shared.pipeline.stage[0U], shared.pipeline.scale[0U], packed_a,
      a_k512_scales_bf16, packed_gate_b, gate_b_k512_scales_bf16,
      packed_up_b, up_b_k512_scales_bf16, m_tile_start,
      absolute_n_tile_start, 0U, input_k512_group_count,
      physical_k64_group_count);
  issue_k64(shared.pipeline.stage[1U], packed_a, packed_gate_b,
            packed_up_b, m_tile_start, absolute_n_tile_start, 1U,
            physical_k64_group_count);
  issue_k64(shared.pipeline.stage[2U], packed_a, packed_gate_b,
            packed_up_b, m_tile_start, absolute_n_tile_start, 2U,
            physical_k64_group_count);
  issue_k64(shared.pipeline.stage[3U], packed_a, packed_gate_b,
            packed_up_b, m_tile_start, absolute_n_tile_start, 3U,
            physical_k64_group_count);

  for (unsigned int group = 0U; group < input_k512_group_count; ++group) {
    Sm87A4W4GateUpDownEdgeM128N64Partials partials{};
    const unsigned int physical_base = 8U * group;

    wait_and_accumulate<3U>(shared.pipeline.stage[0U], partials);
    issue_k64(shared.pipeline.stage[0U], packed_a, packed_gate_b,
              packed_up_b, m_tile_start, absolute_n_tile_start,
              physical_base + 4U, physical_k64_group_count);
    wait_and_accumulate<3U>(shared.pipeline.stage[1U], partials);
    issue_k64(shared.pipeline.stage[1U], packed_a, packed_gate_b,
              packed_up_b, m_tile_start, absolute_n_tile_start,
              physical_base + 5U, physical_k64_group_count);
    wait_and_accumulate<3U>(shared.pipeline.stage[2U], partials);
    issue_k64(shared.pipeline.stage[2U], packed_a, packed_gate_b,
              packed_up_b, m_tile_start, absolute_n_tile_start,
              physical_base + 6U, physical_k64_group_count);
    wait_and_accumulate<3U>(shared.pipeline.stage[3U], partials);
    issue_k64(shared.pipeline.stage[3U], packed_a, packed_gate_b,
              packed_up_b, m_tile_start, absolute_n_tile_start,
              physical_base + 7U, physical_k64_group_count);

    const unsigned int next_group = group + 1U;
    if (next_group < input_k512_group_count) {
      wait_and_accumulate<3U>(shared.pipeline.stage[0U], partials);
      issue_k64_and_scales(
          shared.pipeline.stage[0U],
          shared.pipeline.scale[
              next_group % kSm87A4W4GateUpDownEdgeM128N64ScaleSlots],
          packed_a, a_k512_scales_bf16, packed_gate_b,
          gate_b_k512_scales_bf16, packed_up_b,
          up_b_k512_scales_bf16, m_tile_start, absolute_n_tile_start,
          next_group, input_k512_group_count,
          physical_k64_group_count);
      wait_and_accumulate<3U>(shared.pipeline.stage[1U], partials);
      issue_k64(shared.pipeline.stage[1U], packed_a, packed_gate_b,
                packed_up_b, m_tile_start, absolute_n_tile_start,
                8U * next_group + 1U, physical_k64_group_count);
      wait_and_accumulate<3U>(shared.pipeline.stage[2U], partials);
      issue_k64(shared.pipeline.stage[2U], packed_a, packed_gate_b,
                packed_up_b, m_tile_start, absolute_n_tile_start,
                8U * next_group + 2U, physical_k64_group_count);
      wait_and_accumulate<3U>(shared.pipeline.stage[3U], partials);
      issue_k64(shared.pipeline.stage[3U], packed_a, packed_gate_b,
                packed_up_b, m_tile_start, absolute_n_tile_start,
                8U * next_group + 3U, physical_k64_group_count);
    } else {
      wait_and_accumulate<3U>(shared.pipeline.stage[0U], partials);
      wait_and_accumulate<2U>(shared.pipeline.stage[1U], partials);
      wait_and_accumulate<1U>(shared.pipeline.stage[2U], partials);
      wait_and_accumulate<0U>(shared.pipeline.stage[3U], partials);
    }
    apply_k512_group(
        outputs, partials,
        shared.pipeline.scale[
            group % kSm87A4W4GateUpDownEdgeM128N64ScaleSlots]);
  }

  store_bf16_product_strip(shared.edge, outputs, m_tile_start,
                           logical_token_count, strip_in_edge);
}

__device__ __forceinline__ void quantize_edge_cell(
    const Sm87A4W4GateUpDownEdgeM128N64Plane& edge,
    const unsigned int m_tile_start, const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int output_physical_k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;

#pragma unroll 1
  for (unsigned int row_iteration = 0U; row_iteration < 8U;
       ++row_iteration) {
    const unsigned int local_row =
        warp + row_iteration * kSm87A4W4GateUpDownEdgeM128N64Warps;
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
    Sm87A4W4GateUpDownEdgeM128N64Shared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile, const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int input_k512_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int m_tile_start =
      m_tile * kSm87A4W4GateUpDownEdgeM128N64TileM;
  const unsigned int edge_n_start =
      edge_group * kSm87A4W4GateUpDownEdgeM128N64ScaleK;
#pragma unroll 1
  for (unsigned int strip = 0U;
       strip < kSm87A4W4GateUpDownEdgeM128N64StripsPerScale; ++strip) {
    compute_n64_strip(
        shared, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, logical_token_count, m_tile_start,
        edge_n_start +
            strip * kSm87A4W4GateUpDownEdgeM128N64TileN,
        strip, input_k512_group_count);
  }
  quantize_edge_cell(
      shared.edge, m_tile_start, edge_group, edge_group_count,
      edge_group_count * 8U, output_clip_ratio, packed_output,
      output_k512_scales_bf16);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpDownEdgeM128N64Threads,
                      kSm87A4W4GateUpDownEdgeM128N64CtasPerSm)
void q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_kernel(
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
  auto& shared =
      *reinterpret_cast<Sm87A4W4GateUpDownEdgeM128N64Shared*>(
          dynamic_shared);
  const unsigned int work_edge_cells = m_tile_count * edge_group_count;
  for (unsigned int ordinal = blockIdx.x; ordinal < work_edge_cells;
       ordinal += gridDim.x) {
    const unsigned int edge_group = ordinal / m_tile_count;
    const unsigned int m_tile = ordinal - edge_group * m_tile_count;
    compute_edge_cell(
        shared, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, logical_token_count, m_tile,
        edge_group, edge_group_count, input_k512_group_count,
        output_clip_ratio, packed_output, output_k512_scales_bf16);
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
          kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes));
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
    const std::size_t input_size, const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool require_model_shape, void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpDownEdgeM128N64Plan plan =
      require_model_shape
          ? sm87_a4w4_gateup_down_edge_m128n64_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size)
          : sm87_a4w4_gateup_down_edge_m128n64_test_plan(
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

  Sm87A4W4GateUpDownEdgeM128N64Resources resources{};
  const int resource_status =
      query_sm87_a4w4_gateup_down_k512_edge_m128n64_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeM128N64Threads),
      kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes, stream>>>(
      packed_a, a_k512_scales_bf16, packed_gate_b,
      gate_b_k512_scales_bf16, packed_up_b, up_b_k512_scales_bf16,
      static_cast<unsigned int>(plan.logical_token_count),
      static_cast<unsigned int>(plan.m_tiles),
      static_cast<unsigned int>(plan.edge_groups),
      static_cast<unsigned int>(plan.input_k512_groups),
      output_clip_ratio, packed_output, output_k512_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_down_k512_edge_m128n64_resources_cuda(
    Sm87A4W4GateUpDownEdgeM128N64Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpDownEdgeM128N64Resources{};
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
      q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_down_k512_edge_m128n64_kernel,
      static_cast<int>(kSm87A4W4GateUpDownEdgeM128N64Threads),
      kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes;
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
              kSm87A4W4GateUpDownEdgeM128N64MaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4GateUpDownEdgeM128N64DynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4GateUpDownEdgeM128N64Threads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(kSm87A4W4GateUpDownEdgeM128N64CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m128n64_cuda(
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
    const std::size_t input_size, const float output_clip_ratio,
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
      static_cast<unsigned int>(
          kSm87A4W4GateUpDownEdgeM128N64PersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_gateup_down_k512_edge_m128n64_test_cuda(
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
    const std::size_t input_size, const float output_clip_ratio,
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
      maximum_launch_ctas, false, cuda_stream);
}

}  // namespace q3x::kernels
