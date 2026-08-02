#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes = 32U;

struct alignas(16) AttentionK256Stage final {
  std::uint8_t a[kSm87A4W4AttentionK256K64PerGroup]
                [kSm87A4W4AttentionK256TileM * kPackedK64Bytes];
  std::uint8_t b[kSm87A4W4AttentionK256K64PerGroup]
                [kSm87A4W4AttentionK256TileN * kPackedK64Bytes];
  std::uint16_t a_scale[kSm87A4W4AttentionK256TileM];
  std::uint16_t b_scale[kSm87A4W4AttentionK256TileN];
};

struct alignas(16) AttentionK256Shared final {
  AttentionK256Stage stage[kSm87A4W4AttentionK256Stages];
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(AttentionK256Stage) ==
              kSm87A4W4AttentionK256StageBytes);
static_assert(sizeof(AttentionK256Shared) ==
              kSm87A4W4AttentionK256SharedBytes);
static_assert(sizeof(Float4) == 16U);

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

// Combined stage: 16 KiB A codes, 32 KiB B codes, 256 B A scales and
// 512 B B scales.  Every thread copies two A and four B code vectors; the
// first 16/32 threads additionally copy one scale vector.
__device__ __forceinline__ void issue_o_stage(
    AttentionK256Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k256_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k256_group,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      kSm87A4W4AttentionK256TileM * kPackedK64Bytes / 16U;

#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector =
        threadIdx.x + iteration * kSm87A4W4AttentionK256Threads;
    const unsigned int plane = vector / kAVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kAVectorsPerPlane;
    const unsigned int row = vector_in_plane >> 1U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int k64 =
        k256_group * kSm87A4W4AttentionK256K64PerGroup + plane;
    cp_async_16(
        stage.a[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                             row, 16U * row_vector),
        packed_a + sm87_a4w4_attention_k256_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       k64, 16U * row_vector,
                       physical_k64_group_count));
  }

  const unsigned int b_row = threadIdx.x >> 1U;
  const unsigned int b_row_vector = threadIdx.x & 1U;
#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4AttentionK256K64PerGroup; ++plane) {
    const unsigned int k64 =
        k256_group * kSm87A4W4AttentionK256K64PerGroup + plane;
    cp_async_16(
        stage.b[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                             b_row, 16U * b_row_vector),
        packed_b + sm87_a4w4_attention_k256_packed_offset(
                       static_cast<std::size_t>(n_tile_start) + b_row,
                       k64, 16U * b_row_vector,
                       physical_k64_group_count));
  }

  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.a_scale + first_row,
        a_k256_scales_bf16 +
            sm87_a4w4_attention_k256_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k256_group, k256_group_count));
  }
  if (threadIdx.x < 32U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.b_scale + first_row,
        b_k256_scales_bf16 +
            sm87_a4w4_attention_k256_scale_offset(
                static_cast<std::size_t>(n_tile_start) + first_row,
                k256_group, k256_group_count));
  }
  cp_async_commit();
}

inline constexpr unsigned int kDescriptorProjectionShift = 24U;
inline constexpr unsigned int kDescriptorPanelMask = 0x00ff'ffffU;

struct alignas(16) DevicePanelRecord final {
  const std::uint8_t* packed_b{};
  const std::uint16_t* scales{};
  std::uint16_t* output{};
  unsigned int stride{};
  unsigned int panel{};
};

static_assert(sizeof(DevicePanelRecord) == 32U);

template <Sm87A4W4AttentionK256Topology Topology>
[[nodiscard]] __device__ __forceinline__ unsigned int fixed_descriptor(
    const unsigned int cell, const unsigned int slot) noexcept {
  static_assert(Topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ ||
                Topology == Sm87A4W4AttentionK256Topology::kFullQkv);
  if constexpr (Topology ==
                Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
    if (cell < 48U) {
      return slot < 2U
                 ? 2U * cell + slot
                 : (1U << kDescriptorProjectionShift) |
                       (2U * cell + slot - 2U);
    }
    return 96U + 4U * (cell - 48U) + slot;
  } else {
    if (cell < 8U) {
      return slot < 2U
                 ? 2U * cell + slot
                 : (1U << kDescriptorProjectionShift) |
                       (2U * cell + slot - 2U);
    }
    if (cell < 16U) {
      const unsigned int mixed = cell - 8U;
      return slot < 2U
                 ? 16U + 2U * mixed + slot
                 : (2U << kDescriptorProjectionShift) |
                       (2U * mixed + slot - 2U);
    }
    return 32U + 4U * (cell - 16U) + slot;
  }
}

[[nodiscard]] __device__ __forceinline__ const std::uint8_t*
select_packed_projection(
    const unsigned int projection,
    const std::uint8_t* const packed_b0,
    const std::uint8_t* const packed_b1,
    const std::uint8_t* const packed_b2) noexcept {
  return projection == 0U ? packed_b0
                          : projection == 1U ? packed_b1 : packed_b2;
}

[[nodiscard]] __device__ __forceinline__ const std::uint16_t*
select_scale_projection(
    const unsigned int projection,
    const std::uint16_t* const scales0,
    const std::uint16_t* const scales1,
    const std::uint16_t* const scales2) noexcept {
  return projection == 0U ? scales0
                          : projection == 1U ? scales1 : scales2;
}

template <Sm87A4W4AttentionK256Topology Topology>
__device__ __forceinline__ void issue_fixed_stage(
    AttentionK256Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::uint8_t* const packed_b0,
    const std::uint16_t* const b_scales0,
    const std::uint8_t* const packed_b1,
    const std::uint16_t* const b_scales1,
    const std::uint8_t* const packed_b2,
    const std::uint16_t* const b_scales2,
    const unsigned int macro_cell,
    const unsigned int m_tile_start,
    const unsigned int k256_group,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      kSm87A4W4AttentionK256TileM * kPackedK64Bytes / 16U;
#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector =
        threadIdx.x + iteration * kSm87A4W4AttentionK256Threads;
    const unsigned int plane = vector / kAVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kAVectorsPerPlane;
    const unsigned int row = vector_in_plane >> 1U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int k64 =
        k256_group * kSm87A4W4AttentionK256K64PerGroup + plane;
    cp_async_16(
        stage.a[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                             row, 16U * row_vector),
        packed_a + sm87_a4w4_attention_k256_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       k64, 16U * row_vector,
                       physical_k64_group_count));
  }

  // Threads are partitioned into four 128-thread panel copy cohorts.  A
  // thread selects one projection/panel once and copies that same N64 row
  // pair through all four ordered K64 planes.
  const unsigned int slot = threadIdx.x >> 7U;
  const unsigned int descriptor =
      fixed_descriptor<Topology>(macro_cell, slot);
  const unsigned int projection =
      descriptor >> kDescriptorProjectionShift;
  const unsigned int panel = descriptor & kDescriptorPanelMask;
  const std::uint8_t* const selected_b = select_packed_projection(
      projection, packed_b0, packed_b1, packed_b2);
  const unsigned int thread_in_panel = threadIdx.x & 127U;
  const unsigned int panel_row = thread_in_panel >> 1U;
  const unsigned int row_vector = thread_in_panel & 1U;
  const unsigned int shared_row = slot * 64U + panel_row;
#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4AttentionK256K64PerGroup; ++plane) {
    const unsigned int k64 =
        k256_group * kSm87A4W4AttentionK256K64PerGroup + plane;
    cp_async_16(
        stage.b[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                             shared_row, 16U * row_vector),
        selected_b + sm87_a4w4_attention_k256_packed_offset(
                         static_cast<std::size_t>(panel) * 64U + panel_row,
                         k64, 16U * row_vector,
                         physical_k64_group_count));
  }

  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.a_scale + first_row,
        a_k256_scales_bf16 +
            sm87_a4w4_attention_k256_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k256_group, k256_group_count));
  }
  if (threadIdx.x < 32U) {
    const unsigned int scale_slot = threadIdx.x >> 3U;
    const unsigned int scale_descriptor =
        fixed_descriptor<Topology>(macro_cell, scale_slot);
    const unsigned int scale_projection =
        scale_descriptor >> kDescriptorProjectionShift;
    const unsigned int scale_panel =
        scale_descriptor & kDescriptorPanelMask;
    const std::uint16_t* const selected_scales =
        select_scale_projection(scale_projection, b_scales0, b_scales1,
                                b_scales2);
    const unsigned int row_vector = threadIdx.x & 7U;
    cp_async_16(
        stage.b_scale + 64U * scale_slot + 8U * row_vector,
        selected_scales +
            sm87_a4w4_attention_k256_scale_offset(
                static_cast<std::size_t>(scale_panel) * 64U +
                    8U * row_vector,
                k256_group, k256_group_count));
  }
  cp_async_commit();
}

__device__ __forceinline__ void issue_arbitrary_stage(
    AttentionK256Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k256_scales_bf16,
    const DevicePanelRecord* const panels,
    const unsigned int macro_cell,
    const unsigned int m_tile_start,
    const unsigned int k256_group,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      kSm87A4W4AttentionK256TileM * kPackedK64Bytes / 16U;
#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector =
        threadIdx.x + iteration * kSm87A4W4AttentionK256Threads;
    const unsigned int plane = vector / kAVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kAVectorsPerPlane;
    const unsigned int row = vector_in_plane >> 1U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int k64 =
        k256_group * kSm87A4W4AttentionK256K64PerGroup + plane;
    cp_async_16(
        stage.a[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                             row, 16U * row_vector),
        packed_a + sm87_a4w4_attention_k256_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       k64, 16U * row_vector,
                       physical_k64_group_count));
  }

  const unsigned int slot = threadIdx.x >> 7U;
  const DevicePanelRecord& panel_record = panels[4U * macro_cell + slot];
  const unsigned int thread_in_panel = threadIdx.x & 127U;
  const unsigned int panel_row = thread_in_panel >> 1U;
  const unsigned int row_vector = thread_in_panel & 1U;
  const unsigned int shared_row = slot * 64U + panel_row;
#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4AttentionK256K64PerGroup; ++plane) {
    const unsigned int k64 =
        k256_group * kSm87A4W4AttentionK256K64PerGroup + plane;
    cp_async_16(
        stage.b[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                             shared_row, 16U * row_vector),
        panel_record.packed_b +
            sm87_a4w4_attention_k256_packed_offset(
                static_cast<std::size_t>(panel_record.panel) * 64U +
                    panel_row,
                k64, 16U * row_vector, physical_k64_group_count));
  }

  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.a_scale + first_row,
        a_k256_scales_bf16 +
            sm87_a4w4_attention_k256_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k256_group, k256_group_count));
  }
  if (threadIdx.x < 32U) {
    const unsigned int scale_slot = threadIdx.x >> 3U;
    const DevicePanelRecord& scale_record =
        panels[4U * macro_cell + scale_slot];
    const unsigned int row_vector = threadIdx.x & 7U;
    cp_async_16(
        stage.b_scale + 64U * scale_slot + 8U * row_vector,
        scale_record.scales +
            sm87_a4w4_attention_k256_scale_offset(
                static_cast<std::size_t>(scale_record.panel) * 64U +
                    8U * row_vector,
                k256_group, k256_group_count));
  }
  cp_async_commit();
}

template <unsigned int FragmentN>
__device__ __forceinline__ void accumulate_n8(
    const AttentionK256Stage& stage,
    const Sm87A4W4AFragment& a0,
    const Sm87A4W4AFragment& a1,
    const Sm87A4W4AFragment& a2,
    const Sm87A4W4AFragment& a3,
    const float a_scale0,
    const float a_scale1,
    const unsigned int local_n_start,
    Float4& output) noexcept {
  static_assert(FragmentN < 16U);
  const unsigned int lane = threadIdx.x & 31U;
  constexpr unsigned int kFragmentOffset = FragmentN * 8U;
  Sm87A4W4Accumulator partial{};
  Sm87A4W4BFragment b = sm87_a4w4_load_b_fragment_swizzled_shared(
      stage.b[0U] + (local_n_start + kFragmentOffset) * kPackedK64Bytes,
      lane);
  sm87_a4w4_mma_m16n8k64(partial, a0, b);
  b = sm87_a4w4_load_b_fragment_swizzled_shared(
      stage.b[1U] + (local_n_start + kFragmentOffset) * kPackedK64Bytes,
      lane);
  sm87_a4w4_mma_m16n8k64(partial, a1, b);
  b = sm87_a4w4_load_b_fragment_swizzled_shared(
      stage.b[2U] + (local_n_start + kFragmentOffset) * kPackedK64Bytes,
      lane);
  sm87_a4w4_mma_m16n8k64(partial, a2, b);
  b = sm87_a4w4_load_b_fragment_swizzled_shared(
      stage.b[3U] + (local_n_start + kFragmentOffset) * kPackedK64Bytes,
      lane);
  sm87_a4w4_mma_m16n8k64(partial, a3, b);

  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const float b_scale0 = decode_bf16(
      stage.b_scale[local_n_start + kFragmentOffset + coordinate0.n]);
  const float b_scale1 = decode_bf16(
      stage.b_scale[local_n_start + kFragmentOffset + coordinate1.n]);
  output.x0 = __fmaf_rn(static_cast<float>(partial.x0),
                        __fmul_rn(a_scale0, b_scale0), output.x0);
  output.x1 = __fmaf_rn(static_cast<float>(partial.x1),
                        __fmul_rn(a_scale0, b_scale1), output.x1);
  output.x2 = __fmaf_rn(static_cast<float>(partial.x2),
                        __fmul_rn(a_scale1, b_scale0), output.x2);
  output.x3 = __fmaf_rn(static_cast<float>(partial.x3),
                        __fmul_rn(a_scale1, b_scale1), output.x3);
}

__device__ __forceinline__ void accumulate_group(
    const AttentionK256Stage& stage,
    Float4 (&output)[16U]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int m16 = warp & 7U;
  const unsigned int n128_half = warp >> 3U;
  const unsigned int local_m_start = m16 * 16U;
  const unsigned int local_n_start = n128_half * 128U;

  // All four ordered A fragments become resident before the first N8
  // partial; they are reused across the entire M16N128 warp tile.
  const Sm87A4W4AFragment a0 =
      sm87_a4w4_load_a_fragment_swizzled_shared(
          stage.a[0U] + local_m_start * kPackedK64Bytes, lane);
  const Sm87A4W4AFragment a1 =
      sm87_a4w4_load_a_fragment_swizzled_shared(
          stage.a[1U] + local_m_start * kPackedK64Bytes, lane);
  const Sm87A4W4AFragment a2 =
      sm87_a4w4_load_a_fragment_swizzled_shared(
          stage.a[2U] + local_m_start * kPackedK64Bytes, lane);
  const Sm87A4W4AFragment a3 =
      sm87_a4w4_load_a_fragment_swizzled_shared(
          stage.a[3U] + local_m_start * kPackedK64Bytes, lane);
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const float a_scale0 =
      decode_bf16(stage.a_scale[local_m_start + coordinate0.m]);
  const float a_scale1 =
      decode_bf16(stage.a_scale[local_m_start + coordinate2.m]);

  accumulate_n8<0U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[0U]);
  accumulate_n8<1U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[1U]);
  accumulate_n8<2U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[2U]);
  accumulate_n8<3U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[3U]);
  accumulate_n8<4U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[4U]);
  accumulate_n8<5U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[5U]);
  accumulate_n8<6U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[6U]);
  accumulate_n8<7U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[7U]);
  accumulate_n8<8U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[8U]);
  accumulate_n8<9U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                    local_n_start, output[9U]);
  accumulate_n8<10U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                     local_n_start, output[10U]);
  accumulate_n8<11U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                     local_n_start, output[11U]);
  accumulate_n8<12U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                     local_n_start, output[12U]);
  accumulate_n8<13U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                     local_n_start, output[13U]);
  accumulate_n8<14U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                     local_n_start, output[14U]);
  accumulate_n8<15U>(stage, a0, a1, a2, a3, a_scale0, a_scale1,
                     local_n_start, output[15U]);
}

template <unsigned int FragmentN>
__device__ __forceinline__ void store_n8(
    const Float4& value,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  static_assert(FragmentN < 16U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int m16 = warp & 7U;
  const unsigned int n128_half = warp >> 3U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  constexpr unsigned int kFragmentOffset = FragmentN * 8U;
  const unsigned int global_n =
      n_tile_start + n128_half * 128U + kFragmentOffset + coordinate0.n;
  const unsigned int global_m0 =
      m_tile_start + m16 * 16U + coordinate0.m;
  const unsigned int global_m1 =
      m_tile_start + m16 * 16U + coordinate2.m;
  const std::uint32_t pair0 =
      static_cast<std::uint32_t>(encode_bf16(value.x0)) |
      (static_cast<std::uint32_t>(encode_bf16(value.x1)) << 16U);
  const std::uint32_t pair1 =
      static_cast<std::uint32_t>(encode_bf16(value.x2)) |
      (static_cast<std::uint32_t>(encode_bf16(value.x3)) << 16U);
  *reinterpret_cast<std::uint32_t*>(
      output_bf16 + static_cast<std::size_t>(global_m0) *
                        output_row_stride_elements +
      global_n) = pair0;
  *reinterpret_cast<std::uint32_t*>(
      output_bf16 + static_cast<std::size_t>(global_m1) *
                        output_row_stride_elements +
      global_n) = pair1;
}

__device__ __forceinline__ void store_o_cell(
    const Float4 (&output)[16U],
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  store_n8<0U>(output[0U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<1U>(output[1U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<2U>(output[2U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<3U>(output[3U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<4U>(output[4U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<5U>(output[5U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<6U>(output[6U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<7U>(output[7U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<8U>(output[8U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<9U>(output[9U], m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  store_n8<10U>(output[10U], m_tile_start, n_tile_start, output_bf16,
                output_row_stride_elements);
  store_n8<11U>(output[11U], m_tile_start, n_tile_start, output_bf16,
                output_row_stride_elements);
  store_n8<12U>(output[12U], m_tile_start, n_tile_start, output_bf16,
                output_row_stride_elements);
  store_n8<13U>(output[13U], m_tile_start, n_tile_start, output_bf16,
                output_row_stride_elements);
  store_n8<14U>(output[14U], m_tile_start, n_tile_start, output_bf16,
                output_row_stride_elements);
  store_n8<15U>(output[15U], m_tile_start, n_tile_start, output_bf16,
                output_row_stride_elements);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t*
select_output_projection(
    const unsigned int projection,
    std::uint16_t* const output0,
    std::uint16_t* const output1,
    std::uint16_t* const output2) noexcept {
  return projection == 0U ? output0
                          : projection == 1U ? output1 : output2;
}

[[nodiscard]] __device__ __forceinline__ unsigned int
select_stride_projection(
    const unsigned int projection,
    const unsigned int stride0,
    const unsigned int stride1,
    const unsigned int stride2) noexcept {
  return projection == 0U ? stride0
                          : projection == 1U ? stride1 : stride2;
}

template <unsigned int FragmentN,
          Sm87A4W4AttentionK256Topology Topology>
__device__ __forceinline__ void store_fixed_n8(
    const Float4& value,
    const unsigned int macro_cell,
    const unsigned int m_tile_start,
    std::uint16_t* const output0,
    const unsigned int stride0,
    std::uint16_t* const output1,
    const unsigned int stride1,
    std::uint16_t* const output2,
    const unsigned int stride2) noexcept {
  static_assert(FragmentN < 16U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int m16 = warp & 7U;
  const unsigned int n128_half = warp >> 3U;
  constexpr unsigned int kPanelInHalf = FragmentN / 8U;
  constexpr unsigned int kFragmentInPanel = FragmentN % 8U;
  const unsigned int slot = 2U * n128_half + kPanelInHalf;
  const unsigned int descriptor =
      fixed_descriptor<Topology>(macro_cell, slot);
  const unsigned int projection =
      descriptor >> kDescriptorProjectionShift;
  const unsigned int panel = descriptor & kDescriptorPanelMask;
  std::uint16_t* const selected_output = select_output_projection(
      projection, output0, output1, output2);
  const unsigned int stride = select_stride_projection(
      projection, stride0, stride1, stride2);
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const unsigned int global_n =
      panel * 64U + kFragmentInPanel * 8U + coordinate0.n;
  const unsigned int global_m0 =
      m_tile_start + m16 * 16U + coordinate0.m;
  const unsigned int global_m1 =
      m_tile_start + m16 * 16U + coordinate2.m;
  const std::uint32_t pair0 =
      static_cast<std::uint32_t>(encode_bf16(value.x0)) |
      (static_cast<std::uint32_t>(encode_bf16(value.x1)) << 16U);
  const std::uint32_t pair1 =
      static_cast<std::uint32_t>(encode_bf16(value.x2)) |
      (static_cast<std::uint32_t>(encode_bf16(value.x3)) << 16U);
  *reinterpret_cast<std::uint32_t*>(
      selected_output + static_cast<std::size_t>(global_m0) * stride +
      global_n) = pair0;
  *reinterpret_cast<std::uint32_t*>(
      selected_output + static_cast<std::size_t>(global_m1) * stride +
      global_n) = pair1;
}

template <Sm87A4W4AttentionK256Topology Topology>
__device__ __forceinline__ void store_fixed_cell(
    const Float4 (&output)[16U],
    const unsigned int macro_cell,
    const unsigned int m_tile_start,
    std::uint16_t* const output0,
    const unsigned int stride0,
    std::uint16_t* const output1,
    const unsigned int stride1,
    std::uint16_t* const output2,
    const unsigned int stride2) noexcept {
#define Q3X_STORE_FIXED_N8(fragment)                                      \
  store_fixed_n8<fragment, Topology>(                                    \
      output[fragment], macro_cell, m_tile_start, output0, stride0,      \
      output1, stride1, output2, stride2)
  Q3X_STORE_FIXED_N8(0U);
  Q3X_STORE_FIXED_N8(1U);
  Q3X_STORE_FIXED_N8(2U);
  Q3X_STORE_FIXED_N8(3U);
  Q3X_STORE_FIXED_N8(4U);
  Q3X_STORE_FIXED_N8(5U);
  Q3X_STORE_FIXED_N8(6U);
  Q3X_STORE_FIXED_N8(7U);
  Q3X_STORE_FIXED_N8(8U);
  Q3X_STORE_FIXED_N8(9U);
  Q3X_STORE_FIXED_N8(10U);
  Q3X_STORE_FIXED_N8(11U);
  Q3X_STORE_FIXED_N8(12U);
  Q3X_STORE_FIXED_N8(13U);
  Q3X_STORE_FIXED_N8(14U);
  Q3X_STORE_FIXED_N8(15U);
#undef Q3X_STORE_FIXED_N8
}

template <unsigned int FragmentN>
__device__ __forceinline__ void store_arbitrary_n8(
    const Float4& value,
    const DevicePanelRecord* const panels,
    const unsigned int macro_cell,
    const unsigned int m_tile_start) noexcept {
  static_assert(FragmentN < 16U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int m16 = warp & 7U;
  const unsigned int n128_half = warp >> 3U;
  constexpr unsigned int kPanelInHalf = FragmentN / 8U;
  constexpr unsigned int kFragmentInPanel = FragmentN % 8U;
  const unsigned int slot = 2U * n128_half + kPanelInHalf;
  const DevicePanelRecord& panel_record =
      panels[4U * macro_cell + slot];
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const unsigned int global_n =
      panel_record.panel * 64U + kFragmentInPanel * 8U + coordinate0.n;
  const unsigned int global_m0 =
      m_tile_start + m16 * 16U + coordinate0.m;
  const unsigned int global_m1 =
      m_tile_start + m16 * 16U + coordinate2.m;
  const std::uint32_t pair0 =
      static_cast<std::uint32_t>(encode_bf16(value.x0)) |
      (static_cast<std::uint32_t>(encode_bf16(value.x1)) << 16U);
  const std::uint32_t pair1 =
      static_cast<std::uint32_t>(encode_bf16(value.x2)) |
      (static_cast<std::uint32_t>(encode_bf16(value.x3)) << 16U);
  *reinterpret_cast<std::uint32_t*>(
      panel_record.output +
      static_cast<std::size_t>(global_m0) * panel_record.stride +
      global_n) = pair0;
  *reinterpret_cast<std::uint32_t*>(
      panel_record.output +
      static_cast<std::size_t>(global_m1) * panel_record.stride +
      global_n) = pair1;
}

__device__ __forceinline__ void store_arbitrary_cell(
    const Float4 (&output)[16U],
    const DevicePanelRecord* const panels,
    const unsigned int macro_cell,
    const unsigned int m_tile_start) noexcept {
#define Q3X_STORE_ARBITRARY_N8(fragment)                                  \
  store_arbitrary_n8<fragment>(output[fragment], panels, macro_cell,      \
                               m_tile_start)
  Q3X_STORE_ARBITRARY_N8(0U);
  Q3X_STORE_ARBITRARY_N8(1U);
  Q3X_STORE_ARBITRARY_N8(2U);
  Q3X_STORE_ARBITRARY_N8(3U);
  Q3X_STORE_ARBITRARY_N8(4U);
  Q3X_STORE_ARBITRARY_N8(5U);
  Q3X_STORE_ARBITRARY_N8(6U);
  Q3X_STORE_ARBITRARY_N8(7U);
  Q3X_STORE_ARBITRARY_N8(8U);
  Q3X_STORE_ARBITRARY_N8(9U);
  Q3X_STORE_ARBITRARY_N8(10U);
  Q3X_STORE_ARBITRARY_N8(11U);
  Q3X_STORE_ARBITRARY_N8(12U);
  Q3X_STORE_ARBITRARY_N8(13U);
  Q3X_STORE_ARBITRARY_N8(14U);
  Q3X_STORE_ARBITRARY_N8(15U);
#undef Q3X_STORE_ARBITRARY_N8
}

}  // namespace

inline constexpr unsigned int kAttentionK256QuantizeThreads = 256U;
inline constexpr unsigned int kAttentionK256QuantizeWarps = 8U;

extern "C" __global__ __launch_bounds__(kAttentionK256QuantizeThreads)
void q3x_sm87_a4_quantize_bf16_k256_kernel(
    const std::uint16_t* const input_bf16,
    const std::size_t input_row_stride_elements,
    const std::size_t logical_token_count,
    const std::size_t k256_group_count,
    const std::size_t group_count,
    const float clip_ratio,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_k256_scales_bf16) {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const std::size_t group_ordinal =
      static_cast<std::size_t>(blockIdx.x) *
          kAttentionK256QuantizeWarps +
      warp;
  if (group_ordinal >= group_count) {
    return;
  }
  const std::size_t row = group_ordinal / k256_group_count;
  const std::size_t group =
      group_ordinal - row * k256_group_count;
  const bool valid_row = row < logical_token_count;
  float values[8U];
  float maximum = 0.0F;
  if (valid_row) {
    const std::size_t input_offset =
        row * input_row_stride_elements +
        group * kSm87A4W4AttentionK256ScaleK + 8U * lane;
#pragma unroll
    for (unsigned int index = 0U; index < 8U; ++index) {
      values[index] = decode_bf16(input_bf16[input_offset + index]);
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
  } else {
#pragma unroll
    for (unsigned int index = 0U; index < 8U; ++index) {
      values[index] = 0.0F;
    }
  }
#pragma unroll
  for (unsigned int delta = 16U; delta != 0U; delta >>= 1U) {
    maximum =
        fmaxf(maximum, __shfl_down_sync(0xffff'ffffU, maximum, delta));
  }
  maximum = __shfl_sync(0xffff'ffffU, maximum, 0U);
  const float clipped_maximum = maximum * clip_ratio;
  std::uint16_t scale_bits =
      encode_bf16(maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
  float stored_scale = decode_bf16(scale_bits);
  if (maximum != 0.0F && stored_scale == 0.0F) {
    scale_bits = 1U;
    stored_scale = decode_bf16(scale_bits);
  }
  const std::size_t physical_k64_group_count =
      k256_group_count * kSm87A4W4AttentionK256K64PerGroup;
  const std::size_t physical_group =
      group * kSm87A4W4AttentionK256K64PerGroup + lane / 8U;
  const std::size_t first_byte = 4U * (lane % 8U);
#pragma unroll
  for (unsigned int pair = 0U; pair < 4U; ++pair) {
    const float even = fminf(fmaxf(values[2U * pair], -clipped_maximum),
                             clipped_maximum);
    const float odd =
        fminf(fmaxf(values[2U * pair + 1U], -clipped_maximum),
              clipped_maximum);
    const int even_rounded = stored_scale == 0.0F
                                 ? 0
                                 : __float2int_rn(even / stored_scale);
    const int odd_rounded = stored_scale == 0.0F
                                ? 0
                                : __float2int_rn(odd / stored_scale);
    const int even_code = even_rounded < -7
                              ? -7
                              : even_rounded > 7 ? 7 : even_rounded;
    const int odd_code = odd_rounded < -7
                             ? -7
                             : odd_rounded > 7 ? 7 : odd_rounded;
    packed_a[sm87_a4w4_attention_k256_packed_offset(
        row, physical_group, first_byte + pair,
        physical_k64_group_count)] =
        sm87_a4w4_pack_signed_pair(even_code, odd_code);
  }
  if (lane == 0U) {
    a_k256_scales_bf16[sm87_a4w4_attention_k256_scale_offset(
        row, group, k256_group_count)] = scale_bits;
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4AttentionK256Threads,
                      kSm87A4W4AttentionK256CtasPerSm)
void q3x_sm87_a4w4_attention_o_k256_m128n256_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k256_scales_bf16,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int work_cell_count) {
  extern __shared__ __align__(16) unsigned char o_dynamic_shared[];
  auto& shared = *reinterpret_cast<AttentionK256Shared*>(o_dynamic_shared);

  for (unsigned int work = blockIdx.x; work < work_cell_count;
       work += gridDim.x) {
    const unsigned int macro_cell = work / m_tile_count;
    const unsigned int m_tile = work - macro_cell * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4AttentionK256TileM;
    const unsigned int n_tile_start =
        macro_cell * kSm87A4W4AttentionK256TileN;
    Float4 output[16U]{};

    issue_o_stage(shared.stage[0U], packed_a, a_k256_scales_bf16,
                  packed_b, b_k256_scales_bf16, m_tile_start,
                  n_tile_start, 0U, k256_group_count,
                  physical_k64_group_count);
    if (k256_group_count > 1U) {
      issue_o_stage(shared.stage[1U], packed_a, a_k256_scales_bf16,
                    packed_b, b_k256_scales_bf16, m_tile_start,
                    n_tile_start, 1U, k256_group_count,
                    physical_k64_group_count);
    }
    if (k256_group_count > 2U) {
      issue_o_stage(shared.stage[2U], packed_a, a_k256_scales_bf16,
                    packed_b, b_k256_scales_bf16, m_tile_start,
                    n_tile_start, 2U, k256_group_count,
                    physical_k64_group_count);
    }

    for (unsigned int group = 0U; group < k256_group_count; ++group) {
      if (group + 2U < k256_group_count) {
        cp_async_wait<2U>();
      } else if (group + 1U < k256_group_count) {
        cp_async_wait<1U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();
      accumulate_group(
          shared.stage[group % kSm87A4W4AttentionK256Stages], output);
      __syncthreads();
      if (group + kSm87A4W4AttentionK256Stages < k256_group_count) {
        const unsigned int next_group =
            group + kSm87A4W4AttentionK256Stages;
        issue_o_stage(
            shared.stage[group % kSm87A4W4AttentionK256Stages], packed_a,
            a_k256_scales_bf16, packed_b, b_k256_scales_bf16,
            m_tile_start, n_tile_start, next_group, k256_group_count,
            physical_k64_group_count);
      }
    }

    store_o_cell(output, m_tile_start, n_tile_start, output_bf16,
                 output_row_stride_elements);
    __syncthreads();
  }
}

template <Sm87A4W4AttentionK256Topology Topology>
__global__ __launch_bounds__(kSm87A4W4AttentionK256Threads,
                             kSm87A4W4AttentionK256CtasPerSm)
void q3x_sm87_a4w4_attention_k256_m128n256_fixed_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::uint8_t* const packed_b0,
    const std::uint16_t* const b_scales0,
    const std::uint8_t* const packed_b1,
    const std::uint16_t* const b_scales1,
    const std::uint8_t* const packed_b2,
    const std::uint16_t* const b_scales2,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output0,
    const unsigned int stride0,
    std::uint16_t* const output1,
    const unsigned int stride1,
    std::uint16_t* const output2,
    const unsigned int stride2,
    const unsigned int m_tile_count,
    const unsigned int work_cell_count) {
  static_assert(Topology ==
                        Sm87A4W4AttentionK256Topology::kLinearQkvZ ||
                    Topology ==
                        Sm87A4W4AttentionK256Topology::kFullQkv);
  extern __shared__ __align__(16) unsigned char fixed_dynamic_shared[];
  auto& shared =
      *reinterpret_cast<AttentionK256Shared*>(fixed_dynamic_shared);

  for (unsigned int work = blockIdx.x; work < work_cell_count;
       work += gridDim.x) {
    const unsigned int macro_cell = work / m_tile_count;
    const unsigned int m_tile = work - macro_cell * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4AttentionK256TileM;
    Float4 output[16U]{};

    issue_fixed_stage<Topology>(
        shared.stage[0U], packed_a, a_k256_scales_bf16, packed_b0,
        b_scales0, packed_b1, b_scales1, packed_b2, b_scales2,
        macro_cell, m_tile_start, 0U, k256_group_count,
        physical_k64_group_count);
    if (k256_group_count > 1U) {
      issue_fixed_stage<Topology>(
          shared.stage[1U], packed_a, a_k256_scales_bf16, packed_b0,
          b_scales0, packed_b1, b_scales1, packed_b2, b_scales2,
          macro_cell, m_tile_start, 1U, k256_group_count,
          physical_k64_group_count);
    }
    if (k256_group_count > 2U) {
      issue_fixed_stage<Topology>(
          shared.stage[2U], packed_a, a_k256_scales_bf16, packed_b0,
          b_scales0, packed_b1, b_scales1, packed_b2, b_scales2,
          macro_cell, m_tile_start, 2U, k256_group_count,
          physical_k64_group_count);
    }

    for (unsigned int group = 0U; group < k256_group_count; ++group) {
      if (group + 2U < k256_group_count) {
        cp_async_wait<2U>();
      } else if (group + 1U < k256_group_count) {
        cp_async_wait<1U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();
      accumulate_group(
          shared.stage[group % kSm87A4W4AttentionK256Stages], output);
      __syncthreads();
      if (group + kSm87A4W4AttentionK256Stages < k256_group_count) {
        const unsigned int next_group =
            group + kSm87A4W4AttentionK256Stages;
        issue_fixed_stage<Topology>(
            shared.stage[group % kSm87A4W4AttentionK256Stages], packed_a,
            a_k256_scales_bf16, packed_b0, b_scales0, packed_b1,
            b_scales1, packed_b2, b_scales2, macro_cell, m_tile_start,
            next_group, k256_group_count, physical_k64_group_count);
      }
    }

    store_fixed_cell<Topology>(
        output, macro_cell, m_tile_start, output0, stride0, output1,
        stride1, output2, stride2);
    __syncthreads();
  }
}

__global__ __launch_bounds__(kSm87A4W4AttentionK256Threads,
                             kSm87A4W4AttentionK256CtasPerSm)
void q3x_sm87_a4w4_attention_k256_m128n256_arbitrary_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k256_scales_bf16,
    const DevicePanelRecord* const panels,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int m_tile_count,
    const unsigned int work_cell_count) {
  extern __shared__ __align__(16) unsigned char arbitrary_dynamic_shared[];
  auto& shared =
      *reinterpret_cast<AttentionK256Shared*>(arbitrary_dynamic_shared);

  for (unsigned int work = blockIdx.x; work < work_cell_count;
       work += gridDim.x) {
    const unsigned int macro_cell = work / m_tile_count;
    const unsigned int m_tile = work - macro_cell * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4AttentionK256TileM;
    Float4 output[16U]{};

    issue_arbitrary_stage(
        shared.stage[0U], packed_a, a_k256_scales_bf16, panels,
        macro_cell, m_tile_start, 0U, k256_group_count,
        physical_k64_group_count);
    if (k256_group_count > 1U) {
      issue_arbitrary_stage(
          shared.stage[1U], packed_a, a_k256_scales_bf16, panels,
          macro_cell, m_tile_start, 1U, k256_group_count,
          physical_k64_group_count);
    }
    if (k256_group_count > 2U) {
      issue_arbitrary_stage(
          shared.stage[2U], packed_a, a_k256_scales_bf16, panels,
          macro_cell, m_tile_start, 2U, k256_group_count,
          physical_k64_group_count);
    }

    for (unsigned int group = 0U; group < k256_group_count; ++group) {
      if (group + 2U < k256_group_count) {
        cp_async_wait<2U>();
      } else if (group + 1U < k256_group_count) {
        cp_async_wait<1U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();
      accumulate_group(
          shared.stage[group % kSm87A4W4AttentionK256Stages], output);
      __syncthreads();
      if (group + kSm87A4W4AttentionK256Stages < k256_group_count) {
        const unsigned int next_group =
            group + kSm87A4W4AttentionK256Stages;
        issue_arbitrary_stage(
            shared.stage[group % kSm87A4W4AttentionK256Stages], packed_a,
            a_k256_scales_bf16, panels, macro_cell, m_tile_start,
            next_group, k256_group_count, physical_k64_group_count);
      }
    }

    store_arbitrary_cell(output, panels, macro_cell, m_tile_start);
    __syncthreads();
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
      properties.multiProcessorCount != static_cast<int>(kRequiredSmCount) ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4AttentionK256SharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_o_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_attention_o_k256_m128n256_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4AttentionK256SharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_attention_o_k256_m128n256_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

template <Sm87A4W4AttentionK256Topology Topology>
[[nodiscard]] cudaError_t configure_fixed_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_attention_k256_m128n256_fixed_kernel<Topology>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4AttentionK256SharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_attention_k256_m128n256_fixed_kernel<Topology>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

[[nodiscard]] cudaError_t configure_arbitrary_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_attention_k256_m128n256_arbitrary_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4AttentionK256SharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_attention_k256_m128n256_arbitrary_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

template <Sm87A4W4AttentionK256Topology Topology>
[[nodiscard]] cudaError_t merge_fixed_resources(
    int* const maximum_registers,
    std::size_t* const maximum_local_bytes,
    int* const minimum_maximum_threads,
    int* const minimum_active_blocks,
    std::size_t* const configured_dynamic_shared_limit) noexcept {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_attention_k256_m128n256_fixed_kernel<Topology>);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_attention_k256_m128n256_fixed_kernel<Topology>,
      static_cast<int>(kSm87A4W4AttentionK256Threads),
      kSm87A4W4AttentionK256SharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  if (attributes.numRegs > *maximum_registers) {
    *maximum_registers = attributes.numRegs;
  }
  if (attributes.localSizeBytes > *maximum_local_bytes) {
    *maximum_local_bytes = attributes.localSizeBytes;
  }
  if (attributes.maxThreadsPerBlock < *minimum_maximum_threads) {
    *minimum_maximum_threads = attributes.maxThreadsPerBlock;
  }
  if (active_blocks < *minimum_active_blocks) {
    *minimum_active_blocks = active_blocks;
  }
  if (static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes) <
      *configured_dynamic_shared_limit) {
    *configured_dynamic_shared_limit =
        static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes);
  }
  return cudaSuccess;
}

[[nodiscard]] cudaError_t merge_arbitrary_resources(
    int* const maximum_registers,
    std::size_t* const maximum_local_bytes,
    int* const minimum_maximum_threads,
    int* const minimum_active_blocks,
    std::size_t* const configured_dynamic_shared_limit) noexcept {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_attention_k256_m128n256_arbitrary_kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_attention_k256_m128n256_arbitrary_kernel,
      static_cast<int>(kSm87A4W4AttentionK256Threads),
      kSm87A4W4AttentionK256SharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  if (attributes.numRegs > *maximum_registers) {
    *maximum_registers = attributes.numRegs;
  }
  if (attributes.localSizeBytes > *maximum_local_bytes) {
    *maximum_local_bytes = attributes.localSizeBytes;
  }
  if (attributes.maxThreadsPerBlock < *minimum_maximum_threads) {
    *minimum_maximum_threads = attributes.maxThreadsPerBlock;
  }
  if (active_blocks < *minimum_active_blocks) {
    *minimum_active_blocks = active_blocks;
  }
  if (static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes) <
      *configured_dynamic_shared_limit) {
    *configured_dynamic_shared_limit =
        static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes);
  }
  return cudaSuccess;
}

}  // namespace

int query_sm87_a4w4_attention_k256_m128n256_resources_cuda(
    Sm87A4W4AttentionK256Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4AttentionK256Resources{};
  cudaDeviceProp properties{};
  const int target_status = validate_sm87(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaError_t configure_status = configure_o_kernel();
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }
  cudaError_t status =
      configure_fixed_kernel<
          Sm87A4W4AttentionK256Topology::kLinearQkvZ>();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = configure_fixed_kernel<
      Sm87A4W4AttentionK256Topology::kFullQkv>();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = configure_arbitrary_kernel();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_attention_o_k256_m128n256_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_attention_o_k256_m128n256_kernel,
      static_cast<int>(kSm87A4W4AttentionK256Threads),
      kSm87A4W4AttentionK256SharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int maximum_registers = attributes.numRegs;
  std::size_t maximum_local_bytes = attributes.localSizeBytes;
  int minimum_maximum_threads = attributes.maxThreadsPerBlock;
  int minimum_active_blocks = active_blocks;
  std::size_t configured_dynamic_shared_limit =
      attributes.maxDynamicSharedSizeBytes;
  status = merge_fixed_resources<
      Sm87A4W4AttentionK256Topology::kLinearQkvZ>(
      &maximum_registers, &maximum_local_bytes,
      &minimum_maximum_threads, &minimum_active_blocks,
      &configured_dynamic_shared_limit);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = merge_fixed_resources<
      Sm87A4W4AttentionK256Topology::kFullQkv>(
      &maximum_registers, &maximum_local_bytes,
      &minimum_maximum_threads, &minimum_active_blocks,
      &configured_dynamic_shared_limit);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = merge_arbitrary_resources(
      &maximum_registers, &maximum_local_bytes,
      &minimum_maximum_threads, &minimum_active_blocks,
      &configured_dynamic_shared_limit);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = maximum_registers;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kSm87A4W4AttentionK256SharedBytes;
  resources->configured_dynamic_shared_limit_bytes =
      configured_dynamic_shared_limit;
  resources->device_optin_shared_limit_bytes =
      properties.sharedMemPerBlockOptin;
  resources->local_bytes = maximum_local_bytes;
  resources->maximum_threads_per_block = minimum_maximum_threads;
  resources->active_blocks_per_sm = minimum_active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(kSm87A4W4AttentionK256MaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4AttentionK256SharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4AttentionK256SharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4AttentionK256SharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4AttentionK256Threads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(kSm87A4W4AttentionK256CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4_quantize_bf16_k256_cuda(
    const std::uint16_t* const input_bf16,
    const std::size_t input_row_stride_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t input_size,
    const float clip_ratio,
    std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const a_k256_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  const std::size_t expected_launch_token_count =
      sm87_a4w4_attention_k256_launch_token_count(logical_token_count);
  if (logical_token_count == 0U || expected_launch_token_count == 0U ||
      launch_token_count != expected_launch_token_count ||
      input_size == 0U ||
      input_size % kSm87A4W4AttentionK256ScaleK != 0U ||
      !(clip_ratio > 0.0F && clip_ratio <= 1.0F) ||
      !aligned(input_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k256_scales_bf16, 16U) ||
      input_row_stride_elements < input_size ||
      !sm87_a4w4_attention_k256_product_fits(
          logical_token_count, input_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t k256_groups =
      input_size / kSm87A4W4AttentionK256ScaleK;
  if (!sm87_a4w4_attention_k256_product_fits(
          launch_token_count, k256_groups)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_packed_bytes =
      sm87_a4w4_attention_k256_packed_capacity_bytes(
          launch_token_count, input_size);
  const std::size_t required_scale_elements =
      sm87_a4w4_attention_k256_scale_capacity_elements(
          launch_token_count, input_size);
  if (required_packed_bytes == 0U || required_scale_elements == 0U ||
      packed_a_capacity_bytes < required_packed_bytes ||
      a_scale_capacity_elements < required_scale_elements ||
      !sm87_a4w4_attention_k256_product_fits(
          required_scale_elements, sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t input_elements =
      logical_token_count * input_row_stride_elements;
  const std::size_t scale_bytes =
      required_scale_elements * sizeof(std::uint16_t);
  if (!sm87_a4w4_attention_k256_product_fits(
          input_elements, sizeof(std::uint16_t)) ||
      byte_ranges_overlap(
          input_bf16, input_elements * sizeof(std::uint16_t), packed_a,
          required_packed_bytes) ||
      byte_ranges_overlap(input_bf16,
                          input_elements * sizeof(std::uint16_t),
                          a_k256_scales_bf16, scale_bytes) ||
      byte_ranges_overlap(packed_a, required_packed_bytes,
                          a_k256_scales_bf16, scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t group_count = launch_token_count * k256_groups;
  const std::size_t blocks =
      (group_count + kAttentionK256QuantizeWarps - 1U) /
      kAttentionK256QuantizeWarps;
  if (blocks == 0U ||
      blocks > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int target_status = validate_sm87();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4_quantize_bf16_k256_kernel<<<
      static_cast<unsigned int>(blocks), kAttentionK256QuantizeThreads,
      0U, stream>>>(input_bf16, input_row_stride_elements,
                    logical_token_count, k256_groups, group_count,
                    clip_ratio, packed_a, a_k256_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

namespace {

[[nodiscard]] bool projection_views_valid(
    const std::uint8_t* const packed_a,
    const std::size_t required_a_bytes,
    const std::uint16_t* const a_scales,
    const std::size_t required_a_scale_elements,
    const std::size_t token_count,
    const std::size_t input_size,
    const Sm87A4W4AttentionK256ProjectionView* const projections,
    const std::size_t projection_count) noexcept {
  if (projections == nullptr || projection_count == 0U ||
      projection_count > 3U ||
      !sm87_a4w4_attention_k256_product_fits(
          required_a_scale_elements, sizeof(std::uint16_t))) {
    return false;
  }
  const std::size_t required_a_scale_bytes =
      required_a_scale_elements * sizeof(std::uint16_t);
  for (std::size_t projection = 0U; projection < projection_count;
       ++projection) {
    const auto& view = projections[projection];
    if (view.output_size == 0U ||
        view.output_size % kSm87A4W4AttentionK256PanelN != 0U ||
        !aligned(view.packed_b, 16U) ||
        !aligned(view.k256_scales_bf16, 16U) ||
        !aligned(view.output_bf16, alignof(std::uint32_t)) ||
        view.output_row_stride_elements < view.output_size ||
        view.output_row_stride_elements % 2U != 0U ||
        view.output_row_stride_elements >
            std::numeric_limits<unsigned int>::max() ||
        !sm87_a4w4_attention_k256_product_fits(
            token_count, view.output_row_stride_elements)) {
      return false;
    }
    const std::size_t required_b_bytes =
        sm87_a4w4_attention_k256_packed_capacity_bytes(
            view.output_size, input_size);
    const std::size_t required_b_scale_elements =
        sm87_a4w4_attention_k256_scale_capacity_elements(
            view.output_size, input_size);
    const std::size_t required_output_elements =
        token_count * view.output_row_stride_elements;
    if (required_b_bytes == 0U || required_b_scale_elements == 0U ||
        view.packed_b_capacity_bytes < required_b_bytes ||
        view.scale_capacity_elements < required_b_scale_elements ||
        view.output_capacity_elements < required_output_elements ||
        !sm87_a4w4_attention_k256_product_fits(
            required_b_scale_elements, sizeof(std::uint16_t)) ||
        !sm87_a4w4_attention_k256_product_fits(
            required_output_elements, sizeof(std::uint16_t))) {
      return false;
    }
    const std::size_t output_bytes =
        required_output_elements * sizeof(std::uint16_t);
    if (byte_ranges_overlap(view.output_bf16, output_bytes, packed_a,
                            required_a_bytes) ||
        byte_ranges_overlap(view.output_bf16, output_bytes, a_scales,
                            required_a_scale_bytes)) {
      return false;
    }
    for (std::size_t source = 0U; source < projection_count; ++source) {
      const auto& input_view = projections[source];
      const std::size_t input_b_bytes =
          sm87_a4w4_attention_k256_packed_capacity_bytes(
              input_view.output_size, input_size);
      const std::size_t input_scale_elements =
          sm87_a4w4_attention_k256_scale_capacity_elements(
              input_view.output_size, input_size);
      if (byte_ranges_overlap(view.output_bf16, output_bytes,
                              input_view.packed_b, input_b_bytes) ||
          byte_ranges_overlap(
              view.output_bf16, output_bytes,
              input_view.k256_scales_bf16,
              input_scale_elements * sizeof(std::uint16_t))) {
        return false;
      }
    }
    for (std::size_t other = 0U; other < projection; ++other) {
      const auto& prior = projections[other];
      const std::size_t prior_output_elements =
          token_count * prior.output_row_stride_elements;
      if (byte_ranges_overlap(
              view.output_bf16, output_bytes, prior.output_bf16,
              prior_output_elements * sizeof(std::uint16_t))) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] int validate_attention_common(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const Sm87A4W4AttentionK256Plan& plan,
    const Sm87A4W4AttentionK256ProjectionView* const projections,
    const std::size_t projection_count) noexcept {
  if (plan.launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k256_scales_bf16, 16U) ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_cells > std::numeric_limits<unsigned int>::max() ||
      plan.k256_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_a_bytes =
      sm87_a4w4_attention_k256_packed_capacity_bytes(
          plan.token_count, plan.input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_attention_k256_scale_capacity_elements(
          plan.token_count, plan.input_size);
  if (required_a_bytes == 0U || required_a_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      !projection_views_valid(packed_a, required_a_bytes,
                              a_k256_scales_bf16, required_a_scales,
                              plan.token_count, plan.input_size,
                              projections, projection_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int launch_sm87_a4w4_attention_k256_m128n256_bf16_cuda(
    const Sm87A4W4AttentionK256Topology topology,
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::size_t token_count,
    const Sm87A4W4AttentionK256ProjectionView* const projections,
    const std::size_t projection_count,
    void* const cuda_stream) noexcept {
  const Sm87A4W4AttentionK256Plan plan =
      sm87_a4w4_attention_k256_fixed_plan(topology, token_count);
  if (projection_count !=
      sm87_a4w4_attention_k256_fixed_projection_count(topology)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  for (std::size_t projection = 0U; projection < projection_count;
       ++projection) {
    if (projections == nullptr ||
        projections[projection].output_size !=
            sm87_a4w4_attention_k256_fixed_projection_panels(
                topology, projection) *
                kSm87A4W4AttentionK256PanelN) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  const int common_status = validate_attention_common(
      packed_a, packed_a_capacity_bytes, a_k256_scales_bf16,
      a_scale_capacity_elements, plan, projections, projection_count);
  if (common_status != static_cast<int>(cudaSuccess)) {
    return common_status;
  }
  const int target_status = validate_sm87();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  cudaError_t configure_status = cudaSuccess;
  if (topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
    configure_status = configure_fixed_kernel<
        Sm87A4W4AttentionK256Topology::kLinearQkvZ>();
  } else if (topology == Sm87A4W4AttentionK256Topology::kFullQkv) {
    configure_status = configure_fixed_kernel<
        Sm87A4W4AttentionK256Topology::kFullQkv>();
  } else {
    configure_status = configure_o_kernel();
  }
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const unsigned int launch_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  (void)cudaGetLastError();
  if (topology == Sm87A4W4AttentionK256Topology::kAttentionO) {
    const auto& o = projections[0U];
    q3x_sm87_a4w4_attention_o_k256_m128n256_kernel<<<
        launch_ctas, kSm87A4W4AttentionK256Threads,
        kSm87A4W4AttentionK256SharedBytes, stream>>>(
        packed_a, a_k256_scales_bf16, o.packed_b,
        o.k256_scales_bf16,
        static_cast<unsigned int>(plan.k256_groups),
        static_cast<unsigned int>(plan.physical_k64_groups),
        o.output_bf16,
        static_cast<unsigned int>(o.output_row_stride_elements),
        static_cast<unsigned int>(plan.m_tiles),
        static_cast<unsigned int>(plan.work_cells));
  } else {
    const auto& p0 = projections[0U];
    const auto& p1 = projections[1U];
    const Sm87A4W4AttentionK256ProjectionView empty{};
    const auto& p2 = projection_count > 2U ? projections[2U] : empty;
#define Q3X_LAUNCH_FIXED(topology_value)                                  \
  q3x_sm87_a4w4_attention_k256_m128n256_fixed_kernel<topology_value>     \
      <<<launch_ctas, kSm87A4W4AttentionK256Threads,                     \
         kSm87A4W4AttentionK256SharedBytes, stream>>>(                   \
          packed_a, a_k256_scales_bf16, p0.packed_b,                    \
          p0.k256_scales_bf16, p1.packed_b, p1.k256_scales_bf16,        \
          p2.packed_b, p2.k256_scales_bf16,                             \
          static_cast<unsigned int>(plan.k256_groups),                  \
          static_cast<unsigned int>(plan.physical_k64_groups),          \
          p0.output_bf16,                                               \
          static_cast<unsigned int>(p0.output_row_stride_elements),     \
          p1.output_bf16,                                               \
          static_cast<unsigned int>(p1.output_row_stride_elements),     \
          p2.output_bf16,                                               \
          static_cast<unsigned int>(p2.output_row_stride_elements),     \
          static_cast<unsigned int>(plan.m_tiles),                      \
          static_cast<unsigned int>(plan.work_cells))
    if (topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
      Q3X_LAUNCH_FIXED(
          Sm87A4W4AttentionK256Topology::kLinearQkvZ);
    } else {
      Q3X_LAUNCH_FIXED(Sm87A4W4AttentionK256Topology::kFullQkv);
    }
#undef Q3X_LAUNCH_FIXED
  }
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_a4w4_attention_k256_m128n256_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t input_size,
    const Sm87A4W4AttentionK256ProjectionView* const projections,
    const std::size_t projection_count,
    const Sm87A4W4AttentionK256PanelDescriptor* const panels,
    const std::size_t macro_cells,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  const Sm87A4W4AttentionK256Plan plan =
      sm87_a4w4_attention_k256_test_plan(token_count, input_size,
                                         macro_cells);
  const int common_status = validate_attention_common(
      packed_a, packed_a_capacity_bytes, a_k256_scales_bf16,
      a_scale_capacity_elements, plan, projections, projection_count);
  if (common_status != static_cast<int>(cudaSuccess) || panels == nullptr ||
      maximum_launch_ctas == 0U ||
      !sm87_a4w4_attention_k256_product_fits(
          macro_cells, kSm87A4W4AttentionK256PanelsPerCell)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t panel_count =
      macro_cells * kSm87A4W4AttentionK256PanelsPerCell;
  std::vector<DevicePanelRecord> host_records(panel_count);
  for (std::size_t index = 0U; index < panel_count; ++index) {
    const auto descriptor = panels[index];
    if (descriptor.projection >= projection_count ||
        descriptor.panel >=
            projections[descriptor.projection].output_size /
                kSm87A4W4AttentionK256PanelN ||
        descriptor.panel > kDescriptorPanelMask) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (panels[prior].projection == descriptor.projection &&
          panels[prior].panel == descriptor.panel) {
        return static_cast<int>(cudaErrorInvalidValue);
      }
    }
    const auto& projection = projections[descriptor.projection];
    host_records[index] = {
        projection.packed_b,
        projection.k256_scales_bf16,
        projection.output_bf16,
        static_cast<unsigned int>(
            projection.output_row_stride_elements),
        static_cast<unsigned int>(descriptor.panel)};
  }
  const int target_status = validate_sm87();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaError_t configure_status = configure_arbitrary_kernel();
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }
  DevicePanelRecord* device_records = nullptr;
  cudaError_t status = cudaMalloc(
      reinterpret_cast<void**>(&device_records),
      panel_count * sizeof(DevicePanelRecord));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  status = cudaMemcpyAsync(device_records, host_records.data(),
                           panel_count * sizeof(DevicePanelRecord),
                           cudaMemcpyHostToDevice, stream);
  if (status == cudaSuccess) {
    const unsigned int plan_ctas =
        static_cast<unsigned int>(plan.launch_ctas);
    const unsigned int launch_ctas =
        plan_ctas < maximum_launch_ctas ? plan_ctas
                                        : maximum_launch_ctas;
    q3x_sm87_a4w4_attention_k256_m128n256_arbitrary_kernel<<<
        launch_ctas, kSm87A4W4AttentionK256Threads,
        kSm87A4W4AttentionK256SharedBytes, stream>>>(
        packed_a, a_k256_scales_bf16, device_records,
        static_cast<unsigned int>(plan.k256_groups),
        static_cast<unsigned int>(plan.physical_k64_groups),
        static_cast<unsigned int>(plan.m_tiles),
        static_cast<unsigned int>(plan.work_cells));
    status = cudaPeekAtLastError();
  }
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(stream);
  }
  const cudaError_t free_status = cudaFree(device_records);
  return static_cast<int>(status != cudaSuccess ? status : free_status);
}

}  // namespace q3x::kernels
