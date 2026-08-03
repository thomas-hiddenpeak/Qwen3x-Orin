#include "q3x/kernels/sm87_a4w4_attention_k256_m128n128_a_exchange_b3.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kTileM = 128U;
inline constexpr unsigned int kTileN = 128U;
inline constexpr unsigned int kWarpTileM = 32U;
inline constexpr unsigned int kWarpTileN = 64U;
inline constexpr unsigned int kK64PerGroup = 4U;
inline constexpr unsigned int kPackedK64Bytes = 32U;
inline constexpr unsigned int kBSlots = 3U;
inline constexpr unsigned int kDescriptorProjectionShift = 24U;
inline constexpr unsigned int kDescriptorPanelMask = 0x00ff'ffffU;
inline constexpr int kRequiredSmCount = 16;

struct alignas(16) AExchangeSlot final {
  std::uint8_t code[kK64PerGroup][kTileM * kPackedK64Bytes];
  std::uint16_t scale[kTileM];
};

struct alignas(16) BRingSlot final {
  std::uint8_t code[kK64PerGroup][kTileN * kPackedK64Bytes];
  std::uint16_t scale[kTileN];
};

struct alignas(16) SharedStorage final {
  AExchangeSlot a;
  BRingSlot b[kBSlots];
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

// All four ordered K64 A fragments are retained while a K256 group consumes
// the eight N8 fragments.  The object is 36 registers/thread; paired with the
// 64-register M32N64 accumulator this leaves enough headroom for B fragments,
// addresses, and scale products under the 128-register occupancy boundary.
struct ResidentA final {
  Sm87A4W4AFragment p0_m0;
  Sm87A4W4AFragment p0_m1;
  Sm87A4W4AFragment p1_m0;
  Sm87A4W4AFragment p1_m1;
  Sm87A4W4AFragment p2_m0;
  Sm87A4W4AFragment p2_m1;
  Sm87A4W4AFragment p3_m0;
  Sm87A4W4AFragment p3_m1;
  float m0_scale0{};
  float m0_scale1{};
  float m1_scale0{};
  float m1_scale1{};
};

static_assert(sizeof(AExchangeSlot) == 16'640U);
static_assert(sizeof(BRingSlot) == 16'640U);
static_assert(sizeof(SharedStorage) == 66'560U);
static_assert(sizeof(Float4) == 16U);
static_assert(sizeof(ResidentA) == 144U);
static_assert(8U * kWarpTileM * kWarpTileN == kTileM * kTileN);

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

template <Sm87A4W4AttentionK256Topology Topology>
[[nodiscard]] __device__ __forceinline__ unsigned int
incumbent_descriptor(const unsigned int cell,
                     const unsigned int slot) noexcept {
  static_assert(Topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ ||
                Topology == Sm87A4W4AttentionK256Topology::kFullQkv ||
                Topology == Sm87A4W4AttentionK256Topology::kAttentionO);
  if constexpr (Topology ==
                Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
    if (cell < 48U) {
      return slot < 2U
                 ? 2U * cell + slot
                 : (1U << kDescriptorProjectionShift) |
                       (2U * cell + slot - 2U);
    }
    return 96U + 4U * (cell - 48U) + slot;
  } else if constexpr (Topology ==
                       Sm87A4W4AttentionK256Topology::kFullQkv) {
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
  } else {
    return 4U * cell + slot;
  }
}

// Two consecutive N128 cells reproduce one canonical N256 cell exactly.
// This retains the current Qwen3.6 projection interleave without introducing
// a descriptor table or generic-shape branches into the resource mirror.
template <Sm87A4W4AttentionK256Topology Topology>
[[nodiscard]] __device__ __forceinline__ unsigned int fixed_descriptor(
    const unsigned int cell, const unsigned int slot) noexcept {
  const unsigned int incumbent_cell = cell >> 1U;
  const unsigned int incumbent_slot = 2U * (cell & 1U) + slot;
  return incumbent_descriptor<Topology>(incumbent_cell, incumbent_slot);
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

__device__ __forceinline__ void issue_a_exchange(
    AExchangeSlot& slot,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k256_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int k256_group,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  const unsigned int row = threadIdx.x >> 1U;
  const unsigned int row_vector = threadIdx.x & 1U;
  const std::size_t shared_offset =
      sm87_a4w4_swizzled_k64_byte_offset(row, 16U * row_vector);
#pragma unroll
  for (unsigned int plane = 0U; plane < kK64PerGroup; ++plane) {
    const unsigned int k64 = k256_group * kK64PerGroup + plane;
    cp_async_16(
        slot.code[plane] + shared_offset,
        packed_a + sm87_a4w4_attention_k256_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       k64, 16U * row_vector,
                       physical_k64_group_count));
  }
  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        slot.scale + first_row,
        a_k256_scales_bf16 +
            sm87_a4w4_attention_k256_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k256_group, k256_group_count));
  }
  cp_async_commit();
}

// Two 128-thread cohorts load one N64 panel each.  Every thread contributes
// one 16-byte code vector per ordered K64 plane; sixteen lanes publish both
// panels' complete K256 scale vectors.
template <Sm87A4W4AttentionK256Topology Topology>
__device__ __forceinline__ void issue_b_ring_slot(
    BRingSlot& slot,
    const std::uint8_t* const packed_b0,
    const std::uint16_t* const b_scales0,
    const std::uint8_t* const packed_b1,
    const std::uint16_t* const b_scales1,
    const std::uint8_t* const packed_b2,
    const std::uint16_t* const b_scales2,
    const unsigned int macro_cell,
    const unsigned int k256_group,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  const unsigned int panel_slot = threadIdx.x >> 7U;
  const unsigned int thread_in_panel = threadIdx.x & 127U;
  const unsigned int descriptor =
      fixed_descriptor<Topology>(macro_cell, panel_slot);
  const unsigned int projection =
      descriptor >> kDescriptorProjectionShift;
  const unsigned int panel = descriptor & kDescriptorPanelMask;
  const std::uint8_t* const selected_b = select_packed_projection(
      projection, packed_b0, packed_b1, packed_b2);
  const unsigned int panel_row = thread_in_panel >> 1U;
  const unsigned int row_vector = thread_in_panel & 1U;
  const unsigned int shared_row = panel_slot * 64U + panel_row;
#pragma unroll
  for (unsigned int plane = 0U; plane < kK64PerGroup; ++plane) {
    const unsigned int k64 = k256_group * kK64PerGroup + plane;
    cp_async_16(
        slot.code[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                               shared_row, 16U * row_vector),
        selected_b + sm87_a4w4_attention_k256_packed_offset(
                         static_cast<std::size_t>(panel) * 64U + panel_row,
                         k64, 16U * row_vector,
                         physical_k64_group_count));
  }
  if (threadIdx.x < 16U) {
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
    const unsigned int row_vector8 = threadIdx.x & 7U;
    cp_async_16(
        slot.scale + 64U * scale_slot + 8U * row_vector8,
        selected_scales + sm87_a4w4_attention_k256_scale_offset(
                              static_cast<std::size_t>(scale_panel) * 64U +
                                  8U * row_vector8,
                              k256_group, k256_group_count));
  }
  cp_async_commit();
}

[[nodiscard]] __device__ __forceinline__ ResidentA load_resident_a(
    const AExchangeSlot& slot) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m32 = warp & 3U;
  const unsigned int local_m = warp_m32 * kWarpTileM;
  ResidentA resident{};
  resident.p0_m0 = load_a_ldmatrix_x4(
      slot.code[0U] + local_m * kPackedK64Bytes, lane);
  resident.p0_m1 = load_a_ldmatrix_x4(
      slot.code[0U] + (local_m + 16U) * kPackedK64Bytes, lane);
  resident.p1_m0 = load_a_ldmatrix_x4(
      slot.code[1U] + local_m * kPackedK64Bytes, lane);
  resident.p1_m1 = load_a_ldmatrix_x4(
      slot.code[1U] + (local_m + 16U) * kPackedK64Bytes, lane);
  resident.p2_m0 = load_a_ldmatrix_x4(
      slot.code[2U] + local_m * kPackedK64Bytes, lane);
  resident.p2_m1 = load_a_ldmatrix_x4(
      slot.code[2U] + (local_m + 16U) * kPackedK64Bytes, lane);
  resident.p3_m0 = load_a_ldmatrix_x4(
      slot.code[3U] + local_m * kPackedK64Bytes, lane);
  resident.p3_m1 = load_a_ldmatrix_x4(
      slot.code[3U] + (local_m + 16U) * kPackedK64Bytes, lane);

  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  resident.m0_scale0 =
      decode_bf16(slot.scale[local_m + coordinate0.m]);
  resident.m0_scale1 =
      decode_bf16(slot.scale[local_m + coordinate2.m]);
  resident.m1_scale0 =
      decode_bf16(slot.scale[local_m + 16U + coordinate0.m]);
  resident.m1_scale1 =
      decode_bf16(slot.scale[local_m + 16U + coordinate2.m]);
  return resident;
}

template <unsigned int FragmentN>
__device__ __forceinline__ void accumulate_n8(
    const ResidentA& a,
    const BRingSlot& b_slot,
    Float4& output_m0,
    Float4& output_m1) noexcept {
  static_assert(FragmentN < 8U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int n64_half = warp >> 2U;
  constexpr unsigned int kFragmentOffset = FragmentN * 8U;
  const unsigned int local_n = n64_half * kWarpTileN + kFragmentOffset;
  Sm87A4W4Accumulator partial_m0{};
  Sm87A4W4Accumulator partial_m1{};

  Sm87A4W4BFragment b = load_b_ldmatrix_x2(
      b_slot.code[0U] + local_n * kPackedK64Bytes, lane);
  sm87_a4w4_mma_m16n8k64(partial_m0, a.p0_m0, b);
  sm87_a4w4_mma_m16n8k64(partial_m1, a.p0_m1, b);
  b = load_b_ldmatrix_x2(
      b_slot.code[1U] + local_n * kPackedK64Bytes, lane);
  sm87_a4w4_mma_m16n8k64(partial_m0, a.p1_m0, b);
  sm87_a4w4_mma_m16n8k64(partial_m1, a.p1_m1, b);
  b = load_b_ldmatrix_x2(
      b_slot.code[2U] + local_n * kPackedK64Bytes, lane);
  sm87_a4w4_mma_m16n8k64(partial_m0, a.p2_m0, b);
  sm87_a4w4_mma_m16n8k64(partial_m1, a.p2_m1, b);
  b = load_b_ldmatrix_x2(
      b_slot.code[3U] + local_n * kPackedK64Bytes, lane);
  sm87_a4w4_mma_m16n8k64(partial_m0, a.p3_m0, b);
  sm87_a4w4_mma_m16n8k64(partial_m1, a.p3_m1, b);

  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const float b_scale0 =
      decode_bf16(b_slot.scale[local_n + coordinate0.n]);
  const float b_scale1 =
      decode_bf16(b_slot.scale[local_n + coordinate1.n]);

#define Q3X_ACCUMULATE_BAND(output, partial, scale0, scale1)              \
  do {                                                                    \
    (output).x0 = __fmaf_rn(static_cast<float>((partial).x0),             \
                            __fmul_rn((scale0), b_scale0), (output).x0);   \
    (output).x1 = __fmaf_rn(static_cast<float>((partial).x1),             \
                            __fmul_rn((scale0), b_scale1), (output).x1);   \
    (output).x2 = __fmaf_rn(static_cast<float>((partial).x2),             \
                            __fmul_rn((scale1), b_scale0), (output).x2);   \
    (output).x3 = __fmaf_rn(static_cast<float>((partial).x3),             \
                            __fmul_rn((scale1), b_scale1), (output).x3);   \
  } while (false)
  Q3X_ACCUMULATE_BAND(output_m0, partial_m0,
                      a.m0_scale0, a.m0_scale1);
  Q3X_ACCUMULATE_BAND(output_m1, partial_m1,
                      a.m1_scale0, a.m1_scale1);
#undef Q3X_ACCUMULATE_BAND
}

__device__ __forceinline__ void accumulate_group(
    const ResidentA& a,
    const BRingSlot& b,
    Float4 (&output_m0)[8U],
    Float4 (&output_m1)[8U]) noexcept {
#define Q3X_ACCUMULATE_N8(fragment)                                      \
  accumulate_n8<fragment>(a, b, output_m0[fragment], output_m1[fragment])
  Q3X_ACCUMULATE_N8(0U);
  Q3X_ACCUMULATE_N8(1U);
  Q3X_ACCUMULATE_N8(2U);
  Q3X_ACCUMULATE_N8(3U);
  Q3X_ACCUMULATE_N8(4U);
  Q3X_ACCUMULATE_N8(5U);
  Q3X_ACCUMULATE_N8(6U);
  Q3X_ACCUMULATE_N8(7U);
#undef Q3X_ACCUMULATE_N8
}

template <unsigned int FragmentN,
          Sm87A4W4AttentionK256Topology Topology>
__device__ __forceinline__ void store_n8(
    const Float4& value_m0,
    const Float4& value_m1,
    const unsigned int macro_cell,
    const unsigned int m_tile_start,
    std::uint16_t* const output0,
    const unsigned int stride0,
    std::uint16_t* const output1,
    const unsigned int stride1,
    std::uint16_t* const output2,
    const unsigned int stride2) noexcept {
  static_assert(FragmentN < 8U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m32 = warp & 3U;
  const unsigned int n64_half = warp >> 2U;
  const unsigned int descriptor =
      fixed_descriptor<Topology>(macro_cell, n64_half);
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
      panel * 64U + FragmentN * 8U + coordinate0.n;
  const unsigned int global_m0 =
      m_tile_start + warp_m32 * kWarpTileM + coordinate0.m;
  const unsigned int global_m1 =
      m_tile_start + warp_m32 * kWarpTileM + coordinate2.m;

#define Q3X_PACK_PAIR(value, first, second)                              \
  (static_cast<std::uint32_t>(encode_bf16((value).first)) |              \
   (static_cast<std::uint32_t>(encode_bf16((value).second)) << 16U))
  const std::uint32_t m0_pair0 = Q3X_PACK_PAIR(value_m0, x0, x1);
  const std::uint32_t m0_pair1 = Q3X_PACK_PAIR(value_m0, x2, x3);
  const std::uint32_t m1_pair0 = Q3X_PACK_PAIR(value_m1, x0, x1);
  const std::uint32_t m1_pair1 = Q3X_PACK_PAIR(value_m1, x2, x3);
#undef Q3X_PACK_PAIR
  *reinterpret_cast<std::uint32_t*>(
      selected_output + static_cast<std::size_t>(global_m0) * stride +
      global_n) = m0_pair0;
  *reinterpret_cast<std::uint32_t*>(
      selected_output + static_cast<std::size_t>(global_m1) * stride +
      global_n) = m0_pair1;
  *reinterpret_cast<std::uint32_t*>(
      selected_output +
      static_cast<std::size_t>(global_m0 + 16U) * stride + global_n) =
      m1_pair0;
  *reinterpret_cast<std::uint32_t*>(
      selected_output +
      static_cast<std::size_t>(global_m1 + 16U) * stride + global_n) =
      m1_pair1;
}

template <Sm87A4W4AttentionK256Topology Topology>
__device__ __forceinline__ void store_cell(
    const Float4 (&output_m0)[8U],
    const Float4 (&output_m1)[8U],
    const unsigned int macro_cell,
    const unsigned int m_tile_start,
    std::uint16_t* const output0,
    const unsigned int stride0,
    std::uint16_t* const output1,
    const unsigned int stride1,
    std::uint16_t* const output2,
    const unsigned int stride2) noexcept {
#define Q3X_STORE_N8(fragment)                                           \
  store_n8<fragment, Topology>(                                         \
      output_m0[fragment], output_m1[fragment], macro_cell,              \
      m_tile_start, output0, stride0, output1, stride1, output2, stride2)
  Q3X_STORE_N8(0U);
  Q3X_STORE_N8(1U);
  Q3X_STORE_N8(2U);
  Q3X_STORE_N8(3U);
  Q3X_STORE_N8(4U);
  Q3X_STORE_N8(5U);
  Q3X_STORE_N8(6U);
  Q3X_STORE_N8(7U);
#undef Q3X_STORE_N8
}

template <Sm87A4W4AttentionK256Topology Topology>
__device__ __forceinline__ void run_fixed_cell_kernel(
    SharedStorage& shared,
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
    const unsigned int work_cell_count) noexcept {
  for (unsigned int work = blockIdx.x; work < work_cell_count;
       work += gridDim.x) {
    const unsigned int macro_cell = work / m_tile_count;
    const unsigned int m_tile = work - macro_cell * m_tile_count;
    const unsigned int m_tile_start = m_tile * kTileM;
    Float4 output_m0[8U]{};
    Float4 output_m1[8U]{};

    // A0 is oldest, followed by B0..B{S-1}.  Waiting for S-1 groups retires
    // A0 and B0 while preserving all future B stages.  The steady state
    // publishes A_next before recycling the consumed B slot, then leaves only
    // the newest future B in flight.  Thus A never occupies a ring stage and
    // its lifetime ends immediately after the register handoff.
    issue_a_exchange(shared.a, packed_a, a_k256_scales_bf16,
                     m_tile_start, 0U, k256_group_count,
                     physical_k64_group_count);
    issue_b_ring_slot<Topology>(
        shared.b[0U], packed_b0, b_scales0, packed_b1, b_scales1,
        packed_b2, b_scales2, macro_cell, 0U, k256_group_count,
        physical_k64_group_count);
    if (k256_group_count > 1U) {
      issue_b_ring_slot<Topology>(
          shared.b[1U], packed_b0, b_scales0, packed_b1, b_scales1,
          packed_b2, b_scales2, macro_cell, 1U, k256_group_count,
          physical_k64_group_count);
    }
    if (k256_group_count > 2U) {
      issue_b_ring_slot<Topology>(
          shared.b[2U], packed_b0, b_scales0, packed_b1, b_scales1,
          packed_b2, b_scales2, macro_cell, 2U, k256_group_count,
          physical_k64_group_count);
    }

    if (k256_group_count > 2U) {
      cp_async_wait<2U>();
    } else if (k256_group_count > 1U) {
      cp_async_wait<1U>();
    } else {
      cp_async_wait<0U>();
    }
    __syncthreads();

    for (unsigned int group = 0U; group < k256_group_count; ++group) {
      const ResidentA resident = load_resident_a(shared.a);
      __syncthreads();  // release A immediately after the register handoff

      if (group + 1U < k256_group_count) {
        issue_a_exchange(shared.a, packed_a, a_k256_scales_bf16,
                         m_tile_start, group + 1U, k256_group_count,
                         physical_k64_group_count);
      }

      accumulate_group(resident, shared.b[group % kBSlots],
                       output_m0, output_m1);
      __syncthreads();  // release the consumed B stage

      if (group + kBSlots < k256_group_count) {
        issue_b_ring_slot<Topology>(
            shared.b[group % kBSlots], packed_b0, b_scales0, packed_b1,
            b_scales1, packed_b2, b_scales2, macro_cell,
            group + kBSlots, k256_group_count,
            physical_k64_group_count);
        cp_async_wait<1U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();
    }

    store_cell<Topology>(output_m0, output_m1, macro_cell, m_tile_start,
                         output0, stride0, output1, stride1, output2,
                         stride2);
    __syncthreads();
  }
}

}  // namespace

#define Q3X_DEFINE_B3_KERNEL(TOPOLOGY, TOPOLOGY_NAME)                    \
  extern "C" __global__                                                  \
      __launch_bounds__(kSm87A4W4AttentionK256M128N128AExchangeThreads,   \
                        kSm87A4W4AttentionK256M128N128RequiredCtasPerSm)   \
  void q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_##             \
      TOPOLOGY_NAME##_kernel(                                            \
          const std::uint8_t* const packed_a,                            \
          const std::uint16_t* const a_k256_scales_bf16,                 \
          const std::uint8_t* const packed_b0,                           \
          const std::uint16_t* const b_scales0,                          \
          const std::uint8_t* const packed_b1,                           \
          const std::uint16_t* const b_scales1,                          \
          const std::uint8_t* const packed_b2,                           \
          const std::uint16_t* const b_scales2,                          \
          const unsigned int k256_group_count,                           \
          const unsigned int physical_k64_group_count,                   \
          std::uint16_t* const output0, const unsigned int stride0,      \
          std::uint16_t* const output1, const unsigned int stride1,      \
          std::uint16_t* const output2, const unsigned int stride2,      \
          const unsigned int m_tile_count,                               \
          const unsigned int work_cell_count) {                          \
    extern __shared__ __align__(16) unsigned char dynamic_storage[];     \
    auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_storage);   \
    run_fixed_cell_kernel<TOPOLOGY>(                                     \
        shared, packed_a, a_k256_scales_bf16, packed_b0, b_scales0,      \
        packed_b1, b_scales1, packed_b2, b_scales2,                      \
        k256_group_count, physical_k64_group_count, output0, stride0,    \
        output1, stride1, output2, stride2, m_tile_count,                 \
        work_cell_count);                                                \
  }

Q3X_DEFINE_B3_KERNEL(
    Sm87A4W4AttentionK256Topology::kLinearQkvZ, linear)
Q3X_DEFINE_B3_KERNEL(
    Sm87A4W4AttentionK256Topology::kFullQkv, full)
Q3X_DEFINE_B3_KERNEL(
    Sm87A4W4AttentionK256Topology::kAttentionO, o)

#undef Q3X_DEFINE_B3_KERNEL

namespace {

template <class Kernel>
[[nodiscard]] cudaError_t configure_resource_kernel(
    const Kernel kernel, const std::size_t dynamic_shared_bytes) noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_shared_bytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

struct ResourceAccumulator final {
  int maximum_registers{};
  std::size_t maximum_static_shared{};
  std::size_t maximum_local{};
  int minimum_maximum_threads{std::numeric_limits<int>::max()};
  int minimum_active_blocks{std::numeric_limits<int>::max()};
};

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

[[nodiscard]] cudaError_t configure_b3_topology(
    const Sm87A4W4AttentionK256Topology topology) noexcept {
  if (topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
    return configure_resource_kernel(
        q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_linear_kernel,
        kSm87A4W4AttentionK256M128N128B3SharedBytes);
  }
  if (topology == Sm87A4W4AttentionK256Topology::kFullQkv) {
    return configure_resource_kernel(
        q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_full_kernel,
        kSm87A4W4AttentionK256M128N128B3SharedBytes);
  }
  if (topology == Sm87A4W4AttentionK256Topology::kAttentionO) {
    return configure_resource_kernel(
        q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_o_kernel,
        kSm87A4W4AttentionK256M128N128B3SharedBytes);
  }
  return cudaErrorInvalidValue;
}

[[nodiscard]] bool b3_projection_views_valid(
    const std::uint8_t* const packed_a,
    const std::size_t required_a_bytes,
    const std::uint16_t* const a_scales,
    const std::size_t required_a_scale_elements,
    const Sm87A4W4AttentionK256Topology topology,
    const std::size_t token_count,
    const std::size_t input_size,
    const Sm87A4W4AttentionK256ProjectionView* const projections,
    const std::size_t projection_count) noexcept {
  const std::size_t required_count =
      sm87_a4w4_attention_k256_fixed_projection_count(topology);
  if (required_count == 0U || projection_count != required_count ||
      projections == nullptr ||
      !sm87_a4w4_attention_k256_product_fits(
          required_a_scale_elements, sizeof(std::uint16_t))) {
    return false;
  }
  const std::size_t required_a_scale_bytes =
      required_a_scale_elements * sizeof(std::uint16_t);
  for (std::size_t projection = 0U; projection < projection_count;
       ++projection) {
    const auto& view = projections[projection];
    const std::size_t output_size =
        sm87_a4w4_attention_k256_fixed_projection_panels(
            topology, projection) *
        kSm87A4W4AttentionK256PanelN;
    const std::size_t required_b_bytes =
        sm87_a4w4_attention_k256_packed_capacity_bytes(
            output_size, input_size);
    const std::size_t required_scale_elements =
        sm87_a4w4_attention_k256_scale_capacity_elements(
            output_size, input_size);
    if (output_size == 0U || required_b_bytes == 0U ||
        required_scale_elements == 0U || view.output_size != output_size ||
        !aligned(view.packed_b, 16U) ||
        view.packed_b_capacity_bytes < required_b_bytes ||
        !aligned(view.k256_scales_bf16, 16U) ||
        view.scale_capacity_elements < required_scale_elements ||
        !aligned(view.output_bf16, alignof(std::uint32_t)) ||
        view.output_row_stride_elements < output_size ||
        view.output_row_stride_elements % 2U != 0U ||
        view.output_row_stride_elements >
            std::numeric_limits<unsigned int>::max() ||
        !sm87_a4w4_attention_k256_product_fits(
            token_count, view.output_row_stride_elements) ||
        !sm87_a4w4_attention_k256_product_fits(
            required_scale_elements, sizeof(std::uint16_t))) {
      return false;
    }
    const std::size_t required_output_elements =
        token_count * view.output_row_stride_elements;
    if (view.output_capacity_elements < required_output_elements ||
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
      const std::size_t source_output_size =
          sm87_a4w4_attention_k256_fixed_projection_panels(
              topology, source) *
          kSm87A4W4AttentionK256PanelN;
      const std::size_t source_b_bytes =
          sm87_a4w4_attention_k256_packed_capacity_bytes(
              source_output_size, input_size);
      const std::size_t source_scale_elements =
          sm87_a4w4_attention_k256_scale_capacity_elements(
              source_output_size, input_size);
      if (!sm87_a4w4_attention_k256_product_fits(
              source_scale_elements, sizeof(std::uint16_t)) ||
          byte_ranges_overlap(view.output_bf16, output_bytes,
                              input_view.packed_b, source_b_bytes) ||
          byte_ranges_overlap(view.output_bf16, output_bytes,
                              input_view.k256_scales_bf16,
                              source_scale_elements *
                                  sizeof(std::uint16_t))) {
        return false;
      }
    }
    for (std::size_t other = 0U; other < projection; ++other) {
      const auto& prior = projections[other];
      if (!sm87_a4w4_attention_k256_product_fits(
              token_count, prior.output_row_stride_elements)) {
        return false;
      }
      const std::size_t prior_elements =
          token_count * prior.output_row_stride_elements;
      if (!sm87_a4w4_attention_k256_product_fits(
              prior_elements, sizeof(std::uint16_t)) ||
          byte_ranges_overlap(view.output_bf16, output_bytes,
                              prior.output_bf16,
                              prior_elements * sizeof(std::uint16_t))) {
        return false;
      }
    }
  }
  return true;
}

struct B3ResourceCache final {
  int device{-2};
  int status{static_cast<int>(cudaErrorUnknown)};
};

thread_local B3ResourceCache g_b3_resource_cache{};

[[nodiscard]] int ensure_b3_resources_cached_for_current_device() noexcept {
  int device = -1;
  const cudaError_t device_status = cudaGetDevice(&device);
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  if (g_b3_resource_cache.device == device) {
    return g_b3_resource_cache.status;
  }
  Sm87A4W4AttentionK256M128N128Resources resources{};
  int status =
      query_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_resources_cuda(
          &resources);
  if (status == static_cast<int>(cudaSuccess) &&
      (resources.registers_per_thread <= 0 ||
       resources.registers_per_thread >
           static_cast<int>(
               kSm87A4W4AttentionK256M128N128MaximumRegisters) ||
       resources.static_shared_bytes != 0U ||
       resources.dynamic_shared_bytes !=
           kSm87A4W4AttentionK256M128N128B3SharedBytes ||
       resources.local_bytes != 0U ||
       resources.active_blocks_per_sm !=
           static_cast<int>(
               kSm87A4W4AttentionK256M128N128RequiredCtasPerSm) ||
       resources.maximum_threads_per_block <
           static_cast<int>(
               kSm87A4W4AttentionK256M128N128AExchangeThreads))) {
    status = static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  g_b3_resource_cache = {device, status};
  return status;
}

template <class Kernel>
[[nodiscard]] cudaError_t merge_resource_kernel(
    const Kernel kernel,
    const std::size_t dynamic_shared_bytes,
    ResourceAccumulator* const accumulator) noexcept {
  cudaError_t status =
      configure_resource_kernel(kernel, dynamic_shared_bytes);
  if (status != cudaSuccess) {
    return status;
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel,
      static_cast<int>(
          kSm87A4W4AttentionK256M128N128AExchangeThreads),
      dynamic_shared_bytes);
  if (status != cudaSuccess) {
    return status;
  }
  accumulator->maximum_registers =
      std::max(accumulator->maximum_registers, attributes.numRegs);
  accumulator->maximum_static_shared =
      std::max(accumulator->maximum_static_shared,
               attributes.sharedSizeBytes);
  accumulator->maximum_local =
      std::max(accumulator->maximum_local, attributes.localSizeBytes);
  accumulator->minimum_maximum_threads =
      std::min(accumulator->minimum_maximum_threads,
               attributes.maxThreadsPerBlock);
  accumulator->minimum_active_blocks =
      std::min(accumulator->minimum_active_blocks, active_blocks);
  return cudaSuccess;
}

}  // namespace

int query_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_resources_cuda(
    Sm87A4W4AttentionK256M128N128Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4AttentionK256M128N128Resources{};
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
  resources->device_compute_major = properties.major;
  resources->device_compute_minor = properties.minor;
  resources->device_multiprocessor_count = properties.multiProcessorCount;
  if (properties.major != kSm87A4W4RequiredComputeMajor ||
      properties.minor != kSm87A4W4RequiredComputeMinor ||
      properties.multiProcessorCount != kRequiredSmCount ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4AttentionK256M128N128B3SharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  ResourceAccumulator b3{};
#define Q3X_MERGE_RESOURCE(KERNEL, BYTES, ACCUMULATOR)                  \
  do {                                                                   \
    status = merge_resource_kernel(KERNEL, BYTES, &(ACCUMULATOR));       \
    if (status != cudaSuccess) {                                         \
      return static_cast<int>(status);                                   \
    }                                                                    \
  } while (false)
  Q3X_MERGE_RESOURCE(
      q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_linear_kernel,
      kSm87A4W4AttentionK256M128N128B3SharedBytes, b3);
  Q3X_MERGE_RESOURCE(
      q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_full_kernel,
      kSm87A4W4AttentionK256M128N128B3SharedBytes, b3);
  Q3X_MERGE_RESOURCE(
      q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_o_kernel,
      kSm87A4W4AttentionK256M128N128B3SharedBytes, b3);
#undef Q3X_MERGE_RESOURCE

  resources->registers_per_thread = b3.maximum_registers;
  resources->static_shared_bytes = b3.maximum_static_shared;
  resources->dynamic_shared_bytes =
      kSm87A4W4AttentionK256M128N128B3SharedBytes;
  resources->local_bytes = b3.maximum_local;
  resources->active_blocks_per_sm = b3.minimum_active_blocks;
  resources->maximum_threads_per_block = b3.minimum_maximum_threads;

  const bool b3_pass =
      resources->registers_per_thread > 0 &&
      resources->registers_per_thread <=
          static_cast<int>(
              kSm87A4W4AttentionK256M128N128MaximumRegisters) &&
      resources->static_shared_bytes == 0U &&
      resources->dynamic_shared_bytes ==
          kSm87A4W4AttentionK256M128N128B3SharedBytes &&
      resources->dynamic_shared_bytes <=
          kSm87A4W4AttentionK256M128N128MaximumDynamicSharedBytes &&
      resources->local_bytes == 0U &&
      resources->active_blocks_per_sm ==
          static_cast<int>(
              kSm87A4W4AttentionK256M128N128RequiredCtasPerSm);
  if (!b3_pass ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4AttentionK256M128N128AExchangeThreads)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_bf16_cuda(
    const Sm87A4W4AttentionK256Topology topology,
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::size_t token_count,
    const Sm87A4W4AttentionK256ProjectionView* const projections,
    const std::size_t projection_count,
    void* const cuda_stream) noexcept {
  const int resource_status = ensure_b3_resources_cached_for_current_device();
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  const std::size_t input_size =
      sm87_a4w4_attention_k256_fixed_input_size(topology);
  const std::size_t incumbent_cells =
      sm87_a4w4_attention_k256_fixed_cell_count(topology);
  if (input_size == 0U || incumbent_cells == 0U || token_count == 0U ||
      token_count % kTileM != 0U || !aligned(packed_a, 16U) ||
      !aligned(a_k256_scales_bf16, 16U)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_a_bytes =
      sm87_a4w4_attention_k256_packed_capacity_bytes(
          token_count, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_attention_k256_scale_capacity_elements(
          token_count, input_size);
  if (required_a_bytes == 0U || required_a_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      !b3_projection_views_valid(
          packed_a, required_a_bytes, a_k256_scales_bf16,
          required_a_scales, topology, token_count, input_size,
          projections, projection_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t m_tiles = token_count / kTileM;
  if (!sm87_a4w4_attention_k256_product_fits(incumbent_cells, 2U) ||
      !sm87_a4w4_attention_k256_product_fits(
          m_tiles, 2U * incumbent_cells)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t candidate_cells = 2U * incumbent_cells;
  const std::size_t work_cells = m_tiles * candidate_cells;
  const std::size_t k256_groups = input_size / 256U;
  const std::size_t physical_k64_groups = input_size / 64U;
  if (m_tiles > std::numeric_limits<unsigned int>::max() ||
      work_cells > std::numeric_limits<unsigned int>::max() ||
      k256_groups > std::numeric_limits<unsigned int>::max() ||
      physical_k64_groups > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t configure_status = configure_b3_topology(topology);
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }

  const dim3 grid(
      static_cast<unsigned int>(
          kSm87A4W4AttentionK256M128N128PersistentCtas));
  const dim3 block(
      static_cast<unsigned int>(
          kSm87A4W4AttentionK256M128N128AExchangeThreads));
  cudaStream_t const stream = static_cast<cudaStream_t>(cuda_stream);
#define Q3X_LAUNCH_B3(KERNEL)                                           \
  KERNEL<<<grid, block,                                                 \
           kSm87A4W4AttentionK256M128N128B3SharedBytes, stream>>>(      \
      packed_a, a_k256_scales_bf16, projections[0U].packed_b,           \
      projections[0U].k256_scales_bf16,                                \
      projection_count > 1U ? projections[1U].packed_b : nullptr,      \
      projection_count > 1U                                           \
          ? projections[1U].k256_scales_bf16                           \
          : nullptr,                                                   \
      projection_count > 2U ? projections[2U].packed_b : nullptr,      \
      projection_count > 2U                                           \
          ? projections[2U].k256_scales_bf16                           \
          : nullptr,                                                   \
      static_cast<unsigned int>(k256_groups),                          \
      static_cast<unsigned int>(physical_k64_groups),                  \
      projections[0U].output_bf16,                                     \
      static_cast<unsigned int>(projections[0U].output_row_stride_elements), \
      projection_count > 1U ? projections[1U].output_bf16 : nullptr,   \
      projection_count > 1U                                           \
          ? static_cast<unsigned int>(                                 \
                projections[1U].output_row_stride_elements)            \
          : 0U,                                                        \
      projection_count > 2U ? projections[2U].output_bf16 : nullptr,   \
      projection_count > 2U                                           \
          ? static_cast<unsigned int>(                                 \
                projections[2U].output_row_stride_elements)            \
          : 0U,                                                        \
      static_cast<unsigned int>(m_tiles),                              \
      static_cast<unsigned int>(work_cells))
  if (topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
    Q3X_LAUNCH_B3(
        q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_linear_kernel);
  } else if (topology == Sm87A4W4AttentionK256Topology::kFullQkv) {
    Q3X_LAUNCH_B3(
        q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_full_kernel);
  } else if (topology == Sm87A4W4AttentionK256Topology::kAttentionO) {
    Q3X_LAUNCH_B3(
        q3x_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_o_kernel);
  } else {
#undef Q3X_LAUNCH_B3
    return static_cast<int>(cudaErrorInvalidValue);
  }
#undef Q3X_LAUNCH_B3
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels

#if defined(Q3X_SM87_A4W4_ATTENTION_M128N128_B3_STANDALONE)
int main() {
  q3x::kernels::Sm87A4W4AttentionK256M128N128Resources resources{};
  const int status = q3x::kernels::
      query_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_resources_cuda(
          &resources);
  std::printf(
      "status=%d sm=%d.%d sm_count=%d threads=%d "
      "regs=%d static=%zu dynamic=%zu local=%zu active=%d\n",
      status, resources.device_compute_major, resources.device_compute_minor,
      resources.device_multiprocessor_count,
      resources.maximum_threads_per_block, resources.registers_per_thread,
      resources.static_shared_bytes, resources.dynamic_shared_bytes,
      resources.local_bytes, resources.active_blocks_per_sm);
  return status == static_cast<int>(cudaSuccess) ? 0 : 1;
}
#endif
