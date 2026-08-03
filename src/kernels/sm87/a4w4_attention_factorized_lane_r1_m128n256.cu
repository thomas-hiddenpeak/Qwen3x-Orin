#include "q3x/kernels/sm87_a4w4_attention_factorized_lane_r1_m128n256.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kThreads = 256U;
inline constexpr unsigned int kWarps = 8U;
inline constexpr unsigned int kTileM = 128U;
inline constexpr unsigned int kTileN = 256U;
inline constexpr unsigned int kWarpTileM = 32U;
inline constexpr unsigned int kWarpTileN = 128U;
inline constexpr unsigned int kK64PerGroup = 4U;
inline constexpr unsigned int kPackedK64Bytes = 32U;
inline constexpr unsigned int kBSlots = 4U;
inline constexpr unsigned int kDescriptorProjectionShift = 24U;
inline constexpr unsigned int kDescriptorPanelMask = 0x00ff'ffffU;
inline constexpr int kRequiredSmCount = 16;

enum class KernelTopology : unsigned int {
  kLinearQkvZ,
  kFullQkv,
  kAttentionO,
  kContiguousTest,
};

struct alignas(16) AExchangeSlot final {
  std::uint8_t code[kK64PerGroup][kTileM * kPackedK64Bytes];
  // Whole-K R1 scale: loaded once, never overwritten by later A exchanges.
  std::uint16_t scale[kTileM];
};

struct alignas(16) BRingSlot final {
  std::uint8_t code[kK64PerGroup][kTileN * kPackedK64Bytes];
};

struct alignas(16) SharedStorage final {
  AExchangeSlot a;
  BRingSlot b[kBSlots];
  // The topology mapper gathers four possibly-disjoint N64 scale panels.
  std::uint16_t b_scale[kTileN];
};

// Eight ordered K64 A fragments are resident while one K256 B slot is
// consumed.  R1 scales are intentionally absent from this hot object: they
// are applied once by the explicit epilogue after the complete K reduction.
struct ResidentA final {
  Sm87A4W4AFragment p0_m0;
  Sm87A4W4AFragment p0_m1;
  Sm87A4W4AFragment p1_m0;
  Sm87A4W4AFragment p1_m1;
  Sm87A4W4AFragment p2_m0;
  Sm87A4W4AFragment p2_m1;
  Sm87A4W4AFragment p3_m0;
  Sm87A4W4AFragment p3_m1;
};

static_assert(kThreads ==
              kSm87A4W4AttentionFactorizedLaneR1Threads);
static_assert(kWarps == kSm87A4W4AttentionFactorizedLaneR1Warps);
static_assert(kWarps * kWarpTileM * kWarpTileN == kTileM * kTileN);
static_assert(sizeof(AExchangeSlot) == 16'640U);
static_assert(sizeof(BRingSlot) == 32'768U);
static_assert(sizeof(SharedStorage) == 148'224U);
static_assert(sizeof(SharedStorage) ==
              kSm87A4W4AttentionFactorizedLaneR1SharedBytes);
static_assert(sizeof(ResidentA) == 128U);

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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.commit_group;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

template <unsigned int Remaining>
__device__ __forceinline__ void cp_async_wait() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
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

template <KernelTopology Topology>
[[nodiscard]] __device__ __forceinline__ unsigned int fixed_descriptor(
    const unsigned int cell, const unsigned int slot) noexcept {
  static_assert(Topology == KernelTopology::kLinearQkvZ ||
                Topology == KernelTopology::kFullQkv ||
                Topology == KernelTopology::kAttentionO ||
                Topology == KernelTopology::kContiguousTest);
  if constexpr (Topology == KernelTopology::kLinearQkvZ) {
    if (cell < 48U) {
      return slot < 2U
                 ? 2U * cell + slot
                 : (1U << kDescriptorProjectionShift) |
                       (2U * cell + slot - 2U);
    }
    return 96U + 4U * (cell - 48U) + slot;
  } else if constexpr (Topology == KernelTopology::kFullQkv) {
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
    // Attention-O and the isolated test topology are both one contiguous
    // projection.  Only their host-side N/K admission differs.
    return 4U * cell + slot;
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

template <KernelTopology Topology>
__device__ __forceinline__ void issue_factor_scales(
    SharedStorage& shared,
    const std::uint16_t* const a_scales,
    const std::uint16_t* const b_scales0,
    const std::uint16_t* const b_scales1,
    const std::uint16_t* const b_scales2,
    const unsigned int macro_cell,
    const unsigned int m_tile_start) noexcept {
  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        shared.a.scale + first_row,
        a_scales +
            sm87_a4w4_attention_factorized_lane_r1_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row));
  }
  if (threadIdx.x < 32U) {
    const unsigned int panel_slot = threadIdx.x >> 3U;
    const unsigned int row_vector = threadIdx.x & 7U;
    const unsigned int descriptor =
        fixed_descriptor<Topology>(macro_cell, panel_slot);
    const unsigned int projection =
        descriptor >> kDescriptorProjectionShift;
    const unsigned int panel = descriptor & kDescriptorPanelMask;
    const std::uint16_t* const selected_scales =
        select_scale_projection(projection, b_scales0, b_scales1,
                                b_scales2);
    cp_async_16(
        shared.b_scale + 64U * panel_slot + 8U * row_vector,
        selected_scales +
            sm87_a4w4_attention_factorized_lane_r1_scale_offset(
                static_cast<std::size_t>(panel) * 64U +
                8U * row_vector));
  }
  // No commit here: scale traffic joins the first A exchange group and is
  // therefore proven resident by the same initial wait.
}

__device__ __forceinline__ void issue_a_exchange(
    AExchangeSlot& slot,
    const std::uint8_t* const packed_a,
    const unsigned int m_tile_start,
    const unsigned int k256_group,
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
  cp_async_commit();
}

template <KernelTopology Topology>
__device__ __forceinline__ void issue_b_ring_slot(
    BRingSlot& slot,
    const std::uint8_t* const packed_b0,
    const std::uint8_t* const packed_b1,
    const std::uint8_t* const packed_b2,
    const unsigned int macro_cell,
    const unsigned int k256_group,
    const unsigned int physical_k64_group_count) noexcept {
  const unsigned int panel_slot = threadIdx.x >> 6U;
  const unsigned int thread_in_panel = threadIdx.x & 63U;
  const unsigned int descriptor =
      fixed_descriptor<Topology>(macro_cell, panel_slot);
  const unsigned int projection =
      descriptor >> kDescriptorProjectionShift;
  const unsigned int panel = descriptor & kDescriptorPanelMask;
  const std::uint8_t* const selected_b = select_packed_projection(
      projection, packed_b0, packed_b1, packed_b2);
#pragma unroll
  for (unsigned int plane = 0U; plane < kK64PerGroup; ++plane) {
#pragma unroll
    for (unsigned int vector_in_thread = 0U; vector_in_thread < 2U;
         ++vector_in_thread) {
      const unsigned int panel_vector =
          thread_in_panel + 64U * vector_in_thread;
      const unsigned int panel_row = panel_vector >> 1U;
      const unsigned int row_vector = panel_vector & 1U;
      const unsigned int shared_row = panel_slot * 64U + panel_row;
      const unsigned int k64 = k256_group * kK64PerGroup + plane;
      cp_async_16(
          slot.code[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                                 shared_row, 16U * row_vector),
          selected_b + sm87_a4w4_attention_k256_packed_offset(
                           static_cast<std::size_t>(panel) * 64U +
                               panel_row,
                           k64, 16U * row_vector,
                           physical_k64_group_count));
    }
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
  return resident;
}

template <unsigned int FragmentN>
__device__ __forceinline__ void accumulate_n8(
    const ResidentA& a,
    const BRingSlot& b_slot,
    Sm87A4W4Accumulator& output_m0,
    Sm87A4W4Accumulator& output_m1) noexcept {
  static_assert(FragmentN < 16U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int n128_half = warp >> 2U;
  constexpr unsigned int kFragmentOffset = FragmentN * 8U;
  const unsigned int local_n =
      n128_half * kWarpTileN + kFragmentOffset;

  Sm87A4W4BFragment b = load_b_ldmatrix_x2(
      b_slot.code[0U] + local_n * kPackedK64Bytes, lane);
  sm87_a4w4_mma_m16n8k64(output_m0, a.p0_m0, b);
  sm87_a4w4_mma_m16n8k64(output_m1, a.p0_m1, b);
  b = load_b_ldmatrix_x2(
      b_slot.code[1U] + local_n * kPackedK64Bytes, lane);
  sm87_a4w4_mma_m16n8k64(output_m0, a.p1_m0, b);
  sm87_a4w4_mma_m16n8k64(output_m1, a.p1_m1, b);
  b = load_b_ldmatrix_x2(
      b_slot.code[2U] + local_n * kPackedK64Bytes, lane);
  sm87_a4w4_mma_m16n8k64(output_m0, a.p2_m0, b);
  sm87_a4w4_mma_m16n8k64(output_m1, a.p2_m1, b);
  b = load_b_ldmatrix_x2(
      b_slot.code[3U] + local_n * kPackedK64Bytes, lane);
  sm87_a4w4_mma_m16n8k64(output_m0, a.p3_m0, b);
  sm87_a4w4_mma_m16n8k64(output_m1, a.p3_m1, b);
}

__device__ __forceinline__ void accumulate_group(
    const ResidentA& a,
    const BRingSlot& b,
    Sm87A4W4Accumulator (&output_m0)[16U],
    Sm87A4W4Accumulator (&output_m1)[16U]) noexcept {
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
  Q3X_ACCUMULATE_N8(8U);
  Q3X_ACCUMULATE_N8(9U);
  Q3X_ACCUMULATE_N8(10U);
  Q3X_ACCUMULATE_N8(11U);
  Q3X_ACCUMULATE_N8(12U);
  Q3X_ACCUMULATE_N8(13U);
  Q3X_ACCUMULATE_N8(14U);
  Q3X_ACCUMULATE_N8(15U);
#undef Q3X_ACCUMULATE_N8
}

[[nodiscard]] __device__ __forceinline__ float scaled_output(
    const std::int32_t whole_k_s32,
    const std::uint16_t a_scale_bits,
    const std::uint16_t b_scale_bits) noexcept {
  return __fmul_rn(
      static_cast<float>(whole_k_s32),
      __fmul_rn(decode_bf16(a_scale_bits),
                decode_bf16(b_scale_bits)));
}

template <unsigned int FragmentN, KernelTopology Topology>
__device__ __forceinline__ void publish_topology_bf16_n8(
    const SharedStorage& shared,
    const Sm87A4W4Accumulator& value_m0,
    const Sm87A4W4Accumulator& value_m1,
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
  const unsigned int warp_m32 = warp & 3U;
  const unsigned int n128_half = warp >> 2U;
  constexpr unsigned int kPanelInHalf = FragmentN / 8U;
  constexpr unsigned int kFragmentInPanel = FragmentN % 8U;
  const unsigned int panel_slot = 2U * n128_half + kPanelInHalf;
  const unsigned int descriptor =
      fixed_descriptor<Topology>(macro_cell, panel_slot);
  const unsigned int projection =
      descriptor >> kDescriptorProjectionShift;
  const unsigned int panel = descriptor & kDescriptorPanelMask;
  std::uint16_t* const selected_output = select_output_projection(
      projection, output0, output1, output2);
  const unsigned int stride = select_stride_projection(
      projection, stride0, stride1, stride2);
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const unsigned int local_m = warp_m32 * kWarpTileM;
  const unsigned int local_n =
      n128_half * kWarpTileN + FragmentN * 8U;
  const unsigned int global_n =
      panel * 64U + kFragmentInPanel * 8U + coordinate0.n;
  const unsigned int global_m0 =
      m_tile_start + local_m + coordinate0.m;
  const unsigned int global_m1 =
      m_tile_start + local_m + coordinate2.m;
  const std::uint16_t b_scale0 =
      shared.b_scale[local_n + coordinate0.n];
  const std::uint16_t b_scale1 =
      shared.b_scale[local_n + coordinate1.n];

#define Q3X_PACK_S32_PAIR(value, a_bits, first, second)                  \
  (static_cast<std::uint32_t>(                                          \
       encode_bf16(scaled_output((value).first, (a_bits), b_scale0))) | \
   (static_cast<std::uint32_t>(                                         \
        encode_bf16(scaled_output((value).second, (a_bits), b_scale1))) \
    << 16U))
  const std::uint16_t a_m0_scale0 =
      shared.a.scale[local_m + coordinate0.m];
  const std::uint16_t a_m0_scale1 =
      shared.a.scale[local_m + coordinate2.m];
  const std::uint16_t a_m1_scale0 =
      shared.a.scale[local_m + 16U + coordinate0.m];
  const std::uint16_t a_m1_scale1 =
      shared.a.scale[local_m + 16U + coordinate2.m];
  const std::uint32_t m0_pair0 =
      Q3X_PACK_S32_PAIR(value_m0, a_m0_scale0, x0, x1);
  const std::uint32_t m0_pair1 =
      Q3X_PACK_S32_PAIR(value_m0, a_m0_scale1, x2, x3);
  const std::uint32_t m1_pair0 =
      Q3X_PACK_S32_PAIR(value_m1, a_m1_scale0, x0, x1);
  const std::uint32_t m1_pair1 =
      Q3X_PACK_S32_PAIR(value_m1, a_m1_scale1, x2, x3);
#undef Q3X_PACK_S32_PAIR

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

template <KernelTopology Topology>
__device__ __forceinline__ void publish_topology_bf16(
    const SharedStorage& shared,
    const Sm87A4W4Accumulator (&output_m0)[16U],
    const Sm87A4W4Accumulator (&output_m1)[16U],
    const unsigned int macro_cell,
    const unsigned int m_tile_start,
    std::uint16_t* const output0,
    const unsigned int stride0,
    std::uint16_t* const output1,
    const unsigned int stride1,
    std::uint16_t* const output2,
    const unsigned int stride2) noexcept {
  // This is the sole currently admitted epilogue.  Consumer-native modes
  // belong beside it and must preserve the same whole-K S32 producer.
#define Q3X_PUBLISH_N8(fragment)                                        \
  publish_topology_bf16_n8<fragment, Topology>(                         \
      shared, output_m0[fragment], output_m1[fragment], macro_cell,     \
      m_tile_start, output0, stride0, output1, stride1, output2, stride2)
  Q3X_PUBLISH_N8(0U);
  Q3X_PUBLISH_N8(1U);
  Q3X_PUBLISH_N8(2U);
  Q3X_PUBLISH_N8(3U);
  Q3X_PUBLISH_N8(4U);
  Q3X_PUBLISH_N8(5U);
  Q3X_PUBLISH_N8(6U);
  Q3X_PUBLISH_N8(7U);
  Q3X_PUBLISH_N8(8U);
  Q3X_PUBLISH_N8(9U);
  Q3X_PUBLISH_N8(10U);
  Q3X_PUBLISH_N8(11U);
  Q3X_PUBLISH_N8(12U);
  Q3X_PUBLISH_N8(13U);
  Q3X_PUBLISH_N8(14U);
  Q3X_PUBLISH_N8(15U);
#undef Q3X_PUBLISH_N8
}

template <KernelTopology Topology>
__device__ __forceinline__ void run_cell_kernel(
    SharedStorage& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_scales,
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
    Sm87A4W4Accumulator output_m0[16U]{};
    Sm87A4W4Accumulator output_m1[16U]{};

    issue_factor_scales<Topology>(
        shared, a_scales, b_scales0, b_scales1, b_scales2, macro_cell,
        m_tile_start);
    // A0 commits the scale loads as part of the same FIFO group.  The B4
    // ring then exactly follows the already-validated K256 A-exchange/B4
    // scheduler; only arithmetic and scale lifetime have changed.
    issue_a_exchange(shared.a, packed_a, m_tile_start, 0U,
                     physical_k64_group_count);
    issue_b_ring_slot<Topology>(
        shared.b[0U], packed_b0, packed_b1, packed_b2, macro_cell, 0U,
        physical_k64_group_count);
    if (k256_group_count > 1U) {
      issue_b_ring_slot<Topology>(
          shared.b[1U], packed_b0, packed_b1, packed_b2, macro_cell, 1U,
          physical_k64_group_count);
    }
    if (k256_group_count > 2U) {
      issue_b_ring_slot<Topology>(
          shared.b[2U], packed_b0, packed_b1, packed_b2, macro_cell, 2U,
          physical_k64_group_count);
    }
    if (k256_group_count > 3U) {
      issue_b_ring_slot<Topology>(
          shared.b[3U], packed_b0, packed_b1, packed_b2, macro_cell, 3U,
          physical_k64_group_count);
      cp_async_wait<3U>();
    } else if (k256_group_count > 2U) {
      cp_async_wait<2U>();
    } else if (k256_group_count > 1U) {
      cp_async_wait<1U>();
    } else {
      cp_async_wait<0U>();
    }
    __syncthreads();  // publish scales, A0, and B0

    for (unsigned int group = 0U; group < k256_group_count; ++group) {
      const ResidentA resident = load_resident_a(shared.a);
      __syncthreads();  // release only A code; whole-K scales persist

      if (group + 1U < k256_group_count) {
        issue_a_exchange(shared.a, packed_a, m_tile_start, group + 1U,
                         physical_k64_group_count);
      }

      accumulate_group(resident, shared.b[group & 3U], output_m0,
                       output_m1);
      __syncthreads();  // release consumed B code slot

      if (group + kBSlots < k256_group_count) {
        issue_b_ring_slot<Topology>(
            shared.b[group & 3U], packed_b0, packed_b1, packed_b2,
            macro_cell, group + kBSlots, physical_k64_group_count);
        cp_async_wait<1U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();  // publish A_next and next B slot
    }

    publish_topology_bf16<Topology>(
        shared, output_m0, output_m1, macro_cell, m_tile_start, output0,
        stride0, output1, stride1, output2, stride2);
    __syncthreads();
  }
}

template <KernelTopology Topology>
__global__ __launch_bounds__(kThreads, 1)
void q3x_sm87_a4w4_attention_factorized_lane_r1_m128n256_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_scales,
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
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_shared);
  run_cell_kernel<Topology>(
      shared, packed_a, a_scales, packed_b0, b_scales0, packed_b1,
      b_scales1, packed_b2, b_scales2, k256_group_count,
      physical_k64_group_count, output0, stride0, output1, stride1,
      output2, stride2, m_tile_count, work_cell_count);
}

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
      properties.multiProcessorCount != kRequiredSmCount ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4AttentionFactorizedLaneR1SharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

template <KernelTopology Topology>
[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_attention_factorized_lane_r1_m128n256_kernel<
          Topology>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4AttentionFactorizedLaneR1SharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_attention_factorized_lane_r1_m128n256_kernel<
          Topology>,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

[[nodiscard]] cudaError_t configure_topology(
    const Sm87A4W4AttentionK256Topology topology) noexcept {
  if (topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
    return configure_kernel<KernelTopology::kLinearQkvZ>();
  }
  if (topology == Sm87A4W4AttentionK256Topology::kFullQkv) {
    return configure_kernel<KernelTopology::kFullQkv>();
  }
  if (topology == Sm87A4W4AttentionK256Topology::kAttentionO) {
    return configure_kernel<KernelTopology::kAttentionO>();
  }
  return cudaErrorInvalidValue;
}

template <KernelTopology Topology>
[[nodiscard]] cudaError_t merge_resources(
    int* const maximum_registers,
    std::size_t* const maximum_static_shared,
    std::size_t* const maximum_local,
    int* const minimum_maximum_threads,
    int* const minimum_active_blocks,
    std::size_t* const minimum_dynamic_limit) noexcept {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_attention_factorized_lane_r1_m128n256_kernel<
          Topology>);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_attention_factorized_lane_r1_m128n256_kernel<
          Topology>,
      static_cast<int>(kThreads),
      kSm87A4W4AttentionFactorizedLaneR1SharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  if (attributes.numRegs > *maximum_registers) {
    *maximum_registers = attributes.numRegs;
  }
  if (attributes.sharedSizeBytes > *maximum_static_shared) {
    *maximum_static_shared = attributes.sharedSizeBytes;
  }
  if (attributes.localSizeBytes > *maximum_local) {
    *maximum_local = attributes.localSizeBytes;
  }
  if (attributes.maxThreadsPerBlock < *minimum_maximum_threads) {
    *minimum_maximum_threads = attributes.maxThreadsPerBlock;
  }
  if (active_blocks < *minimum_active_blocks) {
    *minimum_active_blocks = active_blocks;
  }
  if (static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes) <
      *minimum_dynamic_limit) {
    *minimum_dynamic_limit =
        static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes);
  }
  return cudaSuccess;
}

[[nodiscard]] bool projection_views_valid(
    const std::uint8_t* const packed_a,
    const std::size_t required_a_bytes,
    const std::uint16_t* const a_scales,
    const std::size_t required_a_scale_elements,
    const std::size_t token_count,
    const std::size_t input_size,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView* const
        projections,
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
        !aligned(view.lane_scales_bf16, 16U) ||
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
    const std::size_t required_b_scales =
        sm87_a4w4_attention_factorized_lane_r1_scale_capacity_elements(
            view.output_size);
    const std::size_t required_output_elements =
        token_count * view.output_row_stride_elements;
    if (required_b_bytes == 0U || required_b_scales == 0U ||
        view.packed_b_capacity_bytes < required_b_bytes ||
        view.scale_capacity_elements < required_b_scales ||
        view.output_capacity_elements < required_output_elements ||
        !sm87_a4w4_attention_k256_product_fits(
            required_b_scales, sizeof(std::uint16_t)) ||
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
      const auto& source_view = projections[source];
      const std::size_t source_b_bytes =
          sm87_a4w4_attention_k256_packed_capacity_bytes(
              source_view.output_size, input_size);
      const std::size_t source_scale_elements =
          sm87_a4w4_attention_factorized_lane_r1_scale_capacity_elements(
              source_view.output_size);
      if (byte_ranges_overlap(view.output_bf16, output_bytes,
                              source_view.packed_b, source_b_bytes) ||
          byte_ranges_overlap(view.output_bf16, output_bytes,
                              source_view.lane_scales_bf16,
                              source_scale_elements *
                                  sizeof(std::uint16_t))) {
        return false;
      }
    }
    for (std::size_t prior_index = 0U; prior_index < projection;
         ++prior_index) {
      const auto& prior = projections[prior_index];
      const std::size_t prior_elements =
          token_count * prior.output_row_stride_elements;
      if (byte_ranges_overlap(view.output_bf16, output_bytes,
                              prior.output_bf16,
                              prior_elements * sizeof(std::uint16_t))) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] int validate_common(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_scales,
    const std::size_t a_scale_capacity_elements,
    const Sm87A4W4AttentionK256Plan& plan,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView* const
        projections,
    const std::size_t projection_count,
    const Sm87A4W4AttentionFactorizedLaneR1EpilogueMode
        epilogue_mode) noexcept {
  if (epilogue_mode !=
          Sm87A4W4AttentionFactorizedLaneR1EpilogueMode::kTopologyBf16 ||
      plan.launch_ctas == 0U || !aligned(packed_a, 16U) ||
      !aligned(a_scales, 16U) ||
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
      sm87_a4w4_attention_factorized_lane_r1_scale_capacity_elements(
          plan.token_count);
  if (required_a_bytes == 0U || required_a_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      !projection_views_valid(
          packed_a, required_a_bytes, a_scales, required_a_scales,
          plan.token_count, plan.input_size, projections,
          projection_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

template <KernelTopology Topology>
void launch_specialization(
    const dim3 grid,
    const cudaStream_t stream,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_scales,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView& p0,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView& p1,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView& p2,
    const Sm87A4W4AttentionK256Plan& plan) noexcept {
  q3x_sm87_a4w4_attention_factorized_lane_r1_m128n256_kernel<Topology>
      <<<grid, dim3(kThreads),
         kSm87A4W4AttentionFactorizedLaneR1SharedBytes, stream>>>(
          packed_a, a_scales, p0.packed_b, p0.lane_scales_bf16,
          p1.packed_b, p1.lane_scales_bf16, p2.packed_b,
          p2.lane_scales_bf16,
          static_cast<unsigned int>(plan.k256_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          p0.output_bf16,
          static_cast<unsigned int>(p0.output_row_stride_elements),
          p1.output_bf16,
          static_cast<unsigned int>(p1.output_row_stride_elements),
          p2.output_bf16,
          static_cast<unsigned int>(p2.output_row_stride_elements),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.work_cells));
}

}  // namespace

int query_sm87_a4w4_attention_factorized_lane_r1_m128n256_resources_cuda(
    Sm87A4W4AttentionFactorizedLaneR1Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4AttentionFactorizedLaneR1Resources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaError_t status = configure_kernel<KernelTopology::kLinearQkvZ>();
  if (status == cudaSuccess) {
    status = configure_kernel<KernelTopology::kFullQkv>();
  }
  if (status == cudaSuccess) {
    status = configure_kernel<KernelTopology::kAttentionO>();
  }
  if (status == cudaSuccess) {
    status = configure_kernel<KernelTopology::kContiguousTest>();
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  int maximum_registers = 0;
  std::size_t maximum_static_shared = 0U;
  std::size_t maximum_local = 0U;
  int minimum_maximum_threads = std::numeric_limits<int>::max();
  int minimum_active_blocks = std::numeric_limits<int>::max();
  std::size_t minimum_dynamic_limit =
      std::numeric_limits<std::size_t>::max();
#define Q3X_MERGE_SPECIALIZATION(topology)                              \
  do {                                                                  \
    status = merge_resources<topology>(                                 \
        &maximum_registers, &maximum_static_shared, &maximum_local,      \
        &minimum_maximum_threads, &minimum_active_blocks,                \
        &minimum_dynamic_limit);                                        \
    if (status != cudaSuccess) {                                        \
      return static_cast<int>(status);                                  \
    }                                                                   \
  } while (false)
  Q3X_MERGE_SPECIALIZATION(KernelTopology::kLinearQkvZ);
  Q3X_MERGE_SPECIALIZATION(KernelTopology::kFullQkv);
  Q3X_MERGE_SPECIALIZATION(KernelTopology::kAttentionO);
  Q3X_MERGE_SPECIALIZATION(KernelTopology::kContiguousTest);
#undef Q3X_MERGE_SPECIALIZATION

  resources->registers_per_thread = maximum_registers;
  resources->static_shared_bytes = maximum_static_shared;
  resources->dynamic_shared_bytes =
      kSm87A4W4AttentionFactorizedLaneR1SharedBytes;
  resources->configured_dynamic_shared_limit_bytes =
      minimum_dynamic_limit;
  resources->device_optin_shared_limit_bytes =
      properties.sharedMemPerBlockOptin;
  resources->local_bytes = maximum_local;
  resources->maximum_threads_per_block = minimum_maximum_threads;
  resources->active_blocks_per_sm = minimum_active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->multiprocessor_count = properties.multiProcessorCount;
  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4AttentionFactorizedLaneR1MaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4AttentionFactorizedLaneR1SharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4AttentionFactorizedLaneR1SharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4AttentionFactorizedLaneR1SharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block < static_cast<int>(kThreads) ||
      resources->active_blocks_per_sm != 1 ||
      resources->compute_major != kSm87A4W4RequiredComputeMajor ||
      resources->compute_minor != kSm87A4W4RequiredComputeMinor ||
      resources->multiprocessor_count != kRequiredSmCount) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_attention_factorized_lane_r1_m128n256_bf16_cuda(
    const Sm87A4W4AttentionK256Topology topology,
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::size_t token_count,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView* const
        projections,
    const std::size_t projection_count,
    const Sm87A4W4AttentionFactorizedLaneR1EpilogueMode epilogue_mode,
    void* const cuda_stream) noexcept {
  const Sm87A4W4AttentionK256Plan plan =
      sm87_a4w4_attention_k256_fixed_plan(topology, token_count);
  if (projections == nullptr ||
      projection_count !=
          sm87_a4w4_attention_k256_fixed_projection_count(topology)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  for (std::size_t projection = 0U; projection < projection_count;
       ++projection) {
    if (projections[projection].output_size !=
        sm87_a4w4_attention_k256_fixed_projection_panels(
            topology, projection) *
            kSm87A4W4AttentionK256PanelN) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  const int common_status = validate_common(
      packed_a, packed_a_capacity_bytes, a_lane_scales_bf16,
      a_scale_capacity_elements, plan, projections, projection_count,
      epilogue_mode);
  if (common_status != static_cast<int>(cudaSuccess)) {
    return common_status;
  }
  const int device_status = validate_sm87();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const cudaError_t configure_status = configure_topology(topology);
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }

  const auto& p0 = projections[0U];
  const Sm87A4W4AttentionFactorizedLaneR1ProjectionView empty{};
  const auto& p1 = projection_count > 1U ? projections[1U] : empty;
  const auto& p2 = projection_count > 2U ? projections[2U] : empty;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const dim3 grid(static_cast<unsigned int>(plan.launch_ctas));
  (void)cudaGetLastError();
  if (topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
    launch_specialization<KernelTopology::kLinearQkvZ>(
        grid, stream, packed_a, a_lane_scales_bf16, p0, p1, p2, plan);
  } else if (topology == Sm87A4W4AttentionK256Topology::kFullQkv) {
    launch_specialization<KernelTopology::kFullQkv>(
        grid, stream, packed_a, a_lane_scales_bf16, p0, p1, p2, plan);
  } else if (topology == Sm87A4W4AttentionK256Topology::kAttentionO) {
    launch_specialization<KernelTopology::kAttentionO>(
        grid, stream, packed_a, a_lane_scales_bf16, p0, p1, p2, plan);
  } else {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_a4w4_attention_factorized_lane_r1_m128n256_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t input_size,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView* const
        projection,
    const std::size_t macro_cells,
    const Sm87A4W4AttentionFactorizedLaneR1EpilogueMode epilogue_mode,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  const Sm87A4W4AttentionK256Plan plan =
      sm87_a4w4_attention_k256_test_plan(token_count, input_size,
                                         macro_cells);
  if (projection == nullptr || maximum_launch_ctas == 0U ||
      !sm87_a4w4_attention_k256_product_fits(
          macro_cells, kSm87A4W4AttentionK256TileN) ||
      projection->output_size !=
          macro_cells * kSm87A4W4AttentionK256TileN) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int common_status = validate_common(
      packed_a, packed_a_capacity_bytes, a_lane_scales_bf16,
      a_scale_capacity_elements, plan, projection, 1U, epilogue_mode);
  if (common_status != static_cast<int>(cudaSuccess)) {
    return common_status;
  }
  const int device_status = validate_sm87();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  const cudaError_t configure_status =
      configure_kernel<KernelTopology::kContiguousTest>();
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }
  const Sm87A4W4AttentionFactorizedLaneR1ProjectionView empty{};
  const unsigned int planned_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      planned_ctas < maximum_launch_ctas ? planned_ctas
                                         : maximum_launch_ctas;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  launch_specialization<KernelTopology::kContiguousTest>(
      dim3(launch_ctas), stream, packed_a, a_lane_scales_bf16,
      *projection, empty, empty, plan);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
