#include "q3x/kernels/sm87_a4w4_down_k512_m128n128_16warp_pairring.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

std::atomic<bool> g_down_k256_pairring16_resources_ready{false};

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kTileM =
    static_cast<unsigned int>(
        kSm87A4W4DownK512M128N128Pairring16TileM);
inline constexpr unsigned int kTileN =
    static_cast<unsigned int>(
        kSm87A4W4DownK512M128N128Pairring16TileN);
inline constexpr unsigned int kThreads =
    static_cast<unsigned int>(
        kSm87A4W4DownK512M128N128Pairring16Threads);
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(kSm87A4W4DownK512PackedRowK64Bytes);

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

[[nodiscard]] constexpr std::size_t k256_consumer_packed_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  const std::size_t groups = sm87_a4w4_k64_group_count(logical_k);
  constexpr std::size_t kGroupBytes =
      kSm87A4W4ConsumerOuterBlock *
      kSm87A4W4ConsumerPackedKBlockBytes;
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_down_k512_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_down_k512_product_fits(block_groups, kGroupBytes)
             ? sm87_a4w4_consumer_packed_capacity_bytes(outer_count,
                                                        logical_k)
             : 0U;
}

struct alignas(16) Stage final {
  std::uint8_t a[kSm87A4W4DownK512M128N128Pairring16K64PerStage]
                [kTileM * kPackedK64Bytes];
  std::uint8_t b[kSm87A4W4DownK512M128N128Pairring16K64PerStage]
                [kTileN * kPackedK64Bytes];
};

struct alignas(16) ScaleSlot final {
  std::uint16_t a[kTileM];
  std::uint16_t b[kTileN];
};

struct alignas(16) SharedStorage final {
  Stage stage[kSm87A4W4DownK512M128N128Pairring16Stages];
  ScaleSlot scale[kSm87A4W4DownK512M128N128Pairring16ScaleSlots];
};

struct alignas(16) K256ScaleStage final {
  Stage codes;
  ScaleSlot scale;
};

struct alignas(16) K256ScaleSharedStorage final {
  K256ScaleStage
      stage[kSm87A4W4DownK256M128N128Pairring16Stages];
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

// Named fragments are intentional: none of the persistent outputs or K512
// partials may acquire a runtime address and be lowered to local memory.
struct Outputs final {
  Float4 m0n0{};
  Float4 m0n1{};
  Float4 m0n2{};
  Float4 m0n3{};
  Float4 m1n0{};
  Float4 m1n1{};
  Float4 m1n2{};
  Float4 m1n3{};
};

struct Partials final {
  Sm87A4W4Accumulator m0n0{};
  Sm87A4W4Accumulator m0n1{};
  Sm87A4W4Accumulator m0n2{};
  Sm87A4W4Accumulator m0n3{};
  Sm87A4W4Accumulator m1n0{};
  Sm87A4W4Accumulator m1n1{};
  Sm87A4W4Accumulator m1n2{};
  Sm87A4W4Accumulator m1n3{};
};

static_assert(sizeof(Stage) ==
              kSm87A4W4DownK512M128N128Pairring16StageBytes);
static_assert(sizeof(ScaleSlot) ==
              kSm87A4W4DownK512M128N128Pairring16ScaleSlotBytes);
static_assert(sizeof(SharedStorage) ==
              kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes);
static_assert(sizeof(K256ScaleStage) ==
              kSm87A4W4DownK256M128N128Pairring16StageBytes);
static_assert(sizeof(K256ScaleSharedStorage) ==
              kSm87A4W4DownK256M128N128Pairring16DynamicSharedBytes);
static_assert(
    kSm87A4W4DownK256M128N128Pairring16K64PerStage ==
    kSm87A4W4DownK512M128N128Pairring16K64PerStage);
static_assert(sizeof(Outputs) == 128U);
static_assert(sizeof(Partials) == 128U);

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

__device__ __forceinline__ void cp_async_cg_16(
    void* const destination, const void* const source) noexcept {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(source)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_ca_16(
    void* const destination, const void* const source) noexcept {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
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

__device__ __forceinline__ void cp_async_wait_all() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.wait_group 0;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

template <unsigned int Remaining>
__device__ __forceinline__ void cp_async_wait_group() noexcept {
  static_assert(Remaining <= 3U);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(Remaining)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

// A K256 stage contains 1,024 A vectors and 1,024 B vectors.  Each of the
// 512 threads publishes two vectors from each operand, or four cp.async
// instructions total per stage.
__device__ __forceinline__ void issue_k256_stage(
    Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane = kTileM * kPackedK64Bytes / 16U;
  constexpr unsigned int kVectorsPerOperand =
      static_cast<unsigned int>(
          kSm87A4W4DownK512M128N128Pairring16K64PerStage) *
      kVectorsPerPlane;
  static_assert(kVectorsPerPlane == 256U);
  static_assert(kVectorsPerOperand == 2U * kThreads);

#pragma unroll
  for (unsigned int linear = threadIdx.x; linear < kVectorsPerOperand;
       linear += kThreads) {
    const unsigned int plane = linear / kVectorsPerPlane;
    const unsigned int plane_vector = linear - plane * kVectorsPerPlane;
    const unsigned int row = plane_vector >> 1U;
    const unsigned int row_vector = plane_vector & 1U;
    const unsigned int physical_k64_group =
        4U * physical_k256_stage + plane;
    const std::size_t shared_offset =
        sm87_a4w4_swizzled_k64_byte_offset(row, 16U * row_vector);
    cp_async_cg_16(
        stage.a[plane] + shared_offset,
        packed_a + sm87_a4w4_down_k512_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64_group, 16U * row_vector,
                       physical_k64_group_count));
    cp_async_cg_16(
        stage.b[plane] + shared_offset,
        packed_b + sm87_a4w4_down_k512_packed_offset(
                       static_cast<std::size_t>(n_tile_start) + row,
                       physical_k64_group, 16U * row_vector,
                       physical_k64_group_count));
  }
  cp_async_commit();
}

// A scale slot contains exactly the 128 A and 128 B BF16 values consumed by
// one K512 group.  Thirty-two lanes publish one aligned 16-byte vector each.
__device__ __forceinline__ void issue_scale_slot(
    ScaleSlot& slot,
    const std::uint16_t* const a_scales,
    const std::uint16_t* const b_scales,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int group,
    const unsigned int group_count) noexcept {
  if (threadIdx.x < 32U) {
    const unsigned int operand = threadIdx.x >> 4U;
    const unsigned int vector = threadIdx.x & 15U;
    const unsigned int row = 8U * vector;
    std::uint16_t* const destination =
        (operand == 0U ? slot.a : slot.b) + row;
    const std::uint16_t* const source = operand == 0U ? a_scales : b_scales;
    const unsigned int outer_start =
        operand == 0U ? m_tile_start : n_tile_start;
    cp_async_ca_16(
        destination,
        source + sm87_a4w4_down_k512_scale_offset(
                     static_cast<std::size_t>(outer_start) + row,
                     group, group_count));
  }
  cp_async_commit();
}

// One logical K256 group is one indivisible ring entry: four packed K64 code
// planes and its A/B BF16 scale vectors share a single cp.async commit group.
// This keeps wait_group accounting aligned with logical numerical epochs.
__device__ __forceinline__ void issue_k256_scale_stage(
    K256ScaleStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k256_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k256_group,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      kTileM * kPackedK64Bytes / 16U;
  constexpr unsigned int kVectorsPerOperand =
      static_cast<unsigned int>(
          kSm87A4W4DownK256M128N128Pairring16K64PerStage) *
      kVectorsPerPlane;
  static_assert(kVectorsPerPlane == 256U);
  static_assert(kVectorsPerOperand == 2U * kThreads);

#pragma unroll
  for (unsigned int linear = threadIdx.x; linear < kVectorsPerOperand;
       linear += kThreads) {
    const unsigned int plane = linear / kVectorsPerPlane;
    const unsigned int plane_vector = linear - plane * kVectorsPerPlane;
    const unsigned int row = plane_vector >> 1U;
    const unsigned int row_vector = plane_vector & 1U;
    const unsigned int physical_k64_group =
        kSm87A4W4DownK256M128N128Pairring16K64PerStage * k256_group +
        plane;
    const std::size_t shared_offset =
        sm87_a4w4_swizzled_k64_byte_offset(row, 16U * row_vector);
    cp_async_cg_16(
        stage.codes.a[plane] + shared_offset,
        packed_a + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64_group, 16U * row_vector,
                       physical_k64_group_count));
    cp_async_cg_16(
        stage.codes.b[plane] + shared_offset,
        packed_b + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(n_tile_start) + row,
                       physical_k64_group, 16U * row_vector,
                       physical_k64_group_count));
  }

  if (threadIdx.x < 32U) {
    const unsigned int operand = threadIdx.x >> 4U;
    const unsigned int vector = threadIdx.x & 15U;
    const unsigned int row = 8U * vector;
    std::uint16_t* const destination =
        (operand == 0U ? stage.scale.a : stage.scale.b) + row;
    const std::uint16_t* const source =
        operand == 0U ? a_k256_scales_bf16 : b_k256_scales_bf16;
    const unsigned int outer_start =
        operand == 0U ? m_tile_start : n_tile_start;
    cp_async_ca_16(
        destination,
        source + sm87_a4w4_attention_k256_scale_offset(
                     static_cast<std::size_t>(outer_start) + row,
                     k256_group, k256_group_count));
  }
  cp_async_commit();
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

__device__ __forceinline__ void accumulate_k256_stage(
    const Stage& stage,
    Partials& partials,
    const unsigned int local_m_start,
    const unsigned int local_n_start,
    const unsigned int lane) noexcept {
#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4DownK512M128N128Pairring16K64PerStage;
       ++plane) {
    const Sm87A4W4AFragment a0 = load_a_ldmatrix_x4(
        stage.a[plane] + local_m_start * kPackedK64Bytes, lane);
    const Sm87A4W4AFragment a1 = load_a_ldmatrix_x4(
        stage.a[plane] + (local_m_start + 16U) * kPackedK64Bytes,
        lane);

    const Sm87A4W4BFragment b0 = load_b_ldmatrix_x2(
        stage.b[plane] + local_n_start * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(partials.m0n0, a0, b0);
    sm87_a4w4_mma_m16n8k64(partials.m1n0, a1, b0);

    const Sm87A4W4BFragment b1 = load_b_ldmatrix_x2(
        stage.b[plane] + (local_n_start + 8U) * kPackedK64Bytes,
        lane);
    sm87_a4w4_mma_m16n8k64(partials.m0n1, a0, b1);
    sm87_a4w4_mma_m16n8k64(partials.m1n1, a1, b1);

    const Sm87A4W4BFragment b2 = load_b_ldmatrix_x2(
        stage.b[plane] + (local_n_start + 16U) * kPackedK64Bytes,
        lane);
    sm87_a4w4_mma_m16n8k64(partials.m0n2, a0, b2);
    sm87_a4w4_mma_m16n8k64(partials.m1n2, a1, b2);

    const Sm87A4W4BFragment b3 = load_b_ldmatrix_x2(
        stage.b[plane] + (local_n_start + 24U) * kPackedK64Bytes,
        lane);
    sm87_a4w4_mma_m16n8k64(partials.m0n3, a0, b3);
    sm87_a4w4_mma_m16n8k64(partials.m1n3, a1, b3);
  }
}

__device__ __forceinline__ void update_one(
    Float4& output,
    const Sm87A4W4Accumulator& partial,
    const float a0,
    const float a1,
    const float b0,
    const float b1) noexcept {
  output.x0 = __fmaf_rn(static_cast<float>(partial.x0),
                        __fmul_rn(a0, b0), output.x0);
  output.x1 = __fmaf_rn(static_cast<float>(partial.x1),
                        __fmul_rn(a0, b1), output.x1);
  output.x2 = __fmaf_rn(static_cast<float>(partial.x2),
                        __fmul_rn(a1, b0), output.x2);
  output.x3 = __fmaf_rn(static_cast<float>(partial.x3),
                        __fmul_rn(a1, b1), output.x3);
}

__device__ __forceinline__ void apply_scales(
    const ScaleSlot& scale,
    Outputs& outputs,
    const Partials& partials,
    const unsigned int local_m_start,
    const unsigned int local_n_start,
    const unsigned int lane) noexcept {
  // Each lane owns one M scale and one N scale.  All repeated output-lane
  // scale traffic is a warp shuffle from these two shared owner loads.
  const float a_owned = decode_bf16(scale.a[local_m_start + lane]);
  const float b_owned = decode_bf16(scale.b[local_n_start + lane]);
  const unsigned int m_low = lane >> 2U;
  const unsigned int m_high = m_low + 8U;
  const unsigned int n_even = 2U * (lane & 3U);
  const unsigned int n_odd = n_even + 1U;
  constexpr unsigned int kMask = 0xffff'ffffU;

  const float a00 = __shfl_sync(kMask, a_owned, m_low);
  const float a01 = __shfl_sync(kMask, a_owned, m_high);
  const float a10 = __shfl_sync(kMask, a_owned, 16U + m_low);
  const float a11 = __shfl_sync(kMask, a_owned, 16U + m_high);

  float b0 = __shfl_sync(kMask, b_owned, n_even);
  float b1 = __shfl_sync(kMask, b_owned, n_odd);
  update_one(outputs.m0n0, partials.m0n0, a00, a01, b0, b1);
  update_one(outputs.m1n0, partials.m1n0, a10, a11, b0, b1);

  b0 = __shfl_sync(kMask, b_owned, 8U + n_even);
  b1 = __shfl_sync(kMask, b_owned, 8U + n_odd);
  update_one(outputs.m0n1, partials.m0n1, a00, a01, b0, b1);
  update_one(outputs.m1n1, partials.m1n1, a10, a11, b0, b1);

  b0 = __shfl_sync(kMask, b_owned, 16U + n_even);
  b1 = __shfl_sync(kMask, b_owned, 16U + n_odd);
  update_one(outputs.m0n2, partials.m0n2, a00, a01, b0, b1);
  update_one(outputs.m1n2, partials.m1n2, a10, a11, b0, b1);

  b0 = __shfl_sync(kMask, b_owned, 24U + n_even);
  b1 = __shfl_sync(kMask, b_owned, 24U + n_odd);
  update_one(outputs.m0n3, partials.m0n3, a00, a01, b0, b1);
  update_one(outputs.m1n3, partials.m1n3, a10, a11, b0, b1);
}

__device__ __forceinline__ void accumulate_k512_group(
    const Stage& first,
    const Stage& second,
    const ScaleSlot& scale,
    Outputs& outputs) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m_start =
      (warp & 3U) *
      static_cast<unsigned int>(
          kSm87A4W4DownK512M128N128Pairring16WarpTileM);
  const unsigned int local_n_start =
      (warp >> 2U) *
      static_cast<unsigned int>(
          kSm87A4W4DownK512M128N128Pairring16WarpTileN);
  Partials partials{};
  accumulate_k256_stage(first, partials, local_m_start, local_n_start,
                        lane);
  accumulate_k256_stage(second, partials, local_m_start, local_n_start,
                        lane);
  apply_scales(scale, outputs, partials, local_m_start, local_n_start,
               lane);
}

__device__ __forceinline__ void accumulate_k256_scale_group(
    const K256ScaleStage& stage,
    Outputs& outputs) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m_start =
      (warp & 3U) *
      static_cast<unsigned int>(
          kSm87A4W4DownK512M128N128Pairring16WarpTileM);
  const unsigned int local_n_start =
      (warp >> 2U) *
      static_cast<unsigned int>(
          kSm87A4W4DownK512M128N128Pairring16WarpTileN);
  Partials partials{};
  accumulate_k256_stage(stage.codes, partials, local_m_start,
                        local_n_start, lane);
  apply_scales(stage.scale, outputs, partials, local_m_start,
               local_n_start, lane);
}

template <unsigned int MFragment, unsigned int NFragment>
__device__ __forceinline__ void store_fragment(
    const Float4& values,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int output_row_stride,
    std::uint16_t* const output) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int base_m =
      m_tile_start + (warp & 3U) * 32U + MFragment * 16U;
  const unsigned int base_n =
      n_tile_start + (warp >> 2U) * 32U + NFragment * 8U;
  const Sm87A4W4AccumulatorCoordinate c0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate c1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate c2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const Sm87A4W4AccumulatorCoordinate c3 =
      sm87_a4w4_accumulator_coordinate(lane, 3U);
  output[static_cast<std::size_t>(base_m + c0.m) * output_row_stride +
         base_n + c0.n] = encode_bf16(values.x0);
  output[static_cast<std::size_t>(base_m + c1.m) * output_row_stride +
         base_n + c1.n] = encode_bf16(values.x1);
  output[static_cast<std::size_t>(base_m + c2.m) * output_row_stride +
         base_n + c2.n] = encode_bf16(values.x2);
  output[static_cast<std::size_t>(base_m + c3.m) * output_row_stride +
         base_n + c3.n] = encode_bf16(values.x3);
}

__device__ __forceinline__ void store_outputs(
    const Outputs& outputs,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int output_row_stride,
    std::uint16_t* const output) noexcept {
  store_fragment<0U, 0U>(outputs.m0n0, m_tile_start, n_tile_start,
                          output_row_stride, output);
  store_fragment<0U, 1U>(outputs.m0n1, m_tile_start, n_tile_start,
                          output_row_stride, output);
  store_fragment<0U, 2U>(outputs.m0n2, m_tile_start, n_tile_start,
                          output_row_stride, output);
  store_fragment<0U, 3U>(outputs.m0n3, m_tile_start, n_tile_start,
                          output_row_stride, output);
  store_fragment<1U, 0U>(outputs.m1n0, m_tile_start, n_tile_start,
                          output_row_stride, output);
  store_fragment<1U, 1U>(outputs.m1n1, m_tile_start, n_tile_start,
                          output_row_stride, output);
  store_fragment<1U, 2U>(outputs.m1n2, m_tile_start, n_tile_start,
                          output_row_stride, output);
  store_fragment<1U, 3U>(outputs.m1n3, m_tile_start, n_tile_start,
                          output_row_stride, output);
}

// Exact incumbent cell body used only by the default-off L2 scheduling
// sibling.  Keeping it separate leaves the production kernel's generated
// compute path untouched; only the sibling's persistent work mapping differs.
__device__ __forceinline__ void compute_l2_macro4x4_cell(
    SharedStorage& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile,
    const unsigned int n_tile) noexcept {
  const unsigned int m_tile_start = m_tile * kTileM;
  const unsigned int n_tile_start = n_tile * kTileN;
  Outputs outputs{};

  issue_k256_stage(shared.stage[0U], packed_a, packed_b, m_tile_start,
                   n_tile_start, 0U, physical_k64_group_count);
  issue_k256_stage(shared.stage[1U], packed_a, packed_b, m_tile_start,
                   n_tile_start, 1U, physical_k64_group_count);
  issue_scale_slot(shared.scale[0U], a_k512_scales_bf16,
                   b_k512_scales_bf16, m_tile_start, n_tile_start, 0U,
                   k512_group_count);
  cp_async_wait_all();
  __syncthreads();

  for (unsigned int group = 0U; group < k512_group_count; ++group) {
    const bool has_next = group + 1U < k512_group_count;
    if (has_next) {
      const unsigned int next = group + 1U;
      const unsigned int pair = 2U * (next & 1U);
      issue_k256_stage(shared.stage[pair], packed_a, packed_b,
                       m_tile_start, n_tile_start, 2U * next,
                       physical_k64_group_count);
      issue_k256_stage(shared.stage[pair + 1U], packed_a, packed_b,
                       m_tile_start, n_tile_start, 2U * next + 1U,
                       physical_k64_group_count);
      issue_scale_slot(shared.scale[next & 1U], a_k512_scales_bf16,
                       b_k512_scales_bf16, m_tile_start, n_tile_start,
                       next, k512_group_count);
    }

    const unsigned int pair = 2U * (group & 1U);
    accumulate_k512_group(shared.stage[pair], shared.stage[pair + 1U],
                          shared.scale[group & 1U], outputs);

    if (has_next) {
      cp_async_wait_all();
      __syncthreads();
    }
  }

  store_outputs(outputs, m_tile_start, n_tile_start,
                output_row_stride_elements, output_bf16);
  __syncthreads();
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownK512M128N128Pairring16Threads, 1)
void q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_storage);

  for (unsigned int work = blockIdx.x; work < work_tile_count;
       work += gridDim.x) {
    const unsigned int n_tile = work / m_tile_count;
    const unsigned int m_tile = work - n_tile * m_tile_count;
    const unsigned int m_tile_start = m_tile * kTileM;
    const unsigned int n_tile_start = n_tile * kTileN;
    Outputs outputs{};

    issue_k256_stage(shared.stage[0U], packed_a, packed_b, m_tile_start,
                     n_tile_start, 0U, physical_k64_group_count);
    issue_k256_stage(shared.stage[1U], packed_a, packed_b, m_tile_start,
                     n_tile_start, 1U, physical_k64_group_count);
    issue_scale_slot(shared.scale[0U], a_k512_scales_bf16,
                     b_k512_scales_bf16, m_tile_start, n_tile_start, 0U,
                     k512_group_count);
    cp_async_wait_all();
    __syncthreads();

    for (unsigned int group = 0U; group < k512_group_count; ++group) {
      const bool has_next = group + 1U < k512_group_count;
      if (has_next) {
        const unsigned int next = group + 1U;
        const unsigned int pair = 2U * (next & 1U);
        issue_k256_stage(shared.stage[pair], packed_a, packed_b,
                         m_tile_start, n_tile_start, 2U * next,
                         physical_k64_group_count);
        issue_k256_stage(shared.stage[pair + 1U], packed_a, packed_b,
                         m_tile_start, n_tile_start, 2U * next + 1U,
                         physical_k64_group_count);
        issue_scale_slot(shared.scale[next & 1U], a_k512_scales_bf16,
                         b_k512_scales_bf16, m_tile_start, n_tile_start,
                         next, k512_group_count);
      }

      const unsigned int pair = 2U * (group & 1U);
      accumulate_k512_group(shared.stage[pair], shared.stage[pair + 1U],
                            shared.scale[group & 1U], outputs);

      if (has_next) {
        cp_async_wait_all();
        __syncthreads();
      }
    }

    store_outputs(outputs, m_tile_start, n_tile_start,
                  output_row_stride_elements, output_bf16);
    __syncthreads();
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownK512M128N128Pairring16Threads, 1)
void q3x_sm87_a4w4_down_k256_m128n128_16warp_pairring_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k256_scales_bf16,
    const unsigned int k256_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count) {
  extern __shared__ __align__(16) unsigned char k256_dynamic_storage[];
  auto& shared =
      *reinterpret_cast<K256ScaleSharedStorage*>(k256_dynamic_storage);

  for (unsigned int work = blockIdx.x; work < work_tile_count;
       work += gridDim.x) {
    const unsigned int n_tile = work / m_tile_count;
    const unsigned int m_tile = work - n_tile * m_tile_count;
    const unsigned int m_tile_start = m_tile * kTileM;
    const unsigned int n_tile_start = n_tile * kTileN;
    Outputs outputs{};

    const unsigned int initial_groups =
        k256_group_count <
                kSm87A4W4DownK256M128N128Pairring16Stages
            ? k256_group_count
            : static_cast<unsigned int>(
                  kSm87A4W4DownK256M128N128Pairring16Stages);
    for (unsigned int group = 0U; group < initial_groups; ++group) {
      issue_k256_scale_stage(
          shared.stage[group], packed_a, a_k256_scales_bf16,
          packed_b, b_k256_scales_bf16, m_tile_start, n_tile_start,
          group, k256_group_count, physical_k64_group_count);
    }

    for (unsigned int group = 0U; group < k256_group_count; ++group) {
      const unsigned int newer_groups = k256_group_count - group - 1U;
      if (newer_groups >= 3U) {
        cp_async_wait_group<3U>();
      } else if (newer_groups == 2U) {
        cp_async_wait_group<2U>();
      } else if (newer_groups == 1U) {
        cp_async_wait_group<1U>();
      } else {
        cp_async_wait_group<0U>();
      }
      __syncthreads();

      const unsigned int slot =
          group % kSm87A4W4DownK256M128N128Pairring16Stages;
      accumulate_k256_scale_group(shared.stage[slot], outputs);
      __syncthreads();

      const unsigned int next_group =
          group + kSm87A4W4DownK256M128N128Pairring16Stages;
      if (next_group < k256_group_count) {
        issue_k256_scale_stage(
            shared.stage[slot], packed_a, a_k256_scales_bf16,
            packed_b, b_k256_scales_bf16, m_tile_start,
            n_tile_start, next_group, k256_group_count,
            physical_k64_group_count);
      }
    }

    store_outputs(outputs, m_tile_start, n_tile_start,
                  output_row_stride_elements, output_bf16);
    __syncthreads();
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownK512M128N128Pairring16Threads, 1)
void q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int n_tile_count) {
  if (gridDim.x !=
          kSm87A4W4DownK512M128N128Pairring16L2Grid ||
      !sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_topology(
          m_tile_count, n_tile_count)) {
    return;
  }

  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_storage);

  // Every iteration is one 16-CTA wave over exactly four A tiles and four B
  // tiles.  The mapping helper puts all complete M macros first and the
  // partial M macro last, while keeping only one persistent loop counter live
  // across the unchanged compute cell.
  const unsigned int wave_count =
      sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_wave_count(
          m_tile_count, n_tile_count);
  for (unsigned int wave = 0U; wave < wave_count; ++wave) {
    const Sm87A4W4DownK512M128N128Pairring16L2Work work =
        sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_work(
            blockIdx.x, wave, m_tile_count, n_tile_count);
    if (work.valid) {
      compute_l2_macro4x4_cell(
          shared, packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16, k512_group_count,
          physical_k64_group_count, output_bf16,
          output_row_stride_elements, work.m_tile, work.n_tile);
    }
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
          kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int admit_dynamic_shared() noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes)));
}

[[nodiscard]] int admit_k256_dynamic_shared() noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k256_m128n128_16warp_pairring_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4DownK256M128N128Pairring16DynamicSharedBytes)));
}

[[nodiscard]] int admit_l2_macro4x4_dynamic_shared() noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes)));
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool production_shape,
    void* const cuda_stream) noexcept {
  if (production_shape &&
      !sm87_a4w4_down_k512_m128n128_16warp_pairring_padding_contract(
          logical_token_count, launch_token_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const Sm87A4W4DownK512Plan plan =
      production_shape
          ? sm87_a4w4_down_k512_plan(launch_token_count, output_size,
                                      input_size)
          : sm87_a4w4_down_k512_test_plan(launch_token_count, output_size,
                                           input_size);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(packed_b, 16U) ||
      !aligned(b_k512_scales_bf16, 16U) ||
      !aligned(output_bf16, alignof(std::uint32_t)) ||
      output_row_stride_elements < output_size ||
      output_row_stride_elements % 2U != 0U ||
      output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      plan.k512_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_down_k512_product_fits(
          launch_token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(launch_token_count,
                                                input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(output_size, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(launch_token_count,
                                                   input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(output_size,
                                                   input_size);
  const std::size_t required_output_elements =
      launch_token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements ||
      !sm87_a4w4_down_k512_product_fits(required_a_scales,
                                         sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_b_scales,
                                         sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_output_elements,
                                         sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t required_output_bytes =
      required_output_elements * sizeof(std::uint16_t);
  if (byte_ranges_overlap(output_bf16, required_output_bytes, packed_a,
                          required_a_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          a_k512_scales_bf16,
                          required_a_scale_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes, packed_b,
                          required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          b_k512_scales_bf16,
                          required_b_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87A4W4DownK512M128N128Pairring16Resources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k512_m128n128_16warp_pairring_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const unsigned int planned_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      planned_ctas < maximum_launch_ctas ? planned_ctas
                                         : maximum_launch_ctas;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_kernel
      <<<launch_ctas,
         static_cast<unsigned int>(
             kSm87A4W4DownK512M128N128Pairring16Threads),
         kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes,
         stream>>>(
          packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          output_bf16,
          static_cast<unsigned int>(output_row_stride_elements),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.work_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

[[nodiscard]] int launch_k256_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k256_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool production_shape,
    void* const cuda_stream) noexcept {
  if (production_shape &&
      !sm87_a4w4_down_k256_m128n128_16warp_pairring_padding_contract(
          logical_token_count, launch_token_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const Sm87A4W4DownK256M128N128Pairring16Plan plan =
      production_shape
          ? sm87_a4w4_down_k256_m128n128_16warp_pairring_plan(
                launch_token_count, output_size, input_size)
          : sm87_a4w4_down_k256_m128n128_16warp_pairring_test_plan(
                launch_token_count, output_size, input_size);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k256_scales_bf16, 16U) ||
      !aligned(packed_b, 16U) ||
      !aligned(b_k256_scales_bf16, 16U) ||
      !aligned(output_bf16, alignof(std::uint32_t)) ||
      output_row_stride_elements < output_size ||
      output_row_stride_elements % 2U != 0U ||
      output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      launch_token_count > std::numeric_limits<unsigned int>::max() ||
      output_size > std::numeric_limits<unsigned int>::max() ||
      plan.k256_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.n_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_down_k512_product_fits(
          launch_token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      k256_consumer_packed_capacity_bytes(launch_token_count,
                                          input_size);
  const std::size_t required_b_bytes =
      k256_consumer_packed_capacity_bytes(output_size, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_attention_k256_scale_capacity_elements(
          launch_token_count, input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_attention_k256_scale_capacity_elements(output_size,
                                                        input_size);
  const std::size_t required_output_elements =
      launch_token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements ||
      !sm87_a4w4_down_k512_product_fits(required_a_scales,
                                         sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_b_scales,
                                         sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_output_elements,
                                         sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t required_output_bytes =
      required_output_elements * sizeof(std::uint16_t);
  const auto overlaps = [](const void* const first,
                           const std::size_t first_bytes,
                           const void* const second,
                           const std::size_t second_bytes) noexcept {
    return byte_ranges_overlap(first, first_bytes, second, second_bytes);
  };
  if (overlaps(packed_a, required_a_bytes, a_k256_scales_bf16,
               required_a_scale_bytes) ||
      overlaps(packed_a, required_a_bytes, packed_b,
               required_b_bytes) ||
      overlaps(packed_a, required_a_bytes, b_k256_scales_bf16,
               required_b_scale_bytes) ||
      overlaps(a_k256_scales_bf16, required_a_scale_bytes, packed_b,
               required_b_bytes) ||
      overlaps(a_k256_scales_bf16, required_a_scale_bytes,
               b_k256_scales_bf16, required_b_scale_bytes) ||
      overlaps(packed_b, required_b_bytes, b_k256_scales_bf16,
               required_b_scale_bytes) ||
      overlaps(output_bf16, required_output_bytes, packed_a,
               required_a_bytes) ||
      overlaps(output_bf16, required_output_bytes,
               a_k256_scales_bf16, required_a_scale_bytes) ||
      overlaps(output_bf16, required_output_bytes, packed_b,
               required_b_bytes) ||
      overlaps(output_bf16, required_output_bytes,
               b_k256_scales_bf16, required_b_scale_bytes)) {
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
    if (!g_down_k256_pairring16_resources_ready.load(
            std::memory_order_acquire)) {
      return static_cast<int>(cudaErrorNotReady);
    }
  } else if (!g_down_k256_pairring16_resources_ready.load(
                 std::memory_order_acquire)) {
    Sm87A4W4DownK256M128N128Pairring16Resources resources{};
    const int resource_status =
        query_sm87_a4w4_down_k256_m128n128_16warp_pairring_resources_cuda(
            &resources);
    if (resource_status != static_cast<int>(cudaSuccess)) {
      return resource_status;
    }
  }

  const unsigned int planned_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      planned_ctas < maximum_launch_ctas ? planned_ctas
                                         : maximum_launch_ctas;
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_k256_m128n128_16warp_pairring_kernel
      <<<launch_ctas,
         static_cast<unsigned int>(
             kSm87A4W4DownK512M128N128Pairring16Threads),
         kSm87A4W4DownK256M128N128Pairring16DynamicSharedBytes,
         stream>>>(
          packed_a, a_k256_scales_bf16, packed_b,
          b_k256_scales_bf16,
          static_cast<unsigned int>(plan.k256_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          output_bf16,
          static_cast<unsigned int>(output_row_stride_elements),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.work_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

[[nodiscard]] int launch_l2_macro4x4_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const bool production_shape,
    void* const cuda_stream) noexcept {
  if (production_shape &&
      !sm87_a4w4_down_k512_m128n128_16warp_pairring_padding_contract(
          logical_token_count, launch_token_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const Sm87A4W4DownK512Plan plan =
      production_shape
          ? sm87_a4w4_down_k512_plan(launch_token_count, output_size,
                                      input_size)
          : sm87_a4w4_down_k512_test_plan(launch_token_count, output_size,
                                           input_size);
  if (plan.launch_ctas == 0U ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.n_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_topology(
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.n_tiles)) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(packed_b, 16U) ||
      !aligned(b_k512_scales_bf16, 16U) ||
      !aligned(output_bf16, alignof(std::uint32_t)) ||
      output_row_stride_elements < output_size ||
      output_row_stride_elements % 2U != 0U ||
      output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      plan.k512_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_down_k512_product_fits(
          launch_token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(launch_token_count,
                                                input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(output_size, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(launch_token_count,
                                                   input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(output_size,
                                                   input_size);
  const std::size_t required_output_elements =
      launch_token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements ||
      !sm87_a4w4_down_k512_product_fits(required_a_scales,
                                         sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_b_scales,
                                         sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_output_elements,
                                         sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t required_output_bytes =
      required_output_elements * sizeof(std::uint16_t);
  if (byte_ranges_overlap(output_bf16, required_output_bytes, packed_a,
                          required_a_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          a_k512_scales_bf16,
                          required_a_scale_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes, packed_b,
                          required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          b_k512_scales_bf16,
                          required_b_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87A4W4DownK512M128N128Pairring16Resources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_kernel
      <<<kSm87A4W4DownK512M128N128Pairring16L2Grid,
         static_cast<unsigned int>(
             kSm87A4W4DownK512M128N128Pairring16Threads),
         kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes,
         stream>>>(
          packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          output_bf16,
          static_cast<unsigned int>(output_row_stride_elements),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.n_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_down_k512_m128n128_16warp_pairring_resources_cuda(
    Sm87A4W4DownK512M128N128Pairring16Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK512M128N128Pairring16Resources{};
  cudaDeviceProp properties{};
  const int target_status = validate_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const int admission_status = admit_dynamic_shared();
  if (admission_status != static_cast<int>(cudaSuccess)) {
    return admission_status;
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_kernel,
      static_cast<int>(
          kSm87A4W4DownK512M128N128Pairring16Threads),
      kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes;
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
              kSm87A4W4DownK512M128N128Pairring16MaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4DownK512M128N128Pairring16Threads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4DownK512M128N128Pairring16CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int query_sm87_a4w4_down_k256_m128n128_16warp_pairring_resources_cuda(
    Sm87A4W4DownK256M128N128Pairring16Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK256M128N128Pairring16Resources{};
  g_down_k256_pairring16_resources_ready.store(false,
                                                std::memory_order_release);
  cudaDeviceProp properties{};
  const int target_status = validate_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  if (properties.sharedMemPerBlockOptin <
      kSm87A4W4DownK256M128N128Pairring16DynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const int admission_status = admit_k256_dynamic_shared();
  if (admission_status != static_cast<int>(cudaSuccess)) {
    return admission_status;
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_down_k256_m128n128_16warp_pairring_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k256_m128n128_16warp_pairring_kernel,
      static_cast<int>(
          kSm87A4W4DownK512M128N128Pairring16Threads),
      kSm87A4W4DownK256M128N128Pairring16DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK256M128N128Pairring16DynamicSharedBytes;
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
              kSm87A4W4DownK512M128N128Pairring16MaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK256M128N128Pairring16DynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK256M128N128Pairring16DynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4DownK256M128N128Pairring16DynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4DownK512M128N128Pairring16Threads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4DownK512M128N128Pairring16CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  g_down_k256_pairring16_resources_ready.store(true,
                                                std::memory_order_release);
  return static_cast<int>(cudaSuccess);
}

int
query_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_resources_cuda(
    Sm87A4W4DownK512M128N128Pairring16Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK512M128N128Pairring16Resources{};
  cudaDeviceProp properties{};
  const int target_status = validate_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const int admission_status = admit_l2_macro4x4_dynamic_shared();
  if (admission_status != static_cast<int>(cudaSuccess)) {
    return admission_status;
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_kernel,
      static_cast<int>(
          kSm87A4W4DownK512M128N128Pairring16Threads),
      kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes;
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
              kSm87A4W4DownK512M128N128Pairring16MaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4DownK512M128N128Pairring16Threads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4DownK512M128N128Pairring16CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_k512_m128n128_16warp_pairring_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements,
      logical_token_count, launch_token_count, output_size, input_size,
      output_bf16, output_row_stride_elements, output_capacity_elements,
      static_cast<unsigned int>(kSm87A4W4DownK512PersistentCtas), true,
      cuda_stream);
}

int launch_sm87_a4w4_down_k256_m128n128_16warp_pairring_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k256_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_k256_impl(
      packed_a, packed_a_capacity_bytes, a_k256_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k256_scales_bf16, b_scale_capacity_elements,
      logical_token_count, launch_token_count, output_size, input_size,
      output_bf16, output_row_stride_elements, output_capacity_elements,
      static_cast<unsigned int>(kSm87A4W4DownK512PersistentCtas), true,
      cuda_stream);
}

int
launch_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_l2_macro4x4_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements,
      logical_token_count, launch_token_count, output_size, input_size,
      output_bf16, output_row_stride_elements, output_capacity_elements,
      true, cuda_stream);
}

int launch_sm87_a4w4_down_k512_m128n128_16warp_pairring_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements, token_count,
      token_count, output_size, input_size, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      maximum_launch_ctas, false, cuda_stream);
}

int launch_sm87_a4w4_down_k256_m128n128_16warp_pairring_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k256_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k256_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_k256_impl(
      packed_a, packed_a_capacity_bytes, a_k256_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k256_scales_bf16, b_scale_capacity_elements, token_count,
      token_count, output_size, input_size, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      maximum_launch_ctas, false, cuda_stream);
}

int
launch_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_l2_macro4x4_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements, token_count,
      token_count, output_size, input_size, output_bf16,
      output_row_stride_elements, output_capacity_elements, false,
      cuda_stream);
}

}  // namespace q3x::kernels
