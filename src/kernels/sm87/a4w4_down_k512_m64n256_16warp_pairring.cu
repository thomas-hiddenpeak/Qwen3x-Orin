#include "q3x/kernels/sm87_a4w4_down_k512_m64n256_16warp_pairring.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kTileM =
    static_cast<unsigned int>(
        kSm87A4W4DownK512M64N256Pairring16TileM);
inline constexpr unsigned int kTileN =
    static_cast<unsigned int>(
        kSm87A4W4DownK512M64N256Pairring16TileN);
inline constexpr unsigned int kThreads =
    static_cast<unsigned int>(
        kSm87A4W4DownK512M64N256Pairring16Threads);
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(kSm87A4W4DownK512PackedRowK64Bytes);

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

struct alignas(16) Stage final {
  std::uint8_t a[kSm87A4W4DownK512M64N256Pairring16K64PerStage]
                [kTileM * kPackedK64Bytes];
  std::uint8_t b[kSm87A4W4DownK512M64N256Pairring16K64PerStage]
                [kTileN * kPackedK64Bytes];
};

struct alignas(16) ScaleSlot final {
  std::uint16_t a[kTileM];
  std::uint16_t b[kTileN];
};

struct alignas(16) SharedStorage final {
  Stage stage[kSm87A4W4DownK512M64N256Pairring16Stages];
  ScaleSlot scale[kSm87A4W4DownK512M64N256Pairring16ScaleSlots];
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

// Named fragments keep all persistent output and K512 partial addresses out
// of generic local arrays.  This is the same 32+32-register ownership floor
// as the retained M128N128/16-warp kernel.
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
              kSm87A4W4DownK512M64N256Pairring16StageBytes);
static_assert(sizeof(ScaleSlot) ==
              kSm87A4W4DownK512M64N256Pairring16ScaleSlotBytes);
static_assert(sizeof(SharedStorage) ==
              kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes);
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

// Publish the scale with the odd K256 stage.  A slot is therefore replaced
// only after the preceding odd stage has consumed and applied that slot.
__device__ __forceinline__ void issue_scale_slot(
    ScaleSlot& slot,
    const std::uint16_t* const a_scales,
    const std::uint16_t* const b_scales,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int group,
    const unsigned int group_count) noexcept {
  if (threadIdx.x < 8U) {
    const unsigned int row = 8U * threadIdx.x;
    cp_async_ca_16(
        slot.a + row,
        a_scales + sm87_a4w4_down_k512_scale_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       group, group_count));
  } else if (threadIdx.x < 40U) {
    const unsigned int row = 8U * (threadIdx.x - 8U);
    cp_async_ca_16(
        slot.b + row,
        b_scales + sm87_a4w4_down_k512_scale_offset(
                       static_cast<std::size_t>(n_tile_start) + row,
                       group, group_count));
  }
}

// One K256 stage contains 512 A vectors and 2,048 B vectors.  Every one of
// the 512 CTA threads therefore issues exactly one A and four B 16-byte
// transfers.  All five operations (plus an odd-stage scale owner operation)
// form one async commit group, so the four shared slots are also exactly four
// outstanding cp.async groups.
template <bool PublishScale>
__device__ __forceinline__ void issue_k256_stage(
    Stage& stage,
    ScaleSlot& scale,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const a_scales,
    const std::uint16_t* const b_scales,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count,
    const unsigned int k512_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      kTileM * kPackedK64Bytes / 16U;
  constexpr unsigned int kBVectorsPerPlane =
      kTileN * kPackedK64Bytes / 16U;
  static_assert(kAVectorsPerPlane *
                        kSm87A4W4DownK512M64N256Pairring16K64PerStage ==
                    kThreads);
  static_assert(kBVectorsPerPlane *
                        kSm87A4W4DownK512M64N256Pairring16K64PerStage ==
                    4U * kThreads);

  const unsigned int a_plane = threadIdx.x >> 7U;
  const unsigned int a_plane_vector = threadIdx.x & 127U;
  const unsigned int a_row = a_plane_vector >> 1U;
  const unsigned int a_row_vector = a_plane_vector & 1U;
  const unsigned int a_physical_k64 =
      4U * physical_k256_stage + a_plane;
  cp_async_cg_16(
      stage.a[a_plane] + sm87_a4w4_swizzled_k64_byte_offset(
                                   a_row, 16U * a_row_vector),
      packed_a + sm87_a4w4_down_k512_packed_offset(
                     static_cast<std::size_t>(m_tile_start) + a_row,
                     a_physical_k64, 16U * a_row_vector,
                     physical_k64_group_count));

  const unsigned int b_row = threadIdx.x >> 1U;
  const unsigned int b_row_vector = threadIdx.x & 1U;
  const std::size_t b_shared_offset =
      sm87_a4w4_swizzled_k64_byte_offset(
          b_row, 16U * b_row_vector);
#pragma unroll
  for (unsigned int b_plane = 0U; b_plane < 4U; ++b_plane) {
    const unsigned int b_physical_k64 =
        4U * physical_k256_stage + b_plane;
    cp_async_cg_16(
        stage.b[b_plane] + b_shared_offset,
        packed_b + sm87_a4w4_down_k512_packed_offset(
                       static_cast<std::size_t>(n_tile_start) + b_row,
                       b_physical_k64, 16U * b_row_vector,
                       physical_k64_group_count));
  }

  if constexpr (PublishScale) {
    issue_scale_slot(scale, a_scales, b_scales, m_tile_start,
                     n_tile_start, physical_k256_stage >> 1U,
                     k512_group_count);
  }
  cp_async_commit();
}

__device__ __forceinline__ void issue_k512_group(
    SharedStorage& shared,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const a_scales,
    const std::uint16_t* const b_scales,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int group,
    const unsigned int physical_k64_group_count,
    const unsigned int k512_group_count) noexcept {
  const unsigned int pair = 2U * (group & 1U);
  issue_k256_stage<false>(
      shared.stage[pair], shared.scale[group & 1U], packed_a, packed_b, a_scales,
      b_scales, m_tile_start, n_tile_start, 2U * group,
      physical_k64_group_count, k512_group_count);
  issue_k256_stage<true>(
      shared.stage[pair + 1U], shared.scale[group & 1U], packed_a,
      packed_b, a_scales, b_scales, m_tile_start, n_tile_start,
      2U * group + 1U, physical_k64_group_count,
      k512_group_count);
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
       plane < kSm87A4W4DownK512M64N256Pairring16K64PerStage;
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

// Two ordered K256 halves form one exact K512 numerical epoch.  No scale or
// FP32 update occurs between the eight ordered K64 S32 MMA steps.
__device__ __forceinline__ void accumulate_k512_group(
    const Stage& first,
    const Stage& second,
    const ScaleSlot& scale,
    Outputs& outputs) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m_start =
      (warp >> 3U) *
      static_cast<unsigned int>(
          kSm87A4W4DownK512M64N256Pairring16WarpTileM);
  const unsigned int local_n_start =
      (warp & 7U) *
      static_cast<unsigned int>(
          kSm87A4W4DownK512M64N256Pairring16WarpTileN);
  Partials partials{};
  accumulate_k256_stage(first, partials, local_m_start,
                        local_n_start, lane);
  accumulate_k256_stage(second, partials, local_m_start,
                        local_n_start, lane);
  apply_scales(scale, outputs, partials, local_m_start,
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
      m_tile_start + (warp >> 3U) * 32U + MFragment * 16U;
  const unsigned int base_n =
      n_tile_start + (warp & 7U) * 32U + NFragment * 8U;
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

__device__ __forceinline__ void compute_cell(
    SharedStorage& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_scales,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_scales,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    std::uint16_t* const output,
    const unsigned int output_row_stride) noexcept {
  Outputs outputs{};

  issue_k512_group(shared, packed_a, packed_b, a_scales, b_scales,
                   m_tile_start, n_tile_start, 0U,
                   physical_k64_group_count, k512_group_count);
  cp_async_wait<0U>();
  __syncthreads();

  for (unsigned int group = 0U; group < k512_group_count; ++group) {
    const bool has_next = group + 1U < k512_group_count;
    if (has_next) {
      issue_k512_group(
          shared, packed_a, packed_b, a_scales, b_scales,
          m_tile_start, n_tile_start, group + 1U,
          physical_k64_group_count, k512_group_count);
    }

    const unsigned int pair = 2U * (group & 1U);
    accumulate_k512_group(shared.stage[pair], shared.stage[pair + 1U],
                          shared.scale[group & 1U], outputs);
    if (has_next) {
      cp_async_wait<0U>();
      __syncthreads();
    }
  }

  store_outputs(outputs, m_tile_start, n_tile_start,
                output_row_stride, output);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownK512M64N256Pairring16Threads,
                      kSm87A4W4DownK512M64N256Pairring16CtasPerSm)
void q3x_sm87_a4w4_down_k512_m64n256_16warp_pairring_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int m_tile_count,
    const unsigned int n_tile_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_shared);
  const unsigned int work_cells = m_tile_count * n_tile_count;

  // Flat complete-cell ownership balances P1920's 600 cells as 37/38 and
  // P2176's 680 cells as 42/43 across the 16 pinned SM owners.
  for (unsigned int flat = blockIdx.x; flat < work_cells;
       flat += gridDim.x) {
    const unsigned int n_tile = flat / m_tile_count;
    const unsigned int m_tile = flat - n_tile * m_tile_count;
    compute_cell(
        shared, packed_a, a_k512_scales_bf16, packed_b,
        b_k512_scales_bf16, k512_group_count,
        physical_k64_group_count, m_tile * kTileM, n_tile * kTileN,
        output_bf16, output_row_stride_elements);
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
      properties.multiProcessorCount != kRequiredSmCount ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_m64n256_16warp_pairring_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_m64n256_16warp_pairring_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
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
  const auto plan =
      production_shape
          ? sm87_a4w4_down_k512_m64n256_16warp_pairring_plan(
                logical_token_count, launch_token_count, output_size,
                input_size)
          : sm87_a4w4_down_k512_m64n256_16warp_pairring_test_plan(
                launch_token_count, output_size, input_size);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      plan.launch_ctas > maximum_launch_ctas ||
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
      plan.n_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_cells > std::numeric_limits<unsigned int>::max() ||
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
      !sm87_a4w4_down_k512_product_fits(
          required_a_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(
          required_b_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(
          required_output_elements, sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_output_bytes =
      required_output_elements * sizeof(std::uint16_t);
  if (byte_ranges_overlap(output_bf16, required_output_bytes, packed_a,
                          required_a_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          a_k512_scales_bf16,
                          required_a_scales * sizeof(std::uint16_t)) ||
      byte_ranges_overlap(output_bf16, required_output_bytes, packed_b,
                          required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          b_k512_scales_bf16,
                          required_b_scales * sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87A4W4DownK512M64N256Pairring16Resources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k512_m64n256_16warp_pairring_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_k512_m64n256_16warp_pairring_kernel
      <<<static_cast<unsigned int>(plan.launch_ctas),
         kSm87A4W4DownK512M64N256Pairring16Threads,
         kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes, stream>>>(
          packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.n_tiles), output_bf16,
          static_cast<unsigned int>(output_row_stride_elements));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_down_k512_m64n256_16warp_pairring_resources_cuda(
    Sm87A4W4DownK512M64N256Pairring16Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK512M64N256Pairring16Resources{};
  cudaDeviceProp properties{};
  const int target_status = validate_sm87(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaError_t configure_status = configure_kernel();
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_down_k512_m64n256_16warp_pairring_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k512_m64n256_16warp_pairring_kernel,
      static_cast<int>(kSm87A4W4DownK512M64N256Pairring16Threads),
      kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes;
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
              kSm87A4W4DownK512M64N256Pairring16MaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4DownK512M64N256Pairring16Threads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(
              kSm87A4W4DownK512M64N256Pairring16CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_k512_m64n256_16warp_pairring_bf16_cuda(
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
      static_cast<unsigned int>(
          kSm87A4W4DownK512M64N256Pairring16PersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_down_k512_m64n256_16warp_pairring_test_bf16_cuda(
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

}  // namespace q3x::kernels
