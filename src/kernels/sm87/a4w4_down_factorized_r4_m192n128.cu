#include "q3x/kernels/sm87_a4w4_down_factorized_r4_m192n128.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kTileM = static_cast<unsigned int>(
    kSm87A4W4DownFactorizedR4M192N128TileM);
inline constexpr unsigned int kTileN = static_cast<unsigned int>(
    kSm87A4W4DownFactorizedR4M192N128TileN);
inline constexpr unsigned int kThreads = static_cast<unsigned int>(
    kSm87A4W4DownFactorizedR4M192N128Threads);
inline constexpr unsigned int kK64PerStage = static_cast<unsigned int>(
    kSm87A4W4DownFactorizedR4M192N128K64PerStage);
inline constexpr unsigned int kPackedK64Bytes = 32U;
inline constexpr unsigned int kPhysicalK64Groups =
    static_cast<unsigned int>(
        kSm87A4W4DownFactorizedR4M192N128ModelInput /
        kSm87A4W4DownFactorizedR4M192N128PhysicalK64);

struct ByteRange final {
  std::uintptr_t begin{};
  std::uintptr_t end{};
};

[[nodiscard]] bool aligned_16(const void* const pointer) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
}

[[nodiscard]] bool make_range(const void* const pointer,
                              const std::size_t bytes,
                              ByteRange* const output) noexcept {
  if (pointer == nullptr || bytes == 0U || output == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *output = {begin, begin + bytes};
  return true;
}

[[nodiscard]] bool all_disjoint(const ByteRange* const ranges,
                                const std::size_t count) noexcept {
  for (std::size_t left = 0U; left < count; ++left) {
    for (std::size_t right = left + 1U; right < count; ++right) {
      if (ranges[left].begin < ranges[right].end &&
          ranges[right].begin < ranges[left].end) {
        return false;
      }
    }
  }
  return true;
}

struct alignas(16) PackedK256Stage final {
  std::uint8_t a[kK64PerStage][kTileM * kPackedK64Bytes];
  std::uint8_t b[kK64PerStage][kTileN * kPackedK64Bytes];
};

struct alignas(16) SharedStorage final {
  PackedK256Stage stage[
      kSm87A4W4DownFactorizedR4M192N128SharedStages];
};

// A warp owns one M16N128 output row.  The named fields keep the whole-lane
// S32 reduction and cross-lane FP32 result in registers without making either
// object address-taken.  The S32 object is reset after every K4352 lane.
struct alignas(16) ProjectionS32 final {
  Sm87A4W4Accumulator n0{};
  Sm87A4W4Accumulator n1{};
  Sm87A4W4Accumulator n2{};
  Sm87A4W4Accumulator n3{};
  Sm87A4W4Accumulator n4{};
  Sm87A4W4Accumulator n5{};
  Sm87A4W4Accumulator n6{};
  Sm87A4W4Accumulator n7{};
  Sm87A4W4Accumulator n8{};
  Sm87A4W4Accumulator n9{};
  Sm87A4W4Accumulator n10{};
  Sm87A4W4Accumulator n11{};
  Sm87A4W4Accumulator n12{};
  Sm87A4W4Accumulator n13{};
  Sm87A4W4Accumulator n14{};
  Sm87A4W4Accumulator n15{};
};

struct alignas(16) FloatAccumulator final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

struct alignas(16) ProjectionFp32 final {
  FloatAccumulator n0{};
  FloatAccumulator n1{};
  FloatAccumulator n2{};
  FloatAccumulator n3{};
  FloatAccumulator n4{};
  FloatAccumulator n5{};
  FloatAccumulator n6{};
  FloatAccumulator n7{};
  FloatAccumulator n8{};
  FloatAccumulator n9{};
  FloatAccumulator n10{};
  FloatAccumulator n11{};
  FloatAccumulator n12{};
  FloatAccumulator n13{};
  FloatAccumulator n14{};
  FloatAccumulator n15{};
};

static_assert(sizeof(PackedK256Stage) ==
              kSm87A4W4DownFactorizedR4M192N128StageBytes);
static_assert(sizeof(SharedStorage) ==
              kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes);
static_assert(sizeof(ProjectionS32) == 64U * sizeof(std::int32_t));
static_assert(sizeof(ProjectionFp32) == 64U * sizeof(float));

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

__device__ __forceinline__ void cp_async_commit() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.commit_group;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

template <unsigned int PendingGroups>
__device__ __forceinline__ void cp_async_wait_group() noexcept {
  static_assert(PendingGroups <= 2U);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  if constexpr (PendingGroups == 0U) {
    asm volatile("cp.async.wait_group 0;" : : : "memory");
  } else if constexpr (PendingGroups == 1U) {
    asm volatile("cp.async.wait_group 1;" : : : "memory");
  } else {
    asm volatile("cp.async.wait_group 2;" : : : "memory");
  }
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void issue_stage(
    PackedK256Stage& destination,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k256_stage) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      kTileM * kPackedK64Bytes / 16U;
  constexpr unsigned int kBVectorsPerPlane =
      kTileN * kPackedK64Bytes / 16U;
  constexpr unsigned int kAVectors =
      kK64PerStage * kAVectorsPerPlane;
  constexpr unsigned int kBVectors =
      kK64PerStage * kBVectorsPerPlane;
  constexpr unsigned int kVectors = kAVectors + kBVectors;
  constexpr unsigned int kIterations =
      (kVectors + kThreads - 1U) / kThreads;
  static_assert(kAVectorsPerPlane == 384U);
  static_assert(kBVectorsPerPlane == 256U);
  static_assert(kVectors == 2'560U);
  static_assert(kIterations == 7U);

#pragma unroll
  for (unsigned int iteration = 0U; iteration < kIterations; ++iteration) {
    const unsigned int vector = threadIdx.x + iteration * kThreads;
    if (vector < kVectors) {
      const bool is_a = vector < kAVectors;
      const unsigned int operand_vector =
          is_a ? vector : vector - kAVectors;
      const unsigned int vectors_per_plane =
          is_a ? kAVectorsPerPlane : kBVectorsPerPlane;
      const unsigned int plane = operand_vector / vectors_per_plane;
      const unsigned int vector_in_plane =
          operand_vector - plane * vectors_per_plane;
      const unsigned int row = vector_in_plane >> 1U;
      const unsigned int row_vector = vector_in_plane & 1U;
      const unsigned int byte_in_k64 = 16U * row_vector;
      const unsigned int outer_start =
          is_a ? m_tile_start : n_tile_start;
      const std::uint8_t* const packed = is_a ? packed_a : packed_b;
      std::uint8_t* const shared_plane =
          is_a ? destination.a[plane] : destination.b[plane];
      const unsigned int physical_k64_group =
          kK64PerStage * k256_stage + plane;
      cp_async_cg_16(
          shared_plane +
              sm87_a4w4_swizzled_k64_byte_offset(row, byte_in_k64),
          packed + sm87_a4w4_consumer_packed_offset(
                       outer_start + row, physical_k64_group,
                       byte_in_k64, kPhysicalK64Groups));
    }
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
      shared_a +
      sm87_a4w4_swizzled_k64_byte_offset(logical_row, logical_byte);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  Sm87A4W4AFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1), "=r"(fragment.x2),
        "=r"(fragment.x3)
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
      shared_b +
      sm87_a4w4_swizzled_k64_byte_offset(logical_row, logical_byte);
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
    const PackedK256Stage& stage,
    const unsigned int warp_m16,
    ProjectionS32& partial) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int local_m = warp_m16 * 16U;

#define Q3X_R4_DOWN_M192N128_MMA(N)                                      \
  do {                                                                    \
    const Sm87A4W4BFragment b = load_b_ldmatrix_x2(                       \
        stage.b[plane] + N##U * 8U * kPackedK64Bytes, lane);             \
    sm87_a4w4_mma_m16n8k64(partial.n##N, a, b);                          \
  } while (false)

#pragma unroll
  for (unsigned int plane = 0U; plane < kK64PerStage; ++plane) {
    const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
        stage.a[plane] + local_m * kPackedK64Bytes, lane);
    Q3X_R4_DOWN_M192N128_MMA(0);
    Q3X_R4_DOWN_M192N128_MMA(1);
    Q3X_R4_DOWN_M192N128_MMA(2);
    Q3X_R4_DOWN_M192N128_MMA(3);
    Q3X_R4_DOWN_M192N128_MMA(4);
    Q3X_R4_DOWN_M192N128_MMA(5);
    Q3X_R4_DOWN_M192N128_MMA(6);
    Q3X_R4_DOWN_M192N128_MMA(7);
    Q3X_R4_DOWN_M192N128_MMA(8);
    Q3X_R4_DOWN_M192N128_MMA(9);
    Q3X_R4_DOWN_M192N128_MMA(10);
    Q3X_R4_DOWN_M192N128_MMA(11);
    Q3X_R4_DOWN_M192N128_MMA(12);
    Q3X_R4_DOWN_M192N128_MMA(13);
    Q3X_R4_DOWN_M192N128_MMA(14);
    Q3X_R4_DOWN_M192N128_MMA(15);
  }

#undef Q3X_R4_DOWN_M192N128_MMA
}

template <unsigned int NFragment>
__device__ __forceinline__ void fold_lane_fragment(
    const Sm87A4W4Accumulator& partial,
    FloatAccumulator& cross,
    const std::uint16_t* const a_scales_bf16,
    const std::uint16_t* const b_scales_bf16,
    const unsigned int factor_lane,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int warp_m16) noexcept {
  static_assert(NFragment < 16U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int m0 = m_tile_start + warp_m16 * 16U + row0;
  const unsigned int m1 = m_tile_start + warp_m16 * 16U + row1;
  const unsigned int n0 =
      n_tile_start + NFragment * 8U + column0;
  const unsigned int n1 = n0 + 1U;
  const float a0 = decode_bf16(a_scales_bf16[
      sm87_a4w4_down_factorized_r4_m192n128_scale_offset(
          m0, factor_lane)]);
  const float a1 = decode_bf16(a_scales_bf16[
      sm87_a4w4_down_factorized_r4_m192n128_scale_offset(
          m1, factor_lane)]);
  const float b0 = decode_bf16(b_scales_bf16[
      sm87_a4w4_down_factorized_r4_m192n128_scale_offset(
          n0, factor_lane)]);
  const float b1 = decode_bf16(b_scales_bf16[
      sm87_a4w4_down_factorized_r4_m192n128_scale_offset(
          n1, factor_lane)]);
  cross.x0 = __fadd_rn(
      cross.x0,
      __fmul_rn(static_cast<float>(partial.x0), __fmul_rn(a0, b0)));
  cross.x1 = __fadd_rn(
      cross.x1,
      __fmul_rn(static_cast<float>(partial.x1), __fmul_rn(a0, b1)));
  cross.x2 = __fadd_rn(
      cross.x2,
      __fmul_rn(static_cast<float>(partial.x2), __fmul_rn(a1, b0)));
  cross.x3 = __fadd_rn(
      cross.x3,
      __fmul_rn(static_cast<float>(partial.x3), __fmul_rn(a1, b1)));
}

__device__ __forceinline__ void fold_lane(
    const ProjectionS32& partial,
    ProjectionFp32& cross,
    const std::uint16_t* const a_scales_bf16,
    const std::uint16_t* const b_scales_bf16,
    const unsigned int factor_lane,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int warp_m16) noexcept {
#define Q3X_R4_DOWN_M192N128_FOLD(N)                                     \
  fold_lane_fragment<N##U>(                                              \
      partial.n##N, cross.n##N, a_scales_bf16, b_scales_bf16,            \
      factor_lane, m_tile_start, n_tile_start, warp_m16)
  Q3X_R4_DOWN_M192N128_FOLD(0);
  Q3X_R4_DOWN_M192N128_FOLD(1);
  Q3X_R4_DOWN_M192N128_FOLD(2);
  Q3X_R4_DOWN_M192N128_FOLD(3);
  Q3X_R4_DOWN_M192N128_FOLD(4);
  Q3X_R4_DOWN_M192N128_FOLD(5);
  Q3X_R4_DOWN_M192N128_FOLD(6);
  Q3X_R4_DOWN_M192N128_FOLD(7);
  Q3X_R4_DOWN_M192N128_FOLD(8);
  Q3X_R4_DOWN_M192N128_FOLD(9);
  Q3X_R4_DOWN_M192N128_FOLD(10);
  Q3X_R4_DOWN_M192N128_FOLD(11);
  Q3X_R4_DOWN_M192N128_FOLD(12);
  Q3X_R4_DOWN_M192N128_FOLD(13);
  Q3X_R4_DOWN_M192N128_FOLD(14);
  Q3X_R4_DOWN_M192N128_FOLD(15);
#undef Q3X_R4_DOWN_M192N128_FOLD
}

template <unsigned int NFragment>
__device__ __forceinline__ void store_fragment(
    const FloatAccumulator& value,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int warp_m16,
    std::uint16_t* const output_bf16) noexcept {
  static_assert(NFragment < 16U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int m0 = m_tile_start + warp_m16 * 16U + row0;
  const unsigned int m1 = m_tile_start + warp_m16 * 16U + row1;
  const unsigned int n0 =
      n_tile_start + NFragment * 8U + column0;
  const unsigned int n1 = n0 + 1U;
  if (m0 < logical_token_count) {
    output_bf16[static_cast<std::size_t>(m0) *
                        kSm87A4W4DownFactorizedR4M192N128ModelOutput +
                    n0] = encode_bf16(value.x0);
    output_bf16[static_cast<std::size_t>(m0) *
                        kSm87A4W4DownFactorizedR4M192N128ModelOutput +
                    n1] = encode_bf16(value.x1);
  }
  if (m1 < logical_token_count) {
    output_bf16[static_cast<std::size_t>(m1) *
                        kSm87A4W4DownFactorizedR4M192N128ModelOutput +
                    n0] = encode_bf16(value.x2);
    output_bf16[static_cast<std::size_t>(m1) *
                        kSm87A4W4DownFactorizedR4M192N128ModelOutput +
                    n1] = encode_bf16(value.x3);
  }
}

__device__ __forceinline__ void store_output(
    const ProjectionFp32& value,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int warp_m16,
    std::uint16_t* const output_bf16) noexcept {
#define Q3X_R4_DOWN_M192N128_STORE(N)                                    \
  store_fragment<N##U>(                                                  \
      value.n##N, logical_token_count, m_tile_start, n_tile_start,        \
      warp_m16, output_bf16)
  Q3X_R4_DOWN_M192N128_STORE(0);
  Q3X_R4_DOWN_M192N128_STORE(1);
  Q3X_R4_DOWN_M192N128_STORE(2);
  Q3X_R4_DOWN_M192N128_STORE(3);
  Q3X_R4_DOWN_M192N128_STORE(4);
  Q3X_R4_DOWN_M192N128_STORE(5);
  Q3X_R4_DOWN_M192N128_STORE(6);
  Q3X_R4_DOWN_M192N128_STORE(7);
  Q3X_R4_DOWN_M192N128_STORE(8);
  Q3X_R4_DOWN_M192N128_STORE(9);
  Q3X_R4_DOWN_M192N128_STORE(10);
  Q3X_R4_DOWN_M192N128_STORE(11);
  Q3X_R4_DOWN_M192N128_STORE(12);
  Q3X_R4_DOWN_M192N128_STORE(13);
  Q3X_R4_DOWN_M192N128_STORE(14);
  Q3X_R4_DOWN_M192N128_STORE(15);
#undef Q3X_R4_DOWN_M192N128_STORE
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownFactorizedR4M192N128Threads,
                      kSm87A4W4DownFactorizedR4M192N128CtasPerSm)
void q3x_sm87_a4w4_down_factorized_r4_m192n128_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_lane_scales_bf16,
    const unsigned int logical_token_count,
    std::uint16_t* const output_bf16) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_storage);
  const unsigned int warp_m16 = threadIdx.x >> 5U;

  // N-major persistence lets consecutive work reuse one weight family.  The
  // ten M192 owners cover the exact P1920 launch surface; every CTA walks the
  // 400-cell space with the fixed sixteen-CTA stride.
  for (unsigned int work = blockIdx.x;
       work < kSm87A4W4DownFactorizedR4M192N128WorkTiles;
       work += gridDim.x) {
    const unsigned int n_tile =
        work / kSm87A4W4DownFactorizedR4M192N128MOwners;
    const unsigned int m_owner =
        work -
        n_tile * kSm87A4W4DownFactorizedR4M192N128MOwners;
    const unsigned int m_tile_start = m_owner * kTileM;
    const unsigned int n_tile_start = n_tile * kTileN;
    ProjectionFp32 cross{};
    ProjectionS32 partial{};

    issue_stage(shared.stage[0], packed_a, packed_b, m_tile_start,
                n_tile_start, 0U);
    issue_stage(shared.stage[1], packed_a, packed_b, m_tile_start,
                n_tile_start, 1U);
    issue_stage(shared.stage[2], packed_a, packed_b, m_tile_start,
                n_tile_start, 2U);
    cp_async_wait_group<2U>();
    __syncthreads();

    for (unsigned int stage = 0U;
         stage <
         kSm87A4W4DownFactorizedR4M192N128TotalK256Stages;
         ++stage) {
      accumulate_k256_stage(
          shared.stage[
              stage %
              kSm87A4W4DownFactorizedR4M192N128SharedStages],
          warp_m16, partial);
      if ((stage + 1U) %
              kSm87A4W4DownFactorizedR4M192N128StagesPerLane ==
          0U) {
        fold_lane(
            partial, cross, a_lane_scales_bf16,
            b_lane_scales_bf16,
            stage /
                kSm87A4W4DownFactorizedR4M192N128StagesPerLane,
            m_tile_start, n_tile_start, warp_m16);
        partial = {};
      }

      // All consumers retire the current slot before its next K256 payload
      // is published.  wait_group 2 sustains the steady three-stage window;
      // the final two iterations drain it with wait_group 1 then 0.
      __syncthreads();
      const unsigned int next_stage =
          stage +
          kSm87A4W4DownFactorizedR4M192N128SharedStages;
      if (next_stage <
          kSm87A4W4DownFactorizedR4M192N128TotalK256Stages) {
        issue_stage(
            shared.stage[
                stage %
                kSm87A4W4DownFactorizedR4M192N128SharedStages],
            packed_a, packed_b, m_tile_start, n_tile_start,
            next_stage);
        cp_async_wait_group<2U>();
      } else if (stage + 2U <
                 kSm87A4W4DownFactorizedR4M192N128TotalK256Stages) {
        cp_async_wait_group<1U>();
      } else if (stage + 1U <
                 kSm87A4W4DownFactorizedR4M192N128TotalK256Stages) {
        cp_async_wait_group<0U>();
      }
      __syncthreads();
    }

    store_output(cross, logical_token_count, m_tile_start, n_tile_start,
                 warp_m16, output_bf16);
    __syncthreads();
  }
}

namespace {

[[nodiscard]] int validate_target(
    cudaDeviceProp* const output = nullptr) noexcept {
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
          kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output != nullptr) {
    *output = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_factorized_r4_m192n128_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_factorized_r4_m192n128_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

struct ResourceCache final {
  Sm87A4W4DownFactorizedR4M192N128Resources resources{};
  int status{static_cast<int>(cudaErrorUnknown)};
};

[[nodiscard]] ResourceCache build_resource_cache() noexcept {
  ResourceCache cache{};
  cudaDeviceProp properties{};
  cache.status = validate_target(&properties);
  if (cache.status != static_cast<int>(cudaSuccess)) {
    return cache;
  }
  const cudaError_t configure_status = configure_kernel();
  if (configure_status != cudaSuccess) {
    cache.status = static_cast<int>(configure_status);
    return cache;
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_down_factorized_r4_m192n128_kernel);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_factorized_r4_m192n128_kernel,
      static_cast<int>(kSm87A4W4DownFactorizedR4M192N128Threads),
      kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }

  auto& resources = cache.resources;
  resources.registers_per_thread = attributes.numRegs;
  resources.static_shared_bytes = attributes.sharedSizeBytes;
  resources.dynamic_shared_bytes =
      kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes;
  resources.configured_dynamic_shared_limit_bytes =
      attributes.maxDynamicSharedSizeBytes;
  resources.device_optin_shared_limit_bytes =
      properties.sharedMemPerBlockOptin;
  resources.local_bytes = attributes.localSizeBytes;
  resources.maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources.active_blocks_per_sm = active_blocks;
  resources.resident_blocks = active_blocks * properties.multiProcessorCount;
  resources.multiprocessor_count = properties.multiProcessorCount;
  resources.compute_major = properties.major;
  resources.compute_minor = properties.minor;

  const bool passes =
      resources.registers_per_thread > 0 &&
      resources.registers_per_thread <=
          static_cast<int>(
              kSm87A4W4DownFactorizedR4M192N128MaximumRegisters) &&
      resources.static_shared_bytes == 0U &&
      resources.dynamic_shared_bytes ==
          kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes &&
      resources.configured_dynamic_shared_limit_bytes >=
          kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes &&
      resources.device_optin_shared_limit_bytes >=
          kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes &&
      resources.local_bytes == 0U &&
      resources.maximum_threads_per_block >=
          static_cast<int>(kSm87A4W4DownFactorizedR4M192N128Threads) &&
      resources.active_blocks_per_sm >=
          static_cast<int>(kSm87A4W4DownFactorizedR4M192N128CtasPerSm) &&
      resources.resident_blocks >=
          static_cast<int>(
              kSm87A4W4DownFactorizedR4M192N128PersistentCtas);
  cache.status = passes ? static_cast<int>(cudaSuccess)
                        : static_cast<int>(cudaErrorLaunchOutOfResources);
  return cache;
}

[[nodiscard]] const ResourceCache& resource_cache() noexcept {
  static const ResourceCache cache = build_resource_cache();
  return cache;
}

}  // namespace

int query_sm87_a4w4_down_factorized_r4_m192n128_resources_cuda(
    Sm87A4W4DownFactorizedR4M192N128Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ResourceCache& cache = resource_cache();
  *resources = cache.resources;
  return cache.status;
}

int launch_sm87_a4w4_down_factorized_r4_m192n128_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_lane_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4DownFactorizedR4M192N128Plan plan =
      sm87_a4w4_down_factorized_r4_m192n128_plan(
          logical_token_count, launch_token_count, output_size, input_size);
  if (!plan ||
      output_row_stride_elements !=
          kSm87A4W4DownFactorizedR4M192N128ModelOutput ||
      !aligned_16(packed_a) || !aligned_16(a_lane_scales_bf16) ||
      !aligned_16(packed_b) || !aligned_16(b_lane_scales_bf16) ||
      !aligned_16(output_bf16)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(
          kSm87A4W4DownFactorizedR4M192N128LaunchTokens,
          kSm87A4W4DownFactorizedR4M192N128ModelInput);
  const std::size_t required_a_scales =
      sm87_a4w4_down_factorized_r4_m192n128_scale_capacity_elements(
          kSm87A4W4DownFactorizedR4M192N128LaunchTokens);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(
          kSm87A4W4DownFactorizedR4M192N128ModelOutput,
          kSm87A4W4DownFactorizedR4M192N128ModelInput);
  const std::size_t required_b_scales =
      sm87_a4w4_down_factorized_r4_m192n128_scale_capacity_elements(
          kSm87A4W4DownFactorizedR4M192N128ModelOutput);
  const std::size_t required_output_elements =
      kSm87A4W4DownFactorizedR4M192N128LaunchTokens *
      kSm87A4W4DownFactorizedR4M192N128ModelOutput;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      packed_b_capacity_bytes < required_b_bytes ||
      b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  ByteRange ranges[5]{};
  const bool ranges_valid =
      make_range(packed_a, required_a_bytes, &ranges[0]) &&
      make_range(a_lane_scales_bf16,
                 required_a_scales * sizeof(std::uint16_t), &ranges[1]) &&
      make_range(packed_b, required_b_bytes, &ranges[2]) &&
      make_range(b_lane_scales_bf16,
                 required_b_scales * sizeof(std::uint16_t), &ranges[3]) &&
      make_range(output_bf16,
                 required_output_elements * sizeof(std::uint16_t),
                 &ranges[4]);
  if (!ranges_valid || !all_disjoint(ranges, 5U)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const ResourceCache& cache = resource_cache();
  if (cache.status != static_cast<int>(cudaSuccess)) {
    return cache.status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_factorized_r4_m192n128_kernel<<<
      kSm87A4W4DownFactorizedR4M192N128PersistentCtas,
      kSm87A4W4DownFactorizedR4M192N128Threads,
      kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes, stream>>>(
      packed_a, a_lane_scales_bf16, packed_b, b_lane_scales_bf16,
      static_cast<unsigned int>(logical_token_count), output_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels

#if defined(Q3X_SM87_A4W4_DOWN_R4_M192N128_RESOURCE_MAIN)
int main() {
  q3x::kernels::Sm87A4W4DownFactorizedR4M192N128Resources resources{};
  const int status =
      q3x::kernels::
          query_sm87_a4w4_down_factorized_r4_m192n128_resources_cuda(
              &resources);
  std::printf(
      "status=%d regs=%d local=%zu static_smem=%zu dynamic_smem=%zu "
      "configured_dynamic=%zu optin=%zu max_threads=%d active_per_sm=%d "
      "resident=%d sms=%d sm=%d%d\n",
      status, resources.registers_per_thread, resources.local_bytes,
      resources.static_shared_bytes, resources.dynamic_shared_bytes,
      resources.configured_dynamic_shared_limit_bytes,
      resources.device_optin_shared_limit_bytes,
      resources.maximum_threads_per_block, resources.active_blocks_per_sm,
      resources.resident_blocks, resources.multiprocessor_count,
      resources.compute_major, resources.compute_minor);
  return status == static_cast<int>(cudaSuccess) ? 0 : 1;
}
#endif
