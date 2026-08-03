#include "q3x/kernels/sm87_a4w4_gateup_factorized_r4_m64n64_2cta.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kTileM = static_cast<unsigned int>(
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileM);
inline constexpr unsigned int kTileN = static_cast<unsigned int>(
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileN);
inline constexpr unsigned int kWarpN = static_cast<unsigned int>(
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWarpN);
inline constexpr unsigned int kThreads = static_cast<unsigned int>(
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaThreads);
inline constexpr unsigned int kK64PerStage = static_cast<unsigned int>(
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaK64PerStage);
inline constexpr unsigned int kPackedK64Bytes = 32U;
inline constexpr unsigned int kPhysicalK64Groups =
    static_cast<unsigned int>(
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelInput /
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPhysicalK64);

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
  std::uint8_t gate[kK64PerStage][kTileN * kPackedK64Bytes];
  std::uint8_t up[kK64PerStage][kTileN * kPackedK64Bytes];
};

struct alignas(16) SharedCrossTail final {
  float value
      [kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossValuesPerThread]
      [kThreads];
};

struct alignas(16) SharedStorage final {
  PackedK256Stage
      stage[kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedStages];
  SharedCrossTail up_cross;
};

// One warp owns M16N32.  These 32 S32 registers cover both projections for
// the complete current R4 lane; no partial leaves registers before folding.
struct alignas(16) ProjectionS32 final {
  Sm87A4W4Accumulator n0{};
  Sm87A4W4Accumulator n1{};
  Sm87A4W4Accumulator n2{};
  Sm87A4W4Accumulator n3{};
};

struct alignas(16) PairedS32 final {
  ProjectionS32 gate;
  ProjectionS32 up;
};

struct alignas(16) FloatAccumulator final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

// Gate's complete FP32 cross-lane state remains registers.  Up's matching
// sixteen values live in SharedCrossTail, field-major by per-thread value.
struct alignas(16) ProjectionFp32 final {
  FloatAccumulator n0{};
  FloatAccumulator n1{};
  FloatAccumulator n2{};
  FloatAccumulator n3{};
};

static_assert(sizeof(PackedK256Stage) ==
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaStageBytes);
static_assert(sizeof(SharedCrossTail) ==
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossBytes);
static_assert(sizeof(SharedStorage) ==
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes);
static_assert(sizeof(PairedS32) == 32U * sizeof(std::int32_t));
static_assert(sizeof(ProjectionFp32) == 16U * sizeof(float));

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
  static_assert(PendingGroups <= 1U);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  if constexpr (PendingGroups == 0U) {
    asm volatile("cp.async.wait_group 0;" : : : "memory");
  } else {
    asm volatile("cp.async.wait_group 1;" : : : "memory");
  }
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void issue_stage(
    PackedK256Stage& destination,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate,
    const std::uint8_t* const packed_up,
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
  constexpr unsigned int kVectors = kAVectors + 2U * kBVectors;
  constexpr unsigned int kIterations = kVectors / kThreads;
  static_assert(kAVectorsPerPlane == 128U);
  static_assert(kBVectorsPerPlane == 128U);
  static_assert(kVectors == 1'536U);
  static_assert(kIterations == 6U);

#pragma unroll
  for (unsigned int iteration = 0U; iteration < kIterations; ++iteration) {
    const unsigned int vector = threadIdx.x + iteration * kThreads;
    const bool is_a = vector < kAVectors;
    const bool is_gate = !is_a && vector < kAVectors + kBVectors;
    const unsigned int operand_vector =
        is_a ? vector
             : (is_gate ? vector - kAVectors
                        : vector - kAVectors - kBVectors);
    const unsigned int vectors_per_plane =
        is_a ? kAVectorsPerPlane : kBVectorsPerPlane;
    const unsigned int plane = operand_vector / vectors_per_plane;
    const unsigned int vector_in_plane =
        operand_vector - plane * vectors_per_plane;
    const unsigned int row = vector_in_plane >> 1U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int byte_in_k64 = 16U * row_vector;
    const unsigned int outer_start = is_a ? m_tile_start : n_tile_start;
    const std::uint8_t* const packed =
        is_a ? packed_a : (is_gate ? packed_gate : packed_up);
    std::uint8_t* const shared_plane =
        is_a ? destination.a[plane]
             : (is_gate ? destination.gate[plane]
                        : destination.up[plane]);
    const unsigned int physical_k64_group =
        kK64PerStage * k256_stage + plane;
    cp_async_cg_16(
        shared_plane +
            sm87_a4w4_swizzled_k64_byte_offset(row, byte_in_k64),
        packed + sm87_a4w4_consumer_packed_offset(
                     outer_start + row, physical_k64_group,
                     byte_in_k64, kPhysicalK64Groups));
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
    const unsigned int warp_n32,
    PairedS32& partial) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int local_m = warp_m16 * 16U;
  const unsigned int local_n = warp_n32 * kWarpN;

#define Q3X_R4_M64N64_2CTA_MMA(N)                                        \
  do {                                                                    \
    const Sm87A4W4BFragment gate_b = load_b_ldmatrix_x2(                  \
        stage.gate[plane] + local_n * kPackedK64Bytes +                   \
            N##U * 8U * kPackedK64Bytes,                                  \
        lane);                                                            \
    sm87_a4w4_mma_m16n8k64(partial.gate.n##N, a, gate_b);                 \
    const Sm87A4W4BFragment up_b = load_b_ldmatrix_x2(                    \
        stage.up[plane] + local_n * kPackedK64Bytes +                     \
            N##U * 8U * kPackedK64Bytes,                                  \
        lane);                                                            \
    sm87_a4w4_mma_m16n8k64(partial.up.n##N, a, up_b);                     \
  } while (false)

#pragma unroll
  for (unsigned int plane = 0U; plane < kK64PerStage; ++plane) {
    const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
        stage.a[plane] + local_m * kPackedK64Bytes, lane);
    Q3X_R4_M64N64_2CTA_MMA(0);
    Q3X_R4_M64N64_2CTA_MMA(1);
    Q3X_R4_M64N64_2CTA_MMA(2);
    Q3X_R4_M64N64_2CTA_MMA(3);
  }

#undef Q3X_R4_M64N64_2CTA_MMA
}

__device__ __forceinline__ void clear_shared_cross_tail(
    SharedCrossTail& cross_tail) noexcept {
#pragma unroll
  for (unsigned int value = 0U;
       value <
       kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossValuesPerThread;
       ++value) {
    cross_tail.value[value][threadIdx.x] = 0.0F;
  }
}

template <unsigned int Base>
[[nodiscard]] __device__ __forceinline__ FloatAccumulator
load_shared_cross_tail(const SharedCrossTail& cross_tail) noexcept {
  static_assert(
      Base + 3U <
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossValuesPerThread);
  return {cross_tail.value[Base + 0U][threadIdx.x],
          cross_tail.value[Base + 1U][threadIdx.x],
          cross_tail.value[Base + 2U][threadIdx.x],
          cross_tail.value[Base + 3U][threadIdx.x]};
}

template <unsigned int Base>
__device__ __forceinline__ void store_shared_cross_tail(
    SharedCrossTail& cross_tail,
    const FloatAccumulator& accumulator) noexcept {
  static_assert(
      Base + 3U <
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossValuesPerThread);
  cross_tail.value[Base + 0U][threadIdx.x] = accumulator.x0;
  cross_tail.value[Base + 1U][threadIdx.x] = accumulator.x1;
  cross_tail.value[Base + 2U][threadIdx.x] = accumulator.x2;
  cross_tail.value[Base + 3U][threadIdx.x] = accumulator.x3;
}

template <unsigned int NFragment>
__device__ __forceinline__ void fold_lane_fragment(
    const Sm87A4W4Accumulator& gate_partial,
    const Sm87A4W4Accumulator& up_partial,
    FloatAccumulator& gate_cross,
    FloatAccumulator& up_cross,
    const std::uint16_t* const a_scales_bf16,
    const std::uint16_t* const gate_scales_bf16,
    const std::uint16_t* const up_scales_bf16,
    const unsigned int factor_lane,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int warp_m16,
    const unsigned int warp_n32) noexcept {
  static_assert(NFragment < 4U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int m0 = m_tile_start + warp_m16 * 16U + row0;
  const unsigned int m1 = m_tile_start + warp_m16 * 16U + row1;
  const unsigned int n0 =
      n_tile_start + warp_n32 * kWarpN + NFragment * 8U + column0;
  const unsigned int n1 = n0 + 1U;
  const float a0 = decode_bf16(a_scales_bf16[
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(
          m0, factor_lane)]);
  const float a1 = decode_bf16(a_scales_bf16[
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(
          m1, factor_lane)]);
  const float gate_b0 = decode_bf16(gate_scales_bf16[
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(
          n0, factor_lane)]);
  const float gate_b1 = decode_bf16(gate_scales_bf16[
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(
          n1, factor_lane)]);
  const float up_b0 = decode_bf16(up_scales_bf16[
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(
          n0, factor_lane)]);
  const float up_b1 = decode_bf16(up_scales_bf16[
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(
          n1, factor_lane)]);

#define Q3X_R4_M64N64_2CTA_FOLD(CROSS, PARTIAL, ASCALE, BSCALE)           \
  CROSS = __fadd_rn(                                                      \
      CROSS, __fmul_rn(static_cast<float>(PARTIAL),                       \
                       __fmul_rn(ASCALE, BSCALE)))
  Q3X_R4_M64N64_2CTA_FOLD(gate_cross.x0, gate_partial.x0, a0, gate_b0);
  Q3X_R4_M64N64_2CTA_FOLD(gate_cross.x1, gate_partial.x1, a0, gate_b1);
  Q3X_R4_M64N64_2CTA_FOLD(gate_cross.x2, gate_partial.x2, a1, gate_b0);
  Q3X_R4_M64N64_2CTA_FOLD(gate_cross.x3, gate_partial.x3, a1, gate_b1);
  Q3X_R4_M64N64_2CTA_FOLD(up_cross.x0, up_partial.x0, a0, up_b0);
  Q3X_R4_M64N64_2CTA_FOLD(up_cross.x1, up_partial.x1, a0, up_b1);
  Q3X_R4_M64N64_2CTA_FOLD(up_cross.x2, up_partial.x2, a1, up_b0);
  Q3X_R4_M64N64_2CTA_FOLD(up_cross.x3, up_partial.x3, a1, up_b1);
#undef Q3X_R4_M64N64_2CTA_FOLD
}

__device__ __forceinline__ void fold_lane(
    const PairedS32& partial,
    ProjectionFp32& gate_cross,
    SharedCrossTail& up_cross_tail,
    const std::uint16_t* const a_scales_bf16,
    const std::uint16_t* const gate_scales_bf16,
    const std::uint16_t* const up_scales_bf16,
    const unsigned int factor_lane,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int warp_m16,
    const unsigned int warp_n32) noexcept {
#define Q3X_R4_M64N64_2CTA_FOLD_FRAGMENT(N)                              \
  do {                                                                    \
    FloatAccumulator up_cross =                                          \
        load_shared_cross_tail<(N##U * 4U)>(up_cross_tail);              \
    fold_lane_fragment<N##U>(                                            \
        partial.gate.n##N, partial.up.n##N, gate_cross.n##N, up_cross,   \
        a_scales_bf16, gate_scales_bf16, up_scales_bf16, factor_lane,    \
        m_tile_start, n_tile_start, warp_m16, warp_n32);                 \
    store_shared_cross_tail<(N##U * 4U)>(up_cross_tail, up_cross);       \
  } while (false)
  Q3X_R4_M64N64_2CTA_FOLD_FRAGMENT(0);
  Q3X_R4_M64N64_2CTA_FOLD_FRAGMENT(1);
  Q3X_R4_M64N64_2CTA_FOLD_FRAGMENT(2);
  Q3X_R4_M64N64_2CTA_FOLD_FRAGMENT(3);
#undef Q3X_R4_M64N64_2CTA_FOLD_FRAGMENT
}

template <unsigned int NFragment>
__device__ __forceinline__ void store_fragment(
    const FloatAccumulator& gate,
    const FloatAccumulator& up,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int output_n_tile_start,
    const unsigned int warp_m16,
    const unsigned int warp_n32,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  static_assert(NFragment < 4U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int m0 = m_tile_start + warp_m16 * 16U + row0;
  const unsigned int m1 = m_tile_start + warp_m16 * 16U + row1;
  const unsigned int n0 = output_n_tile_start + warp_n32 * kWarpN +
                          NFragment * 8U + column0;
  const unsigned int n1 = n0 + 1U;
  if (m0 < logical_token_count) {
    output_bf16[static_cast<std::size_t>(m0) *
                        output_row_stride_elements +
                    n0] = encode_bf16(silu_product(gate.x0, up.x0));
    output_bf16[static_cast<std::size_t>(m0) *
                        output_row_stride_elements +
                    n1] = encode_bf16(silu_product(gate.x1, up.x1));
  }
  if (m1 < logical_token_count) {
    output_bf16[static_cast<std::size_t>(m1) *
                        output_row_stride_elements +
                    n0] = encode_bf16(silu_product(gate.x2, up.x2));
    output_bf16[static_cast<std::size_t>(m1) *
                        output_row_stride_elements +
                    n1] = encode_bf16(silu_product(gate.x3, up.x3));
  }
}

__device__ __forceinline__ void store_output(
    const ProjectionFp32& gate_cross,
    const SharedCrossTail& up_cross_tail,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int output_n_tile_start,
    const unsigned int warp_m16,
    const unsigned int warp_n32,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
#define Q3X_R4_M64N64_2CTA_STORE(N)                                      \
  do {                                                                    \
    const FloatAccumulator up_cross =                                    \
        load_shared_cross_tail<(N##U * 4U)>(up_cross_tail);              \
    store_fragment<N##U>(gate_cross.n##N, up_cross,                      \
                         logical_token_count, m_tile_start,              \
                         output_n_tile_start, warp_m16, warp_n32,        \
                         output_bf16, output_row_stride_elements);       \
  } while (false)
  Q3X_R4_M64N64_2CTA_STORE(0);
  Q3X_R4_M64N64_2CTA_STORE(1);
  Q3X_R4_M64N64_2CTA_STORE(2);
  Q3X_R4_M64N64_2CTA_STORE(3);
#undef Q3X_R4_M64N64_2CTA_STORE
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaThreads,
                      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaCtasPerSm)
void q3x_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::uint8_t* const packed_gate,
    const std::uint16_t* const gate_lane_scales_bf16,
    const std::uint8_t* const packed_up,
    const std::uint16_t* const up_lane_scales_bf16,
    const unsigned int logical_token_count,
    std::uint16_t* const primary_output_bf16,
    std::uint16_t* const secondary_output_bf16) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_storage);
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m16 = warp >> 1U;
  const unsigned int warp_n32 = warp & 1U;

  for (unsigned int work = blockIdx.x;
       work < kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWorkTiles;
       work += gridDim.x) {
    const unsigned int n_tile =
        work / kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMOwners;
    const unsigned int m_owner =
        work - n_tile *
                   kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMOwners;
    const unsigned int m_tile_start = m_owner * kTileM;
    const unsigned int n_tile_start = n_tile * kTileN;
    ProjectionFp32 gate_cross{};
    PairedS32 partial{};
    clear_shared_cross_tail(shared.up_cross);

    issue_stage(shared.stage[0], packed_a, packed_gate, packed_up,
                m_tile_start, n_tile_start, 0U);
    issue_stage(shared.stage[1], packed_a, packed_gate, packed_up,
                m_tile_start, n_tile_start, 1U);
    cp_async_wait_group<1U>();
    __syncthreads();

    for (unsigned int stage = 0U;
         stage <
         kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTotalK256Stages;
         ++stage) {
      accumulate_k256_stage(
          shared.stage[
              stage %
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedStages],
          warp_m16, warp_n32, partial);
      if ((stage + 1U) %
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaStagesPerLane ==
          0U) {
        fold_lane(
            partial, gate_cross, shared.up_cross, a_lane_scales_bf16,
            gate_lane_scales_bf16, up_lane_scales_bf16,
            stage /
                kSm87A4W4GateUpFactorizedR4M64N64TwoCtaStagesPerLane,
            m_tile_start, n_tile_start, warp_m16, warp_n32);
        partial = {};
      }

      __syncthreads();
      const unsigned int next_stage =
          stage +
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedStages;
      if (next_stage <
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTotalK256Stages) {
        issue_stage(
            shared.stage[
                stage %
                kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedStages],
            packed_a, packed_gate, packed_up, m_tile_start, n_tile_start,
            next_stage);
        cp_async_wait_group<1U>();
      } else if (stage + 1U <
                 kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTotalK256Stages) {
        cp_async_wait_group<0U>();
      }
      __syncthreads();
    }

    const bool primary =
        n_tile_start <
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPrimaryWidth;
    std::uint16_t* const output =
        primary ? primary_output_bf16 : secondary_output_bf16;
    const unsigned int output_stride = static_cast<unsigned int>(
        primary
            ? kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPrimaryStride
            : kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSecondaryStride);
    const unsigned int output_n_tile_start =
        primary
            ? n_tile_start
            : n_tile_start - static_cast<unsigned int>(
                                 kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPrimaryWidth);
    store_output(gate_cross, shared.up_cross, logical_token_count,
                 m_tile_start, output_n_tile_start, warp_m16, warp_n32,
                 output, output_stride);
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
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output != nullptr) {
    *output = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

struct ResourceCache final {
  Sm87A4W4GateUpFactorizedR4M64N64TwoCtaResources resources{};
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
      q3x_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_kernel);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_kernel,
      static_cast<int>(
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaThreads),
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }

  auto& resources = cache.resources;
  resources.registers_per_thread = attributes.numRegs;
  resources.static_shared_bytes = attributes.sharedSizeBytes;
  resources.dynamic_shared_bytes =
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes;
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
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMaximumRegisters) &&
      resources.static_shared_bytes == 0U &&
      resources.dynamic_shared_bytes ==
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes &&
      resources.configured_dynamic_shared_limit_bytes >=
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes &&
      resources.device_optin_shared_limit_bytes >=
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes &&
      resources.local_bytes == 0U &&
      resources.maximum_threads_per_block >=
          static_cast<int>(
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaThreads) &&
      resources.active_blocks_per_sm >=
          static_cast<int>(
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaCtasPerSm) &&
      resources.resident_blocks >=
          static_cast<int>(
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPersistentCtas);
  cache.status = passes ? static_cast<int>(cudaSuccess)
                        : static_cast<int>(cudaErrorLaunchOutOfResources);
  return cache;
}

[[nodiscard]] const ResourceCache& resource_cache() noexcept {
  static const ResourceCache cache = build_resource_cache();
  return cache;
}

}  // namespace

int query_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_resources_cuda(
    Sm87A4W4GateUpFactorizedR4M64N64TwoCtaResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ResourceCache& cache = resource_cache();
  *resources = cache.resources;
  return cache.status;
}

int launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate,
    const std::size_t packed_gate_capacity_bytes,
    const std::uint16_t* const gate_lane_scales_bf16,
    const std::size_t gate_scale_capacity_elements,
    const std::uint8_t* const packed_up,
    const std::size_t packed_up_capacity_bytes,
    const std::uint16_t* const up_lane_scales_bf16,
    const std::size_t up_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    std::uint16_t* const primary_output_bf16,
    const std::size_t primary_output_row_stride_elements,
    const std::size_t primary_output_capacity_elements,
    std::uint16_t* const secondary_output_bf16,
    const std::size_t secondary_output_row_stride_elements,
    const std::size_t secondary_output_capacity_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpFactorizedR4M64N64TwoCtaPlan plan =
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_plan(
          logical_token_count, launch_token_count, intermediate_size,
          input_size);
  if (!plan ||
      primary_output_row_stride_elements !=
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPrimaryStride ||
      secondary_output_row_stride_elements !=
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSecondaryStride ||
      !aligned_16(packed_a) || !aligned_16(a_lane_scales_bf16) ||
      !aligned_16(packed_gate) || !aligned_16(gate_lane_scales_bf16) ||
      !aligned_16(packed_up) || !aligned_16(up_lane_scales_bf16) ||
      !aligned_16(primary_output_bf16) ||
      !aligned_16(secondary_output_bf16)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaunchTokens,
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelInput);
  const std::size_t required_a_scales =
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_capacity_elements(
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaunchTokens);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelIntermediate,
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelInput);
  const std::size_t required_b_scales =
      sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_capacity_elements(
          kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelIntermediate);
  const std::size_t required_primary_elements =
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaunchTokens *
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPrimaryStride;
  const std::size_t required_secondary_elements =
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaunchTokens *
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSecondaryStride;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      packed_gate_capacity_bytes < required_b_bytes ||
      gate_scale_capacity_elements < required_b_scales ||
      packed_up_capacity_bytes < required_b_bytes ||
      up_scale_capacity_elements < required_b_scales ||
      primary_output_capacity_elements < required_primary_elements ||
      secondary_output_capacity_elements < required_secondary_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  ByteRange ranges[8]{};
  const bool ranges_valid =
      make_range(packed_a, required_a_bytes, &ranges[0]) &&
      make_range(a_lane_scales_bf16,
                 required_a_scales * sizeof(std::uint16_t), &ranges[1]) &&
      make_range(packed_gate, required_b_bytes, &ranges[2]) &&
      make_range(gate_lane_scales_bf16,
                 required_b_scales * sizeof(std::uint16_t), &ranges[3]) &&
      make_range(packed_up, required_b_bytes, &ranges[4]) &&
      make_range(up_lane_scales_bf16,
                 required_b_scales * sizeof(std::uint16_t), &ranges[5]) &&
      make_range(primary_output_bf16,
                 required_primary_elements * sizeof(std::uint16_t),
                 &ranges[6]) &&
      make_range(secondary_output_bf16,
                 required_secondary_elements * sizeof(std::uint16_t),
                 &ranges[7]);
  if (!ranges_valid || !all_disjoint(ranges, 8U)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const ResourceCache& cache = resource_cache();
  if (cache.status != static_cast<int>(cudaSuccess)) {
    return cache.status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_kernel<<<
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPersistentCtas,
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaThreads,
      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes,
      stream>>>(
      packed_a, a_lane_scales_bf16, packed_gate,
      gate_lane_scales_bf16, packed_up, up_lane_scales_bf16,
      static_cast<unsigned int>(logical_token_count), primary_output_bf16,
      secondary_output_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
