#include "q3x/kernels/sm87_a4w4_gateup_factorized_lane.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kTileM =
    static_cast<unsigned int>(kSm87A4W4GateUpFactorizedTileM);
inline constexpr unsigned int kTileN =
    static_cast<unsigned int>(kSm87A4W4GateUpFactorizedTileN);
inline constexpr unsigned int kK64PerStage =
    static_cast<unsigned int>(kSm87A4W4GateUpFactorizedK64PerStage);
inline constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87A4W4GateUpFactorizedThreads);
inline constexpr unsigned int kPackedK64Bytes = 32U;

struct alignas(16) PackedStage final {
  std::uint8_t a[kK64PerStage][kTileM * kPackedK64Bytes];
  std::uint8_t gate[kK64PerStage][kTileN * kPackedK64Bytes];
  std::uint8_t up[kK64PerStage][kTileN * kPackedK64Bytes];
};

struct alignas(16) FactorizedScales final {
  std::uint16_t a[kTileM];
  std::uint16_t gate[kTileN];
  std::uint16_t up[kTileN];
};

struct alignas(16) SharedStorage final {
  PackedStage stage[kSm87A4W4GateUpFactorizedStages];
  FactorizedScales scales;
};

// One M32N32 warp owns eight m16n8 fragments per projection.  Gate and Up
// stay as 64 named S32 registers/thread for the complete K reduction.
struct alignas(16) ProjectionAccumulators final {
  Sm87A4W4Accumulator m0n0{};
  Sm87A4W4Accumulator m0n1{};
  Sm87A4W4Accumulator m0n2{};
  Sm87A4W4Accumulator m0n3{};
  Sm87A4W4Accumulator m1n0{};
  Sm87A4W4Accumulator m1n1{};
  Sm87A4W4Accumulator m1n2{};
  Sm87A4W4Accumulator m1n3{};
};

struct alignas(16) PairedAccumulators final {
  ProjectionAccumulators gate;
  ProjectionAccumulators up;
};

static_assert(sizeof(PackedStage) ==
              kSm87A4W4GateUpFactorizedStageBytes);
static_assert(sizeof(FactorizedScales) ==
              kSm87A4W4GateUpFactorizedScaleBytes);
static_assert(sizeof(SharedStorage) ==
              kSm87A4W4GateUpFactorizedDynamicSharedBytes);
static_assert(sizeof(ProjectionAccumulators) ==
              32U * sizeof(std::int32_t));
static_assert(sizeof(PairedAccumulators) ==
              64U * sizeof(std::int32_t));

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

__device__ __forceinline__ void issue_stage(
    PackedStage& destination, const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate,
    const std::uint8_t* const packed_up,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k256_stage,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      kTileM * kPackedK64Bytes / 16U;
  constexpr unsigned int kIterations =
      kK64PerStage * kVectorsPerPlane / kThreads;
  static_assert(kVectorsPerPlane == 256U);
  static_assert(kIterations == 2U);

#pragma unroll
  for (unsigned int iteration = 0U; iteration < kIterations; ++iteration) {
    const unsigned int vector = threadIdx.x + iteration * kThreads;
    const unsigned int plane = vector / kVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kVectorsPerPlane;
    const unsigned int row = vector_in_plane >> 1U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int physical_k64_group =
        kK64PerStage * k256_stage + plane;
    const unsigned int byte_in_row = 16U * row_vector;
    const std::size_t shared_offset =
        sm87_a4w4_swizzled_k64_byte_offset(row, byte_in_row);

    cp_async_cg_16(
        destination.a[plane] + shared_offset,
        packed_a + sm87_a4w4_gateup_factorized_packed_offset(
                       m_tile_start + row, physical_k64_group,
                       byte_in_row, physical_k64_group_count));
    cp_async_cg_16(
        destination.gate[plane] + shared_offset,
        packed_gate + sm87_a4w4_gateup_factorized_packed_offset(
                            n_tile_start + row, physical_k64_group,
                            byte_in_row, physical_k64_group_count));
    cp_async_cg_16(
        destination.up[plane] + shared_offset,
        packed_up + sm87_a4w4_gateup_factorized_packed_offset(
                          n_tile_start + row, physical_k64_group,
                          byte_in_row, physical_k64_group_count));
  }
  cp_async_commit();
}

__device__ __forceinline__ void issue_scales(
    FactorizedScales& destination,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::uint16_t* const gate_lane_scales_bf16,
    const std::uint16_t* const up_lane_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start) noexcept {
  if (threadIdx.x < 48U) {
    const unsigned int operand = threadIdx.x >> 4U;
    const unsigned int vector = threadIdx.x & 15U;
    const unsigned int row = 8U * vector;
    const bool is_a = operand == 0U;
    const std::uint16_t* const source_base =
        is_a ? a_lane_scales_bf16
             : (operand == 1U ? gate_lane_scales_bf16
                              : up_lane_scales_bf16);
    std::uint16_t* const target_base =
        is_a ? destination.a
             : (operand == 1U ? destination.gate : destination.up);
    const unsigned int outer_start =
        is_a ? m_tile_start : n_tile_start;
    cp_async_ca_16(
        target_base + row,
        source_base + sm87_a4w4_gateup_factorized_scale_offset(
                          outer_start + row));
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
    const PackedStage& stage, const unsigned int warp_m32,
    const unsigned int warp_n32,
    PairedAccumulators& accumulators) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int local_m = warp_m32 * 32U;
  const unsigned int local_n = warp_n32 * 32U;

#pragma unroll
  for (unsigned int plane = 0U; plane < kK64PerStage; ++plane) {
    const Sm87A4W4AFragment a0 = load_a_ldmatrix_x4(
        stage.a[plane] + (local_m + 0U) * kPackedK64Bytes, lane);
    const Sm87A4W4AFragment a1 = load_a_ldmatrix_x4(
        stage.a[plane] + (local_m + 16U) * kPackedK64Bytes, lane);

    const Sm87A4W4BFragment gate0 = load_b_ldmatrix_x2(
        stage.gate[plane] + (local_n + 0U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.gate.m0n0, a0, gate0);
    sm87_a4w4_mma_m16n8k64(accumulators.gate.m1n0, a1, gate0);
    const Sm87A4W4BFragment up0 = load_b_ldmatrix_x2(
        stage.up[plane] + (local_n + 0U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.up.m0n0, a0, up0);
    sm87_a4w4_mma_m16n8k64(accumulators.up.m1n0, a1, up0);

    const Sm87A4W4BFragment gate1 = load_b_ldmatrix_x2(
        stage.gate[plane] + (local_n + 8U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.gate.m0n1, a0, gate1);
    sm87_a4w4_mma_m16n8k64(accumulators.gate.m1n1, a1, gate1);
    const Sm87A4W4BFragment up1 = load_b_ldmatrix_x2(
        stage.up[plane] + (local_n + 8U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.up.m0n1, a0, up1);
    sm87_a4w4_mma_m16n8k64(accumulators.up.m1n1, a1, up1);

    const Sm87A4W4BFragment gate2 = load_b_ldmatrix_x2(
        stage.gate[plane] + (local_n + 16U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.gate.m0n2, a0, gate2);
    sm87_a4w4_mma_m16n8k64(accumulators.gate.m1n2, a1, gate2);
    const Sm87A4W4BFragment up2 = load_b_ldmatrix_x2(
        stage.up[plane] + (local_n + 16U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.up.m0n2, a0, up2);
    sm87_a4w4_mma_m16n8k64(accumulators.up.m1n2, a1, up2);

    const Sm87A4W4BFragment gate3 = load_b_ldmatrix_x2(
        stage.gate[plane] + (local_n + 24U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.gate.m0n3, a0, gate3);
    sm87_a4w4_mma_m16n8k64(accumulators.gate.m1n3, a1, gate3);
    const Sm87A4W4BFragment up3 = load_b_ldmatrix_x2(
        stage.up[plane] + (local_n + 24U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.up.m0n3, a0, up3);
    sm87_a4w4_mma_m16n8k64(accumulators.up.m1n3, a1, up3);
  }
}

template <unsigned int MFragment, unsigned int NFragment,
          bool TrackR1TileMaximum>
__device__ __forceinline__ void store_fragment(
    const Sm87A4W4Accumulator& gate,
    const Sm87A4W4Accumulator& up,
    const FactorizedScales& scales,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int output_n_tile_start,
    const unsigned int global_n_tile_start,
    const unsigned int warp_m32,
    const unsigned int warp_n32,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const float* const down_inverse_alpha_fp32,
    float* const warp_tile_maxima) noexcept {
  static_assert(MFragment < 2U);
  static_assert(NFragment < 4U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int column1 = column0 + 1U;
  const unsigned int local_m =
      warp_m32 * 32U + MFragment * 16U;
  const unsigned int local_n =
      warp_n32 * 32U + NFragment * 8U;
  const unsigned int base_m = m_tile_start + local_m;
  const unsigned int base_n = output_n_tile_start + local_n;
  const unsigned int global_n = global_n_tile_start + local_n;
  const float a0 = decode_bf16(scales.a[local_m + row0]);
  const float a1 = decode_bf16(scales.a[local_m + row1]);
  const float gate_b0 = decode_bf16(scales.gate[local_n + column0]);
  const float gate_b1 = decode_bf16(scales.gate[local_n + column1]);
  const float up_b0 = decode_bf16(scales.up[local_n + column0]);
  const float up_b1 = decode_bf16(scales.up[local_n + column1]);

  float maximum0 = 0.0F;
  float maximum1 = 0.0F;
  if (base_m + row0 < logical_token_count) {
    const float gate0 = __fmul_rn(static_cast<float>(gate.x0),
                                  __fmul_rn(a0, gate_b0));
    const float up0 = __fmul_rn(static_cast<float>(up.x0),
                                __fmul_rn(a0, up_b0));
    const std::uint16_t product0 =
        encode_bf16(silu_product(gate0, up0));
    output_bf16[static_cast<std::size_t>(base_m + row0) *
                        output_row_stride_elements +
                    base_n + column0] = product0;
    if constexpr (TrackR1TileMaximum) {
      maximum0 = fmaxf(
          maximum0,
          fabsf(decode_bf16(product0) *
                down_inverse_alpha_fp32[global_n + column0]));
    }
    const float gate1 = __fmul_rn(static_cast<float>(gate.x1),
                                  __fmul_rn(a0, gate_b1));
    const float up1 = __fmul_rn(static_cast<float>(up.x1),
                                __fmul_rn(a0, up_b1));
    const std::uint16_t product1 =
        encode_bf16(silu_product(gate1, up1));
    output_bf16[static_cast<std::size_t>(base_m + row0) *
                        output_row_stride_elements +
                    base_n + column1] = product1;
    if constexpr (TrackR1TileMaximum) {
      maximum0 = fmaxf(
          maximum0,
          fabsf(decode_bf16(product1) *
                down_inverse_alpha_fp32[global_n + column1]));
    }
  }
  if (base_m + row1 < logical_token_count) {
    const float gate0 = __fmul_rn(static_cast<float>(gate.x2),
                                  __fmul_rn(a1, gate_b0));
    const float up0 = __fmul_rn(static_cast<float>(up.x2),
                                __fmul_rn(a1, up_b0));
    const std::uint16_t product0 =
        encode_bf16(silu_product(gate0, up0));
    output_bf16[static_cast<std::size_t>(base_m + row1) *
                        output_row_stride_elements +
                    base_n + column0] = product0;
    if constexpr (TrackR1TileMaximum) {
      maximum1 = fmaxf(
          maximum1,
          fabsf(decode_bf16(product0) *
                down_inverse_alpha_fp32[global_n + column0]));
    }
    const float gate1 = __fmul_rn(static_cast<float>(gate.x3),
                                  __fmul_rn(a1, gate_b1));
    const float up1 = __fmul_rn(static_cast<float>(up.x3),
                                __fmul_rn(a1, up_b1));
    const std::uint16_t product1 =
        encode_bf16(silu_product(gate1, up1));
    output_bf16[static_cast<std::size_t>(base_m + row1) *
                        output_row_stride_elements +
                    base_n + column1] = product1;
    if constexpr (TrackR1TileMaximum) {
      maximum1 = fmaxf(
          maximum1,
          fabsf(decode_bf16(product1) *
                down_inverse_alpha_fp32[global_n + column1]));
    }
  }

  if constexpr (TrackR1TileMaximum) {
    // Four adjacent lanes own one row's eight values in this N8 fragment.
    // Their NFragment calls are serialized within the warp, so the leader
    // can update a shared warp-local maximum without atomics.
    constexpr unsigned int kFullMask = 0xffff'ffffU;
    maximum0 = fmaxf(maximum0,
                     __shfl_xor_sync(kFullMask, maximum0, 1U));
    maximum0 = fmaxf(maximum0,
                     __shfl_xor_sync(kFullMask, maximum0, 2U));
    maximum1 = fmaxf(maximum1,
                     __shfl_xor_sync(kFullMask, maximum1, 1U));
    maximum1 = fmaxf(maximum1,
                     __shfl_xor_sync(kFullMask, maximum1, 2U));
    if ((lane & 3U) == 0U) {
      const unsigned int warp_maximum_base = warp_n32 * kTileM;
      const unsigned int row_index0 = local_m + row0;
      const unsigned int row_index1 = local_m + row1;
      warp_tile_maxima[warp_maximum_base + row_index0] =
          fmaxf(warp_tile_maxima[warp_maximum_base + row_index0],
                maximum0);
      warp_tile_maxima[warp_maximum_base + row_index1] =
          fmaxf(warp_tile_maxima[warp_maximum_base + row_index1],
                maximum1);
    }
  }
}

template <bool TrackR1TileMaximum>
__device__ __forceinline__ void store_accumulators(
    const PairedAccumulators& accumulators,
    const FactorizedScales& scales,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int output_n_tile_start,
    const unsigned int global_n_tile_start,
    const unsigned int warp_m32,
    const unsigned int warp_n32,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const float* const down_inverse_alpha_fp32,
    float* const warp_tile_maxima) noexcept {
#define Q3X_STORE(M, N)                                                    \
  store_fragment<M##U, N##U, TrackR1TileMaximum>(                         \
      accumulators.gate.m##M##n##N, accumulators.up.m##M##n##N, scales,  \
      logical_token_count, m_tile_start, output_n_tile_start,              \
      global_n_tile_start, warp_m32, warp_n32, output_bf16,                \
      output_row_stride_elements, down_inverse_alpha_fp32,                 \
      warp_tile_maxima)
  Q3X_STORE(0, 0);
  Q3X_STORE(0, 1);
  Q3X_STORE(0, 2);
  Q3X_STORE(0, 3);
  Q3X_STORE(1, 0);
  Q3X_STORE(1, 1);
  Q3X_STORE(1, 2);
  Q3X_STORE(1, 3);
#undef Q3X_STORE
}

template <bool TrackR1TileMaximum>
__device__ __forceinline__ void gateup_factorized_lane_body(
    SharedStorage& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_lane_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_lane_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int primary_width,
    const unsigned int physical_k64_group_count,
    const unsigned int k256_stage_count,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count,
    std::uint16_t* const primary_output_bf16,
    const unsigned int primary_output_row_stride_elements,
    std::uint16_t* const secondary_output_bf16,
    const unsigned int secondary_output_row_stride_elements,
    const float* const down_inverse_alpha_fp32,
    float* const r1_product_tile_maxima_fp32) noexcept {
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m32 = warp & 3U;
  const unsigned int warp_n32 = warp >> 2U;

  // N-major persistent order keeps each CTA wave on one adjacent family of
  // Gate/Up weight tiles.  A full slice, including the ABI plane boundary,
  // is covered by this one grid.
  for (unsigned int work = blockIdx.x; work < work_tile_count;
       work += gridDim.x) {
    const unsigned int n_tile = work / m_tile_count;
    const unsigned int m_tile = work - n_tile * m_tile_count;
    const unsigned int m_tile_start = m_tile * kTileM;
    const unsigned int n_tile_start = n_tile * kTileN;
    PairedAccumulators accumulators{};

    issue_scales(shared.scales, a_lane_scales_bf16,
                 gate_b_lane_scales_bf16, up_b_lane_scales_bf16,
                 m_tile_start, n_tile_start);
    issue_stage(shared.stage[0], packed_a, packed_gate_b, packed_up_b,
                m_tile_start, n_tile_start, 0U,
                physical_k64_group_count);
    cp_async_wait_all();
    __syncthreads();

    for (unsigned int stage = 0U; stage < k256_stage_count; ++stage) {
      const unsigned int current_slot = stage & 1U;
      const unsigned int next_stage = stage + 1U;
      if (next_stage < k256_stage_count) {
        issue_stage(shared.stage[next_stage & 1U], packed_a,
                    packed_gate_b, packed_up_b, m_tile_start,
                    n_tile_start, next_stage,
                    physical_k64_group_count);
      }
      accumulate_k256_stage(shared.stage[current_slot], warp_m32,
                            warp_n32, accumulators);
      if (next_stage < k256_stage_count) {
        cp_async_wait_all();
        __syncthreads();
      }
    }

    const bool primary = n_tile_start < primary_width;
    std::uint16_t* const output =
        primary ? primary_output_bf16 : secondary_output_bf16;
    const unsigned int output_stride =
        primary ? primary_output_row_stride_elements
                : secondary_output_row_stride_elements;
    const unsigned int output_n_tile_start =
        primary ? n_tile_start : n_tile_start - primary_width;
    float* warp_tile_maxima = nullptr;
    if constexpr (TrackR1TileMaximum) {
      // All stage reads must finish before the first 2 KiB of stage storage
      // is reused as [warp_n32][M128] scratch.
      __syncthreads();
      warp_tile_maxima = reinterpret_cast<float*>(shared.stage);
      static_assert(kThreads ==
                    kSm87A4W4GateUpFactorizedWarpColumns * kTileM);
      warp_tile_maxima[threadIdx.x] = 0.0F;
      __syncthreads();
    }
    store_accumulators<TrackR1TileMaximum>(
        accumulators, shared.scales, logical_token_count, m_tile_start,
        output_n_tile_start, n_tile_start, warp_m32, warp_n32, output,
        output_stride, down_inverse_alpha_fp32, warp_tile_maxima);
    if constexpr (TrackR1TileMaximum) {
      __syncthreads();
      // The four warp-N owners publish disjoint shared slots.  Warp-N zero
      // then folds those four slots for all 128 rows and performs the sole
      // global partial write for this (row,N128) tile.
      if (warp_n32 == 0U) {
        const unsigned int local_row = warp_m32 * 32U +
                                       (threadIdx.x & 31U);
        float maximum = warp_tile_maxima[local_row];
#pragma unroll
        for (unsigned int warp_n = 1U;
             warp_n < kSm87A4W4GateUpFactorizedWarpColumns; ++warp_n) {
          maximum = fmaxf(
              maximum, warp_tile_maxima[warp_n * kTileM + local_row]);
        }
        const unsigned int global_row = m_tile_start + local_row;
        r1_product_tile_maxima_fp32[
            static_cast<std::size_t>(global_row) *
                kSm87A4W4GateUpFactorizedR1ProductPartialTiles +
            n_tile] = maximum;
      }
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpFactorizedThreads,
                      kSm87A4W4GateUpFactorizedCtasPerSm)
void q3x_sm87_a4w4_gateup_factorized_lane_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_lane_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_lane_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int primary_width,
    const unsigned int physical_k64_group_count,
    const unsigned int k256_stage_count,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count,
    std::uint16_t* const primary_output_bf16,
    const unsigned int primary_output_row_stride_elements,
    std::uint16_t* const secondary_output_bf16,
    const unsigned int secondary_output_row_stride_elements) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_storage);
  gateup_factorized_lane_body<false>(
      shared, packed_a, a_lane_scales_bf16, packed_gate_b,
      gate_b_lane_scales_bf16, packed_up_b, up_b_lane_scales_bf16,
      logical_token_count, primary_width, physical_k64_group_count,
      k256_stage_count, m_tile_count, work_tile_count,
      primary_output_bf16, primary_output_row_stride_elements,
      secondary_output_bf16, secondary_output_row_stride_elements, nullptr,
      nullptr);
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpFactorizedThreads,
                      kSm87A4W4GateUpFactorizedCtasPerSm)
void q3x_sm87_a4w4_gateup_factorized_lane_r1_tile_max_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_lane_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_lane_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int primary_width,
    const unsigned int physical_k64_group_count,
    const unsigned int k256_stage_count,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count,
    std::uint16_t* const primary_output_bf16,
    const unsigned int primary_output_row_stride_elements,
    std::uint16_t* const secondary_output_bf16,
    const unsigned int secondary_output_row_stride_elements,
    const float* const down_inverse_alpha_fp32,
    float* const r1_product_tile_maxima_fp32) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_storage);
  gateup_factorized_lane_body<true>(
      shared, packed_a, a_lane_scales_bf16, packed_gate_b,
      gate_b_lane_scales_bf16, packed_up_b, up_b_lane_scales_bf16,
      logical_token_count, primary_width, physical_k64_group_count,
      k256_stage_count, m_tile_count, work_tile_count,
      primary_output_bf16, primary_output_row_stride_elements,
      secondary_output_bf16, secondary_output_row_stride_elements,
      down_inverse_alpha_fp32, r1_product_tile_maxima_fp32);
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
      properties.multiProcessorCount != static_cast<int>(kRequiredSmCount) ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4GateUpFactorizedDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_factorized_lane_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4GateUpFactorizedDynamicSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_factorized_lane_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

[[nodiscard]] cudaError_t configure_r1_tile_max_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_factorized_lane_r1_tile_max_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4GateUpFactorizedDynamicSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_factorized_lane_r1_tile_max_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

struct ResourceCache final {
  Sm87A4W4GateUpFactorizedLaneResources resources{};
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
      &attributes, q3x_sm87_a4w4_gateup_factorized_lane_kernel);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_gateup_factorized_lane_kernel,
      static_cast<int>(kThreads),
      kSm87A4W4GateUpFactorizedDynamicSharedBytes);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }

  cache.resources.registers_per_thread = attributes.numRegs;
  cache.resources.static_shared_bytes = attributes.sharedSizeBytes;
  cache.resources.dynamic_shared_bytes =
      kSm87A4W4GateUpFactorizedDynamicSharedBytes;
  cache.resources.configured_dynamic_shared_limit_bytes =
      attributes.maxDynamicSharedSizeBytes;
  cache.resources.device_optin_shared_limit_bytes =
      properties.sharedMemPerBlockOptin;
  cache.resources.local_bytes = attributes.localSizeBytes;
  cache.resources.maximum_threads_per_block =
      attributes.maxThreadsPerBlock;
  cache.resources.active_blocks_per_sm = active_blocks;
  cache.resources.compute_major = properties.major;
  cache.resources.compute_minor = properties.minor;

  cache.status =
      cache.resources.registers_per_thread <= 0 ||
              cache.resources.registers_per_thread >
                  static_cast<int>(
                      kSm87A4W4GateUpFactorizedMaximumRegisters) ||
              cache.resources.static_shared_bytes != 0U ||
              cache.resources.dynamic_shared_bytes !=
                  kSm87A4W4GateUpFactorizedDynamicSharedBytes ||
              cache.resources.configured_dynamic_shared_limit_bytes <
                  kSm87A4W4GateUpFactorizedDynamicSharedBytes ||
              cache.resources.device_optin_shared_limit_bytes <
                  kSm87A4W4GateUpFactorizedDynamicSharedBytes ||
              cache.resources.local_bytes != 0U ||
              cache.resources.maximum_threads_per_block <
                  static_cast<int>(kThreads) ||
              cache.resources.active_blocks_per_sm <
                  static_cast<int>(
                      kSm87A4W4GateUpFactorizedCtasPerSm)
          ? static_cast<int>(cudaErrorLaunchOutOfResources)
          : static_cast<int>(cudaSuccess);
  return cache;
}

[[nodiscard]] const ResourceCache& resource_cache() noexcept {
  static const ResourceCache cache = build_resource_cache();
  return cache;
}

[[nodiscard]] ResourceCache build_r1_tile_max_resource_cache() noexcept {
  ResourceCache cache{};
  cudaDeviceProp properties{};
  cache.status = validate_target(&properties);
  if (cache.status != static_cast<int>(cudaSuccess)) {
    return cache;
  }
  const cudaError_t configure_status = configure_r1_tile_max_kernel();
  if (configure_status != cudaSuccess) {
    cache.status = static_cast<int>(configure_status);
    return cache;
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_gateup_factorized_lane_r1_tile_max_kernel);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_factorized_lane_r1_tile_max_kernel,
      static_cast<int>(kThreads),
      kSm87A4W4GateUpFactorizedDynamicSharedBytes);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }

  cache.resources.registers_per_thread = attributes.numRegs;
  cache.resources.static_shared_bytes = attributes.sharedSizeBytes;
  cache.resources.dynamic_shared_bytes =
      kSm87A4W4GateUpFactorizedDynamicSharedBytes;
  cache.resources.configured_dynamic_shared_limit_bytes =
      attributes.maxDynamicSharedSizeBytes;
  cache.resources.device_optin_shared_limit_bytes =
      properties.sharedMemPerBlockOptin;
  cache.resources.local_bytes = attributes.localSizeBytes;
  cache.resources.maximum_threads_per_block = attributes.maxThreadsPerBlock;
  cache.resources.active_blocks_per_sm = active_blocks;
  cache.resources.compute_major = properties.major;
  cache.resources.compute_minor = properties.minor;
  cache.status =
      cache.resources.registers_per_thread <= 0 ||
              cache.resources.registers_per_thread >
                  static_cast<int>(
                      kSm87A4W4GateUpFactorizedMaximumRegisters) ||
              cache.resources.static_shared_bytes != 0U ||
              cache.resources.dynamic_shared_bytes !=
                  kSm87A4W4GateUpFactorizedDynamicSharedBytes ||
              cache.resources.configured_dynamic_shared_limit_bytes <
                  kSm87A4W4GateUpFactorizedDynamicSharedBytes ||
              cache.resources.device_optin_shared_limit_bytes <
                  kSm87A4W4GateUpFactorizedDynamicSharedBytes ||
              cache.resources.local_bytes != 0U ||
              cache.resources.maximum_threads_per_block <
                  static_cast<int>(kThreads) ||
              cache.resources.active_blocks_per_sm <
                  static_cast<int>(
                      kSm87A4W4GateUpFactorizedCtasPerSm)
          ? static_cast<int>(cudaErrorLaunchOutOfResources)
          : static_cast<int>(cudaSuccess);
  return cache;
}

[[nodiscard]] const ResourceCache& r1_tile_max_resource_cache() noexcept {
  static const ResourceCache cache = build_r1_tile_max_resource_cache();
  return cache;
}

std::atomic<bool> g_r1_tile_max_resources_ready{false};

[[nodiscard]] bool output_overlaps_input(
    const void* const output, const std::size_t output_bytes,
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_bytes,
    const std::uint16_t* const a_scales,
    const std::size_t a_scale_bytes,
    const std::uint8_t* const packed_gate,
    const std::size_t packed_gate_bytes,
    const std::uint16_t* const gate_scales,
    const std::size_t gate_scale_bytes,
    const std::uint8_t* const packed_up,
    const std::size_t packed_up_bytes,
    const std::uint16_t* const up_scales,
    const std::size_t up_scale_bytes) noexcept {
  return byte_ranges_overlap(output, output_bytes, packed_a,
                             packed_a_bytes) ||
         byte_ranges_overlap(output, output_bytes, a_scales,
                             a_scale_bytes) ||
         byte_ranges_overlap(output, output_bytes, packed_gate,
                             packed_gate_bytes) ||
         byte_ranges_overlap(output, output_bytes, gate_scales,
                             gate_scale_bytes) ||
         byte_ranges_overlap(output, output_bytes, packed_up,
                             packed_up_bytes) ||
         byte_ranges_overlap(output, output_bytes, up_scales,
                             up_scale_bytes);
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_lane_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_lane_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t primary_width,
    std::uint16_t* const primary_output_bf16,
    const std::size_t primary_output_row_stride_elements,
    const std::size_t primary_output_capacity_elements,
    std::uint16_t* const secondary_output_bf16,
    const std::size_t secondary_output_row_stride_elements,
    const std::size_t secondary_output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool production_shape,
    const bool publish_r1_tile_maxima,
    const float* const authenticated_down_inverse_alpha_fp32,
    const std::size_t down_inverse_alpha_capacity_elements,
    float* const r1_product_tile_maxima_fp32,
    const std::size_t r1_product_tile_maxima_capacity_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpFactorizedLanePlan plan =
      production_shape
          ? sm87_a4w4_gateup_factorized_lane_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size)
          : sm87_a4w4_gateup_factorized_lane_test_plan(
                logical_token_count, launch_token_count,
                intermediate_size, input_size, primary_width);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      (production_shape &&
       (primary_width != kSm87A4W4GateUpFactorizedPrimaryWidth ||
        primary_output_row_stride_elements !=
            kSm87A4W4GateUpFactorizedPrimaryStride ||
        secondary_output_row_stride_elements !=
            kSm87A4W4GateUpFactorizedSecondaryStride)) ||
      (publish_r1_tile_maxima && !production_shape) ||
      (publish_r1_tile_maxima &&
       (!aligned(authenticated_down_inverse_alpha_fp32, 16U) ||
        !aligned(r1_product_tile_maxima_fp32, 16U))) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_lane_scales_bf16, 16U) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_lane_scales_bf16, 16U) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_lane_scales_bf16, 16U) ||
      !aligned(primary_output_bf16, 16U) ||
      !aligned(secondary_output_bf16, 16U) ||
      primary_output_row_stride_elements < plan.primary_width ||
      secondary_output_row_stride_elements < plan.secondary_width ||
      primary_output_row_stride_elements % 2U != 0U ||
      secondary_output_row_stride_elements % 2U != 0U ||
      primary_output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      secondary_output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      plan.logical_token_count >
          std::numeric_limits<unsigned int>::max() ||
      plan.primary_width > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.k256_stages > std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_gateup_factorized_product_fits(
          logical_token_count, primary_output_row_stride_elements) ||
      !sm87_a4w4_gateup_factorized_product_fits(
          logical_token_count, secondary_output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_gateup_factorized_packed_capacity_bytes(
          launch_token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_gateup_factorized_packed_capacity_bytes(
          intermediate_size, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_gateup_factorized_scale_capacity_elements(
          launch_token_count);
  const std::size_t required_b_scales =
      sm87_a4w4_gateup_factorized_scale_capacity_elements(
          intermediate_size);
  const std::size_t required_primary_elements =
      logical_token_count * primary_output_row_stride_elements;
  const std::size_t required_secondary_elements =
      logical_token_count * secondary_output_row_stride_elements;
  const std::size_t required_r1_tile_maxima_elements =
      publish_r1_tile_maxima
          ? sm87_a4w4_gateup_factorized_r1_product_partial_capacity_elements(
                launch_token_count)
          : 0U;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      gate_b_scale_capacity_elements < required_b_scales ||
      up_b_scale_capacity_elements < required_b_scales ||
      primary_output_capacity_elements < required_primary_elements ||
      secondary_output_capacity_elements < required_secondary_elements ||
      (publish_r1_tile_maxima &&
       (down_inverse_alpha_capacity_elements < intermediate_size ||
        required_r1_tile_maxima_elements == 0U ||
        r1_product_tile_maxima_capacity_elements <
            required_r1_tile_maxima_elements)) ||
      !sm87_a4w4_gateup_factorized_product_fits(
          required_a_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_gateup_factorized_product_fits(
          required_b_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_gateup_factorized_product_fits(
          required_primary_elements, sizeof(std::uint16_t)) ||
      !sm87_a4w4_gateup_factorized_product_fits(
          required_secondary_elements, sizeof(std::uint16_t)) ||
      (publish_r1_tile_maxima &&
       (!sm87_a4w4_gateup_factorized_product_fits(
            intermediate_size, sizeof(float)) ||
        !sm87_a4w4_gateup_factorized_product_fits(
            required_r1_tile_maxima_elements, sizeof(float))))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t required_primary_bytes =
      required_primary_elements * sizeof(std::uint16_t);
  const std::size_t required_secondary_bytes =
      required_secondary_elements * sizeof(std::uint16_t);
  const std::size_t required_down_inverse_alpha_bytes =
      publish_r1_tile_maxima ? intermediate_size * sizeof(float) : 0U;
  const std::size_t required_r1_tile_maxima_bytes =
      publish_r1_tile_maxima
          ? required_r1_tile_maxima_elements * sizeof(float)
          : 0U;
  if (output_overlaps_input(
          primary_output_bf16, required_primary_bytes, packed_a,
          required_a_bytes, a_lane_scales_bf16,
          required_a_scale_bytes, packed_gate_b, required_b_bytes,
          gate_b_lane_scales_bf16, required_b_scale_bytes, packed_up_b,
          required_b_bytes, up_b_lane_scales_bf16,
          required_b_scale_bytes) ||
      output_overlaps_input(
          secondary_output_bf16, required_secondary_bytes, packed_a,
          required_a_bytes, a_lane_scales_bf16,
          required_a_scale_bytes, packed_gate_b, required_b_bytes,
          gate_b_lane_scales_bf16, required_b_scale_bytes, packed_up_b,
          required_b_bytes, up_b_lane_scales_bf16,
          required_b_scale_bytes) ||
      byte_ranges_overlap(primary_output_bf16, required_primary_bytes,
                          secondary_output_bf16,
                          required_secondary_bytes) ||
      (publish_r1_tile_maxima &&
       (output_overlaps_input(
            r1_product_tile_maxima_fp32,
            required_r1_tile_maxima_bytes, packed_a, required_a_bytes,
            a_lane_scales_bf16, required_a_scale_bytes, packed_gate_b,
            required_b_bytes, gate_b_lane_scales_bf16,
            required_b_scale_bytes, packed_up_b, required_b_bytes,
            up_b_lane_scales_bf16, required_b_scale_bytes) ||
        byte_ranges_overlap(r1_product_tile_maxima_fp32,
                            required_r1_tile_maxima_bytes,
                            primary_output_bf16,
                            required_primary_bytes) ||
        byte_ranges_overlap(r1_product_tile_maxima_fp32,
                            required_r1_tile_maxima_bytes,
                            secondary_output_bf16,
                            required_secondary_bytes) ||
        byte_ranges_overlap(r1_product_tile_maxima_fp32,
                            required_r1_tile_maxima_bytes,
                            authenticated_down_inverse_alpha_fp32,
                            required_down_inverse_alpha_bytes) ||
        byte_ranges_overlap(authenticated_down_inverse_alpha_fp32,
                            required_down_inverse_alpha_bytes,
                            primary_output_bf16,
                            required_primary_bytes) ||
        byte_ranges_overlap(authenticated_down_inverse_alpha_fp32,
                            required_down_inverse_alpha_bytes,
                            secondary_output_bf16,
                            required_secondary_bytes)))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const unsigned int planned_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      planned_ctas < maximum_launch_ctas ? planned_ctas
                                         : maximum_launch_ctas;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (publish_r1_tile_maxima) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (stream != nullptr) {
      const cudaError_t capture_query =
          cudaStreamIsCapturing(stream, &capture_status);
      if (capture_query != cudaSuccess) {
        return static_cast<int>(capture_query);
      }
    }
    if (capture_status != cudaStreamCaptureStatusNone &&
        !g_r1_tile_max_resources_ready.load(std::memory_order_acquire)) {
      return static_cast<int>(cudaErrorNotReady);
    }
    if (!g_r1_tile_max_resources_ready.load(std::memory_order_acquire)) {
      Sm87A4W4GateUpFactorizedLaneResources resources{};
      const int status =
          query_sm87_a4w4_gateup_factorized_lane_r1_tile_max_resources_cuda(
              &resources);
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
  } else {
    const ResourceCache& cache = resource_cache();
    if (cache.status != static_cast<int>(cudaSuccess)) {
      return cache.status;
    }
  }
  // Clear a caller's stale launch error immediately before publishing this
  // kernel's own launch status.
  (void)cudaGetLastError();
  if (publish_r1_tile_maxima) {
    q3x_sm87_a4w4_gateup_factorized_lane_r1_tile_max_kernel
        <<<launch_ctas, kThreads,
           kSm87A4W4GateUpFactorizedDynamicSharedBytes, stream>>>(
            packed_a, a_lane_scales_bf16, packed_gate_b,
            gate_b_lane_scales_bf16, packed_up_b,
            up_b_lane_scales_bf16,
            static_cast<unsigned int>(plan.logical_token_count),
            static_cast<unsigned int>(plan.primary_width),
            static_cast<unsigned int>(plan.physical_k64_groups),
            static_cast<unsigned int>(plan.k256_stages),
            static_cast<unsigned int>(plan.m_tiles),
            static_cast<unsigned int>(plan.work_tiles),
            primary_output_bf16,
            static_cast<unsigned int>(
                primary_output_row_stride_elements),
            secondary_output_bf16,
            static_cast<unsigned int>(
                secondary_output_row_stride_elements),
            authenticated_down_inverse_alpha_fp32,
            r1_product_tile_maxima_fp32);
  } else {
    q3x_sm87_a4w4_gateup_factorized_lane_kernel
        <<<launch_ctas, kThreads,
           kSm87A4W4GateUpFactorizedDynamicSharedBytes, stream>>>(
            packed_a, a_lane_scales_bf16, packed_gate_b,
            gate_b_lane_scales_bf16, packed_up_b,
            up_b_lane_scales_bf16,
            static_cast<unsigned int>(plan.logical_token_count),
            static_cast<unsigned int>(plan.primary_width),
            static_cast<unsigned int>(plan.physical_k64_groups),
            static_cast<unsigned int>(plan.k256_stages),
            static_cast<unsigned int>(plan.m_tiles),
            static_cast<unsigned int>(plan.work_tiles),
            primary_output_bf16,
            static_cast<unsigned int>(
                primary_output_row_stride_elements),
            secondary_output_bf16,
            static_cast<unsigned int>(
                secondary_output_row_stride_elements));
  }
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_factorized_lane_resources_cuda(
    Sm87A4W4GateUpFactorizedLaneResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ResourceCache& cache = resource_cache();
  *resources = cache.resources;
  return cache.status;
}

int query_sm87_a4w4_gateup_factorized_lane_r1_tile_max_resources_cuda(
    Sm87A4W4GateUpFactorizedLaneResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ResourceCache& cache = r1_tile_max_resource_cache();
  *resources = cache.resources;
  if (cache.status == static_cast<int>(cudaSuccess)) {
    g_r1_tile_max_resources_ready.store(true, std::memory_order_release);
  }
  return cache.status;
}

int launch_sm87_a4w4_gateup_factorized_lane_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_lane_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_lane_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
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
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_lane_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_lane_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_lane_scales_bf16,
      up_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size,
      kSm87A4W4GateUpFactorizedPrimaryWidth, primary_output_bf16,
      primary_output_row_stride_elements,
      primary_output_capacity_elements, secondary_output_bf16,
      secondary_output_row_stride_elements,
      secondary_output_capacity_elements,
      static_cast<unsigned int>(kSm87A4W4GateUpFactorizedPersistentCtas),
      true, false, nullptr, 0U, nullptr, 0U, cuda_stream);
}

int launch_sm87_a4w4_gateup_factorized_lane_r1_tile_max_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_lane_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_lane_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
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
    const float* const authenticated_down_inverse_alpha_fp32,
    const std::size_t down_inverse_alpha_capacity_elements,
    float* const r1_product_tile_maxima_fp32,
    const std::size_t r1_product_tile_maxima_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_lane_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_lane_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_lane_scales_bf16,
      up_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size,
      kSm87A4W4GateUpFactorizedPrimaryWidth, primary_output_bf16,
      primary_output_row_stride_elements,
      primary_output_capacity_elements, secondary_output_bf16,
      secondary_output_row_stride_elements,
      secondary_output_capacity_elements,
      static_cast<unsigned int>(kSm87A4W4GateUpFactorizedPersistentCtas),
      true, true, authenticated_down_inverse_alpha_fp32,
      down_inverse_alpha_capacity_elements, r1_product_tile_maxima_fp32,
      r1_product_tile_maxima_capacity_elements, cuda_stream);
}

int launch_sm87_a4w4_gateup_factorized_lane_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_lane_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_lane_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t primary_width,
    std::uint16_t* const primary_output_bf16,
    const std::size_t primary_output_row_stride_elements,
    const std::size_t primary_output_capacity_elements,
    std::uint16_t* const secondary_output_bf16,
    const std::size_t secondary_output_row_stride_elements,
    const std::size_t secondary_output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_lane_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_lane_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_lane_scales_bf16,
      up_b_scale_capacity_elements, logical_token_count,
      launch_token_count, intermediate_size, input_size, primary_width,
      primary_output_bf16, primary_output_row_stride_elements,
      primary_output_capacity_elements, secondary_output_bf16,
      secondary_output_row_stride_elements,
      secondary_output_capacity_elements, maximum_launch_ctas, false,
      false, nullptr, 0U, nullptr, 0U, cuda_stream);
}

}  // namespace q3x::kernels
