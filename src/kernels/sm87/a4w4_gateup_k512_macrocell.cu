#include "q3x/kernels/sm87_a4w4_gateup_k512_macrocell.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(
        kSm87A4W4GateUpK512MacroPackedK64Bytes);

struct alignas(16) Sm87A4W4GateUpK512MacroStage final {
  std::uint8_t a[kSm87A4W4GateUpK512MacroK64PerCopy]
                [kSm87A4W4GateUpK512MacroTileM *
                 kSm87A4W4GateUpK512MacroPackedK64Bytes];
  std::uint8_t gate[kSm87A4W4GateUpK512MacroK64PerCopy]
                   [kSm87A4W4GateUpK512MacroTileN *
                    kSm87A4W4GateUpK512MacroPackedK64Bytes];
  std::uint8_t up[kSm87A4W4GateUpK512MacroK64PerCopy]
                 [kSm87A4W4GateUpK512MacroTileN *
                  kSm87A4W4GateUpK512MacroPackedK64Bytes];
};

struct alignas(16) Sm87A4W4GateUpK512MacroScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpK512MacroTileM];
  std::uint16_t gate[kSm87A4W4GateUpK512MacroTileN];
  std::uint16_t up[kSm87A4W4GateUpK512MacroTileN];
};

struct alignas(16) Sm87A4W4GateUpK512MacroPipeline final {
  Sm87A4W4GateUpK512MacroStage
      stage[kSm87A4W4GateUpK512MacroStages];
  Sm87A4W4GateUpK512MacroScaleSlot
      scale[kSm87A4W4GateUpK512MacroScaleSlots];
};

union alignas(16) Sm87A4W4GateUpK512MacroShared final {
  Sm87A4W4GateUpK512MacroPipeline pipeline;
  float product[kSm87A4W4GateUpK512MacroTileM]
               [kSm87A4W4GateUpK512MacroTileN];
};

struct alignas(16) Sm87A4W4GateUpK512MacroFloat4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(Sm87A4W4GateUpK512MacroStage) ==
              kSm87A4W4GateUpK512MacroStageBytes);
static_assert(sizeof(Sm87A4W4GateUpK512MacroScaleSlot) ==
              kSm87A4W4GateUpK512MacroScaleSlotBytes);
static_assert(sizeof(Sm87A4W4GateUpK512MacroPipeline) ==
              kSm87A4W4GateUpK512MacroSharedBytes);
static_assert(sizeof(Sm87A4W4GateUpK512MacroShared) ==
              kSm87A4W4GateUpK512MacroSharedBytes);
static_assert(sizeof(Sm87A4W4GateUpK512MacroFloat4) == 16U);

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

// One K256 stage consists of four independently swizzled K64 planes.  Across
// 512 threads, every lane issues one A vector and two vectors for each of Gate
// and Up: five aligned 16-byte cp.async.cg operations, with no duplicate A.
__device__ __forceinline__ void issue_k256_codes(
    Sm87A4W4GateUpK512MacroStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int physical_k256_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512MacroTileM * kPackedK64Bytes / 16U);
  constexpr unsigned int kBVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512MacroTileN * kPackedK64Bytes / 16U);
  constexpr unsigned int kAVectors =
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512MacroK64PerCopy) *
      kAVectorsPerPlane;
  constexpr unsigned int kBVectors =
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512MacroK64PerCopy) *
      kBVectorsPerPlane;
  static_assert(kAVectors == kSm87A4W4GateUpK512MacroThreads);
  static_assert(kBVectors ==
                2U * kSm87A4W4GateUpK512MacroThreads);

  const unsigned int a_vector = threadIdx.x;
  const unsigned int a_plane = a_vector / kAVectorsPerPlane;
  const unsigned int a_vector_in_plane =
      a_vector - a_plane * kAVectorsPerPlane;
  const unsigned int a_row = a_vector_in_plane / 2U;
  const unsigned int a_row_vector = a_vector_in_plane % 2U;
  const unsigned int a_physical_k64 =
      physical_k256_group *
          static_cast<unsigned int>(
              kSm87A4W4GateUpK512MacroK64PerCopy) +
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
        iteration * kSm87A4W4GateUpK512MacroThreads;
    const unsigned int b_plane = b_vector / kBVectorsPerPlane;
    const unsigned int b_vector_in_plane =
        b_vector - b_plane * kBVectorsPerPlane;
    const unsigned int b_row = b_vector_in_plane / 2U;
    const unsigned int b_row_vector = b_vector_in_plane % 2U;
    const unsigned int b_physical_k64 =
        physical_k256_group *
            static_cast<unsigned int>(
                kSm87A4W4GateUpK512MacroK64PerCopy) +
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
    Sm87A4W4GateUpK512MacroScaleSlot& slot,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  constexpr unsigned int kRowsPerVector = 16U / sizeof(std::uint16_t);
  constexpr unsigned int kAVectors =
      kSm87A4W4GateUpK512MacroTileM / kRowsPerVector;
  constexpr unsigned int kBVectors =
      kSm87A4W4GateUpK512MacroTileN / kRowsPerVector;
  static_assert(kAVectors == 8U);
  static_assert(kBVectors == 16U);

  if (threadIdx.x < kAVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    cp_async_16(
        slot.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_k512_macro_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x < kBVectors) {
    const unsigned int first_row = kRowsPerVector * threadIdx.x;
    const std::size_t source_offset =
        sm87_a4w4_gateup_k512_macro_scale_offset(
            static_cast<std::size_t>(absolute_n_tile_start) + first_row,
            k512_group, k512_group_count);
    cp_async_16(slot.gate + first_row,
                gate_b_k512_scales_bf16 + source_offset);
    cp_async_16(slot.up + first_row,
                up_b_k512_scales_bf16 + source_offset);
  }
}

__device__ __forceinline__ void issue_even_k256_and_scales(
    Sm87A4W4GateUpK512MacroStage& stage,
    Sm87A4W4GateUpK512MacroScaleSlot& scale,
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
    Sm87A4W4GateUpK512MacroStage& stage,
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

__device__ __forceinline__ void accumulate_k256_stage(
    const Sm87A4W4GateUpK512MacroStage& stage,
    Sm87A4W4Accumulator (&partials)[8U]) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int projection_warp =
      warp % kSm87A4W4GateUpK512MacroProjectionWarps;
  const unsigned int fragment_m_start =
      (projection_warp / 2U) * 16U;
  const unsigned int fragment_n_half =
      (projection_warp % 2U) * 64U;
  const std::uint8_t (*b)[kSm87A4W4GateUpK512MacroTileN *
                          kSm87A4W4GateUpK512MacroPackedK64Bytes] =
      warp < kSm87A4W4GateUpK512MacroProjectionWarps
          ? stage.gate
          : stage.up;

#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4GateUpK512MacroK64PerCopy; ++plane) {
    const Sm87A4W4AFragment a_fragment =
        sm87_a4w4_load_a_fragment_swizzled_shared(
            stage.a[plane] + fragment_m_start * kPackedK64Bytes,
            lane);
#pragma unroll
    for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
      const Sm87A4W4BFragment b_fragment =
          sm87_a4w4_load_b_fragment_swizzled_shared(
              b[plane] +
                  (fragment_n_half + fragment_n * 8U) *
                      kPackedK64Bytes,
              lane);
      sm87_a4w4_mma_m16n8k64(partials[fragment_n], a_fragment,
                             b_fragment);
    }
  }
}

__device__ __forceinline__ void apply_k512_group(
    Sm87A4W4GateUpK512MacroFloat4 (&accumulators)[8U],
    const Sm87A4W4Accumulator (&partials)[8U],
    const Sm87A4W4GateUpK512MacroScaleSlot& scale) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int projection_warp =
      warp % kSm87A4W4GateUpK512MacroProjectionWarps;
  const unsigned int fragment_m_start =
      (projection_warp / 2U) * 16U;
  const unsigned int fragment_n_half =
      (projection_warp % 2U) * 64U;
  const std::uint16_t* const b_scale =
      warp < kSm87A4W4GateUpK512MacroProjectionWarps
          ? scale.gate
          : scale.up;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const float a_scale0 =
      decode_bf16(scale.a[fragment_m_start + coordinate0.m]);
  const float a_scale1 =
      decode_bf16(scale.a[fragment_m_start + coordinate2.m]);

#pragma unroll
  for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
    const unsigned int local_n0 =
        fragment_n_half + fragment_n * 8U + coordinate0.n;
    const unsigned int local_n1 =
        fragment_n_half + fragment_n * 8U + coordinate1.n;
    const float b_scale0 = decode_bf16(b_scale[local_n0]);
    const float b_scale1 = decode_bf16(b_scale[local_n1]);
    const float scale00 = __fmul_rn(a_scale0, b_scale0);
    const float scale01 = __fmul_rn(a_scale0, b_scale1);
    const float scale10 = __fmul_rn(a_scale1, b_scale0);
    const float scale11 = __fmul_rn(a_scale1, b_scale1);
    accumulators[fragment_n].x0 = __fmaf_rn(
        static_cast<float>(partials[fragment_n].x0), scale00,
        accumulators[fragment_n].x0);
    accumulators[fragment_n].x1 = __fmaf_rn(
        static_cast<float>(partials[fragment_n].x1), scale01,
        accumulators[fragment_n].x1);
    accumulators[fragment_n].x2 = __fmaf_rn(
        static_cast<float>(partials[fragment_n].x2), scale10,
        accumulators[fragment_n].x2);
    accumulators[fragment_n].x3 = __fmaf_rn(
        static_cast<float>(partials[fragment_n].x3), scale11,
        accumulators[fragment_n].x3);
  }
}

__device__ __forceinline__ void exchange_and_store_product(
    Sm87A4W4GateUpK512MacroShared& shared,
    const Sm87A4W4GateUpK512MacroFloat4 (&accumulators)[8U],
    const unsigned int m_tile_start,
    const unsigned int output_n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int projection_warp =
      warp % kSm87A4W4GateUpK512MacroProjectionWarps;
  const unsigned int fragment_m_start =
      (projection_warp / 2U) * 16U;
  const unsigned int fragment_n_half =
      (projection_warp % 2U) * 64U;

  if (warp < kSm87A4W4GateUpK512MacroProjectionWarps) {
#pragma unroll
    for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        const Sm87A4W4AccumulatorCoordinate coordinate =
            sm87_a4w4_accumulator_coordinate(lane, output);
        const float values[4U] = {
            accumulators[fragment_n].x0,
            accumulators[fragment_n].x1,
            accumulators[fragment_n].x2,
            accumulators[fragment_n].x3};
        shared.product[fragment_m_start + coordinate.m]
                      [fragment_n_half + fragment_n * 8U + coordinate.n] =
            values[output];
      }
    }
  }
  __syncthreads();

  if (warp >= kSm87A4W4GateUpK512MacroProjectionWarps) {
#pragma unroll
    for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
#pragma unroll
      for (unsigned int output = 0U; output < 4U; ++output) {
        const Sm87A4W4AccumulatorCoordinate coordinate =
            sm87_a4w4_accumulator_coordinate(lane, output);
        const unsigned int local_m = fragment_m_start + coordinate.m;
        const unsigned int local_n =
            fragment_n_half + fragment_n * 8U + coordinate.n;
        const float values[4U] = {
            accumulators[fragment_n].x0,
            accumulators[fragment_n].x1,
            accumulators[fragment_n].x2,
            accumulators[fragment_n].x3};
        output_bf16[
            static_cast<std::size_t>(m_tile_start + local_m) *
                output_row_stride_elements +
            output_n_tile_start + local_n] =
            encode_bf16(silu_product(shared.product[local_m][local_n],
                                     values[output]));
      }
    }
  }
  // Up must finish every read before the next N tile reuses the union as the
  // raw-code pipeline.
  __syncthreads();
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpK512MacroThreads,
                      kSm87A4W4GateUpK512MacroCtasPerSm)
void q3x_sm87_a4w4_gateup_k512_m64n128_macrocell_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int absolute_n_start,
    const unsigned int n_tile_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared = *reinterpret_cast<Sm87A4W4GateUpK512MacroShared*>(
      dynamic_shared);

  // M-owner scheduling keeps all 16 SMs on the same N128 weight window.  This
  // gives Gate/Up B a single lockstep L2 stream while each CTA retains its M64
  // ownership for the full N sweep.  At P2048 each CTA handles two M tiles.
  for (unsigned int m_tile = blockIdx.x; m_tile < m_tile_count;
       m_tile += gridDim.x) {
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4GateUpK512MacroTileM;
    for (unsigned int n_tile = 0U; n_tile < n_tile_count; ++n_tile) {
      const unsigned int absolute_n_tile_start =
          absolute_n_start +
          n_tile * kSm87A4W4GateUpK512MacroTileN;
      const unsigned int output_n_tile_start =
          n_tile * kSm87A4W4GateUpK512MacroTileN;
      Sm87A4W4GateUpK512MacroFloat4 accumulators[8U]{};

      issue_even_k256_and_scales(
          shared.pipeline.stage[0U], shared.pipeline.scale[0U],
          packed_a, a_k512_scales_bf16, packed_gate_b,
          gate_b_k512_scales_bf16, packed_up_b,
          up_b_k512_scales_bf16, m_tile_start,
          absolute_n_tile_start, 0U, k512_group_count,
          physical_k64_group_count);
      issue_odd_k256(
          shared.pipeline.stage[1U], packed_a, packed_gate_b,
          packed_up_b, m_tile_start, absolute_n_tile_start, 0U,
          physical_k64_group_count);

      for (unsigned int group = 0U; group < k512_group_count; ++group) {
        Sm87A4W4Accumulator partials[8U]{};

        // The even K256 stage (and its K512 scales) retires while the odd
        // stage remains eligible to overlap.
        cp_async_wait<1U>();
        __syncthreads();
        accumulate_k256_stage(shared.pipeline.stage[0U], partials);
        __syncthreads();

        const unsigned int next_group = group + 1U;
        if (next_group < k512_group_count) {
          issue_even_k256_and_scales(
              shared.pipeline.stage[0U],
              shared.pipeline.scale[
                  next_group % kSm87A4W4GateUpK512MacroScaleSlots],
              packed_a, a_k512_scales_bf16, packed_gate_b,
              gate_b_k512_scales_bf16, packed_up_b,
              up_b_k512_scales_bf16, m_tile_start,
              absolute_n_tile_start, next_group, k512_group_count,
              physical_k64_group_count);
          cp_async_wait<1U>();
        } else {
          cp_async_wait<0U>();
        }
        __syncthreads();
        accumulate_k256_stage(shared.pipeline.stage[1U], partials);

        // All eight K64 MMA terms are now present in S32.  This is the only
        // FP32 dequantization boundary for the current K512 group.
        apply_k512_group(
            accumulators, partials,
            shared.pipeline.scale[
                group % kSm87A4W4GateUpK512MacroScaleSlots]);
        __syncthreads();

        if (next_group < k512_group_count) {
          issue_odd_k256(
              shared.pipeline.stage[1U], packed_a, packed_gate_b,
              packed_up_b, m_tile_start, absolute_n_tile_start,
              next_group, physical_k64_group_count);
        }
      }

      exchange_and_store_product(
          shared, accumulators, m_tile_start, output_n_tile_start,
          output_bf16, output_row_stride_elements);
    }
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
      properties.multiProcessorCount != static_cast<int>(kRequiredSmCount)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_k512_m64n128_macrocell_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4GateUpK512MacroSharedBytes));
}

[[nodiscard]] int launch_sm87_a4w4_gateup_k512_macrocell_impl(
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
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool require_model_shape,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpK512MacroPlan plan =
      sm87_a4w4_gateup_k512_macro_plan(
          token_count, intermediate_size, input_size, n_start, n_count);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      (require_model_shape &&
       !sm87_a4w4_gateup_k512_macro_is_model_plan(plan)) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k512_scales_bf16, 16U) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k512_scales_bf16, 16U) ||
      !aligned(output_bf16, 16U) ||
      output_row_stride_elements < n_count ||
      output_row_stride_elements % 2U != 0U ||
      token_count > std::numeric_limits<unsigned int>::max() ||
      intermediate_size > std::numeric_limits<unsigned int>::max() ||
      input_size > std::numeric_limits<unsigned int>::max() ||
      n_start > std::numeric_limits<unsigned int>::max() ||
      plan.n_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.k512_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_gateup_k512_macro_product_fits(
          token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  // A is sized to MxK.  Gate and Up capacities cover the complete weight
  // matrices, even when this launch selects only one output window.
  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(intermediate_size,
                                                input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_gateup_k512_macro_scale_capacity_elements(token_count,
                                                           input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_gateup_k512_macro_scale_capacity_elements(
          intermediate_size, input_size);
  const std::size_t required_output_elements =
      token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      gate_b_scale_capacity_elements < required_b_scales ||
      up_b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements ||
      !sm87_a4w4_gateup_k512_macro_product_fits(
          required_a_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_gateup_k512_macro_product_fits(
          required_b_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_gateup_k512_macro_product_fits(
          required_output_elements, sizeof(std::uint16_t))) {
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
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          packed_gate_b, required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          gate_b_k512_scales_bf16,
                          required_b_scale_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          packed_up_b, required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          up_b_k512_scales_bf16,
                          required_b_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int target_status = validate_sm87();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaError_t shared_status = configure_dynamic_shared();
  if (shared_status != cudaSuccess) {
    return static_cast<int>(shared_status);
  }

  const unsigned int persistent_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      persistent_ctas < maximum_launch_ctas
          ? persistent_ctas
          : maximum_launch_ctas;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_gateup_k512_m64n128_macrocell_kernel
      <<<launch_ctas,
         static_cast<unsigned int>(kSm87A4W4GateUpK512MacroThreads),
         kSm87A4W4GateUpK512MacroSharedBytes, stream>>>(
          packed_a, a_k512_scales_bf16, packed_gate_b,
          gate_b_k512_scales_bf16, packed_up_b,
          up_b_k512_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          static_cast<unsigned int>(plan.n_start),
          static_cast<unsigned int>(plan.n_tiles), output_bf16,
          static_cast<unsigned int>(output_row_stride_elements),
          static_cast<unsigned int>(plan.m_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_k512_macrocell_resources_cuda(
    Sm87A4W4GateUpK512MacroResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpK512MacroResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_sm87(&properties);
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
      q3x_sm87_a4w4_gateup_k512_m64n128_macrocell_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_k512_m64n128_macrocell_kernel,
      static_cast<int>(kSm87A4W4GateUpK512MacroThreads),
      kSm87A4W4GateUpK512MacroSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpK512MacroSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4GateUpK512MacroMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      attributes.maxDynamicSharedSizeBytes <
          static_cast<int>(kSm87A4W4GateUpK512MacroSharedBytes) ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4GateUpK512MacroSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4GateUpK512MacroThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4GateUpK512MacroCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_k512_macrocell_bf16_cuda(
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
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_sm87_a4w4_gateup_k512_macrocell_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_k512_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_k512_scales_bf16,
      up_b_scale_capacity_elements, token_count, intermediate_size,
      input_size, n_start, n_count, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512MacroPersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_gateup_k512_macrocell_test_bf16_cuda(
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
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_sm87_a4w4_gateup_k512_macrocell_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_gate_b,
      packed_gate_b_capacity_bytes, gate_b_k512_scales_bf16,
      gate_b_scale_capacity_elements, packed_up_b,
      packed_up_b_capacity_bytes, up_b_k512_scales_bf16,
      up_b_scale_capacity_elements, token_count, intermediate_size,
      input_size, n_start, n_count, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      maximum_launch_ctas, false, cuda_stream);
}

}  // namespace q3x::kernels
