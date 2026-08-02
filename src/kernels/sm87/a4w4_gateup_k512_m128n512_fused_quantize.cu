#include "q3x/kernels/sm87_a4w4_gateup_k512_m128n512_fused_quantize.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kPackedK64Bytes = 32U;

// CUDA Graph capture must not discover or configure kernel resources.  A
// successful eager resource query authenticates the exact compiled kernel
// once; subsequent launches, including capture, consume that proof.
std::atomic<bool> g_fused_quantize_resources_ready{false};

struct alignas(16) FusedQuantizeStage final {
  std::uint8_t a[kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage]
                [kSm87A4W4GateUpK512M128N512FusedQuantizeTileM *
                 kPackedK64Bytes];
  std::uint8_t gate
      [kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage]
      [kSm87A4W4GateUpK512M128N512FusedQuantizeComputeTileN *
       kPackedK64Bytes];
  std::uint8_t up
      [kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage]
      [kSm87A4W4GateUpK512M128N512FusedQuantizeComputeTileN *
       kPackedK64Bytes];
};

struct alignas(16) FusedQuantizeScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpK512M128N512FusedQuantizeTileM];
  std::uint16_t gate
      [kSm87A4W4GateUpK512M128N512FusedQuantizeComputeTileN];
  std::uint16_t up
      [kSm87A4W4GateUpK512M128N512FusedQuantizeComputeTileN];
};

struct alignas(16) FusedQuantizePipeline final {
  FusedQuantizeStage
      stage[kSm87A4W4GateUpK512M128N512FusedQuantizeStages];
  FusedQuantizeScaleSlot
      scale[kSm87A4W4GateUpK512M128N512FusedQuantizeScaleSlots];
};

// Component-major MMA ownership.  One word holds the adjacent x0/x1 or
// x2/x3 BF16 pair of one lane.  A fixed component therefore spans 32
// consecutive banks instead of aliasing a row-major plane.
struct alignas(16) FusedQuantizeSharedProducts final {
  std::uint32_t word[2U]
                    [kSm87A4W4GateUpK512M128N512FusedQuantizeWarps]
                    [kSm87A4W4GateUpK512M128N512FusedQuantizeN8FragmentsPerWarp]
                    [2U][32U];
};

struct alignas(16) FusedQuantizeShared final {
  FusedQuantizePipeline pipeline;
  FusedQuantizeSharedProducts product;
};

struct alignas(16) FusedQuantizeFinalStage final {
  std::uint8_t a[kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage]
                [kSm87A4W4GateUpK512M128N512FusedQuantizeTileM *
                 kPackedK64Bytes];
  std::uint8_t gate
      [kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage]
      [64U * kPackedK64Bytes];
  std::uint8_t up
      [kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage]
      [64U * kPackedK64Bytes];
};

struct alignas(16) FusedQuantizeFinalScaleSlot final {
  std::uint16_t a[kSm87A4W4GateUpK512M128N512FusedQuantizeTileM];
  std::uint16_t gate[64U];
  std::uint16_t up[64U];
};

struct alignas(16) FusedQuantizeFinalPipeline final {
  FusedQuantizeFinalStage
      stage[kSm87A4W4GateUpK512M128N512FusedQuantizeStages];
  FusedQuantizeFinalScaleSlot
      scale[kSm87A4W4GateUpK512M128N512FusedQuantizeScaleSlots];
};

struct alignas(16) FusedQuantizeThirdSharedProduct final {
  std::uint32_t word
      [kSm87A4W4GateUpK512M128N512FusedQuantizeWarps]
      [kSm87A4W4GateUpK512M128N512FusedQuantizeN8FragmentsPerWarp]
      [2U][32U];
};

// This phase overlays the original 99,840-byte pipeline after the third
// subcell has consumed it.  The reduced N64 pipeline and the third product
// plane are disjoint and together still fit in that original envelope.
struct alignas(16) FusedQuantizeFinalPhase final {
  FusedQuantizeFinalPipeline pipeline;
  FusedQuantizeThirdSharedProduct third;
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

struct PackedProduct final {
  std::uint32_t word
      [kSm87A4W4GateUpK512M128N512FusedQuantizeN8FragmentsPerWarp]
      [2U];
};

struct PackedHalfProduct final {
  std::uint32_t word[4U][2U];
};

struct QuantizeScratch final {
  float half_max[kSm87A4W4GateUpK512M128N512FusedQuantizeTileM][2U];
  float clipped_max[kSm87A4W4GateUpK512M128N512FusedQuantizeTileM];
  float stored_scale[kSm87A4W4GateUpK512M128N512FusedQuantizeTileM];
};

static_assert(sizeof(FusedQuantizeStage) ==
              kSm87A4W4GateUpK512M128N512FusedQuantizeStageBytes);
static_assert(sizeof(FusedQuantizeScaleSlot) ==
              kSm87A4W4GateUpK512M128N512FusedQuantizeScaleSlotBytes);
static_assert(sizeof(FusedQuantizePipeline) ==
              kSm87A4W4GateUpK512M128N512FusedQuantizePipelineBytes);
static_assert(sizeof(FusedQuantizeSharedProducts) ==
              kSm87A4W4GateUpK512M128N512FusedQuantizeProductBytes);
static_assert(sizeof(FusedQuantizeShared) ==
              kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes);
static_assert(sizeof(FusedQuantizeFinalStage) == 32'768U);
static_assert(sizeof(FusedQuantizeFinalScaleSlot) == 512U);
static_assert(sizeof(FusedQuantizeFinalPipeline) == 66'560U);
static_assert(sizeof(FusedQuantizeThirdSharedProduct) == 32'768U);
static_assert(sizeof(FusedQuantizeFinalPhase) == 99'328U);
static_assert(sizeof(FusedQuantizeFinalPhase) <=
              kSm87A4W4GateUpK512M128N512FusedQuantizePipelineBytes);
static_assert(sizeof(PackedProduct) == 64U);
static_assert(sizeof(PackedHalfProduct) == 32U);
static_assert(sizeof(QuantizeScratch) <=
              kSm87A4W4GateUpK512M128N512FusedQuantizePipelineBytes);

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

__device__ __forceinline__ void cp_async_wait_all() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void issue_k256_codes(
    FusedQuantizeStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m_tile_start,
    const unsigned int absolute_n128_start,
    const unsigned int physical_k256_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      kSm87A4W4GateUpK512M128N512FusedQuantizeTileM *
      kPackedK64Bytes / 16U;
  constexpr unsigned int kVectors =
      kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage *
      kVectorsPerPlane;
  static_assert(kVectors ==
                2U *
                    kSm87A4W4GateUpK512M128N512FusedQuantizeThreads);
#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector =
        threadIdx.x +
        iteration *
            kSm87A4W4GateUpK512M128N512FusedQuantizeThreads;
    const unsigned int plane = vector / kVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kVectorsPerPlane;
    const unsigned int row = vector_in_plane / 2U;
    const unsigned int row_vector = vector_in_plane & 1U;
    const unsigned int byte_in_row = 16U * row_vector;
    const unsigned int physical_k64 =
        physical_k256_group *
            kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage +
        plane;
    const std::size_t destination =
        sm87_a4w4_swizzled_k64_byte_offset(row, byte_in_row);
    cp_async_16(
        stage.a[plane] + destination,
        packed_a + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64, byte_in_row,
                       physical_k64_group_count));
    const std::size_t b_source =
        sm87_a4w4_consumer_packed_offset(
            static_cast<std::size_t>(absolute_n128_start) + row,
            physical_k64, byte_in_row, physical_k64_group_count);
    cp_async_16(stage.gate[plane] + destination,
                packed_gate_b + b_source);
    cp_async_16(stage.up[plane] + destination,
                packed_up_b + b_source);
  }
}

__device__ __forceinline__ void issue_k512_scales(
    FusedQuantizeScaleSlot& scale,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n128_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        scale.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k512_group, k512_group_count));
    const std::size_t b_offset =
        sm87_a4w4_gateup_down_edge_scale_offset(
            static_cast<std::size_t>(absolute_n128_start) + first_row,
            k512_group, k512_group_count);
    cp_async_16(scale.gate + first_row,
                gate_b_k512_scales_bf16 + b_offset);
    cp_async_16(scale.up + first_row,
                up_b_k512_scales_bf16 + b_offset);
  }
}

__device__ __forceinline__ void issue_k512_group(
    FusedQuantizePipeline& pipeline,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n128_start,
    const unsigned int group,
    const unsigned int group_count,
    const unsigned int physical_k64_group_count) noexcept {
  issue_k256_codes(pipeline.stage[0U], packed_a, packed_gate_b,
                   packed_up_b, m_tile_start, absolute_n128_start,
                   2U * group, physical_k64_group_count);
  issue_k512_scales(
      pipeline.scale[group & 1U], a_k512_scales_bf16,
      gate_b_k512_scales_bf16, up_b_k512_scales_bf16,
      m_tile_start, absolute_n128_start, group, group_count);
  cp_async_commit();
  issue_k256_codes(pipeline.stage[1U], packed_a, packed_gate_b,
                   packed_up_b, m_tile_start, absolute_n128_start,
                   2U * group + 1U, physical_k64_group_count);
  cp_async_commit();
}

__device__ __forceinline__ void issue_final_k256_codes(
    FusedQuantizeFinalStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_gate_b,
    const std::uint8_t* const packed_up_b,
    const unsigned int m_tile_start,
    const unsigned int absolute_n64_start,
    const unsigned int physical_k256_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      kSm87A4W4GateUpK512M128N512FusedQuantizeTileM *
      kPackedK64Bytes / 16U;
#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector =
        threadIdx.x +
        iteration *
            kSm87A4W4GateUpK512M128N512FusedQuantizeThreads;
    const unsigned int plane = vector / kAVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kAVectorsPerPlane;
    const unsigned int row = vector_in_plane / 2U;
    const unsigned int byte_in_row = 16U * (vector_in_plane & 1U);
    const unsigned int physical_k64 =
        physical_k256_group *
            kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage +
        plane;
    cp_async_16(
        stage.a[plane] +
            sm87_a4w4_swizzled_k64_byte_offset(row, byte_in_row),
        packed_a + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64, byte_in_row,
                       physical_k64_group_count));
  }

  constexpr unsigned int kBVectorsPerPlane =
      64U * kPackedK64Bytes / 16U;
  const unsigned int vector = threadIdx.x;
  const unsigned int plane = vector / kBVectorsPerPlane;
  const unsigned int vector_in_plane =
      vector - plane * kBVectorsPerPlane;
  const unsigned int row = vector_in_plane / 2U;
  const unsigned int byte_in_row = 16U * (vector_in_plane & 1U);
  const unsigned int physical_k64 =
      physical_k256_group *
          kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage +
      plane;
  const std::size_t destination =
      sm87_a4w4_swizzled_k64_byte_offset(row, byte_in_row);
  const std::size_t source = sm87_a4w4_consumer_packed_offset(
      static_cast<std::size_t>(absolute_n64_start) + row,
      physical_k64, byte_in_row, physical_k64_group_count);
  cp_async_16(stage.gate[plane] + destination,
              packed_gate_b + source);
  cp_async_16(stage.up[plane] + destination, packed_up_b + source);
}

__device__ __forceinline__ void issue_final_k512_scales(
    FusedQuantizeFinalScaleSlot& scale,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n64_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        scale.a + first_row,
        a_k512_scales_bf16 +
            sm87_a4w4_gateup_down_edge_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x < 8U) {
    const unsigned int first_row = 8U * threadIdx.x;
    const std::size_t offset =
        sm87_a4w4_gateup_down_edge_scale_offset(
            static_cast<std::size_t>(absolute_n64_start) + first_row,
            k512_group, k512_group_count);
    cp_async_16(scale.gate + first_row,
                gate_b_k512_scales_bf16 + offset);
    cp_async_16(scale.up + first_row,
                up_b_k512_scales_bf16 + offset);
  }
}

__device__ __forceinline__ void issue_final_k512_group(
    FusedQuantizeFinalPipeline& pipeline,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n64_start,
    const unsigned int group,
    const unsigned int group_count,
    const unsigned int physical_k64_group_count) noexcept {
  issue_final_k256_codes(
      pipeline.stage[0U], packed_a, packed_gate_b, packed_up_b,
      m_tile_start, absolute_n64_start, 2U * group,
      physical_k64_group_count);
  issue_final_k512_scales(
      pipeline.scale[group & 1U], a_k512_scales_bf16,
      gate_b_k512_scales_bf16, up_b_k512_scales_bf16,
      m_tile_start, absolute_n64_start, group, group_count);
  cp_async_commit();
  issue_final_k256_codes(
      pipeline.stage[1U], packed_a, packed_gate_b, packed_up_b,
      m_tile_start, absolute_n64_start, 2U * group + 1U,
      physical_k64_group_count);
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

template <typename ScaleSlot>
__device__ __forceinline__ void apply_fragment(
    Float4& gate_accumulator,
    Float4& up_accumulator,
    const Sm87A4W4Accumulator& gate_partial,
    const Sm87A4W4Accumulator& up_partial,
    const ScaleSlot& scale,
    const unsigned int local_m_start,
    const unsigned int local_n_start) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const float a0 =
      decode_bf16(scale.a[local_m_start + coordinate0.m]);
  const float a1 =
      decode_bf16(scale.a[local_m_start + coordinate2.m]);
  const float gate0 =
      decode_bf16(scale.gate[local_n_start + coordinate0.n]);
  const float gate1 =
      decode_bf16(scale.gate[local_n_start + coordinate1.n]);
  const float up0 =
      decode_bf16(scale.up[local_n_start + coordinate0.n]);
  const float up1 =
      decode_bf16(scale.up[local_n_start + coordinate1.n]);
  gate_accumulator.x0 = __fmaf_rn(
      static_cast<float>(gate_partial.x0), __fmul_rn(a0, gate0),
      gate_accumulator.x0);
  gate_accumulator.x1 = __fmaf_rn(
      static_cast<float>(gate_partial.x1), __fmul_rn(a0, gate1),
      gate_accumulator.x1);
  gate_accumulator.x2 = __fmaf_rn(
      static_cast<float>(gate_partial.x2), __fmul_rn(a1, gate0),
      gate_accumulator.x2);
  gate_accumulator.x3 = __fmaf_rn(
      static_cast<float>(gate_partial.x3), __fmul_rn(a1, gate1),
      gate_accumulator.x3);
  up_accumulator.x0 = __fmaf_rn(
      static_cast<float>(up_partial.x0), __fmul_rn(a0, up0),
      up_accumulator.x0);
  up_accumulator.x1 = __fmaf_rn(
      static_cast<float>(up_partial.x1), __fmul_rn(a0, up1),
      up_accumulator.x1);
  up_accumulator.x2 = __fmaf_rn(
      static_cast<float>(up_partial.x2), __fmul_rn(a1, up0),
      up_accumulator.x2);
  up_accumulator.x3 = __fmaf_rn(
      static_cast<float>(up_partial.x3), __fmul_rn(a1, up1),
      up_accumulator.x3);
}

__device__ __forceinline__ void compute_subcell(
    FusedQuantizePipeline& pipeline,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n128_start,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    Float4 (&gate_accumulators)[8U],
    Float4 (&up_accumulators)[8U]) noexcept {
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 8U; ++fragment) {
    gate_accumulators[fragment] = Float4{};
    up_accumulators[fragment] = Float4{};
  }
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m = warp >> 1U;
  const unsigned int warp_n = warp & 1U;
  const unsigned int local_m_start = warp_m * 16U;
  const unsigned int local_n_half = warp_n * 64U;

  for (unsigned int group = 0U; group < k512_group_count; ++group) {
    issue_k512_group(
        pipeline, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, m_tile_start, absolute_n128_start,
        group, k512_group_count, physical_k64_group_count);
    cp_async_wait_all();
    __syncthreads();

    // Only one N8 Gate/Up partial pair is live.  Both K256 stages contribute
    // before the exact K512 scale is applied.
#pragma unroll
    for (unsigned int fragment = 0U; fragment < 8U; ++fragment) {
      Sm87A4W4Accumulator gate_partial{};
      Sm87A4W4Accumulator up_partial{};
#pragma unroll
      for (unsigned int stage = 0U; stage < 2U; ++stage) {
#pragma unroll
        for (unsigned int plane = 0U; plane < 4U; ++plane) {
          const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
              pipeline.stage[stage].a[plane] +
                  local_m_start * kPackedK64Bytes,
              lane);
          const unsigned int fragment_n =
              local_n_half + fragment * 8U;
          const Sm87A4W4BFragment gate = load_b_ldmatrix_x2(
              pipeline.stage[stage].gate[plane] +
                  fragment_n * kPackedK64Bytes,
              lane);
          const Sm87A4W4BFragment up = load_b_ldmatrix_x2(
              pipeline.stage[stage].up[plane] +
                  fragment_n * kPackedK64Bytes,
              lane);
          sm87_a4w4_mma_m16n8k64(gate_partial, a, gate);
          sm87_a4w4_mma_m16n8k64(up_partial, a, up);
        }
      }
      apply_fragment(
          gate_accumulators[fragment], up_accumulators[fragment],
          gate_partial, up_partial, pipeline.scale[group & 1U],
          local_m_start, local_n_half + fragment * 8U);
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void compute_final_half(
    FusedQuantizeFinalPipeline& pipeline,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n64_start,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    Float4 (&gate_accumulators)[4U],
    Float4 (&up_accumulators)[4U]) noexcept {
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
    gate_accumulators[fragment] = Float4{};
    up_accumulators[fragment] = Float4{};
  }
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m = warp >> 1U;
  const unsigned int warp_n = warp & 1U;
  const unsigned int local_m_start = warp_m * 16U;
  const unsigned int local_n_half = warp_n * 32U;

  for (unsigned int group = 0U; group < k512_group_count; ++group) {
    issue_final_k512_group(
        pipeline, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, m_tile_start, absolute_n64_start,
        group, k512_group_count, physical_k64_group_count);
    cp_async_wait_all();
    __syncthreads();

#pragma unroll
    for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
      Sm87A4W4Accumulator gate_partial{};
      Sm87A4W4Accumulator up_partial{};
#pragma unroll
      for (unsigned int stage = 0U; stage < 2U; ++stage) {
#pragma unroll
        for (unsigned int plane = 0U; plane < 4U; ++plane) {
          const Sm87A4W4AFragment a = load_a_ldmatrix_x4(
              pipeline.stage[stage].a[plane] +
                  local_m_start * kPackedK64Bytes,
              lane);
          const unsigned int fragment_n =
              local_n_half + fragment * 8U;
          const Sm87A4W4BFragment gate = load_b_ldmatrix_x2(
              pipeline.stage[stage].gate[plane] +
                  fragment_n * kPackedK64Bytes,
              lane);
          const Sm87A4W4BFragment up = load_b_ldmatrix_x2(
              pipeline.stage[stage].up[plane] +
                  fragment_n * kPackedK64Bytes,
              lane);
          sm87_a4w4_mma_m16n8k64(gate_partial, a, gate);
          sm87_a4w4_mma_m16n8k64(up_partial, a, up);
        }
      }
      apply_fragment(
          gate_accumulators[fragment], up_accumulators[fragment],
          gate_partial, up_partial, pipeline.scale[group & 1U],
          local_m_start, local_n_half + fragment * 8U);
    }
    __syncthreads();
  }
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t pack_product(
    const Float4& gate,
    const Float4& up,
    const bool low_pair,
    const bool valid) noexcept {
  if (!valid) {
    return 0U;
  }
  const float even_gate = low_pair ? gate.x0 : gate.x2;
  const float odd_gate = low_pair ? gate.x1 : gate.x3;
  const float even_up = low_pair ? up.x0 : up.x2;
  const float odd_up = low_pair ? up.x1 : up.x3;
  return static_cast<std::uint32_t>(
             encode_bf16(silu_product(even_gate, even_up))) |
         (static_cast<std::uint32_t>(
              encode_bf16(silu_product(odd_gate, odd_up)))
          << 16U);
}

__device__ __forceinline__ void store_shared_product(
    FusedQuantizeSharedProducts& products,
    const unsigned int product_slot,
    const Float4 (&gate)[8U],
    const Float4 (&up)[8U],
    const unsigned int m_tile_start,
    const unsigned int logical_token_count) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m = (warp >> 1U) * 16U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const bool valid0 =
      m_tile_start + local_m + coordinate0.m < logical_token_count;
  const bool valid1 =
      m_tile_start + local_m + coordinate2.m < logical_token_count;
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 8U; ++fragment) {
    products.word[product_slot][warp][fragment][0U][lane] =
        pack_product(gate[fragment], up[fragment], true, valid0);
    products.word[product_slot][warp][fragment][1U][lane] =
        pack_product(gate[fragment], up[fragment], false, valid1);
  }
  __syncthreads();
}

__device__ __forceinline__ void pack_register_product(
    PackedProduct& products,
    const Float4 (&gate)[8U],
    const Float4 (&up)[8U],
    const unsigned int m_tile_start,
    const unsigned int logical_token_count) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m = (warp >> 1U) * 16U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const bool valid0 =
      m_tile_start + local_m + coordinate0.m < logical_token_count;
  const bool valid1 =
      m_tile_start + local_m + coordinate2.m < logical_token_count;
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 8U; ++fragment) {
    products.word[fragment][0U] =
        pack_product(gate[fragment], up[fragment], true, valid0);
    products.word[fragment][1U] =
        pack_product(gate[fragment], up[fragment], false, valid1);
  }
}

__device__ __forceinline__ void store_third_register_product(
    FusedQuantizeThirdSharedProduct& destination,
    const PackedProduct& products) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 8U; ++fragment) {
    destination.word[warp][fragment][0U][lane] =
        products.word[fragment][0U];
    destination.word[warp][fragment][1U][lane] =
        products.word[fragment][1U];
  }
  __syncthreads();
}

__device__ __forceinline__ void pack_half_register_product(
    PackedHalfProduct& products,
    const Float4 (&gate)[4U],
    const Float4 (&up)[4U],
    const unsigned int m_tile_start,
    const unsigned int logical_token_count) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int local_m = (warp >> 1U) * 16U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const bool valid0 =
      m_tile_start + local_m + coordinate0.m < logical_token_count;
  const bool valid1 =
      m_tile_start + local_m + coordinate2.m < logical_token_count;
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
    products.word[fragment][0U] =
        pack_product(gate[fragment], up[fragment], true, valid0);
    products.word[fragment][1U] =
        pack_product(gate[fragment], up[fragment], false, valid1);
  }
}

[[nodiscard]] __device__ __forceinline__ float word_maximum(
    const std::uint32_t word) noexcept {
  return fmaxf(
      fabsf(decode_bf16(static_cast<std::uint16_t>(word))),
      fabsf(decode_bf16(static_cast<std::uint16_t>(word >> 16U))));
}

__device__ __forceinline__ void prepare_quantization(
    FusedQuantizeShared& shared,
    const FusedQuantizeFinalPhase& final_phase,
    const PackedHalfProduct& fourth_first,
    const PackedHalfProduct& fourth_second,
    const unsigned int m_tile_start,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const float output_clip_ratio,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  auto& scratch = *reinterpret_cast<QuantizeScratch*>(&shared.pipeline);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m = warp >> 1U;
  const unsigned int warp_n = warp & 1U;
  const unsigned int local_m = warp_m * 16U;
  float maximum0 = 0.0F;
  float maximum1 = 0.0F;
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 8U; ++fragment) {
    maximum0 = fmaxf(
        maximum0,
        word_maximum(shared.product.word[0U][warp][fragment][0U][lane]));
    maximum1 = fmaxf(
        maximum1,
        word_maximum(shared.product.word[0U][warp][fragment][1U][lane]));
    maximum0 = fmaxf(
        maximum0,
        word_maximum(shared.product.word[1U][warp][fragment][0U][lane]));
    maximum1 = fmaxf(
        maximum1,
        word_maximum(shared.product.word[1U][warp][fragment][1U][lane]));
    maximum0 = fmaxf(
        maximum0,
        word_maximum(final_phase.third.word[warp][fragment][0U][lane]));
    maximum1 = fmaxf(
        maximum1,
        word_maximum(final_phase.third.word[warp][fragment][1U][lane]));
  }
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
    maximum0 = fmaxf(
        maximum0, word_maximum(fourth_first.word[fragment][0U]));
    maximum1 = fmaxf(
        maximum1, word_maximum(fourth_first.word[fragment][1U]));
    maximum0 = fmaxf(
        maximum0, word_maximum(fourth_second.word[fragment][0U]));
    maximum1 = fmaxf(
        maximum1, word_maximum(fourth_second.word[fragment][1U]));
  }
  maximum0 = fmaxf(maximum0,
                   __shfl_xor_sync(0xffff'ffffU, maximum0, 1U));
  maximum0 = fmaxf(maximum0,
                   __shfl_xor_sync(0xffff'ffffU, maximum0, 2U));
  maximum1 = fmaxf(maximum1,
                   __shfl_xor_sync(0xffff'ffffU, maximum1, 1U));
  maximum1 = fmaxf(maximum1,
                   __shfl_xor_sync(0xffff'ffffU, maximum1, 2U));
  if ((lane & 3U) == 0U) {
    const unsigned int row = local_m + lane / 4U;
    scratch.half_max[row][warp_n] = maximum0;
    scratch.half_max[row + 8U][warp_n] = maximum1;
  }
  __syncthreads();

  if (warp_n == 0U && (lane & 3U) == 0U) {
    const unsigned int row0 = local_m + lane / 4U;
    const unsigned int rows[2U] = {row0, row0 + 8U};
#pragma unroll
    for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
      const unsigned int row = rows[iteration];
      const float maximum =
          fmaxf(scratch.half_max[row][0U], scratch.half_max[row][1U]);
      const float clipped = maximum * output_clip_ratio;
      std::uint16_t scale_bits =
          encode_bf16(maximum == 0.0F ? 1.0F : clipped / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      scratch.clipped_max[row] = clipped;
      scratch.stored_scale[row] = stored_scale;
      output_k512_scales_bf16[
          sm87_a4w4_gateup_down_edge_scale_offset(
              static_cast<std::size_t>(m_tile_start) + row,
              edge_group, edge_group_count)] = scale_bits;
    }
  }
  __syncthreads();
}

[[nodiscard]] __device__ __forceinline__ std::uint8_t quantize_word(
    const std::uint32_t word,
    const float clipped_maximum,
    const float stored_scale) noexcept {
  const float even_value =
      decode_bf16(static_cast<std::uint16_t>(word));
  const float odd_value =
      decode_bf16(static_cast<std::uint16_t>(word >> 16U));
  const float even =
      fminf(fmaxf(even_value, -clipped_maximum), clipped_maximum);
  const float odd =
      fminf(fmaxf(odd_value, -clipped_maximum), clipped_maximum);
  const int even_rounded =
      stored_scale == 0.0F ? 0 : __float2int_rn(even / stored_scale);
  const int odd_rounded =
      stored_scale == 0.0F ? 0 : __float2int_rn(odd / stored_scale);
  const int even_code =
      even_rounded < -7 ? -7 : (even_rounded > 7 ? 7 : even_rounded);
  const int odd_code =
      odd_rounded < -7 ? -7 : (odd_rounded > 7 ? 7 : odd_rounded);
  return sm87_a4w4_pack_signed_pair(even_code, odd_code);
}

__device__ __forceinline__ void publish_quantized_edge(
    FusedQuantizeShared& shared,
    const FusedQuantizeFinalPhase& final_phase,
    const PackedHalfProduct& fourth_first,
    const PackedHalfProduct& fourth_second,
    const unsigned int m_tile_start,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    std::uint8_t* const packed_output) noexcept {
  const auto& scratch =
      *reinterpret_cast<const QuantizeScratch*>(&shared.pipeline);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m = warp >> 1U;
  const unsigned int warp_n = warp & 1U;
  const unsigned int local_m = warp_m * 16U;
  const unsigned int pair_in_n8 = lane & 3U;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const unsigned int rows[2U] = {
      local_m + coordinate0.m, local_m + coordinate2.m};

#pragma unroll
  for (unsigned int cell = 0U; cell < 3U; ++cell) {
#pragma unroll
    for (unsigned int fragment = 0U; fragment < 8U; ++fragment) {
      const unsigned int local_n =
          cell * 128U + warp_n * 64U + fragment * 8U +
          2U * pair_in_n8;
      const unsigned int physical_group =
          edge_group * 8U + local_n / 64U;
      const unsigned int byte_in_group = (local_n & 63U) / 2U;
#pragma unroll
      for (unsigned int row_pair = 0U; row_pair < 2U; ++row_pair) {
        const std::uint32_t word =
            cell < 2U
                ? shared.product.word[cell][warp][fragment][row_pair][lane]
                : final_phase.third.word[warp][fragment][row_pair][lane];
        const unsigned int row = rows[row_pair];
        packed_output[sm87_a4w4_gateup_down_edge_packed_offset(
            static_cast<std::size_t>(m_tile_start) + row,
            physical_group, byte_in_group, edge_group_count * 8U)] =
            quantize_word(word, scratch.clipped_max[row],
                          scratch.stored_scale[row]);
      }
    }
  }

#pragma unroll
  for (unsigned int half = 0U; half < 2U; ++half) {
#pragma unroll
    for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
      const unsigned int local_n =
          384U + half * 64U + warp_n * 32U + fragment * 8U +
          2U * pair_in_n8;
      const unsigned int physical_group =
          edge_group * 8U + local_n / 64U;
      const unsigned int byte_in_group = (local_n & 63U) / 2U;
#pragma unroll
      for (unsigned int row_pair = 0U; row_pair < 2U; ++row_pair) {
        const std::uint32_t word =
            half == 0U ? fourth_first.word[fragment][row_pair]
                       : fourth_second.word[fragment][row_pair];
        const unsigned int row = rows[row_pair];
        packed_output[sm87_a4w4_gateup_down_edge_packed_offset(
            static_cast<std::size_t>(m_tile_start) + row,
            physical_group, byte_in_group, edge_group_count * 8U)] =
            quantize_word(word, scratch.clipped_max[row],
                          scratch.stored_scale[row]);
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void compute_edge(
    FusedQuantizeShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile,
    const unsigned int edge_group,
    const unsigned int edge_group_count,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) noexcept {
  const unsigned int m_tile_start =
      m_tile * kSm87A4W4GateUpK512M128N512FusedQuantizeTileM;
  const unsigned int edge_n_start =
      edge_group * kSm87A4W4GateUpK512M128N512FusedQuantizeTileN;
  {
    Float4 gate[8U];
    Float4 up[8U];

    compute_subcell(
        shared.pipeline, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, m_tile_start, edge_n_start,
        k512_group_count, physical_k64_group_count, gate, up);
    store_shared_product(shared.product, 0U, gate, up, m_tile_start,
                         logical_token_count);

    compute_subcell(
        shared.pipeline, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, m_tile_start, edge_n_start + 128U,
        k512_group_count, physical_k64_group_count, gate, up);
    store_shared_product(shared.product, 1U, gate, up, m_tile_start,
                         logical_token_count);

    compute_subcell(
        shared.pipeline, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, m_tile_start, edge_n_start + 256U,
        k512_group_count, physical_k64_group_count, gate, up);
    PackedProduct third{};
    pack_register_product(third, gate, up, m_tile_start,
                          logical_token_count);
    auto& final_phase =
        *reinterpret_cast<FusedQuantizeFinalPhase*>(&shared.pipeline);
    store_third_register_product(final_phase.third, third);
  }

  // The third product replaces the dead upper full-width pipeline.  The last
  // N128 is then split into two N64 phases, cutting the live Gate+Up state in
  // half while keeping the first phase in eight packed registers.
  auto& final_phase =
      *reinterpret_cast<FusedQuantizeFinalPhase*>(&shared.pipeline);
  Float4 final_gate[4U];
  Float4 final_up[4U];
  compute_final_half(
      final_phase.pipeline, packed_a, a_k512_scales_bf16,
      packed_gate_b, gate_b_k512_scales_bf16, packed_up_b,
      up_b_k512_scales_bf16, m_tile_start, edge_n_start + 384U,
      k512_group_count, physical_k64_group_count, final_gate,
      final_up);
  PackedHalfProduct fourth_first{};
  pack_half_register_product(fourth_first, final_gate, final_up,
                             m_tile_start, logical_token_count);

  compute_final_half(
      final_phase.pipeline, packed_a, a_k512_scales_bf16,
      packed_gate_b, gate_b_k512_scales_bf16, packed_up_b,
      up_b_k512_scales_bf16, m_tile_start, edge_n_start + 448U,
      k512_group_count, physical_k64_group_count, final_gate,
      final_up);
  PackedHalfProduct fourth_second{};
  pack_half_register_product(fourth_second, final_gate, final_up,
                             m_tile_start, logical_token_count);

  // The reduced pipeline is dead and becomes the row-max exchange.  Its
  // scratch prefix does not overlap the third product plane.
  prepare_quantization(
      shared, final_phase, fourth_first, fourth_second, m_tile_start,
      edge_group, edge_group_count, output_clip_ratio,
      output_k512_scales_bf16);
  publish_quantized_edge(
      shared, final_phase, fourth_first, fourth_second, m_tile_start,
      edge_group, edge_group_count, packed_output);
}

}  // namespace

extern "C" __global__
    __launch_bounds__(
        kSm87A4W4GateUpK512M128N512FusedQuantizeThreads,
        kSm87A4W4GateUpK512M128N512FusedQuantizeCtasPerSm)
void q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k512_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int m_tile_count,
    const unsigned int edge_group_count,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k512_scales_bf16) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<FusedQuantizeShared*>(dynamic_shared);
  const unsigned int work_cells = m_tile_count * edge_group_count;
  for (unsigned int cell = blockIdx.x; cell < work_cells;
       cell += gridDim.x) {
    const unsigned int edge_group = cell / m_tile_count;
    const unsigned int m_tile = cell - edge_group * m_tile_count;
    compute_edge(
        shared, packed_a, a_k512_scales_bf16, packed_gate_b,
        gate_b_k512_scales_bf16, packed_up_b,
        up_b_k512_scales_bf16, logical_token_count, m_tile,
        edge_group, edge_group_count, k512_group_count,
        physical_k64_group_count, output_clip_ratio, packed_output,
        output_k512_scales_bf16);
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
      properties.multiProcessorCount != kRequiredSmCount) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes));
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
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k512_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool require_model_shape,
    void* const cuda_stream) noexcept {
  const auto plan = require_model_shape
                        ? sm87_a4w4_gateup_k512_m128n512_fused_quantize_plan(
                              logical_token_count, launch_token_count,
                              intermediate_size, input_size)
                        : sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
                              logical_token_count, launch_token_count,
                              intermediate_size, input_size,
                              maximum_launch_ctas);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k512_scales_bf16, 16U) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k512_scales_bf16, 16U) ||
      !aligned(packed_output, 16U) ||
      !aligned(output_k512_scales_bf16, 16U) ||
      !std::isfinite(output_clip_ratio) || output_clip_ratio <= 0.0F ||
      output_clip_ratio > 1.0F ||
      logical_token_count > std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.edge_groups > std::numeric_limits<unsigned int>::max() ||
      plan.input_k512_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.input_physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.work_cells > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          launch_token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          intermediate_size, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          launch_token_count, input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          intermediate_size, input_size);
  const std::size_t required_output_bytes =
      sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          launch_token_count, intermediate_size);
  const std::size_t required_output_scales =
      sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          launch_token_count, intermediate_size);
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      required_output_bytes == 0U || required_output_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      gate_b_scale_capacity_elements < required_b_scales ||
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

  const auto output_overlaps =
      [&](const void* const pointer, const std::size_t bytes) noexcept {
        return byte_ranges_overlap(packed_output, required_output_bytes,
                                   pointer, bytes) ||
               byte_ranges_overlap(output_k512_scales_bf16,
                                   required_output_scale_bytes, pointer,
                                   bytes);
      };
  if (output_overlaps(packed_a, required_a_bytes) ||
      output_overlaps(a_k512_scales_bf16, required_a_scale_bytes) ||
      output_overlaps(packed_gate_b, required_b_bytes) ||
      output_overlaps(gate_b_k512_scales_bf16,
                      required_b_scale_bytes) ||
      output_overlaps(packed_up_b, required_b_bytes) ||
      output_overlaps(up_b_k512_scales_bf16,
                      required_b_scale_bytes) ||
      byte_ranges_overlap(packed_output, required_output_bytes,
                          output_k512_scales_bf16,
                          required_output_scale_bytes)) {
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
    if (!g_fused_quantize_resources_ready.load(
            std::memory_order_acquire)) {
      return static_cast<int>(cudaErrorNotReady);
    }
  } else if (!g_fused_quantize_resources_ready.load(
                 std::memory_order_acquire)) {
    Sm87A4W4GateUpK512M128N512FusedQuantizeResources resources{};
    const int resource_status =
        query_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resources_cuda(
            &resources);
    if (resource_status != static_cast<int>(cudaSuccess)) {
      return resource_status;
    }
  }
  const unsigned int planned = static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      planned < maximum_launch_ctas ? planned : maximum_launch_ctas;
  (void)cudaGetLastError();
  q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_kernel
      <<<launch_ctas,
         static_cast<unsigned int>(
             kSm87A4W4GateUpK512M128N512FusedQuantizeThreads),
         kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes,
         stream>>>(
          packed_a, a_k512_scales_bf16, packed_gate_b,
          gate_b_k512_scales_bf16, packed_up_b,
          up_b_k512_scales_bf16,
          static_cast<unsigned int>(logical_token_count),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.edge_groups),
          static_cast<unsigned int>(plan.input_k512_groups),
          static_cast<unsigned int>(plan.input_physical_k64_groups),
          output_clip_ratio, packed_output, output_k512_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resources_cuda(
    Sm87A4W4GateUpK512M128N512FusedQuantizeResources* const resources)
    noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpK512M128N512FusedQuantizeResources{};
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
      q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_k512_m128n512_fused_quantize_kernel,
      static_cast<int>(
          kSm87A4W4GateUpK512M128N512FusedQuantizeThreads),
      kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes;
  resources->configured_dynamic_shared_limit_bytes =
      static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes);
  resources->device_optin_shared_limit_bytes =
      static_cast<std::size_t>(properties.sharedMemPerBlockOptin);
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4GateUpK512M128N512FusedQuantizeMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpK512M128N512FusedQuantizeThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(
              kSm87A4W4GateUpK512M128N512FusedQuantizeCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  g_fused_quantize_resources_ready.store(true,
                                         std::memory_order_release);
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_k512_m128n512_fused_quantize_cuda(
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
    const std::size_t input_size,
    const float output_clip_ratio,
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
          kSm87A4W4GateUpK512M128N512FusedQuantizePersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_cuda(
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
    const std::size_t input_size,
    const float output_clip_ratio,
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
