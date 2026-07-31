#include "q3x/kernels/sm87_a4w4_gateup_paired.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr std::size_t kPackedK64Bytes =
    kSm87A4W4GateUpTileK / 2U;
inline constexpr std::size_t kStageABytes =
    kSm87A4W4GateUpTileM * kPackedK64Bytes;
inline constexpr std::size_t kStageBBytes =
    kSm87A4W4GateUpTileN * kPackedK64Bytes;
inline constexpr std::size_t kPackedVectorsPerStage =
    (kStageABytes + 2U * kStageBBytes) / 16U;
inline constexpr int kRequiredSmCount = 16;

struct alignas(16) Sm87A4W4GateUpPipelineStage final {
  std::uint8_t a[kStageABytes];
  std::uint8_t gate_b[kStageBBytes];
  std::uint8_t up_b[kStageBBytes];
  std::uint16_t a_scales[kSm87A4W4GateUpTileM];
  std::uint16_t gate_b_scales[kSm87A4W4GateUpTileN];
  std::uint16_t up_b_scales[kSm87A4W4GateUpTileN];
};

struct alignas(16) Sm87A4W4GateUpSharedStorage final {
  Sm87A4W4GateUpPipelineStage
      pipeline[kSm87A4W4GateUpPipelineStages];
  float product[kSm87A4W4GateUpTileM][kSm87A4W4GateUpTileN];
};

static_assert(kPackedVectorsPerStage == 576U);
static_assert(sizeof(Sm87A4W4GateUpPipelineStage) == 9'792U);
static_assert(sizeof(Sm87A4W4GateUpSharedStorage) == 45'760U);
static_assert(sizeof(Sm87A4W4GateUpSharedStorage) *
                  kSm87A4W4GateUpCtasPerSm <=
              96U * 1'024U);

inline constexpr std::size_t kLargeMStageABytes =
    kSm87A4W4GateUpLargeMTileM * kPackedK64Bytes;
inline constexpr std::size_t kLargeMStageBBytes =
    kSm87A4W4GateUpLargeMTileN * kPackedK64Bytes;
inline constexpr std::size_t kLargeMPackedVectorsPerStage =
    (kLargeMStageABytes + 2U * kLargeMStageBBytes) / 16U;

struct alignas(16) Sm87A4W4GateUpLargeMPipelineStage final {
  std::uint8_t a[kLargeMStageABytes];
  std::uint8_t gate_b[kLargeMStageBBytes];
  std::uint8_t up_b[kLargeMStageBBytes];
  std::uint16_t a_scales[kSm87A4W4GateUpLargeMTileM];
  std::uint16_t gate_b_scales[kSm87A4W4GateUpLargeMTileN];
  std::uint16_t up_b_scales[kSm87A4W4GateUpLargeMTileN];
};

struct alignas(16) Sm87A4W4GateUpLargeMSharedStorage final {
  Sm87A4W4GateUpLargeMPipelineStage
      pipeline[kSm87A4W4GateUpPipelineStages];
  float product[kSm87A4W4GateUpLargeMTileM]
               [kSm87A4W4GateUpLargeMTileN];
};

static_assert(kLargeMPackedVectorsPerStage == 384U);
static_assert(sizeof(Sm87A4W4GateUpLargeMPipelineStage) == 6'528U);
static_assert(sizeof(Sm87A4W4GateUpLargeMSharedStorage) == 35'968U);
static_assert(sizeof(Sm87A4W4GateUpLargeMSharedStorage) *
                  kSm87A4W4GateUpCtasPerSm <=
              96U * 1'024U);

inline constexpr std::size_t kWideLargeMStageABytes =
    kSm87A4W4GateUpWideLargeMTileM * kPackedK64Bytes;
inline constexpr std::size_t kWideLargeMStageBBytes =
    kSm87A4W4GateUpWideLargeMTileN * kPackedK64Bytes;
inline constexpr std::size_t kWideLargeMPackedVectorsPerK64 =
    (kWideLargeMStageABytes + 2U * kWideLargeMStageBBytes) / 16U;
inline constexpr std::size_t kWideLargeMK64PerLogicalStage =
    kSm87A4W4GateUpWideLargeMLogicalTileK /
    kSm87A4W4GateUpWideLargeMTileK;

struct alignas(16) Sm87A4W4GateUpWideLargeMK64Stage final {
  std::uint8_t a[kWideLargeMStageABytes];
  std::uint8_t gate_b[kWideLargeMStageBBytes];
  std::uint8_t up_b[kWideLargeMStageBBytes];
  std::uint16_t a_scales[kSm87A4W4GateUpWideLargeMTileM];
  std::uint16_t gate_b_scales[kSm87A4W4GateUpWideLargeMTileN];
  std::uint16_t up_b_scales[kSm87A4W4GateUpWideLargeMTileN];
};

struct alignas(16) Sm87A4W4GateUpWideLargeMLogicalStage final {
  Sm87A4W4GateUpWideLargeMK64Stage
      k64[kWideLargeMK64PerLogicalStage];
};

struct alignas(16) Sm87A4W4GateUpWideLargeMPipeline final {
  Sm87A4W4GateUpWideLargeMLogicalStage
      slot[kSm87A4W4GateUpWideLargeMPipelineSlots];
};

// Pipeline bytes and the FP32 SiLU product never overlap in lifetime.  The
// union keeps the two logical K128 slots resident for load/compute overlap
// without paying their 42.5-KiB footprint again during the quantizing
// epilogue.  This is the structural condition that preserves 2 CTA/SM.
union alignas(16) Sm87A4W4GateUpWideLargeMSharedStorage final {
  Sm87A4W4GateUpWideLargeMPipeline pipeline;
  float product[kSm87A4W4GateUpWideLargeMTileM]
               [kSm87A4W4GateUpWideLargeMTileN];
};

static_assert(kWideLargeMK64PerLogicalStage == 2U);
static_assert(kWideLargeMPackedVectorsPerK64 == 640U);
static_assert(sizeof(Sm87A4W4GateUpWideLargeMK64Stage) == 10'880U);
static_assert(sizeof(Sm87A4W4GateUpWideLargeMLogicalStage) == 21'760U);
static_assert(sizeof(Sm87A4W4GateUpWideLargeMPipeline) == 43'520U);
static_assert(sizeof(Sm87A4W4GateUpWideLargeMSharedStorage) == 43'520U);
static_assert(sizeof(Sm87A4W4GateUpWideLargeMSharedStorage) *
                  kSm87A4W4GateUpCtasPerSm <=
              96U * 1'024U);

inline constexpr std::size_t kK128StageABytes =
    kSm87A4W4GateUpK128TileM * kPackedK64Bytes;
inline constexpr std::size_t kK128StageBBytes =
    kSm87A4W4GateUpK128TileN * kPackedK64Bytes;
inline constexpr std::size_t kK128PackedVectorsPerPhysicalK64 =
    (kK128StageABytes + 2U * kK128StageBBytes) / 16U;

// One logical K128 stage retains two independently swizzled physical K64
// code planes but only one scale plane.  Its packed-code byte count matches
// the generic M64N256 K128 stage: paired Gate+Up substitutes two N128 B
// operands for the generic N256 B operand.
struct alignas(16) Sm87A4W4GateUpK128Stage final {
  std::uint8_t
      a[kSm87A4W4PhysicalK64BlocksPerSharedScale][kK128StageABytes];
  std::uint8_t
      gate_b[kSm87A4W4PhysicalK64BlocksPerSharedScale][kK128StageBBytes];
  std::uint8_t
      up_b[kSm87A4W4PhysicalK64BlocksPerSharedScale][kK128StageBBytes];
  std::uint16_t a_scales[kSm87A4W4GateUpK128TileM];
  std::uint16_t gate_b_scales[kSm87A4W4GateUpK128TileN];
  std::uint16_t up_b_scales[kSm87A4W4GateUpK128TileN];
};

struct alignas(16) Sm87A4W4GateUpK128Pipeline final {
  Sm87A4W4GateUpK128Stage slot[kSm87A4W4GateUpK128PipelineSlots];
};

// The code/scale pipeline is dead before the fused product is materialized,
// so the epilogue reuses the same shared allocation.  At 42,240 bytes this
// leaves the exact two-CTA/SM shared-memory budget intact.
union alignas(16) Sm87A4W4GateUpK128SharedStorage final {
  Sm87A4W4GateUpK128Pipeline pipeline;
  float product[kSm87A4W4GateUpK128TileM]
               [kSm87A4W4GateUpK128TileN];
};

static_assert(kK128PackedVectorsPerPhysicalK64 == 640U);
static_assert(sizeof(Sm87A4W4GateUpK128Stage) == 21'120U);
static_assert(sizeof(Sm87A4W4GateUpK128Pipeline) == 42'240U);
static_assert(sizeof(Sm87A4W4GateUpK128SharedStorage) == 42'240U);
static_assert(sizeof(Sm87A4W4GateUpK128SharedStorage) <= 48U * 1'024U);
static_assert(sizeof(Sm87A4W4GateUpK128SharedStorage) *
                  kSm87A4W4GateUpCtasPerSm <=
              96U * 1'024U);

[[nodiscard]] constexpr bool aligned(const void* const pointer,
                                     const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] constexpr bool product_fits(const std::size_t first,
                                          const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr bool consumer_capacity_fits(
    const std::size_t outer_count,
    const std::size_t k64_group_count) noexcept {
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return blocks != 0U && k64_group_count != 0U &&
         product_fits(blocks, k64_group_count) &&
         product_fits(blocks * k64_group_count,
                      kSm87A4W4ConsumerOuterBlock) &&
         product_fits(blocks * k64_group_count *
                          kSm87A4W4ConsumerOuterBlock,
                      kSm87A4W4ConsumerPackedKBlockBytes);
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
  // This branch avoids exp overflow for large negative Gate values while
  // preserving an FP32 SiLU boundary before output quantization.
  if (gate >= 0.0F) {
    return (gate / (1.0F + expf(-gate))) * up;
  }
  const float exponential = expf(gate);
  return (gate * exponential / (1.0F + exponential)) * up;
}

__device__ __forceinline__ void cp_async_16(
    void* const destination, const void* const source,
    const unsigned int source_bytes) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(shared_address), "l"(source), "r"(source_bytes)
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

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(PendingGroups)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void prefetch_stage(
    Sm87A4W4GateUpPipelineStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t token_count,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t k64_group,
    const std::size_t k64_group_count) noexcept {
  // The three global consumer blocks are contiguous in N64 halves. A is
  // staged once and subsequently consumed by both projection branches.
  for (std::size_t vector = threadIdx.x; vector < kPackedVectorsPerStage;
       vector += blockDim.x) {
    if (vector < kStageABytes / 16U) {
      const std::size_t row = vector / 2U;
      const std::size_t row_vector = vector % 2U;
      const std::size_t global_row = m_tile_start + row;
      const bool valid = global_row < token_count;
      const std::uint8_t* const source =
          valid ? packed_a + sm87_a4w4_consumer_packed_offset(
                                 global_row, k64_group,
                                 row_vector * 16U, k64_group_count)
                : packed_a;
      cp_async_16(
          stage.a + sm87_a4w4_swizzled_k64_byte_offset(
                        row, row_vector * 16U),
          source, valid ? 16U : 0U);
    } else if (vector < (kStageABytes + kStageBBytes) / 16U) {
      const std::size_t gate_vector = vector - kStageABytes / 16U;
      const std::size_t row = gate_vector / 2U;
      const std::size_t row_vector = gate_vector % 2U;
      const std::uint8_t* const source =
          packed_gate_b + sm87_a4w4_consumer_packed_offset(
                              n_tile_start + row, k64_group,
                              row_vector * 16U, k64_group_count);
      cp_async_16(
          stage.gate_b + sm87_a4w4_swizzled_k64_byte_offset(
                             row, row_vector * 16U),
          source, 16U);
    } else {
      const std::size_t up_vector =
          vector - (kStageABytes + kStageBBytes) / 16U;
      const std::size_t row = up_vector / 2U;
      const std::size_t row_vector = up_vector % 2U;
      const std::uint8_t* const source =
          packed_up_b + sm87_a4w4_consumer_packed_offset(
                            n_tile_start + row, k64_group,
                            row_vector * 16U, k64_group_count);
      cp_async_16(
          stage.up_b + sm87_a4w4_swizzled_k64_byte_offset(
                           row, row_vector * 16U),
          source, 16U);
    }
  }

  if (threadIdx.x < kSm87A4W4GateUpTileM) {
    const std::size_t global_row = m_tile_start + threadIdx.x;
    stage.a_scales[threadIdx.x] =
        global_row < token_count
            ? a_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                  global_row, k64_group, k64_group_count)]
            : static_cast<std::uint16_t>(0U);
  }
  if (threadIdx.x < kSm87A4W4GateUpTileN) {
    const std::size_t global_row = n_tile_start + threadIdx.x;
    stage.gate_b_scales[threadIdx.x] =
        gate_b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_row, k64_group, k64_group_count)];
    stage.up_b_scales[threadIdx.x] =
        up_b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_row, k64_group, k64_group_count)];
  }
  cp_async_commit();
}

__device__ __forceinline__ void prefetch_large_m_stage(
    Sm87A4W4GateUpLargeMPipelineStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t k64_group,
    const std::size_t k64_group_count) noexcept {
  // M is guaranteed to be an exact multiple of 64 on this path, so all A
  // transfers are full-width.  The CTA stages half as many Gate/Up rows as
  // M32N128 while retaining the same three-stage pipeline.
  for (std::size_t vector = threadIdx.x;
       vector < kLargeMPackedVectorsPerStage; vector += blockDim.x) {
    if (vector < kLargeMStageABytes / 16U) {
      const std::size_t row = vector / 2U;
      const std::size_t row_vector = vector % 2U;
      const std::uint8_t* const source =
          packed_a + sm87_a4w4_consumer_packed_offset(
                         m_tile_start + row, k64_group,
                         row_vector * 16U, k64_group_count);
      cp_async_16(
          stage.a + sm87_a4w4_swizzled_k64_byte_offset(
                        row, row_vector * 16U),
          source, 16U);
    } else if (vector <
               (kLargeMStageABytes + kLargeMStageBBytes) / 16U) {
      const std::size_t gate_vector =
          vector - kLargeMStageABytes / 16U;
      const std::size_t row = gate_vector / 2U;
      const std::size_t row_vector = gate_vector % 2U;
      const std::uint8_t* const source =
          packed_gate_b + sm87_a4w4_consumer_packed_offset(
                              n_tile_start + row, k64_group,
                              row_vector * 16U, k64_group_count);
      cp_async_16(
          stage.gate_b + sm87_a4w4_swizzled_k64_byte_offset(
                             row, row_vector * 16U),
          source, 16U);
    } else {
      const std::size_t up_vector =
          vector - (kLargeMStageABytes + kLargeMStageBBytes) / 16U;
      const std::size_t row = up_vector / 2U;
      const std::size_t row_vector = up_vector % 2U;
      const std::uint8_t* const source =
          packed_up_b + sm87_a4w4_consumer_packed_offset(
                            n_tile_start + row, k64_group,
                            row_vector * 16U, k64_group_count);
      cp_async_16(
          stage.up_b + sm87_a4w4_swizzled_k64_byte_offset(
                           row, row_vector * 16U),
          source, 16U);
    }
  }

  if (threadIdx.x < kSm87A4W4GateUpLargeMTileM) {
    stage.a_scales[threadIdx.x] =
        a_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            m_tile_start + threadIdx.x, k64_group, k64_group_count)];
  }
  if (threadIdx.x < kSm87A4W4GateUpLargeMTileN) {
    const std::size_t global_row = n_tile_start + threadIdx.x;
    stage.gate_b_scales[threadIdx.x] =
        gate_b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_row, k64_group, k64_group_count)];
    stage.up_b_scales[threadIdx.x] =
        up_b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_row, k64_group, k64_group_count)];
  }
  cp_async_commit();
}

__device__ __forceinline__ void prefetch_wide_large_m_k64_stage(
    Sm87A4W4GateUpWideLargeMK64Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t k64_group,
    const std::size_t k64_group_count) noexcept {
  const bool valid_group = k64_group < k64_group_count;
  for (std::size_t vector = threadIdx.x;
       vector < kWideLargeMPackedVectorsPerK64; vector += blockDim.x) {
    if (vector < kWideLargeMStageABytes / 16U) {
      const std::size_t row = vector / 2U;
      const std::size_t row_vector = vector % 2U;
      const std::uint8_t* const source =
          valid_group
              ? packed_a + sm87_a4w4_consumer_packed_offset(
                               m_tile_start + row, k64_group,
                               row_vector * 16U, k64_group_count)
              : packed_a;
      cp_async_16(
          stage.a + sm87_a4w4_swizzled_k64_byte_offset(
                        row, row_vector * 16U),
          source, valid_group ? 16U : 0U);
    } else if (vector <
               (kWideLargeMStageABytes + kWideLargeMStageBBytes) / 16U) {
      const std::size_t gate_vector =
          vector - kWideLargeMStageABytes / 16U;
      const std::size_t row = gate_vector / 2U;
      const std::size_t row_vector = gate_vector % 2U;
      const std::uint8_t* const source =
          valid_group
              ? packed_gate_b + sm87_a4w4_consumer_packed_offset(
                                      n_tile_start + row, k64_group,
                                      row_vector * 16U, k64_group_count)
              : packed_gate_b;
      cp_async_16(
          stage.gate_b + sm87_a4w4_swizzled_k64_byte_offset(
                             row, row_vector * 16U),
          source, valid_group ? 16U : 0U);
    } else {
      const std::size_t up_vector =
          vector -
          (kWideLargeMStageABytes + kWideLargeMStageBBytes) / 16U;
      const std::size_t row = up_vector / 2U;
      const std::size_t row_vector = up_vector % 2U;
      const std::uint8_t* const source =
          valid_group
              ? packed_up_b + sm87_a4w4_consumer_packed_offset(
                                    n_tile_start + row, k64_group,
                                    row_vector * 16U, k64_group_count)
              : packed_up_b;
      cp_async_16(
          stage.up_b + sm87_a4w4_swizzled_k64_byte_offset(
                           row, row_vector * 16U),
          source, valid_group ? 16U : 0U);
    }
  }

  if (threadIdx.x < kSm87A4W4GateUpWideLargeMTileM) {
    stage.a_scales[threadIdx.x] =
        valid_group
            ? a_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                  m_tile_start + threadIdx.x, k64_group,
                  k64_group_count)]
            : static_cast<std::uint16_t>(0U);
  }
  if (threadIdx.x < kSm87A4W4GateUpWideLargeMTileN) {
    const std::size_t global_row = n_tile_start + threadIdx.x;
    stage.gate_b_scales[threadIdx.x] =
        valid_group
            ? gate_b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                  global_row, k64_group, k64_group_count)]
            : static_cast<std::uint16_t>(0U);
    stage.up_b_scales[threadIdx.x] =
        valid_group
            ? up_b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                  global_row, k64_group, k64_group_count)]
            : static_cast<std::uint16_t>(0U);
  }
  // Each K64 half remains its own async group even though two halves share a
  // logical K128 slot.  That pins both scale association and accumulation
  // order to the established K64 numerical contract.
  cp_async_commit();
}

__device__ __forceinline__ void prefetch_wide_large_m_logical_stage(
    Sm87A4W4GateUpWideLargeMLogicalStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t logical_stage,
    const std::size_t k64_group_count) noexcept {
#pragma unroll
  for (unsigned int half = 0U;
       half < kWideLargeMK64PerLogicalStage; ++half) {
    prefetch_wide_large_m_k64_stage(
        stage.k64[half], packed_a, a_k64_scales_bf16, packed_gate_b,
        gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
        m_tile_start, n_tile_start,
        logical_stage * kWideLargeMK64PerLogicalStage + half,
        k64_group_count);
  }
}

__device__ __forceinline__ void accumulate_wide_large_m_k64_stage(
    const Sm87A4W4GateUpWideLargeMK64Stage& stage,
    const std::size_t warp_m,
    const std::size_t warp_n,
    const unsigned int lane,
    float (&gate_accumulators)[2U][4U][4U],
    float (&up_accumulators)[2U][4U][4U]) noexcept {
  Sm87A4W4AFragment a_fragments[2U];
  float a_scales[2U][2U];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
    const std::size_t panel_m = warp_m * 32U + m_panel * 16U;
    a_fragments[m_panel] = sm87_a4w4_load_a_fragment_swizzled_shared(
        stage.a + panel_m * kPackedK64Bytes, lane);
    const std::size_t local_m0 = panel_m + lane / 4U;
    const std::size_t local_m1 = local_m0 + 8U;
    a_scales[m_panel][0U] = decode_bf16(stage.a_scales[local_m0]);
    a_scales[m_panel][1U] = decode_bf16(stage.a_scales[local_m1]);
  }

#pragma unroll
  for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
    const std::size_t fragment_n = warp_n * 32U + fragment * 8U;
    const Sm87A4W4BFragment gate_fragment =
        sm87_a4w4_load_b_fragment_swizzled_shared(
            stage.gate_b + fragment_n * kPackedK64Bytes, lane);
    const Sm87A4W4BFragment up_fragment =
        sm87_a4w4_load_b_fragment_swizzled_shared(
            stage.up_b + fragment_n * kPackedK64Bytes, lane);
    const std::size_t local_n0 = fragment_n + 2U * (lane % 4U);
    const std::size_t local_n1 = local_n0 + 1U;
    const float gate_scale0 =
        decode_bf16(stage.gate_b_scales[local_n0]);
    const float gate_scale1 =
        decode_bf16(stage.gate_b_scales[local_n1]);
    const float up_scale0 = decode_bf16(stage.up_b_scales[local_n0]);
    const float up_scale1 = decode_bf16(stage.up_b_scales[local_n1]);

#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
      Sm87A4W4Accumulator gate_partial{};
      Sm87A4W4Accumulator up_partial{};
      sm87_a4w4_mma_m16n8k64(
          gate_partial, a_fragments[m_panel], gate_fragment);
      sm87_a4w4_mma_m16n8k64(
          up_partial, a_fragments[m_panel], up_fragment);
      gate_accumulators[m_panel][fragment][0U] +=
          static_cast<float>(gate_partial.x0) * a_scales[m_panel][0U] *
          gate_scale0;
      gate_accumulators[m_panel][fragment][1U] +=
          static_cast<float>(gate_partial.x1) * a_scales[m_panel][0U] *
          gate_scale1;
      gate_accumulators[m_panel][fragment][2U] +=
          static_cast<float>(gate_partial.x2) * a_scales[m_panel][1U] *
          gate_scale0;
      gate_accumulators[m_panel][fragment][3U] +=
          static_cast<float>(gate_partial.x3) * a_scales[m_panel][1U] *
          gate_scale1;
      up_accumulators[m_panel][fragment][0U] +=
          static_cast<float>(up_partial.x0) * a_scales[m_panel][0U] *
          up_scale0;
      up_accumulators[m_panel][fragment][1U] +=
          static_cast<float>(up_partial.x1) * a_scales[m_panel][0U] *
          up_scale1;
      up_accumulators[m_panel][fragment][2U] +=
          static_cast<float>(up_partial.x2) * a_scales[m_panel][1U] *
          up_scale0;
      up_accumulators[m_panel][fragment][3U] +=
          static_cast<float>(up_partial.x3) * a_scales[m_panel][1U] *
          up_scale1;
    }
  }
}

__device__ __forceinline__ void prefetch_gateup_k128_stage(
    Sm87A4W4GateUpK128Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr std::size_t kVectorCount =
      kSm87A4W4PhysicalK64BlocksPerSharedScale *
      kK128PackedVectorsPerPhysicalK64;
  static_assert(kVectorCount == 5U * kSm87A4W4GateUpThreads);

  for (std::size_t vector = threadIdx.x; vector < kVectorCount;
       vector += blockDim.x) {
    const std::size_t half =
        vector / kK128PackedVectorsPerPhysicalK64;
    const std::size_t group_vector =
        vector - half * kK128PackedVectorsPerPhysicalK64;
    const std::size_t physical_group =
        static_cast<std::size_t>(k128_group) *
            kSm87A4W4PhysicalK64BlocksPerSharedScale +
        half;
    if (group_vector < kK128StageABytes / 16U) {
      const std::size_t row = group_vector / 2U;
      const std::size_t row_vector = group_vector % 2U;
      cp_async_16(
          stage.a[half] + sm87_a4w4_swizzled_k64_byte_offset(
                              row, row_vector * 16U),
          packed_a + sm87_a4w4_consumer_packed_offset(
                         static_cast<std::size_t>(m_tile_start) + row,
                         physical_group, row_vector * 16U,
                         physical_k64_group_count),
          16U);
    } else if (group_vector <
               (kK128StageABytes + kK128StageBBytes) / 16U) {
      const std::size_t gate_vector =
          group_vector - kK128StageABytes / 16U;
      const std::size_t row = gate_vector / 2U;
      const std::size_t row_vector = gate_vector % 2U;
      cp_async_16(
          stage.gate_b[half] + sm87_a4w4_swizzled_k64_byte_offset(
                                   row, row_vector * 16U),
          packed_gate_b + sm87_a4w4_consumer_packed_offset(
                                static_cast<std::size_t>(n_tile_start) + row,
                                physical_group, row_vector * 16U,
                                physical_k64_group_count),
          16U);
    } else {
      const std::size_t up_vector =
          group_vector -
          (kK128StageABytes + kK128StageBBytes) / 16U;
      const std::size_t row = up_vector / 2U;
      const std::size_t row_vector = up_vector % 2U;
      cp_async_16(
          stage.up_b[half] + sm87_a4w4_swizzled_k64_byte_offset(
                                 row, row_vector * 16U),
          packed_up_b + sm87_a4w4_consumer_packed_offset(
                              static_cast<std::size_t>(n_tile_start) + row,
                              physical_group, row_vector * 16U,
                              physical_k64_group_count),
          16U);
    }
  }

  if (threadIdx.x < kSm87A4W4GateUpK128TileM) {
    stage.a_scales[threadIdx.x] =
        a_k128_scales_bf16[sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(m_tile_start) + threadIdx.x,
            k128_group, k128_group_count)];
  }
  if (threadIdx.x < kSm87A4W4GateUpK128TileN) {
    const std::size_t global_n =
        static_cast<std::size_t>(n_tile_start) + threadIdx.x;
    stage.gate_b_scales[threadIdx.x] =
        gate_b_k128_scales_bf16[
            sm87_a4w4_consumer_k128_scale_offset(
                global_n, k128_group, k128_group_count)];
    stage.up_b_scales[threadIdx.x] =
        up_b_k128_scales_bf16[
            sm87_a4w4_consumer_k128_scale_offset(
                global_n, k128_group, k128_group_count)];
  }
  cp_async_commit();
}

__device__ __forceinline__ void accumulate_gateup_k128_stage(
    const Sm87A4W4GateUpK128Stage& stage,
    const unsigned int warp_m,
    const unsigned int warp_n,
    const unsigned int lane,
    float (&gate_accumulators)[2U][4U][4U],
    float (&up_accumulators)[2U][4U][4U]) noexcept {
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
    const unsigned int fragment_n =
        warp_n * 32U + fragment * 8U;
    const unsigned int local_n0 =
        fragment_n + 2U * (lane % 4U);
    const unsigned int local_n1 = local_n0 + 1U;
    const float gate_scale0 =
        decode_bf16(stage.gate_b_scales[local_n0]);
    const float gate_scale1 =
        decode_bf16(stage.gate_b_scales[local_n1]);
    const float up_scale0 = decode_bf16(stage.up_b_scales[local_n0]);
    const float up_scale1 = decode_bf16(stage.up_b_scales[local_n1]);
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
      Sm87A4W4Accumulator gate_partial{};
      Sm87A4W4Accumulator up_partial{};
      const unsigned int panel_m =
          warp_m * 32U + m_panel * 16U;
#pragma unroll
      for (unsigned int half = 0U;
           half < kSm87A4W4PhysicalK64BlocksPerSharedScale; ++half) {
        const Sm87A4W4AFragment a_fragment =
            sm87_a4w4_load_a_fragment_swizzled_shared(
                stage.a[half] + panel_m * kPackedK64Bytes, lane);
        const Sm87A4W4BFragment gate_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.gate_b[half] + fragment_n * kPackedK64Bytes, lane);
        const Sm87A4W4BFragment up_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.up_b[half] + fragment_n * kPackedK64Bytes, lane);
        // The second native K64 operation consumes the first operation's S32
        // result as C.  Both halves therefore form one exact K128 partial.
        sm87_a4w4_mma_m16n8k64(
            gate_partial, a_fragment, gate_fragment);
        sm87_a4w4_mma_m16n8k64(up_partial, a_fragment, up_fragment);
      }
      const unsigned int local_m0 = panel_m + lane / 4U;
      const unsigned int local_m1 = local_m0 + 8U;
      const float a_scale0 = decode_bf16(stage.a_scales[local_m0]);
      const float a_scale1 = decode_bf16(stage.a_scales[local_m1]);
      // Each S32 component crosses to FP32 exactly once, after both K64 MMAs.
      gate_accumulators[m_panel][fragment][0U] +=
          static_cast<float>(gate_partial.x0) * (a_scale0 * gate_scale0);
      gate_accumulators[m_panel][fragment][1U] +=
          static_cast<float>(gate_partial.x1) * (a_scale0 * gate_scale1);
      gate_accumulators[m_panel][fragment][2U] +=
          static_cast<float>(gate_partial.x2) * (a_scale1 * gate_scale0);
      gate_accumulators[m_panel][fragment][3U] +=
          static_cast<float>(gate_partial.x3) * (a_scale1 * gate_scale1);
      up_accumulators[m_panel][fragment][0U] +=
          static_cast<float>(up_partial.x0) * (a_scale0 * up_scale0);
      up_accumulators[m_panel][fragment][1U] +=
          static_cast<float>(up_partial.x1) * (a_scale0 * up_scale1);
      up_accumulators[m_panel][fragment][2U] +=
          static_cast<float>(up_partial.x2) * (a_scale1 * up_scale0);
      up_accumulators[m_panel][fragment][3U] +=
          static_cast<float>(up_partial.x3) * (a_scale1 * up_scale1);
    }
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpThreads,
                      kSm87A4W4GateUpCtasPerSm)
void q3x_sm87_a4w4_gateup_paired_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t token_count,
    const std::size_t k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k64_scales_bf16,
    const std::size_t output_k64_group_count,
    const std::size_t m_tile_count,
    const std::size_t work_tile_count) {
  __shared__ Sm87A4W4GateUpSharedStorage shared;

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const std::size_t warp_m = warp % 2U;
  const std::size_t warp_n = warp / 2U;

  for (std::size_t work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const std::size_t n_tile = work_tile / m_tile_count;
    const std::size_t m_tile = work_tile - n_tile * m_tile_count;
    const std::size_t m_tile_start = m_tile * kSm87A4W4GateUpTileM;
    const std::size_t n_tile_start = n_tile * kSm87A4W4GateUpTileN;

    float gate_accumulators[4U][4U]{};
    float up_accumulators[4U][4U]{};

    prefetch_stage(
        shared.pipeline[0U], packed_a, a_k64_scales_bf16, packed_gate_b,
        gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
        token_count, m_tile_start, n_tile_start, 0U, k64_group_count);
    if (k64_group_count > 1U) {
      prefetch_stage(
          shared.pipeline[1U], packed_a, a_k64_scales_bf16, packed_gate_b,
          gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
          token_count, m_tile_start, n_tile_start, 1U, k64_group_count);
    }

    for (std::size_t group = 0U; group < k64_group_count; ++group) {
      if (group + 2U < k64_group_count) {
        prefetch_stage(
            shared.pipeline[(group + 2U) %
                            kSm87A4W4GateUpPipelineStages],
            packed_a, a_k64_scales_bf16, packed_gate_b,
            gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
            token_count, m_tile_start, n_tile_start, group + 2U,
            k64_group_count);
      }
      if (group + 1U == k64_group_count) {
        cp_async_wait<0>();
      } else {
        cp_async_wait<1>();
      }
      __syncthreads();

      const Sm87A4W4GateUpPipelineStage& stage =
          shared.pipeline[group % kSm87A4W4GateUpPipelineStages];
      const std::uint8_t* const warp_a =
          stage.a + warp_m * 16U * kPackedK64Bytes;
      const Sm87A4W4AFragment a =
          sm87_a4w4_load_a_fragment_swizzled_shared(warp_a, lane);
      const std::size_t local_m0 = warp_m * 16U + lane / 4U;
      const std::size_t local_m1 = local_m0 + 8U;
      const float a_scale0 = decode_bf16(stage.a_scales[local_m0]);
      const float a_scale1 = decode_bf16(stage.a_scales[local_m1]);

#pragma unroll
      for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
        const std::size_t fragment_n = warp_n * 32U + fragment * 8U;
        const std::uint8_t* const gate_b =
            stage.gate_b + fragment_n * kPackedK64Bytes;
        const std::uint8_t* const up_b =
            stage.up_b + fragment_n * kPackedK64Bytes;
        const Sm87A4W4BFragment gate_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(gate_b, lane);
        const Sm87A4W4BFragment up_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(up_b, lane);
        Sm87A4W4Accumulator gate_partial{};
        Sm87A4W4Accumulator up_partial{};
        sm87_a4w4_mma_m16n8k64(gate_partial, a, gate_fragment);
        sm87_a4w4_mma_m16n8k64(up_partial, a, up_fragment);

        const std::size_t local_n0 = fragment_n + 2U * (lane % 4U);
        const std::size_t local_n1 = local_n0 + 1U;
        const float gate_scale0 =
            decode_bf16(stage.gate_b_scales[local_n0]);
        const float gate_scale1 =
            decode_bf16(stage.gate_b_scales[local_n1]);
        const float up_scale0 = decode_bf16(stage.up_b_scales[local_n0]);
        const float up_scale1 = decode_bf16(stage.up_b_scales[local_n1]);

        gate_accumulators[fragment][0U] +=
            static_cast<float>(gate_partial.x0) * a_scale0 * gate_scale0;
        gate_accumulators[fragment][1U] +=
            static_cast<float>(gate_partial.x1) * a_scale0 * gate_scale1;
        gate_accumulators[fragment][2U] +=
            static_cast<float>(gate_partial.x2) * a_scale1 * gate_scale0;
        gate_accumulators[fragment][3U] +=
            static_cast<float>(gate_partial.x3) * a_scale1 * gate_scale1;
        up_accumulators[fragment][0U] +=
            static_cast<float>(up_partial.x0) * a_scale0 * up_scale0;
        up_accumulators[fragment][1U] +=
            static_cast<float>(up_partial.x1) * a_scale0 * up_scale1;
        up_accumulators[fragment][2U] +=
            static_cast<float>(up_partial.x2) * a_scale1 * up_scale0;
        up_accumulators[fragment][3U] +=
            static_cast<float>(up_partial.x3) * a_scale1 * up_scale1;
      }
      __syncthreads();
    }

    const std::size_t local_m0 = warp_m * 16U + lane / 4U;
    const std::size_t local_m1 = local_m0 + 8U;
#pragma unroll
    for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
      const std::size_t local_n0 =
          warp_n * 32U + fragment * 8U + 2U * (lane % 4U);
      const std::size_t local_n1 = local_n0 + 1U;
      shared.product[local_m0][local_n0] = silu_product(
          gate_accumulators[fragment][0U], up_accumulators[fragment][0U]);
      shared.product[local_m0][local_n1] = silu_product(
          gate_accumulators[fragment][1U], up_accumulators[fragment][1U]);
      shared.product[local_m1][local_n0] = silu_product(
          gate_accumulators[fragment][2U], up_accumulators[fragment][2U]);
      shared.product[local_m1][local_n1] = silu_product(
          gate_accumulators[fragment][3U], up_accumulators[fragment][3U]);
    }
    __syncthreads();

    // Each warp owns four M rows and quantizes both N64 halves. A complete
    // output K64 group stays inside this CTA; no BF16 Gate or Up tensor
    // reaches global memory.
#pragma unroll
    for (unsigned int row_iteration = 0U; row_iteration < 4U;
         ++row_iteration) {
      const std::size_t local_m = warp + row_iteration * 8U;
      const std::size_t global_m = m_tile_start + local_m;
      if (global_m < token_count) {
        for (unsigned int output_group = 0U; output_group < 2U;
             ++output_group) {
          float even =
              shared.product[local_m][output_group * 64U + 2U * lane];
          float odd = shared.product[local_m]
                                    [output_group * 64U + 2U * lane + 1U];
          float maximum = fmaxf(fabsf(even), fabsf(odd));
#pragma unroll
          for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffU, maximum, delta));
          }
          maximum = __shfl_sync(0xffffffffU, maximum, 0U);
          const float clipped_maximum = maximum * output_clip_ratio;
          std::uint16_t scale_bits = encode_bf16(
              maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
          float stored_scale = decode_bf16(scale_bits);
          if (maximum != 0.0F && stored_scale == 0.0F) {
            scale_bits = 1U;
            stored_scale = decode_bf16(scale_bits);
          }
          even = fminf(fmaxf(even, -clipped_maximum), clipped_maximum);
          odd = fminf(fmaxf(odd, -clipped_maximum), clipped_maximum);
          int even_code = stored_scale == 0.0F
                              ? 0
                              : __float2int_rn(even / stored_scale);
          int odd_code = stored_scale == 0.0F
                             ? 0
                             : __float2int_rn(odd / stored_scale);
          even_code = even_code < -7 ? -7
                                     : (even_code > 7 ? 7 : even_code);
          odd_code = odd_code < -7 ? -7
                                   : (odd_code > 7 ? 7 : odd_code);
          const std::size_t global_output_group =
              n_tile_start / 64U + output_group;
          packed_output[sm87_a4w4_consumer_packed_offset(
              global_m, global_output_group, lane,
              output_k64_group_count)] =
              sm87_a4w4_pack_signed_pair(even_code, odd_code);
          if (lane == 0U) {
            output_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                global_m, global_output_group, output_k64_group_count)] =
                scale_bits;
          }
        }
      }
    }
    __syncthreads();
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpThreads,
                      kSm87A4W4GateUpCtasPerSm)
void q3x_sm87_a4w4_gateup_paired_large_m_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k64_scales_bf16,
    const std::size_t output_k64_group_count,
    const std::size_t m_tile_count,
    const std::size_t work_tile_count) {
  __shared__ Sm87A4W4GateUpLargeMSharedStorage shared;

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  // Eight warps cover 4x M16 and 2x N32.  Each warp therefore retains the
  // established pair of [4][4] branch accumulators.
  const std::size_t warp_m = warp % 4U;
  const std::size_t warp_n = warp / 4U;

  for (std::size_t work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const std::size_t n_tile = work_tile / m_tile_count;
    const std::size_t m_tile = work_tile - n_tile * m_tile_count;
    const std::size_t m_tile_start =
        m_tile * kSm87A4W4GateUpLargeMTileM;
    const std::size_t n_tile_start =
        n_tile * kSm87A4W4GateUpLargeMTileN;

    float gate_accumulators[4U][4U]{};
    float up_accumulators[4U][4U]{};

    prefetch_large_m_stage(
        shared.pipeline[0U], packed_a, a_k64_scales_bf16, packed_gate_b,
        gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
        m_tile_start, n_tile_start, 0U, k64_group_count);
    if (k64_group_count > 1U) {
      prefetch_large_m_stage(
          shared.pipeline[1U], packed_a, a_k64_scales_bf16, packed_gate_b,
          gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
          m_tile_start, n_tile_start, 1U, k64_group_count);
    }

    for (std::size_t group = 0U; group < k64_group_count; ++group) {
      if (group + 2U < k64_group_count) {
        prefetch_large_m_stage(
            shared.pipeline[(group + 2U) %
                            kSm87A4W4GateUpPipelineStages],
            packed_a, a_k64_scales_bf16, packed_gate_b,
            gate_b_k64_scales_bf16, packed_up_b,
            up_b_k64_scales_bf16, m_tile_start, n_tile_start,
            group + 2U, k64_group_count);
      }
      if (group + 1U == k64_group_count) {
        cp_async_wait<0>();
      } else {
        cp_async_wait<1>();
      }
      __syncthreads();

      const Sm87A4W4GateUpLargeMPipelineStage& stage =
          shared.pipeline[group % kSm87A4W4GateUpPipelineStages];
      const std::uint8_t* const warp_a =
          stage.a + warp_m * 16U * kPackedK64Bytes;
      const Sm87A4W4AFragment a =
          sm87_a4w4_load_a_fragment_swizzled_shared(warp_a, lane);
      const std::size_t local_m0 = warp_m * 16U + lane / 4U;
      const std::size_t local_m1 = local_m0 + 8U;
      const float a_scale0 = decode_bf16(stage.a_scales[local_m0]);
      const float a_scale1 = decode_bf16(stage.a_scales[local_m1]);

#pragma unroll
      for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
        const std::size_t fragment_n = warp_n * 32U + fragment * 8U;
        const Sm87A4W4BFragment gate_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.gate_b + fragment_n * kPackedK64Bytes, lane);
        const Sm87A4W4BFragment up_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.up_b + fragment_n * kPackedK64Bytes, lane);
        Sm87A4W4Accumulator gate_partial{};
        Sm87A4W4Accumulator up_partial{};
        sm87_a4w4_mma_m16n8k64(gate_partial, a, gate_fragment);
        sm87_a4w4_mma_m16n8k64(up_partial, a, up_fragment);

        const std::size_t local_n0 = fragment_n + 2U * (lane % 4U);
        const std::size_t local_n1 = local_n0 + 1U;
        const float gate_scale0 =
            decode_bf16(stage.gate_b_scales[local_n0]);
        const float gate_scale1 =
            decode_bf16(stage.gate_b_scales[local_n1]);
        const float up_scale0 = decode_bf16(stage.up_b_scales[local_n0]);
        const float up_scale1 = decode_bf16(stage.up_b_scales[local_n1]);

        gate_accumulators[fragment][0U] +=
            static_cast<float>(gate_partial.x0) * a_scale0 * gate_scale0;
        gate_accumulators[fragment][1U] +=
            static_cast<float>(gate_partial.x1) * a_scale0 * gate_scale1;
        gate_accumulators[fragment][2U] +=
            static_cast<float>(gate_partial.x2) * a_scale1 * gate_scale0;
        gate_accumulators[fragment][3U] +=
            static_cast<float>(gate_partial.x3) * a_scale1 * gate_scale1;
        up_accumulators[fragment][0U] +=
            static_cast<float>(up_partial.x0) * a_scale0 * up_scale0;
        up_accumulators[fragment][1U] +=
            static_cast<float>(up_partial.x1) * a_scale0 * up_scale1;
        up_accumulators[fragment][2U] +=
            static_cast<float>(up_partial.x2) * a_scale1 * up_scale0;
        up_accumulators[fragment][3U] +=
            static_cast<float>(up_partial.x3) * a_scale1 * up_scale1;
      }
      __syncthreads();
    }

    const std::size_t local_m0 = warp_m * 16U + lane / 4U;
    const std::size_t local_m1 = local_m0 + 8U;
#pragma unroll
    for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
      const std::size_t local_n0 =
          warp_n * 32U + fragment * 8U + 2U * (lane % 4U);
      const std::size_t local_n1 = local_n0 + 1U;
      shared.product[local_m0][local_n0] = silu_product(
          gate_accumulators[fragment][0U], up_accumulators[fragment][0U]);
      shared.product[local_m0][local_n1] = silu_product(
          gate_accumulators[fragment][1U], up_accumulators[fragment][1U]);
      shared.product[local_m1][local_n0] = silu_product(
          gate_accumulators[fragment][2U], up_accumulators[fragment][2U]);
      shared.product[local_m1][local_n1] = silu_product(
          gate_accumulators[fragment][3U], up_accumulators[fragment][3U]);
    }
    __syncthreads();

    // One warp owns eight M rows; one lane pair owns two adjacent values of
    // the CTA's single N64 output group.
#pragma unroll
    for (unsigned int row_iteration = 0U; row_iteration < 8U;
         ++row_iteration) {
      const std::size_t local_m = warp + row_iteration * 8U;
      const std::size_t global_m = m_tile_start + local_m;
      float even = shared.product[local_m][2U * lane];
      float odd = shared.product[local_m][2U * lane + 1U];
      float maximum = fmaxf(fabsf(even), fabsf(odd));
#pragma unroll
      for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
        maximum = fmaxf(
            maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
      }
      maximum = __shfl_sync(0xffffffffU, maximum, 0U);
      const float clipped_maximum = maximum * output_clip_ratio;
      std::uint16_t scale_bits = encode_bf16(
          maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      even = fminf(fmaxf(even, -clipped_maximum), clipped_maximum);
      odd = fminf(fmaxf(odd, -clipped_maximum), clipped_maximum);
      int even_code = stored_scale == 0.0F
                          ? 0
                          : __float2int_rn(even / stored_scale);
      int odd_code = stored_scale == 0.0F
                         ? 0
                         : __float2int_rn(odd / stored_scale);
      even_code = even_code < -7 ? -7 : (even_code > 7 ? 7 : even_code);
      odd_code = odd_code < -7 ? -7 : (odd_code > 7 ? 7 : odd_code);
      const std::size_t global_output_group = n_tile_start / 64U;
      packed_output[sm87_a4w4_consumer_packed_offset(
          global_m, global_output_group, lane,
          output_k64_group_count)] =
          sm87_a4w4_pack_signed_pair(even_code, odd_code);
      if (lane == 0U) {
        output_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_m, global_output_group, output_k64_group_count)] =
            scale_bits;
      }
    }
    __syncthreads();
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpThreads,
                      kSm87A4W4GateUpCtasPerSm)
void q3x_sm87_a4w4_gateup_paired_wide_large_m_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k64_scales_bf16,
    const std::size_t output_k64_group_count,
    const std::size_t m_tile_count,
    const std::size_t work_tile_count) {
  __shared__ Sm87A4W4GateUpWideLargeMSharedStorage shared;

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  // Eight warps cover 2x M32 and 4x N32.  A warp keeps both of its M16
  // panels resident, producing 32 Gate and 32 Up FP32 accumulators/thread.
  const std::size_t warp_m = warp % 2U;
  const std::size_t warp_n = warp / 2U;
  const std::size_t logical_stage_count =
      (k64_group_count + kWideLargeMK64PerLogicalStage - 1U) /
      kWideLargeMK64PerLogicalStage;

  for (std::size_t work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const Sm87A4W4GateUpPairedWorkTile coordinates =
        sm87_a4w4_gateup_paired_n_major_work_tile(
            work_tile, m_tile_count, work_tile_count);
    const std::size_t m_tile_start =
        coordinates.m_tile * kSm87A4W4GateUpWideLargeMTileM;
    const std::size_t n_tile_start =
        coordinates.n_tile * kSm87A4W4GateUpWideLargeMTileN;

    float gate_accumulators[2U][4U][4U]{};
    float up_accumulators[2U][4U][4U]{};

    prefetch_wide_large_m_logical_stage(
        shared.pipeline.slot[0U], packed_a, a_k64_scales_bf16,
        packed_gate_b, gate_b_k64_scales_bf16, packed_up_b,
        up_b_k64_scales_bf16, m_tile_start, n_tile_start, 0U,
        k64_group_count);
    if (logical_stage_count > 1U) {
      prefetch_wide_large_m_logical_stage(
          shared.pipeline.slot[1U], packed_a, a_k64_scales_bf16,
          packed_gate_b, gate_b_k64_scales_bf16, packed_up_b,
          up_b_k64_scales_bf16, m_tile_start, n_tile_start, 1U,
          k64_group_count);
    }

    for (std::size_t logical_stage = 0U;
         logical_stage < logical_stage_count; ++logical_stage) {
      // Two async groups belong to each logical K128 slot.  Keeping at most
      // the next slot's two groups pending publishes the current slot while
      // retaining copy/compute overlap.
      if (logical_stage + 1U < logical_stage_count) {
        cp_async_wait<2>();
      } else {
        cp_async_wait<0>();
      }
      __syncthreads();

      const Sm87A4W4GateUpWideLargeMLogicalStage& current =
          shared.pipeline.slot[
              logical_stage % kSm87A4W4GateUpWideLargeMPipelineSlots];
      accumulate_wide_large_m_k64_stage(
          current.k64[0U], warp_m, warp_n, lane, gate_accumulators,
          up_accumulators);
      if (logical_stage * kWideLargeMK64PerLogicalStage + 1U <
          k64_group_count) {
        accumulate_wide_large_m_k64_stage(
            current.k64[1U], warp_m, warp_n, lane, gate_accumulators,
            up_accumulators);
      }
      __syncthreads();

      if (logical_stage + kSm87A4W4GateUpWideLargeMPipelineSlots <
          logical_stage_count) {
        const std::size_t future =
            logical_stage + kSm87A4W4GateUpWideLargeMPipelineSlots;
        prefetch_wide_large_m_logical_stage(
            shared.pipeline.slot[
                logical_stage % kSm87A4W4GateUpWideLargeMPipelineSlots],
            packed_a, a_k64_scales_bf16, packed_gate_b,
            gate_b_k64_scales_bf16, packed_up_b,
            up_b_k64_scales_bf16, m_tile_start, n_tile_start, future,
            k64_group_count);
      }
    }

    // All async groups have completed and the final CTA barrier above closes
    // the pipeline lifetime.  The product union member is now exclusively
    // active for SiLU and output quantization.
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
      const std::size_t panel_m = warp_m * 32U + m_panel * 16U;
      const std::size_t local_m0 = panel_m + lane / 4U;
      const std::size_t local_m1 = local_m0 + 8U;
#pragma unroll
      for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
        const std::size_t local_n0 =
            warp_n * 32U + fragment * 8U + 2U * (lane % 4U);
        const std::size_t local_n1 = local_n0 + 1U;
        shared.product[local_m0][local_n0] = silu_product(
            gate_accumulators[m_panel][fragment][0U],
            up_accumulators[m_panel][fragment][0U]);
        shared.product[local_m0][local_n1] = silu_product(
            gate_accumulators[m_panel][fragment][1U],
            up_accumulators[m_panel][fragment][1U]);
        shared.product[local_m1][local_n0] = silu_product(
            gate_accumulators[m_panel][fragment][2U],
            up_accumulators[m_panel][fragment][2U]);
        shared.product[local_m1][local_n1] = silu_product(
            gate_accumulators[m_panel][fragment][3U],
            up_accumulators[m_panel][fragment][3U]);
      }
    }
    __syncthreads();

    // One warp owns eight M rows and emits both complete N64 output groups.
#pragma unroll
    for (unsigned int row_iteration = 0U; row_iteration < 8U;
         ++row_iteration) {
      const std::size_t local_m = warp + row_iteration * 8U;
      const std::size_t global_m = m_tile_start + local_m;
#pragma unroll
      for (unsigned int output_group = 0U; output_group < 2U;
           ++output_group) {
        float even =
            shared.product[local_m][output_group * 64U + 2U * lane];
        float odd = shared.product[local_m]
                                  [output_group * 64U + 2U * lane + 1U];
        float maximum = fmaxf(fabsf(even), fabsf(odd));
#pragma unroll
        for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
          maximum = fmaxf(
              maximum,
              __shfl_down_sync(0xffffffffU, maximum, delta));
        }
        maximum = __shfl_sync(0xffffffffU, maximum, 0U);
        const float clipped_maximum = maximum * output_clip_ratio;
        std::uint16_t scale_bits = encode_bf16(
            maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
        float stored_scale = decode_bf16(scale_bits);
        if (maximum != 0.0F && stored_scale == 0.0F) {
          scale_bits = 1U;
          stored_scale = decode_bf16(scale_bits);
        }
        even = fminf(fmaxf(even, -clipped_maximum), clipped_maximum);
        odd = fminf(fmaxf(odd, -clipped_maximum), clipped_maximum);
        int even_code = stored_scale == 0.0F
                            ? 0
                            : __float2int_rn(even / stored_scale);
        int odd_code = stored_scale == 0.0F
                           ? 0
                           : __float2int_rn(odd / stored_scale);
        even_code = even_code < -7 ? -7
                                   : (even_code > 7 ? 7 : even_code);
        odd_code = odd_code < -7 ? -7
                                 : (odd_code > 7 ? 7 : odd_code);
        const std::size_t global_output_group =
            n_tile_start / 64U + output_group;
        packed_output[sm87_a4w4_consumer_packed_offset(
            global_m, global_output_group, lane,
            output_k64_group_count)] =
            sm87_a4w4_pack_signed_pair(even_code, odd_code);
        if (lane == 0U) {
          output_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
              global_m, global_output_group,
              output_k64_group_count)] = scale_bits;
        }
      }
    }
    __syncthreads();
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpThreads,
                      kSm87A4W4GateUpCtasPerSm)
void q3x_sm87_a4w4_gateup_paired_k128_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const unsigned int k128_group_count,
    const unsigned int physical_k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k128_scales_bf16,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count) {
  __shared__ Sm87A4W4GateUpK128SharedStorage shared;

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m = warp % 2U;
  const unsigned int warp_n = warp / 2U;
  const unsigned int output_k128_group_count =
      work_tile_count / m_tile_count;
  const unsigned int output_physical_k64_group_count =
      output_k128_group_count *
      kSm87A4W4PhysicalK64BlocksPerSharedScale;

  for (unsigned int work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const unsigned int n_tile = work_tile / m_tile_count;
    const unsigned int m_tile = work_tile - n_tile * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4GateUpK128TileM;
    const unsigned int n_tile_start =
        n_tile * kSm87A4W4GateUpK128TileN;

    float gate_accumulators[2U][4U][4U]{};
    float up_accumulators[2U][4U][4U]{};

    prefetch_gateup_k128_stage(
        shared.pipeline.slot[0U], packed_a, a_k128_scales_bf16,
        packed_gate_b, gate_b_k128_scales_bf16, packed_up_b,
        up_b_k128_scales_bf16, m_tile_start, n_tile_start, 0U,
        physical_k64_group_count, k128_group_count);
    cp_async_wait<0>();
    __syncthreads();

    for (unsigned int group = 0U; group < k128_group_count; ++group) {
      const bool has_next = group + 1U < k128_group_count;
      if (has_next) {
        prefetch_gateup_k128_stage(
            shared.pipeline.slot[
                (group + 1U) % kSm87A4W4GateUpK128PipelineSlots],
            packed_a, a_k128_scales_bf16, packed_gate_b,
            gate_b_k128_scales_bf16, packed_up_b,
            up_b_k128_scales_bf16, m_tile_start, n_tile_start,
            group + 1U, physical_k64_group_count, k128_group_count);
      }

      accumulate_gateup_k128_stage(
          shared.pipeline.slot[
              group % kSm87A4W4GateUpK128PipelineSlots],
          warp_m, warp_n, lane, gate_accumulators, up_accumulators);
      if (has_next) {
        cp_async_wait<0>();
      }
      // Publish the alternate slot and prevent a fast warp from recycling
      // the current slot until every warp has consumed both K64 code planes.
      __syncthreads();
    }

    // The final barrier closes the pipeline lifetime; activate the product
    // union member for SiLU and shared-K128 output quantization.
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 2U; ++m_panel) {
      const unsigned int panel_m =
          warp_m * 32U + m_panel * 16U;
      const unsigned int local_m0 = panel_m + lane / 4U;
      const unsigned int local_m1 = local_m0 + 8U;
#pragma unroll
      for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
        const unsigned int local_n0 =
            warp_n * 32U + fragment * 8U + 2U * (lane % 4U);
        const unsigned int local_n1 = local_n0 + 1U;
        shared.product[local_m0][local_n0] = silu_product(
            gate_accumulators[m_panel][fragment][0U],
            up_accumulators[m_panel][fragment][0U]);
        shared.product[local_m0][local_n1] = silu_product(
            gate_accumulators[m_panel][fragment][1U],
            up_accumulators[m_panel][fragment][1U]);
        shared.product[local_m1][local_n0] = silu_product(
            gate_accumulators[m_panel][fragment][2U],
            up_accumulators[m_panel][fragment][2U]);
        shared.product[local_m1][local_n1] = silu_product(
            gate_accumulators[m_panel][fragment][3U],
            up_accumulators[m_panel][fragment][3U]);
      }
    }
    __syncthreads();

    // One warp owns eight rows.  Every lane contributes its adjacent pair
    // from both physical N64 blocks to one 128-value maximum, then writes one
    // packed byte in each block under the single persisted K128 scale.
#pragma unroll
    for (unsigned int row_iteration = 0U; row_iteration < 8U;
         ++row_iteration) {
      const unsigned int local_m = warp + row_iteration * 8U;
      const unsigned int global_m = m_tile_start + local_m;
      float value0 = shared.product[local_m][2U * lane];
      float value1 = shared.product[local_m][2U * lane + 1U];
      float value2 = shared.product[local_m][64U + 2U * lane];
      float value3 = shared.product[local_m][64U + 2U * lane + 1U];
      float maximum = fmaxf(
          fmaxf(fabsf(value0), fabsf(value1)),
          fmaxf(fabsf(value2), fabsf(value3)));
#pragma unroll
      for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
        maximum = fmaxf(
            maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
      }
      maximum = __shfl_sync(0xffffffffU, maximum, 0U);
      const float clipped_maximum = maximum * output_clip_ratio;
      std::uint16_t scale_bits = encode_bf16(
          maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      value0 = fminf(fmaxf(value0, -clipped_maximum), clipped_maximum);
      value1 = fminf(fmaxf(value1, -clipped_maximum), clipped_maximum);
      value2 = fminf(fmaxf(value2, -clipped_maximum), clipped_maximum);
      value3 = fminf(fmaxf(value3, -clipped_maximum), clipped_maximum);
      int code0 = stored_scale == 0.0F
                      ? 0
                      : __float2int_rn(value0 / stored_scale);
      int code1 = stored_scale == 0.0F
                      ? 0
                      : __float2int_rn(value1 / stored_scale);
      int code2 = stored_scale == 0.0F
                      ? 0
                      : __float2int_rn(value2 / stored_scale);
      int code3 = stored_scale == 0.0F
                      ? 0
                      : __float2int_rn(value3 / stored_scale);
      code0 = code0 < -7 ? -7 : (code0 > 7 ? 7 : code0);
      code1 = code1 < -7 ? -7 : (code1 > 7 ? 7 : code1);
      code2 = code2 < -7 ? -7 : (code2 > 7 ? 7 : code2);
      code3 = code3 < -7 ? -7 : (code3 > 7 ? 7 : code3);
      const unsigned int first_physical_group =
          n_tile_start / kSm87A4W4ConsumerKBlock;
      packed_output[sm87_a4w4_consumer_packed_offset(
          global_m, first_physical_group, lane,
          output_physical_k64_group_count)] =
          sm87_a4w4_pack_signed_pair(code0, code1);
      packed_output[sm87_a4w4_consumer_packed_offset(
          global_m, first_physical_group + 1U, lane,
          output_physical_k64_group_count)] =
          sm87_a4w4_pack_signed_pair(code2, code3);
      if (lane == 0U) {
        output_k128_scales_bf16[
            sm87_a4w4_consumer_k128_scale_offset(
                global_m,
                n_tile_start / kSm87A4W4GateUpK128TileK,
                output_k128_group_count)] = scale_bits;
      }
    }
    __syncthreads();
  }
}

[[nodiscard]] int validate_sm87(
    cudaDeviceProp* const properties = nullptr) noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp local{};
  status = cudaGetDeviceProperties(&local, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (local.major != kSm87A4W4RequiredComputeMajor ||
      local.minor != kSm87A4W4RequiredComputeMinor ||
      local.multiProcessorCount != kRequiredSmCount) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int query_sm87_a4w4_gateup_paired_resources_cuda(
    Sm87A4W4GateUpPairedResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpPairedResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_gateup_paired_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_gateup_paired_kernel,
      static_cast<int>(kSm87A4W4GateUpThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = 0U;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4GateUpCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4GateUpThreads) ||
      resources->static_shared_bytes !=
          sizeof(Sm87A4W4GateUpSharedStorage)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int query_sm87_a4w4_gateup_paired_large_m_resources_cuda(
    Sm87A4W4GateUpPairedResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpPairedResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_gateup_paired_large_m_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_gateup_paired_large_m_kernel,
      static_cast<int>(kSm87A4W4GateUpThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = 0U;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4GateUpCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4GateUpThreads) ||
      resources->static_shared_bytes !=
          sizeof(Sm87A4W4GateUpLargeMSharedStorage)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int query_sm87_a4w4_gateup_paired_wide_large_m_resources_cuda(
    Sm87A4W4GateUpPairedResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpPairedResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_gateup_paired_wide_large_m_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_gateup_paired_wide_large_m_kernel,
      static_cast<int>(kSm87A4W4GateUpThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = 0U;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4GateUpCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4GateUpThreads) ||
      resources->static_shared_bytes !=
          sizeof(Sm87A4W4GateUpWideLargeMSharedStorage)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int query_sm87_a4w4_gateup_paired_k128_resources_cuda(
    Sm87A4W4GateUpPairedResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpPairedResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_gateup_paired_k128_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_gateup_paired_k128_kernel,
      static_cast<int>(kSm87A4W4GateUpThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = 0U;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread >
          static_cast<int>(kSm87A4W4GateUpK128MaximumRegisters) ||
      resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4GateUpCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4GateUpThreads) ||
      resources->static_shared_bytes !=
          sizeof(Sm87A4W4GateUpK128SharedStorage) ||
      resources->static_shared_bytes > 48U * 1'024U) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_paired_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k64_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpPairedPlan plan =
      sm87_a4w4_gateup_paired_plan(token_count, intermediate_size,
                                  input_size);
  if (plan.launch_ctas == 0U ||
      !(output_clip_ratio > 0.0F && output_clip_ratio <= 1.0F) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_output, 16U) ||
      !aligned(output_k64_scales_bf16, alignof(std::uint16_t)) ||
      !consumer_capacity_fits(token_count, plan.k64_groups) ||
      !consumer_capacity_fits(intermediate_size, plan.k64_groups) ||
      !consumer_capacity_fits(
          token_count, intermediate_size / kSm87A4W4ConsumerKBlock)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_a_scale_elements =
      sm87_a4w4_consumer_scale_capacity_elements(token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(intermediate_size, input_size);
  const std::size_t required_b_scale_elements =
      sm87_a4w4_consumer_scale_capacity_elements(intermediate_size,
                                                 input_size);
  const std::size_t required_output_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count,
                                               intermediate_size);
  const std::size_t required_output_scale_elements =
      sm87_a4w4_consumer_scale_capacity_elements(token_count,
                                                 intermediate_size);
  if (packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scale_elements ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      gate_b_scale_capacity_elements < required_b_scale_elements ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      up_b_scale_capacity_elements < required_b_scale_elements ||
      packed_output_capacity_bytes < required_output_bytes ||
      output_scale_capacity_elements < required_output_scale_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int device_status = validate_sm87();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  Sm87A4W4GateUpPairedResources resources{};
  int resource_status = static_cast<int>(cudaErrorInvalidValue);
  switch (plan.kernel) {
    case Sm87A4W4GateUpPairedKernel::kM32N128K64:
      resource_status =
          query_sm87_a4w4_gateup_paired_resources_cuda(&resources);
      break;
    case Sm87A4W4GateUpPairedKernel::kM64N64K64:
      resource_status =
          query_sm87_a4w4_gateup_paired_large_m_resources_cuda(
              &resources);
      break;
    case Sm87A4W4GateUpPairedKernel::kM64N128K64:
      resource_status =
          query_sm87_a4w4_gateup_paired_wide_large_m_resources_cuda(
              &resources);
      break;
  }
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (plan.kernel == Sm87A4W4GateUpPairedKernel::kM64N128K64) {
    q3x_sm87_a4w4_gateup_paired_wide_large_m_kernel<<<
        static_cast<unsigned int>(kSm87A4W4GateUpPersistentCtas),
        static_cast<unsigned int>(kSm87A4W4GateUpThreads), 0U, stream>>>(
        packed_a, a_k64_scales_bf16, packed_gate_b,
        gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
        plan.k64_groups, output_clip_ratio, packed_output,
        output_k64_scales_bf16,
        intermediate_size / kSm87A4W4ConsumerKBlock, plan.m_tiles,
        plan.work_tiles);
    return static_cast<int>(cudaPeekAtLastError());
  }
  if (plan.kernel == Sm87A4W4GateUpPairedKernel::kM64N64K64) {
    q3x_sm87_a4w4_gateup_paired_large_m_kernel<<<
        static_cast<unsigned int>(kSm87A4W4GateUpPersistentCtas),
        static_cast<unsigned int>(kSm87A4W4GateUpThreads), 0U, stream>>>(
        packed_a, a_k64_scales_bf16, packed_gate_b,
        gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
        plan.k64_groups, output_clip_ratio, packed_output,
        output_k64_scales_bf16,
        intermediate_size / kSm87A4W4ConsumerKBlock, plan.m_tiles,
        plan.work_tiles);
    return static_cast<int>(cudaPeekAtLastError());
  }
  q3x_sm87_a4w4_gateup_paired_kernel<<<
      static_cast<unsigned int>(kSm87A4W4GateUpPersistentCtas),
      static_cast<unsigned int>(kSm87A4W4GateUpThreads), 0U, stream>>>(
      packed_a, a_k64_scales_bf16, packed_gate_b,
      gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
      token_count, plan.k64_groups, output_clip_ratio, packed_output,
      output_k64_scales_bf16,
      intermediate_size / kSm87A4W4ConsumerKBlock,
      plan.m_tiles, plan.work_tiles);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_a4w4_gateup_paired_k128_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k128_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpPairedK128Plan plan =
      sm87_a4w4_gateup_paired_k128_plan(
          token_count, intermediate_size, input_size);
  if (plan.launch_ctas == 0U ||
      !(output_clip_ratio > 0.0F && output_clip_ratio <= 1.0F) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k128_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k128_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k128_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_output, 16U) ||
      !aligned(output_k128_scales_bf16, alignof(std::uint16_t)) ||
      !consumer_capacity_fits(token_count, plan.physical_k64_groups) ||
      !consumer_capacity_fits(intermediate_size,
                              plan.physical_k64_groups) ||
      !consumer_capacity_fits(
          token_count,
          intermediate_size / kSm87A4W4ConsumerKBlock) ||
      plan.k128_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      intermediate_size / kSm87A4W4GateUpK128TileK >
          std::numeric_limits<unsigned int>::max() ||
      intermediate_size / kSm87A4W4ConsumerKBlock >
          std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_a_scale_elements =
      sm87_a4w4_consumer_k128_scale_capacity_elements(
          token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(intermediate_size,
                                               input_size);
  const std::size_t required_b_scale_elements =
      sm87_a4w4_consumer_k128_scale_capacity_elements(
          intermediate_size, input_size);
  const std::size_t required_output_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count,
                                               intermediate_size);
  const std::size_t required_output_scale_elements =
      sm87_a4w4_consumer_k128_scale_capacity_elements(
          token_count, intermediate_size);
  if (required_a_bytes == 0U || required_a_scale_elements == 0U ||
      required_b_bytes == 0U || required_b_scale_elements == 0U ||
      required_output_bytes == 0U ||
      required_output_scale_elements == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scale_elements ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      gate_b_scale_capacity_elements < required_b_scale_elements ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      up_b_scale_capacity_elements < required_b_scale_elements ||
      packed_output_capacity_bytes < required_output_bytes ||
      output_scale_capacity_elements < required_output_scale_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int device_status = validate_sm87();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  Sm87A4W4GateUpPairedResources resources{};
  const int resource_status =
      query_sm87_a4w4_gateup_paired_k128_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_gateup_paired_k128_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(kSm87A4W4GateUpThreads), 0U, stream>>>(
      packed_a, a_k128_scales_bf16, packed_gate_b,
      gate_b_k128_scales_bf16, packed_up_b,
      up_b_k128_scales_bf16,
      static_cast<unsigned int>(plan.k128_groups),
      static_cast<unsigned int>(plan.physical_k64_groups),
      output_clip_ratio, packed_output, output_k128_scales_bf16,
      static_cast<unsigned int>(plan.m_tiles),
      static_cast<unsigned int>(plan.work_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
