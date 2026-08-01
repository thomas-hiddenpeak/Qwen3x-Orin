#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_DOWN_K512_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_DOWN_K512_HOST_DEVICE
#endif

namespace q3x::kernels {

// Isolated Down-projection throughput macrocell.  This API describes a new
// K512 quantization contract and is intentionally incompatible with the
// authenticated K128 production ABI.  It is not selected by any runtime.
//
// The production entry point is pinned to Qwen3.6-27B Down and complete M128
// extents through the external P4K API capacity:
//
//   M in [128,4096] and divisible by 128, N=5120, K=17408.
//
// Sixteen persistent CTAs (one per SM on the pinned 16-SM SM87 target) each
// own one M128 tile at M2048 or two M128 tiles at M4096 and synchronously walk
// all forty N128 tiles. The N-major wave makes all CTAs request the same B
// tile at approximately the same time, so the 16 copies can be served from L2
// rather than traversing unrelated N tiles. One cell is M128N128; its eight
// warps form a 4x2 grid of M32N64 warp tiles.
//
// A physical stage contains four packed K64 code planes (K256) for both A
// and B.  Four 32-KiB stages form a 128-KiB opt-in dynamic-shared ring.  Two
// consecutive stages form one logical K512 group.  For every M16N8 fragment,
// two S32 partial tuples (the two resident M16 rows) accumulate all eight K64
// MMAs before exactly one K512 scale product is applied to resident FP32
// output accumulators.
inline constexpr std::size_t kSm87A4W4DownK512MinimumTokenCount = 128U;
inline constexpr std::size_t kSm87A4W4DownK512MaximumTokenCount = 4'096U;
inline constexpr std::size_t kSm87A4W4DownK512OutputSize = 5'120U;
inline constexpr std::size_t kSm87A4W4DownK512InputSize = 17'408U;
inline constexpr std::size_t kSm87A4W4DownK512TileM = 128U;
inline constexpr std::size_t kSm87A4W4DownK512TileN = 128U;
inline constexpr std::size_t kSm87A4W4DownK512CopyK = 256U;
inline constexpr std::size_t kSm87A4W4DownK512ScaleK = 512U;
inline constexpr std::size_t kSm87A4W4DownK512PhysicalK64 = 64U;
inline constexpr std::size_t kSm87A4W4DownK512K64PerStage = 4U;
inline constexpr std::size_t kSm87A4W4DownK512StagesPerScale = 2U;
inline constexpr std::size_t kSm87A4W4DownK512Stages = 4U;
inline constexpr std::size_t kSm87A4W4DownK512Threads = 256U;
inline constexpr std::size_t kSm87A4W4DownK512Warps = 8U;
inline constexpr std::size_t kSm87A4W4DownK512WarpTileM = 32U;
inline constexpr std::size_t kSm87A4W4DownK512WarpTileN = 64U;
inline constexpr std::size_t kSm87A4W4DownK512WarpRows = 4U;
inline constexpr std::size_t kSm87A4W4DownK512WarpColumns = 2U;
inline constexpr std::size_t kSm87A4W4DownK512PackedRowK64Bytes = 32U;
inline constexpr std::size_t kSm87A4W4DownK512OuterBlock = 64U;
inline constexpr std::size_t kSm87A4W4DownK512StageABytes =
    kSm87A4W4DownK512TileM * kSm87A4W4DownK512CopyK / 2U;
inline constexpr std::size_t kSm87A4W4DownK512StageBBytes =
    kSm87A4W4DownK512TileN * kSm87A4W4DownK512CopyK / 2U;
inline constexpr std::size_t kSm87A4W4DownK512StageBytes =
    kSm87A4W4DownK512StageABytes + kSm87A4W4DownK512StageBBytes;
inline constexpr std::size_t kSm87A4W4DownK512DynamicSharedBytes =
    kSm87A4W4DownK512Stages * kSm87A4W4DownK512StageBytes;
inline constexpr std::size_t kSm87A4W4DownK512PersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4DownK512CtasPerSm = 1U;
// One-CTA residency deliberately spends the full architectural register
// allowance on the 64 resident FP32 outputs/thread.  Admission is governed by
// zero local memory and measured one-block occupancy, not by a two-CTA-era
// register cap.
inline constexpr std::size_t kSm87A4W4DownK512MaximumRegisters = 255U;

[[nodiscard]] constexpr bool sm87_a4w4_down_k512_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_k512_outer_block_count(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U
             ? 0U
             : 1U + (outer_count - 1U) /
                        kSm87A4W4DownK512OuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_k512_k64_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4DownK512PhysicalK64 == 0U
             ? logical_k / kSm87A4W4DownK512PhysicalK64
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_k512_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4DownK512ScaleK == 0U
             ? logical_k / kSm87A4W4DownK512ScaleK
             : 0U;
}

// Packed S4 layout: [ceil(outer/64)][K/64][64][32].
[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_k512_packed_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_down_k512_outer_block_count(outer_count);
  const std::size_t groups =
      sm87_a4w4_down_k512_k64_group_count(logical_k);
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_down_k512_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  constexpr std::size_t group_bytes =
      kSm87A4W4DownK512OuterBlock *
      kSm87A4W4DownK512PackedRowK64Bytes;
  return sm87_a4w4_down_k512_product_fits(block_groups, group_bytes)
             ? block_groups * group_bytes
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_DOWN_K512_HOST_DEVICE constexpr std::size_t
sm87_a4w4_down_k512_packed_offset(
    const std::size_t outer_coordinate, const std::size_t k64_group,
    const std::size_t byte_in_k64,
    const std::size_t k64_group_count) noexcept {
  return (((outer_coordinate / kSm87A4W4DownK512OuterBlock) *
               k64_group_count +
           k64_group) *
              kSm87A4W4DownK512OuterBlock +
          outer_coordinate % kSm87A4W4DownK512OuterBlock) *
             kSm87A4W4DownK512PackedRowK64Bytes +
         byte_in_k64;
}

// Independent BF16 K512 scale layout: [ceil(outer/64)][K/512][64].
[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_k512_scale_capacity_elements(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_down_k512_outer_block_count(outer_count);
  const std::size_t groups =
      sm87_a4w4_down_k512_group_count(logical_k);
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_down_k512_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_down_k512_product_fits(
             block_groups, kSm87A4W4DownK512OuterBlock)
             ? block_groups * kSm87A4W4DownK512OuterBlock
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_DOWN_K512_HOST_DEVICE constexpr std::size_t
sm87_a4w4_down_k512_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate / kSm87A4W4DownK512OuterBlock) *
              k512_group_count +
          k512_group) *
             kSm87A4W4DownK512OuterBlock +
         outer_coordinate % kSm87A4W4DownK512OuterBlock;
}

struct Sm87A4W4DownK512Plan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k256_stages{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

// Generic complete-cell plan used solely by correctness admission.  It has
// no tail handling and retains the same N-major persistent mapping.
[[nodiscard]] constexpr Sm87A4W4DownK512Plan
sm87_a4w4_down_k512_test_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4DownK512TileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4DownK512TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4DownK512ScaleK != 0U) {
    return {};
  }
  const std::size_t m_tiles = token_count / kSm87A4W4DownK512TileM;
  const std::size_t n_tiles = output_size / kSm87A4W4DownK512TileN;
  if (!sm87_a4w4_down_k512_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4DownK512ScaleK,
          input_size / kSm87A4W4DownK512CopyK,
          input_size / kSm87A4W4DownK512PhysicalK64,
          work_tiles,
          work_tiles < kSm87A4W4DownK512PersistentCtas
              ? work_tiles
              : kSm87A4W4DownK512PersistentCtas};
}

// Production admits complete M128 extents through the P4K API capacity. This
// covers ceil128(logical prompt M) without forcing the rest of the K128
// projection path to use the same padding. N/K and the 16-CTA B-wave remain
// fixed.
[[nodiscard]] constexpr Sm87A4W4DownK512Plan
sm87_a4w4_down_k512_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  return token_count >= kSm87A4W4DownK512MinimumTokenCount &&
                 token_count <= kSm87A4W4DownK512MaximumTokenCount &&
                 token_count % kSm87A4W4DownK512TileM == 0U &&
                 output_size == kSm87A4W4DownK512OutputSize &&
                 input_size == kSm87A4W4DownK512InputSize
             ? sm87_a4w4_down_k512_test_plan(token_count, output_size,
                                               input_size)
             : Sm87A4W4DownK512Plan{};
}

struct Sm87A4W4DownK512Resources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t configured_dynamic_shared_limit_bytes{};
  std::size_t device_optin_shared_limit_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// Sets cudaFuncAttributeMaxDynamicSharedMemorySize before inspecting the
// kernel.  Success guarantees the 128-KiB request is admitted, localSizeBytes
// is zero, one block/SM is resident, and the device is the pinned 16-SM SM87.
[[nodiscard]] int query_sm87_a4w4_down_k512_macrocell_resources_cuda(
    Sm87A4W4DownK512Resources* resources) noexcept;

[[nodiscard]] int launch_sm87_a4w4_down_k512_macrocell_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Correctness-only generic complete-cell launcher.  It exists so synthetic
// exhaustive cases need not allocate or evaluate the full real Down shape.
// No performance decision may be made from this entry point.
[[nodiscard]] int launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4DownK512StageABytes == 16U * 1'024U);
static_assert(kSm87A4W4DownK512StageBBytes == 16U * 1'024U);
static_assert(kSm87A4W4DownK512StageBytes == 32U * 1'024U);
static_assert(kSm87A4W4DownK512DynamicSharedBytes == 128U * 1'024U);
static_assert(kSm87A4W4DownK512WarpRows *
                      kSm87A4W4DownK512WarpColumns ==
                  kSm87A4W4DownK512Warps);
static_assert(kSm87A4W4DownK512WarpTileM *
                      kSm87A4W4DownK512WarpRows ==
                  kSm87A4W4DownK512TileM);
static_assert(kSm87A4W4DownK512WarpTileN *
                      kSm87A4W4DownK512WarpColumns ==
                  kSm87A4W4DownK512TileN);
static_assert(sm87_a4w4_down_k512_plan(2'048U, 5'120U, 17'408U)
                  .k512_groups == 34U);
static_assert(sm87_a4w4_down_k512_plan(2'048U, 5'120U, 17'408U)
                  .launch_ctas == 16U);
static_assert(sm87_a4w4_down_k512_plan(4'096U, 5'120U, 17'408U)
                  .m_tiles == 32U);
static_assert(sm87_a4w4_down_k512_plan(4'096U, 5'120U, 17'408U)
                  .work_tiles == 1'280U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_DOWN_K512_HOST_DEVICE
