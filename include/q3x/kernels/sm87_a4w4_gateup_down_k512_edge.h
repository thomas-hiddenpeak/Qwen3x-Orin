#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_DOWN_EDGE_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_DOWN_EDGE_HOST_DEVICE
#endif

namespace q3x::kernels {

// Default-off Gate+Up -> Down-input edge for the Qwen3.6-27B K512 path.
// No runtime selects this kernel merely by linking it.  It is an independent
// candidate whose first authority is bit identity with the current
// Gate+Up-macrocell + split-K512-quantizer seam.
//
// A 512-thread CTA owns an M64 tile.  Its sixteen warps form a 4x4 grid of
// M16N32 warp tiles; every warp computes matching Gate and Up fragments and
// applies SiLU(Gate)*Up itself.  Four consecutive N128 compute cells form one
// logical K512 Down-input edge cell.  Products cross the existing numerical
// seam explicitly:
//
//   product_bits = bf16_rne(SiLU(gate_fp32) * up_fp32)
//   product_fp32 = decode_bf16(product_bits)
//   one signed-A4 scale/code group covers all 512 product_fp32 values
//
// The 64x512 BF16 edge plane co-resides with the authenticated K256
// double-buffered Gate+Up pipeline.  Complete edge cells are never split
// across CTAs, so no inter-CTA reduction or global intermediate is required.
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeTileM = 64U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeComputeTileN = 128U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeScaleK = 512U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeCellsPerScale = 4U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeCopyK = 256U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgePhysicalK64 = 64U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeK64PerCopy = 4U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeCopiesPerScale = 2U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeThreads = 512U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeWarps = 16U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeWarpTileM = 16U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeWarpTileN = 32U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeWarpRows = 4U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeWarpColumns = 4U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeStages = 2U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeScaleSlots = 2U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgePackedK64Bytes = 32U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeOuterBlock = 64U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeLaunchAlignment = 128U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgePersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeCtasPerSm = 1U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeMaximumRegisters = 128U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeModelIntermediate =
    17'408U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeModelInput = 5'120U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeMaximumLaunchTokens =
    4'096U;

inline constexpr std::size_t kSm87A4W4GateUpDownEdgeAStageBytes =
    kSm87A4W4GateUpDownEdgeTileM *
    kSm87A4W4GateUpDownEdgeCopyK / 2U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeBStageBytes =
    kSm87A4W4GateUpDownEdgeComputeTileN *
    kSm87A4W4GateUpDownEdgeCopyK / 2U;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeStageBytes =
    kSm87A4W4GateUpDownEdgeAStageBytes +
    2U * kSm87A4W4GateUpDownEdgeBStageBytes;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeScaleSlotBytes =
    (kSm87A4W4GateUpDownEdgeTileM +
     2U * kSm87A4W4GateUpDownEdgeComputeTileN) * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87A4W4GateUpDownEdgePipelineBytes =
    kSm87A4W4GateUpDownEdgeStages *
        kSm87A4W4GateUpDownEdgeStageBytes +
    kSm87A4W4GateUpDownEdgeScaleSlots *
        kSm87A4W4GateUpDownEdgeScaleSlotBytes;
inline constexpr std::size_t kSm87A4W4GateUpDownEdgePlaneBytes =
    kSm87A4W4GateUpDownEdgeTileM *
    kSm87A4W4GateUpDownEdgeScaleK * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87A4W4GateUpDownEdgeDynamicSharedBytes =
    kSm87A4W4GateUpDownEdgePipelineBytes +
    kSm87A4W4GateUpDownEdgePlaneBytes;

[[nodiscard]] constexpr bool sm87_a4w4_gateup_down_edge_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_down_edge_ceil_div(
    const std::size_t value, const std::size_t divisor) noexcept {
  return value == 0U || divisor == 0U
             ? 0U
             : 1U + (value - 1U) / divisor;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_down_edge_launch_token_count(
    const std::size_t logical_token_count) noexcept {
  const std::size_t tiles = sm87_a4w4_gateup_down_edge_ceil_div(
      logical_token_count, kSm87A4W4GateUpDownEdgeLaunchAlignment);
  return tiles != 0U &&
                 sm87_a4w4_gateup_down_edge_product_fits(
                     tiles, kSm87A4W4GateUpDownEdgeLaunchAlignment)
             ? tiles * kSm87A4W4GateUpDownEdgeLaunchAlignment
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_down_edge_outer_block_count(
    const std::size_t outer_count) noexcept {
  return sm87_a4w4_gateup_down_edge_ceil_div(
      outer_count, kSm87A4W4GateUpDownEdgeOuterBlock);
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  if (logical_k == 0U ||
      logical_k % kSm87A4W4GateUpDownEdgePhysicalK64 != 0U) {
    return 0U;
  }
  const std::size_t blocks =
      sm87_a4w4_gateup_down_edge_outer_block_count(outer_count);
  const std::size_t groups =
      logical_k / kSm87A4W4GateUpDownEdgePhysicalK64;
  constexpr std::size_t group_bytes =
      kSm87A4W4GateUpDownEdgeOuterBlock *
      kSm87A4W4GateUpDownEdgePackedK64Bytes;
  return blocks != 0U && groups != 0U &&
                 sm87_a4w4_gateup_down_edge_product_fits(blocks, groups) &&
                 sm87_a4w4_gateup_down_edge_product_fits(
                     blocks * groups, group_bytes)
             ? blocks * groups * group_bytes
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_down_edge_scale_capacity_elements(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  if (logical_k == 0U ||
      logical_k % kSm87A4W4GateUpDownEdgeScaleK != 0U) {
    return 0U;
  }
  const std::size_t blocks =
      sm87_a4w4_gateup_down_edge_outer_block_count(outer_count);
  const std::size_t groups = logical_k / kSm87A4W4GateUpDownEdgeScaleK;
  return blocks != 0U && groups != 0U &&
                 sm87_a4w4_gateup_down_edge_product_fits(blocks, groups) &&
                 sm87_a4w4_gateup_down_edge_product_fits(
                     blocks * groups, kSm87A4W4GateUpDownEdgeOuterBlock)
             ? blocks * groups * kSm87A4W4GateUpDownEdgeOuterBlock
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_DOWN_EDGE_HOST_DEVICE constexpr
    std::size_t
sm87_a4w4_gateup_down_edge_packed_offset(
    const std::size_t outer_coordinate, const std::size_t k64_group,
    const std::size_t byte_in_k64,
    const std::size_t k64_group_count) noexcept {
  return (((outer_coordinate / kSm87A4W4GateUpDownEdgeOuterBlock) *
               k64_group_count +
           k64_group) *
              kSm87A4W4GateUpDownEdgeOuterBlock +
          outer_coordinate % kSm87A4W4GateUpDownEdgeOuterBlock) *
             kSm87A4W4GateUpDownEdgePackedK64Bytes +
         byte_in_k64;
}

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_DOWN_EDGE_HOST_DEVICE constexpr
    std::size_t
sm87_a4w4_gateup_down_edge_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate / kSm87A4W4GateUpDownEdgeOuterBlock) *
              k512_group_count +
          k512_group) *
             kSm87A4W4GateUpDownEdgeOuterBlock +
         outer_coordinate % kSm87A4W4GateUpDownEdgeOuterBlock;
}

struct Sm87A4W4GateUpDownEdgePlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t edge_groups{};
  std::size_t compute_cells_per_edge{};
  std::size_t input_k512_groups{};
  std::size_t input_physical_k64_groups{};
  std::size_t output_physical_k64_groups{};
  std::size_t work_edge_cells{};
  std::size_t launch_ctas{};
  std::size_t base_waves{};
  std::size_t residual_m_tiles{};
  std::size_t residual_edge_cells{};
};

// Generic complete-cell planner used by the isolated correctness launcher.
// Base waves preserve one M64 owner per CTA across the complete N sweep.
// Remaining M tiles are decomposed into complete (M64,K512) edge cells and
// distributed N-major across all resident CTAs.
[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgePlan
sm87_a4w4_gateup_down_edge_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t maximum_launch_ctas =
        kSm87A4W4GateUpDownEdgePersistentCtas) noexcept {
  if (logical_token_count == 0U ||
      launch_token_count != sm87_a4w4_gateup_down_edge_launch_token_count(
                                logical_token_count) ||
      launch_token_count % kSm87A4W4GateUpDownEdgeTileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % kSm87A4W4GateUpDownEdgeScaleK != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpDownEdgeScaleK != 0U ||
      maximum_launch_ctas == 0U ||
      maximum_launch_ctas > kSm87A4W4GateUpDownEdgePersistentCtas) {
    return {};
  }
  const std::size_t m_tiles =
      launch_token_count / kSm87A4W4GateUpDownEdgeTileM;
  const std::size_t edge_groups =
      intermediate_size / kSm87A4W4GateUpDownEdgeScaleK;
  if (!sm87_a4w4_gateup_down_edge_product_fits(m_tiles, edge_groups)) {
    return {};
  }
  const std::size_t work_edge_cells = m_tiles * edge_groups;
  const std::size_t launch_ctas =
      work_edge_cells < maximum_launch_ctas ? work_edge_cells
                                            : maximum_launch_ctas;
  const std::size_t base_waves = m_tiles / launch_ctas;
  const std::size_t residual_m_tiles = m_tiles % launch_ctas;
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          m_tiles,
          edge_groups,
          kSm87A4W4GateUpDownEdgeCellsPerScale,
          input_size / kSm87A4W4GateUpDownEdgeScaleK,
          input_size / kSm87A4W4GateUpDownEdgePhysicalK64,
          intermediate_size / kSm87A4W4GateUpDownEdgePhysicalK64,
          work_edge_cells,
          launch_ctas,
          base_waves,
          residual_m_tiles,
          residual_m_tiles * edge_groups};
}

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgePlan
sm87_a4w4_gateup_down_edge_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return launch_token_count <= kSm87A4W4GateUpDownEdgeMaximumLaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpDownEdgeModelIntermediate &&
                 input_size == kSm87A4W4GateUpDownEdgeModelInput
             ? sm87_a4w4_gateup_down_edge_test_plan(
                   logical_token_count, launch_token_count,
                   intermediate_size, input_size)
             : Sm87A4W4GateUpDownEdgePlan{};
}

struct Sm87A4W4GateUpDownEdgeResources final {
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

// Success means <=128 registers/thread, zero local storage, the exact
// 148,736-byte dynamic allocation, and one resident 512-thread CTA/SM on the
// pinned 16-SM SM87 target.
[[nodiscard]] int query_sm87_a4w4_gateup_down_k512_edge_resources_cuda(
    Sm87A4W4GateUpDownEdgeResources* resources) noexcept;

// Production-shape surface.  The output is the current Down K512 consumer
// ABI: packed signed-A4 codes plus one independent BF16 scale per row/K512.
// No BF16 Gate/Up product matrix is written to global memory.
[[nodiscard]] int launch_sm87_a4w4_gateup_down_k512_edge_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_gate_b,
    std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* gate_b_k512_scales_bf16,
    std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* packed_up_b,
    std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* up_b_k512_scales_bf16,
    std::size_t up_b_scale_capacity_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    float output_clip_ratio,
    std::uint8_t* packed_output,
    std::size_t packed_output_capacity_bytes,
    std::uint16_t* output_k512_scales_bf16,
    std::size_t output_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Synthetic-correctness-only generic complete-cell surface.  The CTA cap is
// explicit so tests can exercise the residual edge-cell schedule without
// allocating the full model shape.  It is never a timing authority.
[[nodiscard]] int launch_sm87_a4w4_gateup_down_k512_edge_test_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_gate_b,
    std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* gate_b_k512_scales_bf16,
    std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* packed_up_b,
    std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* up_b_k512_scales_bf16,
    std::size_t up_b_scale_capacity_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    float output_clip_ratio,
    std::uint8_t* packed_output,
    std::size_t packed_output_capacity_bytes,
    std::uint16_t* output_k512_scales_bf16,
    std::size_t output_scale_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4GateUpDownEdgeAStageBytes == 8'192U);
static_assert(kSm87A4W4GateUpDownEdgeBStageBytes == 16'384U);
static_assert(kSm87A4W4GateUpDownEdgeStageBytes == 40'960U);
static_assert(kSm87A4W4GateUpDownEdgeScaleSlotBytes == 640U);
static_assert(kSm87A4W4GateUpDownEdgePipelineBytes == 83'200U);
static_assert(kSm87A4W4GateUpDownEdgePlaneBytes == 65'536U);
static_assert(kSm87A4W4GateUpDownEdgeDynamicSharedBytes == 148'736U);
static_assert(kSm87A4W4GateUpDownEdgeWarpRows *
                      kSm87A4W4GateUpDownEdgeWarpColumns ==
                  kSm87A4W4GateUpDownEdgeWarps);
static_assert(kSm87A4W4GateUpDownEdgeWarpTileM *
                      kSm87A4W4GateUpDownEdgeWarpRows ==
                  kSm87A4W4GateUpDownEdgeTileM);
static_assert(kSm87A4W4GateUpDownEdgeWarpTileN *
                      kSm87A4W4GateUpDownEdgeWarpColumns ==
                  kSm87A4W4GateUpDownEdgeComputeTileN);
static_assert(kSm87A4W4GateUpDownEdgeComputeTileN *
                      kSm87A4W4GateUpDownEdgeCellsPerScale ==
                  kSm87A4W4GateUpDownEdgeScaleK);
static_assert(sm87_a4w4_gateup_down_edge_plan(
                  2'148U, 2'176U, 17'408U, 5'120U)
                  .residual_edge_cells == 68U);
static_assert(sm87_a4w4_gateup_down_edge_plan(
                  512U, 512U, 17'408U, 5'120U)
                  .launch_ctas == 16U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_DOWN_EDGE_HOST_DEVICE
