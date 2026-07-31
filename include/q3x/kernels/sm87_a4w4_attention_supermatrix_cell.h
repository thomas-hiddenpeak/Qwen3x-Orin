#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Standalone input-side supermatrix complete cell for authenticated K128 A4
// attention projections.  One 256-thread CTA owns M128 and two independent
// N64 output panels.  All eight warps retain one M16 strip from both panels,
// so the pair shares one A stage while preserving separate BF16 output ABIs.
//
// The target Linear Attention shape pairs QKV [10240,5120] with
// Z [6144,5120].  The first 96 cells pair one QKV panel with one Z panel; the
// remaining 32 cells pair the remaining QKV panels.  This is a new M128
// complete-cell kernel, not a wrapper around the generic M64N256 launcher.
inline constexpr std::size_t kSm87A4W4AttentionCellTileM = 128U;
inline constexpr std::size_t kSm87A4W4AttentionCellPanelN = 64U;
inline constexpr std::size_t kSm87A4W4AttentionCellPanelsPerCell = 2U;
inline constexpr std::size_t kSm87A4W4AttentionCellTileK = 128U;
inline constexpr std::size_t kSm87A4W4AttentionCellThreads = 256U;
inline constexpr std::size_t kSm87A4W4AttentionCellWarps = 8U;
inline constexpr std::size_t kSm87A4W4AttentionCellAStages = 2U;
inline constexpr std::size_t kSm87A4W4AttentionCellBPairStages = 3U;
inline constexpr std::size_t kSm87A4W4AttentionCellPersistentCtas = 32U;
inline constexpr std::size_t kSm87A4W4AttentionCellCtasPerSm = 2U;
inline constexpr std::size_t kSm87A4W4AttentionCellMaximumRegisters = 128U;
inline constexpr std::size_t kSm87A4W4AttentionCellStageBytes = 8'448U;
inline constexpr std::size_t kSm87A4W4AttentionCellSharedBytes = 42'240U;

inline constexpr std::size_t kQwen36LinearQkvOutputSize = 10'240U;
inline constexpr std::size_t kQwen36LinearZOutputSize = 6'144U;
inline constexpr std::size_t kQwen36AttentionInputSize = 5'120U;
inline constexpr std::size_t kQwen36LinearQkvPanels =
    kQwen36LinearQkvOutputSize / kSm87A4W4AttentionCellPanelN;
inline constexpr std::size_t kQwen36LinearZPanels =
    kQwen36LinearZOutputSize / kSm87A4W4AttentionCellPanelN;

// The same cell topology also covers the future Full Attention input-side
// supermatrix: Q=192 N64 panels and K/V=16 panels each, or 112 pair cells.
// That three-output launcher is deliberately not implemented by this first
// vertical slice.
inline constexpr std::size_t kQwen36FullQOutputSize = 12'288U;
inline constexpr std::size_t kQwen36FullKvOutputSize = 1'024U;
inline constexpr std::size_t kQwen36AttentionOInputSize = 6'144U;
inline constexpr std::size_t kQwen36AttentionOOutputSize = 5'120U;
inline constexpr std::size_t kQwen36FullQPanels =
    kQwen36FullQOutputSize / kSm87A4W4AttentionCellPanelN;
inline constexpr std::size_t kQwen36FullKvPanels =
    kQwen36FullKvOutputSize / kSm87A4W4AttentionCellPanelN;
inline constexpr std::size_t kQwen36FullAttentionPairCells =
    (kQwen36FullQPanels + 2U * kQwen36FullKvPanels) /
    kSm87A4W4AttentionCellPanelsPerCell;

struct Sm87A4W4AttentionPairPanel final {
  std::uint8_t projection{};
  std::size_t panel{};
};

struct Sm87A4W4AttentionSupermatrixPlan final {
  std::size_t token_count{};
  std::size_t first_output_size{};
  std::size_t second_output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t first_panels{};
  std::size_t second_panels{};
  std::size_t paired_panels{};
  std::size_t pair_cells{};
  std::size_t k128_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4AttentionSupermatrixPlan
sm87_a4w4_attention_supermatrix_plan(
    const std::size_t token_count, const std::size_t first_output_size,
    const std::size_t second_output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4AttentionCellTileM != 0U ||
      first_output_size == 0U ||
      first_output_size % kSm87A4W4AttentionCellPanelN != 0U ||
      second_output_size == 0U ||
      second_output_size % kSm87A4W4AttentionCellPanelN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4AttentionCellTileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4AttentionCellTileM;
  const std::size_t first_panels =
      first_output_size / kSm87A4W4AttentionCellPanelN;
  const std::size_t second_panels =
      second_output_size / kSm87A4W4AttentionCellPanelN;
  const std::size_t paired_panels =
      first_panels < second_panels ? first_panels : second_panels;
  const std::size_t remaining_panels =
      first_panels > second_panels ? first_panels - second_panels
                                   : second_panels - first_panels;
  if (remaining_panels % kSm87A4W4AttentionCellPanelsPerCell != 0U) {
    return {};
  }
  const std::size_t pair_cells =
      paired_panels +
      remaining_panels / kSm87A4W4AttentionCellPanelsPerCell;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (m_tiles == 0U || pair_cells == 0U || m_tiles > maximum / pair_cells) {
    return {};
  }
  const std::size_t work_cells = m_tiles * pair_cells;
  return {token_count,
          first_output_size,
          second_output_size,
          input_size,
          m_tiles,
          first_panels,
          second_panels,
          paired_panels,
          pair_cells,
          input_size / kSm87A4W4AttentionCellTileK,
          input_size / 64U,
          work_cells,
          work_cells < kSm87A4W4AttentionCellPersistentCtas
              ? work_cells
              : kSm87A4W4AttentionCellPersistentCtas};
}

[[nodiscard]] constexpr Sm87A4W4AttentionPairPanel
sm87_a4w4_attention_supermatrix_panel(
    const Sm87A4W4AttentionSupermatrixPlan& plan,
    const std::size_t pair_cell, const std::size_t slot) noexcept {
  if (pair_cell >= plan.pair_cells ||
      slot >= kSm87A4W4AttentionCellPanelsPerCell) {
    return {2U, 0U};
  }
  if (pair_cell < plan.paired_panels) {
    return {static_cast<std::uint8_t>(slot), pair_cell};
  }
  const std::size_t remaining_cell = pair_cell - plan.paired_panels;
  if (plan.first_panels > plan.second_panels) {
    return {0U, plan.paired_panels + 2U * remaining_cell + slot};
  }
  return {1U, plan.paired_panels + 2U * remaining_cell + slot};
}

struct Sm87A4W4AttentionSupermatrixResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// Queries the fixed real-shape Linear QKV+Z specialization.  Success requires
// <=128 registers/thread, exactly 42,240 B static shared memory, zero local
// frame/spill, and at least two active CTAs/SM on the 16-SM SM87 target.
[[nodiscard]] int
query_sm87_a4w4_linear_qkv_z_supermatrix_resources_cuda(
    Sm87A4W4AttentionSupermatrixResources* resources) noexcept;

// Fixed Qwen3.6 Linear Attention launcher.  Both weight payloads use the
// authenticated K128 A4 consumer ABI: two physical K64 code blocks share one
// BF16 scale. Outputs remain independent row-major BF16 matrices.
[[nodiscard]] int launch_sm87_a4w4_linear_qkv_z_supermatrix_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k128_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_qkv_b,
    std::size_t packed_qkv_b_capacity_bytes,
    const std::uint16_t* qkv_b_k128_scales_bf16,
    std::size_t qkv_b_scale_capacity_elements,
    const std::uint8_t* packed_z_b,
    std::size_t packed_z_b_capacity_bytes,
    const std::uint16_t* z_b_k128_scales_bf16,
    std::size_t z_b_scale_capacity_elements,
    std::size_t token_count,
    std::uint16_t* qkv_output_bf16,
    std::size_t qkv_output_row_stride_elements,
    std::size_t qkv_output_capacity_elements,
    std::uint16_t* z_output_bf16,
    std::size_t z_output_row_stride_elements,
    std::size_t z_output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Synthetic correctness-only specialization: N256 plus N128 and arbitrary
// K128 multiples. Its first two cells cross-pair the projections and its last
// cell pairs the two remaining first-projection panels, exercising both paths
// used by the fixed real-shape topology.
[[nodiscard]] int
launch_sm87_a4w4_attention_pair_supermatrix_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k128_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_first_b,
    std::size_t packed_first_b_capacity_bytes,
    const std::uint16_t* first_b_k128_scales_bf16,
    std::size_t first_b_scale_capacity_elements,
    const std::uint8_t* packed_second_b,
    std::size_t packed_second_b_capacity_bytes,
    const std::uint16_t* second_b_k128_scales_bf16,
    std::size_t second_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t input_size,
    std::uint16_t* first_output_bf16,
    std::size_t first_output_row_stride_elements,
    std::size_t first_output_capacity_elements,
    std::uint16_t* second_output_bf16,
    std::size_t second_output_row_stride_elements,
    std::size_t second_output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4AttentionCellStageBytes *
                      (kSm87A4W4AttentionCellAStages +
                       kSm87A4W4AttentionCellBPairStages) ==
                  kSm87A4W4AttentionCellSharedBytes);
static_assert(kSm87A4W4AttentionCellSharedBytes *
                  kSm87A4W4AttentionCellCtasPerSm <=
              96U * 1'024U);
static_assert(kQwen36LinearQkvPanels == 160U);
static_assert(kQwen36LinearZPanels == 96U);
static_assert(kQwen36FullQPanels == 192U);
static_assert(kQwen36FullKvPanels == 16U);
static_assert(kQwen36FullAttentionPairCells == 112U);
static_assert(sm87_a4w4_attention_supermatrix_plan(
                  2'048U, kQwen36LinearQkvOutputSize,
                  kQwen36LinearZOutputSize, kQwen36AttentionInputSize)
                  .pair_cells == 128U);
static_assert(sm87_a4w4_attention_supermatrix_plan(
                  2'048U, kQwen36LinearQkvOutputSize,
                  kQwen36LinearZOutputSize, kQwen36AttentionInputSize)
                  .work_cells == 2'048U);
static_assert(sm87_a4w4_attention_supermatrix_plan(
                  2'048U, kQwen36LinearQkvOutputSize,
                  kQwen36LinearZOutputSize, kQwen36AttentionInputSize)
                  .launch_ctas == 32U);
static_assert(sm87_a4w4_attention_supermatrix_plan(
                  2'048U, kQwen36LinearQkvOutputSize,
                  kQwen36LinearZOutputSize, kQwen36AttentionInputSize)
                  .k128_groups == 40U);

}  // namespace q3x::kernels
