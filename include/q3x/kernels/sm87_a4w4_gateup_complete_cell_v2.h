#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Gate+Up complete-cell v2 for the K128 consumer ABI.  One 256-thread CTA
// owns M64N128 as two sequential N64 output phases.  Within each output phase
// A is retained in a two-slot logical-K128 ring while Gate and Up are
// independent phases in a four-slot B ring.  The first N64 FP32 product stays
// in shared memory while the second phase runs; the epilogue still reduces
// all 128 values into one exact Down-input scale.
inline constexpr std::size_t kSm87A4W4GateUpCellV2TileM = 64U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2TileN = 128U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2TileK = 128U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2PhaseN = 64U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2Threads = 256U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2AStages = 2U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2BPhaseStages = 4U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2PersistentCtas = 32U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2CtasPerSm = 2U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2MaximumRegisters = 128U;
inline constexpr std::size_t kSm87A4W4GateUpCellV2SharedBytes = 41'728U;

struct Sm87A4W4GateUpCellV2Plan final {
  std::size_t token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_phases{};
  std::size_t k128_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
  std::size_t output_physical_k64_groups{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpCellV2Plan
sm87_a4w4_gateup_cell_v2_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4GateUpCellV2TileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % kSm87A4W4GateUpCellV2TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpCellV2TileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4GateUpCellV2TileM;
  const std::size_t n_phases =
      intermediate_size / kSm87A4W4GateUpCellV2TileN;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (m_tiles == 0U || n_phases == 0U || m_tiles > maximum / n_phases) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_phases;
  const std::size_t launch_ctas =
      m_tiles < kSm87A4W4GateUpCellV2PersistentCtas
          ? m_tiles
          : kSm87A4W4GateUpCellV2PersistentCtas;
  return {token_count,
          intermediate_size,
          input_size,
          m_tiles,
          n_phases,
          input_size / kSm87A4W4GateUpCellV2TileK,
          input_size / 64U,
          work_cells,
          launch_ctas,
          intermediate_size / 64U};
}

struct Sm87A4W4GateUpCellV2Resources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// This is an admission gate.  Success means <=128 registers/thread, exactly
// 41,728 B static shared memory, no local frame/spill, and >=2 CTAs/SM on the
// pinned 16-SM SM87 target.
[[nodiscard]] int query_sm87_a4w4_gateup_cell_v2_resources_cuda(
    Sm87A4W4GateUpCellV2Resources* resources) noexcept;

// The output is the established Down-input ABI: two physical K64 packed-code
// planes share one BF16 K128 scale.  No Gate/Up BF16 or global split-K
// intermediate is materialized.
[[nodiscard]] int launch_sm87_a4w4_gateup_cell_v2_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k128_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_gate_b,
    std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* gate_b_k128_scales_bf16,
    std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* packed_up_b,
    std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* up_b_k128_scales_bf16,
    std::size_t up_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    float output_clip_ratio,
    std::uint8_t* packed_output,
    std::size_t packed_output_capacity_bytes,
    std::uint16_t* output_k128_scales_bf16,
    std::size_t output_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(sm87_a4w4_gateup_cell_v2_plan(64U, 128U, 128U)
                  .work_cells == 1U);
static_assert(sm87_a4w4_gateup_cell_v2_plan(2'048U, 17'408U, 5'120U)
                  .m_tiles == 32U);
static_assert(sm87_a4w4_gateup_cell_v2_plan(2'048U, 17'408U, 5'120U)
                  .n_phases == 136U);
static_assert(sm87_a4w4_gateup_cell_v2_plan(2'048U, 17'408U, 5'120U)
                  .k128_groups == 40U);
static_assert(sm87_a4w4_gateup_cell_v2_plan(2'048U, 17'408U, 5'120U)
                  .work_cells == 4'352U);
static_assert(sm87_a4w4_gateup_cell_v2_plan(2'048U, 17'408U, 5'120U)
                  .launch_ctas == 32U);
static_assert(sm87_a4w4_gateup_cell_v2_plan(2'048U, 17'408U, 5'120U)
                  .output_physical_k64_groups == 272U);
static_assert(sm87_a4w4_gateup_cell_v2_plan(65U, 128U, 128U)
                  .launch_ctas == 0U);

}  // namespace q3x::kernels
