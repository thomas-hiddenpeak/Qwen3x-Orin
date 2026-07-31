#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Test-only Gate+Up projection v3 for the authenticated signed-S4/K128 ABI.
//
// One 256-thread CTA owns M64N128.  Warps 0..3 compute Gate M16N128 strips
// while warps 4..7 compute the matching Up strips.  Both projection crews
// consume one shared A stage, so A codes/scales cross global memory once per
// N128/K128 cell rather than once per N64 phase.  Two complete logical stages
// form a 42,240-byte ring.  Once the K loop drains, the dead ring is reused as
// the Gate -> SiLU(Gate)*Up exchange tile and then by the exact K128 epilogue.
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3TileM = 64U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3TileN = 128U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3TileK = 128U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3Threads = 256U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3Warps = 8U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3ProjectionWarps = 4U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3Stages = 2U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3PersistentCtas = 32U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3CtasPerSm = 2U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3MaximumRegisters =
    128U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3AStageBytes = 4'224U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3BStageBytes = 8'448U;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3LogicalStageBytes =
    kSm87A4W4GateUpProjectionV3AStageBytes +
    2U * kSm87A4W4GateUpProjectionV3BStageBytes;
inline constexpr std::size_t kSm87A4W4GateUpProjectionV3SharedBytes =
    kSm87A4W4GateUpProjectionV3Stages *
    kSm87A4W4GateUpProjectionV3LogicalStageBytes;

struct Sm87A4W4GateUpProjectionV3Plan final {
  std::size_t token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k128_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
  std::size_t output_physical_k64_groups{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpProjectionV3Plan
sm87_a4w4_gateup_projection_v3_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4GateUpProjectionV3TileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % kSm87A4W4GateUpProjectionV3TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpProjectionV3TileK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4GateUpProjectionV3TileM;
  const std::size_t n_tiles =
      intermediate_size / kSm87A4W4GateUpProjectionV3TileN;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  if (m_tiles == 0U || n_tiles == 0U || m_tiles > maximum / n_tiles) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_tiles;
  return {token_count,
          intermediate_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4GateUpProjectionV3TileK,
          input_size / 64U,
          work_cells,
          m_tiles < kSm87A4W4GateUpProjectionV3PersistentCtas
              ? m_tiles
              : kSm87A4W4GateUpProjectionV3PersistentCtas,
          intermediate_size / 64U};
}

struct Sm87A4W4GateUpProjectionV3Resources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// Success means <=128 registers/thread, exactly 42,240 B static shared, no
// local allocation, and at least two active CTAs/SM on the pinned 16-SM SM87.
[[nodiscard]] int query_sm87_a4w4_gateup_projection_v3_resources_cuda(
    Sm87A4W4GateUpProjectionV3Resources* resources) noexcept;

// All inputs and the output use the established consumer-block signed-S4
// layout.  Each pair of physical output K64 planes shares one BF16 K128 scale.
// This isolated cell has no tail policy and fails closed on non-tile shapes.
[[nodiscard]] int launch_sm87_a4w4_gateup_projection_v3_cuda(
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

static_assert(kSm87A4W4GateUpProjectionV3LogicalStageBytes == 21'120U);
static_assert(kSm87A4W4GateUpProjectionV3SharedBytes == 42'240U);
static_assert(kSm87A4W4GateUpProjectionV3SharedBytes *
                      kSm87A4W4GateUpProjectionV3CtasPerSm <=
                  96U * 1'024U);
static_assert(sm87_a4w4_gateup_projection_v3_plan(64U, 128U, 128U)
                  .work_cells == 1U);
static_assert(sm87_a4w4_gateup_projection_v3_plan(
                  2'048U, 17'408U, 5'120U)
                  .m_tiles == 32U);
static_assert(sm87_a4w4_gateup_projection_v3_plan(
                  2'048U, 17'408U, 5'120U)
                  .n_tiles == 136U);
static_assert(sm87_a4w4_gateup_projection_v3_plan(
                  2'048U, 17'408U, 5'120U)
                  .k128_groups == 40U);
static_assert(sm87_a4w4_gateup_projection_v3_plan(
                  2'048U, 17'408U, 5'120U)
                  .work_cells == 4'352U);
static_assert(sm87_a4w4_gateup_projection_v3_plan(
                  2'048U, 17'408U, 5'120U)
                  .launch_ctas == 32U);
static_assert(sm87_a4w4_gateup_projection_v3_plan(65U, 128U, 128U)
                  .launch_ctas == 0U);

}  // namespace q3x::kernels
