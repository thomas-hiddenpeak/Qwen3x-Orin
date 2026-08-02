#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off structural admission for a wide Gate+Up -> Down-input cell.
// One 256-thread CTA owns M128N512.  Its eight warps sweep eight N64 cells;
// warp w owns M128N8, loads each canonical-v1 Gate/Up B fragment directly
// through L1, and reuses it across all eight M16 panels.  A alone is staged
// in the established XOR-16B layout and every MMA A tuple is supplied by one
// ldmatrix.x4 instruction.  The N64 sweep reloads A for each cell; it does
// not claim A persistence across N512.  For an equal M128N512 output region
// this first v1 probe therefore trades roughly 2x incumbent A code traffic
// for roughly 1/2 incumbent B code traffic.  Its job is to prove the wide
// ownership/LDSM/publication skeleton before the paired-B production backend
// is judged on the real API path.
//
// The first M64N512 BF16 product half lives beside the two-stage A pipeline
// in dynamic shared memory.  The second half lives in caller-owned, stable
// 64-KiB-per-CTA scratch until the first half is quantized.  The scratch half
// is then copied into the same shared edge plane and quantized through the
// identical K512 Down-v1 publication boundary.  Merely linking this target
// never changes runtime selection.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixTileN = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixComputeTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixCellsPerEdge = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixWarpTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixWarpTileN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixM16Panels = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixK64PerScale = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixAStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixAStageBytes =
        kSm87A4W4GateUpDownEdgeM128N512LdmatrixTileM *
        kSm87A4W4GateUpDownEdgeScaleK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixScaleSlotBytes =
        (kSm87A4W4GateUpDownEdgeM128N512LdmatrixTileM +
         2U * kSm87A4W4GateUpDownEdgeM128N512LdmatrixComputeTileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixPipelineBytes =
        kSm87A4W4GateUpDownEdgeM128N512LdmatrixAStages *
        (kSm87A4W4GateUpDownEdgeM128N512LdmatrixAStageBytes +
         kSm87A4W4GateUpDownEdgeM128N512LdmatrixScaleSlotBytes);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixEdgePlaneBytes =
        64U * kSm87A4W4GateUpDownEdgeScaleK * sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixDynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeM128N512LdmatrixPipelineBytes +
        kSm87A4W4GateUpDownEdgeM128N512LdmatrixEdgePlaneBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixScratchBytesPerCta =
        kSm87A4W4GateUpDownEdgeM128N512LdmatrixEdgePlaneBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixMaximumRegisters = 255U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixPersistentCtas = 16U;

struct Sm87A4W4GateUpDownEdgeM128N512LdmatrixPlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t edge_groups{};
  std::size_t input_k512_groups{};
  std::size_t input_physical_k64_groups{};
  std::size_t output_physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
  std::size_t base_waves{};
  std::size_t residual_m_tiles{};
  std::size_t residual_cells{};
  std::size_t required_scratch_bytes{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM128N512LdmatrixPlan
sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t maximum_launch_ctas =
        kSm87A4W4GateUpDownEdgeM128N512LdmatrixPersistentCtas) noexcept {
  const Sm87A4W4GateUpDownEdgePlan base =
      sm87_a4w4_gateup_down_edge_test_plan(
          logical_token_count, launch_token_count, intermediate_size,
          input_size, maximum_launch_ctas);
  if (base.launch_ctas == 0U ||
      launch_token_count %
              kSm87A4W4GateUpDownEdgeM128N512LdmatrixTileM !=
          0U) {
    return {};
  }
  const std::size_t m_tiles =
      launch_token_count /
      kSm87A4W4GateUpDownEdgeM128N512LdmatrixTileM;
  if (!sm87_a4w4_gateup_down_edge_product_fits(
          m_tiles, base.edge_groups)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * base.edge_groups;
  const std::size_t launch_ctas =
      work_cells < maximum_launch_ctas ? work_cells : maximum_launch_ctas;
  if (!sm87_a4w4_gateup_down_edge_product_fits(
          launch_ctas,
          kSm87A4W4GateUpDownEdgeM128N512LdmatrixScratchBytesPerCta)) {
    return {};
  }
  const std::size_t base_waves = m_tiles / launch_ctas;
  const std::size_t residual_m_tiles = m_tiles % launch_ctas;
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          m_tiles,
          base.edge_groups,
          base.input_k512_groups,
          base.input_physical_k64_groups,
          base.output_physical_k64_groups,
          work_cells,
          launch_ctas,
          base_waves,
          residual_m_tiles,
          residual_m_tiles * base.edge_groups,
          launch_ctas *
              kSm87A4W4GateUpDownEdgeM128N512LdmatrixScratchBytesPerCta};
}

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM128N512LdmatrixPlan
sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return launch_token_count <= kSm87A4W4GateUpDownEdgeMaximumLaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpDownEdgeModelIntermediate &&
                 input_size == kSm87A4W4GateUpDownEdgeModelInput
             ? sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_test_plan(
                   logical_token_count, launch_token_count,
                   intermediate_size, input_size)
             : Sm87A4W4GateUpDownEdgeM128N512LdmatrixPlan{};
}

struct Sm87A4W4GateUpDownEdgeM128N512LdmatrixResources final {
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

[[nodiscard]] int
query_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_resources_cuda(
    Sm87A4W4GateUpDownEdgeM128N512LdmatrixResources* resources) noexcept;

#define Q3X_SM87_A4W4_GATEUP_DOWN_M128N512_LDMATRIX_LAUNCH_ARGUMENTS       \
    const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,     \
    const std::uint16_t* a_k512_scales_bf16,                              \
    std::size_t a_scale_capacity_elements,                                \
    const std::uint8_t* packed_gate_b,                                    \
    std::size_t packed_gate_b_capacity_bytes,                             \
    const std::uint16_t* gate_b_k512_scales_bf16,                         \
    std::size_t gate_b_scale_capacity_elements,                           \
    const std::uint8_t* packed_up_b, std::size_t packed_up_b_capacity_bytes, \
    const std::uint16_t* up_b_k512_scales_bf16,                           \
    std::size_t up_b_scale_capacity_elements,                             \
    std::size_t logical_token_count, std::size_t launch_token_count,      \
    std::size_t intermediate_size, std::size_t input_size,                \
    float output_clip_ratio, std::uint8_t* cta_scratch,                   \
    std::size_t cta_scratch_capacity_bytes, std::uint8_t* packed_output,  \
    std::size_t packed_output_capacity_bytes,                             \
    std::uint16_t* output_k512_scales_bf16,                               \
    std::size_t output_scale_capacity_elements

[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_cuda(
    Q3X_SM87_A4W4_GATEUP_DOWN_M128N512_LDMATRIX_LAUNCH_ARGUMENTS,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_test_cuda(
    Q3X_SM87_A4W4_GATEUP_DOWN_M128N512_LDMATRIX_LAUNCH_ARGUMENTS,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

#undef Q3X_SM87_A4W4_GATEUP_DOWN_M128N512_LDMATRIX_LAUNCH_ARGUMENTS

static_assert(
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixAStageBytes == 32'768U);
static_assert(
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixScaleSlotBytes == 512U);
static_assert(
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixPipelineBytes == 66'560U);
static_assert(
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixEdgePlaneBytes == 65'536U);
static_assert(
    kSm87A4W4GateUpDownEdgeM128N512LdmatrixDynamicSharedBytes == 132'096U);

}  // namespace q3x::kernels
