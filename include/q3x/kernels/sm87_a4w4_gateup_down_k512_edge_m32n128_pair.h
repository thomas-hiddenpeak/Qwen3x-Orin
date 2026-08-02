#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off structural Gate+Up experiment over the authenticated v1 K512
// payload.  A resident pair of M32N128 CTAs owns one logical M64 row tile.
// Within each CTA, warps 0..3 compute Gate and warps 4..7 compute Up over
// matching M32N32 strips while sharing the same three-stage A ring.  B codes
// and both scale operands are consumed directly from the v1 global layout;
// this experiment introduces no converter or alternate weight ABI.
//
// Gate crews publish one M32N128 FP32 exchange after the complete K sweep.
// Up crews consume it, apply the production SiLU/product/BF16 seam, and fill
// one quarter of the existing M32N512 edge plane.  The exchange lifetime and
// A-ring lifetime are disjoint, so their shared allocation is a union:
//
//   max(3 * M32*K256/2, M32*N128*sizeof(float)) + M32*K512*sizeof(bf16)
//   = max(12,288, 16,384) + 32,768 = 49,152 bytes.
//
// The launch is deliberately cooperative and test-only: exactly 32 CTAs are
// launched on the pinned 16-SM target.  %smid plus an atomic ticket assigns
// the two resident CTAs different M32 halves; a per-SM rendezvous gives every
// pair the same M64/K512 work cell.  No runtime dispatcher can select this
// kernel merely because its object is linked.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairPairTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairTileN = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairEdgeK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairPhysicalK64 = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairCopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairK64PerCopy = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairCopiesPerScale = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairStages = 3U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairProjectionWarps = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairWarpTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairWarpTileN = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairCellsPerEdge = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairCtasPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSmCount = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairPersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairWorkspaceAlignment = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairWorkspaceBytes = 144U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairAStageBytes =
        kSm87A4W4GateUpDownEdgeM32N128PairTileM *
        kSm87A4W4GateUpDownEdgeM32N128PairCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairARingBytes =
        kSm87A4W4GateUpDownEdgeM32N128PairStages *
        kSm87A4W4GateUpDownEdgeM32N128PairAStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairGateExchangeBytes =
        kSm87A4W4GateUpDownEdgeM32N128PairTileM *
        kSm87A4W4GateUpDownEdgeM32N128PairTileN * sizeof(float);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairEdgePlaneBytes =
        kSm87A4W4GateUpDownEdgeM32N128PairTileM *
        kSm87A4W4GateUpDownEdgeM32N128PairEdgeK *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeM32N128PairGateExchangeBytes +
        kSm87A4W4GateUpDownEdgeM32N128PairEdgePlaneBytes;

struct Sm87A4W4GateUpDownEdgeM32N128PairPlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t pair_m_tiles{};
  std::size_t edge_groups{};
  std::size_t input_k512_groups{};
  std::size_t input_physical_k64_groups{};
  std::size_t output_physical_k64_groups{};
  std::size_t work_edge_cells{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM32N128PairPlan
sm87_a4w4_gateup_down_edge_m32n128_pair_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count > launch_token_count ||
      launch_token_count == 0U ||
      launch_token_count %
              kSm87A4W4GateUpDownEdgeM32N128PairPairTileM !=
          0U ||
      intermediate_size == 0U ||
      intermediate_size %
              kSm87A4W4GateUpDownEdgeM32N128PairEdgeK !=
          0U ||
      input_size == 0U ||
      input_size %
              kSm87A4W4GateUpDownEdgeM32N128PairEdgeK !=
          0U) {
    return {};
  }
  const std::size_t pair_m_tiles =
      launch_token_count /
      kSm87A4W4GateUpDownEdgeM32N128PairPairTileM;
  const std::size_t edge_groups =
      intermediate_size /
      kSm87A4W4GateUpDownEdgeM32N128PairEdgeK;
  if (!sm87_a4w4_gateup_down_edge_product_fits(
          pair_m_tiles, edge_groups)) {
    return {};
  }
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          pair_m_tiles,
          edge_groups,
          input_size /
              kSm87A4W4GateUpDownEdgeM32N128PairEdgeK,
          input_size /
              kSm87A4W4GateUpDownEdgeM32N128PairPhysicalK64,
          intermediate_size /
              kSm87A4W4GateUpDownEdgeM32N128PairPhysicalK64,
          pair_m_tiles * edge_groups,
          kSm87A4W4GateUpDownEdgeM32N128PairPersistentCtas};
}

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM32N128PairPlan
sm87_a4w4_gateup_down_edge_m32n128_pair_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return launch_token_count <=
                     kSm87A4W4GateUpDownEdgeMaximumLaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpDownEdgeModelIntermediate &&
                 input_size == kSm87A4W4GateUpDownEdgeModelInput
             ? sm87_a4w4_gateup_down_edge_m32n128_pair_test_plan(
                   logical_token_count, launch_token_count,
                   intermediate_size, input_size)
             : Sm87A4W4GateUpDownEdgeM32N128PairPlan{};
}

struct Sm87A4W4GateUpDownEdgeM32N128PairResources final {
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
query_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_resources_cuda(
    Sm87A4W4GateUpDownEdgeM32N128PairResources* resources) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_cuda(
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
    void* cooperative_workspace,
    std::size_t cooperative_workspace_capacity_bytes,
    void* cuda_stream = nullptr) noexcept;

// Synthetic-correctness-only generic surface.  It still launches the exact
// 32-CTA cooperative schedule so resource and rendezvous failures cannot be
// hidden by a smaller test grid.
[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_test_cuda(
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

static_assert(kSm87A4W4GateUpDownEdgeM32N128PairAStageBytes ==
              4'096U);
static_assert(kSm87A4W4GateUpDownEdgeM32N128PairARingBytes ==
              12'288U);
static_assert(kSm87A4W4GateUpDownEdgeM32N128PairGateExchangeBytes ==
              16'384U);
static_assert(kSm87A4W4GateUpDownEdgeM32N128PairEdgePlaneBytes ==
              32'768U);
static_assert(kSm87A4W4GateUpDownEdgeM32N128PairDynamicSharedBytes ==
              49'152U);
static_assert(kSm87A4W4GateUpDownEdgeM32N128PairWarps ==
              2U * kSm87A4W4GateUpDownEdgeM32N128PairProjectionWarps);
static_assert(kSm87A4W4GateUpDownEdgeM32N128PairM16PerWarp * 16U ==
              kSm87A4W4GateUpDownEdgeM32N128PairWarpTileM);
static_assert(kSm87A4W4GateUpDownEdgeM32N128PairN8PerWarp * 8U ==
              kSm87A4W4GateUpDownEdgeM32N128PairWarpTileN);
static_assert(
    sm87_a4w4_gateup_down_edge_m32n128_pair_test_plan(
        64U, 64U, 512U, 512U)
        .work_edge_cells == 1U);
static_assert(
    sm87_a4w4_gateup_down_edge_m32n128_pair_test_plan(
        96U, 128U, 512U, 1'536U)
        .work_edge_cells == 2U);

}  // namespace q3x::kernels
