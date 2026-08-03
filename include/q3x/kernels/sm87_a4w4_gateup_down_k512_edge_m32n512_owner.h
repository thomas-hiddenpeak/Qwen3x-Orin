#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Resource-first structural mirror for a producer-owned M32N512 Gate+Up
// cell.  One 256-thread CTA visits eight M32N64 subcells.  Warp w owns N8
// and temporally covers two M16 panels, retaining matching Gate and Up
// accumulators until every exact K512 group has been applied.  The resulting
// BF16 products remain in one CTA-owned M32N512 edge plane and are published
// directly through the canonical Down-input A4/K512 ABI.
//
// Four K128 cp.async stages keep the code payload at the same 40,960-byte
// footprint as a two-stage K256 ring while exposing twice the async depth.
// Including two scale slots, the complete pipeline is 41,600 bytes.  The two
// K64 planes in all four stages still accumulate into one S32 partial before
// the exact K512 scale/FMA boundary.
//
// This remains an independent default-off candidate.  It has strict explicit
// launchers and a child runtime selector; merely linking the object cannot
// alter production unless both the pair-feed parent and child are selected.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerTileN = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerCellsPerEdge = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerCopyK = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerStages = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlots = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerWarpTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerWarpTileN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerM16PanelsPerWarp = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerPersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerTeams = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerLaunchAlignment = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerCtasPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerPackedK64Bytes = 32U;

// A logical scheduler unit is an adjacent pair of M32 owners.  The two CTAs
// share only traversal order for opportunistic L2 reuse; they never exchange
// data and correctness does not require physical co-residency.
struct Sm87A4W4GateUpDownEdgeM32N512OwnerPlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m64_pairs{};
  std::size_t edge_groups{};
  std::size_t input_k512_groups{};
  std::size_t input_physical_k64_groups{};
  std::size_t output_physical_k64_groups{};
  std::size_t launch_ctas{};
  std::size_t teams{};
  std::size_t base_waves{};
  std::size_t residual_pairs{};
  std::size_t residual_edge_cells{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM32N512OwnerPlan
sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  if (logical_token_count == 0U ||
      launch_token_count != sm87_a4w4_gateup_down_edge_launch_token_count(
                                logical_token_count) ||
      launch_token_count %
              (2U * kSm87A4W4GateUpDownEdgeM32N512OwnerTileM) !=
          0U ||
      intermediate_size == 0U ||
      intermediate_size %
              kSm87A4W4GateUpDownEdgeM32N512OwnerTileN !=
          0U ||
      input_size == 0U ||
      input_size %
              kSm87A4W4GateUpDownEdgeM32N512OwnerScaleK !=
          0U) {
    return {};
  }
  const std::size_t m64_pairs =
      launch_token_count /
      (2U * kSm87A4W4GateUpDownEdgeM32N512OwnerTileM);
  const std::size_t edge_groups =
      intermediate_size /
      kSm87A4W4GateUpDownEdgeM32N512OwnerTileN;
  if (!sm87_a4w4_gateup_down_edge_product_fits(
          m64_pairs, edge_groups)) {
    return {};
  }
  const std::size_t base_waves =
      m64_pairs / kSm87A4W4GateUpDownEdgeM32N512OwnerTeams;
  const std::size_t residual_pairs =
      m64_pairs % kSm87A4W4GateUpDownEdgeM32N512OwnerTeams;
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          m64_pairs,
          edge_groups,
          input_size /
              kSm87A4W4GateUpDownEdgeM32N512OwnerScaleK,
          input_size / kSm87A4W4GateUpDownEdgePhysicalK64,
          intermediate_size / kSm87A4W4GateUpDownEdgePhysicalK64,
          kSm87A4W4GateUpDownEdgeM32N512OwnerPersistentCtas,
          kSm87A4W4GateUpDownEdgeM32N512OwnerTeams,
          base_waves,
          residual_pairs,
          residual_pairs * edge_groups};
}

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM32N512OwnerPlan
sm87_a4w4_gateup_down_k512_edge_m32n512_owner_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return launch_token_count <=
                     kSm87A4W4GateUpDownEdgeMaximumLaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpDownEdgeModelIntermediate &&
                 input_size == kSm87A4W4GateUpDownEdgeModelInput
             ? sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_plan(
                   logical_token_count, launch_token_count,
                   intermediate_size, input_size)
             : Sm87A4W4GateUpDownEdgeM32N512OwnerPlan{};
}

inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerAStageBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerTileM *
        kSm87A4W4GateUpDownEdgeM32N512OwnerCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerBStageBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN *
        kSm87A4W4GateUpDownEdgeM32N512OwnerCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerStageBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerAStageBytes +
        2U * kSm87A4W4GateUpDownEdgeM32N512OwnerBStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlotBytes =
        (kSm87A4W4GateUpDownEdgeM32N512OwnerTileM +
         2U * kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerPipelineBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerStages *
            kSm87A4W4GateUpDownEdgeM32N512OwnerStageBytes +
        kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlots *
            kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlotBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerEdgePlaneBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerTileM *
        kSm87A4W4GateUpDownEdgeM32N512OwnerTileN *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerPipelineBytes +
        kSm87A4W4GateUpDownEdgeM32N512OwnerEdgePlaneBytes;

struct Sm87A4W4GateUpDownEdgeM32N512OwnerResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t configured_dynamic_shared_limit_bytes{};
  std::size_t device_optin_shared_limit_bytes{};
  std::size_t device_shared_per_sm_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int
query_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resources_cuda(
    Sm87A4W4GateUpDownEdgeM32N512OwnerResources* resources) noexcept;

// Production-shape launcher.  It always submits exactly 32 CTAs (16 logical
// M64 pairs) and emits the canonical packed-A4/BF16-K512 Down-input ABI.
[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_cuda(
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

// Generic-shape correctness launcher.  It retains the exact same fixed
// 32-CTA scheduler and admission contract; only model dimensions are relaxed.
[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_cuda(
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

static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerAStageBytes == 2'048U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerBStageBytes == 4'096U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerStageBytes == 10'240U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlotBytes == 320U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerPipelineBytes == 41'600U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerEdgePlaneBytes == 32'768U);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes == 74'368U);
static_assert(
    2U * kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes ==
    kSm87A4W4GateUpDownEdgeDynamicSharedBytes);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N512OwnerPersistentCtas ==
    2U * kSm87A4W4GateUpDownEdgeM32N512OwnerTeams);
static_assert(
    sm87_a4w4_gateup_down_k512_edge_m32n512_owner_test_plan(
        117U, 128U, 1'024U, 1'536U)
            .residual_edge_cells == 4U);
static_assert(
    sm87_a4w4_gateup_down_k512_edge_m32n512_owner_plan(
        512U, 512U, 17'408U, 5'120U)
            .launch_ctas == 32U);

}  // namespace q3x::kernels
