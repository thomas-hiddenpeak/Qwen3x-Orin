#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off, one-CTA/SM Gate+Up owner for the complete M64N512
// quantization edge.  One 256-thread CTA executes two temporal M64N256
// cells.  Warp w owns M64N32, and each B fragment feeds four ordered M16
// IMMA panels before it dies.
//
// Three raw K64 cp.async stages expose a consumer-order/XOR/LDSM feed for
// A64 and the complete Gate/Up B256 planes.  Both projections' 128 FP32
// output registers remain live for the complete input K.  The exact K512
// S32 terms for two complete N8 stripes stay in registers.  The third uses
// a reusable 32-KiB CTA scratch, and the fourth temporarily borrows exactly
// the current, not-yet-published 32-KiB M64N256 half-edge.
// Cell zero may touch only half zero and publishes it before cell one starts;
// cell one then borrows only half one, so already-published BF16 is never
// overwritten.  At each group boundary scratch is consumed, and after the
// last group the same half becomes BF16.  Nothing crosses a global seam.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinTileN = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinCellsPerEdge = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinWarpTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinWarpTileN = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinM16PanelsPerWarp = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinN8PerWarp = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinN8PerPhase = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinRegisterN8 = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinScratchN8 = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinPhysicalK = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinCopyK = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinK64PerScale = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinK64PerStage = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinStages = 3U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinScaleSlots = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinPersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinMaximumRegisters = 255U;

inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinAStageBytes =
        kSm87A4W4GateUpDownEdgeM64N256MarlinTileM *
        kSm87A4W4GateUpDownEdgeM64N256MarlinCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinBStageBytes =
        kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN *
        kSm87A4W4GateUpDownEdgeM64N256MarlinCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinStageBytes =
        kSm87A4W4GateUpDownEdgeM64N256MarlinAStageBytes +
        2U * kSm87A4W4GateUpDownEdgeM64N256MarlinBStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinScaleSlotBytes =
        (kSm87A4W4GateUpDownEdgeM64N256MarlinTileM +
         2U * kSm87A4W4GateUpDownEdgeM64N256MarlinComputeTileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinPipelineBytes =
        kSm87A4W4GateUpDownEdgeM64N256MarlinStages *
            kSm87A4W4GateUpDownEdgeM64N256MarlinStageBytes +
        kSm87A4W4GateUpDownEdgeM64N256MarlinScaleSlots *
            kSm87A4W4GateUpDownEdgeM64N256MarlinScaleSlotBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinEdgePlaneBytes =
        kSm87A4W4GateUpDownEdgeM64N256MarlinTileM *
        kSm87A4W4GateUpDownEdgeM64N256MarlinTileN *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinAuxScratchBytes =
        2U * kSm87A4W4GateUpDownEdgeM64N256MarlinTileM *
        (kSm87A4W4GateUpDownEdgeM64N256MarlinWarpTileN /
         kSm87A4W4GateUpDownEdgeM64N256MarlinN8PerWarp) *
        kSm87A4W4GateUpDownEdgeM64N256MarlinWarps *
        sizeof(std::int32_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeM64N256MarlinPipelineBytes +
        kSm87A4W4GateUpDownEdgeM64N256MarlinEdgePlaneBytes +
        kSm87A4W4GateUpDownEdgeM64N256MarlinAuxScratchBytes;

struct Sm87A4W4GateUpDownEdgeM64N256MarlinPlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m64_tiles{};
  std::size_t edge_groups{};
  std::size_t input_k512_groups{};
  std::size_t input_physical_k64_groups{};
  std::size_t output_physical_k64_groups{};
  std::size_t work_edges{};
  std::size_t launch_ctas{};
  std::size_t minimum_edges_per_cta{};
  std::size_t maximum_edges_per_cta{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM64N256MarlinPlan
sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  if (logical_token_count == 0U ||
      launch_token_count != sm87_a4w4_gateup_down_edge_launch_token_count(
                                logical_token_count) ||
      launch_token_count %
              kSm87A4W4GateUpDownEdgeM64N256MarlinTileM !=
          0U ||
      intermediate_size == 0U ||
      intermediate_size %
              kSm87A4W4GateUpDownEdgeM64N256MarlinTileN !=
          0U ||
      input_size == 0U ||
      input_size %
              kSm87A4W4GateUpDownEdgeM64N256MarlinScaleK !=
          0U) {
    return {};
  }
  const std::size_t m64_tiles =
      launch_token_count /
      kSm87A4W4GateUpDownEdgeM64N256MarlinTileM;
  const std::size_t edge_groups =
      intermediate_size /
      kSm87A4W4GateUpDownEdgeM64N256MarlinTileN;
  if (!sm87_a4w4_gateup_down_edge_product_fits(m64_tiles,
                                                edge_groups)) {
    return {};
  }
  const std::size_t work_edges = m64_tiles * edge_groups;
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          m64_tiles,
          edge_groups,
          input_size /
              kSm87A4W4GateUpDownEdgeM64N256MarlinScaleK,
          input_size /
              kSm87A4W4GateUpDownEdgeM64N256MarlinPhysicalK,
          intermediate_size / kSm87A4W4GateUpDownEdgePhysicalK64,
          work_edges,
          kSm87A4W4GateUpDownEdgeM64N256MarlinPersistentCtas,
          work_edges /
              kSm87A4W4GateUpDownEdgeM64N256MarlinPersistentCtas,
          (work_edges +
           kSm87A4W4GateUpDownEdgeM64N256MarlinPersistentCtas - 1U) /
              kSm87A4W4GateUpDownEdgeM64N256MarlinPersistentCtas};
}

[[nodiscard]] constexpr Sm87A4W4GateUpDownEdgeM64N256MarlinPlan
sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return launch_token_count <=
                     kSm87A4W4GateUpDownEdgeMaximumLaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpDownEdgeModelIntermediate &&
                 input_size == kSm87A4W4GateUpDownEdgeModelInput
             ? sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_test_plan(
                   logical_token_count, launch_token_count,
                   intermediate_size, input_size)
             : Sm87A4W4GateUpDownEdgeM64N256MarlinPlan{};
}

struct Sm87A4W4GateUpDownEdgeM64N256MarlinResources final {
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
query_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_resources_cuda(
    Sm87A4W4GateUpDownEdgeM64N256MarlinResources* resources) noexcept;

#define Q3X_SM87_A4W4_GATEUP_DOWN_M64N256_MARLIN_ARGUMENTS               \
  const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,      \
      const std::uint16_t* a_k512_scales_bf16,                           \
      std::size_t a_scale_capacity_elements,                             \
      const std::uint8_t* packed_gate_b,                                 \
      std::size_t packed_gate_b_capacity_bytes,                          \
      const std::uint16_t* gate_b_k512_scales_bf16,                      \
      std::size_t gate_b_scale_capacity_elements,                        \
      const std::uint8_t* packed_up_b,                                   \
      std::size_t packed_up_b_capacity_bytes,                            \
      const std::uint16_t* up_b_k512_scales_bf16,                        \
      std::size_t up_b_scale_capacity_elements,                          \
      std::size_t logical_token_count, std::size_t launch_token_count,   \
      std::size_t intermediate_size, std::size_t input_size,             \
      float output_clip_ratio, std::uint8_t* packed_output,              \
      std::size_t packed_output_capacity_bytes,                          \
      std::uint16_t* output_k512_scales_bf16,                            \
      std::size_t output_scale_capacity_elements

[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_cuda(
    Q3X_SM87_A4W4_GATEUP_DOWN_M64N256_MARLIN_ARGUMENTS,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_test_cuda(
    Q3X_SM87_A4W4_GATEUP_DOWN_M64N256_MARLIN_ARGUMENTS,
    void* cuda_stream = nullptr) noexcept;

#undef Q3X_SM87_A4W4_GATEUP_DOWN_M64N256_MARLIN_ARGUMENTS

static_assert(kSm87A4W4GateUpDownEdgeM64N256MarlinAStageBytes ==
              2'048U);
static_assert(kSm87A4W4GateUpDownEdgeM64N256MarlinBStageBytes ==
              8'192U);
static_assert(kSm87A4W4GateUpDownEdgeM64N256MarlinStageBytes ==
              18'432U);
static_assert(kSm87A4W4GateUpDownEdgeM64N256MarlinScaleSlotBytes ==
              1'152U);
static_assert(kSm87A4W4GateUpDownEdgeM64N256MarlinPipelineBytes ==
              57'600U);
static_assert(kSm87A4W4GateUpDownEdgeM64N256MarlinEdgePlaneBytes ==
              65'536U);
static_assert(kSm87A4W4GateUpDownEdgeM64N256MarlinAuxScratchBytes ==
              32'768U);
static_assert(
    kSm87A4W4GateUpDownEdgeM64N256MarlinEdgePlaneBytes /
        kSm87A4W4GateUpDownEdgeM64N256MarlinCellsPerEdge ==
    2U * kSm87A4W4GateUpDownEdgeM64N256MarlinTileM *
        (kSm87A4W4GateUpDownEdgeM64N256MarlinWarpTileN /
         kSm87A4W4GateUpDownEdgeM64N256MarlinN8PerWarp) *
        kSm87A4W4GateUpDownEdgeM64N256MarlinWarps * sizeof(std::int32_t));
static_assert(kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes ==
              155'904U);
static_assert(
    kSm87A4W4GateUpDownEdgeM64N256MarlinDynamicSharedBytes <=
    165'376U);
static_assert(
    sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_test_plan(
        117U, 128U, 1'024U, 1'536U)
            .work_edges == 4U);
static_assert(
    sm87_a4w4_gateup_down_k512_edge_m64n256_marlin_plan(
        512U, 512U, 17'408U, 5'120U)
            .launch_ctas == 16U);

}  // namespace q3x::kernels
