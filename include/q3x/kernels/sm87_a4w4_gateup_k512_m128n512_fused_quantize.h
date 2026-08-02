#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Isolated authenticated-v1 structural experiment.  A persistent 512-thread
// CTA owns one M128N512 output quantization cell and visits four M128N128
// Gate+Up subcells.  Each warp owns M16N64 inside a subcell.  Both K256
// halves of one exact K512 group are made visible together; an N8 fragment is
// then accumulated through all eight K64 planes before its one scale FMA.
// Gate and Up therefore share every A publication without retaining a full
// N64 S32 plane.
//
// The first two BF16 product subcells occupy the 64-KiB shared product plane.
// Once the third is rounded to BF16, its 32-KiB packed plane replaces the
// dead upper part of the full-width operand pipeline.  The remaining N128 is
// then visited as two N64 halves with an M16N32 warp state and a reduced
// 66,560-byte pipeline.  Only the first half's eight packed words remain live
// while the second half accumulates.  Warp-local maxima are finally merged
// through storage that aliases the now-dead reduced pipeline, and all four
// subcells publish one exact K512 activation scale and A4 code row.
// Linking this file does not alter runtime dispatch.
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeTileN = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeComputeTileN = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeSubcells = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeWarpM = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeWarpN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeN8FragmentsPerWarp = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeK64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeScaleSlots = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeStageBytes = 49'152U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeScaleSlotBytes = 768U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizePipelineBytes = 99'840U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeProductBytes = 65'536U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes =
        kSm87A4W4GateUpK512M128N512FusedQuantizePipelineBytes +
        kSm87A4W4GateUpK512M128N512FusedQuantizeProductBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizeCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M128N512FusedQuantizePersistentCtas = 16U;

struct Sm87A4W4GateUpK512M128N512FusedQuantizePlan final {
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
};

[[nodiscard]] constexpr Sm87A4W4GateUpK512M128N512FusedQuantizePlan
sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t maximum_launch_ctas =
        kSm87A4W4GateUpK512M128N512FusedQuantizePersistentCtas) noexcept {
  const Sm87A4W4GateUpDownEdgePlan base =
      sm87_a4w4_gateup_down_edge_test_plan(
          logical_token_count, launch_token_count, intermediate_size,
          input_size, maximum_launch_ctas);
  if (base.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      launch_token_count %
              kSm87A4W4GateUpK512M128N512FusedQuantizeTileM !=
          0U ||
      intermediate_size %
              kSm87A4W4GateUpK512M128N512FusedQuantizeTileN !=
          0U) {
    return {};
  }
  const std::size_t m_tiles =
      launch_token_count /
      kSm87A4W4GateUpK512M128N512FusedQuantizeTileM;
  const std::size_t edge_groups =
      intermediate_size /
      kSm87A4W4GateUpK512M128N512FusedQuantizeTileN;
  if (!sm87_a4w4_gateup_down_edge_product_fits(m_tiles,
                                                edge_groups)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * edge_groups;
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          m_tiles,
          edge_groups,
          base.input_k512_groups,
          base.input_physical_k64_groups,
          base.output_physical_k64_groups,
          work_cells,
          work_cells < maximum_launch_ctas ? work_cells
                                            : maximum_launch_ctas};
}

[[nodiscard]] constexpr Sm87A4W4GateUpK512M128N512FusedQuantizePlan
sm87_a4w4_gateup_k512_m128n512_fused_quantize_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return launch_token_count <= kSm87A4W4GateUpDownEdgeMaximumLaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpDownEdgeModelIntermediate &&
                 input_size == kSm87A4W4GateUpDownEdgeModelInput
             ? sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_plan(
                   logical_token_count, launch_token_count,
                   intermediate_size, input_size)
             : Sm87A4W4GateUpK512M128N512FusedQuantizePlan{};
}

struct Sm87A4W4GateUpK512M128N512FusedQuantizeResources final {
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
query_sm87_a4w4_gateup_k512_m128n512_fused_quantize_resources_cuda(
    Sm87A4W4GateUpK512M128N512FusedQuantizeResources* resources)
    noexcept;

#define Q3X_SM87_A4W4_GATEUP_M128N512_FUSED_QUANTIZE_ARGUMENTS             \
    const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,      \
    const std::uint16_t* a_k512_scales_bf16,                               \
    std::size_t a_scale_capacity_elements,                                 \
    const std::uint8_t* packed_gate_b,                                     \
    std::size_t packed_gate_b_capacity_bytes,                              \
    const std::uint16_t* gate_b_k512_scales_bf16,                          \
    std::size_t gate_b_scale_capacity_elements,                            \
    const std::uint8_t* packed_up_b, std::size_t packed_up_b_capacity_bytes, \
    const std::uint16_t* up_b_k512_scales_bf16,                            \
    std::size_t up_b_scale_capacity_elements,                              \
    std::size_t logical_token_count, std::size_t launch_token_count,       \
    std::size_t intermediate_size, std::size_t input_size,                 \
    float output_clip_ratio, std::uint8_t* packed_output,                  \
    std::size_t packed_output_capacity_bytes,                              \
    std::uint16_t* output_k512_scales_bf16,                                \
    std::size_t output_scale_capacity_elements

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_m128n512_fused_quantize_cuda(
    Q3X_SM87_A4W4_GATEUP_M128N512_FUSED_QUANTIZE_ARGUMENTS,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_m128n512_fused_quantize_test_cuda(
    Q3X_SM87_A4W4_GATEUP_M128N512_FUSED_QUANTIZE_ARGUMENTS,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

#undef Q3X_SM87_A4W4_GATEUP_M128N512_FUSED_QUANTIZE_ARGUMENTS

static_assert(
    kSm87A4W4GateUpK512M128N512FusedQuantizeDynamicSharedBytes ==
    165'376U);
static_assert(
    sm87_a4w4_gateup_k512_m128n512_fused_quantize_plan(
        1'853U, 1'920U, 17'408U, 5'120U)
        .launch_ctas == 16U);

}  // namespace q3x::kernels
