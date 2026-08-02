#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off one-CTA/SM Gate+Up experiment over the existing v2 paired
// payload.  It is intentionally disconnected from production dispatch.
//
// A CTA owns M128 x N64.  Warps 0..7 own the upper M64 and warps 8..15
// own the lower M64.  Warp pairs share one N8 column strip:
//
//   warp_n8 = warp % 8; m_half = warp / 8.
//
// A two-stage K256 ring stages 16 KiB of A and 16 KiB of paired Gate+Up B
// per stage.  Each v2 B record is loaded once and reused by the upper/lower
// warp pair, removing the direct global-load/use edge.  Every warp retains
// Gate and Up S32/FP32 fragments for four M16 panels and performs same-warp
// SwiGLU, so no shared epilogue exchange is required.
//
// Two 512-byte scale slots bring total dynamic shared storage to 66,560 B.
// launch_bounds pins 512 threads and one CTA/SM.  Admission requires at most
// 128 registers/thread and zero local storage; 120 registers is the design
// target rather than a relaxed correctness threshold.
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedWarpM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedWarpN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedMmaN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedPhysicalK = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedCopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedK64PerCopy = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedAStageBytes =
        kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileM *
        kSm87A4W4GateUpK512FragmentNativeM128N64StagedCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedBStageBytes =
        kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileN *
        kSm87A4W4GateUpK512FragmentNativeM128N64StagedCopyK;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedStageBytes =
        kSm87A4W4GateUpK512FragmentNativeM128N64StagedAStageBytes +
        kSm87A4W4GateUpK512FragmentNativeM128N64StagedBStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedScaleSlotBytes = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedSharedBytes =
        kSm87A4W4GateUpK512FragmentNativeM128N64StagedStages *
            (kSm87A4W4GateUpK512FragmentNativeM128N64StagedStageBytes +
             kSm87A4W4GateUpK512FragmentNativeM128N64StagedScaleSlotBytes);
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedTargetRegisters = 120U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedBarriersPerK512 = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedEpilogueBarriers = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedPersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedMinimumTokens = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedMaximumTokens = 4'096U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedModelIntermediate =
        17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedWindow0Start = 0U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedWindow0Count = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedWindow1Start = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedWindow1Count = 5'120U;

static_assert(
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedAStageBytes ==
    16'384U);
static_assert(
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedBStageBytes ==
    16'384U);
static_assert(
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedStageBytes ==
    32'768U);
static_assert(
    kSm87A4W4GateUpK512FragmentNativeM128N64StagedSharedBytes ==
    66'560U);

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_barriers_per_cell(
    const std::size_t k512_groups) noexcept {
  return k512_groups == 0U
             ? 0U
             : k512_groups *
                       kSm87A4W4GateUpK512FragmentNativeM128N64StagedBarriersPerK512 +
                   kSm87A4W4GateUpK512FragmentNativeM128N64StagedEpilogueBarriers;
}

static_assert(
    sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_barriers_per_cell(
        10U) == 41U);

struct Sm87A4W4GateUpK512FragmentNativeM128N64StagedPlan final {
  std::size_t token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t n_start{};
  std::size_t n_count{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
  std::size_t minimum_cells_per_cta{};
  std::size_t maximum_cells_per_cta{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpK512FragmentNativeM128N64StagedPlan
sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count) noexcept {
  if (token_count <
          kSm87A4W4GateUpK512FragmentNativeM128N64StagedMinimumTokens ||
      token_count >
          kSm87A4W4GateUpK512FragmentNativeM128N64StagedMaximumTokens ||
      token_count %
              kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileM !=
          0U ||
      intermediate_size == 0U || intermediate_size % 64U != 0U ||
      input_size == 0U ||
      input_size %
              kSm87A4W4GateUpK512FragmentNativeM128N64StagedScaleK !=
          0U ||
      n_start %
              kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileN !=
          0U ||
      n_count == 0U ||
      n_count %
              kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileN !=
          0U ||
      n_start > intermediate_size ||
      n_count > intermediate_size - n_start) {
    return {};
  }
  const std::size_t m_tiles =
      token_count /
      kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileM;
  const std::size_t n_tiles =
      n_count /
      kSm87A4W4GateUpK512FragmentNativeM128N64StagedTileN;
  if (!sm87_a4w4_gateup_k512_fragment_native_product_fits(
          m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      m_tiles <
              kSm87A4W4GateUpK512FragmentNativeM128N64StagedPersistentCtas
          ? m_tiles
          : kSm87A4W4GateUpK512FragmentNativeM128N64StagedPersistentCtas;
  const std::size_t minimum_m_tiles = m_tiles / launch_ctas;
  const std::size_t maximum_m_tiles =
      minimum_m_tiles + (m_tiles % launch_ctas != 0U ? 1U : 0U);
  return {token_count,
          intermediate_size,
          input_size,
          n_start,
          n_count,
          m_tiles,
          n_tiles,
          input_size /
              kSm87A4W4GateUpK512FragmentNativeM128N64StagedScaleK,
          input_size /
              kSm87A4W4GateUpK512FragmentNativeM128N64StagedPhysicalK,
          work_cells,
          launch_ctas,
          minimum_m_tiles * n_tiles,
          maximum_m_tiles * n_tiles};
}

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_is_model_plan(
    const Sm87A4W4GateUpK512FragmentNativeM128N64StagedPlan& plan) noexcept {
  const bool model_window =
      (plan.n_start ==
           kSm87A4W4GateUpK512FragmentNativeM128N64StagedWindow0Start &&
       plan.n_count ==
           kSm87A4W4GateUpK512FragmentNativeM128N64StagedWindow0Count) ||
      (plan.n_start ==
           kSm87A4W4GateUpK512FragmentNativeM128N64StagedWindow1Start &&
       plan.n_count ==
           kSm87A4W4GateUpK512FragmentNativeM128N64StagedWindow1Count);
  return plan.launch_ctas != 0U &&
         plan.intermediate_size ==
             kSm87A4W4GateUpK512FragmentNativeM128N64StagedModelIntermediate &&
         plan.input_size ==
             kSm87A4W4GateUpK512FragmentNativeM128N64StagedModelInput &&
         model_window;
}

struct Sm87A4W4GateUpK512FragmentNativeM128N64StagedResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int
query_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_resources_cuda(
    Sm87A4W4GateUpK512FragmentNativeM128N64StagedResources* resources) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* paired_b_codes,
    std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* paired_b_scales_bf16,
    std::size_t paired_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* paired_b_codes,
    std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* paired_b_scales_bf16,
    std::size_t paired_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
