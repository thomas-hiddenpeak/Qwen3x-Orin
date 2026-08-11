#include "q3x/kernels/sm87_nvfp4_marlin_p40_parity.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {

using q3x::kernels::Sm87NvFp4MarlinP40ParityLaunchCounters;
using q3x::kernels::Sm87NvFp4MarlinP40ParityResources;
using q3x::kernels::Sm87NvFp4MarlinP40ParityRole;
using q3x::kernels::Sm87NvFp4MarlinP40ParitySegmentKind;
using q3x::kernels::kSm87NvFp4MarlinP40ParityLegacySegmentCount;
using q3x::kernels::kSm87NvFp4MarlinP40ParityLegacySegmentTokens;
using q3x::kernels::kSm87NvFp4MarlinP40ParityActivatedBytes;
using q3x::kernels::kSm87NvFp4MarlinP40ParityDownTailSplitOutputTiles;
using q3x::kernels::kSm87NvFp4MarlinP40ParityGateTailSplitOutputTiles;
using q3x::kernels::kSm87NvFp4MarlinP40ParityGateUpLayout;
using q3x::kernels::kSm87NvFp4MarlinP40ParityGateUpRowStrideBytes;
using q3x::kernels::kSm87NvFp4MarlinP40ParityLockBytes;
using q3x::kernels::kSm87NvFp4MarlinP40ParityMergedGateUpBytes;
using q3x::kernels::kSm87NvFp4MarlinP40ParityReductionBytes;
using q3x::kernels::kSm87NvFp4MarlinP40ParitySegmentCount;
using q3x::kernels::kSm87NvFp4MarlinP40ParityTailTokens;
using q3x::kernels::kSm87NvFp4MarlinP40ParityTokens;
using q3x::kernels::kSm87NvFp4MarlinP40ParityUpColumnByteOffset;
using q3x::kernels::sm87_nvfp4_marlin_p40_parity_plan;
using q3x::kernels::sm87_nvfp4_marlin_p40_parity_segment;

constexpr auto kGatePlan = sm87_nvfp4_marlin_p40_parity_plan(
    Sm87NvFp4MarlinP40ParityRole::kGateUp,
    kSm87NvFp4MarlinP40ParityTokens);
constexpr auto kDownPlan = sm87_nvfp4_marlin_p40_parity_plan(
    Sm87NvFp4MarlinP40ParityRole::kDown,
    kSm87NvFp4MarlinP40ParityTokens);

static_assert(kGatePlan.valid() && kDownPlan.valid());
static_assert(kSm87NvFp4MarlinP40ParityGateUpLayout.valid());
static_assert(kSm87NvFp4MarlinP40ParityGateUpRowStrideBytes == 69'632U &&
              kSm87NvFp4MarlinP40ParityUpColumnByteOffset == 34'816U);
static_assert(kSm87NvFp4MarlinP40ParityMergedGateUpBytes ==
                  2'785'280'000U &&
              kSm87NvFp4MarlinP40ParityActivatedBytes == 1'392'640'000U);
static_assert(kSm87NvFp4MarlinP40ParityReductionBytes == 1'048'576U &&
              kSm87NvFp4MarlinP40ParityLockBytes == 64U);
static_assert(
    kSm87NvFp4MarlinP40ParityGateUpLayout.gate_linear_index(0U, 0U) == 0U &&
    kSm87NvFp4MarlinP40ParityGateUpLayout.up_linear_index(0U, 0U) ==
        17'408U &&
    kSm87NvFp4MarlinP40ParityGateUpLayout.gate_linear_index(1U, 0U) ==
        34'816U &&
    kSm87NvFp4MarlinP40ParityGateUpLayout.up_linear_index(1U, 0U) ==
        52'224U);
static_assert(
    kSm87NvFp4MarlinP40ParityGateUpLayout.gate_linear_index(1U, 0U) !=
    17'408U);
static_assert(
    kSm87NvFp4MarlinP40ParityGateUpLayout
            .up_linear_index(kSm87NvFp4MarlinP40ParityTokens - 1U,
                             17'407U) +
        1U ==
    kSm87NvFp4MarlinP40ParityMergedGateUpBytes / sizeof(std::uint16_t));
static_assert(kGatePlan.input_features == 5'120U &&
              kGatePlan.weight_output_features == 34'816U &&
              kGatePlan.n_tiles == 136U && kGatePlan.k_tiles == 80U);
static_assert(kDownPlan.input_features == 17'408U &&
              kDownPlan.weight_output_features == 5'120U &&
              kDownPlan.n_tiles == 20U && kDownPlan.k_tiles == 272U);
static_assert(kGatePlan.segment_count == 40U &&
              kGatePlan.legacy_full_k_m1024_launches == 39U &&
              kGatePlan.legacy_split_k_m64_launches == 1U &&
              kGatePlan.physical_projection_launches == 40U);
static_assert(kGatePlan.tail_part2_tiles == 8U &&
              kGatePlan.tail_full_k_output_tiles == 128U &&
              kGatePlan.tail_split_k_output_tiles ==
                  kSm87NvFp4MarlinP40ParityGateTailSplitOutputTiles &&
              kGatePlan.tail_split_k_partial_slices == 16U);
static_assert(kDownPlan.tail_part2_tiles == 20U &&
              kDownPlan.tail_full_k_output_tiles == 8U &&
              kDownPlan.tail_split_k_output_tiles ==
                  kSm87NvFp4MarlinP40ParityDownTailSplitOutputTiles &&
              kDownPlan.tail_split_k_partial_slices == 24U);
static_assert(kGatePlan.requires_canonical_gate_then_up &&
              kGatePlan.required_standalone_silu_launches == 1U &&
              kGatePlan.bf16_projection_publication &&
              !kGatePlan.fused_silu && !kGatePlan.fused_residual &&
              kGatePlan.split_k && kGatePlan.uses_locks &&
              kGatePlan.uses_reduction_workspace &&
              kGatePlan.requires_zero_initialized_locks &&
              !kGatePlan.atomic_add && kGatePlan.fp32_reduce &&
              kGatePlan.required_reduction_workspace_bytes ==
                  kSm87NvFp4MarlinP40ParityReductionBytes &&
              kGatePlan.required_lock_bytes ==
                  kSm87NvFp4MarlinP40ParityLockBytes);
static_assert(!kDownPlan.requires_canonical_gate_then_up &&
              kDownPlan.required_standalone_silu_launches == 0U &&
              kDownPlan.bf16_projection_publication &&
              !kDownPlan.fused_residual && kDownPlan.split_k &&
              kDownPlan.uses_locks &&
              kDownPlan.uses_reduction_workspace &&
              !kDownPlan.atomic_add && kDownPlan.fp32_reduce);

constexpr auto kGateFirst =
    sm87_nvfp4_marlin_p40_parity_segment(kGatePlan, 0U);
constexpr auto kGateLastLegacy = sm87_nvfp4_marlin_p40_parity_segment(
    kGatePlan, kSm87NvFp4MarlinP40ParityLegacySegmentCount - 1U);
constexpr auto kGateTail = sm87_nvfp4_marlin_p40_parity_segment(
    kGatePlan, kSm87NvFp4MarlinP40ParityLegacySegmentCount);
constexpr auto kDownFirst =
    sm87_nvfp4_marlin_p40_parity_segment(kDownPlan, 0U);
constexpr auto kDownTail = sm87_nvfp4_marlin_p40_parity_segment(
    kDownPlan, kSm87NvFp4MarlinP40ParityLegacySegmentCount);

static_assert(kGateFirst.valid() && kGateFirst.token_offset == 0U &&
              kGateFirst.token_count == 1'024U &&
              kGateFirst.m_tiles == 16U &&
              kGateFirst.output_tile_count == 2'176U &&
              kGateFirst.physical_tile_slice_count == 2'176U &&
              kGateFirst.full_k_output_tile_count == 2'176U &&
              kGateFirst.split_k_output_tile_count == 0U &&
              kGateFirst.legacy_part2_tiles == 16U &&
              kGateFirst.kind ==
                  Sm87NvFp4MarlinP40ParitySegmentKind::
                      kLegacyStripeFullKM1024 &&
              kGateFirst.all_output_tiles_full_k_owned &&
              !kGateFirst.split_k && !kGateFirst.uses_locks &&
              !kGateFirst.uses_reduction_workspace &&
              !kGateFirst.atomic_add && kGateFirst.fp32_reduce);
static_assert(kGateLastLegacy.valid() &&
              kGateLastLegacy.token_offset == 38U * 1'024U);
static_assert(kGateTail.valid() && kGateTail.token_offset == 39'936U &&
              kGateTail.token_count == 64U && kGateTail.m_tiles == 1U &&
              kGateTail.output_tile_count == 136U &&
              kGateTail.physical_tile_slice_count == 144U &&
              kGateTail.full_k_output_tile_count == 128U &&
              kGateTail.split_k_output_tile_count == 8U &&
              kGateTail.split_k_partial_slice_count == 16U &&
              kGateTail.legacy_part2_tiles == 8U &&
              kGateTail.kind == Sm87NvFp4MarlinP40ParitySegmentKind::
                                    kLegacyStripeSplitKM64 &&
              !kGateTail.all_output_tiles_full_k_owned &&
              kGateTail.split_k && kGateTail.uses_locks &&
              kGateTail.uses_reduction_workspace &&
              !kGateTail.atomic_add && kGateTail.fp32_reduce);
static_assert(kDownFirst.valid() &&
              kDownFirst.output_tile_count == 320U &&
              kDownFirst.legacy_part2_tiles == 16U);
static_assert(kDownTail.valid() &&
              kDownTail.output_tile_count == 20U &&
              kDownTail.physical_tile_slice_count == 32U &&
              kDownTail.full_k_output_tile_count == 8U &&
              kDownTail.split_k_output_tile_count == 12U &&
              kDownTail.split_k_partial_slice_count == 24U &&
              kDownTail.legacy_part2_tiles == 20U &&
              kDownTail.kind == Sm87NvFp4MarlinP40ParitySegmentKind::
                                    kLegacyStripeSplitKM64);
static_assert(!sm87_nvfp4_marlin_p40_parity_plan(
                   Sm87NvFp4MarlinP40ParityRole::kGateUp, 39'999U)
                   .valid());
static_assert(!sm87_nvfp4_marlin_p40_parity_plan(
                   static_cast<Sm87NvFp4MarlinP40ParityRole>(0xffU),
                   kSm87NvFp4MarlinP40ParityTokens)
                   .valid());
static_assert(!sm87_nvfp4_marlin_p40_parity_segment(
                   kGatePlan, kSm87NvFp4MarlinP40ParitySegmentCount)
                   .valid());
constexpr Sm87NvFp4MarlinP40ParityLaunchCounters kCompleteGateCounters{
    kSm87NvFp4MarlinP40ParityLegacySegmentCount,
    1U,
    kSm87NvFp4MarlinP40ParitySegmentCount,
    kSm87NvFp4MarlinP40ParityGateTailSplitOutputTiles,
    2U * kSm87NvFp4MarlinP40ParityGateTailSplitOutputTiles,
    true};
static_assert(kCompleteGateCounters.matches(kGatePlan));

using GateUpPreparer = int (*)(
    const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
    const std::uint8_t*, const float*, float, std::uint8_t*, std::uint8_t*,
    float*, void*, std::size_t, void*) noexcept;
using GateLauncher = int (*)(
    const std::uint16_t*, const std::uint8_t*, const std::uint8_t*,
    const float*, std::uint16_t*, float*, std::size_t, std::int32_t*,
    std::size_t, Sm87NvFp4MarlinP40ParityLaunchCounters*, void*) noexcept;
using SiluLauncher = int (*)(const std::uint16_t*, std::uint16_t*,
                            void*) noexcept;
using DownLauncher = GateLauncher;
using ResourceQuery = int (*)(Sm87NvFp4MarlinP40ParityRole,
                              Sm87NvFp4MarlinP40ParityResources*) noexcept;

static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           prepare_sm87_nvfp4_marlin_p40_parity_gate_up_cuda),
              GateUpPreparer>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_nvfp4_marlin_p40_parity_gate_up_cuda),
              GateLauncher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_nvfp4_marlin_p40_parity_silu_cuda),
              SiluLauncher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_nvfp4_marlin_p40_parity_down_cuda),
              DownLauncher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           query_sm87_nvfp4_marlin_p40_parity_resources_cuda),
              ResourceQuery>);

[[nodiscard]] bool check(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

[[nodiscard]] bool complete_contiguous_schedule(
    const q3x::kernels::Sm87NvFp4MarlinP40ParityPlan& plan) {
  std::size_t cursor = 0U;
  std::size_t full_k_bulk = 0U;
  std::size_t split_k_tail = 0U;
  for (std::size_t index = 0U; index < plan.segment_count; ++index) {
    const auto segment =
        sm87_nvfp4_marlin_p40_parity_segment(plan, index);
    if (!segment.valid() || segment.token_offset != cursor ||
        segment.atomic_add || !segment.fp32_reduce) {
      return false;
    }
    cursor += segment.token_count;
    if (segment.kind ==
        Sm87NvFp4MarlinP40ParitySegmentKind::kLegacyStripeFullKM1024) {
      if (!segment.all_output_tiles_full_k_owned || segment.split_k ||
          segment.uses_locks || segment.uses_reduction_workspace) {
        return false;
      }
      ++full_k_bulk;
    } else if (segment.kind == Sm87NvFp4MarlinP40ParitySegmentKind::
                                   kLegacyStripeSplitKM64) {
      if (segment.all_output_tiles_full_k_owned || !segment.split_k ||
          !segment.uses_locks || !segment.uses_reduction_workspace ||
          segment.split_k_output_tile_count !=
              plan.tail_split_k_output_tiles ||
          segment.split_k_partial_slice_count !=
              plan.tail_split_k_partial_slices) {
        return false;
      }
      ++split_k_tail;
    } else {
      return false;
    }
  }
  return cursor == plan.token_count &&
         full_k_bulk == kSm87NvFp4MarlinP40ParityLegacySegmentCount &&
         split_k_tail == 1U;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= check(complete_contiguous_schedule(kGatePlan),
              "GateUp P40 parity schedule must preserve stock split-K tail");
  ok &= check(complete_contiguous_schedule(kDownPlan),
              "Down P40 parity schedule must preserve stock split-K tail");
  ok &= check(kSm87NvFp4MarlinP40ParityLegacySegmentCount *
                      kSm87NvFp4MarlinP40ParityLegacySegmentTokens +
                  kSm87NvFp4MarlinP40ParityTailTokens ==
              kSm87NvFp4MarlinP40ParityTokens,
              "39xM1024 plus M64 must cover P40000 exactly");
  return ok ? 0 : 1;
}
