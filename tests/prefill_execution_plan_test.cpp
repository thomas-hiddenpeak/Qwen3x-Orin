#include "q3x/runtime/prefill_execution_plan.h"
#include "q3x/runtime/request_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {

namespace model = q3x::model;
namespace runtime = q3x::runtime;

static_assert(noexcept(
    runtime::is_valid_unbound_layer_major_prefill_execution_plan(
        std::declval<const runtime::PrefillExecutionPlan&>())));
static_assert(static_cast<std::uint8_t>(
                  runtime::LayerMajorPrefillProjectionTactic::
                      kNativePromptWideP40PackedProjection) == 8U);
static_assert(static_cast<std::uint8_t>(
                  runtime::LayerMajorPrefillProjectionTactic::
                      kNativePromptWideP40PackedNvfp4V2) == 9U);
static_assert(static_cast<std::uint8_t>(
                  runtime::LayerMajorPrefillProjectionTactic::
                      kNativePromptWideP40VllmMarlinParity) == 10U);
static_assert(static_cast<std::uint8_t>(
                  runtime::LayerMajorPrefillProjectionTactic::
                      kNativePromptWideP40MacroFeedV3) == 11U);
static_assert(static_cast<std::uint8_t>(
                  runtime::LayerMajorPrefillMlpScheduleTactic::
                      kPromptWideP40PackedProjection) == 4U);
static_assert(static_cast<std::uint8_t>(
                  runtime::LayerMajorPrefillMlpScheduleTactic::
                      kPromptWideP40PackedNvfp4V2) == 5U);
static_assert(static_cast<std::uint8_t>(
                  runtime::LayerMajorPrefillMlpScheduleTactic::
                      kPromptWideP40VllmMarlinParity) == 6U);
static_assert(static_cast<std::uint8_t>(
                  runtime::LayerMajorPrefillMlpScheduleTactic::
                      kPromptWideP40MacroFeedV3) == 7U);
static_assert(static_cast<std::uint8_t>(
                  runtime::PrefillNvFp4ArithmeticTactic::
                      kP40000VllmMarlinProjectionHostDispatchGateThenUpSiluDownResidual) ==
              7U);

class TestContext {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] runtime::PrefillExecutionPlanResult build_plan(
    const std::uint64_t prompt_token_count,
    const std::uint64_t first_position = 0U,
    const std::uint64_t max_sequence_length =
        runtime::kLayerMajorPrefillMaximumSequenceTokens,
    const runtime::LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic =
        runtime::LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel) {
  runtime::PrefillExecutionPlanOptions options;
  options.first_position = first_position;
  options.prompt_token_count = prompt_token_count;
  options.max_sequence_length = max_sequence_length;
  options.mlp_schedule_tactic = mlp_schedule_tactic;
  return runtime::build_unbound_layer_major_prefill_execution_plan(options);
}

void expect_panel_shape(TestContext& test,
                        const std::uint64_t prompt_token_count,
                        const std::size_t expected_panel_count,
                        const std::uint32_t expected_tail_tokens) {
  const runtime::PrefillExecutionPlanResult result =
      build_plan(prompt_token_count);
  const bool valid = result &&
                     result.value->panel_count == expected_panel_count &&
                     result.value->panels[expected_panel_count - 1U]
                             .token_count == expected_tail_tokens &&
                     result.value->panels[expected_panel_count - 1U]
                             .end_position == prompt_token_count;
  if (!valid) {
    std::cerr << "  panel shape mismatch: prompt=" << prompt_token_count
              << " expected_count=" << expected_panel_count
              << " expected_tail=" << expected_tail_tokens << '\n';
  }
  test.expect(valid, "prompt decomposes into the expected C8192 panels");
}

[[nodiscard]] bool same_progress(
    const runtime::PrefillExecutionProgress& left,
    const runtime::PrefillExecutionProgress& right) {
  return left.kv_visible_end == right.kv_visible_end &&
         left.gdn_advanced_end == right.gdn_advanced_end &&
         left.completed_panels == right.completed_panels &&
         left.completed_mlp_phases == right.completed_mlp_phases &&
         left.completed_fill_panels == right.completed_fill_panels &&
         left.completed_prompt_core_phases ==
             right.completed_prompt_core_phases &&
         left.completed_drain_panels == right.completed_drain_panels &&
         left.next_layer == right.next_layer &&
         left.next_panel == right.next_panel &&
         left.final_hidden_ready == right.final_hidden_ready &&
         left.prefill_state_committed == right.prefill_state_committed;
}

void test_public_tile_and_operator_panel_are_independent(TestContext& test) {
  test.expect(
      runtime::is_valid_layer_major_prefill_full_attention_tactic(
          runtime::LayerMajorPrefillFullAttentionTactic::
              kExactSegmentedC512) &&
          runtime::is_valid_layer_major_prefill_full_attention_tactic(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ64Panel) &&
          runtime::is_valid_layer_major_prefill_full_attention_tactic(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ128V4Panel) &&
          runtime::is_valid_layer_major_prefill_full_attention_tactic(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeFlashInferExactPanel) &&
          runtime::is_valid_layer_major_prefill_full_attention_tactic(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeFlashInferExactWholePrompt) &&
          !runtime::is_valid_layer_major_prefill_full_attention_tactic(
              static_cast<runtime::LayerMajorPrefillFullAttentionTactic>(
                  0xffU)),
      "layer-major Attention tactics are a closed engine-lifetime set");
  test.expect(
      runtime::to_string(
          runtime::LayerMajorPrefillFullAttentionTactic::
              kExactSegmentedC512) == "exact-segmented" &&
          runtime::to_string(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ64Panel) == "native-group-q64-panel" &&
          runtime::to_string(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ128V4Panel) ==
              "native-group-q128-v4-panel" &&
          runtime::to_string(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeFlashInferExactPanel) ==
              "native-flashinfer-exact-panel" &&
          runtime::to_string(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeFlashInferExactWholePrompt) ==
              "native-flashinfer-exact-whole-prompt",
      "Attention tactic names preserve exact Q64/Q128/FlashInfer route "
      "identity");
  test.expect(
      runtime::is_valid_layer_major_prefill_projection_tactic(
          runtime::LayerMajorPrefillProjectionTactic::kExactSegmentedC512) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kSegmentedMarlinOperatorPanel) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativeQuantizedLargeMOperatorPanel) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativeNvfp4TrueLargeMOperatorPanel) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativeNvfp4G2D2LargeMOperatorPanel) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativeNvfp4PersistentP40LayerWideMlp) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40WholeCore) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40ProjectionReset) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40PackedProjection) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40PackedNvfp4V2) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40VllmMarlinParity) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40MacroFeedV3) &&
          !runtime::is_valid_layer_major_prefill_projection_tactic(
              static_cast<runtime::LayerMajorPrefillProjectionTactic>(
                  0xffU)),
      "layer-major projection tactics are a closed engine-lifetime set");
  test.expect(
      runtime::to_string(
          runtime::LayerMajorPrefillProjectionTactic::
              kExactSegmentedC512) == "exact-segmented" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kSegmentedMarlinOperatorPanel) ==
              "segmented-marlin-operator-panel" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativeQuantizedLargeMOperatorPanel) ==
              "native-quantized-large-m-operator-panel" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativeNvfp4TrueLargeMOperatorPanel) ==
              "native-nvfp4-true-large-m-operator-panel" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativeNvfp4G2D2LargeMOperatorPanel) ==
              "native-nvfp4-g2-d2-large-m-operator-panel" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativeNvfp4PersistentP40LayerWideMlp) ==
              "native-nvfp4-persistent-p40-layer-wide-mlp" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40WholeCore) ==
              "native-prompt-wide-p40-whole-core" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40ProjectionReset) ==
              "native-prompt-wide-p40-projection-reset" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40PackedProjection) ==
              "native-prompt-wide-p40-packed-projection" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40PackedNvfp4V2) ==
              "native-prompt-wide-p40-packed-nvfp4-v2" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40VllmMarlinParity) ==
              "native-prompt-wide-p40-vllm-marlin-parity" &&
          runtime::to_string(
              runtime::LayerMajorPrefillProjectionTactic::
                  kNativePromptWideP40MacroFeedV3) ==
              "native-prompt-wide-p40-macrofeed-v3",
      "projection tactic names preserve exact, segmented, native large-M, "
      "true-large-M, and G2/D2 NVFP4 route identity");
  const runtime::PrefillExecutionPlanResult result = build_plan(513U);
  test.expect(result &&
                  result.value->legacy_public_tile_limit == 512U &&
                  result.value->operator_panel_capacity == 8'192U &&
                  result.value->panel_count == 1U &&
                  result.value->panels[0].token_count == 513U &&
                  !result.value->operator_bindings_complete &&
                  !result.value->executable(),
              "P513 is one unbound operator panel and never inherits C512 "
              "execution semantics");

  test.expect(runtime::kMaximumRequestPrefillChunkSize == 512U &&
                  runtime::kLayerMajorPrefillLegacyPublicTileTokens == 512U &&
                  runtime::kAbsoluteRequestMaxSequenceLength ==
                      runtime::kLayerMajorPrefillMaximumSequenceTokens &&
                  runtime::kRequestLayerCount ==
                      runtime::kLayerMajorPrefillLayerCount &&
                  runtime::kLayerMajorPrefillOperatorPanelTokens == 8'192U &&
                  runtime::kLayerMajorPrefillMaximumPanelCount == 32U,
              "legacy tile and layer-major panel constants remain distinct");
}

void test_prompt_wide_p40_whole_core_schedule(TestContext& test) {
  using MlpSchedule = runtime::LayerMajorPrefillMlpScheduleTactic;
  constexpr auto kWholeCore = MlpSchedule::kPromptWideP40WholeCore;
  test.expect(
      runtime::is_valid_layer_major_prefill_mlp_schedule_tactic(kWholeCore) &&
          runtime::to_string(kWholeCore) ==
              "prompt-wide-p40-whole-core" &&
          runtime::prefill_route_layer_pass_count(5U, kWholeCore) == 1U &&
          runtime::kLayerMajorPrefillPromptWideP40PanelCount == 5U &&
          runtime::kLayerMajorPrefillPromptWideP40PanelTokens == 8'000U &&
          runtime::kLayerMajorPrefillPromptWideP40PanelTokens % 64U == 0U &&
          runtime::kLayerMajorPrefillPromptWideP40PanelTokens <=
              runtime::kLayerMajorPrefillOperatorPanelTokens &&
          runtime::kLayerMajorPrefillPromptWideP40PanelCount *
                  runtime::kLayerMajorPrefillPromptWideP40PanelTokens ==
              runtime::kLayerMajorPrefillPromptWideP40Tokens &&
          runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens ==
              40'001U,
      "whole-core identity locks exact P40000/P40001 and five aligned M8000 "
      "panels");

  const auto candidate = build_plan(
      runtime::kLayerMajorPrefillPromptWideP40Tokens, 0U,
      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
      kWholeCore);
  if (!runtime::prompt_wide_p40_whole_core_prefill_plan_enabled()) {
    test.expect(
        !candidate &&
            candidate.error ==
                runtime::PrefillExecutionPlanError::kInvalidArgument,
        "default build fails closed on the independent whole-core tactic");
    return;
  }

  bool panels_exact = candidate && candidate.value->panel_count == 5U;
  for (std::size_t panel = 0U; panels_exact && panel < 5U; ++panel) {
    const runtime::PrefillOperatorPanel& geometry =
        candidate.value->panels[panel];
    panels_exact = geometry.ordinal == panel &&
                   geometry.first_position == panel * 8'000U &&
                   geometry.token_count == 8'000U &&
                   geometry.end_position == (panel + 1U) * 8'000U;
  }
  const runtime::PrefillWholeCoreSchedulePlan* schedule =
      candidate ? &candidate.value->whole_core_schedule : nullptr;
  test.expect(
      panels_exact && schedule != nullptr && schedule->enabled &&
          schedule->fill_panel_phase_count_per_layer == 5U &&
          schedule->prompt_core_phase_count_per_layer == 1U &&
          schedule->drain_panel_phase_count_per_layer == 5U &&
          schedule->persistent_mlp_phase_count_per_layer == 1U &&
          schedule->panel_token_count == 8'000U &&
          schedule->prompt_core_token_count == 40'000U &&
          schedule->request_capacity_tokens == 40'001U &&
          schedule->route_pass_count == 1U &&
          schedule->fp8_single_launch_per_projection_required &&
          schedule->bf16_ab_prompt_wide_required &&
          schedule->gdn_prompt_wide_required &&
          schedule->flashinfer_whole_prompt_required &&
          candidate.value->mlp_schedule.maximum_standalone_silu_launches_per_layer ==
              0U &&
          candidate.value->mlp_schedule.minimum_total_kernel_launches_per_layer ==
              2U &&
          candidate.value->mlp_schedule.maximum_total_kernel_launches_per_layer ==
              2U &&
          runtime::is_valid_unbound_layer_major_prefill_execution_plan(
              *candidate.value),
      "whole-core plan is exactly 5 fill, 1 core, 5 drain, and 1 persistent "
      "MLP per layer");

  test.expect(
      !build_plan(runtime::kLayerMajorPrefillPromptWideP40Tokens, 0U,
                  runtime::kLayerMajorPrefillMaximumSequenceTokens,
                  kWholeCore) &&
          !build_plan(39'936U, 0U,
                      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
                      kWholeCore) &&
          !build_plan(runtime::kLayerMajorPrefillPromptWideP40Tokens, 64U,
                      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
                      kWholeCore),
      "whole-core refuses a generalized capacity, prompt length, or warm "
      "position");

  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(*candidate.value);
  test.expect(
      runtime::advance_prompt_wide_p40_prompt_core_progress_after_completion(
          *candidate.value, progress, 0U) ==
          runtime::PrefillExecutionProgressError::kOutOfOrder,
      "whole-core cannot publish its core before all five fill panels");
  bool ordered = true;
  for (std::size_t layer = 0U;
       ordered && layer < candidate.value->layers.size(); ++layer) {
    for (std::size_t panel = 0U; ordered && panel < 5U; ++panel) {
      ordered =
          runtime::advance_prompt_wide_p40_fill_progress_after_completion(
              *candidate.value, progress, layer, panel) ==
          runtime::PrefillExecutionProgressError::kNone;
    }
    ordered = ordered &&
              runtime::advance_prompt_wide_p40_prompt_core_progress_after_completion(
                  *candidate.value, progress, layer) ==
                  runtime::PrefillExecutionProgressError::kNone;
    for (std::size_t panel = 0U; ordered && panel < 5U; ++panel) {
      ordered =
          runtime::advance_prompt_wide_p40_drain_progress_after_completion(
              *candidate.value, progress, layer, panel) ==
          runtime::PrefillExecutionProgressError::kNone;
    }
    ordered = ordered &&
              runtime::advance_prompt_wide_p40_persistent_mlp_progress_after_completion(
                  *candidate.value, progress, layer) ==
                  runtime::PrefillExecutionProgressError::kNone;
  }
  test.expect(
      ordered && progress.next_layer == candidate.value->layers.size() &&
          runtime::mark_prefill_final_hidden_ready(*candidate.value,
                                                   progress) ==
              runtime::PrefillExecutionProgressError::kNone &&
          runtime::prefill_final_commit_ready(*candidate.value, progress),
      "whole-core progress reaches final commit only after every ordered "
      "phase in all 64 layers");

  runtime::PrefillExecutionPlan mutated = *candidate.value;
  mutated.panels[1U].token_count = 8'192U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "whole-core authority rejects incumbent 8192/7712 geometry relabeling");
}

void test_prompt_wide_p40_macrofeed_v3_schedule(TestContext& test) {
  using MlpSchedule = runtime::LayerMajorPrefillMlpScheduleTactic;
  using Projection = runtime::LayerMajorPrefillProjectionTactic;
  constexpr auto kMacroFeedV3 = MlpSchedule::kPromptWideP40MacroFeedV3;
  constexpr auto kMacroFeedV3Projection =
      Projection::kNativePromptWideP40MacroFeedV3;
  test.expect(
      runtime::is_valid_layer_major_prefill_projection_tactic(
          kMacroFeedV3Projection) &&
          runtime::to_string(kMacroFeedV3Projection) ==
              "native-prompt-wide-p40-macrofeed-v3" &&
          runtime::is_valid_layer_major_prefill_mlp_schedule_tactic(
              kMacroFeedV3) &&
          runtime::to_string(kMacroFeedV3) ==
              "prompt-wide-p40-macrofeed-v3" &&
          runtime::prefill_route_layer_pass_count(5U, kMacroFeedV3) == 1U &&
          runtime::kLayerMajorPrefillMacroFeedV3GateUpPhysicalLaunchesPerRequest ==
              64U &&
          runtime::kLayerMajorPrefillMacroFeedV3DownPhysicalLaunchesPerRequest ==
              64U &&
          runtime::kLayerMajorPrefillMacroFeedV3GdnLayerCount == 48U &&
          runtime::kLayerMajorPrefillMacroFeedV3GdnPhysicalLaunchesPerLayer ==
              9U &&
          runtime::kLayerMajorPrefillMacroFeedV3GdnPhysicalLaunchesPerRequest ==
              432U &&
          runtime::kLayerMajorPrefillMacroFeedV3TrackedPhysicalLaunchesPerRequest ==
              560U,
      "MacroFeed-v3 has append-only projection/MLP identities and exact "
      "GateUp64, Down64, and GDN48x9 physical schedule counts");

  const auto candidate = build_plan(
      runtime::kLayerMajorPrefillPromptWideP40Tokens, 0U,
      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
      kMacroFeedV3);
  if (!runtime::prompt_wide_p40_macrofeed_v3_prefill_plan_enabled()) {
    test.expect(
        !candidate &&
            candidate.error ==
                runtime::PrefillExecutionPlanError::kInvalidArgument,
        "default builds fail closed on the MacroFeed-v3 overlay");
    return;
  }

  const runtime::PrefillExecutionPlan* plan =
      candidate ? &*candidate.value : nullptr;
  const runtime::PrefillWholeCoreSchedulePlan* whole_core =
      plan != nullptr ? &plan->whole_core_schedule : nullptr;
  const runtime::PrefillP40MacroFeedV3SchedulePlan* schedule =
      plan != nullptr ? &plan->macrofeed_v3_schedule : nullptr;
  bool panels_exact = plan != nullptr && plan->panel_count == 5U;
  for (std::size_t panel = 0U; panels_exact && panel < 5U; ++panel) {
    const runtime::PrefillOperatorPanel& geometry = plan->panels[panel];
    panels_exact = geometry.ordinal == panel &&
                   geometry.first_position == panel * 8'000U &&
                   geometry.token_count == 8'000U &&
                   geometry.end_position == (panel + 1U) * 8'000U;
  }
  test.expect(
      panels_exact && whole_core != nullptr && whole_core->enabled &&
          whole_core->fill_panel_phase_count_per_layer == 5U &&
          whole_core->prompt_core_phase_count_per_layer == 1U &&
          whole_core->drain_panel_phase_count_per_layer == 5U &&
          whole_core->persistent_mlp_phase_count_per_layer == 1U &&
          whole_core->panel_token_count == 8'000U &&
          whole_core->prompt_core_token_count == 40'000U &&
          whole_core->request_capacity_tokens == 40'001U &&
          schedule != nullptr && schedule->enabled &&
          schedule->request_memory_profile ==
              runtime::RequestMemoryProfile::kLayerMajorP40WholeCore &&
          schedule->gate_up_physical_launches_per_request == 64U &&
          schedule->down_physical_launches_per_request == 64U &&
          schedule->gdn_layer_count == 48U &&
          schedule->gdn_physical_launches_per_layer == 9U &&
          schedule->gdn_physical_launches_per_request == 432U &&
          schedule->tracked_physical_launches_per_request == 560U &&
          schedule->v10_whole_core_topology_required &&
          schedule->v10_request_memory_profile_required &&
          schedule->role_specific_noncooperative_macro_projections_required &&
          schedule->exact_per_token_bf16_gdn_state_required &&
          schedule->full_model_physical_receipt_required &&
          schedule->production_accuracy_required &&
          schedule->approximate_numerics_forbidden &&
          schedule->fallback_forbidden && schedule->mtp_forbidden &&
          schedule->cublaslt_forbidden &&
          schedule->request_time_jit_forbidden &&
          schedule->request_time_repack_forbidden &&
          schedule->development_only &&
          schedule->production_dispatch_forbidden &&
          !plan->projection_reset_schedule.enabled &&
          !plan->packed_projection_schedule.enabled &&
          !plan->packed_nvfp4_v2_schedule.enabled &&
          !plan->vllm_marlin_parity_schedule.enabled &&
          plan->mlp_schedule.required_gate_up_projection_launches_per_layer ==
              1U &&
          plan->mlp_schedule.required_down_projection_launches_per_layer ==
              1U &&
          plan->mlp_schedule.minimum_total_kernel_launches_per_layer == 2U &&
          plan->mlp_schedule.maximum_total_kernel_launches_per_layer == 2U &&
          !plan->operator_bindings_complete && !plan->executable() &&
          runtime::is_valid_unbound_layer_major_prefill_execution_plan(*plan),
      "MacroFeed-v3 reuses the exact v10 five-panel/P40 memory substrate "
      "while sealing a distinct default-off physical overlay");

  test.expect(
      !build_plan(runtime::kLayerMajorPrefillPromptWideP40Tokens, 0U,
                  runtime::kLayerMajorPrefillMaximumSequenceTokens,
                  kMacroFeedV3) &&
          !build_plan(39'936U, 0U,
                      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
                      kMacroFeedV3) &&
          !build_plan(runtime::kLayerMajorPrefillPromptWideP40Tokens, 64U,
                      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
                      kMacroFeedV3),
      "MacroFeed-v3 refuses generalized capacity, M, and warm-position "
      "relabeling");

  runtime::PrefillExecutionPlan mutated = *plan;
  mutated.macrofeed_v3_schedule.request_memory_profile =
      runtime::RequestMemoryProfile::kSm87BulkV2P40Owner;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 cannot acquire the retired V2 request owner");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.gate_up_physical_launches_per_request = 63U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 rejects one missing Gate+Up physical launch");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.down_physical_launches_per_request = 63U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 rejects one missing Down physical launch");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.gdn_layer_count = 47U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 rejects incomplete GDN layer coverage");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.gdn_physical_launches_per_layer = 8U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 rejects an incomplete nine-kernel GDN layer graph");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.fallback_forbidden = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 cannot weaken the no-fallback boundary");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.mtp_forbidden = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 cannot weaken the non-MTP boundary");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.cublaslt_forbidden = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 cannot acquire a cuBLASLt path");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.request_time_jit_forbidden = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 cannot acquire request-time JIT");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.request_time_repack_forbidden = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 cannot acquire request-time repack");
  mutated = *plan;
  mutated.macrofeed_v3_schedule.development_only = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "the unbound MacroFeed-v3 slice cannot claim release status");
  mutated = *plan;
  mutated.whole_core_schedule.enabled = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "MacroFeed-v3 cannot detach from the retained whole-core topology");
  mutated = *plan;
  mutated.macrofeed_v3_schedule = {};
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "the v10 topology alone cannot be relabeled as MacroFeed-v3");

  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(*plan);
  test.expect(
      runtime::advance_prefill_progress_after_completion(
          *plan, progress, 0U, 0U) ==
          runtime::PrefillExecutionProgressError::kOutOfOrder,
      "MacroFeed-v3 cannot bypass its retained whole-core phase order");
  bool ordered = true;
  for (std::size_t layer = 0U; ordered && layer < plan->layers.size();
       ++layer) {
    for (std::size_t panel = 0U; ordered && panel < 5U; ++panel) {
      ordered = runtime::advance_prompt_wide_p40_fill_progress_after_completion(
                    *plan, progress, layer, panel) ==
                runtime::PrefillExecutionProgressError::kNone;
    }
    ordered = ordered &&
              runtime::advance_prompt_wide_p40_prompt_core_progress_after_completion(
                  *plan, progress, layer) ==
                  runtime::PrefillExecutionProgressError::kNone;
    for (std::size_t panel = 0U; ordered && panel < 5U; ++panel) {
      ordered =
          runtime::advance_prompt_wide_p40_drain_progress_after_completion(
              *plan, progress, layer, panel) ==
          runtime::PrefillExecutionProgressError::kNone;
    }
    ordered = ordered &&
              runtime::advance_prompt_wide_p40_persistent_mlp_progress_after_completion(
                  *plan, progress, layer) ==
                  runtime::PrefillExecutionProgressError::kNone;
  }
  test.expect(
      ordered && progress.next_layer == plan->layers.size() &&
          runtime::mark_prefill_final_hidden_ready(*plan, progress) ==
              runtime::PrefillExecutionProgressError::kNone &&
          runtime::prefill_final_commit_ready(*plan, progress),
      "MacroFeed-v3 retains v10 ordered progress and one final state commit");
}

void test_prompt_wide_p40_projection_reset_schedule(TestContext& test) {
  using MlpSchedule = runtime::LayerMajorPrefillMlpScheduleTactic;
  constexpr auto kProjectionReset =
      MlpSchedule::kPromptWideP40ProjectionReset;
  test.expect(
      runtime::is_valid_layer_major_prefill_mlp_schedule_tactic(
          kProjectionReset) &&
          runtime::to_string(kProjectionReset) ==
              "prompt-wide-p40-projection-reset" &&
          runtime::prefill_route_layer_pass_count(5U, kProjectionReset) ==
              1U &&
          runtime::kLayerMajorPrefillProjectionResetFp8GroupedInputLaunchesPerLayer ==
              1U &&
          runtime::kLayerMajorPrefillProjectionResetFp8OutputLaunchesPerLayer ==
              1U &&
          runtime::kLayerMajorPrefillProjectionResetFp8PhysicalLaunchesPerRequest ==
              128U &&
          runtime::kLayerMajorPrefillProjectionResetFp8TensorRoleHitsPerRequest ==
              208U &&
          runtime::kLayerMajorPrefillProjectionResetNvFp4PhysicalLaunchesPerRequest ==
              128U,
      "projection reset has a distinct exact-P40000 schedule identity and "
      "frozen physical/logical route counts");

  const auto candidate = build_plan(
      runtime::kLayerMajorPrefillPromptWideP40Tokens, 0U,
      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
      kProjectionReset);
  if (!runtime::prompt_wide_p40_projection_reset_prefill_plan_enabled()) {
    test.expect(
        !candidate &&
            candidate.error ==
                runtime::PrefillExecutionPlanError::kInvalidArgument,
        "default builds fail closed on the independent projection reset");
    return;
  }

  const runtime::PrefillP40ProjectionResetSchedulePlan* schedule =
      candidate ? &candidate.value->projection_reset_schedule : nullptr;
  test.expect(
      candidate && schedule != nullptr && schedule->enabled &&
          candidate.value->panel_count == 5U &&
          schedule->input_preparation_panel_count_per_layer == 5U &&
          schedule->prompt_core_phase_count_per_layer == 1U &&
          schedule->persistent_mlp_phase_count_per_layer == 1U &&
          schedule->panel_token_count == 8'000U &&
          schedule->projection_m_tokens == 40'000U &&
          schedule->request_capacity_tokens == 40'001U &&
          schedule->route_pass_count == 1U &&
          schedule->fp8_grouped_input_launches_per_layer == 1U &&
          schedule->fp8_output_launches_per_layer == 1U &&
          schedule->fp8_physical_launches_per_request == 128U &&
          schedule->fp8_tensor_role_hits_per_request == 208U &&
          schedule->nvfp4_gate_up_launches_per_layer == 1U &&
          schedule->nvfp4_down_launches_per_layer == 1U &&
          schedule->nvfp4_physical_launches_per_request == 128U &&
          schedule->fp8_grouped_full_prompt_input_required &&
          schedule->fp8_full_prompt_output_required &&
          schedule->nvfp4_full_prompt_required &&
          schedule->internal_m_segmentation_forbidden &&
          schedule->production_accuracy_required &&
          schedule->approximate_numerics_forbidden &&
          schedule->mtp_forbidden && schedule->cublaslt_forbidden &&
          !candidate.value->whole_core_schedule.enabled &&
          candidate.value->whole_core_schedule
                  .fill_panel_phase_count_per_layer == 0U &&
          candidate.value->whole_core_schedule
                  .fp8_single_launch_per_projection_required == false &&
          candidate.value->mlp_schedule
                  .required_gate_up_projection_launches_per_layer == 1U &&
          candidate.value->mlp_schedule
                  .maximum_standalone_silu_launches_per_layer == 0U &&
          candidate.value->mlp_schedule
                  .required_down_projection_launches_per_layer == 1U &&
          candidate.value->mlp_schedule
                  .minimum_total_kernel_launches_per_layer == 2U &&
          candidate.value->mlp_schedule
                  .maximum_total_kernel_launches_per_layer == 2U &&
          runtime::is_valid_unbound_layer_major_prefill_execution_plan(
              *candidate.value),
      "projection reset groups every FP8 input role into one P40 launch, "
      "retains one P40 O and Gate/Down launch, and cannot masquerade as "
      "whole-core");

  test.expect(
      !build_plan(runtime::kLayerMajorPrefillPromptWideP40Tokens, 0U,
                  runtime::kLayerMajorPrefillMaximumSequenceTokens,
                  kProjectionReset) &&
          !build_plan(39'936U, 0U,
                      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
                      kProjectionReset) &&
          !build_plan(runtime::kLayerMajorPrefillPromptWideP40Tokens, 64U,
                      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
                      kProjectionReset),
      "projection reset refuses generalized capacity, M, and warm-position "
      "relabeling");

  runtime::PrefillExecutionPlan mutated = *candidate.value;
  mutated.projection_reset_schedule.fp8_physical_launches_per_request =
      1'040U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "the old panel-owned FP8 physical inventory cannot be relabeled as "
      "projection reset");
  mutated = *candidate.value;
  mutated.projection_reset_schedule.fp8_tensor_role_hits_per_request = 128U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "FP8 launch grouping cannot erase the 208 logical tensor-role hits");
  mutated = *candidate.value;
  mutated.projection_reset_schedule.mtp_forbidden = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "projection reset cannot weaken the non-MTP contract");
  mutated = *candidate.value;
  mutated.whole_core_schedule.enabled = true;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "projection reset cannot acquire the retired whole-core schedule "
      "identity");

  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(*candidate.value);
  test.expect(
      runtime::advance_prefill_progress_after_completion(
          *candidate.value, progress, 0U, 0U) ==
          runtime::PrefillExecutionProgressError::kOutOfOrder,
      "projection reset cannot publish a partial M8000 panel");
  bool ordered = true;
  for (std::size_t layer = 0U;
       ordered && layer < candidate.value->layers.size(); ++layer) {
    ordered =
        runtime::advance_prompt_wide_p40_projection_reset_layer_progress_after_completion(
            *candidate.value, progress, layer) ==
        runtime::PrefillExecutionProgressError::kNone;
  }
  test.expect(
      ordered && progress.next_layer == candidate.value->layers.size() &&
          runtime::mark_prefill_final_hidden_ready(*candidate.value,
                                                   progress) ==
              runtime::PrefillExecutionProgressError::kNone &&
          runtime::prefill_final_commit_ready(*candidate.value, progress),
      "projection reset publishes progress only after each complete grouped "
      "P40 layer and reaches one final commit");
}

void test_prompt_wide_p40_packed_projection_schedule(TestContext& test) {
  using MlpSchedule = runtime::LayerMajorPrefillMlpScheduleTactic;
  constexpr auto kPacked = MlpSchedule::kPromptWideP40PackedProjection;
  test.expect(
      runtime::is_valid_layer_major_prefill_mlp_schedule_tactic(kPacked) &&
          runtime::to_string(kPacked) ==
              "prompt-wide-p40-packed-projection" &&
          runtime::prefill_route_layer_pass_count(5U, kPacked) == 1U &&
          runtime::kLayerMajorPrefillPackedProjectionFp8PhysicalLaunchesPerRequest ==
              128U &&
          runtime::kLayerMajorPrefillPackedProjectionFp8TensorRoleHitsPerRequest ==
              208U &&
          runtime::kLayerMajorPrefillPackedProjectionNvFp4PhysicalLaunchesPerRequest ==
              128U &&
          runtime::kLayerMajorPrefillPackedProjectionArtifactCount == 256U &&
          runtime::kLayerMajorPrefillPackedProjectionAuthenticatedSourceCount ==
              400U,
      "packed projection has an independent P40 identity and complete model "
      "artifact/route counts");

  const auto candidate = build_plan(
      runtime::kLayerMajorPrefillPromptWideP40Tokens, 0U,
      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens, kPacked);
  if (!runtime::prompt_wide_p40_packed_projection_prefill_plan_enabled()) {
    test.expect(
        !candidate &&
            candidate.error ==
                runtime::PrefillExecutionPlanError::kInvalidArgument,
        "default builds fail closed on the independent packed route");
    return;
  }

  const runtime::PrefillP40PackedProjectionSchedulePlan* schedule =
      candidate ? &candidate.value->packed_projection_schedule : nullptr;
  test.expect(
      candidate && schedule != nullptr && schedule->enabled &&
          candidate.value->panel_count == 5U &&
          schedule->input_preparation_panel_count_per_layer == 5U &&
          schedule->prompt_core_phase_count_per_layer == 1U &&
          schedule->packed_mlp_phase_count_per_layer == 1U &&
          schedule->panel_token_count == 8'000U &&
          schedule->projection_m_tokens == 40'000U &&
          schedule->request_capacity_tokens == 40'001U &&
          schedule->route_pass_count == 1U &&
          schedule->fp8_physical_launches_per_request == 128U &&
          schedule->fp8_tensor_role_hits_per_request == 208U &&
          schedule->nvfp4_physical_launches_per_request == 128U &&
          schedule->authenticated_artifact_count == 256U &&
          schedule->authenticated_source_count == 400U &&
          schedule->stream_k_slice_count == 1U &&
          schedule->packed_operands_retained_to_register_decode &&
          schedule->role_specific_tactics_required &&
          schedule->request_time_repack_forbidden &&
          schedule->request_time_tactic_selection_forbidden &&
          schedule->internal_m_segmentation_forbidden &&
          schedule->production_accuracy_required &&
          schedule->approximate_numerics_forbidden &&
          schedule->mtp_forbidden && schedule->cublaslt_forbidden &&
          !candidate.value->projection_reset_schedule.enabled &&
          !candidate.value->whole_core_schedule.enabled &&
          runtime::is_valid_unbound_layer_major_prefill_execution_plan(
              *candidate.value),
      "packed plan admits only full-K AOT packed consumers and cannot inherit "
      "an older P40 schedule");

  runtime::PrefillExecutionPlan mutated = *candidate.value;
  mutated.packed_projection_schedule.stream_k_slice_count = 2U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "unproven Stream-K accumulation cannot enter packed v1");
  mutated = *candidate.value;
  mutated.packed_projection_schedule.authenticated_artifact_count = 255U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "one missing packed role invalidates the whole route");
  mutated = *candidate.value;
  mutated.packed_projection_schedule.request_time_repack_forbidden = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "request-time repack cannot masquerade as an AOT packed route");

  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(*candidate.value);
  test.expect(
      runtime::advance_prefill_progress_after_completion(
          *candidate.value, progress, 0U, 0U) ==
          runtime::PrefillExecutionProgressError::kOutOfOrder,
      "packed projection cannot publish a partial M8000 panel");
  bool ordered = true;
  for (std::size_t layer = 0U;
       ordered && layer < candidate.value->layers.size(); ++layer) {
    ordered =
        runtime::advance_prompt_wide_p40_packed_projection_layer_progress_after_completion(
            *candidate.value, progress, layer) ==
        runtime::PrefillExecutionProgressError::kNone;
  }
  test.expect(
      ordered && progress.next_layer == candidate.value->layers.size() &&
          runtime::mark_prefill_final_hidden_ready(*candidate.value,
                                                   progress) ==
              runtime::PrefillExecutionProgressError::kNone &&
          runtime::prefill_final_commit_ready(*candidate.value, progress),
      "packed route publishes one atomic completed layer and one final "
      "request commit");
}

void test_prompt_wide_p40_packed_nvfp4_v2_schedule(TestContext& test) {
  using MlpSchedule = runtime::LayerMajorPrefillMlpScheduleTactic;
  constexpr auto kPackedV2 = MlpSchedule::kPromptWideP40PackedNvfp4V2;
  test.expect(
      runtime::is_valid_layer_major_prefill_mlp_schedule_tactic(kPackedV2) &&
          runtime::to_string(kPackedV2) ==
              "prompt-wide-p40-packed-nvfp4-v2" &&
          runtime::prefill_route_layer_pass_count(5U, kPackedV2) == 1U &&
          runtime::is_valid_layer_major_prefill_arithmetic_contract(
              runtime::
                  kLayerMajorPrefillPromptWideP40PackedNvfp4V2ArithmeticContract) &&
          runtime::
                  kLayerMajorPrefillPromptWideP40PackedNvfp4V2ArithmeticContract
                      .version == 9U &&
          runtime::
                  kLayerMajorPrefillPromptWideP40PackedNvfp4V2ArithmeticContract
                      .fp8 ==
              runtime::PrefillFp8ArithmeticTactic::
                  kP8000FillDrainSingleBulk &&
          runtime::
                  kLayerMajorPrefillPromptWideP40PackedNvfp4V2ArithmeticContract
                      .p8000_fp8_fill_drain_single_bulk,
      "packed NVFP4 v2 owns append-only NVFP4 and v10 FP8 arithmetic "
      "identities");

  const auto candidate = build_plan(
      runtime::kLayerMajorPrefillPromptWideP40Tokens, 0U,
      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
      kPackedV2);
  if (!runtime::prompt_wide_p40_packed_nvfp4_v2_prefill_plan_enabled()) {
    test.expect(
        !candidate &&
            candidate.error ==
                runtime::PrefillExecutionPlanError::kInvalidArgument,
        "default builds fail closed on packed NVFP4 v2");
    return;
  }

  const runtime::PrefillP40PackedNvfp4V2SchedulePlan* schedule =
      candidate ? &candidate.value->packed_nvfp4_v2_schedule : nullptr;
  test.expect(
      candidate && schedule != nullptr && schedule->enabled &&
          schedule->input_preparation_panel_count_per_layer == 5U &&
          schedule->prompt_core_phase_count_per_layer == 1U &&
          schedule->packed_mlp_phase_count_per_layer == 1U &&
          schedule->panel_token_count == 8'000U &&
          schedule->projection_m_tokens == 40'000U &&
          schedule->request_capacity_tokens == 40'001U &&
          schedule->fp8_physical_launches_per_request == 1'040U &&
          schedule->fp8_tensor_role_hits_per_request == 1'040U &&
          schedule->nvfp4_physical_launches_per_request == 128U &&
          schedule->authenticated_artifact_count == 128U &&
          schedule->authenticated_source_count == 192U &&
          schedule->stream_k_slice_count == 1U &&
          schedule->packed_operands_retained_to_register_decode &&
          schedule->role_specific_tactics_required &&
          schedule->shape_specific_gate_up_required &&
          schedule->shape_specific_down_required &&
          schedule->request_time_repack_forbidden &&
          schedule->request_time_tactic_selection_forbidden &&
          schedule->internal_m_segmentation_forbidden &&
          schedule->production_accuracy_required &&
          schedule->approximate_numerics_forbidden &&
          schedule->mtp_forbidden && schedule->cublaslt_forbidden &&
          !candidate.value->packed_projection_schedule.enabled &&
          !candidate.value->projection_reset_schedule.enabled &&
          !candidate.value->whole_core_schedule.enabled,
      "packed NVFP4 v2 authenticates only its NVFP4 payload while retaining "
      "the v10 panel-wise FP8 topology");

  runtime::PrefillExecutionPlan mutated = *candidate.value;
  mutated.packed_nvfp4_v2_schedule.shape_specific_down_required = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "packed NVFP4 v2 rejects a missing shape-specific Down executor");

  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(*candidate.value);
  bool ordered = runtime::advance_prefill_progress_after_completion(
                     *candidate.value, progress, 0U, 0U) ==
                 runtime::PrefillExecutionProgressError::kOutOfOrder;
  for (std::size_t layer = 0U;
       ordered && layer < candidate.value->layers.size(); ++layer) {
    ordered = runtime::
                  advance_prompt_wide_p40_packed_nvfp4_v2_layer_progress_after_completion(
                      *candidate.value, progress, layer) ==
              runtime::PrefillExecutionProgressError::kNone;
  }
  test.expect(
      ordered && progress.next_layer == candidate.value->layers.size() &&
          runtime::mark_prefill_final_hidden_ready(*candidate.value,
                                                   progress) ==
              runtime::PrefillExecutionProgressError::kNone &&
          runtime::prefill_final_commit_ready(*candidate.value, progress),
      "packed NVFP4 v2 publishes only atomic completed layers");
}

void test_prompt_wide_p40_vllm_marlin_parity_schedule(TestContext& test) {
  using MlpSchedule = runtime::LayerMajorPrefillMlpScheduleTactic;
  constexpr auto kParity = MlpSchedule::kPromptWideP40VllmMarlinParity;
  const auto& arithmetic = runtime::
      kLayerMajorPrefillPromptWideP40VllmMarlinParityArithmeticContract;
  test.expect(
      runtime::is_valid_layer_major_prefill_mlp_schedule_tactic(kParity) &&
          runtime::to_string(kParity) ==
              "prompt-wide-p40-vllm-marlin-parity" &&
          runtime::prefill_route_layer_pass_count(5U, kParity) == 1U &&
          runtime::is_valid_layer_major_prefill_arithmetic_contract(
              arithmetic) &&
          arithmetic.version == 10U &&
          arithmetic.fp8 == runtime::PrefillFp8ArithmeticTactic::
                                kP8000FillDrainSingleBulk &&
          arithmetic.nvfp4 ==
              runtime::PrefillNvFp4ArithmeticTactic::
                  kP40000VllmMarlinProjectionHostDispatchGateThenUpSiluDownResidual &&
          arithmetic.p8000_fp8_fill_drain_single_bulk &&
          arithmetic.nvfp4_gate_up_down_coupled &&
          arithmetic.p40000_post_attention_norm_prompt_wide &&
          !arithmetic.p40000_persistent_gate_up_silu &&
          !arithmetic.p40000_persistent_down_residual,
      "vLLM/Marlin parity owns append-only v10 arithmetic without relabelling "
      "the packed-v2 persistent contract");

  runtime::LayerMajorPrefillArithmeticContract relabelled = arithmetic;
  relabelled.nvfp4 = runtime::PrefillNvFp4ArithmeticTactic::
      kP40000PackedShapeSpecificV2GateUpSiluDownResidual;
  test.expect(
      !runtime::is_valid_layer_major_prefill_arithmetic_contract(relabelled),
      "parity arithmetic cannot be relabelled as packed-v2 NVFP4");

  test.expect(
      runtime::kLayerMajorPrefillVllmMarlinParityFullSegmentsPerProjection ==
              39U &&
          runtime::kLayerMajorPrefillVllmMarlinParityTailSegmentsPerProjection ==
              1U &&
          runtime::kLayerMajorPrefillVllmMarlinParityFullSegmentTokens ==
              1'024U &&
          runtime::kLayerMajorPrefillVllmMarlinParityTailSegmentTokens ==
              64U &&
          runtime::kLayerMajorPrefillVllmMarlinParityNvFp4PhysicalLaunchesPerRequest ==
              5'120U &&
          runtime::kLayerMajorPrefillVllmMarlinParityArtifactCount == 128U &&
          runtime::kLayerMajorPrefillVllmMarlinParityAuthenticatedSourceCount ==
              192U,
      "parity constants seal exact P40000 segmentation and canonical "
      "sidecar inventory");

  const auto candidate = build_plan(
      runtime::kLayerMajorPrefillPromptWideP40Tokens, 0U,
      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens,
      kParity);
  if (!runtime::prompt_wide_p40_vllm_marlin_parity_prefill_plan_enabled()) {
    test.expect(
        !candidate &&
            candidate.error ==
                runtime::PrefillExecutionPlanError::kInvalidArgument,
        "default builds fail closed on vLLM/Marlin parity");
    return;
  }

  const runtime::PrefillP40VllmMarlinParitySchedulePlan* schedule =
      candidate ? &candidate.value->vllm_marlin_parity_schedule : nullptr;
  const runtime::PrefillMlpSchedulePlan* mlp =
      candidate ? &candidate.value->mlp_schedule : nullptr;
  test.expect(
      candidate && schedule != nullptr && mlp != nullptr &&
          schedule->enabled && candidate.value->panel_count == 5U &&
          schedule->input_preparation_panel_count_per_layer == 5U &&
          schedule->prompt_core_phase_count_per_layer == 1U &&
          schedule->segmented_mlp_phase_count_per_layer == 1U &&
          schedule->panel_token_count == 8'000U &&
          schedule->projection_m_tokens == 40'000U &&
          schedule->request_capacity_tokens == 40'001U &&
          schedule->route_pass_count == 1U &&
          schedule->fp8_physical_launches_per_request == 1'040U &&
          schedule->fp8_tensor_role_hits_per_request == 1'040U &&
          schedule->gate_up_segments_per_layer == 40U &&
          schedule->down_segments_per_layer == 40U &&
          schedule->nvfp4_physical_launches_per_request == 5'120U &&
          schedule->gate_up_logical_role_hits_per_request == 64U &&
          schedule->down_logical_role_hits_per_request == 64U &&
          schedule->standalone_silu_launches_per_layer == 1U &&
          schedule->standalone_residual_launches_per_layer == 1U &&
          schedule->lock_clear_operations_per_request == 1U &&
          schedule->full_m1024_segments_per_projection == 39U &&
          schedule->tail_m64_segments_per_projection == 1U &&
          schedule->gate_up_tail_output_tiles == 136U &&
          schedule->gate_up_tail_split_output_tiles == 8U &&
          schedule->down_tail_output_tiles == 20U &&
          schedule->down_tail_split_output_tiles == 12U &&
          schedule->full_segment_m_tokens == 1'024U &&
          schedule->tail_segment_m_tokens == 64U &&
          schedule->gate_up_input_features == 5'120U &&
          schedule->merged_gate_up_output_features == 34'816U &&
          schedule->gate_output_features == 17'408U &&
          schedule->up_output_features == 17'408U &&
          schedule->down_input_features == 17'408U &&
          schedule->down_output_features == 5'120U &&
          schedule->authenticated_artifact_count == 128U &&
          schedule->authenticated_source_count == 192U &&
          schedule->independent_canonical_marlin_sidecars_required &&
          schedule->canonical_token_major_gate_then_up_rows_required &&
          schedule->independent_activated_buffer_required &&
          schedule->bf16_projection_publication_required &&
          schedule->internal_m_segmentation_required &&
          schedule->m64_tail_is_final_segment_required &&
          schedule->m1024_segments_full_k_required &&
          schedule->m64_tail_split_k_required &&
          schedule->m64_tail_locks_required &&
          schedule->m64_tail_zero_initialized_locks_required &&
          schedule->m64_tail_fp32_reduction_workspace_required &&
          schedule->m64_tail_in_kernel_global_reduction_required &&
          schedule->request_time_repack_forbidden &&
          schedule->request_time_tactic_selection_forbidden &&
          schedule->production_accuracy_required &&
          schedule->approximate_numerics_forbidden &&
          schedule->mtp_forbidden && schedule->cublaslt_forbidden &&
          mlp->required_gate_up_projection_launches_per_layer == 40U &&
          mlp->maximum_standalone_silu_launches_per_layer == 1U &&
          mlp->required_down_projection_launches_per_layer == 40U &&
          mlp->minimum_total_kernel_launches_per_layer == 82U &&
          mlp->maximum_total_kernel_launches_per_layer == 82U &&
          !mlp->internal_m_segmentation_forbidden &&
          !candidate.value->packed_nvfp4_v2_schedule.enabled &&
          !candidate.value->packed_projection_schedule.enabled &&
          !candidate.value->projection_reset_schedule.enabled &&
          !candidate.value->whole_core_schedule.enabled &&
          runtime::is_valid_unbound_layer_major_prefill_execution_plan(
              *candidate.value),
      "parity plan requires canonical GateThenUp, stock M64 FP32 reduction, "
      "and 39xM1024+M64 projection host dispatch");

  runtime::PrefillExecutionPlan mutated = *candidate.value;
  mutated.vllm_marlin_parity_schedule
      .canonical_token_major_gate_then_up_rows_required = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "tensor-major Gate and Up planes cannot satisfy canonical GateThenUp");
  mutated = *candidate.value;
  mutated.vllm_marlin_parity_schedule.independent_activated_buffer_required =
      false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "parity rejects an activation alias into the merged GateUp publication");
  mutated = *candidate.value;
  mutated.vllm_marlin_parity_schedule.m1024_segments_full_k_required = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "parity requires full-K ownership for every M1024 segment tile");
  mutated = *candidate.value;
  mutated.vllm_marlin_parity_schedule.m64_tail_split_k_required = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "parity requires the stock M64 split-K tail");
  mutated = *candidate.value;
  mutated.vllm_marlin_parity_schedule
      .m64_tail_zero_initialized_locks_required = false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "parity requires zeroed ordered locks before the first M64 tail");
  mutated = *candidate.value;
  mutated.vllm_marlin_parity_schedule.lock_clear_operations_per_request = 0U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "parity records one request-level ordered-lock clear operation");
  mutated = *candidate.value;
  mutated.vllm_marlin_parity_schedule.down_tail_split_output_tiles = 11U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "parity rejects a non-stock Down tail split topology");
  mutated = *candidate.value;
  mutated.vllm_marlin_parity_schedule.gate_up_segments_per_layer = 39U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "parity rejects a missing M64 GateUp tail");
  mutated = *candidate.value;
  mutated.vllm_marlin_parity_schedule.m64_tail_is_final_segment_required =
      false;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "parity rejects an M64 tail that is not the final segment");

  const auto make_completion_receipt = [](const std::size_t layer) {
    runtime::PrefillP40VllmMarlinParityLayerCompletionReceipt receipt;
    receipt.layer_index = layer;
    receipt.request_lock_clear_operations = layer == 0U ? 1U : 0U;
    receipt.gate_up_full_m1024_launches = 39U;
    receipt.gate_up_split_m64_launches = 1U;
    receipt.standalone_silu_launches = 1U;
    receipt.down_full_m1024_launches = 39U;
    receipt.down_split_m64_launches = 1U;
    receipt.standalone_residual_launches = 1U;
    receipt.retained_prompt_core_complete = true;
    receipt.canonical_gate_then_up_bf16_published = true;
    receipt.activated_bf16_published = true;
    receipt.down_bf16_published = true;
    receipt.stable_lock_owner_bound = true;
    receipt.lock_owner_alias_exclusion_proved = true;
    receipt.ordered_lock_protocol_completed = true;
    receipt.request_stream_completion_observed = true;
    return receipt;
  };

  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(*candidate.value);
  bool ordered = runtime::advance_prefill_progress_after_completion(
                     *candidate.value, progress, 0U, 0U) ==
                 runtime::PrefillExecutionProgressError::kOutOfOrder;
  ordered = ordered &&
            runtime::
                    advance_prompt_wide_p40_packed_nvfp4_v2_layer_progress_after_completion(
                        *candidate.value, progress, 0U) ==
                runtime::PrefillExecutionProgressError::kOutOfOrder;
  runtime::PrefillP40VllmMarlinParityLayerCompletionReceipt incomplete =
      make_completion_receipt(0U);
  --incomplete.gate_up_full_m1024_launches;
  ordered = ordered &&
            runtime::
                    advance_prompt_wide_p40_vllm_marlin_parity_layer_progress_after_completion(
                        *candidate.value, progress, 0U, incomplete) ==
                runtime::PrefillExecutionProgressError::
                    kInvalidCompletionReceipt &&
            progress.next_layer == 0U;
  for (std::size_t layer = 0U;
       ordered && layer < candidate.value->layers.size(); ++layer) {
    const auto receipt = make_completion_receipt(layer);
    ordered = runtime::
                  advance_prompt_wide_p40_vllm_marlin_parity_layer_progress_after_completion(
                      *candidate.value, progress, layer, receipt) ==
              runtime::PrefillExecutionProgressError::kNone;
  }
  test.expect(
      ordered && progress.next_layer == candidate.value->layers.size() &&
          runtime::mark_prefill_final_hidden_ready(*candidate.value,
                                                   progress) ==
              runtime::PrefillExecutionProgressError::kNone &&
          runtime::prefill_final_commit_ready(*candidate.value, progress),
      "parity publishes only a fully completed atomic layer and final "
      "request commit");
}

void test_target_panel_matrix(TestContext& test) {
  test.expect(
      runtime::is_nvfp4_true_large_m_prefill_panel_tokens(8'192U) &&
          runtime::is_nvfp4_true_large_m_prefill_panel_tokens(7'712U) &&
          !runtime::is_nvfp4_true_large_m_prefill_panel_tokens(513U) &&
          !runtime::is_nvfp4_true_large_m_prefill_panel_tokens(8'191U),
      "the true-large-M NVFP4 route admits only complete M8192/M7712 "
      "logical panels");
  expect_panel_shape(test, 1U, 1U, 1U);
  expect_panel_shape(test, 512U, 1U, 512U);
  expect_panel_shape(test, 513U, 1U, 513U);
  expect_panel_shape(test, 8'192U, 1U, 8'192U);
  expect_panel_shape(test, 8'193U, 2U, 4'096U);
  expect_panel_shape(test, 40'000U, 5U, 7'712U);
  expect_panel_shape(test, 60'000U, 8U, 5'424U);
  expect_panel_shape(test, 130'000U, 16U, 7'656U);
  expect_panel_shape(test, 262'144U, 32U, 8'192U);

  const runtime::PrefillExecutionPlanResult p40 = build_plan(40'000U);
  bool p40_true_large_m_complete = static_cast<bool>(p40) &&
                                   p40.value->panel_count == 5U;
  if (p40) {
    for (std::size_t panel = 0U; panel < p40.value->panel_count; ++panel) {
      p40_true_large_m_complete =
          p40_true_large_m_complete &&
          runtime::is_nvfp4_true_large_m_prefill_panel_tokens(
              p40.value->panels[panel].token_count);
    }
  }
  test.expect(
      p40_true_large_m_complete &&
          p40.value->panels[0].token_count == 8'192U &&
          p40.value->panels[1].token_count == 8'192U &&
          p40.value->panels[2].token_count == 8'192U &&
          p40.value->panels[3].token_count == 7'712U &&
          p40.value->panels[4].token_count == 7'712U,
      "P40K is exactly three M8192 plus two M7712 panels for the first "
      "true-large-M product gate");

  const runtime::PrefillExecutionPlanResult offset =
      build_plan(8'193U, 512U, 16'384U);
  test.expect(offset && offset.value->first_position == 512U &&
                  offset.value->panels[0].first_position == 512U &&
                  offset.value->panels[0].end_position == 4'609U &&
                  offset.value->panels[1].first_position == 4'609U &&
                  offset.value->panels[1].end_position == 8'705U &&
                  offset.value->final_position == 8'705U,
              "nonzero request positions preserve continuous absolute panel "
              "coordinates");
}

[[nodiscard]] std::vector<std::size_t> physical_segment_schedule(
    std::size_t remaining_panel_tokens) {
  std::vector<std::size_t> schedule;
  while (remaining_panel_tokens != 0U) {
    const std::size_t segment =
        runtime::next_prefill_physical_segment_token_count(
            remaining_panel_tokens);
    if (!runtime::is_prefill_physical_segment_token_count(segment) ||
        segment > remaining_panel_tokens) {
      return {};
    }
    schedule.push_back(segment);
    remaining_panel_tokens -= segment;
  }
  return schedule;
}

[[nodiscard]] std::vector<std::size_t> layer_major_physical_segment_schedule(
    std::size_t remaining_panel_tokens) {
  std::vector<std::size_t> schedule;
  while (remaining_panel_tokens != 0U) {
    const std::size_t segment =
        runtime::next_layer_major_prefill_physical_segment_token_count(
            remaining_panel_tokens);
    if (!runtime::is_layer_major_prefill_physical_segment_token_count(
            segment) ||
        segment > remaining_panel_tokens) {
      return {};
    }
    schedule.push_back(segment);
    remaining_panel_tokens -= segment;
  }
  return schedule;
}

void test_balanced_physical_segment_contract(TestContext& test) {
  test.expect(
      physical_segment_schedule(33U) ==
              std::vector<std::size_t>({32U, 1U}) &&
          physical_segment_schedule(257U) ==
              std::vector<std::size_t>({256U, 1U}) &&
          physical_segment_schedule(513U) ==
              std::vector<std::size_t>({512U, 1U}),
      "the legacy canonical physical schedule remains byte-stable");

  test.expect(
      layer_major_physical_segment_schedule(1U) ==
              std::vector<std::size_t>({1U}) &&
          layer_major_physical_segment_schedule(33U) ==
              std::vector<std::size_t>({33U}) &&
          layer_major_physical_segment_schedule(255U) ==
              std::vector<std::size_t>({255U}) &&
          layer_major_physical_segment_schedule(257U) ==
              std::vector<std::size_t>({257U}) &&
          layer_major_physical_segment_schedule(511U) ==
              std::vector<std::size_t>({511U}) &&
          layer_major_physical_segment_schedule(512U) ==
              std::vector<std::size_t>({512U}) &&
          layer_major_physical_segment_schedule(513U) ==
              std::vector<std::size_t>({257U, 256U}) &&
          layer_major_physical_segment_schedule(1'024U) ==
              std::vector<std::size_t>({512U, 512U}) &&
          layer_major_physical_segment_schedule(1'025U) ==
              std::vector<std::size_t>({512U, 257U, 256U}),
      "layer-major segments preserve full blocks and balance only the final "
      "full-plus-tail pair");

  constexpr std::array<std::uint64_t, 5U> kPanelBoundaryPrompts{
      8'191U, 8'192U, 8'193U, 130'000U,
      runtime::kLayerMajorPrefillMaximumSequenceTokens};
  for (const std::uint64_t prompt_tokens : kPanelBoundaryPrompts) {
    const runtime::PrefillExecutionPlanResult result = build_plan(prompt_tokens);
    bool exact = result.ok();
    std::uint64_t next_position = 0U;
    if (result) {
      for (std::size_t panel_index = 0U;
           panel_index < result.value->panel_count; ++panel_index) {
        const runtime::PrefillOperatorPanel& panel =
            result.value->panels[panel_index];
        exact = exact && panel.first_position == next_position;
        std::size_t remaining = panel.token_count;
        std::uint64_t segment_position = panel.first_position;
        while (remaining != 0U) {
          const std::size_t segment =
              runtime::next_layer_major_prefill_physical_segment_token_count(
                  remaining);
          exact = exact &&
                  runtime::is_layer_major_prefill_physical_segment_token_count(
                      segment) &&
                  segment != 0U && segment <= remaining &&
                  segment_position + segment <= panel.end_position;
          segment_position += segment;
          remaining -= segment;
        }
        exact = exact && segment_position == panel.end_position;
        next_position = panel.end_position;
      }
    }
    test.expect(exact && next_position == prompt_tokens,
                "physical segments are continuous and stay inside C8192 panels");
  }

  test.expect(
      runtime::next_layer_major_prefill_physical_segment_token_count(0U) ==
              0U &&
          !runtime::is_layer_major_prefill_physical_segment_token_count(0U) &&
          runtime::is_layer_major_prefill_physical_segment_token_count(33U) &&
          runtime::is_layer_major_prefill_physical_segment_token_count(255U) &&
          runtime::is_layer_major_prefill_physical_segment_token_count(511U) &&
          !runtime::is_layer_major_prefill_physical_segment_token_count(513U),
      "the layer-major schedule admits every nonzero C1..C512 geometry");
}

void expect_arithmetic_span_ledger(
    TestContext& test, const std::size_t panel_tokens,
    const std::vector<std::size_t>& expected_counts) {
  const runtime::LayerMajorPrefillArithmeticSpanLedger ledger =
      runtime::make_layer_major_prefill_arithmetic_span_ledger(panel_tokens);
  bool exact =
      runtime::is_valid_layer_major_prefill_arithmetic_span_ledger(ledger) &&
      ledger.token_count == panel_tokens &&
      ledger.span_count == expected_counts.size();
  std::size_t expected_offset = 0U;
  if (exact) {
    for (std::size_t index = 0U; index < expected_counts.size(); ++index) {
      exact = exact && ledger.spans[index].token_offset == expected_offset &&
              ledger.spans[index].token_count == expected_counts[index];
      expected_offset += expected_counts[index];
    }
  }
  test.expect(exact && expected_offset == panel_tokens &&
                  expected_counts ==
                      layer_major_physical_segment_schedule(panel_tokens),
              "arithmetic ledger exactly preserves the compatibility span "
              "sequence");
}

void test_exact_arithmetic_span_ledgers(TestContext& test) {
  expect_arithmetic_span_ledger(test, 513U, {257U, 256U});
  expect_arithmetic_span_ledger(test, 1'025U, {512U, 257U, 256U});
  expect_arithmetic_span_ledger(
      test, 8'192U, std::vector<std::size_t>(
                        runtime::kLayerMajorPrefillMaximumArithmeticSpanCount,
                        512U));

  const runtime::PrefillExecutionPlanResult p8193 = build_plan(8'193U);
  bool balanced_panels = p8193 && p8193.value->panel_count == 2U;
  if (balanced_panels) {
    const runtime::PrefillOperatorPanel& first = p8193.value->panels[0];
    const runtime::PrefillOperatorPanel& second = p8193.value->panels[1];
    balanced_panels = first.first_position == 0U &&
                      first.token_count == 4'097U &&
                      first.end_position == 4'097U &&
                      second.first_position == first.end_position &&
                      second.token_count == 4'096U &&
                      second.end_position == 8'193U;
    expect_arithmetic_span_ledger(
        test, first.token_count,
        {512U, 512U, 512U, 512U, 512U, 512U, 512U, 257U, 256U});
    expect_arithmetic_span_ledger(
        test, second.token_count,
        {512U, 512U, 512U, 512U, 512U, 512U, 512U, 512U});
    const runtime::LayerMajorPrefillArithmeticSpanLedger first_ledger =
        runtime::make_layer_major_prefill_arithmetic_span_ledger(
            first.token_count);
    const runtime::LayerMajorPrefillArithmeticSpanLedger second_ledger =
        runtime::make_layer_major_prefill_arithmetic_span_ledger(
            second.token_count);
    balanced_panels =
        balanced_panels &&
        first.first_position + first_ledger.token_count ==
            first.end_position &&
        second.first_position + second_ledger.token_count ==
            second.end_position;
  }
  test.expect(balanced_panels,
              "P8193 panel and arithmetic ledgers continuously cover the "
              "whole prompt");

  runtime::LayerMajorPrefillArithmeticSpanLedger noncanonical =
      runtime::make_layer_major_prefill_arithmetic_span_ledger(513U);
  noncanonical.spans[0] = {0U, 256U};
  noncanonical.spans[1] = {256U, 257U};
  test.expect(
      !runtime::is_valid_layer_major_prefill_arithmetic_span_ledger(
          noncanonical),
      "ledger validator rejects a contiguous but noncanonical P513 split");
  test.expect(runtime::is_valid_layer_major_prefill_arithmetic_contract(
                  runtime::kLayerMajorPrefillExactArithmeticContract),
              "the bound arithmetic contract is explicit and immutable");
  test.expect(
      runtime::is_valid_layer_major_prefill_arithmetic_contract(
          runtime::kLayerMajorPrefillExactMarlinM8192ArithmeticContract) &&
          runtime::kLayerMajorPrefillExactMarlinM8192ArithmeticContract
                  .version == 2U &&
          runtime::kLayerMajorPrefillExactMarlinM8192ArithmeticContract
              .m8192_single_bulk_projection &&
          runtime::kLayerMajorPrefillExactMarlinM8192ArithmeticContract
              .m8192_fp8_resets_locks_once &&
          runtime::kLayerMajorPrefillExactMarlinM8192ArithmeticContract
              .m8192_nvfp4_uses_independent_down_workspace &&
          runtime::kLayerMajorPrefillExactMarlinM8192ArithmeticContract
              .m8192_nvfp4_residual_once_after_bulk,
      "the M8192 Marlin tactic owns a distinct sealed arithmetic contract");
  test.expect(
      runtime::is_valid_layer_major_prefill_arithmetic_contract(
          runtime::kLayerMajorPrefillSegmentedMarlinArithmeticContract) &&
          runtime::kLayerMajorPrefillSegmentedMarlinArithmeticContract
                  .version == 3U &&
          !runtime::kLayerMajorPrefillSegmentedMarlinArithmeticContract
               .nvfp4_interleaves_gate_silu_down_per_span &&
          !runtime::kLayerMajorPrefillSegmentedMarlinArithmeticContract
               .nvfp4_down_reuses_gate_up_locks &&
          !runtime::kLayerMajorPrefillSegmentedMarlinArithmeticContract
               .nvfp4_residual_follows_down_per_span,
      "the segmented Marlin tactic does not masquerade as the exact oracle "
      "arithmetic contract");
  test.expect(
      runtime::is_valid_layer_major_prefill_arithmetic_contract(
          runtime::kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract) &&
          runtime::kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract
                  .version == 4U &&
          runtime::kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract
              .nvfp4_true_large_m_m8192 &&
          runtime::kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract
              .nvfp4_true_large_m_m7712 &&
          runtime::kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract
              .nvfp4_gate_up_down_coupled &&
          !runtime::kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract
               .nvfp4_interleaves_gate_silu_down_per_span &&
          !runtime::kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract
               .m8192_nvfp4_uses_independent_down_workspace,
      "the true-large-M NVFP4 tactic seals M8192/M7712 Gate+Up and Down as "
      "one route without an oracle-span fallback");

  const auto& whole_core =
      runtime::kLayerMajorPrefillPromptWideP40WholeCoreArithmeticContract;
  const auto& projection_reset = runtime::
      kLayerMajorPrefillPromptWideP40ProjectionResetArithmeticContract;
  test.expect(
      runtime::is_valid_layer_major_prefill_arithmetic_contract(whole_core) &&
          runtime::is_valid_layer_major_prefill_arithmetic_contract(
              projection_reset) &&
          whole_core.version == 6U && projection_reset.version == 7U &&
          whole_core.fp8 == runtime::PrefillFp8ArithmeticTactic::
                                kP8000FillDrainSingleBulk &&
          projection_reset.fp8 == runtime::PrefillFp8ArithmeticTactic::
                                      kP40000GroupedInputAndOutputSingleBulk &&
          whole_core.p8000_fp8_fill_drain_single_bulk &&
          !projection_reset.p8000_fp8_fill_drain_single_bulk &&
          &whole_core != &projection_reset,
      "projection reset owns a distinct v7 grouped-P40000 FP8 arithmetic "
      "contract instead of reusing the v6 M8000 contract");

  runtime::LayerMajorPrefillArithmeticContract reset_with_v6_fp8 =
      projection_reset;
  reset_with_v6_fp8.fp8 =
      runtime::PrefillFp8ArithmeticTactic::kP8000FillDrainSingleBulk;
  reset_with_v6_fp8.p8000_fp8_fill_drain_single_bulk = true;
  runtime::LayerMajorPrefillArithmeticContract whole_core_with_reset_fp8 =
      whole_core;
  whole_core_with_reset_fp8.fp8 = runtime::PrefillFp8ArithmeticTactic::
      kP40000GroupedInputAndOutputSingleBulk;
  whole_core_with_reset_fp8.p8000_fp8_fill_drain_single_bulk = false;
  test.expect(
      !runtime::is_valid_layer_major_prefill_arithmetic_contract(
          reset_with_v6_fp8) &&
          !runtime::is_valid_layer_major_prefill_arithmetic_contract(
              whole_core_with_reset_fp8),
      "v6 and v7 arithmetic identities cannot be relabeled by changing only "
      "their FP8 boundary metadata");
}

void test_fixed_layer_schedule(TestContext& test) {
  const runtime::PrefillExecutionPlanResult result = build_plan(40'000U);
  test.expect(result.ok(), "40K layer-major topology builds");
  if (!result) {
    return;
  }

  const runtime::PrefillExecutionPlan& plan = *result.value;
  std::size_t linear_layers = 0U;
  std::size_t full_layers = 0U;
  bool exact_schedule = true;
  for (std::size_t layer_index = 0U; layer_index < plan.layers.size();
       ++layer_index) {
    const runtime::PrefillLayerExecution& layer = plan.layers[layer_index];
    const bool expected_full = (layer_index + 1U) % 4U == 0U;
    const model::LayerType expected_type =
        expected_full ? model::LayerType::kFullAttention
                      : model::LayerType::kLinearAttention;
    const runtime::PrefillProgressDomain expected_progress =
        expected_full ? runtime::PrefillProgressDomain::kKvCache
                      : runtime::PrefillProgressDomain::kGdnState;
    exact_schedule = exact_schedule && layer.layer_index == layer_index &&
                     layer.layer_type == expected_type &&
                     layer.progress_domain == expected_progress &&
                     layer.panel_count == plan.panel_count;
    if (expected_full) {
      ++full_layers;
    } else {
      ++linear_layers;
    }
  }
  test.expect(exact_schedule && linear_layers == 48U && full_layers == 16U &&
                  plan.layers[3].layer_type ==
                      model::LayerType::kFullAttention &&
                  plan.layers[63].layer_type ==
                      model::LayerType::kFullAttention,
              "all 64 layers retain the fixed 48-GDN/16-attention schedule");
  test.expect(plan.final_commit.expected_initial_sequence_length == 0U &&
                  plan.final_commit.committed_sequence_length == 40'000U &&
                  plan.final_commit.commit_count == 1U,
              "the immutable plan declares exactly one final state commit");
}

void test_public_unbound_topology_validator(TestContext& test) {
  const runtime::PrefillExecutionPlanResult result =
      build_plan(40'000U, 512U, 65'536U);
  test.expect(result.ok(), "unbound validator fixture builds");
  if (!result) {
    return;
  }

  const runtime::PrefillExecutionPlan baseline = *result.value;
  test.expect(
      runtime::is_valid_unbound_layer_major_prefill_execution_plan(baseline),
      "the public validator accepts a complete unbound layer-major plan");

  const auto expect_rejected = [&](runtime::PrefillExecutionPlan candidate,
                                   const char* const message) {
    test.expect(
        !runtime::is_valid_unbound_layer_major_prefill_execution_plan(
            candidate),
        message);
  };

  runtime::PrefillExecutionPlan candidate = baseline;
  candidate.traversal = static_cast<runtime::PrefillTraversalOrder>(0xffU);
  expect_rejected(candidate, "the unbound validator rejects traversal drift");

  candidate = baseline;
  candidate.legacy_public_tile_limit += 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects legacy capacity drift");

  candidate = baseline;
  candidate.operator_panel_capacity -= 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects operator capacity drift");

  candidate = baseline;
  candidate.prompt_token_count -= 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects prompt-span drift");

  candidate = baseline;
  candidate.panel_count -= 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects an incomplete panel set");

  candidate = baseline;
  candidate.panels[0].token_count -= 1U;
  candidate.panels[0].end_position -= 1U;
  candidate.panels[1].first_position -= 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects non-canonical panel sizes");

  candidate = baseline;
  candidate.panels[1].ordinal = 0U;
  expect_rejected(candidate,
                  "the unbound validator rejects panel-order drift");

  candidate = baseline;
  candidate.layers[7].layer_index = 8U;
  expect_rejected(candidate,
                  "the unbound validator rejects layer-order drift");

  candidate = baseline;
  candidate.layers[3].layer_type = model::LayerType::kLinearAttention;
  expect_rejected(candidate,
                  "the unbound validator rejects the fixed layer schedule");

  candidate = baseline;
  candidate.layers[3].progress_domain =
      runtime::PrefillProgressDomain::kGdnState;
  expect_rejected(candidate,
                  "the unbound validator rejects progress-domain drift");

  candidate = baseline;
  candidate.final_commit.commit_count = 2U;
  expect_rejected(candidate,
                  "the unbound validator rejects final-commit drift");

  candidate = baseline;
  candidate.operator_bindings_complete = true;
  expect_rejected(candidate,
                  "the unbound validator rejects operator-bound plans");

  candidate = baseline;
  candidate.first_position =
      runtime::kLayerMajorPrefillMaximumSequenceTokens;
  candidate.final_position = candidate.first_position + 1U;
  candidate.prompt_token_count = 1U;
  candidate.panel_count = 1U;
  candidate.panels[0] = runtime::PrefillOperatorPanel{
      0U, candidate.first_position, 1U, candidate.final_position};
  candidate.final_commit = runtime::PrefillFinalCommitPlan{
      candidate.first_position, candidate.final_position, 1U};
  for (runtime::PrefillLayerExecution& layer : candidate.layers) {
    layer.panel_count = 1U;
  }
  expect_rejected(candidate,
                  "the unbound validator rejects an out-of-capacity span");
}

void test_strict_layer_major_progress(TestContext& test) {
  const runtime::PrefillExecutionPlanResult result = build_plan(8'193U);
  test.expect(result.ok(), "two-panel progress topology builds");
  if (!result) {
    return;
  }
  const runtime::PrefillExecutionPlan& plan = *result.value;
  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(plan);

  bool initialized = progress.next_layer == 0U &&
                     progress.next_panel == 0U &&
                     !progress.final_hidden_ready &&
                     !progress.prefill_state_committed;
  for (std::size_t layer = 0U; layer < plan.layers.size(); ++layer) {
    initialized = initialized && progress.kv_visible_end[layer] == 0U &&
                  progress.gdn_advanced_end[layer] == 0U &&
                  progress.completed_panels[layer] == 0U;
  }
  test.expect(initialized,
              "request-owned KV/GDN progress begins at the admitted position");

  runtime::PrefillExecutionProgress before = progress;
  auto status = runtime::advance_prefill_progress_after_completion(
      plan, progress, 1U, 0U);
  test.expect(status == runtime::PrefillExecutionProgressError::kOutOfOrder &&
                  same_progress(progress, before),
              "a later layer cannot begin before the current layer closes");

  status = runtime::advance_prefill_progress_after_completion(
      plan, progress, 0U, 1U);
  test.expect(status == runtime::PrefillExecutionProgressError::kOutOfOrder &&
                  same_progress(progress, before),
              "a later panel cannot skip its predecessor");

  status = runtime::advance_prefill_progress_after_completion(
      plan, progress, plan.layers.size(), 0U);
  test.expect(status ==
                      runtime::PrefillExecutionProgressError::kLayerOutOfRange &&
                  same_progress(progress, before),
              "an out-of-range layer fails without changing progress");

  status = runtime::advance_prefill_progress_after_completion(
      plan, progress, 0U, plan.panel_count);
  test.expect(status ==
                      runtime::PrefillExecutionProgressError::kPanelOutOfRange &&
                  same_progress(progress, before),
              "an out-of-range panel fails without changing progress");

  test.expect(runtime::mark_prefill_final_hidden_ready(plan, progress) ==
                      runtime::PrefillExecutionProgressError::
                          kExecutionIncomplete &&
                  runtime::publish_prefill_state_committed(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kCommitNotReady,
              "final hidden and commit remain unavailable before all layers");

  bool ordered = true;
  for (std::size_t layer = 0U; layer < plan.layers.size(); ++layer) {
    for (std::size_t panel = 0U; panel < plan.panel_count; ++panel) {
      ordered = ordered &&
                runtime::advance_prefill_progress_after_completion(
                    plan, progress, layer, panel) ==
                    runtime::PrefillExecutionProgressError::kNone;
    }
  }
  test.expect(ordered && progress.next_layer == plan.layers.size() &&
                  progress.next_panel == 0U,
              "progress follows strict layer-then-panel traversal");

  bool domains_exact = true;
  for (std::size_t layer = 0U; layer < plan.layers.size(); ++layer) {
    const bool full = plan.layers[layer].progress_domain ==
                      runtime::PrefillProgressDomain::kKvCache;
    domains_exact =
        domains_exact && progress.completed_panels[layer] == plan.panel_count &&
        (full ? progress.kv_visible_end[layer] == plan.final_position
              : progress.kv_visible_end[layer] == plan.first_position) &&
        (full ? progress.gdn_advanced_end[layer] == plan.first_position
              : progress.gdn_advanced_end[layer] == plan.final_position);
  }
  test.expect(domains_exact,
              "each layer advances only its declared KV or GDN progress");

  test.expect(!runtime::prefill_final_commit_ready(plan, progress) &&
                  runtime::publish_prefill_state_committed(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kCommitNotReady,
              "complete state progress still requires the final hidden handoff");
  test.expect(runtime::mark_prefill_final_hidden_ready(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kNone &&
                  runtime::prefill_final_commit_ready(plan, progress),
              "final hidden readiness opens the single logical commit gate");
  test.expect(runtime::publish_prefill_state_committed(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kNone &&
                  progress.prefill_state_committed &&
                  !runtime::prefill_final_commit_ready(plan, progress) &&
                  runtime::publish_prefill_state_committed(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kAlreadyCommitted,
              "the logical PrefillStateCommitted transition is single-shot");
}

void test_layer_wide_p40_mlp_schedule(TestContext& test) {
  using MlpSchedule = runtime::LayerMajorPrefillMlpScheduleTactic;
  test.expect(
      runtime::is_valid_layer_major_prefill_mlp_schedule_tactic(
          MlpSchedule::kPerOperatorPanel) &&
          runtime::is_valid_layer_major_prefill_mlp_schedule_tactic(
              MlpSchedule::kLayerWideP40ExactFullM) &&
          !runtime::is_valid_layer_major_prefill_mlp_schedule_tactic(
              static_cast<MlpSchedule>(0xffU)) &&
          runtime::to_string(MlpSchedule::kPerOperatorPanel) ==
              "per-operator-panel" &&
          runtime::to_string(MlpSchedule::kLayerWideP40ExactFullM) ==
              "layer-wide-p40-exact-full-m" &&
          runtime::prefill_route_layer_pass_count(
              5U, MlpSchedule::kPerOperatorPanel) == 5U &&
          runtime::prefill_route_layer_pass_count(
              5U, MlpSchedule::kLayerWideP40ExactFullM) == 1U,
      "MLP schedules have closed, durable topology identities");

  const auto incumbent =
      build_plan(runtime::kLayerMajorPrefillLayerWideMlpP40Tokens);
  test.expect(
      incumbent && incumbent.value->panel_count == 5U &&
          incumbent.value->mlp_schedule.tactic ==
              MlpSchedule::kPerOperatorPanel &&
          incumbent.value->mlp_schedule
                  .operator_panel_phase_count_per_layer == 5U &&
          incumbent.value->mlp_schedule
                  .mlp_phase_submission_count_per_layer == 5U &&
          incumbent.value->mlp_schedule.maximum_m_per_mlp_submission ==
              runtime::kLayerMajorPrefillOperatorPanelTokens &&
          incumbent.value->mlp_schedule
                  .required_gate_up_projection_launches_per_layer == 0U &&
          !incumbent.value->mlp_schedule.waits_for_all_operator_panels,
      "the default P40 topology preserves five panel-local MLP phases and "
      "makes no unbound physical-launch claim");

  const auto candidate = build_plan(
      runtime::kLayerMajorPrefillLayerWideMlpP40Tokens, 0U,
      runtime::kLayerMajorPrefillMaximumSequenceTokens,
      MlpSchedule::kLayerWideP40ExactFullM);
  if (!runtime::layer_wide_p40_mlp_prefill_plan_enabled()) {
    test.expect(
        !candidate &&
            candidate.error ==
                runtime::PrefillExecutionPlanError::kInvalidArgument,
        "production/default builds fail closed on the test-only P40 plan");
    return;
  }

  test.expect(
      candidate && candidate.value->panel_count == 5U &&
          candidate.value->mlp_schedule.tactic ==
              MlpSchedule::kLayerWideP40ExactFullM &&
          candidate.value->mlp_schedule
                  .operator_panel_phase_count_per_layer == 5U &&
          candidate.value->mlp_schedule
                  .mlp_phase_submission_count_per_layer == 1U &&
          candidate.value->mlp_schedule.maximum_m_per_mlp_submission ==
              runtime::kLayerMajorPrefillLayerWideMlpP40Tokens &&
          candidate.value->mlp_schedule
                  .required_gate_up_projection_launches_per_layer == 1U &&
          candidate.value->mlp_schedule
                  .maximum_standalone_silu_launches_per_layer == 1U &&
          candidate.value->mlp_schedule
                  .required_down_projection_launches_per_layer == 1U &&
          candidate.value->mlp_schedule
                  .minimum_total_kernel_launches_per_layer == 2U &&
          candidate.value->mlp_schedule
                  .maximum_total_kernel_launches_per_layer == 3U &&
          candidate.value->mlp_schedule.waits_for_all_operator_panels &&
          candidate.value->mlp_schedule
              .post_attention_residual_completed_panelwise &&
          candidate.value->mlp_schedule
              .post_attention_norm_is_prompt_wide &&
          candidate.value->mlp_schedule
              .exact_full_m_binding_required &&
          candidate.value->mlp_schedule.internal_m_segmentation_forbidden &&
          !candidate.value->operator_bindings_complete &&
          !candidate.value->executable() &&
          runtime::is_valid_unbound_layer_major_prefill_execution_plan(
              *candidate.value),
      "P40 completes five Attention/GDN panels then requires one exact "
      "unsplit full-M MLP phase with a two-to-three launch envelope");

  test.expect(
      !build_plan(39'968U, 0U,
                  runtime::kLayerMajorPrefillMaximumSequenceTokens,
                  MlpSchedule::kLayerWideP40ExactFullM) &&
          !build_plan(40'064U, 0U,
                      runtime::kLayerMajorPrefillMaximumSequenceTokens,
                      MlpSchedule::kLayerWideP40ExactFullM) &&
          !build_plan(runtime::kLayerMajorPrefillLayerWideMlpP40Tokens, 64U,
                      runtime::kLayerMajorPrefillMaximumSequenceTokens,
                      MlpSchedule::kLayerWideP40ExactFullM),
      "the test-only schedule admits only a cold, exact P40000 prompt");

  runtime::PrefillExecutionPlan mutated = *candidate.value;
  mutated.mlp_schedule.required_down_projection_launches_per_layer = 2U;
  test.expect(
      !runtime::is_valid_unbound_layer_major_prefill_execution_plan(mutated),
      "a binder cannot relabel an internally segmented Down projection as "
      "the required one-launch P40 contract");

  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(*candidate.value);
  test.expect(
      runtime::advance_layer_wide_p40_mlp_progress_after_completion(
          *candidate.value, progress, 0U) ==
          runtime::PrefillExecutionProgressError::kOutOfOrder,
      "full-M MLP cannot publish before every panel phase in its layer");
  bool ordered = true;
  for (std::size_t layer = 0U;
       ordered && layer < candidate.value->layers.size(); ++layer) {
    for (std::size_t panel = 0U;
         ordered && panel < candidate.value->panel_count; ++panel) {
      ordered = runtime::advance_prefill_progress_after_completion(
                    *candidate.value, progress, layer, panel) ==
                runtime::PrefillExecutionProgressError::kNone;
    }
    ordered = ordered && progress.next_layer == layer &&
              progress.next_panel == 0U &&
              progress.completed_mlp_phases[layer] == 0U &&
              runtime::advance_layer_wide_p40_mlp_progress_after_completion(
                  *candidate.value, progress, layer) ==
                  runtime::PrefillExecutionProgressError::kNone &&
              progress.completed_mlp_phases[layer] == 1U;
  }
  test.expect(
      ordered && progress.next_layer == candidate.value->layers.size() &&
          runtime::mark_prefill_final_hidden_ready(*candidate.value,
                                                   progress) ==
              runtime::PrefillExecutionProgressError::kNone,
      "progress cannot advance to the next layer or final hidden until its "
      "single full-M MLP phase completes");
}

void test_fail_closed_inputs(TestContext& test) {
  test.expect(!build_plan(0U) &&
                  build_plan(0U).error ==
                      runtime::PrefillExecutionPlanError::kInvalidArgument,
              "zero-token plans fail closed");
  test.expect(!build_plan(1U, 0U, 0U) &&
                  build_plan(1U, 0U, 0U).error ==
                      runtime::PrefillExecutionPlanError::kInvalidArgument,
              "zero sequence capacity fails closed");
  test.expect(!build_plan(1U, 0U,
                          runtime::kLayerMajorPrefillMaximumSequenceTokens +
                              1U) &&
                  build_plan(1U, 0U,
                             runtime::kLayerMajorPrefillMaximumSequenceTokens +
                                 1U)
                          .error == runtime::PrefillExecutionPlanError::
                                        kInvalidArgument,
              "capacity above the fixed request contract fails closed");
  test.expect(!build_plan(2U, 9U, 10U) &&
                  build_plan(2U, 9U, 10U).error ==
                      runtime::PrefillExecutionPlanError::kCapacityExceeded,
              "prompt spans cannot exceed the admitted request capacity");
  test.expect(!build_plan(
                  runtime::kLayerMajorPrefillMaximumSequenceTokens + 1U) &&
                  build_plan(
                      runtime::kLayerMajorPrefillMaximumSequenceTokens + 1U)
                          .error == runtime::PrefillExecutionPlanError::
                                        kCapacityExceeded,
              "prompt spans cannot exceed the absolute request capacity");

  runtime::PrefillExecutionPlanOptions overflow;
  overflow.first_position = std::numeric_limits<std::uint64_t>::max();
  overflow.prompt_token_count = 1U;
  overflow.max_sequence_length =
      runtime::kLayerMajorPrefillMaximumSequenceTokens;
  const runtime::PrefillExecutionPlanResult overflow_result =
      runtime::build_unbound_layer_major_prefill_execution_plan(overflow);
  test.expect(!overflow_result &&
                  overflow_result.error ==
                      runtime::PrefillExecutionPlanError::kArithmeticOverflow,
              "position arithmetic overflow fails before narrowing");

  runtime::PrefillExecutionPlan invalid_plan;
  runtime::PrefillExecutionProgress invalid_progress =
      runtime::make_prefill_execution_progress(invalid_plan);
  test.expect(runtime::advance_prefill_progress_after_completion(
                  invalid_plan, invalid_progress, 0U, 0U) ==
                  runtime::PrefillExecutionProgressError::kInvalidPlan,
              "default or malformed topologies cannot advance");
}

}  // namespace

int main() {
  TestContext test;
  test_public_tile_and_operator_panel_are_independent(test);
  test_target_panel_matrix(test);
  test_balanced_physical_segment_contract(test);
  test_exact_arithmetic_span_ledgers(test);
  test_fixed_layer_schedule(test);
  test_public_unbound_topology_validator(test);
  test_strict_layer_major_progress(test);
  test_layer_wide_p40_mlp_schedule(test);
  test_prompt_wide_p40_whole_core_schedule(test);
  test_prompt_wide_p40_macrofeed_v3_schedule(test);
  test_prompt_wide_p40_projection_reset_schedule(test);
  test_prompt_wide_p40_packed_projection_schedule(test);
  test_prompt_wide_p40_packed_nvfp4_v2_schedule(test);
  test_prompt_wide_p40_vllm_marlin_parity_schedule(test);
  test_fail_closed_inputs(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " prefill execution plan test(s) failed\n";
    return 1;
  }
  std::cout << "All prefill execution plan tests passed\n";
  return 0;
}
