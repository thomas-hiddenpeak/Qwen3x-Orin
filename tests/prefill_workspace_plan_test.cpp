#include "q3x/runtime/prefill_workspace_plan.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

namespace runtime = q3x::runtime;

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

[[nodiscard]] runtime::LayerMajorPrefillWorkspacePlanResult build_plan(
    const std::uint64_t sequence_capacity,
    const runtime::PrefillHiddenStrategy hidden =
        runtime::PrefillHiddenStrategy::kSinglePromptWideConditional,
    const runtime::PrefillOperatorScratchStrategy scratch =
        runtime::PrefillOperatorScratchStrategy::
            kC8192FamilyOverlayConditional,
    const std::uint64_t arena_limit =
        runtime::kMaximumRequestArenaBytes) {
  runtime::LayerMajorPrefillWorkspaceOptions options;
  options.sequence_capacity_tokens = sequence_capacity;
  options.request_arena_limit_bytes = arena_limit;
  options.hidden_strategy = hidden;
  options.scratch_strategy = scratch;
  return runtime::build_unbound_layer_major_prefill_workspace_plan(options);
}

void test_target_capacity_matrix(TestContext& test) {
  struct Case {
    std::uint64_t tokens;
    std::uint64_t selected_bytes;
    std::uint64_t conservative_bytes;
    bool fits;
  };
  constexpr std::array<Case, 5U> cases{{
      {40'000U, 3'975'364'608U, 5'453'731'840U, true},
      {60'000U, 5'496'004'608U, 7'181'091'840U, true},
      {130'000U, 10'818'244'608U, 13'226'851'840U, true},
      {131'072U, 10'899'750'912U, 13'319'438'336U, true},
      {262'144U, 20'865'417'216U, 24'639'864'832U, false},
  }};

  for (const Case& current : cases) {
    const auto result = build_plan(current.tokens);
    const bool valid =
        result &&
        result.value->sequence_capacity_tokens == current.tokens &&
        result.value->minimum_conditional.required_bytes ==
            current.selected_bytes &&
        result.value->selected.required_bytes == current.selected_bytes &&
        result.value->conservative.required_bytes ==
            current.conservative_bytes &&
        (result.value->selected.capacity ==
         (current.fits
              ? runtime::PrefillMemoryCapacityVerdict::kFitsDeclaredLimit
              : runtime::PrefillMemoryCapacityVerdict::
                    kExceedsDeclaredLimit)) &&
        (result.value->conservative.capacity ==
         (current.fits
              ? runtime::PrefillMemoryCapacityVerdict::kFitsDeclaredLimit
              : runtime::PrefillMemoryCapacityVerdict::
                    kExceedsDeclaredLimit));
    if (!valid) {
      std::cerr << "  target capacity mismatch at " << current.tokens
                << " tokens\n";
    }
    test.expect(valid,
                "target bucket has checked lower/selected/conservative bytes");
  }
}

void test_selected_dependency_plan(TestContext& test) {
  const auto result = build_plan(130'000U);
  test.expect(result.ok(), "130K selected workspace plan builds");
  if (!result) {
    return;
  }
  const auto& plan = *result.value;
  const auto& hidden = plan.prompt_wide_hidden;
  const auto& scratch = plan.operator_scratch;

  test.expect(hidden.minimum_conditional.buffer_count == 1U &&
                  hidden.minimum_conditional.requires_panelwise_in_place_contract &&
                  hidden.minimum_conditional.aggregate_bf16.required_bytes ==
                      1'331'200'000U &&
                  hidden.minimum_conditional.aggregate_bf16.alias_condition ==
                      runtime::PrefillMemoryAliasCondition::
                          kPanelwiseInputConsumedBeforeOutputOverwrite,
              "selected one-hidden plan exposes its unbound in-place contract");
  test.expect(hidden.conservative.buffer_count == 2U &&
                  !hidden.conservative.requires_panelwise_in_place_contract &&
                  hidden.conservative.aggregate_bf16.required_bytes ==
                      2'662'400'000U &&
                  hidden.conservative.aggregate_bf16.alias_condition ==
                      runtime::PrefillMemoryAliasCondition::kDisjoint,
              "two-hidden plan remains the disjoint conservative upper case");

  test.expect(scratch.c8192_family_overlay_conditional.total_required_bytes ==
                      855'638'016U &&
                  scratch.c8192_family_overlay_conditional
                      .requires_family_completion_events &&
                  scratch.c8192_family_overlay_conditional
                      .requires_legacy_route_exclusion &&
                  scratch.c8192_family_overlay_conditional.aggregate
                          .alias_condition ==
                      runtime::PrefillMemoryAliasCondition::
                          kSequentialFamilyLiveSetOverlay,
              "bounded C8192 scratch is derived from sequential live sets");
  test.expect(plan.selected.hidden_strategy ==
                      runtime::PrefillHiddenStrategy::
                          kSinglePromptWideConditional &&
                  plan.selected.scratch_strategy ==
                      runtime::PrefillOperatorScratchStrategy::
                          kC8192FamilyOverlayConditional &&
                  plan.selected.requires_unbound_alias_or_route_contract,
              "selected profile names rather than hides every dependency");

  test.expect(plan.resident_model.owner ==
                      runtime::PrefillMemoryOwner::kEngineResidentModel &&
                  plan.derived_sidecars.owner ==
                      runtime::PrefillMemoryOwner::kDeploymentPlanSidecar &&
                  !plan.resident_model.required_bytes.has_value() &&
                  !plan.derived_sidecars.required_bytes.has_value() &&
                  !plan.whole_process_required_bytes.has_value() &&
                  plan.whole_process_capacity ==
                      runtime::PrefillMemoryCapacityVerdict::kIndeterminate &&
                  !plan.request_arena_reservation_bound &&
                  !plan.operator_bindings_complete && !plan.executable(),
              "unknown model/sidecar residency fails closed at whole process");
}

void test_real_family_live_sets(TestContext& test) {
  const auto result = build_plan(130'000U);
  test.expect(result.ok(), "family live-set plan builds");
  if (!result) {
    return;
  }
  const auto& scratch = result.value->operator_scratch;
  constexpr std::array<std::uint64_t,
                       runtime::kLayerMajorPrefillScratchFamilyCount>
      expected_bytes{270'008'320U, 372'244'480U, 335'544'320U,
                     855'638'016U};
  constexpr std::array<runtime::PrefillScratchFamily,
                       runtime::kLayerMajorPrefillScratchFamilyCount>
      expected_families{
          runtime::PrefillScratchFamily::kGdnMergedInputProjection,
          runtime::PrefillScratchFamily::kGdnFusedPostConvPrep,
          runtime::PrefillScratchFamily::kFullAttentionProjectionAndCore,
          runtime::PrefillScratchFamily::kMlpMergedGateUp};

  bool exact = true;
  for (std::size_t index = 0U; index < expected_bytes.size(); ++index) {
    const auto& live_set = scratch.c8192_family_live_sets[index];
    exact = exact && live_set.family == expected_families[index] &&
            live_set.aggregate.required_bytes == expected_bytes[index] &&
            live_set.aggregate.lifetime ==
                runtime::PrefillMemoryLifetime::kOperatorPanel &&
            live_set.aggregate.alias_condition ==
                runtime::PrefillMemoryAliasCondition::kDisjoint &&
            !live_set.aggregate.allocation_bound &&
            !live_set.aggregate.alias_contract_bound;
  }
  test.expect(exact,
              "GDN/Attention/GateUp producer-consumer live sets are explicit");
  test.expect(
      scratch.c8192_family_live_sets[0].producer ==
              runtime::PrefillScratchProducer::
                  kGdnInProjQkvzAndInProjBa &&
          scratch.c8192_family_live_sets[0].last_consumer ==
              runtime::PrefillScratchLastConsumer::kGdnFusedPostConvPrep &&
          scratch.c8192_family_live_sets[1].producer ==
              runtime::PrefillScratchProducer::
                  kGdnFusedPostConvPrepQkvGBeta &&
          scratch.c8192_family_live_sets[1].last_consumer ==
              runtime::PrefillScratchLastConsumer::
                  kGdnRecurrentUpdateNormAndOutputProjection &&
          scratch.c8192_family_live_sets[3].producer ==
              runtime::PrefillScratchProducer::
                  kMlpMergedGateUpProjection &&
          scratch.c8192_family_live_sets[3].last_consumer ==
              runtime::PrefillScratchLastConsumer::
                  kMlpSiluGateAndDownProjection,
      "merged producers retain named last-consumer reuse boundaries");
}

void test_legacy_overlay_and_shape_separation(TestContext& test) {
  const auto result = build_plan(
      130'000U,
      runtime::PrefillHiddenStrategy::kDoublePromptWideConservative,
      runtime::PrefillOperatorScratchStrategy::
          kOverlayLegacyC512MutuallyExclusive);
  test.expect(result.ok(), "legacy-overlay plan builds");
  if (!result) {
    return;
  }
  const auto& scratch = result.value->operator_scratch;
  test.expect(scratch.legacy_c512_hidden_bf16.required_bytes == 15'728'640U &&
                  scratch.legacy_c512_projection_bf16.required_bytes ==
                      71'303'168U &&
                  scratch.legacy_c512_linear_scalar_bf16.required_bytes ==
                      98'304U &&
                  scratch.legacy_c512_fp32_scratch.required_bytes ==
                      12'480'000U &&
                  scratch.legacy_c512_only.total_required_bytes ==
                      99'610'112U,
              "legacy C512 projection/fp32 requirements are independently visible");
  test.expect(scratch.mutually_exclusive_overlay.total_required_bytes ==
                      855'638'016U &&
                  scratch.mutually_exclusive_overlay
                      .requires_family_completion_events &&
                  scratch.mutually_exclusive_overlay
                      .requires_route_mutual_exclusion_event &&
                  scratch.mutually_exclusive_overlay.aggregate
                          .alias_condition ==
                      runtime::PrefillMemoryAliasCondition::
                          kMutuallyExclusiveRouteOverlay &&
                  result.value->selected.requires_unbound_alias_or_route_contract,
              "C512/C8192 overlay remains conditional on route exclusion events");

  test.expect(scratch.gate_up.maximum_m == 8'192U &&
                  scratch.gate_up.n == 17'408U &&
                  scratch.gate_up.k == 5'120U &&
                  scratch.gate_up.logical_matrix_count == 2U &&
                  !scratch.gate_up.tactic_bound &&
                  scratch.down.maximum_m == 8'192U &&
                  scratch.down.n == 5'120U &&
                  scratch.down.k == 17'408U &&
                  scratch.down.logical_matrix_count == 1U &&
                  !scratch.down.tactic_bound,
              "Gate/Up and Down keep independent asymmetric shape fields");
}

void test_explicit_safe_upper_and_capacity_boundary(TestContext& test) {
  const auto fully_disjoint = build_plan(
      130'000U,
      runtime::PrefillHiddenStrategy::kDoublePromptWideConservative,
      runtime::PrefillOperatorScratchStrategy::
          kDisjointAllFamiliesAndLegacyC512);
  test.expect(fully_disjoint &&
                  fully_disjoint.value->selected.required_bytes ==
                      fully_disjoint.value->conservative.required_bytes &&
                  !fully_disjoint.value->selected
                       .requires_unbound_alias_or_route_contract &&
                  !fully_disjoint.value->executable(),
              "fully disjoint selection removes alias assumptions but not binding gates");

  constexpr std::uint64_t selected_130k = 10'818'244'608U;
  const auto exact = build_plan(130'000U,
                                runtime::PrefillHiddenStrategy::
                                    kSinglePromptWideConditional,
                                runtime::PrefillOperatorScratchStrategy::
                                    kC8192FamilyOverlayConditional,
                                selected_130k);
  const auto short_by_one = build_plan(
      130'000U,
      runtime::PrefillHiddenStrategy::kSinglePromptWideConditional,
      runtime::PrefillOperatorScratchStrategy::
          kC8192FamilyOverlayConditional,
      selected_130k - 1U);
  test.expect(exact && short_by_one &&
                  exact.value->selected.capacity ==
                      runtime::PrefillMemoryCapacityVerdict::
                          kFitsDeclaredLimit &&
                  short_by_one.value->selected.capacity ==
                      runtime::PrefillMemoryCapacityVerdict::
                          kExceedsDeclaredLimit,
              "selected request-arena capacity is exact at the byte boundary");
}

void test_fail_closed_inputs(TestContext& test) {
  runtime::LayerMajorPrefillWorkspaceOptions defaults;
  defaults.sequence_capacity_tokens = 40'000U;
  const auto unselected =
      runtime::build_unbound_layer_major_prefill_workspace_plan(defaults);
  test.expect(!unselected &&
                  unselected.error ==
                      runtime::PrefillWorkspacePlanError::kInvalidArgument,
              "unselected buffering and scratch strategies fail closed");
  test.expect(!build_plan(0U) &&
                  build_plan(0U).error ==
                      runtime::PrefillWorkspacePlanError::kInvalidArgument,
              "zero sequence capacity fails closed");
  test.expect(!build_plan(40'000U,
                          runtime::PrefillHiddenStrategy::
                              kSinglePromptWideConditional,
                          runtime::PrefillOperatorScratchStrategy::
                              kC8192FamilyOverlayConditional,
                          0U) &&
                  build_plan(40'000U,
                             runtime::PrefillHiddenStrategy::
                                 kSinglePromptWideConditional,
                             runtime::PrefillOperatorScratchStrategy::
                                 kC8192FamilyOverlayConditional,
                             0U)
                          .error == runtime::PrefillWorkspacePlanError::
                                        kInvalidArgument,
              "zero arena limit fails closed");
  test.expect(!build_plan(262'145U) &&
                  build_plan(262'145U).error ==
                      runtime::PrefillWorkspacePlanError::kCapacityExceeded,
              "capacity beyond the fixed Qwen3.6 contract fails closed");
  test.expect(!build_plan(std::numeric_limits<std::uint64_t>::max()) &&
                  build_plan(std::numeric_limits<std::uint64_t>::max()).error ==
                      runtime::PrefillWorkspacePlanError::kArithmeticOverflow,
              "dimension arithmetic overflow fails before narrowing");
}

}  // namespace

int main() {
  TestContext test;
  test_target_capacity_matrix(test);
  test_selected_dependency_plan(test);
  test_real_family_live_sets(test);
  test_legacy_overlay_and_shape_separation(test);
  test_explicit_safe_upper_and_capacity_boundary(test);
  test_fail_closed_inputs(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " prefill workspace plan test(s) failed\n";
    return 1;
  }
  std::cout << "All prefill workspace plan tests passed\n";
  return 0;
}
