#include "q3x/runtime/prefill_workspace_plan.h"

#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"
#include "q3x/kernels/sm87_nvfp4_marlin.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

namespace runtime = q3x::runtime;
namespace kernels = q3x::kernels;

static_assert(runtime::kLayerMajorPrefillGdnC64NativeWorkspaceBytes ==
              kernels::kGdnPrefillChunk64NativeWorkspaceBytes);

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
        runtime::kMaximumRequestArenaBytes,
    const runtime::PrefillGdnPhysicalTactic gdn_tactic =
        runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv,
    const runtime::PrefillLegacyGdnPhysicalTactic legacy_gdn_tactic =
        runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite,
    const runtime::PrefillMlpPhysicalTactic mlp_tactic =
        runtime::PrefillMlpPhysicalTactic::kSeparateGateUpAndSilu) {
  runtime::LayerMajorPrefillWorkspaceOptions options;
  options.sequence_capacity_tokens = sequence_capacity;
  options.request_arena_limit_bytes = arena_limit;
  options.hidden_strategy = hidden;
  options.scratch_strategy = scratch;
  options.gdn_tactic = gdn_tactic;
  options.legacy_gdn_tactic = legacy_gdn_tactic;
  options.mlp_tactic = mlp_tactic;
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
      {40'000U, 3'975'374'848U, 5'324'963'840U, true},
      {60'000U, 5'496'014'848U, 7'052'323'840U, true},
      {130'000U, 10'818'254'848U, 13'098'083'840U, true},
      {131'072U, 10'899'761'152U, 13'190'670'336U, true},
      {262'144U, 20'865'427'456U, 24'511'096'832U, false},
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

  test.expect(plan.final_hidden_handoff_bf16.required_bytes == 10'240U &&
                  plan.final_hidden_handoff_bf16.element_capacity == 5'120U &&
                  plan.final_hidden_handoff_bf16.element_size_bytes == 2U &&
                  plan.final_hidden_handoff_bf16.lifetime ==
                      runtime::PrefillMemoryLifetime::kRequestPrefill &&
                  plan.final_hidden_handoff_bf16.alias_condition ==
                      runtime::PrefillMemoryAliasCondition::kDisjoint,
              "final hidden handoff is a stable independent BF16 slot");

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
      expected_bytes{446'365'696U, 402'653'184U, 855'638'016U};
  constexpr std::array<runtime::PrefillScratchFamily,
                       runtime::kLayerMajorPrefillScratchFamilyCount>
      expected_families{
          runtime::PrefillScratchFamily::kGdnProjectionAndRecurrentCore,
          runtime::PrefillScratchFamily::kFullAttentionProjectionAndCore,
          runtime::PrefillScratchFamily::kMlpGateUpAndDown};

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
      scratch.gdn_tactic ==
              runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv &&
          scratch.legacy_gdn_tactic ==
              runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite &&
          scratch.mlp_tactic ==
              runtime::PrefillMlpPhysicalTactic::kSeparateGateUpAndSilu &&
          scratch.gdn_projection_phase.required_bytes == 353'894'400U &&
          scratch.gdn_recurrent_phase.required_bytes == 446'365'696U &&
          scratch.gdn_c64_native_workspace.required_bytes == 75'694'080U &&
          !scratch.gdn_token_parallel_c512_conv_output.has_value() &&
          scratch.full_attention_preprocess_phase.required_bytes ==
              402'653'184U &&
          scratch.mlp_gate_up_down_phase.required_bytes == 855'638'016U &&
          scratch.shared_projection_reduction_and_locks.required_bytes ==
              1'048'832U,
      "physical tactics expose exact GDN phases, Attention preprocess, MLP, "
      "and shared Marlin workspace without treating test-only native as "
      "legacy Release state");
  test.expect(
      scratch.c8192_family_live_sets[0].producer ==
              runtime::PrefillScratchProducer::
                  kGdnInProjQkvzAndInProjBa &&
          scratch.c8192_family_live_sets[0].last_consumer ==
              runtime::PrefillScratchLastConsumer::
                  kGdnRecurrentUpdateNormAndOutputProjection &&
          scratch.c8192_family_live_sets[2].producer ==
              runtime::PrefillScratchProducer::
                  kMlpMergedGateUpProjection &&
          scratch.c8192_family_live_sets[2].last_consumer ==
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
                  fully_disjoint.value->selected
                       .requires_unbound_alias_or_route_contract &&
                  !fully_disjoint.value->executable(),
              "disjoint families retain explicit intra-family phase binding gates");

  constexpr std::uint64_t selected_130k = 10'818'254'848U;
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

void test_physical_tactic_matrix(TestContext& test) {
  const auto token_parallel = build_plan(
      130'000U,
      runtime::PrefillHiddenStrategy::kSinglePromptWideConditional,
      runtime::PrefillOperatorScratchStrategy::
          kC8192FamilyOverlayConditional,
      runtime::kMaximumRequestArenaBytes,
      runtime::PrefillGdnPhysicalTactic::kC64NativeTokenParallelConv);
  test.expect(
      token_parallel &&
          token_parallel.value->operator_scratch.gdn_recurrent_phase
                  .required_bytes == 456'851'456U &&
          token_parallel.value->operator_scratch
                  .gdn_token_parallel_c512_conv_output.has_value() &&
          token_parallel.value->operator_scratch
                  .gdn_token_parallel_c512_conv_output->required_bytes ==
              10'485'760U &&
          token_parallel.value->operator_scratch.c8192_disjoint_families
                  .total_required_bytes == 1'715'142'656U &&
          token_parallel.value->selected.required_bytes ==
              10'818'254'848U &&
          token_parallel.value->conservative.required_bytes ==
              13'108'569'600U,
      "token-parallel C64 tactic adds exactly one reusable C512 conv output "
      "without multiplying native workspace by logical panel segments");

  const auto native_legacy = build_plan(
      130'000U,
      runtime::PrefillHiddenStrategy::kDoublePromptWideConservative,
      runtime::PrefillOperatorScratchStrategy::
          kDisjointAllFamiliesAndLegacyC512,
      runtime::kMaximumRequestArenaBytes,
      runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv,
      runtime::PrefillLegacyGdnPhysicalTactic::kC64Native);
  test.expect(
      native_legacy &&
          native_legacy.value->operator_scratch
                  .legacy_c512_gdn_native_workspace.has_value() &&
          native_legacy.value->operator_scratch
                  .legacy_c512_gdn_native_workspace->required_bytes ==
              75'694'080U &&
          native_legacy.value->operator_scratch.legacy_c512_only
                  .total_required_bytes == 175'304'192U &&
          native_legacy.value->selected.required_bytes ==
              13'173'777'920U &&
          native_legacy.value->conservative.required_bytes ==
              13'173'777'920U &&
          native_legacy.value->operator_scratch.disjoint_conservative
                  .gdn_native_workspace_backing.identity ==
              runtime::PrefillWorkspaceBackingIdentity::
                  kDisjointC8192LegacyGdnNativeArenas &&
          native_legacy.value->operator_scratch.disjoint_conservative
                  .gdn_native_workspace_backing.minimum_instance_count == 2U,
      "literal disjoint C8192 and test-only native legacy routes count two "
      "independent C64 workspaces");

  const auto fused_mlp = build_plan(
      130'000U,
      runtime::PrefillHiddenStrategy::kSinglePromptWideConditional,
      runtime::PrefillOperatorScratchStrategy::
          kC8192FamilyOverlayConditional,
      runtime::kMaximumRequestArenaBytes,
      runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv,
      runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite,
      runtime::PrefillMlpPhysicalTactic::kFusedGateUpEpilogue);
  test.expect(
      fused_mlp &&
          fused_mlp.value->operator_scratch.mlp_gate_up_down_phase
                  .required_bytes == 940'572'928U &&
          fused_mlp.value->operator_scratch
                  .c8192_family_overlay_conditional.total_required_bytes ==
              940'572'928U &&
          fused_mlp.value->selected.required_bytes == 10'903'189'760U &&
          fused_mlp.value->conservative.required_bytes == 13'183'018'752U,
      "fused GateUp epilogue cannot inherit the smaller separate-SiLU live "
      "set");
}

void test_request_state_overlay_with_disjoint_legacy(TestContext& test) {
  struct Case {
    std::uint64_t tokens;
    std::uint64_t selected_bytes;
    std::uint64_t conservative_bytes;
  };
  constexpr std::array<Case, 3U> cases{{
      {40'000U, 4'066'344'960U, 5'324'963'840U},
      {60'000U, 5'588'904'960U, 7'052'323'840U},
      {130'000U, 10'917'864'960U, 13'098'083'840U},
  }};

  for (const Case& current : cases) {
    const auto result = build_plan(
        current.tokens,
        runtime::PrefillHiddenStrategy::kSinglePromptWideConditional,
        runtime::PrefillOperatorScratchStrategy::
            kC8192FamilyOverlayWithDisjointLegacyC512);
    const bool exact =
        result && result.value->selected.required_bytes ==
                      current.selected_bytes &&
        result.value->conservative.required_bytes ==
            current.conservative_bytes &&
        result.value->operator_scratch
                .c8192_family_overlay_with_disjoint_legacy_c512
                .total_required_bytes ==
            result.value->operator_scratch
                    .c8192_family_overlay_conditional.total_required_bytes +
                result.value->operator_scratch.legacy_c512_only
                    .total_required_bytes &&
        !result.value->operator_scratch
             .c8192_family_overlay_with_disjoint_legacy_c512
             .requires_legacy_route_exclusion &&
        !result.value->operator_scratch
             .c8192_family_overlay_with_disjoint_legacy_c512
             .requires_route_mutual_exclusion_event &&
        result.value->operator_scratch
            .c8192_family_overlay_with_disjoint_legacy_c512
            .requires_family_completion_events;
    test.expect(exact,
                "RequestState strategy overlays C8192 families but keeps "
                "legacy C512 physically disjoint");
  }
}

void test_layer_wide_p40_mlp_memory_plan(TestContext& test) {
  const auto candidate = build_plan(
      runtime::kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens,
      runtime::PrefillHiddenStrategy::kSinglePromptWideConditional,
      runtime::PrefillOperatorScratchStrategy::
          kC8192FamilyOverlayConditional,
      runtime::kMaximumRequestArenaBytes,
      runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv,
      runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite,
      runtime::PrefillMlpPhysicalTactic::kLayerWideP40AlignedMarlin);
  if (!runtime::layer_wide_p40_mlp_prefill_plan_enabled()) {
    test.expect(
        !candidate &&
            candidate.error ==
                runtime::PrefillWorkspacePlanError::kInvalidArgument,
        "production/default workspace planning rejects layer-wide P40 MLP");
    return;
  }

  constexpr std::uint64_t kFullMThreeBf16Spans =
      3U * runtime::kLayerMajorPrefillLayerWideMlpP40Tokens * 17'408U * 2U;
  static_assert(kFullMThreeBf16Spans == 4'177'920'000U);
  test.expect(
      candidate &&
          candidate.value->operator_scratch.mlp_tactic ==
              runtime::PrefillMlpPhysicalTactic::
                  kLayerWideP40AlignedMarlin &&
          candidate.value->operator_scratch.mlp_capacity_tokens == 40'000U &&
          candidate.value->operator_scratch.mlp_gate_up_down_phase
                  .required_bytes == kFullMThreeBf16Spans &&
          candidate.value->operator_scratch
                  .c8192_family_overlay_conditional.total_required_bytes ==
              kFullMThreeBf16Spans &&
          candidate.value->operator_scratch.gate_up.maximum_m == 40'000U &&
          candidate.value->operator_scratch.down.maximum_m == 40'000U &&
          candidate.value->selected.required_bytes == 7'297'733'120U &&
          candidate.value->conservative.required_bytes == 8'647'332'608U &&
          candidate.value->selected.capacity ==
              runtime::PrefillMemoryCapacityVerdict::kFitsDeclaredLimit &&
          !candidate.value->request_arena_reservation_bound &&
          !candidate.value->operator_bindings_complete &&
          !candidate.value->executable(),
      "P40 reserves three exact prompt-wide BF16 MLP spans while leaving "
      "allocation and launch bindings unbound");

  const auto marlin_parity = build_plan(
      runtime::kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens,
      runtime::PrefillHiddenStrategy::kSinglePromptWideConditional,
      runtime::PrefillOperatorScratchStrategy::
          kC8192FamilyOverlayConditional,
      runtime::kMaximumRequestArenaBytes,
      runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv,
      runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite,
      runtime::PrefillMlpPhysicalTactic::
          kLayerWideP40MarlinParityMergedGateUp);
  constexpr std::uint64_t kParityMlpLiveSet = kFullMThreeBf16Spans;
  static_assert(kParityMlpLiveSet == 4'177'920'000U);
  if (!runtime::
          prompt_wide_p40_vllm_marlin_parity_prefill_plan_enabled()) {
    test.expect(
        !marlin_parity &&
            marlin_parity.error ==
                runtime::PrefillWorkspacePlanError::kInvalidArgument,
        "old layer-wide admission does not enable Marlin parity implicitly");
  } else {
    test.expect(
        marlin_parity &&
            marlin_parity.value->operator_scratch.mlp_tactic ==
                runtime::PrefillMlpPhysicalTactic::
                    kLayerWideP40MarlinParityMergedGateUp &&
            marlin_parity.value->operator_scratch.mlp_capacity_tokens ==
                40'000U &&
            marlin_parity.value->operator_scratch.mlp_gate_up_down_phase
                    .required_bytes == kParityMlpLiveSet &&
            marlin_parity.value->operator_scratch
                    .c8192_family_overlay_conditional.total_required_bytes ==
                kParityMlpLiveSet &&
            marlin_parity.value->selected.required_bytes == 7'297'733'120U &&
            marlin_parity.value->conservative.required_bytes ==
                8'647'332'608U &&
            !marlin_parity.value->executable(),
        "Marlin parity owns merged GateUp plus independent activation without "
        "adding Ctmp, locks, or reduction workspace to its MLP live set");
  }

  const auto persistent = build_plan(
      runtime::kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens,
      runtime::PrefillHiddenStrategy::kSinglePromptWideConditional,
      runtime::PrefillOperatorScratchStrategy::
          kC8192FamilyOverlayConditional,
      runtime::kMaximumRequestArenaBytes,
      runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv,
      runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite,
      runtime::PrefillMlpPhysicalTactic::
          kLayerWideP40PersistentFusedGateUp);
  constexpr std::uint64_t kPersistentNormalizedAndActivated =
      runtime::kLayerMajorPrefillLayerWideMlpP40Tokens *
      (5'120U + 17'408U) * 2U;
  static_assert(kPersistentNormalizedAndActivated == 1'802'240'000U);
  test.expect(
      persistent &&
          persistent.value->operator_scratch.mlp_capacity_tokens == 40'000U &&
          persistent.value->operator_scratch.mlp_gate_up_down_phase
                  .required_bytes == kPersistentNormalizedAndActivated &&
          persistent.value->operator_scratch
                  .c8192_family_overlay_conditional.total_required_bytes ==
              kPersistentNormalizedAndActivated &&
          persistent.value->selected.required_bytes == 4'922'053'120U &&
          persistent.value->conservative.required_bytes == 6'271'652'608U &&
          !persistent.value->executable(),
      "target persistent P40 ownership materializes only normalized and "
      "activated spans and reuses dead normalized storage for Down output");

  test.expect(
      !build_plan(39'968U,
                  runtime::PrefillHiddenStrategy::
                      kSinglePromptWideConditional,
                  runtime::PrefillOperatorScratchStrategy::
                      kC8192FamilyOverlayConditional,
                  runtime::kMaximumRequestArenaBytes,
                  runtime::PrefillGdnPhysicalTactic::
                      kC64NativeInPlaceConv,
                  runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite,
                  runtime::PrefillMlpPhysicalTactic::
                      kLayerWideP40AlignedMarlin) &&
          !build_plan(40'064U,
                      runtime::PrefillHiddenStrategy::
                          kSinglePromptWideConditional,
                      runtime::PrefillOperatorScratchStrategy::
                          kC8192FamilyOverlayConditional,
                      runtime::kMaximumRequestArenaBytes,
                      runtime::PrefillGdnPhysicalTactic::
                          kC64NativeInPlaceConv,
                      runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite,
                      runtime::PrefillMlpPhysicalTactic::
                          kLayerWideP40AlignedMarlin) &&
          !build_plan(39'968U,
                      runtime::PrefillHiddenStrategy::
                          kSinglePromptWideConditional,
                      runtime::PrefillOperatorScratchStrategy::
                          kC8192FamilyOverlayConditional,
                      runtime::kMaximumRequestArenaBytes,
                      runtime::PrefillGdnPhysicalTactic::
                          kC64NativeInPlaceConv,
                      runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite,
                      runtime::PrefillMlpPhysicalTactic::
                          kLayerWideP40MarlinParityMergedGateUp),
      "layer-wide MLP memory cannot be generalized beyond exact P40000");
}

void test_machine_readable_workspace_backings(TestContext& test) {
  constexpr std::uint64_t maximum_reduction_bytes =
      kernels::kSm87NvFp4MarlinReductionBytes >
              kernels::kSm87Fp8MarlinReductionBytes
          ? kernels::kSm87NvFp4MarlinReductionBytes
          : kernels::kSm87Fp8MarlinReductionBytes;
  constexpr std::uint64_t maximum_lock_bytes =
      kernels::kSm87NvFp4MarlinLockBytes >
              kernels::kSm87Fp8MarlinLockBytes
          ? kernels::kSm87NvFp4MarlinLockBytes
          : kernels::kSm87Fp8MarlinLockBytes;
  constexpr std::uint64_t shared_bytes =
      (maximum_reduction_bytes + maximum_lock_bytes + 255U) & ~255U;
  static_assert(shared_bytes == 1'048'832U);

  const auto baseline = build_plan(130'000U);
  test.expect(baseline.ok(), "machine-readable backing baseline builds");
  if (!baseline) {
    return;
  }
  const auto& scratch = baseline.value->operator_scratch;
  const auto& family_overlay = scratch.c8192_family_overlay_conditional;
  test.expect(
      scratch.prompt_token_ids_u32.element_capacity == 8'192U &&
          scratch.prompt_token_ids_u32.element_size_bytes == 4U &&
          scratch.prompt_token_ids_u32.required_bytes == 32'768U &&
          scratch.prompt_token_ids_u32.alias_condition ==
              runtime::PrefillMemoryAliasCondition::
                  kPromptTokenIdsConsumedBeforeOperatorScratchReuse &&
          family_overlay.requires_prompt_token_ids_consumed_event &&
          family_overlay.prompt_token_ids_backing.identity ==
              runtime::PrefillWorkspaceBackingIdentity::
                  kOperatorArenaPromptTokenPrefix &&
          family_overlay.prompt_token_ids_backing.phase_ownership ==
              runtime::PrefillWorkspacePhaseOwnership::
                  kPromptTokenIdsBeforeOperatorFamilies &&
          family_overlay.prompt_token_ids_backing.bytes_per_instance ==
              32'768U &&
          family_overlay.prompt_token_ids_backing.minimum_instance_count ==
              1U,
      "panel token IDs own one operator-prefix instance until embedding "
      "gather completes");
  test.expect(
      scratch.shared_projection_reduction_and_locks.required_bytes ==
              shared_bytes &&
          family_overlay.projection_workspace_backing.identity ==
              runtime::PrefillWorkspaceBackingIdentity::
                  kC8192SequentialFamilyPhaseArena &&
          family_overlay.projection_workspace_backing.bytes_per_instance ==
              shared_bytes &&
          family_overlay.projection_workspace_backing.minimum_instance_count ==
              1U &&
          scratch.c8192_disjoint_families.projection_workspace_backing
                  .identity ==
              runtime::PrefillWorkspaceBackingIdentity::
                  kC8192DisjointFamilyPhaseArenas &&
          scratch.c8192_disjoint_families.projection_workspace_backing
                  .minimum_instance_count == 3U,
      "Marlin reduction/locks derive from public kernel ABI and expose "
      "sequential versus disjoint backing counts");
  test.expect(
      family_overlay.gdn_native_workspace_backing.identity ==
              runtime::PrefillWorkspaceBackingIdentity::
                  kC8192GdnNativeArena &&
          family_overlay.gdn_native_workspace_backing.bytes_per_instance ==
              kernels::kGdnPrefillChunk64NativeWorkspaceBytes &&
          family_overlay.gdn_native_workspace_backing.minimum_instance_count ==
              1U &&
          !scratch.legacy_c512_gdn_native_workspace.has_value() &&
          scratch.legacy_c512_only.gdn_native_workspace_backing
                  .minimum_instance_count == 0U,
      "default C8192 native owns one ABI-sized workspace while Release C16 "
      "legacy owns none");

  const auto route_native = build_plan(
      130'000U,
      runtime::PrefillHiddenStrategy::kSinglePromptWideConditional,
      runtime::PrefillOperatorScratchStrategy::
          kOverlayLegacyC512MutuallyExclusive,
      runtime::kMaximumRequestArenaBytes,
      runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv,
      runtime::PrefillLegacyGdnPhysicalTactic::kC64Native);
  test.expect(
      route_native &&
          route_native.value->selected.required_bytes == 10'818'254'848U &&
          route_native.value->operator_scratch.mutually_exclusive_overlay
                  .gdn_native_workspace_backing.identity ==
              runtime::PrefillWorkspaceBackingIdentity::
                  kMutuallyExclusiveC8192LegacyGdnNativeArena &&
          route_native.value->operator_scratch.mutually_exclusive_overlay
                  .gdn_native_workspace_backing.minimum_instance_count == 1U &&
          route_native.value->operator_scratch.mutually_exclusive_overlay
              .requires_route_mutual_exclusion_event,
      "native legacy route overlay records exactly one shared native backing "
      "behind the route event");
}

void test_p40_whole_core_workspace_plan(TestContext& test) {
  const auto result =
      runtime::build_unbound_layer_major_p40_whole_core_workspace_plan();
  test.expect(result.ok(), "exact P40 whole-core workspace plan builds");
  if (!result) {
    return;
  }
  const runtime::LayerMajorP40WholeCoreWorkspacePlan& plan = *result.value;
  test.expect(
      plan.prompt_token_count == 40'000U &&
          plan.request_sequence_capacity_tokens == 40'001U &&
          plan.logical_panel_capacity_tokens == 8'000U &&
          plan.logical_panel_count == 5U &&
          plan.persistent_and_kv.required_bytes == 2'699'952'128U &&
          plan.prompt_residual_bf16.required_bytes == 409'610'240U &&
          plan.whole_core_family_arena.required_bytes ==
              runtime::kLayerMajorP40WholeCoreFamilyArenaBytes &&
          plan.legacy_c512_workspace.required_bytes == 90'970'112U &&
          plan.final_hidden_handoff_bf16.required_bytes == 10'240U &&
          plan.rope_cos_sin_fp32.required_bytes == 10'240'256U &&
          plan.required_bytes == 8'640'542'976U &&
          plan.capacity ==
              runtime::PrefillMemoryCapacityVerdict::kFitsDeclaredLimit &&
          !plan.request_arena_reservation_bound &&
          !plan.lifetime_alias_contracts_bound &&
          !plan.operator_bindings_complete && !plan.executable(),
      "whole-core planner fixes the exact six-part request arena ledger");

  test.expect(
      plan.linear_raw_qkv_bf16.family_relative_offset == 0U &&
          plan.linear_raw_qkv_bf16.memory.required_bytes == 819'200'000U &&
          plan.linear_conv_qkv_bf16.family_relative_offset == 819'200'000U &&
          plan.linear_conv_qkv_bf16.memory.required_bytes == 819'200'000U &&
          plan.linear_z_bf16.family_relative_offset == 1'638'400'000U &&
          plan.linear_z_bf16.memory.required_bytes == 491'520'000U &&
          plan.linear_a_bf16.family_relative_offset == 2'129'920'000U &&
          plan.linear_a_bf16.memory.required_bytes == 3'840'000U &&
          plan.linear_b_bf16.family_relative_offset == 2'133'760'000U &&
          plan.linear_b_bf16.memory.required_bytes == 3'840'000U &&
          plan.linear_prompt_wide_workspace.family_relative_offset ==
              2'137'600'000U &&
          plan.linear_prompt_wide_workspace.memory.required_bytes ==
              2'800'640'000U &&
          plan.linear_output_bf16.family_relative_offset ==
              4'938'240'000U &&
          plan.linear_output_bf16.memory.required_bytes == 491'520'000U,
      "whole-core planner records the exact linear/GDN family offsets");

  test.expect(
      runtime::kLayerMajorP40MarlinParityMergedGateUpOffset == 0U &&
          runtime::kLayerMajorP40MarlinParityMergedGateUpFeatures ==
              34'816U &&
          runtime::kLayerMajorP40MarlinParityMergedGateUpRowStrideElements ==
              34'816U &&
          runtime::kLayerMajorP40MarlinParityMergedGateUpRowStrideBytes ==
              69'632U &&
          runtime::kLayerMajorP40MarlinParityUpColumnOffsetElements ==
              17'408U &&
          runtime::kLayerMajorP40MarlinParityUpColumnOffsetBytes == 34'816U &&
          runtime::kLayerMajorP40MarlinParityMergedGateUpBytes ==
              2'785'280'000U &&
          runtime::kLayerMajorP40MarlinParityActivatedOffset ==
              2'785'280'000U &&
          runtime::kLayerMajorP40MarlinParityActivatedBytes ==
              1'392'640'000U &&
          runtime::kLayerMajorP40MarlinParityTemporaryOffset ==
              4'177'920'000U &&
          runtime::kLayerMajorP40MarlinParityReductionWorkspaceOffset ==
              runtime::kLayerMajorP40MarlinParityTemporaryOffset &&
          runtime::kLayerMajorP40MarlinParityReductionWorkspaceBytes ==
              1'048'576U &&
          runtime::kLayerMajorP40MarlinParityLockBytes == 64U &&
          runtime::kLayerMajorP40MarlinParityTemporaryPayloadBytes ==
              1'048'576U &&
          runtime::kLayerMajorP40MarlinParityTemporaryBytes ==
              1'048'832U &&
          runtime::kLayerMajorP40MarlinParityTemporaryOffset +
                  runtime::kLayerMajorP40MarlinParityTemporaryBytes <=
              runtime::kLayerMajorP40MarlinParityNormalizedOffset &&
          runtime::kLayerMajorP40MarlinParityNormalizedOffset +
                  runtime::kLayerMajorP40MarlinParityNormalizedBytes <=
              plan.whole_core_family_arena.required_bytes,
      "whole-core high-water contains canonical GateThenUp, activation, "
      "stock tail temporary, and normalized/Down-branch ranges");

  test.expect(
      plan.prompt_token_ids_u32.family_relative_offset == 2'137'600'000U &&
          plan.prompt_token_ids_u32.memory.required_bytes == 160'000U &&
          plan.prompt_token_ids_u32.memory.alias_condition ==
              runtime::PrefillMemoryAliasCondition::
                  kPromptTokenIdsConsumedBeforeOperatorScratchReuse &&
          plan.prompt_token_ids_u32.family_relative_offset !=
              plan.linear_raw_qkv_bf16.family_relative_offset,
      "prompt IDs stage inside the pre-GDN workspace lifetime, never raw-QKV");

  runtime::LayerMajorP40WholeCoreWorkspaceOptions one_byte_short;
  one_byte_short.request_arena_limit_bytes = 8'640'542'975U;
  const auto limited =
      runtime::build_unbound_layer_major_p40_whole_core_workspace_plan(
          one_byte_short);
  test.expect(
      limited &&
          limited.value->capacity == runtime::PrefillMemoryCapacityVerdict::
                                         kExceedsDeclaredLimit &&
          limited.value->required_bytes == 8'640'542'976U,
      "one-byte-short arena preserves arithmetic and reports explicit overflow");

  runtime::LayerMajorP40WholeCoreWorkspaceOptions wrong = one_byte_short;
  wrong.request_arena_limit_bytes = runtime::kMaximumRequestArenaBytes;
  wrong.logical_panel_capacity_tokens = 8'192U;
  const auto wrong_panel =
      runtime::build_unbound_layer_major_p40_whole_core_workspace_plan(wrong);
  wrong.logical_panel_capacity_tokens = 8'000U;
  wrong.prompt_token_count = 39'999U;
  const auto wrong_prompt =
      runtime::build_unbound_layer_major_p40_whole_core_workspace_plan(wrong);
  test.expect(
      !wrong_panel && !wrong_prompt &&
          wrong_panel.error ==
              runtime::PrefillWorkspacePlanError::kInvalidArgument &&
          wrong_prompt.error ==
              runtime::PrefillWorkspacePlanError::kInvalidArgument,
      "whole-core planner rejects C8192 and every non-P40000 geometry");
}

void test_fail_closed_inputs(TestContext& test) {
  runtime::LayerMajorPrefillWorkspaceOptions defaults;
  defaults.sequence_capacity_tokens = 40'000U;
  const auto unselected =
      runtime::build_unbound_layer_major_prefill_workspace_plan(defaults);
  test.expect(!unselected &&
                  unselected.error ==
                      runtime::PrefillWorkspacePlanError::kInvalidArgument,
              "unselected strategies and physical tactics fail closed");

  runtime::LayerMajorPrefillWorkspaceOptions missing_tactic;
  missing_tactic.sequence_capacity_tokens = 40'000U;
  missing_tactic.hidden_strategy =
      runtime::PrefillHiddenStrategy::kSinglePromptWideConditional;
  missing_tactic.scratch_strategy = runtime::PrefillOperatorScratchStrategy::
      kC8192FamilyOverlayConditional;
  missing_tactic.gdn_tactic =
      runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv;
  missing_tactic.legacy_gdn_tactic =
      runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite;
  const auto no_mlp_tactic =
      runtime::build_unbound_layer_major_prefill_workspace_plan(
          missing_tactic);
  test.expect(!no_mlp_tactic &&
                  no_mlp_tactic.error ==
                      runtime::PrefillWorkspacePlanError::kInvalidArgument,
              "an omitted physical tactic keeps the byte total undefined");
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
  test_physical_tactic_matrix(test);
  test_request_state_overlay_with_disjoint_legacy(test);
  test_layer_wide_p40_mlp_memory_plan(test);
  test_machine_readable_workspace_backings(test);
  test_p40_whole_core_workspace_plan(test);
  test_fail_closed_inputs(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " prefill workspace plan test(s) failed\n";
    return 1;
  }
  std::cout << "All prefill workspace plan tests passed\n";
  return 0;
}
