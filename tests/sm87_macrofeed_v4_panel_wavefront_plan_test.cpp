#include "q3x/runtime/sm87_macrofeed_v4_panel_wavefront_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

namespace runtime = q3x::runtime;

struct Test final {
  bool ok = true;

  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ok = false;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
};

[[nodiscard]] bool has_issue(
    const runtime::Sm87MacroFeedV4PlanValidation& validation,
    const runtime::Sm87MacroFeedV4PlanIssue issue) {
  return runtime::has_sm87_macrofeed_v4_plan_issue(validation, issue);
}

}  // namespace

int main() {
  Test test;
  const auto plan =
      runtime::make_sm87_macrofeed_v4_p40_panel_wavefront_plan();
  const auto validation =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(plan);

  test.expect(validation.valid() &&
                  runtime::sm87_macrofeed_v4_p40_panel_wavefront_plan_valid(
                      plan),
              "canonical V4 host plan validates");
  test.expect(plan.candidate_id == runtime::kSm87MacroFeedV4CandidateId &&
                  plan.deployment_plan_id ==
                      runtime::kSm87MacroFeedV4P40DeploymentPlanId &&
                  plan.api.route_id == runtime::kSm87MacroFeedV4P40ApiRouteId,
              "candidate, AOT plan, and API P40 identities are independent");
  test.expect(plan.api.endpoint == "/v1/completions" &&
                  plan.api.prompt_tokens == 40'000U &&
                  plan.api.maximum_output_tokens == 1U &&
                  plan.api.batch_size == 1U && plan.api.cold_request &&
                  plan.api.prefix_cache_disabled &&
                  plan.api.kv_reuse_disabled &&
                  plan.api.full_prompt_consumption_required,
              "OpenAI P40 identity is exact cold/no-cache batch one");

  test.expect(plan.prompt_tokens == 40'000U &&
                  plan.panel_tokens == 8'000U && plan.panel_count == 5U &&
                  plan.panel_tokens * plan.panel_count == plan.prompt_tokens,
              "P40000 is exactly five C8000 panels");
  test.expect(plan.traversal ==
                      runtime::Sm87MacroFeedV4Traversal::
                          kPanelMajorLayerWavefront &&
                  plan.panel_loop_is_outermost &&
                  plan.layer_loop_is_natural_order_innermost,
              "panel-major wavefront traversal is explicit");

  test.expect(plan.workspace.transient_arena_bytes ==
                      runtime::kSm87MacroFeedV4TransientArenaBytes &&
                  plan.workspace.maximum_temporary_tokens == 8'000U &&
                  plan.workspace.ping_pong_hidden &&
                  plan.workspace.scratch_reused_by_phase &&
                  !plan.workspace.full_p40_temporary_plane_allowed &&
                  plan.workspace.persistent_kv_is_outside_transient_arena &&
                  plan.workspace
                      .persistent_conv_gdn_state_is_outside_transient_arena,
              "transient ownership is C8000-only and excludes persistent state");
  test.expect(
      plan.workspace.buffers[0U].role ==
              runtime::Sm87MacroFeedV4WorkspaceRole::kPingHidden &&
          plan.workspace.buffers[1U].role ==
              runtime::Sm87MacroFeedV4WorkspaceRole::kPongHidden &&
          plan.workspace.buffers[0U].storage_identity !=
              plan.workspace.buffers[1U].storage_identity &&
          plan.workspace.buffers[0U].bytes ==
              runtime::kSm87MacroFeedV4HiddenPanelBytes &&
          plan.workspace.buffers[1U].bytes ==
              runtime::kSm87MacroFeedV4HiddenPanelBytes &&
          plan.workspace.buffers[2U].bytes ==
              runtime::kSm87MacroFeedV4PanelScratchBytes,
      "ping, pong, and phase scratch have disjoint exact panel extents");
  for (const auto& buffer : plan.workspace.buffers) {
    test.expect(buffer.token_capacity == 8'000U && buffer.panel_local &&
                    buffer.reuse_waits_for_completion,
                "every temporary allocation is one reusable C8000 plane");
  }
  test.expect(
      runtime::kSm87MacroFeedV4AttentionLegacyLiveColumns == 24'576U &&
          runtime::kSm87MacroFeedV4AttentionLegacyLiveColumns >
              runtime::kSm87MacroFeedV4Intermediate &&
          runtime::kSm87MacroFeedV4GdnAliasedLiveColumns == 16'480U &&
          plan.phase_aliasing.attention_q_preprocess_overwrites_raw_q_gate &&
          plan.phase_aliasing.attention_online_core_reuses_processed_q &&
          plan.phase_aliasing.gdn_recurrent_reuses_consumed_qkv &&
          plan.phase_aliasing.gate_up_activation_owns_panel_scratch &&
          plan.phase_aliasing.every_phase_fits_one_panel_scratch,
      "phase aliases are mandatory because the incumbent Attention live set does not fit C8000 scratch");
  test.expect(
      plan.state_ownership.recurrent_epoch_bank_count == 2U &&
          plan.state_ownership.recurrent_epoch_bytes ==
              runtime::kSm87MacroFeedV4RecurrentEpochBytes &&
          plan.state_ownership.recurrent_storage_bytes ==
              runtime::kSm87MacroFeedV4RecurrentStorageBytes &&
          plan.state_ownership.active_recurrent_storage_identity != 0U &&
          plan.state_ownership.candidate_recurrent_storage_identity != 0U &&
          plan.state_ownership.active_recurrent_storage_identity !=
              plan.state_ownership.candidate_recurrent_storage_identity &&
          plan.state_ownership.private_kv_valid_end_storage_identity != 0U &&
          plan.state_ownership.panel_commit_event_identity != 0U &&
          plan.state_ownership.final_publish_event_identity != 0U &&
          plan.state_ownership.panel_commit_event_identity !=
              plan.state_ownership.final_publish_event_identity &&
          plan.state_ownership.private_kv_valid_end &&
          plan.state_ownership.candidate_epoch_copies_active_before_panel &&
          plan.state_ownership.active_candidate_swap_after_layer_63 &&
          plan.state_ownership.panel_failure_discards_candidate_epoch &&
          plan.state_ownership.canonical_recurrent_publish_after_final_panel &&
          plan.state_ownership.sequence_length_is_final_visibility_fence &&
          plan.state_ownership.no_fallible_work_after_sequence_publication,
      "private recurrent epochs and KV visibility isolate every fallible panel");

  std::size_t total_steps = 0U;
  std::size_t total_gdn_layers = 0U;
  std::size_t total_attention_layers = 0U;
  for (std::size_t panel_index = 0U;
       panel_index < plan.panels.size(); ++panel_index) {
    const auto& panel = plan.panels[panel_index];
    test.expect(panel.panel_index == panel_index &&
                    panel.token_begin == panel_index * 8'000U &&
                    panel.token_count == 8'000U &&
                    panel.sequence_begin == panel_index * 64U &&
                    panel.sequence_end == (panel_index + 1U) * 64U,
                "panel covers one exact contiguous C8000 token range");
    test.expect(
        panel.initial_workspace ==
                runtime::Sm87MacroFeedV4WorkspaceRole::kPingHidden &&
            panel.final_workspace ==
                runtime::Sm87MacroFeedV4WorkspaceRole::kPingHidden &&
            panel.embedding_publishes_initial_workspace &&
            panel.workspace_reuse_waits_for_panel_commit,
        "each panel begins and ends on the declared ping ownership epoch");

    std::size_t panel_gdn = 0U;
    std::size_t panel_attention = 0U;
    for (std::size_t layer = 0U; layer < panel.layers.size(); ++layer) {
      const auto& step = panel.layers[layer];
      const bool expected_gdn = ((layer + 1U) % 4U) != 0U;
      const auto expected_input =
          (layer % 2U) == 0U
              ? runtime::Sm87MacroFeedV4WorkspaceRole::kPingHidden
              : runtime::Sm87MacroFeedV4WorkspaceRole::kPongHidden;
      const auto expected_output =
          (layer % 2U) == 0U
              ? runtime::Sm87MacroFeedV4WorkspaceRole::kPongHidden
              : runtime::Sm87MacroFeedV4WorkspaceRole::kPingHidden;
      test.expect(step.panel_index == panel_index &&
                      step.layer_index == layer &&
                      step.sequence_ordinal == panel_index * 64U + layer &&
                      step.token_begin == panel_index * 8'000U &&
                      step.token_count == 8'000U,
                  "every panel advances layer 0 through 63 in natural order");
      test.expect(step.input_workspace == expected_input &&
                      step.output_workspace == expected_output &&
                      step.input_workspace != step.output_workspace &&
                      step.input_consumed_before_output_publication &&
                      step.output_reuse_waits_for_completion,
                  "layer ownership alternates the C8000 ping/pong pair");
      test.expect(
          step.state_write_mode ==
                  runtime::Sm87MacroFeedV4StateWriteMode::kPrivatePanelStage &&
              step.stages_kv == !expected_gdn &&
              step.stages_conv_state == expected_gdn &&
              step.stages_gdn_state == expected_gdn &&
              !step.publishes_state_to_decode,
          "layer state writes remain private to the panel transaction");
      panel_gdn += expected_gdn ? 1U : 0U;
      panel_attention += expected_gdn ? 0U : 1U;
      ++total_steps;
    }
    total_gdn_layers += panel_gdn;
    total_attention_layers += panel_attention;
    test.expect(panel_gdn == 48U && panel_attention == 16U,
                "every panel executes the exact 48 GDN / 16 Attention topology");

    const auto& transaction = panel.state_transaction;
    test.expect(
        transaction.panel_index == panel_index &&
            transaction.token_begin == panel_index * 8'000U &&
            transaction.token_end == (panel_index + 1U) * 8'000U &&
            transaction.incoming_state_epoch == panel_index &&
            transaction.outgoing_state_epoch == panel_index + 1U &&
            transaction.commit_dependency_sequence_ordinal ==
                (panel_index + 1U) * 64U - 1U,
        "panel transaction advances exactly one state epoch after layer 63");
    test.expect(
        transaction.kv_layer_count == 16U &&
            transaction.conv_layer_count == 48U &&
            transaction.gdn_layer_count == 48U &&
            transaction.kv_uses_disjoint_final_token_slice &&
            transaction.conv_and_gdn_use_private_next_epoch &&
            transaction.atomic_kv_conv_gdn_commit &&
            transaction.commit_after_layer_63 &&
            transaction.next_panel_waits_for_commit &&
            transaction.rollback_discards_uncommitted_panel_state &&
            transaction.state_private_to_prefill_until_request_commit &&
            !transaction.state_visible_to_decode,
        "KV, convolution, and GDN state commit atomically per panel");
  }
  test.expect(total_steps == 320U && total_gdn_layers == 240U &&
                  total_attention_layers == 80U,
              "five complete 64-layer panel wavefronts are represented");

  test.expect(plan.route.sm87_only && plan.route.real_checkpoint_required &&
                  plan.route.authenticated_aot_deployment_plan &&
                  plan.route.startup_bound_tactics &&
                  !plan.route.request_time_jit_allowed &&
                  !plan.route.request_time_repack_allowed &&
                  !plan.route.request_time_autotune_allowed &&
                  !plan.route.fallback_allowed &&
                  !plan.route.cublaslt_allowed && !plan.route.mtp_allowed &&
                  !plan.route.approximate_numerics_allowed,
              "AOT route excludes JIT, repack, autotune, fallback, cuBLASLt, MTP, and approximation");
  test.expect(plan.route.default_off && plan.route.test_only_contract &&
                  !plan.route.selector_bound && !plan.route.launcher_present &&
                  !plan.route.production_dispatch_eligible &&
                  !plan.route.numerical_qualification_complete,
              "host contract grants no selector, launcher, numerical, or production authority");

  auto malformed = plan;
  malformed.prompt_tokens = 39'999U;
  auto rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(rejected,
                        runtime::Sm87MacroFeedV4PlanIssue::kGeometry),
              "non-P40000 geometry is rejected");

  malformed = plan;
  malformed.panels[2U].token_begin += 1U;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(rejected,
                        runtime::Sm87MacroFeedV4PlanIssue::kTraversal) &&
                  rejected.first_bad_panel == 2U,
              "panel gap is rejected and attributed");

  malformed = plan;
  malformed.panels[1U].layers[17U].layer_index = 18U;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(rejected,
                        runtime::Sm87MacroFeedV4PlanIssue::kLayerSchedule) &&
                  rejected.first_bad_panel == 1U &&
                  rejected.first_bad_layer == 17U,
              "out-of-order layer is rejected and attributed");

  malformed = plan;
  malformed.workspace.full_p40_temporary_plane_allowed = true;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(
                  rejected,
                  runtime::Sm87MacroFeedV4PlanIssue::kWholeP40Temporary),
              "full-P40 temporary-plane permission is rejected");

  malformed = plan;
  malformed.workspace.buffers[0U].token_capacity = 40'000U;
  malformed.workspace.buffers[0U].bytes =
      static_cast<std::uint64_t>(40'000U) * 5'120U * 2U;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(
                  rejected,
                  runtime::Sm87MacroFeedV4PlanIssue::kWholeP40Temporary) &&
                  has_issue(rejected,
                            runtime::Sm87MacroFeedV4PlanIssue::kWorkspace),
              "materialized P40000 hidden plane is rejected");

  malformed = plan;
  malformed.workspace.buffers[1U].storage_identity =
      malformed.workspace.buffers[0U].storage_identity;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(rejected,
                        runtime::Sm87MacroFeedV4PlanIssue::kWorkspace),
              "ping/pong alias is rejected");

  malformed = plan;
  malformed.phase_aliasing.attention_q_preprocess_overwrites_raw_q_gate =
      false;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(rejected,
                        runtime::Sm87MacroFeedV4PlanIssue::kPhaseAliasing),
              "an unfused incumbent Attention live set is rejected");

  malformed = plan;
  malformed.state_ownership.recurrent_epoch_bank_count = 1U;
  malformed.state_ownership.candidate_recurrent_storage_identity =
      malformed.state_ownership.active_recurrent_storage_identity;
  malformed.state_ownership.no_fallible_work_after_sequence_publication =
      false;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(rejected,
                        runtime::Sm87MacroFeedV4PlanIssue::kStateOwnership),
              "single-bank recurrent mutation or post-publication failure is rejected");

  malformed = plan;
  malformed.panels[3U]
      .state_transaction.commit_dependency_sequence_ordinal -= 1U;
  malformed.panels[3U].state_transaction.atomic_kv_conv_gdn_commit = false;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(
                  rejected,
                  runtime::Sm87MacroFeedV4PlanIssue::kStateTransaction) &&
                  rejected.first_bad_panel == 3U,
              "early/non-atomic panel state commit is rejected");

  malformed = plan;
  malformed.route.request_time_jit_allowed = true;
  malformed.route.request_time_repack_allowed = true;
  malformed.route.request_time_autotune_allowed = true;
  malformed.route.fallback_allowed = true;
  malformed.route.cublaslt_allowed = true;
  malformed.route.mtp_allowed = true;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(rejected,
                        runtime::Sm87MacroFeedV4PlanIssue::kForbiddenRoute),
              "all forbidden request routes are rejected");

  malformed = plan;
  malformed.route.default_off = false;
  malformed.route.selector_bound = true;
  malformed.route.launcher_present = true;
  malformed.route.production_dispatch_eligible = true;
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(rejected,
                        runtime::Sm87MacroFeedV4PlanIssue::kDispatchBoundary),
              "selector or production authority cannot be synthesized");

  malformed = plan;
  malformed.api.route_id = "wrong-api-route";
  rejected =
      runtime::validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(malformed);
  test.expect(has_issue(rejected,
                        runtime::Sm87MacroFeedV4PlanIssue::kApiIdentity),
              "API P40 identity tamper is rejected");

  if (test.ok) {
    std::cout << "sm87_macrofeed_v4_panel_wavefront_plan_test: PASS\n";
  }
  return test.ok ? 0 : 1;
}
