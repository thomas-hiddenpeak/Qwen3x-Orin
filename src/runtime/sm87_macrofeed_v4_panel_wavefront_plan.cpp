#include "q3x/runtime/sm87_macrofeed_v4_panel_wavefront_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime {
namespace {

inline constexpr std::uint64_t kPingStorageIdentity =
    0x5133'4d46'5634'5001ULL;
inline constexpr std::uint64_t kPongStorageIdentity =
    0x5133'4d46'5634'5002ULL;
inline constexpr std::uint64_t kScratchStorageIdentity =
    0x5133'4d46'5634'5003ULL;
inline constexpr std::uint64_t kActiveRecurrentStorageIdentity =
    0x5133'4d46'5634'5301ULL;
inline constexpr std::uint64_t kCandidateRecurrentStorageIdentity =
    0x5133'4d46'5634'5302ULL;
inline constexpr std::uint64_t kPrivateKvValidEndStorageIdentity =
    0x5133'4d46'5634'5303ULL;
inline constexpr std::uint64_t kPanelCommitEventIdentity =
    0x5133'4d46'5634'4501ULL;
inline constexpr std::uint64_t kFinalPublishEventIdentity =
    0x5133'4d46'5634'4502ULL;

[[nodiscard]] constexpr bool magic_equal(
    const std::array<std::uint8_t, 8U>& left,
    const std::array<std::uint8_t, 8U>& right) noexcept {
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (left[index] != right[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool ranges_overlap(
    const Sm87MacroFeedV4TransientBuffer& left,
    const Sm87MacroFeedV4TransientBuffer& right) noexcept {
  return left.offset < right.end() && right.offset < left.end();
}

void add_issue(Sm87MacroFeedV4PlanValidation* const validation,
               const Sm87MacroFeedV4PlanIssue issue,
               const std::size_t panel = kSm87MacroFeedV4PanelCount,
               const std::size_t layer = kSm87MacroFeedV4LayerCount) noexcept {
  validation->issue_mask |= static_cast<std::uint32_t>(issue);
  if (validation->first_bad_panel == kSm87MacroFeedV4PanelCount &&
      panel < kSm87MacroFeedV4PanelCount) {
    validation->first_bad_panel = panel;
  }
  if (validation->first_bad_layer == kSm87MacroFeedV4LayerCount &&
      layer < kSm87MacroFeedV4LayerCount) {
    validation->first_bad_layer = layer;
  }
}

[[nodiscard]] constexpr Sm87MacroFeedV4WorkspaceRole input_workspace(
    const std::size_t layer) noexcept {
  return (layer % 2U) == 0U ? Sm87MacroFeedV4WorkspaceRole::kPingHidden
                            : Sm87MacroFeedV4WorkspaceRole::kPongHidden;
}

[[nodiscard]] constexpr Sm87MacroFeedV4WorkspaceRole output_workspace(
    const std::size_t layer) noexcept {
  return (layer % 2U) == 0U ? Sm87MacroFeedV4WorkspaceRole::kPongHidden
                            : Sm87MacroFeedV4WorkspaceRole::kPingHidden;
}

[[nodiscard]] constexpr bool same_buffer(
    const Sm87MacroFeedV4TransientBuffer& actual,
    const Sm87MacroFeedV4TransientBuffer& expected) noexcept {
  return actual.role == expected.role &&
         actual.storage_identity == expected.storage_identity &&
         actual.offset == expected.offset && actual.bytes == expected.bytes &&
         actual.token_capacity == expected.token_capacity &&
         actual.row_width == expected.row_width &&
         actual.panel_local == expected.panel_local &&
         actual.reuse_waits_for_completion ==
             expected.reuse_waits_for_completion;
}

}  // namespace

Sm87MacroFeedV4PanelWavefrontPlan
make_sm87_macrofeed_v4_p40_panel_wavefront_plan() noexcept {
  Sm87MacroFeedV4PanelWavefrontPlan plan;
  plan.magic = kSm87MacroFeedV4PanelWavefrontMagic;
  plan.abi_major = kSm87MacroFeedV4PanelWavefrontAbiMajor;
  plan.abi_minor = kSm87MacroFeedV4PanelWavefrontAbiMinor;
  plan.candidate_id = kSm87MacroFeedV4CandidateId;
  plan.deployment_plan_id = kSm87MacroFeedV4P40DeploymentPlanId;

  plan.api.route_id = kSm87MacroFeedV4P40ApiRouteId;
  plan.api.endpoint = kSm87MacroFeedV4P40Endpoint;
  plan.api.served_model = kSm87MacroFeedV4P40Model;
  plan.api.prompt_tokens = kSm87MacroFeedV4P40Tokens;
  plan.api.maximum_output_tokens = 1U;
  plan.api.batch_size = 1U;
  plan.api.openai_compatible = true;
  plan.api.exact_token_ids = true;
  plan.api.cold_request = true;
  plan.api.prefix_cache_disabled = true;
  plan.api.kv_reuse_disabled = true;
  plan.api.streaming_first_committed_token = true;
  plan.api.full_prompt_consumption_required = true;

  plan.traversal = Sm87MacroFeedV4Traversal::kPanelMajorLayerWavefront;
  plan.prompt_tokens = kSm87MacroFeedV4P40Tokens;
  plan.panel_tokens = kSm87MacroFeedV4PanelTokens;
  plan.panel_count = kSm87MacroFeedV4PanelCount;
  plan.layer_count = kSm87MacroFeedV4LayerCount;

  plan.workspace.buffers[0U] = {
      Sm87MacroFeedV4WorkspaceRole::kPingHidden,
      kPingStorageIdentity,
      0U,
      kSm87MacroFeedV4HiddenPanelBytes,
      kSm87MacroFeedV4PanelTokens,
      kSm87MacroFeedV4Hidden,
      true,
      true,
  };
  plan.workspace.buffers[1U] = {
      Sm87MacroFeedV4WorkspaceRole::kPongHidden,
      kPongStorageIdentity,
      kSm87MacroFeedV4HiddenPanelBytes,
      kSm87MacroFeedV4HiddenPanelBytes,
      kSm87MacroFeedV4PanelTokens,
      kSm87MacroFeedV4Hidden,
      true,
      true,
  };
  plan.workspace.buffers[2U] = {
      Sm87MacroFeedV4WorkspaceRole::kPanelScratch,
      kScratchStorageIdentity,
      2U * kSm87MacroFeedV4HiddenPanelBytes,
      kSm87MacroFeedV4PanelScratchBytes,
      kSm87MacroFeedV4PanelTokens,
      kSm87MacroFeedV4Intermediate,
      true,
      true,
  };
  plan.workspace.transient_arena_bytes =
      kSm87MacroFeedV4TransientArenaBytes;
  plan.workspace.maximum_temporary_tokens =
      kSm87MacroFeedV4PanelTokens;
  plan.workspace.ping_pong_hidden = true;
  plan.workspace.scratch_reused_by_phase = true;
  plan.workspace.full_p40_temporary_plane_allowed = false;
  plan.workspace.persistent_kv_is_outside_transient_arena = true;
  plan.workspace.persistent_conv_gdn_state_is_outside_transient_arena = true;

  plan.phase_aliasing.attention_q_preprocess_overwrites_raw_q_gate = true;
  plan.phase_aliasing.attention_online_core_reuses_processed_q = true;
  plan.phase_aliasing.gdn_recurrent_reuses_consumed_qkv = true;
  plan.phase_aliasing.gate_up_activation_owns_panel_scratch = true;
  plan.phase_aliasing.every_phase_fits_one_panel_scratch = true;

  plan.state_ownership.recurrent_epoch_bank_count = 2U;
  plan.state_ownership.recurrent_epoch_bytes =
      kSm87MacroFeedV4RecurrentEpochBytes;
  plan.state_ownership.recurrent_storage_bytes =
      kSm87MacroFeedV4RecurrentStorageBytes;
  plan.state_ownership.active_recurrent_storage_identity =
      kActiveRecurrentStorageIdentity;
  plan.state_ownership.candidate_recurrent_storage_identity =
      kCandidateRecurrentStorageIdentity;
  plan.state_ownership.private_kv_valid_end_storage_identity =
      kPrivateKvValidEndStorageIdentity;
  plan.state_ownership.panel_commit_event_identity =
      kPanelCommitEventIdentity;
  plan.state_ownership.final_publish_event_identity =
      kFinalPublishEventIdentity;
  plan.state_ownership.private_kv_valid_end = true;
  plan.state_ownership.candidate_epoch_copies_active_before_panel = true;
  plan.state_ownership.active_candidate_swap_after_layer_63 = true;
  plan.state_ownership.panel_failure_discards_candidate_epoch = true;
  plan.state_ownership.canonical_recurrent_publish_after_final_panel = true;
  plan.state_ownership.sequence_length_is_final_visibility_fence = true;
  plan.state_ownership.no_fallible_work_after_sequence_publication = true;

  for (std::size_t panel_index = 0U;
       panel_index < kSm87MacroFeedV4PanelCount; ++panel_index) {
    auto& panel = plan.panels[panel_index];
    panel.panel_index = panel_index;
    panel.token_begin = panel_index * kSm87MacroFeedV4PanelTokens;
    panel.token_count = kSm87MacroFeedV4PanelTokens;
    panel.sequence_begin = panel_index * kSm87MacroFeedV4LayerCount;
    panel.sequence_end = panel.sequence_begin + kSm87MacroFeedV4LayerCount;
    panel.initial_workspace = Sm87MacroFeedV4WorkspaceRole::kPingHidden;
    panel.final_workspace = Sm87MacroFeedV4WorkspaceRole::kPingHidden;
    panel.embedding_publishes_initial_workspace = true;
    panel.workspace_reuse_waits_for_panel_commit = true;

    for (std::size_t layer = 0U; layer < kSm87MacroFeedV4LayerCount;
         ++layer) {
      auto& step = panel.layers[layer];
      step.panel_index = panel_index;
      step.layer_index = layer;
      step.sequence_ordinal = panel.sequence_begin + layer;
      step.token_begin = panel.token_begin;
      step.token_count = kSm87MacroFeedV4PanelTokens;
      step.layer_kind = sm87_macrofeed_v4_expected_layer_kind(layer);
      step.input_workspace = input_workspace(layer);
      step.output_workspace = output_workspace(layer);
      step.state_write_mode =
          Sm87MacroFeedV4StateWriteMode::kPrivatePanelStage;
      step.input_consumed_before_output_publication = true;
      step.output_reuse_waits_for_completion = true;
      step.stages_kv =
          step.layer_kind == Sm87MacroFeedV4LayerKind::kFullAttention;
      step.stages_conv_state =
          step.layer_kind == Sm87MacroFeedV4LayerKind::kGdn;
      step.stages_gdn_state =
          step.layer_kind == Sm87MacroFeedV4LayerKind::kGdn;
      step.publishes_state_to_decode = false;
    }

    auto& transaction = panel.state_transaction;
    transaction.panel_index = panel_index;
    transaction.token_begin = panel.token_begin;
    transaction.token_end = panel.token_begin + panel.token_count;
    transaction.incoming_state_epoch = panel_index;
    transaction.outgoing_state_epoch = panel_index + 1U;
    transaction.commit_dependency_sequence_ordinal = panel.sequence_end - 1U;
    transaction.kv_layer_count =
        kSm87MacroFeedV4FullAttentionLayerCount;
    transaction.conv_layer_count = kSm87MacroFeedV4GdnLayerCount;
    transaction.gdn_layer_count = kSm87MacroFeedV4GdnLayerCount;
    transaction.kv_uses_disjoint_final_token_slice = true;
    transaction.conv_and_gdn_use_private_next_epoch = true;
    transaction.atomic_kv_conv_gdn_commit = true;
    transaction.commit_after_layer_63 = true;
    transaction.next_panel_waits_for_commit = true;
    transaction.rollback_discards_uncommitted_panel_state = true;
    transaction.state_private_to_prefill_until_request_commit = true;
    transaction.state_visible_to_decode = false;
  }

  plan.route.sm87_only = true;
  plan.route.real_checkpoint_required = true;
  plan.route.authenticated_aot_deployment_plan = true;
  plan.route.startup_bound_tactics = true;
  plan.route.request_time_jit_allowed = false;
  plan.route.request_time_repack_allowed = false;
  plan.route.request_time_autotune_allowed = false;
  plan.route.fallback_allowed = false;
  plan.route.cublaslt_allowed = false;
  plan.route.mtp_allowed = false;
  plan.route.approximate_numerics_allowed = false;
  plan.route.default_off = true;
  plan.route.test_only_contract = true;
  plan.route.selector_bound = false;
  plan.route.launcher_present = false;
  plan.route.production_dispatch_eligible = false;
  plan.route.numerical_qualification_complete = false;

  plan.panel_loop_is_outermost = true;
  plan.layer_loop_is_natural_order_innermost = true;
  plan.final_request_commit_after_all_panels = true;
  plan.partial_panel_commit_visible_to_decode = false;
  return plan;
}

Sm87MacroFeedV4PlanValidation
validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(
    const Sm87MacroFeedV4PanelWavefrontPlan& plan) noexcept {
  Sm87MacroFeedV4PlanValidation validation;
  const auto expected = make_sm87_macrofeed_v4_p40_panel_wavefront_plan();

  if (!magic_equal(plan.magic, kSm87MacroFeedV4PanelWavefrontMagic) ||
      plan.abi_major != kSm87MacroFeedV4PanelWavefrontAbiMajor ||
      plan.abi_minor != kSm87MacroFeedV4PanelWavefrontAbiMinor ||
      plan.candidate_id != kSm87MacroFeedV4CandidateId ||
      plan.deployment_plan_id != kSm87MacroFeedV4P40DeploymentPlanId) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kIdentity);
  }

  const auto& api = plan.api;
  if (api.route_id != kSm87MacroFeedV4P40ApiRouteId ||
      api.endpoint != kSm87MacroFeedV4P40Endpoint ||
      api.served_model != kSm87MacroFeedV4P40Model ||
      api.prompt_tokens != kSm87MacroFeedV4P40Tokens ||
      api.maximum_output_tokens != 1U || api.batch_size != 1U ||
      !api.openai_compatible || !api.exact_token_ids ||
      !api.cold_request || !api.prefix_cache_disabled ||
      !api.kv_reuse_disabled || !api.streaming_first_committed_token ||
      !api.full_prompt_consumption_required) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kApiIdentity);
  }

  if (plan.prompt_tokens != kSm87MacroFeedV4P40Tokens ||
      plan.panel_tokens != kSm87MacroFeedV4PanelTokens ||
      plan.panel_count != kSm87MacroFeedV4PanelCount ||
      plan.layer_count != kSm87MacroFeedV4LayerCount ||
      plan.panel_count * plan.panel_tokens != plan.prompt_tokens) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kGeometry);
  }

  if (plan.traversal !=
          Sm87MacroFeedV4Traversal::kPanelMajorLayerWavefront ||
      !plan.panel_loop_is_outermost ||
      !plan.layer_loop_is_natural_order_innermost ||
      !plan.final_request_commit_after_all_panels ||
      plan.partial_panel_commit_visible_to_decode) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kTraversal);
  }

  const auto& workspace = plan.workspace;
  bool workspace_bad =
      workspace.transient_arena_bytes !=
          kSm87MacroFeedV4TransientArenaBytes ||
      workspace.maximum_temporary_tokens != kSm87MacroFeedV4PanelTokens ||
      !workspace.ping_pong_hidden || !workspace.scratch_reused_by_phase ||
      !workspace.persistent_kv_is_outside_transient_arena ||
      !workspace.persistent_conv_gdn_state_is_outside_transient_arena;
  for (std::size_t index = 0U; index < workspace.buffers.size(); ++index) {
    const auto& buffer = workspace.buffers[index];
    workspace_bad |= !same_buffer(buffer, expected.workspace.buffers[index]);
    workspace_bad |= buffer.storage_identity == 0U ||
                     buffer.bytes == 0U || !buffer.panel_local ||
                     !buffer.reuse_waits_for_completion ||
                     buffer.offset > workspace.transient_arena_bytes ||
                     buffer.bytes >
                         workspace.transient_arena_bytes - buffer.offset;
    const std::uint64_t full_prompt_bytes =
        static_cast<std::uint64_t>(kSm87MacroFeedV4P40Tokens) *
        buffer.row_width * kSm87MacroFeedV4Bf16Bytes;
    if (buffer.token_capacity > kSm87MacroFeedV4PanelTokens ||
        (buffer.row_width != 0U && buffer.bytes >= full_prompt_bytes)) {
      add_issue(&validation,
                Sm87MacroFeedV4PlanIssue::kWholeP40Temporary);
    }
  }
  for (std::size_t first = 0U; first < workspace.buffers.size(); ++first) {
    for (std::size_t second = first + 1U;
         second < workspace.buffers.size(); ++second) {
      workspace_bad |=
          workspace.buffers[first].storage_identity ==
              workspace.buffers[second].storage_identity ||
          ranges_overlap(workspace.buffers[first], workspace.buffers[second]);
    }
  }
  if (workspace.full_p40_temporary_plane_allowed ||
      workspace.maximum_temporary_tokens > kSm87MacroFeedV4PanelTokens) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kWholeP40Temporary);
  }
  if (workspace_bad) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kWorkspace);
  }

  const auto& phase_aliasing = plan.phase_aliasing;
  if (!phase_aliasing.attention_q_preprocess_overwrites_raw_q_gate ||
      !phase_aliasing.attention_online_core_reuses_processed_q ||
      !phase_aliasing.gdn_recurrent_reuses_consumed_qkv ||
      !phase_aliasing.gate_up_activation_owns_panel_scratch ||
      !phase_aliasing.every_phase_fits_one_panel_scratch) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kPhaseAliasing);
  }

  const auto& state_ownership = plan.state_ownership;
  if (state_ownership.recurrent_epoch_bank_count != 2U ||
      state_ownership.recurrent_epoch_bytes !=
          kSm87MacroFeedV4RecurrentEpochBytes ||
      state_ownership.recurrent_storage_bytes !=
          kSm87MacroFeedV4RecurrentStorageBytes ||
      state_ownership.active_recurrent_storage_identity !=
          expected.state_ownership.active_recurrent_storage_identity ||
      state_ownership.candidate_recurrent_storage_identity !=
          expected.state_ownership.candidate_recurrent_storage_identity ||
      state_ownership.active_recurrent_storage_identity ==
          state_ownership.candidate_recurrent_storage_identity ||
      state_ownership.private_kv_valid_end_storage_identity !=
          expected.state_ownership.private_kv_valid_end_storage_identity ||
      state_ownership.panel_commit_event_identity !=
          expected.state_ownership.panel_commit_event_identity ||
      state_ownership.final_publish_event_identity !=
          expected.state_ownership.final_publish_event_identity ||
      state_ownership.panel_commit_event_identity ==
          state_ownership.final_publish_event_identity ||
      !state_ownership.private_kv_valid_end ||
      !state_ownership.candidate_epoch_copies_active_before_panel ||
      !state_ownership.active_candidate_swap_after_layer_63 ||
      !state_ownership.panel_failure_discards_candidate_epoch ||
      !state_ownership.canonical_recurrent_publish_after_final_panel ||
      !state_ownership.sequence_length_is_final_visibility_fence ||
      !state_ownership.no_fallible_work_after_sequence_publication) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kStateOwnership);
  }

  for (std::size_t panel_index = 0U;
       panel_index < kSm87MacroFeedV4PanelCount; ++panel_index) {
    const auto& panel = plan.panels[panel_index];
    const std::size_t expected_token_begin =
        panel_index * kSm87MacroFeedV4PanelTokens;
    const std::size_t expected_sequence_begin =
        panel_index * kSm87MacroFeedV4LayerCount;
    if (panel.panel_index != panel_index ||
        panel.token_begin != expected_token_begin ||
        panel.token_count != kSm87MacroFeedV4PanelTokens ||
        panel.sequence_begin != expected_sequence_begin ||
        panel.sequence_end !=
            expected_sequence_begin + kSm87MacroFeedV4LayerCount ||
        panel.initial_workspace !=
            Sm87MacroFeedV4WorkspaceRole::kPingHidden ||
        panel.final_workspace !=
            Sm87MacroFeedV4WorkspaceRole::kPingHidden ||
        !panel.embedding_publishes_initial_workspace ||
        !panel.workspace_reuse_waits_for_panel_commit) {
      add_issue(&validation, Sm87MacroFeedV4PlanIssue::kTraversal,
                panel_index);
    }

    std::size_t gdn_layers = 0U;
    std::size_t attention_layers = 0U;
    for (std::size_t layer = 0U; layer < kSm87MacroFeedV4LayerCount;
         ++layer) {
      const auto& step = panel.layers[layer];
      const auto expected_kind =
          sm87_macrofeed_v4_expected_layer_kind(layer);
      const bool gdn = expected_kind == Sm87MacroFeedV4LayerKind::kGdn;
      gdn_layers += gdn ? 1U : 0U;
      attention_layers += gdn ? 0U : 1U;
      if (step.panel_index != panel_index || step.layer_index != layer ||
          step.sequence_ordinal != expected_sequence_begin + layer ||
          step.token_begin != expected_token_begin ||
          step.token_count != kSm87MacroFeedV4PanelTokens ||
          step.layer_kind != expected_kind ||
          step.input_workspace != input_workspace(layer) ||
          step.output_workspace != output_workspace(layer) ||
          step.input_workspace == step.output_workspace ||
          step.state_write_mode !=
              Sm87MacroFeedV4StateWriteMode::kPrivatePanelStage ||
          !step.input_consumed_before_output_publication ||
          !step.output_reuse_waits_for_completion ||
          step.stages_kv != !gdn || step.stages_conv_state != gdn ||
          step.stages_gdn_state != gdn ||
          step.publishes_state_to_decode) {
        add_issue(&validation, Sm87MacroFeedV4PlanIssue::kLayerSchedule,
                  panel_index, layer);
      }
    }
    if (gdn_layers != kSm87MacroFeedV4GdnLayerCount ||
        attention_layers != kSm87MacroFeedV4FullAttentionLayerCount) {
      add_issue(&validation, Sm87MacroFeedV4PlanIssue::kLayerSchedule,
                panel_index);
    }

    const auto& transaction = panel.state_transaction;
    if (transaction.panel_index != panel_index ||
        transaction.token_begin != expected_token_begin ||
        transaction.token_end !=
            expected_token_begin + kSm87MacroFeedV4PanelTokens ||
        transaction.incoming_state_epoch != panel_index ||
        transaction.outgoing_state_epoch != panel_index + 1U ||
        transaction.commit_dependency_sequence_ordinal !=
            expected_sequence_begin + kSm87MacroFeedV4LayerCount - 1U ||
        transaction.kv_layer_count !=
            kSm87MacroFeedV4FullAttentionLayerCount ||
        transaction.conv_layer_count != kSm87MacroFeedV4GdnLayerCount ||
        transaction.gdn_layer_count != kSm87MacroFeedV4GdnLayerCount ||
        !transaction.kv_uses_disjoint_final_token_slice ||
        !transaction.conv_and_gdn_use_private_next_epoch ||
        !transaction.atomic_kv_conv_gdn_commit ||
        !transaction.commit_after_layer_63 ||
        !transaction.next_panel_waits_for_commit ||
        !transaction.rollback_discards_uncommitted_panel_state ||
        !transaction.state_private_to_prefill_until_request_commit ||
        transaction.state_visible_to_decode) {
      add_issue(&validation, Sm87MacroFeedV4PlanIssue::kStateTransaction,
                panel_index);
    }
  }

  const auto& route = plan.route;
  if (!route.sm87_only || !route.real_checkpoint_required ||
      !route.authenticated_aot_deployment_plan ||
      !route.startup_bound_tactics || route.request_time_jit_allowed ||
      route.request_time_repack_allowed ||
      route.request_time_autotune_allowed || route.fallback_allowed ||
      route.cublaslt_allowed || route.mtp_allowed ||
      route.approximate_numerics_allowed) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kForbiddenRoute);
  }
  if (!route.default_off || !route.test_only_contract ||
      route.selector_bound || route.launcher_present ||
      route.production_dispatch_eligible ||
      route.numerical_qualification_complete) {
    add_issue(&validation, Sm87MacroFeedV4PlanIssue::kDispatchBoundary);
  }
  return validation;
}

bool sm87_macrofeed_v4_p40_panel_wavefront_plan_valid(
    const Sm87MacroFeedV4PanelWavefrontPlan& plan) noexcept {
  return validate_sm87_macrofeed_v4_p40_panel_wavefront_plan(plan).valid();
}

}  // namespace q3x::runtime
