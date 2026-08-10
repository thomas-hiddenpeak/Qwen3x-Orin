#include "q3x/runtime/prefill_execution_plan.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

[[nodiscard]] PrefillExecutionPlanResult plan_failure(
    const PrefillExecutionPlanError error) noexcept {
  PrefillExecutionPlanResult result;
  result.error = error;
  return result;
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] model::LayerType layer_type_for_index(
    const std::size_t layer_index) noexcept {
  return (layer_index + 1U) % 4U == 0U
             ? model::LayerType::kFullAttention
             : model::LayerType::kLinearAttention;
}

[[nodiscard]] constexpr bool layer_wide_p40_mlp_build_enabled() noexcept {
#if defined(Q3X_ENABLE_LAYER_WIDE_P40_MLP_ADMISSION)
  return true;
#else
  return false;
#endif
}

[[nodiscard]] constexpr bool prompt_wide_p40_whole_core_build_enabled()
    noexcept {
#if defined(Q3X_ENABLE_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION)
  return true;
#else
  return false;
#endif
}

[[nodiscard]] constexpr bool prompt_wide_p40_projection_reset_build_enabled()
    noexcept {
#if defined(Q3X_ENABLE_P40_PROJECTION_RESET_ADMISSION)
  return true;
#else
  return false;
#endif
}

[[nodiscard]] bool supported_mlp_schedule_tactic(
    const LayerMajorPrefillMlpScheduleTactic tactic) noexcept {
  return tactic ==
             LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel ||
         (tactic == LayerMajorPrefillMlpScheduleTactic::
                        kLayerWideP40ExactFullM &&
          layer_wide_p40_mlp_build_enabled()) ||
         (tactic == LayerMajorPrefillMlpScheduleTactic::
                        kPromptWideP40WholeCore &&
          prompt_wide_p40_whole_core_build_enabled()) ||
         (tactic == LayerMajorPrefillMlpScheduleTactic::
                        kPromptWideP40ProjectionReset &&
          prompt_wide_p40_projection_reset_build_enabled());
}

[[nodiscard]] bool valid_whole_core_schedule(
    const PrefillExecutionPlan& plan) noexcept {
  const PrefillWholeCoreSchedulePlan& schedule = plan.whole_core_schedule;
  const bool whole_core =
      plan.mlp_schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore;
  if (!whole_core) {
    return !schedule.enabled &&
           schedule.fill_panel_phase_count_per_layer == 0U &&
           schedule.prompt_core_phase_count_per_layer == 0U &&
           schedule.drain_panel_phase_count_per_layer == 0U &&
           schedule.persistent_mlp_phase_count_per_layer == 0U &&
           schedule.panel_token_count == 0U &&
           schedule.prompt_core_token_count == 0U &&
           schedule.request_capacity_tokens == 0U &&
           schedule.route_pass_count == 0U &&
           !schedule.fp8_single_launch_per_projection_required &&
           !schedule.bf16_ab_prompt_wide_required &&
           !schedule.gdn_prompt_wide_required &&
           !schedule.flashinfer_whole_prompt_required;
  }
  return schedule.enabled &&
         plan.first_position == 0U &&
         plan.prompt_token_count == kLayerMajorPrefillPromptWideP40Tokens &&
         plan.final_position == kLayerMajorPrefillPromptWideP40Tokens &&
         plan.panel_count == kLayerMajorPrefillPromptWideP40PanelCount &&
         schedule.fill_panel_phase_count_per_layer == plan.panel_count &&
         schedule.prompt_core_phase_count_per_layer == 1U &&
         schedule.drain_panel_phase_count_per_layer == plan.panel_count &&
         schedule.persistent_mlp_phase_count_per_layer == 1U &&
         schedule.panel_token_count ==
             kLayerMajorPrefillPromptWideP40PanelTokens &&
         schedule.prompt_core_token_count ==
             kLayerMajorPrefillPromptWideP40Tokens &&
         schedule.request_capacity_tokens ==
             kLayerMajorPrefillPromptWideP40RequestCapacityTokens &&
         schedule.route_pass_count == 1U &&
         schedule.fp8_single_launch_per_projection_required &&
         schedule.bf16_ab_prompt_wide_required &&
         schedule.gdn_prompt_wide_required &&
         schedule.flashinfer_whole_prompt_required;
}

[[nodiscard]] bool valid_projection_reset_schedule(
    const PrefillExecutionPlan& plan) noexcept {
  const PrefillP40ProjectionResetSchedulePlan& schedule =
      plan.projection_reset_schedule;
  const bool projection_reset =
      plan.mlp_schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40ProjectionReset;
  if (!projection_reset) {
    return !schedule.enabled &&
           schedule.input_preparation_panel_count_per_layer == 0U &&
           schedule.prompt_core_phase_count_per_layer == 0U &&
           schedule.persistent_mlp_phase_count_per_layer == 0U &&
           schedule.panel_token_count == 0U &&
           schedule.projection_m_tokens == 0U &&
           schedule.request_capacity_tokens == 0U &&
           schedule.route_pass_count == 0U &&
           schedule.fp8_grouped_input_launches_per_layer == 0U &&
           schedule.fp8_output_launches_per_layer == 0U &&
           schedule.fp8_physical_launches_per_request == 0U &&
           schedule.fp8_tensor_role_hits_per_request == 0U &&
           schedule.nvfp4_gate_up_launches_per_layer == 0U &&
           schedule.nvfp4_down_launches_per_layer == 0U &&
           schedule.nvfp4_physical_launches_per_request == 0U &&
           !schedule.fp8_grouped_full_prompt_input_required &&
           !schedule.fp8_full_prompt_output_required &&
           !schedule.nvfp4_full_prompt_required &&
           !schedule.internal_m_segmentation_forbidden &&
           !schedule.production_accuracy_required &&
           !schedule.approximate_numerics_forbidden &&
           !schedule.mtp_forbidden && !schedule.cublaslt_forbidden;
  }
  return schedule.enabled && plan.first_position == 0U &&
         plan.prompt_token_count == kLayerMajorPrefillPromptWideP40Tokens &&
         plan.final_position == kLayerMajorPrefillPromptWideP40Tokens &&
         plan.panel_count == kLayerMajorPrefillPromptWideP40PanelCount &&
         schedule.input_preparation_panel_count_per_layer == plan.panel_count &&
         schedule.prompt_core_phase_count_per_layer == 1U &&
         schedule.persistent_mlp_phase_count_per_layer == 1U &&
         schedule.panel_token_count ==
             kLayerMajorPrefillPromptWideP40PanelTokens &&
         schedule.projection_m_tokens ==
             kLayerMajorPrefillPromptWideP40Tokens &&
         schedule.request_capacity_tokens ==
             kLayerMajorPrefillPromptWideP40RequestCapacityTokens &&
         schedule.route_pass_count == 1U &&
         schedule.fp8_grouped_input_launches_per_layer ==
             kLayerMajorPrefillProjectionResetFp8GroupedInputLaunchesPerLayer &&
         schedule.fp8_output_launches_per_layer ==
             kLayerMajorPrefillProjectionResetFp8OutputLaunchesPerLayer &&
         schedule.fp8_physical_launches_per_request ==
             kLayerMajorPrefillProjectionResetFp8PhysicalLaunchesPerRequest &&
         schedule.fp8_tensor_role_hits_per_request ==
             kLayerMajorPrefillProjectionResetFp8TensorRoleHitsPerRequest &&
         schedule.nvfp4_gate_up_launches_per_layer ==
             kLayerMajorPrefillProjectionResetNvFp4GateUpLaunchesPerLayer &&
         schedule.nvfp4_down_launches_per_layer ==
             kLayerMajorPrefillProjectionResetNvFp4DownLaunchesPerLayer &&
         schedule.nvfp4_physical_launches_per_request ==
             kLayerMajorPrefillProjectionResetNvFp4PhysicalLaunchesPerRequest &&
         schedule.fp8_grouped_full_prompt_input_required &&
         schedule.fp8_full_prompt_output_required &&
         schedule.nvfp4_full_prompt_required &&
         schedule.internal_m_segmentation_forbidden &&
         schedule.production_accuracy_required &&
         schedule.approximate_numerics_forbidden && schedule.mtp_forbidden &&
         schedule.cublaslt_forbidden;
}

[[nodiscard]] bool valid_mlp_schedule(
    const PrefillExecutionPlan& plan) noexcept {
  const PrefillMlpSchedulePlan& schedule = plan.mlp_schedule;
  if (!supported_mlp_schedule_tactic(schedule.tactic) ||
      schedule.operator_panel_phase_count_per_layer != plan.panel_count ||
      !schedule.post_attention_residual_completed_panelwise) {
    return false;
  }

  if (schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel) {
    return schedule.mlp_phase_submission_count_per_layer == plan.panel_count &&
           schedule.maximum_m_per_mlp_submission ==
               kLayerMajorPrefillOperatorPanelTokens &&
           schedule.required_gate_up_projection_launches_per_layer == 0U &&
           schedule.maximum_standalone_silu_launches_per_layer == 0U &&
           schedule.required_down_projection_launches_per_layer == 0U &&
           schedule.minimum_total_kernel_launches_per_layer == 0U &&
           schedule.maximum_total_kernel_launches_per_layer == 0U &&
           !schedule.waits_for_all_operator_panels &&
           !schedule.post_attention_norm_is_prompt_wide &&
           !schedule.exact_full_m_binding_required &&
           !schedule.internal_m_segmentation_forbidden;
  }

  const bool layer_wide_mlp_only =
      schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kLayerWideP40ExactFullM;
  const bool whole_core =
      schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore;
  const bool projection_reset =
      schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40ProjectionReset;
  const bool fused_full_prompt = whole_core || projection_reset;
  return (layer_wide_mlp_only || fused_full_prompt) &&
         plan.first_position == 0U &&
         plan.prompt_token_count ==
             kLayerMajorPrefillLayerWideMlpP40Tokens &&
         plan.final_position == kLayerMajorPrefillLayerWideMlpP40Tokens &&
         plan.prompt_token_count %
                 kLayerMajorPrefillLayerWideMlpAlignmentTokens ==
             0U &&
         schedule.mlp_phase_submission_count_per_layer == 1U &&
         schedule.maximum_m_per_mlp_submission ==
             kLayerMajorPrefillLayerWideMlpP40Tokens &&
         schedule.required_gate_up_projection_launches_per_layer == 1U &&
         schedule.maximum_standalone_silu_launches_per_layer ==
             (fused_full_prompt ? 0U : 1U) &&
         schedule.required_down_projection_launches_per_layer == 1U &&
         schedule.minimum_total_kernel_launches_per_layer == 2U &&
         schedule.maximum_total_kernel_launches_per_layer ==
             (fused_full_prompt ? 2U : 3U) &&
         schedule.waits_for_all_operator_panels &&
         schedule.post_attention_norm_is_prompt_wide &&
         schedule.exact_full_m_binding_required &&
         schedule.internal_m_segmentation_forbidden;
}

[[nodiscard]] bool valid_plan_topology(
    const PrefillExecutionPlan& plan) noexcept {
  if (plan.traversal != PrefillTraversalOrder::kLayerMajor ||
      plan.legacy_public_tile_limit !=
          kLayerMajorPrefillLegacyPublicTileTokens ||
      plan.operator_panel_capacity !=
          kLayerMajorPrefillOperatorPanelTokens ||
      plan.prompt_token_count == 0U || plan.panel_count == 0U ||
      plan.panel_count > plan.panels.size() ||
      plan.prompt_token_count >
          kLayerMajorPrefillMaximumSequenceTokens ||
      plan.final_position <= plan.first_position ||
      plan.final_position > kLayerMajorPrefillMaximumSequenceTokens ||
      plan.final_position - plan.first_position !=
          plan.prompt_token_count ||
      plan.final_commit.expected_initial_sequence_length !=
          plan.first_position ||
      plan.final_commit.committed_sequence_length != plan.final_position ||
      plan.final_commit.commit_count != 1U ||
      plan.operator_bindings_complete || !valid_mlp_schedule(plan) ||
      !valid_whole_core_schedule(plan) ||
      !valid_projection_reset_schedule(plan)) {
    return false;
  }

  const bool whole_core =
      plan.mlp_schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore;
  const bool projection_reset =
      plan.mlp_schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40ProjectionReset;
  const bool fixed_p40_geometry = whole_core || projection_reset;
  const std::size_t expected_panel_count =
      fixed_p40_geometry
          ? kLayerMajorPrefillPromptWideP40PanelCount
          : (static_cast<std::size_t>(plan.prompt_token_count) +
             kLayerMajorPrefillOperatorPanelTokens - 1U) /
                kLayerMajorPrefillOperatorPanelTokens;
  if (plan.panel_count != expected_panel_count) {
    return false;
  }

  std::uint32_t next_position = plan.first_position;
  std::size_t remaining_tokens = plan.prompt_token_count;
  for (std::size_t panel_index = 0U; panel_index < plan.panel_count;
       ++panel_index) {
    const PrefillOperatorPanel& panel = plan.panels[panel_index];
    const std::uint32_t expected_token_count =
        fixed_p40_geometry
            ? kLayerMajorPrefillPromptWideP40PanelTokens
            : static_cast<std::uint32_t>(
                  next_layer_major_prefill_operator_panel_token_count(
                      remaining_tokens));
    if (panel.ordinal != panel_index || panel.first_position != next_position ||
        panel.token_count != expected_token_count ||
        panel.end_position <= panel.first_position ||
        panel.end_position - panel.first_position != panel.token_count) {
      return false;
    }
    next_position = panel.end_position;
    remaining_tokens -= panel.token_count;
  }
  if (remaining_tokens != 0U || next_position != plan.final_position) {
    return false;
  }

  std::size_t linear_layers = 0U;
  std::size_t full_layers = 0U;
  for (std::size_t layer_index = 0U; layer_index < plan.layers.size();
       ++layer_index) {
    const PrefillLayerExecution& layer = plan.layers[layer_index];
    const model::LayerType expected = layer_type_for_index(layer_index);
    const PrefillProgressDomain expected_progress =
        expected == model::LayerType::kFullAttention
            ? PrefillProgressDomain::kKvCache
            : PrefillProgressDomain::kGdnState;
    if (layer.layer_index != layer_index || layer.layer_type != expected ||
        layer.progress_domain != expected_progress ||
        layer.panel_count != plan.panel_count) {
      return false;
    }
    if (expected == model::LayerType::kFullAttention) {
      ++full_layers;
    } else {
      ++linear_layers;
    }
  }
  return linear_layers == kLayerMajorPrefillLinearLayerCount &&
         full_layers == kLayerMajorPrefillFullLayerCount;
}

[[nodiscard]] bool execution_complete(
    const PrefillExecutionPlan& plan,
    const PrefillExecutionProgress& progress) noexcept {
  if (!valid_plan_topology(plan) ||
      progress.next_layer != plan.layers.size() ||
      progress.next_panel != 0U) {
    return false;
  }
  for (std::size_t layer_index = 0U; layer_index < plan.layers.size();
       ++layer_index) {
    if (progress.completed_panels[layer_index] != plan.panel_count ||
        progress.completed_mlp_phases[layer_index] !=
            plan.mlp_schedule.mlp_phase_submission_count_per_layer) {
      return false;
    }
    if (plan.mlp_schedule.tactic ==
        LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore) {
      if (progress.completed_fill_panels[layer_index] != plan.panel_count ||
          progress.completed_prompt_core_phases[layer_index] != 1U ||
          progress.completed_drain_panels[layer_index] != plan.panel_count) {
        return false;
      }
    } else if (progress.completed_fill_panels[layer_index] != 0U ||
               progress.completed_prompt_core_phases[layer_index] != 0U ||
               progress.completed_drain_panels[layer_index] != 0U) {
      return false;
    }
    if (plan.layers[layer_index].progress_domain ==
        PrefillProgressDomain::kKvCache) {
      if (progress.kv_visible_end[layer_index] != plan.final_position ||
          progress.gdn_advanced_end[layer_index] != plan.first_position) {
        return false;
      }
    } else if (progress.gdn_advanced_end[layer_index] !=
                   plan.final_position ||
               progress.kv_visible_end[layer_index] != plan.first_position) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool layer_wide_p40_mlp_prefill_plan_enabled() noexcept {
  return layer_wide_p40_mlp_build_enabled();
}

bool prompt_wide_p40_whole_core_prefill_plan_enabled() noexcept {
  return prompt_wide_p40_whole_core_build_enabled();
}

bool prompt_wide_p40_projection_reset_prefill_plan_enabled() noexcept {
  return prompt_wide_p40_projection_reset_build_enabled();
}

PrefillExecutionPlanResult build_unbound_layer_major_prefill_execution_plan(
    const PrefillExecutionPlanOptions& options) noexcept {
  if (options.prompt_token_count == 0U ||
      options.max_sequence_length == 0U ||
      options.max_sequence_length >
          kLayerMajorPrefillMaximumSequenceTokens ||
      !is_valid_layer_major_prefill_mlp_schedule_tactic(
          options.mlp_schedule_tactic) ||
      !supported_mlp_schedule_tactic(options.mlp_schedule_tactic)) {
    return plan_failure(PrefillExecutionPlanError::kInvalidArgument);
  }

  const bool whole_core =
      options.mlp_schedule_tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore;
  const bool projection_reset =
      options.mlp_schedule_tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPromptWideP40ProjectionReset;
  const bool fixed_p40_geometry = whole_core || projection_reset;

  if (options.mlp_schedule_tactic !=
          LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel &&
      (options.first_position != 0U ||
       options.prompt_token_count !=
           kLayerMajorPrefillLayerWideMlpP40Tokens ||
       options.prompt_token_count %
               kLayerMajorPrefillLayerWideMlpAlignmentTokens !=
           0U)) {
    return plan_failure(PrefillExecutionPlanError::kInvalidArgument);
  }
  if (fixed_p40_geometry &&
      options.max_sequence_length !=
          kLayerMajorPrefillPromptWideP40RequestCapacityTokens) {
    return plan_failure(PrefillExecutionPlanError::kInvalidArgument);
  }

  std::uint64_t final_position = 0U;
  if (!checked_add(options.first_position, options.prompt_token_count,
                   final_position)) {
    return plan_failure(PrefillExecutionPlanError::kArithmeticOverflow);
  }
  if (final_position > options.max_sequence_length ||
      final_position > kLayerMajorPrefillMaximumSequenceTokens) {
    return plan_failure(PrefillExecutionPlanError::kCapacityExceeded);
  }

  const std::uint64_t panel_count =
      fixed_p40_geometry
          ? kLayerMajorPrefillPromptWideP40PanelCount
          : (options.prompt_token_count +
             kLayerMajorPrefillOperatorPanelTokens - 1U) /
                kLayerMajorPrefillOperatorPanelTokens;
  if (panel_count == 0U ||
      panel_count > kLayerMajorPrefillMaximumPanelCount) {
    return plan_failure(PrefillExecutionPlanError::kInvalidTopology);
  }

  PrefillExecutionPlan plan;
  plan.first_position = static_cast<std::uint32_t>(options.first_position);
  plan.prompt_token_count =
      static_cast<std::uint32_t>(options.prompt_token_count);
  plan.final_position = static_cast<std::uint32_t>(final_position);
  plan.panel_count = static_cast<std::size_t>(panel_count);

  std::uint64_t next_position = options.first_position;
  std::uint64_t remaining = options.prompt_token_count;
  for (std::size_t panel_index = 0U; panel_index < plan.panel_count;
       ++panel_index) {
    const std::uint64_t token_count =
        fixed_p40_geometry
            ? kLayerMajorPrefillPromptWideP40PanelTokens
            : next_layer_major_prefill_operator_panel_token_count(
                  static_cast<std::size_t>(remaining));
    const std::uint64_t end_position = next_position + token_count;
    plan.panels[panel_index] = PrefillOperatorPanel{
        static_cast<std::uint32_t>(panel_index),
        static_cast<std::uint32_t>(next_position),
        static_cast<std::uint32_t>(token_count),
        static_cast<std::uint32_t>(end_position)};
    next_position = end_position;
    remaining -= token_count;
  }

  for (std::size_t layer_index = 0U; layer_index < plan.layers.size();
       ++layer_index) {
    const model::LayerType layer_type = layer_type_for_index(layer_index);
    plan.layers[layer_index] = PrefillLayerExecution{
        layer_index, layer_type,
        layer_type == model::LayerType::kFullAttention
            ? PrefillProgressDomain::kKvCache
            : PrefillProgressDomain::kGdnState,
        plan.panel_count};
  }
  plan.mlp_schedule.tactic = options.mlp_schedule_tactic;
  plan.mlp_schedule.operator_panel_phase_count_per_layer = plan.panel_count;
  if (options.mlp_schedule_tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel) {
    plan.mlp_schedule.mlp_phase_submission_count_per_layer = plan.panel_count;
    plan.mlp_schedule.maximum_m_per_mlp_submission =
        kLayerMajorPrefillOperatorPanelTokens;
  } else {
    plan.mlp_schedule.mlp_phase_submission_count_per_layer = 1U;
    plan.mlp_schedule.maximum_m_per_mlp_submission =
        kLayerMajorPrefillLayerWideMlpP40Tokens;
    plan.mlp_schedule.required_gate_up_projection_launches_per_layer = 1U;
    plan.mlp_schedule.maximum_standalone_silu_launches_per_layer =
        fixed_p40_geometry ? 0U : 1U;
    plan.mlp_schedule.required_down_projection_launches_per_layer = 1U;
    plan.mlp_schedule.minimum_total_kernel_launches_per_layer = 2U;
    plan.mlp_schedule.maximum_total_kernel_launches_per_layer =
        fixed_p40_geometry ? 2U : 3U;
    plan.mlp_schedule.waits_for_all_operator_panels = true;
    plan.mlp_schedule.post_attention_norm_is_prompt_wide = true;
    plan.mlp_schedule.exact_full_m_binding_required = true;
    plan.mlp_schedule.internal_m_segmentation_forbidden = true;
  }
  if (whole_core) {
    plan.whole_core_schedule.enabled = true;
    plan.whole_core_schedule.fill_panel_phase_count_per_layer =
        plan.panel_count;
    plan.whole_core_schedule.prompt_core_phase_count_per_layer = 1U;
    plan.whole_core_schedule.drain_panel_phase_count_per_layer =
        plan.panel_count;
    plan.whole_core_schedule.persistent_mlp_phase_count_per_layer = 1U;
    plan.whole_core_schedule.panel_token_count =
        kLayerMajorPrefillPromptWideP40PanelTokens;
    plan.whole_core_schedule.prompt_core_token_count =
        kLayerMajorPrefillPromptWideP40Tokens;
    plan.whole_core_schedule.request_capacity_tokens =
        kLayerMajorPrefillPromptWideP40RequestCapacityTokens;
    plan.whole_core_schedule.route_pass_count = 1U;
    plan.whole_core_schedule.fp8_single_launch_per_projection_required = true;
    plan.whole_core_schedule.bf16_ab_prompt_wide_required = true;
    plan.whole_core_schedule.gdn_prompt_wide_required = true;
    plan.whole_core_schedule.flashinfer_whole_prompt_required = true;
  }
  if (projection_reset) {
    plan.projection_reset_schedule.enabled = true;
    plan.projection_reset_schedule.input_preparation_panel_count_per_layer =
        plan.panel_count;
    plan.projection_reset_schedule.prompt_core_phase_count_per_layer = 1U;
    plan.projection_reset_schedule.persistent_mlp_phase_count_per_layer = 1U;
    plan.projection_reset_schedule.panel_token_count =
        kLayerMajorPrefillPromptWideP40PanelTokens;
    plan.projection_reset_schedule.projection_m_tokens =
        kLayerMajorPrefillPromptWideP40Tokens;
    plan.projection_reset_schedule.request_capacity_tokens =
        kLayerMajorPrefillPromptWideP40RequestCapacityTokens;
    plan.projection_reset_schedule.route_pass_count = 1U;
    plan.projection_reset_schedule.fp8_grouped_input_launches_per_layer =
        kLayerMajorPrefillProjectionResetFp8GroupedInputLaunchesPerLayer;
    plan.projection_reset_schedule.fp8_output_launches_per_layer =
        kLayerMajorPrefillProjectionResetFp8OutputLaunchesPerLayer;
    plan.projection_reset_schedule.fp8_physical_launches_per_request =
        kLayerMajorPrefillProjectionResetFp8PhysicalLaunchesPerRequest;
    plan.projection_reset_schedule.fp8_tensor_role_hits_per_request =
        kLayerMajorPrefillProjectionResetFp8TensorRoleHitsPerRequest;
    plan.projection_reset_schedule.nvfp4_gate_up_launches_per_layer =
        kLayerMajorPrefillProjectionResetNvFp4GateUpLaunchesPerLayer;
    plan.projection_reset_schedule.nvfp4_down_launches_per_layer =
        kLayerMajorPrefillProjectionResetNvFp4DownLaunchesPerLayer;
    plan.projection_reset_schedule.nvfp4_physical_launches_per_request =
        kLayerMajorPrefillProjectionResetNvFp4PhysicalLaunchesPerRequest;
    plan.projection_reset_schedule.fp8_grouped_full_prompt_input_required =
        true;
    plan.projection_reset_schedule.fp8_full_prompt_output_required = true;
    plan.projection_reset_schedule.nvfp4_full_prompt_required = true;
    plan.projection_reset_schedule.internal_m_segmentation_forbidden = true;
    plan.projection_reset_schedule.production_accuracy_required = true;
    plan.projection_reset_schedule.approximate_numerics_forbidden = true;
    plan.projection_reset_schedule.mtp_forbidden = true;
    plan.projection_reset_schedule.cublaslt_forbidden = true;
  }
  plan.final_commit = PrefillFinalCommitPlan{
      plan.first_position, plan.final_position, 1U};
  plan.operator_bindings_complete = false;

  if (remaining != 0U || next_position != final_position ||
      !valid_plan_topology(plan)) {
    return plan_failure(PrefillExecutionPlanError::kInvalidTopology);
  }

  PrefillExecutionPlanResult result;
  result.value.emplace(plan);
  return result;
}

bool is_valid_unbound_layer_major_prefill_execution_plan(
    const PrefillExecutionPlan& plan) noexcept {
  return valid_plan_topology(plan);
}

PrefillExecutionProgress make_prefill_execution_progress(
    const PrefillExecutionPlan& plan) noexcept {
  PrefillExecutionProgress progress;
  progress.kv_visible_end.fill(plan.first_position);
  progress.gdn_advanced_end.fill(plan.first_position);
  return progress;
}

PrefillExecutionProgressError advance_prefill_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    const std::size_t layer_index,
    const std::size_t panel_index) noexcept {
  if (!valid_plan_topology(plan)) {
    return PrefillExecutionProgressError::kInvalidPlan;
  }
  if (plan.mlp_schedule.tactic ==
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore ||
      plan.mlp_schedule.tactic == LayerMajorPrefillMlpScheduleTactic::
                                      kPromptWideP40ProjectionReset) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }
  if (layer_index >= plan.layers.size()) {
    return PrefillExecutionProgressError::kLayerOutOfRange;
  }
  if (panel_index >= plan.panel_count) {
    return PrefillExecutionProgressError::kPanelOutOfRange;
  }
  if (progress.prefill_state_committed ||
      layer_index != progress.next_layer ||
      panel_index != progress.next_panel ||
      progress.completed_panels[layer_index] != panel_index) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }

  const PrefillOperatorPanel& panel = plan.panels[panel_index];
  const PrefillProgressDomain domain =
      plan.layers[layer_index].progress_domain;
  std::uint32_t* const progress_end =
      domain == PrefillProgressDomain::kKvCache
          ? &progress.kv_visible_end[layer_index]
          : &progress.gdn_advanced_end[layer_index];
  if (*progress_end != panel.first_position) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }

  *progress_end = panel.end_position;
  ++progress.completed_panels[layer_index];
  if (plan.mlp_schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel) {
    ++progress.completed_mlp_phases[layer_index];
  }
  ++progress.next_panel;
  if (progress.next_panel == plan.panel_count) {
    progress.next_panel = 0U;
    if (plan.mlp_schedule.tactic ==
        LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel) {
      ++progress.next_layer;
    }
  }
  return PrefillExecutionProgressError::kNone;
}

PrefillExecutionProgressError
advance_layer_wide_p40_mlp_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    const std::size_t layer_index) noexcept {
  if (!valid_plan_topology(plan)) {
    return PrefillExecutionProgressError::kInvalidPlan;
  }
  if (layer_index >= plan.layers.size()) {
    return PrefillExecutionProgressError::kLayerOutOfRange;
  }
  if (plan.mlp_schedule.tactic != LayerMajorPrefillMlpScheduleTactic::
                                      kLayerWideP40ExactFullM ||
      progress.prefill_state_committed ||
      progress.next_layer != layer_index || progress.next_panel != 0U ||
      progress.completed_panels[layer_index] != plan.panel_count ||
      progress.completed_mlp_phases[layer_index] != 0U) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }
  progress.completed_mlp_phases[layer_index] = 1U;
  ++progress.next_layer;
  return PrefillExecutionProgressError::kNone;
}

PrefillExecutionProgressError
advance_prompt_wide_p40_fill_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    const std::size_t layer_index,
    const std::size_t panel_index) noexcept {
  if (!valid_plan_topology(plan)) {
    return PrefillExecutionProgressError::kInvalidPlan;
  }
  if (layer_index >= plan.layers.size()) {
    return PrefillExecutionProgressError::kLayerOutOfRange;
  }
  if (panel_index >= plan.panel_count) {
    return PrefillExecutionProgressError::kPanelOutOfRange;
  }
  if (plan.mlp_schedule.tactic !=
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore ||
      progress.prefill_state_committed || progress.next_layer != layer_index ||
      progress.next_panel != panel_index ||
      progress.completed_fill_panels[layer_index] != panel_index ||
      progress.completed_prompt_core_phases[layer_index] != 0U ||
      progress.completed_drain_panels[layer_index] != 0U ||
      progress.completed_panels[layer_index] != 0U ||
      progress.completed_mlp_phases[layer_index] != 0U) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }
  ++progress.completed_fill_panels[layer_index];
  ++progress.next_panel;
  if (progress.next_panel == plan.panel_count) {
    progress.next_panel = 0U;
  }
  return PrefillExecutionProgressError::kNone;
}

PrefillExecutionProgressError
advance_prompt_wide_p40_prompt_core_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    const std::size_t layer_index) noexcept {
  if (!valid_plan_topology(plan)) {
    return PrefillExecutionProgressError::kInvalidPlan;
  }
  if (layer_index >= plan.layers.size()) {
    return PrefillExecutionProgressError::kLayerOutOfRange;
  }
  if (plan.mlp_schedule.tactic !=
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore ||
      progress.prefill_state_committed || progress.next_layer != layer_index ||
      progress.next_panel != 0U ||
      progress.completed_fill_panels[layer_index] != plan.panel_count ||
      progress.completed_prompt_core_phases[layer_index] != 0U ||
      progress.completed_drain_panels[layer_index] != 0U ||
      progress.completed_panels[layer_index] != 0U ||
      progress.completed_mlp_phases[layer_index] != 0U) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }
  const PrefillProgressDomain domain =
      plan.layers[layer_index].progress_domain;
  std::uint32_t* const progress_end =
      domain == PrefillProgressDomain::kKvCache
          ? &progress.kv_visible_end[layer_index]
          : &progress.gdn_advanced_end[layer_index];
  if (*progress_end != plan.first_position) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }
  *progress_end = plan.final_position;
  progress.completed_prompt_core_phases[layer_index] = 1U;
  return PrefillExecutionProgressError::kNone;
}

PrefillExecutionProgressError
advance_prompt_wide_p40_drain_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    const std::size_t layer_index,
    const std::size_t panel_index) noexcept {
  if (!valid_plan_topology(plan)) {
    return PrefillExecutionProgressError::kInvalidPlan;
  }
  if (layer_index >= plan.layers.size()) {
    return PrefillExecutionProgressError::kLayerOutOfRange;
  }
  if (panel_index >= plan.panel_count) {
    return PrefillExecutionProgressError::kPanelOutOfRange;
  }
  if (plan.mlp_schedule.tactic !=
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore ||
      progress.prefill_state_committed || progress.next_layer != layer_index ||
      progress.next_panel != panel_index ||
      progress.completed_fill_panels[layer_index] != plan.panel_count ||
      progress.completed_prompt_core_phases[layer_index] != 1U ||
      progress.completed_drain_panels[layer_index] != panel_index ||
      progress.completed_panels[layer_index] != panel_index ||
      progress.completed_mlp_phases[layer_index] != 0U) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }
  ++progress.completed_drain_panels[layer_index];
  ++progress.completed_panels[layer_index];
  ++progress.next_panel;
  if (progress.next_panel == plan.panel_count) {
    progress.next_panel = 0U;
  }
  return PrefillExecutionProgressError::kNone;
}

PrefillExecutionProgressError
advance_prompt_wide_p40_persistent_mlp_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    const std::size_t layer_index) noexcept {
  if (!valid_plan_topology(plan)) {
    return PrefillExecutionProgressError::kInvalidPlan;
  }
  if (layer_index >= plan.layers.size()) {
    return PrefillExecutionProgressError::kLayerOutOfRange;
  }
  if (plan.mlp_schedule.tactic !=
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore ||
      progress.prefill_state_committed || progress.next_layer != layer_index ||
      progress.next_panel != 0U ||
      progress.completed_fill_panels[layer_index] != plan.panel_count ||
      progress.completed_prompt_core_phases[layer_index] != 1U ||
      progress.completed_drain_panels[layer_index] != plan.panel_count ||
      progress.completed_panels[layer_index] != plan.panel_count ||
      progress.completed_mlp_phases[layer_index] != 0U) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }
  progress.completed_mlp_phases[layer_index] = 1U;
  ++progress.next_layer;
  return PrefillExecutionProgressError::kNone;
}

PrefillExecutionProgressError
advance_prompt_wide_p40_projection_reset_layer_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    const std::size_t layer_index) noexcept {
  if (!valid_plan_topology(plan)) {
    return PrefillExecutionProgressError::kInvalidPlan;
  }
  if (layer_index >= plan.layers.size()) {
    return PrefillExecutionProgressError::kLayerOutOfRange;
  }
  if (plan.mlp_schedule.tactic != LayerMajorPrefillMlpScheduleTactic::
                                      kPromptWideP40ProjectionReset ||
      progress.prefill_state_committed || progress.next_layer != layer_index ||
      progress.next_panel != 0U ||
      progress.completed_panels[layer_index] != 0U ||
      progress.completed_mlp_phases[layer_index] != 0U ||
      progress.completed_fill_panels[layer_index] != 0U ||
      progress.completed_prompt_core_phases[layer_index] != 0U ||
      progress.completed_drain_panels[layer_index] != 0U) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }

  const PrefillProgressDomain domain =
      plan.layers[layer_index].progress_domain;
  std::uint32_t* const progress_end =
      domain == PrefillProgressDomain::kKvCache
          ? &progress.kv_visible_end[layer_index]
          : &progress.gdn_advanced_end[layer_index];
  if (*progress_end != plan.first_position) {
    return PrefillExecutionProgressError::kOutOfOrder;
  }
  *progress_end = plan.final_position;
  progress.completed_panels[layer_index] = plan.panel_count;
  progress.completed_mlp_phases[layer_index] = 1U;
  ++progress.next_layer;
  return PrefillExecutionProgressError::kNone;
}

PrefillExecutionProgressError mark_prefill_final_hidden_ready(
    const PrefillExecutionPlan& plan,
    PrefillExecutionProgress& progress) noexcept {
  if (!execution_complete(plan, progress)) {
    return PrefillExecutionProgressError::kExecutionIncomplete;
  }
  if (progress.prefill_state_committed) {
    return PrefillExecutionProgressError::kAlreadyCommitted;
  }
  progress.final_hidden_ready = true;
  return PrefillExecutionProgressError::kNone;
}

bool prefill_final_commit_ready(
    const PrefillExecutionPlan& plan,
    const PrefillExecutionProgress& progress) noexcept {
  return !progress.prefill_state_committed && progress.final_hidden_ready &&
         execution_complete(plan, progress);
}

PrefillExecutionProgressError publish_prefill_state_committed(
    const PrefillExecutionPlan& plan,
    PrefillExecutionProgress& progress) noexcept {
  if (progress.prefill_state_committed) {
    return PrefillExecutionProgressError::kAlreadyCommitted;
  }
  if (!prefill_final_commit_ready(plan, progress)) {
    return PrefillExecutionProgressError::kCommitNotReady;
  }
  progress.prefill_state_committed = true;
  return PrefillExecutionProgressError::kNone;
}

}  // namespace q3x::runtime
