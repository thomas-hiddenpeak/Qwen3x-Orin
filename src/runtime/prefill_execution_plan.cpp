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
      plan.operator_bindings_complete) {
    return false;
  }

  const std::size_t expected_panel_count =
      (static_cast<std::size_t>(plan.prompt_token_count) +
       kLayerMajorPrefillOperatorPanelTokens - 1U) /
      kLayerMajorPrefillOperatorPanelTokens;
  if (plan.panel_count != expected_panel_count) {
    return false;
  }

  std::uint32_t next_position = plan.first_position;
  for (std::size_t panel_index = 0U; panel_index < plan.panel_count;
       ++panel_index) {
    const PrefillOperatorPanel& panel = plan.panels[panel_index];
    const std::uint32_t expected_token_count =
        panel_index + 1U < plan.panel_count
            ? kLayerMajorPrefillOperatorPanelTokens
            : plan.final_position - next_position;
    if (panel.ordinal != panel_index || panel.first_position != next_position ||
        panel.token_count != expected_token_count ||
        panel.end_position <= panel.first_position ||
        panel.end_position - panel.first_position != panel.token_count) {
      return false;
    }
    next_position = panel.end_position;
  }
  if (next_position != plan.final_position) {
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
    if (progress.completed_panels[layer_index] != plan.panel_count) {
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

PrefillExecutionPlanResult build_unbound_layer_major_prefill_execution_plan(
    const PrefillExecutionPlanOptions& options) noexcept {
  if (options.prompt_token_count == 0U ||
      options.max_sequence_length == 0U ||
      options.max_sequence_length >
          kLayerMajorPrefillMaximumSequenceTokens) {
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
      (options.prompt_token_count + kLayerMajorPrefillOperatorPanelTokens - 1U) /
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
        remaining < kLayerMajorPrefillOperatorPanelTokens
            ? remaining
            : kLayerMajorPrefillOperatorPanelTokens;
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
  ++progress.next_panel;
  if (progress.next_panel == plan.panel_count) {
    progress.next_panel = 0U;
    ++progress.next_layer;
  }
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
