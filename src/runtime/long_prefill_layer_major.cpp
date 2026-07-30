#include "q3x/runtime/long_prefill_layer_major.h"

#include <limits>

namespace q3x::runtime {
namespace {

[[nodiscard]] constexpr std::uint32_t tile_count_for(
    const std::uint32_t prompt_token_count,
    const std::uint32_t tile_token_count) noexcept {
  return prompt_token_count == 0U || tile_token_count == 0U
             ? 0U
             : 1U + (prompt_token_count - 1U) / tile_token_count;
}

[[nodiscard]] bool valid_plan(
    const LongPrefillLayerMajorPlan& plan) noexcept {
  if (plan.prompt_token_count == 0U ||
      plan.prompt_token_count > kLongPrefillLayerMajorMaximumTokens ||
      plan.hidden_token_capacity < plan.prompt_token_count ||
      plan.hidden_token_capacity > kLongPrefillLayerMajorMaximumTokens ||
      plan.tile_token_count != kLongPrefillLayerMajorTileTokens ||
      plan.embedding_output_hidden_buffer != 0U ||
      plan.final_hidden_buffer != kRequestLayerCount %
                                      kRequestLongPrefillHiddenBufferCount) {
    return false;
  }
  const std::uint32_t expected_tile_count =
      tile_count_for(plan.prompt_token_count, plan.tile_token_count);
  const std::uint32_t expected_full_tiles =
      plan.prompt_token_count / plan.tile_token_count;
  const std::uint32_t expected_tail =
      plan.prompt_token_count % plan.tile_token_count;
  return plan.tile_count == expected_tile_count &&
         plan.full_tile_count == expected_full_tiles &&
         plan.tail_token_count == expected_tail &&
         plan.work_item_count ==
             static_cast<std::size_t>(expected_tile_count) *
                 kRequestLayerCount;
}

}  // namespace

bool long_prefill_layer_major_build_enabled() noexcept {
#if defined(Q3X_ENABLE_LONG_PREFILL_LAYER_MAJOR_ADMISSION)
  return true;
#else
  return false;
#endif
}

LongPrefillLayerMajorRoute select_long_prefill_layer_major_route(
    const LongPrefillLayerMajorRouteQuery& query) noexcept {
  const bool selected =
      long_prefill_layer_major_build_enabled() && query.runtime_enabled &&
      query.projection_backend == ProjectionBackend::kSm87WeightOnly &&
      !query.capture_trace &&
      query.prompt_token_count > kLongPrefillLayerMajorTileTokens &&
      query.prompt_token_count <= kLongPrefillLayerMajorMaximumTokens &&
      query.prefill_chunk_size == kLongPrefillLayerMajorTileTokens &&
      query.hidden_token_capacity >= query.prompt_token_count &&
      query.hidden_token_capacity <= kLongPrefillLayerMajorMaximumTokens &&
      query.hidden_buffer_count == kRequestLongPrefillHiddenBufferCount;
  return selected ? LongPrefillLayerMajorRoute::kLayerMajorAdmission
                  : LongPrefillLayerMajorRoute::kTileMajorFallback;
}

LongPrefillLayerMajorPlanResult build_long_prefill_layer_major_plan(
    const LongPrefillLayerMajorOptions& options) noexcept {
  LongPrefillLayerMajorPlanResult result;
  if (!long_prefill_layer_major_build_enabled()) {
    result.error = LongPrefillLayerMajorPlanError::kBuildDisabled;
    return result;
  }
  if (options.prompt_token_count == 0U ||
      options.tile_token_count != kLongPrefillLayerMajorTileTokens) {
    result.error = LongPrefillLayerMajorPlanError::kInvalidOption;
    return result;
  }
  if (options.prompt_token_count > kLongPrefillLayerMajorMaximumTokens ||
      options.hidden_token_capacity < options.prompt_token_count ||
      options.hidden_token_capacity > kLongPrefillLayerMajorMaximumTokens) {
    result.error = LongPrefillLayerMajorPlanError::kCapacityExceeded;
    return result;
  }

  const std::uint32_t tile_count =
      tile_count_for(options.prompt_token_count, options.tile_token_count);
  if (tile_count >
      std::numeric_limits<std::size_t>::max() / kRequestLayerCount) {
    result.error = LongPrefillLayerMajorPlanError::kArithmeticOverflow;
    return result;
  }

  LongPrefillLayerMajorPlan plan;
  plan.prompt_token_count = options.prompt_token_count;
  plan.hidden_token_capacity = options.hidden_token_capacity;
  plan.tile_token_count = options.tile_token_count;
  plan.tile_count = tile_count;
  plan.full_tile_count =
      options.prompt_token_count / options.tile_token_count;
  plan.tail_token_count =
      options.prompt_token_count % options.tile_token_count;
  plan.work_item_count =
      static_cast<std::size_t>(tile_count) * kRequestLayerCount;
  plan.embedding_output_hidden_buffer = 0U;
  plan.final_hidden_buffer =
      kRequestLayerCount % kRequestLongPrefillHiddenBufferCount;
  result.value.emplace(plan);
  return result;
}

bool long_prefill_layer_major_work_item(
    const LongPrefillLayerMajorPlan& plan, const std::size_t ordinal,
    LongPrefillLayerMajorWorkItem& item) noexcept {
  if (!valid_plan(plan) || ordinal >= plan.work_item_count) {
    return false;
  }
  const std::size_t layer = ordinal / plan.tile_count;
  const std::uint32_t tile =
      static_cast<std::uint32_t>(ordinal % plan.tile_count);
  const std::uint32_t first_position = tile * plan.tile_token_count;
  const std::uint32_t remaining =
      plan.prompt_token_count - first_position;
  const std::uint32_t token_count =
      remaining < plan.tile_token_count ? remaining : plan.tile_token_count;
  const model::LayerType layer_type =
      ((layer + 1U) % 4U) == 0U
          ? model::LayerType::kFullAttention
          : model::LayerType::kLinearAttention;

  item = {};
  item.ordinal = ordinal;
  item.layer_index = layer;
  item.layer_type = layer_type;
  item.tile_index = tile;
  item.first_position = first_position;
  item.token_count = token_count;
  item.input_hidden_buffer =
      layer % kRequestLongPrefillHiddenBufferCount;
  item.output_hidden_buffer =
      (layer + 1U) % kRequestLongPrefillHiddenBufferCount;
  item.first_tile_for_layer = tile == 0U;
  item.last_tile_for_layer = tile + 1U == plan.tile_count;
  item.updates_recurrent_state =
      layer_type == model::LayerType::kLinearAttention;
  item.appends_kv = layer_type == model::LayerType::kFullAttention;
  return true;
}

LongPrefillLayerMajorExecutionResult run_long_prefill_layer_major(
    const LongPrefillLayerMajorPlan& plan,
    const LongPrefillLayerMajorCallbacks& callbacks) noexcept {
  LongPrefillLayerMajorExecutionResult result;
  if (!valid_plan(plan)) {
    result.error = LongPrefillLayerMajorExecutionError::kInvalidPlan;
    return result;
  }
  if (callbacks.prepare_hidden == nullptr ||
      callbacks.execute_tile == nullptr ||
      callbacks.finish_prompt == nullptr) {
    result.error = LongPrefillLayerMajorExecutionError::kMissingCallback;
    return result;
  }
  if (!callbacks.prepare_hidden(callbacks.context,
                                plan.embedding_output_hidden_buffer,
                                plan.prompt_token_count)) {
    result.error =
        LongPrefillLayerMajorExecutionError::kPrepareHiddenFailed;
    return result;
  }
  for (std::size_t ordinal = 0U; ordinal < plan.work_item_count; ++ordinal) {
    LongPrefillLayerMajorWorkItem item;
    if (!long_prefill_layer_major_work_item(plan, ordinal, item)) {
      result.error = LongPrefillLayerMajorExecutionError::kInvalidPlan;
      return result;
    }
    if (!callbacks.execute_tile(callbacks.context, item)) {
      result.error = LongPrefillLayerMajorExecutionError::kExecuteTileFailed;
      result.failed_work_item.emplace(item);
      return result;
    }
    ++result.completed_work_items;
  }
  if (!callbacks.finish_prompt(callbacks.context, plan.prompt_token_count,
                               plan.final_hidden_buffer)) {
    result.error = LongPrefillLayerMajorExecutionError::kFinishPromptFailed;
  }
  return result;
}

std::string_view to_string(
    const LongPrefillLayerMajorPlanError error) noexcept {
  switch (error) {
    case LongPrefillLayerMajorPlanError::kNone:
      return "none";
    case LongPrefillLayerMajorPlanError::kBuildDisabled:
      return "build_disabled";
    case LongPrefillLayerMajorPlanError::kInvalidOption:
      return "invalid_option";
    case LongPrefillLayerMajorPlanError::kCapacityExceeded:
      return "capacity_exceeded";
    case LongPrefillLayerMajorPlanError::kArithmeticOverflow:
      return "arithmetic_overflow";
  }
  return "unknown";
}

std::string_view to_string(
    const LongPrefillLayerMajorExecutionError error) noexcept {
  switch (error) {
    case LongPrefillLayerMajorExecutionError::kNone:
      return "none";
    case LongPrefillLayerMajorExecutionError::kInvalidPlan:
      return "invalid_plan";
    case LongPrefillLayerMajorExecutionError::kMissingCallback:
      return "missing_callback";
    case LongPrefillLayerMajorExecutionError::kPrepareHiddenFailed:
      return "prepare_hidden_failed";
    case LongPrefillLayerMajorExecutionError::kExecuteTileFailed:
      return "execute_tile_failed";
    case LongPrefillLayerMajorExecutionError::kFinishPromptFailed:
      return "finish_prompt_failed";
  }
  return "unknown";
}

}  // namespace q3x::runtime
