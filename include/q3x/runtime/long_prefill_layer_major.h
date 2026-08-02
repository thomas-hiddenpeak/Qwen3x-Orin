#pragma once

#include "q3x/model/model_config.h"
#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/request_state.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string_view>

namespace q3x::runtime {

inline constexpr std::uint32_t kLongPrefillLayerMajorTileTokens = 512U;
// The K128 whole-span executor can safely pad a natural short prompt to its
// M64 projection contract without exposing the padded rows to recurrent,
// attention, cache, or residual state.  Keep this deliberately narrow: the
// ordinary short-prompt schedule remains the default below this boundary.
inline constexpr std::uint32_t kShortPrefillLayerMajorMinimumTokens = 480U;
inline constexpr std::uint32_t kLongPrefillProjectionSpanDefaultTokens =
    4'096U;
inline constexpr std::uint32_t kLongPrefillLayerMajorMaximumTokens =
    kRequestLongPrefillAdmissionMaximumTokens;

// Production comparator switch. This intentionally controls dispatch only:
// callers must still authenticate and retain optimized weights and reserve
// the identical request arena so enabled/disabled measurements use one ELF
// and one lifetime topology.
[[nodiscard]] inline bool optimized_prefill_dispatch_disabled() noexcept {
  static const bool disabled = []() noexcept {
    const char* const value =
        std::getenv("Q3X_DISABLE_OPTIMIZED_PREFILL");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return disabled;
}

// Experimental same-ELF selector for the narrow P480..P512 cliff.  Route
// selection additionally requires a complete authenticated K128 A4 inventory;
// setting this variable alone can never admit K64 or a partial publication.
[[nodiscard]] inline bool
short_prefill_layer_major_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

enum class LongPrefillLayerMajorRoute : std::uint8_t {
  kTileMajorFallback = 0,
  kLayerMajorAdmission,
};

// Runtime selection remains separate from code availability. Production
// callers enable this by default after reserving the required arena; trace,
// backend, shape, and capacity mismatches still fall back safely.
struct LongPrefillLayerMajorRouteQuery {
  bool runtime_enabled = false;
  bool short_prompt_admission_enabled = false;
  bool authenticated_a4_prefill = false;
  ProjectionBackend projection_backend = ProjectionBackend::kReference;
  bool capture_trace = false;
  std::uint32_t prompt_token_count = 0U;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
  std::uint32_t hidden_token_capacity = 0U;
  std::size_t hidden_buffer_count = 0U;
};

[[nodiscard]] bool long_prefill_layer_major_build_enabled() noexcept;

[[nodiscard]] LongPrefillLayerMajorRoute select_long_prefill_layer_major_route(
    const LongPrefillLayerMajorRouteQuery& query) noexcept;

// Pure selector for the whole-M A4 projection-span executor. The enclosing
// layer-major selector owns backend/workspace validation; this helper keeps
// aligned, non-aligned, disabled, and out-of-range boundaries host-testable.
// The workspace span remains a C512 multiple, but the prompt itself need not
// be: the plan owns a final short projection span/state tile.
[[nodiscard]] constexpr bool use_long_prefill_projection_span_route(
    const std::uint32_t prompt_token_count,
    const std::uint32_t projection_span_token_count,
    const bool a4_inventory_enabled,
    const bool optimized_prefill_disabled,
    const bool short_prompt_admission_enabled = false,
    const bool authenticated_a4_prefill = false) noexcept {
  const bool long_prompt =
      prompt_token_count > kLongPrefillLayerMajorTileTokens;
  const bool admitted_short_prompt =
      short_prompt_admission_enabled && authenticated_a4_prefill &&
      prompt_token_count >= kShortPrefillLayerMajorMinimumTokens &&
      prompt_token_count <= kLongPrefillLayerMajorTileTokens;
  return !optimized_prefill_disabled && a4_inventory_enabled &&
         (long_prompt || admitted_short_prompt) &&
         prompt_token_count <= kLongPrefillLayerMajorMaximumTokens &&
         projection_span_token_count >=
             kLongPrefillLayerMajorTileTokens &&
         projection_span_token_count %
                 kLongPrefillLayerMajorTileTokens ==
             0U;
}

enum class LongPrefillLayerMajorPlanError : std::uint8_t {
  kNone = 0,
  kBuildDisabled,
  kInvalidOption,
  kCapacityExceeded,
  kArithmeticOverflow,
};

struct LongPrefillLayerMajorOptions {
  std::uint32_t prompt_token_count = 0U;
  std::uint32_t hidden_token_capacity = 0U;
  std::uint32_t tile_token_count = kLongPrefillLayerMajorTileTokens;
};

struct LongPrefillLayerMajorPlan {
  std::uint32_t prompt_token_count = 0U;
  std::uint32_t hidden_token_capacity = 0U;
  std::uint32_t tile_token_count = 0U;
  std::uint32_t tile_count = 0U;
  std::uint32_t full_tile_count = 0U;
  std::uint32_t tail_token_count = 0U;
  std::size_t work_item_count = 0U;
  std::size_t embedding_output_hidden_buffer = 0U;
  std::size_t final_hidden_buffer = 0U;
};

struct LongPrefillLayerMajorPlanResult {
  std::optional<LongPrefillLayerMajorPlan> value;
  LongPrefillLayerMajorPlanError error =
      LongPrefillLayerMajorPlanError::kNone;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && error == LongPrefillLayerMajorPlanError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// One layer/tile work item. Iterating ordinal 0..work_item_count-1 is strictly
// layer-major: every tile of layer L completes before layer L+1 begins. Within
// a linear-attention layer, increasing first_position preserves recurrent GDN
// state order. Within a full-attention layer, the same order appends exactly
// the global K/V interval consumed by that tile's causal attention.
struct LongPrefillLayerMajorWorkItem {
  std::size_t ordinal = 0U;
  std::size_t layer_index = 0U;
  model::LayerType layer_type = model::LayerType::kInvalid;
  std::uint32_t tile_index = 0U;
  std::uint32_t first_position = 0U;
  std::uint32_t token_count = 0U;
  std::size_t input_hidden_buffer = 0U;
  std::size_t output_hidden_buffer = 0U;
  bool first_tile_for_layer = false;
  bool last_tile_for_layer = false;
  bool updates_recurrent_state = false;
  bool appends_kv = false;
};

[[nodiscard]] LongPrefillLayerMajorPlanResult
build_long_prefill_layer_major_plan(
    const LongPrefillLayerMajorOptions& options) noexcept;

[[nodiscard]] bool long_prefill_layer_major_work_item(
    const LongPrefillLayerMajorPlan& plan, std::size_t ordinal,
    LongPrefillLayerMajorWorkItem& item) noexcept;

// Opt-in whole-M schedule. Unlike LongPrefillLayerMajorPlan, whose callback
// granularity remains one C512 state tile for compatibility, this plan emits
// one work item per layer/projection span. The consumer performs whole-span
// projections around the ordered C512 state tiles described by
// long_prefill_projection_span_state_tile(). Projection spans must be an
// integral number of C512 tiles; only the final span and its final state tile
// may be short.
struct LongPrefillProjectionSpanOptions {
  std::uint32_t prompt_token_count = 0U;
  std::uint32_t hidden_token_capacity = 0U;
  std::uint32_t projection_span_token_count =
      kLongPrefillProjectionSpanDefaultTokens;
  std::uint32_t state_tile_token_count =
      kLongPrefillLayerMajorTileTokens;
};

struct LongPrefillProjectionSpanPlan {
  std::uint32_t prompt_token_count = 0U;
  std::uint32_t hidden_token_capacity = 0U;
  std::uint32_t projection_span_token_count = 0U;
  std::uint32_t projection_span_count = 0U;
  std::uint32_t full_projection_span_count = 0U;
  std::uint32_t projection_tail_token_count = 0U;
  std::uint32_t state_tile_token_count = 0U;
  std::uint32_t state_tile_count = 0U;
  std::uint32_t full_state_tile_count = 0U;
  std::uint32_t state_tail_token_count = 0U;
  std::size_t work_item_count = 0U;
  std::size_t embedding_output_hidden_buffer = 0U;
  std::size_t final_hidden_buffer = 0U;
};

struct LongPrefillProjectionSpanPlanResult {
  std::optional<LongPrefillProjectionSpanPlan> value;
  LongPrefillLayerMajorPlanError error =
      LongPrefillLayerMajorPlanError::kNone;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && error == LongPrefillLayerMajorPlanError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct LongPrefillProjectionSpanWorkItem {
  std::size_t ordinal = 0U;
  std::size_t layer_index = 0U;
  model::LayerType layer_type = model::LayerType::kInvalid;
  std::uint32_t projection_span_index = 0U;
  std::uint32_t first_position = 0U;
  std::uint32_t token_count = 0U;
  std::uint32_t first_state_tile_index = 0U;
  std::uint32_t state_tile_count = 0U;
  std::uint32_t full_state_tile_count = 0U;
  std::uint32_t state_tail_token_count = 0U;
  std::size_t input_hidden_buffer = 0U;
  std::size_t output_hidden_buffer = 0U;
  bool first_projection_span_for_layer = false;
  bool last_projection_span_for_layer = false;
  bool updates_recurrent_state = false;
  bool appends_kv = false;
};

struct LongPrefillProjectionSpanStateTile {
  std::size_t projection_span_ordinal = 0U;
  std::size_t layer_index = 0U;
  model::LayerType layer_type = model::LayerType::kInvalid;
  std::uint32_t projection_span_index = 0U;
  std::uint32_t state_tile_index = 0U;
  std::uint32_t state_tile_index_in_span = 0U;
  std::uint32_t first_position = 0U;
  std::uint32_t token_count = 0U;
  bool first_state_tile_for_span = false;
  bool last_state_tile_for_span = false;
  bool first_state_tile_for_layer = false;
  bool last_state_tile_for_layer = false;
  bool updates_recurrent_state = false;
  bool appends_kv = false;
};

[[nodiscard]] LongPrefillProjectionSpanPlanResult
build_long_prefill_projection_span_plan(
    const LongPrefillProjectionSpanOptions& options) noexcept;

[[nodiscard]] bool long_prefill_projection_span_work_item(
    const LongPrefillProjectionSpanPlan& plan, std::size_t ordinal,
    LongPrefillProjectionSpanWorkItem& item) noexcept;

[[nodiscard]] bool long_prefill_projection_span_state_tile(
    const LongPrefillProjectionSpanPlan& plan,
    std::size_t projection_span_ordinal,
    std::uint32_t state_tile_index_in_span,
    LongPrefillProjectionSpanStateTile& tile) noexcept;

using LongPrefillPrepareHiddenFunction = bool (*)(
    void* context, std::size_t output_hidden_buffer,
    std::uint32_t prompt_token_count) noexcept;
using LongPrefillExecuteTileFunction = bool (*)(
    void* context, const LongPrefillLayerMajorWorkItem& item) noexcept;
using LongPrefillFinishPromptFunction = bool (*)(
    void* context, std::uint32_t sequence_length,
    std::size_t final_hidden_buffer) noexcept;

struct LongPrefillLayerMajorCallbacks {
  void* context = nullptr;
  LongPrefillPrepareHiddenFunction prepare_hidden = nullptr;
  LongPrefillExecuteTileFunction execute_tile = nullptr;
  LongPrefillFinishPromptFunction finish_prompt = nullptr;
};

enum class LongPrefillLayerMajorExecutionError : std::uint8_t {
  kNone = 0,
  kInvalidPlan,
  kMissingCallback,
  kPrepareHiddenFailed,
  kExecuteTileFailed,
  kFinishPromptFailed,
};

struct LongPrefillLayerMajorExecutionResult {
  LongPrefillLayerMajorExecutionError error =
      LongPrefillLayerMajorExecutionError::kNone;
  std::size_t completed_work_items = 0U;
  std::optional<LongPrefillLayerMajorWorkItem> failed_work_item;

  [[nodiscard]] bool ok() const noexcept {
    return error == LongPrefillLayerMajorExecutionError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pure-host executor seam. The future CUDA binding supplies one embedding
// callback and one per-layer/tile callback; this function owns the global
// ordering and introduces no tile-level synchronization by itself.
[[nodiscard]] LongPrefillLayerMajorExecutionResult
run_long_prefill_layer_major(
    const LongPrefillLayerMajorPlan& plan,
    const LongPrefillLayerMajorCallbacks& callbacks) noexcept;

[[nodiscard]] std::string_view to_string(
    LongPrefillLayerMajorPlanError error) noexcept;
[[nodiscard]] std::string_view to_string(
    LongPrefillLayerMajorExecutionError error) noexcept;

}  // namespace q3x::runtime
