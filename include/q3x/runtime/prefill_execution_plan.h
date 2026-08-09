#pragma once

#include "q3x/model/model_config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::runtime {

// AC-PREFILL-LAYERMAJOR-8K-v1 keeps the existing public C512 runner tile
// contract separate from its internal operator-panel capacity. This header is
// currently a pure-host, unbound topology contract: it contains no launcher,
// device pointer, stream, event, allocation, or production route selector.
inline constexpr std::uint32_t kPrefillPhysicalSegmentMaximumTokens = 512U;
inline constexpr std::uint32_t kPrefillPhysicalSegmentM256Tokens = 256U;
inline constexpr std::uint32_t kPrefillPhysicalSegmentM64Tokens = 64U;
inline constexpr std::uint32_t kPrefillPhysicalSegmentM32Tokens = 32U;
inline constexpr std::uint32_t kPrefillPhysicalSegmentTailMaximumTokens = 31U;
inline constexpr std::uint32_t kLayerMajorPrefillLegacyPublicTileTokens =
    kPrefillPhysicalSegmentMaximumTokens;
inline constexpr std::uint32_t kLayerMajorPrefillOperatorPanelTokens =
    8'192U;
inline constexpr std::uint32_t kLayerMajorPrefillMaximumSequenceTokens =
    262'144U;
inline constexpr std::size_t kLayerMajorPrefillLayerCount = 64U;
inline constexpr std::size_t kLayerMajorPrefillLinearLayerCount = 48U;
inline constexpr std::size_t kLayerMajorPrefillFullLayerCount = 16U;
inline constexpr std::size_t kLayerMajorPrefillMaximumPanelCount =
    (kLayerMajorPrefillMaximumSequenceTokens +
     kLayerMajorPrefillOperatorPanelTokens - 1U) /
    kLayerMajorPrefillOperatorPanelTokens;

static_assert(kLayerMajorPrefillLegacyPublicTileTokens == 512U);
static_assert(kLayerMajorPrefillOperatorPanelTokens == 8'192U);
static_assert(kLayerMajorPrefillMaximumPanelCount == 32U);

// Canonical exact-route subdivision for one already selected logical panel.
// The caller supplies only the tokens remaining inside that panel, so the
// returned segment cannot cross its C8192 boundary. The whole executor and
// the legacy engine scheduler must share this order rather than inventing
// numerically distinct arbitrary-M segment boundaries.
[[nodiscard]] constexpr std::size_t
next_prefill_physical_segment_token_count(
    const std::size_t remaining_panel_tokens) noexcept {
  if (remaining_panel_tokens >= kPrefillPhysicalSegmentMaximumTokens) {
    return kPrefillPhysicalSegmentMaximumTokens;
  }
  if (remaining_panel_tokens >= kPrefillPhysicalSegmentM256Tokens) {
    return kPrefillPhysicalSegmentM256Tokens;
  }
  if (remaining_panel_tokens >= kPrefillPhysicalSegmentM64Tokens) {
    return kPrefillPhysicalSegmentM64Tokens;
  }
  if (remaining_panel_tokens >= kPrefillPhysicalSegmentM32Tokens) {
    return kPrefillPhysicalSegmentM32Tokens;
  }
  return remaining_panel_tokens;
}

[[nodiscard]] constexpr bool is_prefill_physical_segment_token_count(
    const std::size_t token_count) noexcept {
  return token_count == kPrefillPhysicalSegmentMaximumTokens ||
         token_count == kPrefillPhysicalSegmentM256Tokens ||
         token_count == kPrefillPhysicalSegmentM64Tokens ||
         token_count == kPrefillPhysicalSegmentM32Tokens ||
         (token_count != 0U &&
          token_count <= kPrefillPhysicalSegmentTailMaximumTokens);
}

enum class PrefillTraversalOrder : std::uint8_t {
  kLayerMajor = 0,
};

enum class PrefillProgressDomain : std::uint8_t {
  kGdnState = 0,
  kKvCache,
};

enum class PrefillExecutionPlanError : std::uint8_t {
  kNone = 0,
  kInvalidArgument,
  kArithmeticOverflow,
  kCapacityExceeded,
  kInvalidTopology,
};

struct PrefillExecutionPlanOptions {
  std::uint64_t first_position = 0U;
  std::uint64_t prompt_token_count = 0U;
  std::uint64_t max_sequence_length =
      kLayerMajorPrefillMaximumSequenceTokens;
};

struct PrefillOperatorPanel {
  std::uint32_t ordinal = 0U;
  std::uint32_t first_position = 0U;
  std::uint32_t token_count = 0U;
  std::uint32_t end_position = 0U;
};

struct PrefillLayerExecution {
  std::size_t layer_index = 0U;
  model::LayerType layer_type = model::LayerType::kInvalid;
  PrefillProgressDomain progress_domain = PrefillProgressDomain::kGdnState;
  std::size_t panel_count = 0U;
};

struct PrefillFinalCommitPlan {
  std::uint32_t expected_initial_sequence_length = 0U;
  std::uint32_t committed_sequence_length = 0U;
  std::uint32_t commit_count = 0U;
};

struct PrefillExecutionPlan {
  PrefillTraversalOrder traversal = PrefillTraversalOrder::kLayerMajor;
  // Descriptive compatibility metadata only. It never determines panels.
  std::uint32_t legacy_public_tile_limit =
      kLayerMajorPrefillLegacyPublicTileTokens;
  std::uint32_t operator_panel_capacity =
      kLayerMajorPrefillOperatorPanelTokens;
  std::uint32_t first_position = 0U;
  std::uint32_t prompt_token_count = 0U;
  std::uint32_t final_position = 0U;
  std::array<PrefillOperatorPanel,
             kLayerMajorPrefillMaximumPanelCount>
      panels{};
  std::size_t panel_count = 0U;
  std::array<PrefillLayerExecution, kLayerMajorPrefillLayerCount> layers{};
  PrefillFinalCommitPlan final_commit;

  // The scaffold deliberately has no mutation or binder that can make this
  // true. A later, separately reviewed BoundPrefillExecutionPlan must own
  // typed launchers before an execution route can become selectable.
  bool operator_bindings_complete = false;

  [[nodiscard]] constexpr bool executable() const noexcept {
    return operator_bindings_complete;
  }
};

struct PrefillExecutionPlanResult {
  std::optional<PrefillExecutionPlan> value;
  PrefillExecutionPlanError error = PrefillExecutionPlanError::kNone;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && error == PrefillExecutionPlanError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Builds only the immutable layer/panel topology. It does not inspect model
// weights, select a kernel, reserve memory, or alter the current C512 route.
[[nodiscard]] PrefillExecutionPlanResult
build_unbound_layer_major_prefill_execution_plan(
    const PrefillExecutionPlanOptions& options) noexcept;

// Allocation-free host validation for an externally supplied immutable
// layer-major topology. This accepts only the unbound design contract:
// operator_bindings_complete must remain false. It does not authenticate or
// convert a bound execution/deployment plan.
[[nodiscard]] bool is_valid_unbound_layer_major_prefill_execution_plan(
    const PrefillExecutionPlan& plan) noexcept;

enum class PrefillExecutionProgressError : std::uint8_t {
  kNone = 0,
  kInvalidPlan,
  kLayerOutOfRange,
  kPanelOutOfRange,
  kOutOfOrder,
  kExecutionIncomplete,
  kCommitNotReady,
  kAlreadyCommitted,
};

// Mutable request-owned progress is intentionally not part of the immutable
// execution plan. Positions are exclusive ends. An executor may call the
// transition helper only after the corresponding device completion/event has
// established visibility; enqueue alone is not completion.
struct PrefillExecutionProgress {
  std::array<std::uint32_t, kLayerMajorPrefillLayerCount> kv_visible_end{};
  std::array<std::uint32_t, kLayerMajorPrefillLayerCount> gdn_advanced_end{};
  std::array<std::size_t, kLayerMajorPrefillLayerCount> completed_panels{};
  std::size_t next_layer = 0U;
  std::size_t next_panel = 0U;
  bool final_hidden_ready = false;
  bool prefill_state_committed = false;
};

[[nodiscard]] PrefillExecutionProgress make_prefill_execution_progress(
    const PrefillExecutionPlan& plan) noexcept;

[[nodiscard]] PrefillExecutionProgressError
advance_prefill_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    std::size_t layer_index, std::size_t panel_index) noexcept;

[[nodiscard]] PrefillExecutionProgressError mark_prefill_final_hidden_ready(
    const PrefillExecutionPlan& plan,
    PrefillExecutionProgress& progress) noexcept;

[[nodiscard]] bool prefill_final_commit_ready(
    const PrefillExecutionPlan& plan,
    const PrefillExecutionProgress& progress) noexcept;

// This is a host-state transition only. It deliberately does not call
// RequestState::set_sequence_length() and therefore cannot alter production
// request state while the plan remains unbound.
[[nodiscard]] PrefillExecutionProgressError publish_prefill_state_committed(
    const PrefillExecutionPlan& plan,
    PrefillExecutionProgress& progress) noexcept;

}  // namespace q3x::runtime
