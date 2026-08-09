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

// Engine-lifetime full-Attention ownership for the layer-major route.  The
// incumbent keeps the established C512 arithmetic spans and their exact
// fixed selector.  The architecture candidate gives one native grouped-Q64
// online-softmax launch ownership of the complete logical operator panel.
// This is part of the sealed plan, never a request-time or environment
// selector.
enum class LayerMajorPrefillFullAttentionTactic : std::uint8_t {
  kExactSegmentedC512 = 0,
  kNativeGroupQ64Panel,
};

[[nodiscard]] constexpr bool
is_valid_layer_major_prefill_full_attention_tactic(
    const LayerMajorPrefillFullAttentionTactic tactic) noexcept {
  return tactic ==
             LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512 ||
         tactic ==
             LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel;
}

// Engine-lifetime projection ownership for the layer-major route. The exact
// incumbent retains the authenticated C512 arithmetic ledger. The segmented
// Marlin operator-panel value is an explicit, default-off dependency screen:
// it submits each logical FP8/NVFP4 projection through the existing wrapper,
// whose large-N kernels still segment at no more than M1024, while leaving
// BF16 A/B on the exact contract. It is not the true native large-M tactic.
// Selection is sealed into the bound plan and never changes per request.
enum class LayerMajorPrefillProjectionTactic : std::uint8_t {
  kExactSegmentedC512 = 0,
  kSegmentedMarlinOperatorPanel,
};

[[nodiscard]] constexpr bool is_valid_layer_major_prefill_projection_tactic(
    const LayerMajorPrefillProjectionTactic tactic) noexcept {
  return tactic ==
             LayerMajorPrefillProjectionTactic::kExactSegmentedC512 ||
         tactic ==
             LayerMajorPrefillProjectionTactic::kSegmentedMarlinOperatorPanel;
}

// Preserve full-capacity work while preventing a final one-token panel or
// physical segment.  Once only the final full-capacity unit plus its tail
// remain, split that suffix into ceil/floor halves.  This keeps the minimum
// number of units, retains every earlier full unit, and prevents the exact
// optimized routes from being defeated by a pathological scalar tail.
[[nodiscard]] constexpr std::size_t
next_layer_major_prefill_operator_panel_token_count(
    const std::size_t remaining_prompt_tokens) noexcept {
  if (remaining_prompt_tokens <= kLayerMajorPrefillOperatorPanelTokens) {
    return remaining_prompt_tokens;
  }
  if (remaining_prompt_tokens <
      2U * kLayerMajorPrefillOperatorPanelTokens) {
    return (remaining_prompt_tokens + 1U) / 2U;
  }
  return kLayerMajorPrefillOperatorPanelTokens;
}

// Canonical exact-route subdivision retained for the legacy scheduling
// contract and its historical evidence.
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

// The layer-major compatibility executor accepts every C1..C512 geometry.
// Keep earlier C512 work full and split only the final C512-plus-tail suffix
// into ceil/floor halves. This avoids a scalar layer tail without nearly
// doubling the masked Marlin launch count over the whole panel.
[[nodiscard]] constexpr std::size_t
next_layer_major_prefill_physical_segment_token_count(
    const std::size_t remaining_panel_tokens) noexcept {
  if (remaining_panel_tokens <= kPrefillPhysicalSegmentMaximumTokens) {
    return remaining_panel_tokens;
  }
  if (remaining_panel_tokens <
      2U * kPrefillPhysicalSegmentMaximumTokens) {
    return (remaining_panel_tokens + 1U) / 2U;
  }
  return kPrefillPhysicalSegmentMaximumTokens;
}

[[nodiscard]] constexpr bool
is_layer_major_prefill_physical_segment_token_count(
    const std::size_t token_count) noexcept {
  return token_count != 0U &&
         token_count <= kPrefillPhysicalSegmentMaximumTokens;
}

// Exact arithmetic is owned by the historical C512-balanced compatibility
// route, even when a layer is submitted as one larger operator panel.  The
// ledger makes those physical ownership boundaries explicit so every
// projection and stateful operator can retain the oracle's masked-tail
// specialization sequence without giving up panel-wide storage and
// scheduling.
inline constexpr std::size_t kLayerMajorPrefillMaximumArithmeticSpanCount =
    (kLayerMajorPrefillOperatorPanelTokens +
     kPrefillPhysicalSegmentMaximumTokens - 1U) /
    kPrefillPhysicalSegmentMaximumTokens;

struct LayerMajorPrefillArithmeticSpan {
  std::uint32_t token_offset = 0U;
  std::uint32_t token_count = 0U;
};

struct LayerMajorPrefillArithmeticSpanLedger {
  std::array<LayerMajorPrefillArithmeticSpan,
             kLayerMajorPrefillMaximumArithmeticSpanCount>
      spans{};
  std::size_t span_count = 0U;
  std::uint32_t token_count = 0U;
};

[[nodiscard]] constexpr LayerMajorPrefillArithmeticSpanLedger
make_layer_major_prefill_arithmetic_span_ledger(
    const std::size_t panel_token_count) noexcept {
  LayerMajorPrefillArithmeticSpanLedger ledger;
  if (panel_token_count == 0U ||
      panel_token_count > kLayerMajorPrefillOperatorPanelTokens) {
    return ledger;
  }

  std::size_t offset = 0U;
  std::size_t remaining = panel_token_count;
  while (remaining != 0U &&
         ledger.span_count < ledger.spans.size()) {
    const std::size_t span_token_count =
        next_layer_major_prefill_physical_segment_token_count(remaining);
    if (!is_layer_major_prefill_physical_segment_token_count(
            span_token_count) ||
        span_token_count > remaining) {
      return {};
    }
    ledger.spans[ledger.span_count++] = LayerMajorPrefillArithmeticSpan{
        static_cast<std::uint32_t>(offset),
        static_cast<std::uint32_t>(span_token_count)};
    offset += span_token_count;
    remaining -= span_token_count;
  }
  if (remaining != 0U || offset != panel_token_count) {
    return {};
  }
  ledger.token_count = static_cast<std::uint32_t>(panel_token_count);
  return ledger;
}

[[nodiscard]] constexpr bool
is_valid_layer_major_prefill_arithmetic_span_ledger(
    const LayerMajorPrefillArithmeticSpanLedger& ledger) noexcept {
  if (ledger.token_count == 0U ||
      ledger.token_count > kLayerMajorPrefillOperatorPanelTokens ||
      ledger.span_count == 0U ||
      ledger.span_count > ledger.spans.size()) {
    return false;
  }
  std::size_t expected_offset = 0U;
  for (std::size_t index = 0U; index < ledger.span_count; ++index) {
    const LayerMajorPrefillArithmeticSpan& span = ledger.spans[index];
    if (expected_offset >= ledger.token_count) {
      return false;
    }
    const std::size_t remaining = ledger.token_count - expected_offset;
    if (span.token_offset != expected_offset ||
        !is_layer_major_prefill_physical_segment_token_count(
            span.token_count) ||
        span.token_count !=
            next_layer_major_prefill_physical_segment_token_count(
                remaining)) {
      return false;
    }
    expected_offset += span.token_count;
  }
  return expected_offset == ledger.token_count;
}

enum class PrefillBf16AbArithmeticTactic : std::uint8_t {
  kEstablishedM32ProjectionPair = 0,
};

enum class PrefillFp8ArithmeticTactic : std::uint8_t {
  kOracleSpanMarlin = 0,
};

enum class PrefillNvFp4ArithmeticTactic : std::uint8_t {
  kOracleSpanGateSiluDownSequence = 0,
};

enum class PrefillGdnArithmeticTactic : std::uint8_t {
  kOracleSpanWholeRawQkv = 0,
};

enum class PrefillAttentionPreprocessArithmeticTactic : std::uint8_t {
  kOracleSpanC16FixedReference256 = 0,
};

struct LayerMajorPrefillArithmeticContract {
  std::uint32_t version = 1U;
  PrefillBf16AbArithmeticTactic bf16_ab =
      PrefillBf16AbArithmeticTactic::kEstablishedM32ProjectionPair;
  PrefillFp8ArithmeticTactic fp8 =
      PrefillFp8ArithmeticTactic::kOracleSpanMarlin;
  PrefillNvFp4ArithmeticTactic nvfp4 =
      PrefillNvFp4ArithmeticTactic::kOracleSpanGateSiluDownSequence;
  PrefillGdnArithmeticTactic gdn =
      PrefillGdnArithmeticTactic::kOracleSpanWholeRawQkv;
  PrefillAttentionPreprocessArithmeticTactic attention_preprocess =
      PrefillAttentionPreprocessArithmeticTactic::
          kOracleSpanC16FixedReference256;
  bool reset_fp8_locks_per_projection_span = true;
  bool nvfp4_interleaves_gate_silu_down_per_span = true;
  bool nvfp4_down_reuses_gate_up_locks = true;
  bool environment_independent = true;
};

inline constexpr LayerMajorPrefillArithmeticContract
    kLayerMajorPrefillExactArithmeticContract{};

[[nodiscard]] constexpr bool is_valid_layer_major_prefill_arithmetic_contract(
    const LayerMajorPrefillArithmeticContract& contract) noexcept {
  return contract.version == 1U &&
         contract.bf16_ab == PrefillBf16AbArithmeticTactic::
                                   kEstablishedM32ProjectionPair &&
         contract.fp8 == PrefillFp8ArithmeticTactic::kOracleSpanMarlin &&
         contract.nvfp4 == PrefillNvFp4ArithmeticTactic::
                                kOracleSpanGateSiluDownSequence &&
         contract.gdn ==
             PrefillGdnArithmeticTactic::kOracleSpanWholeRawQkv &&
         contract.attention_preprocess ==
             PrefillAttentionPreprocessArithmeticTactic::
                 kOracleSpanC16FixedReference256 &&
         contract.reset_fp8_locks_per_projection_span &&
         contract.nvfp4_interleaves_gate_silu_down_per_span &&
         contract.nvfp4_down_reuses_gate_up_locks &&
         contract.environment_independent;
}

static_assert(kLayerMajorPrefillMaximumArithmeticSpanCount == 16U);
static_assert(is_valid_layer_major_prefill_arithmetic_contract(
    kLayerMajorPrefillExactArithmeticContract));
static_assert(is_valid_layer_major_prefill_arithmetic_span_ledger(
    make_layer_major_prefill_arithmetic_span_ledger(513U)));

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
