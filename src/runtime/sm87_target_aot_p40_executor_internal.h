#pragma once

#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/sm87_target_aot_request_state.h"
#include "sm87_target_aot_projection_complete_execution_access_internal.h"
#include "sm87_target_aot_request_state_access_internal.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace q3x::runtime::sm87_target_aot_p40_executor_detail {

// This executor is an admission-only whole-request candidate.  Merely linking
// this header never opens it; all three owner/executor build gates must be
// present in the same binary.
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION) &&       \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_REQUEST_STATE_V1_ADMISSION)
inline constexpr bool kSm87TargetAotP40ExecutorAdmissionCompiled = true;
#else
inline constexpr bool kSm87TargetAotP40ExecutorAdmissionCompiled = false;
#endif

inline constexpr std::size_t kSm87TargetAotP40HandoffTokens = 1U;
inline constexpr std::size_t kSm87TargetAotP40FullAttentionPanelTokens =
    8'000U;
inline constexpr std::size_t kSm87TargetAotP40FullAttentionPanels = 5U;
inline constexpr std::size_t kSm87TargetAotP40ProjectionAssets = 256U;
inline constexpr std::size_t kSm87TargetAotP40ProducerLanes = 2U;
inline constexpr std::size_t kSm87TargetAotP40ProducerJoinEventSlots = 2U;
inline constexpr std::size_t kSm87TargetAotP40Vocabulary = 248'320U;
inline constexpr std::size_t kSm87TargetAotP40Bf16LogitsBytes =
    kSm87TargetAotP40Vocabulary * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87TargetAotP40GreedyWorkspaceResults = 33U;
inline constexpr std::size_t kSm87TargetAotP40GreedyResultBytes = 8U;
inline constexpr std::size_t kSm87TargetAotP40GreedyWorkspaceBytes =
    kSm87TargetAotP40GreedyWorkspaceResults *
    kSm87TargetAotP40GreedyResultBytes;
inline constexpr std::size_t kSm87TargetAotP40FinalHandoffScratchBytes =
    kSm87TargetAotP40Bf16LogitsBytes +
    kSm87TargetAotP40GreedyWorkspaceBytes;

// The logits and greedy workspace reuse the family arena only after the
// terminal Down and LayerComplete events.  No logits byte aliases persistent
// state, the live residual, or final_hidden.  This is a lifetime plan rather
// than a raw arena accessor: the executor derives the physical span from the
// terminal layer's owner-bound, now-dead MLP activation view.
struct Sm87TargetAotP40FinalHandoffScratchContract final {
  std::size_t logits_offset_bytes = 0U;
  std::size_t logits_bytes = 0U;
  std::size_t greedy_workspace_offset_bytes = 0U;
  std::size_t greedy_workspace_bytes = 0U;
  std::size_t required_bytes = 0U;
  std::size_t family_span_bytes = 0U;
  bool begins_at_family_arena = false;
  bool live_only_after_all_layers = false;
  bool overlaps_final_hidden = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return logits_offset_bytes == 0U &&
           logits_bytes == kSm87TargetAotP40Bf16LogitsBytes &&
           greedy_workspace_offset_bytes == logits_bytes &&
           greedy_workspace_offset_bytes % alignof(std::uint32_t) == 0U &&
           greedy_workspace_bytes ==
               kSm87TargetAotP40GreedyWorkspaceBytes &&
           required_bytes ==
               greedy_workspace_offset_bytes + greedy_workspace_bytes &&
           required_bytes == kSm87TargetAotP40FinalHandoffScratchBytes &&
           required_bytes <= family_span_bytes && begins_at_family_arena &&
           live_only_after_all_layers && !overlaps_final_hidden;
  }
};

[[nodiscard]] constexpr Sm87TargetAotP40FinalHandoffScratchContract
sm87_target_aot_p40_final_handoff_scratch_contract() noexcept {
  return {
      0U,
      kSm87TargetAotP40Bf16LogitsBytes,
      kSm87TargetAotP40Bf16LogitsBytes,
      kSm87TargetAotP40GreedyWorkspaceBytes,
      kSm87TargetAotP40FinalHandoffScratchBytes,
      static_cast<std::size_t>(kSm87TargetAotP40FamilyArenaBytes),
      true,
      true,
      false,
  };
}

// Host-visible frozen DAG facts.  They are deliberately independent of CUDA
// so the exact route can be audited without manufacturing device readiness.
struct Sm87TargetAotP40ExecutionContract final {
  std::size_t prompt_tokens = 0U;
  std::size_t handoff_tokens = 0U;
  std::size_t layers = 0U;
  std::size_t gdn_layers = 0U;
  std::size_t full_attention_layers = 0U;
  std::size_t full_attention_panel_tokens = 0U;
  std::size_t full_attention_panels_per_layer = 0U;
  std::size_t projection_assets = 0U;
  std::size_t layer_completion_events = 0U;
  std::size_t global_completion_events = 0U;
  std::size_t producer_lanes = 0U;
  std::size_t producer_join_event_slots = 0U;
  bool exact_p40000_only = false;
  bool cold_request_only = false;
  bool one_owner_stream = false;
  bool producer_launches_initially_serial = false;
  bool producer_parallelism_claimed = true;
  bool mtp_permitted = true;
  bool fallback_permitted = true;
  bool cublaslt_permitted = true;
  bool jit_permitted = true;
  bool final_handoff_callable = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return prompt_tokens == kSm87TargetAotP40PromptTokens &&
           handoff_tokens == kSm87TargetAotP40HandoffTokens &&
           layers == kSm87TargetAotP40LayerCount &&
           gdn_layers == kSm87TargetAotP40GdnLayerCount &&
           full_attention_layers == kSm87TargetAotP40FullLayerCount &&
           gdn_layers + full_attention_layers == layers &&
           full_attention_panel_tokens ==
               kSm87TargetAotP40FullAttentionPanelTokens &&
           full_attention_panels_per_layer ==
               kSm87TargetAotP40FullAttentionPanels &&
           full_attention_panel_tokens * full_attention_panels_per_layer ==
               prompt_tokens &&
           projection_assets == kSm87TargetAotP40ProjectionAssets &&
           layer_completion_events ==
               kSm87TargetAotP40LayerCount *
                   kSm87TargetAotP40LayerEventCount &&
           global_completion_events ==
               kSm87TargetAotP40GlobalEventCount &&
           producer_lanes == kSm87TargetAotP40ProducerLanes &&
           producer_join_event_slots ==
               kSm87TargetAotP40ProducerJoinEventSlots &&
           exact_p40000_only && cold_request_only && one_owner_stream &&
           producer_launches_initially_serial &&
           !producer_parallelism_claimed && !mtp_permitted &&
           !fallback_permitted && !cublaslt_permitted && !jit_permitted &&
           final_handoff_callable;
  }
};

[[nodiscard]] constexpr Sm87TargetAotP40ExecutionContract
sm87_target_aot_p40_execution_contract() noexcept {
  return {
      kSm87TargetAotP40PromptTokens,
      kSm87TargetAotP40HandoffTokens,
      kSm87TargetAotP40LayerCount,
      kSm87TargetAotP40GdnLayerCount,
      kSm87TargetAotP40FullLayerCount,
      kSm87TargetAotP40FullAttentionPanelTokens,
      kSm87TargetAotP40FullAttentionPanels,
      kSm87TargetAotP40ProjectionAssets,
      kSm87TargetAotP40LayerCount * kSm87TargetAotP40LayerEventCount,
      kSm87TargetAotP40GlobalEventCount,
      kSm87TargetAotP40ProducerLanes,
      kSm87TargetAotP40ProducerJoinEventSlots,
      true,
      true,
      true,
      true,
      false,
      false,
      false,
      false,
      false,
      true,
  };
}

[[nodiscard]] constexpr bool sm87_target_aot_p40_exact_request_shape(
    const std::size_t prompt_tokens,
    const std::size_t handoff_tokens) noexcept {
  return prompt_tokens == kSm87TargetAotP40PromptTokens &&
         handoff_tokens == kSm87TargetAotP40HandoffTokens;
}

// Engine-owned RoPE tables.  Their lifetime must cover bind, execution, and
// transaction drain.  identity is an opaque engine-lifetime identity, not a
// caller-generated readiness assertion.
struct Sm87TargetAotP40EngineRope final {
  const float* cosines = nullptr;
  const float* sines = nullptr;
  std::size_t position_count = 0U;
  std::size_t rotary_pairs = 0U;
  std::uint64_t identity = 0U;
};

enum class Sm87TargetAotP40ExecutorError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kInvalidModelWeights,
  kInvalidRequestOwner,
  kInvalidEngineRope,
  kIncompleteProjectionAssets,
  kStaticResourcePreflightFailure,
  kHostAllocationFailure,
  kInvalidExecuteInput,
  kTransactionBeginFailure,
  kLiveOwnerValidationFailure,
  kLiveAssetValidationFailure,
  kCudaOperationFailure,
  kTransactionRecordFailure,
  kInvalidHandoffResult,
  kTransactionCancelFailure,
};

struct Sm87TargetAotP40ExecutorStatus final {
  Sm87TargetAotP40ExecutorError code = Sm87TargetAotP40ExecutorError::kNone;
  const char* context = "none";
  std::size_t layer = kSm87TargetAotP40LayerCount;
  int cuda_error = 0;
  sm87_target_aot_request_detail::ExecutionTransactionError
      transaction_error =
          sm87_target_aot_request_detail::ExecutionTransactionError::kNone;

  [[nodiscard]] explicit operator bool() const noexcept {
    return code == Sm87TargetAotP40ExecutorError::kNone;
  }
};

enum class Sm87TargetAotP40Finalization : std::uint8_t {
  kNotStarted = 0U,
  kFinalHiddenReady,
  kLogitsReady,
  kHandoffReady,
  kCommitted,
};

struct Sm87TargetAotP40ExecutionReceipt final {
  std::size_t prompt_tokens = 0U;
  std::size_t requested_handoff_tokens = 0U;
  std::size_t completed_layers = 0U;
  std::size_t completed_gdn_layers = 0U;
  std::size_t completed_full_attention_layers = 0U;
  std::size_t completed_attention_panels = 0U;
  std::size_t recorded_layer_events = 0U;
  std::size_t recorded_global_events = 0U;
  std::uint64_t admission_epoch = 0U;
  std::uint64_t transaction_epoch = 0U;
  std::uint32_t handoff_token_id = 0U;
  std::uint16_t handoff_value_bits = 0U;
  std::uint16_t handoff_has_nonfinite = 0U;
  Sm87TargetAotP40Finalization finalization =
      Sm87TargetAotP40Finalization::kNotStarted;
  bool transaction_started = false;
  bool transaction_cancelled = false;
  bool transaction_committed = false;
  bool handoff_result_observed = false;
  bool handoff_complete = false;
  bool used_fallback = false;
  bool used_mtp = false;
  bool used_cublaslt = false;
  bool used_jit = false;
};

struct Sm87TargetAotP40ExecutionResult final {
  Sm87TargetAotP40ExecutorStatus status{};
  Sm87TargetAotP40ExecutionReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.transaction_committed &&
           receipt.handoff_result_observed && receipt.handoff_complete &&
           receipt.handoff_has_nonfinite == 0U &&
           receipt.handoff_token_id < kSm87TargetAotP40Vocabulary &&
           receipt.finalization == Sm87TargetAotP40Finalization::kCommitted;
  }
};

class Sm87TargetAotP40Executor;

struct Sm87TargetAotP40ExecutorBindResult final {
  std::unique_ptr<Sm87TargetAotP40Executor> executor;
  Sm87TargetAotP40ExecutorStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return executor != nullptr && static_cast<bool>(status);
  }
};

// Source-private exact-P40000 executor.  It owns no model, request, RoPE, or
// device storage.  The three owners supplied to bind must outlive it and every
// queued operation.  execute() has no generic route, fallback, MTP, cuBLASLt,
// JIT, or autotune escape hatch.
class Sm87TargetAotP40Executor final {
 public:
  Sm87TargetAotP40Executor(const Sm87TargetAotP40Executor&) = delete;
  Sm87TargetAotP40Executor& operator=(const Sm87TargetAotP40Executor&) =
      delete;
  Sm87TargetAotP40Executor(Sm87TargetAotP40Executor&&) = delete;
  Sm87TargetAotP40Executor& operator=(Sm87TargetAotP40Executor&&) = delete;
  ~Sm87TargetAotP40Executor();

  [[nodiscard]] static Sm87TargetAotP40ExecutorBindResult bind(
      const ModelWeights& model_weights,
      Sm87TargetAotP40RequestState& request_owner,
      const Sm87TargetAotP40EngineRope& engine_rope) noexcept;

  [[nodiscard]] Sm87TargetAotP40ExecutionResult execute(
      const std::uint32_t* host_prompt_token_ids,
      std::size_t prompt_tokens,
      std::size_t requested_handoff_tokens =
          kSm87TargetAotP40HandoffTokens) noexcept;

 private:
  using ProjectionAccess =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAccess;

  Sm87TargetAotP40Executor(
      const ModelWeights& model_weights,
      Sm87TargetAotP40RequestState& request_owner,
      const Sm87TargetAotP40EngineRope& engine_rope,
      ProjectionAccess projection_access,
      std::int32_t device_ordinal,
      void* pinned_handoff_result) noexcept
      : model_weights_(&model_weights),
        request_owner_(&request_owner),
        engine_rope_(engine_rope),
        projection_access_(std::move(projection_access)),
        device_ordinal_(device_ordinal),
        pinned_handoff_result_(pinned_handoff_result) {}

  const ModelWeights* model_weights_ = nullptr;
  Sm87TargetAotP40RequestState* request_owner_ = nullptr;
  Sm87TargetAotP40EngineRope engine_rope_{};
  ProjectionAccess projection_access_;
  std::int32_t device_ordinal_ = -1;
  // Exactly one pinned 8-byte Bf16GreedyArgmaxResult.  All large logits and
  // reduction workspace stay inside the request owner's dead family arena.
  void* pinned_handoff_result_ = nullptr;
};

static_assert(sm87_target_aot_p40_execution_contract().valid());
static_assert(kSm87TargetAotP40FullAttentionPanelTokens *
                      kSm87TargetAotP40FullAttentionPanels ==
                  kSm87TargetAotP40PromptTokens);
static_assert(kSm87TargetAotP40ProjectionAssets ==
              kSm87TargetAotCompleteProjectionDeviceArtifactCount);
static_assert(sm87_target_aot_p40_final_handoff_scratch_contract().valid());
static_assert(kSm87TargetAotP40Bf16LogitsBytes %
                      alignof(std::uint32_t) ==
                  0U);

}  // namespace q3x::runtime::sm87_target_aot_p40_executor_detail
