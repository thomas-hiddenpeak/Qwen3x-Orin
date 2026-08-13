#pragma once

#include "q3x/runtime/sm87_target_aot_request_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::runtime::sm87_target_aot_request_detail {

struct BoundSpan final {
  void* device_data = nullptr;
  std::uint64_t byte_size = 0U;
  std::uint64_t physical_span_identity = 0U;
  std::uint64_t lifetime_identity = 0U;
  std::uint64_t publication_identity = 0U;
};

struct BoundEvent final {
  const void* cuda_event = nullptr;
  std::uint64_t event_identity = 0U;
};

struct OwnerEpochs final {
  std::uint64_t admission_epoch = 0U;
  std::uint64_t request_transaction_epoch = 0U;
  std::uint64_t cold_reset_epoch = 0U;
  std::uint64_t event_reset_epoch = 0U;
  std::uint64_t staged_kv_epoch = 0U;
  std::uint64_t position_rope_epoch = 0U;
};

struct OwnerTableIdentities final {
  std::uint64_t request_memory_plan = 0U;
  std::uint64_t layer_residual = 0U;
  std::uint64_t input_norm = 0U;
  std::uint64_t gdn_raw_qkvz = 0U;
  std::uint64_t gdn_bf16_ab = 0U;
  std::uint64_t gdn_output = 0U;
  std::uint64_t attention_raw_qkv = 0U;
  std::uint64_t attention_processed_qkv = 0U;
  std::uint64_t attention_output = 0U;
  std::uint64_t mlp_activated = 0U;
  std::uint64_t down_branch = 0U;
  std::uint64_t staged_kv_transaction = 0U;
  std::uint64_t conv_history_transaction = 0U;
  std::uint64_t recurrent_state_transaction = 0U;
  std::uint64_t layer_completion_events = 0U;
  std::uint64_t request_transaction_events = 0U;
  std::uint64_t final_hidden = 0U;
};

struct GdnLayerSnapshot final {
  BoundSpan raw_qkvz{};
  BoundSpan bf16_ab{};
  BoundSpan output{};
  BoundSpan o_branch{};
  BoundSpan final_conv_history{};
  BoundSpan final_recurrent_state{};
};

struct AttentionLayerSnapshot final {
  BoundSpan raw_q_gate{};
  BoundSpan raw_k{};
  BoundSpan raw_v{};
  BoundSpan processed_q{};
  BoundSpan processed_gate{};
  BoundSpan processed_k{};
  BoundSpan processed_v{};
  BoundSpan pre_gate_output{};
  BoundSpan gated_output{};
  BoundSpan o_branch{};
  bool k_in_place_transaction_alias_verified = false;
  bool v_in_place_transaction_alias_verified = false;
};

struct MlpLayerSnapshot final {
  BoundSpan normalized_input{};
  BoundSpan activated{};
  BoundSpan down_branch{};
};

struct LayerSnapshot final {
  Sm87TargetAotRequestLayerKind kind =
      Sm87TargetAotRequestLayerKind::kInvalid;
  std::size_t family_ordinal = 0U;
  BoundSpan residual{};
  BoundSpan input_normalized{};
  GdnLayerSnapshot gdn{};
  AttentionLayerSnapshot attention{};
  MlpLayerSnapshot mlp{};
  std::array<BoundEvent, kSm87TargetAotP40LayerEventCount>
      completion_events{};
};

struct OwnerSnapshot final {
  const Sm87TargetAotP40RequestMemoryPlan* plan = nullptr;
  void* allocation_base = nullptr;
  std::uint64_t allocation_bytes = 0U;
  std::uint64_t allocation_identity = 0U;
  OwnerEpochs epochs{};
  OwnerTableIdentities tables{};
  BoundSpan token_ids{};
  std::array<LayerSnapshot, kSm87TargetAotP40LayerCount> layers{};
  BoundSpan final_hidden{};
  std::array<BoundEvent, kSm87TargetAotP40GlobalEventCount> global_events{};
  Sm87TargetAotRequestTransactionPhase transaction_phase =
      Sm87TargetAotRequestTransactionPhase::kInvalid;
  bool engine_rope_is_external = false;
  bool all_kv_in_place_aliases_verified = false;
};

// These completion points are deliberately ordered.  The executor may only
// record the next point on the owner stream; a caller cannot turn an arbitrary
// host-side readiness assertion into publication authority.
enum class LayerCompletionPoint : std::uint8_t {
  kInputProjections = 0U,
  kStateOrAttentionPreparation,
  kCore,
  kOutputProjection,
  kPostCoreResidual,
  kGateUp,
  kDown,
  kLayerComplete,
};

enum class GlobalCompletionPoint : std::uint8_t {
  kTokenIdsReady = 0U,
  kEmbeddingComplete,
  kAllLayersComplete,
  kFinalNormComplete,
  kFinalHiddenComplete,
  kPersistentStateStaged,
  kRequestCommit,
};

static_assert(static_cast<std::size_t>(LayerCompletionPoint::kLayerComplete) +
                  1U ==
              kSm87TargetAotP40LayerEventCount);
static_assert(static_cast<std::size_t>(GlobalCompletionPoint::kRequestCommit) +
                  1U ==
              kSm87TargetAotP40GlobalEventCount);

enum class ExecutionTransactionError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kOwnerInvalid,
  kInvalidTransition,
  kConcurrentAccess,
  kTransactionMismatch,
  kCompletionOutOfOrder,
  kCompletionIncomplete,
  kCudaEventRecordFailure,
  kCudaEventSynchronizeFailure,
  kCudaStreamSynchronizeFailure,
};

struct ExecutionTransactionStatus final {
  ExecutionTransactionError code = ExecutionTransactionError::kNone;
  const char* context = "none";
  int cuda_error = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return code == ExecutionTransactionError::kNone;
  }
};

// One begin() call mints this owner-derived capability.  Its private
// constructor prevents callers from manufacturing an epoch/stream pairing.
class OwnerBoundExecutionTransaction final {
 public:
  [[nodiscard]] void* cuda_stream() const noexcept { return cuda_stream_; }
  // Device-visible, owner-lifetime control word.  A nonzero value requests
  // bounded cancellation at the next constituent-kernel safe point.  The
  // caller may observe or pass it to an authenticated launcher but cannot
  // obtain the host publication address.
  [[nodiscard]] const std::uint32_t* device_cancellation_signal()
      const noexcept {
    return device_cancellation_signal_;
  }
  [[nodiscard]] std::uint64_t transaction_epoch() const noexcept {
    return transaction_epoch_;
  }
  [[nodiscard]] std::uint64_t admission_epoch() const noexcept {
    return admission_epoch_;
  }

 private:
  OwnerBoundExecutionTransaction(
      Sm87TargetAotP40RequestState* owner, void* cuda_stream,
      const std::uint64_t stream_identity,
      const std::uint64_t allocation_identity,
      const std::uint32_t* const device_cancellation_signal,
      const std::uint64_t cancellation_signal_identity,
      const std::uint64_t admission_epoch,
      const std::uint64_t transaction_epoch) noexcept
      : owner_(owner),
        cuda_stream_(cuda_stream),
        stream_identity_(stream_identity),
        allocation_identity_(allocation_identity),
        device_cancellation_signal_(device_cancellation_signal),
        cancellation_signal_identity_(cancellation_signal_identity),
        admission_epoch_(admission_epoch),
        transaction_epoch_(transaction_epoch) {}

  Sm87TargetAotP40RequestState* owner_ = nullptr;
  void* cuda_stream_ = nullptr;
  std::uint64_t stream_identity_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  const std::uint32_t* device_cancellation_signal_ = nullptr;
  std::uint64_t cancellation_signal_identity_ = 0U;
  std::uint64_t admission_epoch_ = 0U;
  std::uint64_t transaction_epoch_ = 0U;

  friend class Sm87TargetAotRequestStateAccess;
};

struct BeginExecutionResult final {
  std::optional<OwnerBoundExecutionTransaction> transaction;
  ExecutionTransactionStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return transaction.has_value() && static_cast<bool>(status);
  }
};

// Shared transition ledger.  Production mutation is performed only after the
// corresponding CUDA operation succeeds.  The host fixture below exercises
// this same ledger without manufacturing device completion.
struct ExecutionTransactionLedger final {
  Sm87TargetAotRequestTransactionPhase phase =
      Sm87TargetAotRequestTransactionPhase::kAdmittedUnpublished;
  std::uint64_t transaction_epoch = 0U;
  std::size_t next_layer = 0U;
  std::size_t next_layer_completion = 0U;
  std::size_t next_global_completion = 0U;
  bool final_event_recorded = false;
};

class HostExecutionTransactionFixture final {
 public:
  [[nodiscard]] Sm87TargetAotRequestTransactionPhase phase() const noexcept {
    return ledger_.phase;
  }
  [[nodiscard]] std::uint64_t transaction_epoch() const noexcept {
    return ledger_.transaction_epoch;
  }

 private:
  ExecutionTransactionLedger ledger_{};
  friend class Sm87TargetAotRequestStateAccess;
};

class Sm87TargetAotRequestStateAccess final {
 public:
  // This gate intentionally exposes only an immutable owner-derived
  // snapshot.  Transaction mutation enters together with the source-private
  // executor so readiness can be issued from owner-bound CUDA completion
  // events; a host-only flag setter is not an admissible commit authority.
  [[nodiscard]] static const OwnerSnapshot* snapshot(
      const Sm87TargetAotP40RequestState& owner) noexcept;

  [[nodiscard]] static BeginExecutionResult begin(
      Sm87TargetAotP40RequestState& owner) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus record_layer_completion(
      const OwnerBoundExecutionTransaction& transaction, std::size_t layer,
      LayerCompletionPoint point) noexcept;
  // Waits only on the owner-issued LayerComplete event for this transaction
  // generation.  The raw event remains private and no ledger transition is
  // manufactured by the host wait.
  [[nodiscard]] static ExecutionTransactionStatus wait_layer_completion(
      const OwnerBoundExecutionTransaction& transaction,
      std::size_t layer) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus record_global_completion(
      const OwnerBoundExecutionTransaction& transaction,
      GlobalCompletionPoint point) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus commit(
      const OwnerBoundExecutionTransaction& transaction) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus cancel(
      const OwnerBoundExecutionTransaction& transaction) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus cancel_admitted(
      Sm87TargetAotP40RequestState& owner) noexcept;

  // Pure-host fixture for illegal-transition coverage.  These methods share
  // the production transition functions but never claim CUDA completion.
  [[nodiscard]] static HostExecutionTransactionFixture host_fixture(
      std::uint64_t transaction_epoch = 1U) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus host_begin(
      HostExecutionTransactionFixture& fixture) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus host_record_layer(
      HostExecutionTransactionFixture& fixture, std::uint64_t epoch,
      std::size_t layer, LayerCompletionPoint point) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus host_record_global(
      HostExecutionTransactionFixture& fixture, std::uint64_t epoch,
      GlobalCompletionPoint point) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus host_commit(
      HostExecutionTransactionFixture& fixture, std::uint64_t epoch) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus host_cancel(
      HostExecutionTransactionFixture& fixture, std::uint64_t epoch) noexcept;
  [[nodiscard]] static ExecutionTransactionStatus host_rearm(
      HostExecutionTransactionFixture& fixture,
      std::uint64_t next_transaction_epoch) noexcept;
};

[[nodiscard]] bool validate_owner_snapshot(
    const OwnerSnapshot& snapshot) noexcept;

}  // namespace q3x::runtime::sm87_target_aot_request_detail
