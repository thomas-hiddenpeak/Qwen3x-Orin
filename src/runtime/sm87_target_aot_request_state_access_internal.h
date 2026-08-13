#pragma once

#include "q3x/runtime/sm87_target_aot_request_state.h"

#include <array>
#include <cstddef>
#include <cstdint>

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
  std::array<LayerSnapshot, kSm87TargetAotP40LayerCount> layers{};
  BoundSpan final_hidden{};
  std::array<BoundEvent, kSm87TargetAotP40GlobalEventCount> global_events{};
  Sm87TargetAotRequestTransactionPhase transaction_phase =
      Sm87TargetAotRequestTransactionPhase::kInvalid;
  bool engine_rope_is_external = false;
  bool all_kv_in_place_aliases_verified = false;
};

class Sm87TargetAotRequestStateAccess final {
 public:
  // This gate intentionally exposes only an immutable owner-derived
  // snapshot.  Transaction mutation enters together with the source-private
  // executor so readiness can be issued from owner-bound CUDA completion
  // events; a host-only flag setter is not an admissible commit authority.
  [[nodiscard]] static const OwnerSnapshot* snapshot(
      const Sm87TargetAotP40RequestState& owner) noexcept;
};

[[nodiscard]] bool validate_owner_snapshot(
    const OwnerSnapshot& snapshot) noexcept;

}  // namespace q3x::runtime::sm87_target_aot_request_detail
