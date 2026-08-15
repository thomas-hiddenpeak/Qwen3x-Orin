#pragma once

#include "q3x/runtime/sm87_macrofeed_v4_panel_wavefront_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>

namespace q3x::runtime {

namespace sm87_macrofeed_v4_execution_events_detail {
class Sm87MacroFeedV4ExecutionEventsOwner;
}

// Internal host-only admission and ownership ledger for the default-off
// MacroFeed-v4
// request transaction.  It owns no CUDA pointer, stream, event, launcher, or
// selector and grants no production authority.
inline constexpr std::array<std::uint8_t, 8U>
    kSm87MacroFeedV4RequestStateMagic{{'Q', '3', 'X', 'M', '4', 'S', 'T',
                                       '1'}};
inline constexpr std::uint16_t kSm87MacroFeedV4RequestStateAbiMajor = 1U;
inline constexpr std::uint16_t kSm87MacroFeedV4RequestStateAbiMinor = 1U;
inline constexpr std::size_t kSm87MacroFeedV4StateLayerCount = 48U;
inline constexpr std::uint64_t kSm87MacroFeedV4ConvLayerBytes =
    kSm87MacroFeedV4ConvEpochBytes / kSm87MacroFeedV4StateLayerCount;
inline constexpr std::uint64_t kSm87MacroFeedV4GdnStateLayerBytes =
    kSm87MacroFeedV4GdnEpochBytes / kSm87MacroFeedV4StateLayerCount;

static_assert(kSm87MacroFeedV4ConvLayerBytes == 61'440U);
static_assert(kSm87MacroFeedV4GdnStateLayerBytes == 1'572'864U);
static_assert(kSm87MacroFeedV4StateLayerCount *
                      kSm87MacroFeedV4ConvLayerBytes ==
                  kSm87MacroFeedV4ConvEpochBytes);
static_assert(kSm87MacroFeedV4StateLayerCount *
                      kSm87MacroFeedV4GdnStateLayerBytes ==
                  kSm87MacroFeedV4GdnEpochBytes);

struct Sm87MacroFeedV4RecurrentBankBinding final {
  std::uint64_t storage_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t allocation_offset = 0U;
  std::uint64_t bytes = 0U;
};

struct Sm87MacroFeedV4RecurrentLayerSlice final {
  std::size_t state_layer_ordinal = kSm87MacroFeedV4StateLayerCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  std::uint64_t conv_offset = 0U;
  std::uint64_t conv_bytes = 0U;
  std::uint64_t gdn_state_offset = 0U;
  std::uint64_t gdn_state_bytes = 0U;
};

// One request owns one contiguous BF16 KV allocation.  Each natural
// Full-Attention layer (3, 7, ..., 63) receives one adjacent full-capacity K
// plane and V plane.  These are allocation-relative byte offsets only; this
// host ledger intentionally exposes no pointer or executable allocation view.
struct Sm87MacroFeedV4FullAttentionKvLayerSlice final {
  std::size_t attention_layer_ordinal =
      kSm87MacroFeedV4FullAttentionLayerCount;
  std::size_t model_layer = kSm87MacroFeedV4LayerCount;
  std::uint64_t key_full_allocation_origin = 0U;
  std::uint64_t value_full_allocation_origin = 0U;
  std::uint64_t key_bytes = 0U;
  std::uint64_t value_bytes = 0U;
};

struct Sm87MacroFeedV4RequestStateAdmission final {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::string_view candidate_id{};
  std::string_view deployment_plan_id{};
  std::string_view api_route_id{};
  std::uint64_t owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t allocation_bytes = 0U;
  std::array<Sm87MacroFeedV4RecurrentBankBinding, 2U> recurrent_banks{};
  std::array<Sm87MacroFeedV4RecurrentLayerSlice,
             kSm87MacroFeedV4StateLayerCount>
      recurrent_layers{};
  std::uint64_t kv_allocation_identity = 0U;
  std::uint64_t kv_allocation_bytes = 0U;
  std::array<Sm87MacroFeedV4FullAttentionKvLayerSlice,
             kSm87MacroFeedV4FullAttentionLayerCount>
      full_attention_kv_layers{};
  // True only for the explicit five-argument maker.  A derived four-argument
  // identity keeps old host/GDN fixtures source-compatible but cannot mint a
  // Full-Attention KV write grant.
  bool kv_physical_owner_bound = false;
  std::uint64_t private_kv_valid_end_identity = 0U;
  std::uint64_t panel_commit_event_identity = 0U;
  std::uint64_t final_publish_event_identity = 0U;
  std::uint64_t owner_drain_event_identity = 0U;
  std::uint64_t canonical_state_publication_identity = 0U;
  std::uint64_t sequence_length_fence_identity = 0U;
  bool conv_history_copies_active_to_candidate_per_layer = false;
  bool gdn_first_update_reads_active_and_writes_candidate = false;
  bool gdn_continuation_reads_and_writes_candidate = false;
  bool candidate_epoch_fully_assigned_before_swap = false;
  bool whole_epoch_copy_forbidden = false;
  bool private_kv_valid_end = false;
  bool canonical_state_publishes_after_final_panel = false;
  bool sequence_length_is_final_nonfallible_fence = false;
  bool host_only = false;
  bool default_off = false;
  bool test_only = false;
  bool cuda_handles_present = true;
  bool selector_bound = true;
  bool launcher_present = true;
  bool production_dispatch_eligible = true;
};

enum class Sm87MacroFeedV4RequestAdmissionIssue : std::uint32_t {
  kNone = 0U,
  kIdentity = 1U << 0U,
  kBankOwnership = 1U << 1U,
  kLayerLayout = 1U << 2U,
  kTransitionContract = 1U << 3U,
  kVisibilityOwnership = 1U << 4U,
  kDispatchBoundary = 1U << 5U,
  kKvArenaOwnership = 1U << 6U,
};

struct Sm87MacroFeedV4RequestAdmissionValidation final {
  std::uint32_t issue_mask = 0U;
  std::size_t first_bad_state_layer = kSm87MacroFeedV4StateLayerCount;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return issue_mask == 0U;
  }
};

[[nodiscard]] constexpr bool has_sm87_macrofeed_v4_request_admission_issue(
    const Sm87MacroFeedV4RequestAdmissionValidation& validation,
    const Sm87MacroFeedV4RequestAdmissionIssue issue) noexcept {
  return (validation.issue_mask & static_cast<std::uint32_t>(issue)) != 0U;
}

[[nodiscard]] Sm87MacroFeedV4RequestStateAdmission
make_sm87_macrofeed_v4_request_state_admission(
    std::uint64_t owner_identity, std::uint64_t allocation_identity,
    std::uint64_t bank_a_storage_identity,
    std::uint64_t bank_b_storage_identity) noexcept;

// Normal owners use this overload to bind the identity of the one physical KV
// allocation retained by the request.  The four-argument maker remains a
// host-only T0 convenience and derives a deterministic nonzero KV identity.
[[nodiscard]] Sm87MacroFeedV4RequestStateAdmission
make_sm87_macrofeed_v4_request_state_admission(
    std::uint64_t owner_identity, std::uint64_t allocation_identity,
    std::uint64_t bank_a_storage_identity,
    std::uint64_t bank_b_storage_identity,
    std::uint64_t kv_allocation_identity) noexcept;

[[nodiscard]] Sm87MacroFeedV4RequestAdmissionValidation
validate_sm87_macrofeed_v4_request_state_admission(
    const Sm87MacroFeedV4RequestStateAdmission& admission) noexcept;

enum class Sm87MacroFeedV4RequestStatePhase : std::uint8_t {
  kInvalid = 0U,
  kAdmittedPrivate,
  kPanelActive,
  kPanelReady,
  kBetweenPanelsPrivate,
  kAllPanelsPrivate,
  kFinalPublicationArmed,
  kCanonicalStatePublished,
  kSequenceLengthPublished,
  kCancelled,
  kFailed,
};

enum class Sm87MacroFeedV4RequestDiscardReason : std::uint8_t {
  kInvalid = 0U,
  kCancelled,
  kFailed,
};

enum class Sm87MacroFeedV4RequestStateError : std::uint8_t {
  kNone = 0U,
  kAdmissionInvalid,
  kAllocationFailure,
  kInvalidTransition,
  kPanelMismatch,
  kLayerMismatch,
  kLayerKindMismatch,
  kDuplicateCompletion,
  kCandidateIncomplete,
  kKvValidEndMismatch,
  kInvalidDiscardReason,
  kFinalPublicationIncomplete,
  kCapabilityMismatch,
  kEventReceiptMissing,
  kEventReceiptMismatch,
  kGdnLayerGrantPending,
  kGdnLayerGrantMismatch,
  kFullAttentionKvGrantPending,
  kFullAttentionKvGrantMismatch,
  kRequestEpochExhausted,
  kPhysicalExecutionRequired,
  kPanelGenerationMismatch,
};

struct Sm87MacroFeedV4RequestStateStatus final {
  Sm87MacroFeedV4RequestStateError error =
      Sm87MacroFeedV4RequestStateError::kNone;
  const char* context = "none";
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::size_t layer = kSm87MacroFeedV4LayerCount;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return error == Sm87MacroFeedV4RequestStateError::kNone;
  }
};

struct Sm87MacroFeedV4RequestStateSnapshot final {
  Sm87MacroFeedV4RequestStatePhase phase =
      Sm87MacroFeedV4RequestStatePhase::kInvalid;
  std::size_t active_bank_index = 2U;
  std::size_t candidate_bank_index = 2U;
  std::uint64_t active_bank_identity = 0U;
  std::uint64_t candidate_bank_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint64_t previous_request_epoch = 0U;
  std::size_t runtime_cold_rearm_count = 0U;
  std::uint64_t runtime_recurrent_zero_bytes = 0U;
  std::uint64_t state_epoch = 0U;
  std::uint64_t pending_event_receipt_identity = 0U;
  std::uint64_t pending_gdn_layer_grant_identity = 0U;
  std::uint64_t pending_full_attention_kv_grant_identity = 0U;
  std::size_t completed_panels = 0U;
  std::size_t active_panel = kSm87MacroFeedV4PanelCount;
  std::size_t next_model_layer = 0U;
  std::size_t panel_conv_layers_prepared = 0U;
  std::size_t panel_gdn_layers_assigned = 0U;
  std::size_t panel_kv_layers_staged = 0U;
  std::uint64_t panel_conv_copy_bytes = 0U;
  std::uint64_t panel_gdn_assignment_bytes = 0U;
  std::uint64_t total_conv_copy_bytes = 0U;
  std::uint64_t total_gdn_assignment_bytes = 0U;
  std::uint64_t whole_epoch_copy_bytes = 0U;
  std::size_t private_kv_valid_end = 0U;
  std::size_t candidate_kv_valid_end = 0U;
  std::size_t canonical_kv_valid_end = 0U;
  std::size_t canonical_sequence_length = 0U;
  std::uint64_t canonical_recurrent_source_identity = 0U;
  std::uint64_t canonical_recurrent_target_identity = 0U;
  std::uint64_t canonical_recurrent_copy_bytes = 0U;
  std::size_t panel_swap_count = 0U;
  std::uint64_t last_committed_panel_generation = 0U;
  std::size_t candidate_discard_count = 0U;
  std::uint64_t last_discarded_candidate_identity = 0U;
  std::uint64_t last_invalidated_gdn_layer_grant_identity = 0U;
  std::uint64_t last_invalidated_full_attention_kv_grant_identity = 0U;
  std::uint64_t physical_owner_drain_receipt_identity = 0U;
  std::uint64_t physical_owner_drain_panel_generation = 0U;
  std::uint64_t physical_final_publish_receipt_identity = 0U;
  std::uint64_t physical_final_publish_panel_generation = 0U;
  bool current_conv_layer_prepared = false;
  bool candidate_epoch_complete = false;
  bool fallible_work_closed = false;
  bool canonical_state_published = false;
  // This is a host-ledger ordering bit only.  This slice never issues a
  // Decode view and never attests physical device execution.
  bool logical_sequence_fence_published = false;
  bool decode_access_issued = false;
  bool physical_execution_receipt_issued = false;
  bool physical_owner_drain_was_poison_terminal = false;
  bool default_off = false;
  bool host_only = false;
  bool production_dispatch_eligible = true;
};

class Sm87MacroFeedV4RequestState;

// Copyable owner-issued capability.  Its constructor and bound owner are
// private, so callers cannot synthesize another request epoch or redirect it
// to another state owner.  It carries identities only and exposes no raw
// allocation, CUDA handle, or executable view.
class Sm87MacroFeedV4RequestStateSealedAccess final {
 public:
  Sm87MacroFeedV4RequestStateSealedAccess() = delete;
  Sm87MacroFeedV4RequestStateSealedAccess(
      const Sm87MacroFeedV4RequestStateSealedAccess&) = default;
  Sm87MacroFeedV4RequestStateSealedAccess& operator=(
      const Sm87MacroFeedV4RequestStateSealedAccess&) = default;

  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::uint64_t allocation_identity() const noexcept {
    return allocation_identity_;
  }
  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }

 private:
  Sm87MacroFeedV4RequestStateSealedAccess(
      const Sm87MacroFeedV4RequestState* owner,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uint64_t request_epoch) noexcept;

  const Sm87MacroFeedV4RequestState* owner_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;

  friend class Sm87MacroFeedV4RequestState;
};

// A runtime cold rearm never mutates a previously issued capability into
// authority for another request.  The Events owner receives a newly minted
// access only after its exact recurrent zero has been accepted and the
// RequestState has atomically advanced to panel 0 of a fresh monotonic epoch.
// Keeping the access in an optional also makes every failure result
// capability-empty.
struct Sm87MacroFeedV4RequestStateRearmResult final {
  Sm87MacroFeedV4RequestStateStatus status{};
  std::optional<Sm87MacroFeedV4RequestStateSealedAccess> access{};
  std::uint64_t previous_request_epoch = 0U;
  std::uint64_t request_epoch = 0U;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && access.has_value() &&
           previous_request_epoch != 0U &&
           request_epoch > previous_request_epoch &&
           access->request_epoch() == request_epoch;
  }
};

// Move-only, owner-minted authority for exactly one natural-order GDN layer.
// It identifies immutable slices within the admitted recurrent allocation;
// it deliberately exposes no host/device pointer, CUDA handle, or bank view.
// Moving a grant invalidates the source, and a successful commit consumes the
// destination.  The RequestState additionally authenticates the live nonce,
// owner, request/state epoch, bank generation, layer, and canonical offsets.
class Sm87MacroFeedV4GdnLayerStateGrant final {
 public:
  Sm87MacroFeedV4GdnLayerStateGrant() = delete;
  Sm87MacroFeedV4GdnLayerStateGrant(
      const Sm87MacroFeedV4GdnLayerStateGrant&) = delete;
  Sm87MacroFeedV4GdnLayerStateGrant& operator=(
      const Sm87MacroFeedV4GdnLayerStateGrant&) = delete;
  Sm87MacroFeedV4GdnLayerStateGrant(
      Sm87MacroFeedV4GdnLayerStateGrant&& other) noexcept;
  Sm87MacroFeedV4GdnLayerStateGrant& operator=(
      Sm87MacroFeedV4GdnLayerStateGrant&&) = delete;

  [[nodiscard]] std::uint64_t grant_identity() const noexcept {
    return grant_identity_;
  }
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::uint64_t allocation_identity() const noexcept {
    return allocation_identity_;
  }
  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }
  [[nodiscard]] std::uint64_t state_epoch() const noexcept {
    return state_epoch_;
  }
  [[nodiscard]] std::size_t panel() const noexcept { return panel_; }
  [[nodiscard]] std::size_t model_layer() const noexcept {
    return model_layer_;
  }
  [[nodiscard]] std::size_t state_layer_ordinal() const noexcept {
    return state_layer_ordinal_;
  }
  [[nodiscard]] std::size_t active_bank_index() const noexcept {
    return active_bank_index_;
  }
  [[nodiscard]] std::size_t candidate_bank_index() const noexcept {
    return candidate_bank_index_;
  }
  [[nodiscard]] std::uint64_t active_conv_allocation_offset() const noexcept {
    return active_conv_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t candidate_conv_allocation_offset()
      const noexcept {
    return candidate_conv_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t conv_bytes() const noexcept {
    return conv_bytes_;
  }
  [[nodiscard]] std::uint64_t active_gdn_state_allocation_offset()
      const noexcept {
    return active_gdn_state_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t candidate_gdn_state_allocation_offset()
      const noexcept {
    return candidate_gdn_state_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t gdn_state_bytes() const noexcept {
    return gdn_state_bytes_;
  }

 private:
  Sm87MacroFeedV4GdnLayerStateGrant(
      std::uint64_t grant_identity, std::uint64_t owner_identity,
      std::uint64_t allocation_identity, std::uint64_t request_epoch,
      std::uint64_t state_epoch, std::size_t panel,
      std::size_t model_layer, std::size_t state_layer_ordinal,
      std::size_t active_bank_index, std::size_t candidate_bank_index,
      std::uint64_t active_conv_allocation_offset,
      std::uint64_t candidate_conv_allocation_offset,
      std::uint64_t conv_bytes,
      std::uint64_t active_gdn_state_allocation_offset,
      std::uint64_t candidate_gdn_state_allocation_offset,
      std::uint64_t gdn_state_bytes) noexcept;

  void invalidate() noexcept;

  std::uint64_t grant_identity_ = 0U;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::uint64_t state_epoch_ = 0U;
  std::size_t panel_ = kSm87MacroFeedV4PanelCount;
  std::size_t model_layer_ = kSm87MacroFeedV4LayerCount;
  std::size_t state_layer_ordinal_ = kSm87MacroFeedV4StateLayerCount;
  std::size_t active_bank_index_ = 2U;
  std::size_t candidate_bank_index_ = 2U;
  std::uint64_t active_conv_allocation_offset_ = 0U;
  std::uint64_t candidate_conv_allocation_offset_ = 0U;
  std::uint64_t conv_bytes_ = 0U;
  std::uint64_t active_gdn_state_allocation_offset_ = 0U;
  std::uint64_t candidate_gdn_state_allocation_offset_ = 0U;
  std::uint64_t gdn_state_bytes_ = 0U;

  friend class Sm87MacroFeedV4RequestState;
};

struct Sm87MacroFeedV4GdnLayerStateAuthorizationResult final {
  Sm87MacroFeedV4RequestStateStatus status{};
  std::optional<Sm87MacroFeedV4GdnLayerStateGrant> grant{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && grant.has_value() &&
           grant->grant_identity() != 0U;
  }
};

// Move-only, owner-minted authority for the exact K/V write span of one
// natural-order Full-Attention layer in one C8000 panel.  Full allocation
// origins are retained because Attention reads the complete causal prefix;
// panel offsets identify the only rows that Full-QKV and preprocess may write
// for this grant.  It contains identities and byte offsets only.
class Sm87MacroFeedV4FullAttentionKvGrant final {
 public:
  Sm87MacroFeedV4FullAttentionKvGrant() = delete;
  Sm87MacroFeedV4FullAttentionKvGrant(
      const Sm87MacroFeedV4FullAttentionKvGrant&) = delete;
  Sm87MacroFeedV4FullAttentionKvGrant& operator=(
      const Sm87MacroFeedV4FullAttentionKvGrant&) = delete;
  Sm87MacroFeedV4FullAttentionKvGrant(
      Sm87MacroFeedV4FullAttentionKvGrant&& other) noexcept;
  Sm87MacroFeedV4FullAttentionKvGrant& operator=(
      Sm87MacroFeedV4FullAttentionKvGrant&&) = delete;

  [[nodiscard]] std::uint64_t grant_identity() const noexcept {
    return grant_identity_;
  }
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::uint64_t request_epoch() const noexcept {
    return request_epoch_;
  }
  [[nodiscard]] std::uint64_t state_epoch() const noexcept {
    return state_epoch_;
  }
  [[nodiscard]] std::uint64_t kv_allocation_identity() const noexcept {
    return kv_allocation_identity_;
  }
  [[nodiscard]] std::size_t panel() const noexcept { return panel_; }
  [[nodiscard]] std::size_t attention_layer_ordinal() const noexcept {
    return attention_layer_ordinal_;
  }
  [[nodiscard]] std::size_t model_layer() const noexcept {
    return model_layer_;
  }
  [[nodiscard]] std::uint64_t key_full_allocation_origin() const noexcept {
    return key_full_allocation_origin_;
  }
  [[nodiscard]] std::uint64_t value_full_allocation_origin() const noexcept {
    return value_full_allocation_origin_;
  }
  [[nodiscard]] std::uint64_t key_panel_allocation_offset() const noexcept {
    return key_panel_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t value_panel_allocation_offset() const noexcept {
    return value_panel_allocation_offset_;
  }
  [[nodiscard]] std::uint64_t panel_bytes() const noexcept {
    return panel_bytes_;
  }
  [[nodiscard]] std::size_t first_position() const noexcept {
    return first_position_;
  }
  [[nodiscard]] std::size_t previous_valid_end() const noexcept {
    return previous_valid_end_;
  }
  [[nodiscard]] std::size_t candidate_end() const noexcept {
    return candidate_end_;
  }

 private:
  Sm87MacroFeedV4FullAttentionKvGrant(
      std::uint64_t grant_identity, std::uint64_t owner_identity,
      std::uint64_t request_epoch, std::uint64_t state_epoch,
      std::uint64_t kv_allocation_identity, std::size_t panel,
      std::size_t attention_layer_ordinal, std::size_t model_layer,
      std::uint64_t key_full_allocation_origin,
      std::uint64_t value_full_allocation_origin,
      std::uint64_t key_panel_allocation_offset,
      std::uint64_t value_panel_allocation_offset, std::uint64_t panel_bytes,
      std::size_t first_position, std::size_t previous_valid_end,
      std::size_t candidate_end) noexcept;

  void invalidate() noexcept;

  std::uint64_t grant_identity_ = 0U;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t request_epoch_ = 0U;
  std::uint64_t state_epoch_ = 0U;
  std::uint64_t kv_allocation_identity_ = 0U;
  std::size_t panel_ = kSm87MacroFeedV4PanelCount;
  std::size_t attention_layer_ordinal_ =
      kSm87MacroFeedV4FullAttentionLayerCount;
  std::size_t model_layer_ = kSm87MacroFeedV4LayerCount;
  std::uint64_t key_full_allocation_origin_ = 0U;
  std::uint64_t value_full_allocation_origin_ = 0U;
  std::uint64_t key_panel_allocation_offset_ = 0U;
  std::uint64_t value_panel_allocation_offset_ = 0U;
  std::uint64_t panel_bytes_ = 0U;
  std::size_t first_position_ = kSm87MacroFeedV4P40Tokens;
  std::size_t previous_valid_end_ = kSm87MacroFeedV4P40Tokens;
  std::size_t candidate_end_ = kSm87MacroFeedV4P40Tokens;

  friend class Sm87MacroFeedV4RequestState;
};

struct Sm87MacroFeedV4FullAttentionKvAuthorizationResult final {
  Sm87MacroFeedV4RequestStateStatus status{};
  std::optional<Sm87MacroFeedV4FullAttentionKvGrant> grant{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && grant.has_value() &&
           grant->grant_identity() != 0U;
  }
};

enum class Sm87MacroFeedV4RequestEventKind : std::uint8_t {
  kInvalid = 0U,
  kPanelCommit,
  kOwnerDrain,
  kFinalCanonicalPublish,
};

// These receipts exist only to test request-state ordering.  Recording one
// does not observe a CUDA event or certify device completion; a future device
// issuer must remain a separate production authority.
struct Sm87MacroFeedV4RequestEventReceipt final {
  std::uint64_t receipt_identity = 0U;
  std::uint64_t event_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint64_t state_epoch = 0U;
  std::size_t panel = kSm87MacroFeedV4PanelCount;
  std::size_t completed_model_layer = kSm87MacroFeedV4LayerCount;
  std::size_t conv_layers = 0U;
  std::size_t gdn_layers = 0U;
  std::size_t kv_layers = 0U;
  std::uint64_t conv_copy_bytes = 0U;
  std::uint64_t gdn_assignment_bytes = 0U;
  std::size_t private_kv_valid_end = 0U;
  std::uint64_t source_bank_identity = 0U;
  std::uint64_t target_bank_identity = 0U;
  std::uint64_t copy_bytes = 0U;
  Sm87MacroFeedV4RequestEventKind kind =
      Sm87MacroFeedV4RequestEventKind::kInvalid;
  bool test_only_host_ledger_completed = false;
  bool physical_device_completion_attested = true;
  bool production_receipt_eligible = true;
};

struct Sm87MacroFeedV4RequestEventResult final {
  Sm87MacroFeedV4RequestStateStatus status{};
  Sm87MacroFeedV4RequestEventReceipt receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && receipt.receipt_identity != 0U &&
           receipt.test_only_host_ledger_completed &&
           !receipt.physical_device_completion_attested &&
           !receipt.production_receipt_eligible;
  }
};

struct Sm87MacroFeedV4RequestStateCreateResult final {
  std::unique_ptr<Sm87MacroFeedV4RequestState> state;
  Sm87MacroFeedV4RequestStateStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept;
};

class Sm87MacroFeedV4RequestState final {
 public:
  Sm87MacroFeedV4RequestState() = delete;
  ~Sm87MacroFeedV4RequestState() = default;
  Sm87MacroFeedV4RequestState(const Sm87MacroFeedV4RequestState&) = delete;
  Sm87MacroFeedV4RequestState& operator=(
      const Sm87MacroFeedV4RequestState&) = delete;
  Sm87MacroFeedV4RequestState(Sm87MacroFeedV4RequestState&&) = delete;
  Sm87MacroFeedV4RequestState& operator=(
      Sm87MacroFeedV4RequestState&&) = delete;

  [[nodiscard]] static Sm87MacroFeedV4RequestStateCreateResult create(
      const Sm87MacroFeedV4RequestStateAdmission& admission) noexcept;

  [[nodiscard]] const Sm87MacroFeedV4RequestStateAdmission& admission()
      const noexcept {
    return admission_;
  }
  [[nodiscard]] Sm87MacroFeedV4RequestStateSnapshot snapshot() const noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateSealedAccess issue_sealed_access()
      const noexcept;

  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus begin_panel(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::size_t panel) noexcept;
  [[nodiscard]] Sm87MacroFeedV4GdnLayerStateAuthorizationResult
  authorize_gdn_layer_state(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::size_t panel, std::size_t model_layer) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  commit_gdn_layer_candidate_enqueued(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      Sm87MacroFeedV4GdnLayerStateGrant&& grant) noexcept;
  [[nodiscard]] Sm87MacroFeedV4FullAttentionKvAuthorizationResult
  authorize_full_attention_kv(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::size_t panel, std::size_t model_layer) noexcept;
  // Consume only after the complete Full-QKV -> preprocess -> Attention -> O
  // -> residual/post-norm -> MLP layer DAG has been enqueued.  Advancing the
  // natural layer cursor on a K/V producer alone would permit unsafe scratch
  // reuse and is outside this authority.
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  commit_full_attention_layer_enqueued(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      Sm87MacroFeedV4FullAttentionKvGrant&& grant) noexcept;

  // Legacy host-ledger split retained only for old T0 fixtures.  A package
  // must use the atomic authorize/whole-layer-enqueue/commit grant path above.
  [[deprecated("T0 compatibility only; use authorize_gdn_layer_state")]]
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  prepare_conv_layer_candidate(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::size_t panel, std::size_t model_layer) noexcept;
  [[deprecated("T0 compatibility only; use commit_gdn_layer_candidate_enqueued")]]
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  assign_gdn_layer_candidate(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::size_t panel, std::size_t model_layer) noexcept;
  // Retained as an ABI-negative seam only.  It always fails after sealed
  // access validation and cannot advance K/V or layer state.
  [[deprecated("disabled; use authorize_full_attention_kv")]]
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  stage_attention_kv_layer(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::size_t panel, std::size_t model_layer,
      std::size_t candidate_valid_end) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestEventResult
  record_test_only_panel_completion(
      const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus commit_panel(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      const Sm87MacroFeedV4RequestEventReceipt& receipt) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestEventResult
  record_test_only_owner_drain_completion(
      const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus discard_active_panel(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      const Sm87MacroFeedV4RequestEventReceipt& drain_receipt,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus abort_unpublished_request(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus begin_final_canonical_copy(
      const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestEventResult
  record_test_only_final_copy_completion(
      const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  discard_unpublished_final_copy(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      const Sm87MacroFeedV4RequestEventReceipt& quiescence_receipt,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  publish_canonical_state(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      const Sm87MacroFeedV4RequestEventReceipt& receipt) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  publish_sequence_length_fence(
      const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept;

 private:
  explicit Sm87MacroFeedV4RequestState(
      const Sm87MacroFeedV4RequestStateAdmission& admission,
      std::uint64_t request_epoch) noexcept;

  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus validate_access(
      const Sm87MacroFeedV4RequestStateSealedAccess& access) const noexcept;
  // Execution admission is deliberately owner-mediated.  The execution-event
  // owner must present both the live RequestState object and its sealed access;
  // validation occurs while this RequestState owns its mutex so a copied
  // access cannot be detached from the object or admitted after the request
  // has left the private boundary.
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  validate_execution_begin_access(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::uint64_t expected_engine_owner_identity,
      std::uint64_t* allocation_identity,
      std::uint64_t* request_epoch) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestEventReceipt mint_event_receipt(
      Sm87MacroFeedV4RequestEventKind kind) noexcept;
  [[nodiscard]] bool event_receipt_matches(
      const Sm87MacroFeedV4RequestEventReceipt& receipt,
      Sm87MacroFeedV4RequestEventKind kind) const noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  discard_active_panel_after_physical_execution_drain(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::uint64_t execution_owner_identity,
      std::uint64_t allocation_identity, std::uint64_t request_epoch,
      std::size_t panel, std::uint64_t panel_generation,
      std::uint64_t physical_receipt_identity, bool poison_terminal,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;
  // Production request reuse is owner-atomic.  The execution owner calls this
  // only while holding its mutex, after a quiescent owner boundary and after
  // enqueueing the exact full recurrent-allocation zero on its Main stream.
  // The initial construction epoch is consumable exactly once; later rearms
  // require a healthy physically attested terminal RequestState.  The fresh
  // request and panel 0 are created in the same RequestState lock acquisition,
  // so no admitted-but-undrainable request can escape between those phases.
  // Physical KV bytes are intentionally not cleared here--only their private
  // logical valid-end is reset in the new epoch.
  [[nodiscard]] Sm87MacroFeedV4RequestStateRearmResult
  rearm_cold_and_begin_panel_zero_after_owner_quiescence(
      std::uint64_t execution_owner_identity,
      std::uint64_t recurrent_allocation_identity,
      std::uint64_t previous_request_epoch,
      std::uint64_t enqueued_recurrent_zero_bytes) noexcept;
  // PanelDone remains device ordered: Events validates its live 48-GDN /
  // 16-Full physical ledger and calls this method without a host wait.  No
  // test receipt or caller-filled completion structure authorizes this path.
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  commit_panel_after_device_ordered_execution(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::uint64_t execution_owner_identity,
      std::uint64_t allocation_identity, std::uint64_t request_epoch,
      std::size_t panel, std::uint64_t panel_generation) noexcept;
  // Final failure/cancellation is likewise owner atomic.  Events must first
  // physically drain the exact final generation; RequestState then
  // terminalizes the private AllPanels/FinalPublicationArmed state without
  // publishing canonical state, sequence length, or Decode access.
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus
  discard_final_after_physical_execution_drain(
      const Sm87MacroFeedV4RequestStateSealedAccess& access,
      std::uint64_t execution_owner_identity,
      std::uint64_t allocation_identity, std::uint64_t request_epoch,
      std::size_t final_panel, std::uint64_t final_panel_generation,
      std::uint64_t physical_receipt_identity, bool poison_terminal,
      Sm87MacroFeedV4RequestDiscardReason reason) noexcept;
  [[nodiscard]] Sm87MacroFeedV4RequestStateStatus commit_panel_locked(
      std::size_t panel, std::uint64_t panel_generation) noexcept;

  Sm87MacroFeedV4RequestStateAdmission admission_{};
  mutable std::mutex owner_mutex_;
  Sm87MacroFeedV4RequestStateSnapshot state_{};

  friend class sm87_macrofeed_v4_execution_events_detail::
      Sm87MacroFeedV4ExecutionEventsOwner;
};

}  // namespace q3x::runtime
