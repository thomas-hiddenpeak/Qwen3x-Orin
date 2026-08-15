#include "sm87_macrofeed_v4_request_state_internal.h"

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

namespace q3x::runtime {
namespace {

inline constexpr std::uint64_t kCanonicalStatePublicationIdentity =
    0x5133'4d46'5634'5401ULL;
inline constexpr std::uint64_t kSequenceLengthFenceIdentity =
    0x5133'4d46'5634'5402ULL;
inline constexpr std::uint64_t kOwnerDrainEventIdentity =
    0x5133'4d46'5634'5403ULL;

std::atomic<std::uint64_t> g_next_request_epoch{1U};
std::atomic<std::uint64_t> g_next_event_receipt_identity{1U};
std::atomic<std::uint64_t> g_next_gdn_layer_grant_identity{1U};
std::atomic<std::uint64_t> g_next_full_attention_kv_grant_identity{1U};

[[nodiscard]] std::uint64_t next_nonzero(
    std::atomic<std::uint64_t>* const source) noexcept {
  std::uint64_t value = source->fetch_add(1U, std::memory_order_relaxed);
  if (value == 0U) {
    value = source->fetch_add(1U, std::memory_order_relaxed);
  }
  return value;
}

[[nodiscard]] constexpr bool magic_equal(
    const std::array<std::uint8_t, 8U>& left,
    const std::array<std::uint8_t, 8U>& right) noexcept {
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (left[index] != right[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool gdn_layer(const std::size_t layer) noexcept {
  return sm87_macrofeed_v4_expected_layer_kind(layer) ==
         Sm87MacroFeedV4LayerKind::kGdn;
}

[[nodiscard]] constexpr bool full_attention_layer(
    const std::size_t layer) noexcept {
  return sm87_macrofeed_v4_expected_layer_kind(layer) ==
         Sm87MacroFeedV4LayerKind::kFullAttention;
}

[[nodiscard]] constexpr std::size_t model_layer_for_attention_ordinal(
    const std::size_t ordinal) noexcept {
  return ordinal < kSm87MacroFeedV4FullAttentionLayerCount
             ? 4U * ordinal + 3U
             : kSm87MacroFeedV4LayerCount;
}

[[nodiscard]] constexpr std::size_t attention_ordinal_for_model_layer(
    const std::size_t model_layer) noexcept {
  return full_attention_layer(model_layer)
             ? (model_layer - 3U) / 4U
             : kSm87MacroFeedV4FullAttentionLayerCount;
}

[[nodiscard]] constexpr std::uint64_t mix_identity(
    std::uint64_t value) noexcept {
  value ^= value >> 30U;
  value *= 0xbf58'476d'1ce4'e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d0'49bb'1331'11ebULL;
  value ^= value >> 31U;
  return value;
}

[[nodiscard]] constexpr std::uint64_t derive_host_only_kv_identity(
    const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::uint64_t bank_a_storage_identity,
    const std::uint64_t bank_b_storage_identity) noexcept {
  std::uint64_t value = mix_identity(owner_identity ^ 0x6b76'2d61'7265'6e61ULL);
  value = mix_identity(value ^ allocation_identity);
  value = mix_identity(value ^ bank_a_storage_identity);
  value = mix_identity(value ^ bank_b_storage_identity);
  return value == 0U ? 0x5133'4d46'5634'4b56ULL : value;
}

[[nodiscard]] constexpr std::size_t model_layer_for_state_ordinal(
    const std::size_t ordinal) noexcept {
  return ordinal < kSm87MacroFeedV4StateLayerCount
             ? ordinal + ordinal / 3U
             : kSm87MacroFeedV4LayerCount;
}

[[nodiscard]] constexpr std::size_t state_ordinal_for_model_layer(
    const std::size_t model_layer) noexcept {
  return gdn_layer(model_layer)
             ? model_layer - model_layer / 4U
             : kSm87MacroFeedV4StateLayerCount;
}

[[nodiscard]] constexpr bool state_layer_mapping_is_complete() noexcept {
  for (std::size_t ordinal = 0U;
       ordinal < kSm87MacroFeedV4StateLayerCount; ++ordinal) {
    const std::size_t model_layer = model_layer_for_state_ordinal(ordinal);
    if (model_layer != ordinal + ordinal / 3U || !gdn_layer(model_layer) ||
        state_ordinal_for_model_layer(model_layer) != ordinal) {
      return false;
    }
  }
  for (std::size_t ordinal = 0U;
       ordinal < kSm87MacroFeedV4FullAttentionLayerCount; ++ordinal) {
    const std::size_t model_layer =
        model_layer_for_attention_ordinal(ordinal);
    if (!full_attention_layer(model_layer) ||
        state_ordinal_for_model_layer(model_layer) !=
            kSm87MacroFeedV4StateLayerCount) {
      return false;
    }
  }
  return model_layer_for_state_ordinal(kSm87MacroFeedV4StateLayerCount) ==
             kSm87MacroFeedV4LayerCount &&
         state_ordinal_for_model_layer(kSm87MacroFeedV4LayerCount) ==
             kSm87MacroFeedV4StateLayerCount;
}

static_assert(state_layer_mapping_is_complete());

void add_issue(Sm87MacroFeedV4RequestAdmissionValidation* const validation,
               const Sm87MacroFeedV4RequestAdmissionIssue issue,
               const std::size_t state_layer =
                   kSm87MacroFeedV4StateLayerCount) noexcept {
  validation->issue_mask |= static_cast<std::uint32_t>(issue);
  if (validation->first_bad_state_layer ==
          kSm87MacroFeedV4StateLayerCount &&
      state_layer < kSm87MacroFeedV4StateLayerCount) {
    validation->first_bad_state_layer = state_layer;
  }
}

[[nodiscard]] constexpr Sm87MacroFeedV4RequestStateStatus ok() noexcept {
  return {};
}

[[nodiscard]] constexpr Sm87MacroFeedV4RequestStateStatus fail(
    const Sm87MacroFeedV4RequestStateError error, const char* const context,
    const std::size_t panel = kSm87MacroFeedV4PanelCount,
    const std::size_t layer = kSm87MacroFeedV4LayerCount) noexcept {
  return {error, context, panel, layer};
}

[[nodiscard]] constexpr bool valid_discard_reason(
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  return reason == Sm87MacroFeedV4RequestDiscardReason::kCancelled ||
         reason == Sm87MacroFeedV4RequestDiscardReason::kFailed;
}

[[nodiscard]] constexpr Sm87MacroFeedV4RequestStatePhase discard_phase(
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  return reason == Sm87MacroFeedV4RequestDiscardReason::kCancelled
             ? Sm87MacroFeedV4RequestStatePhase::kCancelled
             : Sm87MacroFeedV4RequestStatePhase::kFailed;
}

}  // namespace

Sm87MacroFeedV4RequestStateAdmission
make_sm87_macrofeed_v4_request_state_admission(
    const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::uint64_t bank_a_storage_identity,
    const std::uint64_t bank_b_storage_identity) noexcept {
  const auto plan = make_sm87_macrofeed_v4_p40_panel_wavefront_plan();
  std::uint64_t kv_allocation_identity = derive_host_only_kv_identity(
      owner_identity, allocation_identity, bank_a_storage_identity,
      bank_b_storage_identity);
  const std::array<std::uint64_t, 10U> forbidden{{
      owner_identity,
      allocation_identity,
      bank_a_storage_identity,
      bank_b_storage_identity,
      plan.state_ownership.private_kv_valid_end_storage_identity,
      plan.state_ownership.panel_commit_event_identity,
      plan.state_ownership.final_publish_event_identity,
      kOwnerDrainEventIdentity,
      kCanonicalStatePublicationIdentity,
      kSequenceLengthFenceIdentity,
  }};
  bool conflict = true;
  while (conflict) {
    conflict = kv_allocation_identity == 0U;
    for (const std::uint64_t identity : forbidden) {
      conflict |= kv_allocation_identity == identity;
    }
    if (conflict) {
      // The odd increment walks every uint64_t value, so at most the finite
      // forbidden set can delay a nonzero host-only identity.
      kv_allocation_identity += 0x9e37'79b9'7f4a'7c15ULL;
    }
  }
  auto admission = make_sm87_macrofeed_v4_request_state_admission(
      owner_identity, allocation_identity, bank_a_storage_identity,
      bank_b_storage_identity, kv_allocation_identity);
  admission.kv_physical_owner_bound = false;
  return admission;
}

Sm87MacroFeedV4RequestStateAdmission
make_sm87_macrofeed_v4_request_state_admission(
    const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::uint64_t bank_a_storage_identity,
    const std::uint64_t bank_b_storage_identity,
    const std::uint64_t kv_allocation_identity) noexcept {
  const auto plan = make_sm87_macrofeed_v4_p40_panel_wavefront_plan();
  Sm87MacroFeedV4RequestStateAdmission admission;
  admission.magic = kSm87MacroFeedV4RequestStateMagic;
  admission.abi_major = kSm87MacroFeedV4RequestStateAbiMajor;
  admission.abi_minor = kSm87MacroFeedV4RequestStateAbiMinor;
  admission.candidate_id = plan.candidate_id;
  admission.deployment_plan_id = plan.deployment_plan_id;
  admission.api_route_id = plan.api.route_id;
  admission.owner_identity = owner_identity;
  admission.allocation_identity = allocation_identity;
  admission.allocation_bytes = kSm87MacroFeedV4RecurrentStorageBytes;
  admission.recurrent_banks[0U] = {
      bank_a_storage_identity,
      owner_identity,
      allocation_identity,
      0U,
      kSm87MacroFeedV4RecurrentEpochBytes,
  };
  admission.recurrent_banks[1U] = {
      bank_b_storage_identity,
      owner_identity,
      allocation_identity,
      kSm87MacroFeedV4RecurrentEpochBytes,
      kSm87MacroFeedV4RecurrentEpochBytes,
  };
  for (std::size_t ordinal = 0U;
       ordinal < kSm87MacroFeedV4StateLayerCount; ++ordinal) {
    admission.recurrent_layers[ordinal] = {
        ordinal,
        model_layer_for_state_ordinal(ordinal),
        ordinal * kSm87MacroFeedV4ConvLayerBytes,
        kSm87MacroFeedV4ConvLayerBytes,
        kSm87MacroFeedV4ConvEpochBytes +
            ordinal * kSm87MacroFeedV4GdnStateLayerBytes,
        kSm87MacroFeedV4GdnStateLayerBytes,
    };
  }
  admission.kv_allocation_identity = kv_allocation_identity;
  admission.kv_allocation_bytes = kSm87MacroFeedV4AttentionKvArenaBytes;
  admission.kv_physical_owner_bound = true;
  for (std::size_t ordinal = 0U;
       ordinal < kSm87MacroFeedV4FullAttentionLayerCount; ++ordinal) {
    const std::uint64_t key_origin =
        ordinal * kSm87MacroFeedV4AttentionKvLayerBytes;
    admission.full_attention_kv_layers[ordinal] = {
        ordinal,
        model_layer_for_attention_ordinal(ordinal),
        key_origin,
        key_origin + kSm87MacroFeedV4AttentionKvPlaneBytes,
        kSm87MacroFeedV4AttentionKvPlaneBytes,
        kSm87MacroFeedV4AttentionKvPlaneBytes,
    };
  }
  admission.private_kv_valid_end_identity =
      plan.state_ownership.private_kv_valid_end_storage_identity;
  admission.panel_commit_event_identity =
      plan.state_ownership.panel_commit_event_identity;
  admission.final_publish_event_identity =
      plan.state_ownership.final_publish_event_identity;
  admission.owner_drain_event_identity = kOwnerDrainEventIdentity;
  admission.canonical_state_publication_identity =
      kCanonicalStatePublicationIdentity;
  admission.sequence_length_fence_identity =
      kSequenceLengthFenceIdentity;
  admission.conv_history_copies_active_to_candidate_per_layer = true;
  admission.gdn_first_update_reads_active_and_writes_candidate = true;
  admission.gdn_continuation_reads_and_writes_candidate = true;
  admission.candidate_epoch_fully_assigned_before_swap = true;
  admission.whole_epoch_copy_forbidden = true;
  admission.private_kv_valid_end = true;
  admission.canonical_state_publishes_after_final_panel = true;
  admission.sequence_length_is_final_nonfallible_fence = true;
  admission.host_only = true;
  admission.default_off = true;
  admission.test_only = true;
  admission.cuda_handles_present = false;
  admission.selector_bound = false;
  admission.launcher_present = false;
  admission.production_dispatch_eligible = false;
  return admission;
}

Sm87MacroFeedV4RequestAdmissionValidation
validate_sm87_macrofeed_v4_request_state_admission(
    const Sm87MacroFeedV4RequestStateAdmission& admission) noexcept {
  Sm87MacroFeedV4RequestAdmissionValidation validation;
  const auto expected = make_sm87_macrofeed_v4_request_state_admission(
      admission.owner_identity, admission.allocation_identity,
      admission.recurrent_banks[0U].storage_identity,
      admission.recurrent_banks[1U].storage_identity,
      admission.kv_allocation_identity);
  const auto expected_host_only =
      make_sm87_macrofeed_v4_request_state_admission(
          admission.owner_identity, admission.allocation_identity,
          admission.recurrent_banks[0U].storage_identity,
          admission.recurrent_banks[1U].storage_identity);
  if (!magic_equal(admission.magic, kSm87MacroFeedV4RequestStateMagic) ||
      admission.abi_major != kSm87MacroFeedV4RequestStateAbiMajor ||
      admission.abi_minor != kSm87MacroFeedV4RequestStateAbiMinor ||
      admission.candidate_id != kSm87MacroFeedV4CandidateId ||
      admission.deployment_plan_id !=
          kSm87MacroFeedV4P40DeploymentPlanId ||
      admission.api_route_id != kSm87MacroFeedV4P40ApiRouteId) {
    add_issue(&validation,
              Sm87MacroFeedV4RequestAdmissionIssue::kIdentity);
  }

  if (admission.owner_identity == 0U || admission.allocation_identity == 0U ||
      admission.owner_identity == admission.allocation_identity ||
      admission.allocation_bytes != kSm87MacroFeedV4RecurrentStorageBytes ||
      admission.owner_identity ==
          admission.recurrent_banks[0U].storage_identity ||
      admission.owner_identity ==
          admission.recurrent_banks[1U].storage_identity ||
      admission.allocation_identity ==
          admission.recurrent_banks[0U].storage_identity ||
      admission.allocation_identity ==
          admission.recurrent_banks[1U].storage_identity ||
      admission.recurrent_banks[0U].storage_identity == 0U ||
      admission.recurrent_banks[1U].storage_identity == 0U ||
      admission.recurrent_banks[0U].storage_identity ==
          admission.recurrent_banks[1U].storage_identity ||
      admission.recurrent_banks[0U].storage_identity !=
          expected.recurrent_banks[0U].storage_identity ||
      admission.recurrent_banks[1U].storage_identity !=
          expected.recurrent_banks[1U].storage_identity ||
      admission.recurrent_banks[0U].owner_identity !=
          admission.owner_identity ||
      admission.recurrent_banks[1U].owner_identity !=
          admission.owner_identity ||
      admission.recurrent_banks[0U].allocation_identity !=
          admission.allocation_identity ||
      admission.recurrent_banks[1U].allocation_identity !=
          admission.allocation_identity ||
      admission.recurrent_banks[0U].allocation_offset != 0U ||
      admission.recurrent_banks[1U].allocation_offset !=
          kSm87MacroFeedV4RecurrentEpochBytes ||
      admission.recurrent_banks[0U].bytes !=
          kSm87MacroFeedV4RecurrentEpochBytes ||
      admission.recurrent_banks[1U].bytes !=
          kSm87MacroFeedV4RecurrentEpochBytes ||
      admission.recurrent_banks[0U].allocation_offset +
              admission.recurrent_banks[0U].bytes >
          admission.allocation_bytes ||
      admission.recurrent_banks[1U].allocation_offset +
              admission.recurrent_banks[1U].bytes >
          admission.allocation_bytes ||
      admission.recurrent_banks[0U].allocation_offset +
              admission.recurrent_banks[0U].bytes >
          admission.recurrent_banks[1U].allocation_offset) {
    add_issue(&validation,
              Sm87MacroFeedV4RequestAdmissionIssue::kBankOwnership);
  }

  for (std::size_t ordinal = 0U;
       ordinal < kSm87MacroFeedV4StateLayerCount; ++ordinal) {
    const auto& slice = admission.recurrent_layers[ordinal];
    const auto& canonical = expected.recurrent_layers[ordinal];
    if (slice.state_layer_ordinal != ordinal ||
        slice.model_layer != canonical.model_layer ||
        slice.conv_offset != canonical.conv_offset ||
        slice.conv_bytes != kSm87MacroFeedV4ConvLayerBytes ||
        slice.gdn_state_offset != canonical.gdn_state_offset ||
        slice.gdn_state_bytes != kSm87MacroFeedV4GdnStateLayerBytes ||
        !gdn_layer(slice.model_layer) ||
        slice.conv_offset + slice.conv_bytes >
            kSm87MacroFeedV4ConvEpochBytes ||
        slice.gdn_state_offset + slice.gdn_state_bytes >
            kSm87MacroFeedV4RecurrentEpochBytes) {
      add_issue(&validation,
                Sm87MacroFeedV4RequestAdmissionIssue::kLayerLayout,
                ordinal);
    }
  }

  bool kv_arena_bad =
      admission.kv_allocation_identity == 0U ||
      admission.kv_allocation_identity == admission.owner_identity ||
      admission.kv_allocation_identity == admission.allocation_identity ||
      admission.kv_allocation_identity ==
          admission.recurrent_banks[0U].storage_identity ||
      admission.kv_allocation_identity ==
          admission.recurrent_banks[1U].storage_identity ||
      admission.kv_allocation_bytes != kSm87MacroFeedV4AttentionKvArenaBytes ||
      (!admission.kv_physical_owner_bound &&
       admission.kv_allocation_identity !=
           expected_host_only.kv_allocation_identity);
  for (std::size_t ordinal = 0U;
       ordinal < kSm87MacroFeedV4FullAttentionLayerCount; ++ordinal) {
    const auto& slice = admission.full_attention_kv_layers[ordinal];
    const auto& canonical = expected.full_attention_kv_layers[ordinal];
    kv_arena_bad |=
        slice.attention_layer_ordinal != ordinal ||
        slice.model_layer != canonical.model_layer ||
        !full_attention_layer(slice.model_layer) ||
        slice.key_full_allocation_origin !=
            canonical.key_full_allocation_origin ||
        slice.value_full_allocation_origin !=
            canonical.value_full_allocation_origin ||
        slice.key_bytes != kSm87MacroFeedV4AttentionKvPlaneBytes ||
        slice.value_bytes != kSm87MacroFeedV4AttentionKvPlaneBytes ||
        slice.key_full_allocation_origin + slice.key_bytes !=
            slice.value_full_allocation_origin ||
        slice.value_full_allocation_origin + slice.value_bytes >
            admission.kv_allocation_bytes;
    if (ordinal + 1U < kSm87MacroFeedV4FullAttentionLayerCount) {
      kv_arena_bad |=
          slice.value_full_allocation_origin + slice.value_bytes !=
          admission.full_attention_kv_layers[ordinal + 1U]
              .key_full_allocation_origin;
    } else {
      kv_arena_bad |=
          slice.value_full_allocation_origin + slice.value_bytes !=
          admission.kv_allocation_bytes;
    }
  }
  if (kv_arena_bad) {
    add_issue(&validation,
              Sm87MacroFeedV4RequestAdmissionIssue::kKvArenaOwnership);
  }

  if (!admission.conv_history_copies_active_to_candidate_per_layer ||
      !admission.gdn_first_update_reads_active_and_writes_candidate ||
      !admission.gdn_continuation_reads_and_writes_candidate ||
      !admission.candidate_epoch_fully_assigned_before_swap ||
      !admission.whole_epoch_copy_forbidden) {
    add_issue(&validation,
              Sm87MacroFeedV4RequestAdmissionIssue::kTransitionContract);
  }

  const std::array<std::uint64_t, 6U> visibility_identities{{
      admission.private_kv_valid_end_identity,
      admission.panel_commit_event_identity,
      admission.final_publish_event_identity,
      admission.owner_drain_event_identity,
      admission.canonical_state_publication_identity,
      admission.sequence_length_fence_identity,
  }};
  bool visibility_bad = !admission.private_kv_valid_end ||
                        !admission.canonical_state_publishes_after_final_panel ||
                        !admission.sequence_length_is_final_nonfallible_fence;
  for (std::size_t first = 0U; first < visibility_identities.size(); ++first) {
    visibility_bad |= visibility_identities[first] == 0U;
    visibility_bad |=
        visibility_identities[first] !=
        std::array<std::uint64_t, 6U>{{
            expected.private_kv_valid_end_identity,
            expected.panel_commit_event_identity,
            expected.final_publish_event_identity,
            expected.owner_drain_event_identity,
            expected.canonical_state_publication_identity,
            expected.sequence_length_fence_identity,
        }}[first];
    for (std::size_t second = first + 1U;
         second < visibility_identities.size(); ++second) {
      visibility_bad |=
          visibility_identities[first] == visibility_identities[second];
    }
    visibility_bad |=
        visibility_identities[first] ==
            admission.recurrent_banks[0U].storage_identity ||
        visibility_identities[first] ==
            admission.recurrent_banks[1U].storage_identity ||
        visibility_identities[first] == admission.kv_allocation_identity ||
        visibility_identities[first] == admission.owner_identity ||
        visibility_identities[first] == admission.allocation_identity;
  }
  if (visibility_bad) {
    add_issue(&validation,
              Sm87MacroFeedV4RequestAdmissionIssue::kVisibilityOwnership);
  }

  if (!admission.host_only || !admission.default_off || !admission.test_only ||
      admission.cuda_handles_present || admission.selector_bound ||
      admission.launcher_present || admission.production_dispatch_eligible) {
    add_issue(&validation,
              Sm87MacroFeedV4RequestAdmissionIssue::kDispatchBoundary);
  }
  return validation;
}

Sm87MacroFeedV4RequestState::Sm87MacroFeedV4RequestState(
    const Sm87MacroFeedV4RequestStateAdmission& admission,
    const std::uint64_t request_epoch) noexcept
    : admission_(admission) {
  state_.phase = Sm87MacroFeedV4RequestStatePhase::kAdmittedPrivate;
  state_.active_bank_index = 0U;
  state_.candidate_bank_index = 1U;
  state_.active_bank_identity = admission_.recurrent_banks[0U].storage_identity;
  state_.candidate_bank_identity =
      admission_.recurrent_banks[1U].storage_identity;
  state_.owner_identity = admission_.owner_identity;
  state_.allocation_identity = admission_.allocation_identity;
  state_.request_epoch = request_epoch;
  state_.default_off = admission_.default_off;
  state_.host_only = admission_.host_only;
  state_.production_dispatch_eligible =
      admission_.production_dispatch_eligible;
}

Sm87MacroFeedV4RequestStateCreateResult
Sm87MacroFeedV4RequestState::create(
    const Sm87MacroFeedV4RequestStateAdmission& admission) noexcept {
  Sm87MacroFeedV4RequestStateCreateResult result;
  if (!validate_sm87_macrofeed_v4_request_state_admission(admission).valid()) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kAdmissionInvalid,
                         "request_state_admission_invalid");
    return result;
  }
  const std::uint64_t request_epoch = next_nonzero(&g_next_request_epoch);
  result.state.reset(new (std::nothrow)
                         Sm87MacroFeedV4RequestState(admission,
                                                    request_epoch));
  if (result.state == nullptr) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kAllocationFailure,
                         "request_state_host_allocation_failed");
    return result;
  }
  result.status = ok();
  return result;
}

Sm87MacroFeedV4RequestStateCreateResult::operator bool() const noexcept {
  return state != nullptr && static_cast<bool>(status) &&
         state->snapshot().phase ==
             Sm87MacroFeedV4RequestStatePhase::kAdmittedPrivate;
}

Sm87MacroFeedV4RequestStateSnapshot
Sm87MacroFeedV4RequestState::snapshot() const noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  return state_;
}

Sm87MacroFeedV4RequestStateSealedAccess::
    Sm87MacroFeedV4RequestStateSealedAccess(
        const Sm87MacroFeedV4RequestState* const owner,
        const std::uint64_t owner_identity,
        const std::uint64_t allocation_identity,
        const std::uint64_t request_epoch) noexcept
    : owner_(owner),
      owner_identity_(owner_identity),
      allocation_identity_(allocation_identity),
      request_epoch_(request_epoch) {}

Sm87MacroFeedV4RequestStateSealedAccess
Sm87MacroFeedV4RequestState::issue_sealed_access() const noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  return Sm87MacroFeedV4RequestStateSealedAccess(
      this, state_.owner_identity, state_.allocation_identity,
      state_.request_epoch);
}

Sm87MacroFeedV4GdnLayerStateGrant::Sm87MacroFeedV4GdnLayerStateGrant(
    const std::uint64_t grant_identity,
    const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::uint64_t request_epoch, const std::uint64_t state_epoch,
    const std::size_t panel, const std::size_t model_layer,
    const std::size_t state_layer_ordinal,
    const std::size_t active_bank_index,
    const std::size_t candidate_bank_index,
    const std::uint64_t active_conv_allocation_offset,
    const std::uint64_t candidate_conv_allocation_offset,
    const std::uint64_t conv_bytes,
    const std::uint64_t active_gdn_state_allocation_offset,
    const std::uint64_t candidate_gdn_state_allocation_offset,
    const std::uint64_t gdn_state_bytes) noexcept
    : grant_identity_(grant_identity),
      owner_identity_(owner_identity),
      allocation_identity_(allocation_identity),
      request_epoch_(request_epoch),
      state_epoch_(state_epoch),
      panel_(panel),
      model_layer_(model_layer),
      state_layer_ordinal_(state_layer_ordinal),
      active_bank_index_(active_bank_index),
      candidate_bank_index_(candidate_bank_index),
      active_conv_allocation_offset_(active_conv_allocation_offset),
      candidate_conv_allocation_offset_(candidate_conv_allocation_offset),
      conv_bytes_(conv_bytes),
      active_gdn_state_allocation_offset_(
          active_gdn_state_allocation_offset),
      candidate_gdn_state_allocation_offset_(
          candidate_gdn_state_allocation_offset),
      gdn_state_bytes_(gdn_state_bytes) {}

Sm87MacroFeedV4GdnLayerStateGrant::Sm87MacroFeedV4GdnLayerStateGrant(
    Sm87MacroFeedV4GdnLayerStateGrant&& other) noexcept
    : grant_identity_(std::exchange(other.grant_identity_, 0U)),
      owner_identity_(std::exchange(other.owner_identity_, 0U)),
      allocation_identity_(std::exchange(other.allocation_identity_, 0U)),
      request_epoch_(std::exchange(other.request_epoch_, 0U)),
      state_epoch_(std::exchange(other.state_epoch_, 0U)),
      panel_(std::exchange(other.panel_, kSm87MacroFeedV4PanelCount)),
      model_layer_(
          std::exchange(other.model_layer_, kSm87MacroFeedV4LayerCount)),
      state_layer_ordinal_(std::exchange(
          other.state_layer_ordinal_, kSm87MacroFeedV4StateLayerCount)),
      active_bank_index_(std::exchange(other.active_bank_index_, 2U)),
      candidate_bank_index_(std::exchange(other.candidate_bank_index_, 2U)),
      active_conv_allocation_offset_(
          std::exchange(other.active_conv_allocation_offset_, 0U)),
      candidate_conv_allocation_offset_(
          std::exchange(other.candidate_conv_allocation_offset_, 0U)),
      conv_bytes_(std::exchange(other.conv_bytes_, 0U)),
      active_gdn_state_allocation_offset_(
          std::exchange(other.active_gdn_state_allocation_offset_, 0U)),
      candidate_gdn_state_allocation_offset_(
          std::exchange(other.candidate_gdn_state_allocation_offset_, 0U)),
      gdn_state_bytes_(std::exchange(other.gdn_state_bytes_, 0U)) {}

void Sm87MacroFeedV4GdnLayerStateGrant::invalidate() noexcept {
  grant_identity_ = 0U;
  owner_identity_ = 0U;
  allocation_identity_ = 0U;
  request_epoch_ = 0U;
  state_epoch_ = 0U;
  panel_ = kSm87MacroFeedV4PanelCount;
  model_layer_ = kSm87MacroFeedV4LayerCount;
  state_layer_ordinal_ = kSm87MacroFeedV4StateLayerCount;
  active_bank_index_ = 2U;
  candidate_bank_index_ = 2U;
  active_conv_allocation_offset_ = 0U;
  candidate_conv_allocation_offset_ = 0U;
  conv_bytes_ = 0U;
  active_gdn_state_allocation_offset_ = 0U;
  candidate_gdn_state_allocation_offset_ = 0U;
  gdn_state_bytes_ = 0U;
}

Sm87MacroFeedV4FullAttentionKvGrant::
    Sm87MacroFeedV4FullAttentionKvGrant(
        const std::uint64_t grant_identity,
        const std::uint64_t owner_identity,
        const std::uint64_t request_epoch, const std::uint64_t state_epoch,
        const std::uint64_t kv_allocation_identity, const std::size_t panel,
        const std::size_t attention_layer_ordinal,
        const std::size_t model_layer,
        const std::uint64_t key_full_allocation_origin,
        const std::uint64_t value_full_allocation_origin,
        const std::uint64_t key_panel_allocation_offset,
        const std::uint64_t value_panel_allocation_offset,
        const std::uint64_t panel_bytes, const std::size_t first_position,
        const std::size_t previous_valid_end,
        const std::size_t candidate_end) noexcept
    : grant_identity_(grant_identity),
      owner_identity_(owner_identity),
      request_epoch_(request_epoch),
      state_epoch_(state_epoch),
      kv_allocation_identity_(kv_allocation_identity),
      panel_(panel),
      attention_layer_ordinal_(attention_layer_ordinal),
      model_layer_(model_layer),
      key_full_allocation_origin_(key_full_allocation_origin),
      value_full_allocation_origin_(value_full_allocation_origin),
      key_panel_allocation_offset_(key_panel_allocation_offset),
      value_panel_allocation_offset_(value_panel_allocation_offset),
      panel_bytes_(panel_bytes),
      first_position_(first_position),
      previous_valid_end_(previous_valid_end),
      candidate_end_(candidate_end) {}

Sm87MacroFeedV4FullAttentionKvGrant::
    Sm87MacroFeedV4FullAttentionKvGrant(
        Sm87MacroFeedV4FullAttentionKvGrant&& other) noexcept
    : grant_identity_(std::exchange(other.grant_identity_, 0U)),
      owner_identity_(std::exchange(other.owner_identity_, 0U)),
      request_epoch_(std::exchange(other.request_epoch_, 0U)),
      state_epoch_(std::exchange(other.state_epoch_, 0U)),
      kv_allocation_identity_(
          std::exchange(other.kv_allocation_identity_, 0U)),
      panel_(std::exchange(other.panel_, kSm87MacroFeedV4PanelCount)),
      attention_layer_ordinal_(std::exchange(
          other.attention_layer_ordinal_,
          kSm87MacroFeedV4FullAttentionLayerCount)),
      model_layer_(
          std::exchange(other.model_layer_, kSm87MacroFeedV4LayerCount)),
      key_full_allocation_origin_(
          std::exchange(other.key_full_allocation_origin_, 0U)),
      value_full_allocation_origin_(
          std::exchange(other.value_full_allocation_origin_, 0U)),
      key_panel_allocation_offset_(
          std::exchange(other.key_panel_allocation_offset_, 0U)),
      value_panel_allocation_offset_(
          std::exchange(other.value_panel_allocation_offset_, 0U)),
      panel_bytes_(std::exchange(other.panel_bytes_, 0U)),
      first_position_(std::exchange(other.first_position_,
                                    kSm87MacroFeedV4P40Tokens)),
      previous_valid_end_(std::exchange(other.previous_valid_end_,
                                        kSm87MacroFeedV4P40Tokens)),
      candidate_end_(std::exchange(other.candidate_end_,
                                   kSm87MacroFeedV4P40Tokens)) {}

void Sm87MacroFeedV4FullAttentionKvGrant::invalidate() noexcept {
  grant_identity_ = 0U;
  owner_identity_ = 0U;
  request_epoch_ = 0U;
  state_epoch_ = 0U;
  kv_allocation_identity_ = 0U;
  panel_ = kSm87MacroFeedV4PanelCount;
  attention_layer_ordinal_ = kSm87MacroFeedV4FullAttentionLayerCount;
  model_layer_ = kSm87MacroFeedV4LayerCount;
  key_full_allocation_origin_ = 0U;
  value_full_allocation_origin_ = 0U;
  key_panel_allocation_offset_ = 0U;
  value_panel_allocation_offset_ = 0U;
  panel_bytes_ = 0U;
  first_position_ = kSm87MacroFeedV4P40Tokens;
  previous_valid_end_ = kSm87MacroFeedV4P40Tokens;
  candidate_end_ = kSm87MacroFeedV4P40Tokens;
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::validate_access(
    const Sm87MacroFeedV4RequestStateSealedAccess& access) const noexcept {
  if (access.owner_ != this || access.owner_identity_ == 0U ||
      access.allocation_identity_ == 0U || access.request_epoch_ == 0U ||
      access.owner_identity_ != state_.owner_identity ||
      access.allocation_identity_ != state_.allocation_identity ||
      access.request_epoch_ != state_.request_epoch) {
    return fail(Sm87MacroFeedV4RequestStateError::kCapabilityMismatch,
                "sealed_access_owner_allocation_or_epoch_mismatch");
  }
  return ok();
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::validate_execution_begin_access(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const std::uint64_t expected_engine_owner_identity,
    std::uint64_t* const allocation_identity,
    std::uint64_t* const request_epoch) const noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (expected_engine_owner_identity == 0U ||
      state_.owner_identity != expected_engine_owner_identity) {
    return fail(Sm87MacroFeedV4RequestStateError::kCapabilityMismatch,
                "execution_owner_identity_mismatch");
  }
  if (state_.phase !=
          Sm87MacroFeedV4RequestStatePhase::kAdmittedPrivate ||
      state_.request_epoch == 0U || state_.allocation_identity == 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "execution_begin_requires_admitted_private_request");
  }
  if (allocation_identity == nullptr || request_epoch == nullptr) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "execution_begin_outputs_required");
  }
  *allocation_identity = state_.allocation_identity;
  *request_epoch = state_.request_epoch;
  return ok();
}

Sm87MacroFeedV4RequestEventReceipt
Sm87MacroFeedV4RequestState::mint_event_receipt(
    const Sm87MacroFeedV4RequestEventKind kind) noexcept {
  Sm87MacroFeedV4RequestEventReceipt receipt;
  receipt.receipt_identity =
      next_nonzero(&g_next_event_receipt_identity);
  receipt.event_identity =
      kind == Sm87MacroFeedV4RequestEventKind::kPanelCommit
          ? admission_.panel_commit_event_identity
          : (kind == Sm87MacroFeedV4RequestEventKind::kOwnerDrain
                 ? admission_.owner_drain_event_identity
                 : admission_.final_publish_event_identity);
  receipt.owner_identity = state_.owner_identity;
  receipt.allocation_identity = state_.allocation_identity;
  receipt.request_epoch = state_.request_epoch;
  receipt.state_epoch = state_.state_epoch;
  receipt.panel = kind == Sm87MacroFeedV4RequestEventKind::kFinalCanonicalPublish
                      ? state_.completed_panels
                      : state_.active_panel;
  receipt.completed_model_layer =
      kind == Sm87MacroFeedV4RequestEventKind::kFinalCanonicalPublish
          ? kSm87MacroFeedV4LayerCount - 1U
          : (state_.next_model_layer == 0U
                 ? kSm87MacroFeedV4LayerCount
                 : state_.next_model_layer - 1U);
  receipt.conv_layers = state_.panel_conv_layers_prepared;
  receipt.gdn_layers = state_.panel_gdn_layers_assigned;
  receipt.kv_layers = state_.panel_kv_layers_staged;
  if (kind == Sm87MacroFeedV4RequestEventKind::kPanelCommit) {
    receipt.conv_copy_bytes = state_.panel_conv_copy_bytes;
    receipt.gdn_assignment_bytes = state_.panel_gdn_assignment_bytes;
    receipt.source_bank_identity = state_.active_bank_identity;
    receipt.target_bank_identity = state_.candidate_bank_identity;
  }
  receipt.private_kv_valid_end =
      kind == Sm87MacroFeedV4RequestEventKind::kPanelCommit
          ? state_.candidate_kv_valid_end
          : state_.private_kv_valid_end;
  if (kind == Sm87MacroFeedV4RequestEventKind::kFinalCanonicalPublish) {
    receipt.source_bank_identity = state_.active_bank_identity;
    receipt.target_bank_identity = state_.candidate_bank_identity;
    receipt.copy_bytes = kSm87MacroFeedV4RecurrentEpochBytes;
  }
  receipt.kind = kind;
  receipt.test_only_host_ledger_completed = true;
  receipt.physical_device_completion_attested = false;
  receipt.production_receipt_eligible = false;
  state_.pending_event_receipt_identity = receipt.receipt_identity;
  return receipt;
}

bool Sm87MacroFeedV4RequestState::event_receipt_matches(
    const Sm87MacroFeedV4RequestEventReceipt& receipt,
    const Sm87MacroFeedV4RequestEventKind kind) const noexcept {
  const std::uint64_t expected_event_identity =
      kind == Sm87MacroFeedV4RequestEventKind::kPanelCommit
          ? admission_.panel_commit_event_identity
          : (kind == Sm87MacroFeedV4RequestEventKind::kOwnerDrain
                 ? admission_.owner_drain_event_identity
                 : admission_.final_publish_event_identity);
  const std::size_t expected_panel =
      kind == Sm87MacroFeedV4RequestEventKind::kFinalCanonicalPublish
          ? state_.completed_panels
          : state_.active_panel;
  return receipt.receipt_identity != 0U &&
         receipt.receipt_identity == state_.pending_event_receipt_identity &&
         receipt.event_identity == expected_event_identity &&
         receipt.owner_identity == state_.owner_identity &&
         receipt.allocation_identity == state_.allocation_identity &&
         receipt.request_epoch == state_.request_epoch &&
         receipt.state_epoch == state_.state_epoch &&
         receipt.panel == expected_panel && receipt.kind == kind &&
         receipt.test_only_host_ledger_completed &&
         !receipt.physical_device_completion_attested &&
         !receipt.production_receipt_eligible;
}

Sm87MacroFeedV4RequestStateStatus Sm87MacroFeedV4RequestState::begin_panel(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const std::size_t panel) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (state_.pending_event_receipt_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "begin_panel_blocked_by_pending_owner_event", panel);
  }
  const bool phase_ok =
      (panel == 0U && state_.phase ==
                          Sm87MacroFeedV4RequestStatePhase::kAdmittedPrivate) ||
      (panel > 0U &&
       state_.phase ==
           Sm87MacroFeedV4RequestStatePhase::kBetweenPanelsPrivate);
  if (!phase_ok) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "begin_panel_requires_private_boundary", panel);
  }
  if (panel >= kSm87MacroFeedV4PanelCount ||
      panel != state_.completed_panels) {
    return fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                "begin_panel_out_of_order", panel);
  }
  if (state_.private_kv_valid_end != panel * kSm87MacroFeedV4PanelTokens ||
      state_.active_bank_identity == 0U ||
      state_.candidate_bank_identity == 0U ||
      state_.active_bank_identity == state_.candidate_bank_identity ||
      state_.pending_gdn_layer_grant_identity != 0U ||
      state_.pending_full_attention_kv_grant_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "begin_panel_private_owner_invalid", panel);
  }
  state_.phase = Sm87MacroFeedV4RequestStatePhase::kPanelActive;
  state_.active_panel = panel;
  state_.next_model_layer = 0U;
  state_.panel_conv_layers_prepared = 0U;
  state_.panel_gdn_layers_assigned = 0U;
  state_.panel_kv_layers_staged = 0U;
  state_.panel_conv_copy_bytes = 0U;
  state_.panel_gdn_assignment_bytes = 0U;
  state_.candidate_kv_valid_end = state_.private_kv_valid_end;
  state_.current_conv_layer_prepared = false;
  state_.candidate_epoch_complete = false;
  return ok();
}

Sm87MacroFeedV4GdnLayerStateAuthorizationResult
Sm87MacroFeedV4RequestState::authorize_gdn_layer_state(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const std::size_t panel, const std::size_t model_layer) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  Sm87MacroFeedV4GdnLayerStateAuthorizationResult result;
  result.status = validate_access(access);
  if (!result.status) {
    return result;
  }
  if (state_.pending_event_receipt_identity != 0U) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
        "gdn_grant_blocked_by_pending_owner_event", panel, model_layer);
    return result;
  }
  if (state_.pending_gdn_layer_grant_identity != 0U ||
      state_.current_conv_layer_prepared) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kGdnLayerGrantPending,
        "gdn_layer_grant_already_pending", panel, model_layer);
    return result;
  }
  if (state_.pending_full_attention_kv_grant_identity != 0U) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kFullAttentionKvGrantPending,
        "gdn_layer_grant_blocked_by_full_attention_kv_grant", panel,
        model_layer);
    return result;
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelActive) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kInvalidTransition,
        "gdn_grant_requires_active_panel", panel, model_layer);
    return result;
  }
  if (panel != state_.active_panel) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                         "gdn_grant_panel_mismatch", panel, model_layer);
    return result;
  }
  if (model_layer != state_.next_model_layer) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kLayerMismatch,
                         "gdn_grant_layer_out_of_order", panel,
                         model_layer);
    return result;
  }
  if (!gdn_layer(model_layer)) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kLayerKindMismatch,
        "gdn_grant_requires_gdn_layer", panel, model_layer);
    return result;
  }
  if (state_.active_bank_index >= admission_.recurrent_banks.size() ||
      state_.candidate_bank_index >= admission_.recurrent_banks.size() ||
      state_.active_bank_index == state_.candidate_bank_index) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kInvalidTransition,
        "gdn_grant_bank_generation_invalid", panel, model_layer);
    return result;
  }
  const std::size_t ordinal = state_ordinal_for_model_layer(model_layer);
  if (ordinal >= admission_.recurrent_layers.size() ||
      admission_.recurrent_layers[ordinal].model_layer != model_layer) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kLayerMismatch,
                         "gdn_grant_state_ordinal_invalid", panel,
                         model_layer);
    return result;
  }

  const auto& slice = admission_.recurrent_layers[ordinal];
  const auto& active_bank =
      admission_.recurrent_banks[state_.active_bank_index];
  const auto& candidate_bank =
      admission_.recurrent_banks[state_.candidate_bank_index];
  const std::uint64_t grant_identity =
      next_nonzero(&g_next_gdn_layer_grant_identity);
  result.grant.emplace(Sm87MacroFeedV4GdnLayerStateGrant(
      grant_identity, state_.owner_identity, state_.allocation_identity,
      state_.request_epoch, state_.state_epoch, panel, model_layer, ordinal,
      state_.active_bank_index, state_.candidate_bank_index,
      active_bank.allocation_offset + slice.conv_offset,
      candidate_bank.allocation_offset + slice.conv_offset, slice.conv_bytes,
      active_bank.allocation_offset + slice.gdn_state_offset,
      candidate_bank.allocation_offset + slice.gdn_state_offset,
      slice.gdn_state_bytes));
  state_.pending_gdn_layer_grant_identity = grant_identity;
  result.status = ok();
  return result;
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::commit_gdn_layer_candidate_enqueued(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    Sm87MacroFeedV4GdnLayerStateGrant&& grant) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  const std::size_t panel = grant.panel_;
  const std::size_t model_layer = grant.model_layer_;
  if (state_.pending_event_receipt_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "gdn_commit_blocked_by_pending_owner_event", panel,
                model_layer);
  }
  if (state_.pending_full_attention_kv_grant_identity != 0U) {
    return fail(
        Sm87MacroFeedV4RequestStateError::kFullAttentionKvGrantPending,
        "gdn_commit_blocked_by_full_attention_kv_grant", panel,
        model_layer);
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelActive) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "gdn_commit_requires_active_panel", panel, model_layer);
  }
  if (panel != state_.active_panel) {
    return fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                "gdn_commit_panel_mismatch", panel, model_layer);
  }
  if (model_layer != state_.next_model_layer) {
    return fail(Sm87MacroFeedV4RequestStateError::kLayerMismatch,
                "gdn_commit_layer_out_of_order", panel, model_layer);
  }
  if (!gdn_layer(model_layer)) {
    return fail(Sm87MacroFeedV4RequestStateError::kLayerKindMismatch,
                "gdn_commit_requires_gdn_layer", panel, model_layer);
  }
  if (state_.pending_gdn_layer_grant_identity == 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kGdnLayerGrantMismatch,
                "gdn_commit_requires_live_grant", panel, model_layer);
  }

  const std::size_t ordinal = state_ordinal_for_model_layer(model_layer);
  if (ordinal >= admission_.recurrent_layers.size() ||
      state_.active_bank_index >= admission_.recurrent_banks.size() ||
      state_.candidate_bank_index >= admission_.recurrent_banks.size()) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "gdn_commit_generation_invalid", panel, model_layer);
  }
  const auto& slice = admission_.recurrent_layers[ordinal];
  const auto& active_bank =
      admission_.recurrent_banks[state_.active_bank_index];
  const auto& candidate_bank =
      admission_.recurrent_banks[state_.candidate_bank_index];
  const bool grant_matches =
      grant.grant_identity_ != 0U &&
      grant.grant_identity_ == state_.pending_gdn_layer_grant_identity &&
      grant.owner_identity_ == state_.owner_identity &&
      grant.allocation_identity_ == state_.allocation_identity &&
      grant.request_epoch_ == state_.request_epoch &&
      grant.state_epoch_ == state_.state_epoch && grant.panel_ == panel &&
      grant.model_layer_ == model_layer &&
      grant.state_layer_ordinal_ == ordinal &&
      grant.active_bank_index_ == state_.active_bank_index &&
      grant.candidate_bank_index_ == state_.candidate_bank_index &&
      grant.active_conv_allocation_offset_ ==
          active_bank.allocation_offset + slice.conv_offset &&
      grant.candidate_conv_allocation_offset_ ==
          candidate_bank.allocation_offset + slice.conv_offset &&
      grant.conv_bytes_ == slice.conv_bytes &&
      grant.active_gdn_state_allocation_offset_ ==
          active_bank.allocation_offset + slice.gdn_state_offset &&
      grant.candidate_gdn_state_allocation_offset_ ==
          candidate_bank.allocation_offset + slice.gdn_state_offset &&
      grant.gdn_state_bytes_ == slice.gdn_state_bytes;
  if (!grant_matches) {
    return fail(Sm87MacroFeedV4RequestStateError::kGdnLayerGrantMismatch,
                "gdn_commit_grant_generation_or_slice_mismatch", panel,
                model_layer);
  }

  state_.pending_gdn_layer_grant_identity = 0U;
  ++state_.panel_conv_layers_prepared;
  ++state_.panel_gdn_layers_assigned;
  state_.panel_conv_copy_bytes += slice.conv_bytes;
  state_.panel_gdn_assignment_bytes += slice.gdn_state_bytes;
  state_.total_conv_copy_bytes += slice.conv_bytes;
  state_.total_gdn_assignment_bytes += slice.gdn_state_bytes;
  ++state_.next_model_layer;
  if (state_.next_model_layer == kSm87MacroFeedV4LayerCount) {
    state_.candidate_epoch_complete =
        state_.panel_conv_layers_prepared ==
            kSm87MacroFeedV4StateLayerCount &&
        state_.panel_gdn_layers_assigned ==
            kSm87MacroFeedV4StateLayerCount &&
        state_.panel_kv_layers_staged ==
            kSm87MacroFeedV4FullAttentionLayerCount;
    if (state_.candidate_epoch_complete) {
      state_.phase = Sm87MacroFeedV4RequestStatePhase::kPanelReady;
    }
  }
  grant.invalidate();
  return ok();
}

Sm87MacroFeedV4FullAttentionKvAuthorizationResult
Sm87MacroFeedV4RequestState::authorize_full_attention_kv(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const std::size_t panel, const std::size_t model_layer) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  Sm87MacroFeedV4FullAttentionKvAuthorizationResult result;
  result.status = validate_access(access);
  if (!result.status) {
    return result;
  }
  if (!admission_.kv_physical_owner_bound) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kCapabilityMismatch,
        "full_attention_kv_grant_requires_explicit_physical_owner", panel,
        model_layer);
    return result;
  }
  if (state_.pending_event_receipt_identity != 0U) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
        "full_attention_kv_grant_blocked_by_pending_owner_event", panel,
        model_layer);
    return result;
  }
  if (state_.pending_gdn_layer_grant_identity != 0U ||
      state_.current_conv_layer_prepared) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kGdnLayerGrantPending,
        "full_attention_kv_grant_blocked_by_gdn_grant", panel,
        model_layer);
    return result;
  }
  if (state_.pending_full_attention_kv_grant_identity != 0U) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kFullAttentionKvGrantPending,
        "full_attention_kv_grant_already_pending", panel, model_layer);
    return result;
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelActive) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kInvalidTransition,
        "full_attention_kv_grant_requires_active_panel", panel,
        model_layer);
    return result;
  }
  if (panel != state_.active_panel) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                         "full_attention_kv_grant_panel_mismatch", panel,
                         model_layer);
    return result;
  }
  if (model_layer != state_.next_model_layer) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kLayerMismatch,
                         "full_attention_kv_grant_layer_out_of_order", panel,
                         model_layer);
    return result;
  }
  if (!full_attention_layer(model_layer)) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kLayerKindMismatch,
        "full_attention_kv_grant_requires_attention_layer", panel,
        model_layer);
    return result;
  }
  const std::size_t ordinal =
      attention_ordinal_for_model_layer(model_layer);
  if (ordinal >= admission_.full_attention_kv_layers.size() ||
      admission_.full_attention_kv_layers[ordinal].model_layer !=
          model_layer ||
      admission_.kv_allocation_identity == 0U ||
      admission_.kv_allocation_bytes !=
          kSm87MacroFeedV4AttentionKvArenaBytes) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kInvalidTransition,
        "full_attention_kv_grant_slice_or_arena_invalid", panel,
        model_layer);
    return result;
  }

  const auto& slice = admission_.full_attention_kv_layers[ordinal];
  const std::size_t first_position =
      panel * kSm87MacroFeedV4PanelTokens;
  const std::size_t candidate_end =
      first_position + kSm87MacroFeedV4PanelTokens;
  if (state_.private_kv_valid_end != first_position ||
      state_.candidate_kv_valid_end != first_position) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kKvValidEndMismatch,
        "full_attention_kv_grant_previous_valid_end_mismatch", panel,
        model_layer);
    return result;
  }
  const std::uint64_t panel_relative_offset =
      static_cast<std::uint64_t>(first_position) *
      kSm87MacroFeedV4AttentionKvRowStride * kSm87MacroFeedV4Bf16Bytes;
  if (panel_relative_offset + kSm87MacroFeedV4AttentionKvPanelBytes >
          slice.key_bytes ||
      panel_relative_offset + kSm87MacroFeedV4AttentionKvPanelBytes >
          slice.value_bytes) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kInvalidTransition,
        "full_attention_kv_grant_panel_span_out_of_slice", panel,
        model_layer);
    return result;
  }

  const std::uint64_t grant_identity =
      next_nonzero(&g_next_full_attention_kv_grant_identity);
  result.grant.emplace(Sm87MacroFeedV4FullAttentionKvGrant(
      grant_identity, state_.owner_identity, state_.request_epoch,
      state_.state_epoch, admission_.kv_allocation_identity, panel, ordinal,
      model_layer, slice.key_full_allocation_origin,
      slice.value_full_allocation_origin,
      slice.key_full_allocation_origin + panel_relative_offset,
      slice.value_full_allocation_origin + panel_relative_offset,
      kSm87MacroFeedV4AttentionKvPanelBytes, first_position,
      state_.private_kv_valid_end, candidate_end));
  state_.pending_full_attention_kv_grant_identity = grant_identity;
  result.status = ok();
  return result;
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::commit_full_attention_layer_enqueued(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    Sm87MacroFeedV4FullAttentionKvGrant&& grant) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (!admission_.kv_physical_owner_bound) {
    return fail(
        Sm87MacroFeedV4RequestStateError::kCapabilityMismatch,
        "full_attention_layer_commit_requires_explicit_physical_owner");
  }
  const std::size_t panel = grant.panel_;
  const std::size_t model_layer = grant.model_layer_;
  if (state_.pending_event_receipt_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "full_attention_kv_commit_blocked_by_pending_owner_event",
                panel, model_layer);
  }
  if (state_.pending_gdn_layer_grant_identity != 0U ||
      state_.current_conv_layer_prepared) {
    return fail(Sm87MacroFeedV4RequestStateError::kGdnLayerGrantPending,
                "full_attention_kv_commit_blocked_by_gdn_grant", panel,
                model_layer);
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelActive) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "full_attention_kv_commit_requires_active_panel", panel,
                model_layer);
  }
  if (state_.pending_full_attention_kv_grant_identity == 0U) {
    return fail(
        Sm87MacroFeedV4RequestStateError::kFullAttentionKvGrantMismatch,
        "full_attention_kv_commit_requires_live_grant", panel, model_layer);
  }
  if (panel != state_.active_panel) {
    return fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                "full_attention_kv_commit_panel_mismatch", panel,
                model_layer);
  }
  if (model_layer != state_.next_model_layer) {
    return fail(Sm87MacroFeedV4RequestStateError::kLayerMismatch,
                "full_attention_kv_commit_layer_out_of_order", panel,
                model_layer);
  }
  if (!full_attention_layer(model_layer)) {
    return fail(Sm87MacroFeedV4RequestStateError::kLayerKindMismatch,
                "full_attention_kv_commit_requires_attention_layer", panel,
                model_layer);
  }

  const std::size_t ordinal =
      attention_ordinal_for_model_layer(model_layer);
  if (ordinal >= admission_.full_attention_kv_layers.size()) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "full_attention_kv_commit_slice_invalid", panel,
                model_layer);
  }
  const auto& slice = admission_.full_attention_kv_layers[ordinal];
  const std::size_t first_position =
      panel * kSm87MacroFeedV4PanelTokens;
  const std::size_t candidate_end =
      first_position + kSm87MacroFeedV4PanelTokens;
  const std::uint64_t panel_relative_offset =
      static_cast<std::uint64_t>(first_position) *
      kSm87MacroFeedV4AttentionKvRowStride * kSm87MacroFeedV4Bf16Bytes;
  const bool grant_matches =
      grant.grant_identity_ != 0U &&
      grant.grant_identity_ ==
          state_.pending_full_attention_kv_grant_identity &&
      grant.owner_identity_ == state_.owner_identity &&
      grant.request_epoch_ == state_.request_epoch &&
      grant.state_epoch_ == state_.state_epoch &&
      grant.kv_allocation_identity_ == admission_.kv_allocation_identity &&
      grant.panel_ == panel &&
      grant.attention_layer_ordinal_ == ordinal &&
      grant.model_layer_ == model_layer &&
      grant.key_full_allocation_origin_ ==
          slice.key_full_allocation_origin &&
      grant.value_full_allocation_origin_ ==
          slice.value_full_allocation_origin &&
      grant.key_panel_allocation_offset_ ==
          slice.key_full_allocation_origin + panel_relative_offset &&
      grant.value_panel_allocation_offset_ ==
          slice.value_full_allocation_origin + panel_relative_offset &&
      grant.panel_bytes_ == kSm87MacroFeedV4AttentionKvPanelBytes &&
      grant.first_position_ == first_position &&
      grant.previous_valid_end_ == state_.private_kv_valid_end &&
      grant.previous_valid_end_ == state_.candidate_kv_valid_end &&
      grant.candidate_end_ == candidate_end;
  if (!grant_matches) {
    return fail(
        Sm87MacroFeedV4RequestStateError::kFullAttentionKvGrantMismatch,
        "full_attention_kv_commit_grant_generation_or_slice_mismatch", panel,
        model_layer);
  }

  state_.pending_full_attention_kv_grant_identity = 0U;
  ++state_.panel_kv_layers_staged;
  if (state_.panel_kv_layers_staged ==
      kSm87MacroFeedV4FullAttentionLayerCount) {
    state_.candidate_kv_valid_end = grant.candidate_end_;
  }
  ++state_.next_model_layer;
  if (state_.next_model_layer == kSm87MacroFeedV4LayerCount) {
    state_.candidate_epoch_complete =
        state_.panel_conv_layers_prepared ==
            kSm87MacroFeedV4StateLayerCount &&
        state_.panel_gdn_layers_assigned ==
            kSm87MacroFeedV4StateLayerCount &&
        state_.panel_kv_layers_staged ==
            kSm87MacroFeedV4FullAttentionLayerCount;
    if (state_.candidate_epoch_complete) {
      state_.phase = Sm87MacroFeedV4RequestStatePhase::kPanelReady;
    }
  }
  grant.invalidate();
  return ok();
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::prepare_conv_layer_candidate(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const std::size_t panel, const std::size_t model_layer) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (state_.pending_event_receipt_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "conv_prepare_blocked_by_pending_owner_event", panel,
                model_layer);
  }
  if (state_.pending_gdn_layer_grant_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kGdnLayerGrantPending,
                "conv_prepare_blocked_by_gdn_grant", panel, model_layer);
  }
  if (state_.pending_full_attention_kv_grant_identity != 0U) {
    return fail(
        Sm87MacroFeedV4RequestStateError::kFullAttentionKvGrantPending,
        "conv_prepare_blocked_by_full_attention_kv_grant", panel,
        model_layer);
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelActive) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "conv_prepare_requires_active_panel", panel, model_layer);
  }
  if (panel != state_.active_panel) {
    return fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                "conv_prepare_panel_mismatch", panel, model_layer);
  }
  if (model_layer != state_.next_model_layer) {
    return fail(Sm87MacroFeedV4RequestStateError::kLayerMismatch,
                "conv_prepare_layer_out_of_order", panel, model_layer);
  }
  if (!gdn_layer(model_layer)) {
    return fail(Sm87MacroFeedV4RequestStateError::kLayerKindMismatch,
                "conv_prepare_requires_gdn_layer", panel, model_layer);
  }
  if (state_.current_conv_layer_prepared) {
    return fail(Sm87MacroFeedV4RequestStateError::kDuplicateCompletion,
                "conv_layer_already_prepared", panel, model_layer);
  }
  state_.current_conv_layer_prepared = true;
  ++state_.panel_conv_layers_prepared;
  state_.panel_conv_copy_bytes += kSm87MacroFeedV4ConvLayerBytes;
  state_.total_conv_copy_bytes += kSm87MacroFeedV4ConvLayerBytes;
  return ok();
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::assign_gdn_layer_candidate(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const std::size_t panel, const std::size_t model_layer) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (state_.pending_event_receipt_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "gdn_assignment_blocked_by_pending_owner_event", panel,
                model_layer);
  }
  if (state_.pending_gdn_layer_grant_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kGdnLayerGrantPending,
                "gdn_assignment_blocked_by_gdn_grant", panel, model_layer);
  }
  if (state_.pending_full_attention_kv_grant_identity != 0U) {
    return fail(
        Sm87MacroFeedV4RequestStateError::kFullAttentionKvGrantPending,
        "gdn_assignment_blocked_by_full_attention_kv_grant", panel,
        model_layer);
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelActive) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "gdn_assignment_requires_active_panel", panel, model_layer);
  }
  if (panel != state_.active_panel) {
    return fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                "gdn_assignment_panel_mismatch", panel, model_layer);
  }
  if (model_layer != state_.next_model_layer) {
    return fail(Sm87MacroFeedV4RequestStateError::kLayerMismatch,
                "gdn_assignment_layer_out_of_order", panel, model_layer);
  }
  if (!gdn_layer(model_layer)) {
    return fail(Sm87MacroFeedV4RequestStateError::kLayerKindMismatch,
                "gdn_assignment_requires_gdn_layer", panel, model_layer);
  }
  if (!state_.current_conv_layer_prepared) {
    return fail(Sm87MacroFeedV4RequestStateError::kCandidateIncomplete,
                "gdn_assignment_requires_conv_prepare", panel, model_layer);
  }
  state_.current_conv_layer_prepared = false;
  ++state_.panel_gdn_layers_assigned;
  state_.panel_gdn_assignment_bytes += kSm87MacroFeedV4GdnStateLayerBytes;
  state_.total_gdn_assignment_bytes += kSm87MacroFeedV4GdnStateLayerBytes;
  ++state_.next_model_layer;
  if (state_.next_model_layer == kSm87MacroFeedV4LayerCount) {
    state_.candidate_epoch_complete =
        state_.panel_conv_layers_prepared ==
            kSm87MacroFeedV4StateLayerCount &&
        state_.panel_gdn_layers_assigned ==
            kSm87MacroFeedV4StateLayerCount &&
        state_.panel_kv_layers_staged ==
            kSm87MacroFeedV4FullAttentionLayerCount;
    if (state_.candidate_epoch_complete) {
      state_.phase = Sm87MacroFeedV4RequestStatePhase::kPanelReady;
    }
  }
  return ok();
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::stage_attention_kv_layer(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const std::size_t panel, const std::size_t model_layer,
    const std::size_t candidate_valid_end) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  (void)candidate_valid_end;
  return fail(Sm87MacroFeedV4RequestStateError::kCapabilityMismatch,
              "legacy_kv_stage_disabled_use_full_attention_grant", panel,
              model_layer);
}

Sm87MacroFeedV4RequestEventResult
Sm87MacroFeedV4RequestState::record_test_only_panel_completion(
    const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  Sm87MacroFeedV4RequestEventResult result;
  result.status = validate_access(access);
  if (!result.status) {
    return result;
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelReady ||
      !state_.candidate_epoch_complete ||
      state_.next_model_layer != kSm87MacroFeedV4LayerCount ||
      state_.pending_gdn_layer_grant_identity != 0U ||
      state_.pending_full_attention_kv_grant_identity != 0U) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kCandidateIncomplete,
        "panel_event_requires_layer63_and_complete_candidate",
        state_.active_panel);
    return result;
  }
  if (state_.pending_event_receipt_identity != 0U) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                         "panel_event_already_pending", state_.active_panel);
    return result;
  }
  result.receipt =
      mint_event_receipt(Sm87MacroFeedV4RequestEventKind::kPanelCommit);
  result.status = ok();
  return result;
}

Sm87MacroFeedV4RequestStateStatus Sm87MacroFeedV4RequestState::commit_panel(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const Sm87MacroFeedV4RequestEventReceipt& receipt) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  const std::size_t panel = receipt.panel;
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelReady) {
    return fail(Sm87MacroFeedV4RequestStateError::kCandidateIncomplete,
                "panel_commit_requires_complete_candidate", panel);
  }
  if (panel != state_.active_panel || panel != state_.completed_panels) {
    return fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                "panel_commit_panel_mismatch", panel);
  }
  if (state_.pending_event_receipt_identity == 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMissing,
                "panel_commit_requires_owner_event", panel);
  }
  if (!event_receipt_matches(
          receipt, Sm87MacroFeedV4RequestEventKind::kPanelCommit) ||
      receipt.completed_model_layer != kSm87MacroFeedV4LayerCount - 1U ||
      receipt.conv_layers != kSm87MacroFeedV4StateLayerCount ||
      receipt.gdn_layers != kSm87MacroFeedV4StateLayerCount ||
      receipt.kv_layers != kSm87MacroFeedV4FullAttentionLayerCount ||
      receipt.conv_copy_bytes != kSm87MacroFeedV4ConvEpochBytes ||
      receipt.gdn_assignment_bytes != kSm87MacroFeedV4GdnEpochBytes ||
      receipt.source_bank_identity != state_.active_bank_identity ||
      receipt.target_bank_identity != state_.candidate_bank_identity ||
      receipt.source_bank_identity == receipt.target_bank_identity ||
      receipt.copy_bytes != 0U ||
      receipt.private_kv_valid_end !=
          (panel + 1U) * kSm87MacroFeedV4PanelTokens) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "panel_commit_event_generation_mismatch", panel);
  }
  const std::size_t expected_end =
      (panel + 1U) * kSm87MacroFeedV4PanelTokens;
  if (!state_.candidate_epoch_complete ||
      state_.next_model_layer != kSm87MacroFeedV4LayerCount ||
      state_.panel_conv_layers_prepared !=
          kSm87MacroFeedV4StateLayerCount ||
      state_.panel_gdn_layers_assigned !=
          kSm87MacroFeedV4StateLayerCount ||
      state_.panel_kv_layers_staged !=
          kSm87MacroFeedV4FullAttentionLayerCount ||
      state_.panel_conv_copy_bytes != kSm87MacroFeedV4ConvEpochBytes ||
      state_.panel_gdn_assignment_bytes != kSm87MacroFeedV4GdnEpochBytes ||
      state_.whole_epoch_copy_bytes != 0U ||
      state_.pending_gdn_layer_grant_identity != 0U ||
      state_.pending_full_attention_kv_grant_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kCandidateIncomplete,
                "panel_commit_candidate_coverage_incomplete", panel);
  }
  if (state_.candidate_kv_valid_end != expected_end) {
    return fail(Sm87MacroFeedV4RequestStateError::kKvValidEndMismatch,
                "panel_commit_private_kv_incomplete", panel);
  }
  std::swap(state_.active_bank_index, state_.candidate_bank_index);
  std::swap(state_.active_bank_identity, state_.candidate_bank_identity);
  state_.private_kv_valid_end = state_.candidate_kv_valid_end;
  ++state_.state_epoch;
  ++state_.completed_panels;
  ++state_.panel_swap_count;
  state_.active_panel = kSm87MacroFeedV4PanelCount;
  state_.next_model_layer = 0U;
  state_.candidate_epoch_complete = false;
  state_.current_conv_layer_prepared = false;
  state_.pending_event_receipt_identity = 0U;
  state_.phase =
      state_.completed_panels == kSm87MacroFeedV4PanelCount
          ? Sm87MacroFeedV4RequestStatePhase::kAllPanelsPrivate
          : Sm87MacroFeedV4RequestStatePhase::kBetweenPanelsPrivate;
  return ok();
}

Sm87MacroFeedV4RequestEventResult
Sm87MacroFeedV4RequestState::record_test_only_owner_drain_completion(
    const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  Sm87MacroFeedV4RequestEventResult result;
  result.status = validate_access(access);
  if (!result.status) {
    return result;
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelActive &&
      state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelReady &&
      state_.phase !=
          Sm87MacroFeedV4RequestStatePhase::kFinalPublicationArmed) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                         "owner_drain_requires_active_producer");
    return result;
  }
  if (state_.pending_event_receipt_identity != 0U) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                         "another_owner_event_is_pending",
                         state_.active_panel);
    return result;
  }
  result.receipt =
      mint_event_receipt(Sm87MacroFeedV4RequestEventKind::kOwnerDrain);
  result.status = ok();
  return result;
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::discard_active_panel(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const Sm87MacroFeedV4RequestEventReceipt& drain_receipt,
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  const std::size_t panel = drain_receipt.panel;
  if (!valid_discard_reason(reason)) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidDiscardReason,
                "discard_reason_invalid", panel);
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelActive &&
      state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelReady) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "discard_requires_active_candidate", panel);
  }
  if (panel != state_.active_panel || panel != state_.completed_panels) {
    return fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                "discard_panel_mismatch", panel);
  }
  if (state_.pending_event_receipt_identity == 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMissing,
                "discard_requires_owner_drain_event", panel);
  }
  if (!event_receipt_matches(
          drain_receipt, Sm87MacroFeedV4RequestEventKind::kOwnerDrain)) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "discard_owner_drain_generation_mismatch", panel);
  }
  state_.last_discarded_candidate_identity = state_.candidate_bank_identity;
  ++state_.candidate_discard_count;
  state_.candidate_kv_valid_end = state_.private_kv_valid_end;
  state_.candidate_epoch_complete = false;
  state_.current_conv_layer_prepared = false;
  state_.last_invalidated_gdn_layer_grant_identity =
      state_.pending_gdn_layer_grant_identity;
  state_.pending_gdn_layer_grant_identity = 0U;
  state_.last_invalidated_full_attention_kv_grant_identity =
      state_.pending_full_attention_kv_grant_identity;
  state_.pending_full_attention_kv_grant_identity = 0U;
  state_.active_panel = kSm87MacroFeedV4PanelCount;
  state_.pending_event_receipt_identity = 0U;
  state_.phase = discard_phase(reason);
  return ok();
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::
    discard_active_panel_after_physical_execution_drain(
        const Sm87MacroFeedV4RequestStateSealedAccess& access,
        const std::uint64_t execution_owner_identity,
        const std::uint64_t allocation_identity,
        const std::uint64_t request_epoch, const std::size_t panel,
        const std::uint64_t panel_generation,
        const std::uint64_t physical_receipt_identity,
        const bool poison_terminal,
        const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (!valid_discard_reason(reason)) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidDiscardReason,
                "physical_discard_reason_invalid", panel);
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelActive &&
      state_.phase != Sm87MacroFeedV4RequestStatePhase::kPanelReady) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "physical_discard_requires_active_candidate", panel);
  }
  if (execution_owner_identity == 0U ||
      execution_owner_identity != admission_.owner_identity ||
      allocation_identity == 0U ||
      allocation_identity != state_.allocation_identity ||
      request_epoch == 0U || request_epoch != state_.request_epoch ||
      panel_generation == 0U || physical_receipt_identity == 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "physical_owner_drain_identity_mismatch", panel);
  }
  if (panel != state_.active_panel || panel != state_.completed_panels) {
    return fail(Sm87MacroFeedV4RequestStateError::kPanelMismatch,
                "physical_discard_panel_mismatch", panel);
  }
  if (state_.pending_event_receipt_identity != 0U ||
      state_.physical_owner_drain_receipt_identity != 0U ||
      state_.physical_owner_drain_panel_generation != 0U ||
      state_.physical_execution_receipt_issued) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "physical_discard_requires_unique_completed_receipt", panel);
  }
  // A grant authorizes writes only to the candidate bank.  Once the exact
  // execution owner has physically drained every producer, an uncommitted
  // grant is safe to invalidate: no active-bank generation, KV visibility, or
  // publication state is advanced.  A moved grant that survives in caller
  // storage is thereafter rejected by both the terminal phase and this cleared
  // live identity.
  state_.last_invalidated_gdn_layer_grant_identity =
      state_.pending_gdn_layer_grant_identity;
  state_.pending_gdn_layer_grant_identity = 0U;
  state_.last_invalidated_full_attention_kv_grant_identity =
      state_.pending_full_attention_kv_grant_identity;
  state_.pending_full_attention_kv_grant_identity = 0U;
  state_.last_discarded_candidate_identity = state_.candidate_bank_identity;
  ++state_.candidate_discard_count;
  state_.candidate_kv_valid_end = state_.private_kv_valid_end;
  state_.candidate_epoch_complete = false;
  state_.current_conv_layer_prepared = false;
  state_.active_panel = kSm87MacroFeedV4PanelCount;
  state_.physical_owner_drain_receipt_identity = physical_receipt_identity;
  state_.physical_owner_drain_panel_generation = panel_generation;
  state_.physical_execution_receipt_issued = true;
  state_.physical_owner_drain_was_poison_terminal = poison_terminal;
  state_.phase = discard_phase(reason);
  return ok();
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::abort_unpublished_request(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (!valid_discard_reason(reason)) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidDiscardReason,
                "abort_reason_invalid");
  }
  const bool private_boundary =
      state_.phase == Sm87MacroFeedV4RequestStatePhase::kAdmittedPrivate ||
      state_.phase ==
          Sm87MacroFeedV4RequestStatePhase::kBetweenPanelsPrivate ||
      state_.phase == Sm87MacroFeedV4RequestStatePhase::kAllPanelsPrivate;
  const bool completed_final_boundary =
      state_.phase ==
          Sm87MacroFeedV4RequestStatePhase::kCanonicalStatePublished &&
      state_.fallible_work_closed && state_.canonical_state_published &&
      state_.pending_event_receipt_identity == 0U;
  if (!private_boundary && !completed_final_boundary) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "abort_requires_quiesced_unpublished_boundary");
  }
  state_.pending_event_receipt_identity = 0U;
  state_.phase = discard_phase(reason);
  return ok();
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::begin_final_canonical_copy(
    const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (state_.phase != Sm87MacroFeedV4RequestStatePhase::kAllPanelsPrivate ||
      state_.completed_panels != kSm87MacroFeedV4PanelCount ||
      state_.state_epoch != kSm87MacroFeedV4PanelCount ||
      state_.private_kv_valid_end != kSm87MacroFeedV4P40Tokens ||
      state_.active_bank_index != 1U || state_.candidate_bank_index != 0U ||
      state_.active_bank_identity == 0U ||
      state_.candidate_bank_identity == 0U ||
      state_.active_bank_identity == state_.candidate_bank_identity ||
      state_.pending_event_receipt_identity != 0U ||
      state_.canonical_state_published ||
      state_.logical_sequence_fence_published) {
    return fail(
        Sm87MacroFeedV4RequestStateError::kFinalPublicationIncomplete,
        "final_copy_requires_private_epoch5_b_to_a");
  }
  state_.fallible_work_closed = false;
  state_.phase = Sm87MacroFeedV4RequestStatePhase::kFinalPublicationArmed;
  return ok();
}

Sm87MacroFeedV4RequestEventResult
Sm87MacroFeedV4RequestState::record_test_only_final_copy_completion(
    const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  Sm87MacroFeedV4RequestEventResult result;
  result.status = validate_access(access);
  if (!result.status) {
    return result;
  }
  if (state_.phase !=
          Sm87MacroFeedV4RequestStatePhase::kFinalPublicationArmed ||
      state_.completed_panels != kSm87MacroFeedV4PanelCount ||
      state_.state_epoch != kSm87MacroFeedV4PanelCount ||
      state_.private_kv_valid_end != kSm87MacroFeedV4P40Tokens ||
      state_.active_bank_index != 1U || state_.candidate_bank_index != 0U ||
      state_.canonical_state_published ||
      state_.logical_sequence_fence_published) {
    result.status = fail(
        Sm87MacroFeedV4RequestStateError::kFinalPublicationIncomplete,
        "final_publication_requires_five_private_panels");
    return result;
  }
  if (state_.pending_event_receipt_identity != 0U) {
    result.status = fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                         "final_publication_event_already_pending");
    return result;
  }
  state_.fallible_work_closed = true;
  result.receipt = mint_event_receipt(
      Sm87MacroFeedV4RequestEventKind::kFinalCanonicalPublish);
  result.status = ok();
  return result;
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::publish_canonical_state(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const Sm87MacroFeedV4RequestEventReceipt& receipt) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (state_.phase !=
          Sm87MacroFeedV4RequestStatePhase::kFinalPublicationArmed ||
      !state_.fallible_work_closed ||
      state_.private_kv_valid_end != kSm87MacroFeedV4P40Tokens ||
      state_.active_bank_identity == 0U ||
      state_.active_bank_identity == state_.candidate_bank_identity) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "canonical_publish_requires_armed_final_state");
  }
  if (state_.pending_event_receipt_identity == 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMissing,
                "canonical_publish_requires_owner_event");
  }
  if (!event_receipt_matches(
          receipt,
          Sm87MacroFeedV4RequestEventKind::kFinalCanonicalPublish) ||
      receipt.panel != kSm87MacroFeedV4PanelCount ||
      receipt.state_epoch != kSm87MacroFeedV4PanelCount ||
      receipt.completed_model_layer != kSm87MacroFeedV4LayerCount - 1U ||
      receipt.private_kv_valid_end != kSm87MacroFeedV4P40Tokens ||
      receipt.source_bank_identity != state_.active_bank_identity ||
      receipt.target_bank_identity != state_.candidate_bank_identity ||
      receipt.source_bank_identity == receipt.target_bank_identity ||
      receipt.copy_bytes != kSm87MacroFeedV4RecurrentEpochBytes ||
      state_.active_bank_index != 1U || state_.candidate_bank_index != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "canonical_publish_event_generation_mismatch");
  }
  state_.canonical_recurrent_source_identity = state_.active_bank_identity;
  state_.canonical_recurrent_target_identity = state_.candidate_bank_identity;
  state_.canonical_recurrent_copy_bytes = receipt.copy_bytes;
  state_.canonical_kv_valid_end = state_.private_kv_valid_end;
  state_.canonical_state_published = true;
  state_.pending_event_receipt_identity = 0U;
  state_.phase =
      Sm87MacroFeedV4RequestStatePhase::kCanonicalStatePublished;
  return ok();
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::discard_unpublished_final_copy(
    const Sm87MacroFeedV4RequestStateSealedAccess& access,
    const Sm87MacroFeedV4RequestEventReceipt& quiescence_receipt,
    const Sm87MacroFeedV4RequestDiscardReason reason) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (!valid_discard_reason(reason)) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidDiscardReason,
                "final_copy_discard_reason_invalid");
  }
  if (state_.phase !=
      Sm87MacroFeedV4RequestStatePhase::kFinalPublicationArmed) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "final_copy_discard_requires_armed_window");
  }
  if (state_.pending_event_receipt_identity == 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMissing,
                "final_copy_discard_requires_quiescence_receipt");
  }
  const bool drain_completed =
      quiescence_receipt.kind == Sm87MacroFeedV4RequestEventKind::kOwnerDrain &&
      event_receipt_matches(quiescence_receipt,
                            Sm87MacroFeedV4RequestEventKind::kOwnerDrain) &&
      quiescence_receipt.panel == kSm87MacroFeedV4PanelCount &&
      quiescence_receipt.state_epoch == kSm87MacroFeedV4PanelCount &&
      quiescence_receipt.private_kv_valid_end ==
          kSm87MacroFeedV4P40Tokens;
  const bool final_copy_completed =
      quiescence_receipt.kind ==
          Sm87MacroFeedV4RequestEventKind::kFinalCanonicalPublish &&
      event_receipt_matches(
          quiescence_receipt,
          Sm87MacroFeedV4RequestEventKind::kFinalCanonicalPublish) &&
      quiescence_receipt.panel == kSm87MacroFeedV4PanelCount &&
      quiescence_receipt.state_epoch == kSm87MacroFeedV4PanelCount &&
      quiescence_receipt.source_bank_identity == state_.active_bank_identity &&
      quiescence_receipt.target_bank_identity ==
          state_.candidate_bank_identity &&
      quiescence_receipt.source_bank_identity !=
          quiescence_receipt.target_bank_identity &&
      quiescence_receipt.copy_bytes ==
          kSm87MacroFeedV4RecurrentEpochBytes;
  if (!drain_completed && !final_copy_completed) {
    return fail(Sm87MacroFeedV4RequestStateError::kEventReceiptMismatch,
                "final_copy_discard_quiescence_generation_mismatch");
  }
  state_.pending_event_receipt_identity = 0U;
  state_.fallible_work_closed = true;
  state_.logical_sequence_fence_published = false;
  state_.phase = discard_phase(reason);
  return ok();
}

Sm87MacroFeedV4RequestStateStatus
Sm87MacroFeedV4RequestState::publish_sequence_length_fence(
    const Sm87MacroFeedV4RequestStateSealedAccess& access) noexcept {
  const std::lock_guard<std::mutex> lock(owner_mutex_);
  const auto capability = validate_access(access);
  if (!capability) {
    return capability;
  }
  if (state_.phase !=
          Sm87MacroFeedV4RequestStatePhase::kCanonicalStatePublished ||
      !state_.fallible_work_closed || !state_.canonical_state_published ||
      state_.canonical_kv_valid_end != kSm87MacroFeedV4P40Tokens ||
      state_.canonical_sequence_length != 0U ||
      state_.logical_sequence_fence_published ||
      state_.pending_event_receipt_identity != 0U) {
    return fail(Sm87MacroFeedV4RequestStateError::kInvalidTransition,
                "sequence_fence_requires_canonical_state");
  }
  state_.canonical_sequence_length = kSm87MacroFeedV4P40Tokens;
  state_.logical_sequence_fence_published = true;
  state_.phase =
      Sm87MacroFeedV4RequestStatePhase::kSequenceLengthPublished;
  return ok();
}

}  // namespace q3x::runtime
