#include "../src/runtime/sm87_macrofeed_v4_request_state_internal.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>

namespace runtime = q3x::runtime;

static_assert(!std::is_default_constructible_v<
              runtime::Sm87MacroFeedV4RequestStateSealedAccess>);
static_assert(!std::is_default_constructible_v<
              runtime::Sm87MacroFeedV4RequestState>);
static_assert(!std::is_copy_constructible_v<
              runtime::Sm87MacroFeedV4RequestState>);
static_assert(!std::is_move_constructible_v<
              runtime::Sm87MacroFeedV4RequestState>);

namespace {

struct Test final {
  int failures = 0;

  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
};

[[nodiscard]] std::uint64_t snapshot_fingerprint(
    const runtime::Sm87MacroFeedV4RequestStateSnapshot& snapshot) noexcept {
  std::uint64_t value = 0xcbf29ce484222325ULL;
  const auto mix = [&value](const std::uint64_t item) noexcept {
    value ^= item;
    value *= 0x100000001b3ULL;
  };
  mix(static_cast<std::uint64_t>(snapshot.phase));
  mix(snapshot.active_bank_index);
  mix(snapshot.candidate_bank_index);
  mix(snapshot.active_bank_identity);
  mix(snapshot.candidate_bank_identity);
  mix(snapshot.owner_identity);
  mix(snapshot.allocation_identity);
  mix(snapshot.request_epoch);
  mix(snapshot.state_epoch);
  mix(snapshot.pending_event_receipt_identity);
  mix(snapshot.completed_panels);
  mix(snapshot.active_panel);
  mix(snapshot.next_model_layer);
  mix(snapshot.panel_conv_layers_prepared);
  mix(snapshot.panel_gdn_layers_assigned);
  mix(snapshot.panel_kv_layers_staged);
  mix(snapshot.panel_conv_copy_bytes);
  mix(snapshot.panel_gdn_assignment_bytes);
  mix(snapshot.total_conv_copy_bytes);
  mix(snapshot.total_gdn_assignment_bytes);
  mix(snapshot.whole_epoch_copy_bytes);
  mix(snapshot.private_kv_valid_end);
  mix(snapshot.candidate_kv_valid_end);
  mix(snapshot.canonical_kv_valid_end);
  mix(snapshot.canonical_sequence_length);
  mix(snapshot.canonical_recurrent_source_identity);
  mix(snapshot.canonical_recurrent_target_identity);
  mix(snapshot.canonical_recurrent_copy_bytes);
  mix(snapshot.panel_swap_count);
  mix(snapshot.candidate_discard_count);
  mix(snapshot.last_discarded_candidate_identity);
  mix(snapshot.current_conv_layer_prepared);
  mix(snapshot.candidate_epoch_complete);
  mix(snapshot.fallible_work_closed);
  mix(snapshot.canonical_state_published);
  mix(snapshot.logical_sequence_fence_published);
  mix(snapshot.decode_access_issued);
  mix(snapshot.physical_execution_receipt_issued);
  mix(snapshot.default_off);
  mix(snapshot.host_only);
  mix(snapshot.production_dispatch_eligible);
  return value;
}

void test_admission(Test& test) {
  const auto admission =
      runtime::make_sm87_macrofeed_v4_request_state_admission(
          101U, 202U, 1'001U, 1'002U);
  const auto validation =
      runtime::validate_sm87_macrofeed_v4_request_state_admission(admission);
  test.expect(validation.valid(), "canonical host-only admission validates");
  test.expect(admission.owner_identity == 101U &&
                  admission.allocation_identity == 202U &&
                  admission.owner_identity != admission.allocation_identity,
              "admission binds distinct owner and allocation identities");
  test.expect(admission.recurrent_banks[0U].storage_identity != 0U &&
                  admission.recurrent_banks[1U].storage_identity != 0U &&
                  admission.recurrent_banks[0U].storage_identity !=
                      admission.recurrent_banks[1U].storage_identity &&
                  admission.recurrent_banks[0U].bytes == 78'446'592U &&
                  admission.recurrent_banks[1U].bytes == 78'446'592U,
              "two exact recurrent epoch banks have distinct identities");
  test.expect(admission.allocation_bytes == 156'893'184U &&
                  admission.recurrent_banks[0U].owner_identity == 101U &&
                  admission.recurrent_banks[1U].owner_identity == 101U &&
                  admission.recurrent_banks[0U].allocation_identity == 202U &&
                  admission.recurrent_banks[1U].allocation_identity == 202U &&
                  admission.recurrent_banks[0U].allocation_offset == 0U &&
                  admission.recurrent_banks[1U].allocation_offset ==
                      78'446'592U,
              "physical bank identities bind one owner allocation without overlap");
  test.expect(admission.recurrent_layers.front().model_layer == 0U &&
                  admission.recurrent_layers.front().conv_offset == 0U &&
                  admission.recurrent_layers.front().gdn_state_offset ==
                      2'949'120U &&
                  admission.recurrent_layers.back().model_layer == 62U &&
                  admission.recurrent_layers.back().conv_offset +
                          admission.recurrent_layers.back().conv_bytes ==
                      2'949'120U &&
                  admission.recurrent_layers.back().gdn_state_offset +
                          admission.recurrent_layers.back().gdn_state_bytes ==
                      78'446'592U,
              "48 layer slices exactly tile Conv and GDN epoch regions");
  test.expect(
      admission.conv_history_copies_active_to_candidate_per_layer &&
          admission.gdn_first_update_reads_active_and_writes_candidate &&
          admission.gdn_continuation_reads_and_writes_candidate &&
          admission.candidate_epoch_fully_assigned_before_swap &&
          admission.whole_epoch_copy_forbidden,
      "admission forbids full epoch copy and requires per-layer assignment");
  test.expect(admission.host_only && admission.default_off &&
                  admission.test_only && !admission.cuda_handles_present &&
                  !admission.selector_bound && !admission.launcher_present &&
                  !admission.production_dispatch_eligible,
              "admission has no CUDA, selector, launcher, or production authority");

  auto malformed = admission;
  malformed.owner_identity = 0U;
  auto rejected =
      runtime::validate_sm87_macrofeed_v4_request_state_admission(malformed);
  test.expect(runtime::has_sm87_macrofeed_v4_request_admission_issue(
                  rejected,
                  runtime::Sm87MacroFeedV4RequestAdmissionIssue::
                      kBankOwnership),
              "zero owner identity fails closed");
  malformed = admission;
  malformed.recurrent_banks[1U].storage_identity =
      malformed.recurrent_banks[0U].storage_identity;
  rejected =
      runtime::validate_sm87_macrofeed_v4_request_state_admission(malformed);
  test.expect(runtime::has_sm87_macrofeed_v4_request_admission_issue(
                  rejected,
                  runtime::Sm87MacroFeedV4RequestAdmissionIssue::
                      kBankOwnership),
              "aliased active/candidate banks fail closed");
  malformed = admission;
  malformed.recurrent_layers[17U].gdn_state_offset += 2U;
  rejected =
      runtime::validate_sm87_macrofeed_v4_request_state_admission(malformed);
  test.expect(runtime::has_sm87_macrofeed_v4_request_admission_issue(
                  rejected,
                  runtime::Sm87MacroFeedV4RequestAdmissionIssue::kLayerLayout) &&
                  rejected.first_bad_state_layer == 17U,
              "per-layer recurrent layout drift is attributed and rejected");
  malformed = admission;
  malformed.whole_epoch_copy_forbidden = false;
  rejected =
      runtime::validate_sm87_macrofeed_v4_request_state_admission(malformed);
  test.expect(runtime::has_sm87_macrofeed_v4_request_admission_issue(
                  rejected,
                  runtime::Sm87MacroFeedV4RequestAdmissionIssue::
                      kTransitionContract),
              "whole-epoch copy permission fails closed");
  malformed = admission;
  malformed.private_kv_valid_end_identity =
      malformed.recurrent_banks[0U].storage_identity;
  rejected =
      runtime::validate_sm87_macrofeed_v4_request_state_admission(malformed);
  test.expect(runtime::has_sm87_macrofeed_v4_request_admission_issue(
                  rejected,
                  runtime::Sm87MacroFeedV4RequestAdmissionIssue::
                      kVisibilityOwnership),
              "KV valid-end cannot alias recurrent storage");
  malformed = admission;
  malformed.launcher_present = true;
  malformed.production_dispatch_eligible = true;
  rejected =
      runtime::validate_sm87_macrofeed_v4_request_state_admission(malformed);
  test.expect(runtime::has_sm87_macrofeed_v4_request_admission_issue(
                  rejected,
                  runtime::Sm87MacroFeedV4RequestAdmissionIssue::
                      kDispatchBoundary),
              "launcher or production authority cannot be synthesized");
}

[[nodiscard]] bool execute_panel_layers(
    Test& test, runtime::Sm87MacroFeedV4RequestState& state,
    const runtime::Sm87MacroFeedV4RequestStateSealedAccess& access,
    const std::size_t panel) {
  if (!state.begin_panel(access, panel)) {
    test.expect(false, "panel begins in exact order");
    return false;
  }
  for (std::size_t layer = 0U;
       layer < runtime::kSm87MacroFeedV4LayerCount; ++layer) {
    if (runtime::sm87_macrofeed_v4_expected_layer_kind(layer) ==
        runtime::Sm87MacroFeedV4LayerKind::kGdn) {
      test.expect(static_cast<bool>(
                      state.prepare_conv_layer_candidate(access, panel, layer)),
                  "GDN layer prepares one Conv history slice");
      test.expect(static_cast<bool>(
                      state.assign_gdn_layer_candidate(access, panel, layer)),
                  "GDN layer fully assigns one candidate state slice");
    } else {
      test.expect(static_cast<bool>(state.stage_attention_kv_layer(
                      access, panel, layer,
                      (panel + 1U) *
                          runtime::kSm87MacroFeedV4PanelTokens)),
                  "Attention layer stages one private KV slice");
      const auto kv_progress = state.snapshot();
      const std::size_t expected_aggregate_end =
          kv_progress.panel_kv_layers_staged ==
                  runtime::kSm87MacroFeedV4FullAttentionLayerCount
              ? (panel + 1U) * runtime::kSm87MacroFeedV4PanelTokens
              : panel * runtime::kSm87MacroFeedV4PanelTokens;
      test.expect(kv_progress.candidate_kv_valid_end ==
                      expected_aggregate_end,
                  "aggregate candidate KV endpoint advances only at 16/16");
    }
  }
  const auto snapshot = state.snapshot();
  test.expect(snapshot.phase ==
                      runtime::Sm87MacroFeedV4RequestStatePhase::kPanelReady &&
                  snapshot.next_model_layer == 64U &&
                  snapshot.panel_conv_layers_prepared == 48U &&
                  snapshot.panel_gdn_layers_assigned == 48U &&
                  snapshot.panel_kv_layers_staged == 16U &&
                  snapshot.panel_conv_copy_bytes == 2'949'120U &&
                  snapshot.panel_gdn_assignment_bytes == 75'497'472U &&
                  snapshot.whole_epoch_copy_bytes == 0U &&
                  snapshot.candidate_epoch_complete,
              "layer63 closes exactly 48 Conv/GDN and 16 KV slices");
  return true;
}

[[nodiscard]] bool drive_to_all_panels_private(
    Test& test, runtime::Sm87MacroFeedV4RequestState& state,
    const runtime::Sm87MacroFeedV4RequestStateSealedAccess& access) {
  for (std::size_t panel = 0U;
       panel < runtime::kSm87MacroFeedV4PanelCount; ++panel) {
    if (!execute_panel_layers(test, state, access, panel)) {
      return false;
    }
    const auto completion = state.record_test_only_panel_completion(access);
    if (!completion || !state.commit_panel(access, completion.receipt)) {
      test.expect(false, "fixture closes panel through its test-only ledger");
      return false;
    }
  }
  return state.snapshot().phase ==
         runtime::Sm87MacroFeedV4RequestStatePhase::kAllPanelsPrivate;
}

void test_capability_and_complete_commit(Test& test) {
  auto created = runtime::Sm87MacroFeedV4RequestState::create(
      runtime::make_sm87_macrofeed_v4_request_state_admission(
          301U, 302U, 3'001U, 3'002U));
  auto foreign = runtime::Sm87MacroFeedV4RequestState::create(
      runtime::make_sm87_macrofeed_v4_request_state_admission(
          401U, 402U, 4'001U, 4'002U));
  test.expect(static_cast<bool>(created) && static_cast<bool>(foreign),
              "two independent host state owners create");
  if (!created || !foreign) {
    return;
  }
  auto& state = *created.state;
  const auto access = state.issue_sealed_access();
  const auto foreign_access = foreign.state->issue_sealed_access();
  const auto initial = state.snapshot();
  test.expect(initial.owner_identity == 301U &&
                  initial.allocation_identity == 302U &&
                  initial.request_epoch != 0U && initial.state_epoch == 0U &&
                  access.owner_identity() == initial.owner_identity &&
                  access.allocation_identity() == initial.allocation_identity &&
                  access.request_epoch() == initial.request_epoch,
              "sealed access binds owner, allocation, and minted request epoch");
  const auto before_foreign = snapshot_fingerprint(initial);
  const auto foreign_result = state.begin_panel(foreign_access, 0U);
  test.expect(!foreign_result &&
                  foreign_result.error ==
                      runtime::Sm87MacroFeedV4RequestStateError::
                          kCapabilityMismatch &&
                  snapshot_fingerprint(state.snapshot()) == before_foreign,
              "foreign sealed access fails before any state mutation");
  test.expect(!state.begin_final_canonical_copy(access),
              "final publication cannot precede five panels");

  for (std::size_t panel = 0U;
       panel < runtime::kSm87MacroFeedV4PanelCount; ++panel) {
    if (!execute_panel_layers(test, state, access, panel)) {
      return;
    }
    const auto before_event = state.snapshot();
    runtime::Sm87MacroFeedV4RequestEventReceipt missing{};
    test.expect(!state.commit_panel(access, missing),
                "panel cannot commit without owner event receipt");
    const auto event = state.record_test_only_panel_completion(access);
    test.expect(static_cast<bool>(event) &&
                    event.receipt.owner_identity == initial.owner_identity &&
                    event.receipt.allocation_identity ==
                        initial.allocation_identity &&
                    event.receipt.request_epoch == initial.request_epoch &&
                    event.receipt.state_epoch == panel &&
                    event.receipt.panel == panel &&
                    event.receipt.completed_model_layer == 63U &&
                    event.receipt.conv_layers == 48U &&
                    event.receipt.gdn_layers == 48U &&
                    event.receipt.kv_layers == 16U &&
                    event.receipt.conv_copy_bytes == 2'949'120U &&
                    event.receipt.gdn_assignment_bytes == 75'497'472U &&
                    event.receipt.source_bank_identity ==
                        before_event.active_bank_identity &&
                    event.receipt.target_bank_identity ==
                        before_event.candidate_bank_identity &&
                    event.receipt.copy_bytes == 0U,
                "panel receipt binds generation, banks, coverage, and layer63");
    auto tampered = event.receipt;
    ++tampered.state_epoch;
    const auto frozen = snapshot_fingerprint(state.snapshot());
    test.expect(!state.commit_panel(access, tampered) &&
                    snapshot_fingerprint(state.snapshot()) == frozen,
                "tampered generation receipt fails without state mutation");
    tampered = event.receipt;
    ++tampered.source_bank_identity;
    test.expect(!state.commit_panel(access, tampered) &&
                    snapshot_fingerprint(state.snapshot()) == frozen,
                "wrong candidate-generation bank receipt fails closed");
    tampered = event.receipt;
    --tampered.conv_copy_bytes;
    test.expect(!state.commit_panel(access, tampered) &&
                    snapshot_fingerprint(state.snapshot()) == frozen,
                "incomplete Conv coverage receipt fails closed");
    test.expect(static_cast<bool>(state.commit_panel(access, event.receipt)),
                "owner event atomically swaps bank and private KV valid-end");
    const auto committed = state.snapshot();
    test.expect(committed.completed_panels == panel + 1U &&
                    committed.state_epoch == panel + 1U &&
                    committed.panel_swap_count == panel + 1U &&
                    committed.private_kv_valid_end == (panel + 1U) * 8'000U &&
                    committed.pending_event_receipt_identity == 0U &&
                    committed.active_bank_identity ==
                        before_event.candidate_bank_identity &&
                    committed.candidate_bank_identity ==
                        before_event.active_bank_identity,
                "bank identity and valid-end advance in one owner transition");
    test.expect(!state.commit_panel(access, event.receipt),
                "consumed receipt is stale after state epoch advances");
  }

  const auto all_private = state.snapshot();
  test.expect(all_private.phase ==
                      runtime::Sm87MacroFeedV4RequestStatePhase::
                          kAllPanelsPrivate &&
                  all_private.private_kv_valid_end == 40'000U &&
                  all_private.canonical_kv_valid_end == 0U &&
                  all_private.canonical_sequence_length == 0U &&
                  !all_private.logical_sequence_fence_published &&
                  all_private.total_conv_copy_bytes == 14'745'600U &&
                  all_private.total_gdn_assignment_bytes == 377'487'360U &&
                  all_private.whole_epoch_copy_bytes == 0U,
              "five panels remain private and never issue a full epoch copy");

  test.expect(static_cast<bool>(state.begin_final_canonical_copy(access)),
              "epoch five arms the exact canonical B-to-A copy");
  const auto final_event = state.record_test_only_final_copy_completion(access);
  test.expect(static_cast<bool>(final_event) &&
                  final_event.receipt.panel == 5U &&
                  final_event.receipt.state_epoch == 5U &&
                  final_event.receipt.private_kv_valid_end == 40'000U &&
                  final_event.receipt.source_bank_identity ==
                      all_private.active_bank_identity &&
                  final_event.receipt.target_bank_identity ==
                      all_private.candidate_bank_identity &&
                  final_event.receipt.copy_bytes == 78'446'592U &&
                  final_event.receipt.test_only_host_ledger_completed &&
                  !final_event.receipt.physical_device_completion_attested &&
                  !final_event.receipt.production_receipt_eligible &&
                  state.snapshot().fallible_work_closed,
              "test-only final ledger binds exact B-to-A copy without device authority");
  auto wrong_final = final_event.receipt;
  wrong_final.request_epoch += 1U;
  const auto before_wrong_final = snapshot_fingerprint(state.snapshot());
  test.expect(!state.publish_canonical_state(access, wrong_final) &&
                  snapshot_fingerprint(state.snapshot()) ==
                      before_wrong_final,
              "stale final request epoch cannot publish canonical state");
  test.expect(static_cast<bool>(
                  state.publish_canonical_state(access, final_event.receipt)),
              "completed final event publishes canonical recurrent and KV state");
  const auto canonical = state.snapshot();
  test.expect(canonical.canonical_state_published &&
                  canonical.canonical_recurrent_source_identity ==
                      canonical.active_bank_identity &&
                  canonical.canonical_recurrent_target_identity ==
                      canonical.candidate_bank_identity &&
                  canonical.canonical_recurrent_copy_bytes == 78'446'592U &&
                  canonical.canonical_kv_valid_end == 40'000U &&
                  canonical.canonical_sequence_length == 0U &&
                  !canonical.logical_sequence_fence_published &&
                  !canonical.decode_access_issued &&
                  !canonical.physical_execution_receipt_issued,
              "canonical copy remains invisible while sequence length is zero");
  const auto before_foreign_fence = snapshot_fingerprint(canonical);
  test.expect(!state.publish_sequence_length_fence(foreign_access) &&
                  snapshot_fingerprint(state.snapshot()) ==
                      before_foreign_fence,
              "foreign capability cannot cross the final visibility fence");
  test.expect(static_cast<bool>(state.publish_sequence_length_fence(access)),
              "sequence length zero-to-P40000 is the final nonfallible fence");
  const auto committed = state.snapshot();
  test.expect(committed.phase ==
                      runtime::Sm87MacroFeedV4RequestStatePhase::
                          kSequenceLengthPublished &&
                  committed.canonical_sequence_length == 40'000U &&
                  committed.logical_sequence_fence_published &&
                  !committed.decode_access_issued &&
                  !committed.physical_execution_receipt_issued &&
                  committed.pending_event_receipt_identity == 0U,
              "logical fence is published without Decode access or physical "
              "receipt");
  test.expect(!state.begin_panel(access, 0U) &&
                  !state.publish_sequence_length_fence(access) &&
                  !state.abort_unpublished_request(
                      access,
                      runtime::Sm87MacroFeedV4RequestDiscardReason::kFailed),
              "committed terminal state has no reuse or fallible transition");
}

void test_ordering_and_discard(Test& test) {
  auto created = runtime::Sm87MacroFeedV4RequestState::create(
      runtime::make_sm87_macrofeed_v4_request_state_admission(
          501U, 502U, 5'001U, 5'002U));
  test.expect(static_cast<bool>(created), "discard fixture creates");
  if (!created) {
    return;
  }
  auto& state = *created.state;
  const auto access = state.issue_sealed_access();
  test.expect(!state.prepare_conv_layer_candidate(access, 0U, 0U),
              "layer completion cannot precede panel begin");
  const auto initial = snapshot_fingerprint(state.snapshot());
  test.expect(!state.begin_panel(access, 1U) &&
                  snapshot_fingerprint(state.snapshot()) == initial,
              "panel gap fails without mutation");
  test.expect(static_cast<bool>(state.begin_panel(access, 0U)),
              "first panel begins");
  test.expect(!state.assign_gdn_layer_candidate(access, 0U, 0U),
              "GDN assignment requires its Conv slice first");
  test.expect(static_cast<bool>(
                  state.prepare_conv_layer_candidate(access, 0U, 0U)) &&
                  !state.prepare_conv_layer_candidate(access, 0U, 0U),
              "duplicate Conv preparation fails closed");
  test.expect(static_cast<bool>(
                  state.assign_gdn_layer_candidate(access, 0U, 0U)),
              "first GDN layer completes after Conv prepare");
  runtime::Sm87MacroFeedV4RequestEventReceipt missing{};
  test.expect(!state.discard_active_panel(
                  access, missing,
                  runtime::Sm87MacroFeedV4RequestDiscardReason::kCancelled),
              "candidate discard requires an owner drain event");
  const auto drain = state.record_test_only_owner_drain_completion(access);
  test.expect(static_cast<bool>(drain) && drain.receipt.panel == 0U &&
                  drain.receipt.state_epoch == 0U &&
                  drain.receipt.request_epoch == access.request_epoch(),
              "owner drain receipt binds request and state generations");
  test.expect(!state.prepare_conv_layer_candidate(access, 0U, 1U),
              "no layer mutation is accepted after drain completion");
  auto stale = drain.receipt;
  stale.allocation_identity += 1U;
  const auto frozen = snapshot_fingerprint(state.snapshot());
  test.expect(!state.discard_active_panel(
                  access, stale,
                  runtime::Sm87MacroFeedV4RequestDiscardReason::kCancelled) &&
                  snapshot_fingerprint(state.snapshot()) == frozen,
              "foreign allocation drain receipt cannot discard state");
  const auto candidate_identity = state.snapshot().candidate_bank_identity;
  test.expect(static_cast<bool>(state.discard_active_panel(
                  access, drain.receipt,
                  runtime::Sm87MacroFeedV4RequestDiscardReason::kCancelled)),
              "drained candidate is discarded without bank swap");
  const auto cancelled = state.snapshot();
  test.expect(cancelled.phase ==
                      runtime::Sm87MacroFeedV4RequestStatePhase::kCancelled &&
                  cancelled.completed_panels == 0U &&
                  cancelled.state_epoch == 0U &&
                  cancelled.panel_swap_count == 0U &&
                  cancelled.private_kv_valid_end == 0U &&
                  cancelled.candidate_discard_count == 1U &&
                  cancelled.last_discarded_candidate_identity ==
                      candidate_identity &&
                  !cancelled.logical_sequence_fence_published,
              "cancel preserves active state and records candidate discard");
  test.expect(!state.begin_panel(access, 0U) &&
                  !state.record_test_only_owner_drain_completion(access) &&
                  !state.abort_unpublished_request(
                      access,
                      runtime::Sm87MacroFeedV4RequestDiscardReason::kFailed),
              "terminal cancellation cannot be rearmed or reused by this slice");
}

void test_final_failure_windows(Test& test) {
  auto armed_failure = runtime::Sm87MacroFeedV4RequestState::create(
      runtime::make_sm87_macrofeed_v4_request_state_admission(
          601U, 602U, 6'001U, 6'002U));
  test.expect(static_cast<bool>(armed_failure),
              "armed-final failure fixture creates");
  if (!armed_failure) {
    return;
  }
  auto& first = *armed_failure.state;
  const auto first_access = first.issue_sealed_access();
  if (!drive_to_all_panels_private(test, first, first_access)) {
    test.expect(false, "armed-final fixture reaches five private panels");
    return;
  }
  test.expect(static_cast<bool>(first.begin_final_canonical_copy(first_access)),
              "failed canonical copy enters the armed window");
  runtime::Sm87MacroFeedV4RequestEventReceipt missing{};
  test.expect(!first.discard_unpublished_final_copy(
                  first_access, missing,
                  runtime::Sm87MacroFeedV4RequestDiscardReason::kFailed),
              "in-flight final copy cannot terminate without quiescence");
  const auto final_drain =
      first.record_test_only_owner_drain_completion(first_access);
  test.expect(static_cast<bool>(final_drain) &&
                  final_drain.receipt.panel == 5U &&
                  final_drain.receipt.state_epoch == 5U &&
                  final_drain.receipt.request_epoch ==
                      first_access.request_epoch(),
              "final drain binds the owner and epoch-five generation");
  auto stale_drain = final_drain.receipt;
  ++stale_drain.request_epoch;
  test.expect(!first.discard_unpublished_final_copy(
                  first_access, stale_drain,
                  runtime::Sm87MacroFeedV4RequestDiscardReason::kFailed),
              "stale final drain cannot terminate the owner");
  test.expect(static_cast<bool>(first.discard_unpublished_final_copy(
                  first_access, final_drain.receipt,
                  runtime::Sm87MacroFeedV4RequestDiscardReason::kFailed)),
              "drained failed canonical copy poisons the unpublished request");
  const auto poisoned = first.snapshot();
  test.expect(poisoned.phase ==
                      runtime::Sm87MacroFeedV4RequestStatePhase::kFailed &&
                  poisoned.canonical_sequence_length == 0U &&
                  !poisoned.logical_sequence_fence_published &&
                  !first.publish_sequence_length_fence(first_access) &&
                  !first.begin_panel(first_access, 0U),
              "poisoned final-copy window is invisible and terminal");

  auto canonical_poison = runtime::Sm87MacroFeedV4RequestState::create(
      runtime::make_sm87_macrofeed_v4_request_state_admission(
          701U, 702U, 7'001U, 7'002U));
  test.expect(static_cast<bool>(canonical_poison),
              "canonical-before-fence failure fixture creates");
  if (!canonical_poison) {
    return;
  }
  auto& second = *canonical_poison.state;
  const auto second_access = second.issue_sealed_access();
  if (!drive_to_all_panels_private(test, second, second_access) ||
      !second.begin_final_canonical_copy(second_access)) {
    test.expect(false, "canonical failure fixture reaches final copy");
    return;
  }
  const auto final_copy =
      second.record_test_only_final_copy_completion(second_access);
  test.expect(static_cast<bool>(final_copy) &&
                  static_cast<bool>(second.publish_canonical_state(
                      second_access, final_copy.receipt)),
              "fixture records the exact canonical copy");
  test.expect(static_cast<bool>(second.abort_unpublished_request(
                  second_access,
                  runtime::Sm87MacroFeedV4RequestDiscardReason::kFailed)),
              "completed canonical copy can fail before its logical fence");
  const auto poisoned_after_publish = second.snapshot();
  test.expect(poisoned_after_publish.phase ==
                      runtime::Sm87MacroFeedV4RequestStatePhase::kFailed &&
                  poisoned_after_publish.canonical_state_published &&
                  poisoned_after_publish.canonical_sequence_length == 0U &&
                  !poisoned_after_publish.logical_sequence_fence_published &&
                  !second.publish_sequence_length_fence(second_access) &&
                  !second.begin_panel(second_access, 0U) &&
                  !second.abort_unpublished_request(
                      second_access,
                      runtime::Sm87MacroFeedV4RequestDiscardReason::kFailed),
              "canonical-published pre-fence failure is invisible and terminal");
}

}  // namespace

int main() {
  Test test;
  test_admission(test);
  test_capability_and_complete_commit(test);
  test_ordering_and_discard(test);
  test_final_failure_windows(test);
  if (test.failures != 0) {
    std::cerr << test.failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "MacroFeed-v4 host-only request-state ownership tests passed\n";
  return 0;
}
