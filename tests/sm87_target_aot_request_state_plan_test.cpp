#include "q3x/runtime/sm87_target_aot_request_state.h"

#include "../src/runtime/sm87_target_aot_request_state_access_internal.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

namespace {

namespace runtime = q3x::runtime;

class TestContext {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <typename T, typename = void>
struct HasArenaData : std::false_type {};

template <typename T>
struct HasArenaData<T,
                    std::void_t<decltype(std::declval<T&>().arena_data())>>
    : std::true_type {};

static_assert(!std::is_default_constructible_v<
              runtime::Sm87TargetAotP40RequestState>);
static_assert(!std::is_copy_constructible_v<
              runtime::Sm87TargetAotP40RequestState>);
static_assert(!std::is_move_constructible_v<
              runtime::Sm87TargetAotP40RequestState>);
static_assert(
    !HasArenaData<runtime::Sm87TargetAotP40RequestState>::value);
static_assert(runtime::kSm87TargetAotP40RequestArenaBytes ==
              5'075'652'608ULL);
static_assert(runtime::kSm87TargetAotP40RequestLocalRopeBytes == 0U);
static_assert(runtime::kSm87TargetAotP40OwnedEventCount == 519U);
static_assert(!std::is_default_constructible_v<
              runtime::sm87_target_aot_request_detail::
                  OwnerBoundExecutionTransaction>);
static_assert(std::is_trivially_copyable_v<
              runtime::Sm87TargetAotRequestRearmResult>);

void test_rearm_receipt_contract(TestContext& test) {
  runtime::Sm87TargetAotRequestRearmResult receipt;
  test.expect(!receipt, "empty rearm receipt is not a successful audit");

  receipt.source_phase =
      runtime::Sm87TargetAotRequestTransactionPhase::kCommitted;
  receipt.result_phase =
      runtime::Sm87TargetAotRequestTransactionPhase::kAdmittedUnpublished;
  receipt.admission_epoch = 11U;
  receipt.previous_transaction_epoch = 12U;
  receipt.request_transaction_epoch = 13U;
  receipt.previous_cold_reset_epoch = 10U;
  receipt.cold_reset_epoch = 14U;
  receipt.event_reset_epoch = 14U;
  receipt.allocation_identity = 15U;
  receipt.stream_identity = 16U;
  receipt.cancellation_signal_identity = 17U;
  receipt.zeroed_arena_bytes =
      runtime::kSm87TargetAotP40RequestArenaBytes;
  receipt.logical_event_count_reset =
      runtime::kSm87TargetAotP40OwnedEventCount;
  receipt.stream_drained = true;
  receipt.addresses_and_identities_preserved = true;
  test.expect(static_cast<bool>(receipt),
              "complete rearm receipt proves cold reset and stable ownership");

  receipt.request_transaction_epoch = receipt.previous_transaction_epoch;
  test.expect(!receipt,
              "rearm receipt rejects a reused request transaction epoch");
  test.expect(
      std::string(runtime::to_string(
          runtime::Sm87TargetAotRequestRearmError::kActiveTransaction)) ==
          "active_transaction",
      "rearm error has a stable audit spelling");
}

void test_exact_ledger(TestContext& test) {
  const auto plan =
      runtime::build_sm87_target_aot_p40_request_memory_plan();
  test.expect(
      runtime::validate_sm87_target_aot_p40_request_memory_plan(plan),
      "canonical target-AOT P40 request plan validates");
  test.expect(
      plan.prompt_tokens == 40'000U &&
          plan.request_capacity_tokens == 40'001U &&
          plan.arena_bytes == 5'075'652'608ULL &&
          plan.persistent_arena.byte_size == 2'699'952'128ULL &&
          plan.residual.storage.byte_size == 409'610'240ULL &&
          plan.family_arena.byte_size == 1'966'080'000ULL &&
          plan.final_hidden.storage.byte_size == 10'240ULL,
      "four physical request regions have the exact audited byte ledger");
  test.expect(
      plan.persistent_arena.arena_offset == 0U &&
          plan.residual.storage.arena_offset == 2'699'952'128ULL &&
          plan.family_arena.arena_offset == 3'109'562'368ULL &&
          plan.final_hidden.storage.arena_offset == 5'075'642'368ULL &&
          plan.final_hidden.storage.arena_offset +
                  plan.final_hidden.storage.byte_size ==
              plan.arena_bytes,
      "single allocation is contiguous and ends at the exact ledger byte");
  test.expect(
      plan.single_allocation && plan.cold_request_only &&
          plan.one_request_wide_commit &&
          plan.cancellation_discards_unpublished &&
          plan.kv_preprocess_in_place_alias_required &&
          !plan.exposes_raw_arena && !plan.permits_legacy_fallback,
      "owner policy is cold-only, transactional, opaque, and fallback-free");
  test.expect(
      !plan.engine_rope.included_in_request_arena &&
          plan.engine_rope.complete_table_bytes == 67'108'864ULL &&
          plan.engine_rope.maximum_position_embeddings == 262'144U &&
          plan.engine_rope.rotary_pairs == 32U,
      "RoPE is an explicit Engine-owned external requirement");
}

void test_persistent_layer_map(TestContext& test) {
  const auto plan =
      runtime::build_sm87_target_aot_p40_request_memory_plan();
  std::size_t gdn = 0U;
  std::size_t full = 0U;
  for (std::size_t layer = 0U; layer < plan.layers.size(); ++layer) {
    const auto& binding = plan.layers[layer];
    if ((layer + 1U) % 4U == 0U) {
      const bool exact =
          binding.kind ==
              runtime::Sm87TargetAotRequestLayerKind::kFullAttention &&
          binding.family_ordinal == full &&
          binding.key.rows == 40'001U && binding.key.columns == 1'024U &&
          binding.key.storage.byte_size == 81'922'048ULL &&
          binding.value.storage.byte_size == 81'922'048ULL &&
          binding.conv_history.byte_size == 0U &&
          binding.recurrent_state.byte_size == 0U;
      test.expect(exact, "full-Attention layer binds one exact K/V pair");
      ++full;
    } else {
      const bool exact =
          binding.kind == runtime::Sm87TargetAotRequestLayerKind::kGdn &&
          binding.family_ordinal == gdn &&
          binding.conv_history.byte_size == 61'440U &&
          binding.recurrent_state.byte_size == 1'572'864U &&
          binding.key.storage.byte_size == 0U &&
          binding.value.storage.byte_size == 0U;
      test.expect(exact, "GDN layer binds one exact state pair");
      ++gdn;
    }
  }
  test.expect(gdn == 48U && full == 16U,
              "all 64 layers have the canonical 48/16 schedule");
}

void test_family_lifetimes_and_aliases(TestContext& test) {
  const auto plan =
      runtime::build_sm87_target_aot_p40_request_memory_plan();
  const std::uint64_t family = plan.family_arena.arena_offset;
  test.expect(
      plan.gdn.raw_qkvz.storage.arena_offset == family &&
          plan.gdn.raw_qkvz.storage.byte_size == 1'310'720'000ULL &&
          plan.gdn.bf16_ab.storage.arena_offset ==
              family + 1'310'720'000ULL &&
          plan.gdn.normalized_input.storage.arena_offset ==
              family + 1'318'400'000ULL &&
          plan.gdn.output.storage.arena_offset ==
              plan.gdn.normalized_input.storage.arena_offset &&
          plan.gdn.o_branch.storage.arena_offset == family,
      "GDN producer, tail transition, and post-core branch are exact");
  test.expect(
      plan.attention.raw_q_gate.storage.arena_offset == family &&
          plan.attention.raw_q_gate.storage.byte_size == 983'040'000ULL &&
          plan.attention.pre_gate_output.storage.arena_offset == family &&
          plan.attention.gated_output.storage.arena_offset ==
              family + 491'520'000ULL &&
          plan.attention.processed_q.storage.arena_offset ==
              family + 983'040'000ULL &&
          plan.attention.processed_gate.storage.arena_offset ==
              family + 1'474'560'000ULL &&
          plan.attention.o_branch.storage.arena_offset ==
              family + 983'040'000ULL,
      "Attention uses two exact 983040000-byte lifetime banks");
  test.expect(
      plan.mlp.activated.storage.arena_offset == family &&
          plan.mlp.activated.storage.byte_size == 1'392'640'000ULL &&
          plan.mlp.normalized_input.storage.arena_offset ==
              family + 1'392'640'000ULL &&
          plan.mlp.down_branch.storage.arena_offset ==
              plan.mlp.normalized_input.storage.arena_offset,
      "MLP materializes only activated output and reclaims post-norm for Down");

  const auto& key_alias = plan.alias_contracts[7U];
  const auto& value_alias = plan.alias_contracts[8U];
  test.expect(
      key_alias.transition == runtime::Sm87TargetAotRequestAliasTransition::
                                  kExactInPlaceKvPreprocess &&
          value_alias.transition ==
              runtime::Sm87TargetAotRequestAliasTransition::
                  kExactInPlaceKvPreprocess &&
          key_alias.family_cardinality == 16U &&
          value_alias.family_cardinality == 16U &&
          key_alias.same_begin_required && key_alias.same_extent_required &&
          value_alias.same_begin_required && value_alias.same_extent_required,
      "all 16 K/V layers require exact in-place transaction aliases");
}

void test_forged_plans_fail_closed(TestContext& test) {
  const auto canonical =
      runtime::build_sm87_target_aot_p40_request_memory_plan();
  auto changed = canonical;
  changed.arena_bytes -= 256U;
  test.expect(
      !runtime::validate_sm87_target_aot_p40_request_memory_plan(changed),
      "short owner allocation fails closed");
  changed = canonical;
  changed.engine_rope.included_in_request_arena = true;
  test.expect(
      !runtime::validate_sm87_target_aot_p40_request_memory_plan(changed),
      "request-local RoPE forgery fails closed");
  changed = canonical;
  changed.persistent.key[0U].storage.arena_offset += 256U;
  test.expect(
      !runtime::validate_sm87_target_aot_p40_request_memory_plan(changed),
      "forged K transaction span fails closed");
  changed = canonical;
  changed.alias_contracts[7U].same_extent_required = false;
  test.expect(
      !runtime::validate_sm87_target_aot_p40_request_memory_plan(changed),
      "weakened K in-place alias fails closed");
  changed = canonical;
  changed.permits_legacy_fallback = true;
  test.expect(
      !runtime::validate_sm87_target_aot_p40_request_memory_plan(changed),
      "legacy fallback cannot enter the dedicated owner plan");
}

void test_default_off_create(TestContext& test) {
#if !defined(Q3X_ENABLE_SM87_TARGET_AOT_REQUEST_STATE_V1_ADMISSION)
  const auto result = runtime::create_sm87_target_aot_p40_request_state();
  test.expect(
      !result && result.value == nullptr &&
          result.diagnostic.code ==
              runtime::Sm87TargetAotRequestStateError::kAdmissionDisabled,
      "default-off owner creation fails before touching the GPU");
#else
  test.expect(runtime::kSm87TargetAotRequestStateV1AdmissionCompiled,
              "admission-enabled build records its compile boundary");
#endif
}

void test_host_execution_transaction_illegal_transitions(TestContext& test) {
  namespace detail = runtime::sm87_target_aot_request_detail;
  using Access = detail::Sm87TargetAotRequestStateAccess;
  using Global = detail::GlobalCompletionPoint;
  using Layer = detail::LayerCompletionPoint;
  constexpr std::uint64_t kEpoch = 0x513U;

  auto fixture = Access::host_fixture(kEpoch);
  test.expect(
      !Access::host_record_global(fixture, kEpoch, Global::kTokenIdsReady),
      "completion cannot be asserted before begin");
  test.expect(!Access::host_commit(fixture, kEpoch),
              "admitted owner cannot commit without execution");
  test.expect(static_cast<bool>(Access::host_begin(fixture)),
              "begin admits exactly one owner-bound execution epoch");
  test.expect(
      fixture.phase() == runtime::Sm87TargetAotRequestTransactionPhase::
                             kPrefillActiveUnpublished,
      "begin moves admitted owner to active unpublished execution");
  test.expect(!Access::host_begin(fixture),
              "a second begin cannot overlap one request");
  test.expect(
      !Access::host_record_layer(fixture, kEpoch, 0U,
                                 Layer::kInputProjections),
      "layer work cannot precede token and embedding completions");
  test.expect(
      !Access::host_record_global(fixture, kEpoch,
                                  Global::kEmbeddingComplete),
      "global completions cannot skip their owner-stream order");
  test.expect(
      !Access::host_record_global(fixture, kEpoch + 1U,
                                  Global::kTokenIdsReady),
      "a forged transaction epoch cannot record completion");
  test.expect(
      Access::host_record_global(fixture, kEpoch, Global::kTokenIdsReady) &&
          Access::host_record_global(fixture, kEpoch,
                                     Global::kEmbeddingComplete),
      "token and embedding completion prefix records in order");
  test.expect(
      !Access::host_record_layer(fixture, kEpoch, 1U,
                                 Layer::kInputProjections),
      "a later layer cannot overtake the current layer");
  test.expect(
      !Access::host_record_layer(
          fixture, kEpoch, 0U, Layer::kStateOrAttentionPreparation),
      "a later layer completion point cannot overtake its producer");

  bool all_layers_recorded = true;
  for (std::size_t layer = 0U;
       layer < runtime::kSm87TargetAotP40LayerCount; ++layer) {
    for (std::size_t point = 0U;
         point < runtime::kSm87TargetAotP40LayerEventCount; ++point) {
      all_layers_recorded =
          all_layers_recorded &&
          static_cast<bool>(Access::host_record_layer(
              fixture, kEpoch, layer, static_cast<Layer>(point)));
    }
  }
  test.expect(all_layers_recorded,
              "all 512 layer completion points record in canonical order");
  test.expect(!Access::host_commit(fixture, kEpoch),
              "layer completion alone cannot publish request state");
  test.expect(
      !Access::host_record_global(fixture, kEpoch,
                                  Global::kRequestCommit),
      "only commit owns the final CUDA event");

  const std::array<Global, 4U> suffix{{
      Global::kAllLayersComplete,
      Global::kFinalNormComplete,
      Global::kFinalHiddenComplete,
      Global::kPersistentStateStaged,
  }};
  bool suffix_recorded = true;
  for (const Global point : suffix) {
    suffix_recorded = suffix_recorded &&
                      static_cast<bool>(Access::host_record_global(
                          fixture, kEpoch, point));
  }
  test.expect(suffix_recorded,
              "all prescribed global prerequisites record in order");
  test.expect(static_cast<bool>(Access::host_commit(fixture, kEpoch)),
              "complete owner event ledger commits exactly once");
  test.expect(
      fixture.phase() ==
          runtime::Sm87TargetAotRequestTransactionPhase::kCommitted,
      "successful commit is the sole publication transition");
  test.expect(!Access::host_commit(fixture, kEpoch) &&
                  !Access::host_cancel(fixture, kEpoch) &&
                  !Access::host_record_layer(
                      fixture, kEpoch, 0U, Layer::kInputProjections),
              "committed owner rejects every later mutation");

  auto admitted_cancel = Access::host_fixture(kEpoch + 1U);
  test.expect(Access::host_cancel(admitted_cancel, kEpoch + 1U) &&
                  admitted_cancel.phase() ==
                      runtime::Sm87TargetAotRequestTransactionPhase::
                          kCancelled &&
                  !Access::host_begin(admitted_cancel),
              "admitted cancellation is terminal and unpublished");

  auto active_cancel = Access::host_fixture(kEpoch + 2U);
  test.expect(Access::host_begin(active_cancel) &&
                  !Access::host_rearm(active_cancel, kEpoch + 3U) &&
                  active_cancel.phase() ==
                      runtime::Sm87TargetAotRequestTransactionPhase::
                          kPrefillActiveUnpublished &&
                  active_cancel.transaction_epoch() == kEpoch + 2U &&
                  Access::host_cancel(active_cancel, kEpoch + 2U) &&
                  !Access::host_commit(active_cancel, kEpoch + 2U),
              "active cancellation prevents later publication");
  test.expect(
      Access::host_rearm(active_cancel, kEpoch + 3U) &&
          active_cancel.phase() ==
              runtime::Sm87TargetAotRequestTransactionPhase::
                  kAdmittedUnpublished &&
          active_cancel.transaction_epoch() == kEpoch + 3U &&
          !Access::host_record_global(active_cancel, kEpoch + 2U,
                                      Global::kTokenIdsReady) &&
          Access::host_begin(active_cancel) &&
          Access::host_record_global(active_cancel, kEpoch + 3U,
                                     Global::kTokenIdsReady),
      "cancelled owner rearms with a fresh epoch and a cleared ledger");

  auto committed_rearm = Access::host_fixture(kEpoch + 4U);
  test.expect(static_cast<bool>(Access::host_begin(committed_rearm)),
              "committed rearm fixture begins");
  bool committed_layers = true;
  committed_layers =
      committed_layers && static_cast<bool>(Access::host_record_global(
                              committed_rearm, kEpoch + 4U,
                              Global::kTokenIdsReady));
  committed_layers =
      committed_layers && static_cast<bool>(Access::host_record_global(
                              committed_rearm, kEpoch + 4U,
                              Global::kEmbeddingComplete));
  for (std::size_t layer = 0U;
       layer < runtime::kSm87TargetAotP40LayerCount; ++layer) {
    for (std::size_t point = 0U;
         point < runtime::kSm87TargetAotP40LayerEventCount; ++point) {
      committed_layers =
          committed_layers &&
          static_cast<bool>(Access::host_record_layer(
              committed_rearm, kEpoch + 4U, layer,
              static_cast<Layer>(point)));
    }
  }
  for (const Global point : suffix) {
    committed_layers =
        committed_layers &&
        static_cast<bool>(Access::host_record_global(
            committed_rearm, kEpoch + 4U, point));
  }
  test.expect(committed_layers &&
                  Access::host_commit(committed_rearm, kEpoch + 4U) &&
                  Access::host_rearm(committed_rearm, kEpoch + 5U) &&
                  committed_rearm.phase() ==
                      runtime::Sm87TargetAotRequestTransactionPhase::
                          kAdmittedUnpublished &&
                  committed_rearm.transaction_epoch() == kEpoch + 5U,
              "committed owner rearms to the exact admitted cold phase");

  auto admitted_rearm = Access::host_fixture(kEpoch + 6U);
  test.expect(!Access::host_rearm(admitted_rearm, 0U) &&
                  !Access::host_rearm(admitted_rearm, kEpoch + 6U) &&
                  Access::host_rearm(admitted_rearm, kEpoch + 7U) &&
                  admitted_rearm.transaction_epoch() == kEpoch + 7U,
              "admitted rearm requires a distinct nonzero epoch");

  auto zero_epoch = Access::host_fixture(0U);
  test.expect(!Access::host_begin(zero_epoch),
              "zero owner epoch fails closed before execution");
}

}  // namespace

int main() {
  TestContext test;
  test_exact_ledger(test);
  test_rearm_receipt_contract(test);
  test_persistent_layer_map(test);
  test_family_lifetimes_and_aliases(test);
  test_forged_plans_fail_closed(test);
  test_default_off_create(test);
  test_host_execution_transaction_illegal_transitions(test);
  return test.failures() == 0 ? 0 : 1;
}
