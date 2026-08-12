#include "q3x/kernels/sm87_target_aot_gdn_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

constexpr auto kP40 = kernels::sm87_target_aot_gdn_plan(40'000U);
constexpr auto kP60 = kernels::sm87_target_aot_gdn_plan(60'000U);
constexpr auto kP130 = kernels::sm87_target_aot_gdn_plan(130'000U);
constexpr auto kP40Owner0 = kernels::sm87_target_aot_gdn_owner_task(kP40, 0U);
constexpr auto kP40Owner15 =
    kernels::sm87_target_aot_gdn_owner_task(kP40, 15U);

static_assert(kP40.valid() && kP60.valid() && kP130.valid());
static_assert(kP40.capacity_bucket ==
                  kernels::Sm87TargetAotCapacityBucket::kP40 &&
              kP60.capacity_bucket ==
                  kernels::Sm87TargetAotCapacityBucket::kP60 &&
              kP130.capacity_bucket ==
                  kernels::Sm87TargetAotCapacityBucket::kP130);
static_assert(kP40.owner_ctas == 16U &&
              kP40.value_head_state_chains == 48U &&
              kP40.qk_groups == 16U && kP40.value_heads_per_owner == 3U &&
              kP40.state_dimension == 128U &&
              kP40.state_bytes_per_owner == 98'304U &&
              kP40.total_state_bytes == 1'572'864U &&
              kP40.packed_state_words_per_owner == 24'576U &&
              kP40.packed_state_words_per_thread == 96U);
static_assert(kP40.kernel_launches_per_layer == 1U &&
              kP40.persistent_ctas == 16U &&
              kP40.threads_per_cta == 256U &&
              kP40.warps_per_cta == 8U && kP40.layer_long_kernel &&
              kP40.packed_register_state_prompt_resident &&
              kP40.same_cta_collective_payload &&
              !kP40.independent_preparation_kernel);
static_assert(
    kP40.completion_contract ==
        kernels::Sm87TargetAotGdnCompletionContract::
            kIndependentOwnersKernelEventThenRequestTransactionAppend &&
    kP40.independent_owner_state_chains &&
    kP40.independent_c16_cancel_observation &&
    !kP40.cooperative_launch_required &&
    !kP40.cross_cta_grid_barrier_required &&
    kP40.cross_cta_barriers_per_layer == 0U &&
    !kP40.in_kernel_commit &&
    kP40.kernel_completion_events_per_layer == 1U &&
    kP40.post_event_request_transaction_appends_per_layer == 1U &&
    kP40.required_owner_receipts == 16U &&
    kP40.kernel_completion_event_waits_all_owners &&
    kP40.request_transaction_append_after_kernel_event &&
    kP40.transaction_spans_prebound_at_request_admission &&
    kP40.stream_ordered_ready_receipt && !kP40.host_callback_required &&
    !kP40.host_synchronization_required &&
    kP40.request_transaction_unpublished_spans &&
    kP40.cancel_discards_unpublished_spans &&
    kP40.pointer_publication_without_copy &&
    kP40.output_projection_read_requires_kernel_completion &&
    kP40.cancelled_span_reclamation_without_publication);
static_assert(kP40.first_position == 0U && kP40.cold_start_only &&
              !kP40.continuation_supported &&
              !kP40.initial_conv_history_device_read &&
              !kP40.initial_recurrent_state_device_read &&
              kP40.register_zero_initialization);
static_assert(
    kP40.numerical_contract ==
        kernels::Sm87TargetAotGdnNumericalContract::
            kQwen36ColdReferenceFp32FmaPerTokenBf16 &&
    kP40.conv_numerical_contract ==
        kernels::Sm87TargetAotGdnConvNumericalContract::
            kFp32FmaOldestToCurrentThenSiluThenBf16Rne &&
    kP40.conv_fp32_fma_oldest_history_to_current &&
    kP40.conv_silu_fp32_before_publication &&
    kP40.conv_output_bf16_rne_before_qkv_use &&
    kP40.z_bit_exact_bypasses_conv &&
    kP40.conv_history_stores_raw_current_bf16 &&
    kP40.qk_l2_fp32_index_order_fma &&
    kP40.q_only_scaled_by_inverse_sqrt_128 &&
    kP40.alpha_softplus_and_beta_sigmoid_fp32 &&
    kP40.prediction_update_output_fp32_index_order_fma &&
    kP40.per_token_bf16_state_rounding &&
    kP40.output_uses_pre_round_fp32_update &&
    kP40.raw_bf16_before_rmsnorm_silu_z);
static_assert(kP40.conv_width == 4U && kP40.conv_history == 3U &&
              kP40.conv_channels_per_owner == 640U &&
              kP40.total_conv_channels == 10'240U &&
              kP40.conv_history_bytes_per_owner == 3'840U &&
              kP40.total_conv_history_bytes == 61'440U &&
              kP40.causal_conv_token_order);
static_assert(
    kP40.raw_partitions[0U].role ==
            kernels::Sm87TargetAotGdnRawPartitionRole::kQ &&
    kP40.raw_partitions[0U].channel_offset == 0U &&
    kP40.raw_partitions[0U].channel_count == 2'048U &&
    kP40.raw_partitions[1U].channel_offset == 2'048U &&
    kP40.raw_partitions[2U].channel_offset == 4'096U &&
    kP40.raw_partitions[2U].channel_count == 6'144U &&
    kP40.raw_partitions[3U].role ==
        kernels::Sm87TargetAotGdnRawPartitionRole::kZ &&
    kP40.raw_partitions[3U].channel_offset == 10'240U &&
    kP40.raw_partitions[3U].channel_count == 6'144U &&
    !kP40.raw_partitions[3U].passes_causal_conv_silu_bf16 &&
    kP40.raw_partitions[3U].bit_exact_direct_gate_input);
static_assert(kP40.q_bytes_per_payload_slot == 4'096U &&
              kP40.k_bytes_per_payload_slot == 4'096U &&
              kP40.v_bytes_per_payload_slot == 12'288U &&
              kP40.z_bytes_per_payload_slot == 12'288U &&
              kP40.a_bytes_per_payload_slot == 96U &&
              kP40.b_bytes_per_payload_slot == 96U &&
              kP40.payload_bytes_per_slot == 32'960U &&
              kP40.private_shared_payload_bytes == 65'920U &&
              kP40.private_shared_c16_payload &&
              kP40.payload_reuse_after_same_cta_consumer_barrier &&
              kP40.producer_ready_event_count == 2U);
static_assert(kP40.exact_c16_blocks == 2'500U &&
              kP40.preparation_c64_macros == 625U &&
              kP40.terminal_macro_c16_blocks == 4U &&
              kP40.terminal_macro_tokens == 64U);
static_assert(kP60.exact_c16_blocks == 3'750U &&
              kP60.preparation_c64_macros == 938U &&
              kP60.terminal_macro_c16_blocks == 2U &&
              kP60.terminal_macro_tokens == 32U);
static_assert(kP130.exact_c16_blocks == 8'125U &&
              kP130.preparation_c64_macros == 2'032U &&
              kP130.terminal_macro_c16_blocks == 1U &&
              kP130.terminal_macro_tokens == 16U);
static_assert(kP40.preparation_slots == 2U &&
              kP40.preparation_ping_pong && !kP40.c64_state_composition &&
              !kP40.chunk_fp32_state_authoritative &&
              !kP40.full_prompt_awu_materialized &&
              kP40.c16_cancel_safe_point && !kP40.resource_qualified &&
              !kP40.production_dispatch_eligible);
static_assert(
    (kP40.policy & kernels::kSm87TargetAotGdnNoWy) != 0U &&
    (kP40.policy & kernels::kSm87TargetAotGdnNoKkt) != 0U &&
    (kP40.policy & kernels::kSm87TargetAotGdnNoSsd) != 0U &&
    (kP40.policy &
     kernels::kSm87TargetAotGdnNoFp32ChunkAuthoritativeState) != 0U &&
    (kP40.policy & kernels::kSm87TargetAotGdnIndependentOwnerStateChains) !=
        0U &&
    (kP40.policy & kernels::kSm87TargetAotGdnNoCrossCtaBarrier) != 0U &&
    (kP40.policy & kernels::kSm87TargetAotGdnNoCooperativeLaunch) != 0U &&
    (kP40.policy &
     kernels::kSm87TargetAotGdnPostEventRequestTransactionAppend) != 0U &&
    (kP40.policy & kernels::kSm87TargetAotGdnNoCopyPointerPublication) != 0U &&
    (kP40.policy &
     kernels::kSm87TargetAotGdnNoInitialStateHistoryDramRead) != 0U &&
    (kP40.policy &
     kernels::kSm87TargetAotGdnOutputProjectionReadsAfterKernelCompletion) !=
        0U &&
    (kP40.policy &
     kernels::kSm87TargetAotGdnCancelDiscardsWholeTransaction) != 0U &&
    (kP40.policy &
     kernels::kSm87TargetAotGdnTransactionSpansPreboundAtAdmission) != 0U &&
    (kP40.policy & kernels::kSm87TargetAotGdnStreamOrderedReadyReceipt) !=
        0U &&
    (kP40.policy & kernels::kSm87TargetAotGdnNoHostCallback) != 0U &&
    (kP40.policy & kernels::kSm87TargetAotGdnNoHostSynchronization) != 0U);
static_assert(kP40Owner0.valid && kP40Owner0.qk_group == 0U &&
              kP40Owner0.first_value_head == 0U &&
              kP40Owner0.q_conv.channel_begin == 0U &&
              kP40Owner0.k_conv.channel_begin == 2'048U &&
              kP40Owner0.v_conv.channel_begin == 4'096U);
static_assert(kP40Owner15.valid && kP40Owner15.qk_group == 15U &&
              kP40Owner15.first_value_head == 45U &&
              kP40Owner15.state_byte_offset == 1'474'560U &&
              kP40Owner15.q_conv.channel_begin == 1'920U &&
              kP40Owner15.k_conv.channel_begin == 3'968U &&
              kP40Owner15.v_conv.channel_begin == 9'856U);
static_assert(!kernels::sm87_target_aot_gdn_plan(39'999U).valid());
static_assert(!kernels::sm87_target_aot_gdn_plan(40'016U).valid());
static_assert(kernels::kSm87TargetAotGdnBindingIdentityCount == 70U);
static_assert(kernels::kSm87TargetAotGdnEpsilonFp32Bits == 0x3586'37bdU);
static_assert(kernels::sm87_target_aot_gdn_input_spec(
                  kernels::Sm87TargetAotGdnInputRole::kConvWeight)
                      .byte_count == 81'920U);
static_assert(kernels::sm87_target_aot_gdn_input_spec(
                  kernels::Sm87TargetAotGdnInputRole::kL2Epsilon)
                      .required_scalar_bits == 0x3586'37bdU);
static_assert(kernels::sm87_target_aot_gdn_input_spec(
                  kernels::Sm87TargetAotGdnInputRole::kNormWeight)
                      .element_count == 128U);

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

void test_owner_bijection(TestContext& test,
                          const kernels::Sm87TargetAotGdnPlan& plan) {
  std::array<std::uint8_t, kernels::kSm87TargetAotGdnValueHeads> heads{};
  std::vector<std::uint8_t> conv_channels(plan.total_conv_channels, 0U);
  std::size_t state_bytes = 0U;
  std::size_t history_bytes = 0U;
  for (std::size_t owner = 0U; owner < plan.owner_ctas; ++owner) {
    const auto task = kernels::sm87_target_aot_gdn_owner_task(plan, owner);
    test.expect(task.valid && task.owner_cta == owner &&
                    task.qk_group == owner &&
                    task.first_value_head ==
                        owner * plan.value_heads_per_owner &&
                    task.value_head_count == plan.value_heads_per_owner &&
                    task.state_byte_offset ==
                        owner * plan.state_bytes_per_owner &&
                    task.state_bytes == plan.state_bytes_per_owner,
                "each owner maps one independent QK/state chain group");
    for (std::size_t head = task.first_value_head;
         head < task.first_value_head + task.value_head_count; ++head) {
      test.expect(head < heads.size() && heads[head] == 0U,
                  "value-head ownership is disjoint");
      if (head < heads.size()) {
        heads[head] = 1U;
      }
    }
    for (const auto& span :
         std::array<kernels::Sm87TargetAotGdnConvSpan, 3U>{
             task.q_conv, task.k_conv, task.v_conv}) {
      test.expect(span.channel_count != 0U &&
                      span.channel_begin + span.channel_count <=
                          conv_channels.size() &&
                      span.history_byte_offset ==
                          span.channel_begin * plan.conv_history *
                              kernels::kSm87TargetAotGdnBf16Bytes &&
                      span.history_bytes ==
                          span.channel_count * plan.conv_history *
                              kernels::kSm87TargetAotGdnBf16Bytes,
                  "owner retains canonical Q/K/V history spans");
      for (std::size_t channel = span.channel_begin;
           channel < span.channel_begin + span.channel_count; ++channel) {
        test.expect(conv_channels[channel] == 0U,
                    "conv-channel ownership is disjoint");
        conv_channels[channel] = 1U;
      }
    }
    state_bytes += task.state_bytes;
    history_bytes += task.conv_history_bytes;
  }
  for (const std::uint8_t seen : heads) {
    test.expect(seen == 1U, "all 48 value heads have one owner");
  }
  for (const std::uint8_t seen : conv_channels) {
    test.expect(seen == 1U, "all canonical conv channels have one owner");
  }
  test.expect(state_bytes == plan.total_state_bytes &&
                  history_bytes == plan.total_conv_history_bytes,
              "owner totals cover exact state/history payloads");
  test.expect(!kernels::sm87_target_aot_gdn_owner_task(plan, plan.owner_ctas)
                   .valid,
              "owner mapping fails closed past CTA 15");
}

void test_ordered_c16_schedule(TestContext& test,
                               const kernels::Sm87TargetAotGdnPlan& plan) {
  std::vector<std::uint8_t> token_seen(plan.token_count, 0U);
  std::size_t expected_c16 = 0U;
  for (std::size_t macro = 0U; macro < plan.preparation_c64_macros;
       ++macro) {
    const auto preparation =
        kernels::sm87_target_aot_gdn_preparation_task(plan, macro);
    test.expect(preparation.valid && !preparation.independent_kernel,
                "C64 is a host grouping inside one layer-long kernel");
    for (std::size_t ordinal = 0U;
         ordinal < preparation.ordered_c16_blocks; ++ordinal) {
      const auto c16 =
          kernels::sm87_target_aot_gdn_c16_task(plan, macro, ordinal);
      test.expect(c16.valid && c16.global_c16_index == expected_c16 &&
                      c16.token_begin ==
                          expected_c16 *
                              kernels::kSm87TargetAotGdnExactRecurrenceTokens &&
                      c16.token_count == 16U &&
                      c16.preparation_slot == expected_c16 % 2U &&
                      c16.same_cta_collective_producer &&
                      c16.same_cta_collective_consumer &&
                      c16.cancel_safe_point_after &&
                      !c16.independent_kernel,
                  "each owner advances an ordered private C16 epoch");
      test.expect(c16.has_predecessor == (expected_c16 != 0U),
                  "C16 recurrence predecessor is exact");
      test.expect(c16.has_prior_same_slot == (expected_c16 >= 2U),
                  "two shared slots retain their private reuse epoch");
      for (std::size_t token = c16.token_begin;
           token < c16.token_begin + c16.token_count; ++token) {
        test.expect(token_seen[token] == 0U,
                    "C16 scheduling never duplicates a token");
        token_seen[token] = 1U;
      }
      ++expected_c16;
    }
    test.expect(!kernels::sm87_target_aot_gdn_c16_task(
                     plan, macro, preparation.ordered_c16_blocks)
                     .valid,
                "terminal C64 macro fails closed past its C16 children");
  }
  test.expect(expected_c16 == plan.exact_c16_blocks,
              "all exact C16 blocks are scheduled once");
  for (const std::uint8_t seen : token_seen) {
    test.expect(seen == 1U, "all target tokens are covered exactly once");
  }
}

[[nodiscard]] kernels::Sm87TargetAotGdnTensorSpanBinding make_span(
    const kernels::Sm87TargetAotGdnTensorSpec& spec,
    const kernels::Sm87TargetAotGdnSpanLifetime lifetime,
    const std::uint64_t span_identity,
    const std::uint64_t lifetime_identity) {
  return {spec.scalar_type, spec.layout, lifetime, spec.element_count,
          spec.byte_count, span_identity, lifetime_identity};
}

[[nodiscard]] kernels::Sm87TargetAotGdnInputBinding make_input_binding(
    const kernels::Sm87TargetAotGdnInputRole role,
    const std::uint64_t identity_base) {
  const auto spec = kernels::sm87_target_aot_gdn_input_spec(role);
  return {spec.role,
          spec.lifetime,
          spec.scalar_type,
          spec.layout,
          spec.element_count,
          spec.byte_count,
          spec.required_scalar_bits,
          identity_base,
          identity_base + 1U,
          identity_base + 2U,
          identity_base + 3U,
          spec.mutable_in_place};
}

[[nodiscard]] kernels::Sm87TargetAotGdnProducerBinding make_producer(
    const kernels::Sm87TargetAotGdnProducerRole role,
    const std::size_t token_count, const std::uint64_t identity_base) {
  return {role,
          identity_base,
          identity_base + 1U,
          make_span(kernels::sm87_target_aot_gdn_producer_spec(role,
                                                               token_count),
                    kernels::Sm87TargetAotGdnSpanLifetime::
                        kProducerReadOnlyUntilKernelCompletion,
                    identity_base + 2U, identity_base + 3U)};
}

[[nodiscard]] kernels::Sm87TargetAotGdnBinding make_complete_binding(
    const kernels::Sm87TargetAotGdnPlan& plan) {
  kernels::Sm87TargetAotGdnBinding binding;
  binding.capacity_bucket = plan.capacity_bucket;
  binding.topology = plan.topology;
  binding.recurrence = plan.recurrence;
  binding.tactic = plan.tactic;
  binding.conv_history_layout = plan.conv_history_layout;
  binding.numerical_contract = plan.numerical_contract;
  binding.conv_numerical_contract = plan.conv_numerical_contract;
  binding.completion_contract = plan.completion_contract;
  binding.gdn_layer_ordinal = kernels::kSm87TargetAotGdnLayers - 1U;
  binding.first_position = 0U;
  binding.raw_qkvz_producer = make_producer(
      kernels::Sm87TargetAotGdnProducerRole::kRawQkvZ, plan.token_count,
      0x1'001U);
  binding.ab_producer = make_producer(
      kernels::Sm87TargetAotGdnProducerRole::kBf16Ab, plan.token_count,
      0x1'005U);
  for (std::size_t input = 0U; input < binding.inputs.size(); ++input) {
    binding.inputs[input] = make_input_binding(
        kernels::kSm87TargetAotGdnInputRoles[input],
        0x4'001U + input * 4U);
  }
  binding.plan_identity = 0x2'001U;
  binding.tactic_identity = 0x2'002U;
  binding.recurrence_identity = 0x2'003U;
  binding.numerical_contract_identity = 0x2'004U;
  binding.conv_numerical_contract_identity = 0x2'016U;
  binding.raw_publication_identity = 0x2'005U;
  binding.norm_gate_publication_identity = 0x2'006U;
  binding.final_conv_state_publication_identity = 0x2'007U;
  binding.c16_cancel_safe_point_identity = 0x2'008U;
  binding.kernel_completion_event_identity = 0x2'009U;
  binding.post_event_request_transaction_append_identity = 0x2'00aU;
  binding.launcher_identity = 0x2'00bU;
  binding.completion_contract_identity = 0x2'00cU;
  binding.request_transaction_identity = 0x2'00dU;
  binding.canonical_cold_zero_identity = 0x2'00eU;
  binding.cold_state_reset_epoch_identity = 0x2'00fU;
  binding.output = make_span(
      {kernels::Sm87TargetAotGdnScalarType::kBf16,
       kernels::Sm87TargetAotGdnTensorLayout::kPromptMajorOutput_T6144,
       plan.token_count * kernels::kSm87TargetAotGdnOutputChannels,
       plan.token_count * kernels::kSm87TargetAotGdnOutputChannels *
           kernels::kSm87TargetAotGdnBf16Bytes},
      kernels::Sm87TargetAotGdnSpanLifetime::
          kLayerIntermediateUntilOutputProjection,
      0x2'010U, 0x2'011U);
  binding.final_conv_history = make_span(
      {kernels::Sm87TargetAotGdnScalarType::kBf16,
       kernels::Sm87TargetAotGdnTensorLayout::kConvHistory_C10240H3,
       kernels::kSm87TargetAotGdnConvHistoryElements,
       kernels::kSm87TargetAotGdnTotalConvHistoryBytes},
      kernels::Sm87TargetAotGdnSpanLifetime::
          kRequestTransactionUnpublishedUntilCommit,
      0x2'012U, 0x2'013U);
  binding.final_recurrent_state = make_span(
      {kernels::Sm87TargetAotGdnScalarType::kBf16,
       kernels::Sm87TargetAotGdnTensorLayout::kRecurrentState_H48D128D128,
       kernels::kSm87TargetAotGdnRecurrentStateElements,
       kernels::kSm87TargetAotGdnTotalStateBytes},
      kernels::Sm87TargetAotGdnSpanLifetime::
          kRequestTransactionUnpublishedUntilCommit,
      0x2'014U, 0x2'015U);
  for (std::size_t owner = 0U;
       owner < binding.final_bf16_state_identities.size(); ++owner) {
    binding.final_bf16_state_identities[owner] = 0x3'001U + owner;
  }
  return binding;
}

[[nodiscard]] kernels::Sm87TargetAotGdnReceipt make_complete_receipt(
    const kernels::Sm87TargetAotGdnPlan& plan,
    const kernels::Sm87TargetAotGdnBinding& binding) {
  kernels::Sm87TargetAotGdnReceipt receipt;
  receipt.capacity_bucket = binding.capacity_bucket;
  receipt.topology = binding.topology;
  receipt.recurrence = binding.recurrence;
  receipt.tactic = binding.tactic;
  receipt.conv_history_layout = binding.conv_history_layout;
  receipt.numerical_contract = binding.numerical_contract;
  receipt.conv_numerical_contract = binding.conv_numerical_contract;
  receipt.completion_contract = binding.completion_contract;
  receipt.gdn_layer_ordinal = binding.gdn_layer_ordinal;
  receipt.first_position = binding.first_position;
  receipt.token_count = plan.token_count;
  receipt.raw_qkvz_producer = binding.raw_qkvz_producer;
  receipt.ab_producer = binding.ab_producer;
  receipt.inputs = binding.inputs;
  receipt.plan_identity = binding.plan_identity;
  receipt.tactic_identity = binding.tactic_identity;
  receipt.recurrence_identity = binding.recurrence_identity;
  receipt.numerical_contract_identity = binding.numerical_contract_identity;
  receipt.conv_numerical_contract_identity =
      binding.conv_numerical_contract_identity;
  receipt.launcher_identity = binding.launcher_identity;
  for (std::size_t owner = 0U; owner < receipt.owners.size(); ++owner) {
    auto& owner_receipt = receipt.owners[owner];
    owner_receipt.completed_preparation_macros =
        plan.preparation_c64_macros;
    owner_receipt.completed_c16_blocks = plan.exact_c16_blocks;
    owner_receipt.completed_tokens = plan.token_count;
    owner_receipt.completed_cancel_safe_points = plan.exact_c16_blocks;
    owner_receipt.completed_conv_channels = plan.conv_channels_per_owner;
    owner_receipt.staged_conv_history_values =
        plan.conv_channels_per_owner * plan.conv_history;
    owner_receipt.staged_gdn_state_chains = plan.value_heads_per_owner;
    owner_receipt.staged_bf16_state_identity =
        binding.final_bf16_state_identities[owner];
  }
  receipt.completed_output_rows = plan.token_count;
  receipt.completed_value_heads = plan.value_head_state_chains;
  receipt.completed_conv_channels = plan.total_conv_channels;
  receipt.staged_conv_history_values =
      plan.total_conv_channels * plan.conv_history;
  receipt.staged_final_gdn_state_chains = plan.value_head_state_chains;
  receipt.completed_owner_count = plan.required_owner_receipts;
  receipt.request_transaction_append_count = 1U;
  receipt.appended_owner_state_count = plan.required_owner_receipts;
  receipt.raw_bf16_output_identity = binding.raw_publication_identity;
  receipt.rmsnorm_silu_z_output_identity =
      binding.norm_gate_publication_identity;
  receipt.final_conv_state_publication_identity =
      binding.final_conv_state_publication_identity;
  receipt.c16_cancel_safe_point_identity =
      binding.c16_cancel_safe_point_identity;
  receipt.kernel_completion_event_identity =
      binding.kernel_completion_event_identity;
  receipt.post_event_request_transaction_append_identity =
      binding.post_event_request_transaction_append_identity;
  receipt.completion_contract_identity = binding.completion_contract_identity;
  receipt.request_transaction_identity = binding.request_transaction_identity;
  receipt.canonical_cold_zero_identity = binding.canonical_cold_zero_identity;
  receipt.cold_state_reset_epoch_identity =
      binding.cold_state_reset_epoch_identity;
  receipt.output = binding.output;
  receipt.final_conv_history = binding.final_conv_history;
  receipt.final_recurrent_state = binding.final_recurrent_state;
  receipt.cold_zero_register_initialization_applied = true;
  receipt.initial_state_history_device_read_observed = false;
  receipt.numerical_contract_applied = true;
  receipt.raw_qkvz_partition_contract_applied = true;
  receipt.conv_silu_fp32_applied = true;
  receipt.conv_output_bf16_rne_before_qkv_use = true;
  receipt.z_bit_exact_conv_bypass_applied = true;
  receipt.per_token_bf16_state_rounding = true;
  receipt.output_used_pre_round_fp32_update = true;
  receipt.raw_bf16_published = true;
  receipt.rmsnorm_complete = true;
  receipt.silu_z_gate_complete = true;
  receipt.independent_owner_completion = true;
  receipt.cooperative_launch_observed = false;
  receipt.cross_cta_barrier_observed = false;
  receipt.in_kernel_commit_observed = false;
  receipt.staged_spans_unpublished_at_kernel_completion = true;
  receipt.kernel_completion_event_recorded = true;
  receipt.kernel_event_after_all_owner_receipts = true;
  receipt.request_transaction_append_after_kernel_event = true;
  receipt.transaction_spans_were_prebound_at_request_admission = true;
  receipt.stream_ordered_ready_receipt_recorded = true;
  receipt.host_callback_observed = false;
  receipt.host_synchronization_observed = false;
  receipt.transaction_spans_appended_unpublished = true;
  receipt.pointer_publication_without_copy = true;
  receipt.output_projection_visibility_after_kernel_event = true;
  receipt.partial_span_publication_observed = false;
  receipt.output_reclaimable_after_output_projection = true;
  receipt.final_conv_history_appended_unpublished = true;
  receipt.final_recurrent_state_appended_unpublished = true;
  receipt.cancellation_observed = false;
  return receipt;
}

void test_typed_binding_and_identity_namespace(TestContext& test) {
  const auto binding = make_complete_binding(kP40);
  test.expect(binding.identity_namespace_valid() && binding.valid(kP40),
              "all 70 typed identities are nonzero and pairwise unique");

  auto changed = binding;
  changed.first_position = 1U;
  test.expect(!changed.valid(kP40),
              "v1 rejects continuation and any nonzero first position");
  changed = binding;
  changed.raw_qkvz_producer.tensor.element_count -= 1U;
  test.expect(!changed.valid(kP40),
              "raw QKVZ [T,16384] element count is exact");
  changed = binding;
  changed.ab_producer.tensor.layout =
      kernels::Sm87TargetAotGdnTensorLayout::kPromptMajorRawQkvZ_T16384;
  test.expect(!changed.valid(kP40), "A/B [T,2,48] layout is exact");
  changed = binding;
  changed.output.byte_count -= 2U;
  test.expect(!changed.valid(kP40),
              "output [T,6144] BF16 byte count is exact");
  changed = binding;
  changed.final_conv_history.lifetime =
      kernels::Sm87TargetAotGdnSpanLifetime::
          kProducerReadOnlyUntilKernelCompletion;
  test.expect(!changed.valid(kP40),
              "final history stays in an unpublished transaction span");
  changed = binding;
  changed.final_recurrent_state.layout =
      kernels::Sm87TargetAotGdnTensorLayout::kConvHistory_C10240H3;
  test.expect(!changed.valid(kP40),
              "final recurrent-state layout cannot alias conv history");

  const std::size_t l2 = kernels::sm87_target_aot_gdn_input_index(
      kernels::Sm87TargetAotGdnInputRole::kL2Epsilon);
  changed = binding;
  changed.inputs[l2].scalar_bits ^= 1U;
  test.expect(!changed.valid(kP40), "L2 epsilon exact FP32 bits are bound");
  const std::size_t norm_epsilon = kernels::sm87_target_aot_gdn_input_index(
      kernels::Sm87TargetAotGdnInputRole::kNormEpsilon);
  changed = binding;
  changed.inputs[norm_epsilon].scalar_type =
      kernels::Sm87TargetAotGdnScalarType::kBf16;
  test.expect(!changed.valid(kP40), "norm epsilon cannot become BF16");

  using IdentityMember =
      std::uint64_t kernels::Sm87TargetAotGdnBinding::*;
  constexpr std::array<IdentityMember, 16U> kScalarIdentities{{
      &kernels::Sm87TargetAotGdnBinding::plan_identity,
      &kernels::Sm87TargetAotGdnBinding::tactic_identity,
      &kernels::Sm87TargetAotGdnBinding::recurrence_identity,
      &kernels::Sm87TargetAotGdnBinding::numerical_contract_identity,
      &kernels::Sm87TargetAotGdnBinding::conv_numerical_contract_identity,
      &kernels::Sm87TargetAotGdnBinding::raw_publication_identity,
      &kernels::Sm87TargetAotGdnBinding::norm_gate_publication_identity,
      &kernels::Sm87TargetAotGdnBinding::final_conv_state_publication_identity,
      &kernels::Sm87TargetAotGdnBinding::c16_cancel_safe_point_identity,
      &kernels::Sm87TargetAotGdnBinding::kernel_completion_event_identity,
      &kernels::Sm87TargetAotGdnBinding::
          post_event_request_transaction_append_identity,
      &kernels::Sm87TargetAotGdnBinding::launcher_identity,
      &kernels::Sm87TargetAotGdnBinding::completion_contract_identity,
      &kernels::Sm87TargetAotGdnBinding::request_transaction_identity,
      &kernels::Sm87TargetAotGdnBinding::canonical_cold_zero_identity,
      &kernels::Sm87TargetAotGdnBinding::cold_state_reset_epoch_identity,
  }};
  for (const IdentityMember identity : kScalarIdentities) {
    changed = binding;
    changed.*identity = binding.raw_qkvz_producer.producer_identity;
    test.expect(!changed.valid(kP40),
                "each scalar contract identity rejects cross-domain aliasing");
  }
  for (std::size_t input = 0U; input < binding.inputs.size(); ++input) {
    changed = binding;
    changed.inputs[input].physical_span_identity = binding.plan_identity;
    test.expect(!changed.valid(kP40),
                "each typed model input span rejects aliasing");
  }
  changed = binding;
  changed.output.physical_span_identity = binding.plan_identity;
  test.expect(!changed.valid(kP40), "transaction output span rejects aliasing");
  changed = binding;
  changed.raw_qkvz_producer.tensor.lifetime_identity = binding.plan_identity;
  test.expect(!changed.valid(kP40), "producer lifetime rejects aliasing");
  changed = binding;
  changed.final_bf16_state_identities[3U] = binding.plan_identity;
  test.expect(!changed.valid(kP40), "owner staged-state ID rejects aliasing");
}

void test_transaction_receipt_fails_closed(TestContext& test) {
  const auto binding = make_complete_binding(kP40);
  const auto receipt = make_complete_receipt(kP40, binding);
  test.expect(receipt.complete(kP40, binding),
              "ordinary completion event then request append is complete");

  auto changed = receipt;
  changed.owners[7U].completed_tokens -= 1U;
  test.expect(!changed.complete(kP40, binding),
              "one incomplete owner invalidates kernel completion");
  changed = receipt;
  changed.completed_owner_count -= 1U;
  test.expect(!changed.complete(kP40, binding),
              "kernel event requires all 16 owner receipts");
  changed = receipt;
  changed.request_transaction_append_count += 1U;
  test.expect(!changed.complete(kP40, binding),
              "exactly one host append stages all owners in the request");
  changed = receipt;
  changed.appended_owner_state_count -= 1U;
  test.expect(!changed.complete(kP40, binding),
              "single request append still covers all 16 owner states");
  changed = receipt;
  changed.kernel_event_after_all_owner_receipts = false;
  test.expect(!changed.complete(kP40, binding),
              "kernel event cannot precede an owner completion");
  changed = receipt;
  changed.request_transaction_append_after_kernel_event = false;
  test.expect(!changed.complete(kP40, binding),
              "ready receipt cannot precede kernel completion event");
  changed = receipt;
  changed.transaction_spans_were_prebound_at_request_admission = false;
  test.expect(!changed.complete(kP40, binding),
              "state/history spans must be bound before GPU execution");
  changed = receipt;
  changed.host_callback_observed = true;
  test.expect(!changed.complete(kP40, binding),
              "per-layer append cannot require a CPU callback");
  changed = receipt;
  changed.host_synchronization_observed = true;
  test.expect(!changed.complete(kP40, binding),
              "per-layer append cannot synchronize the host");
  changed = receipt;
  changed.staged_spans_unpublished_at_kernel_completion = false;
  test.expect(!changed.complete(kP40, binding),
              "kernel completion alone cannot expose transaction spans");
  changed = receipt;
  changed.pointer_publication_without_copy = false;
  test.expect(!changed.complete(kP40, binding),
              "v1 publication is pointer-only with no state copy");
  changed = receipt;
  changed.output_projection_visibility_after_kernel_event = false;
  test.expect(!changed.complete(kP40, binding),
              "O projection cannot read output before kernel completion");
  changed = receipt;
  changed.partial_span_publication_observed = true;
  test.expect(!changed.complete(kP40, binding),
              "state/history/output can never be partially published");
  changed = receipt;
  changed.cancellation_observed = true;
  test.expect(!changed.complete(kP40, binding),
              "cancelled transaction cannot publish success");
  changed = receipt;
  changed.cross_cta_barrier_observed = true;
  test.expect(!changed.complete(kP40, binding),
              "receipt rejects any cross-CTA barrier");
  changed = receipt;
  changed.cooperative_launch_observed = true;
  test.expect(!changed.complete(kP40, binding),
              "receipt rejects cooperative launch");
  changed = receipt;
  changed.in_kernel_commit_observed = true;
  test.expect(!changed.complete(kP40, binding),
              "receipt rejects owner-0 or any in-kernel commit");
  changed = receipt;
  changed.initial_state_history_device_read_observed = true;
  test.expect(!changed.complete(kP40, binding),
              "cold path rejects initial state/history DRAM reads");
  changed = receipt;
  changed.cold_zero_register_initialization_applied = false;
  test.expect(!changed.complete(kP40, binding),
              "cold reset must initialize owner registers to zero");
  changed = receipt;
  changed.post_event_request_transaction_append_identity += 1U;
  test.expect(!changed.complete(kP40, binding),
              "post-event request-transaction append identity is exact");
  changed = receipt;
  changed.kernel_completion_event_identity += 1U;
  test.expect(!changed.complete(kP40, binding),
              "ordinary kernel completion event identity is exact");
  changed = receipt;
  changed.output.element_count -= 1U;
  test.expect(!changed.complete(kP40, binding),
              "receipt echoes exact transaction output span");
  changed = receipt;
  changed.numerical_contract_applied = false;
  test.expect(!changed.complete(kP40, binding),
              "receipt cannot omit intended recurrence operation order");
  changed = receipt;
  changed.conv_silu_fp32_applied = false;
  test.expect(!changed.complete(kP40, binding),
              "convolution SiLU is part of the exact numerical boundary");
  changed = receipt;
  changed.conv_output_bf16_rne_before_qkv_use = false;
  test.expect(!changed.complete(kP40, binding),
              "Q/K/V consume the BF16-RNE convolution publication");
  changed = receipt;
  changed.final_recurrent_state_appended_unpublished = false;
  test.expect(!changed.complete(kP40, binding),
              "recurrent state cannot be omitted from atomic publication");
  changed = receipt;
  changed.forbidden_transform_hits = 1U;
  test.expect(!changed.complete(kP40, binding),
              "WY/KKT/SSD or another forbidden transform fails closed");
}

void test_plan_mutations_fail_closed(TestContext& test) {
  auto changed = kP40;
  changed.topology = kernels::Sm87TargetAotGdnTopology::
      kValueHead48OwnersRequiresResourceAudit;
  test.expect(!changed.valid(), "48-owner fallback is not selectable in v1");
  changed = kP40;
  changed.first_position = 1U;
  test.expect(!changed.valid(), "v1 is cold first_position=0 only");
  changed = kP40;
  changed.cross_cta_barriers_per_layer = 1U;
  test.expect(!changed.valid(), "any cross-CTA barrier changes the plan");
  changed = kP40;
  changed.cooperative_launch_required = true;
  test.expect(!changed.valid(), "cooperative launch is explicitly forbidden");
  changed = kP40;
  changed.in_kernel_commit = true;
  test.expect(!changed.valid(), "no owner CTA may publish a commit");
  changed = kP40;
  changed.kernel_completion_events_per_layer = 2U;
  test.expect(!changed.valid(), "one ordinary kernel completion event is exact");
  changed = kP40;
  changed.required_owner_receipts = 15U;
  test.expect(!changed.valid(), "request append requires all 16 owners");
  changed = kP40;
  changed.initial_recurrent_state_device_read = true;
  test.expect(!changed.valid(), "cold v1 cannot read initial recurrent state");
  changed = kP40;
  changed.output_projection_read_requires_kernel_completion = false;
  test.expect(!changed.valid(),
              "O-projection visibility is guarded by kernel completion");
  changed = kP40;
  changed.host_callback_required = true;
  test.expect(!changed.valid(),
              "stream-ordered state readiness cannot require CPU callbacks");
  changed = kP40;
  changed.cancelled_span_reclamation_without_publication = false;
  test.expect(!changed.valid(),
              "cancel must discard the entire unpublished transaction");
  changed = kP40;
  changed.continuation_supported = true;
  test.expect(!changed.valid(), "continuation requires a separately named plan");
  changed = kP40;
  changed.qk_l2_fp32_index_order_fma = false;
  test.expect(!changed.valid(), "numerical operation order is frozen");
  changed = kP40;
  changed.conv_silu_fp32_before_publication = false;
  test.expect(!changed.valid(), "convolution SiLU cannot be elided");
  changed = kP40;
  changed.raw_partitions[3U].channel_offset -= 1U;
  test.expect(!changed.valid(), "Z partition offset is exact and typed");
  changed = kP40;
  changed.preparation_slots = 3U;
  test.expect(!changed.valid(), "a third preparation slot changes the plan");
  changed = kP40;
  changed.kernel_launches_per_layer = 2U;
  test.expect(!changed.valid(), "layer-long tactic is one kernel");
  changed = kP40;
  changed.per_token_bf16_state_rounding = false;
  test.expect(!changed.valid(), "delayed state rounding fails closed");
  changed = kP40;
  changed.chunk_fp32_state_authoritative = true;
  test.expect(!changed.valid(), "FP32 chunk state is forbidden");
  changed = kP40;
  changed.resource_qualified = true;
  test.expect(!changed.valid(), "resource qualification needs CUDA evidence");
  changed = kP40;
  changed.production_dispatch_eligible = true;
  test.expect(!changed.valid(), "host plan cannot self-promote to production");
}

}  // namespace

int main() {
  TestContext test;
  for (const auto& plan : std::array<kernels::Sm87TargetAotGdnPlan, 3U>{
           kP40, kP60, kP130}) {
    test_owner_bijection(test, plan);
    test_ordered_c16_schedule(test, plan);
  }
  test_typed_binding_and_identity_namespace(test);
  test_transaction_receipt_fails_closed(test);
  test_plan_mutations_fail_closed(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "sm87 target AOT GDN plan checks passed\n";
  return 0;
}
