#include "q3x/kernels/sm87_target_aot_attention_plan.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

constexpr auto kP40 = kernels::sm87_target_aot_attention_plan(40'000U);
constexpr auto kP60 = kernels::sm87_target_aot_attention_plan(60'000U);
constexpr auto kP130 = kernels::sm87_target_aot_attention_plan(130'000U);

constexpr std::size_t buffer_index(
    const kernels::Sm87TargetAotAttentionBufferRole role) {
  for (std::size_t index = 0U;
       index < kernels::kSm87TargetAotAttentionBufferRoles.size(); ++index) {
    if (kernels::kSm87TargetAotAttentionBufferRoles[index] == role) {
      return index;
    }
  }
  return kernels::kSm87TargetAotAttentionBufferCount;
}

constexpr std::size_t kRawQGate = buffer_index(
    kernels::Sm87TargetAotAttentionBufferRole::kRawQGate);
constexpr std::size_t kRawK =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kRawK);
constexpr std::size_t kRawV =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kRawV);
constexpr std::size_t kProcessedQ =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kProcessedQ);
constexpr std::size_t kProcessedGate =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kProcessedGate);
constexpr std::size_t kProcessedK =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kProcessedK);
constexpr std::size_t kProcessedV =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kProcessedV);
constexpr std::size_t kQNormWeight =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kQNormWeight);
constexpr std::size_t kKNormWeight =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kKNormWeight);
constexpr std::size_t kRopeCos =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kRopeCos);
constexpr std::size_t kRopeSin =
    buffer_index(kernels::Sm87TargetAotAttentionBufferRole::kRopeSin);
constexpr std::size_t kBf16AttentionOutput = buffer_index(
    kernels::Sm87TargetAotAttentionBufferRole::kBf16AttentionOutput);
constexpr std::size_t kSigmoidGatedOutput = buffer_index(
    kernels::Sm87TargetAotAttentionBufferRole::kSigmoidGatedOutput);

static_assert(kP40.valid() && kP60.valid() && kP130.valid());
static_assert(kP40.flattened_query_rows == 240'000U &&
              kP40.query_tiles_per_kv_head == 1'875U &&
              kP40.total_ctas == 7'500U &&
              kP40.query_tail_rows == 128U && kP40.kv_tiles == 1'250U &&
              kP40.kv_tail_tokens == 32U);
static_assert(kP60.query_tiles_per_kv_head == 2'813U &&
              kP60.total_ctas == 11'252U &&
              kP60.query_tail_rows == 64U && kP60.kv_tiles == 1'875U);
static_assert(kP130.query_tiles_per_kv_head == 6'094U &&
              kP130.total_ctas == 24'376U &&
              kP130.query_tail_rows == 96U &&
              kP130.kv_tiles == 4'063U && kP130.kv_tail_tokens == 16U);
static_assert(kP40.dynamic_shared_bytes == 131'072U &&
              kP40.split_kv_workspace_bytes == 0U &&
              !kP40.cuda_implementation_present &&
              !kP40.static_resources_qualified &&
              !kP40.numerical_reduction_qualified &&
              !kP40.numerical_contract_qualified &&
              !kP40.production_dispatch_eligible);
static_assert(kP40.q_norm_weight_elements == 256U &&
              kP40.k_norm_weight_elements == 256U &&
              kP40.rms_epsilon_fp32_bits == 0x3586'37bdU);
static_assert(kP40.rotary_elements == 64U && kP40.rotary_pairs == 32U &&
              kP40.rope_passthrough_elements == 192U &&
              kP40.rope_table_elements_per_position == 32U &&
              kP40.rope_theta == 10'000'000ULL);
static_assert(kP40.attention_scale_fp32_bits == 0x3d80'0000U &&
              kP40.attention_scale_numerator == 1U &&
              kP40.attention_scale_denominator == 16U);
static_assert(
    kP40.numerical_execution.exp_contract ==
        kernels::Sm87TargetAotAttentionExpContract::
            kEx2ApproxF32AfterFp32Log2eMultiply &&
    kP40.numerical_execution.qk_traversal ==
        kernels::Sm87TargetAotAttentionQkTraversalContract::
            kMmaSyncM16N16K16DimensionAscendingScoreSubtilesAscending &&
    kP40.numerical_execution.probability_publication ==
        kernels::Sm87TargetAotAttentionProbabilityContract::
            kExpF32ToBf16RneBeforeDenominatorAndPv &&
    kP40.numerical_execution.denominator_update ==
        kernels::Sm87TargetAotAttentionDenominatorContract::
            kRescalePriorThenRegister0145ThenXor1Xor2 &&
    kP40.numerical_execution.pv_traversal ==
        kernels::Sm87TargetAotAttentionPvTraversalContract::
            kMmaSyncBf16ProbabilityBf16VToFp32OutputDimensionAscending &&
    kP40.numerical_execution.output_publication ==
        kernels::Sm87TargetAotAttentionOutputContract::
            kFp32ReciprocalThenNumeratorMultiplyThenBf16Rne &&
    kP40.numerical_execution.sigmoid_publication ==
        kernels::Sm87TargetAotAttentionSigmoidContract::
            kStableSignBranchEx2ApproxF32TimesBf16AttentionThenBf16Rne);
static_assert(
    kP40.numerical_execution.log2e_fp32_bits == 0x3fb8'aa3bU &&
    kP40.numerical_execution.mma_m == 16U &&
    kP40.numerical_execution.mma_n == 16U &&
    kP40.numerical_execution.mma_k == 16U &&
    kP40.numerical_execution.qk_head_dimension_tiles == 16U &&
    kP40.numerical_execution.kv_stage_score_subtiles == 2U &&
    kP40.numerical_execution.pv_output_dimension_tiles == 16U &&
    kP40.numerical_execution.denominator_register_offsets[0U] == 0U &&
    kP40.numerical_execution.denominator_register_offsets[1U] == 1U &&
    kP40.numerical_execution.denominator_register_offsets[2U] == 4U &&
    kP40.numerical_execution.denominator_register_offsets[3U] == 5U &&
    kP40.numerical_execution.denominator_shuffle_xor_masks[0U] == 1U &&
    kP40.numerical_execution.denominator_shuffle_xor_masks[1U] == 2U);
static_assert(
    kP40.numerical_execution.attention_scale_after_qk_accumulation &&
    kP40.numerical_execution
        .prior_denominator_rescaled_before_probability_add &&
    kP40.numerical_execution.prior_output_rescaled_before_pv_mma &&
    kP40.numerical_execution.probability_bf16_reused_by_denominator &&
    kP40.numerical_execution.probability_bf16_reused_by_pv_mma &&
    kP40.numerical_execution.bf16_attention_reused_by_sigmoid_epilogue);
static_assert(
    (kP40.policy &
     kernels::kSm87TargetAotAttentionFinitePrecisionExecutionFrozen) != 0U &&
    (kP40.policy &
     kernels::kSm87TargetAotAttentionStableSigmoidEx2Bf16Rne) != 0U);
static_assert(
    kP40.preprocess_numerical.rms_reduction ==
        kernels::Sm87TargetAotAttentionRmsReductionContract::
            kPromptWide128FmafDAndD128ThenShared64Pair32Pair96Shuffle16To1 &&
    kP40.preprocess_numerical.inverse_rms ==
        kernels::Sm87TargetAotAttentionRsqrtContract::
            kRsqrtfFp32SumDiv256PlusFp32Epsilon &&
    kP40.preprocess_numerical.norm_publication ==
        kernels::Sm87TargetAotAttentionNormPublicationContract::
            kBf16WeightPlusOneValueTimesInverseThenGammaBf16Rne &&
    kP40.preprocess_numerical.rope_fma ==
        kernels::Sm87TargetAotAttentionRopeFmaContract::
            kNeoxSeparateSinProductThenCosFmaBf16Rne &&
    kP40.preprocess_numerical.rope_mapping ==
        kernels::Sm87TargetAotAttentionRopeMappingContract::
            kRotateD0To31WithDPlus32Tail64To255Passthrough);
static_assert(
    kP40.preprocess_numerical.threads == 128U &&
    kP40.preprocess_numerical.head_dimension == 256U &&
    kP40.preprocess_numerical.thread_dimension_offsets[0U] == 0U &&
    kP40.preprocess_numerical.thread_dimension_offsets[1U] == 128U &&
    kP40.preprocess_numerical.shared_tree_offsets[0U] == 0U &&
    kP40.preprocess_numerical.shared_tree_offsets[1U] == 64U &&
    kP40.preprocess_numerical.shared_tree_offsets[2U] == 32U &&
    kP40.preprocess_numerical.shared_tree_offsets[3U] == 96U &&
    kP40.preprocess_numerical.shuffle_down_strides[0U] == 16U &&
    kP40.preprocess_numerical.shuffle_down_strides[1U] == 8U &&
    kP40.preprocess_numerical.shuffle_down_strides[2U] == 4U &&
    kP40.preprocess_numerical.shuffle_down_strides[3U] == 2U &&
    kP40.preprocess_numerical.shuffle_down_strides[4U] == 1U);
static_assert(
    kP40.preprocess_numerical.rope_pair_offsets[0U] == 0U &&
    kP40.preprocess_numerical.rope_pair_offsets[1U] == 32U &&
    kP40.preprocess_numerical.rope_pair_count == 32U &&
    kP40.preprocess_numerical.rope_tail_begin == 64U &&
    kP40.preprocess_numerical.rope_tail_end == 256U &&
    kP40.preprocess_numerical.square_uses_fmaf_with_positive_zero &&
    kP40.preprocess_numerical.pair_add_low_square_before_high_square &&
    kP40.preprocess_numerical
        .epsilon_added_after_divide_by_head_dimension &&
    kP40.preprocess_numerical
        .norm_weight_decoded_from_bf16_then_fp32_plus_one &&
    kP40.preprocess_numerical
        .norm_multiplies_value_by_inverse_before_gamma &&
    kP40.preprocess_numerical
        .normalized_qk_published_bf16_rne_before_rope &&
    kP40.preprocess_numerical.rope_consumes_published_bf16_qk &&
    kP40.preprocess_numerical.rope_sine_product_rounded_before_fma &&
    kP40.preprocess_numerical.rope_output_published_bf16_rne &&
    kP40.preprocess_numerical.rope_tail_is_bit_exact_normalized_bf16);
static_assert(kP40.gate_bit_exact_split_copy &&
              kP40.gate_bypasses_qk_norm && kP40.gate_bypasses_rope &&
              kP40.v_bit_exact_passthrough &&
              kP40.ordered_kv_transaction_stage &&
              kP40.cold_first_position_zero_only);
static_assert(kRawQGate < kernels::kSm87TargetAotAttentionBufferCount &&
              kRopeSin < kernels::kSm87TargetAotAttentionBufferCount);
static_assert(kP40.buffers[kRawQGate].layout ==
                  kernels::Sm87TargetAotAttentionBufferLayout::
                      kRawQGateT24x2xD256PerHeadInterleaved &&
              kP40.buffers[kRawQGate].elements == 491'520'000U &&
              kP40.buffers[kRawQGate].bytes == 983'040'000U);
static_assert(kP40.buffers[kRawK].elements == 40'960'000U &&
              kP40.buffers[kRawV].elements == 40'960'000U &&
              kP40.buffers[kProcessedQ].elements == 245'760'000U &&
              kP40.buffers[kProcessedGate].elements == 245'760'000U);
static_assert(kP40.buffers[kProcessedK].layout ==
                  kernels::Sm87TargetAotAttentionBufferLayout::
                      kProcessedKeyNhdT4xD256 &&
              kP40.buffers[kProcessedV].layout ==
                  kernels::Sm87TargetAotAttentionBufferLayout::
                      kProcessedValueNhdT4xD256 &&
              kP40.buffers[kProcessedK].lifetime ==
                  kernels::Sm87TargetAotAttentionBufferLifetime::
                      kRequestTransactionUnpublishedUntilCommit &&
              kP40.buffers[kProcessedV].publication ==
                  kernels::Sm87TargetAotAttentionBufferPublication::
                      kOrderedKvTransactionStage);
static_assert(kP40.buffers[kQNormWeight].elements == 256U &&
              kP40.buffers[kKNormWeight].bytes == 512U &&
              kP40.buffers[kRopeCos].elements == 1'280'000U &&
              kP40.buffers[kRopeSin].bytes == 5'120'000U);
static_assert(
    kP40.buffers[kBf16AttentionOutput].layout ==
        kernels::Sm87TargetAotAttentionBufferLayout::
            kBf16AttentionT24xD256 &&
    kP40.buffers[kBf16AttentionOutput].publication ==
        kernels::Sm87TargetAotAttentionBufferPublication::
            kAttentionCoreOutput &&
    kP40.buffers[kSigmoidGatedOutput].layout ==
        kernels::Sm87TargetAotAttentionBufferLayout::
            kSigmoidGatedT24xD256 &&
    kP40.buffers[kSigmoidGatedOutput].lifetime ==
        kernels::Sm87TargetAotAttentionBufferLifetime::
            kLayerProjectionConsumer &&
    kP40.buffers[kSigmoidGatedOutput].elements == 245'760'000U);
static_assert(!kernels::sm87_target_aot_attention_plan(39'999U).valid());
static_assert(!kernels::Sm87TargetAotAttentionReceipt{}.complete(
    kP40, kernels::Sm87TargetAotAttentionBinding{}));

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

kernels::Sm87TargetAotAttentionBinding make_binding(
    const kernels::Sm87TargetAotAttentionPlan& plan) {
  kernels::Sm87TargetAotAttentionBinding binding;
  binding.capacity_bucket = plan.capacity_bucket;
  binding.topology = plan.topology;
  binding.layer_index = 3U;
  binding.first_position = 0U;
  binding.initial_kv_length = 0U;
  binding.cold_reset_epoch = 7U;
  binding.staged_kv_epoch = 7U;
  std::uint64_t identity = 100U;
  binding.plan_identity = identity++;
  binding.tactic_identity = identity++;
  binding.reduction_identity = identity++;
  binding.launcher_identity = identity++;
  binding.qk_preprocess_identity = identity++;
  binding.gate_split_copy_identity = identity++;
  binding.v_passthrough_identity = identity++;
  binding.attention_scale_identity = identity++;
  binding.position_reset_identity = identity++;
  binding.kv_cache_reset_epoch_identity = identity++;
  binding.ordered_kv_transaction_stage_identity = identity++;
  binding.core_publication_identity = identity++;
  binding.sigmoid_gate_identity = identity++;
  binding.core_completion_event_identity = identity++;
  for (std::size_t index = 0U; index < binding.buffers.size(); ++index) {
    binding.buffers[index].contract = plan.buffers[index];
    binding.buffers[index].publication_identity = identity++;
    binding.buffers[index].ready_event_identity = identity++;
  }
  return binding;
}

kernels::Sm87TargetAotAttentionReceipt make_receipt(
    const kernels::Sm87TargetAotAttentionPlan& plan,
    const kernels::Sm87TargetAotAttentionBinding& binding) {
  kernels::Sm87TargetAotAttentionReceipt receipt;
  receipt.binding = binding;
  receipt.token_count = plan.token_count;
  for (std::size_t index = 0U;
       index < receipt.completed_buffer_elements.size(); ++index) {
    receipt.completed_buffer_elements[index] =
        binding.buffers[index].contract.elements;
  }
  receipt.completed_q_rmsnorm_head_rows = plan.q_rmsnorm_head_rows;
  receipt.completed_k_rmsnorm_head_rows = plan.k_rmsnorm_head_rows;
  receipt.completed_q_rope_head_rows = plan.q_rope_head_rows;
  receipt.completed_k_rope_head_rows = plan.k_rope_head_rows;
  receipt.completed_processed_q_gate_head_rows =
      plan.processed_q_gate_head_rows;
  receipt.completed_processed_k_head_rows = plan.processed_k_head_rows;
  receipt.completed_published_v_head_rows = plan.published_v_head_rows;
  receipt.completed_bf16_attention_output_head_rows =
      plan.bf16_attention_output_head_rows;
  receipt.completed_sigmoid_gated_output_head_rows =
      plan.sigmoid_gated_output_head_rows;
  receipt.completed_position_rows = plan.position_rows;
  receipt.processed_kv_core_visible_begin = binding.first_position;
  receipt.processed_kv_core_visible_end =
      binding.first_position + plan.token_count;
  receipt.staged_kv_epoch = binding.staged_kv_epoch;
  receipt.completed_ctas = plan.total_ctas;
  receipt.completed_query_head_rows =
      plan.token_count * kernels::kSm87TargetAotAttentionQueryHeads;
  receipt.completed_token_rows = plan.token_count;
  receipt.core_kv_visible_end = plan.token_count;
  receipt.core_publication_identity = binding.core_publication_identity;
  receipt.sigmoid_gate_identity = binding.sigmoid_gate_identity;
  receipt.bf16_attention_output_publication_identity =
      binding.buffers[kBf16AttentionOutput].publication_identity;
  receipt.bf16_attention_output_ready_event_identity =
      binding.buffers[kBf16AttentionOutput].ready_event_identity;
  receipt.sigmoid_gated_output_publication_identity =
      binding.buffers[kSigmoidGatedOutput].publication_identity;
  receipt.sigmoid_gated_output_ready_event_identity =
      binding.buffers[kSigmoidGatedOutput].ready_event_identity;
  receipt.core_completion_event_identity =
      binding.core_completion_event_identity;
  receipt.position_reset_complete = true;
  receipt.qk_rmsnorm_complete = true;
  receipt.rope_position_complete = true;
  receipt.gate_bit_exact_split_copy_complete = true;
  receipt.gate_bypassed_qk_norm_and_rope = true;
  receipt.v_bit_exact_passthrough_complete = true;
  receipt.processed_q_gate_published_before_core = true;
  receipt.processed_kv_staged_core_visible_before_core = true;
  receipt.ordered_kv_transaction_stage_complete = true;
  receipt.bf16_attention_published = true;
  receipt.sigmoid_gate_after_bf16_ready = true;
  receipt.sigmoid_gate_applied = true;
  return receipt;
}

void test_cta_mapping(TestContext& test,
                      const kernels::Sm87TargetAotAttentionPlan& plan) {
  std::vector<std::uint8_t> visited(
      plan.token_count * kernels::kSm87TargetAotAttentionQueryHeads, 0U);
  for (std::size_t cta = 0U; cta < plan.total_ctas; ++cta) {
    const auto task = kernels::sm87_target_aot_attention_task(plan, cta);
    test.expect(
        task.valid &&
            task.role == kernels::Sm87TargetAotAttentionTaskRole::
                             kCausalQ128AgainstOrderedNhdKv &&
            task.raw_q_gate_layout ==
                kernels::Sm87TargetAotAttentionBufferLayout::
                    kRawQGateT24x2xD256PerHeadInterleaved &&
            task.processed_k_layout ==
                kernels::Sm87TargetAotAttentionBufferLayout::
                    kProcessedKeyNhdT4xD256 &&
            task.processed_v_layout ==
                kernels::Sm87TargetAotAttentionBufferLayout::
                    kProcessedValueNhdT4xD256 &&
            task.kv_head < kernels::kSm87TargetAotAttentionKvHeads &&
            task.query_tile < plan.query_tiles_per_kv_head &&
            task.flattened_query_end ==
                task.flattened_query_begin + task.flattened_query_rows &&
            task.flattened_query_rows > 0U &&
            task.flattened_query_rows <=
                kernels::kSm87TargetAotAttentionQueryRows &&
            task.kv_position_span_begin == 0U &&
            task.kv_position_span_end == plan.token_count,
        "every Attention CTA binds the canonical layouts and exact spans");
    for (std::size_t row_index = 0U;
         row_index < task.flattened_query_rows; ++row_index) {
      const auto row = kernels::sm87_target_aot_attention_query_row(
          plan, task, row_index);
      const std::size_t raw_head_offset =
          row.token * plan.raw_q_gate_token_stride_elements +
          row.query_head * 2U *
              kernels::kSm87TargetAotAttentionHeadDimension;
      const std::size_t raw_kv_offset =
          row.token * plan.kv_token_stride_elements +
          row.kv_head * kernels::kSm87TargetAotAttentionHeadDimension;
      test.expect(
          row.valid && row.token < plan.token_count &&
              row.kv_head == task.kv_head &&
              row.local_query_head <
                  kernels::kSm87TargetAotAttentionQueriesPerKv &&
              row.query_head ==
                  row.kv_head *
                          kernels::kSm87TargetAotAttentionQueriesPerKv +
                      row.local_query_head &&
              row.raw_query_element_offset == raw_head_offset &&
              row.raw_gate_element_offset ==
                  raw_head_offset +
                      kernels::kSm87TargetAotAttentionHeadDimension &&
              row.raw_k_element_offset == raw_kv_offset &&
              row.raw_v_element_offset == raw_kv_offset &&
              row.query_element_offset ==
                  row.token * plan.query_token_stride_elements +
                      row.query_head *
                          kernels::kSm87TargetAotAttentionHeadDimension &&
              row.gate_element_offset == row.query_element_offset &&
              row.processed_k_element_offset == raw_kv_offset &&
              row.processed_v_element_offset == raw_kv_offset &&
              row.absolute_position == row.token &&
              row.visible_kv_token_end == row.token + 1U &&
              row.gate_is_bit_exact_split_copy,
          "row mapping preserves interleaved raw Q/Gate and ordered NHD KV");
      if (row.valid && row.token < plan.token_count &&
          row.query_head < kernels::kSm87TargetAotAttentionQueryHeads) {
        const std::size_t canonical =
            row.token * kernels::kSm87TargetAotAttentionQueryHeads +
            row.query_head;
        test.expect(visited[canonical] == 0U,
                    "Attention row ownership never overlaps");
        visited[canonical] = 1U;
      }
    }
  }
  for (const std::uint8_t count : visited) {
    test.expect(count == 1U,
                "CTA map covers every token and query head exactly once");
  }
  test.expect(!kernels::sm87_target_aot_attention_task(plan, plan.total_ctas)
                   .valid,
              "Attention task map fails closed past its CTA grid");
}

void test_forged_tasks_fail_closed(TestContext& test) {
  const auto canonical = kernels::sm87_target_aot_attention_task(kP40, 0U);
  auto forged = canonical;
  forged.role = kernels::Sm87TargetAotAttentionTaskRole::kInvalid;
  test.expect(!kernels::sm87_target_aot_attention_query_row(kP40, forged, 0U)
                   .valid,
              "forged Attention task role fails closed");
  forged = canonical;
  forged.raw_q_gate_layout =
      kernels::Sm87TargetAotAttentionBufferLayout::kProcessedQT24xD256;
  test.expect(!kernels::sm87_target_aot_attention_query_row(kP40, forged, 0U)
                   .valid,
              "contiguous-Q fiction cannot replace raw per-head Q/Gate");
  forged = canonical;
  forged.processed_k_layout =
      kernels::Sm87TargetAotAttentionBufferLayout::
          kProcessedValueNhdT4xD256;
  forged.processed_v_layout =
      kernels::Sm87TargetAotAttentionBufferLayout::
          kProcessedKeyNhdT4xD256;
  test.expect(!kernels::sm87_target_aot_attention_query_row(kP40, forged, 0U)
                   .valid,
              "K/V publication-order layout swap fails closed");
  forged = canonical;
  forged.flattened_query_end += 1U;
  test.expect(!kernels::sm87_target_aot_attention_query_row(kP40, forged, 0U)
                   .valid,
              "forged flattened query span fails closed");
  forged = canonical;
  forged.kv_position_span_begin = 1U;
  test.expect(!kernels::sm87_target_aot_attention_query_row(kP40, forged, 0U)
                   .valid,
              "forged causal KV span fails closed");
}

void test_binding_and_receipt(TestContext& test) {
  const auto binding = make_binding(kP40);
  test.expect(binding.valid(kP40),
              "full-Attention layer accepts the complete typed binding");

  auto changed = binding;
  changed.first_position = 1U;
  test.expect(!changed.valid(kP40),
              "cold Attention binding rejects a nonzero first position");
  changed = binding;
  changed.staged_kv_epoch += 1U;
  test.expect(!changed.valid(kP40),
              "current KV epoch must equal the cold reset epoch");
  changed = binding;
  changed.initial_kv_length = 1U;
  test.expect(!changed.valid(kP40),
              "cold KV cache must start at zero visible length");
  changed = binding;
  changed.core_completion_event_identity = changed.tactic_identity;
  test.expect(!changed.valid(kP40),
              "forged core completion event cannot alias another semantic ID");
  changed = binding;
  changed.buffers[kRawQGate].contract.layout =
      kernels::Sm87TargetAotAttentionBufferLayout::kProcessedQT24xD256;
  test.expect(!changed.valid(kP40),
              "raw interleaved Q/Gate cannot be relabeled contiguous Q");
  changed = binding;
  changed.buffers[kRawQGate].contract.span_end_elements -= 1U;
  test.expect(!changed.valid(kP40),
              "short typed-buffer span fails closed");
  changed = binding;
  changed.buffers[kRawQGate].ready_event_identity =
      changed.buffers[kRawK].publication_identity;
  test.expect(!changed.valid(kP40),
              "forged ready event cannot alias a publication identity");
  changed = binding;
  std::swap(changed.buffers[kProcessedK], changed.buffers[kProcessedV]);
  test.expect(!changed.valid(kP40),
              "typed K/V buffer swap fails closed before core execution");
  changed = binding;
  changed.buffers[kProcessedK].contract.role =
      kernels::Sm87TargetAotAttentionBufferRole::kProcessedV;
  test.expect(!changed.valid(kP40),
              "forged buffer role fails closed");
  changed = binding;
  std::swap(changed.buffers[kBf16AttentionOutput],
            changed.buffers[kSigmoidGatedOutput]);
  test.expect(!changed.valid(kP40),
              "pre-gate and gated output spans cannot be swapped");

  const auto complete = make_receipt(kP40, binding);
  test.expect(complete.complete(kP40, binding),
              "receipt matches every typed buffer and core completion event");
  auto receipt = complete;
  receipt.core_completion_event_identity += 1U;
  test.expect(!receipt.complete(kP40, binding),
              "receipt cannot forge the bound core completion event");
  receipt = complete;
  receipt.binding.buffers[kProcessedK].ready_event_identity += 10'000U;
  test.expect(!receipt.complete(kP40, binding),
              "receipt cannot forge a typed-buffer ready event");
  receipt = complete;
  std::swap(receipt.binding.buffers[kProcessedK],
            receipt.binding.buffers[kProcessedV]);
  test.expect(!receipt.complete(kP40, binding),
              "receipt cannot swap ordered K/V publications");
  receipt = complete;
  receipt.completed_buffer_elements[kProcessedV] -= 1U;
  test.expect(!receipt.complete(kP40, binding),
              "partial typed-buffer publication fails closed");
  receipt = complete;
  receipt.staged_kv_epoch += 1U;
  test.expect(!receipt.complete(kP40, binding),
              "receipt KV epoch must match the current cold epoch");
  receipt = complete;
  receipt.sigmoid_gated_output_publication_identity =
      receipt.bf16_attention_output_publication_identity;
  test.expect(!receipt.complete(kP40, binding),
              "O-projection publication cannot point at pre-gate Attention");
  receipt = complete;
  receipt.completed_sigmoid_gated_output_head_rows -= 1U;
  test.expect(!receipt.complete(kP40, binding),
              "partial sigmoid-gated output publication fails closed");
  receipt = complete;
  receipt.bf16_attention_output_ready_event_identity += 1U;
  test.expect(!receipt.complete(kP40, binding),
              "sigmoid gate must consume the bound BF16-ready event");
  receipt = complete;
  receipt.sigmoid_gate_after_bf16_ready = false;
  test.expect(!receipt.complete(kP40, binding),
              "sigmoid gate cannot run before the BF16 core publication");
}

void test_plan_mutations_fail_closed(TestContext& test) {
  auto changed = kP40;
  changed.topology = kernels::Sm87TargetAotAttentionTopology::
      kQ256RequiresNewProducerConsumer;
  test.expect(!changed.valid(), "Q256 placeholder is not selectable");
  changed = kP40;
  changed.split_kv_workspace_bytes = 1U;
  test.expect(!changed.valid(), "split-KV workspace mutation fails closed");
  changed = kP40;
  changed.bf16_attention_round_before_sigmoid_gate = false;
  test.expect(!changed.valid(),
              "Attention/gate BF16 publication mutation fails closed");
  changed = kP40;
  changed.rms_epsilon_fp32_bits += 1U;
  test.expect(!changed.valid(), "centered RMSNorm epsilon is bit-bound");
  changed = kP40;
  changed.rotary_elements = 128U;
  changed.rotary_pairs = 64U;
  changed.rope_passthrough_elements = 128U;
  changed.rope_table_elements_per_position = 64U;
  test.expect(!changed.valid(),
              "false 64-pair/128-element RoPE contract is rejected");
  changed = kP40;
  changed.attention_scale_fp32_bits += 1U;
  test.expect(!changed.valid(), "Attention 1/16 FP32 scale is bit-bound");
  changed = kP40;
  changed.gate_bypasses_rope = false;
  test.expect(!changed.valid(), "Gate may not enter Q/K RoPE");
  changed = kP40;
  changed.v_bit_exact_passthrough = false;
  test.expect(!changed.valid(), "V passthrough cannot become arithmetic");
  changed = kP40;
  changed.kv_cache_layout =
      kernels::Sm87TargetAotAttentionKvCacheLayout::kInvalid;
  test.expect(!changed.valid(), "KV cache must retain the NHD contract");
  changed = kP40;
  changed.numerical_execution.exp_contract =
      kernels::Sm87TargetAotAttentionExpContract::kInvalid;
  test.expect(!changed.valid(), "online-softmax exp backend is bit-bound");
  changed = kP40;
  changed.numerical_execution.qk_traversal =
      kernels::Sm87TargetAotAttentionQkTraversalContract::kInvalid;
  test.expect(!changed.valid(), "QK mma.sync traversal is frozen");
  changed = kP40;
  changed.numerical_execution.probability_publication =
      kernels::Sm87TargetAotAttentionProbabilityContract::kInvalid;
  test.expect(!changed.valid(),
              "probability must publish BF16 RNE before both consumers");
  changed = kP40;
  changed.numerical_execution.denominator_register_offsets[1U] = 4U;
  changed.numerical_execution.denominator_register_offsets[2U] = 1U;
  test.expect(!changed.valid(),
              "denominator register accumulation order is frozen");
  changed = kP40;
  changed.numerical_execution.denominator_shuffle_xor_masks[0U] = 2U;
  changed.numerical_execution.denominator_shuffle_xor_masks[1U] = 1U;
  test.expect(!changed.valid(),
              "denominator lane-reduction order is frozen");
  changed = kP40;
  changed.numerical_execution.probability_bf16_reused_by_denominator = false;
  test.expect(!changed.valid(),
              "denominator cannot consume unrounded FP32 probability");
  changed = kP40;
  changed.numerical_execution.probability_bf16_reused_by_pv_mma = false;
  test.expect(!changed.valid(),
              "PV mma.sync must consume the published BF16 probability");
  changed = kP40;
  changed.numerical_execution.pv_traversal =
      kernels::Sm87TargetAotAttentionPvTraversalContract::kInvalid;
  test.expect(!changed.valid(), "PV mma.sync traversal is frozen");
  changed = kP40;
  changed.numerical_execution.output_publication =
      kernels::Sm87TargetAotAttentionOutputContract::kInvalid;
  test.expect(!changed.valid(),
              "normalized Attention output must publish BF16 RNE");
  changed = kP40;
  changed.numerical_execution.sigmoid_publication =
      kernels::Sm87TargetAotAttentionSigmoidContract::kInvalid;
  test.expect(!changed.valid(),
              "stable sigmoid backend and final BF16 RNE are frozen");
  changed = kP40;
  changed.numerical_execution.bf16_attention_reused_by_sigmoid_epilogue =
      false;
  test.expect(!changed.valid(),
              "sigmoid epilogue cannot consume the FP32 numerator directly");
  changed = kP40;
  changed.preprocess_numerical.rms_reduction =
      kernels::Sm87TargetAotAttentionRmsReductionContract::kInvalid;
  test.expect(!changed.valid(), "D256 centered-RMS reduction tree is frozen");
  changed = kP40;
  changed.preprocess_numerical.shared_tree_offsets[1U] = 32U;
  changed.preprocess_numerical.shared_tree_offsets[2U] = 64U;
  test.expect(!changed.valid(),
              "centered-RMS shared reduction reassociation fails closed");
  changed = kP40;
  changed.preprocess_numerical.shuffle_down_strides[0U] = 8U;
  changed.preprocess_numerical.shuffle_down_strides[1U] = 16U;
  test.expect(!changed.valid(),
              "centered-RMS warp reduction reassociation fails closed");
  changed = kP40;
  changed.preprocess_numerical.inverse_rms =
      kernels::Sm87TargetAotAttentionRsqrtContract::kInvalid;
  test.expect(!changed.valid(), "rsqrtf placement/backend is frozen");
  changed = kP40;
  changed.preprocess_numerical
      .norm_weight_decoded_from_bf16_then_fp32_plus_one = false;
  test.expect(!changed.valid(),
              "centered BF16 norm weight must receive FP32 plus one");
  changed = kP40;
  changed.preprocess_numerical
      .normalized_qk_published_bf16_rne_before_rope = false;
  test.expect(!changed.valid(),
              "Q/K norm must publish BF16 RNE before RoPE consumption");
  changed = kP40;
  changed.preprocess_numerical.rope_fma =
      kernels::Sm87TargetAotAttentionRopeFmaContract::kInvalid;
  test.expect(!changed.valid(), "RoPE FMA operand order is frozen");
  changed = kP40;
  changed.preprocess_numerical.rope_pair_offsets[1U] = 64U;
  test.expect(!changed.valid(), "NeoX D/D+32 pair mapping is frozen");
  changed = kP40;
  changed.preprocess_numerical.rope_sine_product_rounded_before_fma = false;
  test.expect(!changed.valid(),
              "RoPE sine product must round before the cosine FMA");
  changed = kP40;
  changed.preprocess_numerical.rope_tail_begin = 128U;
  test.expect(!changed.valid(), "RoPE passthrough tail begins at D64");
  changed = kP40;
  changed.preprocess_numerical.rope_tail_is_bit_exact_normalized_bf16 = false;
  test.expect(!changed.valid(),
              "RoPE D64..D255 tail must remain the normalized BF16 bits");
}

}  // namespace

int main() {
  TestContext test;
  test_cta_mapping(test, kP40);
  test_cta_mapping(test, kP60);
  test_cta_mapping(test, kP130);
  test_forged_tasks_fail_closed(test);
  test_binding_and_receipt(test);
  test_plan_mutations_fail_closed(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "sm87 target AOT Attention plan checks passed\n";
  return 0;
}
