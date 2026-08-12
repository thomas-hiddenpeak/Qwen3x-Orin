#include "q3x/runtime/sm87_aot_prefill_system_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

namespace kernels = q3x::kernels;
namespace runtime = q3x::runtime;

static_assert(runtime::kPrefillRequiredOperatorRoleCount == 17U);
static_assert(runtime::kSm87AotPrefillSystemProjectionDataflowCount == 6U);
static_assert(runtime::kSm87AotPrefillTypedResourceEdgeCount == 39U);
static_assert(runtime::kSm87AotPrefillTypedEventEdgeCount == 13U);
static_assert(static_cast<std::uint8_t>(
                  runtime::PrefillBindingRole::kNvfp4GateUp) == 0U);
static_assert(static_cast<std::uint8_t>(
                  runtime::PrefillBindingRole::kFinalHandoff) == 16U);
static_assert(!runtime::Sm87AotPrefillBf16AbPlan{}.valid());

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

void bind_schema_identities(runtime::Sm87AotPrefillSystemPlan& plan) {
  std::uint64_t next = 1U;
  const auto bind = [&next](std::uint64_t& identity) { identity = next++; };
  bind(plan.binary_identity);
  bind(plan.checkpoint_identity);
  bind(plan.platform_identity);
  bind(plan.numerical_contract_identity);
  bind(plan.request_memory_plan_identity);
  bind(plan.stream_plan_identity);
  bind(plan.handoff_plan_identity);
  for (std::size_t index = 0U; index < plan.execution_group_count; ++index) {
    bind(plan.execution_groups[index].tactic_identity);
    bind(plan.execution_groups[index].binding_identity);
  }
  auto& attention = plan.attention_dataflow;
  bind(attention.raw_q_gate_input_span_table_identity);
  bind(attention.raw_k_input_span_table_identity);
  bind(attention.raw_v_input_span_table_identity);
  bind(attention.raw_q_gate_ready_event_table_identity);
  bind(attention.raw_k_ready_event_table_identity);
  bind(attention.raw_v_ready_event_table_identity);
  bind(attention.q_norm_weight_table_identity);
  bind(attention.k_norm_weight_table_identity);
  bind(attention.rope_position_contract_identity);
  bind(attention.processed_q_gate_publication_table_identity);
  bind(attention.processed_k_publication_table_identity);
  bind(attention.processed_v_publication_table_identity);
  bind(attention.preparation_ready_event_table_identity);
  bind(attention.attention_q_gate_input_span_table_identity);
  bind(attention.attention_k_input_span_table_identity);
  bind(attention.attention_v_input_span_table_identity);
  bind(attention.attention_core_input_ready_event_table_identity);
  bind(attention.kv_cache_arena_table_identity);
  bind(attention.kv_cache_lifetime_contract_identity);
  bind(attention.staged_kv_transaction_publication_table_identity);
  bind(attention.staged_kv_transaction_ready_event_table_identity);
  bind(attention.position_rope_epoch_publication_table_identity);
  bind(attention.position_rope_epoch_ready_event_table_identity);
  bind(attention.pre_gate_bf16_output_span_table_identity);
  bind(attention.pre_gate_bf16_publication_table_identity);
  bind(attention.pre_gate_bf16_lifetime_contract_identity);
  bind(attention.pre_gate_bf16_completion_event_table_identity);
  bind(attention.gated_output_span_table_identity);
  bind(attention.gated_output_publication_table_identity);
  bind(attention.gated_output_lifetime_contract_identity);
  bind(attention.gated_output_completion_event_table_identity);

  auto& gdn = plan.gdn_dataflow;
  bind(gdn.raw_qkv_input_span_table_identity);
  bind(gdn.raw_z_input_span_table_identity);
  bind(gdn.raw_qkvz_ready_event_table_identity);
  bind(gdn.a_input_span_table_identity);
  bind(gdn.a_ready_event_table_identity);
  bind(gdn.b_input_span_table_identity);
  bind(gdn.b_ready_event_table_identity);
  bind(gdn.conv_weight_table_identity);
  bind(gdn.a_log_table_identity);
  bind(gdn.dt_bias_table_identity);
  bind(gdn.output_norm_weight_table_identity);
  bind(gdn.raw_bf16_publication_table_identity);
  bind(gdn.norm_silu_z_publication_table_identity);
  bind(gdn.output_span_table_identity);
  bind(gdn.output_publication_table_identity);
  bind(gdn.final_conv_history_publication_table_identity);
  bind(gdn.final_recurrent_state_publication_table_identity);
  bind(gdn.ordinary_kernel_completion_event_table_identity);
  bind(gdn.post_kernel_stream_ordered_state_ready_receipt_identity);
  bind(gdn.request_transaction_lifetime_contract_identity);
  bind(gdn.reset_zero_epoch_table_identity);

  for (auto& projection : plan.projection_dataflows) {
    bind(projection.input_span_table_identity);
    if (projection.role ==
        kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput) {
      bind(projection.input_ready_event_table_identity);
    }
    if (projection.role ==
        kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
      bind(projection.aggregate_completion_event_table_identity);
    }
    for (std::size_t index = 0U; index < projection.partition_count; ++index) {
      auto& partition = projection.partitions[index];
      bind(partition.source_packed_asset_identity);
      bind(partition.independent_scale_identity);
      bind(partition.raw_bits_contract_identity);
      bind(partition.output_span_identity);
      bind(partition.publication_identity);
      bind(partition.completion_event_identity);
    }
  }

  auto& model = plan.model_dataflow;
  bind(model.token_id_span_identity);
  bind(model.embedding_token_input_span_identity);
  bind(model.embedding_weight_table_identity);
  bind(model.embedding_publication_table_identity);
  bind(model.layer_input_span_table_identity);
  bind(model.input_norm_input_span_table_identity);
  bind(model.input_norm_weight_table_identity);
  bind(model.input_norm_publication_table_identity);
  bind(model.input_norm_epsilon_contract_identity);
  bind(model.gdn_o_residual_bypass_input_span_identity);
  bind(model.full_o_residual_bypass_input_span_identity);
  bind(model.gdn_o_projection_residual_input_span_identity);
  bind(model.full_o_projection_residual_input_span_identity);
  bind(model.gdn_attention_residual_publication_identity);
  bind(model.full_attention_residual_publication_identity);
  bind(model.gdn_post_norm_input_span_identity);
  bind(model.full_post_norm_input_span_identity);
  bind(model.gdn_down_residual_bypass_input_span_identity);
  bind(model.full_down_residual_bypass_input_span_identity);
  bind(model.post_attention_norm_weight_table_identity);
  bind(model.post_attention_norm_publication_table_identity);
  bind(model.post_attention_norm_epsilon_contract_identity);
  bind(model.silu_gate_input_span_table_identity);
  bind(model.silu_up_input_span_table_identity);
  bind(model.silu_times_up_publication_table_identity);
  bind(model.down_branch_residual_input_span_identity);
  bind(model.next_layer_input_publication_table_identity);
  bind(model.residual_lifetime_contract_identity);
  bind(model.bf16_ab_input_span_table_identity);
  bind(model.bf16_a_weight_table_identity);
  bind(model.bf16_b_weight_table_identity);
  bind(model.bf16_a_publication_table_identity);
  bind(model.bf16_b_publication_table_identity);
  bind(model.bf16_a_completion_event_table_identity);
  bind(model.bf16_b_completion_event_table_identity);
  bind(model.bf16_a_rne_publication_event_identity);
  bind(model.bf16_b_rne_publication_event_identity);
  bind(model.final_norm_weight_identity);
  bind(model.final_norm_input_span_identity);
  bind(model.final_norm_publication_identity);
  bind(model.final_norm_completion_event_identity);
  bind(model.final_norm_epsilon_contract_identity);
  bind(model.final_handoff_identity);

  auto& transaction = plan.request_transaction;
  bind(transaction.staged_kv_input_table_identity);
  bind(transaction.conv_history_input_table_identity);
  bind(transaction.recurrent_state_input_table_identity);
  bind(transaction.position_rope_epoch_input_identity);
  bind(transaction.final_hidden_input_identity);
  bind(transaction.staged_kv_ready_input_event_identity);
  bind(transaction.gdn_state_ready_input_event_identity);
  bind(transaction.position_rope_epoch_ready_input_event_identity);
  bind(transaction.final_hidden_ready_input_event_identity);
  bind(transaction.transaction_lifetime_contract_identity);
  bind(transaction.prefill_state_committed_receipt_schema_identity);
  bind(transaction.prefill_state_committed_receipt_identity);
  bind(transaction.decode_handoff_receipt_input_identity);

  // Bind the exact producer/consumer storage equivalence classes. The
  // canonical plan owns six dataflow bindings because the identical O shape
  // has separate GDN and full-Attention input families.
  auto& gate = plan.projection_dataflows[0U].partitions[0U];
  auto& up = plan.projection_dataflows[0U].partitions[1U];
  auto& down = plan.projection_dataflows[1U].partitions[0U];
  auto& gdn_qkv = plan.projection_dataflows[2U].partitions[0U];
  auto& gdn_z = plan.projection_dataflows[2U].partitions[1U];
  auto& full_q_gate = plan.projection_dataflows[3U].partitions[0U];
  auto& full_k = plan.projection_dataflows[3U].partitions[1U];
  auto& full_v = plan.projection_dataflows[3U].partitions[2U];
  auto& gdn_o = plan.projection_dataflows[4U].partitions[0U];
  auto& full_o = plan.projection_dataflows[5U].partitions[0U];

  attention.raw_q_gate_input_span_table_identity =
      full_q_gate.publication_identity;
  attention.raw_k_input_span_table_identity = full_k.publication_identity;
  attention.raw_v_input_span_table_identity = full_v.publication_identity;
  attention.attention_q_gate_input_span_table_identity =
      attention.processed_q_gate_publication_table_identity;
  attention.attention_k_input_span_table_identity =
      attention.processed_k_publication_table_identity;
  attention.attention_v_input_span_table_identity =
      attention.processed_v_publication_table_identity;

  gdn.raw_qkv_input_span_table_identity = gdn_qkv.publication_identity;
  gdn.raw_z_input_span_table_identity = gdn_z.publication_identity;
  gdn.a_input_span_table_identity = model.bf16_a_publication_table_identity;
  gdn.b_input_span_table_identity = model.bf16_b_publication_table_identity;

  model.embedding_token_input_span_identity = model.token_id_span_identity;
  const auto layer_residual_identity = model.layer_input_span_table_identity;
  model.embedding_publication_table_identity = layer_residual_identity;
  model.next_layer_input_publication_table_identity = layer_residual_identity;
  model.input_norm_input_span_table_identity = layer_residual_identity;
  model.gdn_o_residual_bypass_input_span_identity = layer_residual_identity;
  model.full_o_residual_bypass_input_span_identity = layer_residual_identity;
  model.final_norm_input_span_identity = layer_residual_identity;

  plan.projection_dataflows[2U].input_span_table_identity =
      model.input_norm_publication_table_identity;
  model.bf16_ab_input_span_table_identity =
      model.input_norm_publication_table_identity;
  plan.projection_dataflows[3U].input_span_table_identity =
      model.input_norm_publication_table_identity;
  model.silu_gate_input_span_table_identity = gate.publication_identity;
  model.silu_up_input_span_table_identity = up.publication_identity;
  plan.projection_dataflows[1U].input_span_table_identity =
      model.silu_times_up_publication_table_identity;
  plan.projection_dataflows[0U].input_span_table_identity =
      model.post_attention_norm_publication_table_identity;
  plan.projection_dataflows[4U].input_span_table_identity =
      gdn.output_publication_table_identity;
  plan.projection_dataflows[5U].input_span_table_identity =
      attention.gated_output_publication_table_identity;
  model.gdn_o_projection_residual_input_span_identity =
      gdn_o.publication_identity;
  model.full_o_projection_residual_input_span_identity =
      full_o.publication_identity;
  model.gdn_post_norm_input_span_identity =
      model.gdn_attention_residual_publication_identity;
  model.gdn_down_residual_bypass_input_span_identity =
      model.gdn_attention_residual_publication_identity;
  model.full_post_norm_input_span_identity =
      model.full_attention_residual_publication_identity;
  model.full_down_residual_bypass_input_span_identity =
      model.full_attention_residual_publication_identity;
  model.down_branch_residual_input_span_identity = down.publication_identity;

  transaction.staged_kv_input_table_identity =
      attention.staged_kv_transaction_publication_table_identity;
  transaction.conv_history_input_table_identity =
      gdn.final_conv_history_publication_table_identity;
  transaction.recurrent_state_input_table_identity =
      gdn.final_recurrent_state_publication_table_identity;
  transaction.position_rope_epoch_input_identity =
      attention.position_rope_epoch_publication_table_identity;
  transaction.final_hidden_input_identity =
      model.final_norm_publication_identity;
  transaction.staged_kv_ready_input_event_identity =
      attention.staged_kv_transaction_ready_event_table_identity;
  transaction.gdn_state_ready_input_event_identity =
      gdn.post_kernel_stream_ordered_state_ready_receipt_identity;
  transaction.position_rope_epoch_ready_input_event_identity =
      attention.position_rope_epoch_ready_event_table_identity;
  transaction.final_hidden_ready_input_event_identity =
      model.final_norm_completion_event_identity;
  transaction.decode_handoff_receipt_input_identity =
      transaction.prefill_state_committed_receipt_identity;
  gdn.request_transaction_lifetime_contract_identity =
      transaction.transaction_lifetime_contract_identity;

  // Event equality is the synchronization contract. Generic BF16 kernel
  // completion stays independent; GDN waits for the BF16-RNE publications.
  gdn.raw_qkvz_ready_event_table_identity =
      plan.projection_dataflows[2U].aggregate_completion_event_table_identity;
  attention.raw_q_gate_ready_event_table_identity =
      full_q_gate.completion_event_identity;
  attention.raw_k_ready_event_table_identity = full_k.completion_event_identity;
  attention.raw_v_ready_event_table_identity = full_v.completion_event_identity;
  gdn.a_ready_event_table_identity = model.bf16_a_rne_publication_event_identity;
  gdn.b_ready_event_table_identity = model.bf16_b_rne_publication_event_identity;
  attention.attention_core_input_ready_event_table_identity =
      attention.preparation_ready_event_table_identity;
  plan.projection_dataflows[5U].input_ready_event_table_identity =
      attention.gated_output_completion_event_table_identity;
  plan.projection_dataflows[4U].input_ready_event_table_identity =
      gdn.ordinary_kernel_completion_event_table_identity;

  for (auto& edge : plan.typed_resource_edges) {
    edge = runtime::sm87_aot_prefill_expected_typed_resource_edge(
        plan, edge.resource);
  }
  for (auto& edge : plan.typed_event_edges) {
    edge = runtime::sm87_aot_prefill_expected_typed_event_edge(
        plan, edge.resource);
  }
}

[[nodiscard]] std::size_t staging_sum(
    const runtime::Sm87AotPrefillSystemCapacityPlan& plan) {
  std::size_t result = 0U;
  for (std::size_t index = 0U; index < plan.staging_span_count; ++index) {
    result += plan.staging_span_tokens[index];
  }
  return result;
}

[[nodiscard]] runtime::Sm87AotPrefillPhysicalExecutionGroup& group_for(
    runtime::Sm87AotPrefillSystemPlan& plan,
    const runtime::Sm87AotPrefillExecutionGroup group) {
  return plan.execution_groups[static_cast<std::size_t>(group)];
}

[[nodiscard]] const runtime::Sm87AotPrefillPhysicalExecutionGroup& group_for(
    const runtime::Sm87AotPrefillSystemPlan& plan,
    const runtime::Sm87AotPrefillExecutionGroup group) {
  return plan.execution_groups[static_cast<std::size_t>(group)];
}

[[nodiscard]] runtime::Sm87AotPrefillTypedResourceEdge* resource_edge_for(
    runtime::Sm87AotPrefillSystemPlan& plan,
    const runtime::Sm87AotPrefillTypedResource resource) {
  for (auto& edge : plan.typed_resource_edges) {
    if (edge.resource == resource) {
      return &edge;
    }
  }
  return nullptr;
}

[[nodiscard]] runtime::Sm87AotPrefillTypedEventEdge* event_edge_for(
    runtime::Sm87AotPrefillSystemPlan& plan,
    const runtime::Sm87AotPrefillTypedResource resource) {
  for (auto& edge : plan.typed_event_edges) {
    if (edge.resource == resource) {
      return &edge;
    }
  }
  return nullptr;
}

void test_capacity_and_constituent_design(TestContext& test) {
  struct Expected {
    kernels::Sm87TargetAotCapacityBucket bucket;
    std::size_t minimum;
    std::size_t witness;
    std::size_t request_capacity;
    std::size_t staging_count;
    std::uint64_t selected_bytes;
    std::uint64_t conservative_bytes;
    std::string_view plan_id;
  };
  constexpr Expected expected[] = {
      {kernels::Sm87TargetAotCapacityBucket::kP40, 1U, 40'000U, 40'001U,
       5U, 3'975'374'848ULL, 5'324'963'840ULL,
       runtime::kSm87AotPrefillSystemP40PlanId},
      {kernels::Sm87TargetAotCapacityBucket::kP60, 40'001U, 60'000U,
       60'001U, 8U, 5'496'014'848ULL, 7'052'323'840ULL,
       runtime::kSm87AotPrefillSystemP60PlanId},
      {kernels::Sm87TargetAotCapacityBucket::kP130, 60'001U, 130'000U,
       130'001U, 16U, 10'818'254'848ULL, 13'098'083'840ULL,
       runtime::kSm87AotPrefillSystemP130PlanId},
  };

  for (const Expected& item : expected) {
    auto plan =
        runtime::make_unbound_sm87_aot_prefill_system_plan(item.bucket);
    const auto validation =
        runtime::validate_sm87_aot_prefill_system_plan(plan);
    test.expect(plan.candidate_id ==
                        runtime::kSm87AotPrefillSystemCandidateId &&
                    plan.deployment_plan_id == item.plan_id,
                "each capacity owns an independent system-plan identity");
    test.expect(plan.capacity.prompt.minimum_prompt_tokens == item.minimum &&
                    plan.capacity.prompt.maximum_prompt_tokens ==
                        item.witness &&
                    plan.capacity.prompt.witness_prompt_tokens ==
                        item.witness &&
                    plan.capacity.prompt.request_capacity_tokens ==
                        item.request_capacity,
                "capacity range and exact witness are canonical");
    test.expect(plan.capacity.staging_span_count == item.staging_count &&
                    staging_sum(plan.capacity) == item.witness,
                "staging spans sum to the complete witness");
    test.expect(plan.capacity.selected_request_memory_bytes ==
                        item.selected_bytes &&
                    plan.capacity.conservative_request_memory_bytes ==
                        item.conservative_bytes,
                "both preliminary host memory values are retained");

    bool projections_valid = true;
    for (std::size_t index = 0U;
         index < runtime::kSm87AotPrefillSystemProjectionPlanCount; ++index) {
      projections_valid =
          projections_valid && plan.constituents.projections[index].valid() &&
          plan.constituents.projections[index].role ==
              runtime::kSm87AotPrefillProjectionRoles[index] &&
          plan.constituents.projections[index].token_count == item.witness;
    }
    test.expect(projections_valid && plan.constituents.bf16_ab.valid() &&
                    plan.constituents.bf16_ab.token_count == item.witness &&
                    plan.constituents.bf16_ab.tile_m == 64U &&
                    plan.constituents.bf16_ab.tile_n == 96U &&
                    plan.constituents.bf16_ab.tile_k == 64U &&
                    plan.constituents.bf16_ab.pipeline_stages == 2U &&
                    plan.constituents.bf16_ab.minimum_active_ctas_per_sm ==
                        2U &&
                    !plan.constituents.bf16_ab.static_resources_qualified &&
                    !plan.constituents.bf16_ab
                         .numerical_reduction_qualified &&
                    plan.constituents.attention.valid() &&
                    plan.constituents.attention.token_count == item.witness &&
                    plan.constituents.gdn.valid() &&
                    plan.constituents.gdn.token_count == item.witness,
                "all five projections, Attention, and GDN are embedded");
    test.expect(validation.capacity_contract_complete &&
                    validation.staging_design_complete &&
                    validation.preliminary_memory_design_complete &&
                    validation.constituent_design_complete &&
                    validation.layer_schedule_design_complete &&
                    validation.dependency_graph_complete &&
                    validation.logical_publication_design_complete &&
                    validation.physical_execution_design_complete &&
                    validation.declared_exact_intent_complete &&
                    validation.native_provider_design_complete &&
                    validation.aot_tactic_design_complete &&
                    validation.expected_physical_work_complete &&
                    validation.logical_physical_ownership_complete &&
                    !validation.typed_dataflow_bindings_complete &&
                    !validation.canonical_design_complete,
                "unbound template is structurally complete but unbound");
    test.expect(!validation.identity_schema_complete &&
                    !validation.physical_group_identities_complete &&
                    !validation.operator_bindings_complete &&
                    !validation.descriptor_schema_complete &&
                    !validation.descriptor_schema_available() &&
                    !validation.executable(),
                "canonical design does not imply a bound or executable route");
    test.expect(!runtime::sm87_aot_prefill_matches_exact_witness_design(
                    plan, item.witness, item.request_capacity),
                "an unbound template cannot satisfy exact-witness design");
    bind_schema_identities(plan);
    const auto bound_validation =
        runtime::validate_sm87_aot_prefill_system_plan(plan);
    test.expect(bound_validation.canonical_design_complete &&
                    bound_validation.typed_dataflow_bindings_complete &&
                    bound_validation.unified_identity_namespace_complete &&
                    runtime::sm87_aot_prefill_matches_exact_witness_design(
                        plan, item.witness, item.request_capacity) &&
                    !runtime::sm87_aot_prefill_matches_exact_witness_design(
                        plan, item.witness - 1U, item.request_capacity - 1U) &&
                    !runtime::sm87_aot_prefill_matches_exact_witness_design(
                        plan, item.witness, item.request_capacity - 1U),
                "design check accepts only the exact target witness");
  }
}

void test_staging_geometry(TestContext& test) {
  const auto p40 = runtime::sm87_aot_prefill_system_capacity_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  test.expect(p40.staging_span_tokens[0U] == 8'192U &&
                  p40.staging_span_tokens[1U] == 8'192U &&
                  p40.staging_span_tokens[2U] == 8'192U &&
                  p40.staging_span_tokens[3U] == 7'712U &&
                  p40.staging_span_tokens[4U] == 7'712U,
              "P40 is 3x8192 plus 2x7712");

  const auto p60 = runtime::sm87_aot_prefill_system_capacity_plan(
      kernels::Sm87TargetAotCapacityBucket::kP60);
  bool p60_exact = true;
  for (std::size_t index = 0U; index < 6U; ++index) {
    p60_exact = p60_exact && p60.staging_span_tokens[index] == 8'192U;
  }
  p60_exact = p60_exact && p60.staging_span_tokens[6U] == 5'424U &&
              p60.staging_span_tokens[7U] == 5'424U;
  test.expect(p60_exact, "P60 is 6x8192 plus 2x5424");

  const auto p130 = runtime::sm87_aot_prefill_system_capacity_plan(
      kernels::Sm87TargetAotCapacityBucket::kP130);
  bool p130_exact = true;
  for (std::size_t index = 0U; index < 14U; ++index) {
    p130_exact = p130_exact && p130.staging_span_tokens[index] == 8'192U;
  }
  p130_exact = p130_exact && p130.staging_span_tokens[14U] == 7'656U &&
               p130.staging_span_tokens[15U] == 7'656U;
  test.expect(p130_exact, "P130 is 14x8192 plus 2x7656");

  auto malformed = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  malformed.capacity.staging_span_tokens[4U] = 7'711U;
  const auto validation =
      runtime::validate_sm87_aot_prefill_system_plan(malformed);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kStagingGeometryMismatch) &&
                  !validation.staging_design_complete &&
                  !validation.canonical_design_complete,
              "one staging mutation invalidates the canonical design");
}

void test_canonical_launch_and_task_ledger(TestContext& test) {
  const auto plan = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  constexpr std::array<std::uint64_t,
                       runtime::kSm87AotPrefillExecutionGroupCount>
      launches{{1U, 64U, 48U, 48U, 48U, 48U,
                16U, 16U, 16U, 16U, 64U, 64U, 64U, 1U}};
  for (std::size_t index = 0U; index < launches.size(); ++index) {
    const auto group =
        static_cast<runtime::Sm87AotPrefillExecutionGroup>(index);
    const std::uint64_t scheduled_occurrences =
        runtime::sm87_aot_prefill_group_layer_occurrences(group);
    const bool singleton_outside_layers =
        group == runtime::Sm87AotPrefillExecutionGroup::kEmbedding ||
        group == runtime::Sm87AotPrefillExecutionGroup::kFinalHandoff;
    test.expect(plan.execution_groups[index].group ==
                        group &&
                    plan.execution_groups[index].work.expected_launches ==
                        launches[index] &&
                    (singleton_outside_layers ||
                     plan.execution_groups[index].work.expected_launches ==
                         scheduled_occurrences) &&
                    plan.execution_groups[index].work.canonical_counts_valid,
                "each physical launch count derives from the layer schedule");
  }

  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kLinearQkvZ)
              .work.projection_tile_tasks ==
          plan.constituents.projections[2U].mma_tile_tasks * 48ULL,
      "linear QKV/Z tasks derive from the fused projection constituent");
  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kLinearAb)
              .work.bf16_ab_pair_tile_tasks ==
          plan.constituents.bf16_ab.logical_pair_tasks * 48ULL,
      "BF16 A/B paired tasks derive from their independent constituent");
  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kLinearOResidual)
              .work.projection_tile_tasks ==
          plan.constituents.projections[4U].mma_tile_tasks * 48ULL,
      "linear O tasks derive from the output projection constituent");
  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kFullQkv)
              .work.projection_tile_tasks ==
          plan.constituents.projections[3U].mma_tile_tasks * 16ULL,
      "full QKV tasks derive from the fused projection constituent");
  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kFullOResidual)
              .work.projection_tile_tasks ==
          plan.constituents.projections[4U].mma_tile_tasks * 16ULL,
      "full O uses the same shape design with a distinct physical group");
  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kGateUp)
              .work.projection_tile_tasks ==
          plan.constituents.projections[0U].mma_tile_tasks * 64ULL &&
          group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kDownResidual)
                  .work.projection_tile_tasks ==
              plan.constituents.projections[1U].mma_tile_tasks * 64ULL,
      "GateUp paired partitions and Down retain independent MMA task domains");
  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::
                          kFullQkNormRopePublish)
                  .work.attention_q_rmsnorm_head_rows ==
              plan.constituents.attention.q_rmsnorm_head_rows * 16ULL &&
          group_for(plan, runtime::Sm87AotPrefillExecutionGroup::
                              kFullQkNormRopePublish)
                  .work.attention_k_rmsnorm_head_rows ==
              plan.constituents.attention.k_rmsnorm_head_rows * 16ULL &&
          group_for(plan, runtime::Sm87AotPrefillExecutionGroup::
                              kFullQkNormRopePublish)
                  .work.attention_position_rows ==
              plan.constituents.attention.position_rows * 16ULL,
      "full-Attention preprocess work binds Q/K norm, RoPE and positions");
  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kAttention)
              .work.attention_cta_tasks ==
          plan.constituents.attention.total_ctas * 16ULL,
      "Attention CTA work derives from the canonical 16-layer plan");
  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kGdn)
                  .work.gdn_preparation_owner_tasks ==
              plan.constituents.gdn.owner_ctas *
                  plan.constituents.gdn.preparation_c64_macros * 48ULL &&
          group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kGdn)
                  .work.gdn_exact_c16_owner_tasks ==
              plan.constituents.gdn.owner_ctas *
                  plan.constituents.gdn.exact_c16_blocks * 48ULL,
      "GDN preparation and exact recurrence have separate task ledgers");
}

void test_fused_logical_physical_ownership(TestContext& test) {
  const auto plan = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  const auto qkvz_mask =
      runtime::sm87_aot_prefill_role_bit(
          runtime::PrefillBindingRole::kLinearFp8Qkv) |
      runtime::sm87_aot_prefill_role_bit(
          runtime::PrefillBindingRole::kLinearFp8Z);
  const auto full_qkv_mask =
      runtime::sm87_aot_prefill_role_bit(runtime::PrefillBindingRole::kFullFp8Q) |
      runtime::sm87_aot_prefill_role_bit(runtime::PrefillBindingRole::kFullFp8K) |
      runtime::sm87_aot_prefill_role_bit(runtime::PrefillBindingRole::kFullFp8V);
  test.expect(group_for(plan,
                        runtime::Sm87AotPrefillExecutionGroup::kLinearQkvZ)
                      .logical_role_mask == qkvz_mask,
              "one physical linear projection serves QKV and Z roles");
  test.expect(group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kFullQkv)
                      .logical_role_mask == full_qkv_mask,
              "one physical full projection serves Q, K, and V roles");
  test.expect(
      (group_for(plan, runtime::Sm87AotPrefillExecutionGroup::
                           kFullQkNormRopePublish)
           .logical_role_mask &
       full_qkv_mask) == full_qkv_mask,
      "preprocess republishes processed Q/gate, K, and V");

  const auto& residual = plan.logical_publications[static_cast<std::size_t>(
      runtime::PrefillBindingRole::kResidual)];
  const auto& normalization =
      plan.logical_publications[static_cast<std::size_t>(
          runtime::PrefillBindingRole::kNormalization)];
  test.expect(residual.execution_group_mask ==
                      (runtime::sm87_aot_prefill_group_bit(
                           runtime::Sm87AotPrefillExecutionGroup::
                               kLinearOResidual) |
                       runtime::sm87_aot_prefill_group_bit(
                           runtime::Sm87AotPrefillExecutionGroup::
                               kFullOResidual) |
                       runtime::sm87_aot_prefill_group_bit(
                           runtime::Sm87AotPrefillExecutionGroup::
                               kDownResidual)) &&
                  residual.expected_logical_publications == 128U,
              "Residual publication is owned by three physical groups");
  test.expect(normalization.execution_group_mask ==
                      (runtime::sm87_aot_prefill_group_bit(
                           runtime::Sm87AotPrefillExecutionGroup::
                               kInputAndLayerNorm) |
                       runtime::sm87_aot_prefill_group_bit(
                           runtime::Sm87AotPrefillExecutionGroup::kGdn) |
                       runtime::sm87_aot_prefill_group_bit(
                           runtime::Sm87AotPrefillExecutionGroup::
                               kFullQkNormRopePublish) |
                       runtime::sm87_aot_prefill_group_bit(
                           runtime::Sm87AotPrefillExecutionGroup::
                               kPostAttentionLayerNorm) |
                       runtime::sm87_aot_prefill_group_bit(
                           runtime::Sm87AotPrefillExecutionGroup::
                               kFinalHandoff)) &&
                  normalization.expected_logical_publications == 209U,
              "Normalization publication is owned by all producing groups");
}

void test_layer_schedule_dependency_and_typed_dataflow(TestContext& test) {
  auto plan = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  std::size_t gdn_layers = 0U;
  std::size_t full_layers = 0U;
  bool schedule_exact = plan.layer_schedule_count == 64U;
  for (std::size_t layer = 0U; layer < plan.layer_schedule_count; ++layer) {
    const auto& entry = plan.layer_schedule[layer];
    const bool should_be_full = (layer + 1U) % 4U == 0U;
    schedule_exact = schedule_exact && entry.layer_index == layer &&
                     entry.input_source ==
                         (layer == 0U
                              ? runtime::Sm87AotPrefillLayerInputSource::
                                    kEmbedding
                              : runtime::Sm87AotPrefillLayerInputSource::
                                    kPriorLayerDownResidual) &&
                     entry.ordered_groups.front() == runtime::
                         Sm87AotPrefillExecutionGroup::kInputAndLayerNorm &&
                     entry.ordered_groups.back() == runtime::
                         Sm87AotPrefillExecutionGroup::kDownResidual;
    if (should_be_full) {
      ++full_layers;
      schedule_exact = schedule_exact &&
                       entry.kind ==
                           runtime::Sm87AotPrefillLayerKind::kFullAttention &&
                       entry.ordered_groups[2U] == runtime::
                           Sm87AotPrefillExecutionGroup::
                               kFullQkNormRopePublish;
    } else {
      ++gdn_layers;
      schedule_exact =
          schedule_exact &&
          entry.kind == runtime::Sm87AotPrefillLayerKind::kGdn;
    }
  }
  test.expect(schedule_exact && gdn_layers == 48U && full_layers == 16U,
              "all 64 layers preserve the exact GDN/full-Attention schedule");

  const auto input_bit = runtime::sm87_aot_prefill_group_bit(
      runtime::Sm87AotPrefillExecutionGroup::kInputAndLayerNorm);
  const auto qkvz_bit = runtime::sm87_aot_prefill_group_bit(
      runtime::Sm87AotPrefillExecutionGroup::kLinearQkvZ);
  const auto ab_bit = runtime::sm87_aot_prefill_group_bit(
      runtime::Sm87AotPrefillExecutionGroup::kLinearAb);
  test.expect(
      group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kLinearQkvZ)
                  .predecessor_group_mask == input_bit &&
          group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kLinearAb)
                  .predecessor_group_mask == input_bit &&
          group_for(plan, runtime::Sm87AotPrefillExecutionGroup::kGdn)
                  .predecessor_group_mask == (qkvz_bit | ab_bit),
      "QKVZ and A/B are independent after input norm and GDN waits for both");

  const auto& layer0_input = plan.layer_schedule[0U].stage_instances[0U];
  const auto& layer1_input = plan.layer_schedule[1U].stage_instances[0U];
  const auto& layer0_o = plan.layer_schedule[0U].stage_instances[4U];
  const auto& layer1_o = plan.layer_schedule[1U].stage_instances[4U];
  const auto& layer3_o = plan.layer_schedule[3U].stage_instances[4U];
  const auto& layer1_down = plan.layer_schedule[1U].stage_instances[7U];
  const auto& layer3_down = plan.layer_schedule[3U].stage_instances[7U];
  test.expect(
      layer0_input.predecessors[0U].source ==
              runtime::Sm87AotPrefillInstanceSource::kEmbedding &&
          layer1_input.predecessors[0U].source ==
              runtime::Sm87AotPrefillInstanceSource::kLayerStage &&
          layer1_input.predecessors[0U].layer_index == 0U &&
          layer1_input.predecessors[0U].stage_index == 7U &&
          layer0_o.predecessor_count == 2U &&
          layer0_o.predecessors[1U].source ==
              runtime::Sm87AotPrefillInstanceSource::kLayerStage &&
          layer0_o.predecessors[1U].edge_kind ==
              runtime::Sm87AotPrefillInstanceEdgeKind::kResidualBypass &&
          layer0_o.predecessors[1U].layer_index == 0U &&
          layer0_o.predecessors[1U].stage_index == 0U &&
          layer0_o.predecessors[1U].group == runtime::
              Sm87AotPrefillExecutionGroup::kInputAndLayerNorm &&
          layer1_o.predecessors[1U].source ==
              runtime::Sm87AotPrefillInstanceSource::kLayerStage &&
          layer1_o.predecessors[1U].layer_index == 1U &&
          layer1_o.predecessors[1U].stage_index == 0U &&
          layer1_o.predecessors[1U].group == runtime::
              Sm87AotPrefillExecutionGroup::kInputAndLayerNorm &&
          layer3_o.predecessors[1U].layer_index == 3U &&
          layer3_o.predecessors[1U].stage_index == 0U &&
          layer1_down.predecessors[1U].layer_index == 1U &&
          layer1_down.predecessors[1U].stage_index == 4U &&
          layer1_down.predecessors[1U].group ==
              runtime::Sm87AotPrefillExecutionGroup::kLinearOResidual &&
          layer3_down.predecessors[1U].layer_index == 3U &&
          layer3_down.predecessors[1U].stage_index == 4U &&
          layer3_down.predecessors[1U].group ==
              runtime::Sm87AotPrefillExecutionGroup::kFullOResidual,
      "instance DAG carries layer-input and post-Attention residual bypasses");

  bind_schema_identities(plan);
  auto validation = runtime::validate_sm87_aot_prefill_system_plan(plan);
  test.expect(validation.layer_schedule_design_complete &&
                  validation.dependency_graph_complete &&
                  validation.typed_dataflow_bindings_complete &&
                  validation.canonical_design_complete,
              "bound typed route closes schedule, DAG and operand schema");

  const auto& handoff = plan.final_handoff_instance;
  const auto* processed_q_gate_edge = resource_edge_for(
      plan,
      runtime::Sm87AotPrefillTypedResource::kProcessedQGateToAttention);
  const auto* processed_k_edge = resource_edge_for(
      plan, runtime::Sm87AotPrefillTypedResource::kProcessedKToAttention);
  const auto* processed_v_edge = resource_edge_for(
      plan, runtime::Sm87AotPrefillTypedResource::kProcessedVToAttention);
  const auto* bf16_a_event = event_edge_for(
      plan, runtime::Sm87AotPrefillTypedResource::kBf16AToGdn);
  const auto* staged_kv_event = event_edge_for(
      plan,
      runtime::Sm87AotPrefillTypedResource::kKvStateToRequestTransaction);
  test.expect(
      handoff.group == runtime::Sm87AotPrefillExecutionGroup::kFinalHandoff &&
          handoff.final_layer_predecessor.source ==
              runtime::Sm87AotPrefillInstanceSource::kLayerStage &&
          handoff.final_layer_predecessor.layer_index == 63U &&
          handoff.final_layer_predecessor.stage_index == 7U &&
          handoff.final_layer_predecessor.group ==
              runtime::Sm87AotPrefillExecutionGroup::kDownResidual &&
          handoff.final_layer_predecessor.edge_kind ==
              runtime::Sm87AotPrefillInstanceEdgeKind::kData,
      "final handoff's sole layer-stage predecessor is layer 63 stage 7 "
      "DownResidual");
  test.expect(
      processed_q_gate_edge != nullptr && processed_k_edge != nullptr &&
          processed_v_edge != nullptr &&
          processed_q_gate_edge->producer_group == runtime::
              Sm87AotPrefillExecutionGroup::kFullQkNormRopePublish &&
          processed_q_gate_edge->consumer_group ==
              runtime::Sm87AotPrefillExecutionGroup::kAttention &&
          processed_k_edge->consumer_group ==
              runtime::Sm87AotPrefillExecutionGroup::kAttention &&
          processed_v_edge->consumer_group ==
              runtime::Sm87AotPrefillExecutionGroup::kAttention,
      "processed Q/gate, K and V each have a typed preprocess-to-core edge");
  test.expect(
      plan.projection_dataflows[4U].family ==
              runtime::Sm87AotPrefillProjectionFamily::kGdnLayers &&
          plan.projection_dataflows[5U].family == runtime::
              Sm87AotPrefillProjectionFamily::kFullAttentionLayers &&
          plan.projection_dataflows[4U].input_span_table_identity ==
              plan.gdn_dataflow.output_publication_table_identity &&
          plan.projection_dataflows[5U].input_span_table_identity ==
              plan.attention_dataflow.gated_output_publication_table_identity &&
          plan.projection_dataflows[4U].input_span_table_identity !=
              plan.projection_dataflows[5U].input_span_table_identity,
      "same-shape O projection keeps distinct GDN and Attention inputs");
  test.expect(
      plan.request_transaction.kv_layer_publication_count == 16U &&
          plan.request_transaction.conv_history_publication_count == 48U &&
          plan.request_transaction.recurrent_state_publication_count == 48U &&
          plan.request_transaction.one_request_wide_commit &&
          plan.request_transaction
              .decode_visible_only_after_prefill_state_committed &&
          plan.request_transaction.cancellation_discards_all_unpublished &&
          plan.attention_dataflow
              .processed_kv_is_nhd_transaction_staged_unpublished &&
          !plan.gdn_dataflow.permits_per_owner_commit &&
          plan.gdn_dataflow
              .state_transaction_spans_prebound_at_request_admission &&
          !plan.gdn_dataflow.requires_cpu_callback_or_host_sync,
      "request state stays private until one final commit without host callbacks");
  test.expect(
      bf16_a_event != nullptr &&
          bf16_a_event->producer_completion_event_identity ==
              plan.model_dataflow.bf16_a_rne_publication_event_identity &&
          bf16_a_event->producer_completion_event_identity !=
              plan.model_dataflow.bf16_a_completion_event_table_identity &&
          staged_kv_event != nullptr &&
          staged_kv_event->producer_completion_event_identity ==
              plan.attention_dataflow
                  .staged_kv_transaction_ready_event_table_identity &&
          staged_kv_event->consumer_ready_event_identity ==
              plan.request_transaction.staged_kv_ready_input_event_identity,
      "typed ready edges bind BF16-RNE and staged transaction visibility");

  auto schedule_mutation = plan;
  schedule_mutation.layer_schedule[3U].ordered_groups[2U] =
      runtime::Sm87AotPrefillExecutionGroup::kAttention;
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(schedule_mutation);
  test.expect(!validation.layer_schedule_design_complete &&
                  !validation.canonical_design_complete,
              "a full-layer preprocess omission fails canonical validation");

  auto instance_mutation = plan;
  instance_mutation.layer_schedule[1U]
      .stage_instances[4U]
      .predecessors[1U]
      .stage_index = 6U;
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(instance_mutation);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kDependencyGraphMismatch) &&
                  !validation.dependency_graph_complete,
              "an instance residual predecessor cannot be rewired");

  auto final_handoff_mutation = plan;
  final_handoff_mutation.final_handoff_instance.final_layer_predecessor
      .layer_index = 62U;
  validation = runtime::validate_sm87_aot_prefill_system_plan(
      final_handoff_mutation);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kDependencyGraphMismatch) &&
                  !validation.dependency_graph_complete,
              "final handoff cannot bypass layer 63 DownResidual");

  auto dependency_mutation = plan;
  group_for(dependency_mutation,
            runtime::Sm87AotPrefillExecutionGroup::kGdn)
      .predecessor_group_mask = qkvz_bit;
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(dependency_mutation);
  test.expect(!validation.dependency_graph_complete &&
                  !validation.canonical_design_complete,
              "GDN cannot drop the independent B producer dependency");

  auto attention_mutation = plan;
  attention_mutation.attention_dataflow.processed_k_publication_table_identity =
      0U;
  validation = runtime::validate_sm87_aot_prefill_system_plan(
      attention_mutation);
  test.expect(!validation.typed_dataflow_bindings_complete &&
                  !validation.canonical_design_complete,
              "processed K publication is mandatory before Attention");

  auto attention_math_mutation = plan;
  attention_math_mutation.attention_dataflow.attention_scale_fp32_bits = 0U;
  validation = runtime::validate_sm87_aot_prefill_system_plan(
      attention_math_mutation);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "Attention score scale is the exact 1/sqrt(256) contract");

  auto residual_mutation = plan;
  residual_mutation.model_dataflow.residual_lifetime_contract_identity = 0U;
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(residual_mutation);
  test.expect(!validation.typed_dataflow_bindings_complete &&
                  !validation.canonical_design_complete,
              "residual bypass lifetime cannot be replaced by a group name");

  auto projection_mutation = plan;
  projection_mutation.projection_dataflows[0U]
      .partitions[0U]
      .independent_scale_identity = 0U;
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(projection_mutation);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "every quantized partition binds its independent scale");

  auto ab_mutation = plan;
  ab_mutation.model_dataflow.bf16_a_role =
      runtime::Sm87AotPrefillBf16AbRole::kInProjB;
  ab_mutation.model_dataflow.bf16_b_role =
      runtime::Sm87AotPrefillBf16AbRole::kInProjA;
  validation = runtime::validate_sm87_aot_prefill_system_plan(ab_mutation);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "A/B semantic roles cannot be swapped despite equal shapes");

  auto duplicate_resource = plan;
  duplicate_resource.typed_resource_edges[1U].resource =
      duplicate_resource.typed_resource_edges[0U].resource;
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(duplicate_resource);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "all 39 typed resources are mandatory and unique");

  auto resource_crosswire = plan;
  if (auto* edge = resource_edge_for(
          resource_crosswire,
          runtime::Sm87AotPrefillTypedResource::kProcessedKToAttention)) {
    edge->consumer_input_identity =
        resource_crosswire.attention_dataflow
            .attention_v_input_span_table_identity;
  }
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(resource_crosswire);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "a typed K edge cannot be cross-wired to V storage");

  auto duplicate_event = plan;
  duplicate_event.typed_event_edges[1U].resource =
      duplicate_event.typed_event_edges[0U].resource;
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(duplicate_event);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "all 13 typed event resources are mandatory and unique");

  auto bf16_event_crosswire = plan;
  if (auto* edge = event_edge_for(
          bf16_event_crosswire,
          runtime::Sm87AotPrefillTypedResource::kBf16AToGdn)) {
    edge->producer_completion_event_identity =
        bf16_event_crosswire.model_dataflow
            .bf16_a_completion_event_table_identity;
  }
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(bf16_event_crosswire);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "generic BF16 kernel completion cannot replace RNE publication");

  auto transaction_event_crosswire = plan;
  if (auto* edge = event_edge_for(
          transaction_event_crosswire,
          runtime::Sm87AotPrefillTypedResource::
              kKvStateToRequestTransaction)) {
    edge->producer_completion_event_identity =
        transaction_event_crosswire.attention_dataflow
            .position_rope_epoch_ready_event_table_identity;
  }
  validation = runtime::validate_sm87_aot_prefill_system_plan(
      transaction_event_crosswire);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "KV transaction readiness cannot borrow the position event");

  auto output_family_mutation = plan;
  output_family_mutation.projection_dataflows[5U].family =
      runtime::Sm87AotPrefillProjectionFamily::kGdnLayers;
  validation = runtime::validate_sm87_aot_prefill_system_plan(
      output_family_mutation);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "GDN and full-Attention O bindings cannot exchange families");

  auto transaction_mutation = plan;
  transaction_mutation.request_transaction.kv_layer_publication_count = 15U;
  transaction_mutation.request_transaction.one_request_wide_commit = false;
  transaction_mutation.request_transaction
      .decode_visible_only_after_prefill_state_committed = false;
  transaction_mutation.request_transaction
      .cancellation_discards_all_unpublished = false;
  validation = runtime::validate_sm87_aot_prefill_system_plan(
      transaction_mutation);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "state counts and one-commit Decode/cancellation rules are fixed");

  auto gdn_commit_mutation = plan;
  gdn_commit_mutation.gdn_dataflow.permits_per_owner_commit = true;
  gdn_commit_mutation.gdn_dataflow.requires_cpu_callback_or_host_sync = true;
  validation =
      runtime::validate_sm87_aot_prefill_system_plan(gdn_commit_mutation);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "GDN cannot publish per owner or require a host callback");

  auto attention_visibility_mutation = plan;
  attention_visibility_mutation.attention_dataflow
      .processed_kv_is_nhd_transaction_staged_unpublished = false;
  validation = runtime::validate_sm87_aot_prefill_system_plan(
      attention_visibility_mutation);
  test.expect(!validation.typed_dataflow_bindings_complete,
              "Attention KV remains staged and unpublished until final commit");
}

void test_missing_duplicate_and_mutated_designs(TestContext& test) {
  auto missing_role = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  --missing_role.logical_publication_count;
  const auto missing_role_validation =
      runtime::validate_sm87_aot_prefill_system_plan(missing_role);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  missing_role_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kMissingLogicalRole) &&
                  !missing_role_validation.logical_publication_design_complete,
              "all 17 logical roles are mandatory");

  auto duplicate_role = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  duplicate_role.logical_publications.back().role =
      runtime::PrefillBindingRole::kNvfp4GateUp;
  const auto duplicate_role_validation =
      runtime::validate_sm87_aot_prefill_system_plan(duplicate_role);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  duplicate_role_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kDuplicateLogicalRole) &&
                  runtime::has_sm87_aot_prefill_system_plan_issue(
                      duplicate_role_validation,
                      runtime::Sm87AotPrefillSystemPlanIssue::
                          kMissingLogicalRole),
              "a duplicate logical role cannot replace another role");

  auto duplicate_group = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  duplicate_group.execution_groups.back().group =
      runtime::Sm87AotPrefillExecutionGroup::kEmbedding;
  const auto duplicate_group_validation =
      runtime::validate_sm87_aot_prefill_system_plan(duplicate_group);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  duplicate_group_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kDuplicateExecutionGroup) &&
                  runtime::has_sm87_aot_prefill_system_plan_issue(
                      duplicate_group_validation,
                      runtime::Sm87AotPrefillSystemPlanIssue::
                          kMissingExecutionGroup),
              "all physical execution groups are mandatory and unique");

  auto ownership = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  ownership.logical_publications[static_cast<std::size_t>(
      runtime::PrefillBindingRole::kResidual)]
      .execution_group_mask = 0U;
  const auto ownership_validation =
      runtime::validate_sm87_aot_prefill_system_plan(ownership);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  ownership_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kLogicalPhysicalOwnershipMismatch) &&
                  !ownership_validation.logical_physical_ownership_complete,
              "logical and reverse physical ownership must agree");

  auto work = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  group_for(work, runtime::Sm87AotPrefillExecutionGroup::kAttention)
      .work.attention_cta_tasks += 1U;
  const auto work_validation =
      runtime::validate_sm87_aot_prefill_system_plan(work);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  work_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kExpectedPhysicalWorkMismatch) &&
                  !work_validation.expected_physical_work_complete,
              "physical work cannot be replaced by an arbitrary count");
}

void test_constituent_mutations_fail_closed(TestContext& test) {
  auto projection = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  projection.constituents.projections[0U].grid_n += 1U;
  const auto projection_validation =
      runtime::validate_sm87_aot_prefill_system_plan(projection);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  projection_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kConstituentDesignMismatch) &&
                  !projection_validation.constituent_design_complete,
              "a projection geometry mutation is rejected");

  auto bf16_ab = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  bf16_ab.constituents.bf16_ab.partitions[0U].role =
      runtime::Sm87AotPrefillBf16AbRole::kInProjB;
  const auto bf16_ab_validation =
      runtime::validate_sm87_aot_prefill_system_plan(bf16_ab);
  test.expect(!bf16_ab_validation.constituent_design_complete,
              "BF16 in_proj_a/in_proj_b partition order is canonical");

  auto attention = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  attention.constituents.attention.total_ctas += 1U;
  const auto attention_validation =
      runtime::validate_sm87_aot_prefill_system_plan(attention);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  attention_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kConstituentDesignMismatch) &&
                  !attention_validation.constituent_design_complete,
              "an Attention topology mutation is rejected");

  auto gdn = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  gdn.constituents.gdn.exact_c16_blocks += 1U;
  const auto gdn_validation =
      runtime::validate_sm87_aot_prefill_system_plan(gdn);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  gdn_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kConstituentDesignMismatch) &&
                  !gdn_validation.constituent_design_complete,
              "a GDN recurrence mutation is rejected");
}

void test_route_modes_identities_and_aliases(TestContext& test) {
  auto plan = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  bind_schema_identities(plan);

  auto non_native = plan;
  group_for(non_native, runtime::Sm87AotPrefillExecutionGroup::kAttention)
      .provider = runtime::PrefillOperatorProvider::kExternalRuntime;
  const auto non_native_validation =
      runtime::validate_sm87_aot_prefill_system_plan(non_native);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  non_native_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kNonNativeExecutionGroup) &&
                  !non_native_validation.native_provider_design_complete,
              "an external physical provider fails closed");

  auto not_aot = plan;
  group_for(not_aot, runtime::Sm87AotPrefillExecutionGroup::kGdn).tactic_mode =
      runtime::PrefillTacticMode::kJit;
  const auto not_aot_validation =
      runtime::validate_sm87_aot_prefill_system_plan(not_aot);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  not_aot_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kExecutionGroupNotAot) &&
                  !not_aot_validation.aot_tactic_design_complete,
              "request JIT cannot satisfy the AOT design");

  auto zero_group = plan;
  group_for(zero_group, runtime::Sm87AotPrefillExecutionGroup::kFullQkv)
      .binding_identity = 0U;
  const auto zero_group_validation =
      runtime::validate_sm87_aot_prefill_system_plan(zero_group);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  zero_group_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kExecutionGroupIdentityUnbound) &&
                  !zero_group_validation.physical_group_identities_complete,
              "a zero physical binding identity fails closed");

  auto zero_system = plan;
  zero_system.stream_plan_identity = 0U;
  const auto zero_system_validation =
      runtime::validate_sm87_aot_prefill_system_plan(zero_system);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  zero_system_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kSystemIdentityUnbound) &&
                  !zero_system_validation.identity_schema_complete,
              "a zero system identity fails closed");

  auto gate_down_alias = plan;
  group_for(gate_down_alias,
            runtime::Sm87AotPrefillExecutionGroup::kDownResidual)
      .tactic_identity =
      group_for(gate_down_alias,
                runtime::Sm87AotPrefillExecutionGroup::kGateUp)
          .tactic_identity;
  const auto gate_down_validation =
      runtime::validate_sm87_aot_prefill_system_plan(gate_down_alias);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  gate_down_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kGateDownTacticAlias) &&
                  !gate_down_validation.physical_group_identities_complete,
              "GateUp and Down require distinct tactic identities");

  auto binding_alias = plan;
  group_for(binding_alias,
            runtime::Sm87AotPrefillExecutionGroup::kInputAndLayerNorm)
      .binding_identity =
      group_for(binding_alias,
                runtime::Sm87AotPrefillExecutionGroup::kEmbedding)
          .binding_identity;
  const auto binding_alias_validation =
      runtime::validate_sm87_aot_prefill_system_plan(binding_alias);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  binding_alias_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kUnexpectedExecutionGroupIdentityAlias),
              "distinct physical owners cannot alias one binding identity");

  auto system_group_alias = plan;
  system_group_alias.binary_identity =
      group_for(system_group_alias,
                runtime::Sm87AotPrefillExecutionGroup::kAttention)
          .binding_identity;
  const auto system_group_alias_validation =
      runtime::validate_sm87_aot_prefill_system_plan(system_group_alias);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  system_group_alias_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kUnexpectedIdentityAlias) &&
                  !system_group_alias_validation
                       .unified_identity_namespace_complete,
              "system and group keys share one fail-closed namespace");

  auto group_dataflow_alias = plan;
  group_dataflow_alias.attention_dataflow.q_norm_weight_table_identity =
      group_for(group_dataflow_alias,
                runtime::Sm87AotPrefillExecutionGroup::kFullQkv)
          .binding_identity;
  const auto group_dataflow_alias_validation =
      runtime::validate_sm87_aot_prefill_system_plan(group_dataflow_alias);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  group_dataflow_alias_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kUnexpectedIdentityAlias),
              "group and typed-dataflow keys cannot silently alias");

  auto allowed_shape_tactic_alias = plan;
  group_for(allowed_shape_tactic_alias,
            runtime::Sm87AotPrefillExecutionGroup::kFullOResidual)
      .tactic_identity =
      group_for(allowed_shape_tactic_alias,
                runtime::Sm87AotPrefillExecutionGroup::kLinearOResidual)
          .tactic_identity;
  const auto allowed_alias_validation =
      runtime::validate_sm87_aot_prefill_system_plan(
          allowed_shape_tactic_alias);
  test.expect(allowed_alias_validation.descriptor_schema_complete &&
                  !runtime::has_sm87_aot_prefill_system_plan_issue(
                      allowed_alias_validation,
                      runtime::Sm87AotPrefillSystemPlanIssue::
                          kUnexpectedExecutionGroupIdentityAlias),
              "same-shape linear/full O groups may share only a tactic");
}

void test_forbidden_legacy_and_qualification_boundaries(TestContext& test) {
  auto forbidden = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  bind_schema_identities(forbidden);
  forbidden.uses_mtp = true;
  forbidden.uses_cublaslt = true;
  forbidden.uses_request_time_jit = true;
  forbidden.allows_fallback = true;
  group_for(forbidden, runtime::Sm87AotPrefillExecutionGroup::kGateUp)
      .allows_fallback = true;
  const auto forbidden_validation =
      runtime::validate_sm87_aot_prefill_system_plan(forbidden);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  forbidden_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kForbiddenBoundaryEnabled) &&
                  !forbidden_validation.forbidden_boundaries_satisfied,
              "MTP, cuBLASLt, request JIT, and fallback remain forbidden");

  auto legacy = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  bind_schema_identities(legacy);
  legacy.deployment_plan_id =
      "q3x.sm87.ac-prefill-prompt-wide-v2.native-p40-whole-core.v1";
  const auto legacy_validation =
      runtime::validate_sm87_aot_prefill_system_plan(legacy);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  legacy_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kDeploymentPlanIdentityMismatch) &&
                  runtime::has_sm87_aot_prefill_system_plan_issue(
                      legacy_validation,
                      runtime::Sm87AotPrefillSystemPlanIssue::
                          kLegacyOrForeignIdentityForbidden) &&
                  !legacy_validation.descriptor_schema_complete,
              "a historical plan identity cannot bind this system");

  auto premature = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  bind_schema_identities(premature);
  premature.accuracy_qualified = true;
  premature.production_dispatch_eligible = true;
  const auto premature_validation =
      runtime::validate_sm87_aot_prefill_system_plan(premature);
  test.expect(runtime::has_sm87_aot_prefill_system_plan_issue(
                  premature_validation,
                  runtime::Sm87AotPrefillSystemPlanIssue::
                      kQualificationBoundaryInvalid) &&
                  !premature_validation.qualification_boundary_satisfied &&
                  !premature_validation.executable(),
              "declared exact intent cannot claim accuracy or production");
}

void test_descriptor_schema_is_never_execution_authority(TestContext& test) {
  auto plan = runtime::make_unbound_sm87_aot_prefill_system_plan(
      kernels::Sm87TargetAotCapacityBucket::kP40);
  bind_schema_identities(plan);
  const auto validation =
      runtime::validate_sm87_aot_prefill_system_plan(plan);
  test.expect(validation.issue_mask == 0U &&
                  validation.identity_schema_complete &&
                  validation.constituent_design_complete &&
                  validation.operator_bindings_complete &&
                  validation.canonical_design_complete &&
                  validation.descriptor_schema_complete,
              "only canonical work plus opaque identities complete the schema");
  test.expect(validation.descriptor_schema_available() ==
                  runtime::kSm87AotPrefillSystemSchemaAdmissionCompiled,
              "schema availability follows only the explicit test build");
  test.expect(!validation.executable() && !plan.accuracy_qualified &&
                  !plan.production_dispatch_eligible &&
                  plan.qualification == runtime::
                      Sm87AotPrefillSystemQualification::
                          kAccuracyUnqualified,
              "a complete host schema has no launcher or execution authority");
}

}  // namespace

int main() {
  TestContext test;
  test_capacity_and_constituent_design(test);
  test_staging_geometry(test);
  test_canonical_launch_and_task_ledger(test);
  test_fused_logical_physical_ownership(test);
  test_layer_schedule_dependency_and_typed_dataflow(test);
  test_missing_duplicate_and_mutated_designs(test);
  test_constituent_mutations_fail_closed(test);
  test_route_modes_identities_and_aliases(test);
  test_forbidden_legacy_and_qualification_boundaries(test);
  test_descriptor_schema_is_never_execution_authority(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " SM87 AOT Prefill system-plan test(s) failed\n";
    return 1;
  }
  std::cout << "All SM87 AOT Prefill system-plan tests passed\n";
  return 0;
}
