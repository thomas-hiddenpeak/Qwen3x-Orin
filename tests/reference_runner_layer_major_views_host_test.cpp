#include "q3x/runtime/prefill_workspace_plan.h"
#include "q3x/runtime/reference_runner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

namespace runtime = q3x::runtime;

class TestContext {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] bool operation_is(
    const runtime::ReferenceLayerMajorRequestDescriptorOutcome& outcome,
    const std::string_view expected) noexcept {
  return outcome.status.operation != nullptr &&
         std::string_view(outcome.status.operation) == expected;
}

void test_descriptor_target_profiles(TestContext& test) {
  constexpr std::array<std::uint64_t, 6U> kSequenceCapacities{
      1U, 40'000U, 40'001U, 60'000U, 130'000U,
      runtime::kAbsoluteRequestMaxSequenceLength};
  for (const std::uint64_t capacity : kSequenceCapacities) {
    runtime::LayerMajorRequestMemoryOptions options;
    options.max_sequence_length = capacity;
    if (capacity == runtime::kAbsoluteRequestMaxSequenceLength) {
      // The explicit profile permits a caller-selected arena ceiling above
      // the 17.4-GB default. The host descriptor must not reapply that default
      // to the valid 262144-token plan (~20.9 GB selected request state).
      options.max_arena_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    }
    const runtime::LayerMajorRequestPlanResult planned =
        runtime::build_layer_major_request_memory_plan(options);
    test.expect(planned.ok(),
                "layer-major source plan builds for host descriptor");
    if (!planned) {
      continue;
    }
    const runtime::ReferenceLayerMajorRequestDescriptorOutcome described =
        runtime::build_reference_layer_major_candidate_binding_descriptor(
            *planned.value);
    test.expect(described.ok(),
                "valid layer-major plan produces one host binding descriptor");
    if (!described) {
      continue;
    }
    const runtime::ReferenceLayerMajorRequestBindingDescriptor& descriptor =
        *described.value;
    test.expect(
        descriptor.disposition == runtime::
                                      ReferenceLayerMajorBindingDisposition::
                                          kUnboundCandidateOnly &&
            descriptor.profile ==
                runtime::RequestMemoryProfile::kLayerMajorC8192 &&
            descriptor.max_sequence_length == capacity &&
            descriptor.operator_panel_capacity_tokens == 8'192U &&
            descriptor.legacy_prefill_chunk_size == 512U &&
            descriptor.arena_bytes == planned.value->common.arena_bytes,
        "descriptor remains candidate-only and preserves capacity identity");
    test.expect(
        descriptor.prompt_residual_bf16.row_capacity == capacity &&
            descriptor.prompt_residual_bf16.columns == 5'120U &&
            descriptor.panel_token_ids_u32.row_capacity == 8'192U &&
            descriptor.panel_token_ids_u32.columns == 1U &&
            descriptor.panel_token_ids_u32.storage.element_size_bytes ==
                sizeof(std::uint32_t) &&
            descriptor.final_hidden_bf16.row_capacity == 1U &&
            descriptor.final_hidden_bf16.columns == 5'120U,
        "prompt residual, token IDs, and final hidden identities are typed");
    test.expect(
        descriptor.gdn.qkv_bf16.columns == 10'240U &&
            descriptor.gdn.recurrent_core_bf16.columns == 6'144U &&
            descriptor.attention.raw_q_gate_bf16.columns == 12'288U &&
            descriptor.attention.processed_q_bf16.columns == 6'144U &&
            descriptor.mlp.gate_bf16.columns == 17'408U &&
            descriptor.mlp.activated_bf16.columns == 17'408U &&
            descriptor.mlp.normalized_input_bf16.columns == 5'120U,
        "all three C8192 operator families retain their distinct typed shapes");
    test.expect(
        descriptor.legacy_c512.hidden_bf16[0U].row_capacity == 512U &&
            descriptor.legacy_c512.projection_bf16[0U].columns == 17'408U &&
            descriptor.conv_state_bf16.byte_size ==
                runtime::kRequestConvStateBytes &&
            descriptor.gdn_state_bf16.byte_size ==
                runtime::kRequestGdnStateBytes &&
            descriptor.rope_cos_fp32.element_capacity == capacity * 32U &&
            descriptor.rope_sin_fp32.element_capacity == capacity * 32U,
        "legacy, persistent, KV schedule, and RoPE identities are retained");
    test.expect(
        descriptor.hidden_strategy ==
                runtime::PrefillHiddenStrategy::
                    kSinglePromptWideConditional &&
            descriptor.scratch_strategy ==
                runtime::PrefillOperatorScratchStrategy::
                    kC8192FamilyOverlayWithDisjointLegacyC512 &&
            descriptor.gdn_tactic ==
                runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv &&
            descriptor.legacy_gdn_tactic ==
                runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite &&
            descriptor.mlp_tactic ==
                runtime::PrefillMlpPhysicalTactic::
                    kSeparateGateUpAndSilu,
        "descriptor binds only the fixed physical tactic identities");
  }
}

void test_descriptor_fails_closed_by_family(TestContext& test) {
  runtime::LayerMajorRequestMemoryOptions options;
  options.max_sequence_length = 40'000U;
  const runtime::LayerMajorRequestPlanResult planned =
      runtime::build_layer_major_request_memory_plan(options);
  test.expect(planned.ok(), "mutation fixture plan builds");
  if (!planned) {
    return;
  }
  const runtime::LayerMajorRequestMemoryPlan baseline = *planned.value;
  test.expect(!baseline.executable(),
              "candidate descriptor fixture is explicitly non-executable");

  runtime::LayerMajorRequestMemoryPlan candidate = baseline;
  candidate.common.profile = runtime::RequestMemoryProfile::kLegacyC512;
  runtime::ReferenceLayerMajorRequestDescriptorOutcome rejected =
      runtime::build_reference_layer_major_candidate_binding_descriptor(
          candidate);
  test.expect(
      !rejected &&
          rejected.access_error ==
              runtime::RequestAccessError::kMemoryProfileMismatch &&
          operation_is(rejected, "layer_major_memory_profile"),
      "legacy plan profile fails closed before any typed interpretation");

  candidate = baseline;
  candidate.common.layers[3U].type =
      q3x::model::LayerType::kLinearAttention;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected &&
                  rejected.status.error ==
                      runtime::ReferenceRunnerError::kInvalidLayerSchedule &&
                  operation_is(rejected, "layer_major_layer_schedule"),
              "wrong 48/16 schedule fails at the schedule identity");

  candidate = baseline;
  --candidate.common.key_cache[0U].element_capacity;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected &&
                  operation_is(rejected, "layer_major_common_identity"),
              "persistent/KV/RoPE common identity fails closed");

  candidate = baseline;
  ++candidate.prompt_residual_bf16.columns;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected &&
                  operation_is(rejected, "layer_major_prompt_residual"),
              "prompt residual shape fails closed");

  candidate = baseline;
  --candidate.c8192_family_phase_arena.byte_size;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected &&
                  operation_is(rejected,
                               "layer_major_family_arena_identity"),
              "C8192 owning capacity is validated but never exported");

  candidate = baseline;
  candidate.panel_token_ids_u32.storage.element_size_bytes = 2U;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected &&
                  operation_is(rejected, "layer_major_panel_token_ids"),
              "token-ID staging dtype fails closed");

  candidate = baseline;
  --candidate.gdn.qkv_bf16.columns;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected && operation_is(rejected, "layer_major_gdn_views"),
              "GDN typed phase failure remains family-local");

  candidate = baseline;
  candidate.gdn.z_bf16.storage.arena_offset +=
      runtime::kRequestArenaAlignment;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected && operation_is(rejected, "layer_major_gdn_views"),
              "GDN qkv-z-a-b-core-workspace sequence cannot be shifted");

  candidate = baseline;
  candidate.gdn.branch_output_bf16.storage.arena_offset +=
      runtime::kRequestArenaAlignment;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected && operation_is(rejected, "layer_major_gdn_views"),
              "GDN dead-QKV branch/output temporary aliases are exact");

  candidate = baseline;
  --candidate.attention.processed_q_bf16.columns;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(
      !rejected && operation_is(rejected, "layer_major_attention_views"),
      "Attention typed phase failure remains family-local");

  candidate = baseline;
  candidate.attention.packed_gate_bf16.storage.arena_offset +=
      runtime::kRequestArenaAlignment;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(
      !rejected && operation_is(rejected, "layer_major_attention_views"),
      "Attention raw-processed-packed topology cannot be shifted");

  candidate = baseline;
  candidate.attention.branch_output_bf16.storage.arena_offset +=
      runtime::kRequestArenaAlignment;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(
      !rejected && operation_is(rejected, "layer_major_attention_views"),
      "Attention core-branch-output-temp dead-span sequence is exact");

  candidate = baseline;
  --candidate.mlp.activated_bf16.columns;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected && operation_is(rejected, "layer_major_mlp_views"),
              "MLP typed phase failure remains family-local");

  candidate = baseline;
  candidate.mlp.up_bf16.storage.arena_offset +=
      runtime::kRequestArenaAlignment;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected && operation_is(rejected, "layer_major_mlp_views"),
              "MLP gate-up-activated topology cannot be shifted");

  candidate = baseline;
  candidate.mlp.down_projection_temporary.arena_offset +=
      runtime::kRequestArenaAlignment;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected && operation_is(rejected, "layer_major_mlp_views"),
              "MLP normalized/branch/down-temporary aliases are exact");

  candidate = baseline;
  --candidate.legacy_c512.hidden_bf16[0U].columns;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(
      !rejected && operation_is(rejected, "layer_major_legacy_c512_views"),
      "disjoint legacy bundle failure remains explicit");

  candidate = baseline;
  constexpr std::uint64_t kExpandedScratchBytes =
      runtime::kRequestArenaAlignment;
  constexpr std::uint64_t kExpandedScratchElements =
      kExpandedScratchBytes / sizeof(float);
  candidate.legacy_c512.fp32_scratch.byte_size += kExpandedScratchBytes;
  candidate.legacy_c512.fp32_scratch.element_capacity +=
      kExpandedScratchElements;
  candidate.common.fp32_scratch = candidate.legacy_c512.fp32_scratch;
  candidate.final_hidden_bf16.storage.arena_offset += kExpandedScratchBytes;
  candidate.common.workspace_bytes += kExpandedScratchBytes;
  candidate.common.rope_offset += kExpandedScratchBytes;
  candidate.common.rope_cos_fp32.arena_offset += kExpandedScratchBytes;
  candidate.common.rope_sin_fp32.arena_offset += kExpandedScratchBytes;
  candidate.common.arena_bytes += kExpandedScratchBytes;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(
      !rejected && operation_is(rejected, "layer_major_legacy_c512_views"),
      "self-consistent enlarged legacy FP32 scratch is not canonical");

  candidate = baseline;
  --candidate.final_hidden_bf16.columns;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(!rejected && operation_is(rejected, "layer_major_final_hidden"),
              "fixed final hidden identity fails closed");

  candidate = baseline;
  candidate.mlp_tactic =
      runtime::PrefillMlpPhysicalTactic::kFusedGateUpEpilogue;
  rejected = runtime::build_reference_layer_major_candidate_binding_descriptor(
      candidate);
  test.expect(
      !rejected &&
          operation_is(rejected, "layer_major_physical_tactic_identity"),
      "unplanned tactic changes fail before view collection");

  using ContractFlag = bool runtime::LayerMajorRequestMemoryPlan::*;
  constexpr std::array<ContractFlag, 6U> kUnboundFlags{
      &runtime::LayerMajorRequestMemoryPlan::
          prompt_residual_in_place_contract_bound,
      &runtime::LayerMajorRequestMemoryPlan::family_completion_events_bound,
      &runtime::LayerMajorRequestMemoryPlan::intra_family_phase_contract_bound,
      &runtime::LayerMajorRequestMemoryPlan::
          prompt_token_ids_consumed_event_bound,
      &runtime::LayerMajorRequestMemoryPlan::
          projection_workspace_subrange_binding_bound,
      &runtime::LayerMajorRequestMemoryPlan::operator_bindings_complete};
  for (const ContractFlag flag : kUnboundFlags) {
    candidate = baseline;
    candidate.*flag = true;
    rejected =
        runtime::build_reference_layer_major_candidate_binding_descriptor(
            candidate);
    test.expect(
        !rejected &&
            operation_is(rejected, "layer_major_unbound_contract_identity"),
        "candidate-only descriptor rejects every claimed binding flag");
  }
}

void test_collector_profile_gate_is_host_only(TestContext& test) {
  const runtime::ReferenceLayerMajorRequestViewsOutcome null_state =
      runtime::collect_reference_layer_major_candidate_views(
          static_cast<runtime::RequestState*>(nullptr));
  test.expect(!null_state &&
                  null_state.status.error ==
                      runtime::ReferenceRunnerError::kInvalidDependency &&
                  null_state.status.cuda_error == 0,
              "null candidate dependency is rejected without CUDA");

  runtime::RequestState empty_legacy_state;
  const runtime::ReferenceLayerMajorRequestViewsOutcome legacy_state =
      runtime::collect_reference_layer_major_candidate_views(
          empty_legacy_state);
  test.expect(
      !legacy_state &&
          legacy_state.status.error ==
              runtime::ReferenceRunnerError::kInvalidRequestState &&
          legacy_state.access_error ==
              runtime::RequestAccessError::kMemoryProfileMismatch &&
          legacy_state.status.operation != nullptr &&
          std::string_view(legacy_state.status.operation) ==
              "layer_major_memory_profile" &&
          legacy_state.status.cuda_error == 0,
      "legacy RequestState reports profile mismatch before empty/CUDA checks");
}

}  // namespace

int main() {
  TestContext test;
  test_descriptor_target_profiles(test);
  test_descriptor_fails_closed_by_family(test);
  test_collector_profile_gate_is_host_only(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " layer-major runner-view host test(s) failed\n";
    return 1;
  }
  std::cout << "Layer-major runner-view host tests passed\n";
  return 0;
}
