#include "q3x/runtime/sm87_bulk_dataflow_v2_p40_plan.h"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace runtime = q3x::runtime;

namespace {

class TestContext final {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] constexpr runtime::Sm87BulkV2P40SealedExecutionAccess
valid_seal() noexcept {
  runtime::Sm87BulkV2P40SealedExecutionAccess access;
  access.plan_magic = runtime::kSm87BulkV2P40PlanMagic;
  access.abi_major = runtime::kSm87BulkV2P40PlanAbiMajor;
  access.abi_minor = runtime::kSm87BulkV2P40PlanAbiMinor;
  access.state = runtime::Sm87BulkV2P40SealState::kSealed;
  access.deployment_identity = 1U;
  access.model_identity = 2U;
  access.allocation_identity = 3U;
  access.stream_event_owner_identity = 4U;
  access.asset_catalog_identity = 5U;
  access.binary_evidence_identity = 6U;
  access.fp8_oracle_evidence_identity = 7U;
  access.attention_oracle_evidence_identity = 8U;
  access.gdn_oracle_evidence_identity = 9U;
  access.nvfp4_oracle_evidence_identity = 10U;
  access.seal_nonce = 11U;
  access.device_ordinal = 0;
  access.all_static_checks_complete = true;
  access.request_time_revalidation_required = false;
  access.production_dispatch_eligible = false;
  return access;
}

void test_plan(TestContext& test) {
  constexpr auto plan = runtime::kSm87BulkV2P40FrozenExecutionPlan;
  test.expect(runtime::sm87_bulk_v2_p40_execution_plan_valid(plan),
              "the frozen whole-request plan validates");
  test.expect(plan.abi_major == 3U,
              "ABI 3 binds the versioned whole-projection successor receipt");
  test.expect(plan.stream_count == 5U && plan.maximum_host_waits == 1U &&
                  plan.maximum_host_drains == 1U &&
                  plan.hot_cuda_resource_queries == 0U,
              "the hot path is five-stream, device-ordered, and query-free");
  test.expect(plan.family_arena.attention_required_extent_bytes ==
                  runtime::kSm87BulkV2P40FamilyArenaBytes &&
                  plan.family_arena.gdn_required_extent_bytes <
                      plan.family_arena.attention_required_extent_bytes &&
                  plan.family_arena.mlp_required_extent_bytes <
                      plan.family_arena.attention_required_extent_bytes &&
                  plan.family_arena.mlp_live_bytes == 418'512'896ULL,
              "Attention remains the exact family-arena capacity peak");
  test.expect(runtime::sm87_bulk_v2_p40_control_plane_plan_valid(
                  plan.control_plane) &&
                  plan.control_plane.device_arena_bytes == 1'280U &&
                  plan.control_plane.mapped_host_reset_bytes == 4U,
              "the separate epoch control plane is frozen exactly");

  auto wrong = plan;
  wrong.maximum_host_waits = 64U;
  test.expect(!runtime::sm87_bulk_v2_p40_execution_plan_valid(wrong),
              "the rejected v1 per-layer host wait cannot enter v2");
  wrong = plan;
  wrong.hot_cuda_resource_queries = 1U;
  test.expect(!runtime::sm87_bulk_v2_p40_execution_plan_valid(wrong),
              "one request-time static CUDA query rejects the plan");
  wrong = plan;
  wrong.production_dispatch_eligible = true;
  test.expect(!runtime::sm87_bulk_v2_p40_execution_plan_valid(wrong),
              "a host contract cannot open production dispatch");
  wrong = plan;
  ++wrong.control_plane.bindings[1U].range.offset;
  test.expect(!runtime::sm87_bulk_v2_p40_execution_plan_valid(wrong),
              "a mutated control-plane offset rejects the plan");
}

void test_live_aliases(TestContext& test) {
  constexpr auto arena = runtime::sm87_bulk_v2_p40_family_arena_plan();
  for (std::size_t first = 0U; first < arena.bindings.size(); ++first) {
    for (std::size_t second = first + 1U; second < arena.bindings.size();
         ++second) {
      const auto& left = arena.bindings[first];
      const auto& right = arena.bindings[second];
      if (runtime::sm87_bulk_v2_p40_ranges_overlap(left.range,
                                                   right.range)) {
        test.expect((left.live_phase_mask & right.live_phase_mask) == 0U,
                    "overlapping family spans have disjoint live phases");
      }
    }
  }
  test.expect(!arena.whole_arena_request_memset_required,
              "v2 does not inherit the 5.075-GB request reset");
  test.expect(arena.data_plane_cold_reset_bytes == 78'446'592ULL &&
                  arena.cold_initial_state_is_persistent_layer_state &&
                  arena.final_gdn_state_copied_after_epilogue,
              "cold reset touches only GDN persistent state and history");

  auto wrong = arena;
  wrong.bindings[0U].role = wrong.bindings[1U].role;
  test.expect(!runtime::sm87_bulk_v2_p40_family_arena_plan_valid(wrong),
              "a duplicate role rejects the exact family plan");
  wrong = arena;
  ++wrong.bindings[7U].range.offset;
  test.expect(!runtime::sm87_bulk_v2_p40_family_arena_plan_valid(wrong),
              "a shifted Attention range rejects the exact family plan");
  wrong = arena;
  wrong.bindings[6U].live_phase_mask = 1U;
  test.expect(!runtime::sm87_bulk_v2_p40_family_arena_plan_valid(wrong),
              "a shortened GDN private lifetime rejects the plan");
}

void test_seal_and_receipt(TestContext& test) {
  constexpr auto access = valid_seal();
  static_assert(runtime::sm87_bulk_v2_p40_sealed_access_valid(access));
  test.expect(runtime::sm87_bulk_v2_p40_sealed_access_valid(access),
              "startup creates one sealed execution capability");
  auto wrong_access = access;
  wrong_access.request_time_revalidation_required = true;
  test.expect(!runtime::sm87_bulk_v2_p40_sealed_access_valid(wrong_access),
              "a seal that leaves static checks for the request is invalid");

  auto completed = runtime::sm87_bulk_v2_p40_request_receipt(access, 7U);
  completed.lifecycle = runtime::Sm87BulkV2P40OwnerLifecycle::kCompleted;
  completed.completed_layers = 64U;
  completed.completed_gdn_layers = 48U;
  completed.completed_full_layers = 16U;
  completed.closed_layer_residuals = 64U;
  completed.closed_gdn_state_publications = 48U;
  completed.logical_projection_roles = 496U;
  completed.fused_outer_operations = 304U;
  completed.projection_conventional_operations =
      1'948'044'492'800'000ULL;
  completed.projection_successor =
      runtime::sm87_bulk_v2_p40_projection_successor_receipt();
  completed.projection_successor.fp8_gdn_input_whole_launches = 48U;
  completed.projection_successor.fp8_full_input_whole_launches = 16U;
  completed.projection_successor.fp8_output_whole_launches = 64U;
  completed.projection_successor.fp8_whole_role_launches = 128U;
  completed.projection_successor.nvfp4_gate_up_whole_launches = 64U;
  completed.projection_successor.nvfp4_down_whole_launches = 64U;
  completed.projection_successor.nvfp4_whole_role_launches = 128U;
  completed.projection_successor.bf16_ab_physical_launches = 48U;
  completed.enqueued_attention_launches = 64U;
  completed.enqueued_attention_preprocess_panels = 80U;
  completed.enqueued_bf16_ab_launches = 48U;
  completed.enqueued_gdn_producer_chunks = 30'000U;
  completed.enqueued_gdn_recurrence_chunks = 30'000U;
  completed.enqueued_gdn_epilogue_chunks = 30'000U;
  completed.enqueued_gdn_persistent_copies = 96U;
  completed.enqueued_final_norm = 1U;
  completed.enqueued_lm_head = 1U;
  completed.enqueued_argmax = 1U;
  completed.enqueued_handoff_d2h = 1U;
  completed.terminal_host_waits = 1U;
  completed.terminal_host_drains = 1U;
  completed.last_submitted_layer = 63U;
  completed.last_submitted_segment = 0U;
  completed.last_submitted_constituent = 3U;
  completed.last_submitted_family =
      runtime::Sm87BulkV2P40FamilyPhase::kFinalHandoff;
  completed.handoff_token_id = 42U;
  completed.handoff_nonfinite = 0U;
  completed.handoff_observed = true;
  completed.submission_started = true;
  completed.all_streams_drained = true;
  completed.state_committed = true;
  test.expect(runtime::sm87_bulk_v2_p40_receipt_terminal_valid(access,
                                                               completed),
              "matched complete work can commit exactly once");

  auto partial = completed;
  --partial.projection_successor.fp8_gdn_input_whole_launches;
  test.expect(!runtime::sm87_bulk_v2_p40_receipt_terminal_valid(access,
                                                                partial),
              "one missing whole-P40000 FP8 role launch prevents commit");
  partial = completed;
  partial.projection_successor.route =
      runtime::Sm87BulkV2P40ProjectionRoute::kExactControlSteppingStones;
  partial.projection_successor.fp8_exact_control_launches = 5'120U;
  partial.projection_successor.nvfp4_exact_control_launches = 2'560U;
  test.expect(!runtime::sm87_bulk_v2_p40_receipt_terminal_valid(access,
                                                                partial),
              "old FP8/NVFP4 exact controls cannot close the whole successor receipt");
  partial = completed;
  partial.all_streams_drained = false;
  test.expect(!runtime::sm87_bulk_v2_p40_receipt_terminal_valid(access,
                                                                partial),
              "completion without whole-owner retirement is invalid");
  partial = completed;
  partial.closed_gdn_state_publications = 47U;
  test.expect(!runtime::sm87_bulk_v2_p40_receipt_terminal_valid(access,
                                                                partial),
              "a layer cannot close before its GDN state publication");
  auto foreign_access = access;
  ++foreign_access.allocation_identity;
  test.expect(!runtime::sm87_bulk_v2_p40_receipt_terminal_valid(
                  foreign_access, completed),
              "a receipt cannot cross its startup allocation seal");

  auto poisoned = runtime::sm87_bulk_v2_p40_request_receipt(access, 8U);
  poisoned.lifecycle = runtime::Sm87BulkV2P40OwnerLifecycle::kPoisoned;
  poisoned.first_error = 719;
  poisoned.submission_started = true;
  poisoned.cancellation_published = true;
  poisoned.all_streams_drained = true;
  poisoned.last_submitted_layer = 0U;
  poisoned.last_submitted_segment = 0U;
  poisoned.last_submitted_constituent = 0U;
  poisoned.last_submitted_family =
      runtime::Sm87BulkV2P40FamilyPhase::kGdnInput;
  test.expect(runtime::sm87_bulk_v2_p40_receipt_terminal_valid(access,
                                                               poisoned),
              "partial submission is valid only as drained poisoned evidence");
  poisoned.state_committed = true;
  test.expect(!runtime::sm87_bulk_v2_p40_receipt_terminal_valid(access,
                                                                poisoned),
              "poisoned work can never publish Prefill state");
}

}  // namespace

int main() {
  TestContext test;
  test_plan(test);
  test_live_aliases(test);
  test_seal_and_receipt(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 bulk-dataflow-v2 whole-P40 plan host contract passed\n";
  return 0;
}
