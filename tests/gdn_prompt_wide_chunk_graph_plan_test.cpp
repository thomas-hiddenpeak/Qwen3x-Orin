#include "q3x/kernels/gdn_prefill_chunk64_workspace_abi.h"
#include "q3x/kernels/gdn_prefill_prompt_wide_chunk_graph_abi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

namespace kernels = q3x::kernels;

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

void test_legacy_c512_abi_is_unchanged(TestContext& test) {
  test.expect(
      kernels::kGdnPrefillChunk64WorkspaceTokenCount == 512U &&
          kernels::kGdnPrefillChunk64WorkspaceChunkCount == 8U &&
          kernels::kGdnPrefillChunk64WorkspaceLayout.q_offset == 0U &&
          kernels::kGdnPrefillChunk64WorkspaceLayout.beta_offset ==
              75'595'776U &&
          kernels::kGdnPrefillChunk64NativeWorkspaceBytes == 75'694'080U,
      "prompt-wide ABI does not alter the established C512 workspace");
}

void test_p40_geometry_and_byte_ledger(TestContext& test) {
  constexpr auto plan =
      kernels::make_gdn_prompt_wide_chunk_graph_workspace_plan(40'000U);
  test.expect(plan.ok(), "P40 is the sole admitted prompt-wide graph");
  test.expect(plan.requested_token_count == 40'000U &&
                  plan.padded_token_count == 40'000U &&
                  plan.chunk_count == 625U &&
                  plan.compact_matrix_count == 10'000U &&
                  plan.value_matrix_count == 30'000U,
              "P40 geometry is one exact 625-chunk graph");
  test.expect(plan.compact_head_token_elements == 81'920'000U &&
                  plan.value_head_token_elements == 245'760'000U &&
                  plan.transform_elements == 122'880'000U &&
                  plan.boundary_state_elements == 491'520'000U &&
                  plan.raw_gram_elements == 40'960'000U &&
                  plan.scalar_elements == 1'920'000U,
              "P40 tensor element ledger matches Qwen3.6 fixed shapes");

  const auto& layout = plan.layout;
  test.expect(layout.compact_q_offset == 0U &&
                  layout.compact_k_offset == 163'840'000U &&
                  layout.transform_offset == 327'680'000U &&
                  layout.raw_gram_offset == 1'310'720'000U &&
                  layout.w_offset == 1'802'240'000U &&
                  layout.u_offset == 2'293'760'000U &&
                  layout.gamma_offset == 2'785'280'000U &&
                  layout.beta_offset == 2'792'960'000U &&
                  layout.total_bytes == 2'800'640'000U,
              "P40 checked partition has exact stable offsets and bytes");
}

void test_phase_aliases_are_explicit_and_aligned(TestContext& test) {
  constexpr auto plan = kernels::kGdnPromptWideChunkGraphP40WorkspacePlan;
  constexpr auto layout = plan.layout;
  test.expect(layout.transform_offset == layout.boundary_state_offset,
              "transform storage becomes boundary-state storage only after WY");
  test.expect(layout.raw_gram_offset == layout.v_new_offset,
              "raw Gram storage becomes v_new storage only after WY");
  test.expect(layout.w_offset == layout.raw_output_offset,
              "W storage becomes raw-output storage only after state update");

  constexpr std::array<std::size_t, 9U> offsets{{
      layout.compact_q_offset,
      layout.compact_k_offset,
      layout.transform_offset,
      layout.raw_gram_offset,
      layout.w_offset,
      layout.u_offset,
      layout.gamma_offset,
      layout.beta_offset,
      layout.total_bytes,
  }};
  bool aligned = true;
  for (const std::size_t offset : offsets) {
    aligned = aligned &&
              offset % kernels::kGdnPromptWideChunkGraphAlignment == 0U;
  }
  test.expect(aligned, "every prompt-wide partition boundary is 256-byte aligned");

  constexpr std::size_t fully_disjoint_production_bytes = 3'701'760'000U;
  test.expect(layout.total_bytes < fully_disjoint_production_bytes &&
                  fully_disjoint_production_bytes - layout.total_bytes ==
                      901'120'000U,
              "phase-dead aliases remove 901120000 bytes without changing values");
}

void test_workspace_lifetime_receipt(TestContext& test) {
  using Region = kernels::GdnPromptWideChunkGraphWorkspaceRegionRole;
  constexpr auto receipt =
      kernels::kGdnPromptWideChunkGraphP40WorkspaceLifetimeReceipt;
  test.expect(receipt.ok(),
              "P40 workspace has one checked phase-lifetime receipt");
  test.expect(
      receipt.range(Region::kTransform).begin ==
              receipt.range(Region::kBoundaryState).begin &&
          kernels::gdn_prompt_wide_range_contains(
              receipt.range(Region::kBoundaryState),
              receipt.range(Region::kTransform)) &&
          receipt.range(Region::kRawGram).begin ==
              receipt.range(Region::kVNew).begin &&
          kernels::gdn_prompt_wide_range_contains(
              receipt.range(Region::kVNew),
              receipt.range(Region::kRawGram)) &&
          kernels::gdn_prompt_wide_ranges_exact(
              receipt.range(Region::kW),
              receipt.range(Region::kRawOutput)),
      "only the three declared dead-phase aliases share workspace storage");

  auto shifted_alias_plan =
      kernels::kGdnPromptWideChunkGraphP40WorkspacePlan;
  shifted_alias_plan.layout.boundary_state_offset +=
      kernels::kGdnPromptWideChunkGraphAlignment;
  const auto shifted_alias =
      kernels::make_gdn_prompt_wide_chunk_graph_workspace_lifetime_receipt(
          shifted_alias_plan);
  test.expect(
      !shifted_alias.ok() &&
          shifted_alias.error ==
              kernels::GdnPromptWideChunkGraphWorkspaceLifetimeError::
                  kUnexpectedAlias &&
          shifted_alias.conflict_left == Region::kTransform &&
          shifted_alias.conflict_right == Region::kBoundaryState,
      "a shifted partial transform/boundary alias fails closed");

  auto physical_overlap_plan =
      kernels::kGdnPromptWideChunkGraphP40WorkspacePlan;
  physical_overlap_plan.layout.compact_k_offset =
      physical_overlap_plan.layout.compact_q_offset +
      kernels::kGdnPromptWideChunkGraphAlignment;
  const auto physical_overlap =
      kernels::make_gdn_prompt_wide_chunk_graph_workspace_lifetime_receipt(
          physical_overlap_plan);
  test.expect(
      !physical_overlap.ok() &&
          physical_overlap.error ==
              kernels::GdnPromptWideChunkGraphWorkspaceLifetimeError::
                  kUnexpectedAlias &&
          physical_overlap.conflict_left == Region::kCompactQ &&
          physical_overlap.conflict_right == Region::kCompactK,
      "offset partial overlap between simultaneous physical owners is rejected");

  auto out_of_bounds_plan =
      kernels::kGdnPromptWideChunkGraphP40WorkspacePlan;
  out_of_bounds_plan.layout.beta_offset =
      out_of_bounds_plan.layout.total_bytes;
  const auto out_of_bounds =
      kernels::make_gdn_prompt_wide_chunk_graph_workspace_lifetime_receipt(
          out_of_bounds_plan);
  test.expect(
      !out_of_bounds.ok() &&
          out_of_bounds.error ==
              kernels::GdnPromptWideChunkGraphWorkspaceLifetimeError::
                  kOutOfBounds &&
          out_of_bounds.conflict_left == Region::kBeta,
      "a logical region extending beyond the checked workspace fails closed");
}

[[nodiscard]] kernels::GdnPromptWideChunkGraphBufferAddresses
make_disjoint_p40_addresses() {
  constexpr std::uintptr_t stride = std::uintptr_t{1} << 32U;
  kernels::GdnPromptWideChunkGraphBufferAddresses addresses{};
  addresses.workspace = stride;
  addresses.workspace_capacity_bytes =
      kernels::kGdnPromptWideChunkGraphP40WorkspaceBytes;
  addresses.raw_qkv = 2U * stride;
  addresses.conv_weight = 3U * stride;
  addresses.conv_history = 4U * stride;
  addresses.conv_qkv_output = 5U * stride;
  addresses.a = 6U * stride;
  addresses.b = 7U * stride;
  addresses.A_log = 8U * stride;
  addresses.dt_bias = 9U * stride;
  addresses.state_input = 10U * stride;
  addresses.state_output = 11U * stride;
  addresses.norm_weight = 12U * stride;
  addresses.silu_gate = 13U * stride;
  addresses.output = 14U * stride;
  return addresses;
}

void test_external_buffer_byte_ledger_and_alias_contract(TestContext& test) {
  using Error = kernels::GdnPromptWideChunkGraphBufferContractError;
  using Role = kernels::GdnPromptWideChunkGraphBufferRole;
  auto addresses = make_disjoint_p40_addresses();
  const auto receipt =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, addresses);
  test.expect(receipt.ok() && !receipt.state_in_place,
              "disjoint fixed-shape P40 buffers pass the host contract");
  test.expect(
      receipt.range(Role::kWorkspace).size_bytes() == 2'800'640'000U &&
          receipt.range(Role::kRawQkv).size_bytes() == 819'200'000U &&
          receipt.range(Role::kConvWeight).size_bytes() == 81'920U &&
          receipt.range(Role::kConvHistory).size_bytes() == 61'440U &&
          receipt.range(Role::kConvQkvOutput).size_bytes() ==
              819'200'000U &&
          receipt.range(Role::kA).size_bytes() == 3'840'000U &&
          receipt.range(Role::kB).size_bytes() == 3'840'000U &&
          receipt.range(Role::kALog).size_bytes() == 96U &&
          receipt.range(Role::kDtBias).size_bytes() == 96U &&
          receipt.range(Role::kStateInput).size_bytes() == 1'572'864U &&
          receipt.range(Role::kStateOutput).size_bytes() == 1'572'864U &&
          receipt.range(Role::kNormWeight).size_bytes() == 256U &&
          receipt.range(Role::kSiluGate).size_bytes() == 491'520'000U &&
          receipt.range(Role::kOutput).size_bytes() == 491'520'000U,
      "every external P40 tensor has the exact checked byte range");

  auto in_place_state_addresses = addresses;
  in_place_state_addresses.state_output =
      in_place_state_addresses.state_input;
  const auto in_place_state =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, in_place_state_addresses);
  test.expect(in_place_state.ok() && in_place_state.state_in_place,
              "exact state-input/state-output alias is the sole external alias");

  auto partial_state_addresses = addresses;
  partial_state_addresses.state_output =
      partial_state_addresses.state_input + sizeof(std::uint16_t);
  const auto partial_state =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, partial_state_addresses);
  test.expect(
      !partial_state.ok() && partial_state.error == Error::kForbiddenOverlap &&
          partial_state.conflict_left == Role::kStateInput &&
          partial_state.conflict_right == Role::kStateOutput,
      "shifted state overlap is rejected even though exact in-place is safe");

  auto exact_conv_alias_addresses = addresses;
  exact_conv_alias_addresses.conv_qkv_output =
      exact_conv_alias_addresses.raw_qkv;
  const auto exact_conv_alias =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, exact_conv_alias_addresses);
  test.expect(
      !exact_conv_alias.ok() &&
          exact_conv_alias.error == Error::kForbiddenOverlap &&
          exact_conv_alias.conflict_left == Role::kRawQkv &&
          exact_conv_alias.conflict_right == Role::kConvQkvOutput,
      "prompt-wide token-parallel convolution forbids raw/output in-place races");

  auto shifted_conv_alias_addresses = addresses;
  shifted_conv_alias_addresses.conv_qkv_output =
      shifted_conv_alias_addresses.raw_qkv + 819'200'000U -
      sizeof(std::uint16_t);
  const auto shifted_conv_alias =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, shifted_conv_alias_addresses);
  test.expect(
      !shifted_conv_alias.ok() &&
          shifted_conv_alias.error == Error::kForbiddenOverlap &&
          shifted_conv_alias.conflict_left == Role::kRawQkv &&
          shifted_conv_alias.conflict_right == Role::kConvQkvOutput,
      "a two-byte shifted raw/output overlap cannot evade validation");

  auto workspace_overlap_addresses = addresses;
  workspace_overlap_addresses.raw_qkv =
      workspace_overlap_addresses.workspace +
      kernels::kGdnPromptWideChunkGraphAlignment;
  const auto workspace_overlap =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, workspace_overlap_addresses);
  test.expect(
      !workspace_overlap.ok() &&
          workspace_overlap.error == Error::kForbiddenOverlap &&
          workspace_overlap.conflict_left == Role::kWorkspace &&
          workspace_overlap.conflict_right == Role::kRawQkv,
      "external tensors cannot overlap the touched workspace span");
}

void test_external_range_boundaries_fail_closed(TestContext& test) {
  using Error = kernels::GdnPromptWideChunkGraphBufferContractError;
  using Role = kernels::GdnPromptWideChunkGraphBufferRole;
  auto insufficient = make_disjoint_p40_addresses();
  --insufficient.workspace_capacity_bytes;
  const auto insufficient_receipt =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, insufficient);
  test.expect(
      !insufficient_receipt.ok() &&
          insufficient_receipt.error == Error::kInsufficientWorkspace &&
          insufficient_receipt.conflict_left == Role::kWorkspace,
      "one byte less than the exact workspace fails before range exposure");

  auto null_gate = make_disjoint_p40_addresses();
  null_gate.silu_gate = 0U;
  const auto null_gate_receipt =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, null_gate);
  test.expect(!null_gate_receipt.ok() &&
                  null_gate_receipt.error == Error::kNullBuffer &&
                  null_gate_receipt.conflict_left == Role::kSiluGate,
              "every required P40 external buffer rejects null");

  auto misaligned_weight = make_disjoint_p40_addresses();
  ++misaligned_weight.conv_weight;
  const auto misaligned_receipt =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, misaligned_weight);
  test.expect(
      !misaligned_receipt.ok() &&
          misaligned_receipt.error == Error::kMisalignedBuffer &&
          misaligned_receipt.conflict_left == Role::kConvWeight,
      "BF16 external ranges enforce element alignment");

  auto overflowing_output = make_disjoint_p40_addresses();
  constexpr std::size_t output_bytes = 491'520'000U;
  overflowing_output.output =
      std::numeric_limits<std::uintptr_t>::max() - output_bytes + 1U;
  const auto overflow_receipt =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, overflowing_output);
  test.expect(
      !overflow_receipt.ok() &&
          overflow_receipt.error == Error::kArithmeticOverflow &&
          overflow_receipt.conflict_left == Role::kOutput,
      "end-address wrap is rejected before overlap arithmetic");

  auto overflowing_workspace = make_disjoint_p40_addresses();
  overflowing_workspace.workspace =
      std::numeric_limits<std::uintptr_t>::max() -
      overflowing_workspace.workspace_capacity_bytes + 1U;
  const auto workspace_overflow_receipt =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          40'000U, overflowing_workspace);
  test.expect(
      !workspace_overflow_receipt.ok() &&
          workspace_overflow_receipt.error == Error::kArithmeticOverflow &&
          workspace_overflow_receipt.conflict_left == Role::kWorkspace,
      "the complete caller-declared workspace capacity has a checked end");

  const auto unowned_shape =
      kernels::make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
          60'000U, make_disjoint_p40_addresses());
  test.expect(!unowned_shape.ok() &&
                  unowned_shape.error == Error::kInvalidPlan,
              "the range helper cannot accidentally admit P60");
}

void test_p60_and_every_unowned_shape_fail_closed(TestContext& test) {
  constexpr auto p60 =
      kernels::make_gdn_prompt_wide_chunk_graph_workspace_plan(60'000U);
  test.expect(
      !p60.ok() &&
          p60.error == kernels::GdnPromptWideChunkGraphPlanError::
                           kP60PartialChunkPending &&
          p60.requested_token_count == 60'000U &&
          p60.padded_token_count == 60'032U && p60.chunk_count == 938U &&
          p60.layout.total_bytes == 0U,
      "P60 reports its partial-C64 geometry but owns no launch/workspace");

  constexpr auto c512 =
      kernels::make_gdn_prompt_wide_chunk_graph_workspace_plan(512U);
  constexpr auto p40_plus_one =
      kernels::make_gdn_prompt_wide_chunk_graph_workspace_plan(40'001U);
  test.expect(
      !c512.ok() &&
          c512.error == kernels::GdnPromptWideChunkGraphPlanError::
                            kInvalidTokenCount &&
          c512.layout.total_bytes == 0U && !p40_plus_one.ok() &&
          p40_plus_one.error ==
              kernels::GdnPromptWideChunkGraphPlanError::kInvalidTokenCount &&
          p40_plus_one.layout.total_bytes == 0U,
      "prompt-wide entry cannot relabel C512 or arbitrary M as admitted");

  constexpr auto overflow =
      kernels::make_gdn_prompt_wide_chunk_graph_workspace_plan(
          std::numeric_limits<std::size_t>::max());
  test.expect(
      !overflow.ok() &&
          overflow.error == kernels::GdnPromptWideChunkGraphPlanError::
                                kArithmeticOverflow &&
          overflow.layout.total_bytes == 0U,
      "geometry overflow fails before any workspace bytes are exposed");
}

}  // namespace

int main() {
  TestContext test;
  test_legacy_c512_abi_is_unchanged(test);
  test_p40_geometry_and_byte_ledger(test);
  test_phase_aliases_are_explicit_and_aligned(test);
  test_workspace_lifetime_receipt(test);
  test_external_buffer_byte_ledger_and_alias_contract(test);
  test_external_range_boundaries_fail_closed(test);
  test_p60_and_every_unowned_shape_fail_closed(test);

  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " prompt-wide GDN chunk-graph plan checks failed\n";
    return 1;
  }
  std::cout << "prompt-wide GDN chunk-graph plan checks passed\n";
  return 0;
}
