#include "q3x/runtime/prefill_execution_plan.h"
#include "q3x/runtime/request_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {

namespace model = q3x::model;
namespace runtime = q3x::runtime;

static_assert(noexcept(
    runtime::is_valid_unbound_layer_major_prefill_execution_plan(
        std::declval<const runtime::PrefillExecutionPlan&>())));

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

[[nodiscard]] runtime::PrefillExecutionPlanResult build_plan(
    const std::uint64_t prompt_token_count,
    const std::uint64_t first_position = 0U,
    const std::uint64_t max_sequence_length =
        runtime::kLayerMajorPrefillMaximumSequenceTokens) {
  runtime::PrefillExecutionPlanOptions options;
  options.first_position = first_position;
  options.prompt_token_count = prompt_token_count;
  options.max_sequence_length = max_sequence_length;
  return runtime::build_unbound_layer_major_prefill_execution_plan(options);
}

void expect_panel_shape(TestContext& test,
                        const std::uint64_t prompt_token_count,
                        const std::size_t expected_panel_count,
                        const std::uint32_t expected_tail_tokens) {
  const runtime::PrefillExecutionPlanResult result =
      build_plan(prompt_token_count);
  const bool valid = result &&
                     result.value->panel_count == expected_panel_count &&
                     result.value->panels[expected_panel_count - 1U]
                             .token_count == expected_tail_tokens &&
                     result.value->panels[expected_panel_count - 1U]
                             .end_position == prompt_token_count;
  if (!valid) {
    std::cerr << "  panel shape mismatch: prompt=" << prompt_token_count
              << " expected_count=" << expected_panel_count
              << " expected_tail=" << expected_tail_tokens << '\n';
  }
  test.expect(valid, "prompt decomposes into the expected C8192 panels");
}

[[nodiscard]] bool same_progress(
    const runtime::PrefillExecutionProgress& left,
    const runtime::PrefillExecutionProgress& right) {
  return left.kv_visible_end == right.kv_visible_end &&
         left.gdn_advanced_end == right.gdn_advanced_end &&
         left.completed_panels == right.completed_panels &&
         left.next_layer == right.next_layer &&
         left.next_panel == right.next_panel &&
         left.final_hidden_ready == right.final_hidden_ready &&
         left.prefill_state_committed == right.prefill_state_committed;
}

void test_public_tile_and_operator_panel_are_independent(TestContext& test) {
  test.expect(
      runtime::is_valid_layer_major_prefill_full_attention_tactic(
          runtime::LayerMajorPrefillFullAttentionTactic::
              kExactSegmentedC512) &&
          runtime::is_valid_layer_major_prefill_full_attention_tactic(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ64Panel) &&
          runtime::is_valid_layer_major_prefill_full_attention_tactic(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ128V4Panel) &&
          !runtime::is_valid_layer_major_prefill_full_attention_tactic(
              static_cast<runtime::LayerMajorPrefillFullAttentionTactic>(
                  0xffU)),
      "layer-major Attention tactics are a closed engine-lifetime set");
  test.expect(
      runtime::to_string(
          runtime::LayerMajorPrefillFullAttentionTactic::
              kExactSegmentedC512) == "exact-segmented" &&
          runtime::to_string(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ64Panel) == "native-group-q64-panel" &&
          runtime::to_string(
              runtime::LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ128V4Panel) ==
              "native-group-q128-v4-panel",
      "Attention tactic names preserve exact Q64/Q128 route identity");
  test.expect(
      runtime::is_valid_layer_major_prefill_projection_tactic(
          runtime::LayerMajorPrefillProjectionTactic::kExactSegmentedC512) &&
          runtime::is_valid_layer_major_prefill_projection_tactic(
              runtime::LayerMajorPrefillProjectionTactic::
                  kSegmentedMarlinOperatorPanel) &&
          !runtime::is_valid_layer_major_prefill_projection_tactic(
              static_cast<runtime::LayerMajorPrefillProjectionTactic>(
                  0xffU)),
      "layer-major projection tactics are a closed engine-lifetime set");
  const runtime::PrefillExecutionPlanResult result = build_plan(513U);
  test.expect(result &&
                  result.value->legacy_public_tile_limit == 512U &&
                  result.value->operator_panel_capacity == 8'192U &&
                  result.value->panel_count == 1U &&
                  result.value->panels[0].token_count == 513U &&
                  !result.value->operator_bindings_complete &&
                  !result.value->executable(),
              "P513 is one unbound operator panel and never inherits C512 "
              "execution semantics");

  test.expect(runtime::kMaximumRequestPrefillChunkSize == 512U &&
                  runtime::kLayerMajorPrefillLegacyPublicTileTokens == 512U &&
                  runtime::kAbsoluteRequestMaxSequenceLength ==
                      runtime::kLayerMajorPrefillMaximumSequenceTokens &&
                  runtime::kRequestLayerCount ==
                      runtime::kLayerMajorPrefillLayerCount &&
                  runtime::kLayerMajorPrefillOperatorPanelTokens == 8'192U &&
                  runtime::kLayerMajorPrefillMaximumPanelCount == 32U,
              "legacy tile and layer-major panel constants remain distinct");
}

void test_target_panel_matrix(TestContext& test) {
  expect_panel_shape(test, 1U, 1U, 1U);
  expect_panel_shape(test, 512U, 1U, 512U);
  expect_panel_shape(test, 513U, 1U, 513U);
  expect_panel_shape(test, 8'192U, 1U, 8'192U);
  expect_panel_shape(test, 8'193U, 2U, 4'096U);
  expect_panel_shape(test, 40'000U, 5U, 7'712U);
  expect_panel_shape(test, 60'000U, 8U, 5'424U);
  expect_panel_shape(test, 130'000U, 16U, 7'656U);
  expect_panel_shape(test, 262'144U, 32U, 8'192U);

  const runtime::PrefillExecutionPlanResult offset =
      build_plan(8'193U, 512U, 16'384U);
  test.expect(offset && offset.value->first_position == 512U &&
                  offset.value->panels[0].first_position == 512U &&
                  offset.value->panels[0].end_position == 4'609U &&
                  offset.value->panels[1].first_position == 4'609U &&
                  offset.value->panels[1].end_position == 8'705U &&
                  offset.value->final_position == 8'705U,
              "nonzero request positions preserve continuous absolute panel "
              "coordinates");
}

[[nodiscard]] std::vector<std::size_t> physical_segment_schedule(
    std::size_t remaining_panel_tokens) {
  std::vector<std::size_t> schedule;
  while (remaining_panel_tokens != 0U) {
    const std::size_t segment =
        runtime::next_prefill_physical_segment_token_count(
            remaining_panel_tokens);
    if (!runtime::is_prefill_physical_segment_token_count(segment) ||
        segment > remaining_panel_tokens) {
      return {};
    }
    schedule.push_back(segment);
    remaining_panel_tokens -= segment;
  }
  return schedule;
}

[[nodiscard]] std::vector<std::size_t> layer_major_physical_segment_schedule(
    std::size_t remaining_panel_tokens) {
  std::vector<std::size_t> schedule;
  while (remaining_panel_tokens != 0U) {
    const std::size_t segment =
        runtime::next_layer_major_prefill_physical_segment_token_count(
            remaining_panel_tokens);
    if (!runtime::is_layer_major_prefill_physical_segment_token_count(
            segment) ||
        segment > remaining_panel_tokens) {
      return {};
    }
    schedule.push_back(segment);
    remaining_panel_tokens -= segment;
  }
  return schedule;
}

void test_balanced_physical_segment_contract(TestContext& test) {
  test.expect(
      physical_segment_schedule(33U) ==
              std::vector<std::size_t>({32U, 1U}) &&
          physical_segment_schedule(257U) ==
              std::vector<std::size_t>({256U, 1U}) &&
          physical_segment_schedule(513U) ==
              std::vector<std::size_t>({512U, 1U}),
      "the legacy canonical physical schedule remains byte-stable");

  test.expect(
      layer_major_physical_segment_schedule(1U) ==
              std::vector<std::size_t>({1U}) &&
          layer_major_physical_segment_schedule(33U) ==
              std::vector<std::size_t>({33U}) &&
          layer_major_physical_segment_schedule(255U) ==
              std::vector<std::size_t>({255U}) &&
          layer_major_physical_segment_schedule(257U) ==
              std::vector<std::size_t>({257U}) &&
          layer_major_physical_segment_schedule(511U) ==
              std::vector<std::size_t>({511U}) &&
          layer_major_physical_segment_schedule(512U) ==
              std::vector<std::size_t>({512U}) &&
          layer_major_physical_segment_schedule(513U) ==
              std::vector<std::size_t>({257U, 256U}) &&
          layer_major_physical_segment_schedule(1'024U) ==
              std::vector<std::size_t>({512U, 512U}) &&
          layer_major_physical_segment_schedule(1'025U) ==
              std::vector<std::size_t>({512U, 257U, 256U}),
      "layer-major segments preserve full blocks and balance only the final "
      "full-plus-tail pair");

  constexpr std::array<std::uint64_t, 5U> kPanelBoundaryPrompts{
      8'191U, 8'192U, 8'193U, 130'000U,
      runtime::kLayerMajorPrefillMaximumSequenceTokens};
  for (const std::uint64_t prompt_tokens : kPanelBoundaryPrompts) {
    const runtime::PrefillExecutionPlanResult result = build_plan(prompt_tokens);
    bool exact = result.ok();
    std::uint64_t next_position = 0U;
    if (result) {
      for (std::size_t panel_index = 0U;
           panel_index < result.value->panel_count; ++panel_index) {
        const runtime::PrefillOperatorPanel& panel =
            result.value->panels[panel_index];
        exact = exact && panel.first_position == next_position;
        std::size_t remaining = panel.token_count;
        std::uint64_t segment_position = panel.first_position;
        while (remaining != 0U) {
          const std::size_t segment =
              runtime::next_layer_major_prefill_physical_segment_token_count(
                  remaining);
          exact = exact &&
                  runtime::is_layer_major_prefill_physical_segment_token_count(
                      segment) &&
                  segment != 0U && segment <= remaining &&
                  segment_position + segment <= panel.end_position;
          segment_position += segment;
          remaining -= segment;
        }
        exact = exact && segment_position == panel.end_position;
        next_position = panel.end_position;
      }
    }
    test.expect(exact && next_position == prompt_tokens,
                "physical segments are continuous and stay inside C8192 panels");
  }

  test.expect(
      runtime::next_layer_major_prefill_physical_segment_token_count(0U) ==
              0U &&
          !runtime::is_layer_major_prefill_physical_segment_token_count(0U) &&
          runtime::is_layer_major_prefill_physical_segment_token_count(33U) &&
          runtime::is_layer_major_prefill_physical_segment_token_count(255U) &&
          runtime::is_layer_major_prefill_physical_segment_token_count(511U) &&
          !runtime::is_layer_major_prefill_physical_segment_token_count(513U),
      "the layer-major schedule admits every nonzero C1..C512 geometry");
}

void expect_arithmetic_span_ledger(
    TestContext& test, const std::size_t panel_tokens,
    const std::vector<std::size_t>& expected_counts) {
  const runtime::LayerMajorPrefillArithmeticSpanLedger ledger =
      runtime::make_layer_major_prefill_arithmetic_span_ledger(panel_tokens);
  bool exact =
      runtime::is_valid_layer_major_prefill_arithmetic_span_ledger(ledger) &&
      ledger.token_count == panel_tokens &&
      ledger.span_count == expected_counts.size();
  std::size_t expected_offset = 0U;
  if (exact) {
    for (std::size_t index = 0U; index < expected_counts.size(); ++index) {
      exact = exact && ledger.spans[index].token_offset == expected_offset &&
              ledger.spans[index].token_count == expected_counts[index];
      expected_offset += expected_counts[index];
    }
  }
  test.expect(exact && expected_offset == panel_tokens &&
                  expected_counts ==
                      layer_major_physical_segment_schedule(panel_tokens),
              "arithmetic ledger exactly preserves the compatibility span "
              "sequence");
}

void test_exact_arithmetic_span_ledgers(TestContext& test) {
  expect_arithmetic_span_ledger(test, 513U, {257U, 256U});
  expect_arithmetic_span_ledger(test, 1'025U, {512U, 257U, 256U});
  expect_arithmetic_span_ledger(
      test, 8'192U, std::vector<std::size_t>(
                        runtime::kLayerMajorPrefillMaximumArithmeticSpanCount,
                        512U));

  const runtime::PrefillExecutionPlanResult p8193 = build_plan(8'193U);
  bool balanced_panels = p8193 && p8193.value->panel_count == 2U;
  if (balanced_panels) {
    const runtime::PrefillOperatorPanel& first = p8193.value->panels[0];
    const runtime::PrefillOperatorPanel& second = p8193.value->panels[1];
    balanced_panels = first.first_position == 0U &&
                      first.token_count == 4'097U &&
                      first.end_position == 4'097U &&
                      second.first_position == first.end_position &&
                      second.token_count == 4'096U &&
                      second.end_position == 8'193U;
    expect_arithmetic_span_ledger(
        test, first.token_count,
        {512U, 512U, 512U, 512U, 512U, 512U, 512U, 257U, 256U});
    expect_arithmetic_span_ledger(
        test, second.token_count,
        {512U, 512U, 512U, 512U, 512U, 512U, 512U, 512U});
    const runtime::LayerMajorPrefillArithmeticSpanLedger first_ledger =
        runtime::make_layer_major_prefill_arithmetic_span_ledger(
            first.token_count);
    const runtime::LayerMajorPrefillArithmeticSpanLedger second_ledger =
        runtime::make_layer_major_prefill_arithmetic_span_ledger(
            second.token_count);
    balanced_panels =
        balanced_panels &&
        first.first_position + first_ledger.token_count ==
            first.end_position &&
        second.first_position + second_ledger.token_count ==
            second.end_position;
  }
  test.expect(balanced_panels,
              "P8193 panel and arithmetic ledgers continuously cover the "
              "whole prompt");

  runtime::LayerMajorPrefillArithmeticSpanLedger noncanonical =
      runtime::make_layer_major_prefill_arithmetic_span_ledger(513U);
  noncanonical.spans[0] = {0U, 256U};
  noncanonical.spans[1] = {256U, 257U};
  test.expect(
      !runtime::is_valid_layer_major_prefill_arithmetic_span_ledger(
          noncanonical),
      "ledger validator rejects a contiguous but noncanonical P513 split");
  test.expect(runtime::is_valid_layer_major_prefill_arithmetic_contract(
                  runtime::kLayerMajorPrefillExactArithmeticContract),
              "the bound arithmetic contract is explicit and immutable");
}

void test_fixed_layer_schedule(TestContext& test) {
  const runtime::PrefillExecutionPlanResult result = build_plan(40'000U);
  test.expect(result.ok(), "40K layer-major topology builds");
  if (!result) {
    return;
  }

  const runtime::PrefillExecutionPlan& plan = *result.value;
  std::size_t linear_layers = 0U;
  std::size_t full_layers = 0U;
  bool exact_schedule = true;
  for (std::size_t layer_index = 0U; layer_index < plan.layers.size();
       ++layer_index) {
    const runtime::PrefillLayerExecution& layer = plan.layers[layer_index];
    const bool expected_full = (layer_index + 1U) % 4U == 0U;
    const model::LayerType expected_type =
        expected_full ? model::LayerType::kFullAttention
                      : model::LayerType::kLinearAttention;
    const runtime::PrefillProgressDomain expected_progress =
        expected_full ? runtime::PrefillProgressDomain::kKvCache
                      : runtime::PrefillProgressDomain::kGdnState;
    exact_schedule = exact_schedule && layer.layer_index == layer_index &&
                     layer.layer_type == expected_type &&
                     layer.progress_domain == expected_progress &&
                     layer.panel_count == plan.panel_count;
    if (expected_full) {
      ++full_layers;
    } else {
      ++linear_layers;
    }
  }
  test.expect(exact_schedule && linear_layers == 48U && full_layers == 16U &&
                  plan.layers[3].layer_type ==
                      model::LayerType::kFullAttention &&
                  plan.layers[63].layer_type ==
                      model::LayerType::kFullAttention,
              "all 64 layers retain the fixed 48-GDN/16-attention schedule");
  test.expect(plan.final_commit.expected_initial_sequence_length == 0U &&
                  plan.final_commit.committed_sequence_length == 40'000U &&
                  plan.final_commit.commit_count == 1U,
              "the immutable plan declares exactly one final state commit");
}

void test_public_unbound_topology_validator(TestContext& test) {
  const runtime::PrefillExecutionPlanResult result =
      build_plan(40'000U, 512U, 65'536U);
  test.expect(result.ok(), "unbound validator fixture builds");
  if (!result) {
    return;
  }

  const runtime::PrefillExecutionPlan baseline = *result.value;
  test.expect(
      runtime::is_valid_unbound_layer_major_prefill_execution_plan(baseline),
      "the public validator accepts a complete unbound layer-major plan");

  const auto expect_rejected = [&](runtime::PrefillExecutionPlan candidate,
                                   const char* const message) {
    test.expect(
        !runtime::is_valid_unbound_layer_major_prefill_execution_plan(
            candidate),
        message);
  };

  runtime::PrefillExecutionPlan candidate = baseline;
  candidate.traversal = static_cast<runtime::PrefillTraversalOrder>(0xffU);
  expect_rejected(candidate, "the unbound validator rejects traversal drift");

  candidate = baseline;
  candidate.legacy_public_tile_limit += 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects legacy capacity drift");

  candidate = baseline;
  candidate.operator_panel_capacity -= 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects operator capacity drift");

  candidate = baseline;
  candidate.prompt_token_count -= 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects prompt-span drift");

  candidate = baseline;
  candidate.panel_count -= 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects an incomplete panel set");

  candidate = baseline;
  candidate.panels[0].token_count -= 1U;
  candidate.panels[0].end_position -= 1U;
  candidate.panels[1].first_position -= 1U;
  expect_rejected(candidate,
                  "the unbound validator rejects non-canonical panel sizes");

  candidate = baseline;
  candidate.panels[1].ordinal = 0U;
  expect_rejected(candidate,
                  "the unbound validator rejects panel-order drift");

  candidate = baseline;
  candidate.layers[7].layer_index = 8U;
  expect_rejected(candidate,
                  "the unbound validator rejects layer-order drift");

  candidate = baseline;
  candidate.layers[3].layer_type = model::LayerType::kLinearAttention;
  expect_rejected(candidate,
                  "the unbound validator rejects the fixed layer schedule");

  candidate = baseline;
  candidate.layers[3].progress_domain =
      runtime::PrefillProgressDomain::kGdnState;
  expect_rejected(candidate,
                  "the unbound validator rejects progress-domain drift");

  candidate = baseline;
  candidate.final_commit.commit_count = 2U;
  expect_rejected(candidate,
                  "the unbound validator rejects final-commit drift");

  candidate = baseline;
  candidate.operator_bindings_complete = true;
  expect_rejected(candidate,
                  "the unbound validator rejects operator-bound plans");

  candidate = baseline;
  candidate.first_position =
      runtime::kLayerMajorPrefillMaximumSequenceTokens;
  candidate.final_position = candidate.first_position + 1U;
  candidate.prompt_token_count = 1U;
  candidate.panel_count = 1U;
  candidate.panels[0] = runtime::PrefillOperatorPanel{
      0U, candidate.first_position, 1U, candidate.final_position};
  candidate.final_commit = runtime::PrefillFinalCommitPlan{
      candidate.first_position, candidate.final_position, 1U};
  for (runtime::PrefillLayerExecution& layer : candidate.layers) {
    layer.panel_count = 1U;
  }
  expect_rejected(candidate,
                  "the unbound validator rejects an out-of-capacity span");
}

void test_strict_layer_major_progress(TestContext& test) {
  const runtime::PrefillExecutionPlanResult result = build_plan(8'193U);
  test.expect(result.ok(), "two-panel progress topology builds");
  if (!result) {
    return;
  }
  const runtime::PrefillExecutionPlan& plan = *result.value;
  runtime::PrefillExecutionProgress progress =
      runtime::make_prefill_execution_progress(plan);

  bool initialized = progress.next_layer == 0U &&
                     progress.next_panel == 0U &&
                     !progress.final_hidden_ready &&
                     !progress.prefill_state_committed;
  for (std::size_t layer = 0U; layer < plan.layers.size(); ++layer) {
    initialized = initialized && progress.kv_visible_end[layer] == 0U &&
                  progress.gdn_advanced_end[layer] == 0U &&
                  progress.completed_panels[layer] == 0U;
  }
  test.expect(initialized,
              "request-owned KV/GDN progress begins at the admitted position");

  runtime::PrefillExecutionProgress before = progress;
  auto status = runtime::advance_prefill_progress_after_completion(
      plan, progress, 1U, 0U);
  test.expect(status == runtime::PrefillExecutionProgressError::kOutOfOrder &&
                  same_progress(progress, before),
              "a later layer cannot begin before the current layer closes");

  status = runtime::advance_prefill_progress_after_completion(
      plan, progress, 0U, 1U);
  test.expect(status == runtime::PrefillExecutionProgressError::kOutOfOrder &&
                  same_progress(progress, before),
              "a later panel cannot skip its predecessor");

  status = runtime::advance_prefill_progress_after_completion(
      plan, progress, plan.layers.size(), 0U);
  test.expect(status ==
                      runtime::PrefillExecutionProgressError::kLayerOutOfRange &&
                  same_progress(progress, before),
              "an out-of-range layer fails without changing progress");

  status = runtime::advance_prefill_progress_after_completion(
      plan, progress, 0U, plan.panel_count);
  test.expect(status ==
                      runtime::PrefillExecutionProgressError::kPanelOutOfRange &&
                  same_progress(progress, before),
              "an out-of-range panel fails without changing progress");

  test.expect(runtime::mark_prefill_final_hidden_ready(plan, progress) ==
                      runtime::PrefillExecutionProgressError::
                          kExecutionIncomplete &&
                  runtime::publish_prefill_state_committed(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kCommitNotReady,
              "final hidden and commit remain unavailable before all layers");

  bool ordered = true;
  for (std::size_t layer = 0U; layer < plan.layers.size(); ++layer) {
    for (std::size_t panel = 0U; panel < plan.panel_count; ++panel) {
      ordered = ordered &&
                runtime::advance_prefill_progress_after_completion(
                    plan, progress, layer, panel) ==
                    runtime::PrefillExecutionProgressError::kNone;
    }
  }
  test.expect(ordered && progress.next_layer == plan.layers.size() &&
                  progress.next_panel == 0U,
              "progress follows strict layer-then-panel traversal");

  bool domains_exact = true;
  for (std::size_t layer = 0U; layer < plan.layers.size(); ++layer) {
    const bool full = plan.layers[layer].progress_domain ==
                      runtime::PrefillProgressDomain::kKvCache;
    domains_exact =
        domains_exact && progress.completed_panels[layer] == plan.panel_count &&
        (full ? progress.kv_visible_end[layer] == plan.final_position
              : progress.kv_visible_end[layer] == plan.first_position) &&
        (full ? progress.gdn_advanced_end[layer] == plan.first_position
              : progress.gdn_advanced_end[layer] == plan.final_position);
  }
  test.expect(domains_exact,
              "each layer advances only its declared KV or GDN progress");

  test.expect(!runtime::prefill_final_commit_ready(plan, progress) &&
                  runtime::publish_prefill_state_committed(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kCommitNotReady,
              "complete state progress still requires the final hidden handoff");
  test.expect(runtime::mark_prefill_final_hidden_ready(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kNone &&
                  runtime::prefill_final_commit_ready(plan, progress),
              "final hidden readiness opens the single logical commit gate");
  test.expect(runtime::publish_prefill_state_committed(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kNone &&
                  progress.prefill_state_committed &&
                  !runtime::prefill_final_commit_ready(plan, progress) &&
                  runtime::publish_prefill_state_committed(plan, progress) ==
                      runtime::PrefillExecutionProgressError::kAlreadyCommitted,
              "the logical PrefillStateCommitted transition is single-shot");
}

void test_fail_closed_inputs(TestContext& test) {
  test.expect(!build_plan(0U) &&
                  build_plan(0U).error ==
                      runtime::PrefillExecutionPlanError::kInvalidArgument,
              "zero-token plans fail closed");
  test.expect(!build_plan(1U, 0U, 0U) &&
                  build_plan(1U, 0U, 0U).error ==
                      runtime::PrefillExecutionPlanError::kInvalidArgument,
              "zero sequence capacity fails closed");
  test.expect(!build_plan(1U, 0U,
                          runtime::kLayerMajorPrefillMaximumSequenceTokens +
                              1U) &&
                  build_plan(1U, 0U,
                             runtime::kLayerMajorPrefillMaximumSequenceTokens +
                                 1U)
                          .error == runtime::PrefillExecutionPlanError::
                                        kInvalidArgument,
              "capacity above the fixed request contract fails closed");
  test.expect(!build_plan(2U, 9U, 10U) &&
                  build_plan(2U, 9U, 10U).error ==
                      runtime::PrefillExecutionPlanError::kCapacityExceeded,
              "prompt spans cannot exceed the admitted request capacity");
  test.expect(!build_plan(
                  runtime::kLayerMajorPrefillMaximumSequenceTokens + 1U) &&
                  build_plan(
                      runtime::kLayerMajorPrefillMaximumSequenceTokens + 1U)
                          .error == runtime::PrefillExecutionPlanError::
                                        kCapacityExceeded,
              "prompt spans cannot exceed the absolute request capacity");

  runtime::PrefillExecutionPlanOptions overflow;
  overflow.first_position = std::numeric_limits<std::uint64_t>::max();
  overflow.prompt_token_count = 1U;
  overflow.max_sequence_length =
      runtime::kLayerMajorPrefillMaximumSequenceTokens;
  const runtime::PrefillExecutionPlanResult overflow_result =
      runtime::build_unbound_layer_major_prefill_execution_plan(overflow);
  test.expect(!overflow_result &&
                  overflow_result.error ==
                      runtime::PrefillExecutionPlanError::kArithmeticOverflow,
              "position arithmetic overflow fails before narrowing");

  runtime::PrefillExecutionPlan invalid_plan;
  runtime::PrefillExecutionProgress invalid_progress =
      runtime::make_prefill_execution_progress(invalid_plan);
  test.expect(runtime::advance_prefill_progress_after_completion(
                  invalid_plan, invalid_progress, 0U, 0U) ==
                  runtime::PrefillExecutionProgressError::kInvalidPlan,
              "default or malformed topologies cannot advance");
}

}  // namespace

int main() {
  TestContext test;
  test_public_tile_and_operator_panel_are_independent(test);
  test_target_panel_matrix(test);
  test_balanced_physical_segment_contract(test);
  test_exact_arithmetic_span_ledgers(test);
  test_fixed_layer_schedule(test);
  test_public_unbound_topology_validator(test);
  test_strict_layer_major_progress(test);
  test_fail_closed_inputs(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " prefill execution plan test(s) failed\n";
    return 1;
  }
  std::cout << "All prefill execution plan tests passed\n";
  return 0;
}
