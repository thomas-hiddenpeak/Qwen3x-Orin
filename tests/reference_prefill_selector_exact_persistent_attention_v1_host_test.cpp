#include "reference_runner_selector_exact_persistent_attention_v1_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

namespace detail =
    q3x::runtime::reference_runner_detail;

class TestContext final {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

void expect_plan(TestContext& test, const std::size_t prompt_tokens,
                 const std::uint32_t expected_spans,
                 const std::uint32_t expected_group_spans,
                 const std::uint32_t expected_generic_spans,
                 const std::uint32_t expected_suffix_offset,
                 const std::uint32_t expected_suffix_tokens,
                 const std::uint32_t expected_submissions,
                 const std::uint32_t expected_minimum_physical_tokens,
                 const std::uint32_t expected_maximum_physical_tokens) {
  const detail::SelectorExactPersistentAttentionV1Plan plan =
      detail::make_selector_exact_persistent_attention_v1_plan(
          0U, prompt_tokens);
  const std::string label = "P" + std::to_string(prompt_tokens);
  test.expect(plan.valid, label + " plan is valid");
  test.expect(plan.first_position == 0U &&
                  plan.token_count == prompt_tokens,
              label + " identity is exact");
  test.expect(plan.arithmetic_span_count == expected_spans,
              label + " arithmetic span count is exact");
  test.expect(plan.group_q64_span_count == expected_group_spans,
              label + " GroupQ64 span count is exact");
  test.expect(plan.generic_qt2_span_count == expected_generic_spans,
              label + " GenericQT2 span count is exact");
  test.expect(plan.generic_suffix_token_offset == expected_suffix_offset &&
                  plan.generic_suffix_first_position ==
                      expected_suffix_offset &&
                  plan.generic_suffix_token_count == expected_suffix_tokens,
              label + " GenericQT2 suffix is one exact contiguous range");
  test.expect(plan.generic_suffix_token_count %
                      detail::kSelectorExactPersistentAttentionV1QueryTokens ==
                  0U,
              label + " GenericQT2 suffix is Q8-aligned");
  test.expect(plan.physical_submission_count == expected_submissions,
              label + " physical submission count is exact");
  test.expect(
      plan.minimum_physical_submission_tokens ==
              expected_minimum_physical_tokens &&
          plan.maximum_physical_submission_tokens ==
              expected_maximum_physical_tokens &&
          plan.logical_panel_tokens == prompt_tokens,
      label + " receipt reports real composite physical M");
  for (std::size_t index = 0U; index < plan.group_q64_span_count; ++index) {
    const auto& span = plan.group_q64_spans[index];
    test.expect(
        q3x::runtime::select_fixed_bulk_causal_gqa_prefill_tactic(
            span.first_position, span.token_count) ==
            q3x::runtime::FixedBulkCausalGqaPrefillTactic::kGroupQ64V3,
        label + " prefix span " + std::to_string(index) +
            " is selected by incumbent GroupQ64");
  }
}

}  // namespace

int main() {
  TestContext test;
  test.expect(
      detail::kSelectorExactPersistentAttentionV1PersistentBlocksPerKvHead ==
              4U &&
          detail::kSelectorExactPersistentAttentionV1PersistentBlockCount ==
              16U,
      "persistent suffix uses a fixed 4x4 CTA grid");
  expect_plan(test, 513U, 2U, 1U, 1U, 257U, 256U, 2U, 256U,
              257U);
  expect_plan(test, 4'096U, 8U, 2U, 6U, 1'024U, 3'072U, 3U,
              512U, 3'072U);
  expect_plan(test, 8'192U, 16U, 2U, 14U, 1'024U, 7'168U, 3U,
              512U, 7'168U);
  expect_plan(test, 40'000U, 80U, 2U, 78U, 1'024U, 38'976U, 3U,
              512U, 38'976U);
  test.expect(
      detail::is_selector_exact_persistent_attention_v1_plan_prompt_tokens(
          40'000U) &&
          !detail::
              is_selector_exact_persistent_attention_v1_runner_prompt_tokens(
                  40'000U),
      "P40000 is an architecture target plan, not a runner-admitted route");
  for (const std::size_t prompt_tokens : {513U, 4'096U, 8'192U}) {
    test.expect(
        detail::
            is_selector_exact_persistent_attention_v1_runner_prompt_tokens(
                prompt_tokens),
        "short qualification shape is runner-admitted");
  }

  std::uint32_t aggregate_minimum = 0U;
  std::uint32_t aggregate_maximum = 0U;
  std::uint32_t aggregate_maximum_logical = 0U;
  for (const std::size_t prompt_tokens : {513U, 4'096U, 8'192U}) {
    const auto plan =
        detail::make_selector_exact_persistent_attention_v1_plan(
            0U, prompt_tokens);
    aggregate_minimum =
        aggregate_minimum == 0U
            ? plan.minimum_physical_submission_tokens
            : std::min(aggregate_minimum,
                       plan.minimum_physical_submission_tokens);
    aggregate_maximum =
        std::max(aggregate_maximum,
                 plan.maximum_physical_submission_tokens);
    aggregate_maximum_logical =
        std::max(aggregate_maximum_logical, plan.logical_panel_tokens);
  }
  test.expect(aggregate_minimum == 256U && aggregate_maximum == 7'168U &&
                  aggregate_maximum_logical == 8'192U,
              "P513/P4096/P8192 aggregate physical-M receipt is exact");

  for (const std::size_t invalid : {0U, 512U, 514U, 1'024U, 8'193U,
                                    39'999U, 40'001U}) {
    test.expect(
        !detail::make_selector_exact_persistent_attention_v1_plan(0U, invalid)
             .valid,
        "unsupported prompt length P" + std::to_string(invalid) +
            " fails closed");
  }
  test.expect(
      !detail::make_selector_exact_persistent_attention_v1_plan(1U, 513U)
           .valid,
      "nonzero first position fails closed");

  test.expect(
      q3x::runtime::select_fixed_bulk_causal_gqa_prefill_tactic(0U, 257U) ==
          q3x::runtime::FixedBulkCausalGqaPrefillTactic::kGroupQ64V3 &&
          q3x::runtime::select_fixed_bulk_causal_gqa_prefill_tactic(
              257U, 256U) ==
              q3x::runtime::FixedBulkCausalGqaPrefillTactic::kGenericQt2 &&
          q3x::runtime::select_fixed_bulk_causal_gqa_prefill_tactic(
              512U, 512U) ==
              q3x::runtime::FixedBulkCausalGqaPrefillTactic::kGroupQ64V3 &&
          q3x::runtime::select_fixed_bulk_causal_gqa_prefill_tactic(
              1'024U, 512U) ==
              q3x::runtime::FixedBulkCausalGqaPrefillTactic::kGenericQt2,
      "incumbent selector remains absolute-position exact");

  test.expect(
      !detail::exchange_selector_exact_persistent_attention_v1_for_test(false),
      "candidate test seam defaults disabled");
  test.expect(
      !detail::exchange_selector_exact_persistent_attention_v1_for_test(true),
      "candidate test seam reports the disabled predecessor");
  test.expect(
      detail::exchange_selector_exact_persistent_attention_v1_for_test(false),
      "candidate test seam round-trips enabled state");

  detail::SelectorExactPersistentAttentionV1RouteReceipt seeded;
  seeded.panel_calls = 7U;
  const auto initial = detail::
      exchange_selector_exact_persistent_attention_v1_route_receipt_for_test(
          seeded);
  const auto returned = detail::
      exchange_selector_exact_persistent_attention_v1_route_receipt_for_test(
          {});
  test.expect(initial.panel_calls == 0U && returned.panel_calls == 7U,
              "candidate route receipt exchange is isolated and exact");

  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " selector-exact persistent Attention host checks failed\n";
    return 1;
  }
  std::cout << "selector-exact-persistent-attention-v1 host contract PASS\n";
  return 0;
}
