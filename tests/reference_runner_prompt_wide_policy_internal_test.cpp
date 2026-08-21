#include "reference_runner_prompt_wide_policy_internal.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <thread>

namespace {

namespace detail = q3x::runtime::reference_runner_detail;

using Policy = detail::ReferenceRunnerPromptWidePolicy;
using TestPolicy = detail::ReferenceRunnerPromptWidePolicyForTest;

static_assert(
    !detail::kReferenceRunnerPromptWideProductionPolicy.embedding);
static_assert(!detail::kReferenceRunnerPromptWideProductionPolicy
                   .full_attention_preprocess);

struct PolicyCase {
  TestPolicy test_policy = TestPolicy::kProductionDefault;
  Policy expected{};
};

[[nodiscard]] constexpr bool same_policy(const Policy left,
                                         const Policy right) noexcept {
  return left.embedding == right.embedding &&
         left.full_attention_preprocess ==
             right.full_attention_preprocess;
}

}  // namespace

int main() {
  constexpr std::array<PolicyCase, 5U> cases{{
      {TestPolicy::kProductionDefault, {false, false}},
      {TestPolicy::kLegacy, {false, false}},
      {TestPolicy::kEmbeddingOnly, {true, false}},
      {TestPolicy::kFullAttentionPreprocessOnly, {false, true}},
      {TestPolicy::kEmbeddingAndFullAttentionPreprocess, {true, true}},
  }};

  bool exact = true;
  for (const PolicyCase& policy_case : cases) {
    exact = detail::is_valid_reference_runner_prompt_wide_policy_for_test(
                policy_case.test_policy) &&
            same_policy(
                detail::select_reference_runner_prompt_wide_policy(
                    policy_case.test_policy),
                policy_case.expected) &&
            exact;
  }
  constexpr TestPolicy invalid = static_cast<TestPolicy>(0xFFU);
  exact =
      !detail::is_valid_reference_runner_prompt_wide_policy_for_test(invalid) &&
      same_policy(detail::select_reference_runner_prompt_wide_policy(invalid),
                  detail::kReferenceRunnerPromptWideProductionPolicy) &&
      exact;
  exact =
      !detail::use_reference_runner_full_attention_prompt_wide_launch(
          {false, false}, 512U) &&
      !detail::use_reference_runner_full_attention_prompt_wide_launch(
          {false, true}, 1U) &&
      detail::use_reference_runner_full_attention_prompt_wide_launch(
          {false, true}, 2U) &&
      detail::use_reference_runner_full_attention_prompt_wide_launch(
          {false, true}, 512U) &&
      exact;
  (void)&detail::
      launch_full_attention_preprocess_24_4_256_64_prompt_wide_128_internal_cuda;

  const TestPolicy original =
      detail::exchange_reference_runner_prompt_wide_policy_for_test(
          TestPolicy::kEmbeddingOnly);
  const std::size_t original_embedding_hits =
      detail::
          exchange_reference_runner_prompt_wide_embedding_launch_hits_for_test(
              7U);
  const std::size_t original_attention_hits =
      detail::
          exchange_reference_runner_prompt_wide_attention_launch_hits_for_test(
              11U);
  bool worker_exact = false;
  std::thread worker([&worker_exact]() {
    const TestPolicy worker_original =
        detail::exchange_reference_runner_prompt_wide_policy_for_test(
            TestPolicy::kFullAttentionPreprocessOnly);
    const TestPolicy worker_previous =
        detail::exchange_reference_runner_prompt_wide_policy_for_test(
            TestPolicy::kProductionDefault);
    worker_exact = worker_original == TestPolicy::kProductionDefault &&
                   worker_previous ==
                       TestPolicy::kFullAttentionPreprocessOnly;
    const std::size_t worker_embedding_hits =
        detail::
            exchange_reference_runner_prompt_wide_embedding_launch_hits_for_test(
                3U);
    const std::size_t worker_attention_hits =
        detail::
            exchange_reference_runner_prompt_wide_attention_launch_hits_for_test(
                5U);
    const std::size_t worker_reset_embedding_hits =
        detail::
            exchange_reference_runner_prompt_wide_embedding_launch_hits_for_test(
                0U);
    const std::size_t worker_reset_attention_hits =
        detail::
            exchange_reference_runner_prompt_wide_attention_launch_hits_for_test(
                0U);
    worker_exact = worker_embedding_hits == 0U &&
                   worker_attention_hits == 0U &&
                   worker_reset_embedding_hits == 3U &&
                   worker_reset_attention_hits == 5U && worker_exact;
    (void)detail::exchange_reference_runner_prompt_wide_policy_for_test(
        worker_original);
  });
  worker.join();

  const TestPolicy embedding_previous =
      detail::exchange_reference_runner_prompt_wide_policy_for_test(
          TestPolicy::kFullAttentionPreprocessOnly);
  const TestPolicy attention_previous =
      detail::exchange_reference_runner_prompt_wide_policy_for_test(
          TestPolicy::kEmbeddingAndFullAttentionPreprocess);
  const TestPolicy both_previous =
      detail::exchange_reference_runner_prompt_wide_policy_for_test(
          TestPolicy::kLegacy);
  const TestPolicy legacy_previous =
      detail::exchange_reference_runner_prompt_wide_policy_for_test(original);
  const std::size_t embedding_hits =
      detail::
          exchange_reference_runner_prompt_wide_embedding_launch_hits_for_test(
              original_embedding_hits);
  const std::size_t attention_hits =
      detail::
          exchange_reference_runner_prompt_wide_attention_launch_hits_for_test(
              original_attention_hits);
  exact = original == TestPolicy::kProductionDefault &&
          original_embedding_hits == 0U && original_attention_hits == 0U &&
          embedding_hits == 7U && attention_hits == 11U &&
          embedding_previous == TestPolicy::kEmbeddingOnly &&
          attention_previous ==
              TestPolicy::kFullAttentionPreprocessOnly &&
          both_previous ==
              TestPolicy::kEmbeddingAndFullAttentionPreprocess &&
          legacy_previous == TestPolicy::kLegacy &&
          worker_exact && exact;

  if (!exact) {
    std::cerr << "reference runner prompt-wide policy matrix failed\n";
    return 1;
  }
  std::cout << "reference runner prompt-wide policy matrix passed\n";
  return 0;
}
