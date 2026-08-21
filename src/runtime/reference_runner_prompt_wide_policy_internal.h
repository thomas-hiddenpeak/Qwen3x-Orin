#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::reference_runner_detail {

// Private route policy for two bit-exact, allocation-free legacy-C512
// mechanisms.  This header is source-local so neither selector becomes part
// of the installed ReferenceRunner ABI.
struct ReferenceRunnerPromptWidePolicy {
  bool embedding = false;
  bool full_attention_preprocess = false;
};

// The Legacy-C512 production route selects both allocation-free mechanisms.
// Promotion is bounded by the real-model P514/P4096/P8192 state, public-logit,
// Decode-handoff, and exact accepted-launch witnesses. Environment variables
// have no authority to compose the release route.
inline constexpr ReferenceRunnerPromptWidePolicy
    kReferenceRunnerPromptWideProductionPolicy{true, true};

// BUILD_TESTING-only A/B authority.  Explicit independent values make every
// baseline/candidate composition visible to the host policy test.
enum class ReferenceRunnerPromptWidePolicyForTest : std::uint8_t {
  kProductionDefault = 0,
  kLegacy,
  kEmbeddingOnly,
  kFullAttentionPreprocessOnly,
  kEmbeddingAndFullAttentionPreprocess,
};

[[nodiscard]] constexpr bool
is_valid_reference_runner_prompt_wide_policy_for_test(
    const ReferenceRunnerPromptWidePolicyForTest policy) noexcept {
  switch (policy) {
    case ReferenceRunnerPromptWidePolicyForTest::kProductionDefault:
    case ReferenceRunnerPromptWidePolicyForTest::kLegacy:
    case ReferenceRunnerPromptWidePolicyForTest::kEmbeddingOnly:
    case ReferenceRunnerPromptWidePolicyForTest::
        kFullAttentionPreprocessOnly:
    case ReferenceRunnerPromptWidePolicyForTest::
        kEmbeddingAndFullAttentionPreprocess:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr ReferenceRunnerPromptWidePolicy
select_reference_runner_prompt_wide_policy(
    const ReferenceRunnerPromptWidePolicyForTest policy) noexcept {
  switch (policy) {
    case ReferenceRunnerPromptWidePolicyForTest::kProductionDefault:
      return kReferenceRunnerPromptWideProductionPolicy;
    case ReferenceRunnerPromptWidePolicyForTest::kLegacy:
      return {};
    case ReferenceRunnerPromptWidePolicyForTest::kEmbeddingOnly:
      return {true, false};
    case ReferenceRunnerPromptWidePolicyForTest::
        kFullAttentionPreprocessOnly:
      return {false, true};
    case ReferenceRunnerPromptWidePolicyForTest::
        kEmbeddingAndFullAttentionPreprocess:
      return {true, true};
  }
  return kReferenceRunnerPromptWideProductionPolicy;
}

// The retained prompt-wide Attention tactic was admitted only for M>=2.  A
// one-token tail keeps the established reference-256 launch, matching the
// historical candidate route without restoring an environment selector.
[[nodiscard]] constexpr bool
use_reference_runner_full_attention_prompt_wide_launch(
    const ReferenceRunnerPromptWidePolicy policy,
    const std::size_t token_count) noexcept {
  return policy.full_attention_preprocess && token_count >= 2U;
}

// Source-local CUDA entry point used by ReferenceRunner after the typed
// policy has selected the prompt-wide tactic.  It is deliberately absent
// from the installed decode-ops header and does not alter the public ABI.
[[nodiscard]] int
launch_full_attention_preprocess_24_4_256_64_prompt_wide_128_internal_cuda(
    const std::uint16_t* interleaved_q_gate, std::uint16_t* key,
    const std::uint16_t* q_weight, const std::uint16_t* k_weight,
    float epsilon, std::uint16_t* query_output,
    std::uint16_t* gate_output, const float* cosines, const float* sines,
    std::size_t first_position, std::size_t token_count,
    void* cuda_stream = nullptr) noexcept;

// Defined only in BUILD_TESTING q3x_kernels.  A production archive contains
// neither this exchange symbol nor mutable policy state.
[[nodiscard]] ReferenceRunnerPromptWidePolicyForTest
exchange_reference_runner_prompt_wide_policy_for_test(
    ReferenceRunnerPromptWidePolicyForTest policy) noexcept;

// BUILD_TESTING-only accepted-launch witnesses. Passing zero atomically
// returns and resets the calling thread's counter for one baseline/candidate
// Generate sample.
[[nodiscard]] std::size_t
exchange_reference_runner_prompt_wide_embedding_launch_hits_for_test(
    std::size_t hits) noexcept;

[[nodiscard]] std::size_t
exchange_reference_runner_prompt_wide_attention_launch_hits_for_test(
    std::size_t hits) noexcept;

}  // namespace q3x::runtime::reference_runner_detail
