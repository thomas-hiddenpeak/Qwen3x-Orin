#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::reference_runner_detail {

// Immutable candidate policy, shared by ordinary ON/OFF builds. Only the
// nonfixed generic suffix changes; GroupQ64 and the fixed launcher do not.
// This source-private identity is a compiled policy fact, not a kernel hit.
inline constexpr bool kOrdinaryExactAttentionScoreFeedEnabled = true;
inline constexpr const char* kOrdinaryExactScoreFeedPlanId =
    "q3x.sm87.legacy-c512.terminal-elision-score-feed.v2";

// Definitions exist only in BUILD_TESTING. Explicit oracle overrides are
// owned by the calling thread; an untouched worker uses the compiled policy.
enum class ExactAttentionScoreFeedForTest : std::uint8_t {
  kIncumbentQt2 = 0,
  kScoreFeed,
  kCompiledDefault,
};

[[nodiscard]] ExactAttentionScoreFeedForTest
exchange_exact_attention_score_feed_for_test(
    ExactAttentionScoreFeedForTest selection) noexcept;

// Counts accepted candidate submissions (including graph capture), not replay.
[[nodiscard]] std::size_t
exchange_exact_attention_score_feed_launch_hits_for_test(
    std::size_t hits) noexcept;

[[nodiscard]] int launch_exact_attention_score_feed_for_test_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, const std::uint16_t* gate,
    std::size_t first_position, std::size_t token_count,
    std::uint16_t* output, void* cuda_stream) noexcept;

[[nodiscard]] int query_exact_attention_score_feed_resources_for_test_cuda(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

}  // namespace q3x::runtime::reference_runner_detail
