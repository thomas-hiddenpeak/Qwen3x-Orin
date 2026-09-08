#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::reference_runner_detail {

// Source-local BUILD_TESTING admission only. The production default, GroupQ64
// prefix, public ABI, and request state are unchanged. The calling inference
// thread owns the selection; graph replay retains the captured kernel.
enum class ExactAttentionScoreFeedForTest : std::uint8_t {
  kIncumbentQt2 = 0,
  kScoreFeed,
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
