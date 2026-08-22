#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::reference_runner_detail {

// Source-local production policy for Decode full-attention after the exact
// fused <=64-token path has had first refusal. The split-KV numerical
// candidate is not bitwise-equivalent to the exact fallback at S65, so the
// production interval is deliberately empty. Keeping this policy compiled
// into the runner removes environment state from release-route selection.
struct ReferenceRunnerDecodeGqaPolicy {
  std::size_t split_kv_minimum_sequence_length = 0U;
  std::size_t split_kv_maximum_sequence_length = 0U;
};

enum class ReferenceRunnerDecodeGqaRoute : std::uint8_t {
  kExactFallback = 0,
  kProductionSplitKv,
};

inline constexpr ReferenceRunnerDecodeGqaPolicy
    kReferenceRunnerDecodeGqaProductionPolicy{1U, 0U};

[[nodiscard]] constexpr ReferenceRunnerDecodeGqaRoute
select_reference_runner_decode_gqa_route(
    const ReferenceRunnerDecodeGqaPolicy policy,
    const std::size_t sequence_length) noexcept {
  return sequence_length >= policy.split_kv_minimum_sequence_length &&
                 sequence_length <= policy.split_kv_maximum_sequence_length
             ? ReferenceRunnerDecodeGqaRoute::kProductionSplitKv
             : ReferenceRunnerDecodeGqaRoute::kExactFallback;
}

}  // namespace q3x::runtime::reference_runner_detail
