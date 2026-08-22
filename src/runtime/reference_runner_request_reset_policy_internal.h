#pragma once

#include "q3x/runtime/request_state.h"

#include <cstdint>

namespace q3x::runtime::reference_runner_request_reset_detail {

// Internal lifecycle fact, never a user- or environment-selected execution
// tactic. Only the ordinary legacy production RequestState profile is eligible
// for the prefix reset; candidate profiles preserve their established full
// reset boundary.
enum class RequestReuseBoundary : std::uint8_t {
  kKnownClean = 0,
  kCommitted,
  kUncertain,
};
static_assert(static_cast<std::uint8_t>(RequestReuseBoundary::kKnownClean) ==
              0U);

struct RequestResetDecision {
  RequestStateResetMode mode = RequestStateResetMode::kConservativeFull;
  std::uint32_t committed_positions = 0U;
};

[[nodiscard]] constexpr RequestResetDecision select_request_reset(
    const RequestReuseBoundary boundary,
    const std::uint32_t committed_positions,
    const std::uint32_t current_position,
    const std::uint32_t max_sequence_length,
    const RequestMemoryProfile memory_profile,
    const bool poisoned,
    const bool whole_request_active) noexcept {
  if (memory_profile != RequestMemoryProfile::kLegacyC512 || poisoned ||
      whole_request_active || max_sequence_length == 0U ||
      current_position > max_sequence_length) {
    return {};
  }
  if (boundary == RequestReuseBoundary::kKnownClean &&
      committed_positions == 0U && current_position == 0U) {
    return {RequestStateResetMode::kAlreadyClean, 0U};
  }
  if (boundary == RequestReuseBoundary::kCommitted &&
      committed_positions != 0U &&
      committed_positions == current_position) {
    return {RequestStateResetMode::kCommittedDirtyPrefix,
            committed_positions};
  }
  return {};
}

}  // namespace q3x::runtime::reference_runner_request_reset_detail
