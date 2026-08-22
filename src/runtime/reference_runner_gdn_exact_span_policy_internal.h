#pragma once

#include "q3x/runtime/model_weights.h"

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::reference_runner_detail {

inline constexpr std::size_t kReferenceRunnerGdnExactSpanAlignment = 16U;

// Source-local production selector. The exact-span recurrence has current
// whole-product authority only for the scheduler's C256/C512 bulk tiles.
[[nodiscard]] constexpr bool should_use_prefill_gdn_exact_span(
    const ProjectionBackend backend, const std::uint32_t first_position,
    const std::size_t token_count) noexcept {
  return backend == ProjectionBackend::kSm87WeightOnly &&
         (token_count == 256U || token_count == 512U) &&
         first_position % kReferenceRunnerGdnExactSpanAlignment == 0U;
}

}  // namespace q3x::runtime::reference_runner_detail
