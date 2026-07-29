#pragma once

#include <cstddef>

namespace q3x::runtime::reference_runner_detail {

// Private test-only controls for the exact C512 NVFP4 Marlin Gate/Up
// architecture admission.  They intentionally stay outside installed public
// headers so the candidate cannot become a supported runtime route before it
// passes the real-model external performance gate.
[[nodiscard]] bool exchange_prefill_nvfp4_marlin_pair_admission_test_enabled(
    bool enabled) noexcept;

[[nodiscard]] std::size_t
exchange_prefill_nvfp4_marlin_pair_admission_test_hits(
    std::size_t hits) noexcept;

}  // namespace q3x::runtime::reference_runner_detail
