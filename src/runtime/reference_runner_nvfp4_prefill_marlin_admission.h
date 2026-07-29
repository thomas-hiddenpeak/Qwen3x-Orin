#pragma once

#include <cstddef>

namespace q3x::runtime::reference_runner_detail {

// Private test-only controls for the exact-C512 fused NVFP4 Gate+Up route.
// They are deliberately absent from the installed ReferenceRunner API.
[[nodiscard]] bool exchange_prefill_nvfp4_marlin_admission_test_enabled(
    bool enabled) noexcept;

[[nodiscard]] std::size_t
exchange_prefill_nvfp4_marlin_admission_test_hits(std::size_t hits) noexcept;

}  // namespace q3x::runtime::reference_runner_detail
