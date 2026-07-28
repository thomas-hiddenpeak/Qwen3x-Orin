#ifndef Q3X_RUNTIME_REFERENCE_RUNNER_GDN_C16_NORM_GATE_ADMISSION_H_
#define Q3X_RUNTIME_REFERENCE_RUNNER_GDN_C16_NORM_GATE_ADMISSION_H_

#include <cstddef>

namespace q3x::runtime::reference_runner_detail {

// Private test-only controls for the exact-C16 admission route. They are not
// installed and deliberately do not extend ReferenceRunner's public API.
bool exchange_prefill_gdn_c16_norm_gate_admission_test_enabled(
    bool enabled) noexcept;

std::size_t exchange_prefill_gdn_c16_norm_gate_admission_test_hits(
    std::size_t hits) noexcept;

}  // namespace q3x::runtime::reference_runner_detail

#endif  // Q3X_RUNTIME_REFERENCE_RUNNER_GDN_C16_NORM_GATE_ADMISSION_H_
