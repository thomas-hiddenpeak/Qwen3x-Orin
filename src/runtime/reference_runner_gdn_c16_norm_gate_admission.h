#ifndef Q3X_RUNTIME_REFERENCE_RUNNER_GDN_C16_NORM_GATE_ADMISSION_H_
#define Q3X_RUNTIME_REFERENCE_RUNNER_GDN_C16_NORM_GATE_ADMISSION_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

class RequestState;

namespace reference_runner_detail {

// Private test-only controls for the exact-C16 admission route. They are not
// installed and deliberately do not extend ReferenceRunner's public API.
bool exchange_prefill_gdn_c16_norm_gate_admission_test_enabled(
    bool enabled) noexcept;

std::size_t exchange_prefill_gdn_c16_norm_gate_admission_test_hits(
    std::size_t hits) noexcept;

enum class PrefillGdnC16NormGateAdmissionSnapshotStage : std::uint8_t {
  kPrefixTile = 0,
  kStep,
};

using PrefillGdnC16NormGateAdmissionSnapshotCallback = void (*)(
    const RequestState& state,
    PrefillGdnC16NormGateAdmissionSnapshotStage stage,
    void* context) noexcept;

struct PrefillGdnC16NormGateAdmissionSnapshotHook {
  PrefillGdnC16NormGateAdmissionSnapshotCallback callback = nullptr;
  void* context = nullptr;
};

// The hook defaults to null, is thread-local in the implementation, and is
// observed only after a synchronous runner call has successfully committed
// its request-state transition. Callback failures are test evidence only and
// deliberately cannot alter the runner's production error semantics.
PrefillGdnC16NormGateAdmissionSnapshotHook
exchange_prefill_gdn_c16_norm_gate_admission_snapshot_hook(
    PrefillGdnC16NormGateAdmissionSnapshotHook hook) noexcept;

}  // namespace reference_runner_detail
}  // namespace q3x::runtime

#endif  // Q3X_RUNTIME_REFERENCE_RUNNER_GDN_C16_NORM_GATE_ADMISSION_H_
