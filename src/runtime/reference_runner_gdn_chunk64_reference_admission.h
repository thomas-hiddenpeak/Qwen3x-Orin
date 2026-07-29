#pragma once

#include <cstddef>

namespace q3x::runtime {

class RequestState;

namespace reference_runner_detail {

[[nodiscard]] bool exchange_prefill_gdn_chunk64_reference_admission_test_enabled(
    bool enabled) noexcept;
[[nodiscard]] std::size_t
exchange_prefill_gdn_chunk64_reference_admission_test_hits(
    std::size_t hits) noexcept;

using PrefillGdnChunk64ReferenceSnapshotCallback = void (*)(
    const RequestState& state, void* context) noexcept;

struct PrefillGdnChunk64ReferenceSnapshotHook {
  PrefillGdnChunk64ReferenceSnapshotCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] PrefillGdnChunk64ReferenceSnapshotHook
exchange_prefill_gdn_chunk64_reference_snapshot_hook(
    PrefillGdnChunk64ReferenceSnapshotHook hook) noexcept;

}  // namespace reference_runner_detail
}  // namespace q3x::runtime
