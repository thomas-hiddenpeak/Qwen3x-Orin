#pragma once

#include <cstddef>

namespace q3x::runtime {

class RequestState;

namespace reference_runner_detail {

[[nodiscard]] bool exchange_prefill_gdn_chunk64_native_admission_test_enabled(
    bool enabled) noexcept;
[[nodiscard]] std::size_t
exchange_prefill_gdn_chunk64_native_admission_test_hits(
    std::size_t hits) noexcept;

using PrefillGdnChunk64NativeSnapshotCallback = void (*)(
    const RequestState& state, void* context) noexcept;

struct PrefillGdnChunk64NativeSnapshotHook {
  PrefillGdnChunk64NativeSnapshotCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] PrefillGdnChunk64NativeSnapshotHook
exchange_prefill_gdn_chunk64_native_snapshot_hook(
    PrefillGdnChunk64NativeSnapshotHook hook) noexcept;

}  // namespace reference_runner_detail
}  // namespace q3x::runtime
