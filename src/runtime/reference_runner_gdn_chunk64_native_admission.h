#pragma once

#include <cstddef>

namespace q3x::runtime {

class RequestState;

namespace reference_runner_detail {

[[nodiscard]] constexpr std::size_t
prefill_gdn_chunk64_native_prefix_token_count(
    const std::size_t token_count) noexcept {
  return token_count - token_count % 64U;
}

[[nodiscard]] constexpr std::size_t
prefill_gdn_chunk64_legacy_tail_token_count(
    const std::size_t token_count) noexcept {
  return token_count % 64U;
}

[[nodiscard]] bool exchange_prefill_gdn_chunk64_native_admission_test_enabled(
    bool enabled) noexcept;
[[nodiscard]] std::size_t
exchange_prefill_gdn_chunk64_native_admission_test_hits(
    std::size_t hits) noexcept;
[[nodiscard]] std::size_t
exchange_gdn_conv_compact_qk_fused_candidate_test_hits(
    std::size_t hits) noexcept;
[[nodiscard]] bool
exchange_gdn_conv_compact_qk_fused_candidate_test_enabled(
    bool enabled) noexcept;

using PrefillGdnChunk64NativeSnapshotCallback = void (*)(
    const RequestState& state, void* context) noexcept;

struct PrefillGdnChunk64NativeSnapshotHook {
  PrefillGdnChunk64NativeSnapshotCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] PrefillGdnChunk64NativeSnapshotHook
exchange_prefill_gdn_chunk64_native_snapshot_hook(
    PrefillGdnChunk64NativeSnapshotHook hook) noexcept;

// A separate final-step observer keeps the established bulk-tile snapshot
// contract intact while allowing the P513 exactness gate to inspect the
// request state after the trailing C1 step commits.
using PrefillGdnChunk64NativeFinalSnapshotCallback = void (*)(
    const RequestState& state, void* context) noexcept;

struct PrefillGdnChunk64NativeFinalSnapshotHook {
  PrefillGdnChunk64NativeFinalSnapshotCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] PrefillGdnChunk64NativeFinalSnapshotHook
exchange_prefill_gdn_chunk64_native_final_snapshot_hook(
    PrefillGdnChunk64NativeFinalSnapshotHook hook) noexcept;

}  // namespace reference_runner_detail
}  // namespace q3x::runtime
