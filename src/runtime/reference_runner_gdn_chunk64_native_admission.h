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

using PrefillGdnChunk64NativeSnapshotCallback = void (*)(
    const RequestState& state, void* context) noexcept;

struct PrefillGdnChunk64NativeSnapshotHook {
  PrefillGdnChunk64NativeSnapshotCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] PrefillGdnChunk64NativeSnapshotHook
exchange_prefill_gdn_chunk64_native_snapshot_hook(
    PrefillGdnChunk64NativeSnapshotHook hook) noexcept;

// Test-only observation point after successful generation has completed all
// prompt work and immediately before ReferenceEngine::generate returns.
using ReferenceEngineGenerateReturnSnapshotCallback = void (*)(
    const RequestState& state, void* context) noexcept;

struct ReferenceEngineGenerateReturnSnapshotHook {
  ReferenceEngineGenerateReturnSnapshotCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] ReferenceEngineGenerateReturnSnapshotHook
exchange_reference_engine_generate_return_snapshot_hook(
    ReferenceEngineGenerateReturnSnapshotHook hook) noexcept;

}  // namespace reference_runner_detail
}  // namespace q3x::runtime
