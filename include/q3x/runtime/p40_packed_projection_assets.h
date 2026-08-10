#pragma once

#include "q3x/kernels/sm87_p40_packed_projection.h"
#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/resident_weights.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace q3x::runtime {

// Exact retained payload extent for AC-PREFILL-P40-PACKED-DATAFLOW-v1.  The
// arena contains payloads only; authenticated 4 KiB logical headers remain in
// the host manifests below and are never inserted between device payloads.
inline constexpr std::uint64_t kP40PackedProjectionArenaBytes =
    16'840'130'560ULL;

// Engine-lifetime owner for the complete fixed P40 projection inventory.
// ModelWeights stores non-owning copies of descriptor views, so this owner
// must be declared before ModelWeights (and therefore destroyed after it), or
// released only while the runner is globally quiescent.  release() does not
// detach views from a still-live ModelWeights object.
class P40PackedProjectionAssets final {
 public:
  P40PackedProjectionAssets() noexcept = default;
  ~P40PackedProjectionAssets();

  P40PackedProjectionAssets(const P40PackedProjectionAssets&) = delete;
  P40PackedProjectionAssets& operator=(
      const P40PackedProjectionAssets&) = delete;
  P40PackedProjectionAssets(P40PackedProjectionAssets&& other) noexcept;
  P40PackedProjectionAssets& operator=(
      P40PackedProjectionAssets&& other) noexcept;

  void release() noexcept;

  [[nodiscard]] bool empty() const noexcept {
    return arena == nullptr && bytes == 0U && descriptor_count == 0U &&
           manifest_count == 0U;
  }

  std::uint8_t* arena = nullptr;
  std::uint64_t bytes = 0U;
  std::array<P40PackedProjectionSidecarDescriptor,
             kernels::kSm87P40PackedProjectionArtifactCount>
      descriptors{};
  std::size_t descriptor_count = 0U;
  std::array<kernels::Sm87P40PackedArtifactManifest,
             kernels::kSm87P40PackedProjectionArtifactCount>
      manifests{};
  std::size_t manifest_count = 0U;
};

struct P40PackedProjectionPreparationStats {
  bool enabled = false;
  bool hard_failure = false;
  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  std::size_t fp8_logical = 0U;
  std::size_t fp8_physical = 0U;
  std::size_t nvfp4_physical = 0U;
  std::uint64_t bytes = 0U;
  int cuda_error = 0;
  std::string message;
};

// Performs a load-time, all-or-nothing preparation of the complete real-model
// inventory.  The ResidentWeights argument closes provenance: every canonical
// source pointer, dtype, shape, and byte range must match its exact resident
// tensor, and the loader-observed shard hashes must match the pinned payload.
// No request path may call this routine.
[[nodiscard]] P40PackedProjectionPreparationStats
prepare_p40_packed_projection_assets(
    const ResidentWeights& resident, ModelWeights& model_weights,
    std::uint64_t minimum_free_bytes_after_prepare,
    P40PackedProjectionAssets& owner);

[[nodiscard]] constexpr bool p40_packed_projection_assets_compiled()
    noexcept {
#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
  return true;
#else
  return false;
#endif
}

static_assert(kP40PackedProjectionArenaBytes ==
              kernels::kSm87P40PackedProjectionLayerCount *
                      (kernels::sm87_p40_packed_projection_plan(
                           kernels::Sm87P40PackedProjectionRole::
                               kNvFp4GateUp)
                           .payload_bytes +
                       kernels::sm87_p40_packed_projection_plan(
                           kernels::Sm87P40PackedProjectionRole::kNvFp4Down)
                           .payload_bytes +
                       kernels::sm87_p40_packed_projection_plan(
                           kernels::Sm87P40PackedProjectionRole::
                               kFp8AttentionOutput)
                           .payload_bytes) +
                  kernels::kSm87P40PackedProjectionLinearLayerCount *
                      kernels::sm87_p40_packed_projection_plan(
                          kernels::Sm87P40PackedProjectionRole::
                              kFp8LinearQkvZ)
                          .payload_bytes +
                  kernels::kSm87P40PackedProjectionFullLayerCount *
                      kernels::sm87_p40_packed_projection_plan(
                          kernels::Sm87P40PackedProjectionRole::kFp8FullQkv)
                          .payload_bytes);

}  // namespace q3x::runtime
