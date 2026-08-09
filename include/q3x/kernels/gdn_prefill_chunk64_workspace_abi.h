#pragma once

#include "q3x/runtime/gdn_decode.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Pure-host ABI for the fixed-shape, test-admission-only C64 GDN workspace.
// The CUDA partitioner consumes these exact offsets; host capacity planners
// consume total_bytes. Keeping both consumers on this layout prevents a
// kernel-region change from silently leaving the request-arena plan stale.
inline constexpr std::size_t kGdnPrefillChunk64WorkspaceTokenCount = 512U;
inline constexpr std::size_t kGdnPrefillChunk64WorkspaceChunkSize = 64U;
inline constexpr std::size_t kGdnPrefillChunk64WorkspaceAlignment = 256U;
inline constexpr std::size_t kGdnPrefillChunk64WorkspaceChunkCount =
    kGdnPrefillChunk64WorkspaceTokenCount /
    kGdnPrefillChunk64WorkspaceChunkSize;
inline constexpr std::size_t kGdnPrefillChunk64WorkspaceMatrixCount =
    kGdnPrefillChunk64WorkspaceChunkCount * runtime::kGdnValueHeadCount;
inline constexpr std::size_t kGdnPrefillChunk64WorkspaceHeadTokenElements =
    kGdnPrefillChunk64WorkspaceMatrixCount *
    kGdnPrefillChunk64WorkspaceChunkSize * runtime::kGdnHeadDimension;
inline constexpr std::size_t kGdnPrefillChunk64WorkspaceChunkMatrixElements =
    kGdnPrefillChunk64WorkspaceMatrixCount *
    kGdnPrefillChunk64WorkspaceChunkSize *
    kGdnPrefillChunk64WorkspaceChunkSize;
inline constexpr std::size_t kGdnPrefillChunk64WorkspaceBoundaryStateElements =
    kGdnPrefillChunk64WorkspaceMatrixCount * runtime::kGdnHeadDimension *
    runtime::kGdnHeadDimension;
inline constexpr std::size_t kGdnPrefillChunk64WorkspaceScalarElements =
    kGdnPrefillChunk64WorkspaceMatrixCount *
    kGdnPrefillChunk64WorkspaceChunkSize;

struct GdnPrefillChunk64WorkspaceLayout {
  std::size_t q_offset = 0U;
  std::size_t k_offset = 0U;
  std::size_t k_g_offset = 0U;
  std::size_t k_decay_offset = 0U;
  std::size_t v_offset = 0U;
  std::size_t transform_offset = 0U;
  std::size_t qk_offset = 0U;
  std::size_t w_offset = 0U;
  std::size_t u_offset = 0U;
  std::size_t v_new_offset = 0U;
  std::size_t boundary_state_offset = 0U;
  std::size_t kkt_offset = 0U;
  std::size_t gamma_offset = 0U;
  std::size_t beta_offset = 0U;
  std::size_t total_bytes = 0U;
};

[[nodiscard]] constexpr std::size_t
align_gdn_prefill_chunk64_workspace(const std::size_t value) noexcept {
  return (value + kGdnPrefillChunk64WorkspaceAlignment - 1U) &
         ~(kGdnPrefillChunk64WorkspaceAlignment - 1U);
}

[[nodiscard]] constexpr std::size_t append_gdn_prefill_chunk64_region(
    const std::size_t offset, const std::size_t elements,
    const std::size_t element_bytes) noexcept {
  return align_gdn_prefill_chunk64_workspace(offset) +
         elements * element_bytes;
}

[[nodiscard]] constexpr GdnPrefillChunk64WorkspaceLayout
make_gdn_prefill_chunk64_workspace_layout() noexcept {
  constexpr std::size_t bf16_bytes = sizeof(std::uint16_t);
  constexpr std::size_t fp32_bytes = sizeof(float);
  GdnPrefillChunk64WorkspaceLayout layout{};
  std::size_t cursor = 0U;

  layout.q_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceHeadTokenElements, bf16_bytes);
  layout.k_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceHeadTokenElements, bf16_bytes);
  layout.k_g_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceHeadTokenElements, bf16_bytes);
  layout.k_decay_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceHeadTokenElements, bf16_bytes);
  layout.v_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceHeadTokenElements, bf16_bytes);

  layout.transform_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceChunkMatrixElements, bf16_bytes);
  layout.qk_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceChunkMatrixElements, bf16_bytes);

  layout.w_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceHeadTokenElements, bf16_bytes);
  layout.u_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceHeadTokenElements, bf16_bytes);
  layout.v_new_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceHeadTokenElements, bf16_bytes);

  layout.boundary_state_offset =
      align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceBoundaryStateElements, bf16_bytes);
  layout.kkt_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceChunkMatrixElements, fp32_bytes);
  layout.gamma_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceScalarElements, fp32_bytes);
  layout.beta_offset = align_gdn_prefill_chunk64_workspace(cursor);
  cursor = append_gdn_prefill_chunk64_region(
      cursor, kGdnPrefillChunk64WorkspaceScalarElements, fp32_bytes);
  layout.total_bytes = align_gdn_prefill_chunk64_workspace(cursor);
  return layout;
}

inline constexpr GdnPrefillChunk64WorkspaceLayout
    kGdnPrefillChunk64WorkspaceLayout =
        make_gdn_prefill_chunk64_workspace_layout();
inline constexpr std::size_t kGdnPrefillChunk64NativeWorkspaceBytes =
    kGdnPrefillChunk64WorkspaceLayout.total_bytes;

static_assert(kGdnPrefillChunk64WorkspaceTokenCount %
                      kGdnPrefillChunk64WorkspaceChunkSize ==
                  0U,
              "C64 workspace requires an integral fixed C512 chunk grid");
static_assert(kGdnPrefillChunk64WorkspaceAlignment == 256U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.q_offset == 0U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.k_offset == 6'291'456U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.k_g_offset == 12'582'912U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.k_decay_offset ==
              18'874'368U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.v_offset == 25'165'824U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.transform_offset ==
              31'457'280U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.qk_offset == 34'603'008U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.w_offset == 37'748'736U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.u_offset == 44'040'192U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.v_new_offset == 50'331'648U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.boundary_state_offset ==
              56'623'104U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.kkt_offset == 69'206'016U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.gamma_offset == 75'497'472U);
static_assert(kGdnPrefillChunk64WorkspaceLayout.beta_offset == 75'595'776U);
static_assert(kGdnPrefillChunk64NativeWorkspaceBytes == 75'694'080U,
              "changing the native CUDA workspace layout must update the "
              "authenticated request-arena contract");

}  // namespace q3x::kernels
