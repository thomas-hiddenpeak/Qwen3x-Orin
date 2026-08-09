#pragma once

#include "q3x/runtime/gdn_decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Host-only ABI for the default-off prompt-wide GDN chunk graph.  This is a
// fixed Qwen3.6/SM87 development contract, not a generic sequence-length
// workspace calculator and not a production admission surface.
inline constexpr std::size_t kGdnPromptWideChunkGraphChunkTokens = 64U;
inline constexpr std::size_t kGdnPromptWideChunkGraphAlignment = 256U;
inline constexpr std::size_t kGdnPromptWideChunkGraphP40Tokens = 40'000U;
inline constexpr std::size_t kGdnPromptWideChunkGraphP60Tokens = 60'000U;

enum class GdnPromptWideChunkGraphPlanError : std::uint8_t {
  kNone = 0,
  kInvalidTokenCount,
  // P60 has a 32-token final C64 tail.  Its geometry is reported, but the
  // first implementation deliberately owns no workspace or launch contract
  // for it.  Callers must not reinterpret this as generic unsupported input.
  kP60PartialChunkPending,
  kArithmeticOverflow,
  kInvalidLayout,
};

// Phase-lifetime layout for one complete linear-attention layer:
//
//   preprocess: compact_q, compact_k, gamma, beta
//   WY:         transform, raw_gram, W, U
//   recurrence: boundary_state, v_new, W, U
//   output:     boundary_state, v_new, compact_q/k, raw_output
//
// The three equal-offset pairs below are intentional same-stream aliases.
// Their preceding value is dead before the following kernel phase begins.
// The legacy C512 ABI is disjoint and byte-stable in its existing header.
struct GdnPromptWideChunkGraphWorkspaceLayout {
  std::size_t compact_q_offset = 0U;
  std::size_t compact_k_offset = 0U;

  std::size_t transform_offset = 0U;
  std::size_t boundary_state_offset = 0U;

  std::size_t raw_gram_offset = 0U;
  std::size_t v_new_offset = 0U;

  std::size_t w_offset = 0U;
  std::size_t raw_output_offset = 0U;

  std::size_t u_offset = 0U;
  std::size_t gamma_offset = 0U;
  std::size_t beta_offset = 0U;
  std::size_t total_bytes = 0U;
};

struct GdnPromptWideChunkGraphWorkspacePlan {
  GdnPromptWideChunkGraphPlanError error =
      GdnPromptWideChunkGraphPlanError::kInvalidTokenCount;
  std::size_t requested_token_count = 0U;
  std::size_t padded_token_count = 0U;
  std::size_t chunk_count = 0U;
  std::size_t compact_matrix_count = 0U;
  std::size_t value_matrix_count = 0U;
  std::size_t compact_head_token_elements = 0U;
  std::size_t value_head_token_elements = 0U;
  std::size_t transform_elements = 0U;
  std::size_t boundary_state_elements = 0U;
  std::size_t raw_gram_elements = 0U;
  std::size_t scalar_elements = 0U;
  GdnPromptWideChunkGraphWorkspaceLayout layout{};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return error == GdnPromptWideChunkGraphPlanError::kNone;
  }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
};

// Every pointer-bearing prompt-wide launch is described as a checked byte
// range before CUDA work is enqueued.  Keeping the role list in the pure-host
// ABI makes the exact P40 tensor contract independently testable and prevents
// pointer-equality checks from accepting shifted/partial overlap.
enum class GdnPromptWideChunkGraphBufferRole : std::uint8_t {
  kWorkspace = 0,
  kRawQkv,
  kConvWeight,
  kConvHistory,
  kConvQkvOutput,
  kA,
  kB,
  kALog,
  kDtBias,
  kStateInput,
  kStateOutput,
  kNormWeight,
  kSiluGate,
  kOutput,
  kCount,
};

inline constexpr std::size_t kGdnPromptWideChunkGraphBufferRoleCount =
    static_cast<std::size_t>(GdnPromptWideChunkGraphBufferRole::kCount);

struct GdnPromptWideChunkGraphByteRange {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept { return end > begin; }

  [[nodiscard]] constexpr std::size_t size_bytes() const noexcept {
    return valid() ? static_cast<std::size_t>(end - begin) : 0U;
  }
};

struct GdnPromptWideChunkGraphBufferAddresses {
  std::uintptr_t workspace = 0U;
  std::size_t workspace_capacity_bytes = 0U;
  std::uintptr_t raw_qkv = 0U;
  std::uintptr_t conv_weight = 0U;
  std::uintptr_t conv_history = 0U;
  std::uintptr_t conv_qkv_output = 0U;
  std::uintptr_t a = 0U;
  std::uintptr_t b = 0U;
  std::uintptr_t A_log = 0U;
  std::uintptr_t dt_bias = 0U;
  std::uintptr_t state_input = 0U;
  std::uintptr_t state_output = 0U;
  std::uintptr_t norm_weight = 0U;
  std::uintptr_t silu_gate = 0U;
  std::uintptr_t output = 0U;
};

enum class GdnPromptWideChunkGraphBufferContractError : std::uint8_t {
  kNone = 0,
  kInvalidPlan,
  kNullBuffer,
  kInsufficientWorkspace,
  kArithmeticOverflow,
  kMisalignedBuffer,
  kForbiddenOverlap,
};

struct GdnPromptWideChunkGraphBufferContractReceipt {
  GdnPromptWideChunkGraphBufferContractError error =
      GdnPromptWideChunkGraphBufferContractError::kInvalidPlan;
  GdnPromptWideChunkGraphBufferRole conflict_left =
      GdnPromptWideChunkGraphBufferRole::kCount;
  GdnPromptWideChunkGraphBufferRole conflict_right =
      GdnPromptWideChunkGraphBufferRole::kCount;
  std::size_t token_count = 0U;
  std::array<std::size_t, kGdnPromptWideChunkGraphBufferRoleCount>
      required_bytes{};
  std::array<GdnPromptWideChunkGraphByteRange,
             kGdnPromptWideChunkGraphBufferRoleCount>
      ranges{};
  // This records the caller-declared capacity separately from the exact
  // workspace bytes touched by the graph.  The declared extent is checked
  // for address overflow, while overlap applies to the touched extent.
  GdnPromptWideChunkGraphByteRange workspace_capacity_range{};
  bool state_in_place = false;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return error == GdnPromptWideChunkGraphBufferContractError::kNone;
  }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }

  [[nodiscard]] constexpr const GdnPromptWideChunkGraphByteRange& range(
      const GdnPromptWideChunkGraphBufferRole role) const noexcept {
    return ranges[static_cast<std::size_t>(role)];
  }
};

enum class GdnPromptWideChunkGraphWorkspaceRegionRole : std::uint8_t {
  kCompactQ = 0,
  kCompactK,
  kTransform,
  kBoundaryState,
  kRawGram,
  kVNew,
  kW,
  kRawOutput,
  kU,
  kGamma,
  kBeta,
  kCount,
};

inline constexpr std::size_t
    kGdnPromptWideChunkGraphWorkspaceRegionRoleCount =
        static_cast<std::size_t>(
            GdnPromptWideChunkGraphWorkspaceRegionRole::kCount);

enum class GdnPromptWideChunkGraphWorkspaceLifetimeError : std::uint8_t {
  kNone = 0,
  kInvalidPlan,
  kArithmeticOverflow,
  kOutOfBounds,
  kUnexpectedAlias,
};

struct GdnPromptWideChunkGraphWorkspaceLifetimeReceipt {
  GdnPromptWideChunkGraphWorkspaceLifetimeError error =
      GdnPromptWideChunkGraphWorkspaceLifetimeError::kInvalidPlan;
  GdnPromptWideChunkGraphWorkspaceRegionRole conflict_left =
      GdnPromptWideChunkGraphWorkspaceRegionRole::kCount;
  GdnPromptWideChunkGraphWorkspaceRegionRole conflict_right =
      GdnPromptWideChunkGraphWorkspaceRegionRole::kCount;
  std::array<GdnPromptWideChunkGraphByteRange,
             kGdnPromptWideChunkGraphWorkspaceRegionRoleCount>
      ranges{};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return error == GdnPromptWideChunkGraphWorkspaceLifetimeError::kNone;
  }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }

  [[nodiscard]] constexpr const GdnPromptWideChunkGraphByteRange& range(
      const GdnPromptWideChunkGraphWorkspaceRegionRole role) const noexcept {
    return ranges[static_cast<std::size_t>(role)];
  }
};

[[nodiscard]] constexpr bool checked_gdn_prompt_wide_add(
    const std::size_t left, const std::size_t right,
    std::size_t& result) noexcept {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] constexpr bool checked_gdn_prompt_wide_multiply(
    const std::size_t left, const std::size_t right,
    std::size_t& result) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] constexpr bool checked_gdn_prompt_wide_align(
    const std::size_t value, std::size_t& result) noexcept {
  std::size_t biased = 0U;
  if (!checked_gdn_prompt_wide_add(
          value, kGdnPromptWideChunkGraphAlignment - 1U, biased)) {
    return false;
  }
  result = biased & ~(kGdnPromptWideChunkGraphAlignment - 1U);
  return true;
}

[[nodiscard]] constexpr bool append_gdn_prompt_wide_region(
    std::size_t& cursor, const std::size_t bytes,
    std::size_t& offset) noexcept {
  if (!checked_gdn_prompt_wide_align(cursor, offset)) {
    return false;
  }
  return checked_gdn_prompt_wide_add(offset, bytes, cursor);
}

[[nodiscard]] constexpr bool make_gdn_prompt_wide_byte_range(
    const std::uintptr_t begin, const std::size_t bytes,
    GdnPromptWideChunkGraphByteRange& range) noexcept {
  if (bytes == 0U ||
      bytes > std::numeric_limits<std::uintptr_t>::max() ||
      begin > std::numeric_limits<std::uintptr_t>::max() -
                  static_cast<std::uintptr_t>(bytes)) {
    return false;
  }
  range.begin = begin;
  range.end = begin + static_cast<std::uintptr_t>(bytes);
  return true;
}

[[nodiscard]] constexpr bool gdn_prompt_wide_ranges_overlap(
    const GdnPromptWideChunkGraphByteRange& left,
    const GdnPromptWideChunkGraphByteRange& right) noexcept {
  return left.valid() && right.valid() && left.begin < right.end &&
         right.begin < left.end;
}

[[nodiscard]] constexpr bool gdn_prompt_wide_ranges_exact(
    const GdnPromptWideChunkGraphByteRange& left,
    const GdnPromptWideChunkGraphByteRange& right) noexcept {
  return left.valid() && right.valid() && left.begin == right.begin &&
         left.end == right.end;
}

[[nodiscard]] constexpr bool gdn_prompt_wide_range_contains(
    const GdnPromptWideChunkGraphByteRange& outer,
    const GdnPromptWideChunkGraphByteRange& inner) noexcept {
  return outer.valid() && inner.valid() && outer.begin <= inner.begin &&
         inner.end <= outer.end;
}

[[nodiscard]] constexpr GdnPromptWideChunkGraphWorkspacePlan
make_gdn_prompt_wide_chunk_graph_workspace_plan(
    const std::size_t token_count) noexcept {
  GdnPromptWideChunkGraphWorkspacePlan plan{};
  plan.requested_token_count = token_count;

  if (token_count == 0U) {
    return plan;
  }

  std::size_t padded_numerator = 0U;
  if (!checked_gdn_prompt_wide_add(
          token_count, kGdnPromptWideChunkGraphChunkTokens - 1U,
          padded_numerator)) {
    plan.error = GdnPromptWideChunkGraphPlanError::kArithmeticOverflow;
    return plan;
  }
  plan.chunk_count =
      padded_numerator / kGdnPromptWideChunkGraphChunkTokens;
  if (!checked_gdn_prompt_wide_multiply(
          plan.chunk_count, kGdnPromptWideChunkGraphChunkTokens,
          plan.padded_token_count)) {
    plan.error = GdnPromptWideChunkGraphPlanError::kArithmeticOverflow;
    return plan;
  }

  if (token_count == kGdnPromptWideChunkGraphP60Tokens) {
    plan.error =
        GdnPromptWideChunkGraphPlanError::kP60PartialChunkPending;
    return plan;
  }
  if (token_count != kGdnPromptWideChunkGraphP40Tokens) {
    plan.error = GdnPromptWideChunkGraphPlanError::kInvalidTokenCount;
    return plan;
  }
  if (plan.padded_token_count != token_count) {
    plan.error = GdnPromptWideChunkGraphPlanError::kInvalidLayout;
    return plan;
  }

  constexpr std::size_t chunk = kGdnPromptWideChunkGraphChunkTokens;
  constexpr std::size_t qk_heads = runtime::kGdnQkHeadCount;
  constexpr std::size_t value_heads = runtime::kGdnValueHeadCount;
  constexpr std::size_t dimension = runtime::kGdnHeadDimension;
  constexpr std::size_t bf16_bytes = sizeof(std::uint16_t);
  constexpr std::size_t fp32_bytes = sizeof(float);

  if (!checked_gdn_prompt_wide_multiply(
          plan.chunk_count, qk_heads, plan.compact_matrix_count) ||
      !checked_gdn_prompt_wide_multiply(
          plan.chunk_count, value_heads, plan.value_matrix_count)) {
    plan.error = GdnPromptWideChunkGraphPlanError::kArithmeticOverflow;
    return plan;
  }

  std::size_t chunk_dimension = 0U;
  std::size_t chunk_matrix = 0U;
  std::size_t state_matrix = 0U;
  if (!checked_gdn_prompt_wide_multiply(chunk, dimension,
                                        chunk_dimension) ||
      !checked_gdn_prompt_wide_multiply(chunk, chunk, chunk_matrix) ||
      !checked_gdn_prompt_wide_multiply(dimension, dimension,
                                        state_matrix) ||
      !checked_gdn_prompt_wide_multiply(
          plan.compact_matrix_count, chunk_dimension,
          plan.compact_head_token_elements) ||
      !checked_gdn_prompt_wide_multiply(
          plan.value_matrix_count, chunk_dimension,
          plan.value_head_token_elements) ||
      !checked_gdn_prompt_wide_multiply(
          plan.value_matrix_count, chunk_matrix,
          plan.transform_elements) ||
      !checked_gdn_prompt_wide_multiply(
          plan.value_matrix_count, state_matrix,
          plan.boundary_state_elements) ||
      !checked_gdn_prompt_wide_multiply(
          plan.compact_matrix_count, chunk_matrix,
          plan.raw_gram_elements) ||
      !checked_gdn_prompt_wide_multiply(
          plan.value_matrix_count, chunk, plan.scalar_elements)) {
    plan.error = GdnPromptWideChunkGraphPlanError::kArithmeticOverflow;
    return plan;
  }

  std::size_t compact_bytes = 0U;
  std::size_t value_bytes = 0U;
  std::size_t transform_bytes = 0U;
  std::size_t boundary_bytes = 0U;
  std::size_t raw_gram_bytes = 0U;
  std::size_t scalar_bytes = 0U;
  if (!checked_gdn_prompt_wide_multiply(
          plan.compact_head_token_elements, bf16_bytes, compact_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.value_head_token_elements, bf16_bytes, value_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.transform_elements, bf16_bytes, transform_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.boundary_state_elements, bf16_bytes, boundary_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.raw_gram_elements, fp32_bytes, raw_gram_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.scalar_elements, fp32_bytes, scalar_bytes)) {
    plan.error = GdnPromptWideChunkGraphPlanError::kArithmeticOverflow;
    return plan;
  }

  auto& layout = plan.layout;
  std::size_t cursor = 0U;
  if (!append_gdn_prompt_wide_region(
          cursor, compact_bytes, layout.compact_q_offset) ||
      !append_gdn_prompt_wide_region(
          cursor, compact_bytes, layout.compact_k_offset) ||
      !append_gdn_prompt_wide_region(
          cursor, boundary_bytes, layout.transform_offset) ||
      !append_gdn_prompt_wide_region(
          cursor, value_bytes, layout.raw_gram_offset) ||
      !append_gdn_prompt_wide_region(
          cursor, value_bytes, layout.w_offset) ||
      !append_gdn_prompt_wide_region(
          cursor, value_bytes, layout.u_offset) ||
      !append_gdn_prompt_wide_region(
          cursor, scalar_bytes, layout.gamma_offset) ||
      !append_gdn_prompt_wide_region(
          cursor, scalar_bytes, layout.beta_offset) ||
      !checked_gdn_prompt_wide_align(cursor, layout.total_bytes)) {
    plan.error = GdnPromptWideChunkGraphPlanError::kArithmeticOverflow;
    return plan;
  }

  // The larger member owns each phase alias.  Keep these assignments after
  // checked placement so a future size change cannot silently overlap the
  // next independent region.
  if (boundary_bytes < transform_bytes || value_bytes < raw_gram_bytes) {
    plan.error = GdnPromptWideChunkGraphPlanError::kInvalidLayout;
    plan.layout = {};
    return plan;
  }
  layout.boundary_state_offset = layout.transform_offset;
  layout.v_new_offset = layout.raw_gram_offset;
  layout.raw_output_offset = layout.w_offset;
  plan.error = GdnPromptWideChunkGraphPlanError::kNone;
  return plan;
}

inline constexpr GdnPromptWideChunkGraphWorkspacePlan
    kGdnPromptWideChunkGraphP40WorkspacePlan =
        make_gdn_prompt_wide_chunk_graph_workspace_plan(
            kGdnPromptWideChunkGraphP40Tokens);
inline constexpr std::size_t kGdnPromptWideChunkGraphP40WorkspaceBytes =
    kGdnPromptWideChunkGraphP40WorkspacePlan.layout.total_bytes;

static_assert((kGdnPromptWideChunkGraphAlignment &
               (kGdnPromptWideChunkGraphAlignment - 1U)) == 0U);
static_assert(kGdnPromptWideChunkGraphP40Tokens %
                      kGdnPromptWideChunkGraphChunkTokens ==
                  0U,
              "P40 candidate owns an exact integral C64 graph");
static_assert(kGdnPromptWideChunkGraphP40WorkspacePlan.ok());
static_assert(kGdnPromptWideChunkGraphP40WorkspacePlan.chunk_count == 625U);
static_assert(kGdnPromptWideChunkGraphP40WorkspacePlan.value_matrix_count ==
              30'000U);
static_assert(kGdnPromptWideChunkGraphP40WorkspaceBytes == 2'800'640'000U);

inline constexpr GdnPromptWideChunkGraphWorkspacePlan
    kGdnPromptWideChunkGraphP60WorkspacePlan =
        make_gdn_prompt_wide_chunk_graph_workspace_plan(
            kGdnPromptWideChunkGraphP60Tokens);
static_assert(kGdnPromptWideChunkGraphP60WorkspacePlan.error ==
              GdnPromptWideChunkGraphPlanError::kP60PartialChunkPending);
static_assert(kGdnPromptWideChunkGraphP60WorkspacePlan.chunk_count == 938U);
static_assert(
    kGdnPromptWideChunkGraphP60WorkspacePlan.padded_token_count == 60'032U);
static_assert(kGdnPromptWideChunkGraphP60WorkspacePlan.layout.total_bytes ==
              0U);

[[nodiscard]] constexpr GdnPromptWideChunkGraphWorkspaceLifetimeReceipt
make_gdn_prompt_wide_chunk_graph_workspace_lifetime_receipt(
    const GdnPromptWideChunkGraphWorkspacePlan& plan) noexcept {
  GdnPromptWideChunkGraphWorkspaceLifetimeReceipt receipt{};
  if (!plan.ok() || plan.layout.total_bytes == 0U) {
    return receipt;
  }

  constexpr std::size_t bf16_bytes = sizeof(std::uint16_t);
  constexpr std::size_t fp32_bytes = sizeof(float);
  std::size_t compact_bytes = 0U;
  std::size_t value_bytes = 0U;
  std::size_t transform_bytes = 0U;
  std::size_t boundary_bytes = 0U;
  std::size_t raw_gram_bytes = 0U;
  std::size_t scalar_bytes = 0U;
  if (!checked_gdn_prompt_wide_multiply(
          plan.compact_head_token_elements, bf16_bytes, compact_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.value_head_token_elements, bf16_bytes, value_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.transform_elements, bf16_bytes, transform_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.boundary_state_elements, bf16_bytes, boundary_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.raw_gram_elements, fp32_bytes, raw_gram_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          plan.scalar_elements, fp32_bytes, scalar_bytes)) {
    receipt.error =
        GdnPromptWideChunkGraphWorkspaceLifetimeError::kArithmeticOverflow;
    return receipt;
  }

  using Region = GdnPromptWideChunkGraphWorkspaceRegionRole;
  constexpr auto index = [](const Region role) constexpr noexcept {
    return static_cast<std::size_t>(role);
  };
  const std::array<std::uintptr_t,
                   kGdnPromptWideChunkGraphWorkspaceRegionRoleCount>
      offsets{{
          plan.layout.compact_q_offset,
          plan.layout.compact_k_offset,
          plan.layout.transform_offset,
          plan.layout.boundary_state_offset,
          plan.layout.raw_gram_offset,
          plan.layout.v_new_offset,
          plan.layout.w_offset,
          plan.layout.raw_output_offset,
          plan.layout.u_offset,
          plan.layout.gamma_offset,
          plan.layout.beta_offset,
      }};
  const std::array<std::size_t,
                   kGdnPromptWideChunkGraphWorkspaceRegionRoleCount>
      sizes{{
          compact_bytes,
          compact_bytes,
          transform_bytes,
          boundary_bytes,
          raw_gram_bytes,
          value_bytes,
          value_bytes,
          value_bytes,
          value_bytes,
          scalar_bytes,
          scalar_bytes,
      }};
  for (std::size_t role = 0U; role < receipt.ranges.size(); ++role) {
    if (!make_gdn_prompt_wide_byte_range(offsets[role], sizes[role],
                                         receipt.ranges[role])) {
      receipt.error =
          GdnPromptWideChunkGraphWorkspaceLifetimeError::kArithmeticOverflow;
      receipt.conflict_left = static_cast<Region>(role);
      return receipt;
    }
    if (receipt.ranges[role].end > plan.layout.total_bytes) {
      receipt.error =
          GdnPromptWideChunkGraphWorkspaceLifetimeError::kOutOfBounds;
      receipt.conflict_left = static_cast<Region>(role);
      return receipt;
    }
  }

  const auto& transform = receipt.ranges[index(Region::kTransform)];
  const auto& boundary = receipt.ranges[index(Region::kBoundaryState)];
  const auto& raw_gram = receipt.ranges[index(Region::kRawGram)];
  const auto& v_new = receipt.ranges[index(Region::kVNew)];
  const auto& w = receipt.ranges[index(Region::kW)];
  const auto& raw_output = receipt.ranges[index(Region::kRawOutput)];
  if (transform.begin != boundary.begin ||
      !gdn_prompt_wide_range_contains(boundary, transform)) {
    receipt.error =
        GdnPromptWideChunkGraphWorkspaceLifetimeError::kUnexpectedAlias;
    receipt.conflict_left = Region::kTransform;
    receipt.conflict_right = Region::kBoundaryState;
    return receipt;
  }
  if (raw_gram.begin != v_new.begin ||
      !gdn_prompt_wide_range_contains(v_new, raw_gram)) {
    receipt.error =
        GdnPromptWideChunkGraphWorkspaceLifetimeError::kUnexpectedAlias;
    receipt.conflict_left = Region::kRawGram;
    receipt.conflict_right = Region::kVNew;
    return receipt;
  }
  if (!gdn_prompt_wide_ranges_exact(w, raw_output)) {
    receipt.error =
        GdnPromptWideChunkGraphWorkspaceLifetimeError::kUnexpectedAlias;
    receipt.conflict_left = Region::kW;
    receipt.conflict_right = Region::kRawOutput;
    return receipt;
  }

  // These are the physical owners.  The smaller transform/raw-Gram logical
  // values are contained at the beginning of the larger boundary/v_new
  // owners and are dead before those owners are written. W and raw_output
  // have exactly the same range and non-overlapping lifetimes.
  constexpr std::array<Region, 8U> physical_owners{{
      Region::kCompactQ,
      Region::kCompactK,
      Region::kBoundaryState,
      Region::kVNew,
      Region::kW,
      Region::kU,
      Region::kGamma,
      Region::kBeta,
  }};
  for (std::size_t left = 0U; left < physical_owners.size(); ++left) {
    for (std::size_t right = left + 1U; right < physical_owners.size();
         ++right) {
      const Region left_role = physical_owners[left];
      const Region right_role = physical_owners[right];
      if (gdn_prompt_wide_ranges_overlap(
              receipt.ranges[index(left_role)],
              receipt.ranges[index(right_role)])) {
        receipt.error =
            GdnPromptWideChunkGraphWorkspaceLifetimeError::kUnexpectedAlias;
        receipt.conflict_left = left_role;
        receipt.conflict_right = right_role;
        return receipt;
      }
    }
  }

  receipt.error = GdnPromptWideChunkGraphWorkspaceLifetimeError::kNone;
  return receipt;
}

[[nodiscard]] constexpr GdnPromptWideChunkGraphBufferContractReceipt
make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
    const std::size_t token_count,
    const GdnPromptWideChunkGraphBufferAddresses& addresses) noexcept {
  GdnPromptWideChunkGraphBufferContractReceipt receipt{};
  receipt.token_count = token_count;
  const auto plan =
      make_gdn_prompt_wide_chunk_graph_workspace_plan(token_count);
  if (!plan.ok() ||
      !make_gdn_prompt_wide_chunk_graph_workspace_lifetime_receipt(plan)
           .ok()) {
    return receipt;
  }

  using Role = GdnPromptWideChunkGraphBufferRole;
  constexpr auto index = [](const Role role) constexpr noexcept {
    return static_cast<std::size_t>(role);
  };

  std::size_t raw_qkv_elements = 0U;
  std::size_t head_scalar_elements = 0U;
  std::size_t gate_elements = 0U;
  std::size_t raw_qkv_bytes = 0U;
  std::size_t conv_weight_bytes = 0U;
  std::size_t conv_history_bytes = 0U;
  std::size_t head_scalar_bytes = 0U;
  std::size_t fixed_head_bytes = 0U;
  std::size_t state_bytes = 0U;
  std::size_t norm_weight_bytes = 0U;
  std::size_t gate_bytes = 0U;
  constexpr std::size_t bf16_bytes = sizeof(std::uint16_t);
  if (!checked_gdn_prompt_wide_multiply(
          token_count, runtime::kGdnQkvChannels, raw_qkv_elements) ||
      !checked_gdn_prompt_wide_multiply(
          token_count, runtime::kGdnValueHeadCount,
          head_scalar_elements) ||
      !checked_gdn_prompt_wide_multiply(
          token_count, runtime::kGdnVElements, gate_elements) ||
      !checked_gdn_prompt_wide_multiply(
          raw_qkv_elements, bf16_bytes, raw_qkv_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          runtime::kGdnQkvChannels * runtime::kGdnConvKernelWidth,
          bf16_bytes, conv_weight_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          runtime::kGdnQkvChannels * runtime::kGdnConvHistoryWidth,
          bf16_bytes, conv_history_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          head_scalar_elements, bf16_bytes, head_scalar_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          runtime::kGdnValueHeadCount, bf16_bytes, fixed_head_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          runtime::kGdnStateElements, bf16_bytes, state_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          runtime::kGdnHeadDimension, bf16_bytes, norm_weight_bytes) ||
      !checked_gdn_prompt_wide_multiply(
          gate_elements, bf16_bytes, gate_bytes)) {
    receipt.error =
        GdnPromptWideChunkGraphBufferContractError::kArithmeticOverflow;
    return receipt;
  }

  receipt.required_bytes = {{
      plan.layout.total_bytes,
      raw_qkv_bytes,
      conv_weight_bytes,
      conv_history_bytes,
      raw_qkv_bytes,
      head_scalar_bytes,
      head_scalar_bytes,
      fixed_head_bytes,
      fixed_head_bytes,
      state_bytes,
      state_bytes,
      norm_weight_bytes,
      gate_bytes,
      gate_bytes,
  }};
  const std::array<std::uintptr_t,
                   kGdnPromptWideChunkGraphBufferRoleCount>
      begins{{
          addresses.workspace,
          addresses.raw_qkv,
          addresses.conv_weight,
          addresses.conv_history,
          addresses.conv_qkv_output,
          addresses.a,
          addresses.b,
          addresses.A_log,
          addresses.dt_bias,
          addresses.state_input,
          addresses.state_output,
          addresses.norm_weight,
          addresses.silu_gate,
          addresses.output,
      }};

  if (addresses.workspace_capacity_bytes <
      receipt.required_bytes[index(Role::kWorkspace)]) {
    receipt.error =
        GdnPromptWideChunkGraphBufferContractError::kInsufficientWorkspace;
    receipt.conflict_left = Role::kWorkspace;
    return receipt;
  }
  if (!make_gdn_prompt_wide_byte_range(
          addresses.workspace, addresses.workspace_capacity_bytes,
          receipt.workspace_capacity_range)) {
    receipt.error = addresses.workspace == 0U
                        ? GdnPromptWideChunkGraphBufferContractError::
                              kNullBuffer
                        : GdnPromptWideChunkGraphBufferContractError::
                              kArithmeticOverflow;
    receipt.conflict_left = Role::kWorkspace;
    return receipt;
  }

  for (std::size_t role = 0U; role < begins.size(); ++role) {
    const auto typed_role = static_cast<Role>(role);
    if (begins[role] == 0U) {
      receipt.error =
          GdnPromptWideChunkGraphBufferContractError::kNullBuffer;
      receipt.conflict_left = typed_role;
      return receipt;
    }
    const std::size_t alignment =
        typed_role == Role::kWorkspace
            ? kGdnPromptWideChunkGraphAlignment
            : alignof(std::uint16_t);
    if ((begins[role] & (alignment - 1U)) != 0U) {
      receipt.error =
          GdnPromptWideChunkGraphBufferContractError::kMisalignedBuffer;
      receipt.conflict_left = typed_role;
      return receipt;
    }
    if (!make_gdn_prompt_wide_byte_range(
            begins[role], receipt.required_bytes[role],
            receipt.ranges[role])) {
      receipt.error =
          GdnPromptWideChunkGraphBufferContractError::kArithmeticOverflow;
      receipt.conflict_left = typed_role;
      return receipt;
    }
  }

  for (std::size_t left = 0U; left < receipt.ranges.size(); ++left) {
    for (std::size_t right = left + 1U; right < receipt.ranges.size();
         ++right) {
      const auto left_role = static_cast<Role>(left);
      const auto right_role = static_cast<Role>(right);
      if (!gdn_prompt_wide_ranges_overlap(receipt.ranges[left],
                                          receipt.ranges[right])) {
        continue;
      }
      const bool is_state_pair =
          left_role == Role::kStateInput &&
          right_role == Role::kStateOutput;
      if (is_state_pair && gdn_prompt_wide_ranges_exact(
                               receipt.ranges[left],
                               receipt.ranges[right])) {
        receipt.state_in_place = true;
        continue;
      }
      receipt.error =
          GdnPromptWideChunkGraphBufferContractError::kForbiddenOverlap;
      receipt.conflict_left = left_role;
      receipt.conflict_right = right_role;
      return receipt;
    }
  }

  receipt.error = GdnPromptWideChunkGraphBufferContractError::kNone;
  return receipt;
}

[[nodiscard]] inline GdnPromptWideChunkGraphBufferContractReceipt
inspect_gdn_prompt_wide_chunk_graph_buffer_contract(
    void* const workspace, const std::size_t workspace_capacity_bytes,
    const std::uint16_t* const raw_qkv,
    const std::size_t token_count,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const conv_history,
    std::uint16_t* const conv_qkv_output,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate,
    std::uint16_t* const output) noexcept {
  const GdnPromptWideChunkGraphBufferAddresses addresses{
      reinterpret_cast<std::uintptr_t>(workspace),
      workspace_capacity_bytes,
      reinterpret_cast<std::uintptr_t>(raw_qkv),
      reinterpret_cast<std::uintptr_t>(conv_weight),
      reinterpret_cast<std::uintptr_t>(conv_history),
      reinterpret_cast<std::uintptr_t>(conv_qkv_output),
      reinterpret_cast<std::uintptr_t>(a),
      reinterpret_cast<std::uintptr_t>(b),
      reinterpret_cast<std::uintptr_t>(A_log),
      reinterpret_cast<std::uintptr_t>(dt_bias),
      reinterpret_cast<std::uintptr_t>(state_input),
      reinterpret_cast<std::uintptr_t>(state_output),
      reinterpret_cast<std::uintptr_t>(norm_weight),
      reinterpret_cast<std::uintptr_t>(silu_gate),
      reinterpret_cast<std::uintptr_t>(output),
  };
  return make_gdn_prompt_wide_chunk_graph_buffer_contract_receipt(
      token_count, addresses);
}

inline constexpr GdnPromptWideChunkGraphWorkspaceLifetimeReceipt
    kGdnPromptWideChunkGraphP40WorkspaceLifetimeReceipt =
        make_gdn_prompt_wide_chunk_graph_workspace_lifetime_receipt(
            kGdnPromptWideChunkGraphP40WorkspacePlan);
static_assert(kGdnPromptWideChunkGraphP40WorkspaceLifetimeReceipt.ok());

}  // namespace q3x::kernels
