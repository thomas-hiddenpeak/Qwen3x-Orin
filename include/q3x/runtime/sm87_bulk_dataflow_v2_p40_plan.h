#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_attention_l2_cohort.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_projection.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_p40_plan.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_projection.h"
#include "q3x/runtime/reference_runner.h"
#include "q3x/runtime/sm87_bulk_dataflow_v2_p40_projection_successor.h"
#include "q3x/runtime/sm87_target_aot_request_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {

// Frozen whole-request host contract for AC-PREFILL-SM87-BULK-DATAFLOW-v2.
// This is deliberately a new identity rather than a selector bolted onto the
// rejected target-AOT-v1 executor.  It describes the exact P40000 execution
// graph, arena ownership, startup seal and request receipt that the executable
// runner must implement.  The contract is default-off and grants no numerical,
// performance, API, or production qualification by itself.
inline constexpr std::array<std::uint8_t, 8U>
    kSm87BulkV2P40PlanMagic{{'Q', '3', 'X', 'B', 'V', '2', 'P', '1'}};
// ABI 3 replaces the segmented FP8/NVFP4 counters in the terminal receipt
// with the versioned whole-P40000 successor receipt.  The old controls remain
// available to numerical oracles but cannot close this request contract.
inline constexpr std::uint16_t kSm87BulkV2P40PlanAbiMajor = 3U;
inline constexpr std::uint16_t kSm87BulkV2P40PlanAbiMinor = 0U;

[[nodiscard]] constexpr bool sm87_bulk_v2_p40_magic_equal(
    const std::array<std::uint8_t, 8U>& left,
    const std::array<std::uint8_t, 8U>& right) noexcept {
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (left[index] != right[index]) {
      return false;
    }
  }
  return true;
}

inline constexpr std::size_t kSm87BulkV2P40Tokens = 40'000U;
inline constexpr std::size_t kSm87BulkV2P40HandoffTokens = 1U;
inline constexpr std::size_t kSm87BulkV2P40Layers = 64U;
inline constexpr std::size_t kSm87BulkV2P40GdnLayers = 48U;
inline constexpr std::size_t kSm87BulkV2P40FullLayers = 16U;
inline constexpr std::size_t kSm87BulkV2P40Hidden = 5'120U;
inline constexpr std::size_t kSm87BulkV2P40Intermediate = 17'408U;
inline constexpr std::size_t kSm87BulkV2P40AttentionWidth = 6'144U;
inline constexpr std::size_t kSm87BulkV2P40GdnRawWidth = 16'384U;
inline constexpr std::size_t kSm87BulkV2P40GdnAbWidth = 96U;
inline constexpr std::size_t kSm87BulkV2P40AttentionQGateWidth = 12'288U;
inline constexpr std::size_t kSm87BulkV2P40Bf16Bytes = 2U;
inline constexpr std::size_t kSm87BulkV2P40ArenaAlignment = 256U;
inline constexpr std::size_t kSm87BulkV2P40Vocabulary = 248'320U;
inline constexpr std::size_t kSm87BulkV2P40LogicalProjectionRoles = 496U;
inline constexpr std::size_t kSm87BulkV2P40FusedOuterOperations = 304U;
inline constexpr std::uint64_t
    kSm87BulkV2P40ProjectionConventionalOperations =
        1'948'044'492'800'000ULL;

[[nodiscard]] constexpr std::uint64_t sm87_bulk_v2_p40_align_up(
    const std::uint64_t value, const std::uint64_t alignment) noexcept {
  return alignment == 0U || value >
                                std::numeric_limits<std::uint64_t>::max() -
                                    (alignment - 1U)
             ? 0U
             : ((value + alignment - 1U) / alignment) * alignment;
}

// The first executor version reuses the already admitted single allocation
// capacity, but rebinds it through this independent v2 lifetime plan.  No v1
// layer-event or host-wait semantics are inherited by that reuse.
inline constexpr std::uint64_t kSm87BulkV2P40PersistentBytes =
    2'699'952'128ULL;
inline constexpr std::uint64_t kSm87BulkV2P40ResidualBytes =
    409'610'240ULL;
inline constexpr std::uint64_t kSm87BulkV2P40FamilyArenaBytes =
    1'966'080'000ULL;
inline constexpr std::uint64_t kSm87BulkV2P40FinalHiddenBytes = 10'240ULL;
inline constexpr std::uint64_t kSm87BulkV2P40RequestArenaBytes =
    5'075'652'608ULL;
inline constexpr std::uint64_t kSm87BulkV2P40FamilyArenaOffset =
    kSm87BulkV2P40PersistentBytes + kSm87BulkV2P40ResidualBytes;

inline constexpr std::uint64_t kSm87BulkV2P40NormalizedBytes =
    kSm87BulkV2P40Tokens * kSm87BulkV2P40Hidden *
    kSm87BulkV2P40Bf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2P40GdnRawBytes =
    kSm87BulkV2P40Tokens * kSm87BulkV2P40GdnRawWidth *
    kSm87BulkV2P40Bf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2P40GdnAbBytes =
    kSm87BulkV2P40Tokens * kSm87BulkV2P40GdnAbWidth *
    kSm87BulkV2P40Bf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2P40GdnOutputBytes =
    kSm87BulkV2P40Tokens * kSm87BulkV2P40AttentionWidth *
    kSm87BulkV2P40Bf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2P40AttentionRawQGateBytes =
    kSm87BulkV2P40Tokens * kSm87BulkV2P40AttentionQGateWidth *
    kSm87BulkV2P40Bf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2P40AttentionQOrGateBytes =
    kSm87BulkV2P40GdnOutputBytes;
inline constexpr std::uint64_t kSm87BulkV2P40NvFp4GroupScratchBytes =
    256ULL * kSm87BulkV2P40Intermediate * kSm87BulkV2P40Bf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2P40GdnPrivateBytes =
    q3x::kernels::kSm87BulkV2GdnP40PrivateBytes;
inline constexpr std::uint64_t kSm87BulkV2P40GdnStateBytesPerLayer =
    1'572'864ULL;
inline constexpr std::uint64_t kSm87BulkV2P40GdnHistoryBytesPerLayer =
    61'440ULL;
inline constexpr std::uint64_t kSm87BulkV2P40ColdResetBytes =
    kSm87BulkV2P40GdnLayers *
    (kSm87BulkV2P40GdnStateBytesPerLayer +
     kSm87BulkV2P40GdnHistoryBytesPerLayer);
inline constexpr std::uint64_t kSm87BulkV2P40TokenIdsBytes =
    kSm87BulkV2P40Tokens * sizeof(std::uint32_t);
inline constexpr std::uint64_t kSm87BulkV2P40FinalLogitsBytes =
    kSm87BulkV2P40Vocabulary * kSm87BulkV2P40Bf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2P40FinalGreedyBytes =
    33ULL * 8ULL;

struct Sm87BulkV2P40Range final {
  std::uint64_t offset = 0U;
  std::uint64_t bytes = 0U;

  [[nodiscard]] constexpr std::uint64_t end() const noexcept {
    return offset + bytes;
  }
  [[nodiscard]] constexpr bool valid(
      const std::uint64_t capacity) const noexcept {
    return bytes != 0U && offset <= capacity && bytes <= capacity - offset;
  }
};

[[nodiscard]] constexpr bool sm87_bulk_v2_p40_ranges_overlap(
    const Sm87BulkV2P40Range& left,
    const Sm87BulkV2P40Range& right) noexcept {
  return left.offset < right.end() && right.offset < left.end();
}

enum class Sm87BulkV2P40FamilyPhase : std::uint8_t {
  kEmbedding = 0U,
  kGdnInput,
  kGdnCore,
  kGdnStatePublish,
  kGdnOutputProjection,
  kFullInput,
  kFullPreprocess,
  kFullAttentionCore,
  kFullOutputProjection,
  kMlp,
  kFinalHandoff,
  kCount,
};

enum class Sm87BulkV2P40BufferRole : std::uint8_t {
  kInvalid = 0U,
  kTokenIds,
  kNormalized,
  kGdnRawQkvz,
  kGdnAb,
  kGdnOutput,
  kGdnOBranch,
  kGdnPrivate,
  kAttentionRawQGate,
  kAttentionProcessedQ,
  kAttentionProcessedGate,
  kAttentionPreGateOutput,
  kAttentionGatedOutput,
  kAttentionOBranch,
  kNvFp4GroupScratch,
  kFinalLogits,
  kFinalGreedyWorkspace,
};

struct Sm87BulkV2P40BufferBinding final {
  Sm87BulkV2P40BufferRole role = Sm87BulkV2P40BufferRole::kInvalid;
  Sm87BulkV2P40Range range{};
  std::uint32_t live_phase_mask = 0U;
};

[[nodiscard]] constexpr std::uint32_t sm87_bulk_v2_p40_phase_bit(
    const Sm87BulkV2P40FamilyPhase phase) noexcept {
  return 1U << static_cast<std::uint32_t>(phase);
}

inline constexpr std::size_t kSm87BulkV2P40BufferBindings = 16U;

struct Sm87BulkV2P40FamilyArenaPlan final {
  std::uint64_t family_bytes = 0U;
  std::array<Sm87BulkV2P40BufferBinding,
             kSm87BulkV2P40BufferBindings>
      bindings{};
  std::uint64_t embedding_required_extent_bytes = 0U;
  std::uint64_t gdn_required_extent_bytes = 0U;
  std::uint64_t attention_required_extent_bytes = 0U;
  std::uint64_t mlp_required_extent_bytes = 0U;
  std::uint64_t mlp_live_bytes = 0U;
  std::uint64_t final_handoff_required_extent_bytes = 0U;
  std::uint64_t data_plane_cold_reset_bytes = 0U;
  bool cold_initial_state_is_persistent_layer_state = false;
  bool final_gdn_state_copied_after_epilogue = false;
  bool whole_arena_request_memset_required = true;
};

[[nodiscard]] constexpr Sm87BulkV2P40FamilyArenaPlan
sm87_bulk_v2_p40_family_arena_plan() noexcept {
  constexpr auto bit = [](const Sm87BulkV2P40FamilyPhase phase) {
    return 1U << static_cast<std::uint32_t>(phase);
  };
  constexpr std::uint64_t kTail =
      kSm87BulkV2P40GdnRawBytes + kSm87BulkV2P40GdnAbBytes;
  constexpr std::uint64_t kPrivate =
      kTail + kSm87BulkV2P40GdnOutputBytes;
  return {
      kSm87BulkV2P40FamilyArenaBytes,
      {{
          {Sm87BulkV2P40BufferRole::kTokenIds,
           {0U, kSm87BulkV2P40TokenIdsBytes},
           bit(Sm87BulkV2P40FamilyPhase::kEmbedding)},
          {Sm87BulkV2P40BufferRole::kNormalized,
           {kTail, kSm87BulkV2P40NormalizedBytes},
           bit(Sm87BulkV2P40FamilyPhase::kGdnInput) |
               bit(Sm87BulkV2P40FamilyPhase::kFullInput) |
               bit(Sm87BulkV2P40FamilyPhase::kMlp)},
          {Sm87BulkV2P40BufferRole::kGdnRawQkvz,
           {0U, kSm87BulkV2P40GdnRawBytes},
           bit(Sm87BulkV2P40FamilyPhase::kGdnInput) |
               bit(Sm87BulkV2P40FamilyPhase::kGdnCore)},
          {Sm87BulkV2P40BufferRole::kGdnAb,
           {kSm87BulkV2P40GdnRawBytes, kSm87BulkV2P40GdnAbBytes},
           bit(Sm87BulkV2P40FamilyPhase::kGdnInput) |
               bit(Sm87BulkV2P40FamilyPhase::kGdnCore)},
          {Sm87BulkV2P40BufferRole::kGdnOutput,
           {kTail, kSm87BulkV2P40GdnOutputBytes},
           bit(Sm87BulkV2P40FamilyPhase::kGdnCore) |
               bit(Sm87BulkV2P40FamilyPhase::kGdnOutputProjection)},
          {Sm87BulkV2P40BufferRole::kGdnOBranch,
           {0U, kSm87BulkV2P40NormalizedBytes},
           bit(Sm87BulkV2P40FamilyPhase::kGdnOutputProjection)},
          {Sm87BulkV2P40BufferRole::kGdnPrivate,
           {kPrivate, kSm87BulkV2P40GdnPrivateBytes},
           bit(Sm87BulkV2P40FamilyPhase::kGdnCore) |
               bit(Sm87BulkV2P40FamilyPhase::kGdnStatePublish)},
          {Sm87BulkV2P40BufferRole::kAttentionRawQGate,
           {0U, kSm87BulkV2P40AttentionRawQGateBytes},
           bit(Sm87BulkV2P40FamilyPhase::kFullInput) |
               bit(Sm87BulkV2P40FamilyPhase::kFullPreprocess)},
          {Sm87BulkV2P40BufferRole::kAttentionProcessedQ,
           {kSm87BulkV2P40AttentionRawQGateBytes,
            kSm87BulkV2P40AttentionQOrGateBytes},
           bit(Sm87BulkV2P40FamilyPhase::kFullPreprocess) |
               bit(Sm87BulkV2P40FamilyPhase::kFullAttentionCore)},
          {Sm87BulkV2P40BufferRole::kAttentionProcessedGate,
           {kSm87BulkV2P40AttentionRawQGateBytes +
                kSm87BulkV2P40AttentionQOrGateBytes,
            kSm87BulkV2P40AttentionQOrGateBytes},
           bit(Sm87BulkV2P40FamilyPhase::kFullPreprocess) |
               bit(Sm87BulkV2P40FamilyPhase::kFullAttentionCore)},
          {Sm87BulkV2P40BufferRole::kAttentionPreGateOutput,
           {0U, kSm87BulkV2P40AttentionQOrGateBytes},
           bit(Sm87BulkV2P40FamilyPhase::kFullAttentionCore)},
          {Sm87BulkV2P40BufferRole::kAttentionGatedOutput,
           {kSm87BulkV2P40AttentionQOrGateBytes,
            kSm87BulkV2P40AttentionQOrGateBytes},
           bit(Sm87BulkV2P40FamilyPhase::kFullAttentionCore) |
               bit(Sm87BulkV2P40FamilyPhase::kFullOutputProjection)},
          {Sm87BulkV2P40BufferRole::kAttentionOBranch,
           {kSm87BulkV2P40AttentionRawQGateBytes,
            kSm87BulkV2P40NormalizedBytes},
           bit(Sm87BulkV2P40FamilyPhase::kFullOutputProjection)},
          {Sm87BulkV2P40BufferRole::kNvFp4GroupScratch,
           {0U, kSm87BulkV2P40NvFp4GroupScratchBytes},
           bit(Sm87BulkV2P40FamilyPhase::kMlp)},
          {Sm87BulkV2P40BufferRole::kFinalLogits,
           {0U, kSm87BulkV2P40FinalLogitsBytes},
           bit(Sm87BulkV2P40FamilyPhase::kFinalHandoff)},
          {Sm87BulkV2P40BufferRole::kFinalGreedyWorkspace,
           {kSm87BulkV2P40FinalLogitsBytes,
            kSm87BulkV2P40FinalGreedyBytes},
           bit(Sm87BulkV2P40FamilyPhase::kFinalHandoff)},
      }},
      kSm87BulkV2P40TokenIdsBytes,
      kPrivate + kSm87BulkV2P40GdnPrivateBytes,
      kSm87BulkV2P40AttentionRawQGateBytes +
          2U * kSm87BulkV2P40AttentionQOrGateBytes,
      kTail + kSm87BulkV2P40NormalizedBytes,
      kSm87BulkV2P40NormalizedBytes +
          kSm87BulkV2P40NvFp4GroupScratchBytes,
      kSm87BulkV2P40FinalLogitsBytes +
          kSm87BulkV2P40FinalGreedyBytes,
      kSm87BulkV2P40ColdResetBytes,
      true,
      true,
      false,
  };
}

[[nodiscard]] constexpr bool sm87_bulk_v2_p40_family_arena_plan_valid(
    const Sm87BulkV2P40FamilyArenaPlan& plan) noexcept {
  constexpr auto expected = sm87_bulk_v2_p40_family_arena_plan();
  if (plan.family_bytes != kSm87BulkV2P40FamilyArenaBytes ||
      plan.embedding_required_extent_bytes !=
          expected.embedding_required_extent_bytes ||
      plan.gdn_required_extent_bytes !=
          expected.gdn_required_extent_bytes ||
      plan.attention_required_extent_bytes != plan.family_bytes ||
      plan.mlp_required_extent_bytes !=
          expected.mlp_required_extent_bytes ||
      plan.mlp_live_bytes != expected.mlp_live_bytes ||
      plan.final_handoff_required_extent_bytes !=
          expected.final_handoff_required_extent_bytes ||
      plan.data_plane_cold_reset_bytes != kSm87BulkV2P40ColdResetBytes ||
      !plan.cold_initial_state_is_persistent_layer_state ||
      !plan.final_gdn_state_copied_after_epilogue ||
      plan.whole_arena_request_memset_required) {
    return false;
  }
  constexpr std::uint32_t kKnownPhaseBits =
      (1U << static_cast<std::uint32_t>(
           Sm87BulkV2P40FamilyPhase::kCount)) -
      1U;
  std::uint32_t observed_roles = 0U;
  for (std::size_t index = 0U; index < plan.bindings.size(); ++index) {
    const auto& binding = plan.bindings[index];
    const auto& canonical = expected.bindings[index];
    if (binding.role == Sm87BulkV2P40BufferRole::kInvalid ||
        binding.live_phase_mask == 0U ||
        (binding.live_phase_mask & ~kKnownPhaseBits) != 0U ||
        !binding.range.valid(plan.family_bytes) ||
        binding.range.offset % kSm87BulkV2P40ArenaAlignment != 0U ||
        binding.role != canonical.role ||
        binding.range.offset != canonical.range.offset ||
        binding.range.bytes != canonical.range.bytes ||
        binding.live_phase_mask != canonical.live_phase_mask) {
      return false;
    }
    const auto role = static_cast<std::uint32_t>(binding.role);
    if (role >= 32U || (observed_roles & (1U << role)) != 0U) {
      return false;
    }
    observed_roles |= 1U << role;
  }
  for (std::size_t first = 0U; first < plan.bindings.size(); ++first) {
    for (std::size_t second = first + 1U; second < plan.bindings.size();
         ++second) {
      const auto& left = plan.bindings[first];
      const auto& right = plan.bindings[second];
      if (sm87_bulk_v2_p40_ranges_overlap(left.range, right.range) &&
          (left.live_phase_mask & right.live_phase_mask) != 0U) {
        return false;
      }
    }
  }
  return observed_roles == ((1U << (kSm87BulkV2P40BufferBindings + 1U)) -
                            2U);
}

// Control-plane storage is independent of the 5.075-GB data allocation. It
// is startup-owned and re-epochized for each request. The exact host-mapped
// cancellation pair is deliberately counted separately, so the 78,446,592-B
// state/history reset is never misreported as the complete request reset.
enum class Sm87BulkV2P40ControlRole : std::uint8_t {
  kInvalid = 0U,
  kNvFp4DeviceControl,
  kRequestProgress,
  kFp8CancellationSnapshot,
  kBf16CancellationSnapshot,
  kAttentionCancellationSnapshot,
  kNvFp4CancellationSnapshot,
};

struct Sm87BulkV2P40ControlBinding final {
  Sm87BulkV2P40ControlRole role = Sm87BulkV2P40ControlRole::kInvalid;
  Sm87BulkV2P40Range range{};
  std::uint64_t alignment = 0U;
  std::uint32_t live_phase_mask = 0U;
};

inline constexpr std::uint64_t kSm87BulkV2P40MappedCancellationBytes =
    sizeof(std::uint32_t);
inline constexpr std::uint64_t kSm87BulkV2P40RequestProgressBytes = 64U;
inline constexpr std::uint64_t kSm87BulkV2P40NvFp4DeviceControlBytes =
    sizeof(q3x::kernels::Sm87BulkV2NvFp4DeviceControl);
inline constexpr std::uint64_t kSm87BulkV2P40ControlArenaBytes = 1'280U;
inline constexpr std::size_t kSm87BulkV2P40ControlBindings = 6U;

struct Sm87BulkV2P40ControlPlanePlan final {
  std::uint64_t device_arena_bytes = 0U;
  std::uint64_t device_reset_bytes = 0U;
  std::uint64_t mapped_host_reset_bytes = 0U;
  std::array<Sm87BulkV2P40ControlBinding,
             kSm87BulkV2P40ControlBindings>
      bindings{};
  std::size_t gdn_cancel_boundary_tokens = 0U;
  std::size_t fp8_cancel_boundary_tokens = 0U;
  std::size_t bf16_ab_cancel_boundary_tokens = 0U;
  std::size_t attention_cancel_boundary_epochs = 0U;
  std::size_t nvfp4_cancel_boundary_tokens = 0U;
  bool exact_mapped_host_device_pair_required = false;
  bool progress_is_request_epoch_scoped = false;
  bool kernels_publish_safe_progress = false;
};

[[nodiscard]] constexpr Sm87BulkV2P40ControlPlanePlan
sm87_bulk_v2_p40_control_plane_plan() noexcept {
  constexpr auto bit = [](const Sm87BulkV2P40FamilyPhase phase) {
    return 1U << static_cast<std::uint32_t>(phase);
  };
  constexpr std::uint32_t kAllPhases =
      (1U << static_cast<std::uint32_t>(
           Sm87BulkV2P40FamilyPhase::kCount)) -
      1U;
  return {
      kSm87BulkV2P40ControlArenaBytes,
      kSm87BulkV2P40ControlArenaBytes,
      kSm87BulkV2P40MappedCancellationBytes,
      {{
          {Sm87BulkV2P40ControlRole::kNvFp4DeviceControl,
           {0U, kSm87BulkV2P40NvFp4DeviceControlBytes}, 64U,
           bit(Sm87BulkV2P40FamilyPhase::kMlp)},
          {Sm87BulkV2P40ControlRole::kRequestProgress,
           {1'152U, kSm87BulkV2P40RequestProgressBytes}, 64U, kAllPhases},
          {Sm87BulkV2P40ControlRole::kFp8CancellationSnapshot,
           {1'216U, sizeof(std::uint32_t)}, 16U,
           bit(Sm87BulkV2P40FamilyPhase::kGdnInput) |
               bit(Sm87BulkV2P40FamilyPhase::kGdnOutputProjection) |
               bit(Sm87BulkV2P40FamilyPhase::kFullInput) |
               bit(Sm87BulkV2P40FamilyPhase::kFullOutputProjection)},
          {Sm87BulkV2P40ControlRole::kBf16CancellationSnapshot,
           {1'232U, sizeof(std::uint32_t)}, 16U,
           bit(Sm87BulkV2P40FamilyPhase::kGdnInput)},
          {Sm87BulkV2P40ControlRole::kAttentionCancellationSnapshot,
           {1'248U, sizeof(std::uint32_t)}, 16U,
           bit(Sm87BulkV2P40FamilyPhase::kFullPreprocess) |
               bit(Sm87BulkV2P40FamilyPhase::kFullAttentionCore)},
          {Sm87BulkV2P40ControlRole::kNvFp4CancellationSnapshot,
           {1'264U, sizeof(std::uint32_t)}, 16U,
           bit(Sm87BulkV2P40FamilyPhase::kMlp)},
      }},
      64U,
      1'024U,
      40'000U,
      1U,
      256U,
      true,
      true,
      true,
  };
}

[[nodiscard]] constexpr bool sm87_bulk_v2_p40_control_plane_plan_valid(
    const Sm87BulkV2P40ControlPlanePlan& plan) noexcept {
  constexpr auto expected = sm87_bulk_v2_p40_control_plane_plan();
  if (plan.device_arena_bytes != expected.device_arena_bytes ||
      plan.device_reset_bytes != expected.device_reset_bytes ||
      plan.mapped_host_reset_bytes != expected.mapped_host_reset_bytes ||
      plan.gdn_cancel_boundary_tokens != 64U ||
      plan.fp8_cancel_boundary_tokens != 1'024U ||
      plan.bf16_ab_cancel_boundary_tokens != 40'000U ||
      plan.attention_cancel_boundary_epochs != 1U ||
      plan.nvfp4_cancel_boundary_tokens != 256U ||
      !plan.exact_mapped_host_device_pair_required ||
      !plan.progress_is_request_epoch_scoped ||
      !plan.kernels_publish_safe_progress) {
    return false;
  }
  std::uint32_t roles = 0U;
  for (std::size_t index = 0U; index < plan.bindings.size(); ++index) {
    const auto& binding = plan.bindings[index];
    const auto& canonical = expected.bindings[index];
    const auto role = static_cast<std::uint32_t>(binding.role);
    if (binding.role != canonical.role ||
        binding.range.offset != canonical.range.offset ||
        binding.range.bytes != canonical.range.bytes ||
        binding.alignment != canonical.alignment ||
        binding.live_phase_mask != canonical.live_phase_mask ||
        !binding.range.valid(plan.device_arena_bytes) ||
        binding.alignment == 0U ||
        binding.range.offset % binding.alignment != 0U || role == 0U ||
        role >= 32U || (roles & (1U << role)) != 0U) {
      return false;
    }
    roles |= 1U << role;
  }
  for (std::size_t first = 0U; first < plan.bindings.size(); ++first) {
    for (std::size_t second = first + 1U; second < plan.bindings.size();
         ++second) {
      if (sm87_bulk_v2_p40_ranges_overlap(plan.bindings[first].range,
                                           plan.bindings[second].range)) {
        return false;
      }
    }
  }
  return roles == ((1U << (kSm87BulkV2P40ControlBindings + 1U)) - 2U);
}

enum class Sm87BulkV2P40Stream : std::uint8_t {
  kMain = 0U,
  kProjectionAndGdnProducer,
  kBf16Ab,
  kGdnRecurrence,
  kGdnEpilogue,
  kCount,
};

enum class Sm87BulkV2P40ReusableEvent : std::uint8_t {
  kNormalizedReady = 0U,
  kProjectionInputReady,
  kBf16AbReady,
  kCoreReady,
  kProjectionOutputReady,
  kGdnPrepared0,
  kGdnPrepared1,
  kGdnRecurrence0,
  kGdnRecurrence1,
  kGdnEpilogue0,
  kGdnEpilogue1,
  kRequestTerminal,
  kCount,
};

inline constexpr std::size_t kSm87BulkV2P40StreamCount =
    static_cast<std::size_t>(Sm87BulkV2P40Stream::kCount);
inline constexpr std::size_t kSm87BulkV2P40ReusableEventCount =
    static_cast<std::size_t>(Sm87BulkV2P40ReusableEvent::kCount);

enum Sm87BulkV2P40Policy : std::uint64_t {
  kSm87BulkV2P40ExactP40000 = 1ULL << 0U,
  kSm87BulkV2P40ColdNoCache = 1ULL << 1U,
  kSm87BulkV2P40FiveStreams = 1ULL << 2U,
  kSm87BulkV2P40DeviceOrderedLayerProgress = 1ULL << 3U,
  kSm87BulkV2P40OneTerminalHostSync = 1ULL << 4U,
  kSm87BulkV2P40NoPerLayerHostDrain = 1ULL << 5U,
  kSm87BulkV2P40StaticChecksAtSeal = 1ULL << 6U,
  kSm87BulkV2P40ZeroHotCudaQueries = 1ULL << 7U,
  kSm87BulkV2P40PartialSubmitDrainsAllStreams = 1ULL << 8U,
  kSm87BulkV2P40PartialSubmitPoisonsOwner = 1ULL << 9U,
  kSm87BulkV2P40AtomicTerminalCommit = 1ULL << 10U,
  kSm87BulkV2P40NoFallback = 1ULL << 11U,
  kSm87BulkV2P40NoMtp = 1ULL << 12U,
  kSm87BulkV2P40NoCublasLt = 1ULL << 13U,
  kSm87BulkV2P40NoRequestJitRepackAutotune = 1ULL << 14U,
  kSm87BulkV2P40NoWholeArenaReset = 1ULL << 15U,
  kSm87BulkV2P40FrozenControlPlane = 1ULL << 16U,
  kSm87BulkV2P40FullApiHandoff = 1ULL << 17U,
};

inline constexpr std::uint64_t kSm87BulkV2P40RequiredPolicy =
    (1ULL << 18U) - 1ULL;

struct Sm87BulkV2P40ExecutionPlan final {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::size_t prompt_tokens = 0U;
  std::size_t handoff_tokens = 0U;
  std::size_t layers = 0U;
  std::size_t gdn_layers = 0U;
  std::size_t full_layers = 0U;
  std::size_t stream_count = 0U;
  std::size_t reusable_event_count = 0U;
  std::size_t maximum_host_waits = 0U;
  std::size_t maximum_host_drains = 0U;
  std::size_t hot_cuda_resource_queries = 0U;
  std::uint64_t policy = 0U;
  Sm87BulkV2P40FamilyArenaPlan family_arena{};
  Sm87BulkV2P40ControlPlanePlan control_plane{};
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr Sm87BulkV2P40ExecutionPlan
sm87_bulk_v2_p40_execution_plan() noexcept {
  return {kSm87BulkV2P40PlanMagic,
          kSm87BulkV2P40PlanAbiMajor,
          kSm87BulkV2P40PlanAbiMinor,
          kSm87BulkV2P40Tokens,
          kSm87BulkV2P40HandoffTokens,
          kSm87BulkV2P40Layers,
          kSm87BulkV2P40GdnLayers,
          kSm87BulkV2P40FullLayers,
          kSm87BulkV2P40StreamCount,
          kSm87BulkV2P40ReusableEventCount,
          1U,
          1U,
          0U,
          kSm87BulkV2P40RequiredPolicy,
          sm87_bulk_v2_p40_family_arena_plan(),
          sm87_bulk_v2_p40_control_plane_plan(),
          false};
}

[[nodiscard]] constexpr bool sm87_bulk_v2_p40_execution_plan_valid(
    const Sm87BulkV2P40ExecutionPlan& plan) noexcept {
  return sm87_bulk_v2_p40_magic_equal(plan.magic,
                                      kSm87BulkV2P40PlanMagic) &&
         plan.abi_major == kSm87BulkV2P40PlanAbiMajor &&
         plan.abi_minor == kSm87BulkV2P40PlanAbiMinor &&
         plan.prompt_tokens == kSm87BulkV2P40Tokens &&
         plan.handoff_tokens == kSm87BulkV2P40HandoffTokens &&
         plan.layers == kSm87BulkV2P40Layers &&
         plan.gdn_layers == kSm87BulkV2P40GdnLayers &&
         plan.full_layers == kSm87BulkV2P40FullLayers &&
         plan.gdn_layers + plan.full_layers == plan.layers &&
         plan.stream_count == kSm87BulkV2P40StreamCount &&
         plan.reusable_event_count == kSm87BulkV2P40ReusableEventCount &&
         plan.maximum_host_waits == 1U && plan.maximum_host_drains == 1U &&
         plan.hot_cuda_resource_queries == 0U &&
         plan.policy == kSm87BulkV2P40RequiredPolicy &&
         sm87_bulk_v2_p40_family_arena_plan_valid(plan.family_arena) &&
         sm87_bulk_v2_p40_control_plane_plan_valid(plan.control_plane) &&
         !plan.production_dispatch_eligible;
}

enum class Sm87BulkV2P40SealState : std::uint8_t {
  kInvalid = 0U,
  kSealed,
};

// Created once at engine readiness after complete real-asset, allocation,
// stream/event, device, resource, pointer-extent and route authentication.
// The nonce and identities are implementation-issued, never caller assertions.
struct Sm87BulkV2P40SealedExecutionAccess final {
  std::array<std::uint8_t, 8U> plan_magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  Sm87BulkV2P40SealState state = Sm87BulkV2P40SealState::kInvalid;
  std::uint64_t deployment_identity = 0U;
  std::uint64_t model_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t stream_event_owner_identity = 0U;
  std::uint64_t asset_catalog_identity = 0U;
  std::uint64_t binary_evidence_identity = 0U;
  std::uint64_t fp8_oracle_evidence_identity = 0U;
  std::uint64_t attention_oracle_evidence_identity = 0U;
  std::uint64_t gdn_oracle_evidence_identity = 0U;
  std::uint64_t nvfp4_oracle_evidence_identity = 0U;
  std::uint64_t seal_nonce = 0U;
  std::int32_t device_ordinal = -1;
  bool all_static_checks_complete = false;
  bool request_time_revalidation_required = true;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_p40_sealed_access_valid(
    const Sm87BulkV2P40SealedExecutionAccess& access) noexcept {
  return sm87_bulk_v2_p40_magic_equal(access.plan_magic,
                                      kSm87BulkV2P40PlanMagic) &&
         access.abi_major == kSm87BulkV2P40PlanAbiMajor &&
         access.abi_minor == kSm87BulkV2P40PlanAbiMinor &&
         access.state == Sm87BulkV2P40SealState::kSealed &&
         access.deployment_identity != 0U && access.model_identity != 0U &&
         access.allocation_identity != 0U &&
         access.stream_event_owner_identity != 0U &&
         access.asset_catalog_identity != 0U &&
         access.binary_evidence_identity != 0U &&
         access.fp8_oracle_evidence_identity != 0U &&
         access.attention_oracle_evidence_identity != 0U &&
         access.gdn_oracle_evidence_identity != 0U &&
         access.nvfp4_oracle_evidence_identity != 0U &&
         access.seal_nonce != 0U &&
         access.device_ordinal >= 0 && access.all_static_checks_complete &&
         !access.request_time_revalidation_required &&
         !access.production_dispatch_eligible;
}

enum class Sm87BulkV2P40OwnerLifecycle : std::uint8_t {
  kReady = 0U,
  kActive,
  kDraining,
  kCompleted,
  kCancelled,
  kPoisoned,
};

struct Sm87BulkV2P40RequestReceipt final {
  std::array<std::uint8_t, 8U> plan_magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  Sm87BulkV2P40OwnerLifecycle lifecycle =
      Sm87BulkV2P40OwnerLifecycle::kReady;
  std::uint64_t seal_nonce = 0U;
  std::uint64_t request_epoch = 0U;
  std::uint64_t deployment_identity = 0U;
  std::uint64_t model_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t stream_event_owner_identity = 0U;
  std::uint64_t asset_catalog_identity = 0U;
  std::size_t completed_layers = 0U;
  std::size_t completed_gdn_layers = 0U;
  std::size_t completed_full_layers = 0U;
  std::size_t closed_layer_residuals = 0U;
  std::size_t closed_gdn_state_publications = 0U;
  std::size_t logical_projection_roles = 0U;
  std::size_t fused_outer_operations = 0U;
  std::uint64_t projection_conventional_operations = 0U;
  Sm87BulkV2P40ProjectionSuccessorReceipt projection_successor{};
  std::size_t enqueued_attention_launches = 0U;
  std::size_t enqueued_attention_preprocess_panels = 0U;
  std::size_t enqueued_bf16_ab_launches = 0U;
  std::size_t enqueued_gdn_producer_chunks = 0U;
  std::size_t enqueued_gdn_recurrence_chunks = 0U;
  std::size_t enqueued_gdn_epilogue_chunks = 0U;
  std::size_t enqueued_gdn_persistent_copies = 0U;
  std::size_t enqueued_final_norm = 0U;
  std::size_t enqueued_lm_head = 0U;
  std::size_t enqueued_argmax = 0U;
  std::size_t enqueued_handoff_d2h = 0U;
  std::size_t terminal_host_waits = 0U;
  std::size_t terminal_host_drains = 0U;
  std::size_t last_submitted_layer =
      std::numeric_limits<std::size_t>::max();
  std::size_t last_submitted_segment =
      std::numeric_limits<std::size_t>::max();
  std::size_t last_submitted_constituent =
      std::numeric_limits<std::size_t>::max();
  Sm87BulkV2P40FamilyPhase last_submitted_family =
      Sm87BulkV2P40FamilyPhase::kCount;
  std::uint32_t handoff_token_id =
      std::numeric_limits<std::uint32_t>::max();
  std::uint32_t handoff_nonfinite = 1U;
  std::int32_t first_error = 0;
  bool submission_started = false;
  bool cancellation_published = false;
  bool all_streams_drained = false;
  bool state_committed = false;
  bool handoff_observed = false;
  bool used_fallback = false;
  bool used_mtp = false;
  bool used_cublaslt = false;
  bool used_request_jit_repack_or_autotune = false;
};

[[nodiscard]] constexpr Sm87BulkV2P40RequestReceipt
sm87_bulk_v2_p40_request_receipt(
    const Sm87BulkV2P40SealedExecutionAccess& access,
    const std::uint64_t request_epoch) noexcept {
  Sm87BulkV2P40RequestReceipt receipt;
  if (!sm87_bulk_v2_p40_sealed_access_valid(access) || request_epoch == 0U) {
    receipt.lifecycle = Sm87BulkV2P40OwnerLifecycle::kPoisoned;
    return receipt;
  }
  receipt.plan_magic = access.plan_magic;
  receipt.abi_major = access.abi_major;
  receipt.abi_minor = access.abi_minor;
  receipt.seal_nonce = access.seal_nonce;
  receipt.request_epoch = request_epoch;
  receipt.deployment_identity = access.deployment_identity;
  receipt.model_identity = access.model_identity;
  receipt.allocation_identity = access.allocation_identity;
  receipt.stream_event_owner_identity =
      access.stream_event_owner_identity;
  receipt.asset_catalog_identity = access.asset_catalog_identity;
  receipt.projection_successor =
      sm87_bulk_v2_p40_projection_successor_receipt();
  return receipt;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_p40_receipt_terminal_valid(
    const Sm87BulkV2P40SealedExecutionAccess& access,
    const Sm87BulkV2P40RequestReceipt& receipt) noexcept {
  if (!sm87_bulk_v2_p40_sealed_access_valid(access) ||
      !sm87_bulk_v2_p40_magic_equal(receipt.plan_magic,
                                    kSm87BulkV2P40PlanMagic) ||
      receipt.abi_major != kSm87BulkV2P40PlanAbiMajor ||
      receipt.abi_minor != kSm87BulkV2P40PlanAbiMinor ||
      receipt.seal_nonce != access.seal_nonce ||
      receipt.request_epoch == 0U ||
      receipt.deployment_identity != access.deployment_identity ||
      receipt.model_identity != access.model_identity ||
      receipt.allocation_identity != access.allocation_identity ||
      receipt.stream_event_owner_identity !=
          access.stream_event_owner_identity ||
      receipt.asset_catalog_identity != access.asset_catalog_identity ||
      receipt.used_fallback || receipt.used_mtp || receipt.used_cublaslt ||
      receipt.used_request_jit_repack_or_autotune) {
    return false;
  }
  if (receipt.lifecycle == Sm87BulkV2P40OwnerLifecycle::kCompleted) {
    return receipt.submission_started && receipt.all_streams_drained &&
           receipt.state_committed && receipt.first_error == 0 &&
           receipt.completed_layers == kSm87BulkV2P40Layers &&
           receipt.completed_gdn_layers == kSm87BulkV2P40GdnLayers &&
           receipt.completed_full_layers == kSm87BulkV2P40FullLayers &&
           receipt.closed_layer_residuals == kSm87BulkV2P40Layers &&
           receipt.closed_gdn_state_publications ==
               kSm87BulkV2P40GdnLayers &&
           receipt.logical_projection_roles ==
               kSm87BulkV2P40LogicalProjectionRoles &&
           receipt.fused_outer_operations ==
               kSm87BulkV2P40FusedOuterOperations &&
           receipt.projection_conventional_operations ==
               kSm87BulkV2P40ProjectionConventionalOperations &&
           sm87_bulk_v2_p40_projection_successor_receipt_complete(
               receipt.projection_successor) &&
           receipt.enqueued_attention_launches ==
               kSm87BulkV2P40FullLayers *
                   q3x::kernels::kSm87BulkV2AttentionKernelLaunches &&
           receipt.enqueued_attention_preprocess_panels == 80U &&
           receipt.enqueued_bf16_ab_launches == 48U &&
           receipt.enqueued_gdn_producer_chunks == 30'000U &&
           receipt.enqueued_gdn_recurrence_chunks == 30'000U &&
           receipt.enqueued_gdn_epilogue_chunks == 30'000U &&
           receipt.enqueued_gdn_persistent_copies == 96U &&
           receipt.enqueued_final_norm == 1U &&
           receipt.enqueued_lm_head == 1U &&
           receipt.enqueued_argmax == 1U &&
           receipt.enqueued_handoff_d2h == 1U &&
           receipt.terminal_host_waits == 1U &&
           receipt.terminal_host_drains == 1U &&
           receipt.last_submitted_layer == kSm87BulkV2P40Layers - 1U &&
           receipt.last_submitted_family ==
               Sm87BulkV2P40FamilyPhase::kFinalHandoff &&
           receipt.last_submitted_segment == 0U &&
           receipt.last_submitted_constituent == 3U &&
           receipt.handoff_observed && receipt.handoff_nonfinite == 0U &&
           receipt.handoff_token_id < kSm87BulkV2P40Vocabulary;
  }
  if (receipt.lifecycle == Sm87BulkV2P40OwnerLifecycle::kCancelled ||
      receipt.lifecycle == Sm87BulkV2P40OwnerLifecycle::kPoisoned) {
    const bool partial_evidence =
        receipt.last_submitted_layer !=
            std::numeric_limits<std::size_t>::max() &&
        receipt.last_submitted_family != Sm87BulkV2P40FamilyPhase::kCount;
    const bool reason_valid =
        receipt.lifecycle == Sm87BulkV2P40OwnerLifecycle::kPoisoned
            ? receipt.first_error != 0
            : receipt.first_error == 0;
    return receipt.submission_started && receipt.cancellation_published &&
           receipt.all_streams_drained && !receipt.state_committed &&
           !receipt.handoff_observed && partial_evidence && reason_valid;
  }
  return false;
}

inline constexpr auto kSm87BulkV2P40FrozenExecutionPlan =
    sm87_bulk_v2_p40_execution_plan();

static_assert(kSm87BulkV2P40FamilyArenaOffset == 3'109'562'368ULL);
static_assert(kSm87BulkV2P40FamilyArenaOffset +
                  kSm87BulkV2P40FamilyArenaBytes +
                  kSm87BulkV2P40FinalHiddenBytes ==
              kSm87BulkV2P40RequestArenaBytes);
static_assert(kSm87BulkV2P40GdnRawBytes == 1'310'720'000ULL);
static_assert(kSm87BulkV2P40GdnAbBytes == 7'680'000ULL);
static_assert(kSm87BulkV2P40GdnOutputBytes == 491'520'000ULL);
static_assert(kSm87BulkV2P40AttentionRawQGateBytes == 983'040'000ULL);
static_assert(kSm87BulkV2P40AttentionQOrGateBytes == 491'520'000ULL);
static_assert(kSm87BulkV2P40NvFp4GroupScratchBytes == 8'912'896ULL);
static_assert(kSm87BulkV2P40LogicalProjectionRoles ==
                  kSm87BulkV2P40ProjectionLogicalRoles &&
              kSm87BulkV2P40FusedOuterOperations ==
                  kSm87BulkV2P40ProjectionFusedOuterOperations &&
              kSm87BulkV2P40ProjectionConventionalOperations ==
                  kSm87BulkV2P40SuccessorProjectionConventionalOperations);
static_assert(kSm87BulkV2P40ColdResetBytes == 78'446'592ULL);
static_assert(kSm87BulkV2P40GdnPrivateBytes == 6'988'032ULL);
static_assert(kSm87BulkV2P40GdnPrivateBytes ==
              q3x::kernels::kSm87BulkV2GdnP40PrivateBytes);
static_assert(kSm87BulkV2P40Tokens ==
                  q3x::kernels::kSm87BulkV2Fp8P40Tokens &&
              kSm87BulkV2P40Tokens ==
                  q3x::kernels::kSm87BulkV2AttentionTokens &&
              kSm87BulkV2P40Tokens ==
                  q3x::kernels::kSm87BulkV2GdnP40Tokens &&
              kSm87BulkV2P40Tokens ==
                  q3x::kernels::kSm87BulkV2NvFp4P40Tokens &&
              kSm87BulkV2P40Tokens == kSm87TargetAotP40PromptTokens);
static_assert(kSm87BulkV2P40Layers ==
                  q3x::kernels::kSm87BulkV2Fp8LayerCount &&
              kSm87BulkV2P40Layers ==
                  q3x::kernels::kSm87BulkV2NvFp4LayerCount &&
              kSm87BulkV2P40Layers == kSm87TargetAotP40LayerCount);
static_assert(kSm87BulkV2P40GdnLayers ==
                  q3x::kernels::kSm87BulkV2GdnP40SessionLayerCount &&
              kSm87BulkV2P40GdnLayers ==
                  kSm87TargetAotP40GdnLayerCount);
static_assert(kSm87BulkV2P40FullLayers ==
              kSm87TargetAotP40FullLayerCount);
static_assert(kSm87BulkV2P40NvFp4GroupScratchBytes ==
              q3x::kernels::kSm87BulkV2NvFp4GroupScratchBytes);
static_assert(kSm87BulkV2P40NvFp4DeviceControlBytes ==
                  sizeof(q3x::kernels::Sm87BulkV2NvFp4DeviceControl) &&
              alignof(q3x::kernels::Sm87BulkV2NvFp4DeviceControl) == 64U);
static_assert(kSm87BulkV2P40Vocabulary == kReferenceVocabularySize);
static_assert(kSm87BulkV2P40PersistentBytes ==
                  kSm87TargetAotP40PersistentBytes &&
              kSm87BulkV2P40ResidualBytes ==
                  kSm87TargetAotP40ResidualBytes &&
              kSm87BulkV2P40FamilyArenaBytes ==
                  kSm87TargetAotP40FamilyArenaBytes &&
              kSm87BulkV2P40RequestArenaBytes ==
                  kSm87TargetAotP40RequestArenaBytes);
static_assert(sm87_bulk_v2_p40_control_plane_plan_valid(
    sm87_bulk_v2_p40_control_plane_plan()));
static_assert(sm87_bulk_v2_p40_execution_plan_valid(
    kSm87BulkV2P40FrozenExecutionPlan));

}  // namespace q3x::runtime
