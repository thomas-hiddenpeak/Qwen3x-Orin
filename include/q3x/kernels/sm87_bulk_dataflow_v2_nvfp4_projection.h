#pragma once

#include "q3x/kernels/sm87_target_aot_projection_cuda.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off exact-control stepping stone for
// AC-PREFILL-SM87-BULK-DATAFLOW-v2.  This file freezes the host ABI,
// authenticated byte view, exact work ownership, cancellation/progress
// protocol, and CUDA experiment envelope.  Its executable oracle proves the
// current arithmetic/control path only.  The implementation intentionally
// claims neither cross-group weight residency nor a sealed P40 hot path, does
// not bind a runner selector, and grants no performance or production
// qualification.
inline constexpr std::array<std::uint8_t, 8U>
    kSm87BulkV2NvFp4Magic{{'Q', '3', 'X', 'N', 'V', '2', 'P', '1'}};
inline constexpr std::uint16_t kSm87BulkV2NvFp4AbiMajor = 2U;
inline constexpr std::uint16_t kSm87BulkV2NvFp4AbiMinor = 0U;

inline constexpr std::uint32_t kSm87BulkV2NvFp4P40Tokens = 40'000U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4MacroTokens = 1'024U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4TailTokens = 64U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4FullMacroSegments = 39U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4SegmentsPerLayer = 40U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4LayerCount = 64U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4RoleCount = 128U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4PhysicalMacroLaunches =
    kSm87BulkV2NvFp4LayerCount * kSm87BulkV2NvFp4SegmentsPerLayer;

inline constexpr std::uint32_t kSm87BulkV2NvFp4Hidden = 5'120U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4Intermediate = 17'408U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4TileM = 64U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4GateTileN = 64U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownTileN = 256U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4TileK = 64U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4GateNTiles =
    kSm87BulkV2NvFp4Intermediate / kSm87BulkV2NvFp4GateTileN;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownNTiles =
    kSm87BulkV2NvFp4Hidden / kSm87BulkV2NvFp4DownTileN;
inline constexpr std::uint32_t kSm87BulkV2NvFp4GateKTiles =
    kSm87BulkV2NvFp4Hidden / kSm87BulkV2NvFp4TileK;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownKTiles =
    kSm87BulkV2NvFp4Intermediate / kSm87BulkV2NvFp4TileK;

inline constexpr std::uint32_t kSm87BulkV2NvFp4GroupTokens = 256U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4RowsPerGroup = 4U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4GroupsPerMacro = 4U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4PersistentCtas = 32U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownOwnerCtas = 20U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DedicatedProducerCtas = 12U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4Threads = 256U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4Warps = 8U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4PipelineStages = 3U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4RegisterStages = 2U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4ReadinessSlots = 32U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4RequiredCtasPerSm = 2U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4SmCount = 16U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4MaximumRegisters = 128U;

inline constexpr std::uint32_t kSm87BulkV2NvFp4ActivationBytesPerStage =
    kSm87BulkV2NvFp4TileM * kSm87BulkV2NvFp4TileK *
    sizeof(std::uint16_t);
inline constexpr std::uint32_t kSm87BulkV2NvFp4GateBranchBytesPerStage =
    kSm87BulkV2NvFp4GateTileN * kSm87BulkV2NvFp4TileK / 2U +
    kSm87BulkV2NvFp4GateTileN *
        (kSm87BulkV2NvFp4TileK / 16U);
inline constexpr std::uint32_t kSm87BulkV2NvFp4GateBytesPerStage =
    kSm87BulkV2NvFp4ActivationBytesPerStage +
    2U * kSm87BulkV2NvFp4GateBranchBytesPerStage;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DownBytesPerStage =
    kSm87BulkV2NvFp4ActivationBytesPerStage +
    kSm87BulkV2NvFp4DownTileN * kSm87BulkV2NvFp4TileK / 2U +
    kSm87BulkV2NvFp4DownTileN *
        (kSm87BulkV2NvFp4TileK / 16U);
inline constexpr std::uint32_t kSm87BulkV2NvFp4GateDynamicSharedBytes =
    kSm87BulkV2NvFp4PipelineStages *
    kSm87BulkV2NvFp4GateBytesPerStage;
inline constexpr std::uint32_t kSm87BulkV2NvFp4DynamicSharedBytes =
    kSm87BulkV2NvFp4PipelineStages *
    kSm87BulkV2NvFp4DownBytesPerStage;

inline constexpr std::uint64_t kSm87BulkV2NvFp4GroupScratchBytes =
    static_cast<std::uint64_t>(kSm87BulkV2NvFp4GroupTokens) *
    kSm87BulkV2NvFp4Intermediate * sizeof(std::uint16_t);
inline constexpr std::uint64_t kSm87BulkV2NvFp4NormalizedGroupBytes =
    static_cast<std::uint64_t>(kSm87BulkV2NvFp4GroupTokens) *
    kSm87BulkV2NvFp4Hidden * sizeof(std::uint16_t);
inline constexpr std::uint64_t kSm87BulkV2NvFp4HotReadyWindowBytesPerRow =
    static_cast<std::uint64_t>(kSm87BulkV2NvFp4ReadinessSlots) *
    kSm87BulkV2NvFp4TileM * kSm87BulkV2NvFp4GateTileN *
    sizeof(std::uint16_t);
// Every one of the four M64 rows owns an independent 32-credit ring.  Count
// the complete concurrently claimable M256 domain; a single-row estimate is
// not a conservative L2 admission bound for this work-stealing schedule.
inline constexpr std::uint64_t kSm87BulkV2NvFp4HotReadyWindowBytes =
    kSm87BulkV2NvFp4RowsPerGroup *
    kSm87BulkV2NvFp4HotReadyWindowBytesPerRow;
inline constexpr std::uint64_t kSm87BulkV2NvFp4AggregateL2BudgetBytes =
    kSm87BulkV2NvFp4NormalizedGroupBytes +
    kSm87BulkV2NvFp4HotReadyWindowBytes;

inline constexpr std::uint64_t kSm87BulkV2NvFp4GatePartitionPayloadBytes =
    50'135'040ULL;
inline constexpr std::uint64_t kSm87BulkV2NvFp4GateUpPayloadBytes =
    100'270'080ULL;
inline constexpr std::uint64_t kSm87BulkV2NvFp4DownPayloadBytes =
    50'135'040ULL;
inline constexpr std::uint64_t kSm87BulkV2NvFp4LayerPayloadBytes =
    kSm87BulkV2NvFp4GateUpPayloadBytes +
    kSm87BulkV2NvFp4DownPayloadBytes;
inline constexpr std::uint64_t kSm87BulkV2NvFp4FamilyPayloadBytes =
    kSm87BulkV2NvFp4LayerCount * kSm87BulkV2NvFp4LayerPayloadBytes;

enum class Sm87BulkV2NvFp4ExecutionIdentity : std::uint16_t {
  kInvalid = 0U,
  kExactControlSteppingStoneM256J64N256K64NoResidencyV2,
};

enum class Sm87BulkV2NvFp4PayloadViewIdentity : std::uint16_t {
  kInvalid = 0U,
  // The bytes retain the authenticated target-AOT
  // [K16][N64][N8][lane][component] representation.  V2 changes only the
  // consumer view and never repacks, decodes, or copies the payload.
  kTargetAotAuthenticatedNvFp4ByteViewV2,
};

enum class Sm87BulkV2NvFp4LogicalPayloadRole : std::uint8_t {
  kInvalid = 0U,
  kGate,
  kUp,
  kDown,
};

enum Sm87BulkV2NvFp4Policy : std::uint64_t {
  kSm87BulkV2NvFp4Bf16Activation = 1ULL << 0U,
  kSm87BulkV2NvFp4Fp32MmaAccumulation = 1ULL << 1U,
  kSm87BulkV2NvFp4AscendingFullKSingleCta = 1ULL << 2U,
  kSm87BulkV2NvFp4IndependentGateUpPartitions = 1ULL << 3U,
  kSm87BulkV2NvFp4PartitionLocalFp32Scale = 1ULL << 4U,
  kSm87BulkV2NvFp4Bf16RnePublication = 1ULL << 5U,
  kSm87BulkV2NvFp4NoSplitK = 1ULL << 6U,
  kSm87BulkV2NvFp4NoReductionWorkspace = 1ULL << 7U,
  kSm87BulkV2NvFp4NoRequestRepack = 1ULL << 8U,
  kSm87BulkV2NvFp4NoRequestJit = 1ULL << 9U,
  kSm87BulkV2NvFp4NoCublasLt = 1ULL << 10U,
  kSm87BulkV2NvFp4NoMtp = 1ULL << 11U,
  kSm87BulkV2NvFp4NoActivationQuantization = 1ULL << 12U,
  kSm87BulkV2NvFp4CooperativeFixed32 = 1ULL << 13U,
  kSm87BulkV2NvFp4TwentyDownOwners = 1ULL << 14U,
  kSm87BulkV2NvFp4TwelveDedicatedProducers = 1ULL << 15U,
  kSm87BulkV2NvFp4ConsumerWorkStealing = 1ULL << 16U,
  kSm87BulkV2NvFp4ReadinessCredit32 = 1ULL << 17U,
  kSm87BulkV2NvFp4CancellationSuppressesPrivatePublication = 1ULL << 18U,
  kSm87BulkV2NvFp4JointReceiptOnly = 1ULL << 19U,
  kSm87BulkV2NvFp4NoProductionSelector = 1ULL << 20U,
  kSm87BulkV2NvFp4AcaBcg = 1ULL << 21U,
  kSm87BulkV2NvFp4ThreeStageG2S = 1ULL << 22U,
  kSm87BulkV2NvFp4TwoStageS2R = 1ULL << 23U,
  kSm87BulkV2NvFp4NoCrossGroupWeightResidencyQualification = 1ULL << 24U,
  kSm87BulkV2NvFp4NoP40HotPathQualification = 1ULL << 25U,
};

inline constexpr std::uint64_t kSm87BulkV2NvFp4RequiredPolicy =
    kSm87BulkV2NvFp4Bf16Activation |
    kSm87BulkV2NvFp4Fp32MmaAccumulation |
    kSm87BulkV2NvFp4AscendingFullKSingleCta |
    kSm87BulkV2NvFp4IndependentGateUpPartitions |
    kSm87BulkV2NvFp4PartitionLocalFp32Scale |
    kSm87BulkV2NvFp4Bf16RnePublication |
    kSm87BulkV2NvFp4NoSplitK |
    kSm87BulkV2NvFp4NoReductionWorkspace |
    kSm87BulkV2NvFp4NoRequestRepack |
    kSm87BulkV2NvFp4NoRequestJit |
    kSm87BulkV2NvFp4NoCublasLt | kSm87BulkV2NvFp4NoMtp |
    kSm87BulkV2NvFp4NoActivationQuantization |
    kSm87BulkV2NvFp4CooperativeFixed32 |
    kSm87BulkV2NvFp4TwentyDownOwners |
    kSm87BulkV2NvFp4TwelveDedicatedProducers |
    kSm87BulkV2NvFp4ConsumerWorkStealing |
    kSm87BulkV2NvFp4ReadinessCredit32 |
    kSm87BulkV2NvFp4CancellationSuppressesPrivatePublication |
    kSm87BulkV2NvFp4JointReceiptOnly |
    kSm87BulkV2NvFp4NoProductionSelector |
    kSm87BulkV2NvFp4AcaBcg |
    kSm87BulkV2NvFp4ThreeStageG2S |
    kSm87BulkV2NvFp4TwoStageS2R |
    kSm87BulkV2NvFp4NoCrossGroupWeightResidencyQualification |
    kSm87BulkV2NvFp4NoP40HotPathQualification;

struct Sm87BulkV2NvFp4SegmentPlan final {
  std::uint32_t segment = 0U;
  std::uint32_t first_token = 0U;
  std::uint32_t token_count = 0U;
  std::uint32_t m_tiles = 0U;
  std::uint32_t group_count = 0U;
  std::uint32_t gate_tasks = 0U;
  std::uint32_t down_tasks = 0U;
  bool tail = false;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4SegmentPlan
sm87_bulk_v2_nvfp4_segment_plan(const std::size_t segment) noexcept {
  if (segment >= kSm87BulkV2NvFp4SegmentsPerLayer) {
    return {};
  }
  const bool tail = segment == kSm87BulkV2NvFp4FullMacroSegments;
  const std::uint32_t tokens =
      tail ? kSm87BulkV2NvFp4TailTokens
           : kSm87BulkV2NvFp4MacroTokens;
  const std::uint32_t m_tiles = tokens / kSm87BulkV2NvFp4TileM;
  const std::uint32_t groups =
      (m_tiles + kSm87BulkV2NvFp4RowsPerGroup - 1U) /
      kSm87BulkV2NvFp4RowsPerGroup;
  return {static_cast<std::uint32_t>(segment),
          static_cast<std::uint32_t>(segment) *
              kSm87BulkV2NvFp4MacroTokens,
          tokens,
          m_tiles,
          groups,
          m_tiles * kSm87BulkV2NvFp4GateNTiles,
          m_tiles * kSm87BulkV2NvFp4DownNTiles,
          tail,
          true};
}

struct Sm87BulkV2NvFp4GroupPlan final {
  std::uint32_t segment = 0U;
  std::uint32_t group = 0U;
  std::uint32_t first_m_tile = 0U;
  std::uint32_t active_rows = 0U;
  std::uint32_t gate_tasks = 0U;
  std::uint32_t down_tasks = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4GroupPlan
sm87_bulk_v2_nvfp4_group_plan(const std::size_t segment,
                              const std::size_t group) noexcept {
  const auto macro = sm87_bulk_v2_nvfp4_segment_plan(segment);
  if (!macro.valid || group >= macro.group_count) {
    return {};
  }
  const std::uint32_t first =
      static_cast<std::uint32_t>(group) *
      kSm87BulkV2NvFp4RowsPerGroup;
  const std::uint32_t remaining = macro.m_tiles - first;
  const std::uint32_t rows =
      remaining < kSm87BulkV2NvFp4RowsPerGroup
          ? remaining
          : kSm87BulkV2NvFp4RowsPerGroup;
  return {static_cast<std::uint32_t>(segment),
          static_cast<std::uint32_t>(group), first, rows,
          rows * kSm87BulkV2NvFp4GateNTiles,
          rows * kSm87BulkV2NvFp4DownNTiles, true};
}

struct Sm87BulkV2NvFp4GateTask final {
  std::uint32_t segment = 0U;
  std::uint32_t group = 0U;
  std::uint32_t local_row = 0U;
  std::uint32_t segment_m_tile = 0U;
  std::uint32_t q = 0U;
  std::uint32_t first_token = 0U;
  std::uint32_t first_output = 0U;
  std::uint32_t k_tiles = 0U;
  std::uint32_t ordinal = 0U;
  bool full_k_single_cta = false;
  bool valid = false;
};

// Gate tasks are row-major.  Every row's readiness ring is independent, so
// producer CTAs can choose an eligible row without changing task identity.
[[nodiscard]] constexpr Sm87BulkV2NvFp4GateTask
sm87_bulk_v2_nvfp4_gate_task(const Sm87BulkV2NvFp4GroupPlan& group,
                             const std::size_t ordinal) noexcept {
  if (!group.valid || ordinal >= group.gate_tasks) {
    return {};
  }
  const std::uint32_t row = static_cast<std::uint32_t>(
      ordinal / kSm87BulkV2NvFp4GateNTiles);
  const std::uint32_t q = static_cast<std::uint32_t>(
      ordinal % kSm87BulkV2NvFp4GateNTiles);
  const auto macro = sm87_bulk_v2_nvfp4_segment_plan(group.segment);
  const std::uint32_t m_tile = group.first_m_tile + row;
  return {group.segment,
          group.group,
          row,
          m_tile,
          q,
          macro.first_token + m_tile * kSm87BulkV2NvFp4TileM,
          q * kSm87BulkV2NvFp4GateTileN,
          kSm87BulkV2NvFp4GateKTiles,
          static_cast<std::uint32_t>(ordinal),
          true,
          true};
}

struct Sm87BulkV2NvFp4DownTask final {
  std::uint32_t segment = 0U;
  std::uint32_t group = 0U;
  std::uint32_t owner_cta = 0U;
  std::uint32_t local_row = 0U;
  std::uint32_t segment_m_tile = 0U;
  std::uint32_t n_tile = 0U;
  std::uint32_t first_token = 0U;
  std::uint32_t first_output = 0U;
  std::uint32_t q_steps = 0U;
  bool q_strictly_ascending = false;
  bool full_k_single_cta = false;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4DownTask
sm87_bulk_v2_nvfp4_down_task(const Sm87BulkV2NvFp4GroupPlan& group,
                             const std::size_t owner_cta,
                             const std::size_t local_row) noexcept {
  if (!group.valid || owner_cta >= kSm87BulkV2NvFp4DownOwnerCtas ||
      local_row >= group.active_rows) {
    return {};
  }
  const auto macro = sm87_bulk_v2_nvfp4_segment_plan(group.segment);
  const std::uint32_t m_tile =
      group.first_m_tile + static_cast<std::uint32_t>(local_row);
  return {group.segment,
          group.group,
          static_cast<std::uint32_t>(owner_cta),
          static_cast<std::uint32_t>(local_row),
          m_tile,
          static_cast<std::uint32_t>(owner_cta),
          macro.first_token + m_tile * kSm87BulkV2NvFp4TileM,
          static_cast<std::uint32_t>(owner_cta) *
              kSm87BulkV2NvFp4DownTileN,
          kSm87BulkV2NvFp4GateNTiles,
          true,
          true,
          true};
}

[[nodiscard]] constexpr std::uint32_t
sm87_bulk_v2_nvfp4_readiness_slot(const std::uint32_t q) noexcept {
  return q % kSm87BulkV2NvFp4ReadinessSlots;
}

[[nodiscard]] constexpr std::uint32_t
sm87_bulk_v2_nvfp4_readiness_generation(
    const std::uint32_t q) noexcept {
  return q / kSm87BulkV2NvFp4ReadinessSlots + 1U;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_nvfp4_credit_available(
    const std::uint32_t next_q, const std::uint32_t retired_q) noexcept {
  return next_q < kSm87BulkV2NvFp4GateNTiles &&
         next_q < retired_q + kSm87BulkV2NvFp4ReadinessSlots;
}

// Selects one eligible row for an atomic-CAS claim.  The returned row is a
// candidate only: the device implementation must still CAS gate_next_q[row]
// and retry on contention.  Per-row cursors make every (row,q) task unique
// without a global queue that can head-of-line block on a full future row.
[[nodiscard]] constexpr std::uint32_t
sm87_bulk_v2_nvfp4_claimable_row(
    const std::array<std::uint32_t,
                     kSm87BulkV2NvFp4RowsPerGroup>& next_q,
    const std::array<std::uint32_t,
                     kSm87BulkV2NvFp4RowsPerGroup>& retired_q,
    const std::uint32_t active_rows,
    const std::uint32_t scan_seed) noexcept {
  if (active_rows == 0U ||
      active_rows > kSm87BulkV2NvFp4RowsPerGroup) {
    return kSm87BulkV2NvFp4RowsPerGroup;
  }
  for (std::uint32_t offset = 0U; offset < active_rows; ++offset) {
    const std::uint32_t row = (scan_seed + offset) % active_rows;
    if (sm87_bulk_v2_nvfp4_credit_available(next_q[row], retired_q[row])) {
      return row;
    }
  }
  return kSm87BulkV2NvFp4RowsPerGroup;
}

struct Sm87BulkV2NvFp4PayloadFragmentView final {
  Sm87BulkV2NvFp4LogicalPayloadRole role =
      Sm87BulkV2NvFp4LogicalPayloadRole::kInvalid;
  std::uint32_t partition_index = 0U;
  std::uint32_t v2_n_tile = 0U;
  std::uint32_t k_tile = 0U;
  std::uint32_t local_n8_panel = 0U;
  std::uint32_t source_n_tile = 0U;
  std::uint32_t source_n_warp = 0U;
  std::uint32_t source_n8_panel = 0U;
  std::uint64_t source_fragment_ordinal = 0U;
  Sm87TargetAotProjectionPackedFragment authenticated{};
  bool same_authenticated_bytes = false;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4PayloadFragmentView
sm87_bulk_v2_nvfp4_payload_fragment(
    const Sm87BulkV2NvFp4LogicalPayloadRole role,
    const std::size_t v2_n_tile, const std::size_t k_tile,
    const std::size_t k16,
    const std::size_t local_n8_panel) noexcept {
  const bool gate_or_up =
      role == Sm87BulkV2NvFp4LogicalPayloadRole::kGate ||
      role == Sm87BulkV2NvFp4LogicalPayloadRole::kUp;
  const bool down = role == Sm87BulkV2NvFp4LogicalPayloadRole::kDown;
  if ((!gate_or_up && !down) || k16 >= 4U) {
    return {};
  }
  const auto projection_role =
      gate_or_up ? Sm87TargetAotProjectionRole::kNvFp4GateUp
                 : Sm87TargetAotProjectionRole::kNvFp4Down;
  const auto layout =
      sm87_target_aot_projection_packed_layout(projection_role);
  const std::uint32_t partition =
      role == Sm87BulkV2NvFp4LogicalPayloadRole::kUp ? 1U : 0U;
  std::uint32_t source_n_tile = 0U;
  std::uint32_t source_n_warp = 0U;
  std::uint32_t source_n8 = 0U;
  if (gate_or_up) {
    if (v2_n_tile >= kSm87BulkV2NvFp4GateNTiles ||
        k_tile >= kSm87BulkV2NvFp4GateKTiles ||
        local_n8_panel >= 8U) {
      return {};
    }
    source_n_tile = static_cast<std::uint32_t>(v2_n_tile / 4U);
    source_n_warp = static_cast<std::uint32_t>(v2_n_tile % 4U);
    source_n8 = static_cast<std::uint32_t>(local_n8_panel);
  } else {
    if (v2_n_tile >= kSm87BulkV2NvFp4DownNTiles ||
        k_tile >= kSm87BulkV2NvFp4DownKTiles ||
        local_n8_panel >= 32U) {
      return {};
    }
    source_n_tile = static_cast<std::uint32_t>(v2_n_tile);
    source_n_warp = static_cast<std::uint32_t>(local_n8_panel / 8U);
    source_n8 = static_cast<std::uint32_t>(local_n8_panel % 8U);
  }
  const auto fragment = sm87_target_aot_projection_packed_fragment(
      layout, partition, source_n_tile, k_tile, k16, source_n_warp,
      source_n8);
  if (!fragment.valid) {
    return {};
  }
  const auto& source = layout.partitions[partition];
  const std::uint64_t fragments_per_cell = 4U * 4U * 8U;
  std::uint64_t partition_fragment_base = 0U;
  for (std::uint32_t index = 0U; index < partition; ++index) {
    const auto& prior = layout.partitions[index];
    partition_fragment_base +=
        static_cast<std::uint64_t>(prior.n_tiles) * prior.k_tiles *
        fragments_per_cell;
  }
  const std::uint64_t ordinal =
      partition_fragment_base +
      (static_cast<std::uint64_t>(source_n_tile) * source.k_tiles +
       k_tile) * fragments_per_cell +
      (k16 * 4U + source_n_warp) * 8U + source_n8;
  return {role,
          partition,
          static_cast<std::uint32_t>(v2_n_tile),
          static_cast<std::uint32_t>(k_tile),
          static_cast<std::uint32_t>(local_n8_panel),
          source_n_tile,
          source_n_warp,
          source_n8,
          ordinal,
          fragment,
          true,
          true};
}

struct Sm87BulkV2NvFp4RoleDescriptor final {
  std::uint32_t ordinal = 0U;
  std::uint32_t layer = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::uint32_t joint_macro_segments = 0U;
  std::uint32_t owned_physical_launches = 0U;
  std::uint64_t payload_bytes = 0U;
  bool joint_with_adjacent_role = false;
  bool owns_joint_launch = false;
};

struct Sm87BulkV2NvFp4FamilyManifest final {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  Sm87BulkV2NvFp4ExecutionIdentity execution_identity =
      Sm87BulkV2NvFp4ExecutionIdentity::kInvalid;
  Sm87BulkV2NvFp4PayloadViewIdentity payload_view_identity =
      Sm87BulkV2NvFp4PayloadViewIdentity::kInvalid;
  std::array<Sm87BulkV2NvFp4RoleDescriptor,
             kSm87BulkV2NvFp4RoleCount>
      roles{};
  std::uint32_t role_count = 0U;
  std::uint32_t layer_count = 0U;
  std::uint32_t physical_macro_launches = 0U;
  std::uint64_t family_payload_bytes = 0U;
  std::uint64_t required_policy = 0U;
  std::uint64_t seal = 0U;
  bool reuses_authenticated_target_aot_bytes = false;
  bool request_time_repack = true;
  bool request_time_jit = true;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr std::uint64_t
sm87_bulk_v2_nvfp4_manifest_hash_byte(std::uint64_t hash,
                                      const std::uint8_t byte) noexcept {
  return (hash ^ byte) * 1'099'511'628'211ULL;
}

[[nodiscard]] constexpr std::uint64_t
sm87_bulk_v2_nvfp4_manifest_hash_u64(std::uint64_t hash,
                                     const std::uint64_t value,
                                     const std::size_t bytes) noexcept {
  for (std::size_t index = 0U; index < bytes; ++index) {
    hash = sm87_bulk_v2_nvfp4_manifest_hash_byte(
        hash, static_cast<std::uint8_t>(value >> (8U * index)));
  }
  return hash;
}

[[nodiscard]] constexpr std::uint64_t
sm87_bulk_v2_nvfp4_compute_manifest_seal(
    const Sm87BulkV2NvFp4FamilyManifest& manifest) noexcept {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  for (const auto byte : manifest.magic) {
    hash = sm87_bulk_v2_nvfp4_manifest_hash_byte(hash, byte);
  }
  const auto add = [&hash](const std::uint64_t value,
                           const std::size_t bytes) constexpr {
    hash = sm87_bulk_v2_nvfp4_manifest_hash_u64(hash, value, bytes);
  };
  add(manifest.abi_major, 2U);
  add(manifest.abi_minor, 2U);
  add(static_cast<std::uint16_t>(manifest.execution_identity), 2U);
  add(static_cast<std::uint16_t>(manifest.payload_view_identity), 2U);
  add(manifest.role_count, 4U);
  add(manifest.layer_count, 4U);
  add(manifest.physical_macro_launches, 4U);
  add(manifest.family_payload_bytes, 8U);
  add(manifest.required_policy, 8U);
  for (const auto& role : manifest.roles) {
    add(role.ordinal, 4U);
    add(role.layer, 4U);
    add(static_cast<std::uint8_t>(role.role), 1U);
    add(role.joint_macro_segments, 4U);
    add(role.owned_physical_launches, 4U);
    add(role.payload_bytes, 8U);
    add(role.joint_with_adjacent_role, 1U);
    add(role.owns_joint_launch, 1U);
  }
  add(manifest.reuses_authenticated_target_aot_bytes, 1U);
  add(manifest.request_time_repack, 1U);
  add(manifest.request_time_jit, 1U);
  add(manifest.numerical_contract_qualified, 1U);
  add(manifest.production_dispatch_eligible, 1U);
  return hash;
}

[[nodiscard]] constexpr Sm87BulkV2NvFp4FamilyManifest
sm87_bulk_v2_nvfp4_family_manifest() noexcept {
  Sm87BulkV2NvFp4FamilyManifest result;
  result.magic = kSm87BulkV2NvFp4Magic;
  result.abi_major = kSm87BulkV2NvFp4AbiMajor;
  result.abi_minor = kSm87BulkV2NvFp4AbiMinor;
  result.execution_identity = Sm87BulkV2NvFp4ExecutionIdentity::
      kExactControlSteppingStoneM256J64N256K64NoResidencyV2;
  result.payload_view_identity = Sm87BulkV2NvFp4PayloadViewIdentity::
      kTargetAotAuthenticatedNvFp4ByteViewV2;
  for (std::uint32_t layer = 0U; layer < kSm87BulkV2NvFp4LayerCount;
       ++layer) {
    const std::uint32_t gate_ordinal = 2U * layer;
    const std::uint32_t down_ordinal = gate_ordinal + 1U;
    result.roles[gate_ordinal] = {
        gate_ordinal, layer, Sm87TargetAotProjectionRole::kNvFp4GateUp,
        kSm87BulkV2NvFp4SegmentsPerLayer,
        kSm87BulkV2NvFp4SegmentsPerLayer,
        kSm87BulkV2NvFp4GateUpPayloadBytes, true, true};
    result.roles[down_ordinal] = {
        down_ordinal, layer, Sm87TargetAotProjectionRole::kNvFp4Down,
        kSm87BulkV2NvFp4SegmentsPerLayer, 0U,
        kSm87BulkV2NvFp4DownPayloadBytes, true, false};
  }
  result.role_count = kSm87BulkV2NvFp4RoleCount;
  result.layer_count = kSm87BulkV2NvFp4LayerCount;
  result.physical_macro_launches = kSm87BulkV2NvFp4PhysicalMacroLaunches;
  result.family_payload_bytes = kSm87BulkV2NvFp4FamilyPayloadBytes;
  result.required_policy = kSm87BulkV2NvFp4RequiredPolicy;
  result.reuses_authenticated_target_aot_bytes = true;
  result.request_time_repack = false;
  result.request_time_jit = false;
  result.numerical_contract_qualified = false;
  result.production_dispatch_eligible = false;
  result.seal = sm87_bulk_v2_nvfp4_compute_manifest_seal(result);
  return result;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_nvfp4_manifest_valid(
    const Sm87BulkV2NvFp4FamilyManifest& manifest) noexcept {
  for (std::size_t index = 0U; index < manifest.magic.size(); ++index) {
    if (manifest.magic[index] != kSm87BulkV2NvFp4Magic[index]) {
      return false;
    }
  }
  if (manifest.abi_major != kSm87BulkV2NvFp4AbiMajor ||
      manifest.abi_minor != kSm87BulkV2NvFp4AbiMinor ||
      manifest.execution_identity != Sm87BulkV2NvFp4ExecutionIdentity::
          kExactControlSteppingStoneM256J64N256K64NoResidencyV2 ||
      manifest.payload_view_identity !=
          Sm87BulkV2NvFp4PayloadViewIdentity::
              kTargetAotAuthenticatedNvFp4ByteViewV2 ||
      manifest.role_count != kSm87BulkV2NvFp4RoleCount ||
      manifest.layer_count != kSm87BulkV2NvFp4LayerCount ||
      manifest.physical_macro_launches !=
          kSm87BulkV2NvFp4PhysicalMacroLaunches ||
      manifest.family_payload_bytes !=
          kSm87BulkV2NvFp4FamilyPayloadBytes ||
      manifest.required_policy != kSm87BulkV2NvFp4RequiredPolicy ||
      !manifest.reuses_authenticated_target_aot_bytes ||
      manifest.request_time_repack || manifest.request_time_jit ||
      manifest.numerical_contract_qualified ||
      manifest.production_dispatch_eligible || manifest.seal == 0U ||
      manifest.seal !=
          sm87_bulk_v2_nvfp4_compute_manifest_seal(manifest)) {
    return false;
  }
  for (std::uint32_t ordinal = 0U; ordinal < manifest.role_count;
       ++ordinal) {
    const auto& role = manifest.roles[ordinal];
    const bool gate = (ordinal & 1U) == 0U;
    if (role.ordinal != ordinal || role.layer != ordinal / 2U ||
        role.role !=
            (gate ? Sm87TargetAotProjectionRole::kNvFp4GateUp
                  : Sm87TargetAotProjectionRole::kNvFp4Down) ||
        role.joint_macro_segments !=
            kSm87BulkV2NvFp4SegmentsPerLayer ||
        role.owned_physical_launches !=
            (gate ? kSm87BulkV2NvFp4SegmentsPerLayer : 0U) ||
        role.payload_bytes !=
            (gate ? kSm87BulkV2NvFp4GateUpPayloadBytes
                  : kSm87BulkV2NvFp4DownPayloadBytes) ||
        !role.joint_with_adjacent_role ||
        role.owns_joint_launch != gate) {
      return false;
    }
  }
  return true;
}

struct Sm87BulkV2NvFp4AuthenticatedAssets final {
  Sm87TargetAotNvFp4CudaAssetView gate_up{};
  Sm87TargetAotNvFp4CudaAssetView down{};
  Sm87BulkV2NvFp4ExecutionIdentity execution_identity =
      Sm87BulkV2NvFp4ExecutionIdentity::kInvalid;
  Sm87BulkV2NvFp4PayloadViewIdentity payload_view_identity =
      Sm87BulkV2NvFp4PayloadViewIdentity::kInvalid;
  bool same_authenticated_payload_bytes = false;
  bool no_request_time_repack = false;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4AuthenticatedAssets
sm87_bulk_v2_nvfp4_bind_authenticated_assets(
    const Sm87TargetAotNvFp4CudaAssetView& gate_up,
    const Sm87TargetAotNvFp4CudaAssetView& down) noexcept {
  if (!sm87_target_aot_nvfp4_cuda_asset_valid(gate_up) ||
      !sm87_target_aot_nvfp4_cuda_asset_valid(down) ||
      gate_up.payload.role !=
          Sm87TargetAotProjectionRole::kNvFp4GateUp ||
      down.payload.role != Sm87TargetAotProjectionRole::kNvFp4Down ||
      gate_up.device_upload_receipt.device_allocation_owner_identity !=
          down.device_upload_receipt.device_allocation_owner_identity) {
    return {};
  }
  return {gate_up,
          down,
          Sm87BulkV2NvFp4ExecutionIdentity::
              kExactControlSteppingStoneM256J64N256K64NoResidencyV2,
          Sm87BulkV2NvFp4PayloadViewIdentity::
              kTargetAotAuthenticatedNvFp4ByteViewV2,
          true,
          true,
          false};
}

[[nodiscard]] constexpr bool sm87_bulk_v2_nvfp4_assets_valid(
    const Sm87BulkV2NvFp4AuthenticatedAssets& assets) noexcept {
  return assets.execution_identity == Sm87BulkV2NvFp4ExecutionIdentity::
             kExactControlSteppingStoneM256J64N256K64NoResidencyV2 &&
         assets.payload_view_identity ==
             Sm87BulkV2NvFp4PayloadViewIdentity::
                 kTargetAotAuthenticatedNvFp4ByteViewV2 &&
         assets.same_authenticated_payload_bytes &&
         assets.no_request_time_repack &&
         !assets.production_dispatch_eligible &&
         sm87_target_aot_nvfp4_cuda_asset_valid(assets.gate_up) &&
         sm87_target_aot_nvfp4_cuda_asset_valid(assets.down) &&
         assets.gate_up.payload.role ==
             Sm87TargetAotProjectionRole::kNvFp4GateUp &&
         assets.down.payload.role ==
             Sm87TargetAotProjectionRole::kNvFp4Down &&
         assets.gate_up.device_upload_receipt
                 .device_allocation_owner_identity ==
             assets.down.device_upload_receipt
                 .device_allocation_owner_identity;
}

// Reused once per M256 group. Readiness is a bounded producer/consumer seam,
// not a numerical reduction or a replacement for the global BF16 H boundary.
struct alignas(64) Sm87BulkV2NvFp4DeviceControl final {
  std::uint32_t gate_next_q[kSm87BulkV2NvFp4RowsPerGroup]{};
  std::uint32_t retired_q[kSm87BulkV2NvFp4RowsPerGroup]{};
  std::uint32_t readiness_generation
      [kSm87BulkV2NvFp4RowsPerGroup]
      [kSm87BulkV2NvFp4ReadinessSlots]{};
  std::uint32_t readers_remaining
      [kSm87BulkV2NvFp4RowsPerGroup]
      [kSm87BulkV2NvFp4ReadinessSlots]{};
  std::uint32_t claimed_gate_tasks = 0U;
  std::uint32_t completed_gate_tasks = 0U;
  std::uint32_t completed_down_tasks = 0U;
  std::uint32_t cancellation_observed = 0U;
  std::uint32_t group_epoch = 0U;
  // Per-macro aggregates survive the four internal M256 groups of an M1024
  // segment.  The per-group counters above are reset at each grid-wide group
  // boundary; these fields make one physical macro launch auditable without
  // expanding the frozen 1,152-byte control ABI.
  // Exact maxima are 4 groups, 4,352 Gate tasks, and 320 Down tasks, so the
  // macro receipt is losslessly representable in 16-bit fields.  The compact
  // receipt pays for an aligned cancellation continuation without growing the
  // frozen control ABI.
  std::uint16_t macro_completed_groups = 0U;
  std::uint16_t macro_claimed_gate_tasks = 0U;
  std::uint16_t macro_completed_gate_tasks = 0U;
  std::uint16_t macro_completed_down_tasks = 0U;
  // These bounds are at most four.  Keeping them as two 16-bit receipt
  // fields lets the fixed 1,152-byte ABI carry the numerical continuation
  // state required by the one-launch M1024 driver without a thread-local
  // continuation frame.
  std::uint16_t active_group = 0U;
  std::uint16_t active_rows = 0U;
  std::uint32_t down_tensor_scale_bits = 0U;
  std::uintptr_t numerical_down_payload = 0U;
  // Residual and input are memory-backed cursors for the active M256 epoch.
  // Thread 0 advances them only after the preceding grid-wide completion
  // boundary, so no group-derived pointer remains live across the numerical
  // body.
  std::uintptr_t numerical_residual = 0U;
  std::uintptr_t numerical_input = 0U;
  std::uintptr_t numerical_gate_payload = 0U;
  std::uintptr_t numerical_h = 0U;
  std::uint32_t gate_tensor_scale_bits = 0U;
  std::uint32_t up_tensor_scale_bits = 0U;
  std::uintptr_t numerical_cancellation_signal = 0U;
};

// The whole-P40 planner allocates this opaque control object once per live
// group.  Freeze the exact ABI here so a later launcher cannot silently bind
// a stale allocation size after the numerical continuation fields change.
static_assert(alignof(Sm87BulkV2NvFp4DeviceControl) == 64U);
static_assert(sizeof(Sm87BulkV2NvFp4DeviceControl) == 1'152U);

enum class Sm87BulkV2NvFp4JointPhase : std::uint8_t {
  kInvalid = 0U,
  kAdmitted,
  kMacroEnqueued,
  kGroupActive,
  kGroupComplete,
  kMacroComplete,
  kCancelled,
  kFailed,
};

enum class Sm87BulkV2NvFp4JointEvent : std::uint8_t {
  kInvalid = 0U,
  kEnqueueMacro,
  kStartGroup,
  kCompleteGroup,
  kCompleteMacro,
  kObserveCancellation,
  kFail,
};

struct Sm87BulkV2NvFp4JointReceipt final {
  std::uint64_t transaction_epoch = 0U;
  std::uint32_t layer = 0U;
  std::uint32_t segment = 0U;
  std::uint32_t active_group = 0U;
  std::uint32_t completed_groups = 0U;
  std::uint32_t observed_gate_tasks = 0U;
  std::uint32_t observed_down_tasks = 0U;
  Sm87BulkV2NvFp4JointPhase phase =
      Sm87BulkV2NvFp4JointPhase::kInvalid;
  Sm87BulkV2NvFp4ExecutionIdentity execution_identity =
      Sm87BulkV2NvFp4ExecutionIdentity::kInvalid;
  bool gate_up_and_down_joint = false;
  bool cancellation_word_observed = false;
  bool unpublished_state_discarded = false;
  bool exact_control_stepping_stone = false;
  bool cross_group_weight_residency_qualified = false;
  bool p40_hot_path_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr Sm87BulkV2NvFp4JointReceipt
sm87_bulk_v2_nvfp4_make_joint_receipt(
    const std::uint64_t transaction_epoch, const std::size_t layer,
    const std::size_t segment) noexcept {
  if (transaction_epoch == 0U || layer >= kSm87BulkV2NvFp4LayerCount ||
      !sm87_bulk_v2_nvfp4_segment_plan(segment).valid) {
    return {};
  }
  Sm87BulkV2NvFp4JointReceipt result;
  result.transaction_epoch = transaction_epoch;
  result.layer = static_cast<std::uint32_t>(layer);
  result.segment = static_cast<std::uint32_t>(segment);
  result.phase = Sm87BulkV2NvFp4JointPhase::kAdmitted;
  result.execution_identity = Sm87BulkV2NvFp4ExecutionIdentity::
      kExactControlSteppingStoneM256J64N256K64NoResidencyV2;
  result.gate_up_and_down_joint = true;
  result.exact_control_stepping_stone = true;
  return result;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_nvfp4_joint_receipt_valid(
    const Sm87BulkV2NvFp4JointReceipt& receipt) noexcept {
  if (receipt.transaction_epoch == 0U ||
      receipt.layer >= kSm87BulkV2NvFp4LayerCount ||
      !sm87_bulk_v2_nvfp4_segment_plan(receipt.segment).valid ||
      receipt.execution_identity != Sm87BulkV2NvFp4ExecutionIdentity::
          kExactControlSteppingStoneM256J64N256K64NoResidencyV2 ||
      !receipt.gate_up_and_down_joint ||
      !receipt.exact_control_stepping_stone ||
      receipt.cross_group_weight_residency_qualified ||
      receipt.p40_hot_path_qualified ||
      receipt.numerical_contract_qualified ||
      receipt.production_dispatch_eligible ||
      receipt.phase == Sm87BulkV2NvFp4JointPhase::kInvalid) {
    return false;
  }
  if (receipt.phase == Sm87BulkV2NvFp4JointPhase::kCancelled) {
    return receipt.cancellation_word_observed &&
           receipt.unpublished_state_discarded;
  }
  return !receipt.cancellation_word_observed &&
         !receipt.unpublished_state_discarded;
}

[[nodiscard]] constexpr Sm87BulkV2NvFp4JointReceipt
sm87_bulk_v2_nvfp4_advance_joint_receipt(
    Sm87BulkV2NvFp4JointReceipt receipt,
    const Sm87BulkV2NvFp4JointEvent event,
    const std::uint32_t observed_gate_tasks = 0U,
    const std::uint32_t observed_down_tasks = 0U) noexcept {
  if (!sm87_bulk_v2_nvfp4_joint_receipt_valid(receipt)) {
    return {};
  }
  const auto macro = sm87_bulk_v2_nvfp4_segment_plan(receipt.segment);
  if (event == Sm87BulkV2NvFp4JointEvent::kFail &&
      receipt.phase != Sm87BulkV2NvFp4JointPhase::kMacroComplete &&
      receipt.phase != Sm87BulkV2NvFp4JointPhase::kCancelled) {
    receipt.phase = Sm87BulkV2NvFp4JointPhase::kFailed;
    return receipt;
  }
  if (event == Sm87BulkV2NvFp4JointEvent::kObserveCancellation &&
      (receipt.phase == Sm87BulkV2NvFp4JointPhase::kMacroEnqueued ||
       receipt.phase == Sm87BulkV2NvFp4JointPhase::kGroupActive ||
       receipt.phase == Sm87BulkV2NvFp4JointPhase::kGroupComplete)) {
    receipt.phase = Sm87BulkV2NvFp4JointPhase::kCancelled;
    receipt.cancellation_word_observed = true;
    receipt.unpublished_state_discarded = true;
    return receipt;
  }
  if (receipt.phase == Sm87BulkV2NvFp4JointPhase::kAdmitted &&
      event == Sm87BulkV2NvFp4JointEvent::kEnqueueMacro) {
    receipt.phase = Sm87BulkV2NvFp4JointPhase::kMacroEnqueued;
    return receipt;
  }
  if ((receipt.phase == Sm87BulkV2NvFp4JointPhase::kMacroEnqueued ||
       receipt.phase == Sm87BulkV2NvFp4JointPhase::kGroupComplete) &&
      event == Sm87BulkV2NvFp4JointEvent::kStartGroup &&
      receipt.completed_groups < macro.group_count) {
    receipt.active_group = receipt.completed_groups;
    receipt.observed_gate_tasks = 0U;
    receipt.observed_down_tasks = 0U;
    receipt.phase = Sm87BulkV2NvFp4JointPhase::kGroupActive;
    return receipt;
  }
  if (receipt.phase == Sm87BulkV2NvFp4JointPhase::kGroupActive &&
      event == Sm87BulkV2NvFp4JointEvent::kCompleteGroup) {
    const auto group = sm87_bulk_v2_nvfp4_group_plan(
        receipt.segment, receipt.active_group);
    if (!group.valid || observed_gate_tasks != group.gate_tasks ||
        observed_down_tasks != group.down_tasks) {
      return {};
    }
    receipt.observed_gate_tasks = observed_gate_tasks;
    receipt.observed_down_tasks = observed_down_tasks;
    ++receipt.completed_groups;
    receipt.phase = Sm87BulkV2NvFp4JointPhase::kGroupComplete;
    return receipt;
  }
  if (receipt.phase == Sm87BulkV2NvFp4JointPhase::kGroupComplete &&
      event == Sm87BulkV2NvFp4JointEvent::kCompleteMacro &&
      receipt.completed_groups == macro.group_count) {
    receipt.phase = Sm87BulkV2NvFp4JointPhase::kMacroComplete;
    return receipt;
  }
  return {};
}

struct Sm87BulkV2NvFp4MacroArguments final {
  std::uint64_t transaction_epoch = 0U;
  std::uint32_t layer = 0U;
  std::uint32_t segment = 0U;
  const std::uint16_t* normalized_input = nullptr;
  std::uint16_t* residual = nullptr;
  std::uint16_t* group_h_scratch = nullptr;
  Sm87BulkV2NvFp4DeviceControl* device_control = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  Sm87BulkV2NvFp4AuthenticatedAssets assets{};
  void* cuda_stream = nullptr;
};

// Diagnostic whole-layer wrapper for the exact-control stepping stone.  It
// enqueues 39 M1024 cooperative macro kernels plus the M64 tail on one
// caller-owned stream and reuses the M256 H scratch/control allocation.  It
// deliberately has neither a startup seal nor a prevalidated hot enqueue and
// must not be identified as the frozen production P40 executor.
struct Sm87BulkV2NvFp4P40LayerArguments final {
  std::uint64_t transaction_epoch = 0U;
  std::uint32_t layer = 0U;
  const std::uint16_t* normalized_input = nullptr;
  std::uint16_t* residual = nullptr;
  std::uint16_t* group_h_scratch = nullptr;
  Sm87BulkV2NvFp4DeviceControl* device_control = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  Sm87BulkV2NvFp4AuthenticatedAssets assets{};
  void* cuda_stream = nullptr;
};

struct Sm87BulkV2NvFp4CodeEvidence final {
  std::uint64_t elf_identity = 0U;
  std::uint64_t canonical_sass_hash = 0U;
  std::uint32_t instruction_rows = 0U;
  std::uint32_t text_bytes = 0U;
  std::uint32_t k_dependent_unrolled_body_copies = 0U;
  std::uint32_t local_load_store_rows = 0U;
  bool launch_bounds_256_2 = false;
  bool cooperative_grid_sync_present = false;
  bool atomic_cas_work_claim_present = false;
  bool cp_async_ca_activation_present = false;
  bool cp_async_cg_payload_present = false;
};

inline constexpr std::uint32_t kSm87BulkV2NvFp4MaximumSassRows = 6'736U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4MaximumTextBytes =
    128U * 1'024U;

[[nodiscard]] constexpr bool sm87_bulk_v2_nvfp4_code_evidence_valid(
    const Sm87BulkV2NvFp4CodeEvidence& evidence) noexcept {
  return evidence.elf_identity != 0U &&
         evidence.canonical_sass_hash != 0U &&
         evidence.instruction_rows > 0U &&
         evidence.instruction_rows <=
             kSm87BulkV2NvFp4MaximumSassRows &&
         evidence.text_bytes > 0U &&
         evidence.text_bytes <= kSm87BulkV2NvFp4MaximumTextBytes &&
         evidence.k_dependent_unrolled_body_copies <= 1U &&
         evidence.local_load_store_rows == 0U &&
         evidence.launch_bounds_256_2 &&
         evidence.cooperative_grid_sync_present &&
         evidence.atomic_cas_work_claim_present &&
         evidence.cp_async_ca_activation_present &&
         evidence.cp_async_cg_payload_present;
}

struct Sm87BulkV2NvFp4KernelResources final {
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int cooperative_grid_capacity = 0;
  Sm87BulkV2NvFp4CodeEvidence code{};
  bool kernel_compiled = false;
  bool cooperative_launch_supported = false;
  bool resource_and_code_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_nvfp4_resources_valid(
    const Sm87BulkV2NvFp4KernelResources& resources) noexcept {
  return resources.binary_version == 87 &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             static_cast<int>(kSm87BulkV2NvFp4MaximumRegisters) &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87BulkV2NvFp4DynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<int>(kSm87BulkV2NvFp4Threads) &&
         resources.active_blocks_per_sm >=
             static_cast<int>(kSm87BulkV2NvFp4RequiredCtasPerSm) &&
         resources.cooperative_grid_capacity >=
             static_cast<int>(kSm87BulkV2NvFp4PersistentCtas) &&
         sm87_bulk_v2_nvfp4_code_evidence_valid(resources.code) &&
         resources.kernel_compiled &&
         resources.cooperative_launch_supported &&
         resources.resource_and_code_gate_passed &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

// Read-only query for the compile-only skeleton. The caller supplies retained
// offline SASS/code-size evidence for the exact ELF; CUDA has no runtime API
// that reports instruction rows or .text byte size. This query never launches
// the kernel and never upgrades numerical or production qualification.
[[nodiscard]] int query_sm87_bulk_dataflow_v2_nvfp4_resources_cuda(
    const Sm87BulkV2NvFp4CodeEvidence* code_evidence,
    Sm87BulkV2NvFp4KernelResources* resources) noexcept;

// Permanently fail-closed compile-only seam. A future numerical body and a
// separately reviewed admission launcher must use a new symbol.
[[nodiscard]] int launch_sm87_bulk_dataflow_v2_nvfp4_compile_only_cuda(
    const Sm87BulkV2NvFp4MacroArguments& arguments) noexcept;

// Separate executable-candidate inventory for the exact M64 tail.  The CUDA
// test target carries the same-ELF v1 differential oracle; this read-only
// resource query cannot itself retain that test result or open production.
struct Sm87BulkV2NvFp4TailNumericalResources final {
  Sm87BulkV2NvFp4KernelResources kernel{};
  bool exact_m64_tail_geometry = false;
  bool authenticated_raw_payload_path = false;
  bool ascending_full_k_without_split_k = false;
  bool numerical_body_compiled = false;
  bool exact_control_stepping_stone = false;
  bool cross_group_weight_residency_qualified = false;
  bool p40_hot_path_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool
sm87_bulk_v2_nvfp4_tail_numerical_resources_valid(
    const Sm87BulkV2NvFp4TailNumericalResources& resources) noexcept {
  return sm87_bulk_v2_nvfp4_resources_valid(resources.kernel) &&
         resources.exact_m64_tail_geometry &&
         resources.authenticated_raw_payload_path &&
         resources.ascending_full_k_without_split_k &&
         resources.numerical_body_compiled &&
         resources.exact_control_stepping_stone &&
         !resources.cross_group_weight_residency_qualified &&
         !resources.p40_hot_path_qualified &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

[[nodiscard]] int
query_sm87_bulk_dataflow_v2_nvfp4_tail_numerical_resources_cuda(
    const Sm87BulkV2NvFp4CodeEvidence* code_evidence,
    Sm87BulkV2NvFp4TailNumericalResources* resources) noexcept;

// Admission-only executable candidate.  It accepts only the frozen P40 M64
// tail segment and authenticated target-AOT bytes.  It remains disconnected
// from the runner and does not grant production qualification.
[[nodiscard]] int
launch_sm87_bulk_dataflow_v2_nvfp4_tail_numerical_cuda(
    const Sm87BulkV2NvFp4MacroArguments& arguments) noexcept;

// Same exact-control stepping stone for either an M1024 macro segment or the
// M64 tail.  M1024 is one cooperative launch with four internal M256 epochs,
// but the current epochs reread projection payload and therefore provide no
// cross-group weight-residency or performance qualification.
[[nodiscard]] int
launch_sm87_bulk_dataflow_v2_nvfp4_macro_numerical_cuda(
    const Sm87BulkV2NvFp4MacroArguments& arguments) noexcept;

// Diagnostic 40-launch layer wrapper.  This is not a sealed/prevalidated P40
// hot path and is permanently ineligible for runner dispatch.
[[nodiscard]] int launch_sm87_bulk_dataflow_v2_nvfp4_p40_layer_cuda(
    const Sm87BulkV2NvFp4P40LayerArguments& arguments) noexcept;

inline constexpr std::uint64_t kSm87BulkV2NvFp4GateTaskWork =
    2ULL * kSm87BulkV2NvFp4TileM * kSm87BulkV2NvFp4GateTileN *
    kSm87BulkV2NvFp4Hidden;
inline constexpr std::uint64_t kSm87BulkV2NvFp4DownTaskWork =
    static_cast<std::uint64_t>(kSm87BulkV2NvFp4TileM) *
    kSm87BulkV2NvFp4DownTileN * kSm87BulkV2NvFp4Intermediate;
inline constexpr std::uint64_t kSm87BulkV2NvFp4FullGroupGateWork =
    static_cast<std::uint64_t>(kSm87BulkV2NvFp4RowsPerGroup) *
    kSm87BulkV2NvFp4GateNTiles * kSm87BulkV2NvFp4GateTaskWork;
inline constexpr std::uint64_t kSm87BulkV2NvFp4FullGroupDownWork =
    static_cast<std::uint64_t>(kSm87BulkV2NvFp4RowsPerGroup) *
    kSm87BulkV2NvFp4DownNTiles * kSm87BulkV2NvFp4DownTaskWork;
inline constexpr std::uint32_t kSm87BulkV2NvFp4StealFractionNumerator = 7U;
inline constexpr std::uint32_t kSm87BulkV2NvFp4StealFractionDenominator =
    15U;

inline constexpr auto kSm87BulkV2NvFp4FrozenManifest =
    sm87_bulk_v2_nvfp4_family_manifest();

static_assert(kSm87BulkV2NvFp4P40Tokens ==
              kSm87BulkV2NvFp4FullMacroSegments *
                      kSm87BulkV2NvFp4MacroTokens +
                  kSm87BulkV2NvFp4TailTokens);
static_assert(kSm87BulkV2NvFp4GateNTiles == 272U &&
              kSm87BulkV2NvFp4DownNTiles == 20U);
static_assert(kSm87BulkV2NvFp4GateKTiles == 80U &&
              kSm87BulkV2NvFp4DownKTiles == 272U);
static_assert(kSm87BulkV2NvFp4GateBytesPerStage == 12'800U);
static_assert(kSm87BulkV2NvFp4DownBytesPerStage == 17'408U);
static_assert(kSm87BulkV2NvFp4GateDynamicSharedBytes == 38'400U);
static_assert(kSm87BulkV2NvFp4DynamicSharedBytes == 52'224U);
static_assert(kSm87BulkV2NvFp4GroupScratchBytes == 8'912'896ULL);
static_assert(kSm87BulkV2NvFp4HotReadyWindowBytes == 1'048'576ULL);
static_assert(kSm87BulkV2NvFp4AggregateL2BudgetBytes == 3'670'016ULL);
static_assert(kSm87BulkV2NvFp4FamilyPayloadBytes == 9'625'927'680ULL);
static_assert(kSm87BulkV2NvFp4PersistentCtas ==
              kSm87BulkV2NvFp4SmCount *
                  kSm87BulkV2NvFp4RequiredCtasPerSm);
static_assert(kSm87BulkV2NvFp4DownOwnerCtas +
                  kSm87BulkV2NvFp4DedicatedProducerCtas ==
              kSm87BulkV2NvFp4PersistentCtas);
static_assert(kSm87BulkV2NvFp4FullGroupGateWork ==
              2U * kSm87BulkV2NvFp4FullGroupDownWork);
static_assert(sm87_bulk_v2_nvfp4_manifest_valid(
    kSm87BulkV2NvFp4FrozenManifest));

}  // namespace q3x::kernels
