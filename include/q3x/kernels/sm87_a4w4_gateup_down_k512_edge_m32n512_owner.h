#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>

namespace q3x::kernels {

// Resource-first structural mirror for a producer-owned M32N512 Gate+Up
// cell.  One 256-thread CTA visits eight M32N64 subcells.  Warp w owns N8
// and temporally covers two M16 panels, retaining matching Gate and Up
// accumulators until every exact K512 group has been applied.  The resulting
// BF16 products remain in one CTA-owned M32N512 edge plane and are published
// directly through the canonical Down-input A4/K512 ABI.
//
// Four K128 cp.async stages keep the operand pipeline at the same 40,960-byte
// code footprint as a two-stage K256 ring while exposing twice the async
// depth.  The two K64 planes in all four stages still accumulate into one S32
// partial before the exact K512 scale/FMA boundary.
//
// This surface is deliberately resource-only.  It has no runtime selector or
// public launch API, and merely linking the object cannot alter production.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerTileN = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerCellsPerEdge = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerCopyK = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerK64PerStage = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerStages = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlots = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerWarpTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerWarpTileN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerM16PanelsPerWarp = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerPersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerCtasPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerPackedK64Bytes = 32U;

inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerAStageBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerTileM *
        kSm87A4W4GateUpDownEdgeM32N512OwnerCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerBStageBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN *
        kSm87A4W4GateUpDownEdgeM32N512OwnerCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerStageBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerAStageBytes +
        2U * kSm87A4W4GateUpDownEdgeM32N512OwnerBStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlotBytes =
        (kSm87A4W4GateUpDownEdgeM32N512OwnerTileM +
         2U * kSm87A4W4GateUpDownEdgeM32N512OwnerComputeTileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerPipelineBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerStages *
            kSm87A4W4GateUpDownEdgeM32N512OwnerStageBytes +
        kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlots *
            kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlotBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerEdgePlaneBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerTileM *
        kSm87A4W4GateUpDownEdgeM32N512OwnerTileN *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeM32N512OwnerPipelineBytes +
        kSm87A4W4GateUpDownEdgeM32N512OwnerEdgePlaneBytes;

struct Sm87A4W4GateUpDownEdgeM32N512OwnerResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t configured_dynamic_shared_limit_bytes{};
  std::size_t device_optin_shared_limit_bytes{};
  std::size_t device_shared_per_sm_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int
query_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_resources_cuda(
    Sm87A4W4GateUpDownEdgeM32N512OwnerResources* resources) noexcept;

static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerAStageBytes == 2'048U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerBStageBytes == 4'096U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerStageBytes == 10'240U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerScaleSlotBytes == 320U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerPipelineBytes == 41'600U);
static_assert(kSm87A4W4GateUpDownEdgeM32N512OwnerEdgePlaneBytes == 32'768U);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes == 74'368U);
static_assert(
    2U * kSm87A4W4GateUpDownEdgeM32N512OwnerDynamicSharedBytes ==
    kSm87A4W4GateUpDownEdgeDynamicSharedBytes);

}  // namespace q3x::kernels
