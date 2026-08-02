#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Resource-only structural experiment over the authenticated v1 K512 ABI.
// One 512-thread CTA owns M64N128.  Warps 0..7 compute Gate and warps 8..15
// compute Up; matching crews own M64N16 (four M16 panels by two N8 fragments).
// Unlike the rejected direct/fragment-native M64N128 route, both B operands
// are published as complete K256 stages alongside A before any consumer MMA.
//
// Two K256 stages each contain 8 KiB A, 16 KiB Gate B, and 16 KiB Up B.
// Two 640-byte K512 scale slots bring the live pipeline to 83,200 bytes.  The
// dead pipeline allocation is then reused as a 32,768-byte FP32 Gate exchange;
// a concurrently live M64K512 BF16 edge plane occupies 65,536 bytes.  Thus the
// exact dynamic allocation remains the incumbent 148,736-byte contract.
//
// Merely linking this object cannot select it.  This first slice exposes only
// a resource query; it has no runtime, converter, correctness, or timing API.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedEdgeK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedCopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedPhysicalK64 = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedK64PerCopy = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedScaleSlots = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedProjectionWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedM16PerWarp = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedN8PerWarp = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedComponents = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedCellsPerEdge = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedPersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedMaximumRegisters = 128U;

inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedAStageBytes =
        kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM *
        kSm87A4W4GateUpDownEdgeM64N128K256StagedCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedBStageBytes =
        kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN *
        kSm87A4W4GateUpDownEdgeM64N128K256StagedCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedStageBytes =
        kSm87A4W4GateUpDownEdgeM64N128K256StagedAStageBytes +
        2U * kSm87A4W4GateUpDownEdgeM64N128K256StagedBStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedScaleSlotBytes =
        (kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM +
         2U * kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedPipelineBytes =
        kSm87A4W4GateUpDownEdgeM64N128K256StagedStages *
            kSm87A4W4GateUpDownEdgeM64N128K256StagedStageBytes +
        kSm87A4W4GateUpDownEdgeM64N128K256StagedScaleSlots *
            kSm87A4W4GateUpDownEdgeM64N128K256StagedScaleSlotBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedGateExchangeBytes =
        kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM *
        kSm87A4W4GateUpDownEdgeM64N128K256StagedTileN * sizeof(float);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedEdgePlaneBytes =
        kSm87A4W4GateUpDownEdgeM64N128K256StagedTileM *
        kSm87A4W4GateUpDownEdgeM64N128K256StagedEdgeK *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeM64N128K256StagedPipelineBytes +
        kSm87A4W4GateUpDownEdgeM64N128K256StagedEdgePlaneBytes;

struct Sm87A4W4GateUpDownEdgeM64N128K256StagedResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t configured_dynamic_shared_limit_bytes{};
  std::size_t device_optin_shared_limit_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int
query_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_staged_resources_cuda(
    Sm87A4W4GateUpDownEdgeM64N128K256StagedResources* resources) noexcept;

static_assert(kSm87A4W4GateUpDownEdgeM64N128K256StagedAStageBytes ==
              8'192U);
static_assert(kSm87A4W4GateUpDownEdgeM64N128K256StagedBStageBytes ==
              16'384U);
static_assert(kSm87A4W4GateUpDownEdgeM64N128K256StagedStageBytes ==
              40'960U);
static_assert(kSm87A4W4GateUpDownEdgeM64N128K256StagedScaleSlotBytes ==
              640U);
static_assert(kSm87A4W4GateUpDownEdgeM64N128K256StagedPipelineBytes ==
              83'200U);
static_assert(kSm87A4W4GateUpDownEdgeM64N128K256StagedGateExchangeBytes ==
              32'768U);
static_assert(kSm87A4W4GateUpDownEdgeM64N128K256StagedEdgePlaneBytes ==
              65'536U);
static_assert(kSm87A4W4GateUpDownEdgeM64N128K256StagedDynamicSharedBytes ==
              148'736U);

}  // namespace q3x::kernels
