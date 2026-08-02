#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off resource experiment over the authenticated v1 K512 payload.
// Two cooperative M32N128 CTAs share an M64 work cell by residing on the same
// SM.  Four warps compute Gate and four compute Up; every warp owns M32N32.
// Packed B is consumed directly with cache-at-all-levels loads while A alone
// occupies a three-slot cp.async.cg M32K256 ring.
//
// The key structural change from the rejected register-accumulating pair is
// that only one K512 S32 delta remains in registers.  Completed FP32 groups
// live in component-major shared planes.  On the final group, Gate updates
// its plane while Up reuses each partial's register slot as its final Float4,
// then consumes final Gate after one seam barrier.  Shared storage is exact:
//
//   3*M32*K256/2 + 2*M32*N128*sizeof(float)
//                    + M32*K512*sizeof(bf16)
//   = 12,288 + 32,768 + 32,768 = 77,824 bytes.
//
// Merely linking the object cannot select it.  This first slice exposes a
// resource query only; runtime, conversion, correctness, and performance
// surfaces remain intentionally absent until the hard resource gate passes.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumPairTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileN = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumEdgeK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumPhysicalK64 = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumK64PerCopy = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumStages = 3U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumProjectionWarps = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumComponents = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCellsPerEdge = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumSmCount = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumPersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCtasPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumWorkspaceAlignment = 16U;

inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumAStageBytes =
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileM *
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumARingBytes =
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumStages *
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumAStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumPlanesBytes =
        2U * kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileM *
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileN *
        sizeof(float);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumEdgePlaneBytes =
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileM *
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumEdgeK *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumARingBytes +
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumPlanesBytes +
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumEdgePlaneBytes;

// Request-local pinned-Orin SM tickets plus one error word, rounded to the
// declared alignment.  Admission beyond the resource gate additionally
// requires a one-time probe proving that the physical SM-ID set is 0..15.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumWorkspaceBytes = 80U;

struct Sm87A4W4GateUpDownEdgeM32N128PairSharedAccumResources final {
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

// Success means <=128 registers/thread, zero local storage, the exact
// 77,824-byte dynamic allocation, and exactly two resident 256-thread CTAs
// per SM on the pinned cooperative-launch SM87 target.
[[nodiscard]] int
query_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_shared_accum_resources_cuda(
    Sm87A4W4GateUpDownEdgeM32N128PairSharedAccumResources* resources) noexcept;

static_assert(
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumAStageBytes == 4'096U);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumARingBytes == 12'288U);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumPlanesBytes == 32'768U);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumEdgePlaneBytes == 32'768U);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumDynamicSharedBytes ==
    77'824U);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumProjectionWarps * 2U ==
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumWarps);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumM16PerWarp * 16U ==
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileM);
static_assert(
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumN8PerWarp * 8U *
        kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumProjectionWarps ==
    kSm87A4W4GateUpDownEdgeM32N128PairSharedAccumTileN);

}  // namespace q3x::kernels
