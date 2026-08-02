#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"

#include <cstddef>

namespace q3x::kernels {

// Default-off resource experiment that retains the incumbent paired M16N32
// warp ownership, authenticated v1 K512 payload/scale ABI, edge layout, and
// quantizer.  Its sole structural variable is an alternating whole-K256-stage
// schedule: one publication/reader-release barrier per K256 stage instead of
// the incumbent four-barrier K512 loop.
//
// One 512-thread CTA owns M64N128.  Each of its sixteen warps owns M16N32 and
// accumulates matching Gate and Up fragments in registers, so SwiGLU requires
// no cross-warp Gate exchange.  Dynamic shared storage is byte-identical to
// the retained edge: two 40,960-byte A+GateB+UpB K256 stages, two 640-byte
// scale slots, and one 65,536-byte M64K512 BF16 product plane.
//
// This first slice deliberately exposes no launcher.  Linking it cannot alter
// runtime selection; only the resource query below is public.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256AlternatingThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256AlternatingCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256AlternatingMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256AlternatingPreferredRegisters = 125U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeDynamicSharedBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256AlternatingStageBarriersPerCell =
        21U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM64N128K256AlternatingBarriersPerEdge =
        4U *
            kSm87A4W4GateUpDownEdgeM64N128K256AlternatingStageBarriersPerCell +
        1U;

struct Sm87A4W4GateUpDownEdgeM64N128K256AlternatingResources final {
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
query_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_resources_cuda(
    Sm87A4W4GateUpDownEdgeM64N128K256AlternatingResources* resources)
    noexcept;

static_assert(
    kSm87A4W4GateUpDownEdgeM64N128K256AlternatingDynamicSharedBytes ==
    148'736U);
static_assert(
    kSm87A4W4GateUpDownEdgeM64N128K256AlternatingBarriersPerEdge == 85U);

}  // namespace q3x::kernels
