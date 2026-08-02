#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix.h"
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// P0 structural combination only.  One 512-thread CTA owns M128N512 and
// sweeps eight N64 cells.  Warps form [m_half=0..1][n8=0..7], so each warp
// owns M64N8 and four M16 panels.  A and paired Gate/Up B use two K256 shared
// stages; A reaches IMMA through ldmatrix.x4 while one paired-B LDS.128 feed
// is reused by both projections and four panels.  The alternating whole-K256
// handoff is 21 CTA barriers per N64 at model K5120, not the rejected staged
// kernel's 41.
//
// This probe deliberately retains the wide edge's honest traffic boundary:
// A is reloaded for every N64 cell; paired B is loaded once per M128N64 cell.
// Products publish the canonical Down-v1 K512 ABI through one shared M64N512
// half and one caller-owned 64-KiB-per-CTA scratch half.
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixTileN = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixComputeTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixCellsPerEdge = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixMHalves = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixN8Warps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPanelsPerWarp = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixK64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixAStageBytes = 16'384U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixBStageBytes = 16'384U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixStageBytes = 32'768U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixScaleSlotBytes = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPipelineBytes =
        kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixStages *
        (kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixStageBytes +
         kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixScaleSlotBytes);
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixEdgePlaneBytes = 65'536U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixDynamicSharedBytes =
        kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPipelineBytes +
        kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixEdgePlaneBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixScratchBytesPerCta =
        kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixEdgePlaneBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixActiveWarpsPerSm = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixBarriersPerN64 = 21U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPublicationBarriersPerEdge =
        3U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixBarriersPerEdge =
        kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixCellsPerEdge *
            kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixBarriersPerN64 +
        kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPublicationBarriersPerEdge;

using Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPlan =
    Sm87A4W4GateUpDownEdgeM128N512LdmatrixPlan;

[[nodiscard]] constexpr
Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPlan
sm87_a4w4_gateup_down_edge_m128n512_paired_ldmatrix_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t maximum_launch_ctas = 16U) noexcept {
  return sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_test_plan(
      logical_token_count, launch_token_count, intermediate_size,
      input_size, maximum_launch_ctas);
}

[[nodiscard]] constexpr
Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPlan
sm87_a4w4_gateup_down_edge_m128n512_paired_ldmatrix_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return sm87_a4w4_gateup_down_edge_m128n512_ldmatrix_plan(
      logical_token_count, launch_token_count, intermediate_size,
      input_size);
}

struct Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixResources final {
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
query_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_resources_cuda(
    Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixResources* resources)
    noexcept;

#define Q3X_SM87_A4W4_GATEUP_DOWN_M128N512_PAIRED_LDMATRIX_ARGUMENTS      \
    const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,    \
    const std::uint16_t* a_k512_scales_bf16,                             \
    std::size_t a_scale_capacity_elements,                               \
    const std::uint8_t* paired_b_codes,                                  \
    std::size_t paired_b_code_capacity_bytes,                            \
    const std::uint16_t* paired_b_scales_bf16,                           \
    std::size_t paired_b_scale_capacity_elements,                        \
    std::size_t logical_token_count, std::size_t launch_token_count,     \
    std::size_t intermediate_size, std::size_t input_size,               \
    float output_clip_ratio, std::uint8_t* cta_scratch,                  \
    std::size_t cta_scratch_capacity_bytes, std::uint8_t* packed_output, \
    std::size_t packed_output_capacity_bytes,                            \
    std::uint16_t* output_k512_scales_bf16,                              \
    std::size_t output_scale_capacity_elements

[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_cuda(
    Q3X_SM87_A4W4_GATEUP_DOWN_M128N512_PAIRED_LDMATRIX_ARGUMENTS,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_test_cuda(
    Q3X_SM87_A4W4_GATEUP_DOWN_M128N512_PAIRED_LDMATRIX_ARGUMENTS,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

#undef Q3X_SM87_A4W4_GATEUP_DOWN_M128N512_PAIRED_LDMATRIX_ARGUMENTS

static_assert(
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixPipelineBytes == 66'560U);
static_assert(
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixDynamicSharedBytes ==
    132'096U);
static_assert(
    kSm87A4W4GateUpDownEdgeM128N512PairedLdmatrixBarriersPerEdge ==
    171U);

}  // namespace q3x::kernels
