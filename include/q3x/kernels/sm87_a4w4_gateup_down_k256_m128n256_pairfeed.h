#pragma once

#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off Gate+Up -> Down-input edge over the authenticated base K256
// sidecar ABI.  One cooperative 16-CTA grid stays phase aligned on one N256
// weight panel while its CTAs sweep complete M128 waves.  A CTA computes four
// temporal M128N64 cells.  Its sixteen warps are [M16=0..7][N32=0..1], and
// every warp owns matching Gate and Up fragments through the exact BF16
// SwiGLU seam before publishing one K256 signed-A4 output group.
//
// Codes retain the canonical physical layout
//
//   [outer / 64][K / 64][64][32]
//
// while every operand scale uses the independent K256 layout
//
//   [outer / 64][K / 256][64].
//
// The two K256 stages contain independent A, Gate-B, and Up-B planes.  Scale
// application happens after exactly four ordered K64 IMMA contributions;
// adjacent K256 groups are never combined in S32.  The complete shared-memory
// contract is fixed at 132,096 bytes:
//
//   2 * (M128*K256/2 + 2*N64*K256/2) = 65,536 code bytes
//   2 * (M128 + 2*N64) * sizeof(bf16) =  1,024 scale bytes
//   M128*K256*sizeof(bf16)            = 65,536 edge bytes
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedEdgeK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedComputeTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedCellsPerEdge = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedCopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedPhysicalK64 = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedK64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedPackedK64Bytes = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedWarpRows = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedWarpColumns = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedWarpTileM = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedWarpTileN = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedScaleSlots = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedPersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedModelIntermediate = 17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedMaximumLaunchTokens = 4'096U;

inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedAStageBytes =
        kSm87A4W4GateUpDownK256M128N256PairfeedTileM *
        kSm87A4W4GateUpDownK256M128N256PairfeedCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedBStageBytes =
        kSm87A4W4GateUpDownK256M128N256PairfeedComputeTileN *
        kSm87A4W4GateUpDownK256M128N256PairfeedCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedStageBytes =
        kSm87A4W4GateUpDownK256M128N256PairfeedAStageBytes +
        2U * kSm87A4W4GateUpDownK256M128N256PairfeedBStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedScaleSlotBytes =
        (kSm87A4W4GateUpDownK256M128N256PairfeedTileM +
         2U * kSm87A4W4GateUpDownK256M128N256PairfeedComputeTileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedPipelineBytes =
        kSm87A4W4GateUpDownK256M128N256PairfeedStages *
            kSm87A4W4GateUpDownK256M128N256PairfeedStageBytes +
        kSm87A4W4GateUpDownK256M128N256PairfeedScaleSlots *
            kSm87A4W4GateUpDownK256M128N256PairfeedScaleSlotBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedEdgePlaneBytes =
        kSm87A4W4GateUpDownK256M128N256PairfeedTileM *
        kSm87A4W4GateUpDownK256M128N256PairfeedEdgeK *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4GateUpDownK256M128N256PairfeedDynamicSharedBytes =
        kSm87A4W4GateUpDownK256M128N256PairfeedPipelineBytes +
        kSm87A4W4GateUpDownK256M128N256PairfeedEdgePlaneBytes;

struct Sm87A4W4GateUpDownK256M128N256PairfeedPlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t m_waves{};
  std::size_t edge_groups{};
  std::size_t input_k256_groups{};
  std::size_t input_physical_k64_groups{};
  std::size_t output_physical_k64_groups{};
  std::size_t work_edge_cells{};
  std::size_t launch_ctas{};
};

// The correctness surface intentionally retains the production cooperative
// topology.  `maximum_launch_ctas` must be exactly sixteen; a smaller grid
// would no longer prove the N-major phase-lock contract.
[[nodiscard]] constexpr Sm87A4W4GateUpDownK256M128N256PairfeedPlan
sm87_a4w4_gateup_down_k256_m128n256_pairfeed_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t maximum_launch_ctas =
        kSm87A4W4GateUpDownK256M128N256PairfeedPersistentCtas) noexcept {
  if (logical_token_count == 0U ||
      launch_token_count !=
          sm87_a4w4_attention_k256_launch_token_count(logical_token_count) ||
      launch_token_count %
              kSm87A4W4GateUpDownK256M128N256PairfeedTileM !=
          0U ||
      intermediate_size == 0U ||
      intermediate_size %
              kSm87A4W4GateUpDownK256M128N256PairfeedEdgeK !=
          0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpDownK256M128N256PairfeedCopyK != 0U ||
      maximum_launch_ctas !=
          kSm87A4W4GateUpDownK256M128N256PairfeedPersistentCtas) {
    return {};
  }
  const std::size_t m_tiles =
      launch_token_count /
      kSm87A4W4GateUpDownK256M128N256PairfeedTileM;
  const std::size_t edge_groups =
      intermediate_size /
      kSm87A4W4GateUpDownK256M128N256PairfeedEdgeK;
  if (!sm87_a4w4_attention_k256_product_fits(m_tiles, edge_groups)) {
    return {};
  }
  const std::size_t m_waves =
      1U + (m_tiles - 1U) /
               kSm87A4W4GateUpDownK256M128N256PairfeedPersistentCtas;
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          m_tiles,
          m_waves,
          edge_groups,
          input_size /
              kSm87A4W4GateUpDownK256M128N256PairfeedCopyK,
          input_size /
              kSm87A4W4GateUpDownK256M128N256PairfeedPhysicalK64,
          intermediate_size /
              kSm87A4W4GateUpDownK256M128N256PairfeedPhysicalK64,
          m_tiles * edge_groups,
          kSm87A4W4GateUpDownK256M128N256PairfeedPersistentCtas};
}

[[nodiscard]] constexpr Sm87A4W4GateUpDownK256M128N256PairfeedPlan
sm87_a4w4_gateup_down_k256_m128n256_pairfeed_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return launch_token_count <=
                     kSm87A4W4GateUpDownK256M128N256PairfeedMaximumLaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpDownK256M128N256PairfeedModelIntermediate &&
                 input_size ==
                     kSm87A4W4GateUpDownK256M128N256PairfeedModelInput
             ? sm87_a4w4_gateup_down_k256_m128n256_pairfeed_test_plan(
                   logical_token_count, launch_token_count,
                   intermediate_size, input_size)
             : Sm87A4W4GateUpDownK256M128N256PairfeedPlan{};
}

struct Sm87A4W4GateUpDownK256M128N256PairfeedResources final {
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
query_sm87_a4w4_gateup_down_k256_m128n256_pairfeed_resources_cuda(
    Sm87A4W4GateUpDownK256M128N256PairfeedResources* resources) noexcept;

#define Q3X_SM87_A4W4_GATEUP_DOWN_K256_M128N256_PAIRFEED_ARGUMENTS       \
  const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,      \
  const std::uint16_t* a_k256_scales_bf16,                               \
  std::size_t a_scale_capacity_elements,                                 \
  const std::uint8_t* packed_gate_b,                                     \
  std::size_t packed_gate_b_capacity_bytes,                              \
  const std::uint16_t* gate_b_k256_scales_bf16,                          \
  std::size_t gate_b_scale_capacity_elements,                            \
  const std::uint8_t* packed_up_b,                                       \
  std::size_t packed_up_b_capacity_bytes,                                \
  const std::uint16_t* up_b_k256_scales_bf16,                            \
  std::size_t up_b_scale_capacity_elements,                              \
  std::size_t logical_token_count, std::size_t launch_token_count,       \
  std::size_t intermediate_size, std::size_t input_size,                 \
  float output_clip_ratio, std::uint8_t* packed_output,                  \
  std::size_t packed_output_capacity_bytes,                              \
  std::uint16_t* output_k256_scales_bf16,                                \
  std::size_t output_scale_capacity_elements

// Production model-shape surface.  Resource admission must be primed before
// this call is captured into a CUDA graph; capture otherwise fails closed.
[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k256_m128n256_pairfeed_cuda(
    Q3X_SM87_A4W4_GATEUP_DOWN_K256_M128N256_PAIRFEED_ARGUMENTS,
    void* cuda_stream = nullptr) noexcept;

// Generic-shape correctness surface.  It still launches the exact 16-CTA
// cooperative grid and therefore is not a synthetic timing authority.
[[nodiscard]] int
launch_sm87_a4w4_gateup_down_k256_m128n256_pairfeed_test_cuda(
    Q3X_SM87_A4W4_GATEUP_DOWN_K256_M128N256_PAIRFEED_ARGUMENTS,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

#undef Q3X_SM87_A4W4_GATEUP_DOWN_K256_M128N256_PAIRFEED_ARGUMENTS

static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedAStageBytes == 16'384U);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedBStageBytes == 8'192U);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedStageBytes == 32'768U);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedScaleSlotBytes == 512U);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedPipelineBytes == 66'560U);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedEdgePlaneBytes == 65'536U);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedDynamicSharedBytes ==
    132'096U);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedWarpRows *
            kSm87A4W4GateUpDownK256M128N256PairfeedWarpColumns ==
        kSm87A4W4GateUpDownK256M128N256PairfeedWarps);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedWarpRows *
            kSm87A4W4GateUpDownK256M128N256PairfeedWarpTileM ==
        kSm87A4W4GateUpDownK256M128N256PairfeedTileM);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedWarpColumns *
            kSm87A4W4GateUpDownK256M128N256PairfeedWarpTileN ==
        kSm87A4W4GateUpDownK256M128N256PairfeedComputeTileN);
static_assert(
    kSm87A4W4GateUpDownK256M128N256PairfeedCellsPerEdge *
            kSm87A4W4GateUpDownK256M128N256PairfeedComputeTileN ==
        kSm87A4W4GateUpDownK256M128N256PairfeedEdgeK);
static_assert(sm87_a4w4_attention_k256_packed_capacity_bytes(
                  1'920U, 5'120U) ==
              sm87_a4w4_consumer_packed_capacity_bytes(
                  1'920U, 5'120U));
static_assert(sm87_a4w4_attention_k256_packed_offset(
                  129U, 7U, 13U, 80U) ==
              sm87_a4w4_consumer_packed_offset(
                  129U, 7U, 13U, 80U));
static_assert(sm87_a4w4_gateup_down_k256_m128n256_pairfeed_plan(
                  1'853U, 1'920U, 17'408U, 5'120U)
                  .m_tiles == 15U);
static_assert(sm87_a4w4_gateup_down_k256_m128n256_pairfeed_plan(
                  4'096U, 4'096U, 17'408U, 5'120U)
                  .m_waves == 2U);

}  // namespace q3x::kernels
