#pragma once

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_FACTOR_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_FACTOR_HOST_DEVICE
#endif

namespace q3x::kernels {

// Default-off R1 paired Gate+Up experiment.  It is deliberately isolated
// from every runner/selector.  Consumer-order signed-A4 codes remain
// [outer/64][K/64][64][32], while the lane-count-one BF16 scale is
// [outer/64][1][64].  Gate and Up each retain a complete whole-K S32 result;
// A/Gate/Up row scales are applied exactly once before the single
// BF16-RNE SiLU(Gate)*Up publication.
inline constexpr std::size_t kSm87A4W4GateUpFactorizedLaneCount = 1U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedTileM = 128U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedTileN = 128U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedWarpM = 32U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedWarpN = 32U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedWarpRows = 4U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedWarpColumns = 4U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedThreads = 512U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedWarps = 16U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedCopyK = 256U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedPhysicalK64 = 64U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedK64PerStage = 4U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedStages = 2U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedPersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedCtasPerSm = 1U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedMaximumRegisters = 128U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedMaximumTokens = 4'096U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedModelIntermediate =
    17'408U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedModelInput = 5'120U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedPrimaryWidth = 12'288U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedPrimaryStride = 12'288U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedSecondaryWidth = 5'120U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedSecondaryStride = 6'144U;
// R1 projection-plane v2 keeps the exact Gate/Up tensor byte count while
// publishing one consumer-native 16-byte B pair per (N8,K64,lane): Gate's
// two-u32 fragment followed by Up's two-u32 fragment.  The layout is
// [N64 block][physical K64][N8 fragment][lane][16 bytes].  Scales are the
// equal-element interleave [N64 block][row][Gate,Up].
inline constexpr std::size_t kSm87A4W4GateUpFactorizedV2N8 = 8U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedV2N8PerN64 = 8U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedV2FragmentLanes = 32U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedV2PairBytes = 16U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedV2ScalePlanes = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR1ProductPartialTiles =
        kSm87A4W4GateUpFactorizedModelIntermediate /
        kSm87A4W4GateUpFactorizedTileN;

inline constexpr std::size_t kSm87A4W4GateUpFactorizedAStageBytes =
    kSm87A4W4GateUpFactorizedTileM *
    kSm87A4W4GateUpFactorizedCopyK / 2U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedBStageBytes =
    kSm87A4W4GateUpFactorizedTileN *
    kSm87A4W4GateUpFactorizedCopyK / 2U;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedStageBytes =
    kSm87A4W4GateUpFactorizedAStageBytes +
    2U * kSm87A4W4GateUpFactorizedBStageBytes;
inline constexpr std::size_t kSm87A4W4GateUpFactorizedScaleBytes =
    (kSm87A4W4GateUpFactorizedTileM +
     2U * kSm87A4W4GateUpFactorizedTileN) *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87A4W4GateUpFactorizedDynamicSharedBytes =
    kSm87A4W4GateUpFactorizedStages *
        kSm87A4W4GateUpFactorizedStageBytes +
    kSm87A4W4GateUpFactorizedScaleBytes;
inline constexpr std::int32_t kSm87A4W4GateUpFactorizedMaximumS32 =
    static_cast<std::int32_t>(
        7U * 7U * kSm87A4W4GateUpFactorizedModelInput);

static_assert(kSm87A4W4GateUpFactorizedThreads ==
              32U * kSm87A4W4GateUpFactorizedWarps);
static_assert(kSm87A4W4GateUpFactorizedWarpRows *
                      kSm87A4W4GateUpFactorizedWarpM ==
                  kSm87A4W4GateUpFactorizedTileM);
static_assert(kSm87A4W4GateUpFactorizedWarpColumns *
                      kSm87A4W4GateUpFactorizedWarpN ==
                  kSm87A4W4GateUpFactorizedTileN);
static_assert(kSm87A4W4GateUpFactorizedAStageBytes == 16'384U);
static_assert(kSm87A4W4GateUpFactorizedBStageBytes == 16'384U);
static_assert(kSm87A4W4GateUpFactorizedStageBytes == 49'152U);
static_assert(kSm87A4W4GateUpFactorizedScaleBytes == 768U);
static_assert(kSm87A4W4GateUpFactorizedDynamicSharedBytes == 99'072U);
static_assert(kSm87A4W4GateUpFactorizedMaximumS32 == 250'880);
static_assert(kSm87A4W4GateUpFactorizedPrimaryWidth +
                      kSm87A4W4GateUpFactorizedSecondaryWidth ==
                  kSm87A4W4GateUpFactorizedModelIntermediate);
static_assert(kSm87A4W4GateUpFactorizedR1ProductPartialTiles == 136U);

[[nodiscard]] constexpr bool sm87_a4w4_gateup_factorized_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_outer_blocks(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U
             ? 0U
             : 1U + (outer_count - 1U) /
                        kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_k64_groups(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4GateUpFactorizedPhysicalK64 == 0U
             ? logical_k / kSm87A4W4GateUpFactorizedPhysicalK64
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_k256_stages(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4GateUpFactorizedCopyK == 0U
             ? logical_k / kSm87A4W4GateUpFactorizedCopyK
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_packed_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_gateup_factorized_outer_blocks(outer_count);
  const std::size_t groups =
      sm87_a4w4_gateup_factorized_k64_groups(logical_k);
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_gateup_factorized_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  constexpr std::size_t kBytesPerBlockGroup =
      kSm87A4W4ConsumerOuterBlock *
      kSm87A4W4ConsumerPackedKBlockBytes;
  return sm87_a4w4_gateup_factorized_product_fits(
             block_groups, kBytesPerBlockGroup)
             ? block_groups * kBytesPerBlockGroup
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_FACTOR_HOST_DEVICE constexpr std::size_t
sm87_a4w4_gateup_factorized_packed_offset(
    const std::size_t outer_coordinate, const std::size_t k64_group,
    const std::size_t byte_in_k64,
    const std::size_t k64_group_count) noexcept {
  return sm87_a4w4_consumer_packed_offset(
      outer_coordinate, k64_group, byte_in_k64, k64_group_count);
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_scale_capacity_elements(
    const std::size_t outer_count) noexcept {
  const std::size_t blocks =
      sm87_a4w4_gateup_factorized_outer_blocks(outer_count);
  return blocks == 0U ||
                 !sm87_a4w4_gateup_factorized_product_fits(
                     blocks, kSm87A4W4ConsumerOuterBlock)
             ? 0U
             : blocks * kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_FACTOR_HOST_DEVICE constexpr std::size_t
sm87_a4w4_gateup_factorized_scale_offset(
    const std::size_t outer_coordinate) noexcept {
  return (outer_coordinate / kSm87A4W4ConsumerOuterBlock) *
             kSm87A4W4ConsumerOuterBlock +
         outer_coordinate % kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_v2_paired_code_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  return outer_count == 0U ||
                 outer_count % kSm87A4W4ConsumerOuterBlock != 0U ||
                 sm87_a4w4_gateup_factorized_k64_groups(logical_k) == 0U ||
                 !sm87_a4w4_gateup_factorized_product_fits(outer_count,
                                                           logical_k)
             ? 0U
             : outer_count * logical_k;
}

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_FACTOR_HOST_DEVICE constexpr std::size_t
sm87_a4w4_gateup_factorized_v2_paired_code_offset(
    const std::size_t outer_coordinate,
    const std::size_t physical_k64_group,
    const std::size_t lane,
    const std::size_t physical_k64_group_count) noexcept {
  const std::size_t outer_block =
      outer_coordinate / kSm87A4W4ConsumerOuterBlock;
  const std::size_t n8_fragment =
      (outer_coordinate % kSm87A4W4ConsumerOuterBlock) /
      kSm87A4W4GateUpFactorizedV2N8;
  return ((((outer_block * physical_k64_group_count +
             physical_k64_group) *
                kSm87A4W4GateUpFactorizedV2N8PerN64 +
            n8_fragment) *
               kSm87A4W4GateUpFactorizedV2FragmentLanes +
           lane) *
          kSm87A4W4GateUpFactorizedV2PairBytes);
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_v2_paired_scale_capacity_elements(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U ||
                 outer_count % kSm87A4W4ConsumerOuterBlock != 0U ||
                 !sm87_a4w4_gateup_factorized_product_fits(
                     outer_count,
                     kSm87A4W4GateUpFactorizedV2ScalePlanes)
             ? 0U
             : outer_count * kSm87A4W4GateUpFactorizedV2ScalePlanes;
}

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_FACTOR_HOST_DEVICE constexpr std::size_t
sm87_a4w4_gateup_factorized_v2_paired_scale_offset(
    const std::size_t outer_coordinate,
    const std::size_t projection) noexcept {
  return ((outer_coordinate / kSm87A4W4ConsumerOuterBlock) *
              kSm87A4W4ConsumerOuterBlock +
          outer_coordinate % kSm87A4W4ConsumerOuterBlock) *
             kSm87A4W4GateUpFactorizedV2ScalePlanes +
         projection;
}

// Candidate-only GateUp -> Down R1 handoff.  One FP32 maximum is published
// for every padded launch row and N128 GateUp output tile.  The row-major
// layout is [launch_token_count][136]; padded rows are explicitly zero.
[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_r1_product_partial_capacity_elements(
    const std::size_t launch_token_count) noexcept {
  return launch_token_count == 0U ||
                 launch_token_count >
                     std::numeric_limits<std::size_t>::max() /
                         kSm87A4W4GateUpFactorizedR1ProductPartialTiles
             ? 0U
             : launch_token_count *
                   kSm87A4W4GateUpFactorizedR1ProductPartialTiles;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_launch_token_count(
    const std::size_t logical_token_count) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count > kSm87A4W4GateUpFactorizedMaximumTokens ||
      logical_token_count >
          std::numeric_limits<std::size_t>::max() -
              (kSm87A4W4GateUpFactorizedTileM - 1U)) {
    return 0U;
  }
  return ((logical_token_count + kSm87A4W4GateUpFactorizedTileM - 1U) /
          kSm87A4W4GateUpFactorizedTileM) *
         kSm87A4W4GateUpFactorizedTileM;
}

struct Sm87A4W4GateUpFactorizedLanePlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t primary_width{};
  std::size_t secondary_width{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k256_stages{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

// Flexible N/K and plane widths are test-only.  Both planes must own at
// least one complete N128 tile so cross-plane routing is always exercised.
[[nodiscard]] constexpr Sm87A4W4GateUpFactorizedLanePlan
sm87_a4w4_gateup_factorized_lane_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t primary_width) noexcept {
  if (launch_token_count !=
          sm87_a4w4_gateup_factorized_launch_token_count(
              logical_token_count) ||
      intermediate_size < 2U * kSm87A4W4GateUpFactorizedTileN ||
      intermediate_size % kSm87A4W4GateUpFactorizedTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpFactorizedCopyK != 0U ||
      primary_width == 0U || primary_width >= intermediate_size ||
      primary_width % kSm87A4W4GateUpFactorizedTileN != 0U ||
      (intermediate_size - primary_width) %
              kSm87A4W4GateUpFactorizedTileN !=
          0U) {
    return {};
  }
  const std::size_t m_tiles =
      launch_token_count / kSm87A4W4GateUpFactorizedTileM;
  const std::size_t n_tiles =
      intermediate_size / kSm87A4W4GateUpFactorizedTileN;
  if (!sm87_a4w4_gateup_factorized_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {logical_token_count,
          launch_token_count,
          intermediate_size,
          input_size,
          primary_width,
          intermediate_size - primary_width,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4GateUpFactorizedCopyK,
          input_size / kSm87A4W4GateUpFactorizedPhysicalK64,
          work_tiles,
          work_tiles < kSm87A4W4GateUpFactorizedPersistentCtas
              ? work_tiles
              : kSm87A4W4GateUpFactorizedPersistentCtas};
}

[[nodiscard]] constexpr Sm87A4W4GateUpFactorizedLanePlan
sm87_a4w4_gateup_factorized_lane_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return intermediate_size == kSm87A4W4GateUpFactorizedModelIntermediate &&
                 input_size == kSm87A4W4GateUpFactorizedModelInput
             ? sm87_a4w4_gateup_factorized_lane_test_plan(
                   logical_token_count, launch_token_count,
                   intermediate_size, input_size,
                   kSm87A4W4GateUpFactorizedPrimaryWidth)
             : Sm87A4W4GateUpFactorizedLanePlan{};
}

struct Sm87A4W4GateUpFactorizedLaneResources final {
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
query_sm87_a4w4_gateup_factorized_lane_resources_cuda(
    Sm87A4W4GateUpFactorizedLaneResources* resources) noexcept;

#define Q3X_SM87_A4W4_GATEUP_FACTORIZED_INPUT_ARGUMENTS                  \
  const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,      \
      const std::uint16_t* a_lane_scales_bf16,                           \
      std::size_t a_scale_capacity_elements,                             \
      const std::uint8_t* packed_gate_b,                                 \
      std::size_t packed_gate_b_capacity_bytes,                          \
      const std::uint16_t* gate_b_lane_scales_bf16,                      \
      std::size_t gate_b_scale_capacity_elements,                        \
      const std::uint8_t* packed_up_b,                                   \
      std::size_t packed_up_b_capacity_bytes,                            \
      const std::uint16_t* up_b_lane_scales_bf16,                        \
      std::size_t up_b_scale_capacity_elements

#define Q3X_SM87_A4W4_GATEUP_FACTORIZED_OUTPUT_ARGUMENTS                 \
  std::uint16_t* primary_output_bf16,                                    \
      std::size_t primary_output_row_stride_elements,                    \
      std::size_t primary_output_capacity_elements,                      \
      std::uint16_t* secondary_output_bf16,                              \
      std::size_t secondary_output_row_stride_elements,                  \
      std::size_t secondary_output_capacity_elements

[[nodiscard]] int launch_sm87_a4w4_gateup_factorized_lane_bf16_cuda(
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_INPUT_ARGUMENTS,
    std::size_t logical_token_count, std::size_t launch_token_count,
    std::size_t intermediate_size, std::size_t input_size,
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_OUTPUT_ARGUMENTS,
    void* cuda_stream = nullptr) noexcept;

// Production-shape, default-off candidate.  The BF16 outputs and all
// incumbent inputs retain the exact launcher contract above.  In addition,
// each GateUp CTA computes abs(BF16-RNE(product) * inverse_alpha) maxima for
// its N128 tile and publishes them to `r1_product_tile_maxima_fp32`.  The
// authenticated inverse-alpha is the exact Down projection R1 payload.
[[nodiscard]] int
launch_sm87_a4w4_gateup_factorized_lane_r1_tile_max_bf16_cuda(
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_INPUT_ARGUMENTS,
    std::size_t logical_token_count, std::size_t launch_token_count,
    std::size_t intermediate_size, std::size_t input_size,
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_OUTPUT_ARGUMENTS,
    const float* authenticated_down_inverse_alpha_fp32,
    std::size_t down_inverse_alpha_capacity_elements,
    float* r1_product_tile_maxima_fp32,
    std::size_t r1_product_tile_maxima_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
query_sm87_a4w4_gateup_factorized_lane_r1_tile_max_resources_cuda(
    Sm87A4W4GateUpFactorizedLaneResources* resources) noexcept;

// Isolated projection-plane-v2 API.  No runner selector references these
// symbols until the complete weight publication and full-model gate are
// wired.  A remains in the incumbent R1 consumer order.  `paired_b_codes`
// and `paired_b_lane_scales_bf16` use the exact equal-byte helpers above.
[[nodiscard]] int
query_sm87_a4w4_gateup_factorized_lane_r1_v2_resources_cuda(
    Sm87A4W4GateUpFactorizedLaneResources* resources) noexcept;

#define Q3X_SM87_A4W4_GATEUP_FACTORIZED_V2_INPUT_ARGUMENTS               \
  const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,      \
      const std::uint16_t* a_lane_scales_bf16,                           \
      std::size_t a_scale_capacity_elements,                             \
      const std::uint8_t* paired_b_codes,                                \
      std::size_t paired_b_code_capacity_bytes,                          \
      const std::uint16_t* paired_b_lane_scales_bf16,                    \
      std::size_t paired_b_scale_capacity_elements

[[nodiscard]] int
launch_sm87_a4w4_gateup_factorized_lane_r1_v2_tile_max_bf16_cuda(
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_V2_INPUT_ARGUMENTS,
    std::size_t logical_token_count, std::size_t launch_token_count,
    std::size_t intermediate_size, std::size_t input_size,
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_OUTPUT_ARGUMENTS,
    const float* authenticated_down_inverse_alpha_fp32,
    std::size_t down_inverse_alpha_capacity_elements,
    float* r1_product_tile_maxima_fp32,
    std::size_t r1_product_tile_maxima_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_factorized_lane_r1_v2_test_bf16_cuda(
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_V2_INPUT_ARGUMENTS,
    std::size_t logical_token_count, std::size_t launch_token_count,
    std::size_t intermediate_size, std::size_t input_size,
    std::size_t primary_width,
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_OUTPUT_ARGUMENTS,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

#undef Q3X_SM87_A4W4_GATEUP_FACTORIZED_V2_INPUT_ARGUMENTS

[[nodiscard]] int launch_sm87_a4w4_gateup_factorized_lane_test_bf16_cuda(
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_INPUT_ARGUMENTS,
    std::size_t logical_token_count, std::size_t launch_token_count,
    std::size_t intermediate_size, std::size_t input_size,
    std::size_t primary_width,
    Q3X_SM87_A4W4_GATEUP_FACTORIZED_OUTPUT_ARGUMENTS,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

#undef Q3X_SM87_A4W4_GATEUP_FACTORIZED_OUTPUT_ARGUMENTS
#undef Q3X_SM87_A4W4_GATEUP_FACTORIZED_INPUT_ARGUMENTS

static_assert(sm87_a4w4_gateup_factorized_lane_plan(
                  1'853U, 1'920U, 17'408U, 5'120U)
                      .work_tiles == 2'040U);
static_assert(sm87_a4w4_gateup_factorized_lane_test_plan(
                  129U, 256U, 256U, 256U, 128U)
                      .m_tiles == 2U);
static_assert(
    sm87_a4w4_gateup_factorized_v2_paired_code_capacity_bytes(
        kSm87A4W4GateUpFactorizedModelIntermediate,
        kSm87A4W4GateUpFactorizedModelInput) ==
    2U * sm87_a4w4_gateup_factorized_packed_capacity_bytes(
             kSm87A4W4GateUpFactorizedModelIntermediate,
             kSm87A4W4GateUpFactorizedModelInput));
static_assert(
    sm87_a4w4_gateup_factorized_v2_paired_scale_capacity_elements(
        kSm87A4W4GateUpFactorizedModelIntermediate) ==
    2U * sm87_a4w4_gateup_factorized_scale_capacity_elements(
             kSm87A4W4GateUpFactorizedModelIntermediate));

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_FACTOR_HOST_DEVICE
