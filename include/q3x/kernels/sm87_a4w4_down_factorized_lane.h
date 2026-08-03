#pragma once

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_DOWN_FACTOR_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_DOWN_FACTOR_HOST_DEVICE
#endif

namespace q3x::kernels {

// Default-off R1 factorized-scale Down experiment.  This is an isolated CUDA
// API: no engine, runner, sidecar selector, or production route references it.
// Packed A/W retain [outer/64][K/64][64][32].  Each operand owns one BF16
// scale per outer row in the lane-count-one specialization of
// [outer/64][lane][64].  The complete K reduction stays S32 and is multiplied
// by the A-row and W-row scales exactly once at the final BF16 store.
inline constexpr std::size_t kSm87A4W4DownFactorizedLaneCount = 1U;
inline constexpr std::size_t kSm87A4W4DownFactorizedTileM = 256U;
inline constexpr std::size_t kSm87A4W4DownFactorizedTileN = 128U;
inline constexpr std::size_t kSm87A4W4DownFactorizedCopyK = 256U;
inline constexpr std::size_t kSm87A4W4DownFactorizedPhysicalK64 = 64U;
inline constexpr std::size_t kSm87A4W4DownFactorizedK64PerStage = 4U;
inline constexpr std::size_t kSm87A4W4DownFactorizedThreads = 512U;
inline constexpr std::size_t kSm87A4W4DownFactorizedWarps = 16U;
inline constexpr std::size_t kSm87A4W4DownFactorizedWarpM = 64U;
inline constexpr std::size_t kSm87A4W4DownFactorizedWarpN = 32U;
inline constexpr std::size_t kSm87A4W4DownFactorizedWarpRows = 4U;
inline constexpr std::size_t kSm87A4W4DownFactorizedWarpColumns = 4U;
inline constexpr std::size_t kSm87A4W4DownFactorizedAStages = 2U;
inline constexpr std::size_t kSm87A4W4DownFactorizedBStages = 2U;
inline constexpr std::size_t kSm87A4W4DownFactorizedStageABytes =
    kSm87A4W4DownFactorizedTileM * kSm87A4W4DownFactorizedCopyK / 2U;
inline constexpr std::size_t kSm87A4W4DownFactorizedStageBBytes =
    kSm87A4W4DownFactorizedTileN * kSm87A4W4DownFactorizedCopyK / 2U;
inline constexpr std::size_t kSm87A4W4DownFactorizedScaleBytes =
    (kSm87A4W4DownFactorizedTileM +
     kSm87A4W4DownFactorizedTileN) *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87A4W4DownFactorizedDynamicSharedBytes =
    kSm87A4W4DownFactorizedAStages *
        kSm87A4W4DownFactorizedStageABytes +
    kSm87A4W4DownFactorizedBStages *
        kSm87A4W4DownFactorizedStageBBytes +
    kSm87A4W4DownFactorizedScaleBytes;
inline constexpr std::size_t kSm87A4W4DownFactorizedPersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4DownFactorizedCtasPerSm = 1U;
inline constexpr std::size_t kSm87A4W4DownFactorizedMaximumRegisters = 128U;
inline constexpr std::size_t kSm87A4W4DownFactorizedMaximumTokens = 4'096U;
inline constexpr std::size_t kSm87A4W4DownFactorizedModelOutput = 5'120U;
inline constexpr std::size_t kSm87A4W4DownFactorizedModelInput = 17'408U;
inline constexpr std::int32_t kSm87A4W4DownFactorizedMaximumS32 =
    static_cast<std::int32_t>(
        7U * 7U * kSm87A4W4DownFactorizedModelInput);

static_assert(kSm87A4W4DownFactorizedThreads ==
              32U * kSm87A4W4DownFactorizedWarps);
static_assert(kSm87A4W4DownFactorizedWarpRows *
                      kSm87A4W4DownFactorizedWarpM ==
                  kSm87A4W4DownFactorizedTileM);
static_assert(kSm87A4W4DownFactorizedWarpColumns *
                      kSm87A4W4DownFactorizedWarpN ==
                  kSm87A4W4DownFactorizedTileN);
static_assert(kSm87A4W4DownFactorizedStageABytes == 32'768U);
static_assert(kSm87A4W4DownFactorizedStageBBytes == 16'384U);
static_assert(kSm87A4W4DownFactorizedScaleBytes == 768U);
static_assert(kSm87A4W4DownFactorizedDynamicSharedBytes == 99'072U);
static_assert(kSm87A4W4DownFactorizedMaximumS32 == 852'992);

[[nodiscard]] constexpr bool sm87_a4w4_down_factorized_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_factorized_outer_blocks(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U
             ? 0U
             : 1U + (outer_count - 1U) /
                        kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_factorized_k64_groups(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4DownFactorizedPhysicalK64 == 0U
             ? logical_k / kSm87A4W4DownFactorizedPhysicalK64
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_factorized_k256_stages(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4DownFactorizedCopyK == 0U
             ? logical_k / kSm87A4W4DownFactorizedCopyK
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_factorized_packed_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_down_factorized_outer_blocks(outer_count);
  const std::size_t groups =
      sm87_a4w4_down_factorized_k64_groups(logical_k);
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_down_factorized_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  constexpr std::size_t kBytesPerBlockGroup =
      kSm87A4W4ConsumerOuterBlock *
      kSm87A4W4ConsumerPackedKBlockBytes;
  return sm87_a4w4_down_factorized_product_fits(
             block_groups, kBytesPerBlockGroup)
             ? block_groups * kBytesPerBlockGroup
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_DOWN_FACTOR_HOST_DEVICE constexpr std::size_t
sm87_a4w4_down_factorized_packed_offset(
    const std::size_t outer_coordinate, const std::size_t k64_group,
    const std::size_t byte_in_k64,
    const std::size_t k64_group_count) noexcept {
  return sm87_a4w4_consumer_packed_offset(
      outer_coordinate, k64_group, byte_in_k64, k64_group_count);
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_factorized_scale_capacity_elements(
    const std::size_t outer_count) noexcept {
  const std::size_t blocks =
      sm87_a4w4_down_factorized_outer_blocks(outer_count);
  return blocks == 0U ||
                 !sm87_a4w4_down_factorized_product_fits(
                     blocks, kSm87A4W4ConsumerOuterBlock)
             ? 0U
             : blocks * kSm87A4W4ConsumerOuterBlock;
}

// R1 specialization of [outer/64][lane][64].
[[nodiscard]] Q3X_SM87_A4W4_DOWN_FACTOR_HOST_DEVICE constexpr std::size_t
sm87_a4w4_down_factorized_scale_offset(
    const std::size_t outer_coordinate) noexcept {
  return (outer_coordinate / kSm87A4W4ConsumerOuterBlock) *
             kSm87A4W4ConsumerOuterBlock +
         outer_coordinate % kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_factorized_launch_token_count(
    const std::size_t logical_token_count) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count >
          std::numeric_limits<std::size_t>::max() -
              (kSm87A4W4DownFactorizedTileM - 1U)) {
    return 0U;
  }
  return ((logical_token_count + kSm87A4W4DownFactorizedTileM - 1U) /
          kSm87A4W4DownFactorizedTileM) *
         kSm87A4W4DownFactorizedTileM;
}

struct Sm87A4W4DownFactorizedLanePlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k256_stages{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

// Flexible shapes exist solely for the isolated correctness executable.
[[nodiscard]] constexpr Sm87A4W4DownFactorizedLanePlan
sm87_a4w4_down_factorized_lane_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count > launch_token_count ||
      launch_token_count == 0U ||
      launch_token_count % kSm87A4W4DownFactorizedTileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4DownFactorizedTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4DownFactorizedCopyK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      launch_token_count / kSm87A4W4DownFactorizedTileM;
  const std::size_t n_tiles =
      output_size / kSm87A4W4DownFactorizedTileN;
  if (!sm87_a4w4_down_factorized_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {logical_token_count,
          launch_token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4DownFactorizedCopyK,
          input_size / kSm87A4W4DownFactorizedPhysicalK64,
          work_tiles,
          work_tiles < kSm87A4W4DownFactorizedPersistentCtas
              ? work_tiles
              : kSm87A4W4DownFactorizedPersistentCtas};
}

[[nodiscard]] constexpr Sm87A4W4DownFactorizedLanePlan
sm87_a4w4_down_factorized_lane_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  return logical_token_count <= kSm87A4W4DownFactorizedMaximumTokens &&
                 launch_token_count ==
                     sm87_a4w4_down_factorized_launch_token_count(
                         logical_token_count) &&
                 output_size == kSm87A4W4DownFactorizedModelOutput &&
                 input_size == kSm87A4W4DownFactorizedModelInput
             ? sm87_a4w4_down_factorized_lane_test_plan(
                   logical_token_count, launch_token_count, output_size,
                   input_size)
             : Sm87A4W4DownFactorizedLanePlan{};
}

struct Sm87A4W4DownFactorizedLaneResources final {
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
query_sm87_a4w4_down_factorized_lane_resources_cuda(
    Sm87A4W4DownFactorizedLaneResources* resources) noexcept;

[[nodiscard]] int launch_sm87_a4w4_down_factorized_lane_bf16_cuda(
    const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements, const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_lane_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t logical_token_count, std::size_t launch_token_count,
    std::size_t output_size, std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_a4w4_down_factorized_lane_test_bf16_cuda(
    const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements, const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_lane_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t logical_token_count, std::size_t launch_token_count,
    std::size_t output_size, std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements, unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

static_assert(sm87_a4w4_down_factorized_lane_plan(
                  1'853U, 2'048U, 5'120U, 17'408U)
                      .work_tiles == 320U);
static_assert(sm87_a4w4_down_factorized_lane_test_plan(
                  257U, 512U, 128U, 256U)
                      .m_tiles == 2U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_DOWN_FACTOR_HOST_DEVICE
