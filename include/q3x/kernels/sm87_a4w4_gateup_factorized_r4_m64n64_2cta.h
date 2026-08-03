#pragma once

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_R4_M64N64_2CTA_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_R4_M64N64_2CTA_HOST_DEVICE
#endif

namespace q3x::kernels {

// Resource-first, default-off R4 Gate+Up experiment for the pinned P1920
// prefill bucket.  This is a distinct two-CTA/SM architecture, not a tuning
// variant of the admitted M128N64 kernel, and is deliberately absent from all
// runner/API selectors.
//
// Eight warps form a 4x2 warp grid.  Each warp owns M16N32 and retains both
// Gate and Up's complete current-lane S32 state in registers.  Gate's sixteen
// cross-lane FP32 values remain registers while Up's exactly sixteen values
// use a field-major shared tail.  This one structural successor to the
// all-register attempt trades 16 KiB shared memory for enough register headroom
// to target the same two-CTA/SM residency without compiler spills.
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaneCount = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWarpM = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWarpN = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaCopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPhysicalK64 = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaK64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaneK = 1'280U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaStagesPerLane = 5U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTotalK256Stages = 20U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaCtasPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMinimumLogicalTokens = 1'793U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaunchTokens = 1'920U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMOwners = 30U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelIntermediate = 17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPrimaryWidth = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPrimaryStride = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSecondaryWidth = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSecondaryStride = 6'144U;

inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaAStageBytes =
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileM *
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaBStageBytes =
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileN *
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaCopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaStageBytes =
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaAStageBytes +
        2U * kSm87A4W4GateUpFactorizedR4M64N64TwoCtaBStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossValuesPerThread = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossBytes =
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossValuesPerThread *
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaThreads * sizeof(float);
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes =
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedStages *
            kSm87A4W4GateUpFactorizedR4M64N64TwoCtaStageBytes +
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaNTiles =
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelIntermediate /
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileN;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWorkTiles =
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMOwners *
        kSm87A4W4GateUpFactorizedR4M64N64TwoCtaNTiles;

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_R4_M64N64_2CTA_HOST_DEVICE constexpr
    std::size_t
sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(
    const std::size_t outer_coordinate,
    const std::size_t factor_lane) noexcept {
  return ((outer_coordinate / kSm87A4W4ConsumerOuterBlock) *
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaneCount +
          factor_lane) *
             kSm87A4W4ConsumerOuterBlock +
         outer_coordinate % kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_capacity_elements(
    const std::size_t outer_count) noexcept {
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return blocks == 0U
             ? 0U
             : blocks *
                   kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaneCount *
                   kSm87A4W4ConsumerOuterBlock;
}

struct Sm87A4W4GateUpFactorizedR4M64N64TwoCtaPlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t m_owners{};
  std::size_t n_tiles{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return launch_ctas != 0U;
  }
};

[[nodiscard]] constexpr Sm87A4W4GateUpFactorizedR4M64N64TwoCtaPlan
sm87_a4w4_gateup_factorized_r4_m64n64_2cta_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return logical_token_count >=
                     kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMinimumLogicalTokens &&
                 logical_token_count <=
                     kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaunchTokens &&
                 launch_token_count ==
                     kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelIntermediate &&
                 input_size ==
                     kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelInput
             ? Sm87A4W4GateUpFactorizedR4M64N64TwoCtaPlan{
                   logical_token_count,
                   launch_token_count,
                   kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMOwners,
                   kSm87A4W4GateUpFactorizedR4M64N64TwoCtaNTiles,
                   kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWorkTiles,
                   kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPersistentCtas}
             : Sm87A4W4GateUpFactorizedR4M64N64TwoCtaPlan{};
}

struct Sm87A4W4GateUpFactorizedR4M64N64TwoCtaResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t configured_dynamic_shared_limit_bytes{};
  std::size_t device_optin_shared_limit_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int resident_blocks{};
  int multiprocessor_count{};
  int compute_major{};
  int compute_minor{};
};

// cudaSuccess means every hard gate passed on the pinned target: <=128
// registers/thread, zero local memory, active>=2 CTA/SM, and >=32 resident
// blocks for the persistent grid.
[[nodiscard]] int
query_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_resources_cuda(
    Sm87A4W4GateUpFactorizedR4M64N64TwoCtaResources* resources) noexcept;

// Test-only launch ABI.  It is intentionally not wired into the runner/API.
[[nodiscard]] int
launch_sm87_a4w4_gateup_factorized_r4_m64n64_2cta_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_gate,
    std::size_t packed_gate_capacity_bytes,
    const std::uint16_t* gate_lane_scales_bf16,
    std::size_t gate_scale_capacity_elements,
    const std::uint8_t* packed_up,
    std::size_t packed_up_capacity_bytes,
    const std::uint16_t* up_lane_scales_bf16,
    std::size_t up_scale_capacity_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::uint16_t* primary_output_bf16,
    std::size_t primary_output_row_stride_elements,
    std::size_t primary_output_capacity_elements,
    std::uint16_t* secondary_output_bf16,
    std::size_t secondary_output_row_stride_elements,
    std::size_t secondary_output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaThreads ==
              32U * kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWarps);
static_assert((kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileM /
               kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWarpM) *
                  (kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileN /
                   kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWarpN) ==
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWarps);
static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaneCount *
                      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaneK ==
                  kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelInput);
static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaStagesPerLane *
                      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaCopyK ==
                  kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaneK);
static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaLaunchTokens ==
              kSm87A4W4GateUpFactorizedR4M64N64TwoCtaMOwners *
                  kSm87A4W4GateUpFactorizedR4M64N64TwoCtaTileM);
static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaPrimaryWidth +
                      kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSecondaryWidth ==
                  kSm87A4W4GateUpFactorizedR4M64N64TwoCtaModelIntermediate);
static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaAStageBytes == 8'192U);
static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaBStageBytes == 8'192U);
static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaStageBytes == 24'576U);
static_assert(
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaSharedCrossBytes == 16'384U);
static_assert(
    kSm87A4W4GateUpFactorizedR4M64N64TwoCtaDynamicSharedBytes == 65'536U);
static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaNTiles == 272U);
static_assert(kSm87A4W4GateUpFactorizedR4M64N64TwoCtaWorkTiles == 8'160U);
static_assert(sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_offset(
                  64U, 3U) == 448U);
static_assert(
    sm87_a4w4_gateup_factorized_r4_m64n64_2cta_scale_capacity_elements(
        1'920U) == 7'680U);
static_assert(sm87_a4w4_gateup_factorized_r4_m64n64_2cta_plan(
                  1'853U, 1'920U, 17'408U, 5'120U)
                      .work_tiles == 8'160U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_R4_M64N64_2CTA_HOST_DEVICE
