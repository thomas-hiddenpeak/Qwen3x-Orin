#pragma once

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_DOWN_R4_M192N128_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_DOWN_R4_M192N128_HOST_DEVICE
#endif

namespace q3x::kernels {

// Resource-first, default-off R4 Down producer for the pinned P1920 prefill
// bucket.  This surface is intentionally absent from CMake, every runner,
// and every selector.  It answers the resource question before a correctness
// harness or any production admission is allowed to exist.
//
// Codes retain canonical signed-A4 consumer order:
//
//   [outer / 64][K / 64][64][32 bytes]
//
// Activation and weight scales use the four-lane factorized order:
//
//   [outer / 64][lane = 4][64] BF16
//
// K=17408 is split into four contiguous K=4352 lanes.  Each lane owns
// exactly seventeen K256 stages.  S32 partials never cross a factor-lane
// boundary: a complete lane is dequantized with its A/W scale pair into an
// FP32 cross-lane result before the integer registers are reset.
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128LaneCount = 4U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128TileM = 192U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128TileN = 128U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128WarpM = 16U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128WarpN = 128U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128Warps = 12U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128Threads = 384U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128CopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128PhysicalK64 = 64U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128K64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128LaneK = 4'352U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128StagesPerLane = 17U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128TotalK256Stages = 68U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128SharedStages = 3U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128PersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128CtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128MaximumRegisters = 168U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128MinimumLogicalTokens = 1'793U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128LaunchTokens = 1'920U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128MOwners = 10U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128ModelOutput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128ModelInput = 17'408U;

inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128AStageBytes =
        kSm87A4W4DownFactorizedR4M192N128TileM *
        kSm87A4W4DownFactorizedR4M192N128CopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128BStageBytes =
        kSm87A4W4DownFactorizedR4M192N128TileN *
        kSm87A4W4DownFactorizedR4M192N128CopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128StageBytes =
        kSm87A4W4DownFactorizedR4M192N128AStageBytes +
        kSm87A4W4DownFactorizedR4M192N128BStageBytes;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes =
        kSm87A4W4DownFactorizedR4M192N128SharedStages *
        kSm87A4W4DownFactorizedR4M192N128StageBytes;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128NTiles =
        kSm87A4W4DownFactorizedR4M192N128ModelOutput /
        kSm87A4W4DownFactorizedR4M192N128TileN;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M192N128WorkTiles =
        kSm87A4W4DownFactorizedR4M192N128MOwners *
        kSm87A4W4DownFactorizedR4M192N128NTiles;

[[nodiscard]] Q3X_SM87_A4W4_DOWN_R4_M192N128_HOST_DEVICE constexpr
    std::size_t
sm87_a4w4_down_factorized_r4_m192n128_scale_offset(
    const std::size_t outer_coordinate,
    const std::size_t lane) noexcept {
  return ((outer_coordinate / kSm87A4W4ConsumerOuterBlock) *
              kSm87A4W4DownFactorizedR4M192N128LaneCount +
          lane) *
             kSm87A4W4ConsumerOuterBlock +
         outer_coordinate % kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_factorized_r4_m192n128_scale_capacity_elements(
    const std::size_t outer_count) noexcept {
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return blocks == 0U
             ? 0U
             : blocks * kSm87A4W4DownFactorizedR4M192N128LaneCount *
                   kSm87A4W4ConsumerOuterBlock;
}

struct Sm87A4W4DownFactorizedR4M192N128Plan final {
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

[[nodiscard]] constexpr Sm87A4W4DownFactorizedR4M192N128Plan
sm87_a4w4_down_factorized_r4_m192n128_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size) noexcept {
  return logical_token_count >=
                     kSm87A4W4DownFactorizedR4M192N128MinimumLogicalTokens &&
                 logical_token_count <=
                     kSm87A4W4DownFactorizedR4M192N128LaunchTokens &&
                 launch_token_count ==
                     kSm87A4W4DownFactorizedR4M192N128LaunchTokens &&
                 output_size ==
                     kSm87A4W4DownFactorizedR4M192N128ModelOutput &&
                 input_size ==
                     kSm87A4W4DownFactorizedR4M192N128ModelInput
             ? Sm87A4W4DownFactorizedR4M192N128Plan{
                   logical_token_count,
                   launch_token_count,
                   kSm87A4W4DownFactorizedR4M192N128MOwners,
                   kSm87A4W4DownFactorizedR4M192N128NTiles,
                   kSm87A4W4DownFactorizedR4M192N128WorkTiles,
                   kSm87A4W4DownFactorizedR4M192N128PersistentCtas}
             : Sm87A4W4DownFactorizedR4M192N128Plan{};
}

struct Sm87A4W4DownFactorizedR4M192N128Resources final {
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

// Returns cudaSuccess only when every resource hard gate passes on the pinned
// target: SM87 with exactly 16 SMs, <=168 registers/thread, zero local bytes,
// one active CTA/SM, and at least sixteen resident CTAs for the fixed grid.
[[nodiscard]] int
query_sm87_a4w4_down_factorized_r4_m192n128_resources_cuda(
    Sm87A4W4DownFactorizedR4M192N128Resources* resources) noexcept;

// Fail-closed launch ABI for the single release shape.  Capacities cover the
// complete P1920 allocation, including the unwritten logical-token tail.
// Every pointer must be 16-byte aligned and every supplied range disjoint.
[[nodiscard]] int
launch_sm87_a4w4_down_factorized_r4_m192n128_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_lane_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4DownFactorizedR4M192N128Threads ==
              32U * kSm87A4W4DownFactorizedR4M192N128Warps);
static_assert(kSm87A4W4DownFactorizedR4M192N128Warps *
                      kSm87A4W4DownFactorizedR4M192N128WarpM ==
                  kSm87A4W4DownFactorizedR4M192N128TileM);
static_assert(kSm87A4W4DownFactorizedR4M192N128WarpN ==
              kSm87A4W4DownFactorizedR4M192N128TileN);
static_assert(kSm87A4W4DownFactorizedR4M192N128LaneCount *
                      kSm87A4W4DownFactorizedR4M192N128LaneK ==
                  kSm87A4W4DownFactorizedR4M192N128ModelInput);
static_assert(kSm87A4W4DownFactorizedR4M192N128StagesPerLane *
                      kSm87A4W4DownFactorizedR4M192N128CopyK ==
                  kSm87A4W4DownFactorizedR4M192N128LaneK);
static_assert(kSm87A4W4DownFactorizedR4M192N128TotalK256Stages ==
              kSm87A4W4DownFactorizedR4M192N128LaneCount *
                  kSm87A4W4DownFactorizedR4M192N128StagesPerLane);
static_assert(kSm87A4W4DownFactorizedR4M192N128LaunchTokens ==
              kSm87A4W4DownFactorizedR4M192N128MOwners *
                  kSm87A4W4DownFactorizedR4M192N128TileM);
static_assert(kSm87A4W4DownFactorizedR4M192N128AStageBytes == 24'576U);
static_assert(kSm87A4W4DownFactorizedR4M192N128BStageBytes == 16'384U);
static_assert(kSm87A4W4DownFactorizedR4M192N128StageBytes == 40'960U);
static_assert(
    kSm87A4W4DownFactorizedR4M192N128DynamicSharedBytes == 122'880U);
static_assert(kSm87A4W4DownFactorizedR4M192N128NTiles == 40U);
static_assert(kSm87A4W4DownFactorizedR4M192N128WorkTiles == 400U);
static_assert(sm87_a4w4_down_factorized_r4_m192n128_scale_offset(
                  64U, 3U) == 448U);
static_assert(
    sm87_a4w4_down_factorized_r4_m192n128_scale_capacity_elements(
        1'920U) == 7'680U);
static_assert(sm87_a4w4_down_factorized_r4_m192n128_plan(
                  1'853U, 1'920U, 5'120U, 17'408U)
                      .work_tiles == 400U);
static_assert(!sm87_a4w4_down_factorized_r4_m192n128_plan(
                   1'792U, 1'920U, 5'120U, 17'408U));
static_assert(!sm87_a4w4_down_factorized_r4_m192n128_plan(
                   1'853U, 2'048U, 5'120U, 17'408U));

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_DOWN_R4_M192N128_HOST_DEVICE
