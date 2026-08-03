#pragma once

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_DOWN_R4_M128N64_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_DOWN_R4_M128N64_HOST_DEVICE
#endif

namespace q3x::kernels {

// Resource-first, default-off R4 Down experiment for the pinned P1920 prefill
// bucket.  The 32-CTA grid is deliberately sized for two resident CTAs on
// each of Orin's sixteen SMs.  This surface has an independent CMake switch
// and remains absent from every runner and runtime selector: it must pass the
// resource gate before a release-shape correctness harness may be admitted.
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
    kSm87A4W4DownFactorizedR4M128N64LaneCount = 4U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64TileM = 128U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64TileN = 64U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64WarpM = 16U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64WarpN = 64U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64Warps = 8U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64Threads = 256U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64CopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64PhysicalK64 = 64U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64K64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64LaneK = 4'352U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64StagesPerLane = 17U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64TotalK256Stages = 68U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64SharedStages = 3U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64PersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64CtasPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64MaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64MaximumDynamicSharedBytes = 83'968U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64MinimumLogicalTokens = 1'793U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64LaunchTokens = 1'920U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64MOwners = 15U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64ModelOutput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64ModelInput = 17'408U;

inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64AStageBytes =
        kSm87A4W4DownFactorizedR4M128N64TileM *
        kSm87A4W4DownFactorizedR4M128N64CopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64BStageBytes =
        kSm87A4W4DownFactorizedR4M128N64TileN *
        kSm87A4W4DownFactorizedR4M128N64CopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64StageBytes =
        kSm87A4W4DownFactorizedR4M128N64AStageBytes +
        kSm87A4W4DownFactorizedR4M128N64BStageBytes;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64DynamicSharedBytes =
        kSm87A4W4DownFactorizedR4M128N64SharedStages *
        kSm87A4W4DownFactorizedR4M128N64StageBytes;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64NTiles =
        kSm87A4W4DownFactorizedR4M128N64ModelOutput /
        kSm87A4W4DownFactorizedR4M128N64TileN;
inline constexpr std::size_t
    kSm87A4W4DownFactorizedR4M128N64WorkTiles =
        kSm87A4W4DownFactorizedR4M128N64MOwners *
        kSm87A4W4DownFactorizedR4M128N64NTiles;

[[nodiscard]] Q3X_SM87_A4W4_DOWN_R4_M128N64_HOST_DEVICE constexpr
    std::size_t
sm87_a4w4_down_factorized_r4_m128n64_scale_offset(
    const std::size_t outer_coordinate,
    const std::size_t lane) noexcept {
  return ((outer_coordinate / kSm87A4W4ConsumerOuterBlock) *
              kSm87A4W4DownFactorizedR4M128N64LaneCount +
          lane) *
             kSm87A4W4ConsumerOuterBlock +
         outer_coordinate % kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_down_factorized_r4_m128n64_scale_capacity_elements(
    const std::size_t outer_count) noexcept {
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return blocks == 0U
             ? 0U
             : blocks * kSm87A4W4DownFactorizedR4M128N64LaneCount *
                   kSm87A4W4ConsumerOuterBlock;
}

struct Sm87A4W4DownFactorizedR4M128N64Plan final {
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

[[nodiscard]] constexpr Sm87A4W4DownFactorizedR4M128N64Plan
sm87_a4w4_down_factorized_r4_m128n64_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size) noexcept {
  return logical_token_count >=
                     kSm87A4W4DownFactorizedR4M128N64MinimumLogicalTokens &&
                 logical_token_count <=
                     kSm87A4W4DownFactorizedR4M128N64LaunchTokens &&
                 launch_token_count ==
                     kSm87A4W4DownFactorizedR4M128N64LaunchTokens &&
                 output_size ==
                     kSm87A4W4DownFactorizedR4M128N64ModelOutput &&
                 input_size ==
                     kSm87A4W4DownFactorizedR4M128N64ModelInput
             ? Sm87A4W4DownFactorizedR4M128N64Plan{
                   logical_token_count,
                   launch_token_count,
                   kSm87A4W4DownFactorizedR4M128N64MOwners,
                   kSm87A4W4DownFactorizedR4M128N64NTiles,
                   kSm87A4W4DownFactorizedR4M128N64WorkTiles,
                   kSm87A4W4DownFactorizedR4M128N64PersistentCtas}
             : Sm87A4W4DownFactorizedR4M128N64Plan{};
}

struct Sm87A4W4DownFactorizedR4M128N64Resources final {
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
// target: SM87 with exactly 16 SMs, <=128 registers/thread, <=83,968 bytes
// dynamic shared memory, zero local bytes, at least two active CTAs/SM, and at
// least thirty-two resident CTAs for the fixed grid.
[[nodiscard]] int
query_sm87_a4w4_down_factorized_r4_m128n64_resources_cuda(
    Sm87A4W4DownFactorizedR4M128N64Resources* resources) noexcept;

// Fail-closed launch ABI for the single release shape.  Capacities cover the
// complete P1920 allocation, including the unwritten logical-token tail.
// Every pointer must be 16-byte aligned and every supplied range disjoint.
[[nodiscard]] int
launch_sm87_a4w4_down_factorized_r4_m128n64_bf16_cuda(
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

static_assert(kSm87A4W4DownFactorizedR4M128N64Threads ==
              32U * kSm87A4W4DownFactorizedR4M128N64Warps);
static_assert(kSm87A4W4DownFactorizedR4M128N64Warps *
                      kSm87A4W4DownFactorizedR4M128N64WarpM ==
                  kSm87A4W4DownFactorizedR4M128N64TileM);
static_assert(kSm87A4W4DownFactorizedR4M128N64WarpN ==
              kSm87A4W4DownFactorizedR4M128N64TileN);
static_assert(kSm87A4W4DownFactorizedR4M128N64LaneCount *
                      kSm87A4W4DownFactorizedR4M128N64LaneK ==
                  kSm87A4W4DownFactorizedR4M128N64ModelInput);
static_assert(kSm87A4W4DownFactorizedR4M128N64StagesPerLane *
                      kSm87A4W4DownFactorizedR4M128N64CopyK ==
                  kSm87A4W4DownFactorizedR4M128N64LaneK);
static_assert(kSm87A4W4DownFactorizedR4M128N64TotalK256Stages ==
              kSm87A4W4DownFactorizedR4M128N64LaneCount *
                  kSm87A4W4DownFactorizedR4M128N64StagesPerLane);
static_assert(kSm87A4W4DownFactorizedR4M128N64LaunchTokens ==
              kSm87A4W4DownFactorizedR4M128N64MOwners *
                  kSm87A4W4DownFactorizedR4M128N64TileM);
static_assert(kSm87A4W4DownFactorizedR4M128N64AStageBytes == 16'384U);
static_assert(kSm87A4W4DownFactorizedR4M128N64BStageBytes == 8'192U);
static_assert(kSm87A4W4DownFactorizedR4M128N64StageBytes == 24'576U);
static_assert(
    kSm87A4W4DownFactorizedR4M128N64DynamicSharedBytes == 73'728U);
static_assert(kSm87A4W4DownFactorizedR4M128N64DynamicSharedBytes <=
              kSm87A4W4DownFactorizedR4M128N64MaximumDynamicSharedBytes);
static_assert(kSm87A4W4DownFactorizedR4M128N64NTiles == 80U);
static_assert(kSm87A4W4DownFactorizedR4M128N64WorkTiles == 1'200U);
static_assert(sm87_a4w4_down_factorized_r4_m128n64_scale_offset(
                  64U, 3U) == 448U);
static_assert(
    sm87_a4w4_down_factorized_r4_m128n64_scale_capacity_elements(
        1'920U) == 7'680U);
static_assert(sm87_a4w4_down_factorized_r4_m128n64_plan(
                  1'853U, 1'920U, 5'120U, 17'408U)
                      .work_tiles == 1'200U);
static_assert(!sm87_a4w4_down_factorized_r4_m128n64_plan(
                   1'792U, 1'920U, 5'120U, 17'408U));
static_assert(!sm87_a4w4_down_factorized_r4_m128n64_plan(
                   1'853U, 2'048U, 5'120U, 17'408U));

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_DOWN_R4_M128N64_HOST_DEVICE
