#pragma once

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_R4_M128N64_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_R4_M128N64_HOST_DEVICE
#endif

namespace q3x::kernels {

// Resource-first, default-off R4 paired Gate+Up experiment for the pinned
// P1920 prefill bucket.  It is intentionally absent from CMake, every runner,
// and every selector.  Gate and Up share each staged A tile and directly
// publish one BF16-RNE SiLU(Gate)*Up product; no Gate scratch is materialized.
//
// Codes retain canonical signed-A4 consumer order:
//
//   [outer / 64][K / 64][64][32 bytes]
//
// A/Gate/Up scales use factorized R4 order:
//
//   [outer / 64][lane = 4][64] BF16
//
// K=5120 is four contiguous K1280 factor lanes.  Each lane owns five K256
// stages.  Its paired S32 partial is dequantized with that lane's A/Gate/Up
// scales into paired FP32 cross-lane state before S32 is reset.
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64LaneCount = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64TileM = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64TileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64WarpM = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64WarpN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64Warps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64Threads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64CopyK = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64PhysicalK64 = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64K64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64LaneK = 1'280U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64StagesPerLane = 5U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64TotalK256Stages = 20U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64SharedStages = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64PersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64CtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64MaximumRegisters = 255U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64MinimumLogicalTokens = 1'793U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64LaunchTokens = 1'920U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64MOwners = 15U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64ModelIntermediate = 17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64ModelInput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64PrimaryWidth = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64PrimaryStride = 12'288U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64SecondaryWidth = 5'120U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64SecondaryStride = 6'144U;

inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64AStageBytes =
        kSm87A4W4GateUpFactorizedR4M128N64TileM *
        kSm87A4W4GateUpFactorizedR4M128N64CopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64BStageBytes =
        kSm87A4W4GateUpFactorizedR4M128N64TileN *
        kSm87A4W4GateUpFactorizedR4M128N64CopyK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64StageBytes =
        kSm87A4W4GateUpFactorizedR4M128N64AStageBytes +
        2U * kSm87A4W4GateUpFactorizedR4M128N64BStageBytes;
// The initial all-register state exceeded the architectural register file.
// Four paired N8 cross-lane FP32 fragments are explicit field-major shared
// state.  They are touched only at four lane folds and the epilogue; the
// complete current-lane S32 partial remains registers throughout every MMA.
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64SharedCrossBytes =
        32U * kSm87A4W4GateUpFactorizedR4M128N64Threads *
        sizeof(float);
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64DynamicSharedBytes =
        kSm87A4W4GateUpFactorizedR4M128N64SharedStages *
            kSm87A4W4GateUpFactorizedR4M128N64StageBytes +
        kSm87A4W4GateUpFactorizedR4M128N64SharedCrossBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64NTiles =
        kSm87A4W4GateUpFactorizedR4M128N64ModelIntermediate /
        kSm87A4W4GateUpFactorizedR4M128N64TileN;
inline constexpr std::size_t
    kSm87A4W4GateUpFactorizedR4M128N64WorkTiles =
        kSm87A4W4GateUpFactorizedR4M128N64MOwners *
        kSm87A4W4GateUpFactorizedR4M128N64NTiles;

[[nodiscard]] Q3X_SM87_A4W4_GATEUP_R4_M128N64_HOST_DEVICE constexpr
    std::size_t
sm87_a4w4_gateup_factorized_r4_m128n64_scale_offset(
    const std::size_t outer_coordinate,
    const std::size_t factor_lane) noexcept {
  return ((outer_coordinate / kSm87A4W4ConsumerOuterBlock) *
              kSm87A4W4GateUpFactorizedR4M128N64LaneCount +
          factor_lane) *
             kSm87A4W4ConsumerOuterBlock +
         outer_coordinate % kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_factorized_r4_m128n64_scale_capacity_elements(
    const std::size_t outer_count) noexcept {
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return blocks == 0U
             ? 0U
             : blocks *
                   kSm87A4W4GateUpFactorizedR4M128N64LaneCount *
                   kSm87A4W4ConsumerOuterBlock;
}

struct Sm87A4W4GateUpFactorizedR4M128N64Plan final {
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

[[nodiscard]] constexpr Sm87A4W4GateUpFactorizedR4M128N64Plan
sm87_a4w4_gateup_factorized_r4_m128n64_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size) noexcept {
  return logical_token_count >=
                     kSm87A4W4GateUpFactorizedR4M128N64MinimumLogicalTokens &&
                 logical_token_count <=
                     kSm87A4W4GateUpFactorizedR4M128N64LaunchTokens &&
                 launch_token_count ==
                     kSm87A4W4GateUpFactorizedR4M128N64LaunchTokens &&
                 intermediate_size ==
                     kSm87A4W4GateUpFactorizedR4M128N64ModelIntermediate &&
                 input_size ==
                     kSm87A4W4GateUpFactorizedR4M128N64ModelInput
             ? Sm87A4W4GateUpFactorizedR4M128N64Plan{
                   logical_token_count,
                   launch_token_count,
                   kSm87A4W4GateUpFactorizedR4M128N64MOwners,
                   kSm87A4W4GateUpFactorizedR4M128N64NTiles,
                   kSm87A4W4GateUpFactorizedR4M128N64WorkTiles,
                   kSm87A4W4GateUpFactorizedR4M128N64PersistentCtas}
             : Sm87A4W4GateUpFactorizedR4M128N64Plan{};
}

struct Sm87A4W4GateUpFactorizedR4M128N64Resources final {
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

// cudaSuccess means all resource hard gates passed on the pinned target:
// <=255 registers/thread, zero local bytes, active>=1 CTA/SM, and at least
// sixteen resident blocks for the ordinary persistent grid.
[[nodiscard]] int
query_sm87_a4w4_gateup_factorized_r4_m128n64_resources_cuda(
    Sm87A4W4GateUpFactorizedR4M128N64Resources* resources) noexcept;

// Fail-closed launch ABI for the single release shape.  Capacities cover the
// complete P1920 allocation, including the unwritten logical-token tail and
// the secondary output's 1024-element row padding.  Every pointer must be
// 16-byte aligned and every supplied range must be disjoint.
[[nodiscard]] int
launch_sm87_a4w4_gateup_factorized_r4_m128n64_bf16_cuda(
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

static_assert(kSm87A4W4GateUpFactorizedR4M128N64Threads ==
              32U * kSm87A4W4GateUpFactorizedR4M128N64Warps);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64Warps *
                      kSm87A4W4GateUpFactorizedR4M128N64WarpM ==
                  kSm87A4W4GateUpFactorizedR4M128N64TileM);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64LaneCount *
                      kSm87A4W4GateUpFactorizedR4M128N64LaneK ==
                  kSm87A4W4GateUpFactorizedR4M128N64ModelInput);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64StagesPerLane *
                      kSm87A4W4GateUpFactorizedR4M128N64CopyK ==
                  kSm87A4W4GateUpFactorizedR4M128N64LaneK);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64TotalK256Stages ==
              kSm87A4W4GateUpFactorizedR4M128N64LaneCount *
                  kSm87A4W4GateUpFactorizedR4M128N64StagesPerLane);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64LaunchTokens ==
              kSm87A4W4GateUpFactorizedR4M128N64MOwners *
                  kSm87A4W4GateUpFactorizedR4M128N64TileM);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64PrimaryWidth +
                      kSm87A4W4GateUpFactorizedR4M128N64SecondaryWidth ==
                  kSm87A4W4GateUpFactorizedR4M128N64ModelIntermediate);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64AStageBytes == 16'384U);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64BStageBytes == 8'192U);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64StageBytes == 32'768U);
static_assert(
    kSm87A4W4GateUpFactorizedR4M128N64SharedCrossBytes == 32'768U);
static_assert(
    kSm87A4W4GateUpFactorizedR4M128N64DynamicSharedBytes == 163'840U);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64NTiles == 272U);
static_assert(kSm87A4W4GateUpFactorizedR4M128N64WorkTiles == 4'080U);
static_assert(sm87_a4w4_gateup_factorized_r4_m128n64_scale_offset(
                  64U, 3U) == 448U);
static_assert(
    sm87_a4w4_gateup_factorized_r4_m128n64_scale_capacity_elements(
        1'920U) == 7'680U);
static_assert(sm87_a4w4_gateup_factorized_r4_m128n64_plan(
                  1'853U, 1'920U, 17'408U, 5'120U)
                      .work_tiles == 4'080U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_R4_M128N64_HOST_DEVICE
