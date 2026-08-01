#pragma once

#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Standalone Down-projection candidate.  No runtime selects this API.
//
// The incumbent M128N128 kernel spends 128 KiB of shared memory and can keep
// only eight warps resident.  This candidate retains the M128N128 cell but
// uses 512 threads.  Sixteen warps form an 8x2 grid of M16N64 warp tiles, so
// every thread owns only 32 FP32 outputs.  A three-stage K256 cp.async ring
// occupies 96 KiB.  Two adjacent slots form one K512 group while the third
// retains the next group's first half.  One S32 tuple accumulates all eight
// K64 MMAs before the one authenticated BF16 K512 scale product is applied.
//
// M128N128x512 was selected over both 256-thread half-cells.  M128N64 doubles
// the activation presentation by doubling N waves; M64N128 doubles B
// presentation by doubling M cells.  Two resident M64N128 CTAs expose the
// same sixteen warps as this one 512-thread CTA, so residency alone does not
// repay that extra traffic.
//
// Scheduling is shape-aware.  Equal owner groups use strict B waves: CTA b
// owns M tile owner_group*grid+b and walks every N tile in increasing order.
// Uneven owner groups use balanced N-major cells instead, preventing the one
// remainder owner at M17 from running a complete forty-N critical-path tail.
inline constexpr std::size_t kSm87A4W4DownK512BStationaryMinimumTokens =
    128U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryMaximumTokens =
    4'096U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryOutputSize =
    5'120U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryInputSize =
    17'408U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryTileM = 128U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryTileN = 128U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryStageK = 256U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryScaleK = 512U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryK64PerStage = 4U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryStagesPerScale = 2U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryStages = 3U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryThreads = 512U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryWarps = 16U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryWarpTileM = 16U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryWarpTileN = 64U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryWarpRows = 8U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryWarpColumns = 2U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryPackedRowK64Bytes =
    32U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryStageABytes =
    kSm87A4W4DownK512BStationaryTileM *
    kSm87A4W4DownK512BStationaryStageK / 2U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryStageBBytes =
    kSm87A4W4DownK512BStationaryTileN *
    kSm87A4W4DownK512BStationaryStageK / 2U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryStageBytes =
    kSm87A4W4DownK512BStationaryStageABytes +
    kSm87A4W4DownK512BStationaryStageBBytes;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryDynamicSharedBytes =
    kSm87A4W4DownK512BStationaryStages *
    kSm87A4W4DownK512BStationaryStageBytes;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryCtasPerSm = 1U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryPersistentCtas =
    16U;
inline constexpr std::size_t kSm87A4W4DownK512BStationaryMaximumRegisters =
    128U;

struct Sm87A4W4DownK512BStationaryPlan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k256_stages{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
  std::size_t owner_groups{};
};

struct Sm87A4W4DownK512BStationaryTile final {
  std::size_t m_tile{};
  std::size_t n_tile{};
  bool valid{};
};

// Equal owner groups use a strict M-owner -> N-wave schedule.  Uneven groups
// (for example ceil128(P2148)=2176, or 17 M tiles) deliberately use a
// balanced N-major schedule so one remainder owner cannot double the kernel's
// critical path merely to retain perfect B phase.
[[nodiscard]] constexpr bool
sm87_a4w4_down_k512_bstationary_uses_exact_owner_groups(
    const Sm87A4W4DownK512BStationaryPlan& plan) noexcept {
  return plan.launch_ctas != 0U &&
         (plan.m_tiles <= plan.launch_ctas ||
          plan.m_tiles % plan.launch_ctas == 0U);
}

[[nodiscard]] constexpr Sm87A4W4DownK512BStationaryPlan
sm87_a4w4_down_k512_bstationary_test_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4DownK512BStationaryTileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4DownK512BStationaryTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4DownK512BStationaryScaleK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4DownK512BStationaryTileM;
  const std::size_t n_tiles =
      output_size / kSm87A4W4DownK512BStationaryTileN;
  if (!sm87_a4w4_down_k512_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      m_tiles < kSm87A4W4DownK512BStationaryPersistentCtas
          ? m_tiles
          : kSm87A4W4DownK512BStationaryPersistentCtas;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4DownK512BStationaryScaleK,
          input_size / kSm87A4W4DownK512BStationaryStageK,
          input_size / kSm87A4W4DownK512PhysicalK64,
          work_tiles,
          launch_ctas,
          (m_tiles + launch_ctas - 1U) / launch_ctas};
}

[[nodiscard]] constexpr Sm87A4W4DownK512BStationaryPlan
sm87_a4w4_down_k512_bstationary_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  // The surrounding Prefill runner pads logical prompts to complete M128
  // extents.  Keep that production contract explicit at this standalone API.
  return token_count >= kSm87A4W4DownK512BStationaryMinimumTokens &&
                 token_count <=
                     kSm87A4W4DownK512BStationaryMaximumTokens &&
                 token_count % 128U == 0U &&
                 output_size ==
                     kSm87A4W4DownK512BStationaryOutputSize &&
                 input_size == kSm87A4W4DownK512BStationaryInputSize
             ? sm87_a4w4_down_k512_bstationary_test_plan(
                   token_count, output_size, input_size)
             : Sm87A4W4DownK512BStationaryPlan{};
}

[[nodiscard]] constexpr Sm87A4W4DownK512BStationaryTile
sm87_a4w4_down_k512_bstationary_tile(
    const Sm87A4W4DownK512BStationaryPlan& plan,
    const std::size_t cta, const std::size_t owner_group,
    const std::size_t n_wave) noexcept {
  if (!sm87_a4w4_down_k512_bstationary_uses_exact_owner_groups(plan) ||
      cta >= plan.launch_ctas ||
      owner_group >= plan.owner_groups || n_wave >= plan.n_tiles) {
    return {};
  }
  const std::size_t m_tile = owner_group * plan.launch_ctas + cta;
  return m_tile < plan.m_tiles
             ? Sm87A4W4DownK512BStationaryTile{m_tile, n_wave, true}
             : Sm87A4W4DownK512BStationaryTile{};
}

// Iteration i of CTA b in the uneven-owner fallback consumes flattened
// N-major cell b+i*launch_ctas.  This is a contract helper for the standalone
// candidate; the production kernel uses the identical arithmetic directly.
[[nodiscard]] constexpr Sm87A4W4DownK512BStationaryTile
sm87_a4w4_down_k512_bstationary_balanced_tile(
    const Sm87A4W4DownK512BStationaryPlan& plan,
    const std::size_t cta, const std::size_t iteration) noexcept {
  if (sm87_a4w4_down_k512_bstationary_uses_exact_owner_groups(plan) ||
      cta >= plan.launch_ctas ||
      !sm87_a4w4_down_k512_product_fits(iteration, plan.launch_ctas)) {
    return {};
  }
  const std::size_t work_base = iteration * plan.launch_ctas;
  if (work_base > std::numeric_limits<std::size_t>::max() - cta) {
    return {};
  }
  const std::size_t work_tile = cta + work_base;
  if (work_tile >= plan.work_tiles) {
    return {};
  }
  const std::size_t n_tile = work_tile / plan.m_tiles;
  return {work_tile - n_tile * plan.m_tiles, n_tile, true};
}

struct Sm87A4W4DownK512BStationaryResources final {
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
query_sm87_a4w4_down_k512_bstationary_resources_cuda(
    Sm87A4W4DownK512BStationaryResources* resources) noexcept;

[[nodiscard]] int launch_sm87_a4w4_down_k512_bstationary_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Correctness-only generic complete-cell launcher.  Synthetic input through
// this API is never a performance result.
[[nodiscard]] int
launch_sm87_a4w4_down_k512_bstationary_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4DownK512BStationaryStageABytes == 16U * 1'024U);
static_assert(kSm87A4W4DownK512BStationaryStageBBytes == 16U * 1'024U);
static_assert(kSm87A4W4DownK512BStationaryStageBytes == 32U * 1'024U);
static_assert(kSm87A4W4DownK512BStationaryDynamicSharedBytes ==
              96U * 1'024U);
static_assert(kSm87A4W4DownK512BStationaryWarpRows *
                      kSm87A4W4DownK512BStationaryWarpColumns ==
                  kSm87A4W4DownK512BStationaryWarps);
static_assert(kSm87A4W4DownK512BStationaryWarpTileM *
                      kSm87A4W4DownK512BStationaryWarpRows ==
                  kSm87A4W4DownK512BStationaryTileM);
static_assert(kSm87A4W4DownK512BStationaryWarpTileN *
                      kSm87A4W4DownK512BStationaryWarpColumns ==
                  kSm87A4W4DownK512BStationaryTileN);
static_assert(sm87_a4w4_down_k512_bstationary_plan(
                  1'920U, 5'120U, 17'408U)
                  .m_tiles == 15U);
static_assert(sm87_a4w4_down_k512_bstationary_plan(
                  1'920U, 5'120U, 17'408U)
                  .launch_ctas == 15U);
static_assert(sm87_a4w4_down_k512_bstationary_plan(
                  2'176U, 5'120U, 17'408U)
                  .m_tiles == 17U);
static_assert(sm87_a4w4_down_k512_bstationary_plan(
                  2'176U, 5'120U, 17'408U)
                  .owner_groups == 2U);
static_assert(!sm87_a4w4_down_k512_bstationary_uses_exact_owner_groups(
    sm87_a4w4_down_k512_bstationary_plan(
        2'176U, 5'120U, 17'408U)));
static_assert(sm87_a4w4_down_k512_bstationary_uses_exact_owner_groups(
    sm87_a4w4_down_k512_bstationary_plan(
        4'096U, 5'120U, 17'408U)));

}  // namespace q3x::kernels
