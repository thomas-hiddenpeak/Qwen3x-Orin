#pragma once

#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_DOWN_K512_M16N64_V2_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_DOWN_K512_M16N64_V2_HOST_DEVICE
#endif

namespace q3x::kernels {

// Full-residency successor to the authenticated Down K512 macrocell.  The
// CTA still owns one exact M128N128 output tile and consumes the same packed
// codes and independent BF16 K512 scales.  Sixteen warps replace the former
// eight-warp M32N64 ownership: warp w owns one M16N64 strip, so the resident
// FP32 output payload is halved without changing any arithmetic boundary.
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2TileM = 128U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2TileN = 128U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2WarpTileM = 16U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2WarpTileN = 64U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2WarpRows = 8U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2WarpColumns = 2U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2Threads = 512U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2Warps = 16U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2CopyK = 256U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2ScaleK = 512U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2K64PerStage = 4U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2StagesPerScale = 2U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2Stages = 4U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2StageABytes =
    kSm87A4W4DownK512M16N64V2TileM *
    kSm87A4W4DownK512M16N64V2CopyK / 2U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2StageBBytes =
    kSm87A4W4DownK512M16N64V2TileN *
    kSm87A4W4DownK512M16N64V2CopyK / 2U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2StageBytes =
    kSm87A4W4DownK512M16N64V2StageABytes +
    kSm87A4W4DownK512M16N64V2StageBBytes;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2DynamicSharedBytes =
    kSm87A4W4DownK512M16N64V2Stages *
    kSm87A4W4DownK512M16N64V2StageBytes;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2PersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2CtasPerSm = 1U;
inline constexpr std::size_t kSm87A4W4DownK512M16N64V2MaximumRegisters =
    128U;

struct Sm87A4W4DownK512M16N64V2Plan final {
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
  std::size_t base_waves{};
  std::size_t base_m_tiles{};
  std::size_t residual_m_tiles{};
  std::size_t residual_work_tiles{};
};

struct Sm87A4W4DownK512M16N64V2WorkTile final {
  std::size_t m_tile{};
  std::size_t n_tile{};
  bool valid{};
};

[[nodiscard]] constexpr Sm87A4W4DownK512M16N64V2Plan
sm87_a4w4_down_k512_m16n64_v2_test_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4DownK512M16N64V2TileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4DownK512M16N64V2TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4DownK512M16N64V2ScaleK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4DownK512M16N64V2TileM;
  const std::size_t n_tiles =
      output_size / kSm87A4W4DownK512M16N64V2TileN;
  constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
  if (m_tiles > maximum / n_tiles) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      m_tiles < kSm87A4W4DownK512M16N64V2PersistentCtas
          ? m_tiles
          : kSm87A4W4DownK512M16N64V2PersistentCtas;
  const std::size_t base_waves = m_tiles / launch_ctas;
  const std::size_t base_m_tiles = base_waves * launch_ctas;
  const std::size_t residual_m_tiles = m_tiles - base_m_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4DownK512M16N64V2ScaleK,
          input_size / kSm87A4W4DownK512M16N64V2CopyK,
          input_size / kSm87A4W4DownK512PhysicalK64,
          work_tiles,
          launch_ctas,
          base_waves,
          base_m_tiles,
          residual_m_tiles,
          residual_m_tiles * n_tiles};
}

[[nodiscard]] constexpr Sm87A4W4DownK512M16N64V2Plan
sm87_a4w4_down_k512_m16n64_v2_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  return token_count >= kSm87A4W4DownK512MinimumTokenCount &&
                 token_count <= kSm87A4W4DownK512MaximumTokenCount &&
                 output_size == kSm87A4W4DownK512OutputSize &&
                 input_size == kSm87A4W4DownK512InputSize
             ? sm87_a4w4_down_k512_m16n64_v2_test_plan(
                   token_count, output_size, input_size)
             : Sm87A4W4DownK512M16N64V2Plan{};
}

// For M<=2048 the launch count equals the M-tile count and every CTA keeps
// one fixed M owner while walking N in lockstep.  Larger spans first execute
// complete fixed-M waves.  Only the final partial M wave is decomposed into
// N-major complete cells and balanced over all sixteen CTAs.
[[nodiscard]] Q3X_SM87_A4W4_DOWN_K512_M16N64_V2_HOST_DEVICE constexpr
    Sm87A4W4DownK512M16N64V2WorkTile
sm87_a4w4_down_k512_m16n64_v2_work_tile(
    const Sm87A4W4DownK512M16N64V2Plan& plan,
    const std::size_t cta, const std::size_t iteration) noexcept {
  if (plan.launch_ctas == 0U || cta >= plan.launch_ctas) {
    return {};
  }
  const std::size_t base_iterations = plan.base_waves * plan.n_tiles;
  if (iteration < base_iterations) {
    const std::size_t wave = iteration / plan.n_tiles;
    const std::size_t n_tile = iteration - wave * plan.n_tiles;
    return {wave * plan.launch_ctas + cta, n_tile, true};
  }
  if (plan.residual_m_tiles == 0U) {
    return {};
  }
  const std::size_t residual_iteration = iteration - base_iterations;
  constexpr std::size_t maximum = ~std::size_t{0};
  if (residual_iteration >
      (maximum - cta) / plan.launch_ctas) {
    return {};
  }
  const std::size_t ordinal =
      cta + residual_iteration * plan.launch_ctas;
  if (ordinal >= plan.residual_work_tiles) {
    return {};
  }
  const std::size_t n_tile = ordinal / plan.residual_m_tiles;
  return {plan.base_m_tiles + ordinal - n_tile * plan.residual_m_tiles,
          n_tile, true};
}

struct Sm87A4W4DownK512M16N64V2Resources final {
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

[[nodiscard]] int query_sm87_a4w4_down_k512_m16n64_v2_resources_cuda(
    Sm87A4W4DownK512M16N64V2Resources* resources) noexcept;

[[nodiscard]] int launch_sm87_a4w4_down_k512_m16n64_v2_bf16_cuda(
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

[[nodiscard]] int launch_sm87_a4w4_down_k512_m16n64_v2_test_bf16_cuda(
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

static_assert(kSm87A4W4DownK512M16N64V2Threads ==
              kSm87A4W4DownK512M16N64V2Warps * 32U);
static_assert(kSm87A4W4DownK512M16N64V2WarpRows *
                      kSm87A4W4DownK512M16N64V2WarpColumns ==
                  kSm87A4W4DownK512M16N64V2Warps);
static_assert(kSm87A4W4DownK512M16N64V2WarpTileM *
                      kSm87A4W4DownK512M16N64V2WarpRows ==
                  kSm87A4W4DownK512M16N64V2TileM);
static_assert(kSm87A4W4DownK512M16N64V2WarpTileN *
                      kSm87A4W4DownK512M16N64V2WarpColumns ==
                  kSm87A4W4DownK512M16N64V2TileN);
static_assert(kSm87A4W4DownK512M16N64V2DynamicSharedBytes ==
              128U * 1'024U);
static_assert(sm87_a4w4_down_k512_m16n64_v2_plan(
                  1'920U, 5'120U, 17'408U)
                  .launch_ctas == 15U);
static_assert(sm87_a4w4_down_k512_m16n64_v2_plan(
                  2'176U, 5'120U, 17'408U)
                  .residual_work_tiles == 40U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_DOWN_K512_M16N64_V2_HOST_DEVICE
