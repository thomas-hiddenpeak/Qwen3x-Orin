#pragma once

#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off structural Down candidate over the authenticated K512 v1
// payload.  One 512-thread CTA owns M64N256.  Its sixteen warps form a 2x8
// grid of M32N32 warp cells, retaining the incumbent 32 FP32 outputs and 32
// K512 S32 partials per thread while widening CTA-level B reuse.
//
// Four combined K256 stages form an asymmetric pair ring.  Every stage holds
// M64K256 A codes and N256K256 B codes; the slot two K512 groups ahead is
// filled while the next resident group computes.  Two BF16 scale slots are
// published with each group's odd K256 stage.  The resulting 165,120-byte
// allocation deliberately admits exactly one 16-warp CTA per SM.
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16TileM = 64U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16TileN = 256U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16WarpTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16WarpTileN = 32U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16WarpRows = 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16WarpColumns = 8U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16Threads = 512U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16Warps = 16U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16StageK = 256U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16ScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16K64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16StagesPerScale = 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16Stages = 4U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16ScaleSlots = 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16AStageBytes =
        kSm87A4W4DownK512M64N256Pairring16TileM *
        kSm87A4W4DownK512M64N256Pairring16StageK / 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16BStageBytes =
        kSm87A4W4DownK512M64N256Pairring16TileN *
        kSm87A4W4DownK512M64N256Pairring16StageK / 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16StageBytes =
        kSm87A4W4DownK512M64N256Pairring16AStageBytes +
        kSm87A4W4DownK512M64N256Pairring16BStageBytes;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16ScaleSlotBytes =
        (kSm87A4W4DownK512M64N256Pairring16TileM +
         kSm87A4W4DownK512M64N256Pairring16TileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes =
        kSm87A4W4DownK512M64N256Pairring16Stages *
            kSm87A4W4DownK512M64N256Pairring16StageBytes +
        kSm87A4W4DownK512M64N256Pairring16ScaleSlots *
            kSm87A4W4DownK512M64N256Pairring16ScaleSlotBytes;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16PersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16CtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16MaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16MaximumFp32OutputsPerThread = 32U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16ModelOutput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4DownK512M64N256Pairring16ModelInput = 17'408U;

static_assert(kSm87A4W4DownK512M64N256Pairring16Threads ==
              32U * kSm87A4W4DownK512M64N256Pairring16Warps);
static_assert(kSm87A4W4DownK512M64N256Pairring16WarpRows *
                      kSm87A4W4DownK512M64N256Pairring16WarpTileM ==
                  kSm87A4W4DownK512M64N256Pairring16TileM);
static_assert(kSm87A4W4DownK512M64N256Pairring16WarpColumns *
                      kSm87A4W4DownK512M64N256Pairring16WarpTileN ==
                  kSm87A4W4DownK512M64N256Pairring16TileN);
static_assert(kSm87A4W4DownK512M64N256Pairring16AStageBytes ==
              8'192U);
static_assert(kSm87A4W4DownK512M64N256Pairring16BStageBytes ==
              32'768U);
static_assert(kSm87A4W4DownK512M64N256Pairring16StageBytes ==
              40'960U);
static_assert(kSm87A4W4DownK512M64N256Pairring16ScaleSlotBytes ==
              640U);
static_assert(kSm87A4W4DownK512M64N256Pairring16DynamicSharedBytes ==
              165'120U);

struct Sm87A4W4DownK512M64N256Pairring16Plan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k256_stages{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
  std::size_t minimum_cells_per_cta{};
  std::size_t maximum_cells_per_cta{};
};

[[nodiscard]] constexpr Sm87A4W4DownK512M64N256Pairring16Plan
sm87_a4w4_down_k512_m64n256_16warp_pairring_test_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4DownK512M64N256Pairring16TileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4DownK512M64N256Pairring16TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4DownK512M64N256Pairring16ScaleK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4DownK512M64N256Pairring16TileM;
  const std::size_t n_tiles =
      output_size / kSm87A4W4DownK512M64N256Pairring16TileN;
  if (!sm87_a4w4_down_k512_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      work_cells < kSm87A4W4DownK512M64N256Pairring16PersistentCtas
          ? work_cells
          : kSm87A4W4DownK512M64N256Pairring16PersistentCtas;
  return {token_count,
          token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4DownK512M64N256Pairring16ScaleK,
          input_size / kSm87A4W4DownK512M64N256Pairring16StageK,
          input_size / kSm87A4W4DownK512PhysicalK64,
          work_cells,
          launch_ctas,
          work_cells / launch_ctas,
          (work_cells + launch_ctas - 1U) / launch_ctas};
}

[[nodiscard]] constexpr bool
sm87_a4w4_down_k512_m64n256_16warp_pairring_padding_contract(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count) noexcept {
  return logical_token_count != 0U &&
         launch_token_count ==
             sm87_a4w4_prefill_k512_launch_token_count(
                 logical_token_count) &&
         launch_token_count >= kSm87A4W4DownK512MinimumTokenCount &&
         launch_token_count <= kSm87A4W4DownK512MaximumTokenCount;
}

[[nodiscard]] constexpr Sm87A4W4DownK512M64N256Pairring16Plan
sm87_a4w4_down_k512_m64n256_16warp_pairring_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (!sm87_a4w4_down_k512_m64n256_16warp_pairring_padding_contract(
          logical_token_count, launch_token_count) ||
      output_size != kSm87A4W4DownK512M64N256Pairring16ModelOutput ||
      input_size != kSm87A4W4DownK512M64N256Pairring16ModelInput) {
    return {};
  }
  auto plan =
      sm87_a4w4_down_k512_m64n256_16warp_pairring_test_plan(
          launch_token_count, output_size, input_size);
  plan.logical_token_count = logical_token_count;
  return plan;
}

struct Sm87A4W4DownK512M64N256Pairring16Cell final {
  std::size_t m_tile{};
  std::size_t n_tile{};
  bool valid{};
};

// Flat N-major persistent ownership gives every CTA floor(W/16) or
// ceil(W/16) complete cells.  In particular, P2176's 34 M64 tiles do not
// create a dedicated final M wave.
[[nodiscard]] constexpr Sm87A4W4DownK512M64N256Pairring16Cell
sm87_a4w4_down_k512_m64n256_16warp_pairring_cell(
    const Sm87A4W4DownK512M64N256Pairring16Plan& plan,
    const std::size_t cta, const std::size_t iteration) noexcept {
  if (plan.launch_ctas == 0U || cta >= plan.launch_ctas ||
      !sm87_a4w4_down_k512_product_fits(iteration, plan.launch_ctas)) {
    return {};
  }
  const std::size_t flat = cta + iteration * plan.launch_ctas;
  return flat < plan.work_cells
             ? Sm87A4W4DownK512M64N256Pairring16Cell{
                   flat % plan.m_tiles, flat / plan.m_tiles, true}
             : Sm87A4W4DownK512M64N256Pairring16Cell{};
}

static_assert(
    sm87_a4w4_down_k512_m64n256_16warp_pairring_plan(
        1'853U, 1'920U, 5'120U, 17'408U)
            .minimum_cells_per_cta == 37U);
static_assert(
    sm87_a4w4_down_k512_m64n256_16warp_pairring_plan(
        1'853U, 1'920U, 5'120U, 17'408U)
            .maximum_cells_per_cta == 38U);
static_assert(
    sm87_a4w4_down_k512_m64n256_16warp_pairring_plan(
        2'049U, 2'176U, 5'120U, 17'408U)
            .minimum_cells_per_cta == 42U);
static_assert(
    sm87_a4w4_down_k512_m64n256_16warp_pairring_plan(
        2'049U, 2'176U, 5'120U, 17'408U)
            .maximum_cells_per_cta == 43U);

struct Sm87A4W4DownK512M64N256Pairring16Resources final {
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
query_sm87_a4w4_down_k512_m64n256_16warp_pairring_resources_cuda(
    Sm87A4W4DownK512M64N256Pairring16Resources* resources) noexcept;

// Production requires the exact ceil128 publication emitted by the K512 A4
// quantizer.  The complete padded output extent is written.
[[nodiscard]] int
launch_sm87_a4w4_down_k512_m64n256_16warp_pairring_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Correctness-only complete-cell launcher.  Synthetic data through this API
// is never a performance authority.
[[nodiscard]] int
launch_sm87_a4w4_down_k512_m64n256_16warp_pairring_test_bf16_cuda(
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

}  // namespace q3x::kernels
