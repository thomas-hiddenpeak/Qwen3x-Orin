#pragma once

#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off structural Down experiment over the authenticated dense
// A4/K512 payload.  One CTA owns M256N128.  The complete K512 code epoch is
// resident in shared memory while sixteen warps keep the FP32 output alive,
// so the N128 B panel is presented once for four M64 panels (equivalently two
// M128 panels).  The 16-warp ownership halves per-thread accumulator state
// relative to the blocked 8-warp/M128N32 predecessor.
// There is no cross-CTA synchronization and no global partial-output scratch.
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentTileM = 256U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentTileN = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentPhysicalK64PerScale = 8U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentWarpM = 64U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentWarpN = 32U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentWarpRows = 4U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentWarpColumns = 4U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentM16PerWarp = 4U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentN16Phases = 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentN8PerPhase = 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentABytes =
        kSm87A4W4DownK512M256N128ResidentTileM *
        kSm87A4W4DownK512M256N128ResidentScaleK / 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentBBytes =
        kSm87A4W4DownK512M256N128ResidentTileN *
        kSm87A4W4DownK512M256N128ResidentScaleK / 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentScaleBytes =
        (kSm87A4W4DownK512M256N128ResidentTileM +
         kSm87A4W4DownK512M256N128ResidentTileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes =
        kSm87A4W4DownK512M256N128ResidentABytes +
        kSm87A4W4DownK512M256N128ResidentBBytes +
        kSm87A4W4DownK512M256N128ResidentScaleBytes;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentPersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentMinimumTokens = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentMaximumTokens = 4'096U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentModelOutput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4DownK512M256N128ResidentModelInput = 17'408U;

static_assert(kSm87A4W4DownK512M256N128ResidentThreads ==
              32U * kSm87A4W4DownK512M256N128ResidentWarps);
static_assert(kSm87A4W4DownK512M256N128ResidentWarpRows *
                      kSm87A4W4DownK512M256N128ResidentWarpM ==
                  kSm87A4W4DownK512M256N128ResidentTileM);
static_assert(kSm87A4W4DownK512M256N128ResidentWarpColumns *
                      kSm87A4W4DownK512M256N128ResidentWarpN ==
                  kSm87A4W4DownK512M256N128ResidentTileN);
static_assert(kSm87A4W4DownK512M256N128ResidentABytes == 65'536U);
static_assert(kSm87A4W4DownK512M256N128ResidentBBytes == 32'768U);
static_assert(kSm87A4W4DownK512M256N128ResidentScaleBytes == 768U);
static_assert(kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes ==
              99'072U);

struct Sm87A4W4DownK512M256N128ResidentPlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4DownK512M256N128ResidentPlan
sm87_a4w4_down_k512_m256n128_k512resident_test_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (logical_token_count == 0U || logical_token_count > launch_token_count ||
      launch_token_count == 0U || launch_token_count % 128U != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4DownK512M256N128ResidentTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4DownK512M256N128ResidentScaleK != 0U) {
    return {};
  }
  const std::size_t m_tiles = sm87_a4w4_ceil_div(
      launch_token_count, kSm87A4W4DownK512M256N128ResidentTileM);
  const std::size_t n_tiles =
      output_size / kSm87A4W4DownK512M256N128ResidentTileN;
  if (!sm87_a4w4_down_k512_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  const std::size_t launch_ctas =
      work_tiles < kSm87A4W4DownK512M256N128ResidentPersistentCtas
          ? work_tiles
          : kSm87A4W4DownK512M256N128ResidentPersistentCtas;
  return {logical_token_count,
          launch_token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4DownK512M256N128ResidentScaleK,
          input_size / kSm87A4W4DownK512PhysicalK64,
          work_tiles,
          launch_ctas};
}

[[nodiscard]] constexpr Sm87A4W4DownK512M256N128ResidentPlan
sm87_a4w4_down_k512_m256n128_k512resident_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  return logical_token_count >=
                     kSm87A4W4DownK512M256N128ResidentMinimumTokens &&
                 logical_token_count <=
                     kSm87A4W4DownK512M256N128ResidentMaximumTokens &&
                 launch_token_count ==
                     sm87_a4w4_prefill_k512_launch_token_count(
                         logical_token_count) &&
                 output_size ==
                     kSm87A4W4DownK512M256N128ResidentModelOutput &&
                 input_size ==
                     kSm87A4W4DownK512M256N128ResidentModelInput
             ? sm87_a4w4_down_k512_m256n128_k512resident_test_plan(
                   logical_token_count, launch_token_count, output_size,
                   input_size)
             : Sm87A4W4DownK512M256N128ResidentPlan{};
}

struct Sm87A4W4DownK512M256N128ResidentResources final {
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
query_sm87_a4w4_down_k512_m256n128_k512resident_resources_cuda(
    Sm87A4W4DownK512M256N128ResidentResources* resources) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_down_k512_m256n128_k512resident_bf16_cuda(
    const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements, const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements, std::size_t logical_token_count,
    std::size_t launch_token_count, std::size_t output_size,
    std::size_t input_size, std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_down_k512_m256n128_k512resident_test_bf16_cuda(
    const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements, const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements, std::size_t logical_token_count,
    std::size_t launch_token_count, std::size_t output_size,
    std::size_t input_size, std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements, unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
