#pragma once

#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_DOWN_PAIRRING16_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_DOWN_PAIRRING16_HOST_DEVICE
#endif

namespace q3x::kernels {

// Default-off Down successor over the authenticated K512 v1 payload.  It
// keeps the incumbent M128N128 cell and exact K512 numerical boundary, but
// maps the cell to a 4x4 grid of M32N32 warps so the one-CTA SM87 residency
// exposes all 16 resident warps.
//
// The production launcher takes both logical_token_count and
// launch_token_count.  The latter must be exactly ceil128(logical M), matching
// launch_sm87_a4_quantize_bf16_k512_cuda.  That producer is the sole padding
// authority: it emits zero A4 codes and BF16-one scales for every padded row.
// Callers must never manufacture a padded extent without that producer.
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16TileM = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16TileN = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16WarpTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16WarpTileN = 32U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16WarpRows = 4U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16WarpColumns = 4U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16Threads = 512U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16Warps = 16U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16K64PerStage = 4U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16Stages = 4U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16StageBytes = 32U * 1'024U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16ScaleSlotBytes = 512U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16ScaleSlots = 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16CtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16MaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes =
        kSm87A4W4DownK512M128N128Pairring16Stages *
            kSm87A4W4DownK512M128N128Pairring16StageBytes +
        kSm87A4W4DownK512M128N128Pairring16ScaleSlots *
            kSm87A4W4DownK512M128N128Pairring16ScaleSlotBytes;

// Default-off L2 scheduling sibling.  It does not change the M128N128
// compute cell.  Exactly sixteen CTAs form a 4-M by 4-N macro-wave; complete
// 4-M macro rows are exhausted first, then the final partial M macro row.
inline constexpr unsigned int
    kSm87A4W4DownK512M128N128Pairring16L2MacroM = 4U;
inline constexpr unsigned int
    kSm87A4W4DownK512M128N128Pairring16L2MacroN = 4U;
inline constexpr unsigned int
    kSm87A4W4DownK512M128N128Pairring16L2Grid = 16U;

struct Sm87A4W4DownK512M128N128Pairring16L2Work final {
  unsigned int m_tile{};
  unsigned int n_tile{};
  bool valid{};
};

[[nodiscard]] Q3X_SM87_A4W4_DOWN_PAIRRING16_HOST_DEVICE constexpr bool
sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_topology(
    const unsigned int m_tile_count,
    const unsigned int n_tile_count) noexcept {
  return m_tile_count != 0U && n_tile_count != 0U &&
         n_tile_count %
                 kSm87A4W4DownK512M128N128Pairring16L2MacroN ==
             0U;
}

[[nodiscard]] Q3X_SM87_A4W4_DOWN_PAIRRING16_HOST_DEVICE constexpr
    unsigned int
sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_wave_count(
    const unsigned int m_tile_count,
    const unsigned int n_tile_count) noexcept {
  if (!sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_topology(
          m_tile_count, n_tile_count)) {
    return 0U;
  }
  const unsigned int m_macros =
      m_tile_count /
          kSm87A4W4DownK512M128N128Pairring16L2MacroM +
      (m_tile_count %
                   kSm87A4W4DownK512M128N128Pairring16L2MacroM !=
               0U
           ? 1U
           : 0U);
  return m_macros *
         (n_tile_count /
          kSm87A4W4DownK512M128N128Pairring16L2MacroN);
}

[[nodiscard]] Q3X_SM87_A4W4_DOWN_PAIRRING16_HOST_DEVICE constexpr
    Sm87A4W4DownK512M128N128Pairring16L2Work
sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_work(
    const unsigned int block,
    const unsigned int wave,
    const unsigned int m_tile_count,
    const unsigned int n_tile_count) noexcept {
  if (block >=
          kSm87A4W4DownK512M128N128Pairring16L2Grid ||
      !sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_topology(
          m_tile_count, n_tile_count)) {
    return {};
  }
  const unsigned int n_macro_count =
      n_tile_count /
      kSm87A4W4DownK512M128N128Pairring16L2MacroN;
  const unsigned int full_m_macro_count =
      m_tile_count /
      kSm87A4W4DownK512M128N128Pairring16L2MacroM;
  const unsigned int full_wave_count =
      full_m_macro_count * n_macro_count;
  const unsigned int local_m =
      block % kSm87A4W4DownK512M128N128Pairring16L2MacroM;
  const unsigned int local_n =
      block / kSm87A4W4DownK512M128N128Pairring16L2MacroM;

  if (wave < full_wave_count) {
    const unsigned int m_macro = wave / n_macro_count;
    const unsigned int n_macro = wave - m_macro * n_macro_count;
    return {m_macro *
                    kSm87A4W4DownK512M128N128Pairring16L2MacroM +
                local_m,
            n_macro *
                    kSm87A4W4DownK512M128N128Pairring16L2MacroN +
                local_n,
            true};
  }

  const unsigned int tail_m =
      m_tile_count %
      kSm87A4W4DownK512M128N128Pairring16L2MacroM;
  const unsigned int tail_wave = wave - full_wave_count;
  if (tail_m == 0U || tail_wave >= n_macro_count ||
      local_m >= tail_m) {
    return {};
  }
  return {full_m_macro_count *
                  kSm87A4W4DownK512M128N128Pairring16L2MacroM +
              local_m,
          tail_wave *
                  kSm87A4W4DownK512M128N128Pairring16L2MacroN +
              local_n,
          true};
}

static_assert(kSm87A4W4DownK512M128N128Pairring16Threads ==
              32U * kSm87A4W4DownK512M128N128Pairring16Warps);
static_assert(kSm87A4W4DownK512M128N128Pairring16WarpRows *
                      kSm87A4W4DownK512M128N128Pairring16WarpTileM ==
                  kSm87A4W4DownK512M128N128Pairring16TileM);
static_assert(kSm87A4W4DownK512M128N128Pairring16WarpColumns *
                      kSm87A4W4DownK512M128N128Pairring16WarpTileN ==
                  kSm87A4W4DownK512M128N128Pairring16TileN);
static_assert(kSm87A4W4DownK512M128N128Pairring16DynamicSharedBytes ==
              132'096U);

[[nodiscard]] constexpr bool
sm87_a4w4_down_k512_m128n128_16warp_pairring_padding_contract(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count) noexcept {
  return logical_token_count != 0U &&
         launch_token_count ==
             sm87_a4w4_prefill_k512_launch_token_count(
                 logical_token_count) &&
         launch_token_count >= kSm87A4W4DownK512MinimumTokenCount &&
         launch_token_count <= kSm87A4W4DownK512MaximumTokenCount;
}

using Sm87A4W4DownK512M128N128Pairring16Resources =
    Sm87A4W4DownK512Resources;

[[nodiscard]] int
query_sm87_a4w4_down_k512_m128n128_16warp_pairring_resources_cuda(
    Sm87A4W4DownK512M128N128Pairring16Resources* resources) noexcept;

[[nodiscard]] int
query_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_resources_cuda(
    Sm87A4W4DownK512M128N128Pairring16Resources* resources) noexcept;

// Production entry point.  `packed_a` and `a_k512_scales_bf16` must be the
// complete ceil128 publication produced from `logical_token_count` by the
// authenticated K512 activation quantizer.
[[nodiscard]] int
launch_sm87_a4w4_down_k512_m128n128_16warp_pairring_bf16_cuda(
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

// Default-off exact-model sibling.  The launcher always emits grid=16 and
// admits only the Qwen3.6-27B Down N/K topology with Ntiles divisible by 4.
[[nodiscard]] int
launch_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_bf16_cuda(
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

// Correctness-only complete-cell launcher.  Synthetic inputs through this
// entry point are never a performance authority.
[[nodiscard]] int
launch_sm87_a4w4_down_k512_m128n128_16warp_pairring_test_bf16_cuda(
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

// Correctness-only complete-cell macro-wave launcher.  It is fixed at
// grid=16 and requires Ntiles%4==0; synthetic timings are not authoritative.
[[nodiscard]] int
launch_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_test_bf16_cuda(
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

#if defined(__CUDACC__)
extern "C" __global__ void
q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_kernel(
    const std::uint8_t* packed_a,
    const std::uint16_t* a_k512_scales_bf16,
    const std::uint8_t* packed_b,
    const std::uint16_t* b_k512_scales_bf16,
    unsigned int k512_group_count,
    unsigned int physical_k64_group_count,
    std::uint16_t* output_bf16,
    unsigned int output_row_stride_elements,
    unsigned int m_tile_count,
    unsigned int work_tile_count);

extern "C" __global__ void
q3x_sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_kernel(
    const std::uint8_t* packed_a,
    const std::uint16_t* a_k512_scales_bf16,
    const std::uint8_t* packed_b,
    const std::uint16_t* b_k512_scales_bf16,
    unsigned int k512_group_count,
    unsigned int physical_k64_group_count,
    std::uint16_t* output_bf16,
    unsigned int output_row_stride_elements,
    unsigned int m_tile_count,
    unsigned int n_tile_count);
#endif

static_assert(
    sm87_a4w4_down_k512_m128n128_16warp_pairring_padding_contract(
        1U, 128U));
static_assert(
    sm87_a4w4_down_k512_m128n128_16warp_pairring_padding_contract(
        1'853U, 1'920U));
static_assert(
    !sm87_a4w4_down_k512_m128n128_16warp_pairring_padding_contract(
        1'853U, 1'853U));
static_assert(
    kSm87A4W4DownK512M128N128Pairring16L2Grid ==
    kSm87A4W4DownK512M128N128Pairring16L2MacroM *
        kSm87A4W4DownK512M128N128Pairring16L2MacroN);
static_assert(
    sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_topology(
        15U, 40U));
static_assert(
    sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_wave_count(
        15U, 40U) == 40U);
static_assert(
    !sm87_a4w4_down_k512_m128n128_16warp_pairring_l2_macro4x4_topology(
        15U, 39U));

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_DOWN_PAIRRING16_HOST_DEVICE
