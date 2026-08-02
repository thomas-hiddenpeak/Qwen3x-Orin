#pragma once

#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"

#include <cstddef>
#include <cstdint>

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

}  // namespace q3x::kernels
