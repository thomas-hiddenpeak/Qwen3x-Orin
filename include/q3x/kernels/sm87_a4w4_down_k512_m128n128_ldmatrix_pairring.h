#pragma once

#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off, shape-specific Down candidate over the authenticated K512 v1
// payload.  It deliberately preserves the incumbent M128N128 CTA, 256-thread
// 4x2 M32N64 warp ownership, four K256 shared stages, K512 S32 numerical
// boundary, packed-code layout, scale layout, and public launch contract.
//
// The two isolated structural changes are:
//
//  * each K64 plane loads its two A fragments once with ldmatrix.x4 and its
//    eight B fragments with ldmatrix.x2 before issuing 16 MMAs; and
//  * the four K256 slots are treated as two complete K512 pairs.  While pair
//    p is consumed, pair p^1 is asynchronously filled, so one inter-group CTA
//    barrier both releases p and publishes p^1.
//
// Merely linking this file never selects it in the runtime.
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128LdmatrixPairringLdmatrixX4PerGroup = 16U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128LdmatrixPairringLdmatrixX2PerGroup = 64U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128LdmatrixPairringImmaPerGroup = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512M128N128LdmatrixPairringInitialBarriers = 1U;

using Sm87A4W4DownK512M128N128LdmatrixPairringResources =
    Sm87A4W4DownK512Resources;

[[nodiscard]] int
query_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_resources_cuda(
    Sm87A4W4DownK512M128N128LdmatrixPairringResources* resources) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_bf16_cuda(
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

// Correctness-only complete-cell launcher.  Synthetic inputs through this
// entry point are never a performance authority.
[[nodiscard]] int
launch_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_test_bf16_cuda(
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

static_assert(kSm87A4W4DownK512TileM == 128U);
static_assert(kSm87A4W4DownK512TileN == 128U);
static_assert(kSm87A4W4DownK512Threads == 256U);
static_assert(kSm87A4W4DownK512Warps == 8U);
static_assert(kSm87A4W4DownK512Stages == 4U);
static_assert(kSm87A4W4DownK512StagesPerScale == 2U);
static_assert(kSm87A4W4DownK512DynamicSharedBytes == 128U * 1'024U);
static_assert(kSm87A4W4DownK512MaximumRegisters == 255U);

}  // namespace q3x::kernels
