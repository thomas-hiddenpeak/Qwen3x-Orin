#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off full-residency Gate+Up experiment over the authenticated v2
// fragment-native payload.  One 512-thread CTA owns M64 x N128.  Warp w owns
// M64 x N8 and keeps both Gate and Up accumulators until the same-warp SwiGLU
// epilogue.  The two adjacent N64 payload blocks are therefore consumed by
// one CTA while the M64 activation ring is loaded only once.
//
// This preserves the exact K512 accumulation and BF16 output boundary of the
// existing M64N64 consumer.  It changes neither payload bytes nor scale order.
// The structural target is the incumbent macrocell's 16 resident warps while
// removing its B shared-memory round trip.
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileN = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaWarpN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaPersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaSharedBytes =
        kSm87A4W4GateUpK512FragmentNativeSharedBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaModelIntermediate =
        17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaModelInput = 5'120U;

[[nodiscard]] constexpr Sm87A4W4GateUpK512FragmentNativePlan
sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count) noexcept {
  if (token_count == 0U ||
      token_count %
              kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileM !=
          0U ||
      intermediate_size == 0U || intermediate_size % 64U != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpK512FragmentNativeScaleK != 0U ||
      n_start %
              kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileN !=
          0U ||
      n_count == 0U ||
      n_count %
              kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileN !=
          0U ||
      n_start > intermediate_size ||
      n_count > intermediate_size - n_start) {
    return {};
  }
  const std::size_t m_tiles =
      token_count /
      kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileM;
  const std::size_t n_tiles =
      n_count /
      kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileN;
  if (!sm87_a4w4_gateup_k512_fragment_native_product_fits(
          m_tiles, n_tiles)) {
    return {};
  }
  return {token_count,
          intermediate_size,
          input_size,
          n_start,
          n_count,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4GateUpK512FragmentNativeScaleK,
          input_size / kSm87A4W4GateUpK512FragmentNativePhysicalK,
          m_tiles * n_tiles,
          m_tiles * n_tiles <
                  kSm87A4W4GateUpK512FragmentNativeM64N1281CtaPersistentCtas
              ? m_tiles * n_tiles
              : kSm87A4W4GateUpK512FragmentNativeM64N1281CtaPersistentCtas};
}

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_is_model_plan(
    const Sm87A4W4GateUpK512FragmentNativePlan& plan) noexcept {
  const bool model_window =
      (plan.n_start == 0U && plan.n_count == 12'288U) ||
      (plan.n_start == 12'288U && plan.n_count == 5'120U);
  return plan.launch_ctas != 0U &&
         plan.intermediate_size ==
             kSm87A4W4GateUpK512FragmentNativeM64N1281CtaModelIntermediate &&
         plan.input_size ==
             kSm87A4W4GateUpK512FragmentNativeM64N1281CtaModelInput &&
         model_window;
}

using Sm87A4W4GateUpK512FragmentNativeM64N1281CtaResources =
    Sm87A4W4GateUpK512FragmentNativeResources;

[[nodiscard]] int
query_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_resources_cuda(
    Sm87A4W4GateUpK512FragmentNativeM64N1281CtaResources* resources) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* paired_b_codes,
    std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* paired_b_scales_bf16,
    std::size_t paired_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* paired_b_codes,
    std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* paired_b_scales_bf16,
    std::size_t paired_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

static_assert(
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaThreads ==
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaWarps * 32U);
static_assert(
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaTileN ==
    kSm87A4W4GateUpK512FragmentNativeM64N1281CtaWarps *
        kSm87A4W4GateUpK512FragmentNativeM64N1281CtaWarpN);
static_assert(
    sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
        1'920U, 17'408U, 5'120U, 0U, 12'288U)
        .launch_ctas == 16U);
static_assert(
    sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
        1'920U, 17'408U, 5'120U, 12'288U, 5'120U)
        .launch_ctas == 16U);

}  // namespace q3x::kernels
