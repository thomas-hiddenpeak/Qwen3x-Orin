#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace q3x::kernels::macrofeed_v3_nvfp4_detail {

// MacroFeed-v3 retains the constituent's original finite-terminal E4M3
// interpretation.  In particular, magnitude 0x7f remains 480 rather than
// being canonicalized to NaN.  The model payload is already authenticated at
// startup; this conversion only removes scalar floating-point reconstruction
// from the MMA hot loop.
[[nodiscard]] __device__ __forceinline__ std::uint16_t
decode_e4m3fn_finite_terminal_to_bf16_bits(
    const std::uint8_t code) noexcept {
  const std::uint16_t sign =
      static_cast<std::uint16_t>(code & 0x80U) << 8U;
  const std::uint16_t magnitude =
      static_cast<std::uint16_t>(code & 0x7fU);
  const std::uint16_t exponent = magnitude >> 3U;
  const std::uint16_t mantissa = magnitude & 0x07U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return sign;
    }
    const std::uint16_t leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const std::uint16_t bf16_exponent = 118U + leading;
    const std::uint16_t bf16_mantissa =
        (mantissa - (1U << leading)) << (7U - leading);
    return static_cast<std::uint16_t>(
        sign | (bf16_exponent << 7U) | bf16_mantissa);
  }
  return static_cast<std::uint16_t>(
      sign | ((120U + exponent) << 7U) | (mantissa << 4U));
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
multiply_bf16x2_bits(const std::uint32_t value_bits,
                     const std::uint16_t scale_bits) noexcept {
  const __nv_bfloat162_raw value_raw{
      static_cast<std::uint16_t>(value_bits),
      static_cast<std::uint16_t>(value_bits >> 16U)};
  const __nv_bfloat162_raw scale_raw{scale_bits, scale_bits};
  const __nv_bfloat162 values(value_raw);
  const __nv_bfloat162 scales(scale_raw);
  const __nv_bfloat162_raw result_raw =
      static_cast<__nv_bfloat162_raw>(__hmul2_rn(values, scales));
  return static_cast<std::uint32_t>(result_raw.x) |
         (static_cast<std::uint32_t>(result_raw.y) << 16U);
}

// Decode the canonical persisted [K0,K8,K1,K9] nibble order directly into
// the m16n8k16.col register order [K0,K1] / [K8,K9].  PRMT constructs exact
// E2M1 BF16 values, and one packed BF16 multiply applies the shared block
// scale to each register pair.  This is mathematically identical to the
// former scalar float decode followed by BF16-RNE.
[[nodiscard]] __device__ __forceinline__ uint2
decode_nvfp4x4_to_bf16x4(const std::uint16_t packed,
                         const std::uint8_t encoded_scale) noexcept {
  constexpr std::uint32_t kLowBytes0To3 = 0xc080'0000U;
  constexpr std::uint32_t kLowBytes4To7 = 0xc080'4000U;
  constexpr std::uint32_t kHighBytes0To3 = 0x3f3f'3f00U;
  constexpr std::uint32_t kHighBytes4To7 = 0x4040'4040U;
  constexpr std::uint32_t kMagnitudeSelectorMask = 0x7777U;
  constexpr std::uint32_t kSignByteMask = 0x8080'8080U;
  constexpr std::uint32_t kFirstFourSignSelector = 0xd9c8U;
  // `packed` persists [K0,K8,K1,K9].  PRMT selectors are written from the
  // least-significant output byte upward, so these produce BF16 pairs
  // [K0,K1] and [K8,K9], respectively.
  constexpr std::uint32_t kFirstPairInterleave = 0x6240U;
  constexpr std::uint32_t kSecondPairInterleave = 0x7351U;

  const std::uint32_t packed32 = static_cast<std::uint32_t>(packed);
  const std::uint32_t selector = packed32 & kMagnitudeSelectorMask;
  const std::uint32_t low =
      __byte_perm(kLowBytes0To3, kLowBytes4To7, selector);
  const std::uint32_t signs =
      __byte_perm(packed32 << 4U, packed32, kFirstFourSignSelector) &
      kSignByteMask;
  const std::uint32_t high =
      __byte_perm(kHighBytes0To3, kHighBytes4To7, selector) | signs;
  const std::uint16_t scale_bits =
      decode_e4m3fn_finite_terminal_to_bf16_bits(encoded_scale);

  uint2 result{};
  result.x = multiply_bf16x2_bits(
      __byte_perm(low, high, kFirstPairInterleave), scale_bits);
  result.y = multiply_bf16x2_bits(
      __byte_perm(low, high, kSecondPairInterleave), scale_bits);
  return result;
}

}  // namespace q3x::kernels::macrofeed_v3_nvfp4_detail
