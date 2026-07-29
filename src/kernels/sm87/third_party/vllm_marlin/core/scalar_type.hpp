/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * This file is a dependency-free subset of vLLM's csrc/core/scalar_type.hpp.
 * It preserves the compile-time identifiers consumed by the vendored Marlin
 * templates while deliberately omitting the PyTorch-facing validation and
 * string/variant helpers.
 *
 * Copyright contributors to the vLLM project
 */
#pragma once

#include <cstdint>

namespace vllm {

class ScalarType {
 public:
  enum NanRepr : std::uint8_t {
    NAN_NONE = 0,
    NAN_IEEE_754 = 1,
    NAN_EXTD_RANGE_MAX_MIN = 2,
  };

  using Id = std::int64_t;

  constexpr ScalarType(const std::uint8_t exponent,
                       const std::uint8_t mantissa,
                       const bool is_signed,
                       const std::int32_t bias,
                       const bool finite_values_only = false,
                       const NanRepr nan_repr = NAN_IEEE_754)
      : exponent_(exponent),
        mantissa_(mantissa),
        signed_(is_signed),
        bias_(bias),
        finite_values_only_(finite_values_only),
        nan_repr_(nan_repr) {}

  [[nodiscard]] static constexpr ScalarType int_(
      const std::uint8_t size_bits, const std::int32_t bias = 0) {
    return ScalarType(0U, static_cast<std::uint8_t>(size_bits - 1U), true,
                      bias);
  }

  [[nodiscard]] static constexpr ScalarType uint(
      const std::uint8_t size_bits, const std::int32_t bias = 0) {
    return ScalarType(0U, size_bits, false, bias);
  }

  [[nodiscard]] static constexpr ScalarType float_IEEE754(
      const std::uint8_t exponent, const std::uint8_t mantissa) {
    return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
  }

  [[nodiscard]] static constexpr ScalarType float_(
      const std::uint8_t exponent, const std::uint8_t mantissa,
      const bool finite_values_only, const NanRepr nan_repr) {
    return ScalarType(exponent, mantissa, true, 0, finite_values_only,
                      nan_repr);
  }

  [[nodiscard]] constexpr Id id() const {
    return static_cast<Id>(exponent_) |
           (static_cast<Id>(mantissa_) << 8U) |
           (static_cast<Id>(signed_) << 16U) |
           ((static_cast<Id>(static_cast<std::uint32_t>(bias_)) &
             0xFFFF'FFFFLL)
            << 17U) |
           (static_cast<Id>(finite_values_only_) << 49U) |
           (static_cast<Id>(nan_repr_) << 50U);
  }

  [[nodiscard]] static constexpr ScalarType from_id(const Id id) {
    return ScalarType(
        static_cast<std::uint8_t>(id & 0xFFLL),
        static_cast<std::uint8_t>((id >> 8U) & 0xFFLL),
        static_cast<bool>((id >> 16U) & 0x1LL),
        static_cast<std::int32_t>((id >> 17U) & 0xFFFF'FFFFLL),
        static_cast<bool>((id >> 49U) & 0x1LL),
        static_cast<NanRepr>((id >> 50U) & 0xFFLL));
  }

  [[nodiscard]] constexpr std::int64_t size_bits() const {
    return static_cast<std::int64_t>(mantissa_) +
           static_cast<std::int64_t>(exponent_) +
           static_cast<std::int64_t>(signed_);
  }

  [[nodiscard]] constexpr bool operator==(const ScalarType& other) const {
    return exponent_ == other.exponent_ && mantissa_ == other.mantissa_ &&
           signed_ == other.signed_ && bias_ == other.bias_ &&
           finite_values_only_ == other.finite_values_only_ &&
           nan_repr_ == other.nan_repr_;
  }

 private:
  std::uint8_t exponent_;
  std::uint8_t mantissa_;
  bool signed_;
  std::int32_t bias_;
  bool finite_values_only_;
  NanRepr nan_repr_;
};

using ScalarTypeId = ScalarType::Id;

inline constexpr auto kS4 = ScalarType::int_(4U);
inline constexpr auto kU4 = ScalarType::uint(4U);
inline constexpr auto kU4B8 = ScalarType::uint(4U, 8);
inline constexpr auto kS8 = ScalarType::int_(8U);
inline constexpr auto kU8 = ScalarType::uint(8U);
inline constexpr auto kU8B128 = ScalarType::uint(8U, 128);
inline constexpr auto kFE2M1f =
    ScalarType::float_(2U, 1U, true, ScalarType::NAN_NONE);
inline constexpr auto kFE4M3fn = ScalarType::float_(
    4U, 3U, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
inline constexpr auto kFE8M0fnu = ScalarType(
    8U, 0U, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
inline constexpr auto kFE8M7 = ScalarType::float_IEEE754(8U, 7U);
inline constexpr auto kFE5M10 = ScalarType::float_IEEE754(5U, 10U);
inline constexpr auto kFloat16 = kFE5M10;
inline constexpr auto kBFloat16 = kFE8M7;

static_assert(kS4.id() == 1'125'899'906'908'928LL);
static_assert(kU4.id() == 1'125'899'906'843'648LL);
static_assert(kU4B8.id() == 1'125'899'907'892'224LL);
static_assert(kS8.id() == 1'125'899'906'909'952LL);
static_assert(kU8.id() == 1'125'899'906'844'672LL);
static_assert(kU8B128.id() == 1'125'899'923'621'888LL);
static_assert(kFE2M1f.id() == 562'949'953'487'106LL);
static_assert(kFE4M3fn.id() == 2'814'749'767'172'868LL);
static_assert(kFE8M0fnu.id() == 2'814'749'767'106'568LL);
static_assert(kFloat16.id() == 1'125'899'906'910'725LL);
static_assert(kBFloat16.id() == 1'125'899'906'909'960LL);

}  // namespace vllm
