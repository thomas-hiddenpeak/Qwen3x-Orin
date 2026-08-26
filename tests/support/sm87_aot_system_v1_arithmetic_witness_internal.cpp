#include "sm87_aot_system_v1_arithmetic_witness_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace q3x::test::sm87_aot_arithmetic_witness {
namespace {

constexpr std::uint64_t kBf16AbLogicalProjectionOps =
    2ULL * kP40Tokens * 5'120ULL * 96ULL *
    runtime::kSm87AotPrefillSystemGdnLayerCount;
constexpr std::size_t kPlaneWords = 5U;

struct DecodedBf16 {
  bool negative = false;
  bool finite = true;
  bool zero = true;
  bool subnormal = false;
  bool infinity = false;
  bool nan = false;
  std::uint16_t significand = 0U;
  int unbiased_exponent = 0;
  int unit_exponent = 0;
  int lowest_set_exponent = 0;
  int highest_set_exponent = 0;
  unsigned int trailing_zeros = 0U;
};

[[nodiscard]] unsigned int count_trailing_zeros(
    const std::uint16_t value) noexcept {
  if (value == 0U) {
    return 0U;
  }
  unsigned int count = 0U;
  std::uint16_t remaining = value;
  while ((remaining & 1U) == 0U) {
    ++count;
    remaining = static_cast<std::uint16_t>(remaining >> 1U);
  }
  return count;
}

[[nodiscard]] unsigned int highest_set_bit(
    const std::uint16_t value) noexcept {
  unsigned int result = 0U;
  std::uint16_t remaining = value;
  while (remaining > 1U) {
    ++result;
    remaining = static_cast<std::uint16_t>(remaining >> 1U);
  }
  return result;
}

[[nodiscard]] unsigned int population_count(std::uint64_t value) noexcept {
  unsigned int result = 0U;
  while (value != 0U) {
    value &= value - 1U;
    ++result;
  }
  return result;
}

[[nodiscard]] DecodedBf16 decode_bf16(const std::uint16_t bits) noexcept {
  DecodedBf16 result;
  result.negative = (bits & 0x8000U) != 0U;
  const std::uint16_t exponent = static_cast<std::uint16_t>(
      (bits >> 7U) & 0xffU);
  const std::uint16_t fraction = static_cast<std::uint16_t>(bits & 0x7fU);
  if (exponent == 0xffU) {
    result.finite = false;
    result.zero = false;
    result.infinity = fraction == 0U;
    result.nan = fraction != 0U;
    return result;
  }
  if (exponent == 0U && fraction == 0U) {
    return result;
  }

  result.zero = false;
  result.subnormal = exponent == 0U;
  result.significand = result.subnormal
                           ? fraction
                           : static_cast<std::uint16_t>(0x80U | fraction);
  result.unbiased_exponent =
      result.subnormal ? -126 : static_cast<int>(exponent) - 127;
  // A normal BF16 value is significand * 2^(unbiased_exponent - 7).
  // A subnormal BF16 value is fraction * 2^-133.
  result.unit_exponent =
      result.subnormal ? -133 : result.unbiased_exponent - 7;
  result.trailing_zeros = count_trailing_zeros(result.significand);
  result.lowest_set_exponent =
      result.unit_exponent + static_cast<int>(result.trailing_zeros);
  result.highest_set_exponent =
      result.unit_exponent + static_cast<int>(highest_set_bit(
                                 result.significand));
  return result;
}

[[nodiscard]] bool checked_add(std::uint64_t& target,
                               const std::uint64_t value) noexcept {
  if (value > std::numeric_limits<std::uint64_t>::max() - target) {
    return false;
  }
  target += value;
  return true;
}

[[nodiscard]] bool checked_increment(std::uint64_t& target) noexcept {
  return checked_add(target, 1U);
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left,
                                    const std::uint64_t right,
                                    std::uint64_t& result) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] bool checked_add_size(std::size_t& target,
                                    const std::size_t value) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - target) {
    return false;
  }
  target += value;
  return true;
}

[[nodiscard]] bool checked_multiply_size(const std::size_t left,
                                         const std::size_t right,
                                         std::size_t& result) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] bool digest_is_nonzero(
    const std::array<std::uint8_t, 32U>& digest) noexcept {
  return std::any_of(digest.begin(), digest.end(),
                     [](const std::uint8_t value) { return value != 0U; });
}

[[nodiscard]] std::uint64_t payload_digest_layer_mask(
    const OperandDomainSummary& summary) noexcept {
  std::uint64_t result = 0U;
  for (std::size_t layer = 0U; layer < summary.layer_payload_sha256.size();
       ++layer) {
    if (digest_is_nonzero(summary.layer_payload_sha256[layer])) {
      result |= 1ULL << layer;
    }
  }
  return result;
}

[[nodiscard]] bool update_max(std::uint32_t& target,
                              const std::size_t value) noexcept {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  target = std::max(target, static_cast<std::uint32_t>(value));
  return true;
}

using PlaneBits = std::array<std::uint64_t, kPlaneWords>;

[[nodiscard]] int compare_plane_bits(const PlaneBits& left,
                                     const PlaneBits& right) noexcept {
  for (std::size_t index = kPlaneWords; index-- > 0U;) {
    if (left[index] < right[index]) {
      return -1;
    }
    if (left[index] > right[index]) {
      return 1;
    }
  }
  return 0;
}

[[nodiscard]] PlaneBits signed_limb_magnitude_bound(
    const std::size_t limbs, const unsigned int payload_bits,
    const bool negative) noexcept {
  PlaneBits result{};
  if (limbs == 0U || payload_bits == 0U) {
    return result;
  }
  if (!negative) {
    const std::size_t bit_count = limbs * payload_bits;
    for (std::size_t bit = 0U; bit < bit_count; ++bit) {
      result[bit / 64U] |= 1ULL << (bit % 64U);
    }
    return result;
  }
  // With radix b=2^payload_bits and signed digits [-b,b-1], L limbs
  // cover negative magnitudes through b+b^2+...+b^L exactly.
  for (std::size_t limb = 1U; limb <= limbs; ++limb) {
    const std::size_t bit = limb * payload_bits;
    if (bit < kPlaneWords * 64U) {
      result[bit / 64U] |= 1ULL << (bit % 64U);
    }
  }
  return result;
}

[[nodiscard]] std::size_t minimum_signed_limb_count(
    const PlaneBits& magnitude, const bool negative,
    const unsigned int payload_bits, const std::size_t maximum_limbs) noexcept {
  for (std::size_t limbs = 1U; limbs <= maximum_limbs; ++limbs) {
    if (compare_plane_bits(
            magnitude,
            signed_limb_magnitude_bound(limbs, payload_bits, negative)) <= 0) {
      return limbs;
    }
  }
  return maximum_limbs + 1U;
}

[[nodiscard]] PlaneBits shifted_significand(const DecodedBf16& value,
                                            const int minimum_exponent) {
  PlaneBits result{};
  if (!value.finite || value.zero) {
    return result;
  }
  const int shift = value.unit_exponent - minimum_exponent;
  for (unsigned int bit = 0U; bit < 8U; ++bit) {
    if ((value.significand & (1U << bit)) == 0U) {
      continue;
    }
    const int plane = shift + static_cast<int>(bit);
    if (plane < 0 || plane >= static_cast<int>(kPlaneWords * 64U)) {
      return {};
    }
    result[static_cast<std::size_t>(plane) / 64U] |=
        1ULL << (static_cast<unsigned int>(plane) % 64U);
  }
  return result;
}

[[nodiscard]] bool analyze_cell(const std::uint16_t* bits,
                                const std::size_t rows,
                                const std::size_t columns,
                                const std::size_t row_stride,
                                CellDomainSummary& output) noexcept {
  if (bits == nullptr || rows == 0U || columns == 0U ||
      columns % 4U != 0U || row_stride < columns) {
    return false;
  }

  int minimum_unbiased_exponent = std::numeric_limits<int>::max();
  int maximum_unbiased_exponent = std::numeric_limits<int>::min();
  int minimum_set_exponent = std::numeric_limits<int>::max();
  int maximum_set_exponent = std::numeric_limits<int>::min();
  std::uint64_t finite_nonzero = 0U;
  std::uint64_t zero = 0U;
  std::uint64_t negative_zero = 0U;
  std::uint64_t subnormal = 0U;
  std::uint64_t infinity = 0U;
  std::uint64_t nan = 0U;
  std::array<std::uint64_t, kTrailingZeroHistogramSize> trailing{};

  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const DecodedBf16 value = decode_bf16(bits[row * row_stride + column]);
      if (value.nan) {
        ++nan;
      } else if (value.infinity) {
        ++infinity;
      } else if (value.zero) {
        ++zero;
        negative_zero += value.negative ? 1U : 0U;
      } else {
        ++finite_nonzero;
        subnormal += value.subnormal ? 1U : 0U;
        minimum_unbiased_exponent =
            std::min(minimum_unbiased_exponent, value.unbiased_exponent);
        maximum_unbiased_exponent =
            std::max(maximum_unbiased_exponent, value.unbiased_exponent);
        minimum_set_exponent =
            std::min(minimum_set_exponent, value.lowest_set_exponent);
        maximum_set_exponent =
            std::max(maximum_set_exponent, value.highest_set_exponent);
        if (value.trailing_zeros >= trailing.size()) {
          return false;
        }
        ++trailing[value.trailing_zeros];
      }
    }
  }

  const std::size_t value_count = rows * columns;
  if (value_count > std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  const bool has_special = infinity != 0U || nan != 0U;
  const std::size_t exponent_span =
      finite_nonzero == 0U
          ? 0U
          : static_cast<std::size_t>(maximum_unbiased_exponent -
                                     minimum_unbiased_exponent);
  const std::size_t aligned_span =
      finite_nonzero == 0U
          ? 0U
          : static_cast<std::size_t>(maximum_set_exponent -
                                     minimum_set_exponent + 1);
  std::size_t int8_limbs = 0U;
  std::size_t int4_limbs = 0U;
  if (finite_nonzero != 0U) {
    for (std::size_t row = 0U; row < rows; ++row) {
      for (std::size_t column = 0U; column < columns; ++column) {
        const DecodedBf16 value =
            decode_bf16(bits[row * row_stride + column]);
        if (!value.finite || value.zero) {
          continue;
        }
        const PlaneBits magnitude =
            shifted_significand(value, minimum_set_exponent);
        int8_limbs = std::max(
            int8_limbs,
            minimum_signed_limb_count(
                magnitude, value.negative, 7U,
                kInt8LimbHistogramSize - 1U));
        int4_limbs = std::max(
            int4_limbs,
            minimum_signed_limb_count(
                magnitude, value.negative, 3U,
                kInt4LimbHistogramSize - 1U));
      }
    }
  }
  if (exponent_span >= output.exponent_span_histogram.size() ||
      aligned_span >= output.aligned_bit_span_histogram.size() ||
      int8_limbs >= output.int8_limb_histogram.size() ||
      int4_limbs >= output.int4_limb_histogram.size()) {
    return false;
  }

  std::uint64_t residual_set_bits = 0U;
  std::uint64_t residual_slots = 0U;
  std::uint64_t exact_two_of_four = 0U;
  std::uint64_t occupied_two_of_four = 0U;
  // A single NaN/Inf forces the whole physical cell onto the exact fallback.
  // Do not let the remaining finite lanes contribute direct-representation
  // or bit-plane sparsity evidence for that cell.
  if (aligned_span != 0U && !has_special) {
    if (!checked_multiply(static_cast<std::uint64_t>(value_count),
                          static_cast<std::uint64_t>(aligned_span),
                          residual_slots)) {
      return false;
    }
    for (std::size_t row = 0U; row < rows; ++row) {
      for (std::size_t column = 0U; column < columns; ++column) {
        const DecodedBf16 value =
            decode_bf16(bits[row * row_stride + column]);
        if (value.finite && !value.zero) {
          residual_set_bits += population_count(value.significand);
        }
      }
      for (std::size_t column = 0U; column < columns; column += 4U) {
        std::array<PlaneBits, 4U> planes{};
        for (std::size_t index = 0U; index < planes.size(); ++index) {
          planes[index] = shifted_significand(
              decode_bf16(bits[row * row_stride + column + index]),
              minimum_set_exponent);
        }
        for (std::size_t word = 0U; word < kPlaneWords; ++word) {
          const std::uint64_t a = planes[0U][word];
          const std::uint64_t b = planes[1U][word];
          const std::uint64_t c = planes[2U][word];
          const std::uint64_t d = planes[3U][word];
          const std::uint64_t occupied = a | b | c | d;
          const std::uint64_t at_least_three =
              (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d);
          occupied_two_of_four += population_count(occupied);
          exact_two_of_four += population_count(occupied & ~at_least_three);
        }
      }
    }
  }

  if (!checked_increment(output.cell_count) ||
      !checked_add(output.value_count,
                   static_cast<std::uint64_t>(value_count)) ||
      !checked_add(output.finite_nonzero_count, finite_nonzero) ||
      !checked_add(output.zero_count, zero) ||
      !checked_add(output.negative_zero_count, negative_zero) ||
      !checked_add(output.subnormal_count, subnormal) ||
      !checked_add(output.infinity_count, infinity) ||
      !checked_add(output.nan_count, nan) ||
      (has_special && !checked_increment(output.special_cell_count)) ||
      !update_max(output.maximum_exponent_span, exponent_span) ||
      !update_max(output.maximum_aligned_bit_span, aligned_span) ||
      !update_max(output.maximum_int8_limb_count,
                  has_special ? 0U : int8_limbs) ||
      !update_max(output.maximum_int4_limb_count,
                  has_special ? 0U : int4_limbs) ||
      !checked_add(output.residual_plane_set_bits, residual_set_bits) ||
      !checked_add(output.residual_plane_slots, residual_slots) ||
      !checked_add(output.exact_bit_plane_two_of_four_groups,
                   exact_two_of_four) ||
      !checked_add(output.occupied_bit_plane_two_of_four_groups,
                   occupied_two_of_four) ||
      !checked_increment(output.exponent_span_histogram[exponent_span]) ||
      !checked_increment(output.aligned_bit_span_histogram[aligned_span])) {
    return false;
  }
  // Special-value cells require an explicit exact fallback and are therefore
  // kept out of the direct-limb histograms rather than being double counted.
  if (!has_special &&
      (!checked_increment(output.int8_limb_histogram[int8_limbs]) ||
       !checked_increment(output.int4_limb_histogram[int4_limbs]))) {
    return false;
  }
  for (std::size_t index = 0U; index < trailing.size(); ++index) {
    if (!checked_add(output.significand_trailing_zero_histogram[index],
                     trailing[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionRole projection_role(
    const OperandRole role) noexcept {
  switch (role) {
    case OperandRole::kNvFp4GateUp:
      return kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp;
    case OperandRole::kNvFp4Down:
      return kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
    case OperandRole::kFp8GdnQkvZ:
      return kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
    case OperandRole::kFp8FullQkv:
      return kernels::Sm87TargetAotProjectionRole::kFp8FullQkv;
    case OperandRole::kFp8AttentionOutput:
      return kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput;
    case OperandRole::kInvalid:
    case OperandRole::kCount:
      return kernels::Sm87TargetAotProjectionRole::kInvalid;
  }
  return kernels::Sm87TargetAotProjectionRole::kInvalid;
}

[[nodiscard]] std::uint64_t expected_k16_cells(
    const OperandRole role) noexcept {
  const std::uint64_t instances = expected_role_instances(role);
  const std::uint64_t features = expected_input_features(role);
  return instances * (kP40Tokens / kMmaRows) * (features / kK16);
}

[[nodiscard]] std::uint64_t expected_k64_cells(
    const OperandRole role) noexcept {
  const std::uint64_t instances = expected_role_instances(role);
  const std::uint64_t features = expected_input_features(role);
  return instances * ((kP40Tokens + kPanelRows - 1U) / kPanelRows) *
         (features / kK64);
}

[[nodiscard]] bool finite_nonnegative(const double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool arithmetic_class_valid(
    const ArithmeticClass value) noexcept {
  switch (value) {
    case ArithmeticClass::kK16SignedInt8Limbs:
    case ArithmeticClass::kK16SignedInt4Limbs:
    case ArithmeticClass::kK16ExactBitPlanes:
    case ArithmeticClass::kK64SignedInt8Limbs:
    case ArithmeticClass::kK64SignedInt4Limbs:
    case ArithmeticClass::kK64ExactBitPlanes:
      return true;
    case ArithmeticClass::kInvalid:
      return false;
  }
  return false;
}

[[nodiscard]] bool arithmetic_class_uses_k16(
    const ArithmeticClass value) noexcept {
  return value == ArithmeticClass::kK16SignedInt8Limbs ||
         value == ArithmeticClass::kK16SignedInt4Limbs ||
         value == ArithmeticClass::kK16ExactBitPlanes;
}

[[nodiscard]] bool cost_envelope_valid(
    const CostEnvelope& cost, const CostScope expected_scope,
    const OperandRole expected_role,
    const ArithmeticClass expected_arithmetic_class,
    const std::array<std::uint8_t, 32U>& expected_mapping_sha256) noexcept {
  const bool identity_matches =
      cost.scope == expected_scope && cost.role == expected_role &&
      cost.arithmetic_class == expected_arithmetic_class &&
      digest_is_nonzero(cost.arithmetic_mapping_sha256) &&
      digest_is_nonzero(cost.measurement_receipt_sha256) &&
      (expected_scope == CostScope::kOrderedBf16Ab ||
       cost.arithmetic_mapping_sha256 == expected_mapping_sha256);
  return identity_matches && cost.measurement_identity_authenticated &&
         cost.calibrated_same_sm87 &&
         cost.measured_sample_count != 0U &&
         std::isfinite(cost.optimistic_throughput_ops_per_second) &&
         cost.optimistic_throughput_ops_per_second > 0.0 &&
         std::isfinite(cost.conservative_throughput_ops_per_second) &&
         cost.conservative_throughput_ops_per_second > 0.0 &&
         cost.conservative_throughput_ops_per_second <=
             cost.optimistic_throughput_ops_per_second &&
         finite_nonnegative(cost.fallback_seconds_lower) &&
         finite_nonnegative(cost.fallback_seconds_upper) &&
         cost.fallback_seconds_upper >= cost.fallback_seconds_lower &&
         finite_nonnegative(cost.representation_seconds_lower) &&
         finite_nonnegative(cost.representation_seconds_upper) &&
         cost.representation_seconds_upper >=
             cost.representation_seconds_lower;
}

template <std::size_t Size>
[[nodiscard]] bool merge_histogram(
    std::array<std::uint64_t, Size>& target,
    const std::array<std::uint64_t, Size>& source) noexcept {
  for (std::size_t index = 0U; index < Size; ++index) {
    if (!checked_add(target[index], source[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool merge_cell_summary(CellDomainSummary& target,
                                      const CellDomainSummary& source) noexcept {
  if (!checked_add(target.cell_count, source.cell_count) ||
      !checked_add(target.value_count, source.value_count) ||
      !checked_add(target.finite_nonzero_count,
                   source.finite_nonzero_count) ||
      !checked_add(target.zero_count, source.zero_count) ||
      !checked_add(target.negative_zero_count, source.negative_zero_count) ||
      !checked_add(target.subnormal_count, source.subnormal_count) ||
      !checked_add(target.infinity_count, source.infinity_count) ||
      !checked_add(target.nan_count, source.nan_count) ||
      !checked_add(target.special_cell_count, source.special_cell_count) ||
      !checked_add(target.residual_plane_set_bits,
                   source.residual_plane_set_bits) ||
      !checked_add(target.residual_plane_slots,
                   source.residual_plane_slots) ||
      !checked_add(target.exact_bit_plane_two_of_four_groups,
                   source.exact_bit_plane_two_of_four_groups) ||
      !checked_add(target.occupied_bit_plane_two_of_four_groups,
                   source.occupied_bit_plane_two_of_four_groups) ||
      !merge_histogram(target.exponent_span_histogram,
                       source.exponent_span_histogram) ||
      !merge_histogram(target.aligned_bit_span_histogram,
                       source.aligned_bit_span_histogram) ||
      !merge_histogram(target.int8_limb_histogram,
                       source.int8_limb_histogram) ||
      !merge_histogram(target.int4_limb_histogram,
                       source.int4_limb_histogram) ||
      !merge_histogram(target.significand_trailing_zero_histogram,
                       source.significand_trailing_zero_histogram)) {
    return false;
  }
  target.maximum_exponent_span =
      std::max(target.maximum_exponent_span, source.maximum_exponent_span);
  target.maximum_aligned_bit_span = std::max(
      target.maximum_aligned_bit_span, source.maximum_aligned_bit_span);
  target.maximum_int8_limb_count = std::max(
      target.maximum_int8_limb_count, source.maximum_int8_limb_count);
  target.maximum_int4_limb_count = std::max(
      target.maximum_int4_limb_count, source.maximum_int4_limb_count);
  return true;
}

template <std::size_t Size>
[[nodiscard]] bool histogram_sum_checked(
    const std::array<std::uint64_t, Size>& values,
    std::uint64_t& result) noexcept {
  result = 0U;
  for (const std::uint64_t value : values) {
    if (!checked_add(result, value)) {
      return false;
    }
  }
  return true;
}

template <std::size_t Size>
[[nodiscard]] std::size_t highest_occupied_histogram_bin(
    const std::array<std::uint64_t, Size>& values) noexcept {
  for (std::size_t index = Size; index-- > 0U;) {
    if (values[index] != 0U) {
      return index;
    }
  }
  return 0U;
}

[[nodiscard]] bool validate_cell_summary(
    const CellDomainSummary& cells, const std::uint64_t expected_cells,
    const std::uint64_t expected_values) noexcept {
  std::uint64_t classified_values = 0U;
  std::uint64_t special_values = 0U;
  std::uint64_t exponent_cells = 0U;
  std::uint64_t aligned_cells = 0U;
  std::uint64_t int8_direct_cells = 0U;
  std::uint64_t int4_direct_cells = 0U;
  std::uint64_t trailing_values = 0U;
  std::uint64_t int8_accounted_cells = 0U;
  std::uint64_t int4_accounted_cells = 0U;
  if (!checked_add(classified_values, cells.finite_nonzero_count) ||
      !checked_add(classified_values, cells.zero_count) ||
      !checked_add(classified_values, cells.infinity_count) ||
      !checked_add(classified_values, cells.nan_count) ||
      !checked_add(special_values, cells.infinity_count) ||
      !checked_add(special_values, cells.nan_count) ||
      !histogram_sum_checked(cells.exponent_span_histogram, exponent_cells) ||
      !histogram_sum_checked(cells.aligned_bit_span_histogram, aligned_cells) ||
      !histogram_sum_checked(cells.int8_limb_histogram,
                             int8_direct_cells) ||
      !histogram_sum_checked(cells.int4_limb_histogram,
                             int4_direct_cells) ||
      !histogram_sum_checked(cells.significand_trailing_zero_histogram,
                             trailing_values)) {
    return false;
  }
  int8_accounted_cells = int8_direct_cells;
  int4_accounted_cells = int4_direct_cells;
  if (!checked_add(int8_accounted_cells, cells.special_cell_count) ||
      !checked_add(int4_accounted_cells, cells.special_cell_count)) {
    return false;
  }
  const bool special_consistent =
      (special_values == 0U) == (cells.special_cell_count == 0U) &&
      cells.special_cell_count <= special_values;
  const bool plane_density_consistent =
      cells.residual_plane_set_bits <= cells.residual_plane_slots &&
      cells.exact_bit_plane_two_of_four_groups <=
          cells.occupied_bit_plane_two_of_four_groups &&
      cells.occupied_bit_plane_two_of_four_groups <=
          cells.residual_plane_slots / 4U;
  return cells.cell_count == expected_cells &&
         cells.value_count == expected_values &&
         classified_values == expected_values &&
         cells.negative_zero_count <= cells.zero_count &&
         cells.subnormal_count <= cells.finite_nonzero_count &&
         special_consistent && exponent_cells == expected_cells &&
         aligned_cells == expected_cells &&
         int8_accounted_cells == expected_cells &&
         int4_accounted_cells == expected_cells &&
         trailing_values == cells.finite_nonzero_count &&
         highest_occupied_histogram_bin(cells.exponent_span_histogram) ==
             cells.maximum_exponent_span &&
         highest_occupied_histogram_bin(cells.aligned_bit_span_histogram) ==
             cells.maximum_aligned_bit_span &&
         highest_occupied_histogram_bin(cells.int8_limb_histogram) ==
             cells.maximum_int8_limb_count &&
         highest_occupied_histogram_bin(cells.int4_limb_histogram) ==
             cells.maximum_int4_limb_count &&
         plane_density_consistent;
}

[[nodiscard]] bool same_value_classification(
    const CellDomainSummary& left,
    const CellDomainSummary& right) noexcept {
  return left.value_count == right.value_count &&
         left.finite_nonzero_count == right.finite_nonzero_count &&
         left.zero_count == right.zero_count &&
         left.negative_zero_count == right.negative_zero_count &&
         left.subnormal_count == right.subnormal_count &&
         left.infinity_count == right.infinity_count &&
         left.nan_count == right.nan_count;
}

[[nodiscard]] bool expected_operand_value_count(
    const OperandRole role, std::uint64_t& result) noexcept {
  std::uint64_t intermediate = 0U;
  return checked_multiply(
             static_cast<std::uint64_t>(expected_role_instances(role)),
             static_cast<std::uint64_t>(kP40Tokens), intermediate) &&
         checked_multiply(
             intermediate,
             static_cast<std::uint64_t>(expected_input_features(role)),
             result);
}

[[nodiscard]] OperandDomainSummary synthetic_complete_summary(
    const OperandRole role, const std::size_t limb_count,
    const std::uint64_t special_cells = 0U) {
  OperandDomainSummary result;
  result.role = role;
  result.capture_point = expected_capture_point(role);
  result.expected_rows = kP40Tokens;
  result.input_features = expected_input_features(role);
  result.next_row = kP40Tokens;
  result.panel_count = (kP40Tokens + kPanelRows - 1U) / kPanelRows;
  result.instance_count = expected_role_instances(role);
  result.next_row *= result.instance_count;
  result.panel_count *= result.instance_count;
  result.model_layer_mask = expected_model_layer_mask(role);
  for (std::size_t layer = 0U;
       layer < runtime::kSm87AotPrefillSystemLayerCount; ++layer) {
    if ((result.model_layer_mask & (1ULL << layer)) != 0U) {
      result.layer_payload_sha256[layer][0U] =
          static_cast<std::uint8_t>(layer + 1U);
    }
  }
  result.capture_identity_authenticated = true;
  result.complete = true;
  result.valid = true;
  std::uint64_t value_count = 0U;
  if (!expected_operand_value_count(role, value_count) ||
      special_cells > expected_k64_cells(role) || limb_count == 0U ||
      limb_count >= kInt8LimbHistogramSize ||
      limb_count >= kInt4LimbHistogramSize) {
    result.valid = false;
    result.complete = false;
    return result;
  }
  const std::uint64_t finite_nonzero =
      expected_k16_cells(role) - special_cells;
  const std::uint64_t special_values = special_cells;
  if (finite_nonzero + special_values > value_count) {
    result.valid = false;
    result.complete = false;
    return result;
  }
  const auto populate = [&](CellDomainSummary& cells,
                            const std::uint64_t cell_count) {
    const std::uint64_t direct_cells = cell_count - special_cells;
    cells.cell_count = cell_count;
    cells.value_count = value_count;
    cells.finite_nonzero_count = finite_nonzero;
    cells.zero_count = value_count - finite_nonzero - special_values;
    cells.nan_count = special_values;
    cells.special_cell_count = special_cells;
    cells.maximum_aligned_bit_span = direct_cells == 0U ? 0U : 1U;
    cells.maximum_int8_limb_count =
        direct_cells == 0U ? 0U : static_cast<std::uint32_t>(limb_count);
    cells.maximum_int4_limb_count = cells.maximum_int8_limb_count;
    cells.residual_plane_set_bits = finite_nonzero;
    cells.residual_plane_slots = finite_nonzero;
    cells.exponent_span_histogram[0U] = cell_count;
    cells.aligned_bit_span_histogram[0U] = special_cells;
    cells.aligned_bit_span_histogram[1U] = direct_cells;
    cells.int8_limb_histogram[limb_count] = direct_cells;
    cells.int4_limb_histogram[limb_count] = direct_cells;
    cells.significand_trailing_zero_histogram[7U] = finite_nonzero;
  };
  populate(result.k16, expected_k16_cells(role));
  populate(result.k64, expected_k64_cells(role));
  return result;
}

}  // namespace

Bf16OperandAccumulator::Bf16OperandAccumulator(
    const OperandInstanceIdentity identity, const std::size_t expected_rows,
    const std::size_t input_features) noexcept {
  summary_.role = identity.role;
  summary_.capture_point = identity.capture_point;
  summary_.expected_rows = expected_rows;
  summary_.input_features = input_features;
  const bool layer_in_role =
      identity.model_layer_index <
          runtime::kSm87AotPrefillSystemLayerCount &&
      (expected_model_layer_mask(identity.role) &
       (1ULL << identity.model_layer_index)) != 0U;
  summary_.valid = identity.role != OperandRole::kInvalid &&
                   identity.role != OperandRole::kCount &&
                   identity.capture_point ==
                       expected_capture_point(identity.role) &&
                   layer_in_role && digest_is_nonzero(identity.payload_sha256) &&
                   identity.checkpoint_identity_authenticated &&
                   identity.route_identity_authenticated &&
                   identity.capture_boundary_authenticated &&
                   expected_rows != 0U &&
                   expected_rows % kMmaRows == 0U && input_features != 0U &&
                   input_features % kK64 == 0U;
  if (summary_.valid) {
    summary_.model_layer_mask = 1ULL << identity.model_layer_index;
    summary_.layer_payload_sha256[identity.model_layer_index] =
        identity.payload_sha256;
    summary_.capture_identity_authenticated = true;
  }
  failed_ = !summary_.valid;
}

bool Bf16OperandAccumulator::consume_panel(
    const std::size_t first_row, const std::uint16_t* bits,
    const std::size_t row_count,
    const std::size_t row_stride_elements) noexcept {
  if (failed_ || finalized_ || bits == nullptr || first_row != summary_.next_row ||
      first_row % kPanelRows != 0U || first_row >= summary_.expected_rows ||
      row_stride_elements < summary_.input_features) {
    failed_ = true;
    return false;
  }
  const std::size_t remaining = summary_.expected_rows - first_row;
  const std::size_t expected_panel_rows =
      std::min(kPanelRows, remaining);
  if (row_count != expected_panel_rows || row_count % kMmaRows != 0U) {
    failed_ = true;
    return false;
  }

  for (std::size_t row = 0U; row < row_count; row += kMmaRows) {
    for (std::size_t column = 0U; column < summary_.input_features;
         column += kK16) {
      if (!analyze_cell(bits + row * row_stride_elements + column,
                        kMmaRows, kK16, row_stride_elements, summary_.k16)) {
        failed_ = true;
        return false;
      }
    }
  }
  for (std::size_t column = 0U; column < summary_.input_features;
       column += kK64) {
    if (!analyze_cell(bits + column, row_count, kK64,
                      row_stride_elements, summary_.k64)) {
      failed_ = true;
      return false;
    }
  }
  summary_.next_row += row_count;
  ++summary_.panel_count;
  return true;
}

OperandDomainSummary Bf16OperandAccumulator::finalize() noexcept {
  if (finalized_) {
    failed_ = true;
  }
  finalized_ = true;
  summary_.complete = !failed_ && summary_.next_row == summary_.expected_rows;
  summary_.valid = summary_.valid && summary_.complete;
  summary_.instance_count = summary_.valid ? 1U : 0U;
  return summary_;
}

bool OperandDomainMerger::add(
    const OperandDomainSummary& instance) noexcept {
  const bool single_layer =
      instance.model_layer_mask != 0U &&
      (instance.model_layer_mask & (instance.model_layer_mask - 1U)) == 0U;
  const bool layer_in_role =
      single_layer &&
      (instance.model_layer_mask & expected_model_layer_mask(instance.role)) !=
          0U;
  const bool payload_inventory_matches =
      payload_digest_layer_mask(instance) == instance.model_layer_mask;
  if (failed_ || finalized_ || !instance.valid || !instance.complete ||
      instance.instance_count != 1U ||
      instance.next_row != instance.expected_rows || !layer_in_role ||
      !payload_inventory_matches ||
      !instance.capture_identity_authenticated ||
      instance.role == OperandRole::kInvalid ||
      instance.role == OperandRole::kCount ||
      instance.capture_point != expected_capture_point(instance.role)) {
    failed_ = true;
    summary_.valid = false;
    summary_.complete = false;
    return false;
  }
  if (!started_) {
    summary_ = instance;
    started_ = true;
    return true;
  }
  OperandDomainSummary candidate = summary_;
  if (!candidate.valid || !candidate.complete ||
      candidate.role != instance.role ||
      candidate.capture_point != instance.capture_point ||
      candidate.expected_rows != instance.expected_rows ||
      candidate.input_features != instance.input_features ||
      (candidate.model_layer_mask & instance.model_layer_mask) != 0U ||
      !checked_add_size(candidate.next_row, instance.next_row) ||
      !checked_add_size(candidate.panel_count, instance.panel_count) ||
      !checked_add_size(candidate.instance_count, 1U) ||
      !merge_cell_summary(candidate.k16, instance.k16) ||
      !merge_cell_summary(candidate.k64, instance.k64)) {
    failed_ = true;
    summary_.valid = false;
    summary_.complete = false;
    return false;
  }
  for (std::size_t layer = 0U;
       layer < candidate.layer_payload_sha256.size(); ++layer) {
    if (digest_is_nonzero(instance.layer_payload_sha256[layer])) {
      candidate.layer_payload_sha256[layer] =
          instance.layer_payload_sha256[layer];
    }
  }
  candidate.model_layer_mask |= instance.model_layer_mask;
  candidate.capture_identity_authenticated =
      candidate.capture_identity_authenticated &&
      instance.capture_identity_authenticated;
  summary_ = candidate;
  return true;
}

OperandDomainSummary OperandDomainMerger::finalize() noexcept {
  if (finalized_ || failed_ || !started_) {
    failed_ = true;
    summary_.valid = false;
    summary_.complete = false;
  }
  finalized_ = true;
  return summary_;
}

std::string_view to_string(const OperandRole role) noexcept {
  switch (role) {
    case OperandRole::kInvalid:
      return "invalid";
    case OperandRole::kNvFp4GateUp:
      return "nvfp4_gate_up";
    case OperandRole::kNvFp4Down:
      return "nvfp4_down";
    case OperandRole::kFp8GdnQkvZ:
      return "fp8_gdn_qkv_z";
    case OperandRole::kFp8FullQkv:
      return "fp8_full_qkv";
    case OperandRole::kFp8AttentionOutput:
      return "fp8_attention_output";
    case OperandRole::kCount:
      return "count";
  }
  return "invalid";
}

std::string_view to_string(const Decision decision) noexcept {
  switch (decision) {
    case Decision::kInconclusive:
      return "INCONCLUSIVE";
    case Decision::kReject:
      return "REJECT";
    case Decision::kPass:
      return "PASS";
  }
  return "INCONCLUSIVE";
}

std::string_view to_string(const ArithmeticClass value) noexcept {
  switch (value) {
    case ArithmeticClass::kInvalid:
      return "invalid";
    case ArithmeticClass::kK16SignedInt8Limbs:
      return "k16_signed_int8_limbs";
    case ArithmeticClass::kK16SignedInt4Limbs:
      return "k16_signed_int4_limbs";
    case ArithmeticClass::kK16ExactBitPlanes:
      return "k16_exact_bit_planes";
    case ArithmeticClass::kK64SignedInt8Limbs:
      return "k64_signed_int8_limbs";
    case ArithmeticClass::kK64SignedInt4Limbs:
      return "k64_signed_int4_limbs";
    case ArithmeticClass::kK64ExactBitPlanes:
      return "k64_exact_bit_planes";
  }
  return "invalid";
}

std::string_view to_string(const OperandCapturePoint value) noexcept {
  switch (value) {
    case OperandCapturePoint::kInvalid:
      return "invalid";
    case OperandCapturePoint::kGateUpInputAfterPostAttentionNorm:
      return "gate_up_input_after_post_attention_norm";
    case OperandCapturePoint::kDownInputAfterGatedActivation:
      return "down_input_after_gated_activation";
    case OperandCapturePoint::kGdnQkvZInputAfterInputNorm:
      return "gdn_qkvz_input_after_input_norm";
    case OperandCapturePoint::kFullQkvInputAfterInputNorm:
      return "full_qkv_input_after_input_norm";
    case OperandCapturePoint::kAttentionOutputInputAfterPinnedCore:
      return "attention_output_input_after_pinned_core";
  }
  return "invalid";
}

std::string_view to_string(const CostScope value) noexcept {
  switch (value) {
    case CostScope::kInvalid:
      return "invalid";
    case CostScope::kOrderedBf16Ab:
      return "ordered_bf16_ab";
    case CostScope::kProjectionRole:
      return "projection_role";
  }
  return "invalid";
}

std::size_t expected_role_instances(const OperandRole role) noexcept {
  switch (role) {
    case OperandRole::kNvFp4GateUp:
    case OperandRole::kNvFp4Down:
    case OperandRole::kFp8AttentionOutput:
      return runtime::kSm87AotPrefillSystemLayerCount;
    case OperandRole::kFp8GdnQkvZ:
      return runtime::kSm87AotPrefillSystemGdnLayerCount;
    case OperandRole::kFp8FullQkv:
      return runtime::kSm87AotPrefillSystemFullAttentionLayerCount;
    case OperandRole::kInvalid:
    case OperandRole::kCount:
      return 0U;
  }
  return 0U;
}

std::size_t expected_input_features(const OperandRole role) noexcept {
  const auto plan = kernels::sm87_target_aot_projection_plan(
      projection_role(role), kP40Tokens);
  return plan.valid() ? plan.input_features : 0U;
}

std::size_t expected_partition_count(const OperandRole role) noexcept {
  const auto plan = kernels::sm87_target_aot_projection_plan(
      projection_role(role), kP40Tokens);
  return plan.valid() ? plan.partition_count : 0U;
}

OperandCapturePoint expected_capture_point(const OperandRole role) noexcept {
  switch (role) {
    case OperandRole::kNvFp4GateUp:
      return OperandCapturePoint::kGateUpInputAfterPostAttentionNorm;
    case OperandRole::kNvFp4Down:
      return OperandCapturePoint::kDownInputAfterGatedActivation;
    case OperandRole::kFp8GdnQkvZ:
      return OperandCapturePoint::kGdnQkvZInputAfterInputNorm;
    case OperandRole::kFp8FullQkv:
      return OperandCapturePoint::kFullQkvInputAfterInputNorm;
    case OperandRole::kFp8AttentionOutput:
      return OperandCapturePoint::kAttentionOutputInputAfterPinnedCore;
    case OperandRole::kInvalid:
    case OperandRole::kCount:
      return OperandCapturePoint::kInvalid;
  }
  return OperandCapturePoint::kInvalid;
}

std::uint64_t expected_model_layer_mask(const OperandRole role) noexcept {
  std::uint64_t result = 0U;
  for (std::size_t layer = 0U;
       layer < runtime::kSm87AotPrefillSystemLayerCount; ++layer) {
    const auto schedule = runtime::sm87_aot_prefill_layer_schedule(layer);
    const bool accepted =
        role == OperandRole::kNvFp4GateUp ||
        role == OperandRole::kNvFp4Down ||
        role == OperandRole::kFp8AttentionOutput ||
        (role == OperandRole::kFp8GdnQkvZ &&
         schedule.kind == runtime::Sm87AotPrefillLayerKind::kGdn) ||
        (role == OperandRole::kFp8FullQkv &&
         schedule.kind == runtime::Sm87AotPrefillLayerKind::kFullAttention);
    if (accepted) {
      result |= 1ULL << layer;
    }
  }
  return result;
}

std::uint64_t role_logical_projection_ops(const OperandRole role) noexcept {
  const auto plan = kernels::sm87_target_aot_projection_plan(
      projection_role(role), kP40Tokens);
  const std::uint64_t instances = expected_role_instances(role);
  std::uint64_t result = 0U;
  std::uint64_t intermediate = 0U;
  if (!plan.valid() ||
      !checked_multiply(2U * kP40Tokens,
                        static_cast<std::uint64_t>(plan.input_features),
                        intermediate) ||
      !checked_multiply(intermediate,
                        static_cast<std::uint64_t>(
                            plan.projected_output_features),
                        intermediate) ||
      !checked_multiply(intermediate, instances, result)) {
    return 0U;
  }
  return result;
}

std::uint64_t total_logical_projection_ops() noexcept {
  std::uint64_t result = kBf16AbLogicalProjectionOps;
  for (std::size_t index = 0U; index < kOperandRoleCount; ++index) {
    if (!checked_add(result, role_logical_projection_ops(
                                 static_cast<OperandRole>(index + 1U)))) {
      return 0U;
    }
  }
  return result;
}

// Synthetic branch exerciser only.  It has internal linkage and cannot issue
// a consumable witness decision.  In particular, its caller-populated
// receipts are not an authority boundary.
static DecisionReceipt evaluate_provisional_for_self_test(
    const Assessment& assessment) noexcept {
  DecisionReceipt result;
  result.logical_projection_ops = total_logical_projection_ops();
  result.ordered_bf16_physical_ops = kBf16AbLogicalProjectionOps;
  result.direct_physical_ops = kBf16AbLogicalProjectionOps;

  std::array<bool, kOperandRoleCount> seen{};
  std::array<std::uint64_t, kOperandRoleCount> fallback_cells_by_role{};
  bool all_joint_receipts_valid = true;
  for (const RoleMapping& mapping : assessment.roles) {
    if (mapping.role == OperandRole::kInvalid ||
        mapping.role == OperandRole::kCount) {
      result.issues |= kWitnessIssueInvalidInput;
      continue;
    }
    const std::size_t index = role_index(mapping.role);
    if (index >= seen.size() || seen[index]) {
      result.issues |= kWitnessIssueInvalidInput;
      continue;
    }
    seen[index] = true;
    const CellDomainSummary& cells = mapping.operand.k16;
    std::size_t expected_next_row = 0U;
    std::size_t expected_panel_count = 0U;
    std::uint64_t expected_values = 0U;
    const bool geometry_products_valid =
        checked_multiply_size(kP40Tokens,
                              expected_role_instances(mapping.role),
                              expected_next_row) &&
        checked_multiply_size(
            (kP40Tokens + kPanelRows - 1U) / kPanelRows,
            expected_role_instances(mapping.role), expected_panel_count) &&
        expected_operand_value_count(mapping.role, expected_values);
    const bool layer_inventory_valid =
        mapping.operand.model_layer_mask ==
            expected_model_layer_mask(mapping.role) &&
        mapping.operand.instance_count ==
            expected_role_instances(mapping.role) &&
        payload_digest_layer_mask(mapping.operand) ==
            expected_model_layer_mask(mapping.role);
    const bool numeric_accounting_valid =
        geometry_products_valid &&
        validate_cell_summary(cells, expected_k16_cells(mapping.role),
                              expected_values) &&
        validate_cell_summary(mapping.operand.k64,
                              expected_k64_cells(mapping.role),
                              expected_values) &&
        same_value_classification(cells, mapping.operand.k64);
    const bool geometry_valid =
        mapping.operand.role == mapping.role && mapping.operand.valid &&
        mapping.operand.complete &&
        mapping.operand.capture_point == expected_capture_point(mapping.role) &&
        mapping.operand.expected_rows == kP40Tokens &&
        mapping.operand.input_features == expected_input_features(mapping.role) &&
        mapping.operand.next_row == expected_next_row &&
        mapping.operand.panel_count == expected_panel_count &&
        layer_inventory_valid && numeric_accounting_valid;
    if (!geometry_valid) {
      result.issues |= kWitnessIssueMissingP40Coverage;
    }
    if (!layer_inventory_valid) {
      result.issues |= kWitnessIssueDuplicateOrWrongLayer;
    }
    if (!mapping.operand.capture_identity_authenticated) {
      result.issues |= kWitnessIssueUnauthenticatedCapture;
    }
    if (!arithmetic_class_valid(mapping.arithmetic_class)) {
      result.issues |= kWitnessIssueInvalidInput;
      all_joint_receipts_valid = false;
      continue;
    }

    const CellDomainSummary& execution_cells =
        arithmetic_class_uses_k16(mapping.arithmetic_class)
            ? mapping.operand.k16
            : mapping.operand.k64;
    std::uint64_t accounted_joint_cells =
        mapping.joint_pass.direct_alignment_cells;
    const bool joint_cell_sum_valid =
        checked_add(accounted_joint_cells,
                    mapping.joint_pass.fallback_alignment_cells);
    const bool joint_valid =
        mapping.joint_pass.role == mapping.role &&
        mapping.joint_pass.arithmetic_class == mapping.arithmetic_class &&
        mapping.joint_pass.source_alignment_cells ==
            execution_cells.cell_count &&
        joint_cell_sum_valid &&
        accounted_joint_cells == execution_cells.cell_count &&
        mapping.joint_pass.fallback_alignment_cells >=
            execution_cells.special_cell_count &&
        mapping.joint_pass.partition_count ==
            expected_partition_count(mapping.role) &&
        digest_is_nonzero(mapping.joint_pass.arithmetic_mapping_sha256) &&
        digest_is_nonzero(mapping.joint_pass.checkpoint_manifest_sha256) &&
        digest_is_nonzero(mapping.joint_pass.pass_receipt_sha256) &&
        mapping.joint_pass.activation_weight_joint_enumeration_complete &&
        mapping.joint_pass.checkpoint_weights_authenticated &&
        mapping.joint_pass.partition_inventory_complete &&
        mapping.joint_pass.encoding_metadata_complete &&
        mapping.joint_pass.pass_accounting_authenticated;
    if (!joint_valid) {
      result.issues |= kWitnessIssueMissingJointPassReceipt;
      all_joint_receipts_valid = false;
      continue;
    }
    if (!checked_add(result.direct_physical_ops,
                     mapping.joint_pass.exact_direct_physical_ops)) {
      result.issues |= kWitnessIssueArithmeticOverflow;
      all_joint_receipts_valid = false;
      continue;
    }
    result.role_physical_ops[index] =
        mapping.joint_pass.exact_direct_physical_ops;
    result.role_source_alignment_cells[index] =
        mapping.joint_pass.source_alignment_cells;
    result.role_direct_alignment_cells[index] =
        mapping.joint_pass.direct_alignment_cells;
    result.role_fallback_alignment_cells[index] =
        mapping.joint_pass.fallback_alignment_cells;
    fallback_cells_by_role[index] =
        mapping.joint_pass.fallback_alignment_cells;
  }
  for (const bool role_seen : seen) {
    if (!role_seen) {
      result.issues |= kWitnessIssueMissingP40Coverage;
    }
  }

  std::size_t expected_projection_instances = 0U;
  bool expected_instance_count_valid = true;
  for (std::size_t index = 0U; index < kOperandRoleCount; ++index) {
    expected_instance_count_valid =
        expected_instance_count_valid &&
        checked_add_size(
            expected_projection_instances,
            expected_role_instances(static_cast<OperandRole>(index + 1U)));
  }
  const ExactnessReceipt& exactness = assessment.exactness;
  const bool oracle_counts_consistent =
      exactness.special_value_mismatches <=
          exactness.special_value_test_cases &&
      exactness.signed_zero_mismatches <= exactness.signed_zero_test_cases &&
      exactness.ordered_fp32_partial_mismatches <=
          exactness.ordered_fp32_partial_checks &&
      exactness.scale_rejoin_mismatches <= exactness.scale_rejoin_checks &&
      exactness.bf16_publication_mismatches <=
          exactness.bf16_publication_checks;
  const bool oracle_complete =
      expected_instance_count_valid &&
      digest_is_nonzero(exactness.oracle_receipt_sha256) &&
      exactness.oracle_identity_authenticated &&
      exactness.compared_projection_instances ==
          expected_projection_instances &&
      exactness.special_value_test_cases >= expected_projection_instances &&
      exactness.signed_zero_test_cases >= expected_projection_instances &&
      exactness.ordered_fp32_partial_checks >=
          expected_projection_instances &&
      exactness.scale_rejoin_checks >= expected_projection_instances &&
      exactness.bf16_publication_checks >= expected_projection_instances &&
      oracle_counts_consistent;
  const bool exactness_mismatch =
      exactness.special_value_mismatches != 0U ||
      exactness.signed_zero_mismatches != 0U ||
      exactness.ordered_fp32_partial_mismatches != 0U ||
      exactness.scale_rejoin_mismatches != 0U ||
      exactness.bf16_publication_mismatches != 0U;
  if (!oracle_complete) {
    result.issues |= kWitnessIssueMissingExactnessOracle;
  } else if (exactness_mismatch) {
    result.issues |= kWitnessIssueExactnessViolation;
  }

  const std::uint64_t projection_logical_ops =
      total_logical_projection_ops() - kBf16AbLogicalProjectionOps;
  result.average_direct_mma_pass_count_excluding_fallback =
      projection_logical_ops == 0U
          ? 0.0
          : static_cast<double>(
                result.direct_physical_ops - kBf16AbLogicalProjectionOps) /
                static_cast<double>(projection_logical_ops);

  bool all_costs_valid = true;
  const std::array<std::uint8_t, 32U> no_projection_mapping{};
  const bool bf16_cost_valid = cost_envelope_valid(
      assessment.ordered_bf16_cost, CostScope::kOrderedBf16Ab,
      OperandRole::kInvalid, ArithmeticClass::kInvalid,
      no_projection_mapping);
  if (!bf16_cost_valid) {
    result.issues |= kWitnessIssueMissingCostEnvelope;
    all_costs_valid = false;
  } else {
    const CostEnvelope& cost = assessment.ordered_bf16_cost;
    result.ordered_bf16_optimistic_seconds_lower =
        static_cast<double>(kBf16AbLogicalProjectionOps) /
            cost.optimistic_throughput_ops_per_second +
        cost.fallback_seconds_lower + cost.representation_seconds_lower;
    result.ordered_bf16_conservative_seconds_upper =
        static_cast<double>(kBf16AbLogicalProjectionOps) /
            cost.conservative_throughput_ops_per_second +
        cost.fallback_seconds_upper + cost.representation_seconds_upper;
    if (cost.representation_component_count == 0U ||
        cost.charged_representation_component_count !=
            cost.representation_component_count ||
        cost.representation_seconds_upper <= 0.0) {
      result.issues |= kWitnessIssueUnchargedRepresentation;
    }
  }

  for (std::size_t index = 0U; index < kOperandRoleCount; ++index) {
    const OperandRole role = static_cast<OperandRole>(index + 1U);
    const RoleMapping* mapping = nullptr;
    for (const RoleMapping& candidate : assessment.roles) {
      if (candidate.role == role) {
        mapping = &candidate;
        break;
      }
    }
    if (mapping == nullptr ||
        mapping->joint_pass.role != role ||
        !all_joint_receipts_valid) {
      all_costs_valid = false;
      continue;
    }
    const CostEnvelope& cost = assessment.role_costs[index];
    const bool role_cost_valid = cost_envelope_valid(
        cost, CostScope::kProjectionRole, role, mapping->arithmetic_class,
        mapping->joint_pass.arithmetic_mapping_sha256);
    if (!role_cost_valid) {
      result.issues |= kWitnessIssueMissingCostEnvelope;
      all_costs_valid = false;
      continue;
    }
    result.role_optimistic_seconds_lower[index] =
        static_cast<double>(result.role_physical_ops[index]) /
            cost.optimistic_throughput_ops_per_second +
        cost.fallback_seconds_lower + cost.representation_seconds_lower;
    result.role_conservative_seconds_upper[index] =
        static_cast<double>(result.role_physical_ops[index]) /
            cost.conservative_throughput_ops_per_second +
        cost.fallback_seconds_upper + cost.representation_seconds_upper;
    if (cost.charged_fallback_cells < fallback_cells_by_role[index] ||
        (fallback_cells_by_role[index] != 0U &&
         cost.fallback_seconds_upper <= 0.0)) {
      result.issues |= kWitnessIssueUnchargedFallback;
    }
    if (cost.representation_component_count == 0U ||
        cost.charged_representation_component_count !=
            cost.representation_component_count ||
        cost.representation_seconds_upper <= 0.0) {
      result.issues |= kWitnessIssueUnchargedRepresentation;
    }
  }

  if (all_costs_valid) {
    result.optimistic_seconds_lower =
        result.ordered_bf16_optimistic_seconds_lower;
    result.conservative_seconds_upper =
        result.ordered_bf16_conservative_seconds_upper;
    for (std::size_t index = 0U; index < kOperandRoleCount; ++index) {
      result.optimistic_seconds_lower +=
          result.role_optimistic_seconds_lower[index];
      result.conservative_seconds_upper +=
          result.role_conservative_seconds_upper[index];
    }
    if (!std::isfinite(result.optimistic_seconds_lower) ||
        !std::isfinite(result.conservative_seconds_upper)) {
      result.issues |= kWitnessIssueArithmeticOverflow;
      all_costs_valid = false;
    }
  }

  constexpr std::uint32_t kEvidenceBlockingIssues =
      kWitnessIssueInvalidInput | kWitnessIssueMissingP40Coverage |
      kWitnessIssueUnauthenticatedCapture |
      kWitnessIssueMissingExactnessOracle |
      kWitnessIssueArithmeticOverflow |
      kWitnessIssueMissingJointPassReceipt |
      kWitnessIssueDuplicateOrWrongLayer;
  if (all_costs_valid && all_joint_receipts_valid &&
      (result.issues & kEvidenceBlockingIssues) == 0U &&
      (result.issues & kWitnessIssueExactnessViolation) == 0U) {
    if (result.optimistic_seconds_lower > kProjectionBudgetSeconds) {
      result.issues |= kWitnessIssueOptimisticBudgetExceeded;
    } else if (result.conservative_seconds_upper > kProjectionBudgetSeconds) {
      result.issues |= kWitnessIssueConservativeBudgetNotClosed;
    }
  }

  const bool authoritative_exactness_rejection =
      (result.issues & kEvidenceBlockingIssues) == 0U &&
      (result.issues & kWitnessIssueExactnessViolation) != 0U;
  const bool authoritative_budget_rejection =
      (result.issues & kEvidenceBlockingIssues) == 0U && all_costs_valid &&
      all_joint_receipts_valid &&
      (result.issues & kWitnessIssueOptimisticBudgetExceeded) != 0U;
  if (authoritative_exactness_rejection ||
      authoritative_budget_rejection) {
    result.decision = Decision::kReject;
  } else if (result.issues == kWitnessIssueNone) {
    result.decision = Decision::kPass;
  } else {
    result.decision = Decision::kInconclusive;
  }
  return result;
}

DecisionReceipt evaluate(const Assessment& assessment) noexcept {
  DecisionReceipt result = evaluate_provisional_for_self_test(assessment);
  result.decision = Decision::kInconclusive;
  result.issues |= kWitnessIssueMissingAuthoritativeEvidenceReader;
  return result;
}

bool run_self_test() noexcept {
  bool passed = true;
  const auto expect = [&passed](const bool value) { passed = passed && value; };
  const auto identity = [](const OperandRole role,
                           const std::size_t layer) {
    OperandInstanceIdentity result;
    result.role = role;
    result.capture_point = expected_capture_point(role);
    result.model_layer_index = layer;
    result.payload_sha256[0U] = static_cast<std::uint8_t>(layer + 1U);
    result.checkpoint_identity_authenticated = true;
    result.route_identity_authenticated = true;
    result.capture_boundary_authenticated = true;
    return result;
  };

  expect(runtime::kSm87AotPrefillSystemSchemaAdmissionCompiled);
  expect(runtime::kSm87AotPrefillSystemCandidateId ==
         "AC-PREFILL-SM87-AOT-SYSTEM-v1");
  expect(total_logical_projection_ops() == 1'948'044'492'800'000ULL);
  expect(population_count(expected_model_layer_mask(
             OperandRole::kFp8GdnQkvZ)) ==
             runtime::kSm87AotPrefillSystemGdnLayerCount &&
         population_count(expected_model_layer_mask(
             OperandRole::kFp8FullQkv)) ==
             runtime::kSm87AotPrefillSystemFullAttentionLayerCount &&
         (expected_model_layer_mask(OperandRole::kFp8GdnQkvZ) &
          expected_model_layer_mask(OperandRole::kFp8FullQkv)) == 0U);
  PlaneBits positive_128{};
  positive_128[0U] = 1ULL << 7U;
  expect(minimum_signed_limb_count(positive_128, false, 7U, 38U) == 2U);
  expect(minimum_signed_limb_count(positive_128, true, 7U, 38U) == 1U);
  PlaneBits positive_8{};
  positive_8[0U] = 1ULL << 3U;
  expect(minimum_signed_limb_count(positive_8, false, 3U, 87U) == 2U);
  expect(minimum_signed_limb_count(positive_8, true, 3U, 87U) == 1U);

  constexpr std::size_t kRows = 128U;
  constexpr std::size_t kColumns = 64U;
  std::vector<std::uint16_t> values(kRows * kColumns, 0U);
  // One exact 2:4 plane per group, plus a second exponent in the same K16
  // cell.  1.0 is 0x3f80 and 0.5 is 0x3f00.
  for (std::size_t row = 0U; row < kRows; ++row) {
    values[row * kColumns] = 0x3f80U;
    values[row * kColumns + 4U] = 0x3f00U;
  }
  Bf16OperandAccumulator accumulator(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(accumulator.consume_panel(0U, values.data(), kRows, kColumns));
  const OperandDomainSummary analyzed = accumulator.finalize();
  expect(analyzed.valid && analyzed.complete && analyzed.panel_count == 1U);
  expect(analyzed.k16.cell_count == 32U &&
         analyzed.k64.cell_count == 1U);
  expect(analyzed.k16.maximum_exponent_span == 1U &&
         analyzed.k16.maximum_aligned_bit_span == 2U &&
         analyzed.k16.maximum_int8_limb_count == 1U &&
         analyzed.k16.maximum_int4_limb_count == 1U);
  expect(analyzed.k16.special_cell_count == 0U &&
         analyzed.k16.infinity_count == 0U && analyzed.k16.nan_count == 0U);
  expect(analyzed.k16.exact_bit_plane_two_of_four_groups ==
         analyzed.k16.occupied_bit_plane_two_of_four_groups);
  Bf16OperandAccumulator layer_one_accumulator(
      identity(OperandRole::kNvFp4GateUp, 1U), kRows, kColumns);
  expect(layer_one_accumulator.consume_panel(0U, values.data(), kRows,
                                             kColumns));
  const OperandDomainSummary layer_one = layer_one_accumulator.finalize();
  OperandDomainMerger merger;
  expect(merger.add(analyzed));
  expect(merger.add(layer_one));
  const OperandDomainSummary merged = merger.finalize();
  expect(merged.valid && merged.complete && merged.instance_count == 2U &&
         merged.next_row == 2U * kRows && merged.panel_count == 2U &&
         merged.model_layer_mask == 3U &&
         merged.k16.cell_count == 2U * analyzed.k16.cell_count &&
         merged.k64.cell_count == 2U * analyzed.k64.cell_count);
  OperandDomainMerger duplicate_layer;
  expect(duplicate_layer.add(analyzed));
  expect(!duplicate_layer.add(analyzed));
  expect(!duplicate_layer.finalize().valid);
  OperandDomainMerger invalid_first;
  OperandDomainSummary partial = analyzed;
  partial.complete = false;
  partial.valid = false;
  expect(!invalid_first.add(partial));
  expect(!invalid_first.add(analyzed));
  expect(!invalid_first.finalize().valid);

  std::fill(values.begin(), values.end(), 0U);
  values[0U] = 0x3f80U;
  values[1U] = 0x3f80U;
  values[2U] = 0x3f80U;
  values[4U] = 0x3580U;  // 2^-20: forces a 21-plane exact alignment.
  Bf16OperandAccumulator sparse_failure(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(sparse_failure.consume_panel(0U, values.data(), kRows, kColumns));
  const OperandDomainSummary sparse_summary = sparse_failure.finalize();
  expect(sparse_summary.k16.maximum_aligned_bit_span == 21U &&
         sparse_summary.k16.maximum_int8_limb_count == 3U &&
         sparse_summary.k16.maximum_int4_limb_count == 7U);
  expect(sparse_summary.k16.exact_bit_plane_two_of_four_groups <
         sparse_summary.k16.occupied_bit_plane_two_of_four_groups);

  Bf16OperandAccumulator wrong_sequence(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(!wrong_sequence.consume_panel(kPanelRows, values.data(), kRows,
                                       kColumns));
  expect(!wrong_sequence.finalize().valid);
  Bf16OperandAccumulator wrong_family_layer(
      identity(OperandRole::kFp8FullQkv, 0U), kRows, kColumns);
  expect(wrong_family_layer.failed() &&
         !wrong_family_layer.finalize().valid);

  std::fill(values.begin(), values.end(), 0U);
  values[0U] = 0x7f80U;
  values[1U] = 0x7fc1U;
  values[2U] = 0x8000U;
  Bf16OperandAccumulator special(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(special.consume_panel(0U, values.data(), kRows, kColumns));
  const OperandDomainSummary special_summary = special.finalize();
  expect(special_summary.k16.special_cell_count == 1U &&
         special_summary.k16.infinity_count == 1U &&
         special_summary.k16.nan_count == 1U &&
         special_summary.k16.negative_zero_count == 1U);
  std::uint64_t special_direct_cells = 0U;
  expect(histogram_sum_checked(special_summary.k16.int8_limb_histogram,
                               special_direct_cells));
  expect(special_direct_cells + special_summary.k16.special_cell_count ==
             special_summary.k16.cell_count &&
         special_summary.k16.residual_plane_slots == 0U &&
         special_summary.k16.occupied_bit_plane_two_of_four_groups == 0U);

  // Bit-plane 2:4 is intentionally not a limb-digit sparsity claim. These
  // values occupy all four lanes as signed limbs, while every individual
  // magnitude plane remains exactly 2:4 encodable.
  std::fill(values.begin(), values.end(), 0U);
  values[0U] = 0x3f80U;
  values[1U] = 0x4000U;
  values[2U] = 0x4080U;
  values[3U] = 0x4040U;
  Bf16OperandAccumulator bit_plane_only(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(bit_plane_only.consume_panel(0U, values.data(), kRows, kColumns));
  const OperandDomainSummary bit_plane_summary = bit_plane_only.finalize();
  expect(bit_plane_summary.k16.exact_bit_plane_two_of_four_groups ==
         bit_plane_summary.k16.occupied_bit_plane_two_of_four_groups);

  Assessment complete;
  complete.ordered_bf16_cost.scope = CostScope::kOrderedBf16Ab;
  complete.ordered_bf16_cost.arithmetic_mapping_sha256[0U] = 0xa1U;
  complete.ordered_bf16_cost.measurement_receipt_sha256[0U] = 0xb1U;
  complete.ordered_bf16_cost.measurement_identity_authenticated = true;
  complete.ordered_bf16_cost.calibrated_same_sm87 = true;
  complete.ordered_bf16_cost.measured_sample_count = 1U;
  complete.ordered_bf16_cost.representation_component_count = 1U;
  complete.ordered_bf16_cost.charged_representation_component_count = 1U;
  complete.ordered_bf16_cost.optimistic_throughput_ops_per_second = 1.0e15;
  complete.ordered_bf16_cost.conservative_throughput_ops_per_second = 5.0e14;
  complete.ordered_bf16_cost.representation_seconds_lower = 0.001;
  complete.ordered_bf16_cost.representation_seconds_upper = 0.002;
  for (std::size_t index = 0U; index < complete.roles.size(); ++index) {
    RoleMapping& mapping = complete.roles[index];
    mapping.role = static_cast<OperandRole>(index + 1U);
    mapping.arithmetic_class = ArithmeticClass::kK16SignedInt8Limbs;
    mapping.operand = synthetic_complete_summary(mapping.role, 1U);
    mapping.joint_pass.role = mapping.role;
    mapping.joint_pass.arithmetic_class = mapping.arithmetic_class;
    mapping.joint_pass.source_alignment_cells = mapping.operand.k16.cell_count;
    mapping.joint_pass.direct_alignment_cells = mapping.operand.k16.cell_count;
    mapping.joint_pass.exact_direct_physical_ops =
        role_logical_projection_ops(mapping.role);
    mapping.joint_pass.partition_count =
        expected_partition_count(mapping.role);
    mapping.joint_pass.arithmetic_mapping_sha256[0U] =
        static_cast<std::uint8_t>(index + 1U);
    mapping.joint_pass.checkpoint_manifest_sha256[0U] = 0xc1U;
    mapping.joint_pass.pass_receipt_sha256[0U] =
        static_cast<std::uint8_t>(0xd0U + index);
    mapping.joint_pass.activation_weight_joint_enumeration_complete = true;
    mapping.joint_pass.checkpoint_weights_authenticated = true;
    mapping.joint_pass.partition_inventory_complete = true;
    mapping.joint_pass.encoding_metadata_complete = true;
    mapping.joint_pass.pass_accounting_authenticated = true;

    CostEnvelope& cost = complete.role_costs[index];
    cost.scope = CostScope::kProjectionRole;
    cost.role = mapping.role;
    cost.arithmetic_class = mapping.arithmetic_class;
    cost.arithmetic_mapping_sha256 =
        mapping.joint_pass.arithmetic_mapping_sha256;
    cost.measurement_receipt_sha256[0U] =
        static_cast<std::uint8_t>(0x40U + index);
    cost.measurement_identity_authenticated = true;
    cost.calibrated_same_sm87 = true;
    cost.measured_sample_count = 1U;
    cost.representation_component_count = 1U;
    cost.charged_representation_component_count = 1U;
    cost.optimistic_throughput_ops_per_second = 1.0e15;
    cost.conservative_throughput_ops_per_second = 5.0e14;
    cost.representation_seconds_lower = 0.001;
    cost.representation_seconds_upper = 0.002;
  }
  std::size_t synthetic_projection_instances = 0U;
  for (std::size_t index = 0U; index < kOperandRoleCount; ++index) {
    synthetic_projection_instances += expected_role_instances(
        static_cast<OperandRole>(index + 1U));
  }
  complete.exactness.oracle_receipt_sha256[0U] = 0xe1U;
  complete.exactness.compared_projection_instances =
      synthetic_projection_instances;
  complete.exactness.special_value_test_cases =
      synthetic_projection_instances;
  complete.exactness.signed_zero_test_cases = synthetic_projection_instances;
  complete.exactness.ordered_fp32_partial_checks =
      synthetic_projection_instances;
  complete.exactness.scale_rejoin_checks = synthetic_projection_instances;
  complete.exactness.bf16_publication_checks =
      synthetic_projection_instances;
  complete.exactness.oracle_identity_authenticated = true;
  const DecisionReceipt passing =
      evaluate_provisional_for_self_test(complete);
  expect(passing.decision == Decision::kPass &&
         passing.issues == kWitnessIssueNone &&
         passing.direct_physical_ops == total_logical_projection_ops() &&
         passing.conservative_seconds_upper < kProjectionBudgetSeconds);

  Assessment no_capture = complete;
  no_capture.roles[0U].operand.capture_identity_authenticated = false;
  const DecisionReceipt inconclusive_capture =
      evaluate_provisional_for_self_test(no_capture);
  expect(inconclusive_capture.decision == Decision::kInconclusive &&
         (inconclusive_capture.issues &
          kWitnessIssueUnauthenticatedCapture) != 0U);

  constexpr std::array<std::uint64_t ExactnessReceipt::*, 5U>
      kExactnessMismatchFields{{
      &ExactnessReceipt::special_value_mismatches,
      &ExactnessReceipt::signed_zero_mismatches,
      &ExactnessReceipt::ordered_fp32_partial_mismatches,
      &ExactnessReceipt::scale_rejoin_mismatches,
      &ExactnessReceipt::bf16_publication_mismatches,
  }};
  for (const auto field : kExactnessMismatchFields) {
    Assessment numerical_failure = complete;
    numerical_failure.exactness.*field = 1U;
    const DecisionReceipt rejected_numerically =
        evaluate_provisional_for_self_test(numerical_failure);
    expect(rejected_numerically.decision == Decision::kReject &&
           (rejected_numerically.issues &
           kWitnessIssueExactnessViolation) != 0U);
  }
  Assessment untrusted_counterexample = complete;
  untrusted_counterexample.roles[0U].operand.capture_identity_authenticated =
      false;
  untrusted_counterexample.exactness.bf16_publication_mismatches = 1U;
  expect(evaluate_provisional_for_self_test(untrusted_counterexample).decision ==
         Decision::kInconclusive);
  Assessment incomplete_oracle = complete;
  incomplete_oracle.exactness.compared_projection_instances -= 1U;
  expect((evaluate_provisional_for_self_test(incomplete_oracle).issues &
          kWitnessIssueMissingExactnessOracle) != 0U);

  Assessment too_slow = complete;
  too_slow.ordered_bf16_cost.optimistic_throughput_ops_per_second = 2.0e14;
  too_slow.ordered_bf16_cost.conservative_throughput_ops_per_second = 1.5e14;
  for (CostEnvelope& cost : too_slow.role_costs) {
    cost.optimistic_throughput_ops_per_second = 2.0e14;
    cost.conservative_throughput_ops_per_second = 1.5e14;
  }
  const DecisionReceipt rejected_by_lower_bound =
      evaluate_provisional_for_self_test(too_slow);
  expect(rejected_by_lower_bound.decision == Decision::kReject &&
         (rejected_by_lower_bound.issues &
          kWitnessIssueOptimisticBudgetExceeded) != 0U);

  Assessment uncertain = complete;
  uncertain.ordered_bf16_cost.optimistic_throughput_ops_per_second = 8.0e14;
  uncertain.ordered_bf16_cost.conservative_throughput_ops_per_second = 2.0e14;
  for (CostEnvelope& cost : uncertain.role_costs) {
    cost.optimistic_throughput_ops_per_second = 8.0e14;
    cost.conservative_throughput_ops_per_second = 2.0e14;
  }
  const DecisionReceipt uncertain_budget =
      evaluate_provisional_for_self_test(uncertain);
  expect(uncertain_budget.decision == Decision::kInconclusive &&
         (uncertain_budget.issues &
          kWitnessIssueConservativeBudgetNotClosed) != 0U);

  Assessment fallback = complete;
  fallback.roles[0U].operand =
      synthetic_complete_summary(fallback.roles[0U].role, 1U, 1U);
  fallback.roles[0U].joint_pass.source_alignment_cells =
      fallback.roles[0U].operand.k16.cell_count;
  fallback.roles[0U].joint_pass.direct_alignment_cells =
      fallback.roles[0U].operand.k16.cell_count - 1U;
  fallback.roles[0U].joint_pass.fallback_alignment_cells = 1U;
  fallback.role_costs[0U].charged_fallback_cells = 0U;
  const DecisionReceipt uncharged_fallback =
      evaluate_provisional_for_self_test(fallback);
  expect(uncharged_fallback.decision == Decision::kInconclusive &&
         (uncharged_fallback.issues & kWitnessIssueUnchargedFallback) != 0U);

  Assessment missing_joint = complete;
  missing_joint.roles[0U].joint_pass.partition_inventory_complete = false;
  const DecisionReceipt missing_joint_receipt =
      evaluate_provisional_for_self_test(missing_joint);
  expect(missing_joint_receipt.decision == Decision::kInconclusive &&
         (missing_joint_receipt.issues &
          kWitnessIssueMissingJointPassReceipt) != 0U);

  Assessment wrong_partition_count = complete;
  wrong_partition_count.roles[0U].joint_pass.partition_count += 1U;
  const DecisionReceipt wrong_partition_receipt =
      evaluate_provisional_for_self_test(wrong_partition_count);
  expect(wrong_partition_receipt.decision == Decision::kInconclusive &&
         (wrong_partition_receipt.issues &
          kWitnessIssueMissingJointPassReceipt) != 0U);

  Assessment mixed_cost_class = complete;
  mixed_cost_class.role_costs[0U].arithmetic_class =
      ArithmeticClass::kK16SignedInt4Limbs;
  const DecisionReceipt mixed_cost =
      evaluate_provisional_for_self_test(mixed_cost_class);
  expect(mixed_cost.decision == Decision::kInconclusive &&
         (mixed_cost.issues & kWitnessIssueMissingCostEnvelope) != 0U);

  Assessment k64_mapping = complete;
  k64_mapping.roles[0U].arithmetic_class =
      ArithmeticClass::kK64SignedInt8Limbs;
  k64_mapping.roles[0U].joint_pass.arithmetic_class =
      ArithmeticClass::kK64SignedInt8Limbs;
  k64_mapping.roles[0U].joint_pass.source_alignment_cells =
      k64_mapping.roles[0U].operand.k64.cell_count;
  k64_mapping.roles[0U].joint_pass.direct_alignment_cells =
      k64_mapping.roles[0U].operand.k64.cell_count;
  k64_mapping.role_costs[0U].arithmetic_class =
      ArithmeticClass::kK64SignedInt8Limbs;
  expect(evaluate_provisional_for_self_test(k64_mapping).decision ==
         Decision::kPass);
  k64_mapping.roles[0U].joint_pass.source_alignment_cells =
      k64_mapping.roles[0U].operand.k16.cell_count;
  expect((evaluate_provisional_for_self_test(k64_mapping).issues &
          kWitnessIssueMissingJointPassReceipt) != 0U);

  Assessment corrupt_k64 = complete;
  corrupt_k64.roles[0U].operand.k64.int8_limb_histogram[1U] -= 1U;
  expect((evaluate_provisional_for_self_test(corrupt_k64).issues &
          kWitnessIssueMissingP40Coverage) != 0U);

  Assessment duplicate_layer_inventory = complete;
  duplicate_layer_inventory.roles[0U].operand.model_layer_mask &= ~1ULL;
  const DecisionReceipt duplicate_layer_receipt =
      evaluate_provisional_for_self_test(duplicate_layer_inventory);
  expect(duplicate_layer_receipt.decision == Decision::kInconclusive &&
         (duplicate_layer_receipt.issues &
          kWitnessIssueDuplicateOrWrongLayer) != 0U);

  const DecisionReceipt public_unissued = evaluate(complete);
  expect(public_unissued.decision == Decision::kInconclusive &&
         (public_unissued.issues &
          kWitnessIssueMissingAuthoritativeEvidenceReader) != 0U);

  return passed;
}

}  // namespace q3x::test::sm87_aot_arithmetic_witness
