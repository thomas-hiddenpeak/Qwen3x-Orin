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
  return summary.layer_payload_digest_mask;
}

constexpr std::string_view kOperandPayloadDigestDomain =
    "q3x.sm87-aot-system-v1/bf16-operand-digest/v1";
constexpr std::string_view kOperandInventoryDigestDomain =
    "q3x.sm87-aot-system-v1/bf16-inventory-digest/v1";
constexpr std::string_view kOperandAnalysisDigestDomain =
    "q3x.sm87-aot-system-v1/bf16-analysis-digest/v1";

[[nodiscard]] std::uint8_t role_wire_id(const OperandRole role) noexcept {
  switch (role) {
    case OperandRole::kNvFp4GateUp:
      return 1U;
    case OperandRole::kNvFp4Down:
      return 2U;
    case OperandRole::kFp8GdnQkvZ:
      return 3U;
    case OperandRole::kFp8FullQkv:
      return 4U;
    case OperandRole::kFp8AttentionOutput:
      return 5U;
    case OperandRole::kInvalid:
    case OperandRole::kCount:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] std::uint8_t capture_wire_id(
    const OperandCapturePoint capture) noexcept {
  switch (capture) {
    case OperandCapturePoint::kGateUpInputAfterPostAttentionNorm:
      return 1U;
    case OperandCapturePoint::kDownInputAfterGatedActivation:
      return 2U;
    case OperandCapturePoint::kGdnQkvZInputAfterInputNorm:
      return 3U;
    case OperandCapturePoint::kFullQkvInputAfterInputNorm:
      return 4U;
    case OperandCapturePoint::kAttentionOutputInputAfterPinnedCore:
      return 5U;
    case OperandCapturePoint::kInvalid:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] bool hash_u8(core::Sha256& hasher,
                           const std::uint8_t value) noexcept {
  return hasher.update(&value, sizeof(value));
}

[[nodiscard]] bool hash_u16le(core::Sha256& hasher,
                              const std::uint16_t value) noexcept {
  const std::array<std::uint8_t, 2U> bytes{{
      static_cast<std::uint8_t>(value),
      static_cast<std::uint8_t>(value >> 8U),
  }};
  return hasher.update(bytes.data(), bytes.size());
}

[[nodiscard]] bool hash_u32le(core::Sha256& hasher,
                              const std::uint32_t value) noexcept {
  std::array<std::uint8_t, 4U> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
  return hasher.update(bytes.data(), bytes.size());
}

[[nodiscard]] bool hash_u64le(core::Sha256& hasher,
                              const std::uint64_t value) noexcept {
  std::array<std::uint8_t, 8U> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
  return hasher.update(bytes.data(), bytes.size());
}

[[nodiscard]] bool hash_string(core::Sha256& hasher,
                               const std::string_view value) noexcept {
  return value.size() <= std::numeric_limits<std::uint32_t>::max() &&
         hash_u32le(hasher, static_cast<std::uint32_t>(value.size())) &&
         hasher.update(value.data(), value.size());
}

template <std::size_t Size>
[[nodiscard]] bool hash_u64_array(
    core::Sha256& hasher,
    const std::array<std::uint64_t, Size>& values) noexcept {
  for (const std::uint64_t value : values) {
    if (!hash_u64le(hasher, value)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool initialize_payload_hash(
    core::Sha256& hasher, const OperandInstanceIdentity& identity,
    const std::size_t rows, const std::size_t columns,
    std::uint64_t& logical_values) noexcept {
  const std::uint8_t role_id = role_wire_id(identity.role);
  const std::uint8_t capture_id = capture_wire_id(identity.capture_point);
  if (role_id == 0U || capture_id == 0U ||
      identity.model_layer_index >
          std::numeric_limits<std::uint32_t>::max() ||
      rows > std::numeric_limits<std::uint64_t>::max() ||
      columns > std::numeric_limits<std::uint64_t>::max() ||
      !checked_multiply(static_cast<std::uint64_t>(rows),
                        static_cast<std::uint64_t>(columns),
                        logical_values)) {
    return false;
  }
  return hash_string(hasher, kOperandPayloadDigestDomain) &&
         hash_string(hasher, runtime::kSm87AotPrefillSystemCandidateId) &&
         hash_string(hasher, runtime::kSm87AotPrefillSystemP40PlanId) &&
         hash_u8(hasher, role_id) && hash_u8(hasher, capture_id) &&
         hash_u32le(
             hasher,
             static_cast<std::uint32_t>(identity.model_layer_index)) &&
         hash_u64le(hasher, static_cast<std::uint64_t>(rows)) &&
         hash_u64le(hasher, static_cast<std::uint64_t>(columns)) &&
         hash_u64le(hasher, logical_values);
}

[[nodiscard]] bool hash_bf16_row_u16le(core::Sha256& hasher,
                                       const std::uint16_t* bits,
                                       const std::size_t count) noexcept {
  if (bits == nullptr && count != 0U) {
    return false;
  }
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  std::size_t bytes = 0U;
  return checked_multiply_size(count, sizeof(std::uint16_t), bytes) &&
         hasher.update(bits, bytes);
#else
  for (std::size_t index = 0U; index < count; ++index) {
    if (!hash_u16le(hasher, bits[index])) {
      return false;
    }
  }
  return true;
#endif
}

[[nodiscard]] bool hash_cell_summary(core::Sha256& hasher,
                                     const CellDomainSummary& cells,
                                     const std::uint8_t cell_width) noexcept {
  return hash_u8(hasher, cell_width) &&
         hash_u64le(hasher, cells.cell_count) &&
         hash_u64le(hasher, cells.value_count) &&
         hash_u64le(hasher, cells.finite_nonzero_count) &&
         hash_u64le(hasher, cells.zero_count) &&
         hash_u64le(hasher, cells.negative_zero_count) &&
         hash_u64le(hasher, cells.subnormal_count) &&
         hash_u64le(hasher, cells.infinity_count) &&
         hash_u64le(hasher, cells.nan_count) &&
         hash_u64le(hasher, cells.special_cell_count) &&
         hash_u32le(hasher, cells.maximum_exponent_span) &&
         hash_u32le(hasher, cells.maximum_aligned_bit_span) &&
         hash_u32le(hasher, cells.maximum_int8_limb_count) &&
         hash_u32le(hasher, cells.maximum_int4_limb_count) &&
         hash_u64le(hasher, cells.residual_plane_set_bits) &&
         hash_u64le(hasher, cells.residual_plane_slots) &&
         hash_u64le(hasher,
                    cells.exact_bit_plane_two_of_four_groups) &&
         hash_u64le(hasher,
                    cells.occupied_bit_plane_two_of_four_groups) &&
         hash_u64_array(hasher, cells.exponent_span_histogram) &&
         hash_u64_array(hasher, cells.aligned_bit_span_histogram) &&
         hash_u64_array(hasher, cells.int8_limb_histogram) &&
         hash_u64_array(hasher, cells.int4_limb_histogram) &&
         hash_u64_array(hasher,
                        cells.significand_trailing_zero_histogram);
}

[[nodiscard]] bool compute_payload_inventory_digest(
    const OperandDomainSummary& summary,
    std::array<std::uint8_t, 32U>& digest) noexcept {
  const std::uint8_t role_id = role_wire_id(summary.role);
  const std::uint8_t capture_id = capture_wire_id(summary.capture_point);
  if (role_id == 0U || capture_id == 0U ||
      summary.expected_rows > std::numeric_limits<std::uint64_t>::max() ||
      summary.input_features > std::numeric_limits<std::uint64_t>::max() ||
      summary.instance_count > std::numeric_limits<std::uint32_t>::max() ||
      summary.model_layer_mask != expected_model_layer_mask(summary.role) ||
      summary.layer_payload_digest_mask !=
          summary.model_layer_mask ||
      summary.instance_count != expected_role_instances(summary.role)) {
    return false;
  }
  core::Sha256 hasher;
  bool valid =
      hash_string(hasher, kOperandInventoryDigestDomain) &&
      hash_string(hasher, runtime::kSm87AotPrefillSystemCandidateId) &&
      hash_string(hasher, runtime::kSm87AotPrefillSystemP40PlanId) &&
      hash_u8(hasher, role_id) && hash_u8(hasher, capture_id) &&
      hash_u64le(hasher,
                 static_cast<std::uint64_t>(summary.expected_rows)) &&
      hash_u64le(hasher,
                 static_cast<std::uint64_t>(summary.input_features)) &&
      hash_u32le(hasher,
                 static_cast<std::uint32_t>(summary.instance_count)) &&
      hash_u64le(hasher, summary.model_layer_mask);
  for (std::size_t layer = 0U;
       valid && layer < summary.layer_payload_sha256.size(); ++layer) {
    if ((summary.model_layer_mask & (1ULL << layer)) == 0U) {
      continue;
    }
    valid = hash_u32le(hasher, static_cast<std::uint32_t>(layer)) &&
            hasher.update(summary.layer_payload_sha256[layer].data(),
                          summary.layer_payload_sha256[layer].size());
  }
  if (!valid) {
    return false;
  }
  digest = hasher.finalize().bytes;
  return true;
}

[[nodiscard]] bool compute_analysis_inventory_digest(
    const OperandDomainSummary& summary,
    std::array<std::uint8_t, 32U>& digest) noexcept {
  if (!summary.payload_inventory_digest_present) {
    return false;
  }
  core::Sha256 hasher;
  if (!hash_string(hasher, kOperandAnalysisDigestDomain) ||
      !hash_string(hasher, runtime::kSm87AotPrefillSystemCandidateId) ||
      !hash_string(hasher, runtime::kSm87AotPrefillSystemP40PlanId) ||
      !hasher.update(summary.payload_inventory_sha256.data(),
                     summary.payload_inventory_sha256.size()) ||
      !hash_cell_summary(hasher, summary.k16, 16U) ||
      !hash_cell_summary(hasher, summary.k64, 64U)) {
    return false;
  }
  digest = hasher.finalize().bytes;
  return true;
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

[[nodiscard]] bool expected_output_tile_count(
    const OperandRole role, std::uint64_t& result) noexcept {
  const auto plan = kernels::sm87_target_aot_projection_plan(
      projection_role(role), kP40Tokens);
  if (!plan.valid() || plan.partition_count == 0U ||
      plan.partition_count > plan.partitions.size()) {
    return false;
  }
  result = 0U;
  std::size_t expected_offset = 0U;
  for (std::size_t index = 0U; index < plan.partition_count; ++index) {
    const auto& partition = plan.partitions[index];
    if (partition.output_offset != expected_offset ||
        partition.output_features == 0U ||
        partition.output_features % kJointOutputTileColumns != 0U ||
        !checked_add(result,
                     static_cast<std::uint64_t>(partition.output_features /
                                                kJointOutputTileColumns))) {
      return false;
    }
    if (!checked_add_size(expected_offset, partition.output_features)) {
      return false;
    }
  }
  return expected_offset == plan.projected_output_features && result != 0U;
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

[[nodiscard]] bool expected_summary_geometry(
    const OperandDomainSummary& summary, std::size_t& next_row,
    std::size_t& panel_count, std::uint64_t& value_count,
    std::uint64_t& k16_cells, std::uint64_t& k64_cells) noexcept {
  std::uint64_t intermediate = 0U;
  std::uint64_t rows = 0U;
  std::uint64_t panels = 0U;
  if (summary.expected_rows == 0U || summary.input_features == 0U ||
      summary.expected_rows % kMmaRows != 0U ||
      summary.input_features % kK64 != 0U ||
      !checked_multiply_size(summary.expected_rows, summary.instance_count,
                             next_row) ||
      !checked_multiply_size(
          (summary.expected_rows + kPanelRows - 1U) / kPanelRows,
          summary.instance_count, panel_count) ||
      !checked_multiply(static_cast<std::uint64_t>(summary.expected_rows),
                        static_cast<std::uint64_t>(summary.instance_count),
                        rows) ||
      !checked_multiply(rows,
                        static_cast<std::uint64_t>(summary.input_features),
                        value_count) ||
      !checked_multiply(
          rows / kMmaRows,
          static_cast<std::uint64_t>(summary.input_features / kK16),
          k16_cells) ||
      !checked_multiply(
          static_cast<std::uint64_t>(
              (summary.expected_rows + kPanelRows - 1U) / kPanelRows),
          static_cast<std::uint64_t>(summary.instance_count), panels) ||
      !checked_multiply(
          panels,
          static_cast<std::uint64_t>(summary.input_features / kK64),
          intermediate)) {
    return false;
  }
  k64_cells = intermediate;
  return true;
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
  result.layer_payload_digest_mask = result.model_layer_mask;
  for (std::size_t layer = 0U;
       layer < runtime::kSm87AotPrefillSystemLayerCount; ++layer) {
    if ((result.model_layer_mask & (1ULL << layer)) != 0U) {
      result.layer_payload_sha256[layer][0U] =
          static_cast<std::uint8_t>(layer + 1U);
    }
  }
  result.capture_identity_authenticated = true;
  result.payload_digest_present = true;
  result.layer_inventory_complete = true;
  result.valid = true;
  std::uint64_t value_count = 0U;
  if (!expected_operand_value_count(role, value_count) ||
      special_cells > expected_k64_cells(role) || limb_count == 0U ||
      limb_count >= kInt8LimbHistogramSize ||
      limb_count >= kInt4LimbHistogramSize) {
    result.valid = false;
    result.layer_inventory_complete = false;
    return result;
  }
  const std::uint64_t finite_nonzero =
      expected_k16_cells(role) - special_cells;
  const std::uint64_t special_values = special_cells;
  if (finite_nonzero + special_values > value_count) {
    result.valid = false;
    result.layer_inventory_complete = false;
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
  result.payload_inventory_digest_present =
      compute_payload_inventory_digest(result,
                                       result.payload_inventory_sha256);
  result.analysis_inventory_digest_present =
      result.payload_inventory_digest_present &&
      compute_analysis_inventory_digest(result,
                                        result.analysis_inventory_sha256);
  result.valid = result.valid && result.analysis_inventory_digest_present;
  result.layer_inventory_complete = result.valid;
  return result;
}

[[nodiscard]] bool populate_synthetic_joint_pass(
    RoleMapping& mapping) noexcept {
  if (!arithmetic_class_valid(mapping.arithmetic_class)) {
    return false;
  }
  const auto plan = kernels::sm87_target_aot_projection_plan(
      projection_role(mapping.role), kP40Tokens);
  const CellDomainSummary& execution_cells =
      arithmetic_class_uses_k16(mapping.arithmetic_class)
          ? mapping.operand.k16
          : mapping.operand.k64;
  if (!plan.valid() || plan.partition_count == 0U ||
      plan.partition_count > mapping.joint_pass.partitions.size() ||
      execution_cells.cell_count == 0U) {
    return false;
  }

  JointPassReceipt receipt;
  receipt.role = mapping.role;
  receipt.arithmetic_class = mapping.arithmetic_class;
  receipt.source_alignment_cells = execution_cells.cell_count;
  receipt.partition_count = static_cast<std::uint32_t>(plan.partition_count);
  for (std::size_t index = 0U; index < plan.partition_count; ++index) {
    const auto& expected = plan.partitions[index];
    JointPartitionReceipt& partition = receipt.partitions[index];
    std::uint64_t n8_tiles = 0U;
    std::uint64_t special_fallback_cells = 0U;
    std::uint64_t direct_cells = 0U;
    std::uint64_t physical_ops = 0U;
    std::uint64_t intermediate = 0U;
    if (expected.output_features == 0U ||
        expected.output_features % kJointOutputTileColumns != 0U) {
      return false;
    }
    n8_tiles = expected.output_features / kJointOutputTileColumns;
    if (n8_tiles > std::numeric_limits<std::uint32_t>::max() ||
        !checked_multiply(execution_cells.cell_count, n8_tiles,
                          partition.joint_mapping_cells) ||
        !checked_multiply(execution_cells.special_cell_count, n8_tiles,
                          special_fallback_cells) ||
        special_fallback_cells > partition.joint_mapping_cells) {
      return false;
    }
    direct_cells = partition.joint_mapping_cells - special_fallback_cells;
    if (!checked_multiply(2U * kP40Tokens,
                          static_cast<std::uint64_t>(plan.input_features),
                          intermediate) ||
        !checked_multiply(
            intermediate,
            static_cast<std::uint64_t>(expected.output_features),
            intermediate) ||
        !checked_multiply(
            intermediate,
            static_cast<std::uint64_t>(expected_role_instances(mapping.role)),
            physical_ops)) {
      return false;
    }
    partition.logical_role = expected.role;
    partition.partition_index = static_cast<std::uint32_t>(index);
    partition.n8_tile_count = static_cast<std::uint32_t>(n8_tiles);
    partition.direct_mapping_cells = direct_cells;
    partition.fallback_mapping_cells = special_fallback_cells;
    partition.exact_direct_physical_ops = physical_ops;
    partition.mapping_sha256[0U] =
        static_cast<std::uint8_t>(0x70U + index);
    partition.enumeration_complete = true;
    if (!checked_add(receipt.joint_mapping_cells,
                     partition.joint_mapping_cells) ||
        !checked_add(receipt.direct_alignment_cells,
                     partition.direct_mapping_cells) ||
        !checked_add(receipt.fallback_alignment_cells,
                     partition.fallback_mapping_cells) ||
        !checked_add(receipt.exact_direct_physical_ops,
                     partition.exact_direct_physical_ops)) {
      return false;
    }
  }
  receipt.arithmetic_mapping_sha256[0U] = 0xa5U;
  receipt.checkpoint_manifest_sha256[0U] = 0xc1U;
  receipt.pass_receipt_sha256[0U] = 0xd1U;
  receipt.activation_weight_joint_enumeration_complete = true;
  receipt.checkpoint_weights_authenticated = true;
  receipt.partition_inventory_complete = true;
  receipt.encoding_metadata_complete = true;
  receipt.pass_accounting_authenticated = true;
  mapping.joint_pass = receipt;
  return receipt.joint_mapping_cells ==
             expected_joint_mapping_cells(mapping.role,
                                          mapping.arithmetic_class) &&
         receipt.exact_direct_physical_ops ==
             role_logical_projection_ops(mapping.role);
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
                   layer_in_role && expected_rows != 0U &&
                   expected_rows % kMmaRows == 0U && input_features != 0U &&
                   input_features % kK64 == 0U;
  if (summary_.valid) {
    summary_.model_layer_mask = 1ULL << identity.model_layer_index;
    source_identity_authenticated_ =
        identity.checkpoint_identity_authenticated &&
        identity.route_identity_authenticated &&
        identity.capture_boundary_authenticated;
    payload_hash_initialized_ = initialize_payload_hash(
        payload_hasher_, identity, expected_rows, input_features,
        expected_logical_value_count_);
  }
  failed_ = !summary_.valid || !payload_hash_initialized_;
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
  std::size_t final_row_offset = 0U;
  if (row_count != expected_panel_rows || row_count % kMmaRows != 0U ||
      !checked_multiply_size(row_count - 1U, row_stride_elements,
                             final_row_offset) ||
      summary_.input_features >
          std::numeric_limits<std::size_t>::max() - final_row_offset) {
    failed_ = true;
    return false;
  }

  std::uint64_t panel_values = 0U;
  if (!checked_multiply(static_cast<std::uint64_t>(row_count),
                        static_cast<std::uint64_t>(summary_.input_features),
                        panel_values) ||
      !checked_add(hashed_logical_value_count_, panel_values)) {
    failed_ = true;
    return false;
  }
  for (std::size_t row = 0U; row < row_count; ++row) {
    if (!hash_bf16_row_u16le(
            payload_hasher_, bits + row * row_stride_elements,
            summary_.input_features)) {
      failed_ = true;
      return false;
    }
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
  summary_.instance_complete =
      !failed_ && summary_.next_row == summary_.expected_rows &&
      hashed_logical_value_count_ == expected_logical_value_count_;
  failed_ = failed_ || !summary_.instance_complete;
  summary_.valid = summary_.valid && summary_.instance_complete;
  if (summary_.valid) {
    std::size_t layer = 0U;
    while (layer < runtime::kSm87AotPrefillSystemLayerCount &&
           (summary_.model_layer_mask & (1ULL << layer)) == 0U) {
      ++layer;
    }
    if (layer == runtime::kSm87AotPrefillSystemLayerCount) {
      summary_.valid = false;
      summary_.instance_complete = false;
    } else {
      summary_.layer_payload_sha256[layer] =
          payload_hasher_.finalize().bytes;
      summary_.layer_payload_digest_mask = 1ULL << layer;
      summary_.payload_digest_present = true;
      summary_.capture_identity_authenticated =
          source_identity_authenticated_;
    }
  }
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
  if (failed_ || finalized_ || !instance.valid ||
      !instance.instance_complete || instance.layer_inventory_complete ||
      !instance.payload_digest_present ||
      instance.payload_inventory_digest_present ||
      instance.analysis_inventory_digest_present ||
      instance.instance_count != 1U ||
      instance.next_row != instance.expected_rows || !layer_in_role ||
      !payload_inventory_matches ||
      instance.role == OperandRole::kInvalid ||
      instance.role == OperandRole::kCount ||
      instance.capture_point != expected_capture_point(instance.role)) {
    failed_ = true;
    summary_.valid = false;
    summary_.instance_complete = false;
    summary_.layer_inventory_complete = false;
    return false;
  }
  if (!started_) {
    summary_ = instance;
    summary_.instance_complete = false;
    started_ = true;
    return true;
  }
  OperandDomainSummary candidate = summary_;
  if (!candidate.valid || candidate.instance_complete ||
      candidate.layer_inventory_complete ||
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
    summary_.instance_complete = false;
    summary_.layer_inventory_complete = false;
    return false;
  }
  for (std::size_t layer = 0U;
       layer < candidate.layer_payload_sha256.size(); ++layer) {
    if ((instance.layer_payload_digest_mask & (1ULL << layer)) != 0U) {
      candidate.layer_payload_sha256[layer] =
          instance.layer_payload_sha256[layer];
    }
  }
  candidate.model_layer_mask |= instance.model_layer_mask;
  candidate.layer_payload_digest_mask |=
      instance.layer_payload_digest_mask;
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
    summary_.layer_inventory_complete = false;
  } else {
    std::size_t expected_next_row = 0U;
    std::size_t expected_panel_count = 0U;
    std::uint64_t expected_values = 0U;
    std::uint64_t expected_k16 = 0U;
    std::uint64_t expected_k64 = 0U;
    const bool complete_inventory =
        summary_.model_layer_mask == expected_model_layer_mask(summary_.role) &&
        summary_.layer_payload_digest_mask == summary_.model_layer_mask &&
        summary_.instance_count == expected_role_instances(summary_.role) &&
        expected_summary_geometry(summary_, expected_next_row,
                                  expected_panel_count, expected_values,
                                  expected_k16, expected_k64) &&
        summary_.next_row == expected_next_row &&
        summary_.panel_count == expected_panel_count &&
        validate_cell_summary(summary_.k16, expected_k16,
                              expected_values) &&
        validate_cell_summary(summary_.k64, expected_k64,
                              expected_values) &&
        same_value_classification(summary_.k16, summary_.k64);
    summary_.layer_inventory_complete = complete_inventory;
    summary_.payload_inventory_digest_present =
        complete_inventory &&
        compute_payload_inventory_digest(
            summary_, summary_.payload_inventory_sha256);
    summary_.analysis_inventory_digest_present =
        summary_.payload_inventory_digest_present &&
        compute_analysis_inventory_digest(
            summary_, summary_.analysis_inventory_sha256);
    summary_.valid = summary_.valid &&
                     summary_.analysis_inventory_digest_present;
    summary_.layer_inventory_complete = summary_.valid;
    failed_ = !summary_.layer_inventory_complete;
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

std::size_t expected_output_features(const OperandRole role) noexcept {
  const auto plan = kernels::sm87_target_aot_projection_plan(
      projection_role(role), kP40Tokens);
  return plan.valid() ? plan.projected_output_features : 0U;
}

std::uint64_t expected_joint_mapping_cells(
    const OperandRole role,
    const ArithmeticClass arithmetic_class) noexcept {
  if (!arithmetic_class_valid(arithmetic_class)) {
    return 0U;
  }
  std::uint64_t output_tiles = 0U;
  std::uint64_t result = 0U;
  const std::uint64_t activation_cells =
      arithmetic_class_uses_k16(arithmetic_class) ? expected_k16_cells(role)
                                                  : expected_k64_cells(role);
  if (activation_cells == 0U ||
      !expected_output_tile_count(role, output_tiles) ||
      !checked_multiply(activation_cells, output_tiles, result)) {
    return 0U;
  }
  return result;
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
    std::array<std::uint8_t, 32U> expected_payload_inventory{};
    std::array<std::uint8_t, 32U> expected_analysis_inventory{};
    OperandDomainSummary canonical_operand = mapping.operand;
    const bool payload_inventory_digest_valid =
        mapping.operand.payload_inventory_digest_present &&
        compute_payload_inventory_digest(mapping.operand,
                                         expected_payload_inventory) &&
        expected_payload_inventory ==
            mapping.operand.payload_inventory_sha256;
    canonical_operand.payload_inventory_sha256 =
        expected_payload_inventory;
    canonical_operand.payload_inventory_digest_present =
        payload_inventory_digest_valid;
    const bool analysis_inventory_digest_valid =
        mapping.operand.analysis_inventory_digest_present &&
        payload_inventory_digest_valid &&
        compute_analysis_inventory_digest(canonical_operand,
                                          expected_analysis_inventory) &&
        expected_analysis_inventory ==
            mapping.operand.analysis_inventory_sha256;
    const bool geometry_valid =
        mapping.operand.role == mapping.role && mapping.operand.valid &&
        !mapping.operand.instance_complete &&
        mapping.operand.layer_inventory_complete &&
        mapping.operand.payload_digest_present &&
        payload_inventory_digest_valid && analysis_inventory_digest_valid &&
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
    const auto projection_plan = kernels::sm87_target_aot_projection_plan(
        projection_role(mapping.role), kP40Tokens);
    const std::uint64_t expected_joint_cells =
        expected_joint_mapping_cells(mapping.role, mapping.arithmetic_class);
    std::uint64_t accounted_joint_cells = 0U;
    std::uint64_t accounted_direct_cells = 0U;
    std::uint64_t accounted_fallback_cells = 0U;
    std::uint64_t accounted_direct_physical_ops = 0U;
    bool partition_receipts_valid =
        projection_plan.valid() && projection_plan.partition_count != 0U &&
        projection_plan.partition_count <=
            mapping.joint_pass.partitions.size();
    for (std::size_t partition_index = 0U;
         partition_index < mapping.joint_pass.partitions.size();
         ++partition_index) {
      const JointPartitionReceipt& receipt =
          mapping.joint_pass.partitions[partition_index];
      if (partition_index >= projection_plan.partition_count) {
        partition_receipts_valid =
            partition_receipts_valid &&
            receipt.logical_role ==
                kernels::Sm87TargetAotLogicalRole::kInvalid &&
            receipt.n8_tile_count == 0U &&
            receipt.joint_mapping_cells == 0U &&
            receipt.direct_mapping_cells == 0U &&
            receipt.fallback_mapping_cells == 0U &&
            receipt.exact_direct_physical_ops == 0U &&
            !digest_is_nonzero(receipt.mapping_sha256) &&
            !receipt.enumeration_complete;
        continue;
      }
      const auto& expected_partition =
          projection_plan.partitions[partition_index];
      const std::uint64_t n8_tiles =
          expected_partition.output_features / kJointOutputTileColumns;
      std::uint64_t expected_partition_cells = 0U;
      std::uint64_t expected_special_fallback_cells = 0U;
      std::uint64_t partition_cell_sum = receipt.direct_mapping_cells;
      const bool partition_arithmetic_valid =
          expected_partition.output_features != 0U &&
          expected_partition.output_features % kJointOutputTileColumns == 0U &&
          n8_tiles <= std::numeric_limits<std::uint32_t>::max() &&
          checked_multiply(execution_cells.cell_count, n8_tiles,
                           expected_partition_cells) &&
          checked_multiply(execution_cells.special_cell_count, n8_tiles,
                           expected_special_fallback_cells) &&
          checked_add(partition_cell_sum,
                      receipt.fallback_mapping_cells);
      partition_receipts_valid =
          partition_receipts_valid && partition_arithmetic_valid &&
          receipt.partition_index == partition_index &&
          receipt.logical_role == expected_partition.role &&
          receipt.n8_tile_count == n8_tiles &&
          receipt.joint_mapping_cells == expected_partition_cells &&
          partition_cell_sum == expected_partition_cells &&
          receipt.fallback_mapping_cells >=
              expected_special_fallback_cells &&
          digest_is_nonzero(receipt.mapping_sha256) &&
          receipt.enumeration_complete &&
          checked_add(accounted_joint_cells, receipt.joint_mapping_cells) &&
          checked_add(accounted_direct_cells, receipt.direct_mapping_cells) &&
          checked_add(accounted_fallback_cells,
                      receipt.fallback_mapping_cells) &&
          checked_add(accounted_direct_physical_ops,
                      receipt.exact_direct_physical_ops);
    }
    std::uint64_t partition_joint_sum = accounted_direct_cells;
    std::uint64_t top_joint_sum = mapping.joint_pass.direct_alignment_cells;
    const bool joint_cell_sum_valid =
        checked_add(partition_joint_sum, accounted_fallback_cells) &&
        checked_add(top_joint_sum,
                    mapping.joint_pass.fallback_alignment_cells);
    const bool joint_valid =
        mapping.joint_pass.role == mapping.role &&
        mapping.joint_pass.arithmetic_class == mapping.arithmetic_class &&
        mapping.joint_pass.source_alignment_cells ==
            execution_cells.cell_count &&
        expected_joint_cells != 0U &&
        mapping.joint_pass.joint_mapping_cells == expected_joint_cells &&
        partition_receipts_valid &&
        accounted_joint_cells == expected_joint_cells &&
        accounted_direct_cells ==
            mapping.joint_pass.direct_alignment_cells &&
        accounted_fallback_cells ==
            mapping.joint_pass.fallback_alignment_cells &&
        accounted_direct_physical_ops ==
            mapping.joint_pass.exact_direct_physical_ops &&
        joint_cell_sum_valid &&
        partition_joint_sum == expected_joint_cells &&
        top_joint_sum == expected_joint_cells &&
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
    result.role_joint_mapping_cells[index] =
        mapping.joint_pass.joint_mapping_cells;
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
    result.checkpoint_identity_authenticated = true;
    result.route_identity_authenticated = true;
    result.capture_boundary_authenticated = true;
    return result;
  };

  expect(runtime::kSm87AotPrefillSystemSchemaAdmissionCompiled);
  expect(runtime::kSm87AotPrefillSystemCandidateId ==
         "AC-PREFILL-SM87-AOT-SYSTEM-v1");
  expect(total_logical_projection_ops() == 1'948'044'492'800'000ULL);
  expect(expected_output_features(OperandRole::kNvFp4GateUp) == 34'816U &&
         expected_output_features(OperandRole::kNvFp4Down) == 5'120U &&
         expected_output_features(OperandRole::kFp8GdnQkvZ) == 16'384U &&
         expected_output_features(OperandRole::kFp8FullQkv) == 14'336U &&
         expected_output_features(OperandRole::kFp8AttentionOutput) == 5'120U);
  expect(expected_joint_mapping_cells(
             OperandRole::kNvFp4GateUp,
             ArithmeticClass::kK16ExactBitPlanes) == 222'822'400'000ULL &&
         expected_joint_mapping_cells(
             OperandRole::kNvFp4Down,
             ArithmeticClass::kK16ExactBitPlanes) == 111'411'200'000ULL &&
         expected_joint_mapping_cells(
             OperandRole::kFp8GdnQkvZ,
             ArithmeticClass::kK16ExactBitPlanes) == 78'643'200'000ULL &&
         expected_joint_mapping_cells(
             OperandRole::kFp8FullQkv,
             ArithmeticClass::kK16ExactBitPlanes) == 22'937'600'000ULL &&
         expected_joint_mapping_cells(
             OperandRole::kFp8AttentionOutput,
             ArithmeticClass::kK16ExactBitPlanes) == 39'321'600'000ULL);
  expect(expected_joint_mapping_cells(
             OperandRole::kNvFp4GateUp,
             ArithmeticClass::kK64ExactBitPlanes) == 6'974'341'120ULL &&
         expected_joint_mapping_cells(
             OperandRole::kNvFp4Down,
             ArithmeticClass::kK64ExactBitPlanes) == 3'487'170'560ULL &&
         expected_joint_mapping_cells(
             OperandRole::kFp8GdnQkvZ,
             ArithmeticClass::kK64ExactBitPlanes) == 2'461'532'160ULL &&
         expected_joint_mapping_cells(
             OperandRole::kFp8FullQkv,
             ArithmeticClass::kK64ExactBitPlanes) == 717'946'880ULL &&
         expected_joint_mapping_cells(
             OperandRole::kFp8AttentionOutput,
             ArithmeticClass::kK64ExactBitPlanes) == 1'230'766'080ULL);
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
  expect(analyzed.valid && analyzed.instance_complete &&
         !analyzed.layer_inventory_complete &&
         analyzed.payload_digest_present &&
         !analyzed.payload_inventory_digest_present &&
         !analyzed.analysis_inventory_digest_present &&
         analyzed.layer_payload_digest_mask == 1U &&
         analyzed.panel_count == 1U);
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
  expect(!accumulator.consume_panel(0U, values.data(), kRows, kColumns));
  expect(!accumulator.finalize().valid);
  Bf16OperandAccumulator duplicate_panel_accumulator(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(duplicate_panel_accumulator.consume_panel(0U, values.data(), kRows,
                                                   kColumns));
  expect(!duplicate_panel_accumulator.consume_panel(0U, values.data(), kRows,
                                                    kColumns));
  expect(!duplicate_panel_accumulator.finalize().valid);
  Bf16OperandAccumulator short_panel_accumulator(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(!short_panel_accumulator.consume_panel(0U, values.data(),
                                                kRows / 2U, kColumns));
  expect(!short_panel_accumulator.finalize().valid);
  Bf16OperandAccumulator layer_one_accumulator(
      identity(OperandRole::kNvFp4GateUp, 1U), kRows, kColumns);
  expect(layer_one_accumulator.consume_panel(0U, values.data(), kRows,
                                             kColumns));
  const OperandDomainSummary layer_one = layer_one_accumulator.finalize();
  expect(layer_one.valid && layer_one.instance_complete &&
         layer_one.layer_payload_sha256[1U] !=
             analyzed.layer_payload_sha256[0U]);

  std::vector<std::uint16_t> padded_values(
      kRows * (kColumns + 8U), 0xa5a5U);
  for (std::size_t row = 0U; row < kRows; ++row) {
    std::copy_n(values.data() + row * kColumns, kColumns,
                padded_values.data() + row * (kColumns + 8U));
  }
  Bf16OperandAccumulator padded_accumulator(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(padded_accumulator.consume_panel(0U, padded_values.data(), kRows,
                                          kColumns + 8U));
  const OperandDomainSummary padded = padded_accumulator.finalize();
  expect(padded.valid && padded.layer_payload_sha256[0U] ==
                             analyzed.layer_payload_sha256[0U]);
  padded_values[kColumns] ^= 0xffffU;
  Bf16OperandAccumulator changed_padding_accumulator(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(changed_padding_accumulator.consume_panel(
      0U, padded_values.data(), kRows, kColumns + 8U));
  expect(changed_padding_accumulator.finalize()
             .layer_payload_sha256[0U] ==
         analyzed.layer_payload_sha256[0U]);
  padded_values[0U] ^= 1U;
  Bf16OperandAccumulator changed_payload_accumulator(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(changed_payload_accumulator.consume_panel(
      0U, padded_values.data(), kRows, kColumns + 8U));
  expect(changed_payload_accumulator.finalize()
             .layer_payload_sha256[0U] !=
         analyzed.layer_payload_sha256[0U]);
  Bf16OperandAccumulator changed_role_accumulator(
      identity(OperandRole::kNvFp4Down, 0U), kRows, kColumns);
  expect(changed_role_accumulator.consume_panel(0U, values.data(), kRows,
                                                kColumns));
  expect(changed_role_accumulator.finalize().layer_payload_sha256[0U] !=
         analyzed.layer_payload_sha256[0U]);
  std::vector<std::uint16_t> wider_values(kRows * (2U * kColumns), 0U);
  for (std::size_t row = 0U; row < kRows; ++row) {
    std::copy_n(values.data() + row * kColumns, kColumns,
                wider_values.data() + row * 2U * kColumns);
  }
  Bf16OperandAccumulator changed_shape_accumulator(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, 2U * kColumns);
  expect(changed_shape_accumulator.consume_panel(
      0U, wider_values.data(), kRows, 2U * kColumns));
  expect(changed_shape_accumulator.finalize().layer_payload_sha256[0U] !=
         analyzed.layer_payload_sha256[0U]);

  core::Sha256 encoded_u16;
  core::Sha256 golden_u16;
  const std::array<std::uint8_t, 2U> golden_u16_bytes{{0x34U, 0x12U}};
  expect(hash_u16le(encoded_u16, 0x1234U) &&
         golden_u16.update(golden_u16_bytes.data(),
                           golden_u16_bytes.size()) &&
         encoded_u16.finalize() == golden_u16.finalize());

  std::array<OperandDomainSummary,
             runtime::kSm87AotPrefillSystemLayerCount>
      gate_instances{};
  gate_instances[0U] = analyzed;
  gate_instances[1U] = layer_one;
  for (std::size_t layer = 2U; layer < gate_instances.size(); ++layer) {
    Bf16OperandAccumulator layer_accumulator(
        identity(OperandRole::kNvFp4GateUp, layer), kRows, kColumns);
    expect(layer_accumulator.consume_panel(0U, values.data(), kRows,
                                           kColumns));
    gate_instances[layer] = layer_accumulator.finalize();
    expect(gate_instances[layer].valid);
  }
  OperandDomainMerger forward_merger;
  for (const OperandDomainSummary& instance : gate_instances) {
    expect(forward_merger.add(instance));
  }
  const OperandDomainSummary merged = forward_merger.finalize();
  expect(merged.valid && !merged.instance_complete &&
         merged.layer_inventory_complete && merged.payload_digest_present &&
         merged.payload_inventory_digest_present &&
         merged.analysis_inventory_digest_present &&
         merged.instance_count == gate_instances.size() &&
         merged.next_row == gate_instances.size() * kRows &&
         merged.panel_count == gate_instances.size() &&
         merged.model_layer_mask == std::numeric_limits<std::uint64_t>::max() &&
         merged.layer_payload_digest_mask == merged.model_layer_mask &&
         merged.k16.cell_count ==
             gate_instances.size() * analyzed.k16.cell_count &&
         merged.k64.cell_count ==
             gate_instances.size() * analyzed.k64.cell_count);
  OperandDomainMerger reverse_merger;
  for (std::size_t layer = gate_instances.size(); layer-- > 0U;) {
    expect(reverse_merger.add(gate_instances[layer]));
  }
  const OperandDomainSummary reverse_merged = reverse_merger.finalize();
  expect(reverse_merged.valid &&
         reverse_merged.payload_inventory_sha256 ==
             merged.payload_inventory_sha256 &&
         reverse_merged.analysis_inventory_sha256 ==
             merged.analysis_inventory_sha256);
  OperandDomainMerger duplicate_layer;
  expect(duplicate_layer.add(analyzed));
  expect(!duplicate_layer.add(analyzed));
  expect(!duplicate_layer.finalize().valid);
  OperandDomainMerger missing_layer;
  for (std::size_t layer = 0U; layer + 1U < gate_instances.size(); ++layer) {
    expect(missing_layer.add(gate_instances[layer]));
  }
  expect(!missing_layer.finalize().valid && missing_layer.failed());
  OperandDomainMerger invalid_first;
  OperandDomainSummary partial = analyzed;
  partial.instance_complete = false;
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
  Bf16OperandAccumulator incomplete_sequence(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(!incomplete_sequence.finalize().valid &&
         incomplete_sequence.failed());
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
  values[1U] = 0x7fc2U;
  Bf16OperandAccumulator changed_nan_payload(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(changed_nan_payload.consume_panel(0U, values.data(), kRows,
                                           kColumns));
  expect(changed_nan_payload.finalize().layer_payload_sha256[0U] !=
         special_summary.layer_payload_sha256[0U]);
  values[1U] = 0x7fc1U;
  values[2U] = 0x0000U;
  Bf16OperandAccumulator changed_signed_zero(
      identity(OperandRole::kNvFp4GateUp, 0U), kRows, kColumns);
  expect(changed_signed_zero.consume_panel(0U, values.data(), kRows,
                                           kColumns));
  expect(changed_signed_zero.finalize().layer_payload_sha256[0U] !=
         special_summary.layer_payload_sha256[0U]);

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
    expect(populate_synthetic_joint_pass(mapping));
    mapping.joint_pass.arithmetic_mapping_sha256[0U] =
        static_cast<std::uint8_t>(index + 1U);
    mapping.joint_pass.pass_receipt_sha256[0U] =
        static_cast<std::uint8_t>(0xd0U + index);

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
  Assessment toy_inventory = complete;
  toy_inventory.roles[0U].operand = merged;
  const DecisionReceipt toy_inventory_result =
      evaluate_provisional_for_self_test(toy_inventory);
  expect(toy_inventory.roles[0U].operand.layer_inventory_complete &&
         (toy_inventory_result.issues & kWitnessIssueMissingP40Coverage) !=
             0U);
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
  expect(populate_synthetic_joint_pass(fallback.roles[0U]));
  fallback.role_costs[0U].arithmetic_mapping_sha256 =
      fallback.roles[0U].joint_pass.arithmetic_mapping_sha256;
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
  expect(populate_synthetic_joint_pass(k64_mapping.roles[0U]));
  k64_mapping.role_costs[0U].arithmetic_class =
      ArithmeticClass::kK64SignedInt8Limbs;
  k64_mapping.role_costs[0U].arithmetic_mapping_sha256 =
      k64_mapping.roles[0U].joint_pass.arithmetic_mapping_sha256;
  expect(evaluate_provisional_for_self_test(k64_mapping).decision ==
         Decision::kPass);
  k64_mapping.roles[0U].joint_pass.source_alignment_cells =
      k64_mapping.roles[0U].operand.k16.cell_count;
  expect((evaluate_provisional_for_self_test(k64_mapping).issues &
          kWitnessIssueMissingJointPassReceipt) != 0U);

  Assessment n256_under_count = complete;
  n256_under_count.roles[0U].joint_pass.joint_mapping_cells /= 32U;
  expect((evaluate_provisional_for_self_test(n256_under_count).issues &
          kWitnessIssueMissingJointPassReceipt) != 0U);

  Assessment missing_partition = complete;
  missing_partition.roles[0U].joint_pass.partitions[1U] = {};
  expect((evaluate_provisional_for_self_test(missing_partition).issues &
          kWitnessIssueMissingJointPassReceipt) != 0U);

  Assessment swapped_partition_roles = complete;
  std::swap(swapped_partition_roles.roles[0U]
                .joint_pass.partitions[0U].logical_role,
            swapped_partition_roles.roles[0U]
                .joint_pass.partitions[1U].logical_role);
  expect((evaluate_provisional_for_self_test(swapped_partition_roles).issues &
          kWitnessIssueMissingJointPassReceipt) != 0U);

  Assessment corrupt_k64 = complete;
  corrupt_k64.roles[0U].operand.k64.int8_limb_histogram[1U] -= 1U;
  expect((evaluate_provisional_for_self_test(corrupt_k64).issues &
          kWitnessIssueMissingP40Coverage) != 0U);

  Assessment changed_payload_inventory = complete;
  changed_payload_inventory.roles[0U]
      .operand.layer_payload_sha256[0U][0U] ^= 1U;
  expect((evaluate_provisional_for_self_test(changed_payload_inventory).issues &
          kWitnessIssueMissingP40Coverage) != 0U);
  Assessment changed_analysis_inventory = complete;
  changed_analysis_inventory.roles[0U]
      .operand.analysis_inventory_sha256[0U] ^= 1U;
  expect((evaluate_provisional_for_self_test(changed_analysis_inventory)
              .issues &
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
  const DecisionReceipt public_rejection_is_unissued = evaluate(too_slow);
  expect(public_rejection_is_unissued.decision == Decision::kInconclusive &&
         (public_rejection_is_unissued.issues &
          kWitnessIssueMissingAuthoritativeEvidenceReader) != 0U);

  return passed;
}

}  // namespace q3x::test::sm87_aot_arithmetic_witness
