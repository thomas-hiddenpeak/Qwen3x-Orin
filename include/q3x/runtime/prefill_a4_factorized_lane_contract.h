#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {

// Experimental host-only ABI foundation for a dense factorized-scale A4
// projection.  This contract is intentionally independent from
// PrefillSidecarKind and has no runtime-selection or production-residency
// authority.
//
// Packed signed-W4 codes retain the existing physical transport order and
// therefore occupy exactly N*K/2 bytes.  The numerical contract is new:
// weight and activation scales are stored once per factorization lane rather
// than once per fixed K group.  No performance or model-accuracy property is
// implied by a structurally valid plan.
inline constexpr std::uint32_t
    kPrefillA4FactorizedLaneContractVersionMajor = 4U;
inline constexpr std::uint32_t
    kPrefillA4FactorizedLaneContractVersionMinor = 0U;
inline constexpr std::uint64_t kPrefillA4FactorizedLaneOuterBlock = 64U;
inline constexpr std::uint64_t kPrefillA4FactorizedLanePackedKBlock = 64U;
inline constexpr std::uint64_t kPrefillA4FactorizedLaneMinimumAlignment =
    256U;
inline constexpr std::uint64_t kPrefillA4FactorizedLaneBf16Bytes = 2U;
inline constexpr std::uint64_t kPrefillA4FactorizedLaneFp32Bytes = 4U;
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneInvalidOffset =
        std::numeric_limits<std::uint64_t>::max();

// Metadata is encoded byte-for-byte rather than as a C++ object so compiler
// padding and host endianness cannot alter the publication ABI:
//
//   [0, 8)   magic "Q3XA4FL4"
//   [8, 12)  little-endian version major
//   [12, 16) little-endian version minor
//   [16, 20) little-endian lane count
//   [20, 24) little-endian inverse-alpha encoding (1 == FP32 LE)
//   [24, 32) little-endian K
//   [32, 64) SHA-256 of the exact inverse-alpha byte payload
//   [64, ...) finite positive inverse-alpha[K], little-endian FP32
//
// The digest permits exact internal verification of the inverse-alpha array.
// Authentication requires a future publication to bind the complete
// projection payload through its ordinary receipt/payload identity before
// residency is eligible.
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataHeaderBytes = 64U;
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataMagicOffset = 0U;
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataMagicBytes = 8U;
inline constexpr char kPrefillA4FactorizedLaneMetadataMagic[9] =
    "Q3XA4FL4";
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataVersionMajorOffset = 8U;
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataVersionMinorOffset = 12U;
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataLaneCountOffset = 16U;
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataEncodingOffset = 20U;
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataInputSizeOffset = 24U;
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataDigestOffset = 32U;
inline constexpr std::uint64_t
    kPrefillA4FactorizedLaneMetadataDigestBytes = 32U;
inline constexpr std::uint32_t
    kPrefillA4FactorizedLaneInverseAlphaFp32Le = 1U;

enum class PrefillA4FactorizedLanePlanError : std::uint8_t {
  kNone = 0,
  kInvalidAlignment,
  kZeroShape,
  kShapeNotConsumerAligned,
  kUnsupportedLaneCount,
  kLaneNotConsumerAligned,
  kArithmeticOverflow,
};

struct PrefillA4FactorizedLaneProjectionLayoutPlan final {
  PrefillA4FactorizedLanePlanError error =
      PrefillA4FactorizedLanePlanError::kZeroShape;
  std::uint64_t output_size = 0U;
  std::uint64_t input_size = 0U;
  std::uint32_t lane_count = 0U;
  std::uint64_t lane_input_size = 0U;
  std::uint64_t alignment = 0U;

  std::uint64_t packed_weight_bytes = 0U;
  std::uint64_t weight_scale_elements = 0U;
  std::uint64_t weight_scale_bytes = 0U;
  std::uint64_t inverse_alpha_elements = 0U;
  std::uint64_t inverse_alpha_bytes = 0U;
  std::uint64_t metadata_bytes = 0U;

  std::uint64_t packed_weight_offset = 0U;
  std::uint64_t weight_scale_offset = 0U;
  std::uint64_t metadata_offset = 0U;
  std::uint64_t metadata_digest_offset = 0U;
  std::uint64_t inverse_alpha_offset = 0U;
  std::uint64_t projection_bytes = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return error == PrefillA4FactorizedLanePlanError::kNone;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
};

namespace prefill_a4_factorized_lane_contract_detail {

[[nodiscard]] constexpr bool is_power_of_two(
    const std::uint64_t value) noexcept {
  return value != 0U && (value & (value - 1U)) == 0U;
}

[[nodiscard]] constexpr bool checked_multiply(
    const std::uint64_t left, const std::uint64_t right,
    std::uint64_t& output) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

[[nodiscard]] constexpr bool checked_add(
    const std::uint64_t left, const std::uint64_t right,
    std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] constexpr bool checked_align_up(
    const std::uint64_t value, const std::uint64_t alignment,
    std::uint64_t& output) noexcept {
  if (!is_power_of_two(alignment)) {
    return false;
  }
  const std::uint64_t mask = alignment - 1U;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return false;
  }
  output = (value + mask) & ~mask;
  return true;
}

}  // namespace prefill_a4_factorized_lane_contract_detail

[[nodiscard]] constexpr PrefillA4FactorizedLaneProjectionLayoutPlan
prefill_a4_factorized_lane_projection_layout_plan(
    const std::uint64_t output_size, const std::uint64_t input_size,
    const std::uint32_t lane_count,
    const std::uint64_t alignment =
        kPrefillA4FactorizedLaneMinimumAlignment) noexcept {
  PrefillA4FactorizedLaneProjectionLayoutPlan plan;
  plan.output_size = output_size;
  plan.input_size = input_size;
  plan.lane_count = lane_count;
  plan.alignment = alignment;

  if (alignment < kPrefillA4FactorizedLaneMinimumAlignment ||
      !prefill_a4_factorized_lane_contract_detail::is_power_of_two(
          alignment)) {
    plan.error = PrefillA4FactorizedLanePlanError::kInvalidAlignment;
    return plan;
  }
  if (output_size == 0U || input_size == 0U) {
    plan.error = PrefillA4FactorizedLanePlanError::kZeroShape;
    return plan;
  }
  if (output_size % kPrefillA4FactorizedLaneOuterBlock != 0U ||
      input_size % kPrefillA4FactorizedLanePackedKBlock != 0U) {
    plan.error =
        PrefillA4FactorizedLanePlanError::kShapeNotConsumerAligned;
    return plan;
  }
  if (lane_count != 1U && lane_count != 2U && lane_count != 4U) {
    plan.error = PrefillA4FactorizedLanePlanError::kUnsupportedLaneCount;
    return plan;
  }
  if (input_size % lane_count != 0U ||
      (input_size / lane_count) %
              kPrefillA4FactorizedLanePackedKBlock !=
          0U) {
    plan.error = PrefillA4FactorizedLanePlanError::kLaneNotConsumerAligned;
    return plan;
  }
  plan.lane_input_size = input_size / lane_count;

  std::uint64_t weight_elements = 0U;
  if (!prefill_a4_factorized_lane_contract_detail::checked_multiply(
          output_size, input_size, weight_elements)) {
    plan.error = PrefillA4FactorizedLanePlanError::kArithmeticOverflow;
    return plan;
  }
  plan.packed_weight_bytes = weight_elements / 2U;
  if (!prefill_a4_factorized_lane_contract_detail::checked_multiply(
          output_size, lane_count, plan.weight_scale_elements) ||
      !prefill_a4_factorized_lane_contract_detail::checked_multiply(
          plan.weight_scale_elements,
          kPrefillA4FactorizedLaneBf16Bytes,
          plan.weight_scale_bytes) ||
      !prefill_a4_factorized_lane_contract_detail::checked_multiply(
          input_size, kPrefillA4FactorizedLaneFp32Bytes,
          plan.inverse_alpha_bytes) ||
      !prefill_a4_factorized_lane_contract_detail::checked_add(
          kPrefillA4FactorizedLaneMetadataHeaderBytes,
          plan.inverse_alpha_bytes, plan.metadata_bytes)) {
    plan.error = PrefillA4FactorizedLanePlanError::kArithmeticOverflow;
    return plan;
  }
  plan.inverse_alpha_elements = input_size;
  plan.packed_weight_offset = 0U;

  if (!prefill_a4_factorized_lane_contract_detail::checked_align_up(
          plan.packed_weight_bytes, alignment, plan.weight_scale_offset)) {
    plan.error = PrefillA4FactorizedLanePlanError::kArithmeticOverflow;
    return plan;
  }
  std::uint64_t scale_end = 0U;
  if (!prefill_a4_factorized_lane_contract_detail::checked_add(
          plan.weight_scale_offset, plan.weight_scale_bytes, scale_end) ||
      !prefill_a4_factorized_lane_contract_detail::checked_align_up(
          scale_end, alignment, plan.metadata_offset) ||
      !prefill_a4_factorized_lane_contract_detail::checked_add(
          plan.metadata_offset,
          kPrefillA4FactorizedLaneMetadataDigestOffset,
          plan.metadata_digest_offset) ||
      !prefill_a4_factorized_lane_contract_detail::checked_add(
          plan.metadata_offset,
          kPrefillA4FactorizedLaneMetadataHeaderBytes,
          plan.inverse_alpha_offset)) {
    plan.error = PrefillA4FactorizedLanePlanError::kArithmeticOverflow;
    return plan;
  }
  std::uint64_t metadata_end = 0U;
  if (!prefill_a4_factorized_lane_contract_detail::checked_add(
          plan.metadata_offset, plan.metadata_bytes, metadata_end) ||
      !prefill_a4_factorized_lane_contract_detail::checked_align_up(
          metadata_end, alignment, plan.projection_bytes)) {
    plan.error = PrefillA4FactorizedLanePlanError::kArithmeticOverflow;
    return plan;
  }

  plan.error = PrefillA4FactorizedLanePlanError::kNone;
  return plan;
}

// Weight scales use [N/64][lane][64] BF16 consumer order.  Invalid plans or
// coordinates return a sentinel instead of wrapping an offset.
[[nodiscard]] constexpr std::uint64_t
prefill_a4_factorized_lane_weight_scale_element_offset(
    const PrefillA4FactorizedLaneProjectionLayoutPlan& plan,
    const std::uint64_t output_coordinate,
    const std::uint32_t lane) noexcept {
  if (!plan.valid() || output_coordinate >= plan.output_size ||
      lane >= plan.lane_count) {
    return kPrefillA4FactorizedLaneInvalidOffset;
  }
  return ((output_coordinate / kPrefillA4FactorizedLaneOuterBlock) *
              plan.lane_count +
          lane) *
             kPrefillA4FactorizedLaneOuterBlock +
         output_coordinate % kPrefillA4FactorizedLaneOuterBlock;
}

// Returns an absolute byte offset within the projection payload.
[[nodiscard]] constexpr std::uint64_t
prefill_a4_factorized_lane_inverse_alpha_byte_offset(
    const PrefillA4FactorizedLaneProjectionLayoutPlan& plan,
    const std::uint64_t input_coordinate) noexcept {
  if (!plan.valid() || input_coordinate >= plan.input_size) {
    return kPrefillA4FactorizedLaneInvalidOffset;
  }
  return plan.inverse_alpha_offset +
         input_coordinate * kPrefillA4FactorizedLaneFp32Bytes;
}

static_assert(kPrefillA4FactorizedLaneMetadataDigestOffset +
                      kPrefillA4FactorizedLaneMetadataDigestBytes ==
                  kPrefillA4FactorizedLaneMetadataHeaderBytes,
              "inverse-alpha digest must close the fixed metadata header");
static_assert(sizeof(kPrefillA4FactorizedLaneMetadataMagic) == 9U &&
                  kPrefillA4FactorizedLaneMetadataMagic[8U] == '\0',
              "metadata magic includes eight ABI bytes and one C terminator");
static_assert(
    prefill_a4_factorized_lane_projection_layout_plan(17'408U, 5'120U,
                                                       1U)
        .valid(),
    "the pinned Gate shape must admit the independent R1 host ABI");
static_assert(
    prefill_a4_factorized_lane_projection_layout_plan(5'120U, 17'408U,
                                                       4U)
        .valid(),
    "the pinned Down shape must admit the independent R4 host ABI");

}  // namespace q3x::runtime
