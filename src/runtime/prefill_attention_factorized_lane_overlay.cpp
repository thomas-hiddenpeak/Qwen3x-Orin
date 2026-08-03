#include "q3x/runtime/prefill_attention_factorized_lane_overlay.h"

#include "q3x/core/sha256.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace q3x::runtime {
namespace {

[[nodiscard]] bool supported_lane_count(
    const std::uint32_t lane_count) noexcept {
  return prefill_attention_factorized_lane_qualification_role(lane_count) !=
         PrefillAttentionFactorizedLaneQualificationRole::kUnsupported;
}

[[nodiscard]] bool structurally_valid_input_size(
    const std::uint64_t input_size,
    const std::uint32_t lane_count) noexcept {
  return input_size != 0U && supported_lane_count(lane_count) &&
         input_size % kPrefillA4FactorizedLanePackedKBlock == 0U &&
         input_size % lane_count == 0U &&
         (input_size / lane_count) %
                 kPrefillA4FactorizedLanePackedKBlock ==
             0U;
}

void write_u32_little_endian(const std::uint32_t value,
                             std::uint8_t* const output) noexcept {
  output[0] = static_cast<std::uint8_t>(value);
  output[1] = static_cast<std::uint8_t>(value >> 8U);
  output[2] = static_cast<std::uint8_t>(value >> 16U);
  output[3] = static_cast<std::uint8_t>(value >> 24U);
}

void write_u64_little_endian(const std::uint64_t value,
                             std::uint8_t* const output) noexcept {
  for (std::size_t index = 0U; index < 8U; ++index) {
    output[index] =
        static_cast<std::uint8_t>(value >> (index * 8U));
  }
}

[[nodiscard]] std::uint32_t read_u32_little_endian(
    const std::uint8_t* const input) noexcept {
  return static_cast<std::uint32_t>(input[0]) |
         (static_cast<std::uint32_t>(input[1]) << 8U) |
         (static_cast<std::uint32_t>(input[2]) << 16U) |
         (static_cast<std::uint32_t>(input[3]) << 24U);
}

[[nodiscard]] std::uint64_t read_u64_little_endian(
    const std::uint8_t* const input) noexcept {
  std::uint64_t result = 0U;
  for (std::size_t index = 0U; index < 8U; ++index) {
    result |= static_cast<std::uint64_t>(input[index]) << (index * 8U);
  }
  return result;
}

void write_float_little_endian(const float value,
                               std::uint8_t* const output) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  write_u32_little_endian(bits, output);
}

[[nodiscard]] float read_float_little_endian(
    const std::uint8_t* const input) noexcept {
  const std::uint32_t bits = read_u32_little_endian(input);
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] std::array<
    std::uint8_t, kPrefillA4FactorizedLaneMetadataDigestBytes>
sha256_bytes(const std::uint8_t* const bytes,
             const std::size_t byte_count) noexcept {
  core::Sha256 hasher;
  (void)hasher.update(bytes, byte_count);
  return hasher.finalize().bytes;
}

[[nodiscard]] bool digest_equal(
    const std::array<std::uint8_t,
                     kPrefillA4FactorizedLaneMetadataDigestBytes>& left,
    const std::uint8_t* const right) noexcept {
  // Do not turn the first differing byte into an observable early exit.  The
  // digest is an integrity boundary today and may become part of an
  // authenticated publication boundary later.
  std::uint8_t difference = 0U;
  for (std::size_t index = 0U; index < left.size(); ++index) {
    difference = static_cast<std::uint8_t>(difference |
                                           (left[index] ^ right[index]));
  }
  return difference == 0U;
}

[[nodiscard]] bool metadata_byte_count(
    const std::uint64_t input_size, std::uint64_t& inverse_alpha_bytes,
    std::uint64_t& metadata_bytes) noexcept {
  using namespace prefill_a4_factorized_lane_contract_detail;
  return checked_multiply(input_size,
                          kPrefillA4FactorizedLaneFp32Bytes,
                          inverse_alpha_bytes) &&
         checked_add(kPrefillA4FactorizedLaneMetadataHeaderBytes,
                     inverse_alpha_bytes, metadata_bytes);
}

}  // namespace

PrefillAttentionFactorizedLaneMetadataSerializationResult
serialize_prefill_attention_factorized_lane_metadata(
    const std::uint32_t lane_count, const float* const inverse_alpha,
    const std::size_t inverse_alpha_count) {
  PrefillAttentionFactorizedLaneMetadataSerializationResult result;
  if (!supported_lane_count(lane_count)) {
    result.error =
        PrefillAttentionFactorizedLaneMetadataError::kUnsupportedLaneCount;
    return result;
  }
  if (inverse_alpha == nullptr && inverse_alpha_count != 0U) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kNullInput;
    return result;
  }
  if (inverse_alpha_count >
      static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kArithmeticOverflow;
    return result;
  }
  const std::uint64_t input_size =
      static_cast<std::uint64_t>(inverse_alpha_count);
  if (!structurally_valid_input_size(input_size, lane_count)) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kInvalidInputSize;
    return result;
  }
  for (std::size_t index = 0U; index < inverse_alpha_count; ++index) {
    if (!std::isfinite(inverse_alpha[index]) ||
        !(inverse_alpha[index] > 0.0F)) {
      result.error =
          PrefillAttentionFactorizedLaneMetadataError::kInvalidInverseAlpha;
      return result;
    }
  }

  std::uint64_t inverse_alpha_bytes = 0U;
  std::uint64_t metadata_bytes = 0U;
  if (!metadata_byte_count(input_size, inverse_alpha_bytes, metadata_bytes) ||
      metadata_bytes >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max())) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kArithmeticOverflow;
    return result;
  }
  try {
    result.bytes.assign(static_cast<std::size_t>(metadata_bytes), 0U);
  } catch (const std::bad_alloc&) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kAllocationFailure;
    return result;
  } catch (const std::length_error&) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kAllocationFailure;
    return result;
  }

  std::copy_n(
      reinterpret_cast<const std::uint8_t*>(
          kPrefillA4FactorizedLaneMetadataMagic),
      static_cast<std::size_t>(kPrefillA4FactorizedLaneMetadataMagicBytes),
      result.bytes.data() + kPrefillA4FactorizedLaneMetadataMagicOffset);
  write_u32_little_endian(
      kPrefillA4FactorizedLaneContractVersionMajor,
      result.bytes.data() +
          kPrefillA4FactorizedLaneMetadataVersionMajorOffset);
  write_u32_little_endian(
      kPrefillA4FactorizedLaneContractVersionMinor,
      result.bytes.data() +
          kPrefillA4FactorizedLaneMetadataVersionMinorOffset);
  write_u32_little_endian(
      lane_count,
      result.bytes.data() +
          kPrefillA4FactorizedLaneMetadataLaneCountOffset);
  write_u32_little_endian(
      kPrefillA4FactorizedLaneInverseAlphaFp32Le,
      result.bytes.data() +
          kPrefillA4FactorizedLaneMetadataEncodingOffset);
  write_u64_little_endian(
      input_size,
      result.bytes.data() +
          kPrefillA4FactorizedLaneMetadataInputSizeOffset);

  auto* const payload =
      result.bytes.data() + kPrefillA4FactorizedLaneMetadataHeaderBytes;
  for (std::size_t index = 0U; index < inverse_alpha_count; ++index) {
    write_float_little_endian(
        inverse_alpha[index],
        payload + index * kPrefillA4FactorizedLaneFp32Bytes);
  }
  result.inverse_alpha_sha256 = sha256_bytes(
      payload, static_cast<std::size_t>(inverse_alpha_bytes));
  std::copy(result.inverse_alpha_sha256.begin(),
            result.inverse_alpha_sha256.end(),
            result.bytes.begin() + static_cast<std::ptrdiff_t>(
                                       kPrefillA4FactorizedLaneMetadataDigestOffset));
  result.error = PrefillAttentionFactorizedLaneMetadataError::kNone;
  return result;
}

PrefillAttentionFactorizedLaneMetadataParseResult
parse_prefill_attention_factorized_lane_metadata(
    const std::uint8_t* const bytes, const std::size_t byte_count,
    const std::uint32_t expected_lane_count,
    const std::uint64_t expected_input_size) {
  PrefillAttentionFactorizedLaneMetadataParseResult result;
  if (bytes == nullptr && byte_count != 0U) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kNullInput;
    return result;
  }
  if (byte_count < kPrefillA4FactorizedLaneMetadataHeaderBytes) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kInvalidByteLength;
    return result;
  }
  if (!std::equal(
          reinterpret_cast<const std::uint8_t*>(
              kPrefillA4FactorizedLaneMetadataMagic),
          reinterpret_cast<const std::uint8_t*>(
              kPrefillA4FactorizedLaneMetadataMagic) +
              kPrefillA4FactorizedLaneMetadataMagicBytes,
          bytes + kPrefillA4FactorizedLaneMetadataMagicOffset)) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kInvalidMagic;
    return result;
  }
  const std::uint32_t version_major = read_u32_little_endian(
      bytes + kPrefillA4FactorizedLaneMetadataVersionMajorOffset);
  const std::uint32_t version_minor = read_u32_little_endian(
      bytes + kPrefillA4FactorizedLaneMetadataVersionMinorOffset);
  if (version_major != kPrefillA4FactorizedLaneContractVersionMajor ||
      version_minor != kPrefillA4FactorizedLaneContractVersionMinor) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kUnsupportedVersion;
    return result;
  }
  result.lane_count = read_u32_little_endian(
      bytes + kPrefillA4FactorizedLaneMetadataLaneCountOffset);
  const std::uint32_t encoding = read_u32_little_endian(
      bytes + kPrefillA4FactorizedLaneMetadataEncodingOffset);
  if (encoding != kPrefillA4FactorizedLaneInverseAlphaFp32Le) {
    result.error =
        PrefillAttentionFactorizedLaneMetadataError::kUnsupportedEncoding;
    return result;
  }
  result.input_size = read_u64_little_endian(
      bytes + kPrefillA4FactorizedLaneMetadataInputSizeOffset);
  if (!supported_lane_count(result.lane_count)) {
    result.error =
        PrefillAttentionFactorizedLaneMetadataError::kUnsupportedLaneCount;
    return result;
  }
  if (!structurally_valid_input_size(result.input_size,
                                     result.lane_count)) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kInvalidInputSize;
    return result;
  }
  if (!supported_lane_count(expected_lane_count) ||
      !structurally_valid_input_size(expected_input_size,
                                     expected_lane_count) ||
      result.lane_count != expected_lane_count ||
      result.input_size != expected_input_size) {
    result.error =
        PrefillAttentionFactorizedLaneMetadataError::kExpectedShapeMismatch;
    return result;
  }

  std::uint64_t inverse_alpha_bytes = 0U;
  std::uint64_t metadata_bytes = 0U;
  if (!metadata_byte_count(result.input_size, inverse_alpha_bytes,
                           metadata_bytes) ||
      metadata_bytes >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max())) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kArithmeticOverflow;
    return result;
  }
  if (byte_count != static_cast<std::size_t>(metadata_bytes)) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kInvalidByteLength;
    return result;
  }

  const auto* const payload =
      bytes + kPrefillA4FactorizedLaneMetadataHeaderBytes;
  result.inverse_alpha_sha256 = sha256_bytes(
      payload, static_cast<std::size_t>(inverse_alpha_bytes));
  if (!digest_equal(
          result.inverse_alpha_sha256,
          bytes + kPrefillA4FactorizedLaneMetadataDigestOffset)) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kDigestMismatch;
    return result;
  }
  try {
    result.inverse_alpha.resize(
        static_cast<std::size_t>(result.input_size));
  } catch (const std::bad_alloc&) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kAllocationFailure;
    return result;
  } catch (const std::length_error&) {
    result.error = PrefillAttentionFactorizedLaneMetadataError::kAllocationFailure;
    return result;
  }
  for (std::size_t index = 0U; index < result.inverse_alpha.size(); ++index) {
    const float value = read_float_little_endian(
        payload + index * kPrefillA4FactorizedLaneFp32Bytes);
    if (!std::isfinite(value) || !(value > 0.0F)) {
      result.inverse_alpha.clear();
      result.error =
          PrefillAttentionFactorizedLaneMetadataError::kInvalidInverseAlpha;
      return result;
    }
    result.inverse_alpha[index] = value;
  }
  result.error = PrefillAttentionFactorizedLaneMetadataError::kNone;
  return result;
}

}  // namespace q3x::runtime
