#include "q3x/runtime/prefill_mlp_factorized_lane_r4_converter.h"

#include "q3x/quantization/nvfp4.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

[[nodiscard]] PrefillMLPFactorizedLaneR4ConverterDiagnostic make_diagnostic(
    const PrefillMLPFactorizedLaneR4ConverterErrorCode code,
    std::string context, std::string message,
    const std::size_t index = 0U) {
  PrefillMLPFactorizedLaneR4ConverterDiagnostic result;
  result.code = code;
  result.context = std::move(context);
  result.message = std::move(message);
  result.index = index;
  return result;
}

[[nodiscard]] bool checked_multiply(const std::size_t left,
                                    const std::size_t right,
                                    std::size_t& output) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

[[nodiscard]] bool spans_overlap(const void* const left,
                                 const std::size_t left_bytes,
                                 const void* const right,
                                 const std::size_t right_bytes) noexcept {
  const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
  const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
  if (left_bytes > std::numeric_limits<std::uintptr_t>::max() - left_begin ||
      right_bytes >
          std::numeric_limits<std::uintptr_t>::max() - right_begin) {
    return true;
  }
  const auto left_end = left_begin + left_bytes;
  const auto right_end = right_begin + right_bytes;
  return left_begin < right_end && right_begin < left_end;
}

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] float bits_float(const std::uint32_t bits) noexcept {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] std::uint16_t float_to_bf16_nearest_even(
    const float value) noexcept {
  std::uint32_t bits = float_bits(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float bf16_to_float(const std::uint16_t value) noexcept {
  return bits_float(static_cast<std::uint32_t>(value) << 16U);
}

void write_u16_little_endian(const std::uint16_t value,
                             std::uint8_t* const output) noexcept {
  output[0] = static_cast<std::uint8_t>(value);
  output[1] = static_cast<std::uint8_t>(value >> 8U);
}

[[nodiscard]] int round_nearest_even(const float value) noexcept {
  const float floor_value = std::floor(value);
  const float fraction = value - floor_value;
  if (fraction < 0.5F) {
    return static_cast<int>(floor_value);
  }
  if (fraction > 0.5F) {
    return static_cast<int>(floor_value) + 1;
  }
  const int floor_integer = static_cast<int>(floor_value);
  return (floor_integer & 1) == 0 ? floor_integer : floor_integer + 1;
}

[[nodiscard]] bool valid_clip_ratio(const double value) noexcept {
  const float narrowed = static_cast<float>(value);
  return std::isfinite(value) &&
         value >= kPrefillMLPFactorizedLaneR4MinimumClipRatio &&
         value <= 1.0 && std::isfinite(narrowed) &&
         narrowed >=
             static_cast<float>(
                 kPrefillMLPFactorizedLaneR4MinimumClipRatio) &&
         narrowed <= 1.0F;
}

[[nodiscard]] PrefillMLPFactorizedLaneR4ConverterDiagnostic validate_alpha(
    const float* const alpha, const std::size_t alpha_count,
    const std::size_t expected_count) {
  if (alpha == nullptr || alpha_count != expected_count) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidAlpha,
        "r4_direct.alpha",
        "alpha must contain exactly one FP32 value per input channel");
  }
  for (std::size_t index = 0U; index < alpha_count; ++index) {
    if (!std::isfinite(alpha[index]) || !(alpha[index] > 0.0F)) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidAlpha,
          "r4_direct.alpha",
          "alpha values must be finite and strictly positive", index);
    }
    const float inverse = 1.0F / alpha[index];
    if (!std::isfinite(inverse) || !(inverse > 0.0F)) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidAlpha,
          "r4_direct.inverse_alpha",
          "the FP32 reciprocal of alpha must be finite and positive", index);
    }
  }
  return {};
}

}  // namespace

PrefillMLPFactorizedLaneR4MetadataResult
build_prefill_mlp_factorized_lane_r4_metadata(
    const float* const alpha, const std::size_t alpha_count) {
  PrefillMLPFactorizedLaneR4MetadataResult result;
  const auto shape = prefill_a4_factorized_lane_projection_layout_plan(
      kPrefillA4FactorizedLaneOuterBlock, alpha_count,
      kPrefillMLPFactorizedLaneR4LaneCount);
  if (!shape) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidShape,
        "r4_metadata.shape",
        "R4 K must split into four nonempty K64-aligned lanes");
    return result;
  }
  result.diagnostic = validate_alpha(alpha, alpha_count, alpha_count);
  if (!result.diagnostic) {
    return result;
  }

  try {
    std::vector<float> inverse_alpha(alpha_count);
    for (std::size_t index = 0U; index < alpha_count; ++index) {
      inverse_alpha[index] = 1.0F / alpha[index];
    }
    result.metadata = serialize_prefill_mlp_factorized_lane_metadata(
        kPrefillMLPFactorizedLaneR4LaneCount, inverse_alpha.data(),
        inverse_alpha.size());
    if (!result.metadata) {
      result.diagnostic = make_diagnostic(
          result.metadata.error ==
                  PrefillMLPFactorizedLaneMetadataError::kAllocationFailure
              ? PrefillMLPFactorizedLaneR4ConverterErrorCode::
                    kAllocationFailure
              : PrefillMLPFactorizedLaneR4ConverterErrorCode::
                    kMetadataFailure,
          "r4_metadata.serialize",
          "factorized-lane v4 metadata serialization failed");
    }
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kAllocationFailure,
        "r4_metadata.inverse_alpha",
        "inverse-alpha allocation failed");
    return result;
  } catch (const std::length_error&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kAllocationFailure,
        "r4_metadata.inverse_alpha",
        "inverse-alpha allocation exceeded the host container limit");
    return result;
  }
}

PrefillMLPFactorizedLaneR4ConverterDiagnostic
transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
    const std::uint8_t* const source_packed_nvfp4,
    const std::size_t source_packed_nvfp4_bytes,
    const std::uint8_t* const source_e4m3_scales,
    const std::size_t source_e4m3_scale_bytes,
    const float source_weight_scale_2, const std::size_t row_count,
    const std::size_t input_size, const float* const alpha,
    const std::size_t alpha_count, const double weight_clip_ratio,
    std::uint8_t* const r4_packed_signed_w4,
    const std::size_t r4_packed_signed_w4_bytes,
    std::uint8_t* const r4_bf16_scales_little_endian,
    const std::size_t r4_bf16_scale_bytes) {
  if (source_packed_nvfp4 == nullptr || source_e4m3_scales == nullptr ||
      r4_packed_signed_w4 == nullptr ||
      r4_bf16_scales_little_endian == nullptr) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidArgument,
        "r4_direct.buffers", "all source and output buffers are required");
  }
  const auto shape = prefill_a4_factorized_lane_projection_layout_plan(
      row_count, input_size, kPrefillMLPFactorizedLaneR4LaneCount);
  if (!shape) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidShape,
        "r4_direct.shape",
        "R4 conversion requires complete N64 blocks and four K64-aligned lanes");
  }
  if (!valid_clip_ratio(weight_clip_ratio)) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidArgument,
        "r4_direct.weight_clip_ratio",
        "an explicit finite clip ratio in [1/256,1] is required");
  }
  if (!std::isfinite(source_weight_scale_2) ||
      source_weight_scale_2 < 0.0F) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidSourceValue,
        "r4_direct.weight_scale_2",
        "the original NVFP4 tensor scale must be finite and nonnegative");
  }
  if (alpha == nullptr || alpha_count != input_size) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidAlpha,
        "r4_direct.alpha",
        "alpha must contain exactly one FP32 value per input channel");
  }
  std::size_t logical_elements = 0U;
  std::size_t expected_source_scales = 0U;
  std::size_t expected_r4_scales = 0U;
  std::size_t alpha_bytes = 0U;
  if (!checked_multiply(row_count, input_size, logical_elements) ||
      !checked_multiply(row_count,
                        kPrefillMLPFactorizedLaneR4LaneCount,
                        expected_r4_scales) ||
      !checked_multiply(expected_r4_scales, sizeof(std::uint16_t),
                        expected_r4_scales) ||
      !checked_multiply(input_size, sizeof(float), alpha_bytes)) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kArithmeticOverflow,
        "r4_direct.shape", "R4 buffer byte count overflowed size_t");
  }
  const std::size_t expected_packed = logical_elements / 2U;
  expected_source_scales = logical_elements / quantization::kNvFp4GroupSize;
  if (source_packed_nvfp4_bytes != expected_packed ||
      source_e4m3_scale_bytes != expected_source_scales ||
      r4_packed_signed_w4_bytes != expected_packed ||
      r4_bf16_scale_bytes != expected_r4_scales) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kBufferSizeMismatch,
        "r4_direct.buffers",
        "source and R4 buffers must have their exact shape-derived byte lengths");
  }
  const struct BufferSpan final {
    const void* data;
    std::size_t bytes;
  } buffers[] = {
      {source_packed_nvfp4, source_packed_nvfp4_bytes},
      {source_e4m3_scales, source_e4m3_scale_bytes},
      {alpha, alpha_bytes},
      {r4_packed_signed_w4, r4_packed_signed_w4_bytes},
      {r4_bf16_scales_little_endian, r4_bf16_scale_bytes},
  };
  for (std::size_t left = 0U; left < std::size(buffers); ++left) {
    for (std::size_t right = left + 1U; right < std::size(buffers); ++right) {
      if (spans_overlap(buffers[left].data, buffers[left].bytes,
                        buffers[right].data, buffers[right].bytes)) {
        return make_diagnostic(
            PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidArgument,
            "r4_direct.alias",
            "source, alpha, packed output, and scale output spans must be pairwise disjoint",
            left * std::size(buffers) + right);
      }
    }
  }
  auto diagnostic = validate_alpha(alpha, alpha_count, input_size);
  if (!diagnostic) {
    return diagnostic;
  }

  const std::size_t packed_source_stride = input_size / 2U;
  const std::size_t source_scale_stride =
      input_size / quantization::kNvFp4GroupSize;
  const std::size_t packed_k64_blocks =
      input_size / kPrefillA4FactorizedLanePackedKBlock;
  const std::size_t lane_input_size =
      input_size / kPrefillMLPFactorizedLaneR4LaneCount;
  const float clip_ratio = static_cast<float>(weight_clip_ratio);

  auto decode_transformed_value = [&](const std::size_t row,
                                      const std::size_t k,
                                      float& output)
      -> PrefillMLPFactorizedLaneR4ConverterDiagnostic {
    const std::size_t scale_index = row * source_scale_stride +
                                    k / quantization::kNvFp4GroupSize;
    const std::uint8_t scale_bits = source_e4m3_scales[scale_index];
    const float block_scale = quantization::decode_e4m3fn(scale_bits);
    if (!std::isfinite(block_scale) || block_scale < 0.0F) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidSourceValue,
          "r4_direct.weight_scale",
          "original NVFP4 block scales must be finite and nonnegative",
          scale_index);
    }
    const std::size_t packed_index = row * packed_source_stride + k / 2U;
    const float decoded = quantization::dequantize_nvfp4_value(
        source_packed_nvfp4[packed_index], (k & 1U) != 0U, scale_bits,
        source_weight_scale_2);
    output = decoded * alpha[k];
    if (!std::isfinite(decoded) || !std::isfinite(output)) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidSourceValue,
          "r4_direct.transformed_weight",
          "directly decoded and alpha-transformed weights must be finite",
          row * input_size + k);
    }
    return {};
  };

  std::size_t decoded_elements = 0U;
  if (!checked_multiply(
          static_cast<std::size_t>(kPrefillA4FactorizedLaneOuterBlock),
          input_size, decoded_elements)) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kArithmeticOverflow,
        "r4_direct.decode_buffer",
        "bounded N64 decode buffer element count overflowed");
  }

  try {
    // One N64 block is decoded exactly once.  This stays bounded for a full
    // projection and avoids turning the direct R4 converter into a second
    // dequantization pass over the authenticated checkpoint bytes.
    std::vector<float> decoded(decoded_elements);
    const std::size_t n64_blocks =
        row_count / kPrefillA4FactorizedLaneOuterBlock;
    for (std::size_t n64 = 0U; n64 < n64_blocks; ++n64) {
      for (std::size_t local_n = 0U;
           local_n < kPrefillA4FactorizedLaneOuterBlock; ++local_n) {
        const std::size_t row =
            n64 * kPrefillA4FactorizedLaneOuterBlock + local_n;
        for (std::size_t k = 0U; k < input_size; ++k) {
          diagnostic = decode_transformed_value(
              row, k, decoded[local_n * input_size + k]);
          if (!diagnostic) {
            return diagnostic;
          }
        }
      }

      for (std::size_t local_n = 0U;
           local_n < kPrefillA4FactorizedLaneOuterBlock; ++local_n) {
        const std::size_t row =
            n64 * kPrefillA4FactorizedLaneOuterBlock + local_n;
        const float* const decoded_row =
            decoded.data() + local_n * input_size;
        for (std::size_t lane = 0U;
             lane < kPrefillMLPFactorizedLaneR4LaneCount; ++lane) {
          const std::size_t lane_begin = lane * lane_input_size;
          float maximum = 0.0F;
          for (std::size_t inner = 0U; inner < lane_input_size; ++inner) {
            maximum = std::max(
                maximum, std::fabs(decoded_row[lane_begin + inner]));
          }

          const float threshold = maximum * clip_ratio;
          std::uint16_t scale_bits = float_to_bf16_nearest_even(
              maximum == 0.0F ? 1.0F : threshold / 7.0F);
          float stored_scale = bf16_to_float(scale_bits);
          if (maximum != 0.0F && stored_scale == 0.0F) {
            scale_bits = 1U;
            stored_scale = bf16_to_float(scale_bits);
          }
          if (!std::isfinite(stored_scale) || !(stored_scale > 0.0F)) {
            return make_diagnostic(
                PrefillMLPFactorizedLaneR4ConverterErrorCode::
                    kQuantizationFailure,
                "r4_direct.lane_scale",
                "the rounded R4 lane BF16 scale must be finite and positive",
                (row * kPrefillMLPFactorizedLaneR4LaneCount) + lane);
          }
          const std::size_t scale_output_index =
              ((n64 * kPrefillMLPFactorizedLaneR4LaneCount + lane) *
                   kPrefillA4FactorizedLaneOuterBlock +
               local_n) *
              sizeof(std::uint16_t);
          write_u16_little_endian(
              scale_bits,
              r4_bf16_scales_little_endian + scale_output_index);

          for (std::size_t pair = 0U; pair < lane_input_size / 2U;
               ++pair) {
            const std::size_t even_k = lane_begin + pair * 2U;
            std::uint8_t encoded = 0U;
            for (std::size_t nibble = 0U; nibble < 2U; ++nibble) {
              const float value = decoded_row[even_k + nibble];
              const float clipped =
                  std::max(-threshold, std::min(threshold, value));
              int code = maximum == 0.0F
                             ? 0
                             : round_nearest_even(clipped / stored_scale);
              code = std::max(-7, std::min(7, code));
              encoded = static_cast<std::uint8_t>(
                  encoded |
                  static_cast<std::uint8_t>(
                      (static_cast<unsigned int>(code) & 0x0fU)
                      << (nibble * 4U)));
            }
            const std::size_t k64 =
                even_k / kPrefillA4FactorizedLanePackedKBlock;
            const std::size_t pair_in_k64 =
                (even_k % kPrefillA4FactorizedLanePackedKBlock) / 2U;
            const std::size_t packed_output_index =
                ((n64 * packed_k64_blocks + k64) *
                     kPrefillA4FactorizedLaneOuterBlock +
                 local_n) *
                    (kPrefillA4FactorizedLanePackedKBlock / 2U) +
                pair_in_k64;
            r4_packed_signed_w4[packed_output_index] = encoded;
          }
        }
      }
    }
  } catch (const std::bad_alloc&) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kAllocationFailure,
        "r4_direct.decode_buffer",
        "bounded N64 transformed-weight buffer allocation failed");
  } catch (const std::length_error&) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4ConverterErrorCode::kAllocationFailure,
        "r4_direct.decode_buffer",
        "bounded N64 transformed-weight buffer exceeded the host container limit");
  }
  return {};
}

std::string_view to_string(
    const PrefillMLPFactorizedLaneR4ConverterErrorCode code) noexcept {
  switch (code) {
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kNone:
      return "none";
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidArgument:
      return "invalid_argument";
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidShape:
      return "invalid_shape";
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kBufferSizeMismatch:
      return "buffer_size_mismatch";
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kArithmeticOverflow:
      return "arithmetic_overflow";
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidSourceValue:
      return "invalid_source_value";
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kInvalidAlpha:
      return "invalid_alpha";
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kQuantizationFailure:
      return "quantization_failure";
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kMetadataFailure:
      return "metadata_failure";
    case PrefillMLPFactorizedLaneR4ConverterErrorCode::kAllocationFailure:
      return "allocation_failure";
  }
  return "unknown";
}

}  // namespace q3x::runtime
