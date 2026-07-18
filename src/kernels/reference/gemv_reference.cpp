#include "q3x/kernels/reference_gemv.h"

#include "q3x/quantization/nvfp4.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace q3x::kernels {
namespace {

static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "reference GEMV requires IEEE 754 binary32 float");

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float decoded = 0.0F;
  static_assert(sizeof(decoded) == sizeof(bits), "float must be binary32");
  std::memcpy(&decoded, &bits, sizeof(decoded));
  return decoded;
}

[[nodiscard]] bool element_count_overflows(const std::size_t rows,
                                           const std::size_t columns) noexcept {
  return columns != 0U &&
         rows > std::numeric_limits<std::size_t>::max() / columns;
}

[[nodiscard]] bool is_empty(const std::size_t rows,
                            const std::size_t columns) noexcept {
  return rows == 0U || columns == 0U;
}

[[nodiscard]] float quantize_e4m3fn_to_float(
    const float value, const float scale) noexcept {
  if (std::isnan(value)) {
    return value;
  }
  const float inverse_scale = 1.0F / scale;
  const float scaled =
      std::fmax(-448.0F, std::fmin(value * inverse_scale, 448.0F));
  const bool negative = std::signbit(scaled);
  const float magnitude = std::fabs(scaled);
  std::uint8_t best_code = 0U;
  float best_distance = magnitude;
  for (std::uint16_t candidate = 1U; candidate <= 0x7eU; ++candidate) {
    const auto code = static_cast<std::uint8_t>(candidate);
    const float decoded = q3x::quantization::decode_e4m3fn(code);
    const float distance = std::fabs(magnitude - decoded);
    if (distance < best_distance ||
        (distance == best_distance && (code & 1U) == 0U &&
         (best_code & 1U) != 0U)) {
      best_code = code;
      best_distance = distance;
    }
  }
  if (negative) {
    best_code = static_cast<std::uint8_t>(best_code | 0x80U);
  }
  return q3x::quantization::decode_e4m3fn(best_code);
}

}  // namespace

const char* gemv_status_string(const GemvStatus status) noexcept {
  switch (status) {
    case GemvStatus::kSuccess:
      return "success";
    case GemvStatus::kInvalidArgument:
      return "invalid argument";
    case GemvStatus::kInvalidColumnCount:
      return "NVFP4 column count must be a multiple of 16";
    case GemvStatus::kSizeOverflow:
      return "matrix element count overflows size_t";
  }
  return "unknown GEMV status";
}

GemvStatus bf16_gemv_reference_cpu(
    const std::uint16_t* const weights,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    float* const output) noexcept {
  if (element_count_overflows(rows, columns)) {
    return GemvStatus::kSizeOverflow;
  }
  if (is_empty(rows, columns)) {
    return GemvStatus::kSuccess;
  }
  if (weights == nullptr || activation == nullptr || output == nullptr) {
    return GemvStatus::kInvalidArgument;
  }

  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    const std::size_t row_offset = row * columns;
    for (std::size_t column = 0; column < columns; ++column) {
      sum += decode_bf16(weights[row_offset + column]) *
             decode_bf16(activation[column]);
    }
    output[row] = sum;
  }
  return GemvStatus::kSuccess;
}

GemvStatus fp8_gemv_reference_cpu(
    const std::uint8_t* const weights,
    const float weight_scale,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    float* const output) noexcept {
  if (!std::isfinite(weight_scale) || weight_scale < 0.0F) {
    return GemvStatus::kInvalidArgument;
  }
  if (element_count_overflows(rows, columns)) {
    return GemvStatus::kSizeOverflow;
  }
  if (is_empty(rows, columns)) {
    return GemvStatus::kSuccess;
  }
  if (weights == nullptr || activation == nullptr || output == nullptr) {
    return GemvStatus::kInvalidArgument;
  }

  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    const std::size_t row_offset = row * columns;
    for (std::size_t column = 0; column < columns; ++column) {
      const float weight = q3x::quantization::decode_e4m3fn(
                               weights[row_offset + column]) *
                           weight_scale;
      sum += weight * decode_bf16(activation[column]);
    }
    output[row] = sum;
  }
  return GemvStatus::kSuccess;
}

GemvStatus fp8_static_gemv_reference_cpu(
    const std::uint8_t* const weights,
    const float weight_scale,
    const float input_scale,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    float* const output) noexcept {
  if (!std::isfinite(weight_scale) || weight_scale < 0.0F ||
      !std::isfinite(input_scale) || input_scale <= 0.0F) {
    return GemvStatus::kInvalidArgument;
  }
  if (element_count_overflows(rows, columns)) {
    return GemvStatus::kSizeOverflow;
  }
  if (is_empty(rows, columns)) {
    return GemvStatus::kSuccess;
  }
  if (weights == nullptr || activation == nullptr || output == nullptr) {
    return GemvStatus::kInvalidArgument;
  }

  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    const std::size_t row_offset = row * columns;
    for (std::size_t column = 0; column < columns; ++column) {
      const float weight = q3x::quantization::decode_e4m3fn(
          weights[row_offset + column]);
      const float quantized_input = quantize_e4m3fn_to_float(
          decode_bf16(activation[column]), input_scale);
      sum += weight * quantized_input;
    }
    output[row] = sum * weight_scale * input_scale;
  }
  return GemvStatus::kSuccess;
}

GemvStatus nvfp4_gemv_reference_cpu(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    const float weight_scale_2,
    const std::uint16_t* const activation,
    const std::size_t rows,
    const std::size_t columns,
    float* const output) noexcept {
  if (columns % q3x::quantization::kNvFp4GroupSize != 0U) {
    return GemvStatus::kInvalidColumnCount;
  }
  if (!std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F) {
    return GemvStatus::kInvalidArgument;
  }
  if (element_count_overflows(rows, columns)) {
    return GemvStatus::kSizeOverflow;
  }
  if (is_empty(rows, columns)) {
    return GemvStatus::kSuccess;
  }
  if (packed_weights == nullptr || block_scales == nullptr ||
      activation == nullptr || output == nullptr) {
    return GemvStatus::kInvalidArgument;
  }

  const std::size_t packed_row_stride =
      columns / q3x::quantization::kNvFp4ValuesPerByte;
  const std::size_t scale_row_stride =
      columns / q3x::quantization::kNvFp4GroupSize;
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    const std::size_t packed_row_offset = row * packed_row_stride;
    const std::size_t scale_row_offset = row * scale_row_stride;
    for (std::size_t column = 0; column < columns; ++column) {
      const std::uint8_t packed =
          packed_weights[packed_row_offset + column / 2U];
      const std::uint8_t scale =
          block_scales[scale_row_offset +
                       column / q3x::quantization::kNvFp4GroupSize];
      const float weight = q3x::quantization::dequantize_nvfp4_value(
          packed, (column & 1U) != 0U, scale, weight_scale_2);
      sum += weight * decode_bf16(activation[column]);
    }
    output[row] = sum;
  }
  return GemvStatus::kSuccess;
}

}  // namespace q3x::kernels
