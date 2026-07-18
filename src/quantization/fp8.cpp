#include "q3x/quantization/fp8.h"

#include "q3x/quantization/nvfp4.h"

#include <cmath>
#include <limits>

namespace q3x::quantization {

const char* fp8_status_string(const Fp8Status status) noexcept {
  switch (status) {
    case Fp8Status::kSuccess:
      return "success";
    case Fp8Status::kInvalidArgument:
      return "invalid argument";
    case Fp8Status::kSizeOverflow:
      return "matrix element count overflows size_t";
  }
  return "unknown FP8 status";
}

Fp8Status dequantize_fp8_reference(const std::uint8_t* const weights,
                                   const float weight_scale,
                                   const std::size_t rows,
                                   const std::size_t columns,
                                   float* const output) noexcept {
  if (!std::isfinite(weight_scale) || weight_scale < 0.0F) {
    return Fp8Status::kInvalidArgument;
  }
  if (columns != 0U &&
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    return Fp8Status::kSizeOverflow;
  }
  if (rows == 0U || columns == 0U) {
    return Fp8Status::kSuccess;
  }
  if (weights == nullptr || output == nullptr) {
    return Fp8Status::kInvalidArgument;
  }

  const std::size_t total = rows * columns;
  for (std::size_t index = 0; index < total; ++index) {
    output[index] = decode_e4m3fn(weights[index]) * weight_scale;
  }
  return Fp8Status::kSuccess;
}

}  // namespace q3x::quantization
