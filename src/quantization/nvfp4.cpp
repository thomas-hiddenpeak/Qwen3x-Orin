#include "q3x/quantization/nvfp4.h"

#include <cmath>
#include <limits>

namespace q3x::quantization {
namespace {

constexpr float kE2m1Values[16] = {
    0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
    -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
};

}  // namespace

std::uint8_t unpack_e2m1_nibble(
    const std::uint8_t packed,
    const bool high_nibble) noexcept {
    return high_nibble ? static_cast<std::uint8_t>((packed >> 4U) & 0x0fU)
                       : static_cast<std::uint8_t>(packed & 0x0fU);
}

float decode_e2m1(const std::uint8_t nibble) noexcept {
    return kE2m1Values[nibble & 0x0fU];
}

float decode_e4m3fn(const std::uint8_t bits) noexcept {
    const std::uint8_t magnitude = bits & 0x7fU;
    const int exponent = static_cast<int>((magnitude >> 3U) & 0x0fU);
    const int mantissa = static_cast<int>(magnitude & 0x07U);

    if (exponent == 0x0f && mantissa == 0x07) {
        return std::copysign(
            std::numeric_limits<float>::quiet_NaN(),
            (bits & 0x80U) != 0U ? -1.0F : 1.0F);
    }

    float value = 0.0F;
    if (exponent == 0) {
        // Subnormal: (mantissa / 8) * 2^(1 - bias), bias = 7.
        value = std::ldexp(static_cast<float>(mantissa), -9);
    } else {
        // E4M3FN extends exponent 0xf with finite values through 448. The
        // sole NaN mantissa was handled above.
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           exponent - 7);
    }

    return std::copysign(value, (bits & 0x80U) != 0U ? -1.0F : 1.0F);
}

float dequantize_nvfp4_value(
    const std::uint8_t packed,
    const bool high_nibble,
    const std::uint8_t weight_scale,
    const float weight_scale_2) noexcept {
    return decode_e2m1(unpack_e2m1_nibble(packed, high_nibble)) *
           decode_e4m3fn(weight_scale) * weight_scale_2;
}

const char* nvfp4_status_string(const NvFp4Status status) noexcept {
    switch (status) {
        case NvFp4Status::kSuccess:
            return "success";
        case NvFp4Status::kInvalidArgument:
            return "invalid argument";
        case NvFp4Status::kInvalidColumnCount:
            return "column count must be a multiple of 16";
        case NvFp4Status::kSizeOverflow:
            return "matrix element count overflows size_t";
    }
    return "unknown NVFP4 status";
}

NvFp4Status dequantize_nvfp4_reference(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const weight_scales,
    const float weight_scale_2,
    const std::size_t rows,
    const std::size_t columns,
    float* const output) noexcept {
    if (columns % kNvFp4GroupSize != 0U) {
        return NvFp4Status::kInvalidColumnCount;
    }
    if (!std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F) {
        return NvFp4Status::kInvalidArgument;
    }
    if (columns != 0U && rows > std::numeric_limits<std::size_t>::max() / columns) {
        return NvFp4Status::kSizeOverflow;
    }
    if (rows == 0U || columns == 0U) {
        return NvFp4Status::kSuccess;
    }
    if (packed_weights == nullptr || weight_scales == nullptr || output == nullptr) {
        return NvFp4Status::kInvalidArgument;
    }

    const std::size_t packed_row_stride = columns / kNvFp4ValuesPerByte;
    const std::size_t scale_row_stride = columns / kNvFp4GroupSize;

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const std::uint8_t packed =
                packed_weights[row * packed_row_stride + column / 2U];
            const std::uint8_t scale =
                weight_scales[row * scale_row_stride + column / kNvFp4GroupSize];
            output[row * columns + column] = dequantize_nvfp4_value(
                packed, (column & 1U) != 0U, scale, weight_scale_2);
        }
    }

    return NvFp4Status::kSuccess;
}

}  // namespace q3x::quantization
