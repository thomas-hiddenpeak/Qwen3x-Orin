#include "q3x/quantization/nvfp4.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::quantization {
namespace {

[[nodiscard]] constexpr std::size_t ceil_divide(
    const std::size_t dividend,
    const std::size_t divisor) noexcept {
    return dividend / divisor + (dividend % divisor != 0U ? 1U : 0U);
}

static_assert(
    ceil_divide(std::numeric_limits<std::size_t>::max(), 256U) ==
        std::numeric_limits<std::size_t>::max() / 256U + 1U,
    "ceil_divide must not overflow at size_t maximum");

__device__ __forceinline__ float decode_e2m1_device(const std::uint8_t nibble) {
    constexpr float values[16] = {
        0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
        -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
    };
    return values[nibble & 0x0fU];
}

__device__ __forceinline__ float decode_e4m3fn_device(const std::uint8_t bits) {
    const std::uint8_t magnitude = bits & 0x7fU;
    const int exponent = static_cast<int>((magnitude >> 3U) & 0x0fU);
    const int mantissa = static_cast<int>(magnitude & 0x07U);

    if (exponent == 0x0f && mantissa == 0x07) {
        return nanf("");
    }

    float value = exponent == 0
                      ? ldexpf(static_cast<float>(mantissa), -9)
                      : ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F,
                               exponent - 7);
    return (bits & 0x80U) != 0U ? -value : value;
}

__global__ void dequantize_nvfp4_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const weight_scales,
    const float weight_scale_2,
    const std::size_t rows,
    const std::size_t columns,
    float* const output) {
    const std::size_t total = rows * columns;
    const std::size_t packed_row_stride = columns / kNvFp4ValuesPerByte;
    const std::size_t scale_row_stride = columns / kNvFp4GroupSize;

    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t row = index / columns;
        const std::size_t column = index - row * columns;
        const std::uint8_t packed =
            packed_weights[row * packed_row_stride + column / 2U];
        const std::uint8_t nibble = (column & 1U) != 0U
                                        ? static_cast<std::uint8_t>(packed >> 4U)
                                        : static_cast<std::uint8_t>(packed & 0x0fU);
        const std::uint8_t scale =
            weight_scales[row * scale_row_stride + column / kNvFp4GroupSize];
        output[index] = decode_e2m1_device(nibble) *
                        decode_e4m3fn_device(scale) * weight_scale_2;
    }
}

}  // namespace

int launch_nvfp4_reference_cuda(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const weight_scales,
    const float weight_scale_2,
    const std::size_t rows,
    const std::size_t columns,
    float* const output) noexcept {
    if (columns % kNvFp4GroupSize != 0U ||
        !std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F ||
        (columns != 0U && rows > std::numeric_limits<std::size_t>::max() / columns)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (rows == 0U || columns == 0U) {
        return static_cast<int>(cudaSuccess);
    }
    if (packed_weights == nullptr || weight_scales == nullptr || output == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    constexpr unsigned int kThreads = 256;
    constexpr std::size_t kMaximumBlocks = 65535;
    const std::size_t total = rows * columns;
    const std::size_t needed_blocks = ceil_divide(total, kThreads);
    const auto blocks = static_cast<unsigned int>(
        needed_blocks < kMaximumBlocks ? needed_blocks : kMaximumBlocks);

    // Attribute the returned launch status to this call rather than to an
    // unrelated earlier CUDA runtime error in the same host thread.
    (void)cudaGetLastError();
    dequantize_nvfp4_kernel<<<blocks, kThreads>>>(
        packed_weights, weight_scales, weight_scale_2, rows, columns, output);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::quantization
