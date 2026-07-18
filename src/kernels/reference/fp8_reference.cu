#include "q3x/quantization/fp8.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::quantization {
namespace {

__device__ __forceinline__ float decode_e4m3fn_device(
    const std::uint8_t bits) {
  const std::uint8_t magnitude = bits & 0x7fU;
  const int exponent = static_cast<int>((magnitude >> 3U) & 0x0fU);
  const int mantissa = static_cast<int>(magnitude & 0x07U);

  if (exponent == 0x0f && mantissa == 0x07) {
    return copysignf(nanf(""), (bits & 0x80U) != 0U ? -1.0F : 1.0F);
  }

  const float value =
      exponent == 0
          ? ldexpf(static_cast<float>(mantissa), -9)
          : ldexpf(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
  return copysignf(value, (bits & 0x80U) != 0U ? -1.0F : 1.0F);
}

__global__ void dequantize_fp8_kernel(const std::uint8_t* const weights,
                                      const float weight_scale,
                                      const std::size_t total,
                                      float* const output) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = decode_e4m3fn_device(weights[index]) * weight_scale;
  }
}

}  // namespace

int launch_fp8_reference_cuda(const std::uint8_t* const weights,
                              const float weight_scale,
                              const std::size_t rows,
                              const std::size_t columns,
                              float* const output) noexcept {
  if (!std::isfinite(weight_scale) || weight_scale < 0.0F ||
      (columns != 0U &&
       rows > std::numeric_limits<std::size_t>::max() / columns)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (rows == 0U || columns == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (weights == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  constexpr unsigned int kThreads = 256;
  constexpr std::size_t kMaximumBlocks = 65535;
  const std::size_t total = rows * columns;
  const std::size_t needed_blocks =
      total / kThreads + (total % kThreads != 0U ? 1U : 0U);
  const auto blocks = static_cast<unsigned int>(
      needed_blocks < kMaximumBlocks ? needed_blocks : kMaximumBlocks);

  // CUDA's per-thread last-error slot can still contain an error from an
  // unrelated earlier runtime call. Clear it immediately before this launch
  // so the returned status belongs to this ownership boundary.
  (void)cudaGetLastError();
  dequantize_fp8_kernel<<<blocks, kThreads>>>(weights, weight_scale, total,
                                              output);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::quantization
