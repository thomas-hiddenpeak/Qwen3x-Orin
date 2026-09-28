#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::mtp_detail {
namespace {
constexpr unsigned kBatch = 8, kThreads = 256;
__device__ float decode(std::uint16_t x) {
  return __uint_as_float(static_cast<unsigned>(x) << 16);
}
__device__ std::uint16_t encode(float x) {
  unsigned bits = __float_as_uint(x);
  if ((bits & 0x7fffffffU) > 0x7f800000U)
    return static_cast<std::uint16_t>((bits >> 16) | 0x40U);
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>(bits >> 16);
}
// Same per-thread K sequence and 256-thread binary reduction as BF16
// reference GEMV. Each decoded weight serves eight independent accumulators.
// No Tensor Core reassociation or change to the BF16 publication boundary.
__global__ void project(const std::uint16_t* weights,
                        const std::uint16_t* input, unsigned count,
                        unsigned rows, unsigned columns,
                        std::uint16_t* output) {
  __shared__ float partial[kBatch][kThreads];
  float sums[kBatch] = {};
  const auto row = blockIdx.x;
  for (unsigned k = threadIdx.x; k < columns; k += kThreads) {
    const float w = decode(weights[static_cast<std::size_t>(row) * columns + k]);
#pragma unroll
    for (unsigned m = 0; m < kBatch; ++m)
      if (m < count) sums[m] = fmaf(w, decode(input[m * columns + k]), sums[m]);
  }
#pragma unroll
  for (unsigned m = 0; m < kBatch; ++m) partial[m][threadIdx.x] = sums[m];
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned m = 0; m < kBatch; ++m)
        partial[m][threadIdx.x] += partial[m][threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (unsigned m = 0; m < count; ++m)
      output[m * rows + row] = encode(partial[m][0]);
}
}  // namespace

// Source-private: operands are disjoint, construction-owned Draft regions.
int launch_mtp_prefill_projection(const std::uint16_t* weights,
    const std::uint16_t* input, unsigned count, unsigned rows,
    unsigned columns, std::uint16_t* output, void* stream) noexcept {
  if (!weights || !input || !output || count == 0 || count > kBatch ||
      !((rows == 5120 && columns == 10240) ||
        (rows == 1024 && columns == 5120))) return cudaErrorInvalidValue;
  (void)cudaGetLastError();
  project<<<rows, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      weights, input, count, rows, columns, output);
  return cudaGetLastError();
}
}  // namespace q3x::runtime::mtp_detail
