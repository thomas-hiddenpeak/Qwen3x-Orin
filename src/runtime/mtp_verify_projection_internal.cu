#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace q3x::runtime::mtp_detail {
namespace {
__device__ __forceinline__ float bf16(std::uint16_t x) {
  return __uint_as_float(static_cast<unsigned>(x) << 16);
}
__device__ __forceinline__ std::uint16_t rounded(float x) {
  unsigned b = __float_as_uint(x);
  if ((b & 0x7fffffffU) > 0x7f800000U) return (b >> 16) | 0x40U;
  return (b + 0x7fffU + ((b >> 16) & 1U)) >> 16;
}
__device__ __forceinline__ float fp8(unsigned bits) {
  const unsigned sign = (bits & 128U) << 24, mag = bits & 127U;
  const unsigned e = mag >> 3, m = mag & 7U;
  if (mag == 127) return __uint_as_float(sign | 0x7fc00000U);
  if (!e) {
    if (!m) return __uint_as_float(sign);
    const unsigned leading = m >= 4 ? 2 : (m >= 2 ? 1 : 0);
    return __uint_as_float(sign | ((118U + leading) << 23) |
                           ((m - (1U << leading)) << (23 - leading)));
  }
  return __uint_as_float(sign | ((120U + e) << 23) | (m << 20));
}
__device__ __forceinline__ float warp_sum(float x) {
  x += __shfl_down_sync(0xffffffffU, x, 16);
  x += __shfl_down_sync(0xffffffffU, x, 8);
  x += __shfl_down_sync(0xffffffffU, x, 4);
  x += __shfl_down_sync(0xffffffffU, x, 2);
  x += __shfl_down_sync(0xffffffffU, x, 1);
  return x;
}
// Preserve scalar Decode's four independent K chains and exact parenthesized
// merge. Generic small-M kernels use one chain and are not this oracle.
__global__ void fp8_rows(const std::uint8_t* weights, float scale,
    const std::uint16_t* x, unsigned count, unsigned n, unsigned k,
    std::uint16_t* y) {
  __shared__ float table[256], partial[4][8];
  table[threadIdx.x] = fp8(threadIdx.x);
  __syncthreads();
  float a[4][4] = {};
  for (unsigned base = threadIdx.x * 4; base < k; base += 1024) {
#pragma unroll
    for (unsigned c = 0; c < 4; ++c) {
      const float w = table[weights[blockIdx.x * k + base + c]];
#pragma unroll
      for (unsigned m = 0; m < 4; ++m)
        if (m < count) a[m][c] = fmaf(w, bf16(x[m*k + base + c]), a[m][c]);
    }
  }
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (unsigned m = 0; m < 4; ++m) {
    const float s = warp_sum((a[m][0]+a[m][1])+(a[m][2]+a[m][3]));
    if (!lane) partial[m][warp] = s;
  }
  __syncthreads();
  if (!warp) {
#pragma unroll
    for (unsigned m = 0; m < 4; ++m) {
      const float s = warp_sum(lane < 8 ? partial[m][lane] : 0.f) * scale;
      if (!lane && m < count) y[m*n + blockIdx.x] = rounded(s);
    }
  }
}
__global__ void nvfp4_rows(const std::uint8_t* weights, const std::uint8_t* scales,
    float scale, const std::uint16_t* x, unsigned count, unsigned n, unsigned k,
    std::uint16_t* y) {
  __shared__ float table[256];
  table[threadIdx.x] = fp8(threadIdx.x);
  __syncthreads();
  constexpr float values[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                               -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
  const unsigned lane = threadIdx.x % 32, row = blockIdx.x*8 + threadIdx.x/32;
  if (row >= n) return;  // Whole warp exits together.
  float a[4][4] = {};
  for (unsigned base = lane*8; base < k; base += 256) {
    const float block_scale = table[scales[row*(k/16) + base/16]];
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
      for (unsigned c = 0; c < 4; ++c) {
        const unsigned column = base + half*4 + c;
        const auto byte = weights[row*(k/2) + column/2];
        const float w = values[(byte >> ((column&1)*4)) & 15] * block_scale;
#pragma unroll
        for (unsigned m = 0; m < 4; ++m)
          if (m < count) a[m][c] = fmaf(w, bf16(x[m*k + column]), a[m][c]);
      }
    }
  }
#pragma unroll
  for (unsigned m = 0; m < 4; ++m) {
    const float s = warp_sum((a[m][0]+a[m][1])+(a[m][2]+a[m][3])) * scale;
    if (!lane && m < count) y[m*n + row] = rounded(s);
  }
}
}  // namespace
int launch_mtp_verify_projection(const std::uint8_t* weights,
    const std::uint8_t* scales, float scale, const std::uint16_t* x,
    unsigned count, unsigned n, unsigned k, std::uint16_t* y, void* stream) noexcept {
  if (!weights || !x || !y || count < 2 || count > 4 || !std::isfinite(scale) || scale < 0)
    return cudaErrorInvalidValue;
  const bool shape = scales ? ((n == 17408 && k == 5120) || (n == 5120 && k == 17408)) :
      ((k == 5120 && (n == 1024 || n == 6144 || n == 10240 || n == 12288)) ||
       (k == 6144 && n == 5120));
  if (!shape) return cudaErrorInvalidValue;
  (void)cudaGetLastError();
  auto s = static_cast<cudaStream_t>(stream);
  if (scales) nvfp4_rows<<<(n+7)/8,256,0,s>>>(weights,scales,scale,x,count,n,k,y);
  else fp8_rows<<<n,256,0,s>>>(weights,scale,x,count,n,k,y);
  return cudaGetLastError();
}
}  // namespace q3x::runtime::mtp_detail
