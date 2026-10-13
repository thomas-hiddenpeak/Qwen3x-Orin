#include "mtp_prefill_internal.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime::mtp_detail {
namespace {
// Private draft arithmetic: the same ascending K16 MMA sequence for every
// count, including initialization and cache-only append. Target arithmetic is
// independent. Weight rows occupy MMA's M axis to bound single-token padding.
__device__ __forceinline__ unsigned swizzled(unsigned row, unsigned col) {
  return row * 128 + (col ^ ((row & 7U) * 8));
}
__device__ __forceinline__ void copy16(std::uint16_t* dst,
    const std::uint16_t* src, unsigned valid_bytes = 16) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
      "r"(address), "l"(src), "r"(valid_bytes) : "memory");
}
template<unsigned C>
__device__ __forceinline__ void stage_draft(std::uint16_t* w,
    std::uint16_t* x, const std::uint16_t* weights,
    const std::uint16_t* input, unsigned count, unsigned k, unsigned tile) {
  for (unsigned i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    const unsigned row = i / 16, col = (i % 16) * 8;
    copy16(w + swizzled(row, col),
        weights + (blockIdx.x * 64 + row) * k + tile * 128 + col);
  }
  for (unsigned i = threadIdx.x; i < C * 16; i += blockDim.x) {
    const unsigned row = i / 16, col = (i % 16) * 8;
    copy16(x + swizzled(row, col), row < count ?
        input + row * k + tile * 128 + col : input, row < count ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template<unsigned C>
__global__ void draft_weight_major(const std::uint16_t* weights,
    const std::uint16_t* input, unsigned count, unsigned n, unsigned k,
    std::uint16_t* output) {
  __shared__ __align__(16) std::uint16_t w[2][64 * 128];
  __shared__ __align__(16) std::uint16_t x[2][C * 128];
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const unsigned wr = (warp % 4) * 16 + lane / 4;
  const unsigned xr = (warp / 4) * 8 + lane / 4;
  const unsigned pair = (lane % 4) * 2;
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  stage_draft<C>(w[0], x[0], weights, input, count, k, 0);
  stage_draft<C>(w[1], x[1], weights, input, count, k, 1);
  asm volatile("cp.async.wait_group 1;" ::: "memory");
  __syncthreads();
  for (unsigned tile = 0; tile < k / 128; ++tile) {
    const unsigned b = tile & 1;
    #pragma unroll
    for (unsigned step = 0; step < 8; ++step) {
      const unsigned q = step * 16 + pair;
      const unsigned a0 = *reinterpret_cast<const unsigned*>(w[b] + swizzled(wr, q));
      const unsigned a1 = *reinterpret_cast<const unsigned*>(w[b] + swizzled(wr + 8, q));
      const unsigned a2 = *reinterpret_cast<const unsigned*>(w[b] + swizzled(wr, q + 8));
      const unsigned a3 = *reinterpret_cast<const unsigned*>(w[b] + swizzled(wr + 8, q + 8));
      const unsigned b0 = *reinterpret_cast<const unsigned*>(x[b] + swizzled(xr, q));
      const unsigned b1 = *reinterpret_cast<const unsigned*>(x[b] + swizzled(xr, q + 8));
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();
    if (tile + 2 < k / 128) {
      stage_draft<C>(w[b], x[b], weights, input, count, k, tile + 2);
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
  }
  const float results[4] = {c0, c1, c2, c3};
  #pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned token = (warp / 4) * 8 + pair + (i & 1);
    const unsigned row = blockIdx.x * 64 + wr + (i >= 2 ? 8 : 0);
    if (token < count) {
      const unsigned bits = __float_as_uint(results[i]);
      const unsigned rounded = (bits & 0x7fffffffU) > 0x7f800000U ?
          (bits >> 16) | 0x40U : (bits + 0x7fffU + ((bits >> 16) & 1)) >> 16;
      output[token * n + row] = static_cast<std::uint16_t>(rounded);
    }
  }
}
__global__ void draft_confidence(const std::uint16_t* logits, unsigned n, float* output) {
  __shared__ float reduction[256];
  float maximum = -CUDART_INF_F;
  unsigned invalid = 0;
  for (unsigned i = threadIdx.x; i < n; i += 256) {
    const float x = __uint_as_float(unsigned(logits[i]) << 16);
    invalid |= !isfinite(x);
    maximum = fmaxf(maximum, x);
  }
  const int bad = __syncthreads_or(invalid);
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = reduction[0];
  __syncthreads();
  float sum = 0;
  for (unsigned i = threadIdx.x; i < n; i += 256)
    sum += expf(__uint_as_float(unsigned(logits[i]) << 16) - maximum);
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) *output = bad ? CUDART_NAN_F : 1.0F / reduction[0];
}
bool overlap(const void* a, std::size_t na, const void* b, std::size_t nb) {
  auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
  const auto max = std::numeric_limits<std::uintptr_t>::max();
  return x > max - na || y > max - nb || (x < y + nb && y < x + na);
}
}
int launch_mtp_draft_confidence(const std::uint16_t* logits, unsigned count,
    float* output, void* stream) noexcept {
  if (!logits || !output || count == 0 || count > 248320 ||
      (reinterpret_cast<std::uintptr_t>(logits) & 1U) ||
      (reinterpret_cast<std::uintptr_t>(output) & 3U) ||
      overlap(logits, 2ULL * count, output, sizeof(float))) return cudaErrorInvalidValue;
  draft_confidence<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(logits, count, output);
  return cudaGetLastError();
}
int launch_mtp_prefill_projection(const std::uint16_t* weights,
    const std::uint16_t* input, unsigned count, unsigned rows,
    unsigned columns, std::uint16_t* output, void* stream) noexcept {
  const bool shape = (rows == 5120 && (columns == 10240 || columns == 6144 || columns == 17408)) ||
      ((rows == 1024 || rows == 12288 || rows == 17408) && columns == 5120);
  if (!weights || !input || !output || count == 0 || count > kDraftPrefillBatch || !shape ||
      ((reinterpret_cast<std::uintptr_t>(weights) | reinterpret_cast<std::uintptr_t>(input) |
        reinterpret_cast<std::uintptr_t>(output)) & 15U)) return cudaErrorInvalidValue;
  const std::size_t nw = 2ULL * rows * columns, ni = 2ULL * count * columns, no = 2ULL * count * rows;
  if (overlap(weights, nw, input, ni) || overlap(weights, nw, output, no) ||
      overlap(input, ni, output, no)) return cudaErrorInvalidValue;
  (void)cudaGetLastError();
  const auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (count <= 8)
    draft_weight_major<8><<<rows / 64, 128, 0, cuda_stream>>>(weights, input, count, rows, columns, output);
  else if (count <= 16)
    draft_weight_major<16><<<rows / 64, 256, 0, cuda_stream>>>(weights, input, count, rows, columns, output);
  else
    draft_weight_major<32><<<rows / 64, 512, 0, cuda_stream>>>(weights, input, count, rows, columns, output);
  return cudaGetLastError();
}
}  // namespace q3x::runtime::mtp_detail
