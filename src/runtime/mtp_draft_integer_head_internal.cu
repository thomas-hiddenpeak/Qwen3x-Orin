#include "mtp_draft_integer_head_internal.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

namespace q3x::runtime::mtp_detail {
namespace integer_head {
constexpr unsigned K = 5120, N = 248320, Groups = K / 16;

__device__ __forceinline__ int unpack_four(unsigned codes) {
  // FP4 magnitudes multiplied by two: 0,1,2,3,4,6,8,12.
  const unsigned magnitude = __byte_perm(0x03020100U, 0x0c080604U, codes & 0x7777U);
  const unsigned sign = __byte_perm(0U, ~0U, (codes >> 1) & 0x4444U);
  return static_cast<int>(__vsub4(magnitude ^ sign, sign));
}

__device__ __forceinline__ float scale_value(unsigned code) {
  const unsigned exponent = (code & 127U) >> 3, mantissa = code & 7U;
  if (exponent == 15 && mantissa == 7) return __int_as_float(0x7fffffff);
  const float magnitude = exponent ?
      __uint_as_float(((exponent + 120U) << 23) | (mantissa << 20)) :
      static_cast<float>(mantissa) * 0x1p-9F;
  return code & 128U ? -magnitude : magnitude;
}

__global__ void quantize(const __nv_bfloat16* input, unsigned* packed, float* scales) {
  const unsigned index = blockIdx.x * 256 + threadIdx.x;
  const unsigned lane = threadIdx.x & 31, group_lane = lane & 15;
  const float x = __bfloat162float(input[index]);
  float maximum = fabsf(x);
  int bad = !isfinite(x);
  #pragma unroll
  for (unsigned shift = 8; shift; shift >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, shift, 16));
    bad |= __shfl_xor_sync(0xffffffffU, bad, shift, 16);
  }
  const float scale = bad ? __int_as_float(0x7fffffff) : __fdiv_rn(maximum, 127.F);
  const int q = bad || maximum == 0.F ? 0 : max(-127, min(127, __float2int_rn(__fdiv_rn(x, scale))));
  const unsigned base = lane & ~3U;
  const unsigned q0 = static_cast<unsigned>(__shfl_sync(0xffffffffU, q, base)) & 255U;
  const unsigned q1 = static_cast<unsigned>(__shfl_sync(0xffffffffU, q, base + 1)) & 255U;
  const unsigned q2 = static_cast<unsigned>(__shfl_sync(0xffffffffU, q, base + 2)) & 255U;
  const unsigned q3 = static_cast<unsigned>(__shfl_sync(0xffffffffU, q, base + 3)) & 255U;
  if (!(lane & 3)) packed[index / 4] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (!group_lane) scales[index / 16] = scale;
}

__global__ void project(const std::uint8_t* weights, const std::uint8_t* scales,
    float tensor_scale, const unsigned* input, const float* input_scales,
    __nv_bfloat16* output) {
  const unsigned lane = threadIdx.x & 31;
  const unsigned row = blockIdx.x * 4 + threadIdx.x / 32;
  float sum = 0.F;
  #pragma unroll
  for (unsigned group = lane; group < Groups; group += 32) {
    const uint2 packed = reinterpret_cast<const uint2*>(weights + row * (K / 2))[group];
    const uint4 activation = reinterpret_cast<const uint4*>(input)[group];
    int dot = __dp4a(unpack_four(packed.x), static_cast<int>(activation.x), 0);
    dot = __dp4a(unpack_four(packed.x >> 16), static_cast<int>(activation.y), dot);
    dot = __dp4a(unpack_four(packed.y), static_cast<int>(activation.z), dot);
    dot = __dp4a(unpack_four(packed.y >> 16), static_cast<int>(activation.w), dot);
    sum = fmaf(static_cast<float>(dot),
        input_scales[group] * scale_value(scales[row * Groups + group]), sum);
  }
  #pragma unroll
  for (unsigned shift = 16; shift; shift >>= 1)
    sum += __shfl_down_sync(0xffffffffU, sum, shift);
  if (!lane) output[row] = __float2bfloat16_rn(sum * (tensor_scale * 0.5F));
}
}  // namespace integer_head

int launch_mtp_integer_draft_head(const std::uint8_t* weights,
    const std::uint8_t* scales, float tensor_scale, const std::uint16_t* input,
    unsigned outputs, unsigned inputs, void* scratch, std::size_t scratch_bytes,
    std::uint16_t* output, void* stream) noexcept {
  using namespace integer_head;
  if (outputs != N || inputs != K || !std::isfinite(tensor_scale) || tensor_scale <= 0 ||
      scratch_bytes < kIntegerDraftHeadScratchBytes) return cudaErrorInvalidValue;
  const void* pointers[] = {weights, scales, input, scratch, output};
  const std::size_t sizes[] = {std::size_t(N) * K / 2, std::size_t(N) * Groups,
      K * 2, kIntegerDraftHeadScratchBytes, N * 2};
  for (unsigned i = 0; i < 5; ++i) {
    const auto a = reinterpret_cast<std::uintptr_t>(pointers[i]);
    if (!a || (a & 15) || a > std::numeric_limits<std::uintptr_t>::max() - sizes[i])
      return cudaErrorInvalidValue;
    for (unsigned j = 0; j < i; ++j) {
      const auto b = reinterpret_cast<std::uintptr_t>(pointers[j]);
      if (a < b + sizes[j] && b < a + sizes[i]) return cudaErrorInvalidValue;
    }
  }
  auto* packed = static_cast<unsigned*>(scratch);
  auto* input_scales = reinterpret_cast<float*>(static_cast<char*>(scratch) + K);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  quantize<<<K / 256, 256, 0, cuda_stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input), packed, input_scales);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  project<<<N / 4, 128, 0, cuda_stream>>>(weights, scales, tensor_scale,
      packed, input_scales, reinterpret_cast<__nv_bfloat16*>(output));
  return cudaGetLastError();
}
}  // namespace q3x::runtime::mtp_detail
