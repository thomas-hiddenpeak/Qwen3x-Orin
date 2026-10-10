#include "mtp_prefill_internal.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime::mtp_detail {
namespace {
// One reduction mapping for full draft steps, live-KV appends and M1..32
// initialization. This draft-only numerical identity is independent of the
// target verifier's exact scalar FMA tree; no split-K or workspace is used.
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 128, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
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
  using B = cutlass::bfloat16_t;
  Gemm::Arguments args({static_cast<int>(count), static_cast<int>(rows), static_cast<int>(columns)},
      {reinterpret_cast<const B*>(input), columns},
      {reinterpret_cast<const B*>(weights), columns},
      {reinterpret_cast<const B*>(output), rows}, {reinterpret_cast<B*>(output), rows},
      {1.0F, 0.0F}, 1);
  if (Gemm::get_workspace_size(args) != 0 || Gemm::can_implement(args) != cutlass::Status::kSuccess)
    return cudaErrorInvalidValue;
  (void)cudaGetLastError();
  Gemm op;
  const auto status = op(args, nullptr, static_cast<cudaStream_t>(stream));
  return status == cutlass::Status::kSuccess ? cudaGetLastError() : cudaErrorLaunchFailure;
}
}  // namespace q3x::runtime::mtp_detail
