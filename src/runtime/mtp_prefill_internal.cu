#include "mtp_prefill_internal.h"
#include <cuda_runtime.h>
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
bool overlap(const void* a, std::size_t na, const void* b, std::size_t nb) {
  auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
  const auto max = std::numeric_limits<std::uintptr_t>::max();
  return x > max - na || y > max - nb || (x < y + nb && y < x + na);
}
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
