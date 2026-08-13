#pragma once

#include "q3x/kernels/sm87_target_aot_projection_cuda.h"
#include "q3x/kernels/sm87_target_aot_projection_fp8_cuda.h"

namespace q3x::kernels::sm87_target_aot_projection_execution_detail {

struct Sm87TargetAotFp8FullQkvScatterArguments final {
  const std::uint16_t* input = nullptr;
  Sm87TargetAotFp8CudaAssetView asset{};
  std::size_t token_count = 0U;
  std::uint16_t* q_gate_output = nullptr;
  std::uint16_t* key_output = nullptr;
  std::uint16_t* value_output = nullptr;
  void* cuda_stream = nullptr;
};

// Source-private execution seam.  The installed launchers remain immutable
// fail-closed sentinels; only the complete-owner executor may construct these
// authenticated arguments after revalidating the live owner borrow and the
// request transaction ranges.
[[nodiscard]] int launch_authenticated_nvfp4(
    const Sm87TargetAotNvFp4CudaArguments& arguments) noexcept;

[[nodiscard]] int launch_authenticated_fp8(
    const Sm87TargetAotFp8CudaArguments& arguments) noexcept;

// Full-QKV uses the same authenticated packed payload and arithmetic as the
// contiguous compile/oracle body, but publishes the three model tensors
// directly into their request-owner spans.  This is not a post-GEMM copy and
// no contiguous [M,14336] scratch is materialized.
[[nodiscard]] int launch_authenticated_fp8_full_qkv_scatter(
    const Sm87TargetAotFp8FullQkvScatterArguments& arguments) noexcept;

}  // namespace q3x::kernels::sm87_target_aot_projection_execution_detail
