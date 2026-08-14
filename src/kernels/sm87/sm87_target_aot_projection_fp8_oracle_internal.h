#pragma once

#include "q3x/kernels/sm87_target_aot_projection_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels::sm87_target_aot_projection_execution_detail {

#if !defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_FP8_PROJECTION_ADMISSION)
#error "The raw v1 FP8 oracle is available only in the default-off bulk-v2 FP8 admission build"
#endif

// Test-only raw-payload control for the existing v1 M128 FP8 implementation.
// It is source-private, accepts only M64, and cannot bind an authenticated
// asset or production selector.  The bulk-v2 CUDA oracle uses it in the same
// ELF to distinguish a shared harness/payload failure from a v2 execution
// failure.
struct Sm87TargetAotFp8RawOracleArguments final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::array<std::uint16_t, 3U> compensated_scale_bf16_bits{};
  std::size_t token_count = 0U;
  std::uint16_t* primary_output = nullptr;
  std::uint16_t* secondary_output = nullptr;
  std::uint16_t* tertiary_output = nullptr;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_raw_fp8_v1_oracle(
    const Sm87TargetAotFp8RawOracleArguments& arguments) noexcept;

}  // namespace q3x::kernels::sm87_target_aot_projection_execution_detail
