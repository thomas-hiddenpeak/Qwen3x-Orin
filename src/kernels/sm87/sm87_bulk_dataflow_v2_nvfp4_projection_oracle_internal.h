#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_projection.h"

namespace q3x::kernels::sm87_bulk_v2_nvfp4_oracle_detail {

// Source-private raw-payload seam used only by the exact CUDA differential
// oracle.  Production-facing code must use the authenticated public
// arguments and cannot include this header.
struct RawTailArguments final {
  const std::uint16_t* normalized_input = nullptr;
  const std::uint8_t* gate_up_payload = nullptr;
  float gate_tensor_scale = 0.0F;
  float up_tensor_scale = 0.0F;
  const std::uint8_t* down_payload = nullptr;
  float down_tensor_scale = 0.0F;
  std::uint16_t* residual = nullptr;
  std::uint16_t* group_h_scratch = nullptr;
  Sm87BulkV2NvFp4DeviceControl* device_control = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  std::uint32_t group_epoch = 0U;
  std::uint32_t token_count = kSm87BulkV2NvFp4TailTokens;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_raw_tail(
    const RawTailArguments& arguments) noexcept;

// Exact M64/M1024 raw seam for the source-private differential oracle.  The
// public admission continues to require authenticated target-AOT assets.
[[nodiscard]] int launch_raw_macro(
    const RawTailArguments& arguments) noexcept;

}  // namespace q3x::kernels::sm87_bulk_v2_nvfp4_oracle_detail
