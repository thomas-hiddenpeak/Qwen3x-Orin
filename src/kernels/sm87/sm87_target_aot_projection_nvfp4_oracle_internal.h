#pragma once

#include "q3x/kernels/sm87_target_aot_projection_plan.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels::sm87_target_aot_nvfp4_oracle_detail {

// Source-private raw byte seam for a same-ELF differential oracle.  It never
// participates in production dispatch and is intentionally restricted to the
// exact M64 tail exercised by bulk-dataflow-v2.
struct RawV1Arguments final {
  Sm87TargetAotProjectionRole role = Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::size_t token_count = 0U;
  float tensor_scale0 = 0.0F;
  float tensor_scale1 = 0.0F;
  std::uint16_t* output_or_residual = nullptr;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_raw_v1(const RawV1Arguments& arguments) noexcept;

}  // namespace q3x::kernels::sm87_target_aot_nvfp4_oracle_detail
