#pragma once

#include "q3x/kernels/sm87_target_aot_gdn_cuda.h"

namespace q3x::kernels::sm87_target_aot_gdn_execution_detail {

// Source-private execution seam.  Its caller must first revalidate the live
// complete projection owner, request-owner allocation/epochs, engine inputs,
// prebound transaction spans, and owner-bound stream.  The CUDA body repeats
// pointer/device/range checks but does not mint ownership authority itself.
[[nodiscard]] int launch_authenticated(
    const Sm87TargetAotGdnCudaArguments& arguments) noexcept;

}  // namespace q3x::kernels::sm87_target_aot_gdn_execution_detail
