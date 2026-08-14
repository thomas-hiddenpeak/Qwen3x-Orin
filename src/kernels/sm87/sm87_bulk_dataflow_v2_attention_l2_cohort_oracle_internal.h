#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_attention_l2_cohort.h"

namespace q3x::kernels::sm87_bulk_v2_attention_oracle_detail {

// Test-only launcher for the 52 final-epoch cohort bodies whose query-tile
// work is intentionally repeated with publication disabled.  The ordinary
// candidate launcher remains the execution authority; this narrow surface
// exists only so the CUDA differential oracle can poison the complete output
// and prove that the exact shared numerical body performs no store for any
// suppressed repeat.
[[nodiscard]] int launch_suppressed_repeat_bodies_cuda(
    const Sm87BulkV2AttentionArguments& arguments) noexcept;

}  // namespace q3x::kernels::sm87_bulk_v2_attention_oracle_detail
