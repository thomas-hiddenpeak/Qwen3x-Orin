#pragma once

#include <cstdint>

namespace q3x::kernels::sm87_target_aot_bf16_ab_execution_detail {

[[nodiscard]] int launch_interleaved_p40(
    const std::uint16_t* a_weights,
    const std::uint16_t* b_weights,
    const std::uint16_t* input,
    std::uint16_t* interleaved_ab_output,
    void* cuda_stream) noexcept;

}  // namespace q3x::kernels::sm87_target_aot_bf16_ab_execution_detail
