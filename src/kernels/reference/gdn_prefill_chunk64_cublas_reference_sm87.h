#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_chunk64_reference_detail {

// This module is a compile-time-isolated architecture reference. It may be
// built only with BUILD_TESTING and has no production/fallback authority.
// The historical filename is retained for build compatibility, but every
// C64/WY GEMM stage is now a native fixed-shape SM87 HMMA kernel; the context
// token carries no external-library handle.
[[nodiscard]] std::size_t workspace_bytes() noexcept;

[[nodiscard]] int create_context(void** context) noexcept;
[[nodiscard]] int destroy_context(void* context) noexcept;

[[nodiscard]] int launch(
    void* context,
    void* workspace,
    std::size_t workspace_capacity_bytes,
    std::size_t token_count,
    const std::uint16_t* conv_qkv,
    const std::uint16_t* a,
    const std::uint16_t* b,
    const std::uint16_t* A_log,
    const std::uint16_t* dt_bias,
    const std::uint16_t* state_input,
    std::uint16_t* state_output,
    float l2_epsilon,
    const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate,
    float norm_epsilon,
    std::uint16_t* output,
    void* cuda_stream) noexcept;

}  // namespace q3x::runtime::gdn_prefill_chunk64_reference_detail
