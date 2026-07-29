#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_chunk64_reference_detail {

// This module is a compile-time-isolated architecture reference. It may be
// built only with BUILD_TESTING and has no production/fallback authority.
// Its external GEMMs establish whether the C64/WY dataflow clears the real
// P513 stop-loss before the stages are replaced by native SM87 HMMA kernels.
[[nodiscard]] std::size_t workspace_bytes() noexcept;

[[nodiscard]] int create_context(void** context) noexcept;
[[nodiscard]] int destroy_context(void* context) noexcept;

[[nodiscard]] int launch(
    void* context,
    void* workspace,
    std::size_t workspace_capacity_bytes,
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
