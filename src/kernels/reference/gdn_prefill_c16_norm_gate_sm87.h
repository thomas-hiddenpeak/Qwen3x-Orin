#ifndef Q3X_KERNELS_REFERENCE_GDN_PREFILL_C16_NORM_GATE_SM87_H_
#define Q3X_KERNELS_REFERENCE_GDN_PREFILL_C16_NORM_GATE_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_c16_norm_gate_detail {

// Internal exact-C16 production launcher and its isolated diagnostic
// controls. This header is not installed and is not part of the public ABI.
[[nodiscard]] int launch_shared_boundary(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, float norm_epsilon,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_shared_boundary_debug(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, float norm_epsilon,
    std::uint16_t* output, std::uint16_t* raw_debug,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_global_boundary(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, float norm_epsilon,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int query_shared_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] int query_global_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::runtime::gdn_prefill_c16_norm_gate_detail

#endif  // Q3X_KERNELS_REFERENCE_GDN_PREFILL_C16_NORM_GATE_SM87_H_
