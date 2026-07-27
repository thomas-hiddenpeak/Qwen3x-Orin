#ifndef Q3X_KERNELS_REFERENCE_GDN_PREFILL_B8_SEQUENTIAL_SM87_H_
#define Q3X_KERNELS_REFERENCE_GDN_PREFILL_B8_SEQUENTIAL_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_b8_detail {

// Private admission-only launcher for the screened sequential FP32-B8
// Prefill candidate. It intentionally is not part of the public GDN ABI and
// is not selected by the production runner. Only exact C256/C512 canonical
// spans are accepted. All eight complete spans must be pairwise disjoint;
// exact state_input == state_output is the only overlap exception.
[[nodiscard]] int launch_gated_delta_net_update_sequential_fp32_b8_exact_cuda(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Private resource query for admission checks. No kernel is launched.
[[nodiscard]] int
query_gated_delta_net_update_sequential_fp32_b8_resources_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::runtime::gdn_prefill_b8_detail

#endif  // Q3X_KERNELS_REFERENCE_GDN_PREFILL_B8_SEQUENTIAL_SM87_H_
