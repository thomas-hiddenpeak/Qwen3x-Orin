#ifndef Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK64_NATIVE_SM87_H_
#define Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK64_NATIVE_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_chunk64_native_detail {

// Admission-only exact-C512 native SM87 WY cell. One 256-thread CTA owns a
// value head for all eight ordered C64 chunks. The launcher has no library
// context and accepts no fallback shapes. The caller-owned workspace is zero;
// QK, the WY transform, corrected/new values, and chunk-boundary state never
// cross the kernel boundary.
[[nodiscard]] constexpr std::size_t workspace_bytes() noexcept { return 0U; }

[[nodiscard]] int launch(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, float norm_epsilon,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Reports the persistent WY cell, not the small post-normalization kernel.
[[nodiscard]] int query_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::runtime::gdn_prefill_chunk64_native_detail

#endif  // Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK64_NATIVE_SM87_H_
