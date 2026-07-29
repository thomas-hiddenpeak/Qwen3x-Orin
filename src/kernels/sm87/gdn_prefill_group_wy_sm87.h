#ifndef Q3X_KERNELS_SM87_GDN_PREFILL_GROUP_WY_SM87_H_
#define Q3X_KERNELS_SM87_GDN_PREFILL_GROUP_WY_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_group_wy_detail {

// Fixed-shape SM87 producer for Qwen3.6 GDN prefill. One CTA owns one
// (C64, Q/K-head) group, forms the shared K K^T Gram matrix once, and then
// emits the three value-head-specific WY pairs in their established BF16
// format. There is deliberately no generic-shape fallback in this module.
[[nodiscard]] int configure() noexcept;

[[nodiscard]] int launch(
    const std::uint16_t* k, const float* cumulative_gate,
    const float* beta, const std::uint16_t* gated_k,
    const std::uint16_t* value, std::size_t chunk_count,
    std::uint16_t* transform, std::uint16_t* w, std::uint16_t* u,
    void* cuda_stream) noexcept;

[[nodiscard]] int query_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::runtime::gdn_prefill_group_wy_detail

#endif  // Q3X_KERNELS_SM87_GDN_PREFILL_GROUP_WY_SM87_H_
