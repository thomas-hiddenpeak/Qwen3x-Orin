#ifndef Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK_O_BV64_SM87_H_
#define Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK_O_BV64_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_chunk_o_bv64_detail {

// Model- and architecture-fixed FLA output stage for Qwen3.6 GDN:
// Hg=16, H=48, K=V=128, BT=BK=BV=64 on SM87.  compact_q/compact_k are
// chunk-major [chunk, Hg, BT, K]. boundary_state and v_new are the established
// BF16 recurrence boundaries. raw_output is an internal BF16 [T,H,V]
// boundary consumed by the exact rows-8 RMSNorm+SiLU epilogue.
[[nodiscard]] int launch(
    const std::uint16_t* compact_q, const std::uint16_t* compact_k,
    const std::uint16_t* boundary_state, const std::uint16_t* v_new,
    const float* cumulative_gate, std::size_t token_count,
    const std::uint16_t* norm_weight, const std::uint16_t* silu_gate,
    float norm_epsilon, std::uint16_t* raw_output,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Correctness-only WMMA oracle for diagnosing inline-PTX fragment mapping.
// Engine dispatch and performance admission never call this entry.
[[nodiscard]] int launch_wmma_oracle(
    const std::uint16_t* compact_q, const std::uint16_t* compact_k,
    const std::uint16_t* boundary_state, const std::uint16_t* v_new,
    const float* cumulative_gate, std::size_t token_count,
    const std::uint16_t* norm_weight, const std::uint16_t* silu_gate,
    float norm_epsilon, std::uint16_t* raw_output,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Correctness-only lane sentinel. `matrix_a` and canonical [N,K] `matrix_b`
// are 16x16 BF16. Output arrays hold 32 lane fragments: A4, B2 and C4.
[[nodiscard]] int launch_fragment_sentinel(
    const std::uint16_t* matrix_a, const std::uint16_t* matrix_b,
    std::uint32_t* loaded_a, std::uint32_t* direct_a,
    std::uint32_t* loaded_b, std::uint32_t* direct_b,
    float* loaded_accumulator, float* direct_accumulator,
    void* cuda_stream = nullptr) noexcept;

// Independent exact epilogue for up to the bounded P4096 prompt-span route.
// row_count rows of D128 BF16 input/gate are normalized without launching
// chunk-o; the C512 component path remains otherwise unchanged.
[[nodiscard]] int launch_norm_rows8(
    const std::uint16_t* raw_output, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, std::size_t row_count,
    float norm_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Resource queries are deliberately split: the chunk-o tensor-core kernel
// and the exact rows-8 epilogue have different launch ownership.
[[nodiscard]] int query_chunk_o_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] int query_norm_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::runtime::gdn_prefill_chunk_o_bv64_detail

#endif  // Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK_O_BV64_SM87_H_
