#ifndef Q3X_KERNELS_SM87_GDN_PREFILL_WY_VLLM_LAYOUT_SM87_H_
#define Q3X_KERNELS_SM87_GDN_PREFILL_WY_VLLM_LAYOUT_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_wy_vllm_layout_detail {

// Forms the compact lower-triangular raw Gram matrices used by the C64 WY
// solve.  compact_k is laid out as [chunk_count, 16, 64, 128] BF16 and
// raw_gram as [chunk_count, 16, 64, 64] FP32.  The final compact-K chunk is
// expected to have already been zero padded by its producer.
//
// Unlike the complete packless route below, this kernel-only boundary accepts
// the full prompt span (up to 4096 tokens / 64 chunks).  chunk_count must be
// exactly ceil(token_count / 64).
[[nodiscard]] int launch_raw_gram(
    const std::uint16_t* compact_k, std::size_t token_count,
    std::size_t chunk_count, float* raw_gram,
    void* cuda_stream) noexcept;

// Production fixed-shape SM87 route modeled after the executed FLA/Triton
// C64 hierarchy.  It is specialized to the authenticated model shape and
// keeps the preceding group-owned implementation as a same-ELF diagnostic
// fallback.
//
// The caller supplies the existing chunk_count * 48 * 64 * 64 FP32 scratch;
// this route populates only its compact chunk_count * 16 prefix.
[[nodiscard]] int launch_packless(
    const std::uint16_t* compact_k, const float* cumulative_gate,
    const float* beta, const std::uint16_t* conv_qkv,
    std::size_t token_count, std::size_t chunk_count,
    float* raw_gram_scratch,
    std::uint16_t* transform, std::uint16_t* w,
    std::uint16_t* u, void* cuda_stream) noexcept;

}  // namespace q3x::runtime::gdn_prefill_wy_vllm_layout_detail

#endif  // Q3X_KERNELS_SM87_GDN_PREFILL_WY_VLLM_LAYOUT_SM87_H_
