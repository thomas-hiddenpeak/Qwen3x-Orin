#ifndef Q3X_KERNELS_SM87_GDN_PREFILL_WY_VLLM_LAYOUT_SM87_H_
#define Q3X_KERNELS_SM87_GDN_PREFILL_WY_VLLM_LAYOUT_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_wy_vllm_layout_detail {

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
    std::size_t chunk_count, float* raw_gram_scratch,
    std::uint16_t* transform, std::uint16_t* w,
    std::uint16_t* u, void* cuda_stream) noexcept;

}  // namespace q3x::runtime::gdn_prefill_wy_vllm_layout_detail

#endif  // Q3X_KERNELS_SM87_GDN_PREFILL_WY_VLLM_LAYOUT_SM87_H_
