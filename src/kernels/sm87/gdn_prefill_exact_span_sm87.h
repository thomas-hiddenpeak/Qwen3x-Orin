#ifndef Q3X_KERNELS_SM87_GDN_PREFILL_EXACT_SPAN_SM87_H_
#define Q3X_KERNELS_SM87_GDN_PREFILL_EXACT_SPAN_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_exact_span_detail {

// Exact-BF16 Prefill recurrence for one contiguous aligned C16..C512 span.
// The production runner selects its proven C256/C512 scheduler shapes. The
// implementation changes only state ownership: two lanes own one complete
// state row and pass the FP32 dot-product accumulator after every K8 slice.
// K0..K127 FMA order and the per-token BF16 state boundary therefore match
// the established exact-C16 kernel. `output` is the raw rounded GDN boundary;
// the caller deliberately owns the existing RMSNorm/SiLU epilogue.
[[nodiscard]] int launch_row16_register_baton(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int query_row16_register_baton_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::runtime::gdn_prefill_exact_span_detail

#endif  // Q3X_KERNELS_SM87_GDN_PREFILL_EXACT_SPAN_SM87_H_
