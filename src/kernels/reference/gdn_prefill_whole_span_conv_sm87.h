#ifndef Q3X_KERNELS_REFERENCE_GDN_PREFILL_WHOLE_SPAN_CONV_SM87_H_
#define Q3X_KERNELS_REFERENCE_GDN_PREFILL_WHOLE_SPAN_CONV_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_whole_span_conv_detail {

inline constexpr std::size_t kMaximumTokenCount = 512U;

using BoundaryInspectionCallback = void (*)(
    const std::uint16_t* conv_qkv, std::size_t conv_qkv_elements,
    const std::uint16_t* history, std::size_t history_elements,
    void* cuda_stream, void* context) noexcept;

struct BoundaryInspectionHook {
  BoundaryInspectionCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] BoundaryInspectionHook exchange_boundary_inspection_hook(
    BoundaryInspectionHook hook) noexcept;

// Private, test-only Prefill launcher. This entry point deliberately stays
// outside the installed/public GDN ABI: the production tile ABI remains
// capped at 16 tokens.
//
// raw_qkv and conv_qkv_output are token-major BF16 [token_count, 10240].
// conv_weight is channel-major BF16 [10240, 4], and history_in_out is
// channel-major BF16 [10240, 3], oldest to newest. The complete span is
// advanced in one launch. Exact raw_qkv == conv_qkv_output aliasing is
// supported; every other operand must be disjoint. History always receives
// the original raw BF16 inputs, never the convolved outputs.
[[nodiscard]] int launch_causal_conv1d_silu_update_whole_span_exact_cuda(
    const std::uint16_t* raw_qkv, std::size_t token_count,
    const std::uint16_t* conv_weight, std::uint16_t* history_in_out,
    std::uint16_t* conv_qkv_output,
    void* cuda_stream = nullptr) noexcept;

// Token-parallel admission for the same exact arithmetic. Unlike the
// established in-place launcher, this form requires a disjoint output so
// independent C8 token tiles can read the immutable raw projection safely.
[[nodiscard]] int
launch_causal_conv1d_silu_update_token_parallel_exact_cuda(
    const std::uint16_t* raw_qkv, std::size_t token_count,
    const std::uint16_t* conv_weight, std::uint16_t* history_in_out,
    std::uint16_t* conv_qkv_output,
    void* cuda_stream = nullptr) noexcept;

// Structural Prefill candidate for the native C64 path.  The C8 channel
// owner keeps the established convolution BF16 boundary, then uses the two
// complete 128-wide heads already resident in each Q/K CTA to perform the
// exact compact-Q/K reduction before those values return to device memory.
// compact_q and compact_k use [chunk, qk_head, 64, 128] BF16 layout and must
// be disjoint from every convolution operand.
[[nodiscard]] int
launch_causal_conv1d_silu_update_token_parallel_compact_qk_exact_cuda(
    const std::uint16_t* raw_qkv, std::size_t token_count,
    const std::uint16_t* conv_weight, std::uint16_t* history_in_out,
    std::uint16_t* conv_qkv_output, float l2_epsilon,
    std::uint16_t* compact_q, std::uint16_t* compact_k,
    void* cuda_stream = nullptr) noexcept;

// Private admission resource query. No kernel is launched.
[[nodiscard]] int
query_causal_conv1d_silu_update_whole_span_resources_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

// CPU-callable resource query for the fused candidate.  It does not launch a
// kernel and is used by the static admission gate before real-model timing.
[[nodiscard]] int
query_causal_conv1d_silu_update_token_parallel_compact_qk_resources_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] const void*
token_parallel_compact_qk_kernel_handle_for_test() noexcept;

}  // namespace q3x::runtime::gdn_prefill_whole_span_conv_detail

#endif  // Q3X_KERNELS_REFERENCE_GDN_PREFILL_WHOLE_SPAN_CONV_SM87_H_
