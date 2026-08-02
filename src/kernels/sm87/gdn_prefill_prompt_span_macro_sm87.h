#ifndef Q3X_KERNELS_SM87_GDN_PREFILL_PROMPT_SPAN_MACRO_SM87_H_
#define Q3X_KERNELS_SM87_GDN_PREFILL_PROMPT_SPAN_MACRO_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_prompt_span_macro_detail {

// Isolated proof slice for the value-head-owned prompt-span macro pipeline.
// The fixed model shape is Hg=16, H=48, K=V=128 and BT=64. raw_gram is
// compact [ceil64(M),16,64,64] FP32; gamma/beta are chunk-major
// [ceil64(M),48,64] FP32; compact_q/compact_k are
// [ceil64(M),16,64,128] BF16; conv_qkv is token-major [M,10240] BF16.
// state may be in-place. output may alias silu_gate exactly, which is the
// final whole-span runner contract used to publish directly into the
// projection-secondary buffer.
[[nodiscard]] int launch_c64(
    const float* raw_gram, const float* gamma, const float* beta,
    const std::uint16_t* compact_q, const std::uint16_t* compact_k,
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* state_input,
    std::uint16_t* state_output, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, float norm_epsilon,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Test-only publication of every incumbent BF16 boundary.  Production uses
// launch_c64(), whose non-diagnostic instantiation has no corresponding
// global stores.  Layouts are transform [C,48,64,64], W/U/Vnew
// [C,48,64,128], boundary_state [C,48,128,128], and raw_output
// [C*64,48,128].
struct DiagnosticBoundaries final {
  std::uint16_t* transform = nullptr;
  std::uint16_t* w = nullptr;
  std::uint16_t* u = nullptr;
  std::uint16_t* v_new = nullptr;
  std::uint16_t* boundary_state = nullptr;
  std::uint16_t* raw_output = nullptr;
};

[[nodiscard]] int launch_c64_diagnostic(
    const float* raw_gram, const float* gamma, const float* beta,
    const std::uint16_t* compact_q, const std::uint16_t* compact_k,
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* state_input,
    std::uint16_t* state_output, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, float norm_epsilon,
    std::uint16_t* output, DiagnosticBoundaries boundaries,
    void* cuda_stream = nullptr) noexcept;

// The resource contract applies to the production (non-diagnostic) kernel:
// 256 threads, one CTA per SM, <=224 registers/thread target (255 absolute),
// 115200 bytes of dynamic shared memory, and zero local/stack storage.
[[nodiscard]] int query_c64_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

[[nodiscard]] const void* c64_kernel_handle_for_test() noexcept;

}  // namespace q3x::runtime::gdn_prefill_prompt_span_macro_detail

#endif  // Q3X_KERNELS_SM87_GDN_PREFILL_PROMPT_SPAN_MACRO_SM87_H_
