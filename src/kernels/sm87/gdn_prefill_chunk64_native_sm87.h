#ifndef Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK64_NATIVE_SM87_H_
#define Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK64_NATIVE_SM87_H_

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_chunk64_native_detail {

// Test-only observation point for proving that a native pipeline rewrite does
// not change its established BF16 boundaries.  A null callback is the
// production default and has no synchronization or copy cost.
using InspectionCallback = void (*)(
    const std::uint16_t* transform, std::size_t transform_elements,
    const std::uint16_t* w, std::size_t w_elements,
    const std::uint16_t* u, std::size_t u_elements,
    const std::uint16_t* v_new, std::size_t v_new_elements,
    const std::uint16_t* boundary_state,
    std::size_t boundary_state_elements,
    const std::uint16_t* compact_k, std::size_t compact_k_elements,
    const float* gamma, std::size_t gamma_elements,
    const std::uint16_t* diagnostic_k_decay,
    std::size_t diagnostic_k_decay_elements,
    const std::uint16_t* diagnostic_post_update_state,
    std::size_t diagnostic_post_update_state_elements,
    const std::uint16_t* state_output, std::size_t state_elements,
    const std::uint16_t* output, std::size_t output_elements,
    void* cuda_stream, void* context) noexcept;

struct InspectionHook {
  InspectionCallback callback = nullptr;
  void* context = nullptr;
};

using PreprocessInspectionCallback = void (*)(
    const std::uint16_t* compact_q, std::size_t compact_q_elements,
    const std::uint16_t* compact_k, std::size_t compact_k_elements,
    void* cuda_stream, void* context) noexcept;

struct PreprocessInspectionHook {
  PreprocessInspectionCallback callback = nullptr;
  void* context = nullptr;
};

// Test-only CUDA-event observation points bracketing the group-owned WY and
// QK kernels.  They deliberately begin after the packed route's pack kernel,
// so this hook attributes only the common core window; end-to-end timing owns
// the omitted pack/state/reconstruct contribution. Null events are the
// production default.
struct WyTimingHook {
  void* begin = nullptr;
  void* after_initial = nullptr;
  void* after_qk = nullptr;
  void* after_final = nullptr;
};

[[nodiscard]] InspectionHook exchange_inspection_hook(
    InspectionHook hook) noexcept;

[[nodiscard]] PreprocessInspectionHook exchange_preprocess_inspection_hook(
    PreprocessInspectionHook hook) noexcept;

[[nodiscard]] WyTimingHook exchange_wy_timing_hook(
    WyTimingHook hook) noexcept;

[[nodiscard]] bool exchange_force_fused_kkt_baseline_for_test(
    bool enabled) noexcept;

[[nodiscard]] bool exchange_force_split_wy_baseline_for_test(
    bool enabled) noexcept;

[[nodiscard]] bool exchange_force_packed_qkv_baseline_for_test(
    bool enabled) noexcept;

[[nodiscard]] bool exchange_force_resident_state_baseline_for_test(
    bool enabled) noexcept;

enum class VllmLayoutWyRouteForTest : std::int8_t {
  kProductionDefault = -1,
  kGroupOwned = 0,
  kVllmLayout = 1,
};

[[nodiscard]] VllmLayoutWyRouteForTest
exchange_vllm_layout_wy_route_for_test(
    VllmLayoutWyRouteForTest route) noexcept;

[[nodiscard]] bool exchange_force_legacy_qk_reconstruct_baseline_for_test(
    bool enabled) noexcept;

[[nodiscard]] bool exchange_force_packless_resident_state_fallback_for_test(
    bool enabled) noexcept;

// Admission-only C64..C512 native SM87 FLA/WY pipeline. The implementation
// is deliberately fixed to the authenticated model shape (Hg=16, H=48,
// K=V=128, BT=64). It has no library context and accepts no fallback shape.
// The caller owns one reusable workspace that carries the explicit FLA stage
// boundaries; no cuBLAS/cuBLASLt handle is present in this interface.
[[nodiscard]] std::size_t workspace_bytes() noexcept;

[[nodiscard]] int launch(
    void* workspace, std::size_t workspace_capacity_bytes,
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, float norm_epsilon,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Same-ELF structural candidate boundary.  The first call performs the
// complete token-parallel causal convolution and writes compact normalized
// Q/K directly into this launcher's private workspace.  The second consumes
// those established BF16 boundaries without launching normalize_qk_kernel.
// They are deliberately separate host calls so the runner can retain its
// existing error attribution and CUDA stream ordering.
[[nodiscard]] int launch_fused_conv_compact_qk_preprocess(
    void* workspace, std::size_t workspace_capacity_bytes,
    const std::uint16_t* raw_qkv, std::size_t token_count,
    const std::uint16_t* conv_weight, std::uint16_t* history_in_out,
    std::uint16_t* conv_qkv_output, float l2_epsilon,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_qk_preprocessed(
    void* workspace, std::size_t workspace_capacity_bytes,
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, float norm_epsilon,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Prompt-span gate producer shared by the fixed-shape WY and state/output
// consumers. The outputs are FP32 chunk-major tensors laid out as
// [ceil(T / 64), 48, 64]. Tail gamma lanes preserve the final real cumulative
// gate while tail beta lanes are zero, matching the established C64 boundary.
// The authenticated prompt span is 1..4096 tokens.
[[nodiscard]] int launch_prompt_span_gate_producer(
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    std::size_t token_count, float* cumulative_gate, float* beta,
    void* cuda_stream = nullptr) noexcept;

// Prompt-span vertical state/output slice.  W/U, compact Q/K, and cumulative
// gamma are the unchanged chunk-major WY boundaries.  The fused CTA writes
// raw_output as padded BF16 [ceil64(T),48,128], publishes only final state,
// then the established rows-8 norm/gate kernel writes logical T rows.
// This entry supports C64 correctness fixtures as well as the independently
// admitted production range; caller-side admission remains default-off.
[[nodiscard]] int launch_prompt_span_state_o(
    const std::uint16_t* w, const std::uint16_t* u,
    const std::uint16_t* compact_q, const std::uint16_t* compact_k,
    const float* cumulative_gate, const std::uint16_t* state_input,
    std::uint16_t* state_output, std::size_t token_count,
    const std::uint16_t* norm_weight, const std::uint16_t* silu_gate,
    float norm_epsilon, std::uint16_t* raw_output,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Correctness-only two-kernel oracle for one C64..C512 segment.  It exposes
// the incumbent global Vnew/boundary-state storage to the fixture and has no
// production dispatch authority.
[[nodiscard]] int launch_prompt_span_state_o_baseline_for_test(
    const std::uint16_t* w, const std::uint16_t* u,
    const std::uint16_t* compact_q, const std::uint16_t* compact_k,
    const float* cumulative_gate, const std::uint16_t* state_input,
    std::uint16_t* state_output, std::size_t token_count,
    std::uint16_t* v_new, std::uint16_t* boundary_state,
    const std::uint16_t* norm_weight, const std::uint16_t* silu_gate,
    float norm_epsilon, std::uint16_t* raw_output,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int query_prompt_span_state_o_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

// Component oracle used only by the exactness harness: launches the admitted
// standalone compact normalize producer on an already-convolved tensor.
[[nodiscard]] int launch_compact_qk_baseline_for_test(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    float l2_epsilon, std::uint16_t* compact_q,
    std::uint16_t* compact_k, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] const void* compact_qk_baseline_kernel_handle_for_test()
    noexcept;

// Reports the largest-resource native FLA stage selected by the launcher.
[[nodiscard]] int query_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::runtime::gdn_prefill_chunk64_native_detail

#endif  // Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK64_NATIVE_SM87_H_
