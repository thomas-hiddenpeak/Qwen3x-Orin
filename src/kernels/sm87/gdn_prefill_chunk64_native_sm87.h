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
    const std::uint16_t* state_output, std::size_t state_elements,
    const std::uint16_t* output, std::size_t output_elements,
    void* cuda_stream, void* context) noexcept;

struct InspectionHook {
  InspectionCallback callback = nullptr;
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

[[nodiscard]] bool exchange_force_chunk_o_bv64_candidate_for_test(
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

// Reports the largest-resource native FLA stage selected by the launcher.
[[nodiscard]] int query_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::runtime::gdn_prefill_chunk64_native_detail

#endif  // Q3X_KERNELS_SM87_GDN_PREFILL_CHUNK64_NATIVE_SM87_H_
