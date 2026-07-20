#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

inline constexpr std::size_t kGdnQkHeadCount = 16U;
inline constexpr std::size_t kGdnValueHeadCount = 48U;
inline constexpr std::size_t kGdnHeadDimension = 128U;
inline constexpr std::size_t kGdnConvHistoryWidth = 3U;
inline constexpr std::size_t kGdnConvKernelWidth = 4U;
inline constexpr std::size_t kGdnMaximumTileTokenCount = 16U;
inline constexpr std::size_t kGdnQElements =
    kGdnQkHeadCount * kGdnHeadDimension;
inline constexpr std::size_t kGdnKElements = kGdnQElements;
inline constexpr std::size_t kGdnVElements =
    kGdnValueHeadCount * kGdnHeadDimension;
inline constexpr std::size_t kGdnQkvChannels =
    kGdnQElements + kGdnKElements + kGdnVElements;
inline constexpr std::size_t kGdnStateElements =
    kGdnValueHeadCount * kGdnHeadDimension * kGdnHeadDimension;

struct GdnDimensions {
  std::size_t qk_head_count = kGdnQkHeadCount;
  std::size_t value_head_count = kGdnValueHeadCount;
  std::size_t head_dimension = kGdnHeadDimension;
};

enum class GdnStatus : std::uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kInvalidDimension,
  kSizeOverflow,
  kInvalidAlias,
};

[[nodiscard]] const char* gdn_status_string(GdnStatus status) noexcept;

// Single-token width-4 depthwise causal convolution over canonical
// [Q(16,128), K(16,128), V(48,128)] BF16 projection output. history_in_out is
// channel-major [10240,3] ordered oldest to newest. conv_weight is
// channel-major [10240,4]. The operation computes in FP32, applies SiLU,
// writes BF16 RNE conv_qkv, then shifts history and stores the original raw
// current value (not the convolved output).
//
// raw_qkv and conv_qkv_output may be exactly the same pointer. History,
// weights, and output must otherwise be disjoint; exact invalid aliases are
// rejected. Partial overlap is outside the API contract.
[[nodiscard]] GdnStatus causal_conv1d_silu_update_reference_cpu(
    const std::uint16_t* raw_qkv, const std::uint16_t* conv_weight,
    std::uint16_t* history_in_out, std::uint16_t* conv_qkv_output,
    GdnDimensions dimensions = {}) noexcept;

// One canonical Gated DeltaNet recurrence step. conv_qkv uses the same Q/K/V
// layout as above. a, b, A_log, and dt_bias are BF16 [48]. state_input and
// state_output use canonical BF16 [48,V=128,K=128] row-major storage; exact
// state_input == state_output is supported. output is BF16 [48,128].
//
// Q/K L2 reductions, decay/gates, state update, and output accumulation are
// FP32. Q alone receives the additional 1/sqrt(128) scale. State is persisted
// in BF16 RNE, while output uses the FP32 updated state before that rounding.
[[nodiscard]] GdnStatus gated_delta_net_update_reference_cpu(
    const std::uint16_t* conv_qkv, const std::uint16_t* a,
    const std::uint16_t* b, const std::uint16_t* A_log,
    const std::uint16_t* dt_bias, const std::uint16_t* state_input,
    std::uint16_t* state_output, float l2_epsilon,
    std::uint16_t* output, GdnDimensions dimensions = {}) noexcept;

// Asynchronous CUDA counterparts. All pointers are device-accessible;
// cuda_stream is cudaStream_t represented as void*. Calls allocate, copy, and
// synchronize nothing. Invalid host-visible arguments return
// cudaErrorInvalidValue. Each valid API clears an unrelated stale CUDA
// last-error immediately before its own launch.
[[nodiscard]] int launch_causal_conv1d_silu_update_reference_cuda(
    const std::uint16_t* raw_qkv, const std::uint16_t* conv_weight,
    std::uint16_t* history_in_out, std::uint16_t* conv_qkv_output,
    GdnDimensions dimensions = {}, void* cuda_stream = nullptr) noexcept;

// Sequence-tile form of the causal convolution launch above. raw_qkv and
// conv_qkv_output are contiguous token-major BF16
// [token_count, kGdnQkvChannels], and token_count must be in [1, 16]. The
// history recurrence is evaluated in token order with the same raw-BF16
// boundary as separate single-token launches. Exact raw/output aliasing is
// supported, and M=1 delegates to the single-token entry point.
[[nodiscard]] int launch_causal_conv1d_silu_update_tile_reference_cuda(
    const std::uint16_t* raw_qkv, std::size_t token_count,
    const std::uint16_t* conv_weight, std::uint16_t* history_in_out,
    std::uint16_t* conv_qkv_output, GdnDimensions dimensions = {},
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_gated_delta_net_update_reference_cuda(
    const std::uint16_t* conv_qkv, const std::uint16_t* a,
    const std::uint16_t* b, const std::uint16_t* A_log,
    const std::uint16_t* dt_bias, const std::uint16_t* state_input,
    std::uint16_t* state_output, float l2_epsilon,
    std::uint16_t* output, GdnDimensions dimensions = {},
    void* cuda_stream = nullptr) noexcept;

// Sequence-tile form of the Gated DeltaNet launch above. conv_qkv is
// token-major BF16 [token_count, kGdnQkvChannels], a and b are token-major
// BF16 [token_count, kGdnValueHeadCount], and output is token-major BF16
// [token_count, kGdnVElements]. A_log and dt_bias remain per-head constants.
// token_count must be in [1, 16]. Each recurrence step reads the BF16 state
// persisted by the preceding step while its output uses that step's FP32
// updated state, exactly as separate single-token launches. M=1 delegates to
// the single-token entry point.
[[nodiscard]] int launch_gated_delta_net_update_tile_reference_cuda(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    GdnDimensions dimensions = {}, void* cuda_stream = nullptr) noexcept;

// Warp-parallel forms of the Gated DeltaNet launches. One 256-thread block
// owns each value head; every warp advances eight state rows together while
// lanes keep each row's 128 BF16 elements coalesced during state updates.
// The FP32 dot products retain the reference kernel's left-to-right FMA order,
// so outputs and persisted state are bitwise equivalent to the reference
// entry points. These calls otherwise share their validation and aliasing
// contract, including in-place state updates.
[[nodiscard]] int launch_gated_delta_net_update_warp_parallel_cuda(
    const std::uint16_t* conv_qkv, const std::uint16_t* a,
    const std::uint16_t* b, const std::uint16_t* A_log,
    const std::uint16_t* dt_bias, const std::uint16_t* state_input,
    std::uint16_t* state_output, float l2_epsilon,
    std::uint16_t* output, GdnDimensions dimensions = {},
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_gated_delta_net_update_tile_warp_parallel_cuda(
    const std::uint16_t* conv_qkv, std::size_t token_count,
    const std::uint16_t* a, const std::uint16_t* b,
    const std::uint16_t* A_log, const std::uint16_t* dt_bias,
    const std::uint16_t* state_input, std::uint16_t* state_output,
    float l2_epsilon, std::uint16_t* output,
    GdnDimensions dimensions = {}, void* cuda_stream = nullptr) noexcept;

// Pure-host selector for the exact single-token composite below. It is true
// only for M=1 and the canonical GDN output interpreted as 48 heads of width
// 128. Pointer alignment, ranges, aliases, and both epsilons remain launch-
// time validation responsibilities.
[[nodiscard]] bool
supports_gated_delta_net_update_plain_rms_norm_silu_gate_fusion(
    std::size_t token_count, GdnDimensions dimensions,
    std::size_t norm_head_count,
    std::size_t norm_head_dimension) noexcept;

namespace gdn_decode_detail {

// Internal exact-shape kernel ABI used by the validated composite dispatcher.
// All pointers are BF16 device spans with the canonical sizes above. The
// output first receives the rounded raw GDN result, is synchronized within
// its owning head CTA, and is then read back and overwritten by plain
// RMSNorm+SiLU(gate). No cross-CTA synchronization is used.
[[nodiscard]] int
launch_gated_delta_net_update_plain_rms_norm_silu_gate_exact_cuda(
    const std::uint16_t* conv_qkv, const std::uint16_t* a,
    const std::uint16_t* b, const std::uint16_t* A_log,
    const std::uint16_t* dt_bias, const std::uint16_t* state_input,
    std::uint16_t* state_output, float l2_epsilon,
    const std::uint16_t* norm_weight, const std::uint16_t* silu_gate,
    float norm_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace gdn_decode_detail

// Validates and launches one canonical GDN update followed by headwise plain
// RMSNorm and SiLU gating. The exact aligned 48x128 M=1 route uses one kernel;
// every other valid norm partition whose product is kGdnVElements preserves
// the existing ordered GDN then reference norm/gate launches. The raw GDN
// output is rounded to BF16 and read back before the in-place final overwrite.
// Exact state_input == state_output is supported. Every writable span must be
// disjoint from every other operand; all arguments and fallback requirements
// are rejected before the first enqueue.
[[nodiscard]] int
launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
    const std::uint16_t* conv_qkv, const std::uint16_t* a,
    const std::uint16_t* b, const std::uint16_t* A_log,
    const std::uint16_t* dt_bias, const std::uint16_t* state_input,
    std::uint16_t* state_output, float l2_epsilon,
    const std::uint16_t* norm_weight, const std::uint16_t* silu_gate,
    std::size_t norm_head_count, std::size_t norm_head_dimension,
    float norm_epsilon, std::uint16_t* output,
    GdnDimensions dimensions = {}, void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::runtime
