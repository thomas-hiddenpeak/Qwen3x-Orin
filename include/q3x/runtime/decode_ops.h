#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

inline constexpr std::size_t kLinearAttentionHeadDimension = 128U;
inline constexpr std::size_t kFullAttentionHeadDimension = 256U;
inline constexpr std::size_t kFusedGqaMaximumSequenceLength = 64U;
inline constexpr std::size_t kQwenRotaryDimension = 64U;
inline constexpr std::size_t kQkRopeTileMaximumTokens = 16U;

enum class DecodeOpStatus : std::uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kInvalidDimension,
  kSizeOverflow,
  kTokenOutOfRange,
  kInsufficientScratch,
};

// Compact device result for BF16 greedy selection. value_bits retains the
// exact selected BF16 payload (including the sign of an earliest zero tie),
// while has_nonfinite is non-zero when any input is NaN or infinity.
struct Bf16GreedyArgmaxResult {
  std::uint32_t index = 0U;
  std::uint16_t value_bits = 0U;
  std::uint16_t has_nonfinite = 0U;
};

static_assert(sizeof(Bf16GreedyArgmaxResult) == 8U);

inline constexpr std::size_t kBf16GreedyArgmaxWorkspaceResults = 33U;

[[nodiscard]] const char* decode_op_status_string(
    DecodeOpStatus status) noexcept;

// CPU correctness APIs. BF16 storage is represented by raw IEEE bfloat16
// uint16_t bits. Computation and reductions use FP32, and BF16 outputs are
// rounded to nearest-even. Empty vector/matrix shapes are successful no-ops.
[[nodiscard]] DecodeOpStatus embedding_gather_reference_cpu(
    const std::uint16_t* embedding_table, std::size_t vocabulary_size,
    std::size_t hidden_size, std::size_t token_id,
    std::uint16_t* output) noexcept;

// Qwen outer RMSNorm uses centered weights: multiplier = 1 + weight[i].
[[nodiscard]] DecodeOpStatus centered_rms_norm_reference_cpu(
    const std::uint16_t* input, const std::uint16_t* weight,
    std::size_t hidden_size, float epsilon, std::uint16_t* output) noexcept;

// Internal GDN norm uses plain weights: multiplier = weight[i].
[[nodiscard]] DecodeOpStatus plain_rms_norm_reference_cpu(
    const std::uint16_t* input, const std::uint16_t* weight,
    std::size_t hidden_size, float epsilon, std::uint16_t* output) noexcept;

// Apply one shared [head_dimension] weight vector independently to every
// head. These are the correct full-attention Q/K and GDN output boundaries;
// concatenated heads must never be normalized as one large vector.
[[nodiscard]] DecodeOpStatus headwise_centered_rms_norm_reference_cpu(
    const std::uint16_t* input, const std::uint16_t* shared_weight,
    std::size_t head_count, std::size_t head_dimension, float epsilon,
    std::uint16_t* output) noexcept;

[[nodiscard]] DecodeOpStatus headwise_plain_rms_norm_reference_cpu(
    const std::uint16_t* input, const std::uint16_t* shared_weight,
    std::size_t head_count, std::size_t head_dimension, float epsilon,
    std::uint16_t* output) noexcept;

// GDN output epilogue: headwise plain RMSNorm followed by elementwise
// SiLU(gate). input, gate and output are [head_count, head_dimension]; the
// plain norm weight is shared [head_dimension].
[[nodiscard]] DecodeOpStatus headwise_plain_rms_norm_silu_gate_reference_cpu(
    const std::uint16_t* input, const std::uint16_t* shared_weight,
    const std::uint16_t* gate, std::size_t head_count,
    std::size_t head_dimension, float epsilon,
    std::uint16_t* output) noexcept;

[[nodiscard]] DecodeOpStatus residual_add_reference_cpu(
    const std::uint16_t* left, const std::uint16_t* right,
    std::size_t element_count, std::uint16_t* output) noexcept;

// Convert FP32 GEMV output to BF16 using round-to-nearest-even. Any FP32 NaN
// remains a quiet BF16 NaN even when its low payload would otherwise round to
// infinity.
[[nodiscard]] DecodeOpStatus fp32_to_bf16_reference_cpu(
    const float* input, std::size_t element_count,
    std::uint16_t* output) noexcept;

[[nodiscard]] DecodeOpStatus silu_mul_reference_cpu(
    const std::uint16_t* gate, const std::uint16_t* up,
    std::size_t element_count, std::uint16_t* output) noexcept;

// Full-attention output gate: value[i] * sigmoid(gate[i]). Output may alias
// value or gate exactly.
[[nodiscard]] DecodeOpStatus sigmoid_gate_reference_cpu(
    const std::uint16_t* value, const std::uint16_t* gate,
    std::size_t element_count, std::uint16_t* output) noexcept;

// Normalize each [head_dimension] vector by sqrt(sum(x^2) + epsilon).
[[nodiscard]] DecodeOpStatus l2_normalize_heads_reference_cpu(
    const std::uint16_t* input, std::size_t head_count,
    std::size_t head_dimension, float epsilon,
    std::uint16_t* output) noexcept;

// Qwen partial NeoX RoPE for full-attention head_dim=256, rotary_dim=64.
// cosines and sines
// each contain 32 FP32 values. Dimensions 0..31 pair with 32..63;
// dimensions 64..255 pass through unchanged.
[[nodiscard]] DecodeOpStatus partial_neox_rope_256_64_reference_cpu(
    const std::uint16_t* input, const float* cosines, const float* sines,
    std::size_t head_count, std::uint16_t* output) noexcept;

// Stable row-wise FP32 softmax. Input and output may alias exactly.
[[nodiscard]] DecodeOpStatus softmax_reference_cpu(
    const float* input, std::size_t rows, std::size_t columns,
    float* output) noexcept;

// Single-token causal GQA over an already-populated KV cache:
//   query:       [query_head_count, head_dimension] BF16
//   key/value:   [sequence_length, kv_head_count, head_dimension] BF16
//   probabilities_scratch: [query_head_count, sequence_length] FP32
//   output:      [query_head_count, head_dimension] BF16
// query_head_count must be divisible by kv_head_count. The caller supplies
// scratch_elements so undersized scratch is rejected before it is used.
[[nodiscard]] DecodeOpStatus gqa_attention_reference_cpu(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, std::size_t query_head_count,
    std::size_t kv_head_count, std::size_t sequence_length,
    std::size_t head_dimension, float attention_scale,
    float* probabilities_scratch, std::size_t scratch_elements,
    std::uint16_t* output) noexcept;

// Asynchronous CUDA counterparts. All pointers must refer to device-accessible
// storage and cuda_stream is a cudaStream_t represented as void*. No launch
// allocates, copies, or synchronizes. Invalid host-visible arguments return
// cudaErrorInvalidValue. Each API clears an unrelated stale CUDA last-error
// immediately before its first kernel and reports its own launch boundary.
[[nodiscard]] int launch_embedding_gather_reference_cuda(
    const std::uint16_t* embedding_table, std::size_t vocabulary_size,
    std::size_t hidden_size, std::size_t token_id, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_centered_rms_norm_reference_cuda(
    const std::uint16_t* input, const std::uint16_t* weight,
    std::size_t hidden_size, float epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_plain_rms_norm_reference_cuda(
    const std::uint16_t* input, const std::uint16_t* weight,
    std::size_t hidden_size, float epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_headwise_centered_rms_norm_reference_cuda(
    const std::uint16_t* input, const std::uint16_t* shared_weight,
    std::size_t head_count, std::size_t head_dimension, float epsilon,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_headwise_plain_rms_norm_reference_cuda(
    const std::uint16_t* input, const std::uint16_t* shared_weight,
    std::size_t head_count, std::size_t head_dimension, float epsilon,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
    const std::uint16_t* input, const std::uint16_t* shared_weight,
    const std::uint16_t* gate, std::size_t head_count,
    std::size_t head_dimension, float epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_residual_add_reference_cuda(
    const std::uint16_t* left, const std::uint16_t* right,
    std::size_t element_count, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed-width decode residual boundary for hidden_size=5120. This is
// bitwise-equivalent to a 5120-element residual add followed by centered
// RMSNorm of the rounded BF16 residual. residual_output must be disjoint from
// every input and normalized_output. normalized_output may alias right
// exactly, but must otherwise be disjoint from all inputs.
[[nodiscard]] int launch_residual_add_centered_rms_norm_5120_cuda(
    const std::uint16_t* left, const std::uint16_t* right,
    const std::uint16_t* weight, float epsilon,
    std::uint16_t* residual_output, std::uint16_t* normalized_output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_fp32_to_bf16_reference_cuda(
    const float* input, std::size_t element_count, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Greedy selection over an already-rounded BF16 vector. The comparison is
// identical to a strict left-to-right FP32 argmax: equal maxima select the
// smallest index, including signed-zero ties. result_workspace must contain
// kBf16GreedyArgmaxWorkspaceResults device-accessible elements and be disjoint
// from input. The final result is written to element zero; remaining elements
// are launch-private scratch.
[[nodiscard]] int launch_bf16_greedy_argmax_cuda(
    const std::uint16_t* input, std::size_t element_count,
    Bf16GreedyArgmaxResult* result_workspace,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_silu_mul_reference_cuda(
    const std::uint16_t* gate, const std::uint16_t* up,
    std::size_t element_count, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sigmoid_gate_reference_cuda(
    const std::uint16_t* value, const std::uint16_t* gate,
    std::size_t element_count, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_l2_normalize_heads_reference_cuda(
    const std::uint16_t* input, std::size_t head_count,
    std::size_t head_dimension, float epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_partial_neox_rope_256_64_reference_cuda(
    const std::uint16_t* input, const float* cosines, const float* sines,
    std::size_t head_count, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed full-attention Q/K tile fast path. query is [token_count, 24, 256]
// BF16 and key is [token_count, 4, 256] BF16; both are rotated in place.
// cosines and sines are base pointers to [position, 32] FP32 tables.
// token_count must be in [1, kQkRopeTileMaximumTokens].
[[nodiscard]] int launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
    std::uint16_t* query, std::uint16_t* key, const float* cosines,
    const float* sines, std::size_t first_position,
    std::size_t token_count, void* cuda_stream = nullptr) noexcept;

// Fixed full-attention preprocessing fast path for Q=24, KV=4, D=256, and
// rotary=64. interleaved_q_gate is [token_count, 24, 2, 256]; query_output and
// gate_output are separate [token_count, 24, 256] arrays, while key is updated
// in place as [token_count, 4, 256]. The operation splits Q/gate raw BF16,
// applies centered headwise Q/K RMSNorm, then rotates Q/K using base
// [position, 32] FP32 tables. token_count must be in [1, 16]. Writable arrays
// must not overlap each other or any read-only input.
[[nodiscard]] int launch_full_attention_preprocess_24_4_256_64_cuda(
    const std::uint16_t* interleaved_q_gate, std::uint16_t* key,
    const std::uint16_t* q_weight, const std::uint16_t* k_weight,
    float epsilon, std::uint16_t* query_output,
    std::uint16_t* gate_output, const float* cosines, const float* sines,
    std::size_t first_position, std::size_t token_count,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_softmax_reference_cuda(
    const float* input, std::size_t rows, std::size_t columns, float* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_gqa_attention_reference_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, std::size_t query_head_count,
    std::size_t kv_head_count, std::size_t sequence_length,
    std::size_t head_dimension, float attention_scale,
    float* probabilities_scratch, std::size_t scratch_elements,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Fixed-shape full-attention fast path for Q=24, KV=4, and D=256 with a
// sequence length in [1, kFusedGqaMaximumSequenceLength]. It is bitwise
// equivalent to launch_gqa_attention_reference_cuda followed by an in-place
// sigmoid gate, including the intermediate BF16 rounding boundary. The final
// FP32 softmax probabilities are written to probabilities_scratch exactly as
// in the reference GQA entry point.
[[nodiscard]] int launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, std::size_t sequence_length,
    float attention_scale, float* probabilities_scratch,
    std::size_t scratch_elements, const std::uint16_t* gate,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::runtime
