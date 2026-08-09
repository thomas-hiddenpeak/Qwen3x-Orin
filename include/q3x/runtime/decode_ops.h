#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

inline constexpr std::size_t kLinearAttentionHeadDimension = 128U;
inline constexpr std::size_t kFullAttentionHeadDimension = 256U;
inline constexpr std::size_t kFusedGqaMaximumSequenceLength = 64U;
inline constexpr std::size_t kDecodeGqaSplitKvMaximumSequenceLength = 4'096U;
inline constexpr std::size_t kDecodeGqaSplitKvMaximumSplits = 8U;
inline constexpr std::size_t kDecodeGqaSplitKvStateElements = 258U;
inline constexpr std::size_t kDecodeGqaSplitKvMaximumWorkspaceElements =
    24U * kDecodeGqaSplitKvMaximumSplits *
    kDecodeGqaSplitKvStateElements;
inline constexpr std::size_t kBulkCausalGqaMaximumSequenceLength = 262'144U;
inline constexpr unsigned int kBulkCausalGqaGroupQ64FirstPositionBits = 18U;
inline constexpr std::size_t kBulkCausalGqaGroupQ64PanelMaximumTokens =
    8'192U;
inline constexpr std::size_t
    kBulkCausalGqaFlashInferExactPanelMaximumTokens = 8'192U;
// Isolated SM87 Attention-v4 bring-up geometry.  This surface is never
// selected by the production runner: one CTA aggregates eight independent
// Q16 warp states so one K/V32 tile is shared by 128 packed GQA queries.
inline constexpr std::size_t kBulkCausalGqaGroupQ128V4PackedQueryTile =
    128U;
inline constexpr std::size_t kBulkCausalGqaGroupQ128V4Threads = 256U;
inline constexpr std::size_t kBulkCausalGqaGroupQ128V4DynamicSharedBytes =
    96U * 1024U;
inline constexpr std::size_t kQwenRotaryDimension = 64U;
inline constexpr std::size_t kQkRopeTileMaximumTokens = 16U;
inline constexpr std::size_t kFullAttentionPreprocessMaximumTokens = 512U;

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

// Prompt-wide counterpart. token_ids must already reside in device-accessible
// storage. The embedding_table, token_ids, and output byte ranges must be
// pairwise disjoint. One CTA owns one token row and copies the complete hidden
// vector, avoiding one launch per prompt token while retaining bitwise copy
// semantics. Token values are not synchronized back for host validation; a
// device token outside vocabulary_size leaves its output row unmodified, so
// callers that require request failure must validate token IDs before launch.
[[nodiscard]] int launch_embedding_gather_prompt_reference_cuda(
    const std::uint16_t* embedding_table, std::size_t vocabulary_size,
    std::size_t hidden_size, const std::uint32_t* token_ids,
    std::size_t token_count, std::uint16_t* output,
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

// left and right must occupy disjoint full byte ranges. output may alias left
// or right only when its complete range is exactly identical to that input;
// every partial overlap is rejected before enqueue. Empty input remains a
// successful no-op, including with null pointers.
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

// Fixed exact-M32 Prefill residual boundary. token_count must be 32 and
// hidden_size must be 5120. This is bitwise-equivalent to residual-left plus
// projection-right, rounded to BF16, followed by per-token centered RMSNorm
// of that rounded residual. residual_output must be disjoint from every input
// and normalized_output. normalized_output may alias right exactly, but must
// otherwise be disjoint from all inputs.
[[nodiscard]] int
launch_residual_add_headwise_centered_rms_norm_m32_5120_cuda(
    const std::uint16_t* left, const std::uint16_t* right,
    const std::uint16_t* weight, std::size_t token_count,
    std::size_t hidden_size, float epsilon,
    std::uint16_t* residual_output, std::uint16_t* normalized_output,
    void* cuda_stream = nullptr) noexcept;

// Prompt-wide form of the exact Prefill residual boundary above. It retains
// the same per-token reduction order and BF16 rounding boundary, but admits a
// complete 1..512-token Prefill tile in one launch instead of decomposing it
// into M32 launches plus a reference tail.
[[nodiscard]] int
launch_residual_add_headwise_centered_rms_norm_prefill_5120_cuda(
    const std::uint16_t* left, const std::uint16_t* right,
    const std::uint16_t* weight, std::size_t token_count,
    std::size_t hidden_size, float epsilon,
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
// [position, 32] FP32 tables. token_count must be in
// [1, kFullAttentionPreprocessMaximumTokens]. Writable arrays must not overlap
// each other or any read-only input.
//
// This explicit entry point always launches the established 256-thread,
// one-head-per-CTA reference tactic. Its tactic selection never reads the
// prompt-wide admission environment or thread-local test state.
[[nodiscard]] int
launch_full_attention_preprocess_24_4_256_64_reference_256_cuda(
    const std::uint16_t* interleaved_q_gate, std::uint16_t* key,
    const std::uint16_t* q_weight, const std::uint16_t* k_weight,
    float epsilon, std::uint16_t* query_output,
    std::uint16_t* gate_output, const float* cosines, const float* sines,
    std::size_t first_position, std::size_t token_count,
    void* cuda_stream = nullptr) noexcept;

// Compatibility/admission wrapper. The prompt-wide M>=2 route requires
// Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION=1; otherwise it
// launches the same established reference-256 tactic as the explicit entry
// point above.
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

// Fixed Qwen3.6 decode-only split-KV GQA path for Q=24, KV=4, D=256.
// Stage one uses four sequence splits through length 512 and eight beyond it.
// Each 192-thread CTA owns one KV head/split; its six Q warps cooperatively
// pipeline six cache rows at a time while retaining a stable online-softmax
// (maximum, denominator, FP32 value accumulator) state.  Stage two merges the
// split states without materializing scores or probabilities, rounds attention
// to BF16, applies the sigmoid gate, and rounds the final result to BF16.
//
// sequence_length must be in (kFusedGqaMaximumSequenceLength,
// kDecodeGqaSplitKvMaximumSequenceLength].  workspace is caller-owned FP32
// storage and must contain at least
// gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements()
// elements for the selected sequence length.  All launches are asynchronous.
[[nodiscard]] std::size_t
gqa_attention_splitkv_sigmoid_gate_24_4_256_workspace_elements(
    std::size_t sequence_length) noexcept;

[[nodiscard]] int
launch_gqa_attention_splitkv_sigmoid_gate_24_4_256_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, std::size_t sequence_length,
    float attention_scale, float* workspace, std::size_t workspace_elements,
    const std::uint16_t* gate, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed-shape bulk causal full-attention path for Q=24, KV=4, and D=256.
// token_count must be in [2, 512]. query_tile, gate_tile, and
// output_tile are tile-local [token_count, 24, 256] BF16 arrays. key_cache
// and value_cache use global contiguous NHD layout
// [position, 4, 256], and first_position is the global append position of
// tile-local token zero. Attention uses the fixed 1/sqrt(256) scale and
// preserves the FP32 attention -> BF16 -> sigmoid Gate -> BF16 boundary.
// All arrays must be disjoint and at least uint32_t aligned. P0/C2..C512 and
// P512/C2..C512 continuations use the grouped-Q64 Tensor Core path; other
// legal append positions use QT2. Both paths mask incomplete tiles. The
// launch is asynchronous, performs no allocation or synchronization, and
// uses no caller-visible scratch.
[[nodiscard]] constexpr bool use_bulk_causal_gqa_group_q64_prefill(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  return (first_position == 0U || first_position == 512U) &&
         token_count >= 2U && token_count <= 512U;
}

// Sealed Prefill owns an explicit, environment-independent tactic contract.
// P0/P512 use the current production-default V3 grouped-Q64 implementation;
// every other legal append position uses the generic QT2 implementation.
// Invalid geometry is represented explicitly so host-only binding tests can
// prove the complete dispatch table without launching CUDA work.
enum class FixedBulkCausalGqaPrefillTactic : std::uint8_t {
  kInvalid = 0,
  kGroupQ64V3,
  kGenericQt2,
};

[[nodiscard]] constexpr FixedBulkCausalGqaPrefillTactic
select_fixed_bulk_causal_gqa_prefill_tactic(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  if (token_count < 2U || token_count > 512U ||
      first_position > kBulkCausalGqaMaximumSequenceLength - token_count) {
    return FixedBulkCausalGqaPrefillTactic::kInvalid;
  }
  return use_bulk_causal_gqa_group_q64_prefill(first_position, token_count)
             ? FixedBulkCausalGqaPrefillTactic::kGroupQ64V3
             : FixedBulkCausalGqaPrefillTactic::kGenericQt2;
}

[[nodiscard]] int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
    const std::uint16_t* query_tile, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, const std::uint16_t* gate_tile,
    std::size_t first_position, std::size_t token_count,
    std::uint16_t* output_tile, void* cuda_stream = nullptr) noexcept;

// Sealed counterpart to the admission-aware wrapper above. This entry never
// reads Q3X_FULL_ATTENTION_FLASHINFER_DIRECT or
// Q3X_FULL_ATTENTION_C512_FORCE_V2_BASELINE. It uses only the tactic returned
// by select_fixed_bulk_causal_gqa_prefill_tactic().
[[nodiscard]] int launch_bulk_causal_gqa_sigmoid_gate_24_4_256_fixed_cuda(
    const std::uint16_t* query_tile, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, const std::uint16_t* gate_tile,
    std::size_t first_position, std::size_t token_count,
    std::uint16_t* output_tile, void* cuda_stream = nullptr) noexcept;

// Architecture-candidate panel-wide counterpart. Query/Gate/output remain
// panel-local [token_count,24,256] BF16 arrays while K/V are global NHD
// caches. The grouped-Q64 Tensor Core online-softmax kernel owns the complete
// logical panel in one launch and scans all causal K/V visible to each query.
// This entry is explicit and environment-independent; it never falls back to
// generic QT2. The arithmetic boundary remains Attention BF16 -> sigmoid
// Gate -> BF16, but its reduction tree is qualified separately from the
// exact segmented incumbent before any production promotion. All five arrays
// must be pairwise disjoint and 16-byte aligned.
[[nodiscard]] constexpr bool
can_launch_bulk_causal_gqa_group_q64_panel(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kFirstPositionCapacity =
      std::size_t{1U} << kBulkCausalGqaGroupQ64FirstPositionBits;
  return token_count >= 2U &&
         token_count <= kBulkCausalGqaGroupQ64PanelMaximumTokens &&
         first_position < kFirstPositionCapacity &&
         first_position <=
             kBulkCausalGqaMaximumSequenceLength - token_count;
}

[[nodiscard]] int
launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
    const std::uint16_t* query_panel, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, const std::uint16_t* gate_panel,
    std::size_t first_position, std::size_t token_count,
    std::uint16_t* output_panel, void* cuda_stream = nullptr) noexcept;

// Default-off Attention-v4 architecture surface.  It preserves the v3
// per-warp Q16, K16 online-softmax update order and BF16 boundaries while
// aggregating two former Q64 CTAs into one Q128/8-warp CTA.  The initial
// bring-up deliberately retains the v3 linear shared layout and alternating
// single-slot K/V cp.async pipeline.  XOR-swizzled direct ldmatrix/mma and a
// true two-stage K/V buffer remain later mechanisms on this same surface.
// This launcher is explicit, environment-independent, and has no default or
// production dispatch/fallback eligibility.  It may be selected only by the
// sealed, default-off, accuracy-unqualified layer-major tactic.
[[nodiscard]] constexpr bool
can_launch_bulk_causal_gqa_group_q128_v4_panel(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  return can_launch_bulk_causal_gqa_group_q64_panel(first_position,
                                                    token_count);
}

[[nodiscard]] constexpr std::size_t
bulk_causal_gqa_group_q128_v4_grid_x(
    const std::size_t token_count) noexcept {
  constexpr std::size_t kQueriesPerKvHead = 6U;
  return (token_count * kQueriesPerKvHead +
          kBulkCausalGqaGroupQ128V4PackedQueryTile - 1U) /
         kBulkCausalGqaGroupQ128V4PackedQueryTile;
}

[[nodiscard]] int
launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
    const std::uint16_t* query_panel, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, const std::uint16_t* gate_panel,
    std::size_t first_position, std::size_t token_count,
    std::uint16_t* output_panel, void* cuda_stream = nullptr) noexcept;

// Default-off exact logical-panel Attention surface backed by the vendored
// FlashInfer single-Prefill dataflow. query_panel, gate_panel, and
// output_panel are panel-local [token_count,24,256] BF16 arrays. key_cache and
// value_cache are global contiguous NHD [first_position+token_count,4,256]
// BF16 arrays. FlashInfer owns the exact causal mask for qo_len=token_count
// and kv_len=first_position+token_count, retains FP32 online-softmax state,
// and writes a BF16 Attention boundary before the existing sigmoid gate and
// final BF16 rounding.
//
// The capability query has no CUDA side effects and reports only whether the
// test-only compile-time admission is present in the linked kernel library.
// The geometry query performs no admission or device query. The launcher is
// explicit and environment-independent: it has no fallback or selector, and
// returns cudaErrorNotSupported when the admission was not compiled. All five
// arrays must be pairwise disjoint and 16-byte aligned.
[[nodiscard]] bool
has_bulk_causal_gqa_flashinfer_exact_panel_cuda() noexcept;

[[nodiscard]] constexpr bool
can_launch_bulk_causal_gqa_flashinfer_exact_panel(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  return token_count >= 2U &&
         token_count <= kBulkCausalGqaFlashInferExactPanelMaximumTokens &&
         first_position <=
             kBulkCausalGqaMaximumSequenceLength - token_count;
}

[[nodiscard]] int
launch_bulk_causal_gqa_sigmoid_gate_24_4_256_flashinfer_exact_panel_fixed_cuda(
    const std::uint16_t* query_panel, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, const std::uint16_t* gate_panel,
    std::size_t first_position, std::size_t token_count,
    std::uint16_t* output_panel, void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::runtime
