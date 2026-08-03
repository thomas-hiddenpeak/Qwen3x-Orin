#pragma once

#include "q3x/model/model_config.h"
#include "q3x/runtime/long_prefill_layer_major.h"
#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/request_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace q3x::runtime {

inline constexpr std::size_t kReferenceVocabularySize = 248'320U;
inline constexpr std::size_t kReferenceHiddenSize = 5'120U;
inline constexpr std::size_t kReferenceIntermediateSize = 17'408U;
inline constexpr std::size_t kReferenceDecoderLayerCount = 64U;
inline constexpr std::size_t kReferenceTraceElements =
    (2U + 2U * kReferenceDecoderLayerCount) * kReferenceHiddenSize;
inline constexpr std::size_t kReferenceNoLayer =
    static_cast<std::size_t>(-1);

enum class ReferenceRunnerError : std::uint8_t {
  kNone = 0,
  kInvalidDependency,
  kInvalidModelWeights,
  kInvalidRequestState,
  kInvalidLayerSchedule,
  kCudaFailure,
  kAllocationFailure,
  kInvalidRunner,
  kPoisoned,
  kTokenOutOfRange,
  kCapacityExceeded,
  kTraceUnavailable,
  kNonFiniteLogits,
  kStateCommitFailure,
  kInvalidStepOptions,
};

struct ReferenceRunnerStatus {
  ReferenceRunnerError error = ReferenceRunnerError::kNone;
  int cuda_error = 0;
  std::size_t layer = kReferenceNoLayer;
  const char* operation = nullptr;

  [[nodiscard]] bool ok() const noexcept {
    return error == ReferenceRunnerError::kNone && cuda_error == 0;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

[[nodiscard]] const char* reference_runner_error_string(
    ReferenceRunnerError error) noexcept;

struct ReferenceRunnerOptions {
  // Reserve one pinned BF16 trace buffer at factory time. A step only copies
  // activations when ReferenceStepOptions::capture_trace is true.
  bool enable_trace = false;
  // Explicitly opt into the SM87 weight-only projection kernels. Correctness
  // reference dispatch remains the stable default.
  ProjectionBackend projection_backend = ProjectionBackend::kReference;
  // Select the authenticated full-model A4 Prefill route for this runner.
  // Engine callers set this only after attaching all 400 sidecars. The
  // environment/test hook remains available for direct runner experiments,
  // but production selection must not depend on thread-local initialization.
  bool enable_a4w4_full_prefill_admission = false;
};

enum class ReferenceLogitsMode : std::uint8_t {
  // Preserve the exact public reference result: chosen logit, stable
  // logsumexp, maximum log-probability, and greedy token id.
  kFullStatistics = 0,
  // Validate finiteness and return only the greedy token id. This is intended
  // for callers such as the CLI that do not expose probability statistics.
  kPredictedTokenOnly,
};

[[nodiscard]] constexpr bool is_valid_reference_logits_mode(
    const ReferenceLogitsMode mode) noexcept {
  return mode == ReferenceLogitsMode::kFullStatistics ||
         mode == ReferenceLogitsMode::kPredictedTokenOnly;
}

struct ReferenceStepOptions {
  // False is the prompt-prefix path: all 64 layers and persistent-state
  // updates still execute, while lm_head and the logits D2H copy are skipped.
  bool compute_logits = true;
  bool capture_trace = false;
  bool measure_timing = false;
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
};

struct ReferenceStepLogits {
  std::uint32_t predicted_token_id = 0U;
  float chosen_logit = 0.0F;
  double max_log_probability = 0.0;
  double logsumexp = 0.0;
};

struct ReferenceStepPrediction {
  std::uint32_t predicted_token_id = 0U;
};

struct ReferenceStepTiming {
  // End-to-end host elapsed time, including the required stream synchronize
  // and (when requested) BF16-logits analysis.
  double elapsed_milliseconds = 0.0;
};

struct ReferenceStepResult {
  std::uint32_t position = 0U;
  std::uint32_t input_token_id = 0U;
  std::optional<ReferenceStepLogits> logits;
  std::optional<ReferenceStepTiming> timing;
  std::optional<ReferenceStepPrediction> prediction;
};

struct ReferenceStepOutcome {
  std::optional<ReferenceStepResult> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

inline constexpr std::size_t kReferenceDecodeGraphP2MaximumSlots = 64U;

// Test-only screening surface for a short fixed-position full Decode CUDA
// Graph cache. It is deliberately bounded and is not the production decode
// scheduler. prepare captures, instantiates, and uploads the current-position
// slot without executing or committing; replay selects that position's slot
// and uses the same synchronize, host prediction validation, and commit
// boundary as step.
struct ReferenceDecodeGraphP1Stats {
  std::uint32_t position = 0U;
  std::uint32_t input_token_id = 0U;
  std::size_t node_count = 0U;
  std::size_t kernel_node_count = 0U;
  std::size_t memcpy_node_count = 0U;
  std::size_t other_node_count = 0U;
  double capture_enqueue_milliseconds = 0.0;
  double topology_inspection_milliseconds = 0.0;
  double instantiate_milliseconds = 0.0;
  double upload_ready_milliseconds = 0.0;
  double total_prepare_milliseconds = 0.0;
};

struct ReferenceDecodeGraphP1PrepareOutcome {
  std::optional<ReferenceDecodeGraphP1Stats> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Engine-lifetime preparation result for a contiguous fixed-position cache.
// graphs[0..graph_count) are packed in ascending position order. The complete
// bank is published only after every requested slot is uploaded and ready.
struct ReferenceDecodeGraphCachePrepareResult {
  std::array<ReferenceDecodeGraphP1Stats,
             kReferenceDecodeGraphP2MaximumSlots>
      graphs{};
  std::size_t graph_count = 0U;
  std::uint64_t prepared_mask = 0U;
};

struct ReferenceDecodeGraphCachePrepareOutcome {
  std::optional<ReferenceDecodeGraphCachePrepareResult> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct ReferencePrefillTileOptions {
  bool measure_timing = false;
  // Test-only whole-prompt admission contract. A successful tile retains the
  // final token's already-normalized hidden row for exactly one subsequent
  // finish_prefill_from_retained_tile call. The tile still commits every
  // persistent KV/GDN/conv update and the complete logical sequence length.
  bool retain_last_hidden_for_logits = false;
};

// A prefix tile never produces logits or trace data. The C512 fixed-capacity
// result keeps the runner boundary allocation-free while retaining
// one position/input record per committed token for the high-level generation
// transcript. When timing is requested, timing contains the aggregate tile
// latency. Individual step timings are absent for M>1; M=1 preserves the
// delegated step timing.
struct ReferencePrefillTileResult {
  std::array<ReferenceStepResult, kMaximumRequestPrefillChunkSize> steps{};
  std::size_t step_count = 0U;
  std::optional<ReferenceStepTiming> timing;
};

struct ReferencePrefillTileOutcome {
  std::optional<ReferencePrefillTileResult> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Whole-prompt layer-major admission result. Unlike the fixed C512 tile
// transcript above, the runner reports only the committed interval and its
// aggregate timing; the host generation controller already owns the prompt
// token ids and materializes their allocation-free step metadata itself.
struct ReferenceLongPrefillResult {
  std::uint32_t first_position = 0U;
  std::size_t token_count = 0U;
  // Independent request-local proof for the incumbent K256 Attention
  // projection implementation.  The physical count is one launch per
  // Linear input, Full input, or Attention-O topology; the logical count
  // expands the fused input topologies back to their two or three planes.
  std::size_t attention_k256_m128n256_incumbent_launch_hits = 0U;
  std::size_t
      attention_k256_m128n256_incumbent_logical_projection_hits = 0U;
  // The A-exchange/B4 successor never aliases the incumbent counters.  This
  // lets a real API request prove a single-selector replacement rather than
  // merely proving that some K256 Attention implementation ran.
  std::size_t attention_k256_m128n256_a_exchange_b4_launch_hits = 0U;
  std::size_t
      attention_k256_m128n256_a_exchange_b4_logical_projection_hits = 0U;
  // Request-local proof that the default-off alternating K256 Gate+Up route
  // owned every decoder layer.  The runner leaves this at zero for every
  // other route; consumers must not infer selection from the environment.
  std::size_t gateup_alternating_launch_hits = 0U;
  // Request-local proof that the default-off LDSM pair-feed Gate+Up route
  // owned every decoder layer.  This is independent from the alternating
  // incumbent counter so production API evidence proves an actual B/C swap.
  std::size_t gateup_ldmatrix_pairfeed_launch_hits = 0U;
  // Request-local proof that the structural K256 MLP package owned Gate+Up,
  // product publication, and Down for every decoder layer.  The package
  // counter is independent from every K512 experiment and increments only
  // after both K256 consumers enqueue successfully.
  std::size_t mlp_k256_m128n256_pairfeed_package_launch_hits = 0U;
  // Request-local proof that the default-off full-projection-serial
  // M128N128 Gate+Up route owned every decoder layer.
  std::size_t gateup_m128n128_projection_serial_launch_hits = 0U;
  // Request-local proof that the default-off M128N64 same-CTA Gate-to-Up
  // route owned every decoder layer.  It never aliases the older projection-
  // serial or pair-feed counters.
  std::size_t gateup_m128n64_same_cta_launch_hits = 0U;
  // Request-local proof that the default-off M128N512 Gate+Up route fused
  // the canonical K512 intermediate publication once for every decoder
  // layer.  This never aliases any older Gate+Up selector's counter.
  std::size_t gateup_m128n512_fused_quantize_launch_hits = 0U;
  // Request-local proof that the paired-GateUp/canonical-Down admission
  // launched its M128N512 Gate+Up kernel once for every decoder layer and
  // projection span.  It remains zero for every other route.
  std::size_t gateup_m128n512_paired_ldmatrix_launch_hits = 0U;
  // Request-local proof that the projection-major publication launched the
  // fused M64N512 Gate+Up -> signed-A4/K512 edge once for every decoder
  // layer and projection span.  It never aliases the older paired-layout
  // counter even though both routes consume a composite MLP publication.
  std::size_t gateup_m64n128_register_pipeline_launch_hits = 0U;
  // Request-local proof that the paired-v2 publication used the M64N8
  // same-warp Gate+Up register pipeline.  This is deliberately independent
  // from both the older paired-layout Gate and the projection-major route.
  std::size_t gateup_m64n8_paired_warp_register_pipeline_launch_hits = 0U;
  // The optional pair-ring Down selector has an independent request-local
  // proof.  Gate-only admission deliberately leaves this at zero.
  std::size_t down_m128n128_ldmatrix_pairring_launch_hits = 0U;
  // Request-local proof for the independent 16-warp pair-ring successor.
  // It is never aliased with the incumbent pair-ring counter, so an external
  // real-API run can prove exactly which Down implementation owned the span.
  std::size_t down_m128n128_16warp_pairring_launch_hits = 0U;
  // Mutually exclusive request-local proof for the incumbent C512 native GDN
  // route and its full-prompt persistent-state successor. Logical-token hits
  // count one token for each Linear-Attention layer that consumed it.
  std::size_t gdn_chunk64_native_launch_hits = 0U;
  std::size_t gdn_chunk64_native_logical_token_hits = 0U;
  std::size_t gdn_prompt_span_macro_launch_hits = 0U;
  std::size_t gdn_prompt_span_macro_logical_token_hits = 0U;
  std::optional<ReferenceStepTiming> timing;
};

struct ReferenceLongPrefillOutcome {
  std::optional<ReferenceLongPrefillResult> value;
  ReferenceRunnerStatus status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && status.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct ConstBf16Span {
  const std::uint16_t* data = nullptr;
  std::size_t size = 0U;

  [[nodiscard]] bool empty() const noexcept { return size == 0U; }
  [[nodiscard]] const std::uint16_t& operator[](
      const std::size_t index) const noexcept {
    return data[index];
  }
};

// Non-owning view of the most recently captured, successfully committed step.
// It is invalidated by reset, runner destruction, or the next captured step.
// Raw layout is embedding, then [hidden_i, residual_i] for i=0..63, then
// final_norm. Each logical vector contains exactly 5120 BF16 elements.
struct ReferenceTraceView {
  std::uint32_t position = 0U;
  std::uint32_t input_token_id = 0U;
  const std::uint16_t* data = nullptr;
  std::size_t element_count = 0U;

  [[nodiscard]] ConstBf16Span raw() const noexcept;
  [[nodiscard]] ConstBf16Span embedding() const noexcept;
  [[nodiscard]] ConstBf16Span layer_hidden(
      std::size_t layer) const noexcept;
  [[nodiscard]] ConstBf16Span layer_residual(
      std::size_t layer) const noexcept;
  [[nodiscard]] ConstBf16Span final_norm() const noexcept;
};

namespace reference_runner_detail {

// Prompt-span native GDN is a model-specific, independently admitted route.
// It keeps the existing C64 algebra and the existing C512 BF16 state boundary,
// but one CTA owns a (value head, value half) pair across all chunks in the
// span.  These constants are part of the host/device allocation contract; the
// route is never allowed to silently grow the legacy C512 workspace.
inline constexpr std::size_t kGdnPromptSpanChunkTokens = 64U;
inline constexpr std::size_t kGdnPromptSpanStateBoundaryChunks = 8U;
inline constexpr std::size_t kGdnPromptSpanMinimumTokens = 513U;
inline constexpr std::size_t kGdnPromptSpanMaximumTokens = 4'096U;
inline constexpr std::size_t kGdnPromptSpanWorkspaceAlignment = 256U;
inline constexpr std::size_t kGdnPromptSpanMaximumWorkspaceBytes =
    228'065'280U;  // 217.5 MiB at M=4096.

struct GdnPromptSpanWorkspaceRegion final {
  std::size_t offset_bytes = 0U;
  std::size_t size_bytes = 0U;

  [[nodiscard]] constexpr std::size_t end_bytes() const noexcept {
    return offset_bytes + size_bytes;
  }
};

// The large-M route deliberately materializes only the arrays required by
// the unchanged conv/WY/norm stages.  In particular, boundary_state and
// v_new are absent: the fused persistent-state + BV64 consumer retains both
// inside the owning CTA.
struct GdnPromptSpanWorkspacePlan final {
  std::size_t logical_token_capacity = 0U;
  std::size_t padded_token_capacity = 0U;
  std::size_t chunk_count = 0U;
  GdnPromptSpanWorkspaceRegion compact_q{};
  GdnPromptSpanWorkspaceRegion compact_k{};
  GdnPromptSpanWorkspaceRegion convolved_v{};
  GdnPromptSpanWorkspaceRegion transform{};
  GdnPromptSpanWorkspaceRegion w{};
  GdnPromptSpanWorkspaceRegion u{};
  GdnPromptSpanWorkspaceRegion raw_gram{};
  GdnPromptSpanWorkspaceRegion gamma{};
  GdnPromptSpanWorkspaceRegion beta{};
  std::size_t total_bytes = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return logical_token_capacity >= kGdnPromptSpanMinimumTokens &&
           logical_token_capacity <= kGdnPromptSpanMaximumTokens &&
           padded_token_capacity ==
               chunk_count * kGdnPromptSpanChunkTokens &&
           padded_token_capacity >= logical_token_capacity &&
           compact_q.end_bytes() <= compact_k.offset_bytes &&
           compact_k.end_bytes() <= convolved_v.offset_bytes &&
           convolved_v.end_bytes() <= transform.offset_bytes &&
           transform.end_bytes() <= w.offset_bytes &&
           w.end_bytes() <= u.offset_bytes &&
           u.end_bytes() <= raw_gram.offset_bytes &&
           raw_gram.end_bytes() <= gamma.offset_bytes &&
           gamma.end_bytes() <= beta.offset_bytes &&
           beta.end_bytes() == total_bytes &&
           total_bytes <= kGdnPromptSpanMaximumWorkspaceBytes;
  }
};

[[nodiscard]] constexpr std::size_t gdn_prompt_span_align_workspace(
    const std::size_t byte_count) noexcept {
  return (byte_count + kGdnPromptSpanWorkspaceAlignment - 1U) /
         kGdnPromptSpanWorkspaceAlignment *
         kGdnPromptSpanWorkspaceAlignment;
}

[[nodiscard]] constexpr GdnPromptSpanWorkspaceRegion
gdn_prompt_span_workspace_region(
    const std::size_t previous_end,
    const std::size_t byte_count) noexcept {
  return {gdn_prompt_span_align_workspace(previous_end), byte_count};
}

[[nodiscard]] constexpr GdnPromptSpanWorkspacePlan
gdn_prompt_span_workspace_plan(
    const std::size_t logical_token_capacity) noexcept {
  if (logical_token_capacity < kGdnPromptSpanMinimumTokens ||
      logical_token_capacity > kGdnPromptSpanMaximumTokens) {
    return {};
  }

  constexpr std::size_t kBf16Bytes = 2U;
  constexpr std::size_t kFp32Bytes = 4U;
  constexpr std::size_t kQueryKeyHeads = 16U;
  constexpr std::size_t kValueHeads = 48U;
  constexpr std::size_t kHeadDimension = 128U;
  constexpr std::size_t kWyRank = 64U;
  const std::size_t chunk_count =
      (logical_token_capacity + kGdnPromptSpanChunkTokens - 1U) /
      kGdnPromptSpanChunkTokens;
  const std::size_t padded_token_capacity =
      chunk_count * kGdnPromptSpanChunkTokens;

  GdnPromptSpanWorkspacePlan plan;
  plan.logical_token_capacity = logical_token_capacity;
  plan.padded_token_capacity = padded_token_capacity;
  plan.chunk_count = chunk_count;
  plan.compact_q = gdn_prompt_span_workspace_region(
      0U, padded_token_capacity * kQueryKeyHeads * kHeadDimension *
              kBf16Bytes);
  plan.compact_k = gdn_prompt_span_workspace_region(
      plan.compact_q.end_bytes(),
      padded_token_capacity * kQueryKeyHeads * kHeadDimension *
          kBf16Bytes);
  plan.convolved_v = gdn_prompt_span_workspace_region(
      plan.compact_k.end_bytes(),
      padded_token_capacity * kValueHeads * kHeadDimension * kBf16Bytes);
  plan.transform = gdn_prompt_span_workspace_region(
      plan.convolved_v.end_bytes(),
      chunk_count * kValueHeads * kWyRank * kWyRank * kBf16Bytes);
  plan.w = gdn_prompt_span_workspace_region(
      plan.transform.end_bytes(),
      chunk_count * kValueHeads * kWyRank * kHeadDimension * kBf16Bytes);
  plan.u = gdn_prompt_span_workspace_region(
      plan.w.end_bytes(),
      chunk_count * kValueHeads * kWyRank * kHeadDimension * kBf16Bytes);
  plan.raw_gram = gdn_prompt_span_workspace_region(
      plan.u.end_bytes(),
      chunk_count * kQueryKeyHeads * kWyRank * kWyRank * kFp32Bytes);
  plan.gamma = gdn_prompt_span_workspace_region(
      plan.raw_gram.end_bytes(),
      chunk_count * kValueHeads * kWyRank * kFp32Bytes);
  plan.beta = gdn_prompt_span_workspace_region(
      plan.gamma.end_bytes(),
      chunk_count * kValueHeads * kWyRank * kFp32Bytes);
  plan.total_bytes = plan.beta.end_bytes();
  return plan.valid() ? plan : GdnPromptSpanWorkspacePlan{};
}

struct GdnPromptSpanNativePlan final {
  std::size_t logical_token_count = 0U;
  std::size_t padded_token_count = 0U;
  std::size_t chunk_count = 0U;
  std::size_t virtual_c512_tile_count = 0U;
  std::size_t intermediate_bf16_state_boundaries = 0U;
  std::size_t workspace_bytes = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return logical_token_count >= kGdnPromptSpanMinimumTokens &&
           logical_token_count <= kGdnPromptSpanMaximumTokens &&
           padded_token_count == chunk_count * kGdnPromptSpanChunkTokens &&
           virtual_c512_tile_count ==
               (logical_token_count + 511U) / 512U &&
           workspace_bytes != 0U &&
           workspace_bytes <= kGdnPromptSpanMaximumWorkspaceBytes;
  }
};

// The incumbent sends a final C512 remainder of 1..31 tokens through the
// exact recurrent tail.  Until a bit-exact tail is part of this route, those
// shapes must remain on the incumbent path.  Every eighth C64 chunk performs
// an in-CTA BF16 round/reload before continuing, preserving the incumbent's
// C512 state boundary without surrendering CTA ownership.
[[nodiscard]] constexpr GdnPromptSpanNativePlan gdn_prompt_span_native_plan(
    const std::size_t logical_token_count) noexcept {
  const std::size_t c512_tail = logical_token_count % 512U;
  if (logical_token_count < kGdnPromptSpanMinimumTokens ||
      logical_token_count > kGdnPromptSpanMaximumTokens ||
      (c512_tail != 0U && c512_tail < 32U)) {
    return {};
  }
  const GdnPromptSpanWorkspacePlan workspace =
      gdn_prompt_span_workspace_plan(logical_token_count);
  if (!workspace.valid()) {
    return {};
  }
  const std::size_t intermediate_boundaries =
      (workspace.chunk_count - 1U) /
      kGdnPromptSpanStateBoundaryChunks;
  return {logical_token_count,
          workspace.padded_token_capacity,
          workspace.chunk_count,
          (logical_token_count + 511U) / 512U,
          intermediate_boundaries,
          workspace.total_bytes};
}

// Pure-host production admission contract. prompt_span_admission_enabled is
// supplied only by the route's own exact-value environment/runner setting and
// therefore defaults false independently of the legacy native-C512 selector.
[[nodiscard]] constexpr bool use_gdn_prompt_span_native_prefill(
    const bool prompt_span_admission_enabled,
    const bool native_kernel_available,
    const ProjectionBackend backend,
    const model::LayerType layer_type,
    const bool capture_trace,
    const bool optimized_prefill_disabled,
    const std::size_t logical_token_count,
    const std::size_t workspace_capacity_bytes) noexcept {
  const GdnPromptSpanNativePlan plan =
      gdn_prompt_span_native_plan(logical_token_count);
  return prompt_span_admission_enabled && native_kernel_available &&
         backend == ProjectionBackend::kSm87WeightOnly &&
         layer_type == model::LayerType::kLinearAttention &&
         !capture_trace && !optimized_prefill_disabled && plan.valid() &&
         workspace_capacity_bytes >= plan.workspace_bytes;
}

// Public, allocation-free test/oracle helpers. The logits analyzer first
// rounds every value in-place to BF16 RNE and expands it back to FP32, matching
// the vLLM BF16 logits boundary. Any NaN or infinity is rejected. Argmax ties
// select the smallest index.
[[nodiscard]] std::uint16_t float_to_bf16_rne(float value) noexcept;
[[nodiscard]] float bf16_to_float(std::uint16_t bits) noexcept;
[[nodiscard]] float round_float_to_bf16(float value) noexcept;

enum class LogitsAnalysisStatus : std::uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kNonFinite,
};

struct LogitsAnalysis {
  LogitsAnalysisStatus status = LogitsAnalysisStatus::kInvalidArgument;
  std::size_t predicted_index = 0U;
  float maximum = 0.0F;
  double logsumexp = 0.0;
  double max_log_probability = 0.0;

  [[nodiscard]] bool ok() const noexcept {
    return status == LogitsAnalysisStatus::kSuccess;
  }
};

[[nodiscard]] LogitsAnalysis analyze_bf16_logits_in_place(
    float* logits, std::size_t element_count) noexcept;
[[nodiscard]] LogitsAnalysis analyze_bf16_logits_bits(
    const std::uint16_t* logits, std::size_t element_count) noexcept;
[[nodiscard]] LogitsAnalysis analyze_bf16_argmax_in_place(
    float* logits, std::size_t element_count) noexcept;
[[nodiscard]] LogitsAnalysis analyze_bf16_argmax_bits(
    const std::uint16_t* logits, std::size_t element_count) noexcept;

// Exact payload/dimension preflight used by the runner factory. Exposed here
// so small fake weights can test FP8/NVFP4 scalar constraints without a model
// arena or a CUDA context.
[[nodiscard]] bool valid_reference_linear_weight_contract(
    const LinearWeight& weight, std::size_t output_size,
    std::size_t input_size) noexcept;

[[nodiscard]] model::LayerType expected_reference_layer_type(
    std::size_t layer) noexcept;

// Pure-host selector for the exact-shape fused GQA/gate path. first_position
// is zero-based; a complete tile is selected only when every token's causal
// sequence length is within the fused kernel limit.
[[nodiscard]] bool use_fused_gqa_sigmoid_gate_tile(
    std::size_t first_position, std::size_t token_count) noexcept;

// Admission-only decode selector.  The legacy score/softmax/value/gate path
// remains in the same ELF and is selected unless the split-KV admission flag
// is enabled and the dynamic sequence length is in [65, 40960].
[[nodiscard]] bool use_decode_gqa_splitkv(
    std::size_t sequence_length) noexcept;

// Returns the leading token count whose causal positions remain within the
// fused GQA/Gate kernel limit. A C256/C512 tile beginning before position 64
// keeps this prefix fused while its suffix follows the reference fallback.
[[nodiscard]] std::size_t fused_gqa_sigmoid_gate_prefix_token_count(
    std::size_t first_position, std::size_t token_count) noexcept;

// Pure-host selector for the production bulk causal full-attention Prefill
// route. Only an explicitly selected SM87 backend, a full-attention layer,
// and a 2..512-token tile whose complete global causal range fits the kernel
// ABI may bypass the established per-token GQA/Gate schedule. The kernel's
// QT2 grid masks the second row of an odd final query pair.
[[nodiscard]] bool use_bulk_causal_gqa_sigmoid_gate_prefill(
    ProjectionBackend backend, model::LayerType layer_type,
    std::size_t first_position, std::size_t token_count) noexcept;

// Pure-host selector for the fixed Q=24, KV=4, D=256, rotary=64 fused
// full-attention preprocessing tile. It also rejects position-table
// arithmetic overflow; callers retain the split/norm/RoPE fallback.
[[nodiscard]] bool use_qk_rope_tile(
    std::size_t first_position, std::size_t token_count) noexcept;

// Pure-host selector for the same fused preprocessing dataflow when the
// independent-token grid spans one complete request Prefill tile. Unlike the
// standalone Q/K RoPE helper, the production fused kernel has no cross-token
// state and can expose all 512 prompt rows in one launch.
[[nodiscard]] bool use_full_attention_preprocess_tile(
    std::size_t first_position, std::size_t token_count) noexcept;

inline constexpr std::size_t kPrefillResidualRmsM32Tokens = 32U;

// Pure-host decomposition of an arbitrary Prefill span into exact-M32 fused
// tiles and a final 1..31-token reference tail. A zero fused prefix means the
// existing all-reference schedule must remain in force.
struct PrefillResidualRmsM32Schedule {
  std::size_t fused_prefix_tokens = 0U;
  std::size_t fallback_tail_tokens = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return fused_prefix_tokens != 0U;
  }
};

[[nodiscard]] constexpr PrefillResidualRmsM32Schedule
prefill_residual_rms_m32_schedule(
    const std::size_t token_count,
    const std::size_t hidden_size) noexcept {
  if (token_count < kPrefillResidualRmsM32Tokens ||
      token_count > kMaximumRequestPrefillChunkSize ||
      hidden_size != kReferenceHiddenSize) {
    return {};
  }
  const std::size_t fused_prefix_tokens =
      token_count - token_count % kPrefillResidualRmsM32Tokens;
  return {fused_prefix_tokens, token_count - fused_prefix_tokens};
}

// Selects the exact-M32 residual-add plus centered-RMSNorm prefix schedule for
// every M=32..512 span. A final 1..31-token suffix uses the established
// residual-add and centered-RMSNorm launches. When selected, layer 0 retains
// its standalone input norm; each MLP residual produces the normalized input
// for the next layer (or the final norm after the final layer), so no
// subsequent whole-span input/final norm is scheduled.
[[nodiscard]] bool use_m32_prefill_residual_rms_fusion(
    std::size_t token_count, std::size_t hidden_size) noexcept;

// Pure-host selector for exact FP8 C256/C512 whole-chunk Prefill projection.
// Only explicitly selected SM87, linear-attention QKV [10240,5120] and Z
// [6144,5120], full-attention Q [12288,5120] and K/V [1024,5120], or
// attention output [5120,6144], and the production weight/input/output
// alignments may bypass the runner's established tiled schedule. Device
// companion-scale pointers are intentionally irrelevant to this kernel-only
// eligibility decision.
[[nodiscard]] bool use_fp8_whole_chunk_prefill_projection(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::uint16_t* output,
    std::size_t token_count) noexcept;

// Pure-host selector for the exact FP8 C64 attention-output projection.
// Aligned [5120,6144] is handed to the tile dispatcher once; C32, decode,
// near-miss alignment/shape, and all other C64 weights preserve the
// established runner schedule.
[[nodiscard]] bool use_fp8_m64_prefill_attention_output_projection(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::uint16_t* output,
    std::size_t token_count) noexcept;

// Pure-host selector for the exact NVFP4 Down whole-chunk projection. Only
// explicitly selected SM87, [5120,17408], C256/C512, and the narrow kernel's
// production alignments may bypass the runner's established C32 schedule.
// Gate/Up is intentionally ineligible for this selector.
[[nodiscard]] bool use_nvfp4_whole_chunk_prefill_down_projection(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::uint16_t* output,
    std::size_t token_count) noexcept;

// Pure-host selector for the exact NVFP4 Gate/Up whole-chunk pair. Only
// explicitly selected SM87, two aligned [17408,5120] branches, C256/C512,
// and non-overlapping complete aligned output spans may use the runner's
// existing auxiliary-stream fork/join. Device companion-scale pointers remain
// launcher-validation state: a malformed selected payload must fail instead of
// becoming a serial fallback.
[[nodiscard]] bool use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* input,
    std::uint16_t* gate_output, std::uint16_t* up_output,
    std::size_t token_count) noexcept;

// Pure-host selector for the narrow C32/C64 NVFP4 MLP scheduling optimization.
// C64 retains two ordered C32 launches per branch. It accepts only the two
// exact aligned direct-output projections with non-overlapping complete output
// spans, so every route that could touch the shared FP32 fallback scratch
// remains serial.
[[nodiscard]] bool use_nvfp4_m32_prefill_gate_up_dual_stream(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* input,
    std::uint16_t* gate_output, std::uint16_t* up_output,
    std::size_t token_count) noexcept;

// Test-admission controls for the scheduler-wide, all-64-layer vLLM-Marlin
// MLP route. It covers <=32/C64/C256/C512 with vLLM's corresponding M tile;
// the build-time admission remains absent from ordinary production binaries.
// These calls provide deterministic route-hit accounting to real generation.
bool exchange_nvfp4_marlin_prefill_admission_test_enabled(
    bool enabled) noexcept;
std::size_t exchange_nvfp4_marlin_prefill_admission_test_hits(
    std::size_t hits) noexcept;

// Test-admission controls for the complete 208-projection FP8 W8A16 Marlin
// route. A/B remain their canonical BF16 pair because the checkpoint stores
// them as BF16. Ordinary builds return false/zero and contain no W8 kernels.
bool exchange_fp8_marlin_prefill_admission_test_enabled(
    bool enabled) noexcept;
std::size_t exchange_fp8_marlin_prefill_admission_test_hits(
    std::size_t hits) noexcept;

// Test-admission controls for the BF16 A/B M64 whole-span route. The route
// remains absent from ordinary builds and disabled by default in admission
// builds. One hit represents one accepted pair launch: a complete layer/span
// in the whole-span executor, or one legacy C512-or-shorter tile. Each launch
// includes its exact final M1..M63 tail.
bool exchange_bf16_ab_large_m_prefill_admission_test_enabled(
    bool enabled) noexcept;
std::size_t exchange_bf16_ab_large_m_prefill_admission_test_hits(
    std::size_t hits) noexcept;

// Scheduler-wide accounting for the gated calibrated A4W4 Prefill plane.
// generic_projection_hits counts one logical projection per generic GEMM;
// paired_gate_up_hits counts one fused Gate+Up launch while
// logical_projection_hits counts both branches. A complete 64-layer tile is
// admitted only when the local delta is exactly 192 quantizations, 272
// generic projections, 64 paired launches, and all 400 logical projections.
// The final three fields independently prove how many of those launches used
// the opt-in M128 stage-major implementations. Down remains separate because
// its long-K dataflow is not the generic stage-major kernel.
struct A4W4FullPrefillAdmissionHits {
  std::size_t activation_quantize_hits = 0U;
  std::size_t generic_projection_hits = 0U;
  std::size_t paired_gate_up_hits = 0U;
  std::size_t logical_projection_hits = 0U;
  std::size_t complete_model_tile_hits = 0U;
  std::size_t m128_stage_major_generic_projection_hits = 0U;
  std::size_t m128_stage_major_down_projection_hits = 0U;
  std::size_t m128_stage_major_paired_gate_up_hits = 0U;
};

// Independent success-only accounting for the default-off Attention
// supermatrix runtime slice.  Input launches count one physical launcher even
// though they cover two (Linear QKV+Z) or three (Full Q/K/V) logical
// projections.  output_launch_hits covers both Linear and Full Attention O.
// logical_projection_hits is therefore exactly
// 2*linear + 3*full + output for every accepted execution.
struct A4W4AttentionSupermatrixAdmissionHits final {
  std::size_t linear_input_launch_hits = 0U;
  std::size_t full_input_launch_hits = 0U;
  std::size_t output_launch_hits = 0U;
  std::size_t logical_projection_hits = 0U;
};

// Immutable consumer ABI selected from the complete authenticated A4
// inventory when the runner is created.  K128 keeps the physical packed-K64
// code layout, but its activation and weight scales describe pairs of K64
// blocks and therefore must never reach a K64-scale launcher.
enum class A4W4PrefillConsumer : std::uint8_t {
  kUnavailable = 0,
  kK64,
  kK128,
  kK256,
};

[[nodiscard]] constexpr std::uint64_t
authenticated_a4_payload_bytes_for_kind(
    const PrefillSidecarKind kind) noexcept {
  return kind == PrefillSidecarKind::kA4K64
             ? kPrefillA4K64SidecarPayloadBytes
             : kind == PrefillSidecarKind::kA4K128
                   ? kPrefillA4K128SidecarPayloadBytes
                   : kind == PrefillSidecarKind::kA4K256
                         ? kPrefillA4K256SidecarPayloadBytes
                         : 0U;
}

[[nodiscard]] constexpr A4W4PrefillConsumer
a4w4_prefill_consumer_from_contract(
    const PrefillSidecarKind kind,
    const std::uint32_t packed_k_group_size,
    const std::uint32_t scale_group_size) noexcept {
  if (kind == PrefillSidecarKind::kA4K64 &&
      packed_k_group_size == 64U && scale_group_size == 64U) {
    return A4W4PrefillConsumer::kK64;
  }
  if (kind == PrefillSidecarKind::kA4K128 &&
      packed_k_group_size == 64U && scale_group_size == 128U) {
    return A4W4PrefillConsumer::kK128;
  }
  if (kind == PrefillSidecarKind::kA4K256 &&
      packed_k_group_size == 64U && scale_group_size == 256U) {
    return A4W4PrefillConsumer::kK256;
  }
  return A4W4PrefillConsumer::kUnavailable;
}

[[nodiscard]] constexpr bool a4w4_prefill_consumer_supports_token_count(
    const A4W4PrefillConsumer consumer,
    const std::size_t token_count) noexcept {
  return token_count != 0U &&
         (consumer == A4W4PrefillConsumer::kK64 ||
          (consumer == A4W4PrefillConsumer::kK128 &&
           token_count % 64U == 0U));
}

// Whole-M projection spans have independent packed/scaled activation and BF16
// output workspaces whose capacity is a C512 multiple.  A K128 publication can
// therefore execute a natural prompt tail by quantizing only the logical rows,
// explicitly zero-filling the packed/scaled rows through the next M128 boundary,
// and launching the incumbent K128 consumer on that padded M.  Hidden state,
// attention, recurrent state, residuals, and the retained final row continue to
// use logical_token_count.  This plan is deliberately unavailable to the C512
// tile path because that path does not own independent padded output storage.
struct A4W4ProjectionSpanPaddingPlan final {
  std::size_t logical_token_count = 0U;
  std::size_t projection_token_count = 0U;
  std::size_t padding_token_count = 0U;
  std::size_t span_capacity = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return logical_token_count != 0U &&
           projection_token_count >= logical_token_count &&
           projection_token_count <= span_capacity &&
           padding_token_count ==
               projection_token_count - logical_token_count;
  }
};

[[nodiscard]] constexpr A4W4ProjectionSpanPaddingPlan
a4w4_projection_span_padding_plan(
    const A4W4PrefillConsumer consumer,
    const std::size_t logical_token_count,
    const std::size_t span_capacity) noexcept {
  if (consumer == A4W4PrefillConsumer::kUnavailable ||
      logical_token_count == 0U || logical_token_count > span_capacity) {
    return {};
  }
  if (consumer == A4W4PrefillConsumer::kK64) {
    return {logical_token_count, logical_token_count, 0U, span_capacity};
  }
  constexpr std::size_t kSharedScaleMAlignment = 128U;
  constexpr std::size_t kMaximum = static_cast<std::size_t>(-1);
  if (logical_token_count > kMaximum - (kSharedScaleMAlignment - 1U)) {
    return {};
  }
  const std::size_t projection_token_count =
      (logical_token_count + kSharedScaleMAlignment - 1U) /
      kSharedScaleMAlignment * kSharedScaleMAlignment;
  if (projection_token_count > span_capacity) {
    return {};
  }
  return {logical_token_count, projection_token_count,
          projection_token_count - logical_token_count, span_capacity};
}

[[nodiscard]] constexpr bool
a4w4_prefill_consumer_supports_projection_span_prompt(
    const A4W4PrefillConsumer consumer,
    const std::size_t prompt_token_count,
    const std::size_t span_capacity) noexcept {
  if (prompt_token_count == 0U || span_capacity == 0U) {
    return false;
  }
  const std::size_t first_span_token_count =
      prompt_token_count < span_capacity ? prompt_token_count : span_capacity;
  if (!a4w4_projection_span_padding_plan(
           consumer, first_span_token_count, span_capacity).valid()) {
    return false;
  }
  const std::size_t tail_token_count = prompt_token_count % span_capacity;
  return tail_token_count == 0U ||
         a4w4_projection_span_padding_plan(
             consumer, tail_token_count, span_capacity).valid();
}

[[nodiscard]] constexpr bool a4w4_prefill_inventory_consumers_match(
    const A4W4PrefillConsumer selected,
    const A4W4PrefillConsumer candidate) noexcept {
  return selected != A4W4PrefillConsumer::kUnavailable &&
         selected == candidate;
}

// Complete-cell v2 is a model-specific Gate+Up route, not a generic GEMM
// selector.  The consumer is the immutable result of authenticating the
// complete 400-projection inventory. projection_token_count is the runner's
// internal M after ceil128 padding; logical prompt tails never reach this
// selector directly.  Capacities may exceed the selected M (the whole-span
// workspace does), but every input, weight, and Down-publication plane must
// cover the complete selected matrix.
struct A4W4GateUpCompleteCellV2RouteQuery final {
  bool admission_enabled = false;
  A4W4PrefillConsumer inventory_consumer =
      A4W4PrefillConsumer::kUnavailable;
  std::size_t projection_token_count = 0U;
  std::size_t gate_output_size = 0U;
  std::size_t gate_input_size = 0U;
  std::size_t up_output_size = 0U;
  std::size_t up_input_size = 0U;
  std::size_t packed_input_capacity_bytes = 0U;
  std::size_t input_scale_capacity_elements = 0U;
  std::size_t gate_weight_capacity_bytes = 0U;
  std::size_t gate_scale_capacity_elements = 0U;
  std::size_t up_weight_capacity_bytes = 0U;
  std::size_t up_scale_capacity_elements = 0U;
  std::size_t packed_output_capacity_bytes = 0U;
  std::size_t output_scale_capacity_elements = 0U;
};

// Projection v3 is an independent default-off successor candidate.  It owns
// the same external M64/N128/K128 ABI as complete-cell v2, but its runtime
// gate is deliberately separate so compiling or enabling either experiment
// cannot silently enable the other.  projection_token_count is the runner's
// internal ceil128 M; the authenticated inventory and every exact capacity are
// mandatory before the new warp-specialized kernel may be selected.
struct A4W4GateUpProjectionV3RouteQuery final {
  bool admission_enabled = false;
  A4W4PrefillConsumer inventory_consumer =
      A4W4PrefillConsumer::kUnavailable;
  std::size_t projection_token_count = 0U;
  std::size_t gate_output_size = 0U;
  std::size_t gate_input_size = 0U;
  std::size_t up_output_size = 0U;
  std::size_t up_input_size = 0U;
  std::size_t packed_input_capacity_bytes = 0U;
  std::size_t input_scale_capacity_elements = 0U;
  std::size_t gate_weight_capacity_bytes = 0U;
  std::size_t gate_scale_capacity_elements = 0U;
  std::size_t up_weight_capacity_bytes = 0U;
  std::size_t up_scale_capacity_elements = 0U;
  std::size_t packed_output_capacity_bytes = 0U;
  std::size_t output_scale_capacity_elements = 0U;
};

[[nodiscard]] constexpr bool a4w4_matrix_capacity_covers(
    const std::size_t capacity, const std::size_t outer,
    const std::size_t inner, const std::size_t inner_group) noexcept {
  if (outer == 0U || inner == 0U || inner_group == 0U ||
      inner % inner_group != 0U) {
    return false;
  }
  const std::size_t groups = inner / inner_group;
  constexpr std::size_t maximum = static_cast<std::size_t>(-1);
  return outer <= maximum / groups && capacity >= outer * groups;
}

enum class A4W4AttentionSupermatrixFamily : std::uint8_t {
  kLinearInput = 0,
  kFullInput,
  kOutput,
};

struct A4W4AttentionSupermatrixProjectionPlane final {
  std::size_t output_size = 0U;
  std::size_t input_size = 0U;
  std::size_t weight_capacity_bytes = 0U;
  std::size_t weight_scale_capacity_elements = 0U;
  std::size_t output_row_stride_elements = 0U;
  std::size_t output_capacity_elements = 0U;
};

// A capacity-aware, model-specific selector for the three Attention complete
// cells.  projection_token_count is the internal M after K128 whole-span
// ceil128 padding, so the candidate always receives a complete M128 tile.
// inventory_consumer is authenticated once from all 400 sidecars, so a mixed
// or partially published K128 plane can never enter this selector.
struct A4W4AttentionSupermatrixRouteQuery final {
  bool admission_enabled = false;
  A4W4PrefillConsumer inventory_consumer =
      A4W4PrefillConsumer::kUnavailable;
  A4W4AttentionSupermatrixFamily family =
      A4W4AttentionSupermatrixFamily::kLinearInput;
  std::size_t projection_token_count = 0U;
  std::size_t packed_input_capacity_bytes = 0U;
  std::size_t input_scale_capacity_elements = 0U;
  std::array<A4W4AttentionSupermatrixProjectionPlane, 3U> projections{};
};

[[nodiscard]] constexpr bool use_a4w4_attention_projection_cell_route(
    const A4W4AttentionSupermatrixRouteQuery& query,
    const A4W4PrefillConsumer required_consumer,
    const std::size_t scale_group_size) noexcept {
  constexpr std::size_t kLinearQkvOutputSize = 10'240U;
  constexpr std::size_t kLinearZOutputSize = 6'144U;
  constexpr std::size_t kFullQOutputSize = 12'288U;
  constexpr std::size_t kFullKvOutputSize = 1'024U;
  constexpr std::size_t kAttentionOInputSize = 6'144U;
  constexpr std::size_t kAttentionOOutputSize = 5'120U;
  if (!query.admission_enabled ||
      query.inventory_consumer != required_consumer ||
      (scale_group_size != 128U && scale_group_size != 256U) ||
      query.projection_token_count == 0U ||
      query.projection_token_count % 128U != 0U ||
      query.projection_token_count - 1U >
          static_cast<std::size_t>(
              std::numeric_limits<std::uint32_t>::max())) {
    return false;
  }

  std::array<std::size_t, 3U> expected_outputs{};
  std::size_t expected_input = 0U;
  std::size_t projection_count = 0U;
  switch (query.family) {
    case A4W4AttentionSupermatrixFamily::kLinearInput:
      expected_outputs = {kLinearQkvOutputSize, kLinearZOutputSize, 0U};
      expected_input = kReferenceHiddenSize;
      projection_count = 2U;
      break;
    case A4W4AttentionSupermatrixFamily::kFullInput:
      expected_outputs = {
          kFullQOutputSize, kFullKvOutputSize, kFullKvOutputSize};
      expected_input = kReferenceHiddenSize;
      projection_count = 3U;
      break;
    case A4W4AttentionSupermatrixFamily::kOutput:
      expected_outputs = {kAttentionOOutputSize, 0U, 0U};
      expected_input = kAttentionOInputSize;
      projection_count = 1U;
      break;
    default:
      return false;
  }

  if (!a4w4_matrix_capacity_covers(
          query.packed_input_capacity_bytes,
          query.projection_token_count, expected_input, 2U) ||
      !a4w4_matrix_capacity_covers(
          query.input_scale_capacity_elements,
          query.projection_token_count, expected_input, scale_group_size)) {
    return false;
  }
  for (std::size_t index = 0U; index < projection_count; ++index) {
    const A4W4AttentionSupermatrixProjectionPlane& plane =
        query.projections[index];
    if (plane.output_size != expected_outputs[index] ||
        plane.input_size != expected_input ||
        plane.output_row_stride_elements != expected_outputs[index] ||
        !a4w4_matrix_capacity_covers(
            plane.weight_capacity_bytes, expected_outputs[index],
            expected_input, 2U) ||
        !a4w4_matrix_capacity_covers(
            plane.weight_scale_capacity_elements, expected_outputs[index],
            expected_input, scale_group_size) ||
        !a4w4_matrix_capacity_covers(
            plane.output_capacity_elements, query.projection_token_count,
            expected_outputs[index], 1U)) {
      return false;
    }
  }
  for (std::size_t index = projection_count;
       index < query.projections.size(); ++index) {
    const A4W4AttentionSupermatrixProjectionPlane& plane =
        query.projections[index];
    if (plane.output_size != 0U || plane.input_size != 0U ||
        plane.weight_capacity_bytes != 0U ||
        plane.weight_scale_capacity_elements != 0U ||
        plane.output_row_stride_elements != 0U ||
        plane.output_capacity_elements != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool use_a4w4_attention_supermatrix_route(
    const A4W4AttentionSupermatrixRouteQuery& query) noexcept {
  return use_a4w4_attention_projection_cell_route(
      query, A4W4PrefillConsumer::kK128, 128U);
}

[[nodiscard]] constexpr bool use_a4w4_attention_k256_m128n256_route(
    const A4W4AttentionSupermatrixRouteQuery& query) noexcept {
  return use_a4w4_attention_projection_cell_route(
      query, A4W4PrefillConsumer::kK256, 256U);
}

// Exactly one implementation selector may own the authenticated K256
// Attention projection route.  The payload/shape selector above remains the
// shared ABI authority; this selector only chooses the consumer dataflow.
enum class A4W4AttentionK256M128N256Implementation : std::uint8_t {
  kDisabled = 0,
  kIncumbent,
  kAExchangeB4,
  kInvalid,
};

struct A4W4AttentionK256M128N256ImplementationQuery final {
  bool incumbent_requested = false;
  bool a_exchange_b4_requested = false;
  bool a_exchange_b4_l2_macro4x4_requested = false;
  bool a_exchange_b3_m128n128_requested = false;
};

[[nodiscard]] constexpr A4W4AttentionK256M128N256Implementation
select_a4w4_attention_k256_m128n256_implementation(
    const A4W4AttentionK256M128N256ImplementationQuery& query) noexcept {
  if ((query.incumbent_requested && query.a_exchange_b4_requested) ||
      (query.a_exchange_b4_l2_macro4x4_requested &&
       !query.a_exchange_b4_requested) ||
      (query.a_exchange_b3_m128n128_requested &&
       !query.a_exchange_b4_requested) ||
      (query.a_exchange_b3_m128n128_requested &&
       query.a_exchange_b4_l2_macro4x4_requested)) {
    return A4W4AttentionK256M128N256Implementation::kInvalid;
  }
  if (query.a_exchange_b4_requested) {
    return A4W4AttentionK256M128N256Implementation::kAExchangeB4;
  }
  if (query.incumbent_requested) {
    return A4W4AttentionK256M128N256Implementation::kIncumbent;
  }
  return A4W4AttentionK256M128N256Implementation::kDisabled;
}

// The M32N512 edge owner is deliberately a child modifier of the proven
// M64N128/K256 LDSM pair-feed route.  Keeping the parent selected preserves
// the authenticated K512 publication and its request-local accounting while
// this selector proves that an orphan child, a non-span dispatch, or a mixed
// experimental schedule can never silently fall back to the parent kernel.
enum class A4W4GateUpK512PairfeedImplementation : std::uint8_t {
  kDisabled = 0,
  kM64N128K256LdmatrixPairfeed,
  kM32N512Owner,
  kInvalid,
};

struct A4W4GateUpK512PairfeedImplementationQuery final {
  bool parent_pairfeed_requested = false;
  bool m32n512_owner_requested = false;
  bool conflicting_gate_sibling_requested = false;
  bool conflicting_macro_requested = false;
  bool projection_span = false;
};

[[nodiscard]] constexpr A4W4GateUpK512PairfeedImplementation
select_a4w4_gateup_k512_pairfeed_implementation(
    const A4W4GateUpK512PairfeedImplementationQuery& query) noexcept {
  if ((query.m32n512_owner_requested &&
       !query.parent_pairfeed_requested) ||
      (query.m32n512_owner_requested &&
       (query.conflicting_gate_sibling_requested ||
        query.conflicting_macro_requested)) ||
      ((query.parent_pairfeed_requested || query.m32n512_owner_requested) &&
       !query.projection_span)) {
    return A4W4GateUpK512PairfeedImplementation::kInvalid;
  }
  if (query.m32n512_owner_requested) {
    return A4W4GateUpK512PairfeedImplementation::kM32N512Owner;
  }
  if (query.parent_pairfeed_requested) {
    return A4W4GateUpK512PairfeedImplementation::
        kM64N128K256LdmatrixPairfeed;
  }
  return A4W4GateUpK512PairfeedImplementation::kDisabled;
}

[[nodiscard]] constexpr bool use_a4w4_gateup_complete_cell_v2_route(
    const A4W4GateUpCompleteCellV2RouteQuery& query) noexcept {
  return query.admission_enabled &&
         query.inventory_consumer == A4W4PrefillConsumer::kK128 &&
         query.projection_token_count != 0U &&
         query.projection_token_count % 64U == 0U &&
         query.gate_output_size == kReferenceIntermediateSize &&
         query.gate_input_size == kReferenceHiddenSize &&
         query.up_output_size == kReferenceIntermediateSize &&
         query.up_input_size == kReferenceHiddenSize &&
         a4w4_matrix_capacity_covers(
             query.packed_input_capacity_bytes,
             query.projection_token_count, kReferenceHiddenSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.input_scale_capacity_elements,
             query.projection_token_count, kReferenceHiddenSize, 128U) &&
         a4w4_matrix_capacity_covers(
             query.gate_weight_capacity_bytes,
             kReferenceIntermediateSize, kReferenceHiddenSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.gate_scale_capacity_elements,
             kReferenceIntermediateSize, kReferenceHiddenSize, 128U) &&
         a4w4_matrix_capacity_covers(
             query.up_weight_capacity_bytes,
             kReferenceIntermediateSize, kReferenceHiddenSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.up_scale_capacity_elements,
             kReferenceIntermediateSize, kReferenceHiddenSize, 128U) &&
         a4w4_matrix_capacity_covers(
             query.packed_output_capacity_bytes,
             query.projection_token_count, kReferenceIntermediateSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.output_scale_capacity_elements,
             query.projection_token_count, kReferenceIntermediateSize,
             128U);
}

[[nodiscard]] constexpr bool use_a4w4_gateup_projection_v3_route(
    const A4W4GateUpProjectionV3RouteQuery& query) noexcept {
  return query.admission_enabled &&
         query.inventory_consumer == A4W4PrefillConsumer::kK128 &&
         query.projection_token_count != 0U &&
         query.projection_token_count % 64U == 0U &&
         query.gate_output_size == kReferenceIntermediateSize &&
         query.gate_input_size == kReferenceHiddenSize &&
         query.up_output_size == kReferenceIntermediateSize &&
         query.up_input_size == kReferenceHiddenSize &&
         a4w4_matrix_capacity_covers(
             query.packed_input_capacity_bytes,
             query.projection_token_count, kReferenceHiddenSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.input_scale_capacity_elements,
             query.projection_token_count, kReferenceHiddenSize, 128U) &&
         a4w4_matrix_capacity_covers(
             query.gate_weight_capacity_bytes,
             kReferenceIntermediateSize, kReferenceHiddenSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.gate_scale_capacity_elements,
             kReferenceIntermediateSize, kReferenceHiddenSize, 128U) &&
         a4w4_matrix_capacity_covers(
             query.up_weight_capacity_bytes,
             kReferenceIntermediateSize, kReferenceHiddenSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.up_scale_capacity_elements,
             kReferenceIntermediateSize, kReferenceHiddenSize, 128U) &&
         a4w4_matrix_capacity_covers(
             query.packed_output_capacity_bytes,
             query.projection_token_count, kReferenceIntermediateSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.output_scale_capacity_elements,
             query.projection_token_count, kReferenceIntermediateSize,
             128U);
}

enum class A4W4K128GateUpPrefillRoute : std::uint8_t {
  kBaseline = 0,
  kProjectionV3,
  kCompleteCellV2,
  kRejectedM128StageMajor,
};

// Both structural experiments remain independently gated.  If an operator
// explicitly enables both, v3 owns the one launch; v2 is the lower-priority
// structural incumbent, followed by the archived M128 diagnostic.  A launch
// failure after selection is returned directly by the runtime.
[[nodiscard]] constexpr A4W4K128GateUpPrefillRoute
select_a4w4_k128_gateup_prefill_route(
    const A4W4GateUpProjectionV3RouteQuery& projection_v3_query,
    const A4W4GateUpCompleteCellV2RouteQuery& complete_cell_v2_query,
    const bool rejected_m128_admission_enabled) noexcept {
  if (use_a4w4_gateup_projection_v3_route(projection_v3_query)) {
    return A4W4K128GateUpPrefillRoute::kProjectionV3;
  }
  if (use_a4w4_gateup_complete_cell_v2_route(complete_cell_v2_query)) {
    return A4W4K128GateUpPrefillRoute::kCompleteCellV2;
  }
  return rejected_m128_admission_enabled &&
                 complete_cell_v2_query.inventory_consumer ==
                     A4W4PrefillConsumer::kK128 &&
                 complete_cell_v2_query.projection_token_count != 0U &&
                 complete_cell_v2_query.projection_token_count % 128U == 0U
             ? A4W4K128GateUpPrefillRoute::kRejectedM128StageMajor
             : A4W4K128GateUpPrefillRoute::kBaseline;
}

// Down complete-cell v2 owns only the authenticated model-specific Down
// projection. projection_token_count is the runner's internal ceil128-padded
// M, never the logical prompt length. Natural spans such as P1853 -> M1920
// therefore reach this complete-M128 route without a separate tail kernel.
struct A4W4DownCompleteCellV2RouteQuery final {
  bool admission_enabled = false;
  A4W4PrefillConsumer inventory_consumer =
      A4W4PrefillConsumer::kUnavailable;
  std::size_t projection_token_count = 0U;
  std::size_t output_size = 0U;
  std::size_t input_size = 0U;
  std::size_t packed_input_capacity_bytes = 0U;
  std::size_t input_scale_capacity_elements = 0U;
  std::size_t weight_capacity_bytes = 0U;
  std::size_t weight_scale_capacity_elements = 0U;
  std::size_t output_capacity_elements = 0U;
};

[[nodiscard]] constexpr bool use_a4w4_down_complete_cell_v2_route(
    const A4W4DownCompleteCellV2RouteQuery& query) noexcept {
  return query.admission_enabled &&
         query.inventory_consumer == A4W4PrefillConsumer::kK128 &&
         query.projection_token_count != 0U &&
         query.projection_token_count % 128U == 0U &&
         query.output_size == kReferenceHiddenSize &&
         query.input_size == kReferenceIntermediateSize &&
         a4w4_matrix_capacity_covers(
             query.packed_input_capacity_bytes,
             query.projection_token_count, kReferenceIntermediateSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.input_scale_capacity_elements,
             query.projection_token_count, kReferenceIntermediateSize,
             128U) &&
         a4w4_matrix_capacity_covers(
             query.weight_capacity_bytes, kReferenceHiddenSize,
             kReferenceIntermediateSize, 2U) &&
         a4w4_matrix_capacity_covers(
             query.weight_scale_capacity_elements, kReferenceHiddenSize,
             kReferenceIntermediateSize, 128U) &&
         a4w4_matrix_capacity_covers(
             query.output_capacity_elements,
             query.projection_token_count, kReferenceHiddenSize, 1U);
}

// v3 intentionally retains the authenticated v2 storage/capacity contract.
// Its independent query instance documents that enabling v3 never mutates
// the v2 switch; only warp ownership inside the selected kernel changes.
using A4W4DownCompleteCellV3RouteQuery =
    A4W4DownCompleteCellV2RouteQuery;

[[nodiscard]] constexpr bool use_a4w4_down_complete_cell_v3_route(
    const A4W4DownCompleteCellV3RouteQuery& query) noexcept {
  return use_a4w4_down_complete_cell_v2_route(query);
}

// The consumer argument is the immutable result of validating all 400 A4
// projection sidecars at runner creation.  Consequently these selectors
// cannot admit a partial or mixed K64/K128 publication.  The candidates have
// no tail kernel: M128 alignment and the explicit worker-local switch are
// mandatory.
[[nodiscard]] constexpr bool a4w4_m128_stage_major_common_route(
    const bool admission_enabled,
    const A4W4PrefillConsumer inventory_consumer,
    const std::size_t token_count) noexcept {
  return admission_enabled && inventory_consumer == A4W4PrefillConsumer::kK128 &&
         token_count != 0U && token_count % 128U == 0U;
}

enum class A4W4K128GenericPrefillRoute : std::uint8_t {
  kBaseline = 0,
  kM128StageMajor,
  kDownM128StageMajor,
};

// The exact model-specific Down shape uses its separate long-K stage-major
// dataflow. Other complete N256/K128 shapes use the generic stage-major
// kernel. All tails and unsupported shapes remain on the incumbent K128
// baseline; a selected candidate's launch status is returned directly by the
// runtime and is never converted into a silent baseline retry.
[[nodiscard]] constexpr A4W4K128GenericPrefillRoute
select_a4w4_k128_generic_prefill_route(
    const bool m128_admission_enabled,
    const bool down_m128_admission_enabled,
    const A4W4PrefillConsumer inventory_consumer,
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  const bool generic_route = a4w4_m128_stage_major_common_route(
      m128_admission_enabled, inventory_consumer, token_count);
  const bool down_route = a4w4_m128_stage_major_common_route(
      down_m128_admission_enabled, inventory_consumer, token_count);
  if ((!generic_route && !down_route) ||
      output_size == 0U || output_size % 256U != 0U || input_size == 0U ||
      input_size % 128U != 0U) {
    return A4W4K128GenericPrefillRoute::kBaseline;
  }
  if (output_size == kReferenceHiddenSize &&
      input_size == kReferenceIntermediateSize) {
    return down_route ? A4W4K128GenericPrefillRoute::kDownM128StageMajor
                      : A4W4K128GenericPrefillRoute::kBaseline;
  }
  return generic_route ? A4W4K128GenericPrefillRoute::kM128StageMajor
                       : A4W4K128GenericPrefillRoute::kBaseline;
}

enum class A4W4K128DownPrefillRoute : std::uint8_t {
  kBaseline = 0,
  kCompleteCellV3,
  kCompleteCellV2,
  kRejectedM128StageMajor,
};

// v3 is the highest-priority complete cell, followed by v2 and the archived
// M128 diagnostic. All switches remain independent and default-off.
[[nodiscard]] constexpr A4W4K128DownPrefillRoute
select_a4w4_k128_down_prefill_route(
    const A4W4DownCompleteCellV3RouteQuery& complete_cell_v3_query,
    const A4W4DownCompleteCellV2RouteQuery& complete_cell_v2_query,
    const bool rejected_m128_admission_enabled) noexcept {
  if (use_a4w4_down_complete_cell_v3_route(complete_cell_v3_query)) {
    return A4W4K128DownPrefillRoute::kCompleteCellV3;
  }
  if (use_a4w4_down_complete_cell_v2_route(complete_cell_v2_query)) {
    return A4W4K128DownPrefillRoute::kCompleteCellV2;
  }
  const A4W4K128GenericPrefillRoute old_route =
      select_a4w4_k128_generic_prefill_route(
          false, rejected_m128_admission_enabled,
          complete_cell_v2_query.inventory_consumer,
          complete_cell_v2_query.projection_token_count,
          complete_cell_v2_query.output_size,
          complete_cell_v2_query.input_size);
  return old_route == A4W4K128GenericPrefillRoute::kDownM128StageMajor
             ? A4W4K128DownPrefillRoute::kRejectedM128StageMajor
             : A4W4K128DownPrefillRoute::kBaseline;
}

// Preserve the established v2-only selector contract for callers that do not
// build the v3 vertical slice.
[[nodiscard]] constexpr A4W4K128DownPrefillRoute
select_a4w4_k128_down_prefill_route(
    const A4W4DownCompleteCellV2RouteQuery& complete_cell_v2_query,
    const bool rejected_m128_admission_enabled) noexcept {
  A4W4DownCompleteCellV3RouteQuery disabled_v3 = complete_cell_v2_query;
  disabled_v3.admission_enabled = false;
  return select_a4w4_k128_down_prefill_route(
      disabled_v3, complete_cell_v2_query,
      rejected_m128_admission_enabled);
}

// A trace-capable runner retains authenticated A4 residency but must execute
// the scalar trace path so every requested activation digest is available.
// The unified comparator likewise changes dispatch only.
[[nodiscard]] constexpr bool use_a4w4_full_prefill_tile_route(
    const bool a4_inventory_enabled, const bool trace_enabled,
    const bool optimized_prefill_disabled) noexcept {
  return a4_inventory_enabled && !trace_enabled &&
         !optimized_prefill_disabled;
}

// The paired-GateUp/canonical-Down publication is a complete production
// route, not a leaf-kernel override.  The factory freezes one of two complete
// Gate/Down pairs: the old M128N512 Gate may use only the old LDSM pair-ring
// Down, while the M64N8 paired-warp Gate may use only the 16-warp pair-ring
// Down.  Exactly one Gate selector is required whenever the publication
// master is requested.  The explicit projection_span flag lets non-span
// entry points fail closed.
enum class A4W4PairedGateUpCanonicalDownRoute : std::uint8_t {
  kDisabled = 0,
  kInvalid,
  kOldGateCanonicalDown,
  kOldGateLdmatrixPairringDown,
  kNewPairedWarpGateCanonicalDown,
  kNewPairedWarpGate16WarpPairringDown,
};

struct A4W4PairedGateUpCanonicalDownSelectorQuery final {
  bool master_requested = false;
  bool old_gate_requested = false;
  bool new_paired_warp_gate_requested = false;
  bool old_ldmatrix_pairring_down_requested = false;
  bool new_16warp_pairring_down_requested = false;
  bool legacy_mlp_requested = false;
  bool legacy_gate_requested = false;
  bool legacy_down_requested = false;
  bool projection_span = false;
};

[[nodiscard]] constexpr A4W4PairedGateUpCanonicalDownRoute
select_a4w4_paired_gateup_canonical_down_route(
    const A4W4PairedGateUpCanonicalDownSelectorQuery& query) noexcept {
  const bool any_new_selector = query.master_requested ||
                                query.old_gate_requested ||
                                query.new_paired_warp_gate_requested ||
                                query.old_ldmatrix_pairring_down_requested ||
                                query.new_16warp_pairring_down_requested;
  if (!any_new_selector) {
    return A4W4PairedGateUpCanonicalDownRoute::kDisabled;
  }
  const bool exactly_one_gate =
      query.old_gate_requested != query.new_paired_warp_gate_requested;
  if (!query.master_requested || !exactly_one_gate ||
      !query.projection_span || query.legacy_mlp_requested ||
      query.legacy_gate_requested || query.legacy_down_requested) {
    return A4W4PairedGateUpCanonicalDownRoute::kInvalid;
  }
  if (query.old_gate_requested) {
    if (query.new_16warp_pairring_down_requested) {
      return A4W4PairedGateUpCanonicalDownRoute::kInvalid;
    }
    return query.old_ldmatrix_pairring_down_requested
               ? A4W4PairedGateUpCanonicalDownRoute::
                     kOldGateLdmatrixPairringDown
               : A4W4PairedGateUpCanonicalDownRoute::kOldGateCanonicalDown;
  }
  if (query.old_ldmatrix_pairring_down_requested) {
    return A4W4PairedGateUpCanonicalDownRoute::kInvalid;
  }
  return query.new_16warp_pairring_down_requested
             ? A4W4PairedGateUpCanonicalDownRoute::
                   kNewPairedWarpGate16WarpPairringDown
             : A4W4PairedGateUpCanonicalDownRoute::
                   kNewPairedWarpGateCanonicalDown;
}

[[nodiscard]] constexpr bool
a4w4_paired_gateup_canonical_down_accounting_valid(
    const A4W4PairedGateUpCanonicalDownRoute route,
    const std::size_t projection_span_count,
    const std::size_t old_gate_hits_before,
    const std::size_t old_gate_hits_after,
    const std::size_t new_gate_hits_before,
    const std::size_t new_gate_hits_after,
    const std::size_t old_down_hits_before,
    const std::size_t old_down_hits_after,
    const std::size_t new_down_hits_before,
    const std::size_t new_down_hits_after) noexcept {
  if (route == A4W4PairedGateUpCanonicalDownRoute::kInvalid ||
      old_gate_hits_after < old_gate_hits_before ||
      new_gate_hits_after < new_gate_hits_before ||
      old_down_hits_after < old_down_hits_before ||
      new_down_hits_after < new_down_hits_before ||
      projection_span_count >
          std::numeric_limits<std::size_t>::max() /
              kReferenceDecoderLayerCount) {
    return false;
  }
  const std::size_t expected =
      projection_span_count * kReferenceDecoderLayerCount;
  const std::size_t old_gate_delta =
      old_gate_hits_after - old_gate_hits_before;
  const std::size_t new_gate_delta =
      new_gate_hits_after - new_gate_hits_before;
  const std::size_t old_down_delta =
      old_down_hits_after - old_down_hits_before;
  const std::size_t new_down_delta =
      new_down_hits_after - new_down_hits_before;
  switch (route) {
    case A4W4PairedGateUpCanonicalDownRoute::kDisabled:
      return old_gate_delta == 0U && new_gate_delta == 0U &&
             old_down_delta == 0U && new_down_delta == 0U;
    case A4W4PairedGateUpCanonicalDownRoute::kOldGateCanonicalDown:
      return old_gate_delta == expected && new_gate_delta == 0U &&
             old_down_delta == 0U && new_down_delta == 0U;
    case A4W4PairedGateUpCanonicalDownRoute::
        kOldGateLdmatrixPairringDown:
      return old_gate_delta == expected && new_gate_delta == 0U &&
             old_down_delta == expected && new_down_delta == 0U;
    case A4W4PairedGateUpCanonicalDownRoute::
        kNewPairedWarpGateCanonicalDown:
      return old_gate_delta == 0U && new_gate_delta == expected &&
             old_down_delta == 0U && new_down_delta == 0U;
    case A4W4PairedGateUpCanonicalDownRoute::
        kNewPairedWarpGate16WarpPairringDown:
      return old_gate_delta == 0U && new_gate_delta == expected &&
             old_down_delta == 0U && new_down_delta == expected;
    case A4W4PairedGateUpCanonicalDownRoute::kInvalid:
      return false;
  }
  return false;
}

// The projection-major-GateUp/canonical-Down publication owns a physically
// distinct production route.  Keep its selector and accounting independent
// from the older paired-layout route so API telemetry proves the exact
// weight layout and kernel that executed.
enum class A4W4ProjectionMajorGateUpCanonicalDownRoute : std::uint8_t {
  kDisabled = 0,
  kInvalid,
  kGateOnly,
  kGateAndDown,
};

struct A4W4ProjectionMajorGateUpCanonicalDownSelectorQuery final {
  bool master_requested = false;
  bool gate_requested = false;
  bool down_requested = false;
  bool legacy_publication_requested = false;
  bool legacy_gate_requested = false;
  bool legacy_down_requested = false;
  bool projection_span = false;
};

[[nodiscard]] constexpr A4W4ProjectionMajorGateUpCanonicalDownRoute
select_a4w4_projection_major_gateup_canonical_down_route(
    const A4W4ProjectionMajorGateUpCanonicalDownSelectorQuery& query)
    noexcept {
  const bool any_new_selector =
      query.master_requested || query.gate_requested || query.down_requested;
  if (!any_new_selector) {
    return A4W4ProjectionMajorGateUpCanonicalDownRoute::kDisabled;
  }
  if (!query.master_requested || !query.gate_requested ||
      !query.projection_span || query.legacy_publication_requested ||
      query.legacy_gate_requested || query.legacy_down_requested) {
    return A4W4ProjectionMajorGateUpCanonicalDownRoute::kInvalid;
  }
  return query.down_requested
             ? A4W4ProjectionMajorGateUpCanonicalDownRoute::kGateAndDown
             : A4W4ProjectionMajorGateUpCanonicalDownRoute::kGateOnly;
}

[[nodiscard]] constexpr bool
a4w4_projection_major_gateup_canonical_down_accounting_valid(
    const A4W4ProjectionMajorGateUpCanonicalDownRoute route,
    const std::size_t projection_span_count,
    const std::size_t gate_hits_before,
    const std::size_t gate_hits_after,
    const std::size_t down_hits_before,
    const std::size_t down_hits_after) noexcept {
  if (route == A4W4ProjectionMajorGateUpCanonicalDownRoute::kInvalid ||
      gate_hits_after < gate_hits_before ||
      down_hits_after < down_hits_before ||
      projection_span_count >
          std::numeric_limits<std::size_t>::max() /
              kReferenceDecoderLayerCount) {
    return false;
  }
  const std::size_t expected =
      projection_span_count * kReferenceDecoderLayerCount;
  const std::size_t gate_delta = gate_hits_after - gate_hits_before;
  const std::size_t down_delta = down_hits_after - down_hits_before;
  switch (route) {
    case A4W4ProjectionMajorGateUpCanonicalDownRoute::kDisabled:
      return gate_delta == 0U && down_delta == 0U;
    case A4W4ProjectionMajorGateUpCanonicalDownRoute::kGateOnly:
      return gate_delta == expected && down_delta == 0U;
    case A4W4ProjectionMajorGateUpCanonicalDownRoute::kGateAndDown:
      return gate_delta == expected && down_delta == expected;
    case A4W4ProjectionMajorGateUpCanonicalDownRoute::kInvalid:
      return false;
  }
  return false;
}

// The pair-ring Down kernel is also an independent leaf replacement for the
// authenticated v1 K512 MLP publication.  Gate+Up ownership is deliberately
// absent from this selector: the retained edge, alternating edge, or the v1
// split Gate+Up route may all feed the same authenticated intermediate A4
// layout.  Only another Down selector conflicts with it.
enum class A4W4DownK512M128N128LdmatrixPairringV1Route : std::uint8_t {
  kDisabled = 0,
  kInvalid,
  kEnabled,
};

struct A4W4DownK512M128N128LdmatrixPairringV1SelectorQuery final {
  bool requested = false;
  bool mlp_k512_v1_requested = false;
  bool fragment_native_requested = false;
  bool hybrid_requested = false;
  bool conflicting_down_requested = false;
  bool projection_span = false;
};

[[nodiscard]] constexpr
    A4W4DownK512M128N128LdmatrixPairringV1Route
    select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
        const A4W4DownK512M128N128LdmatrixPairringV1SelectorQuery&
            query) noexcept {
  if (!query.requested) {
    return A4W4DownK512M128N128LdmatrixPairringV1Route::kDisabled;
  }
  if (!query.mlp_k512_v1_requested || query.fragment_native_requested ||
      query.hybrid_requested || query.conflicting_down_requested ||
      !query.projection_span) {
    return A4W4DownK512M128N128LdmatrixPairringV1Route::kInvalid;
  }
  return A4W4DownK512M128N128LdmatrixPairringV1Route::kEnabled;
}

[[nodiscard]] constexpr bool
a4w4_down_k512_m128n128_ldmatrix_pairring_v1_accounting_valid(
    const A4W4DownK512M128N128LdmatrixPairringV1Route route,
    const std::size_t projection_span_count,
    const std::size_t mlp_k512_v1_delta,
    const std::size_t down_hits_before,
    const std::size_t down_hits_after) noexcept {
  if (route ==
          A4W4DownK512M128N128LdmatrixPairringV1Route::kInvalid ||
      down_hits_after < down_hits_before ||
      projection_span_count >
          std::numeric_limits<std::size_t>::max() /
              kReferenceDecoderLayerCount) {
    return false;
  }
  const std::size_t down_delta = down_hits_after - down_hits_before;
  if (route ==
      A4W4DownK512M128N128LdmatrixPairringV1Route::kDisabled) {
    return down_delta == 0U;
  }
  const std::size_t expected =
      projection_span_count * kReferenceDecoderLayerCount;
  return down_delta == expected && down_delta == mlp_k512_v1_delta;
}

// Independent 16-warp successor to the incumbent pair-ring Down leaf.  The
// route is valid only under the authenticated v1 K512 MLP publication and a
// projection span.  It cannot be selected together with the incumbent
// pair-ring, hybrid publication, fragment-native publication, or any other
// Down replacement.  Exact logical/ceil128 padding is revalidated by the
// production launcher at every invocation.
enum class A4W4DownK512M128N128Pairring16V1Route : std::uint8_t {
  kDisabled = 0,
  kInvalid,
  kEnabled,
};

struct A4W4DownK512M128N128Pairring16V1SelectorQuery final {
  bool requested = false;
  bool mlp_k512_v1_requested = false;
  bool fragment_native_requested = false;
  bool hybrid_requested = false;
  bool incumbent_pairring_requested = false;
  bool conflicting_down_requested = false;
  bool projection_span = false;
};

[[nodiscard]] constexpr A4W4DownK512M128N128Pairring16V1Route
select_a4w4_down_k512_m128n128_16warp_pairring_v1_route(
    const A4W4DownK512M128N128Pairring16V1SelectorQuery& query) noexcept {
  if (!query.requested) {
    return A4W4DownK512M128N128Pairring16V1Route::kDisabled;
  }
  if (!query.mlp_k512_v1_requested || query.fragment_native_requested ||
      query.hybrid_requested || query.incumbent_pairring_requested ||
      query.conflicting_down_requested || !query.projection_span) {
    return A4W4DownK512M128N128Pairring16V1Route::kInvalid;
  }
  return A4W4DownK512M128N128Pairring16V1Route::kEnabled;
}

[[nodiscard]] constexpr bool
a4w4_down_k512_m128n128_16warp_pairring_v1_accounting_valid(
    const A4W4DownK512M128N128Pairring16V1Route route,
    const std::size_t projection_span_count,
    const std::size_t mlp_k512_v1_delta,
    const std::size_t down_hits_before,
    const std::size_t down_hits_after) noexcept {
  if (route == A4W4DownK512M128N128Pairring16V1Route::kInvalid ||
      down_hits_after < down_hits_before ||
      projection_span_count >
          std::numeric_limits<std::size_t>::max() /
              kReferenceDecoderLayerCount) {
    return false;
  }
  const std::size_t down_delta = down_hits_after - down_hits_before;
  if (route == A4W4DownK512M128N128Pairring16V1Route::kDisabled) {
    return down_delta == 0U;
  }
  const std::size_t expected =
      projection_span_count * kReferenceDecoderLayerCount;
  return down_delta == expected && down_delta == mlp_k512_v1_delta;
}

bool exchange_a4w4_full_prefill_admission_test_enabled(
    bool enabled) noexcept;
bool exchange_a4w4_m128_stage_major_admission_test_enabled(
    bool enabled) noexcept;
bool exchange_a4w4_down_m128_stage_major_admission_test_enabled(
    bool enabled) noexcept;
bool exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
    bool enabled) noexcept;
std::size_t exchange_a4w4_down_complete_cell_v2_admission_test_hits(
    std::size_t hits) noexcept;
bool exchange_a4w4_down_complete_cell_v3_admission_test_enabled(
    bool enabled) noexcept;
std::size_t exchange_a4w4_down_complete_cell_v3_admission_test_hits(
    std::size_t hits) noexcept;
bool exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
    bool enabled) noexcept;
std::size_t
exchange_a4w4_gateup_complete_cell_v2_admission_test_hits(
    std::size_t hits) noexcept;
bool exchange_a4w4_gateup_projection_v3_admission_test_enabled(
    bool enabled) noexcept;
std::size_t exchange_a4w4_gateup_projection_v3_admission_test_hits(
    std::size_t hits) noexcept;
bool exchange_a4w4_attention_supermatrix_admission_test_enabled(
    bool enabled) noexcept;
A4W4AttentionSupermatrixAdmissionHits
exchange_a4w4_attention_supermatrix_admission_test_hits(
    A4W4AttentionSupermatrixAdmissionHits hits) noexcept;
bool exchange_a4w4_attention_k256_m128n256_admission_test_enabled(
    bool enabled) noexcept;
A4W4AttentionSupermatrixAdmissionHits
exchange_a4w4_attention_k256_m128n256_admission_test_hits(
    A4W4AttentionSupermatrixAdmissionHits hits) noexcept;
bool
exchange_a4w4_attention_k256_m128n256_a_exchange_b4_admission_test_enabled(
    bool enabled) noexcept;
bool exchange_a4w4_attention_k256_m128n128_a_exchange_b3_admission_test_enabled(
    bool enabled) noexcept;
bool exchange_a4w4_gateup_down_k512_edge_m32n512_owner_admission_test_enabled(
    bool enabled) noexcept;
A4W4AttentionSupermatrixAdmissionHits
exchange_a4w4_attention_k256_m128n256_a_exchange_b4_admission_test_hits(
    A4W4AttentionSupermatrixAdmissionHits hits) noexcept;
A4W4FullPrefillAdmissionHits
exchange_a4w4_full_prefill_admission_test_hits(
    A4W4FullPrefillAdmissionHits hits) noexcept;

[[nodiscard]] bool use_fp8_marlin_prefill_projection(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::uint16_t* output,
    std::size_t token_count) noexcept;

// Pure-host validation entry used by tests and factory preflight. It checks
// exact batch-one workspace, cache, RoPE, and 48/16 schedule capacities.
[[nodiscard]] ReferenceRunnerError validate_reference_workspace_plan(
    const RequestMemoryPlan& plan) noexcept;

}  // namespace reference_runner_detail

struct ReferenceRunnerFactoryResult;

// Correctness-first, batch-one CUDA runner. ModelWeights and RequestState are
// non-owning dependencies: both exact objects, their backing CUDA allocations,
// and all bound ResidentWeights storage must outlive this runner. They must not
// be moved, reset externally, or used from another stream while it is alive.
class ReferenceRunner {
 public:
  ~ReferenceRunner();

  ReferenceRunner(const ReferenceRunner&) = delete;
  ReferenceRunner& operator=(const ReferenceRunner&) = delete;
  ReferenceRunner(ReferenceRunner&& other) noexcept;
  ReferenceRunner& operator=(ReferenceRunner&& other) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }
  [[nodiscard]] std::uint32_t current_position() const noexcept;
  [[nodiscard]] ProjectionBackend projection_backend() const noexcept {
    return projection_backend_;
  }
  // True only after runner creation has validated one complete K128 A4
  // inventory and enabled its full-Prefill dispatch contract.  Host routing
  // uses this instead of trusting paths, filenames, or environment state.
  [[nodiscard]] bool authenticated_a4w4_k128_prefill_enabled()
      const noexcept {
    return a4w4_full_prefill_admission_enabled_ &&
           a4w4_prefill_consumer_ ==
               reference_runner_detail::A4W4PrefillConsumer::kK128;
  }
  [[nodiscard]] bool authenticated_a4w4_prefill_enabled() const noexcept {
    return a4w4_full_prefill_admission_enabled_ &&
           (a4w4_prefill_consumer_ ==
                reference_runner_detail::A4W4PrefillConsumer::kK128 ||
            a4w4_prefill_consumer_ ==
                reference_runner_detail::A4W4PrefillConsumer::kK256);
  }

  [[nodiscard]] ReferenceStepOutcome step(
      std::uint32_t input_token_id,
      const ReferenceStepOptions& options = {}) noexcept;

  // Strict SM87 predicted-token-only experiment. Each prepared slot fixes one
  // current_position(); replay updates its root embedding node even when the
  // token is unchanged. Slots are indexed by positions [0, 64). reset may
  // restore a prepared logical position before another replay; graph pointers
  // remain valid because RequestState storage is stable for the runner
  // lifetime.
  [[nodiscard]] ReferenceDecodeGraphP1PrepareOutcome
  prepare_fixed_position_decode_graph_p1(
      std::uint32_t input_token_id) noexcept;
  // Transactionally prepares every position in [first_position,
  // last_position]. The range must currently be empty. Failure leaves the
  // live graph bank unchanged and restores the entry sequence length.
  [[nodiscard]] ReferenceDecodeGraphCachePrepareOutcome
  prepare_fixed_position_decode_graph_cache(
      std::uint32_t first_position, std::uint32_t last_position,
      std::uint32_t input_token_id) noexcept;
  [[nodiscard]] std::uint64_t
  fixed_position_decode_graph_cache_mask() const noexcept;
  // Synchronizes both owned streams before detaching the complete bank. This
  // does not reset request state, trace state, or an existing poison marker.
  [[nodiscard]] ReferenceRunnerStatus
  clear_fixed_position_decode_graph_cache() noexcept;
  [[nodiscard]] bool has_fixed_position_decode_graph_p1(
      std::uint32_t position) const noexcept;
  [[nodiscard]] std::optional<ReferenceDecodeGraphP1Stats>
  fixed_position_decode_graph_p1_stats(
      std::uint32_t position) const noexcept;
  // Callers select their serial fallback with has_* before replay. Directly
  // replaying a missing slot is an invalid experimental operation.
  [[nodiscard]] ReferenceStepOutcome replay_fixed_position_decode_graph_p1(
      std::uint32_t input_token_id, bool measure_timing = false) noexcept;

  // Executes 1..512 non-logit prompt-prefix tokens in layer-major order. The
  // request plan must reserve at least token_count workspace rows. Operations
  // with a narrower kernel contract are enqueued as ordered subtiles.
  // Persistent conv/GDN/KV state is updated in token order. The exact aligned
  // SM87 FP8 C64 attention-output projection uses one exact kernel. Exact
  // aligned C256/C512 FP8 QKV/Z/O and NVFP4 Down projections each use one
  // whole-chunk self-hosted grid. Exact aligned NVFP4 C512 Gate/Up use the
  // native fork/join path when the auxiliary stream is available and the
  // native serial path otherwise; Down remains on the native main-stream
  // route. C32/C64 retains the M32 dual-stream schedule and C64 preserves two
  // ordered C32 launches on each branch. External-library comparators are not
  // linked into the runner and cannot be selected or used as fallbacks.
  // Exact SM87 C256/C512 full-attention tiles use one bulk causal GQA/Gate
  // launch with tile-local Q/Gate/output and global NHD K/V caches. Every
  // fallback remains on the main stream, and the logical request length is
  // committed only after the complete tile synchronizes.
  [[nodiscard]] ReferencePrefillTileOutcome prefill_prefix_tile(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      const ReferencePrefillTileOptions& options = {}) noexcept;

  // Test-build/runtime-gated P513..P40960 path. Embeddings are gathered once
  // into the first full hidden slab, then all C512/tail tiles of layer L run
  // before layer L+1. Existing projection, GDN, and Attention kernel ABIs are
  // reused with explicit global positions. The prompt owns exactly one
  // successful stream synchronization and one logical-length commit.
  [[nodiscard]] ReferenceLongPrefillOutcome prefill_layer_major_prompt(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      bool measure_timing = false) noexcept;

  // Test-only final-prompt admission boundary. Consumes the retained final
  // normalized hidden row from the immediately preceding marked prefill tile
  // and runs only lm_head/logits analysis. It never gathers an embedding,
  // executes a decoder layer, updates persistent model state, or advances the
  // logical request position.
  [[nodiscard]] ReferenceStepOutcome finish_prefill_from_retained_tile(
      std::uint32_t input_token_id,
      const ReferenceStepOptions& options = {}) noexcept;

  // A successful reset synchronizes the owned streams, clears all persistent
  // request state through RequestState::reset_async, clears poison, and
  // invalidates the prior trace. Reset is the only poison recovery operation.
  [[nodiscard]] ReferenceRunnerStatus reset() noexcept;

  [[nodiscard]] std::optional<ReferenceTraceView> last_trace() const noexcept;

 private:
  friend struct ReferenceRunnerFactoryResult;
  friend ReferenceRunnerFactoryResult create_reference_runner(
      const ModelWeights*, RequestState*, const ReferenceRunnerOptions&) noexcept;

  struct Views {
    std::uint16_t* hidden[3]{};
    std::uint16_t* long_prefill_hidden[2]{};
    std::uint16_t* long_prefill_projection_primary = nullptr;
    std::uint16_t* long_prefill_projection_secondary = nullptr;
    std::uint16_t* projection[4]{};
    std::uint8_t* prefill_a4_hidden_packed = nullptr;
    std::uint16_t* prefill_a4_hidden_scales = nullptr;
    std::uint8_t* prefill_a4_intermediate_packed = nullptr;
    std::uint16_t* prefill_a4_intermediate_scales = nullptr;
    std::uint8_t* prefill_a4_gateup_cta_scratch = nullptr;
    std::size_t prefill_a4_gateup_cta_scratch_bytes = 0U;
    std::uint16_t* linear_a = nullptr;
    std::uint16_t* linear_b = nullptr;
    float* fp32_scratch = nullptr;
    std::size_t fp32_scratch_elements = 0U;
    std::uint16_t* conv_state[kReferenceDecoderLayerCount]{};
    std::uint16_t* gdn_state[kReferenceDecoderLayerCount]{};
    std::uint16_t* key_cache[kReferenceDecoderLayerCount]{};
    std::uint16_t* value_cache[kReferenceDecoderLayerCount]{};
    const float* rope_cos = nullptr;
    const float* rope_sin = nullptr;
  };

  [[nodiscard]] static ReferenceRunnerStatus collect_request_views(
      RequestState* state, Views& views) noexcept;
  ReferenceRunner() noexcept = default;
  void release() noexcept;
  [[nodiscard]] ReferenceStepOutcome fail_step(
      ReferenceRunnerStatus status) noexcept;
  [[nodiscard]] ReferencePrefillTileOutcome fail_prefill_tile(
      ReferenceRunnerStatus status) noexcept;
  [[nodiscard]] ReferenceLongPrefillOutcome fail_long_prefill(
      ReferenceRunnerStatus status) noexcept;

  struct LongPrefillLayerTileInvocation {
    LongPrefillLayerMajorWorkItem item;
    const std::uint16_t* input_hidden = nullptr;
    std::uint16_t* output_hidden = nullptr;
  };
  [[nodiscard]] ReferencePrefillTileOutcome prefill_prefix_tile_impl(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      const ReferencePrefillTileOptions& options,
      const LongPrefillLayerTileInvocation* layer_tile) noexcept;
  [[nodiscard]] ReferenceRunnerStatus execute_long_prefill_projection_span(
      const LongPrefillProjectionSpanPlan& plan,
      const LongPrefillProjectionSpanWorkItem& item) noexcept;

  enum class DecodeGraphP1Action : std::uint8_t {
    kDisabled = 0,
    kCaptureOnly,
    kReplay,
  };
  struct DecodeGraphP1Slot;
  [[nodiscard]] ReferenceStepOutcome step_impl(
      std::uint32_t input_token_id, const ReferenceStepOptions& options,
      DecodeGraphP1Action graph_action,
      DecodeGraphP1Slot* capture_destination = nullptr) noexcept;

  struct DecodeGraphP1KernelLaunch {
    void* function = nullptr;
    std::array<unsigned int, 3U> grid{};
    std::array<unsigned int, 3U> block{};
    unsigned int shared_memory_bytes = 0U;
  };

  struct DecodeGraphP1Slot {
    void* graph = nullptr;
    void* exec = nullptr;
    void* embedding_node = nullptr;
    ReferenceDecodeGraphP1Stats stats{};
    DecodeGraphP1KernelLaunch embedding_launch{};
  };

  [[nodiscard]] static int destroy_decode_graph_p1_slot(
      DecodeGraphP1Slot& slot) noexcept;
  void destroy_decode_graph_p1_slot(std::size_t position) noexcept;
  void destroy_decode_graph_p1() noexcept;

  const ModelWeights* weights_ = nullptr;
  RequestState* state_ = nullptr;
  void* stream_ = nullptr;
  void* prefill_auxiliary_stream_ = nullptr;
  void* prefill_branch_ready_event_ = nullptr;
  void* prefill_branch_done_event_ = nullptr;
  void* prefill_gdn_chunk64_reference_context_ = nullptr;
  void* prefill_gdn_chunk64_reference_workspace_ = nullptr;
  std::size_t prefill_gdn_chunk64_reference_workspace_bytes_ = 0U;
  void* prefill_gdn_chunk64_native_workspace_ = nullptr;
  std::size_t prefill_gdn_chunk64_native_workspace_bytes_ = 0U;
  void* pinned_logits_ = nullptr;
  std::uint16_t* pinned_trace_ = nullptr;
  std::array<DecodeGraphP1Slot, kReferenceDecodeGraphP2MaximumSlots>
      decode_graph_p1_slots_{};
  bool decode_graph_capture_active_ = false;
  Views views_{};
  ProjectionBackend projection_backend_ = ProjectionBackend::kReference;
  reference_runner_detail::A4W4PrefillConsumer a4w4_prefill_consumer_ =
      reference_runner_detail::A4W4PrefillConsumer::kUnavailable;
  // Factory-authenticated immutable route decision. Long-Prefill execution
  // must never re-read thread-local environment selectors and silently pick
  // a different publication or kernel on a worker thread.
  reference_runner_detail::A4W4ProjectionMajorGateUpCanonicalDownRoute
      a4w4_projection_major_gateup_canonical_down_route_ =
          reference_runner_detail::
              A4W4ProjectionMajorGateUpCanonicalDownRoute::kDisabled;
  reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute
      a4w4_paired_gateup_canonical_down_route_ =
          reference_runner_detail::
              A4W4PairedGateUpCanonicalDownRoute::kDisabled;
  bool a4w4_full_prefill_admission_enabled_ = false;
  bool trace_enabled_ = false;
  bool trace_valid_ = false;
  bool poisoned_ = false;
  bool retained_prefill_hidden_valid_ = false;
  std::uint32_t retained_prefill_position_ = 0U;
  std::uint32_t retained_prefill_input_token_ = 0U;
  std::size_t retained_prefill_hidden_row_ = 0U;
  std::uint32_t trace_position_ = 0U;
  std::uint32_t trace_input_token_ = 0U;
};

struct ReferenceRunnerFactoryResult {
  std::optional<ReferenceRunner> value;
  ReferenceRunnerStatus diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pointer form deliberately permits deterministic null-dependency factory
// error tests without creating CUDA state. Valid runners retain non-owning
// references exactly as documented on ReferenceRunner.
[[nodiscard]] ReferenceRunnerFactoryResult create_reference_runner(
    const ModelWeights* weights, RequestState* state,
    const ReferenceRunnerOptions& options = {}) noexcept;

[[nodiscard]] inline ReferenceRunnerFactoryResult create_reference_runner(
    const ModelWeights& weights, RequestState& state,
    const ReferenceRunnerOptions& options = {}) noexcept {
  return create_reference_runner(&weights, &state, options);
}

}  // namespace q3x::runtime
