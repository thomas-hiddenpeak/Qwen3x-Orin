#pragma once

#include "q3x/model/model_config.h"
#include "q3x/runtime/long_prefill_layer_major.h"
#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/request_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
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
    std::uint16_t* projection[4]{};
    std::uint8_t* prefill_a4_hidden_packed = nullptr;
    std::uint16_t* prefill_a4_hidden_scales = nullptr;
    std::uint8_t* prefill_a4_intermediate_packed = nullptr;
    std::uint16_t* prefill_a4_intermediate_scales = nullptr;
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
