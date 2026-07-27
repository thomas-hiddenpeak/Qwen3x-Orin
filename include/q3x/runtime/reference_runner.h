#pragma once

#include "q3x/model/model_config.h"
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
};

// A prefix tile never produces logits or trace data. The 32-entry fixed-
// capacity result keeps the runner boundary allocation-free while retaining
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

// Pure-host selector for the fixed Q=24, KV=4, D=256, rotary=64 fused
// full-attention preprocessing tile. It also rejects position-table
// arithmetic overflow; callers retain the split/norm/RoPE fallback.
[[nodiscard]] bool use_qk_rope_tile(
    std::size_t first_position, std::size_t token_count) noexcept;

// Selects the exact-M32 residual-add plus centered-RMSNorm schedule. When
// selected, layer 0 retains its standalone input norm; each MLP residual
// produces the normalized input for the next layer (or the final norm after
// the final layer), so no subsequent standalone input/final norm is scheduled.
[[nodiscard]] bool use_m32_prefill_residual_rms_fusion(
    std::size_t token_count, std::size_t hidden_size) noexcept;

// Pure-host selector for the narrow C32 NVFP4 MLP scheduling optimization.
// It accepts only the two exact aligned direct-output projections, so every
// route that could touch the shared FP32 fallback scratch remains serial.
[[nodiscard]] bool use_nvfp4_m32_prefill_gate_up_dual_stream(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* input,
    std::uint16_t* gate_output, std::uint16_t* up_output,
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

  // Executes 1..32 non-logit prompt-prefix tokens in layer-major order. The
  // request plan must reserve at least token_count workspace rows. Operations
  // with a 16-token kernel contract are enqueued as ordered subtiles.
  // Persistent conv/GDN/KV state is updated in token order. The exact aligned
  // SM87 NVFP4 C32 MLP gate/up pair may use one owned auxiliary stream and an
  // event join; every fallback remains on the main stream, and the logical
  // request length is committed only after the complete tile synchronizes.
  [[nodiscard]] ReferencePrefillTileOutcome prefill_prefix_tile(
      const std::uint32_t* input_token_ids, std::size_t token_count,
      const ReferencePrefillTileOptions& options = {}) noexcept;

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
    std::uint16_t* projection[4]{};
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
