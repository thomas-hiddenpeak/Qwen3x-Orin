#pragma once

#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/reference_runner.h"
#include "q3x/runtime/request_state.h"
#include "q3x/runtime/resident_weights.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

inline constexpr std::uint32_t kQwen36ImEndTokenId = 248'046U;

enum class ReferenceEngineError : std::uint8_t {
  kNone = 0,
  kInvalidArgument,
  kCapacityExceeded,
  kArithmeticOverflow,
  kTokenizerFailure,
  kResidentLoadFailure,
  kWeightBindFailure,
  kRequestStateFailure,
  kRunnerFactoryFailure,
  kRunnerStepFailure,
  kRunnerResetFailure,
  kMissingLogits,
  kMissingTiming,
  kDecodeFailure,
  kTraceFailure,
  kAllocationFailure,
  kMissingPrediction,
};

struct ReferenceEngineDiagnostic {
  ReferenceEngineError code = ReferenceEngineError::kNone;
  std::string stage;
  std::string message;
  std::string context;
  int dependency_error = 0;
  int cuda_error = 0;
  std::size_t layer = kReferenceNoLayer;
  std::string operation;
};

enum class ReferenceDecodeGraphCachePolicy : std::uint8_t {
  kDisabled = 0,
  kSm87ShortPositions,
};

[[nodiscard]] constexpr bool is_valid_reference_decode_graph_cache_policy(
    const ReferenceDecodeGraphCachePolicy policy) noexcept {
  return policy == ReferenceDecodeGraphCachePolicy::kDisabled ||
         policy == ReferenceDecodeGraphCachePolicy::kSm87ShortPositions;
}

struct ReferenceEngineOptions {
  ResidentLoadOptions resident_options;
  RequestMemoryOptions request_options;
  bool enable_trace = false;
  ProjectionBackend projection_backend = ProjectionBackend::kReference;
  // Engine-lifetime resource policy. The selected short-position cache is
  // prepared during engine creation and never lazily inside generate().
  ReferenceDecodeGraphCachePolicy decode_graph_cache_policy =
      ReferenceDecodeGraphCachePolicy::kDisabled;
};

// Text-only messages accepted by the pinned Qwen 3.6 chat formatter. The
// tokenizer deliberately fails closed on unsupported roles, non-alternating
// histories, tool payloads, and multimodal content.
struct ReferenceChatMessage {
  std::string role;
  std::string content;
};

// A token is reported only after its Prefill/Decode step has completed and
// the id has been committed to the generation transcript. String views are
// valid only for the duration of the callback. text_delta may be empty when a
// token contributes no visible text (including the configured stop token).
struct ReferenceTokenEvent {
  std::size_t index = 0U;
  std::uint32_t token_id = 0U;
  std::string_view text_delta;
  std::string_view generated_text;
  double elapsed_milliseconds = 0.0;
  bool is_stop_token = false;
};

// Returning false requests cancellation at the current committed-token
// boundary. The observer must not throw and is invoked synchronously on the
// engine's serialized generation thread.
using ReferenceTokenObserver = bool (*)(
    void* context, const ReferenceTokenEvent& event) noexcept;

struct ReferenceGenerateOptions {
  std::uint32_t max_new_tokens = 16U;
  bool capture_trace = false;
  std::uint32_t stop_token_id = kQwen36ImEndTokenId;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
  // Full statistics is the compatibility default. Prediction-only compute
  // steps populate ReferenceStepResult::prediction instead of logits.
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
  // Emit static NVTX ranges around the host-side prefill/decode runner
  // boundaries. Disabled by default so ordinary generation pays one
  // predictable dispatch branch.
  bool emit_nvtx_phase_ranges = false;
  // Test-only production-alignment hook. When enabled, predicted-only Decode
  // replays an already-prepared position-specialized graph and falls back to
  // the ordinary serial step on every incompatible option or cache miss.
  // Graph preparation is explicit and never occurs in a timed request.
  bool use_prepared_decode_graph_cache = false;
  // Optional synchronous per-token observer used by the streaming gateway.
  // Keep the callback and context at the end for aggregate-initializer source
  // compatibility with the pre-existing options surface.
  ReferenceTokenObserver token_observer = nullptr;
  void* token_observer_context = nullptr;
};

enum class ReferenceStopReason : std::uint8_t {
  kImEnd,
  kMaxNewTokens,
  kCancelled,
};

struct ReferenceGenerationTiming {
  // One aggregate duration for each prefix_step or prefix_tile execution, in
  // execution order. Under the test-only whole-prompt admission, the final
  // prompt token is included here; its logits-only finalize duration remains
  // in finish_prefill_milliseconds below.
  std::vector<double> prefix_execution_milliseconds;
  double finish_prefill_milliseconds = 0.0;
  double prompt_prefill_milliseconds = 0.0;
  double time_to_first_token_milliseconds = 0.0;
  std::vector<double> subsequent_token_milliseconds;
  double decode_after_first_milliseconds = 0.0;
  double total_generation_milliseconds = 0.0;
};

struct ReferenceTraceDigest {
  std::uint32_t position = 0U;
  std::uint32_t input_token_id = 0U;
  std::size_t element_count = 0U;
  std::string full_sha256;
  std::string embedding_sha256;
  std::array<std::string, kReferenceDecoderLayerCount> layer_hidden_sha256;
  std::array<std::string, kReferenceDecoderLayerCount> layer_residual_sha256;
  std::string final_norm_sha256;
};

struct ReferenceGeneration {
  std::string rendered_prompt;
  std::vector<std::uint32_t> prompt_token_ids;
  std::vector<std::uint32_t> generated_token_ids;
  std::string generated_text;
  ReferenceStopReason stop_reason = ReferenceStopReason::kMaxNewTokens;
  std::uint32_t requested_prefill_chunk_size =
      kDefaultRequestPrefillChunkSize;
  std::uint32_t effective_prefill_chunk_size =
      kDefaultRequestPrefillChunkSize;
  // True only when every prompt token was committed by prefill tiles and the
  // final prediction was produced from the retained last hidden row.
  bool all_prompt_tokens_prefilled_by_tiles = false;
  // True only for the test admission that dispatches each request-sized
  // prefix chunk as one arbitrary 1..512-token layer-major tile instead of
  // decomposing it into canonical C512/C256/C64/C32/tail executions.
  bool single_arbitrary_prefill_tiles = false;
  // True only when one whole prompt used the build- and runtime-gated
  // P<=4096 layer-major runner path.
  bool layer_major_prefill = false;
  ReferenceGenerationTiming timing;
  std::vector<ReferenceStepResult> steps;
  std::vector<ReferenceTraceDigest> traces;
  std::size_t decode_graph_replays = 0U;
  std::size_t decode_graph_serial_fallbacks = 0U;
};

struct ReferenceEngineLoadStats {
  double tokenizer_milliseconds = 0.0;
  double resident_load_milliseconds = 0.0;
  double weight_bind_milliseconds = 0.0;
  double request_state_milliseconds = 0.0;
  double fp8_output_sidecar_milliseconds = 0.0;
  double nvfp4_down_scale6_sidecar_milliseconds = 0.0;
  double nvfp4_down_consumer_order_sidecar_milliseconds = 0.0;
  double nvfp4_gate_up_coupled_feed_milliseconds = 0.0;
  double fp8_prefill_qkv_sidecar_milliseconds = 0.0;
  double fp8_prefill_supermatrix_sidecar_milliseconds = 0.0;
  double fp8_marlin_prefill_sidecar_milliseconds = 0.0;
  double nvfp4_marlin_prefill_sidecar_milliseconds = 0.0;
  double runner_factory_milliseconds = 0.0;
  ReferenceDecodeGraphCachePolicy decode_graph_cache_requested_policy =
      ReferenceDecodeGraphCachePolicy::kDisabled;
  ReferenceDecodeGraphCachePolicy decode_graph_cache_effective_policy =
      ReferenceDecodeGraphCachePolicy::kDisabled;
  std::uint32_t decode_graph_cache_first_position = 0U;
  std::uint32_t decode_graph_cache_last_position = 0U;
  std::size_t decode_graph_cache_slot_count = 0U;
  double decode_graph_cache_capture_enqueue_milliseconds = 0.0;
  double decode_graph_cache_topology_inspection_milliseconds = 0.0;
  double decode_graph_cache_instantiate_milliseconds = 0.0;
  double decode_graph_cache_upload_ready_milliseconds = 0.0;
  double decode_graph_cache_prepare_milliseconds = 0.0;
  std::uint64_t decode_graph_cache_free_bytes_before = 0U;
  std::uint64_t decode_graph_cache_free_bytes_after = 0U;
  std::uint64_t decode_graph_cache_free_drop_bytes = 0U;
  // Empty when the requested cache was completely prepared or when the
  // policy was disabled. Any preparation/admission failure is rolled back
  // before this stable serial-fallback reason is published.
  std::string decode_graph_cache_fallback_reason;
  double total_milliseconds = 0.0;
  ResidentLoadStats resident;
  WeightBindingStats binding;
  std::uint64_t request_arena_bytes = 0U;
  std::uint32_t request_max_sequence_length = 0U;
  std::uint32_t request_prefill_chunk_size =
      kDefaultRequestPrefillChunkSize;
  bool fp8_output_sidecars_enabled = false;
  std::size_t fp8_output_sidecar_layers = 0U;
  std::uint64_t fp8_output_sidecar_bytes = 0U;
  // Empty when the SM87 sidecar path was enabled or was not requested.
  // Optional allocation/memory-gate failures retain the canonical M1 route
  // and record a stable reason here instead of failing engine creation.
  std::string fp8_output_sidecar_fallback_reason;
  // The down-only scale6 inventory is derived from all 64 canonical NVFP4
  // block-scale tensors. Eligible layers receive compact sidecars; fallback
  // layers retain their canonical scales. bytes is nonzero only when the
  // complete eligible inventory was attached successfully.
  bool nvfp4_down_scale6_sidecars_enabled = false;
  std::size_t nvfp4_down_scale6_sidecar_eligible_layers = 0U;
  std::size_t nvfp4_down_scale6_sidecar_fallback_layers = 0U;
  std::uint64_t nvfp4_down_scale6_sidecar_bytes = 0U;
  // Empty when the optional SM87 sidecar inventory was attached or was not
  // requested. Admission/allocation failures preserve canonical execution.
  std::string nvfp4_down_scale6_sidecar_fallback_reason;
  // Explicit same-ELF Decode Down K512 consumer-order admission. The
  // equal-byte weight arena is attached only to scale6-eligible layers and
  // is prepared after all production/Prefill sidecars so it cannot displace
  // them. A requested admission is all-or-nothing and fails engine creation
  // instead of silently benchmarking the production kernel.
  bool nvfp4_down_consumer_order_sidecars_requested = false;
  bool nvfp4_down_consumer_order_sidecars_enabled = false;
  std::size_t nvfp4_down_consumer_order_sidecar_layers = 0U;
  std::uint64_t nvfp4_down_consumer_order_sidecar_bytes = 0U;

  // Explicit Decode admission only.  The arena is an equal-byte Gate+Up
  // weight+scale permutation and therefore coexists with canonical tensors.
  bool nvfp4_gate_up_coupled_feed_requested = false;
  bool nvfp4_gate_up_coupled_feed_enabled = false;
  std::size_t nvfp4_gate_up_coupled_feed_layers = 0U;
  std::uint64_t nvfp4_gate_up_coupled_feed_bytes = 0U;
  // The exact C512 register-feed layout is optional and intentionally
  // co-resident with the canonical 48 linear-attention QKV tensors. A
  // capacity miss preserves the canonical Prefill route; all non-capacity
  // preparation failures remain engine-creation failures.
  bool fp8_prefill_qkv_sidecars_enabled = false;
  std::size_t fp8_prefill_qkv_sidecar_layers = 0U;
  std::uint64_t fp8_prefill_qkv_sidecar_bytes = 0U;
  // Empty when the complete 48-layer inventory was attached or when the
  // exact SM87 C512 sidecar route was not requested.
  std::string fp8_prefill_qkv_sidecar_fallback_reason;
  // The fixed 208-projection exact-C512 inventory replaces the legacy
  // QKV-only sidecars when its complete arena is admitted.
  bool fp8_prefill_supermatrix_sidecars_enabled = false;
  std::size_t fp8_prefill_supermatrix_sidecar_projections = 0U;
  std::uint64_t fp8_prefill_supermatrix_sidecar_bytes = 0U;
  // Populated only by the direct W8A16 test-admission build. The retained
  // compact weight and BF16 channel-scale arenas replace the equal-byte
  // production supermatrix arena; disposable transpose scratch is excluded.
  bool fp8_marlin_prefill_sidecars_enabled = false;
  std::size_t fp8_marlin_prefill_sidecar_projections = 0U;
  std::uint64_t fp8_marlin_prefill_sidecar_bytes = 0U;
  // Populated only by the dedicated test-admission build. This is the sum of
  // all retained Gate+Up and Down Marlin weights, processed scales, and
  // global scales; the disposable load-time transpose scratch is excluded.
  bool nvfp4_marlin_prefill_sidecars_enabled = false;
  std::size_t nvfp4_marlin_prefill_sidecar_layers = 0U;
  std::uint64_t nvfp4_marlin_prefill_sidecar_bytes = 0U;
  // True only when tokenizer parsing and resident loading actually executed
  // concurrently. When true, total_milliseconds is wall time and phase
  // timings intentionally overlap.
  bool tokenizer_resident_overlap = false;
};

struct ReferenceGenerateResult {
  std::optional<ReferenceGeneration> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

inline constexpr std::size_t kReferenceDecodeGraphP1ScreenRounds = 5U;

struct ReferenceDecodeGraphP1ScreenOptions {
  std::uint32_t expected_position = 19U;
  std::uint32_t expected_input_token_id = 77'517U;
  std::uint32_t expected_prediction = 220U;
  std::uint32_t alternate_input_token_id = 220U;
  std::uint32_t expected_alternate_prediction = 52'965U;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
};

struct ReferenceDecodeGraphP1RoundTiming {
  double serial_first_milliseconds = 0.0;
  double graph_first_milliseconds = 0.0;
  double graph_second_milliseconds = 0.0;
  double serial_second_milliseconds = 0.0;
};

struct ReferenceDecodeGraphP1ScreenResult {
  ReferenceDecodeGraphP1Stats graph;
  std::uint32_t serial_prediction = 0U;
  std::uint32_t graph_prediction = 0U;
  std::uint32_t alternate_serial_prediction = 0U;
  std::uint32_t alternate_graph_prediction = 0U;
  bool primary_arena_exact = false;
  bool alternate_arena_exact = false;
  std::uint64_t compared_arena_bytes = 0U;
  std::array<ReferenceDecodeGraphP1RoundTiming,
             kReferenceDecodeGraphP1ScreenRounds>
      rounds{};
};

struct ReferenceDecodeGraphP1ScreenOutcome {
  std::optional<ReferenceDecodeGraphP1ScreenResult> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

inline constexpr std::size_t kReferenceDecodeGraphP2ScreenRounds = 5U;
inline constexpr std::size_t kReferenceDecodeGraphP2ContinuousSteps = 25U;
// Decode Graph admission remains pinned to the frozen P19/C32 baseline even
// when the independent request Prefill capacity grows.
inline constexpr std::uint32_t kReferenceDecodeGraphScreenPrefillChunkSize =
    32U;

struct ReferenceDecodeGraphP2ScreenOptions {
  std::uint32_t first_decode_position = 19U;
  std::uint32_t last_decode_position = 43U;
  std::uint32_t boundary_graph_position = 63U;
  std::uint32_t capture_input_token_id = 0U;
  std::uint32_t boundary_input_token_id = 9'419U;
  std::uint32_t max_new_tokens = 26U;
  std::uint32_t prefill_chunk_size =
      kReferenceDecodeGraphScreenPrefillChunkSize;
};

struct ReferenceDecodeGraphP2RoundTiming {
  double serial_first_milliseconds_per_token = 0.0;
  double graph_first_milliseconds_per_token = 0.0;
  double graph_second_milliseconds_per_token = 0.0;
  double serial_second_milliseconds_per_token = 0.0;
};

struct ReferenceDecodeGraphP2ScreenResult {
  std::array<ReferenceDecodeGraphP1Stats,
             kReferenceDecodeGraphP2ContinuousSteps>
      continuous_graphs{};
  ReferenceDecodeGraphP1Stats boundary_graph;
  double cache_prepare_milliseconds = 0.0;
  std::uint64_t cache_free_bytes_before = 0U;
  std::uint64_t cache_free_bytes_after = 0U;
  std::uint64_t cache_cuda_free_drop_bytes = 0U;
  std::uint64_t boundary_free_bytes_after = 0U;
  std::uint64_t boundary_cuda_free_drop_bytes = 0U;
  std::uint64_t cache_plus_boundary_cuda_free_drop_bytes = 0U;
  std::uint64_t cache_host_private_bytes_before = 0U;
  std::uint64_t cache_host_private_bytes_after = 0U;
  std::uint64_t cache_host_private_increase_bytes = 0U;
  bool cache_host_private_observed = false;
  bool cache_prepare_arena_exact = false;
  bool boundary_prepare_arena_exact = false;
  ReferenceGeneration serial_generation;
  ReferenceGeneration graph_generation;
  bool continuous_generation_exact = false;
  bool continuous_arena_exact = false;
  bool cache_miss_fallback_exact = false;
  bool full_statistics_fallback_exact = false;
  bool trace_fallback_exact = false;
  bool boundary_cache_hit = false;
  bool boundary_cache_miss_fallback = false;
  std::uint32_t boundary_serial_first_prediction = 0U;
  std::uint32_t boundary_serial_second_prediction = 0U;
  std::uint32_t boundary_graph_first_prediction = 0U;
  std::uint32_t boundary_fallback_second_prediction = 0U;
  bool boundary_arena_exact = false;
  std::uint64_t compared_arena_bytes = 0U;
  std::array<ReferenceDecodeGraphP2RoundTiming,
             kReferenceDecodeGraphP2ScreenRounds>
      rounds{};
};

struct ReferenceDecodeGraphP2ScreenOutcome {
  std::optional<ReferenceDecodeGraphP2ScreenResult> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct ReferenceEngineCreateResult;
struct ReferenceOneShotOptions;
struct ReferenceOneShotResult;

// High-level owner for the exact lifetime chain:
// ResidentWeights -> ModelWeights -> RequestState -> ReferenceRunner.
// The implementation is heap-stable so moving ReferenceEngine never changes
// the addresses retained by the non-owning runner.
class ReferenceEngine {
 public:
  ~ReferenceEngine();
  ReferenceEngine(const ReferenceEngine&) = delete;
  ReferenceEngine& operator=(const ReferenceEngine&) = delete;
  ReferenceEngine(ReferenceEngine&&) noexcept;
  ReferenceEngine& operator=(ReferenceEngine&&) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] const ReferenceEngineLoadStats& load_stats() const noexcept;
  [[nodiscard]] std::uint32_t max_sequence_length() const noexcept;

  // Formats exactly one user message with thinking=false, encodes it with the
  // pinned tokenizer, resets request state, then performs sequential batch-one
  // prefill and greedy decode.
  [[nodiscard]] ReferenceGenerateResult generate(
      std::string_view user_prompt,
      const ReferenceGenerateOptions& options = {});

  // Formats a supported text-only chat history with the pinned template,
  // resets request state, then performs the same batch-one greedy path as
  // generate(). The final message must be a user message.
  [[nodiscard]] ReferenceGenerateResult generate_chat(
      const std::vector<ReferenceChatMessage>& messages,
      const ReferenceGenerateOptions& options = {});

  // Raw completion surfaces. generate_prompt() encodes bytes without a chat
  // template; generate_prompt_token_ids() executes caller-supplied pinned
  // vocabulary ids exactly. They are used by /v1/completions and must never
  // be emulated through generate_chat().
  [[nodiscard]] ReferenceGenerateResult generate_prompt(
      std::string_view prompt,
      const ReferenceGenerateOptions& options = {});
  [[nodiscard]] ReferenceGenerateResult generate_prompt_token_ids(
      const std::vector<std::uint32_t>& prompt_token_ids,
      const ReferenceGenerateOptions& options = {});

  // Test-only, fixed-position CUDA Graph P1 screen. It first performs an exact
  // one-token predicted-only generation so the owned runner is left at the
  // prompt boundary, then snapshots the complete request arena. Serial and
  // graph steps are restored from that snapshot before every correctness or
  // timing sample. The graph remains a single-position experiment; no
  // production graph cache or generation dispatch is installed.
  [[nodiscard]] ReferenceDecodeGraphP1ScreenOutcome
  screen_fixed_position_decode_graph_p1(
      std::string_view user_prompt,
      const ReferenceDecodeGraphP1ScreenOptions& options = {});

  // Test-only P2 screen for a pre-uploaded short-position GraphExec cache.
  // It validates continuous generation and serial fallbacks from identical
  // complete arena snapshots before measuring hot subsequent-token latency.
  [[nodiscard]] ReferenceDecodeGraphP2ScreenOutcome
  screen_short_decode_graph_cache_p2(
      std::string_view user_prompt,
      const ReferenceDecodeGraphP2ScreenOptions& options = {});

 private:
  friend struct ReferenceEngineCreateResult;
  friend ReferenceEngineCreateResult create_reference_engine(
      const std::filesystem::path&, const ReferenceEngineOptions&);
  friend ReferenceOneShotResult generate_reference(
      const std::filesystem::path&, std::string_view,
      const ReferenceOneShotOptions&);
  struct Impl;
  explicit ReferenceEngine(std::unique_ptr<Impl> impl) noexcept;
  [[nodiscard]] ReferenceGenerateResult generate_tokenized(
      std::string rendered_prompt,
      std::vector<std::uint32_t> prompt_token_ids,
      const ReferenceGenerateOptions& options);

  std::unique_ptr<Impl> impl_;
};

struct ReferenceEngineCreateResult {
  std::optional<ReferenceEngine> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

[[nodiscard]] ReferenceEngineCreateResult create_reference_engine(
    const std::filesystem::path& model_directory,
    const ReferenceEngineOptions& options = {});

struct ReferenceOneShotOptions {
  ResidentLoadOptions resident_options;
  std::uint64_t request_max_arena_bytes =
      2ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t request_min_free_bytes_after_create =
      8ULL * 1024ULL * 1024ULL * 1024ULL;
  ProjectionBackend projection_backend = ProjectionBackend::kReference;
  ReferenceGenerateOptions generation;
  // Tokenizer parsing is CPU-only and independent of checkpoint
  // authentication/copy, so the production one-shot path overlaps them by
  // default. Disable this only for diagnostics or controlled benchmarks.
  bool overlap_tokenizer_and_resident_load = true;
  // Keep new one-shot policy fields at the end so positional aggregate
  // initialization of the pre-existing surface remains source-compatible.
  ReferenceDecodeGraphCachePolicy decode_graph_cache_policy =
      ReferenceDecodeGraphCachePolicy::kDisabled;
};

struct ReferenceOneShotGeneration {
  ReferenceEngineLoadStats load;
  ReferenceGeneration generation;
};

struct ReferenceOneShotResult {
  std::optional<ReferenceOneShotGeneration> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// One-shot production path used by the CLI. It loads and validates the pinned
// tokenizer while the resident checkpoint loads, derives the smallest request
// capacity that can execute the prompt plus max_new_tokens, then creates the
// four-stage native lifetime chain without loading either asset a second time.
[[nodiscard]] ReferenceOneShotResult generate_reference(
    const std::filesystem::path& model_directory,
    std::string_view user_prompt,
    const ReferenceOneShotOptions& options = {});

[[nodiscard]] std::string_view to_string(ReferenceEngineError error) noexcept;
[[nodiscard]] std::string_view to_string(ReferenceStopReason reason) noexcept;

namespace reference_engine_detail {

inline constexpr std::size_t kOptimizedPrefillSubtileTokens = 32U;
inline constexpr std::size_t kPrefillM64TileTokens = 64U;
inline constexpr std::size_t kPrefillM256TileTokens = 256U;
static_assert(kMaximumRequestPrefillChunkSize == 512U);

// The public C512 request boundary is independent from individual kernel
// limits. Scheduler calls use only exact C512/C256/C64/C32 tiles plus an
// ordered <=31 tail. Non-canonical request caps such as C128, C192, and C320
// therefore decompose into the same explicit production sizes.
[[nodiscard]] constexpr std::size_t next_prefix_tile_token_count(
    const std::size_t remaining_tokens,
    const std::size_t requested_chunk_size) noexcept {
  if (remaining_tokens == 0U || requested_chunk_size == 0U) {
    return 0U;
  }
  const std::size_t candidate = remaining_tokens < requested_chunk_size
                                    ? remaining_tokens
                                    : requested_chunk_size;
  if (candidate >= kMaximumRequestPrefillChunkSize) {
    return kMaximumRequestPrefillChunkSize;
  }
  if (candidate >= kPrefillM256TileTokens) {
    return kPrefillM256TileTokens;
  }
  if (candidate >= kPrefillM64TileTokens) {
    return kPrefillM64TileTokens;
  }
  if (candidate >= kOptimizedPrefillSubtileTokens) {
    return kOptimizedPrefillSubtileTokens;
  }
  return candidate;
}

[[nodiscard]] constexpr std::size_t prefix_execution_count(
    const std::size_t prefix_token_count,
    const std::size_t effective_prefill_chunk_size) noexcept {
  if (effective_prefill_chunk_size == 0U) {
    return 0U;
  }
  if (effective_prefill_chunk_size == 1U) {
    return prefix_token_count;
  }
  std::size_t count = 0U;
  std::size_t remaining = prefix_token_count;
  while (remaining != 0U) {
    remaining -= next_prefix_tile_token_count(
        remaining, effective_prefill_chunk_size);
    ++count;
  }
  return count;
}

// Test-only scheduler used to measure whether eliminating repeated
// layer/weight traversal is the missing system-level prefill architecture.
// Individual subkernels remain free to use their established ordered
// fallbacks inside this one public layer-major tile.
[[nodiscard]] constexpr std::size_t
next_single_arbitrary_prefix_tile_token_count(
    const std::size_t remaining_tokens,
    const std::size_t requested_chunk_size) noexcept {
  if (remaining_tokens == 0U || requested_chunk_size == 0U) {
    return 0U;
  }
  return remaining_tokens < requested_chunk_size ? remaining_tokens
                                                 : requested_chunk_size;
}

[[nodiscard]] constexpr std::size_t
single_arbitrary_prefix_execution_count(
    const std::size_t prefix_token_count,
    const std::size_t effective_prefill_chunk_size) noexcept {
  if (effective_prefill_chunk_size == 0U) {
    return 0U;
  }
  return prefix_token_count / effective_prefill_chunk_size +
         (prefix_token_count % effective_prefill_chunk_size != 0U ? 1U : 0U);
}

enum class GenerationControlError : std::uint8_t {
  kNone = 0,
  kInvalidArgument,
  kCapacityExceeded,
  kArithmeticOverflow,
  kRunnerFailure,
  kUnexpectedStep,
  kMissingLogits,
  kMissingTiming,
  kAllocationFailure,
  kMissingPrediction,
};

using StepFunction = ReferenceStepOutcome (*)(
    void* context, std::uint32_t input_token_id,
    const ReferenceStepOptions& options);
using PrefillTileFunction = ReferencePrefillTileOutcome (*)(
    void* context, const std::uint32_t* input_token_ids,
    std::size_t token_count, const ReferencePrefillTileOptions& options);
using LongPrefillFunction = ReferenceLongPrefillOutcome (*)(
    void* context, const std::uint32_t* input_token_ids,
    std::size_t token_count, bool measure_timing);
using CommittedTokenFunction = bool (*)(
    void* context, std::uint32_t token_id, std::size_t token_index,
    double elapsed_milliseconds) noexcept;

// Host-side execution seams for the two generation phases. The first
// implementation may route every callback to the same runner and stream; the
// separate plans make the final-prompt boundary explicit without changing the
// runner, request-state, workspace, or public engine contracts. Both plans
// must provide every scalar callback even when a particular request would not
// exercise it; prefix_tile is required only for a chunked non-empty prefix.
struct PrefillPlan {
  void* context = nullptr;
  StepFunction prefix_step = nullptr;
  StepFunction finish_prefill = nullptr;
  PrefillTileFunction prefix_tile = nullptr;
  // Required only by the test-only whole-prompt tile admission. This callback
  // finalizes logits from the last prompt step already committed by a marked
  // prefix tile; it must not append or commit another model-state step.
  StepFunction finish_prefill_from_tile = nullptr;
  // Required only for the whole-prompt P<=4096 layer-major admission.
  LongPrefillFunction layer_major_prompt = nullptr;
};

struct DecodePlan {
  void* context = nullptr;
  StepFunction decode_step = nullptr;
};

struct GenerationControlOptions {
  std::uint32_t max_new_tokens = 0U;
  std::uint32_t stop_token_id = kQwen36ImEndTokenId;
  std::uint32_t max_sequence_length = 0U;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
  bool capture_trace = false;
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
  bool emit_nvtx_phase_ranges = false;
  // Test-only admission. When enabled with a chunk size greater than one, all
  // prompt tokens are submitted to prefix tiles and the final tile retains its
  // last normalized hidden row for finish_prefill_from_tile. Default-off keeps
  // the production prompt-(P-1) plus scalar-final-step schedule unchanged.
  bool prefill_all_prompt_tokens = false;
  // Test-only admission layered on top of prefill_all_prompt_tokens. Each
  // request-sized chunk is one arbitrary 1..512-token prefix_tile call. It is
  // invalid unless whole-prompt admission is also enabled.
  bool prefill_single_arbitrary_tile = false;
  // Mutually exclusive with the existing C512 tile schedulers above. The
  // callback commits all prompt tokens and retains the last normalized row.
  bool prefill_layer_major_prompt = false;
  void* committed_token_context = nullptr;
  CommittedTokenFunction committed_token = nullptr;
};

struct GenerationControl {
  std::vector<std::uint32_t> generated_token_ids;
  std::vector<ReferenceStepResult> steps;
  ReferenceStopReason stop_reason = ReferenceStopReason::kMaxNewTokens;
  ReferenceGenerationTiming timing;
};

struct GenerationControlResult {
  std::optional<GenerationControl> value;
  GenerationControlError error = GenerationControlError::kNone;
  ReferenceRunnerStatus runner_status;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && error == GenerationControlError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pure host generation state machine. The callback is the only runner boundary
// and makes token ordering/prefill policy testable without CUDA or model files.
[[nodiscard]] GenerationControlResult run_generation_control(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationControlOptions& options,
    const PrefillPlan& prefill_plan,
    const DecodePlan& decode_plan);

// Compatibility entry point for callers that intentionally use one callback
// context for both phases. New engine code should pass explicit plans above.
[[nodiscard]] GenerationControlResult run_generation_control(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationControlOptions& options,
    void* step_context,
    StepFunction step_function,
    PrefillTileFunction prefill_tile_function = nullptr);

// Returns the prefix that should be decoded for user-visible text. The exact
// generated id sequence retains a terminal stop id for oracle comparisons;
// only a stop that was actually observed is hidden from the text view.
[[nodiscard]] std::size_t generated_text_token_count(
    const std::vector<std::uint32_t>& generated_token_ids,
    ReferenceStopReason stop_reason,
    std::uint32_t stop_token_id) noexcept;

[[nodiscard]] std::string_view to_string(
    GenerationControlError error) noexcept;

}  // namespace reference_engine_detail

}  // namespace q3x::runtime
