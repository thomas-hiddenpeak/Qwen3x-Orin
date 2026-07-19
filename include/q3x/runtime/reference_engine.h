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

struct ReferenceEngineOptions {
  ResidentLoadOptions resident_options;
  RequestMemoryOptions request_options;
  bool enable_trace = false;
  ProjectionBackend projection_backend = ProjectionBackend::kReference;
};

struct ReferenceGenerateOptions {
  std::uint32_t max_new_tokens = 16U;
  bool capture_trace = false;
  std::uint32_t stop_token_id = kQwen36ImEndTokenId;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
  // Full statistics is the compatibility default. Prediction-only compute
  // steps populate ReferenceStepResult::prediction instead of logits.
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
};

enum class ReferenceStopReason : std::uint8_t {
  kImEnd,
  kMaxNewTokens,
};

struct ReferenceGenerationTiming {
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
  ReferenceGenerationTiming timing;
  std::vector<ReferenceStepResult> steps;
  std::vector<ReferenceTraceDigest> traces;
};

struct ReferenceEngineLoadStats {
  double tokenizer_milliseconds = 0.0;
  double resident_load_milliseconds = 0.0;
  double weight_bind_milliseconds = 0.0;
  double request_state_milliseconds = 0.0;
  double runner_factory_milliseconds = 0.0;
  double total_milliseconds = 0.0;
  ResidentLoadStats resident;
  WeightBindingStats binding;
  std::uint64_t request_arena_bytes = 0U;
  std::uint32_t request_max_sequence_length = 0U;
  std::uint32_t request_prefill_chunk_size =
      kDefaultRequestPrefillChunkSize;
};

struct ReferenceGenerateResult {
  std::optional<ReferenceGeneration> value;
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

 private:
  friend struct ReferenceEngineCreateResult;
  friend ReferenceEngineCreateResult create_reference_engine(
      const std::filesystem::path&, const ReferenceEngineOptions&);
  friend ReferenceOneShotResult generate_reference(
      const std::filesystem::path&, std::string_view,
      const ReferenceOneShotOptions&);
  struct Impl;
  explicit ReferenceEngine(std::unique_ptr<Impl> impl) noexcept;

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
// tokenizer first, derives the smallest request capacity that can execute the
// prompt plus max_new_tokens, then creates the four-stage native lifetime chain
// without parsing tokenizer.json a second time.
[[nodiscard]] ReferenceOneShotResult generate_reference(
    const std::filesystem::path& model_directory,
    std::string_view user_prompt,
    const ReferenceOneShotOptions& options = {});

[[nodiscard]] std::string_view to_string(ReferenceEngineError error) noexcept;
[[nodiscard]] std::string_view to_string(ReferenceStopReason reason) noexcept;

namespace reference_engine_detail {

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

struct GenerationControlOptions {
  std::uint32_t max_new_tokens = 0U;
  std::uint32_t stop_token_id = kQwen36ImEndTokenId;
  std::uint32_t max_sequence_length = 0U;
  std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
  bool capture_trace = false;
  ReferenceLogitsMode logits_mode = ReferenceLogitsMode::kFullStatistics;
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
