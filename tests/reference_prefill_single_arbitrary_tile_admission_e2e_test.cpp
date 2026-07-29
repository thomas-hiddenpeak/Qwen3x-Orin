#include "q3x/runtime/reference_engine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

namespace runtime = q3x::runtime;
namespace detail = q3x::runtime::reference_engine_detail;

constexpr char kAllPromptEnvironment[] =
    "Q3X_RUN_PREFILL_ALL_PROMPT_TOKENS_ADMISSION";
constexpr char kSingleTileEnvironment[] =
    "Q3X_RUN_PREFILL_SINGLE_ARBITRARY_TILE_ADMISSION";
constexpr std::uint32_t kPrefillChunkTokens =
    runtime::kMaximumRequestPrefillChunkSize;
constexpr std::array<std::size_t, 2U> kPromptTokenCounts{407U, 481U};

class EnvironmentRestore final {
 public:
  explicit EnvironmentRestore(const char* const name) : name_(name) {
    const char* const value = std::getenv(name_);
    if (value != nullptr) {
      original_.emplace(value);
    }
  }

  ~EnvironmentRestore() {
    if (original_.has_value()) {
      (void)setenv(name_, original_->c_str(), 1);
    } else {
      (void)unsetenv(name_);
    }
  }

  EnvironmentRestore(const EnvironmentRestore&) = delete;
  EnvironmentRestore& operator=(const EnvironmentRestore&) = delete;

 private:
  const char* name_ = nullptr;
  std::optional<std::string> original_;
};

[[nodiscard]] std::string model_directory_from(
    const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* const environment = std::getenv("Q3X_E2E_MODEL_DIR");
  return environment == nullptr ? std::string{} : std::string(environment);
}

[[nodiscard]] std::string repeated_hello_prompt(
    const std::size_t prompt_token_count) {
  // The pinned non-thinking chat template maps "hello" followed by P-13
  // copies of " hello" to exactly P prompt tokens.
  const std::size_t repetitions = prompt_token_count - 12U;
  std::string prompt;
  prompt.reserve(6U * repetitions);
  for (std::size_t index = 0U; index < repetitions; ++index) {
    if (index != 0U) {
      prompt.push_back(' ');
    }
    prompt.append("hello");
  }
  return prompt;
}

void print_diagnostic(
    const runtime::ReferenceEngineDiagnostic& diagnostic) {
  std::cerr << "code=" << runtime::to_string(diagnostic.code)
            << " stage=" << diagnostic.stage
            << " message=" << diagnostic.message;
  if (!diagnostic.context.empty()) {
    std::cerr << " context=" << diagnostic.context;
  }
  if (diagnostic.cuda_error != 0) {
    std::cerr << " cuda_error=" << diagnostic.cuda_error;
  }
  if (diagnostic.layer != runtime::kReferenceNoLayer) {
    std::cerr << " layer=" << diagnostic.layer;
  }
  if (!diagnostic.operation.empty()) {
    std::cerr << " operation=" << diagnostic.operation;
  }
  std::cerr << '\n';
}

[[nodiscard]] bool same_step_transcript(
    const runtime::ReferenceGeneration& baseline,
    const runtime::ReferenceGeneration& candidate) {
  if (baseline.steps.size() != candidate.steps.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < baseline.steps.size(); ++index) {
    const runtime::ReferenceStepResult& left = baseline.steps[index];
    const runtime::ReferenceStepResult& right = candidate.steps[index];
    if (left.position != right.position ||
        left.input_token_id != right.input_token_id ||
        left.logits.has_value() != right.logits.has_value() ||
        left.prediction.has_value() != right.prediction.has_value()) {
      return false;
    }
    if (left.logits.has_value() &&
        left.logits->predicted_token_id !=
            right.logits->predicted_token_id) {
      return false;
    }
    if (left.prediction.has_value() &&
        left.prediction->predicted_token_id !=
            right.prediction->predicted_token_id) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool run_case(runtime::ReferenceEngine& engine,
                            const std::size_t prompt_token_count) {
  const std::string prompt = repeated_hello_prompt(prompt_token_count);
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = 1U;
  options.prefill_chunk_size = kPrefillChunkTokens;
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;

  if (setenv(kAllPromptEnvironment, "1", 1) != 0 ||
      unsetenv(kSingleTileEnvironment) != 0) {
    std::cerr << "failed to select canonical whole-prompt baseline\n";
    return false;
  }
  const runtime::ReferenceGenerateResult baseline =
      engine.generate(prompt, options);
  if (!baseline) {
    std::cerr << "P" << prompt_token_count
              << " baseline generation failed: ";
    print_diagnostic(baseline.diagnostic);
    return false;
  }

  if (setenv(kSingleTileEnvironment, "1", 1) != 0) {
    std::cerr << "failed to select single-arbitrary candidate\n";
    return false;
  }
  const runtime::ReferenceGenerateResult candidate =
      engine.generate(prompt, options);
  if (!candidate) {
    std::cerr << "P" << prompt_token_count
              << " candidate generation failed: ";
    print_diagnostic(candidate.diagnostic);
    return false;
  }

  const runtime::ReferenceGeneration& left = *baseline.value;
  const runtime::ReferenceGeneration& right = *candidate.value;
  const std::size_t expected_baseline_executions =
      detail::prefix_execution_count(prompt_token_count,
                                     kPrefillChunkTokens);
  const std::size_t expected_candidate_executions =
      detail::single_arbitrary_prefix_execution_count(
          prompt_token_count, kPrefillChunkTokens);
  const bool exact =
      left.prompt_token_ids.size() == prompt_token_count &&
      right.prompt_token_ids.size() == prompt_token_count &&
      left.prompt_token_ids == right.prompt_token_ids &&
      left.generated_token_ids == right.generated_token_ids &&
      left.generated_text == right.generated_text &&
      left.stop_reason == right.stop_reason &&
      same_step_transcript(left, right) &&
      left.all_prompt_tokens_prefilled_by_tiles &&
      right.all_prompt_tokens_prefilled_by_tiles &&
      !left.single_arbitrary_prefill_tiles &&
      right.single_arbitrary_prefill_tiles &&
      left.timing.prefix_execution_milliseconds.size() ==
          expected_baseline_executions &&
      right.timing.prefix_execution_milliseconds.size() ==
          expected_candidate_executions &&
      expected_candidate_executions == 1U;

  std::cout << "P" << prompt_token_count
            << " baseline_prefix_ms="
            << left.timing.prompt_prefill_milliseconds -
                   left.timing.finish_prefill_milliseconds
            << " candidate_prefix_ms="
            << right.timing.prompt_prefill_milliseconds -
                   right.timing.finish_prefill_milliseconds
            << " baseline_ttft_ms="
            << left.timing.time_to_first_token_milliseconds
            << " candidate_ttft_ms="
            << right.timing.time_to_first_token_milliseconds
            << " baseline_executions="
            << left.timing.prefix_execution_milliseconds.size()
            << " candidate_executions="
            << right.timing.prefix_execution_milliseconds.size()
            << " generated_id="
            << (right.generated_token_ids.empty()
                    ? runtime::kReferenceVocabularySize
                    : right.generated_token_ids.front())
            << " exact=" << (exact ? "true" : "false") << '\n';
  return exact;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr
        << "usage: q3x_reference_prefill_single_arbitrary_tile_admission_"
           "e2e_test [MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }

  const EnvironmentRestore restore_all_prompt(kAllPromptEnvironment);
  const EnvironmentRestore restore_single_tile(kSingleTileEnvironment);
  runtime::ReferenceEngineOptions options;
  options.request_options.prefill_chunk_size = kPrefillChunkTokens;
  options.request_options.max_sequence_length =
      static_cast<std::uint32_t>(kPromptTokenCounts.back());
  options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(
          std::filesystem::path(model_directory), options);
  if (!created) {
    std::cerr << "engine creation failed: ";
    print_diagnostic(created.diagnostic);
    return 1;
  }

  bool exact = true;
  for (const std::size_t prompt_token_count : kPromptTokenCounts) {
    exact = run_case(*created.value, prompt_token_count) && exact;
  }
  return exact ? 0 : 1;
}
