#include "q3x/runtime/reference_engine.h"
#include "reference_runner_nvfp4_prefill_marlin_admission.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

namespace {

namespace runtime = q3x::runtime;
namespace detail = q3x::runtime::reference_runner_detail;

constexpr std::size_t kPromptTokens = 513U;
constexpr std::uint32_t kPrefixTokens = 512U;
constexpr std::uint32_t kExpectedGeneratedId = 9'419U;
constexpr std::string_view kExpectedGeneratedText = "Hello";
constexpr std::size_t kExpectedRouteHits = runtime::kRequestLayerCount;

struct Sample {
  double prefix_milliseconds = 0.0;
  double ttft_milliseconds = 0.0;
  std::size_t route_hits = 0U;
  bool semantic_oracle = false;
};

class ScopedAdmission final {
 public:
  explicit ScopedAdmission(const bool enabled) noexcept
      : previous_(
            detail::exchange_prefill_nvfp4_marlin_admission_test_enabled(
                enabled)) {}

  ~ScopedAdmission() {
    (void)detail::exchange_prefill_nvfp4_marlin_admission_test_enabled(
        previous_);
  }

  ScopedAdmission(const ScopedAdmission&) = delete;
  ScopedAdmission& operator=(const ScopedAdmission&) = delete;

 private:
  bool previous_ = false;
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

[[nodiscard]] std::string repeated_hello_prompt() {
  // The pinned non-thinking chat template maps 501 repetitions to P513.
  constexpr std::size_t kRepetitions = kPromptTokens - 12U;
  std::string prompt;
  prompt.reserve(kRepetitions * 6U);
  for (std::size_t index = 0U; index < kRepetitions; ++index) {
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

[[nodiscard]] bool expected_generation(
    const runtime::ReferenceGeneration& generation) {
  return generation.prompt_token_ids.size() == kPromptTokens &&
         generation.generated_token_ids.size() == 1U &&
         generation.generated_token_ids.front() == kExpectedGeneratedId &&
         generation.generated_text == kExpectedGeneratedText &&
         generation.stop_reason ==
             runtime::ReferenceStopReason::kMaxNewTokens &&
         generation.requested_prefill_chunk_size == kPrefixTokens &&
         generation.effective_prefill_chunk_size == kPrefixTokens &&
         generation.timing.prefix_execution_milliseconds.size() == 1U &&
         generation.steps.size() == kPromptTokens;
}

[[nodiscard]] bool run_sample(runtime::ReferenceEngine& engine,
                              const std::string& prompt,
                              const bool candidate,
                              const std::string_view phase,
                              Sample& sample) {
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = 1U;
  options.prefill_chunk_size = kPrefixTokens;
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;

  (void)detail::exchange_prefill_nvfp4_marlin_admission_test_hits(0U);
  runtime::ReferenceGenerateResult result;
  {
    const ScopedAdmission admission(candidate);
    result = engine.generate(prompt, options);
  }
  sample.route_hits =
      detail::exchange_prefill_nvfp4_marlin_admission_test_hits(0U);
  if (!result) {
    std::cerr << "NVFP4_PREFILL_MARLIN_SAMPLE phase=" << phase
              << " route=" << (candidate ? "candidate" : "baseline")
              << " generation failed: ";
    print_diagnostic(result.diagnostic);
    return false;
  }

  for (const double elapsed :
       result.value->timing.prefix_execution_milliseconds) {
    sample.prefix_milliseconds += elapsed;
  }
  sample.ttft_milliseconds =
      result.value->timing.time_to_first_token_milliseconds;
  sample.semantic_oracle = expected_generation(*result.value);
  const std::size_t expected_hits = candidate ? kExpectedRouteHits : 0U;
  const bool structural_oracle =
      std::isfinite(sample.prefix_milliseconds) &&
      sample.prefix_milliseconds > 0.0 &&
      std::isfinite(sample.ttft_milliseconds) &&
      sample.ttft_milliseconds > 0.0 &&
      sample.route_hits == expected_hits;

  std::cout << "NVFP4_PREFILL_MARLIN_SAMPLE phase=" << phase
            << " route=" << (candidate ? "candidate" : "baseline")
            << " prefix_ms=" << sample.prefix_milliseconds
            << " ttft_ms=" << sample.ttft_milliseconds
            << " generated_token="
            << (result.value->generated_token_ids.empty()
                    ? runtime::kReferenceVocabularySize
                    : result.value->generated_token_ids.front())
            << " generated_text=" << result.value->generated_text
            << " route_hits=" << sample.route_hits
            << " expected_hits=" << expected_hits
            << " structural_oracle="
            << (structural_oracle ? "PASS" : "FAIL")
            << " semantic_oracle="
            << (sample.semantic_oracle ? "PASS" : "FAIL") << '\n';
  return structural_oracle;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: "
                 "q3x_reference_nvfp4_prefill_marlin_admission_e2e_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }
  const char* const enabled =
      std::getenv("Q3X_RUN_NVFP4_PREFILL_MARLIN_ADMISSION");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "SKIP: set Q3X_RUN_NVFP4_PREFILL_MARLIN_ADMISSION=1 "
                 "to run the P513 real-model direction screen\n";
    return 77;
  }
  (void)detail::exchange_prefill_nvfp4_marlin_admission_test_enabled(false);
  (void)detail::exchange_prefill_nvfp4_marlin_admission_test_hits(0U);

  runtime::ReferenceEngineOptions options;
  options.request_options.prefill_chunk_size = kPrefixTokens;
  options.request_options.max_sequence_length = kPromptTokens + 1U;
  options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(std::filesystem::path(model_directory),
                                       options);
  if (!created) {
    std::cerr << "engine creation failed: ";
    print_diagnostic(created.diagnostic);
    return 1;
  }

  std::cout << std::fixed << std::setprecision(9);
  const std::string prompt = repeated_hello_prompt();
  Sample baseline_warmup;
  Sample candidate_warmup;
  Sample baseline;
  Sample candidate;
  const bool valid =
      run_sample(*created.value, prompt, false, "warmup", baseline_warmup) &&
      run_sample(*created.value, prompt, true, "warmup", candidate_warmup) &&
      run_sample(*created.value, prompt, false, "measured", baseline) &&
      run_sample(*created.value, prompt, true, "measured", candidate);
  if (!valid) {
    std::cout << "NVFP4_PREFILL_MARLIN_DIRECTION result=INVALID\n";
    return 1;
  }

  const double prefix_saved =
      baseline.prefix_milliseconds - candidate.prefix_milliseconds;
  const double ttft_saved =
      baseline.ttft_milliseconds - candidate.ttft_milliseconds;
  const bool positive = prefix_saved > 0.0 && ttft_saved > 0.0;
  const bool semantic_oracle =
      baseline.semantic_oracle && candidate.semantic_oracle;
  std::cout << "NVFP4_PREFILL_MARLIN_DIRECTION"
            << " baseline_prefix_ms=" << baseline.prefix_milliseconds
            << " candidate_prefix_ms=" << candidate.prefix_milliseconds
            << " prefix_saved_ms=" << prefix_saved
            << " prefix_speedup="
            << baseline.prefix_milliseconds / candidate.prefix_milliseconds
            << " baseline_ttft_ms=" << baseline.ttft_milliseconds
            << " candidate_ttft_ms=" << candidate.ttft_milliseconds
            << " ttft_saved_ms=" << ttft_saved
            << " ttft_speedup="
            << baseline.ttft_milliseconds / candidate.ttft_milliseconds
            << " direction=" << (positive ? "POSITIVE" : "NEGATIVE")
            << " semantic_oracle="
            << (semantic_oracle ? "PASS" : "FAIL")
            << " authority=DIRECTION_SCREEN_ONLY"
            << " production_unchanged=true\n";
  if (!semantic_oracle) {
    return 4;
  }
  return positive ? 0 : 3;
}
