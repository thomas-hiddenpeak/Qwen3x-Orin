#include "q3x/kernels/sm87_nvfp4_prefill_marlin.h"
#include "q3x/runtime/reference_engine.h"
#include "reference_runner_nvfp4_prefill_marlin_pair_admission.h"

#include <cuda_runtime_api.h>

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

namespace kernels = q3x::kernels;
namespace runtime = q3x::runtime;
namespace detail = q3x::runtime::reference_runner_detail;

constexpr std::size_t kPromptTokens = 513U;
constexpr std::size_t kPrefixTokens = 512U;
constexpr std::size_t kExpectedRouteHits =
    runtime::kQwen36DenseLayerCount;
constexpr std::uint32_t kExpectedGeneratedToken = 9'419U;
constexpr std::string_view kExpectedGeneratedText = "Hello";
constexpr std::string_view kRunEnvironment =
    "Q3X_RUN_NVFP4_PREFILL_MARLIN_PAIR_ADMISSION";
constexpr std::string_view kMarker = "NVFP4_PREFILL_MARLIN_PAIR";

[[nodiscard]] bool exchange_admission(const bool enabled) noexcept {
  return detail::exchange_prefill_nvfp4_marlin_pair_admission_test_enabled(
      enabled);
}

[[nodiscard]] std::size_t exchange_hits(const std::size_t hits) noexcept {
  return detail::exchange_prefill_nvfp4_marlin_pair_admission_test_hits(hits);
}

class ScopedAdmission {
 public:
  explicit ScopedAdmission(const bool enabled) noexcept
      : previous_(exchange_admission(enabled)) {}

  ~ScopedAdmission() { (void)exchange_admission(previous_); }

  ScopedAdmission(const ScopedAdmission&) = delete;
  ScopedAdmission& operator=(const ScopedAdmission&) = delete;

 private:
  bool previous_ = false;
};

struct Sample {
  double prefix_milliseconds = 0.0;
  double ttft_milliseconds = 0.0;
  std::size_t route_hits = 0U;
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
  constexpr std::size_t kWords = kPromptTokens - 12U;
  std::string prompt;
  prompt.reserve(kWords * 6U);
  for (std::size_t index = 0U; index < kWords; ++index) {
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
            << " message=" << diagnostic.message
            << " context=" << diagnostic.context
            << " cuda_error=" << diagnostic.cuda_error
            << " layer=" << diagnostic.layer
            << " operation=" << diagnostic.operation << '\n';
}

[[nodiscard]] bool expected_generation(
    const runtime::ReferenceGeneration& generation) {
  return generation.prompt_token_ids.size() == kPromptTokens &&
         generation.generated_token_ids.size() == 1U &&
         generation.generated_token_ids.front() == kExpectedGeneratedToken &&
         std::string_view(generation.generated_text) == kExpectedGeneratedText &&
         generation.stop_reason ==
             runtime::ReferenceStopReason::kMaxNewTokens &&
         generation.requested_prefill_chunk_size == kPrefixTokens &&
         generation.effective_prefill_chunk_size == kPrefixTokens &&
         generation.steps.size() == kPromptTokens;
}

[[nodiscard]] bool validate_kernel_resources() {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  const int status =
      kernels::query_sm87_nvfp4_prefill_marlin_pair_resources(
          &registers_per_thread, &static_shared_bytes, &dynamic_shared_bytes,
          &local_bytes, &maximum_threads_per_block, &active_blocks_per_sm);
  const bool passed = status == static_cast<int>(cudaSuccess) &&
                      registers_per_thread > 0 &&
                      registers_per_thread <= 255 && local_bytes == 0U &&
                      maximum_threads_per_block >= 256 &&
                      active_blocks_per_sm >= 1;
  std::cout << kMarker << "_RESOURCES"
            << " cuda_status=" << status
            << " registers_per_thread=" << registers_per_thread
            << " static_shared_bytes=" << static_shared_bytes
            << " dynamic_shared_bytes=" << dynamic_shared_bytes
            << " local_bytes=" << local_bytes
            << " maximum_threads_per_block=" << maximum_threads_per_block
            << " active_blocks_per_sm=" << active_blocks_per_sm
            << " gate=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}

[[nodiscard]] bool validate_sidecar_load(
    const runtime::ReferenceEngine& engine) {
  const runtime::ReferenceEngineLoadStats& load = engine.load_stats();
  const std::size_t bytes_per_projection =
      kernels::sm87_nvfp4_prefill_marlin_sidecar_bytes_per_projection();
  const std::uint64_t expected_bytes =
      static_cast<std::uint64_t>(bytes_per_projection) * 2U *
      runtime::kQwen36DenseLayerCount;
  const bool passed = bytes_per_projection != 0U &&
                      load.nvfp4_prefill_marlin_pair_sidecars_enabled &&
                      load.nvfp4_prefill_marlin_pair_sidecar_layers ==
                          runtime::kQwen36DenseLayerCount &&
                      load.nvfp4_prefill_marlin_pair_sidecar_bytes ==
                          expected_bytes &&
                      std::isfinite(
                          load.nvfp4_prefill_marlin_pair_sidecar_milliseconds) &&
                      load.nvfp4_prefill_marlin_pair_sidecar_milliseconds >
                          0.0;
  std::cout << kMarker << "_SIDECAR"
            << " enabled="
            << (load.nvfp4_prefill_marlin_pair_sidecars_enabled ? 1 : 0)
            << " layers="
            << load.nvfp4_prefill_marlin_pair_sidecar_layers
            << " bytes=" << load.nvfp4_prefill_marlin_pair_sidecar_bytes
            << " expected_bytes=" << expected_bytes
            << " prepare_ms="
            << load.nvfp4_prefill_marlin_pair_sidecar_milliseconds
            << " gate=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
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

  (void)exchange_hits(0U);
  runtime::ReferenceGenerateResult result;
  {
    const ScopedAdmission admission(candidate);
    result = engine.generate(prompt, options);
  }
  sample.route_hits = exchange_hits(0U);
  if (!result) {
    std::cerr << kMarker << "_SAMPLE phase=" << phase
              << " route=" << (candidate ? "candidate" : "baseline")
              << " generation_failed ";
    print_diagnostic(result.diagnostic);
    return false;
  }

  for (const double elapsed :
       result.value->timing.prefix_execution_milliseconds) {
    sample.prefix_milliseconds += elapsed;
  }
  sample.ttft_milliseconds =
      result.value->timing.time_to_first_token_milliseconds;
  const std::size_t expected_hits = candidate ? kExpectedRouteHits : 0U;
  const bool structural_oracle =
      result.value->timing.prefix_execution_milliseconds.size() == 1U &&
      std::isfinite(sample.prefix_milliseconds) &&
      sample.prefix_milliseconds > 0.0 &&
      std::isfinite(sample.ttft_milliseconds) &&
      sample.ttft_milliseconds > 0.0 && sample.route_hits == expected_hits;
  const bool semantic_oracle = expected_generation(*result.value);

  std::cout << kMarker << "_SAMPLE"
            << " phase=" << phase
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
            << (semantic_oracle ? "PASS" : "FAIL") << '\n';
  return structural_oracle && semantic_oracle;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: "
                 "q3x_reference_nvfp4_prefill_marlin_pair_engine_e2e_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }
  const char* const enabled = std::getenv(kRunEnvironment.data());
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "SKIP: set " << kRunEnvironment
              << "=1 to prepare and run the isolated architecture screen\n";
    return 77;
  }

  (void)exchange_admission(false);
  if (!validate_kernel_resources()) {
    return 1;
  }

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
  if (!validate_sidecar_load(*created.value)) {
    return 1;
  }

  std::cout << std::fixed << std::setprecision(9);
  const std::string prompt = repeated_hello_prompt();
  Sample baseline_warmup;
  Sample candidate_warmup;
  Sample baseline;
  Sample candidate;
  const bool passed =
      run_sample(*created.value, prompt, false, "warmup", baseline_warmup) &&
      run_sample(*created.value, prompt, true, "warmup", candidate_warmup) &&
      run_sample(*created.value, prompt, false, "measured", baseline) &&
      run_sample(*created.value, prompt, true, "measured", candidate);
  if (!passed) {
    std::cout << kMarker << "_DIRECTION result=INVALID\n";
    return 1;
  }

  const double prefix_saved =
      baseline.prefix_milliseconds - candidate.prefix_milliseconds;
  const double ttft_saved =
      baseline.ttft_milliseconds - candidate.ttft_milliseconds;
  const bool positive = prefix_saved > 0.0 && ttft_saved > 0.0;
  std::cout << kMarker << "_DIRECTION"
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
            << " authority=REAL_P513_DIRECTION_ONLY"
            << " same_elf=true production_unchanged=true\n";
  return positive ? 0 : 3;
}
