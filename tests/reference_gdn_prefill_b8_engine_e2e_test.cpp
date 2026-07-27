#include "q3x/runtime/reference_engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime::reference_runner_detail {

[[nodiscard]] bool exchange_prefill_gdn_b8_admission_test_enabled(
    bool enabled) noexcept;
[[nodiscard]] std::size_t exchange_prefill_gdn_b8_admission_test_hits(
    std::size_t hits) noexcept;

}  // namespace q3x::runtime::reference_runner_detail

namespace {

namespace runtime = q3x::runtime;

constexpr std::array<std::uint32_t, 1U> kP257Expected{9'419U};
constexpr std::array<std::uint32_t, 8U> kP513Expected{
    9'419U, 0U, 1'049U, 5'686U, 1'040U, 488U, 599U, 264U};

class ScopedB8Admission {
 public:
  explicit ScopedB8Admission(const bool enabled) noexcept
      : previous_(runtime::reference_runner_detail::
                      exchange_prefill_gdn_b8_admission_test_enabled(
                          enabled)) {}
  ~ScopedB8Admission() {
    (void)runtime::reference_runner_detail::
        exchange_prefill_gdn_b8_admission_test_enabled(previous_);
  }

  ScopedB8Admission(const ScopedB8Admission&) = delete;
  ScopedB8Admission& operator=(const ScopedB8Admission&) = delete;

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

[[nodiscard]] std::string repeated_hello_prompt(
    const std::size_t prompt_token_count) {
  const std::size_t words = prompt_token_count - 12U;
  std::string prompt;
  prompt.reserve(words * 6U);
  for (std::size_t index = 0U; index < words; ++index) {
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

template <std::size_t GeneratedCount>
[[nodiscard]] bool run_profile(
    runtime::ReferenceEngine& engine,
    const std::size_t prompt_token_count,
    const std::array<std::uint32_t, GeneratedCount>& expected) {
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = static_cast<std::uint32_t>(expected.size());
  options.prefill_chunk_size =
      prompt_token_count == 257U ? 256U : 512U;
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  const std::string prompt = repeated_hello_prompt(prompt_token_count);

  runtime::ReferenceGenerateResult baseline;
  (void)runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_hits(0U);
  {
    const ScopedB8Admission admission(false);
    baseline = engine.generate(prompt, options);
  }
  if (!baseline) {
    std::cerr << "baseline generation failed P" << prompt_token_count
              << ": ";
    print_diagnostic(baseline.diagnostic);
    return false;
  }
  const std::size_t baseline_hits = runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_hits(0U);
  if (baseline_hits != 0U) {
    std::cerr << "baseline unexpectedly hit B8 route P"
              << prompt_token_count << '\n';
    return false;
  }

  runtime::ReferenceGenerateResult candidate;
  (void)runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_hits(0U);
  {
    const ScopedB8Admission admission(true);
    candidate = engine.generate(prompt, options);
  }
  if (!candidate) {
    std::cerr << "candidate generation failed P" << prompt_token_count
              << ": ";
    print_diagnostic(candidate.diagnostic);
    return false;
  }
  const std::size_t candidate_hits = runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_hits(0U);
  if (candidate_hits != runtime::kRequestLinearLayerCount) {
    std::cerr << "candidate B8 route-hit mismatch P" << prompt_token_count
              << " expected=" << runtime::kRequestLinearLayerCount
              << " actual=" << candidate_hits << '\n';
    return false;
  }

  const runtime::ReferenceGeneration& baseline_value = *baseline.value;
  const runtime::ReferenceGeneration& candidate_value = *candidate.value;
  const bool expected_baseline =
      baseline_value.prompt_token_ids.size() == prompt_token_count &&
      baseline_value.generated_token_ids.size() == expected.size() &&
      std::equal(expected.begin(), expected.end(),
                 baseline_value.generated_token_ids.begin()) &&
      baseline_value.stop_reason == runtime::ReferenceStopReason::kMaxNewTokens;
  const bool exact_candidate =
      candidate_value.prompt_token_ids == baseline_value.prompt_token_ids &&
      candidate_value.generated_token_ids ==
          baseline_value.generated_token_ids &&
      candidate_value.generated_text == baseline_value.generated_text &&
      candidate_value.stop_reason == baseline_value.stop_reason &&
      candidate_value.steps.size() == baseline_value.steps.size() &&
      candidate_value.steps.size() ==
          prompt_token_count + expected.size() - 1U;
  std::cout << "GDN_B8_ENGINE_E2E prompt_tokens=" << prompt_token_count
            << " chunk=" << options.prefill_chunk_size
            << " generated_ids=";
  for (std::size_t index = 0U;
       index < candidate_value.generated_token_ids.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << candidate_value.generated_token_ids[index];
  }
  std::cout << " generated_text=" << candidate_value.generated_text
            << " steps=" << candidate_value.steps.size()
            << " expected_baseline="
            << (expected_baseline ? "true" : "false")
            << " candidate_exact="
            << (exact_candidate ? "true" : "false")
            << " b8_route_hits=" << candidate_hits << '\n';
  return expected_baseline && exact_candidate;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_gdn_prefill_b8_engine_e2e_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }
  const char* const enabled = std::getenv("Q3X_RUN_GDN_B8_ADMISSION");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "SKIP: set Q3X_RUN_GDN_B8_ADMISSION=1 to run the B8 "
                 "production-engine ability check\n";
    return 77;
  }
  (void)runtime::reference_runner_detail::
      exchange_prefill_gdn_b8_admission_test_enabled(false);

  runtime::ReferenceEngineOptions options;
  options.request_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  options.request_options.max_sequence_length = 1'040U;
  options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(std::filesystem::path(model_directory),
                                       options);
  if (!created) {
    std::cerr << "engine creation failed: ";
    print_diagnostic(created.diagnostic);
    return 1;
  }
  const runtime::ReferenceEngineLoadStats& load = created.value->load_stats();
  std::cout << "GDN_B8_ENGINE_LOAD fp8_output_sidecars="
            << (load.fp8_output_sidecars_enabled ? "true" : "false")
            << " fp8_output_sidecar_layers="
            << load.fp8_output_sidecar_layers
            << " nvfp4_down_scale6_sidecars="
            << (load.nvfp4_down_scale6_sidecars_enabled ? "true" : "false")
            << " nvfp4_down_scale6_eligible_layers="
            << load.nvfp4_down_scale6_sidecar_eligible_layers
            << " request_arena_bytes=" << load.request_arena_bytes << '\n';

  const bool p257 =
      run_profile(*created.value, 257U, kP257Expected);
  const bool p513 =
      run_profile(*created.value, 513U, kP513Expected);
  const bool passed = p257 && p513;
  std::cout << "GDN_B8_ENGINE_ABILITY "
            << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
