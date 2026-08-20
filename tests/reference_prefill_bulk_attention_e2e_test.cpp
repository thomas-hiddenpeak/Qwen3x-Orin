#include "q3x/core/sha256.h"
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

namespace {

namespace runtime = q3x::runtime;
namespace runner_detail = q3x::runtime::reference_runner_detail;

struct Profile {
  std::size_t prompt_token_count = 0U;
  std::uint32_t candidate_chunk = 0U;
  std::string_view token_csv_sha256;
};

constexpr std::array<Profile, 2U> kProfiles{{
    {257U, 256U,
     "e24c73f282d6dabf42b48ea8fb9b71a452cdbaadc9fc0d4eebf7672ee584e2ce"},
    {513U, 512U,
     "45ae21468cb8a0b0ac64566985daba4c0b78e429b99cac3453a98664587efd6d"},
}};
constexpr std::uint32_t kBaselinePrefillChunk = 64U;
constexpr std::uint32_t kExpectedGeneratedId = 9'419U;
constexpr std::string_view kExpectedGeneratedText = "Hello";
constexpr std::array<std::uint32_t, 4U> kProfileOpening{
    248'045U, 846U, 198U, 14'556U};
constexpr std::uint32_t kProfileBodyToken = 23'066U;
constexpr std::array<std::uint32_t, 9U> kProfileSuffix{
    248'046U, 198U, 248'045U, 74'455U, 198U,
    248'068U, 271U, 248'069U, 271U};

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
  // This is the matched native/vLLM profile construction: "hello" followed
  // by P-13 copies of " hello". The pinned non-thinking chat template maps
  // it to the fixed opening/body/suffix contract checked below.
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

[[nodiscard]] std::string token_ids_csv_sha256(
    const std::vector<std::uint32_t>& token_ids) {
  std::string csv;
  csv.reserve(7U * token_ids.size());
  for (const std::uint32_t token : token_ids) {
    if (!csv.empty()) {
      csv.push_back(',');
    }
    csv.append(std::to_string(token));
  }
  return q3x::core::sha256(csv).hex();
}

[[nodiscard]] bool valid_profile_tokens(
    const std::vector<std::uint32_t>& token_ids,
    const Profile& profile) {
  if (token_ids.size() != profile.prompt_token_count ||
      !std::equal(kProfileOpening.begin(), kProfileOpening.end(),
                  token_ids.begin())) {
    return false;
  }
  const std::size_t body_begin = kProfileOpening.size();
  const std::size_t body_end =
      body_begin + profile.prompt_token_count - 13U;
  return std::all_of(token_ids.begin() + body_begin,
                     token_ids.begin() + body_end,
                     [](const std::uint32_t token) {
                       return token == kProfileBodyToken;
                     }) &&
         std::equal(kProfileSuffix.begin(), kProfileSuffix.end(),
                    token_ids.begin() + body_end) &&
         token_ids_csv_sha256(token_ids) == profile.token_csv_sha256;
}

void print_diagnostic(
    const runtime::ReferenceEngineDiagnostic& diagnostic) {
  std::cerr << "code=" << runtime::to_string(diagnostic.code)
            << " stage=" << diagnostic.stage
            << " message=" << diagnostic.message;
  if (!diagnostic.context.empty()) {
    std::cerr << " context=" << diagnostic.context;
  }
  if (diagnostic.dependency_error != 0) {
    std::cerr << " dependency_error=" << diagnostic.dependency_error;
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

[[nodiscard]] bool valid_generation(
    const runtime::ReferenceGeneration& value, const Profile& profile,
    const std::uint32_t expected_chunk,
    const std::size_t expected_prefix_executions) {
  bool ordered_steps =
      value.prompt_token_ids.size() == profile.prompt_token_count &&
      value.steps.size() == profile.prompt_token_count;
  for (std::size_t index = 0U; ordered_steps && index < value.steps.size();
       ++index) {
    ordered_steps =
        value.steps[index].position == index &&
        value.steps[index].input_token_id == value.prompt_token_ids[index];
  }
  return valid_profile_tokens(value.prompt_token_ids, profile) &&
         value.generated_token_ids.size() == 1U &&
         value.generated_token_ids.front() == kExpectedGeneratedId &&
         value.generated_text == kExpectedGeneratedText &&
         value.stop_reason == runtime::ReferenceStopReason::kMaxNewTokens &&
         value.requested_prefill_chunk_size == expected_chunk &&
         value.effective_prefill_chunk_size == expected_chunk &&
         value.timing.prefix_execution_milliseconds.size() ==
             expected_prefix_executions &&
         ordered_steps;
}

[[nodiscard]] bool run_case(runtime::ReferenceEngine& engine,
                            const Profile& profile) {
  const std::string prompt =
      repeated_hello_prompt(profile.prompt_token_count);
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = 1U;
  options.prefill_chunk_size = kBaselinePrefillChunk;
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  (void)runner_detail::exchange_prefill_embedding_batch_test_hits(0U);
  const runtime::ReferenceGenerateResult baseline =
      engine.generate(prompt, options);
  const std::size_t baseline_embedding_launches =
      runner_detail::exchange_prefill_embedding_batch_test_hits(0U);
  if (!baseline) {
    std::cerr << "P" << profile.prompt_token_count
              << " C64 baseline generation failed: ";
    print_diagnostic(baseline.diagnostic);
    return false;
  }

  options.prefill_chunk_size = profile.candidate_chunk;
  (void)runner_detail::exchange_prefill_embedding_batch_test_hits(0U);
  const runtime::ReferenceGenerateResult candidate =
      engine.generate(prompt, options);
  const std::size_t candidate_embedding_launches =
      runner_detail::exchange_prefill_embedding_batch_test_hits(0U);
  if (!candidate) {
    std::cerr << "P" << profile.prompt_token_count
              << " bulk candidate generation failed: ";
    print_diagnostic(candidate.diagnostic);
    return false;
  }

  const runtime::ReferenceGeneration& baseline_value = *baseline.value;
  const runtime::ReferenceGeneration& candidate_value = *candidate.value;
  const std::size_t baseline_executions =
      (profile.prompt_token_count - 1U) / kBaselinePrefillChunk;
  const bool exact =
      valid_generation(baseline_value, profile, kBaselinePrefillChunk,
                       baseline_executions) &&
      valid_generation(candidate_value, profile, profile.candidate_chunk,
                       1U) &&
      baseline_embedding_launches == baseline_executions &&
      candidate_embedding_launches == 1U &&
      baseline_value.prompt_token_ids == candidate_value.prompt_token_ids &&
      baseline_value.generated_token_ids ==
          candidate_value.generated_token_ids &&
      baseline_value.generated_text == candidate_value.generated_text;
  std::cout << "P" << profile.prompt_token_count
            << " baseline_chunk=" << kBaselinePrefillChunk
            << " candidate_chunk=" << profile.candidate_chunk
            << " generated_id="
            << (candidate_value.generated_token_ids.empty()
                    ? runtime::kReferenceVocabularySize
                    : candidate_value.generated_token_ids.front())
            << " generated_text=" << candidate_value.generated_text
            << " steps=" << candidate_value.steps.size()
            << " token_csv_sha256="
            << token_ids_csv_sha256(candidate_value.prompt_token_ids)
            << " baseline_embedding_launches="
            << baseline_embedding_launches
            << " candidate_embedding_launches="
            << candidate_embedding_launches
            << " exact=" << (exact ? "true" : "false") << '\n';
  return exact;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_prefill_bulk_attention_e2e_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }

  runtime::ReferenceEngineOptions options;
  options.request_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  options.request_options.max_sequence_length =
      kProfiles.back().prompt_token_count;
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
  for (const Profile& profile : kProfiles) {
    exact = run_case(*created.value, profile) && exact;
  }
  return exact ? 0 : 1;
}
