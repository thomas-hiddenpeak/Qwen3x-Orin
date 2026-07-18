#include "q3x/runtime/reference_engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace runtime = q3x::runtime;

constexpr std::string_view kPrompt =
    u8"用一句话解释 CUDA 是什么。";
constexpr std::string_view kExpectedText =
    u8"CUDA 是 NVIDIA 开发的一种并行计算平台和编程模型，旨在利用 GPU 的强大算力来加速通用计算任务。";

constexpr std::array<std::uint32_t, 19U> kExpectedPromptIds = {
    248045U, 846U,    198U,    95826U,  110827U,
    98682U,  52965U,  220U,    98661U,  1710U,
    248046U, 198U,    248045U, 74455U,  198U,
    248068U, 271U,    248069U, 271U,
};

constexpr std::array<std::uint32_t, 26U> kExpectedGeneratedIds = {
    77517U,  220U,    95761U,  32449U,  220U,    97039U,  101692U,
    119548U, 97792U,  125574U, 102027U, 103725U, 3709U,   109238U,
    97327U,  21966U,  220U,    127361U, 130111U, 95860U,  103806U,
    108751U, 97792U,  97995U,  1710U,   runtime::kQwen36ImEndTokenId,
};

class TestContext {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <std::size_t Size>
bool exact_tokens(const std::vector<std::uint32_t>& actual,
                  const std::array<std::uint32_t, Size>& expected,
                  const std::string_view label) {
  const bool matches =
      actual.size() == expected.size() &&
      std::equal(actual.begin(), actual.end(), expected.begin());
  if (matches) {
    return true;
  }

  std::cerr << "  " << label << " mismatch: expected_count="
            << expected.size() << " actual_count=" << actual.size() << '\n';
  const std::size_t common = std::min(actual.size(), expected.size());
  for (std::size_t index = 0U; index < common; ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << "  first_mismatch index=" << index
                << " expected=" << expected[index]
                << " actual=" << actual[index] << '\n';
      break;
    }
  }
  return false;
}

std::string model_directory_from(const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* const environment = std::getenv("Q3X_E2E_MODEL_DIR");
  if (environment != nullptr && environment[0] != '\0') {
    return environment;
  }
  return {};
}

bool projection_backend_from(
    const int argc, char** const argv,
    runtime::ProjectionBackend& projection_backend) noexcept {
  projection_backend = runtime::ProjectionBackend::kReference;
  if (argc < 3) {
    return true;
  }
  if (argv[2] == nullptr) {
    return false;
  }

  const std::string_view value(argv[2]);
  if (value == "reference") {
    return true;
  }
  if (value == "sm87") {
    projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
    return true;
  }
  return false;
}

void print_diagnostic(const runtime::ReferenceEngineDiagnostic& diagnostic) {
  std::cerr << "generate_reference failed: code="
            << runtime::to_string(diagnostic.code)
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
  if (!diagnostic.operation.empty()) {
    std::cerr << " operation=" << diagnostic.operation;
  }
  std::cerr << '\n';
}

void check_step_sequence(TestContext& test,
                         const runtime::ReferenceGeneration& generation) {
  constexpr std::size_t kExpectedStepCount = 44U;
  test.expect(generation.steps.size() == kExpectedStepCount,
              "generation performs exactly 44 sequential runner steps");
  if (generation.steps.size() != kExpectedStepCount) {
    return;
  }

  bool coherent = true;
  for (std::size_t index = 0U; index < generation.steps.size(); ++index) {
    const std::uint32_t expected_input =
        index < kExpectedPromptIds.size()
            ? kExpectedPromptIds[index]
            : kExpectedGeneratedIds[index - kExpectedPromptIds.size()];
    const auto& step = generation.steps[index];
    if (step.position != static_cast<std::uint32_t>(index) ||
        step.input_token_id != expected_input) {
      std::cerr << "  step mismatch index=" << index
                << " position=" << step.position
                << " input_token_id=" << step.input_token_id
                << " expected_input_token_id=" << expected_input << '\n';
      coherent = false;
      break;
    }
  }
  test.expect(coherent, "all 44 steps have exact positions and input tokens");
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 3) {
    std::cerr << "usage: q3x_reference_engine_e2e_test "
                 "[MODEL_DIR|-] [reference|sm87]\n";
    return 2;
  }

  runtime::ProjectionBackend projection_backend;
  if (!projection_backend_from(argc, argv, projection_backend)) {
    std::cerr << "invalid projection backend: expected reference or sm87\n";
    return 2;
  }

  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set CMake cache Q3X_E2E_MODEL_DIR or environment "
                 "Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }

  runtime::ReferenceOneShotOptions options;
  options.generation.max_new_tokens =
      static_cast<std::uint32_t>(kExpectedGeneratedIds.size());
  options.generation.stop_token_id = runtime::kQwen36ImEndTokenId;
  options.generation.capture_trace = false;
  options.projection_backend = projection_backend;

  const runtime::ReferenceOneShotResult result = runtime::generate_reference(
      model_directory, kPrompt, options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }

  const runtime::ReferenceGeneration& generation = result.value->generation;
  TestContext test;
  test.expect(exact_tokens(generation.prompt_token_ids, kExpectedPromptIds,
                           "prompt_token_ids"),
              "all 19 prompt token ids match the pinned oracle");
  test.expect(exact_tokens(generation.generated_token_ids,
                           kExpectedGeneratedIds, "generated_token_ids"),
              "all 26 generated token ids match the pinned oracle");
  test.expect(generation.generated_text == kExpectedText,
              "generated UTF-8 text matches the pinned oracle exactly");
  test.expect(generation.stop_reason == runtime::ReferenceStopReason::kImEnd,
              "generation stops because im_end was produced");
  test.expect(!generation.generated_token_ids.empty() &&
                  generation.generated_token_ids.back() ==
                      runtime::kQwen36ImEndTokenId,
              "generated ids retain terminal im_end");
  check_step_sequence(test, generation);

  if (test.failures() == 0) {
    std::cout << "PASS: pinned Qwen3.6-27B reference generation matched "
                 "19 prompt ids, 26 generated ids, exact text, im_end, and "
                 "44 steps with projection_backend="
              << runtime::to_string(projection_backend) << '\n';
    return 0;
  }
  return 1;
}
