#include "q3x/runtime/reference_engine.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
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

constexpr std::uint32_t kExpectedRequestSequenceLength =
    static_cast<std::uint32_t>(kExpectedPromptIds.size() +
                               kExpectedGeneratedIds.size() - 1U);

std::uint64_t expected_request_arena_bytes(
    const std::uint32_t prefill_chunk_size) noexcept {
  switch (prefill_chunk_size) {
    case 1U:
      return 82'505'216U;
    case 8U:
      return 83'696'128U;
    case 16U:
      return 85'057'536U;
    case 32U:
      return 87'780'352U;
    case 64U:
      return 93'225'984U;
    case 256U:
      return 125'899'776U;
    case 512U:
      return 169'464'832U;
    default:
      return 0U;
  }
}

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

bool prefill_chunk_from(const int argc, char** const argv,
                        std::uint32_t& prefill_chunk_size) noexcept {
  prefill_chunk_size = 1U;
  if (argc < 4) {
    return true;
  }
  if (argv[3] == nullptr) {
    return false;
  }
  const std::string_view value(argv[3]);
  std::uint32_t parsed = 0U;
  const auto conversion =
      std::from_chars(value.data(), value.data() + value.size(), parsed, 10);
  if (conversion.ec != std::errc{} ||
      conversion.ptr != value.data() + value.size() || parsed == 0U ||
      parsed > runtime::kMaximumRequestPrefillChunkSize) {
    return false;
  }
  prefill_chunk_size = parsed;
  return true;
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
  if (diagnostic.layer != runtime::kReferenceNoLayer) {
    std::cerr << " layer=" << diagnostic.layer;
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

void check_default_full_statistics_arms(
    TestContext& test, const runtime::ReferenceGeneration& generation) {
  constexpr std::size_t kPrefixStepCount = kExpectedPromptIds.size() - 1U;
  bool coherent = true;
  for (std::size_t index = 0U; index < generation.steps.size(); ++index) {
    const bool expects_logits = index >= kPrefixStepCount;
    const auto& step = generation.steps[index];
    if (step.logits.has_value() != expects_logits ||
        step.prediction.has_value()) {
      std::cerr << "  result-arm mismatch index=" << index
                << " has_logits=" << step.logits.has_value()
                << " has_prediction=" << step.prediction.has_value()
                << " expected_logits=" << expects_logits << '\n';
      coherent = false;
      break;
    }
  }
  test.expect(coherent,
              "default full-statistics mode returns only the logits arm on "
              "compute steps");
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 4) {
    std::cerr << "usage: q3x_reference_engine_e2e_test "
                 "[MODEL_DIR|-] [reference|sm87] [PREFILL_CHUNK]\n";
    return 2;
  }

  runtime::ProjectionBackend projection_backend;
  if (!projection_backend_from(argc, argv, projection_backend)) {
    std::cerr << "invalid projection backend: expected reference or sm87\n";
    return 2;
  }
  std::uint32_t prefill_chunk_size = 1U;
  if (!prefill_chunk_from(argc, argv, prefill_chunk_size)) {
    std::cerr << "invalid prefill chunk: expected an integer in [1,512]\n";
    return 2;
  }

  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set CMake cache Q3X_E2E_MODEL_DIR or environment "
                 "Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }

  runtime::ReferenceOneShotOptions options;
  const char* const serial_startup =
      std::getenv("Q3X_E2E_SERIAL_STARTUP");
  options.overlap_tokenizer_and_resident_load =
      serial_startup == nullptr || std::string_view(serial_startup) != "1";
  options.generation.max_new_tokens =
      static_cast<std::uint32_t>(kExpectedGeneratedIds.size());
  options.generation.stop_token_id = runtime::kQwen36ImEndTokenId;
  options.generation.capture_trace = false;
  options.generation.prefill_chunk_size = prefill_chunk_size;
  options.projection_backend = projection_backend;

  const runtime::ReferenceOneShotResult result = runtime::generate_reference(
      model_directory, kPrompt, options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }

  const runtime::ReferenceGeneration& generation = result.value->generation;
  const runtime::ReferenceEngineLoadStats& load = result.value->load;
  TestContext test;
  // This executable is also the controlled startup-performance gate: a
  // requested overlap that fell back to serial must remain visible as a test
  // failure instead of silently weakening the benchmark.
  test.expect(load.tokenizer_resident_overlap ==
                  options.overlap_tokenizer_and_resident_load,
              "load statistics report the selected startup overlap mode");
  const bool finite_nonnegative_timings =
      std::isfinite(load.tokenizer_milliseconds) &&
      load.tokenizer_milliseconds >= 0.0 &&
      std::isfinite(load.resident_load_milliseconds) &&
      load.resident_load_milliseconds >= 0.0 &&
      std::isfinite(load.weight_bind_milliseconds) &&
      load.weight_bind_milliseconds >= 0.0 &&
      std::isfinite(load.request_state_milliseconds) &&
      load.request_state_milliseconds >= 0.0 &&
      std::isfinite(load.fp8_output_sidecar_milliseconds) &&
      load.fp8_output_sidecar_milliseconds >= 0.0 &&
      std::isfinite(load.nvfp4_down_scale6_sidecar_milliseconds) &&
      load.nvfp4_down_scale6_sidecar_milliseconds >= 0.0 &&
      std::isfinite(load.fp8_prefill_qkv_sidecar_milliseconds) &&
      load.fp8_prefill_qkv_sidecar_milliseconds >= 0.0 &&
      std::isfinite(load.runner_factory_milliseconds) &&
      load.runner_factory_milliseconds >= 0.0 &&
      std::isfinite(load.total_milliseconds) &&
      load.total_milliseconds >= 0.0;
  test.expect(finite_nonnegative_timings,
              "all engine load timings are finite and nonnegative");
  test.expect(load.total_milliseconds >=
                  std::max(load.tokenizer_milliseconds,
                           load.resident_load_milliseconds),
              "load wall time covers every individual startup phase");
  constexpr std::uint64_t kExpectedFp8OutputSidecarBytes =
      2'013'265'920ULL;
  if (load.fp8_output_sidecars_enabled) {
    test.expect(projection_backend ==
                        runtime::ProjectionBackend::kSm87WeightOnly &&
                    load.fp8_output_sidecar_layers == 64U &&
                    load.fp8_output_sidecar_bytes ==
                        kExpectedFp8OutputSidecarBytes &&
                    load.fp8_output_sidecar_fallback_reason.empty(),
                "enabled FP8 output sidecars report the exact 64-layer "
                "allocation");
  } else {
    test.expect(load.fp8_output_sidecar_layers == 0U &&
                    load.fp8_output_sidecar_bytes == 0U &&
                    (projection_backend ==
                             runtime::ProjectionBackend::kSm87WeightOnly ||
                         load.fp8_output_sidecar_fallback_reason.empty()),
                "disabled FP8 output sidecars retain an internally "
                "consistent fallback report");
  }
  constexpr std::uint64_t kExpectedNvFp4DownScale6SidecarBytes =
      221'429'760ULL;
  if (load.nvfp4_down_scale6_sidecars_enabled) {
    test.expect(projection_backend ==
                        runtime::ProjectionBackend::kSm87WeightOnly &&
                    load.nvfp4_down_scale6_sidecar_eligible_layers == 53U &&
                    load.nvfp4_down_scale6_sidecar_fallback_layers == 11U &&
                    load.nvfp4_down_scale6_sidecar_bytes ==
                        kExpectedNvFp4DownScale6SidecarBytes &&
                    load.nvfp4_down_scale6_sidecar_fallback_reason.empty(),
                "enabled NVFP4 down scale6 sidecars report the exact "
                "53-layer compact allocation and 11 canonical fallbacks");
  } else if (projection_backend ==
             runtime::ProjectionBackend::kSm87WeightOnly) {
    test.expect(load.nvfp4_down_scale6_sidecar_bytes == 0U &&
                    load.nvfp4_down_scale6_sidecar_eligible_layers +
                            load.nvfp4_down_scale6_sidecar_fallback_layers ==
                        64U &&
                    !load.nvfp4_down_scale6_sidecar_fallback_reason.empty(),
                "disabled SM87 NVFP4 down scale6 sidecars retain the "
                "complete inventory and an explicit fallback reason");
  } else {
    test.expect(load.nvfp4_down_scale6_sidecar_eligible_layers == 0U &&
                    load.nvfp4_down_scale6_sidecar_fallback_layers == 0U &&
                    load.nvfp4_down_scale6_sidecar_bytes == 0U &&
                    load.nvfp4_down_scale6_sidecar_fallback_reason.empty(),
                "unrequested NVFP4 down scale6 sidecars retain an empty "
                "fallback report");
  }
  constexpr std::uint64_t kExpectedFp8PrefillQkvSidecarBytes =
      2'516'582'400ULL;
  const bool fp8_prefill_qkv_sidecars_requested =
      projection_backend == runtime::ProjectionBackend::kSm87WeightOnly &&
      prefill_chunk_size == runtime::kMaximumRequestPrefillChunkSize;
  if (load.fp8_prefill_qkv_sidecars_enabled) {
    test.expect(fp8_prefill_qkv_sidecars_requested &&
                    load.fp8_prefill_qkv_sidecar_layers == 48U &&
                    load.fp8_prefill_qkv_sidecar_bytes ==
                        kExpectedFp8PrefillQkvSidecarBytes &&
                    load.fp8_prefill_qkv_sidecar_fallback_reason.empty(),
                "enabled FP8 Prefill QKV sidecars report the exact "
                "48-layer allocation");
  } else if (fp8_prefill_qkv_sidecars_requested) {
    test.expect(load.fp8_prefill_qkv_sidecar_layers == 0U &&
                    load.fp8_prefill_qkv_sidecar_bytes == 0U &&
                    !load.fp8_prefill_qkv_sidecar_fallback_reason.empty(),
                "disabled SM87 FP8 Prefill QKV sidecars retain an explicit "
                "and internally consistent fallback report");
  } else {
    test.expect(load.fp8_prefill_qkv_sidecar_layers == 0U &&
                    load.fp8_prefill_qkv_sidecar_bytes == 0U &&
                    load.fp8_prefill_qkv_sidecar_fallback_reason.empty(),
                "unrequested FP8 Prefill QKV sidecars retain an empty "
                "fallback report");
  }
  if (!load.tokenizer_resident_overlap) {
    test.expect(load.total_milliseconds >=
                    load.tokenizer_milliseconds +
                        load.resident_load_milliseconds,
                "serial load wall time covers tokenizer plus resident phases");
  }
  const std::uint64_t expected_arena =
      expected_request_arena_bytes(prefill_chunk_size);
  test.expect(load.request_max_sequence_length ==
                      kExpectedRequestSequenceLength &&
                  load.request_prefill_chunk_size == prefill_chunk_size &&
                  (expected_arena == 0U ||
                   load.request_arena_bytes == expected_arena),
              "one-shot request capacity, chunk policy, and known C1/C8/C16/C32/C64 "
              "arena sizes remain exact");
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
  test.expect(generation.requested_prefill_chunk_size == prefill_chunk_size &&
                  generation.effective_prefill_chunk_size ==
                      prefill_chunk_size,
              "generation reports the requested and effective prefill chunk");
  check_step_sequence(test, generation);
  check_default_full_statistics_arms(test, generation);

  if (test.failures() == 0) {
    std::cout << "PASS: pinned Qwen3.6-27B reference generation matched "
                 "19 prompt ids, 26 generated ids, exact text, im_end, and "
                 "44 steps with projection_backend="
              << runtime::to_string(projection_backend)
              << " startup_overlap="
              << (result.value->load.tokenizer_resident_overlap ? 1 : 0)
              << " load_total_ms=" << load.total_milliseconds
              << " tokenizer_ms=" << load.tokenizer_milliseconds
              << " resident_ms=" << load.resident_load_milliseconds
              << " fp8_output_sidecars_enabled="
              << (load.fp8_output_sidecars_enabled ? 1 : 0)
              << " fp8_output_sidecar_layers="
              << load.fp8_output_sidecar_layers
              << " fp8_output_sidecar_bytes="
              << load.fp8_output_sidecar_bytes
              << " fp8_output_sidecar_ms="
              << load.fp8_output_sidecar_milliseconds
              << " fp8_output_sidecar_fallback_reason="
              << load.fp8_output_sidecar_fallback_reason
              << " nvfp4_down_scale6_sidecars_enabled="
              << (load.nvfp4_down_scale6_sidecars_enabled ? 1 : 0)
              << " nvfp4_down_scale6_sidecar_eligible_layers="
              << load.nvfp4_down_scale6_sidecar_eligible_layers
              << " nvfp4_down_scale6_sidecar_fallback_layers="
              << load.nvfp4_down_scale6_sidecar_fallback_layers
              << " nvfp4_down_scale6_sidecar_bytes="
              << load.nvfp4_down_scale6_sidecar_bytes
              << " nvfp4_down_scale6_sidecar_ms="
              << load.nvfp4_down_scale6_sidecar_milliseconds
              << " nvfp4_down_scale6_sidecar_fallback_reason="
              << load.nvfp4_down_scale6_sidecar_fallback_reason
              << " fp8_prefill_qkv_sidecars_enabled="
              << (load.fp8_prefill_qkv_sidecars_enabled ? 1 : 0)
              << " fp8_prefill_qkv_sidecar_layers="
              << load.fp8_prefill_qkv_sidecar_layers
              << " fp8_prefill_qkv_sidecar_bytes="
              << load.fp8_prefill_qkv_sidecar_bytes
              << " fp8_prefill_qkv_sidecar_ms="
              << load.fp8_prefill_qkv_sidecar_milliseconds
              << " fp8_prefill_qkv_sidecar_fallback_reason="
              << load.fp8_prefill_qkv_sidecar_fallback_reason << '\n';
    return 0;
  }
  return 1;
}
