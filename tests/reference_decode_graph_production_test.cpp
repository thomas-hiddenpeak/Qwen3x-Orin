#include "q3x/runtime/reference_engine.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
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

constexpr std::uint32_t kFirstDecodePosition = 19U;
constexpr std::uint32_t kLastDecodePosition = 43U;
constexpr std::size_t kDecodeGraphSlotCount = 25U;
constexpr std::uint32_t kCanonicalMaxNewTokens = 26U;
constexpr std::uint32_t kBoundaryMaxNewTokens = 27U;
constexpr std::uint32_t kRequestMaxSequenceLength = 45U;
constexpr std::uint32_t kPrefillChunkSize = 32U;
constexpr std::uint32_t kBoundaryStopTokenId = 248056U;
constexpr std::uint64_t kExpectedRequestArenaBytes = 87'846'400ULL;
constexpr double kMaximumPrepareMilliseconds = 1'000.0;
constexpr std::uint64_t kMaximumFreeDropBytes =
    256ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMinimumFreeBytesAfterCreate =
    8ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr int kSkipReturnCode = 77;

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

std::string model_directory_from(const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* environment =
      std::getenv("Q3X_DECODE_GRAPH_PRODUCTION_MODEL_DIR");
  if (environment != nullptr && environment[0] != '\0') {
    return environment;
  }
  environment = std::getenv("Q3X_E2E_MODEL_DIR");
  return environment != nullptr && environment[0] != '\0'
             ? std::string(environment)
             : std::string{};
}

void print_diagnostic(const std::string_view label,
                      const runtime::ReferenceEngineDiagnostic& diagnostic) {
  std::cerr << label << " failed: code="
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

template <std::size_t Size>
bool exact_tokens(const std::vector<std::uint32_t>& actual,
                  const std::array<std::uint32_t, Size>& expected) {
  return actual.size() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

bool exact_token_prefix(
    const std::vector<std::uint32_t>& actual,
    const std::array<std::uint32_t, kExpectedGeneratedIds.size()>& expected) {
  return actual.size() >= expected.size() &&
         std::equal(expected.begin(), expected.end(), actual.begin());
}

bool same_generation_semantics(
    const runtime::ReferenceGeneration& expected,
    const runtime::ReferenceGeneration& actual) noexcept {
  if (expected.rendered_prompt != actual.rendered_prompt ||
      expected.prompt_token_ids != actual.prompt_token_ids ||
      expected.generated_token_ids != actual.generated_token_ids ||
      expected.generated_text != actual.generated_text ||
      expected.stop_reason != actual.stop_reason ||
      expected.requested_prefill_chunk_size !=
          actual.requested_prefill_chunk_size ||
      expected.effective_prefill_chunk_size !=
          actual.effective_prefill_chunk_size ||
      expected.steps.size() != actual.steps.size() ||
      expected.traces.size() != actual.traces.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < expected.steps.size(); ++index) {
    const runtime::ReferenceStepResult& left = expected.steps[index];
    const runtime::ReferenceStepResult& right = actual.steps[index];
    if (left.position != right.position ||
        left.input_token_id != right.input_token_id ||
        left.logits.has_value() != right.logits.has_value() ||
        left.prediction.has_value() != right.prediction.has_value()) {
      return false;
    }
    if (left.logits.has_value() &&
        (left.logits->predicted_token_id !=
             right.logits->predicted_token_id ||
         left.logits->chosen_logit != right.logits->chosen_logit ||
         left.logits->max_log_probability !=
             right.logits->max_log_probability ||
         left.logits->logsumexp != right.logits->logsumexp)) {
      return false;
    }
    if (left.prediction.has_value() &&
        left.prediction->predicted_token_id !=
            right.prediction->predicted_token_id) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < expected.traces.size(); ++index) {
    const runtime::ReferenceTraceDigest& left = expected.traces[index];
    const runtime::ReferenceTraceDigest& right = actual.traces[index];
    if (left.position != right.position ||
        left.input_token_id != right.input_token_id ||
        left.element_count != right.element_count ||
        left.full_sha256 != right.full_sha256 ||
        left.embedding_sha256 != right.embedding_sha256 ||
        left.layer_hidden_sha256 != right.layer_hidden_sha256 ||
        left.layer_residual_sha256 != right.layer_residual_sha256 ||
        left.final_norm_sha256 != right.final_norm_sha256) {
      return false;
    }
  }
  return true;
}

bool exact_step_shape(const runtime::ReferenceGeneration& generation,
                      const runtime::ReferenceLogitsMode logits_mode,
                      const bool capture_trace,
                      const std::size_t generated_count) noexcept {
  const std::size_t expected_steps =
      kExpectedPromptIds.size() + generated_count - 1U;
  if (generation.steps.size() != expected_steps ||
      generation.timing.subsequent_token_milliseconds.size() !=
          generated_count - 1U ||
      generation.traces.size() != (capture_trace ? expected_steps : 0U)) {
    return false;
  }
  for (std::size_t index = 0U; index < generation.steps.size(); ++index) {
    const std::uint32_t expected_input =
        index < kExpectedPromptIds.size()
            ? kExpectedPromptIds[index]
            : generation.generated_token_ids[
                  index - kExpectedPromptIds.size()];
    const runtime::ReferenceStepResult& step = generation.steps[index];
    const bool compute = index + 1U >= kExpectedPromptIds.size();
    const bool expects_logits =
        compute && logits_mode == runtime::ReferenceLogitsMode::kFullStatistics;
    const bool expects_prediction =
        compute &&
        logits_mode == runtime::ReferenceLogitsMode::kPredictedTokenOnly;
    const bool expects_timing = compute || capture_trace;
    if (step.position != static_cast<std::uint32_t>(index) ||
        step.input_token_id != expected_input ||
        step.logits.has_value() != expects_logits ||
        step.prediction.has_value() != expects_prediction ||
        step.timing.has_value() != expects_timing ||
        (step.timing.has_value() &&
         (!std::isfinite(step.timing->elapsed_milliseconds) ||
          step.timing->elapsed_milliseconds < 0.0))) {
      return false;
    }
    if (compute) {
      const std::size_t generated_index =
          index + 1U - kExpectedPromptIds.size();
      const std::uint32_t prediction =
          expects_logits ? step.logits->predicted_token_id
                         : step.prediction->predicted_token_id;
      if (prediction != generation.generated_token_ids[generated_index]) {
        return false;
      }
    }
    if (capture_trace) {
      const runtime::ReferenceTraceDigest& trace = generation.traces[index];
      if (trace.position != step.position ||
          trace.input_token_id != step.input_token_id ||
          trace.element_count != runtime::kReferenceTraceElements ||
          trace.full_sha256.empty() || trace.embedding_sha256.empty() ||
          trace.final_norm_sha256.empty()) {
        return false;
      }
    }
  }
  return true;
}

bool exact_canonical_generation(
    const runtime::ReferenceGeneration& generation,
    const runtime::ReferenceLogitsMode logits_mode,
    const bool capture_trace, const std::size_t graph_replays,
    const std::size_t graph_fallbacks) noexcept {
  return exact_tokens(generation.prompt_token_ids, kExpectedPromptIds) &&
         exact_tokens(generation.generated_token_ids,
                      kExpectedGeneratedIds) &&
         generation.generated_text == kExpectedText &&
         generation.stop_reason == runtime::ReferenceStopReason::kImEnd &&
         generation.requested_prefill_chunk_size == kPrefillChunkSize &&
         generation.effective_prefill_chunk_size ==
             (capture_trace ? runtime::kDefaultRequestPrefillChunkSize
                            : kPrefillChunkSize) &&
         generation.decode_graph_replays == graph_replays &&
         generation.decode_graph_serial_fallbacks == graph_fallbacks &&
         exact_step_shape(generation, logits_mode, capture_trace,
                          kExpectedGeneratedIds.size());
}

bool exact_boundary_generation_shape(
    const runtime::ReferenceGeneration& generation,
    const std::size_t graph_replays,
    const std::size_t graph_fallbacks) noexcept {
  return exact_tokens(generation.prompt_token_ids, kExpectedPromptIds) &&
         generation.generated_token_ids.size() == kBoundaryMaxNewTokens &&
         exact_token_prefix(generation.generated_token_ids,
                            kExpectedGeneratedIds) &&
         generation.stop_reason == runtime::ReferenceStopReason::kMaxNewTokens &&
         generation.requested_prefill_chunk_size == kPrefillChunkSize &&
         generation.effective_prefill_chunk_size == kPrefillChunkSize &&
         generation.decode_graph_replays == graph_replays &&
         generation.decode_graph_serial_fallbacks == graph_fallbacks &&
         exact_step_shape(generation,
                          runtime::ReferenceLogitsMode::kPredictedTokenOnly,
                          false, kBoundaryMaxNewTokens);
}

runtime::ReferenceGenerateOptions canonical_options(
    const runtime::ReferenceLogitsMode logits_mode,
    const bool capture_trace) noexcept {
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = kCanonicalMaxNewTokens;
  options.stop_token_id = runtime::kQwen36ImEndTokenId;
  options.prefill_chunk_size = kPrefillChunkSize;
  options.logits_mode = logits_mode;
  options.capture_trace = capture_trace;
  return options;
}

runtime::ReferenceGenerateOptions boundary_options() noexcept {
  runtime::ReferenceGenerateOptions options = canonical_options(
      runtime::ReferenceLogitsMode::kPredictedTokenOnly, false);
  options.max_new_tokens = kBoundaryMaxNewTokens;
  // The image-pad token is outside this text-only generation, allowing
  // im_end to be consumed at P44 and proving the first position beyond the
  // admitted cache.
  options.stop_token_id = kBoundaryStopTokenId;
  return options;
}

runtime::ReferenceEngineOptions engine_options(
    const runtime::ReferenceDecodeGraphCachePolicy policy) noexcept {
  runtime::ReferenceEngineOptions options;
  options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  options.enable_trace = true;
  options.decode_graph_cache_policy = policy;
  options.request_options.max_sequence_length = kRequestMaxSequenceLength;
  options.request_options.prefill_chunk_size = kPrefillChunkSize;
  options.request_options.min_free_bytes_after_create =
      kMinimumFreeBytesAfterCreate;
  return options;
}

bool finite_nonnegative(const double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

void print_bool(const std::string_view key, const bool value) {
  std::cout << key << '=' << (value ? 1 : 0) << '\n';
}

}  // namespace

int main(const int argc, char** const argv) {
  const char* const enabled =
      std::getenv("Q3X_RUN_SM87_DECODE_GRAPH_PRODUCTION");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "decode_graph_production.status=skip\n"
              << "SKIP: set Q3X_RUN_SM87_DECODE_GRAPH_PRODUCTION=1 to run "
                 "the full-model production Decode Graph test\n";
    return kSkipReturnCode;
  }
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_decode_graph_production_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }

  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "decode_graph_production.status=skip\n"
              << "SKIP: provide MODEL_DIR, "
                 "Q3X_DECODE_GRAPH_PRODUCTION_MODEL_DIR, or "
                 "Q3X_E2E_MODEL_DIR\n";
    return kSkipReturnCode;
  }

  int device = 0;
  (void)cudaGetLastError();
  cudaError_t cuda_status = cudaGetDevice(&device);
  cudaDeviceProp properties{};
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaGetDeviceProperties(&properties, device);
  }
  if (cuda_status != cudaSuccess || properties.major != 8 ||
      properties.minor != 7) {
    std::cout << "decode_graph_production.status=skip\n";
    if (cuda_status == cudaSuccess) {
      std::cout << "decode_graph_production.device.sm=" << properties.major
                << properties.minor << '\n';
    }
    std::cout << "SKIP: production Decode Graph requires an SM87 device\n";
    return kSkipReturnCode;
  }

  runtime::RequestMemoryOptions plan_options;
  plan_options.max_sequence_length = kRequestMaxSequenceLength;
  plan_options.prefill_chunk_size = kPrefillChunkSize;
  const runtime::RequestPlanResult expected_plan =
      runtime::build_request_memory_plan(plan_options);
  if (!expected_plan ||
      expected_plan.value->arena_bytes != kExpectedRequestArenaBytes) {
    std::cerr << "fixed C32/P45 request arena oracle changed: actual="
              << (expected_plan ? expected_plan.value->arena_bytes : 0U)
              << '\n';
    return 1;
  }

  // Build the disabled SM87 engine first and retain only compact generation
  // semantics before constructing the production-policy engine. This keeps
  // the test's peak memory at one resident 27B model.
  runtime::ReferenceEngineCreateResult serial_created =
      runtime::create_reference_engine(
          model_directory,
          engine_options(runtime::ReferenceDecodeGraphCachePolicy::kDisabled));
  if (!serial_created) {
    print_diagnostic("disabled engine creation", serial_created.diagnostic);
    return 1;
  }
  const runtime::ReferenceEngineLoadStats serial_load =
      serial_created.value->load_stats();

  runtime::ReferenceGenerateResult serial_canonical_result =
      serial_created.value->generate(
          kPrompt,
          canonical_options(
              runtime::ReferenceLogitsMode::kPredictedTokenOnly, false));
  runtime::ReferenceGenerateResult serial_full_statistics_result =
      serial_created.value->generate(
          kPrompt,
          canonical_options(runtime::ReferenceLogitsMode::kFullStatistics,
                            false));
  runtime::ReferenceGenerateResult serial_trace_result =
      serial_created.value->generate(
          kPrompt,
          canonical_options(
              runtime::ReferenceLogitsMode::kPredictedTokenOnly, true));
  runtime::ReferenceGenerateResult serial_boundary_result =
      serial_created.value->generate(kPrompt, boundary_options());
  if (!serial_canonical_result || !serial_full_statistics_result ||
      !serial_trace_result || !serial_boundary_result) {
    if (!serial_canonical_result) {
      print_diagnostic("disabled canonical generation",
                       serial_canonical_result.diagnostic);
    } else if (!serial_full_statistics_result) {
      print_diagnostic("disabled full-statistics generation",
                       serial_full_statistics_result.diagnostic);
    } else if (!serial_trace_result) {
      print_diagnostic("disabled trace generation",
                       serial_trace_result.diagnostic);
    } else {
      print_diagnostic("disabled P44 generation",
                       serial_boundary_result.diagnostic);
    }
    return 1;
  }

  runtime::ReferenceGeneration serial_canonical =
      std::move(*serial_canonical_result.value);
  runtime::ReferenceGeneration serial_full_statistics =
      std::move(*serial_full_statistics_result.value);
  runtime::ReferenceGeneration serial_trace =
      std::move(*serial_trace_result.value);
  runtime::ReferenceGeneration serial_boundary =
      std::move(*serial_boundary_result.value);
  serial_created.value.reset();
  cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    std::cerr << "failed to synchronize after disabled engine release: "
              << cudaGetErrorString(cuda_status) << '\n';
    return 1;
  }

  runtime::ReferenceEngineCreateResult production_created =
      runtime::create_reference_engine(
          model_directory,
          engine_options(
              runtime::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions));
  if (!production_created) {
    print_diagnostic("production engine creation",
                     production_created.diagnostic);
    return 1;
  }
  const runtime::ReferenceEngineLoadStats& production_load =
      production_created.value->load_stats();

  runtime::ReferenceGenerateResult production_canonical_result =
      production_created.value->generate(
          kPrompt,
          canonical_options(
              runtime::ReferenceLogitsMode::kPredictedTokenOnly, false));
  // generate() owns reset; a second request proves reset preserves the
  // engine-lifetime graph bank and its dispatch policy.
  runtime::ReferenceGenerateResult production_after_reset_result =
      production_created.value->generate(
          kPrompt,
          canonical_options(
              runtime::ReferenceLogitsMode::kPredictedTokenOnly, false));
  runtime::ReferenceGenerateResult production_full_statistics_result =
      production_created.value->generate(
          kPrompt,
          canonical_options(runtime::ReferenceLogitsMode::kFullStatistics,
                            false));
  runtime::ReferenceGenerateResult production_trace_result =
      production_created.value->generate(
          kPrompt,
          canonical_options(
              runtime::ReferenceLogitsMode::kPredictedTokenOnly, true));
  runtime::ReferenceGenerateResult production_boundary_result =
      production_created.value->generate(kPrompt, boundary_options());
  if (!production_canonical_result || !production_after_reset_result ||
      !production_full_statistics_result || !production_trace_result ||
      !production_boundary_result) {
    if (!production_canonical_result) {
      print_diagnostic("production canonical generation",
                       production_canonical_result.diagnostic);
    } else if (!production_after_reset_result) {
      print_diagnostic("production post-reset generation",
                       production_after_reset_result.diagnostic);
    } else if (!production_full_statistics_result) {
      print_diagnostic("production full-statistics generation",
                       production_full_statistics_result.diagnostic);
    } else if (!production_trace_result) {
      print_diagnostic("production trace generation",
                       production_trace_result.diagnostic);
    } else {
      print_diagnostic("production P44 generation",
                       production_boundary_result.diagnostic);
    }
    return 1;
  }

  const runtime::ReferenceGeneration& production_canonical =
      *production_canonical_result.value;
  const runtime::ReferenceGeneration& production_after_reset =
      *production_after_reset_result.value;
  const runtime::ReferenceGeneration& production_full_statistics =
      *production_full_statistics_result.value;
  const runtime::ReferenceGeneration& production_trace =
      *production_trace_result.value;
  const runtime::ReferenceGeneration& production_boundary =
      *production_boundary_result.value;

  const bool serial_oracles_exact =
      exact_canonical_generation(
          serial_canonical,
          runtime::ReferenceLogitsMode::kPredictedTokenOnly, false, 0U,
          0U) &&
      exact_canonical_generation(
          serial_full_statistics,
          runtime::ReferenceLogitsMode::kFullStatistics, false, 0U, 0U) &&
      exact_canonical_generation(
          serial_trace,
          runtime::ReferenceLogitsMode::kPredictedTokenOnly, true, 0U,
          0U) &&
      exact_boundary_generation_shape(serial_boundary, 0U, 0U);
  const bool load_contract_exact =
      production_load.decode_graph_cache_requested_policy ==
          runtime::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions &&
      production_load.decode_graph_cache_effective_policy ==
          runtime::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions &&
      production_load.decode_graph_cache_first_position ==
          kFirstDecodePosition &&
      production_load.decode_graph_cache_last_position ==
          kLastDecodePosition &&
      production_load.decode_graph_cache_slot_count ==
          kDecodeGraphSlotCount &&
      production_load.decode_graph_cache_fallback_reason.empty();
  const double prepare_phase_sum =
      production_load.decode_graph_cache_capture_enqueue_milliseconds +
      production_load.decode_graph_cache_topology_inspection_milliseconds +
      production_load.decode_graph_cache_instantiate_milliseconds +
      production_load.decode_graph_cache_upload_ready_milliseconds;
  const bool cold_time_gate =
      finite_nonnegative(
          production_load.decode_graph_cache_capture_enqueue_milliseconds) &&
      finite_nonnegative(
          production_load
              .decode_graph_cache_topology_inspection_milliseconds) &&
      finite_nonnegative(
          production_load.decode_graph_cache_instantiate_milliseconds) &&
      finite_nonnegative(
          production_load.decode_graph_cache_upload_ready_milliseconds) &&
      finite_nonnegative(
          production_load.decode_graph_cache_prepare_milliseconds) &&
      prepare_phase_sum <=
          production_load.decode_graph_cache_prepare_milliseconds &&
      production_load.decode_graph_cache_prepare_milliseconds <=
          kMaximumPrepareMilliseconds;
  const std::uint64_t expected_free_drop =
      production_load.decode_graph_cache_free_bytes_before >=
              production_load.decode_graph_cache_free_bytes_after
          ? production_load.decode_graph_cache_free_bytes_before -
                production_load.decode_graph_cache_free_bytes_after
          : 0U;
  const bool memory_gate =
      production_load.decode_graph_cache_free_drop_bytes ==
          expected_free_drop &&
      production_load.decode_graph_cache_free_drop_bytes <=
          kMaximumFreeDropBytes &&
      production_load.decode_graph_cache_free_bytes_after >=
          kMinimumFreeBytesAfterCreate;
  const bool arena_layout_exact =
      serial_load.request_arena_bytes == kExpectedRequestArenaBytes &&
      production_load.request_arena_bytes == kExpectedRequestArenaBytes &&
      serial_load.request_arena_bytes == production_load.request_arena_bytes &&
      production_load.request_max_sequence_length ==
          kRequestMaxSequenceLength &&
      production_load.request_prefill_chunk_size == kPrefillChunkSize;
  const bool canonical_exact =
      exact_canonical_generation(
          production_canonical,
          runtime::ReferenceLogitsMode::kPredictedTokenOnly, false,
          kDecodeGraphSlotCount, 0U) &&
      same_generation_semantics(serial_canonical, production_canonical);
  const bool post_reset_exact =
      exact_canonical_generation(
          production_after_reset,
          runtime::ReferenceLogitsMode::kPredictedTokenOnly, false,
          kDecodeGraphSlotCount, 0U) &&
      same_generation_semantics(serial_canonical, production_after_reset) &&
      same_generation_semantics(production_canonical,
                                production_after_reset);
  const bool full_statistics_serial_exact =
      exact_canonical_generation(
          production_full_statistics,
          runtime::ReferenceLogitsMode::kFullStatistics, false, 0U, 0U) &&
      same_generation_semantics(serial_full_statistics,
                                production_full_statistics);
  const bool trace_serial_exact =
      exact_canonical_generation(
          production_trace,
          runtime::ReferenceLogitsMode::kPredictedTokenOnly, true, 0U,
          0U) &&
      same_generation_semantics(serial_trace, production_trace);
  const bool p44_miss_exact =
      exact_boundary_generation_shape(production_boundary,
                                      kDecodeGraphSlotCount, 1U) &&
      same_generation_semantics(serial_boundary, production_boundary) &&
      production_boundary.steps.back().position == 44U &&
      production_boundary.steps.back().input_token_id ==
          runtime::kQwen36ImEndTokenId;

  TestContext test;
  test.expect(serial_oracles_exact,
              "disabled SM87 serial generations match the pinned oracle");
  test.expect(load_contract_exact,
              "production policy admits the exact P19-P43 25-slot cache");
  test.expect(cold_time_gate,
              "production cache preparation stays within the 1s gate");
  test.expect(memory_gate,
              "production cache stays within the 256MiB device-memory gate");
  test.expect(arena_layout_exact,
              "disabled and production engines retain the exact C32/P45 "
              "request arena layout");
  test.expect(canonical_exact,
              "ordinary predicted-only generate is serial-exact with 25/0 "
              "production dispatch");
  test.expect(post_reset_exact,
              "a second ordinary generate preserves exact 25/0 dispatch");
  test.expect(full_statistics_serial_exact,
              "full statistics stays on the exact serial path with 0/0 "
              "dispatch counters");
  test.expect(trace_serial_exact,
              "trace capture stays on the exact serial path with 0/0 "
              "dispatch counters");
  test.expect(p44_miss_exact,
              "max27 executes P19-P43 as graph hits and P44 as one exact "
              "serial miss");

  const bool passed = test.failures() == 0;
  std::cout << std::fixed << std::setprecision(6)
            << "decode_graph_production.cache.first_position="
            << production_load.decode_graph_cache_first_position << '\n'
            << "decode_graph_production.cache.last_position="
            << production_load.decode_graph_cache_last_position << '\n'
            << "decode_graph_production.cache.slot_count="
            << production_load.decode_graph_cache_slot_count << '\n'
            << "decode_graph_production.prepare.capture_ms="
            << production_load.decode_graph_cache_capture_enqueue_milliseconds
            << '\n'
            << "decode_graph_production.prepare.topology_ms="
            << production_load
                   .decode_graph_cache_topology_inspection_milliseconds
            << '\n'
            << "decode_graph_production.prepare.instantiate_ms="
            << production_load.decode_graph_cache_instantiate_milliseconds
            << '\n'
            << "decode_graph_production.prepare.upload_ready_ms="
            << production_load.decode_graph_cache_upload_ready_milliseconds
            << '\n'
            << "decode_graph_production.prepare.phase_sum_ms="
            << prepare_phase_sum << '\n'
            << "decode_graph_production.prepare.wall_ms="
            << production_load.decode_graph_cache_prepare_milliseconds << '\n'
            << "decode_graph_production.memory.free_before_bytes="
            << production_load.decode_graph_cache_free_bytes_before << '\n'
            << "decode_graph_production.memory.free_after_bytes="
            << production_load.decode_graph_cache_free_bytes_after << '\n'
            << "decode_graph_production.memory.free_drop_bytes="
            << production_load.decode_graph_cache_free_drop_bytes << '\n'
            << "decode_graph_production.arena.bytes="
            << production_load.request_arena_bytes << '\n'
            << "decode_graph_production.canonical.replays="
            << production_canonical.decode_graph_replays << '\n'
            << "decode_graph_production.canonical.fallbacks="
            << production_canonical.decode_graph_serial_fallbacks << '\n'
            << "decode_graph_production.reset.replays="
            << production_after_reset.decode_graph_replays << '\n'
            << "decode_graph_production.reset.fallbacks="
            << production_after_reset.decode_graph_serial_fallbacks << '\n'
            << "decode_graph_production.full_statistics.replays="
            << production_full_statistics.decode_graph_replays << '\n'
            << "decode_graph_production.full_statistics.fallbacks="
            << production_full_statistics.decode_graph_serial_fallbacks
            << '\n'
            << "decode_graph_production.trace.replays="
            << production_trace.decode_graph_replays << '\n'
            << "decode_graph_production.trace.fallbacks="
            << production_trace.decode_graph_serial_fallbacks << '\n'
            << "decode_graph_production.p44.replays="
            << production_boundary.decode_graph_replays << '\n'
            << "decode_graph_production.p44.fallbacks="
            << production_boundary.decode_graph_serial_fallbacks << '\n';
  print_bool("decode_graph_production.serial_oracles.exact",
             serial_oracles_exact);
  print_bool("decode_graph_production.load_contract.exact",
             load_contract_exact);
  print_bool("decode_graph_production.arena_layout.exact",
             arena_layout_exact);
  print_bool("decode_graph_production.canonical.exact", canonical_exact);
  print_bool("decode_graph_production.reset.exact", post_reset_exact);
  print_bool("decode_graph_production.full_statistics.exact",
             full_statistics_serial_exact);
  print_bool("decode_graph_production.trace.exact", trace_serial_exact);
  print_bool("decode_graph_production.p44_miss.exact", p44_miss_exact);
  print_bool("decode_graph_production.gate.cold_time", cold_time_gate);
  print_bool("decode_graph_production.gate.memory", memory_gate);
  print_bool("decode_graph_production.gate.pass", passed);
  std::cout << "decode_graph_production.status="
            << (passed ? "pass" : "failure") << '\n';
  return passed ? 0 : 1;
}
