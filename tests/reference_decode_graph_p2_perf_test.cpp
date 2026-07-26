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
constexpr std::uint32_t kBoundaryGraphPosition = 63U;
constexpr std::uint32_t kBoundarySeedPrediction = 9'419U;
constexpr std::uint32_t kBoundaryGraphPrediction = 0U;
constexpr std::uint32_t kBoundaryFallbackPrediction = 59'720U;
constexpr std::size_t kExpectedNodes = 390U;
constexpr std::size_t kExpectedKernelNodes = 389U;
constexpr std::size_t kExpectedMemcpyNodes = 1U;
constexpr std::size_t kExpectedOtherNodes = 0U;
constexpr double kMinimumMedianDeltaMilliseconds = 0.70;
constexpr double kMaximumCachePrepareMilliseconds = 1'000.0;
constexpr std::uint64_t kMaximumObservedMemoryIncreaseBytes =
    256ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMinimumWinningRounds =
    runtime::kReferenceDecodeGraphP2ScreenRounds;
constexpr int kSkipReturnCode = 77;

std::string model_directory_from(const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* environment =
      std::getenv("Q3X_DECODE_GRAPH_P2_MODEL_DIR");
  if (environment != nullptr && environment[0] != '\0') {
    return environment;
  }
  environment = std::getenv("Q3X_E2E_MODEL_DIR");
  if (environment != nullptr && environment[0] != '\0') {
    return environment;
  }
  return {};
}

void print_diagnostic(const runtime::ReferenceEngineDiagnostic& diagnostic) {
  std::cerr << "decode graph P2 screen failed: code="
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

bool exact_generation(const runtime::ReferenceGeneration& generation,
                      const std::size_t graph_replays,
                      const std::size_t graph_fallbacks) {
  if (!exact_tokens(generation.prompt_token_ids, kExpectedPromptIds) ||
      !exact_tokens(generation.generated_token_ids,
                    kExpectedGeneratedIds) ||
      generation.generated_text != kExpectedText ||
      generation.stop_reason != runtime::ReferenceStopReason::kImEnd ||
      generation.steps.size() != 44U ||
      generation.timing.subsequent_token_milliseconds.size() !=
          runtime::kReferenceDecodeGraphP2ContinuousSteps ||
      generation.decode_graph_replays != graph_replays ||
      generation.decode_graph_serial_fallbacks != graph_fallbacks) {
    return false;
  }
  for (std::size_t index = 0U; index < generation.steps.size(); ++index) {
    const std::uint32_t expected_input =
        index < kExpectedPromptIds.size()
            ? kExpectedPromptIds[index]
            : kExpectedGeneratedIds[index - kExpectedPromptIds.size()];
    const runtime::ReferenceStepResult& step = generation.steps[index];
    const bool expects_prediction = index + 1U >= kExpectedPromptIds.size();
    const bool expects_scalar_timing = expects_prediction;
    if (step.position != static_cast<std::uint32_t>(index) ||
        step.input_token_id != expected_input || step.logits.has_value() ||
        step.prediction.has_value() != expects_prediction ||
        step.timing.has_value() != expects_scalar_timing ||
        (step.timing.has_value() &&
         (!std::isfinite(step.timing->elapsed_milliseconds) ||
          step.timing->elapsed_milliseconds < 0.0))) {
      return false;
    }
    if (expects_prediction) {
      const std::size_t generated_index =
          index + 1U - kExpectedPromptIds.size();
      if (step.prediction->predicted_token_id !=
          kExpectedGeneratedIds[generated_index]) {
        return false;
      }
    }
  }
  return true;
}

bool same_generation_without_timing(
    const runtime::ReferenceGeneration& left,
    const runtime::ReferenceGeneration& right) {
  if (left.rendered_prompt != right.rendered_prompt ||
      left.prompt_token_ids != right.prompt_token_ids ||
      left.generated_token_ids != right.generated_token_ids ||
      left.generated_text != right.generated_text ||
      left.stop_reason != right.stop_reason ||
      left.requested_prefill_chunk_size !=
          right.requested_prefill_chunk_size ||
      left.effective_prefill_chunk_size !=
          right.effective_prefill_chunk_size ||
      !left.traces.empty() || !right.traces.empty() ||
      left.steps.size() != right.steps.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.steps.size(); ++index) {
    const runtime::ReferenceStepResult& expected = left.steps[index];
    const runtime::ReferenceStepResult& actual = right.steps[index];
    if (expected.position != actual.position ||
        expected.input_token_id != actual.input_token_id ||
        expected.logits.has_value() != actual.logits.has_value() ||
        expected.prediction.has_value() != actual.prediction.has_value()) {
      return false;
    }
    if (expected.logits.has_value() &&
        (expected.logits->predicted_token_id !=
             actual.logits->predicted_token_id ||
         expected.logits->chosen_logit != actual.logits->chosen_logit ||
         expected.logits->max_log_probability !=
             actual.logits->max_log_probability ||
         expected.logits->logsumexp != actual.logits->logsumexp)) {
      return false;
    }
    if (expected.prediction.has_value() &&
        expected.prediction->predicted_token_id !=
            actual.prediction->predicted_token_id) {
      return false;
    }
  }
  return true;
}

template <std::size_t Size>
double median(std::array<double, Size> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

void print_bool(const std::string_view key, const bool value) {
  std::cout << key << '=' << (value ? 1 : 0) << '\n';
}

bool finite_prepare_stats(
    const runtime::ReferenceDecodeGraphP1Stats& stats) noexcept {
  return std::isfinite(stats.capture_enqueue_milliseconds) &&
         stats.capture_enqueue_milliseconds >= 0.0 &&
         std::isfinite(stats.topology_inspection_milliseconds) &&
         stats.topology_inspection_milliseconds >= 0.0 &&
         std::isfinite(stats.instantiate_milliseconds) &&
         stats.instantiate_milliseconds >= 0.0 &&
         std::isfinite(stats.upload_ready_milliseconds) &&
         stats.upload_ready_milliseconds >= 0.0 &&
         std::isfinite(stats.total_prepare_milliseconds) &&
         stats.total_prepare_milliseconds >= 0.0;
}

bool exact_graph_stats(const runtime::ReferenceDecodeGraphP1Stats& stats,
                       const std::uint32_t position) noexcept {
  return stats.position == position && stats.input_token_id == 0U &&
         stats.node_count == kExpectedNodes &&
         stats.kernel_node_count == kExpectedKernelNodes &&
         stats.memcpy_node_count == kExpectedMemcpyNodes &&
         stats.other_node_count == kExpectedOtherNodes &&
         finite_prepare_stats(stats);
}

}  // namespace

int main(const int argc, char** const argv) {
  const char* const enabled =
      std::getenv("Q3X_RUN_SM87_DECODE_GRAPH_P2_PERF");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "decode_graph_p2.status=skip\n"
              << "SKIP: set Q3X_RUN_SM87_DECODE_GRAPH_P2_PERF=1 to run "
                 "the full-model SM87 CUDA Graph P2 screen\n";
    return kSkipReturnCode;
  }
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_decode_graph_p2_perf_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }

  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "decode_graph_p2.status=error\n";
    std::cerr << "Q3X_RUN_SM87_DECODE_GRAPH_P2_PERF=1 requires MODEL_DIR, "
                 "Q3X_DECODE_GRAPH_P2_MODEL_DIR, or Q3X_E2E_MODEL_DIR\n";
    return 2;
  }

  int device = 0;
  (void)cudaGetLastError();
  cudaError_t cuda_status = cudaGetDevice(&device);
  cudaDeviceProp properties{};
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaGetDeviceProperties(&properties, device);
  }
  if (cuda_status != cudaSuccess) {
    std::cout << "decode_graph_p2.status=error\n";
    std::cerr << "failed to inspect CUDA device: "
              << cudaGetErrorString(cuda_status) << '\n';
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "decode_graph_p2.status=skip\n"
              << "decode_graph_p2.device.sm=" << properties.major
              << properties.minor << '\n'
              << "SKIP: decode graph P2 requires an SM87 device\n";
    return kSkipReturnCode;
  }

  runtime::ReferenceEngineOptions engine_options;
  engine_options.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  engine_options.enable_trace = true;
  engine_options.request_options.max_sequence_length = 65U;
  engine_options.request_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(model_directory, engine_options);
  if (!created) {
    std::cout << "decode_graph_p2.status=error\n";
    print_diagnostic(created.diagnostic);
    return 1;
  }

  runtime::ReferenceDecodeGraphP2ScreenOptions screen_options;
  screen_options.first_decode_position = kFirstDecodePosition;
  screen_options.last_decode_position = kLastDecodePosition;
  screen_options.boundary_graph_position = kBoundaryGraphPosition;
  screen_options.capture_input_token_id = 0U;
  screen_options.boundary_input_token_id = kBoundarySeedPrediction;
  screen_options.max_new_tokens =
      static_cast<std::uint32_t>(kExpectedGeneratedIds.size());
  screen_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  runtime::ReferenceDecodeGraphP2ScreenOutcome screened =
      created.value->screen_short_decode_graph_cache_p2(
          kPrompt, screen_options);
  if (!screened) {
    std::cout << "decode_graph_p2.status=error\n";
    print_diagnostic(screened.diagnostic);
    return 1;
  }

  const runtime::ReferenceDecodeGraphP2ScreenResult& result =
      *screened.value;
  const std::uint64_t sm87_request_arena_bytes =
      created.value->load_stats().request_arena_bytes;

  // Exercise the public generation dispatcher on the real reference backend,
  // where no graph can exist. Release the SM87 engine first so this hard
  // fallback gate does not require two resident copies of the 27B model.
  created.value.reset();
  cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    std::cout << "decode_graph_p2.status=error\n";
    std::cerr << "failed to synchronize after releasing SM87 engine: "
              << cudaGetErrorString(cuda_status) << '\n';
    return 1;
  }

  runtime::ReferenceEngineOptions reference_engine_options;
  reference_engine_options.projection_backend =
      runtime::ProjectionBackend::kReference;
  reference_engine_options.request_options.max_sequence_length = 64U;
  reference_engine_options.request_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  runtime::ReferenceEngineCreateResult reference_created =
      runtime::create_reference_engine(model_directory,
                                       reference_engine_options);
  if (!reference_created) {
    std::cout << "decode_graph_p2.status=error\n";
    print_diagnostic(reference_created.diagnostic);
    return 1;
  }

  runtime::ReferenceGenerateOptions reference_generate_options;
  reference_generate_options.max_new_tokens = 2U;
  reference_generate_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  reference_generate_options.logits_mode =
      runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  runtime::ReferenceGenerateResult reference_cache_off =
      reference_created.value->generate(kPrompt,
                                        reference_generate_options);
  if (!reference_cache_off) {
    std::cout << "decode_graph_p2.status=error\n";
    print_diagnostic(reference_cache_off.diagnostic);
    return 1;
  }
  reference_generate_options.use_prepared_decode_graph_cache = true;
  runtime::ReferenceGenerateResult reference_cache_on =
      reference_created.value->generate(kPrompt,
                                        reference_generate_options);
  if (!reference_cache_on) {
    std::cout << "decode_graph_p2.status=error\n";
    print_diagnostic(reference_cache_on.diagnostic);
    return 1;
  }

  constexpr std::array<std::uint32_t, 2U> kExpectedReferenceGeneratedIds = {
      kExpectedGeneratedIds[0U], kExpectedGeneratedIds[1U]};
  const runtime::ReferenceGeneration& reference_off =
      *reference_cache_off.value;
  const runtime::ReferenceGeneration& reference_on =
      *reference_cache_on.value;
  const bool reference_backend_fallback_exact =
      exact_tokens(reference_off.prompt_token_ids, kExpectedPromptIds) &&
      exact_tokens(reference_off.generated_token_ids,
                   kExpectedReferenceGeneratedIds) &&
      same_generation_without_timing(reference_off, reference_on) &&
      reference_off.decode_graph_replays == 0U &&
      reference_off.decode_graph_serial_fallbacks == 0U &&
      reference_on.decode_graph_replays == 0U &&
      reference_on.decode_graph_serial_fallbacks == 1U;
  reference_created.value.reset();

  bool topology_exact = true;
  double capture_sum = 0.0;
  double topology_sum = 0.0;
  double instantiate_sum = 0.0;
  double upload_ready_sum = 0.0;
  double slot_total_sum = 0.0;
  for (std::size_t index = 0U;
       index < result.continuous_graphs.size(); ++index) {
    const runtime::ReferenceDecodeGraphP1Stats& stats =
        result.continuous_graphs[index];
    topology_exact =
        topology_exact &&
        exact_graph_stats(stats,
                          kFirstDecodePosition +
                              static_cast<std::uint32_t>(index));
    capture_sum += stats.capture_enqueue_milliseconds;
    topology_sum += stats.topology_inspection_milliseconds;
    instantiate_sum += stats.instantiate_milliseconds;
    upload_ready_sum += stats.upload_ready_milliseconds;
    slot_total_sum += stats.total_prepare_milliseconds;
  }
  topology_exact =
      topology_exact &&
      exact_graph_stats(result.boundary_graph, kBoundaryGraphPosition);

  const bool serial_oracle_exact =
      exact_generation(result.serial_generation, 0U, 0U);
  const bool graph_oracle_exact = exact_generation(
      result.graph_generation,
      runtime::kReferenceDecodeGraphP2ContinuousSteps, 0U);
  const runtime::ReferenceStepResult& position31_step =
      result.graph_generation.steps[31U];
  const runtime::ReferenceStepResult& position32_step =
      result.graph_generation.steps[32U];
  const bool position31_exact =
      exact_graph_stats(result.continuous_graphs[12U], 31U) &&
      position31_step.position == 31U &&
      position31_step.input_token_id == 3'709U &&
      position31_step.prediction.has_value() &&
      position31_step.prediction->predicted_token_id == 109'238U;
  const bool position32_exact =
      exact_graph_stats(result.continuous_graphs[13U], 32U) &&
      position32_step.position == 32U &&
      position32_step.input_token_id == 109'238U &&
      position32_step.prediction.has_value() &&
      position32_step.prediction->predicted_token_id == 97'327U;
  const bool arena_size_exact =
      result.compared_arena_bytes != 0U &&
      result.compared_arena_bytes == sm87_request_arena_bytes;
  const bool boundary_exact =
      result.boundary_graph_first_prediction ==
          result.boundary_serial_first_prediction &&
      result.boundary_graph_first_prediction == kBoundaryGraphPrediction &&
      result.boundary_fallback_second_prediction ==
          result.boundary_serial_second_prediction &&
      result.boundary_fallback_second_prediction ==
          kBoundaryFallbackPrediction &&
      result.boundary_cache_hit && result.boundary_cache_miss_fallback &&
      result.boundary_arena_exact;
  const bool correctness_gate =
      topology_exact && result.cache_prepare_arena_exact &&
      result.boundary_prepare_arena_exact &&
      result.cache_miss_fallback_exact && serial_oracle_exact &&
      graph_oracle_exact && position31_exact && position32_exact &&
      result.continuous_generation_exact &&
      result.continuous_arena_exact &&
      result.full_statistics_fallback_exact &&
      result.trace_fallback_exact && reference_backend_fallback_exact &&
      boundary_exact && arena_size_exact;

  const bool cold_time_gate =
      std::isfinite(result.cache_prepare_milliseconds) &&
      result.cache_prepare_milliseconds >= slot_total_sum &&
      result.cache_prepare_milliseconds <=
          kMaximumCachePrepareMilliseconds;
  const bool memory_gate =
      result.cache_cuda_free_drop_bytes <=
          kMaximumObservedMemoryIncreaseBytes;
  const bool host_private_budget_observed =
      result.cache_host_private_observed &&
      result.cache_host_private_increase_bytes <=
          kMaximumObservedMemoryIncreaseBytes;

  const double first_serial_milliseconds =
      result.serial_generation.timing.decode_after_first_milliseconds /
      static_cast<double>(runtime::kReferenceDecodeGraphP2ContinuousSteps);
  const double first_graph_milliseconds =
      result.graph_generation.timing.decode_after_first_milliseconds /
      static_cast<double>(runtime::kReferenceDecodeGraphP2ContinuousSteps);
  const double first_delta_milliseconds =
      first_serial_milliseconds - first_graph_milliseconds;
  const bool first_distinct_chain_gate =
      std::isfinite(first_delta_milliseconds) &&
      first_delta_milliseconds >= kMinimumMedianDeltaMilliseconds;

  std::array<double, runtime::kReferenceDecodeGraphP2ScreenRounds>
      serial_pairs{};
  std::array<double, runtime::kReferenceDecodeGraphP2ScreenRounds>
      graph_pairs{};
  std::array<double, runtime::kReferenceDecodeGraphP2ScreenRounds> deltas{};
  std::size_t winning_rounds = 0U;
  std::cout << std::fixed << std::setprecision(6);
  for (std::size_t round_index = 0U;
       round_index < result.rounds.size(); ++round_index) {
    const runtime::ReferenceDecodeGraphP2RoundTiming& round =
        result.rounds[round_index];
    serial_pairs[round_index] =
        (round.serial_first_milliseconds_per_token +
         round.serial_second_milliseconds_per_token) /
        2.0;
    graph_pairs[round_index] =
        (round.graph_first_milliseconds_per_token +
         round.graph_second_milliseconds_per_token) /
        2.0;
    deltas[round_index] =
        serial_pairs[round_index] - graph_pairs[round_index];
    if (deltas[round_index] > 0.0) {
      ++winning_rounds;
    }
    const std::string prefix =
        "decode_graph_p2.round." + std::to_string(round_index) + '.';
    std::cout << prefix << "serial_first_ms_per_token="
              << round.serial_first_milliseconds_per_token << '\n'
              << prefix << "graph_first_ms_per_token="
              << round.graph_first_milliseconds_per_token << '\n'
              << prefix << "graph_second_ms_per_token="
              << round.graph_second_milliseconds_per_token << '\n'
              << prefix << "serial_second_ms_per_token="
              << round.serial_second_milliseconds_per_token << '\n'
              << prefix << "serial_pair_ms_per_token="
              << serial_pairs[round_index] << '\n'
              << prefix << "graph_pair_ms_per_token="
              << graph_pairs[round_index] << '\n'
              << prefix << "delta_ms_per_token="
              << deltas[round_index] << '\n';
  }

  const double serial_median = median(serial_pairs);
  const double graph_median = median(graph_pairs);
  const double delta_median = median(deltas);
  const double speedup =
      graph_median > 0.0 ? serial_median / graph_median : 0.0;
  const double graph_tokens_per_second =
      graph_median > 0.0 ? 1'000.0 / graph_median : 0.0;
  const bool timing_valid =
      std::isfinite(serial_median) && serial_median >= 0.0 &&
      std::isfinite(graph_median) && graph_median >= 0.0 &&
      std::isfinite(delta_median) && std::isfinite(speedup);
  const bool hot_performance_gate =
      timing_valid &&
      delta_median >= kMinimumMedianDeltaMilliseconds &&
      winning_rounds >= kMinimumWinningRounds;
  const bool target_100ms_met =
      graph_median <= 100.0 && graph_tokens_per_second >= 10.0;
  const bool passed =
      correctness_gate && cold_time_gate && memory_gate &&
      hot_performance_gate;

  std::cout << "decode_graph_p2.slots.continuous="
            << result.continuous_graphs.size() << '\n'
            << "decode_graph_p2.slots.boundary=1\n"
            << "decode_graph_p2.resource_gate.scope=P19-P43\n"
            << "decode_graph_p2.nodes.per_slot=" << kExpectedNodes << '\n'
            << "decode_graph_p2.kernels.per_slot=" << kExpectedKernelNodes
            << '\n'
            << "decode_graph_p2.prepare.capture_sum_ms=" << capture_sum
            << '\n'
            << "decode_graph_p2.prepare.topology_sum_ms=" << topology_sum
            << '\n'
            << "decode_graph_p2.prepare.instantiate_sum_ms="
            << instantiate_sum << '\n'
            << "decode_graph_p2.prepare.upload_ready_sum_ms="
            << upload_ready_sum << '\n'
            << "decode_graph_p2.prepare.slot_total_sum_ms="
            << slot_total_sum << '\n'
            << "decode_graph_p2.prepare.wall_ms="
            << result.cache_prepare_milliseconds << '\n'
            << "decode_graph_p2.boundary_prepare.total_ms="
            << result.boundary_graph.total_prepare_milliseconds << '\n'
            << "decode_graph_p2.memory.cuda_free_before_bytes="
            << result.cache_free_bytes_before << '\n'
            << "decode_graph_p2.memory.cuda_free_after_bytes="
            << result.cache_free_bytes_after << '\n'
            << "decode_graph_p2.memory.cuda_free_drop_bytes="
            << result.cache_cuda_free_drop_bytes << '\n'
            << "decode_graph_p2.memory.boundary_free_after_bytes="
            << result.boundary_free_bytes_after << '\n'
            << "decode_graph_p2.memory.boundary_increment_bytes="
            << result.boundary_cuda_free_drop_bytes << '\n'
            << "decode_graph_p2.memory.cache_plus_boundary_drop_bytes="
            << result.cache_plus_boundary_cuda_free_drop_bytes << '\n'
            << "decode_graph_p2.memory.host_private_before_bytes="
            << result.cache_host_private_bytes_before << '\n'
            << "decode_graph_p2.memory.host_private_after_bytes="
            << result.cache_host_private_bytes_after << '\n'
            << "decode_graph_p2.memory.host_private_increase_bytes="
            << result.cache_host_private_increase_bytes << '\n'
            << "decode_graph_p2.first_use.serial_ms_per_token="
            << first_serial_milliseconds << '\n'
            << "decode_graph_p2.first_use.graph_ms_per_token="
            << first_graph_milliseconds << '\n'
            << "decode_graph_p2.first_use.delta_ms_per_token="
            << first_delta_milliseconds << '\n'
            << "decode_graph_p2.boundary.serial_first_prediction="
            << result.boundary_serial_first_prediction << '\n'
            << "decode_graph_p2.boundary.graph_first_prediction="
            << result.boundary_graph_first_prediction << '\n'
            << "decode_graph_p2.boundary.serial_second_prediction="
            << result.boundary_serial_second_prediction << '\n'
            << "decode_graph_p2.boundary.fallback_second_prediction="
            << result.boundary_fallback_second_prediction << '\n'
            << "decode_graph_p2.arena.compared_bytes="
            << result.compared_arena_bytes << '\n'
            << "decode_graph_p2.reference_backend.cache_off_replays="
            << reference_off.decode_graph_replays << '\n'
            << "decode_graph_p2.reference_backend.cache_off_fallbacks="
            << reference_off.decode_graph_serial_fallbacks << '\n'
            << "decode_graph_p2.reference_backend.cache_on_replays="
            << reference_on.decode_graph_replays << '\n'
            << "decode_graph_p2.reference_backend.cache_on_fallbacks="
            << reference_on.decode_graph_serial_fallbacks << '\n'
            << "decode_graph_p2.serial_median_ms=" << serial_median << '\n'
            << "decode_graph_p2.graph_median_ms=" << graph_median << '\n'
            << "decode_graph_p2.median_delta_ms=" << delta_median << '\n'
            << "decode_graph_p2.median_speedup=" << speedup << '\n'
            << "decode_graph_p2.graph_tokens_per_second="
            << graph_tokens_per_second << '\n'
            << "decode_graph_p2.winning_rounds=" << winning_rounds << '\n'
            << "decode_graph_p2.gate.min_median_delta_ms="
            << kMinimumMedianDeltaMilliseconds << '\n'
            << "decode_graph_p2.gate.min_winning_rounds="
            << kMinimumWinningRounds << '\n'
            << "decode_graph_p2.gate.max_prepare_ms="
            << kMaximumCachePrepareMilliseconds << '\n'
            << "decode_graph_p2.gate.max_memory_increase_bytes="
            << kMaximumObservedMemoryIncreaseBytes << '\n';
  print_bool("decode_graph_p2.topology.exact", topology_exact);
  print_bool("decode_graph_p2.prepare.arena_exact",
             result.cache_prepare_arena_exact);
  print_bool("decode_graph_p2.boundary_prepare.arena_exact",
             result.boundary_prepare_arena_exact);
  print_bool("decode_graph_p2.cache_miss_fallback.exact",
             result.cache_miss_fallback_exact);
  print_bool("decode_graph_p2.generation.serial_oracle_exact",
             serial_oracle_exact);
  print_bool("decode_graph_p2.generation.graph_oracle_exact",
             graph_oracle_exact);
  print_bool("decode_graph_p2.position31.exact", position31_exact);
  print_bool("decode_graph_p2.position32.exact", position32_exact);
  print_bool("decode_graph_p2.generation.continuous_exact",
             result.continuous_generation_exact);
  print_bool("decode_graph_p2.arena.continuous_exact",
             result.continuous_arena_exact);
  print_bool("decode_graph_p2.full_statistics_fallback.exact",
             result.full_statistics_fallback_exact);
  print_bool("decode_graph_p2.trace_fallback.exact",
             result.trace_fallback_exact);
  print_bool("decode_graph_p2.reference_backend_fallback.exact",
             reference_backend_fallback_exact);
  print_bool("decode_graph_p2.boundary.exact", boundary_exact);
  print_bool("decode_graph_p2.arena.size_exact", arena_size_exact);
  print_bool("decode_graph_p2.memory.host_private_observed",
             result.cache_host_private_observed);
  print_bool("decode_graph_p2.memory.host_private_budget_observed",
             host_private_budget_observed);
  print_bool("decode_graph_p2.gate.correctness", correctness_gate);
  print_bool("decode_graph_p2.gate.cold_time", cold_time_gate);
  print_bool("decode_graph_p2.gate.memory", memory_gate);
  print_bool("decode_graph_p2.diagnostic.first_distinct_chain_ge_0_70ms",
             first_distinct_chain_gate);
  print_bool("decode_graph_p2.gate.hot_performance",
             hot_performance_gate);
  print_bool("decode_graph_p2.target_100ms_met", target_100ms_met);
  print_bool("decode_graph_p2.gate.pass", passed);
  std::cout << "decode_graph_p2.status="
            << (passed ? "pass" : correctness_gate ? "stop_loss"
                                                   : "correctness_failure")
            << '\n';
  return passed ? 0 : 1;
}
