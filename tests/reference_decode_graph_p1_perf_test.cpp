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

namespace {

namespace runtime = q3x::runtime;

constexpr std::string_view kPrompt =
    u8"用一句话解释 CUDA 是什么。";
constexpr std::uint32_t kExpectedPosition = 19U;
constexpr std::uint32_t kExpectedInputTokenId = 77'517U;
constexpr std::uint32_t kExpectedPrediction = 220U;
constexpr std::uint32_t kAlternateInputTokenId = 220U;
constexpr std::uint32_t kExpectedAlternatePrediction = 52'965U;
constexpr std::size_t kExpectedGraphNodes = 390U;
constexpr std::size_t kExpectedKernelNodes = 389U;
constexpr std::size_t kExpectedMemcpyNodes = 1U;
constexpr std::size_t kExpectedOtherNodes = 0U;
constexpr double kMinimumMedianDeltaMilliseconds = 0.30;
constexpr std::size_t kMinimumWinningRounds =
    runtime::kReferenceDecodeGraphP1ScreenRounds;
constexpr int kSkipReturnCode = 77;

std::string model_directory_from(const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* environment =
      std::getenv("Q3X_DECODE_GRAPH_P1_MODEL_DIR");
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
  std::cerr << "decode graph P1 screen failed: code="
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

double median(std::array<double, runtime::kReferenceDecodeGraphP1ScreenRounds>
                  values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

void print_bool(const std::string_view key, const bool value) {
  std::cout << key << '=' << (value ? 1 : 0) << '\n';
}

}  // namespace

int main(const int argc, char** const argv) {
  const char* const enabled =
      std::getenv("Q3X_RUN_SM87_DECODE_GRAPH_P1_PERF");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "decode_graph_p1.status=skip\n"
              << "SKIP: set Q3X_RUN_SM87_DECODE_GRAPH_P1_PERF=1 to run "
                 "the full-model SM87 CUDA Graph P1 screen\n";
    return kSkipReturnCode;
  }
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_decode_graph_p1_perf_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }

  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "decode_graph_p1.status=error\n";
    std::cerr << "Q3X_RUN_SM87_DECODE_GRAPH_P1_PERF=1 requires MODEL_DIR, "
                 "Q3X_DECODE_GRAPH_P1_MODEL_DIR, or Q3X_E2E_MODEL_DIR\n";
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
    std::cout << "decode_graph_p1.status=error\n";
    std::cerr << "failed to inspect CUDA device: "
              << cudaGetErrorString(cuda_status) << '\n';
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "decode_graph_p1.status=skip\n"
              << "decode_graph_p1.device.sm=" << properties.major
              << properties.minor << '\n'
              << "SKIP: decode graph P1 requires an SM87 device\n";
    return kSkipReturnCode;
  }

  runtime::ReferenceEngineOptions engine_options;
  engine_options.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  engine_options.request_options.max_sequence_length = 64U;
  engine_options.request_options.prefill_chunk_size =
      runtime::kReferenceDecodeGraphScreenPrefillChunkSize;
  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(model_directory, engine_options);
  if (!created) {
    std::cout << "decode_graph_p1.status=error\n";
    print_diagnostic(created.diagnostic);
    return 1;
  }

  runtime::ReferenceDecodeGraphP1ScreenOptions screen_options;
  screen_options.expected_position = kExpectedPosition;
  screen_options.expected_input_token_id = kExpectedInputTokenId;
  screen_options.expected_prediction = kExpectedPrediction;
  screen_options.alternate_input_token_id = kAlternateInputTokenId;
  screen_options.expected_alternate_prediction =
      kExpectedAlternatePrediction;
  screen_options.prefill_chunk_size =
      runtime::kReferenceDecodeGraphScreenPrefillChunkSize;
  runtime::ReferenceDecodeGraphP1ScreenOutcome screened =
      created.value->screen_fixed_position_decode_graph_p1(
          kPrompt, screen_options);
  if (!screened) {
    std::cout << "decode_graph_p1.status=error\n";
    print_diagnostic(screened.diagnostic);
    return 1;
  }

  const runtime::ReferenceDecodeGraphP1ScreenResult& result =
      *screened.value;
  const bool topology_exact =
      result.graph.position == kExpectedPosition &&
      result.graph.input_token_id == kExpectedInputTokenId &&
      result.graph.node_count == kExpectedGraphNodes &&
      result.graph.kernel_node_count == kExpectedKernelNodes &&
      result.graph.memcpy_node_count == kExpectedMemcpyNodes &&
      result.graph.other_node_count == kExpectedOtherNodes;
  const bool primary_prediction_exact =
      result.serial_prediction == kExpectedPrediction &&
      result.graph_prediction == kExpectedPrediction;
  const bool alternate_prediction_exact =
      result.alternate_serial_prediction == kExpectedAlternatePrediction &&
      result.alternate_graph_prediction == kExpectedAlternatePrediction &&
      result.alternate_serial_prediction != result.serial_prediction;
  const bool arena_size_exact =
      result.compared_arena_bytes != 0U &&
      result.compared_arena_bytes ==
          created.value->load_stats().request_arena_bytes;
  const bool correctness_exact =
      primary_prediction_exact && alternate_prediction_exact &&
      result.primary_arena_exact && result.alternate_arena_exact &&
      arena_size_exact;

  std::array<double, runtime::kReferenceDecodeGraphP1ScreenRounds>
      serial_pairs{};
  std::array<double, runtime::kReferenceDecodeGraphP1ScreenRounds>
      graph_pairs{};
  std::array<double, runtime::kReferenceDecodeGraphP1ScreenRounds> deltas{};
  std::size_t winning_rounds = 0U;
  std::cout << std::fixed << std::setprecision(6);
  for (std::size_t round_index = 0U;
       round_index < result.rounds.size(); ++round_index) {
    const runtime::ReferenceDecodeGraphP1RoundTiming& round =
        result.rounds[round_index];
    serial_pairs[round_index] =
        (round.serial_first_milliseconds +
         round.serial_second_milliseconds) /
        2.0;
    graph_pairs[round_index] =
        (round.graph_first_milliseconds +
         round.graph_second_milliseconds) /
        2.0;
    deltas[round_index] =
        serial_pairs[round_index] - graph_pairs[round_index];
    if (deltas[round_index] > 0.0) {
      ++winning_rounds;
    }
    const std::string prefix =
        "decode_graph_p1.round." + std::to_string(round_index) + '.';
    std::cout << prefix << "serial_first_ms="
              << round.serial_first_milliseconds << '\n'
              << prefix << "graph_first_ms="
              << round.graph_first_milliseconds << '\n'
              << prefix << "graph_second_ms="
              << round.graph_second_milliseconds << '\n'
              << prefix << "serial_second_ms="
              << round.serial_second_milliseconds << '\n'
              << prefix << "serial_pair_ms=" << serial_pairs[round_index]
              << '\n'
              << prefix << "graph_pair_ms=" << graph_pairs[round_index]
              << '\n'
              << prefix << "delta_ms=" << deltas[round_index] << '\n';
  }

  const double serial_median = median(serial_pairs);
  const double graph_median = median(graph_pairs);
  const double delta_median = median(deltas);
  const double speedup = graph_median > 0.0
                             ? serial_median / graph_median
                             : 0.0;
  const bool timing_valid =
      std::isfinite(serial_median) && serial_median >= 0.0 &&
      std::isfinite(graph_median) && graph_median >= 0.0 &&
      std::isfinite(delta_median) && std::isfinite(speedup);
  const bool performance_gate =
      timing_valid &&
      delta_median >= kMinimumMedianDeltaMilliseconds &&
      winning_rounds >= kMinimumWinningRounds;
  const bool hard_gate = topology_exact && correctness_exact;
  const bool passed = hard_gate && performance_gate;
  const double graph_tokens_per_second =
      graph_median > 0.0 ? 1'000.0 / graph_median : 0.0;
  const bool target_100ms_met =
      graph_median <= 100.0 && graph_tokens_per_second >= 10.0;

  std::cout << "decode_graph_p1.position=" << result.graph.position << '\n'
            << "decode_graph_p1.input_token_id="
            << result.graph.input_token_id << '\n'
            << "decode_graph_p1.nodes.total=" << result.graph.node_count
            << '\n'
            << "decode_graph_p1.nodes.kernel="
            << result.graph.kernel_node_count << '\n'
            << "decode_graph_p1.nodes.memcpy="
            << result.graph.memcpy_node_count << '\n'
            << "decode_graph_p1.nodes.other="
            << result.graph.other_node_count << '\n'
            << "decode_graph_p1.prediction.serial="
            << result.serial_prediction << '\n'
            << "decode_graph_p1.prediction.graph="
            << result.graph_prediction << '\n'
            << "decode_graph_p1.prediction.alternate_serial="
            << result.alternate_serial_prediction << '\n'
            << "decode_graph_p1.prediction.alternate_graph="
            << result.alternate_graph_prediction << '\n'
            << "decode_graph_p1.arena.compared_bytes="
            << result.compared_arena_bytes << '\n'
            << "decode_graph_p1.serial_median_ms=" << serial_median << '\n'
            << "decode_graph_p1.graph_median_ms=" << graph_median << '\n'
            << "decode_graph_p1.median_delta_ms=" << delta_median << '\n'
            << "decode_graph_p1.median_speedup=" << speedup << '\n'
            << "decode_graph_p1.graph_tokens_per_second="
            << graph_tokens_per_second << '\n'
            << "decode_graph_p1.winning_rounds=" << winning_rounds << '\n'
            << "decode_graph_p1.gate.min_median_delta_ms="
            << kMinimumMedianDeltaMilliseconds << '\n'
            << "decode_graph_p1.gate.min_winning_rounds="
            << kMinimumWinningRounds << '\n';
  print_bool("decode_graph_p1.topology.exact", topology_exact);
  print_bool("decode_graph_p1.prediction.primary_exact",
             primary_prediction_exact);
  print_bool("decode_graph_p1.prediction.alternate_exact",
             alternate_prediction_exact);
  print_bool("decode_graph_p1.arena.primary_exact",
             result.primary_arena_exact);
  print_bool("decode_graph_p1.arena.alternate_exact",
             result.alternate_arena_exact);
  print_bool("decode_graph_p1.arena.size_exact", arena_size_exact);
  print_bool("decode_graph_p1.gate.correctness", hard_gate);
  print_bool("decode_graph_p1.gate.performance", performance_gate);
  print_bool("decode_graph_p1.target_100ms_met", target_100ms_met);
  print_bool("decode_graph_p1.gate.pass", passed);
  std::cout << "decode_graph_p1.status="
            << (passed ? "pass"
                       : hard_gate ? "stop_loss" : "correctness_failure")
            << '\n';
  return passed ? 0 : 1;
}
