#include "q3x/runtime/reference_benchmark.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

using reference_benchmark_detail::DeviceMemoryProbeResult;
using reference_benchmark_detail::DeviceMemorySnapshot;

[[nodiscard]] ReferenceBenchmarkDiagnostic benchmark_diagnostic(
    const ReferenceBenchmarkError code, std::string message) {
  ReferenceBenchmarkDiagnostic result;
  result.code = code;
  result.message = std::move(message);
  return result;
}

[[nodiscard]] ReferenceGenerateResult generate_with_engine(
    void* const context, const std::string_view prompt,
    const ReferenceGenerateOptions& options) {
  return static_cast<ReferenceEngine*>(context)->generate(prompt, options);
}

[[nodiscard]] DeviceMemoryProbeResult probe_cuda_memory(void*) {
  DeviceMemoryProbeResult result;
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (error != cudaSuccess) {
    result.cuda_error = static_cast<int>(error);
    result.message = cudaGetErrorString(error);
    return result;
  }
  DeviceMemorySnapshot snapshot;
  snapshot.free_bytes = static_cast<std::uint64_t>(free_bytes);
  snapshot.total_bytes = static_cast<std::uint64_t>(total_bytes);
  result.value.emplace(snapshot);
  return result;
}

[[nodiscard]] ReferenceBenchmarkStep benchmark_step(
    const ReferenceStepResult& step) {
  ReferenceBenchmarkStep result;
  result.position = step.position;
  result.input_token_id = step.input_token_id;
  if (step.logits.has_value()) {
    result.predicted_token_id = step.logits->predicted_token_id;
  } else if (step.prediction.has_value()) {
    result.predicted_token_id = step.prediction->predicted_token_id;
  }
  return result;
}

[[nodiscard]] ReferenceBenchmarkPromptReport prompt_report(
    const std::string& prompt, const ReferenceGeneration& generation) {
  ReferenceBenchmarkPromptReport result;
  result.prompt = prompt;
  result.prompt_token_ids = generation.prompt_token_ids;
  result.generated_token_ids = generation.generated_token_ids;
  result.generated_text = generation.generated_text;
  result.stop_reason = generation.stop_reason;
  result.decode_graph_replays = generation.decode_graph_replays;
  result.decode_graph_serial_fallbacks =
      generation.decode_graph_serial_fallbacks;
  result.step_sequence.reserve(generation.steps.size());
  for (const ReferenceStepResult& step : generation.steps) {
    result.step_sequence.push_back(benchmark_step(step));
  }
  return result;
}

[[nodiscard]] std::string logits_mode_mismatch_field(
    const ReferenceGeneration& generation,
    const ReferenceLogitsMode mode) {
  if (generation.prompt_token_ids.empty() ||
      generation.steps.size() < generation.prompt_token_ids.size()) {
    return "step_sequence.size";
  }
  const std::size_t first_compute_step =
      generation.prompt_token_ids.size() - 1U;
  for (std::size_t index = 0U; index < generation.steps.size(); ++index) {
    const ReferenceStepResult& step = generation.steps[index];
    const bool compute_logits = index >= first_compute_step;
    const bool has_logits = step.logits.has_value();
    const bool has_prediction = step.prediction.has_value();
    bool valid = !has_logits && !has_prediction;
    if (compute_logits) {
      valid = mode == ReferenceLogitsMode::kFullStatistics
                  ? has_logits && !has_prediction
                  : !has_logits && has_prediction;
    }
    if (!valid) {
      return "step_sequence[" + std::to_string(index) +
             "].logits_mode";
    }
  }
  return {};
}

[[nodiscard]] bool checked_product(const std::size_t lhs,
                                   const std::uint32_t rhs,
                                   std::size_t& result) noexcept {
  if (rhs != 0U &&
      lhs > std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(rhs)) {
    return false;
  }
  result = lhs * static_cast<std::size_t>(rhs);
  return true;
}

[[nodiscard]] bool checked_sum(const std::size_t lhs,
                               const std::size_t rhs,
                               std::size_t& result) noexcept {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

[[nodiscard]] std::string decode_graph_counter_validation_error(
    const ReferenceGeneration& generation) {
  const std::size_t decode_step_count =
      generation.generated_token_ids.empty()
          ? 0U
          : generation.generated_token_ids.size() - 1U;
  std::size_t dispatch_count = 0U;
  if (!checked_sum(generation.decode_graph_replays,
                   generation.decode_graph_serial_fallbacks,
                   dispatch_count)) {
    return "decode_graph_dispatch_count.overflow";
  }
  // A generation selects one Decode callback for its full lifetime. The
  // ordinary serial callback records no graph dispatches; the cache-aware
  // callback records exactly one replay or fallback for every Decode step.
  if (dispatch_count != 0U && dispatch_count != decode_step_count) {
    return "decode_graph_dispatch_count";
  }
  return {};
}

[[nodiscard]] bool valid_latency(const double milliseconds) noexcept {
  return std::isfinite(milliseconds) && milliseconds >= 0.0;
}

[[nodiscard]] bool approximately_equal_latency(const double lhs,
                                                const double rhs) noexcept {
  constexpr double kAbsoluteToleranceMilliseconds = 1.0e-6;
  constexpr double kRelativeTolerance = 1.0e-6;
  const double scale = std::max(std::abs(lhs), std::abs(rhs));
  return std::abs(lhs - rhs) <=
         kAbsoluteToleranceMilliseconds + kRelativeTolerance * scale;
}

// Returns the offending field/invariant, or an empty string for a valid
// decomposition. The prefix total is returned for phase aggregation.
[[nodiscard]] std::string timing_validation_error(
    const ReferenceGeneration& generation,
    double& prompt_prefix_milliseconds) {
  const ReferenceGenerationTiming& timing = generation.timing;
  if (generation.prompt_token_ids.empty() ||
      generation.effective_prefill_chunk_size == 0U ||
      generation.effective_prefill_chunk_size >
          kMaximumRequestPrefillChunkSize) {
    return "prefix_execution_milliseconds.size";
  }
  const std::size_t prefix_token_count =
      generation.prompt_token_ids.size() -
      (generation.all_prompt_tokens_prefilled_by_tiles ? 0U : 1U);
  const std::size_t effective_prefill_chunk_size =
      generation.effective_prefill_chunk_size;
  const std::size_t expected_prefix_execution_count =
      reference_engine_detail::prefix_execution_count(
          prefix_token_count, effective_prefill_chunk_size);
  if (timing.prefix_execution_milliseconds.size() !=
      expected_prefix_execution_count) {
    return "prefix_execution_milliseconds.size";
  }
  if (generation.generated_token_ids.empty() ||
      timing.subsequent_token_milliseconds.size() !=
          generation.generated_token_ids.size() - 1U) {
    return "subsequent_token_milliseconds.size";
  }

  prompt_prefix_milliseconds = 0.0;
  for (std::size_t index = 0U;
       index < timing.prefix_execution_milliseconds.size(); ++index) {
    const double milliseconds = timing.prefix_execution_milliseconds[index];
    if (!valid_latency(milliseconds)) {
      return "prefix_execution_milliseconds[" + std::to_string(index) +
             "]";
    }
    prompt_prefix_milliseconds += milliseconds;
    if (!std::isfinite(prompt_prefix_milliseconds)) {
      return "prefix_execution_milliseconds.sum";
    }
  }
  if (!valid_latency(timing.finish_prefill_milliseconds)) {
    return "finish_prefill_milliseconds";
  }
  if (!valid_latency(timing.prompt_prefill_milliseconds)) {
    return "prompt_prefill_milliseconds";
  }
  if (!valid_latency(timing.time_to_first_token_milliseconds)) {
    return "time_to_first_token_milliseconds";
  }
  double subsequent_token_total = 0.0;
  for (std::size_t index = 0U;
       index < timing.subsequent_token_milliseconds.size(); ++index) {
    const double milliseconds =
        timing.subsequent_token_milliseconds[index];
    if (!valid_latency(milliseconds)) {
      return "subsequent_token_milliseconds[" + std::to_string(index) +
             "]";
    }
    subsequent_token_total += milliseconds;
    if (!std::isfinite(subsequent_token_total)) {
      return "subsequent_token_milliseconds.sum";
    }
  }
  if (!valid_latency(timing.decode_after_first_milliseconds)) {
    return "decode_after_first_milliseconds";
  }
  if (!valid_latency(timing.total_generation_milliseconds)) {
    return "total_generation_milliseconds";
  }

  const double reconstructed_prefill =
      prompt_prefix_milliseconds + timing.finish_prefill_milliseconds;
  if (!std::isfinite(reconstructed_prefill) ||
      !approximately_equal_latency(reconstructed_prefill,
                                   timing.prompt_prefill_milliseconds)) {
    return "prefix_execution_plus_finish_prefill";
  }
  if (!approximately_equal_latency(timing.prompt_prefill_milliseconds,
                                   timing.time_to_first_token_milliseconds)) {
    return "prompt_prefill_equals_time_to_first_token";
  }
  if (!approximately_equal_latency(subsequent_token_total,
                                   timing.decode_after_first_milliseconds)) {
    return "subsequent_token_equals_decode_after_first";
  }
  const double reconstructed_total = timing.prompt_prefill_milliseconds +
                                     timing.decode_after_first_milliseconds;
  if (!std::isfinite(reconstructed_total) ||
      !approximately_equal_latency(reconstructed_total,
                                   timing.total_generation_milliseconds)) {
    return "prompt_prefill_plus_decode_after_first";
  }
  return {};
}

}  // namespace

namespace reference_benchmark_detail {

std::optional<ReferenceLatencyStatistics> compute_latency_statistics(
    const std::vector<double>& milliseconds) {
  ReferenceLatencyStatistics result;
  result.count = milliseconds.size();
  if (milliseconds.empty()) {
    return result;
  }
  for (const double value : milliseconds) {
    if (!std::isfinite(value) || value < 0.0) {
      return std::nullopt;
    }
  }

  std::vector<double> sorted = milliseconds;
  std::sort(sorted.begin(), sorted.end());
  result.minimum_milliseconds = sorted.front();
  result.maximum_milliseconds = sorted.back();
  const std::size_t count = sorted.size();
  if ((count & 1U) != 0U) {
    result.median_milliseconds = sorted[count / 2U];
  } else {
    const double lower = sorted[count / 2U - 1U];
    const double upper = sorted[count / 2U];
    result.median_milliseconds = lower + (upper - lower) / 2.0;
  }
  const std::size_t p95_rank =
      (count / 100U) * 95U + ((count % 100U) * 95U + 99U) / 100U;
  result.p95_milliseconds = sorted[p95_rank - 1U];
  return result;
}

std::string generation_mismatch_field(const ReferenceGeneration& expected,
                                      const ReferenceGeneration& actual) {
  if (expected.prompt_token_ids != actual.prompt_token_ids) {
    return "prompt_token_ids";
  }
  if (expected.generated_token_ids != actual.generated_token_ids) {
    return "generated_token_ids";
  }
  if (expected.generated_text != actual.generated_text) {
    return "generated_text";
  }
  if (expected.stop_reason != actual.stop_reason) {
    return "stop_reason";
  }
  if (expected.requested_prefill_chunk_size !=
      actual.requested_prefill_chunk_size) {
    return "requested_prefill_chunk_size";
  }
  if (expected.effective_prefill_chunk_size !=
      actual.effective_prefill_chunk_size) {
    return "effective_prefill_chunk_size";
  }
  if (expected.all_prompt_tokens_prefilled_by_tiles !=
      actual.all_prompt_tokens_prefilled_by_tiles) {
    return "all_prompt_tokens_prefilled_by_tiles";
  }
  if (expected.decode_graph_replays != actual.decode_graph_replays) {
    return "decode_graph_replays";
  }
  if (expected.decode_graph_serial_fallbacks !=
      actual.decode_graph_serial_fallbacks) {
    return "decode_graph_serial_fallbacks";
  }
  if (expected.steps.size() != actual.steps.size()) {
    return "step_sequence.size";
  }
  for (std::size_t index = 0U; index < expected.steps.size(); ++index) {
    const ReferenceStepResult& expected_step = expected.steps[index];
    const ReferenceStepResult& actual_step = actual.steps[index];
    const std::string prefix = "step_sequence[" + std::to_string(index) + "]";
    if (expected_step.position != actual_step.position) {
      return prefix + ".position";
    }
    if (expected_step.input_token_id != actual_step.input_token_id) {
      return prefix + ".input_token_id";
    }
    if (expected_step.logits.has_value() != actual_step.logits.has_value()) {
      return prefix + ".predicted_token_id.presence";
    }
    if (expected_step.prediction.has_value() !=
        actual_step.prediction.has_value()) {
      return prefix + ".predicted_token_id.presence";
    }
    if (expected_step.logits.has_value() &&
        expected_step.logits->predicted_token_id !=
            actual_step.logits->predicted_token_id) {
      return prefix + ".predicted_token_id";
    }
    if (expected_step.prediction.has_value() &&
        expected_step.prediction->predicted_token_id !=
            actual_step.prediction->predicted_token_id) {
      return prefix + ".predicted_token_id";
    }
  }
  return {};
}

ReferenceBenchmarkResult run_benchmark_control(
    const std::vector<std::string>& prompts,
    const ReferenceBenchmarkOptions& options,
    void* const generate_context,
    const GenerateFunction generate,
    void* const memory_context,
    const DeviceMemoryProbeFunction probe_memory) {
  ReferenceBenchmarkResult result;
  if (generate == nullptr || probe_memory == nullptr || prompts.empty() ||
      options.measured_rounds == 0U || options.max_new_tokens == 0U ||
      options.stop_token_id >= kReferenceVocabularySize ||
      options.prefill_chunk_size == 0U ||
      options.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
      !is_valid_reference_logits_mode(options.logits_mode)) {
    result.diagnostic = benchmark_diagnostic(
        ReferenceBenchmarkError::kInvalidArgument,
        "callbacks and prompts must be present; measured_rounds and "
        "max_new_tokens must be positive; stop_token_id and "
        "prefill_chunk_size must be valid");
    return result;
  }
  for (std::size_t index = 0U; index < prompts.size(); ++index) {
    if (prompts[index].empty()) {
      result.diagnostic = benchmark_diagnostic(
          ReferenceBenchmarkError::kInvalidArgument,
          "benchmark prompts must not be empty");
      result.diagnostic.prompt_index = index;
      return result;
    }
  }

  std::size_t sample_count = 0U;
  if (!checked_product(prompts.size(), options.measured_rounds,
                       sample_count)) {
    result.diagnostic = benchmark_diagnostic(
        ReferenceBenchmarkError::kInvalidArgument,
        "prompt count times measured rounds overflows size_t");
    return result;
  }

  try {
    const DeviceMemoryProbeResult initial_memory =
        probe_memory(memory_context);
    if (!initial_memory) {
      result.diagnostic = benchmark_diagnostic(
          ReferenceBenchmarkError::kDeviceMemoryProbeFailure,
          initial_memory.message.empty() ? "initial cudaMemGetInfo failed"
                                         : initial_memory.message);
      result.diagnostic.cuda_error = initial_memory.cuda_error;
      return result;
    }

    ReferenceBenchmarkReport report;
    report.warmup_rounds = options.warmup_rounds;
    report.measured_rounds = options.measured_rounds;
    report.logits_mode = options.logits_mode;
    report.nvtx_phase_ranges_emitted = options.emit_nvtx_phase_ranges;
    report.max_new_tokens = options.max_new_tokens;
    report.stop_token_id = options.stop_token_id;
    report.prefill_chunk_size = options.prefill_chunk_size;
    report.samples.reserve(sample_count);
    report.device_memory.start_free_bytes = initial_memory.value->free_bytes;
    report.device_memory.end_free_bytes = initial_memory.value->free_bytes;
    report.device_memory.minimum_free_bytes = initial_memory.value->free_bytes;
    report.device_memory.total_bytes = initial_memory.value->total_bytes;
    report.device_memory.drop_tolerance_bytes =
        options.device_memory_drop_tolerance_bytes;

    std::vector<std::optional<ReferenceGeneration>> baselines(prompts.size());
    std::vector<std::vector<double>> prompt_prompt_prefix(prompts.size());
    std::vector<std::vector<double>> prompt_finish_prefill(prompts.size());
    std::vector<std::vector<double>> prompt_prompt_prefill(prompts.size());
    std::vector<std::vector<double>> prompt_decode_after_first(prompts.size());
    std::vector<std::vector<double>> prompt_ttft(prompts.size());
    std::vector<std::vector<double>> prompt_total(prompts.size());
    std::vector<std::vector<double>> prompt_subsequent(prompts.size());
    std::vector<double> all_prompt_prefix;
    std::vector<double> all_finish_prefill;
    std::vector<double> all_prompt_prefill;
    std::vector<double> all_decode_after_first;
    std::vector<double> all_ttft;
    std::vector<double> all_total;
    std::vector<double> all_subsequent;
    all_prompt_prefix.reserve(sample_count);
    all_finish_prefill.reserve(sample_count);
    all_prompt_prefill.reserve(sample_count);
    all_decode_after_first.reserve(sample_count);
    all_ttft.reserve(sample_count);
    all_total.reserve(sample_count);

    ReferenceGenerateOptions generation_options;
    generation_options.max_new_tokens = options.max_new_tokens;
    generation_options.stop_token_id = options.stop_token_id;
    generation_options.capture_trace = false;
    generation_options.prefill_chunk_size = options.prefill_chunk_size;
    generation_options.logits_mode = options.logits_mode;
    generation_options.emit_nvtx_phase_ranges = options.emit_nvtx_phase_ranges;

    auto run_phase = [&](const std::uint32_t rounds,
                         const bool warmup) -> bool {
      for (std::uint32_t round = 0U; round < rounds; ++round) {
        for (std::size_t prompt_index = 0U;
             prompt_index < prompts.size(); ++prompt_index) {
          ReferenceGenerateResult generated = generate(
              generate_context, prompts[prompt_index], generation_options);
          if (!generated) {
            result.diagnostic = benchmark_diagnostic(
                ReferenceBenchmarkError::kGenerationFailure,
                "generation failed during benchmark");
            result.diagnostic.prompt_index = prompt_index;
            result.diagnostic.round = round;
            result.diagnostic.warmup = warmup;
            result.diagnostic.generation = std::move(generated.diagnostic);
            return false;
          }

          const std::string mode_mismatch = logits_mode_mismatch_field(
              *generated.value, options.logits_mode);
          if (!mode_mismatch.empty()) {
            result.diagnostic = benchmark_diagnostic(
                ReferenceBenchmarkError::kGenerationFailure,
                "generation result arms do not match the requested logits "
                "mode");
            result.diagnostic.prompt_index = prompt_index;
            result.diagnostic.round = round;
            result.diagnostic.warmup = warmup;
            result.diagnostic.mismatch_field = mode_mismatch;
            return false;
          }

          const std::string graph_counter_error =
              decode_graph_counter_validation_error(*generated.value);
          if (!graph_counter_error.empty()) {
            result.diagnostic = benchmark_diagnostic(
                ReferenceBenchmarkError::kGenerationFailure,
                "generation returned inconsistent Decode Graph dispatcher "
                "counts");
            result.diagnostic.prompt_index = prompt_index;
            result.diagnostic.round = round;
            result.diagnostic.warmup = warmup;
            result.diagnostic.mismatch_field = graph_counter_error;
            return false;
          }

          double prompt_prefix_milliseconds = 0.0;
          const std::string timing_error = timing_validation_error(
              *generated.value, prompt_prefix_milliseconds);
          if (!timing_error.empty()) {
            result.diagnostic = benchmark_diagnostic(
                ReferenceBenchmarkError::kInvalidTiming,
                "generation returned an invalid or inconsistent timing "
                "decomposition");
            result.diagnostic.prompt_index = prompt_index;
            result.diagnostic.round = round;
            result.diagnostic.warmup = warmup;
            result.diagnostic.mismatch_field = timing_error;
            return false;
          }

          if (!baselines[prompt_index].has_value()) {
            baselines[prompt_index] = *generated.value;
          } else {
            const std::string mismatch = generation_mismatch_field(
                *baselines[prompt_index], *generated.value);
            if (!mismatch.empty()) {
              result.diagnostic = benchmark_diagnostic(
                  ReferenceBenchmarkError::kRepeatabilityFailure,
                  "generation changed across benchmark invocations");
              result.diagnostic.prompt_index = prompt_index;
              result.diagnostic.round = round;
              result.diagnostic.warmup = warmup;
              result.diagnostic.mismatch_field = mismatch;
              return false;
            }
          }

          if (!warmup) {
            std::size_t aggregate_replays = 0U;
            std::size_t aggregate_fallbacks = 0U;
            if (!checked_sum(report.decode_graph_replays,
                             generated.value->decode_graph_replays,
                             aggregate_replays) ||
                !checked_sum(report.decode_graph_serial_fallbacks,
                             generated.value->decode_graph_serial_fallbacks,
                             aggregate_fallbacks)) {
              result.diagnostic = benchmark_diagnostic(
                  ReferenceBenchmarkError::kGenerationFailure,
                  "Decode Graph dispatcher counter aggregate overflowed");
              result.diagnostic.prompt_index = prompt_index;
              result.diagnostic.round = round;
              result.diagnostic.warmup = false;
              result.diagnostic.mismatch_field =
                  "decode_graph_dispatch_count.sum";
              return false;
            }
            ReferenceBenchmarkSample sample;
            sample.prompt_index = prompt_index;
            sample.measured_round = round;
            sample.timing = generated.value->timing;
            sample.decode_graph_replays =
                generated.value->decode_graph_replays;
            sample.decode_graph_serial_fallbacks =
                generated.value->decode_graph_serial_fallbacks;
            report.samples.push_back(std::move(sample));
            report.decode_graph_replays = aggregate_replays;
            report.decode_graph_serial_fallbacks = aggregate_fallbacks;
            const double finish_prefill =
                generated.value->timing.finish_prefill_milliseconds;
            const double prompt_prefill =
                generated.value->timing.prompt_prefill_milliseconds;
            const double decode_after_first =
                generated.value->timing.decode_after_first_milliseconds;
            const double ttft =
                generated.value->timing.time_to_first_token_milliseconds;
            const double total =
                generated.value->timing.total_generation_milliseconds;
            prompt_prompt_prefix[prompt_index].push_back(
                prompt_prefix_milliseconds);
            prompt_finish_prefill[prompt_index].push_back(finish_prefill);
            prompt_prompt_prefill[prompt_index].push_back(prompt_prefill);
            prompt_decode_after_first[prompt_index].push_back(
                decode_after_first);
            prompt_ttft[prompt_index].push_back(ttft);
            prompt_total[prompt_index].push_back(total);
            all_prompt_prefix.push_back(prompt_prefix_milliseconds);
            all_finish_prefill.push_back(finish_prefill);
            all_prompt_prefill.push_back(prompt_prefill);
            all_decode_after_first.push_back(decode_after_first);
            all_ttft.push_back(ttft);
            all_total.push_back(total);
            for (const double latency :
                 generated.value->timing.subsequent_token_milliseconds) {
              prompt_subsequent[prompt_index].push_back(latency);
              all_subsequent.push_back(latency);
            }
          }

          const DeviceMemoryProbeResult memory = probe_memory(memory_context);
          if (!memory) {
            result.diagnostic = benchmark_diagnostic(
                ReferenceBenchmarkError::kDeviceMemoryProbeFailure,
                memory.message.empty() ? "cudaMemGetInfo failed"
                                       : memory.message);
            result.diagnostic.prompt_index = prompt_index;
            result.diagnostic.round = round;
            result.diagnostic.warmup = warmup;
            result.diagnostic.cuda_error = memory.cuda_error;
            return false;
          }
          if (memory.value->total_bytes !=
              report.device_memory.total_bytes) {
            result.diagnostic = benchmark_diagnostic(
                ReferenceBenchmarkError::kDeviceMemoryProbeFailure,
                "CUDA total memory changed during benchmark");
            result.diagnostic.prompt_index = prompt_index;
            result.diagnostic.round = round;
            result.diagnostic.warmup = warmup;
            return false;
          }
          report.device_memory.end_free_bytes = memory.value->free_bytes;
          report.device_memory.minimum_free_bytes = std::min(
              report.device_memory.minimum_free_bytes,
              memory.value->free_bytes);
        }
      }
      return true;
    };

    if (!run_phase(options.warmup_rounds, true) ||
        !run_phase(options.measured_rounds, false)) {
      return result;
    }
    report.all_prompt_tokens_prefilled_by_tiles =
        baselines.front()->all_prompt_tokens_prefilled_by_tiles;

    const auto prompt_prefix_stats =
        compute_latency_statistics(all_prompt_prefix);
    const auto finish_prefill_stats =
        compute_latency_statistics(all_finish_prefill);
    const auto prompt_prefill_stats =
        compute_latency_statistics(all_prompt_prefill);
    const auto decode_after_first_stats =
        compute_latency_statistics(all_decode_after_first);
    const auto ttft_stats = compute_latency_statistics(all_ttft);
    const auto total_stats = compute_latency_statistics(all_total);
    const auto subsequent_stats =
        compute_latency_statistics(all_subsequent);
    if (!prompt_prefix_stats.has_value() ||
        !finish_prefill_stats.has_value() ||
        !prompt_prefill_stats.has_value() ||
        !decode_after_first_stats.has_value() || !ttft_stats.has_value() ||
        !total_stats.has_value() || !subsequent_stats.has_value()) {
      result.diagnostic = benchmark_diagnostic(
          ReferenceBenchmarkError::kInvalidTiming,
          "generation returned a negative or non-finite timing");
      return result;
    }
    report.prompt_prefix = *prompt_prefix_stats;
    report.finish_prefill = *finish_prefill_stats;
    report.prompt_prefill = *prompt_prefill_stats;
    report.decode_after_first = *decode_after_first_stats;
    report.time_to_first_token = *ttft_stats;
    report.total_generation = *total_stats;
    report.subsequent_token = *subsequent_stats;

    report.prompts.reserve(prompts.size());
    for (std::size_t index = 0U; index < prompts.size(); ++index) {
      ReferenceBenchmarkPromptReport prompt =
          prompt_report(prompts[index], *baselines[index]);
      const auto per_prompt_prompt_prefix =
          compute_latency_statistics(prompt_prompt_prefix[index]);
      const auto per_prompt_finish_prefill =
          compute_latency_statistics(prompt_finish_prefill[index]);
      const auto per_prompt_prompt_prefill =
          compute_latency_statistics(prompt_prompt_prefill[index]);
      const auto per_prompt_decode_after_first =
          compute_latency_statistics(prompt_decode_after_first[index]);
      const auto per_prompt_ttft =
          compute_latency_statistics(prompt_ttft[index]);
      const auto per_prompt_total =
          compute_latency_statistics(prompt_total[index]);
      const auto per_prompt_subsequent =
          compute_latency_statistics(prompt_subsequent[index]);
      if (!per_prompt_prompt_prefix.has_value() ||
          !per_prompt_finish_prefill.has_value() ||
          !per_prompt_prompt_prefill.has_value() ||
          !per_prompt_decode_after_first.has_value() ||
          !per_prompt_ttft.has_value() || !per_prompt_total.has_value() ||
          !per_prompt_subsequent.has_value()) {
        result.diagnostic = benchmark_diagnostic(
            ReferenceBenchmarkError::kInvalidTiming,
            "generation returned a negative or non-finite timing");
        result.diagnostic.prompt_index = index;
        return result;
      }
      prompt.prompt_prefix = *per_prompt_prompt_prefix;
      prompt.finish_prefill = *per_prompt_finish_prefill;
      prompt.prompt_prefill = *per_prompt_prompt_prefill;
      prompt.decode_after_first = *per_prompt_decode_after_first;
      prompt.time_to_first_token = *per_prompt_ttft;
      prompt.total_generation = *per_prompt_total;
      prompt.subsequent_token = *per_prompt_subsequent;
      report.prompts.push_back(std::move(prompt));
    }

    report.device_memory.persistent_drop_bytes =
        report.device_memory.start_free_bytes >
                report.device_memory.end_free_bytes
            ? report.device_memory.start_free_bytes -
                  report.device_memory.end_free_bytes
            : 0U;
    report.device_memory.maximum_observed_drop_bytes =
        report.device_memory.start_free_bytes >
                report.device_memory.minimum_free_bytes
            ? report.device_memory.start_free_bytes -
                  report.device_memory.minimum_free_bytes
            : 0U;
    report.device_memory.persistent_drop_detected =
        report.device_memory.persistent_drop_bytes >
        report.device_memory.drop_tolerance_bytes;
    result.value.emplace(std::move(report));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = benchmark_diagnostic(
        ReferenceBenchmarkError::kAllocationFailure,
        "host allocation failed during benchmark");
    return result;
  } catch (const std::length_error& error) {
    result.diagnostic = benchmark_diagnostic(
        ReferenceBenchmarkError::kAllocationFailure, error.what());
    return result;
  } catch (const std::exception& error) {
    result.diagnostic = benchmark_diagnostic(
        ReferenceBenchmarkError::kInvalidArgument, error.what());
    return result;
  }
}

}  // namespace reference_benchmark_detail

ReferenceBenchmarkResult benchmark_reference_engine(
    ReferenceEngine& engine, const std::vector<std::string>& prompts,
    const ReferenceBenchmarkOptions& options) {
  if (!engine) {
    ReferenceBenchmarkResult result;
    result.diagnostic = benchmark_diagnostic(
        ReferenceBenchmarkError::kInvalidArgument,
        "reference engine is empty");
    return result;
  }
  return reference_benchmark_detail::run_benchmark_control(
      prompts, options, &engine, generate_with_engine, nullptr,
      probe_cuda_memory);
}

std::string_view to_string(const ReferenceBenchmarkError error) noexcept {
  switch (error) {
    case ReferenceBenchmarkError::kNone:
      return "none";
    case ReferenceBenchmarkError::kInvalidArgument:
      return "invalid_argument";
    case ReferenceBenchmarkError::kGenerationFailure:
      return "generation_failure";
    case ReferenceBenchmarkError::kRepeatabilityFailure:
      return "repeatability_failure";
    case ReferenceBenchmarkError::kDeviceMemoryProbeFailure:
      return "device_memory_probe_failure";
    case ReferenceBenchmarkError::kInvalidTiming:
      return "invalid_timing";
    case ReferenceBenchmarkError::kAllocationFailure:
      return "allocation_failure";
  }
  return "unknown";
}

}  // namespace q3x::runtime
