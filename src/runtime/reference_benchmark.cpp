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
  result.step_sequence.reserve(generation.steps.size());
  for (const ReferenceStepResult& step : generation.steps) {
    result.step_sequence.push_back(benchmark_step(step));
  }
  return result;
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
    if (expected_step.logits.has_value() &&
        expected_step.logits->predicted_token_id !=
            actual_step.logits->predicted_token_id) {
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
      options.stop_token_id >= kReferenceVocabularySize) {
    result.diagnostic = benchmark_diagnostic(
        ReferenceBenchmarkError::kInvalidArgument,
        "callbacks and prompts must be present; measured_rounds and "
        "max_new_tokens must be positive; stop_token_id must be valid");
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
    report.max_new_tokens = options.max_new_tokens;
    report.stop_token_id = options.stop_token_id;
    report.samples.reserve(sample_count);
    report.device_memory.start_free_bytes = initial_memory.value->free_bytes;
    report.device_memory.end_free_bytes = initial_memory.value->free_bytes;
    report.device_memory.minimum_free_bytes = initial_memory.value->free_bytes;
    report.device_memory.total_bytes = initial_memory.value->total_bytes;
    report.device_memory.drop_tolerance_bytes =
        options.device_memory_drop_tolerance_bytes;

    std::vector<std::optional<ReferenceGeneration>> baselines(prompts.size());
    std::vector<std::vector<double>> prompt_ttft(prompts.size());
    std::vector<std::vector<double>> prompt_total(prompts.size());
    std::vector<std::vector<double>> prompt_subsequent(prompts.size());
    std::vector<double> all_ttft;
    std::vector<double> all_total;
    std::vector<double> all_subsequent;
    all_ttft.reserve(sample_count);
    all_total.reserve(sample_count);

    ReferenceGenerateOptions generation_options;
    generation_options.max_new_tokens = options.max_new_tokens;
    generation_options.stop_token_id = options.stop_token_id;
    generation_options.capture_trace = false;

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
            ReferenceBenchmarkSample sample;
            sample.prompt_index = prompt_index;
            sample.measured_round = round;
            sample.timing = generated.value->timing;
            report.samples.push_back(std::move(sample));
            const double ttft =
                generated.value->timing.time_to_first_token_milliseconds;
            const double total =
                generated.value->timing.total_generation_milliseconds;
            prompt_ttft[prompt_index].push_back(ttft);
            prompt_total[prompt_index].push_back(total);
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

    const auto ttft_stats = compute_latency_statistics(all_ttft);
    const auto total_stats = compute_latency_statistics(all_total);
    const auto subsequent_stats =
        compute_latency_statistics(all_subsequent);
    if (!ttft_stats.has_value() || !total_stats.has_value() ||
        !subsequent_stats.has_value()) {
      result.diagnostic = benchmark_diagnostic(
          ReferenceBenchmarkError::kInvalidTiming,
          "generation returned a negative or non-finite timing");
      return result;
    }
    report.time_to_first_token = *ttft_stats;
    report.total_generation = *total_stats;
    report.subsequent_token = *subsequent_stats;

    report.prompts.reserve(prompts.size());
    for (std::size_t index = 0U; index < prompts.size(); ++index) {
      ReferenceBenchmarkPromptReport prompt =
          prompt_report(prompts[index], *baselines[index]);
      const auto per_prompt_ttft =
          compute_latency_statistics(prompt_ttft[index]);
      const auto per_prompt_total =
          compute_latency_statistics(prompt_total[index]);
      const auto per_prompt_subsequent =
          compute_latency_statistics(prompt_subsequent[index]);
      if (!per_prompt_ttft.has_value() || !per_prompt_total.has_value() ||
          !per_prompt_subsequent.has_value()) {
        result.diagnostic = benchmark_diagnostic(
            ReferenceBenchmarkError::kInvalidTiming,
            "generation returned a negative or non-finite timing");
        result.diagnostic.prompt_index = index;
        return result;
      }
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
