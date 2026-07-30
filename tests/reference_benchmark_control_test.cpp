#include "q3x/runtime/reference_benchmark.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace runtime = q3x::runtime;
namespace detail = q3x::runtime::reference_benchmark_detail;

class TestContext {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }
  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

runtime::ReferenceGeneration make_generation(const std::string_view prompt,
                                             const double ttft,
                                             const double total,
                                             std::vector<double> subsequent) {
  const std::uint32_t discriminator = prompt == "alpha" ? 1U : 2U;
  runtime::ReferenceGeneration generation;
  generation.prompt_token_ids = {100U, discriminator};
  if (subsequent.empty()) {
    generation.generated_token_ids = {runtime::kQwen36ImEndTokenId};
  } else {
    generation.generated_token_ids.push_back(200U + discriminator);
    for (std::size_t index = 1U; index < subsequent.size(); ++index) {
      generation.generated_token_ids.push_back(
          300U + discriminator * 16U + static_cast<std::uint32_t>(index));
    }
    generation.generated_token_ids.push_back(runtime::kQwen36ImEndTokenId);
  }
  generation.generated_text = prompt == "alpha" ? "A" : "B";
  generation.stop_reason = runtime::ReferenceStopReason::kImEnd;
  generation.timing.prefix_execution_milliseconds = {ttft / 4.0};
  generation.timing.finish_prefill_milliseconds = ttft * 3.0 / 4.0;
  generation.timing.prompt_prefill_milliseconds = ttft;
  generation.timing.time_to_first_token_milliseconds = ttft;
  generation.timing.subsequent_token_milliseconds = std::move(subsequent);
  generation.timing.decode_after_first_milliseconds = total - ttft;
  generation.timing.total_generation_milliseconds = total;

  runtime::ReferenceStepResult prefix;
  prefix.position = 0U;
  prefix.input_token_id = 100U;
  generation.steps.push_back(prefix);
  runtime::ReferenceStepResult first;
  first.position = 1U;
  first.input_token_id = discriminator;
  runtime::ReferenceStepLogits first_logits;
  first_logits.predicted_token_id = generation.generated_token_ids.front();
  first.logits.emplace(first_logits);
  generation.steps.push_back(first);
  for (std::size_t index = 1U;
       index < generation.generated_token_ids.size(); ++index) {
    runtime::ReferenceStepResult decode;
    decode.position = static_cast<std::uint32_t>(1U + index);
    decode.input_token_id = generation.generated_token_ids[index - 1U];
    runtime::ReferenceStepLogits decode_logits;
    decode_logits.predicted_token_id = generation.generated_token_ids[index];
    decode.logits.emplace(decode_logits);
    generation.steps.push_back(std::move(decode));
  }
  return generation;
}

void replace_prompt_steps(runtime::ReferenceGeneration& generation,
                          const std::size_t prompt_token_count) {
  generation.prompt_token_ids.resize(prompt_token_count);
  generation.steps.clear();
  generation.steps.reserve(prompt_token_count +
                           generation.generated_token_ids.size() - 1U);
  for (std::size_t index = 0U; index < prompt_token_count; ++index) {
    generation.prompt_token_ids[index] =
        static_cast<std::uint32_t>(1'000U + index);
    runtime::ReferenceStepResult step;
    step.position = static_cast<std::uint32_t>(index);
    step.input_token_id = generation.prompt_token_ids[index];
    if (index + 1U == prompt_token_count) {
      runtime::ReferenceStepLogits logits;
      logits.predicted_token_id = generation.generated_token_ids.front();
      step.logits.emplace(logits);
    }
    generation.steps.push_back(std::move(step));
  }
  for (std::size_t index = 1U;
       index < generation.generated_token_ids.size(); ++index) {
    runtime::ReferenceStepResult step;
    step.position =
        static_cast<std::uint32_t>(prompt_token_count + index - 1U);
    step.input_token_id = generation.generated_token_ids[index - 1U];
    runtime::ReferenceStepLogits logits;
    logits.predicted_token_id = generation.generated_token_ids[index];
    step.logits.emplace(logits);
    generation.steps.push_back(std::move(step));
  }
}

struct FakeGenerator {
  std::size_t calls = 0U;
  bool mismatch = false;
  bool fail = false;
  bool invalid_timing = false;
  bool nan_phase_timing = false;
  bool inconsistent_phase_timing = false;
  bool inconsistent_decode_phase_timing = false;
  bool invalid_prefix_cardinality = false;
  bool invalid_subsequent_cardinality = false;
  std::uint32_t expected_prefill_chunk_size = 4U;
  runtime::ReferenceLogitsMode expected_logits_mode =
      runtime::ReferenceLogitsMode::kFullStatistics;
  bool expected_emit_nvtx_phase_ranges = false;
  bool preserve_full_arm = false;
  bool add_both_arms = false;
  bool add_prefix_prediction = false;
  bool c64_partial_prefix = false;
  bool single_arbitrary_prefill = false;
  bool layer_major_prefill = false;
  std::vector<std::string> prompts;
};

runtime::ReferenceGenerateResult fake_generate(
    void* const context, const std::string_view prompt,
    const runtime::ReferenceGenerateOptions& options) {
  auto& fake = *static_cast<FakeGenerator*>(context);
  const std::size_t call = fake.calls++;
  fake.prompts.emplace_back(prompt);
  runtime::ReferenceGenerateResult result;
  if (fake.fail) {
    result.diagnostic.code = runtime::ReferenceEngineError::kRunnerStepFailure;
    result.diagnostic.stage = "fake";
    return result;
  }
  if (options.max_new_tokens != 8U || options.capture_trace ||
      options.prefill_chunk_size != fake.expected_prefill_chunk_size ||
      options.stop_token_id != runtime::kQwen36ImEndTokenId ||
      options.logits_mode != fake.expected_logits_mode ||
      options.emit_nvtx_phase_ranges !=
          fake.expected_emit_nvtx_phase_ranges) {
    result.diagnostic.code = runtime::ReferenceEngineError::kInvalidArgument;
    return result;
  }

  static const double kTtft[] = {1.0, 2.0, 10.0, 20.0, 30.0, 40.0};
  static const double kTotal[] = {2.0, 4.0, 20.0, 40.0, 60.0, 80.0};
  static const std::vector<double> kSubsequent[] = {
      {0.25, 0.25, 0.5}, {0.5, 1.5}, {3.0, 4.0, 3.0},
      {5.0, 15.0}, {6.0, 7.0, 17.0}, {10.0, 30.0}};
  runtime::ReferenceGeneration generation = make_generation(
      prompt, kTtft[call], kTotal[call], kSubsequent[call]);
  if (fake.c64_partial_prefix) {
    constexpr std::size_t kPromptTokens = 128U;
    replace_prompt_steps(generation, kPromptTokens);
    generation.timing.prefix_execution_milliseconds =
        {kTtft[call] / 4.0, kTtft[call] / 4.0,
         kTtft[call] / 4.0};
    generation.timing.finish_prefill_milliseconds = kTtft[call] / 4.0;
  }
  if (fake.single_arbitrary_prefill) {
    constexpr std::size_t kPromptTokens = 407U;
    replace_prompt_steps(generation, kPromptTokens);
    generation.all_prompt_tokens_prefilled_by_tiles = true;
    generation.single_arbitrary_prefill_tiles = true;
  }
  if (fake.layer_major_prefill) {
    constexpr std::size_t kPromptTokens = 4'096U;
    replace_prompt_steps(generation, kPromptTokens);
    generation.all_prompt_tokens_prefilled_by_tiles = true;
    generation.layer_major_prefill = true;
  }
  if (options.logits_mode ==
          runtime::ReferenceLogitsMode::kPredictedTokenOnly &&
      !fake.preserve_full_arm) {
    for (runtime::ReferenceStepResult& step : generation.steps) {
      if (step.logits.has_value()) {
        step.prediction.emplace(runtime::ReferenceStepPrediction{
            step.logits->predicted_token_id});
        step.logits.reset();
      }
    }
  }
  if (fake.add_both_arms) {
    for (runtime::ReferenceStepResult& step : generation.steps) {
      if (step.prediction.has_value()) {
        runtime::ReferenceStepLogits logits;
        logits.predicted_token_id = step.prediction->predicted_token_id;
        step.logits.emplace(logits);
      }
    }
  }
  if (fake.add_prefix_prediction) {
    generation.steps.front().prediction.emplace(
        runtime::ReferenceStepPrediction{999U});
  }
  generation.requested_prefill_chunk_size = options.prefill_chunk_size;
  generation.effective_prefill_chunk_size = options.prefill_chunk_size;
  if (fake.invalid_timing) {
    generation.timing.time_to_first_token_milliseconds = -1.0;
  }
  if (fake.nan_phase_timing) {
    generation.timing.prefix_execution_milliseconds.front() =
        std::numeric_limits<double>::quiet_NaN();
  }
  if (fake.inconsistent_phase_timing) {
    generation.timing.finish_prefill_milliseconds += 1.0;
  }
  if (fake.inconsistent_decode_phase_timing) {
    generation.timing.decode_after_first_milliseconds += 1.0;
  }
  if (fake.invalid_prefix_cardinality) {
    generation.timing.prefix_execution_milliseconds.push_back(0.0);
  }
  if (fake.invalid_subsequent_cardinality) {
    generation.timing.subsequent_token_milliseconds.push_back(0.0);
  }
  if (fake.mismatch && call == 2U) {
    generation.generated_text = "changed";
  }
  result.value.emplace(std::move(generation));
  return result;
}

struct FakeMemory {
  std::vector<std::uint64_t> free_bytes;
  std::size_t calls = 0U;
  std::size_t fail_at = std::numeric_limits<std::size_t>::max();
};

detail::DeviceMemoryProbeResult fake_memory(void* const context) {
  auto& fake = *static_cast<FakeMemory*>(context);
  detail::DeviceMemoryProbeResult result;
  if (fake.calls == fake.fail_at) {
    ++fake.calls;
    result.cuda_error = 17;
    result.message = "fake CUDA failure";
    return result;
  }
  detail::DeviceMemorySnapshot snapshot;
  snapshot.free_bytes = fake.free_bytes.at(fake.calls++);
  snapshot.total_bytes = 2'000U;
  result.value.emplace(snapshot);
  return result;
}

runtime::ReferenceBenchmarkOptions benchmark_options() {
  runtime::ReferenceBenchmarkOptions options;
  options.warmup_rounds = 1U;
  options.measured_rounds = 2U;
  options.max_new_tokens = 8U;
  options.prefill_chunk_size = 4U;
  options.device_memory_drop_tolerance_bytes = 64U;
  return options;
}

FakeMemory benchmark_memory() {
  FakeMemory memory;
  memory.free_bytes = {1'000U, 990U, 980U, 970U, 960U, 950U, 900U};
  return memory;
}

void test_statistics(TestContext& test) {
  const auto odd = detail::compute_latency_statistics({9.0, 1.0, 5.0});
  test.expect(odd.has_value() && odd->count == 3U &&
                  odd->minimum_milliseconds == 1.0 &&
                  odd->median_milliseconds == 5.0 &&
                  odd->p95_milliseconds == 9.0 &&
                  odd->maximum_milliseconds == 9.0,
              "odd statistics use sorted median and nearest-rank p95");
  const auto even =
      detail::compute_latency_statistics({40.0, 10.0, 30.0, 20.0});
  test.expect(even.has_value() && even->median_milliseconds == 25.0 &&
                  even->p95_milliseconds == 40.0,
              "even median averages its two middle values");
  const auto empty = detail::compute_latency_statistics({});
  test.expect(empty.has_value() && empty->count == 0U,
              "an empty optional-token distribution is valid");
  test.expect(!detail::compute_latency_statistics(
                   {-1.0, std::numeric_limits<double>::infinity()})
                   .has_value(),
              "negative and non-finite timings are rejected");
}

void test_control_success(TestContext& test) {
  FakeGenerator generator;
  FakeMemory memory = benchmark_memory();
  const auto result = detail::run_benchmark_control(
      {"alpha", "beta"}, benchmark_options(), &generator, fake_generate,
      &memory, fake_memory);
  test.expect(result.ok(), "host benchmark control succeeds");
  if (!result) {
    return;
  }
  const auto& report = *result.value;
  test.expect(generator.prompts ==
                  std::vector<std::string>({"alpha", "beta", "alpha",
                                            "beta", "alpha", "beta"}),
              "warmup and measured rounds execute in round-major order");
  test.expect(report.samples.size() == 4U &&
                  report.stop_token_id == runtime::kQwen36ImEndTokenId &&
                  report.prefill_chunk_size == 4U &&
                  report.samples[0].prompt_index == 0U &&
                  report.samples[0].measured_round == 0U &&
                  report.samples[3].prompt_index == 1U &&
                  report.samples[3].measured_round == 1U,
              "policy and measured invocations become an ordered report");
  test.expect(report.time_to_first_token.count == 4U &&
                  report.time_to_first_token.minimum_milliseconds == 10.0 &&
                  report.time_to_first_token.median_milliseconds == 25.0 &&
                  report.time_to_first_token.p95_milliseconds == 40.0 &&
                  report.time_to_first_token.maximum_milliseconds == 40.0,
              "aggregate TTFT statistics exclude warmups");
  test.expect(report.prompt_prefix.count == 4U &&
                  report.prompt_prefix.minimum_milliseconds == 2.5 &&
                  report.prompt_prefix.median_milliseconds == 6.25 &&
                  report.prompt_prefix.p95_milliseconds == 10.0 &&
                  report.finish_prefill.median_milliseconds == 18.75 &&
                  report.finish_prefill.p95_milliseconds == 30.0 &&
                  report.prompt_prefill.median_milliseconds == 25.0 &&
                  report.decode_after_first.median_milliseconds == 25.0,
              "aggregate phase statistics use one prefix sum per sample");
  test.expect(report.total_generation.median_milliseconds == 50.0 &&
                  report.total_generation.p95_milliseconds == 80.0 &&
                  report.subsequent_token.count == 10U &&
                  report.subsequent_token.median_milliseconds == 6.5 &&
                  report.subsequent_token.p95_milliseconds == 30.0,
              "total and flattened subsequent-token statistics are exact");
  test.expect(report.prompts.size() == 2U &&
                  report.prompts[0].prompt_prefix.median_milliseconds == 5.0 &&
                  report.prompts[1].finish_prefill.median_milliseconds ==
                      22.5 &&
                  report.prompts[0].prompt_prefill.median_milliseconds ==
                      20.0 &&
                  report.prompts[1]
                          .decode_after_first
                          .median_milliseconds == 30.0 &&
                  report.prompts[0].time_to_first_token.median_milliseconds ==
                      20.0 &&
                  report.prompts[1].time_to_first_token.median_milliseconds ==
                      30.0 &&
                  report.prompts[0].step_sequence.size() == 5U &&
                  report.prompts[0].step_sequence[1].predicted_token_id ==
                      std::optional<std::uint32_t>(201U),
              "per-prompt statistics and replay step signature are retained");
  test.expect(report.samples[0].timing.prefix_execution_milliseconds ==
                      std::vector<double>({2.5}) &&
                  report.samples[0].timing.finish_prefill_milliseconds == 7.5 &&
                  report.samples[0]
                          .timing.prompt_prefill_milliseconds == 10.0 &&
                  report.samples[0]
                          .timing.decode_after_first_milliseconds == 10.0,
              "sample reports retain the complete phase decomposition");
  test.expect(memory.calls == 7U &&
                  report.device_memory.start_free_bytes == 1'000U &&
                  report.device_memory.end_free_bytes == 900U &&
                  report.device_memory.minimum_free_bytes == 900U &&
                  report.device_memory.persistent_drop_bytes == 100U &&
                  report.device_memory.maximum_observed_drop_bytes == 100U &&
                  report.device_memory.persistent_drop_detected,
              "memory is sampled at start and after every invocation");
}

void test_prediction_only_mode(TestContext& test) {
  FakeGenerator generator;
  generator.expected_logits_mode =
      runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  FakeMemory memory = benchmark_memory();
  runtime::ReferenceBenchmarkOptions options = benchmark_options();
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  const auto result = detail::run_benchmark_control(
      {"alpha", "beta"}, options, &generator, fake_generate,
      &memory, fake_memory);
  test.expect(result &&
                  result.value->logits_mode ==
                      runtime::ReferenceLogitsMode::kPredictedTokenOnly &&
                  result.value->prompts[0]
                          .step_sequence[1]
                          .predicted_token_id ==
                      std::optional<std::uint32_t>(201U),
              "benchmark propagates and records prediction-only mode");

  for (int malformed = 0; malformed < 3; ++malformed) {
    generator = {};
    generator.expected_logits_mode =
        runtime::ReferenceLogitsMode::kPredictedTokenOnly;
    generator.preserve_full_arm = malformed == 0;
    generator.add_both_arms = malformed == 1;
    generator.add_prefix_prediction = malformed == 2;
    memory = benchmark_memory();
    const auto rejected = detail::run_benchmark_control(
        {"alpha"}, options, &generator, fake_generate,
        &memory, fake_memory);
    test.expect(!rejected &&
                    rejected.diagnostic.code ==
                        runtime::ReferenceBenchmarkError::
                            kGenerationFailure &&
                    rejected.diagnostic.mismatch_field.find(
                        ".logits_mode") != std::string::npos,
                "benchmark rejects a result arm inconsistent with logits mode");
  }
}

void test_nvtx_phase_option_propagation(TestContext& test) {
  FakeGenerator generator;
  generator.expected_emit_nvtx_phase_ranges = true;
  FakeMemory memory;
  memory.free_bytes = {1'000U, 1'000U};
  runtime::ReferenceBenchmarkOptions options = benchmark_options();
  options.warmup_rounds = 0U;
  options.measured_rounds = 1U;
  options.emit_nvtx_phase_ranges = true;
  const auto result = detail::run_benchmark_control(
      {"alpha"}, options, &generator, fake_generate, &memory, fake_memory);
  test.expect(result && generator.calls == 1U &&
                  result.value->nvtx_phase_ranges_emitted,
              "benchmark propagates explicit NVTX phase range emission");
}

void test_repeatability_and_failures(TestContext& test) {
  FakeGenerator generator;
  generator.mismatch = true;
  FakeMemory memory = benchmark_memory();
  auto result = detail::run_benchmark_control(
      {"alpha", "beta"}, benchmark_options(), &generator, fake_generate,
      &memory, fake_memory);
  test.expect(!result &&
                  result.diagnostic.code ==
                      runtime::ReferenceBenchmarkError::kRepeatabilityFailure &&
                  result.diagnostic.prompt_index == 0U &&
                  result.diagnostic.round == 0U &&
                  !result.diagnostic.warmup &&
                  result.diagnostic.mismatch_field == "generated_text",
              "the first changed deterministic field identifies its sample");

  generator = {};
  generator.fail = true;
  memory = benchmark_memory();
  result = detail::run_benchmark_control(
      {"alpha", "beta"}, benchmark_options(), &generator, fake_generate,
      &memory, fake_memory);
  test.expect(!result &&
                  result.diagnostic.code ==
                      runtime::ReferenceBenchmarkError::kGenerationFailure &&
                  result.diagnostic.generation.code ==
                      runtime::ReferenceEngineError::kRunnerStepFailure,
              "generation failure preserves the engine diagnostic");

  generator = {};
  memory = benchmark_memory();
  memory.fail_at = 2U;
  result = detail::run_benchmark_control(
      {"alpha", "beta"}, benchmark_options(), &generator, fake_generate,
      &memory, fake_memory);
  test.expect(!result &&
                  result.diagnostic.code == runtime::ReferenceBenchmarkError::
                                                kDeviceMemoryProbeFailure &&
                  result.diagnostic.cuda_error == 17,
              "memory probe errors retain their CUDA status");

  generator = {};
  memory = benchmark_memory();
  runtime::ReferenceBenchmarkOptions invalid = benchmark_options();
  invalid.measured_rounds = 0U;
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidArgument,
              "zero measured rounds are rejected before callbacks");

  invalid = benchmark_options();
  invalid.prefill_chunk_size = 0U;
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidArgument,
              "zero prefill chunk size is rejected before callbacks");

  invalid.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize + 1U;
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidArgument,
              "oversized prefill chunk size is rejected before callbacks");

  invalid = benchmark_options();
  invalid.logits_mode =
      static_cast<runtime::ReferenceLogitsMode>(255U);
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidArgument,
              "unknown benchmark logits mode is rejected before callbacks");

  generator = {};
  generator.invalid_timing = true;
  memory.free_bytes = {1'000U, 1'000U};
  memory.calls = 0U;
  invalid = benchmark_options();
  invalid.warmup_rounds = 0U;
  invalid.measured_rounds = 1U;
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidTiming &&
                  result.diagnostic.mismatch_field ==
                      "time_to_first_token_milliseconds",
              "negative measured timing identifies the invalid phase");

  generator = {};
  generator.nan_phase_timing = true;
  memory.free_bytes = {1'000U, 1'000U};
  memory.calls = 0U;
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidTiming &&
                  result.diagnostic.mismatch_field ==
                      "prefix_execution_milliseconds[0]",
              "a NaN phase timing is rejected before aggregation");

  generator = {};
  generator.inconsistent_phase_timing = true;
  memory.calls = 0U;
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidTiming &&
                  result.diagnostic.mismatch_field ==
                      "prefix_execution_plus_finish_prefill",
              "an inconsistent phase decomposition is rejected");

  generator = {};
  generator.inconsistent_decode_phase_timing = true;
  memory.calls = 0U;
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidTiming &&
                  result.diagnostic.mismatch_field ==
                      "subsequent_token_equals_decode_after_first",
              "an inconsistent decode-phase sum is rejected");

  generator = {};
  generator.invalid_prefix_cardinality = true;
  memory.calls = 0U;
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidTiming &&
                  result.diagnostic.mismatch_field ==
                      "prefix_execution_milliseconds.size",
              "a prefix timing count inconsistent with dispatch is rejected");

  generator = {};
  generator.invalid_subsequent_cardinality = true;
  memory.calls = 0U;
  result = detail::run_benchmark_control(
      {"alpha"}, invalid, &generator, fake_generate, &memory, fake_memory);
  test.expect(!result && result.diagnostic.code ==
                             runtime::ReferenceBenchmarkError::kInvalidTiming &&
                  result.diagnostic.mismatch_field ==
                      "subsequent_token_milliseconds.size",
              "a decode timing count inconsistent with tokens is rejected");
}

void test_zero_warmup_and_memory_boundary(TestContext& test) {
  FakeGenerator generator;
  FakeMemory memory;
  memory.free_bytes = {1'000U, 936U};
  runtime::ReferenceBenchmarkOptions options = benchmark_options();
  options.warmup_rounds = 0U;
  options.measured_rounds = 1U;
  options.device_memory_drop_tolerance_bytes = 64U;
  const auto result = detail::run_benchmark_control(
      {"alpha"}, options, &generator, fake_generate, &memory, fake_memory);
  test.expect(result && generator.calls == 1U &&
                  result.value->samples.size() == 1U &&
                  result.value->device_memory.persistent_drop_bytes == 64U &&
                  !result.value->device_memory.persistent_drop_detected,
              "zero warmup succeeds and a drop equal to tolerance is allowed");
}

void test_maximum_prefill_chunk_boundary(TestContext& test) {
  FakeGenerator generator;
  generator.expected_prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  FakeMemory memory;
  memory.free_bytes = {1'000U, 1'000U};
  runtime::ReferenceBenchmarkOptions options = benchmark_options();
  options.warmup_rounds = 0U;
  options.measured_rounds = 1U;
  options.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize;
  const auto result = detail::run_benchmark_control(
      {"alpha"}, options, &generator, fake_generate, &memory, fake_memory);
  test.expect(result && generator.calls == 1U &&
                  result.value->prefill_chunk_size == 512U &&
                  result.value->samples.size() == 1U,
              "C512 is accepted and preserved by benchmark control");
}

void test_c64_partial_prefix_timing_cardinality(TestContext& test) {
  FakeGenerator generator;
  generator.expected_prefill_chunk_size = 64U;
  generator.c64_partial_prefix = true;
  FakeMemory memory;
  memory.free_bytes = {1'000U, 1'000U};
  runtime::ReferenceBenchmarkOptions options = benchmark_options();
  options.warmup_rounds = 0U;
  options.measured_rounds = 1U;
  options.prefill_chunk_size = 64U;
  const auto result = detail::run_benchmark_control(
      {"alpha"}, options, &generator, fake_generate, &memory, fake_memory);
  test.expect(result && result.value->samples.size() == 1U &&
                  result.value->samples.front()
                          .timing.prefix_execution_milliseconds.size() == 3U,
              "C64 benchmark timing accepts the optimized 64+32+31 "
              "execution schedule for a 127-token prefix");

  generator = {};
  generator.expected_prefill_chunk_size = 64U;
  generator.c64_partial_prefix = true;
  generator.invalid_prefix_cardinality = true;
  memory = {};
  memory.free_bytes = {1'000U, 1'000U};
  const auto invalid = detail::run_benchmark_control(
      {"alpha"}, options, &generator, fake_generate, &memory, fake_memory);
  test.expect(!invalid &&
                  invalid.diagnostic.code ==
                      runtime::ReferenceBenchmarkError::kInvalidTiming &&
                  invalid.diagnostic.mismatch_field ==
                      "prefix_execution_milliseconds.size",
              "C64 benchmark timing rejects a cardinality that differs from "
              "the shared controller schedule");
}

void test_single_arbitrary_prefill_timing_cardinality(TestContext& test) {
  FakeGenerator generator;
  generator.expected_prefill_chunk_size = 512U;
  generator.single_arbitrary_prefill = true;
  FakeMemory memory;
  memory.free_bytes = {1'000U, 1'000U};
  runtime::ReferenceBenchmarkOptions options = benchmark_options();
  options.warmup_rounds = 0U;
  options.measured_rounds = 1U;
  options.prefill_chunk_size = 512U;
  const auto result = detail::run_benchmark_control(
      {"alpha"}, options, &generator, fake_generate, &memory, fake_memory);
  test.expect(
      result && result.value->samples.size() == 1U &&
          result.value->all_prompt_tokens_prefilled_by_tiles &&
          result.value->single_arbitrary_prefill_tiles &&
          result.value->samples.front()
                  .timing.prefix_execution_milliseconds.size() == 1U,
      "benchmark timing accepts one arbitrary P407 layer-major tile");

  generator = {};
  generator.expected_prefill_chunk_size = 512U;
  generator.single_arbitrary_prefill = true;
  generator.invalid_prefix_cardinality = true;
  memory = {};
  memory.free_bytes = {1'000U, 1'000U};
  const auto invalid = detail::run_benchmark_control(
      {"alpha"}, options, &generator, fake_generate, &memory, fake_memory);
  test.expect(!invalid &&
                  invalid.diagnostic.code ==
                      runtime::ReferenceBenchmarkError::kInvalidTiming &&
                  invalid.diagnostic.mismatch_field ==
                      "prefix_execution_milliseconds.size",
              "single-arbitrary benchmark timing rejects canonical or extra "
              "prefix cardinality");
}

void test_layer_major_prefill_timing_cardinality(TestContext& test) {
  FakeGenerator generator;
  generator.expected_prefill_chunk_size = 512U;
  generator.layer_major_prefill = true;
  FakeMemory memory;
  memory.free_bytes = {1'000U, 1'000U};
  runtime::ReferenceBenchmarkOptions options = benchmark_options();
  options.warmup_rounds = 0U;
  options.measured_rounds = 1U;
  options.prefill_chunk_size = 512U;
  const auto result = detail::run_benchmark_control(
      {"alpha"}, options, &generator, fake_generate, &memory, fake_memory);
  test.expect(
      result && result.value->samples.size() == 1U &&
          result.value->all_prompt_tokens_prefilled_by_tiles &&
          result.value->layer_major_prefill &&
          !result.value->single_arbitrary_prefill_tiles &&
          result.value->samples.front()
                  .timing.prefix_execution_milliseconds.size() == 1U,
      "benchmark timing accepts one whole P4096 layer-major invocation");
}

void test_step_comparison(TestContext& test) {
  runtime::ReferenceGeneration expected =
      make_generation("alpha", 1.0, 2.0, {1.0});
  runtime::ReferenceGeneration actual =
      make_generation("alpha", 99.0, 100.0, {88.0});
  test.expect(detail::generation_mismatch_field(expected, actual).empty(),
              "latency differences do not fail deterministic replay");
  actual.steps[1].logits->predicted_token_id = 999U;
  test.expect(detail::generation_mismatch_field(expected, actual) ==
                  "step_sequence[1].predicted_token_id",
              "step prediction changes are localized");
  expected.steps[1].prediction.emplace(
      runtime::ReferenceStepPrediction{201U});
  expected.steps[1].logits.reset();
  actual = expected;
  actual.steps[1].prediction->predicted_token_id = 998U;
  test.expect(detail::generation_mismatch_field(expected, actual) ==
                  "step_sequence[1].predicted_token_id",
              "prediction-only step changes are localized");
  actual = expected;
  actual.effective_prefill_chunk_size = 2U;
  test.expect(detail::generation_mismatch_field(expected, actual) ==
                  "effective_prefill_chunk_size",
              "prefill dispatch policy changes are localized");
  actual = expected;
  actual.all_prompt_tokens_prefilled_by_tiles = true;
  test.expect(detail::generation_mismatch_field(expected, actual) ==
                  "all_prompt_tokens_prefilled_by_tiles",
              "whole-prompt admission policy changes are localized");
  actual = expected;
  actual.single_arbitrary_prefill_tiles = true;
  test.expect(detail::generation_mismatch_field(expected, actual) ==
                  "single_arbitrary_prefill_tiles",
              "single-arbitrary tile policy changes are localized");
  actual = expected;
  actual.layer_major_prefill = true;
  test.expect(detail::generation_mismatch_field(expected, actual) ==
                  "layer_major_prefill",
              "whole-prompt layer-major policy changes are localized");
  test.expect(runtime::to_string(
                  runtime::ReferenceBenchmarkError::kRepeatabilityFailure) ==
                  "repeatability_failure",
              "benchmark diagnostic names are stable");
}

}  // namespace

int main() {
  TestContext test;
  test_statistics(test);
  test_control_success(test);
  test_prediction_only_mode(test);
  test_nvtx_phase_option_propagation(test);
  test_repeatability_and_failures(test);
  test_zero_warmup_and_memory_boundary(test);
  test_maximum_prefill_chunk_boundary(test);
  test_c64_partial_prefix_timing_cardinality(test);
  test_single_arbitrary_prefill_timing_cardinality(test);
  test_layer_major_prefill_timing_cardinality(test);
  test_step_comparison(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " benchmark control assertion(s) failed\n";
    return 1;
  }
  std::cout << "reference benchmark host control tests passed\n";
  return 0;
}
