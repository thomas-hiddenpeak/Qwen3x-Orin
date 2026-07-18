#include "q3x/runtime/reference_engine.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

namespace runtime = q3x::runtime;
namespace detail = q3x::runtime::reference_engine_detail;

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

struct FakeRunner {
  std::vector<std::uint32_t> predictions;
  std::vector<std::uint32_t> inputs;
  std::vector<runtime::ReferenceStepOptions> options;
  std::size_t next_prediction = 0U;
  std::size_t fail_at = static_cast<std::size_t>(-1);
  bool omit_timing = false;
  bool omit_logits = false;
  bool wrong_position = false;
};

runtime::ReferenceStepOutcome fake_step(
    void* const context,
    const std::uint32_t input_token,
    const runtime::ReferenceStepOptions& options) {
  auto& fake = *static_cast<FakeRunner*>(context);
  const std::size_t call = fake.inputs.size();
  fake.inputs.push_back(input_token);
  fake.options.push_back(options);
  runtime::ReferenceStepOutcome outcome;
  if (call == fake.fail_at) {
    outcome.status.error = runtime::ReferenceRunnerError::kPoisoned;
    outcome.status.layer = 7U;
    outcome.status.operation = "fake_step";
    return outcome;
  }
  runtime::ReferenceStepResult step;
  step.position = static_cast<std::uint32_t>(
      fake.wrong_position ? call + 1U : call);
  step.input_token_id = input_token;
  if (!fake.omit_timing) {
    runtime::ReferenceStepTiming timing;
    timing.elapsed_milliseconds = static_cast<double>(call + 1U);
    step.timing.emplace(timing);
  }
  if (options.compute_logits && !fake.omit_logits) {
    runtime::ReferenceStepLogits logits;
    logits.predicted_token_id = fake.predictions.at(fake.next_prediction++);
    logits.chosen_logit = 1.0F;
    logits.max_log_probability = -0.5;
    logits.logsumexp = 1.5;
    step.logits.emplace(logits);
  }
  outcome.value.emplace(std::move(step));
  return outcome;
}

detail::GenerationControlOptions options(const std::uint32_t max_new_tokens,
                                         const std::uint32_t capacity,
                                         const bool trace = false) {
  detail::GenerationControlOptions result;
  result.max_new_tokens = max_new_tokens;
  result.max_sequence_length = capacity;
  result.capture_trace = trace;
  result.stop_token_id = runtime::kQwen36ImEndTokenId;
  return result;
}

void test_prefill_decode_and_stop(TestContext& test) {
  FakeRunner fake;
  fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
  const auto result = detail::run_generation_control(
      {10U, 11U, 12U}, options(8U, 10U, true), &fake, fake_step);
  test.expect(result.ok(), "host generation control succeeds");
  if (!result) {
    return;
  }
  test.expect(result.value->generated_token_ids ==
                      std::vector<std::uint32_t>(
                          {42U, runtime::kQwen36ImEndTokenId}) &&
                  result.value->stop_reason ==
                      runtime::ReferenceStopReason::kImEnd,
              "greedy decode includes im_end and stops immediately");
  test.expect(fake.inputs == std::vector<std::uint32_t>({10U, 11U, 12U, 42U}),
              "runner inputs are sequential prompt then prior prediction");
  test.expect(fake.options.size() == 4U &&
                  !fake.options[0].compute_logits &&
                  !fake.options[1].compute_logits &&
                  fake.options[2].compute_logits &&
                  fake.options[3].compute_logits,
              "only prompt prefixes skip logits and last prompt produces first token");
  bool flags = true;
  for (const auto& step_options : fake.options) {
    flags = flags && step_options.capture_trace &&
            step_options.measure_timing;
  }
  test.expect(flags, "trace and timing flags reach every step");
  test.expect(result.value->steps.size() == 4U &&
                  result.value->timing.prompt_prefill_milliseconds == 6.0 &&
                  result.value->timing.time_to_first_token_milliseconds == 6.0 &&
                  result.value->timing.subsequent_token_milliseconds ==
                      std::vector<double>({4.0}) &&
                  result.value->timing.total_generation_milliseconds == 10.0,
              "TTFT includes all prefill and later token timing is separate");
}

void test_max_tokens_and_first_stop(TestContext& test) {
  FakeRunner fake;
  fake.predictions = {21U, 22U, 23U};
  auto result = detail::run_generation_control(
      {9U}, options(3U, 3U), &fake, fake_step);
  test.expect(result && result.value->generated_token_ids ==
                            std::vector<std::uint32_t>({21U, 22U, 23U}) &&
                  result.value->stop_reason ==
                      runtime::ReferenceStopReason::kMaxNewTokens &&
                  fake.inputs == std::vector<std::uint32_t>({9U, 21U, 22U}),
              "max_new_tokens bounds decode without an extra runner step");

  fake = {};
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  result = detail::run_generation_control(
      {7U, 8U}, options(5U, 6U), &fake, fake_step);
  test.expect(result && result.value->generated_token_ids.size() == 1U &&
                  fake.inputs == std::vector<std::uint32_t>({7U, 8U}),
              "im_end as first prediction performs no decode-after-first step");
}

void test_validation_and_runner_failures(TestContext& test) {
  FakeRunner fake;
  fake.predictions = {1U};
  auto result = detail::run_generation_control(
      {}, options(1U, 1U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kInvalidArgument,
              "empty prompt is rejected");
  result = detail::run_generation_control(
      {1U}, options(0U, 1U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kInvalidArgument,
              "zero max_new_tokens is rejected");
  result = detail::run_generation_control(
      {1U, 2U}, options(3U, 3U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kCapacityExceeded,
              "prompt plus future input steps must fit request capacity");
  result = detail::run_generation_control(
      {static_cast<std::uint32_t>(runtime::kReferenceVocabularySize)},
      options(1U, 1U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kInvalidArgument,
              "out-of-vocabulary prompt token is rejected before callback");

  fake = {};
  fake.fail_at = 0U;
  result = detail::run_generation_control(
      {1U}, options(1U, 1U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kRunnerFailure &&
                  result.runner_status.error == runtime::ReferenceRunnerError::kPoisoned &&
                  result.runner_status.layer == 7U,
              "runner failure preserves structured nested status");

  fake = {};
  fake.predictions = {1U};
  fake.omit_logits = true;
  result = detail::run_generation_control(
      {1U}, options(1U, 1U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kMissingLogits,
              "missing logits on a compute step is rejected");

  fake = {};
  fake.predictions = {1U};
  fake.omit_timing = true;
  result = detail::run_generation_control(
      {1U}, options(1U, 1U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kMissingTiming,
              "missing requested timing is rejected");

  fake = {};
  fake.predictions = {1U};
  fake.wrong_position = true;
  result = detail::run_generation_control(
      {1U}, options(1U, 1U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kUnexpectedStep,
              "non-sequential runner position is rejected");
  test.expect(detail::to_string(detail::GenerationControlError::kMissingLogits) ==
                  "missing_logits",
              "control diagnostic names are stable");
}

void test_generated_text_stop_semantics(TestContext& test) {
  const std::vector<std::uint32_t> stopped = {
      10U, 11U, runtime::kQwen36ImEndTokenId};
  test.expect(detail::generated_text_token_count(
                  stopped, runtime::ReferenceStopReason::kImEnd,
                  runtime::kQwen36ImEndTokenId) == 2U,
              "an observed terminal stop id is excluded from text");
  test.expect(detail::generated_text_token_count(
                  stopped, runtime::ReferenceStopReason::kMaxNewTokens,
                  runtime::kQwen36ImEndTokenId) == stopped.size(),
              "the same id is retained when stop was not observed");
  test.expect(detail::generated_text_token_count(
                  {10U, 11U}, runtime::ReferenceStopReason::kImEnd,
                  runtime::kQwen36ImEndTokenId) == 2U,
              "a mismatched terminal token is never removed");
}

}  // namespace

int main() {
  TestContext test;
  test_prefill_decode_and_stop(test);
  test_max_tokens_and_first_stop(test);
  test_validation_and_runner_failures(test);
  test_generated_text_stop_semantics(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " reference engine control test(s) failed\n";
    return 1;
  }
  std::cout << "All reference engine control tests passed\n";
  return 0;
}
