#include "q3x/runtime/reference_engine.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
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
  std::vector<std::vector<std::uint32_t>> tile_inputs;
  std::vector<runtime::ReferencePrefillTileOptions> tile_options;
  std::size_t next_prediction = 0U;
  std::size_t next_position = 0U;
  std::size_t fail_at = static_cast<std::size_t>(-1);
  std::size_t tile_fail_at = static_cast<std::size_t>(-1);
  bool omit_timing = false;
  bool omit_logits = false;
  bool wrong_position = false;
  bool tile_omit_timing = false;
  bool tile_wrong_count = false;
  bool tile_wrong_position = false;
  bool tile_wrong_token = false;
  bool tile_add_logits = false;
  double tile_elapsed_milliseconds = 10.0;
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
      fake.wrong_position ? fake.next_position + 1U : fake.next_position);
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
  ++fake.next_position;
  outcome.value.emplace(std::move(step));
  return outcome;
}

runtime::ReferencePrefillTileOutcome fake_prefill_tile(
    void* const context, const std::uint32_t* const input_tokens,
    const std::size_t token_count,
    const runtime::ReferencePrefillTileOptions& options) {
  auto& fake = *static_cast<FakeRunner*>(context);
  const std::size_t call = fake.tile_inputs.size();
  fake.tile_inputs.emplace_back(input_tokens, input_tokens + token_count);
  fake.tile_options.push_back(options);

  runtime::ReferencePrefillTileOutcome outcome;
  if (call == fake.tile_fail_at) {
    outcome.status.error = runtime::ReferenceRunnerError::kPoisoned;
    outcome.status.layer = 11U;
    outcome.status.operation = "fake_prefill_tile";
    return outcome;
  }

  runtime::ReferencePrefillTileResult tile;
  tile.step_count = fake.tile_wrong_count ? token_count + 1U : token_count;
  for (std::size_t index = 0U; index < token_count; ++index) {
    runtime::ReferenceStepResult& step = tile.steps[index];
    step.position = static_cast<std::uint32_t>(
        fake.next_position + index +
        ((fake.tile_wrong_position && index == 0U) ? 1U : 0U));
    step.input_token_id =
        input_tokens[index] +
        ((fake.tile_wrong_token && index == 0U) ? 1U : 0U);
    if (fake.tile_add_logits && index == 0U) {
      step.logits.emplace();
    }
  }
  if (!fake.tile_omit_timing) {
    runtime::ReferenceStepTiming timing;
    timing.elapsed_milliseconds = fake.tile_elapsed_milliseconds;
    tile.timing.emplace(timing);
  }
  fake.next_position += token_count;
  outcome.value.emplace(std::move(tile));
  return outcome;
}

detail::GenerationControlOptions options(const std::uint32_t max_new_tokens,
                                         const std::uint32_t capacity,
                                         const bool trace = false,
                                         const std::uint32_t chunk_size = 1U) {
  detail::GenerationControlOptions result;
  result.max_new_tokens = max_new_tokens;
  result.max_sequence_length = capacity;
  result.capture_trace = trace;
  result.prefill_chunk_size = chunk_size;
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

void test_chunked_prefix_tiles(TestContext& test) {
  std::vector<std::uint32_t> prompt;
  for (std::uint32_t token = 100U; token < 119U; ++token) {
    prompt.push_back(token);
  }

  FakeRunner fake;
  fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
  const auto result = detail::run_generation_control(
      prompt, options(2U, 20U, false, 8U), &fake, fake_step,
      fake_prefill_tile);
  test.expect(result.ok(), "chunked prompt-prefix generation succeeds");
  if (!result) {
    return;
  }

  test.expect(fake.tile_inputs.size() == 3U &&
                  fake.tile_inputs[0] ==
                      std::vector<std::uint32_t>(prompt.begin(),
                                                 prompt.begin() + 8) &&
                  fake.tile_inputs[1] ==
                      std::vector<std::uint32_t>(prompt.begin() + 8,
                                                 prompt.begin() + 16) &&
                  fake.tile_inputs[2] ==
                      std::vector<std::uint32_t>(prompt.begin() + 16,
                                                 prompt.begin() + 18),
              "a 19-token prompt tiles only its 18-token prefix as 8+8+2");
  bool measured_tiles = fake.tile_options.size() == 3U;
  for (const auto& tile_options : fake.tile_options) {
    measured_tiles = measured_tiles && tile_options.measure_timing;
  }
  test.expect(measured_tiles,
              "every prefix tile requests one whole-tile timing");
  test.expect(fake.inputs ==
                  std::vector<std::uint32_t>(
                      {118U, 42U}),
              "the final prompt token and decode remain scalar steps");
  test.expect(result.value->steps.size() == 20U &&
                  result.value->steps.front().position == 0U &&
                  result.value->steps[17].position == 17U &&
                  result.value->steps[18].position == 18U &&
                  result.value->steps[19].position == 19U &&
                  !result.value->steps[0].timing.has_value() &&
                  !result.value->steps[17].timing.has_value() &&
                  result.value->steps[18].timing.has_value() &&
                  result.value->steps[19].timing.has_value(),
              "tile metadata expands into continuous per-token transcript steps");
  test.expect(result.value->timing.prompt_prefill_milliseconds == 31.0 &&
                  result.value->timing.time_to_first_token_milliseconds ==
                      31.0 &&
                  result.value->timing.subsequent_token_milliseconds ==
                      std::vector<double>({2.0}) &&
                  result.value->timing.total_generation_milliseconds == 33.0,
              "TTFT sums whole-tile times plus the scalar final-prompt step");

  std::vector<std::uint32_t> c16_prompt;
  for (std::uint32_t token = 200U; token < 221U; ++token) {
    c16_prompt.push_back(token);
  }
  fake = {};
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  const auto c16_result = detail::run_generation_control(
      c16_prompt, options(1U, 21U, false, 16U), &fake, fake_step,
      fake_prefill_tile);
  test.expect(c16_result && fake.tile_inputs.size() == 2U &&
                  fake.tile_inputs[0] ==
                      std::vector<std::uint32_t>(c16_prompt.begin(),
                                                 c16_prompt.begin() + 16) &&
                  fake.tile_inputs[1] ==
                      std::vector<std::uint32_t>(c16_prompt.begin() + 16,
                                                 c16_prompt.begin() + 20) &&
                  fake.inputs == std::vector<std::uint32_t>({220U}),
              "chunk sixteen routes a 20-token prefix as 16+4 before the scalar final prompt token");
}

void test_chunk_fallbacks_and_callback_requirement(TestContext& test) {
  FakeRunner fake;
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  auto result = detail::run_generation_control(
      {7U}, options(1U, 1U, false, 8U), &fake, fake_step);
  test.expect(result && fake.inputs == std::vector<std::uint32_t>({7U}) &&
                  fake.tile_inputs.empty(),
              "a one-token prompt needs no prefix-tile callback");

  fake = {};
  result = detail::run_generation_control(
      {7U, 8U}, options(1U, 2U, false, 8U), &fake, fake_step);
  test.expect(!result &&
                  result.error == detail::GenerationControlError::kInvalidArgument &&
                  fake.inputs.empty(),
              "chunked prefix execution fails closed without a tile callback");

  fake = {};
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  result = detail::run_generation_control(
      {7U, 8U, 9U, 10U}, options(1U, 4U, true, 8U), &fake,
      fake_step, fake_prefill_tile);
  bool traced_steps = fake.options.size() == 4U;
  for (const auto& step_options : fake.options) {
    traced_steps = traced_steps && step_options.capture_trace;
  }
  test.expect(result && fake.tile_inputs.empty() && traced_steps &&
                  fake.inputs ==
                      std::vector<std::uint32_t>({7U, 8U, 9U, 10U}),
              "trace capture forces the exact scalar prefill path");

  fake = {};
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  result = detail::run_generation_control(
      {7U, 8U}, options(1U, 2U, false, 1U), &fake, fake_step,
      fake_prefill_tile);
  test.expect(result && fake.tile_inputs.empty() &&
                  fake.inputs == std::vector<std::uint32_t>({7U, 8U}),
              "chunk size one preserves scalar callback ordering");
}

void test_tile_failures_and_malformed_results(TestContext& test) {
  auto run = [](FakeRunner& fake) {
    return detail::run_generation_control(
        {10U, 11U, 12U}, options(1U, 3U, false, 8U), &fake,
        fake_step, fake_prefill_tile);
  };

  FakeRunner fake;
  fake.tile_fail_at = 0U;
  auto result = run(fake);
  test.expect(!result &&
                  result.error == detail::GenerationControlError::kRunnerFailure &&
                  result.runner_status.error ==
                      runtime::ReferenceRunnerError::kPoisoned &&
                  result.runner_status.layer == 11U,
              "tile failure preserves structured nested runner status");

  fake = {};
  fake.tile_wrong_count = true;
  result = run(fake);
  test.expect(!result &&
                  result.error == detail::GenerationControlError::kUnexpectedStep,
              "wrong tile result count is rejected");

  fake = {};
  fake.tile_wrong_position = true;
  result = run(fake);
  test.expect(!result &&
                  result.error == detail::GenerationControlError::kUnexpectedStep,
              "non-contiguous tile position is rejected");

  fake = {};
  fake.tile_wrong_token = true;
  result = run(fake);
  test.expect(!result &&
                  result.error == detail::GenerationControlError::kUnexpectedStep,
              "mismatched tile input token is rejected");

  fake = {};
  fake.tile_omit_timing = true;
  result = run(fake);
  test.expect(!result &&
                  result.error == detail::GenerationControlError::kMissingTiming,
              "missing whole-tile timing is rejected");

  for (const double bad_elapsed : {
           -1.0, std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity()}) {
    fake = {};
    fake.tile_elapsed_milliseconds = bad_elapsed;
    result = run(fake);
    test.expect(!result &&
                    result.error == detail::GenerationControlError::kUnexpectedStep,
                "non-finite or negative whole-tile timing is rejected");
  }

  fake = {};
  fake.tile_add_logits = true;
  result = run(fake);
  test.expect(!result &&
                  result.error == detail::GenerationControlError::kUnexpectedStep,
              "prefix-tile steps cannot smuggle logits into the transcript");
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
  result = detail::run_generation_control(
      {1U}, options(1U, 1U, false, 0U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kInvalidArgument,
              "zero prefill chunk size is rejected");
  result = detail::run_generation_control(
      {1U}, options(1U, 1U, false, 17U), &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kInvalidArgument,
              "prefill chunk size above fixed capacity is rejected");

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

void test_engine_backend_validation(TestContext& test) {
  runtime::ReferenceEngineOptions engine_options;
  engine_options.projection_backend =
      static_cast<runtime::ProjectionBackend>(0xffU);
  const runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine("unused-model-directory",
                                       engine_options);
  test.expect(!created &&
                  created.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  created.diagnostic.stage == "projection_backend",
              "engine rejects an unknown projection backend before I/O");
}

}  // namespace

int main() {
  TestContext test;
  test_prefill_decode_and_stop(test);
  test_max_tokens_and_first_stop(test);
  test_chunked_prefix_tiles(test);
  test_chunk_fallbacks_and_callback_requirement(test);
  test_tile_failures_and_malformed_results(test);
  test_validation_and_runner_failures(test);
  test_generated_text_stop_semantics(test);
  test_engine_backend_validation(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " reference engine control test(s) failed\n";
    return 1;
  }
  std::cout << "All reference engine control tests passed\n";
  return 0;
}
