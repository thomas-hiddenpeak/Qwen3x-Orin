#include "q3x/runtime/reference_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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

bool has_consistent_prefill_timing(
    const runtime::ReferenceGenerationTiming& timing) {
  double prefix_sum = 0.0;
  for (const double elapsed : timing.prefix_execution_milliseconds) {
    prefix_sum += elapsed;
  }
  return prefix_sum + timing.finish_prefill_milliseconds +
                 timing.commit_prefill_milliseconds ==
             timing.prompt_prefill_milliseconds &&
         timing.prompt_prefill_milliseconds ==
             timing.time_to_first_token_milliseconds;
}

enum class PhaseCall : std::uint8_t {
  kPrefixStep,
  kPrefixTile,
  kWholeRequestPrefill,
  kFinishPrefill,
  kFinishPrefillFromTile,
  kFinishWholeRequestFromUncommittedRetained,
  kWholeRequestCommit,
  kDecodeStep,
};

enum class PromptResultMutation : std::uint8_t {
  kNone,
  kRunnerFailure,
  kMissingTiming,
  kWrongLogicalPanelCount,
  kMissingPanel,
  kWrongPanelOffset,
  kWrongPanelEnd,
  kWrongStepPosition,
  kWrongStepToken,
  kStepTiming,
  kStepLogits,
  kStepPrediction,
  kProgressMissingLayer,
  kProgressFinalHiddenNotReady,
  kProgressAlreadyCommitted,
  kProgressWrongDomain,
  kProgressWrongCursor,
  kThrowException,
};

struct FakeRunner {
  std::vector<std::uint32_t> predictions;
  std::vector<std::uint32_t> inputs;
  std::vector<runtime::ReferenceStepOptions> options;
  std::vector<std::vector<std::uint32_t>> tile_inputs;
  std::vector<runtime::ReferencePrefillTileOptions> tile_options;
  std::vector<PhaseCall> phase_calls;
  std::size_t next_prediction = 0U;
  std::size_t next_position = 0U;
  std::size_t fail_at = static_cast<std::size_t>(-1);
  std::size_t tile_fail_at = static_cast<std::size_t>(-1);
  bool omit_timing = false;
  bool override_timing = false;
  double timing_milliseconds = 0.0;
  bool omit_logits = false;
  bool force_full_logits_arm = false;
  bool force_prediction_arm = false;
  bool add_opposite_result_arm = false;
  bool wrong_position = false;
  bool tile_omit_timing = false;
  bool tile_wrong_count = false;
  bool tile_wrong_position = false;
  bool tile_wrong_token = false;
  bool tile_add_logits = false;
  double tile_elapsed_milliseconds = 10.0;
  std::size_t prompt_call_count = 0U;
  std::size_t prompt_token_count = 0U;
  std::size_t prompt_panel_count = 0U;
  std::size_t prompt_tail_token_count = 0U;
  bool prompt_topology_was_unbound = false;
  std::vector<detail::PrefillPromptOptions> prompt_options;
  PromptResultMutation prompt_result_mutation = PromptResultMutation::kNone;
  double prompt_elapsed_milliseconds = 25.0;
  std::uint32_t whole_retained_position = 0U;
  std::uint32_t whole_retained_token = 0U;
  std::size_t whole_finalizer_call_count = 0U;
  bool whole_finalizer_fail = false;
  bool whole_finalizer_throw = false;
  bool whole_finalizer_advance_state = false;
  double whole_finalizer_elapsed_milliseconds = 1.0;
  std::size_t prompt_commit_call_count = 0U;
  std::size_t prompt_commit_success_count = 0U;
  bool prompt_commit_force_failure = false;
};

struct PhaseContext {
  FakeRunner* runner = nullptr;
};

struct CommittedTokenRecorder {
  std::array<std::uint32_t, 8U> token_ids{};
  std::array<std::size_t, 8U> indices{};
  std::array<double, 8U> elapsed_milliseconds{};
  std::size_t count = 0U;
  std::size_t cancel_after = static_cast<std::size_t>(-1);
};

bool record_committed_token(void* const context,
                            const std::uint32_t token_id,
                            const std::size_t token_index,
                            const double elapsed_milliseconds) noexcept {
  auto& recorder = *static_cast<CommittedTokenRecorder*>(context);
  if (recorder.count >= recorder.token_ids.size()) {
    return false;
  }
  recorder.token_ids[recorder.count] = token_id;
  recorder.indices[recorder.count] = token_index;
  recorder.elapsed_milliseconds[recorder.count] = elapsed_milliseconds;
  ++recorder.count;
  return recorder.count < recorder.cancel_after;
}

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
    timing.elapsed_milliseconds =
        fake.override_timing ? fake.timing_milliseconds
                             : static_cast<double>(call + 1U);
    step.timing.emplace(timing);
  }
  if (options.compute_logits && !fake.omit_logits) {
    const std::uint32_t predicted =
        fake.predictions.at(fake.next_prediction++);
    const auto add_full_logits = [&]() {
      runtime::ReferenceStepLogits logits;
      logits.predicted_token_id = predicted;
      logits.chosen_logit = 1.0F;
      logits.max_log_probability = -0.5;
      logits.logsumexp = 1.5;
      step.logits.emplace(logits);
    };
    const bool use_prediction_arm =
        fake.force_prediction_arm ||
        (options.logits_mode ==
             runtime::ReferenceLogitsMode::kPredictedTokenOnly &&
         !fake.force_full_logits_arm);
    if (use_prediction_arm) {
      step.prediction.emplace(runtime::ReferenceStepPrediction{predicted});
    } else {
      add_full_logits();
    }
    if (fake.add_opposite_result_arm) {
      if (!step.logits.has_value()) {
        add_full_logits();
      }
      if (!step.prediction.has_value()) {
        step.prediction.emplace(runtime::ReferenceStepPrediction{predicted});
      }
    }
  }
  ++fake.next_position;
  outcome.value.emplace(std::move(step));
  return outcome;
}

runtime::ReferenceStepOutcome fake_prefix_step(
    void* const context,
    const std::uint32_t input_token,
    const runtime::ReferenceStepOptions& options) {
  FakeRunner& fake = *static_cast<PhaseContext*>(context)->runner;
  fake.phase_calls.push_back(PhaseCall::kPrefixStep);
  return fake_step(&fake, input_token, options);
}

runtime::ReferenceStepOutcome fake_finish_prefill(
    void* const context,
    const std::uint32_t input_token,
    const runtime::ReferenceStepOptions& options) {
  FakeRunner& fake = *static_cast<PhaseContext*>(context)->runner;
  fake.phase_calls.push_back(PhaseCall::kFinishPrefill);
  return fake_step(&fake, input_token, options);
}

runtime::ReferenceStepOutcome fake_decode_step(
    void* const context,
    const std::uint32_t input_token,
    const runtime::ReferenceStepOptions& options) {
  FakeRunner& fake = *static_cast<PhaseContext*>(context)->runner;
  fake.phase_calls.push_back(PhaseCall::kDecodeStep);
  return fake_step(&fake, input_token, options);
}

runtime::ReferenceStepOutcome fake_finish_prefill_from_tile(
    void* const context,
    const std::uint32_t input_token,
    const runtime::ReferenceStepOptions& options) {
  FakeRunner& fake = *static_cast<PhaseContext*>(context)->runner;
  fake.phase_calls.push_back(PhaseCall::kFinishPrefillFromTile);
  runtime::ReferenceStepOutcome outcome;
  if (fake.next_position == 0U) {
    outcome.status.error = runtime::ReferenceRunnerError::kInvalidStepOptions;
    outcome.status.operation = "fake_retained_prefill_position";
    return outcome;
  }
  // The real callback returns logits for the final position already committed
  // by the tile and never advances request state. Reuse the scalar result
  // builder at position P-1 while preserving next_position == P.
  --fake.next_position;
  return fake_step(&fake, input_token, options);
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

runtime::ReferencePrefillTileOutcome fake_prefix_tile(
    void* const context,
    const std::uint32_t* const input_tokens,
    const std::size_t token_count,
    const runtime::ReferencePrefillTileOptions& options) {
  FakeRunner& fake = *static_cast<PhaseContext*>(context)->runner;
  fake.phase_calls.push_back(PhaseCall::kPrefixTile);
  return fake_prefill_tile(&fake, input_tokens, token_count, options);
}

detail::PrefillPromptOutcome fake_whole_request_prefill(
    void* const context,
    const std::uint32_t* const input_tokens,
    const std::size_t token_count,
    const runtime::PrefillExecutionPlan& unbound_immutable_topology,
    const detail::PrefillPromptOptions& options) {
  FakeRunner& fake = *static_cast<PhaseContext*>(context)->runner;
  fake.phase_calls.push_back(PhaseCall::kWholeRequestPrefill);
  ++fake.prompt_call_count;
  fake.prompt_token_count = token_count;
  fake.prompt_panel_count = unbound_immutable_topology.panel_count;
  fake.prompt_tail_token_count =
      unbound_immutable_topology
          .panels[unbound_immutable_topology.panel_count - 1U]
          .token_count;
  fake.prompt_topology_was_unbound =
      !unbound_immutable_topology.executable() &&
      !unbound_immutable_topology.operator_bindings_complete;
  fake.prompt_options.push_back(options);

  detail::PrefillPromptOutcome outcome;
  if (fake.prompt_result_mutation ==
      PromptResultMutation::kRunnerFailure) {
    outcome.status.error = runtime::ReferenceRunnerError::kPoisoned;
    outcome.status.layer = 23U;
    outcome.status.operation = "fake_whole_request_prefill";
    return outcome;
  }
  if (fake.prompt_result_mutation == PromptResultMutation::kThrowException) {
    throw std::runtime_error("fake whole-request callback exception");
  }

  detail::PrefillPromptResult prompt;
  prompt.logical_panel_count = unbound_immutable_topology.panel_count;
  prompt.prompt_token_count = token_count;
  prompt.progress = runtime::make_prefill_execution_progress(
      unbound_immutable_topology);
  fake.whole_retained_position =
      unbound_immutable_topology.final_position - 1U;
  fake.whole_retained_token = input_tokens[token_count - 1U];
  prompt.panels.reserve(unbound_immutable_topology.panel_count);
  for (std::size_t panel_index = 0U;
       panel_index < unbound_immutable_topology.panel_count;
       ++panel_index) {
    const runtime::PrefillOperatorPanel& topology_panel =
        unbound_immutable_topology.panels[panel_index];
    detail::PrefillPromptPanelResult panel;
    panel.logical_panel_ordinal = panel_index;
    panel.prompt_token_offset = topology_panel.first_position;
    panel.first_position = topology_panel.first_position;
    panel.end_position = topology_panel.end_position;
    panel.steps.reserve(topology_panel.token_count);
    for (std::size_t panel_step = 0U;
         panel_step < topology_panel.token_count; ++panel_step) {
      runtime::ReferenceStepResult step;
      step.position = static_cast<std::uint32_t>(
          topology_panel.first_position + panel_step);
      step.input_token_id =
          input_tokens[panel.prompt_token_offset + panel_step];
      panel.steps.emplace_back(std::move(step));
    }
    prompt.panels.emplace_back(std::move(panel));
  }
  for (std::size_t layer_index = 0U;
       layer_index < unbound_immutable_topology.layers.size();
       ++layer_index) {
    for (std::size_t panel_index = 0U;
         panel_index < unbound_immutable_topology.panel_count;
         ++panel_index) {
      if (runtime::advance_prefill_progress_after_completion(
              unbound_immutable_topology, prompt.progress, layer_index,
              panel_index) !=
          runtime::PrefillExecutionProgressError::kNone) {
        outcome.status.error =
            runtime::ReferenceRunnerError::kStateCommitFailure;
        outcome.status.operation = "fake_whole_request_progress";
        return outcome;
      }
    }
  }
  if (runtime::mark_prefill_final_hidden_ready(
          unbound_immutable_topology, prompt.progress) !=
      runtime::PrefillExecutionProgressError::kNone) {
    outcome.status.error = runtime::ReferenceRunnerError::kStateCommitFailure;
    outcome.status.operation = "fake_whole_request_final_hidden";
    return outcome;
  }
  if (fake.prompt_result_mutation !=
      PromptResultMutation::kMissingTiming) {
    runtime::ReferenceStepTiming timing;
    timing.elapsed_milliseconds = fake.prompt_elapsed_milliseconds;
    prompt.timing.emplace(timing);
  }
  switch (fake.prompt_result_mutation) {
    case PromptResultMutation::kNone:
    case PromptResultMutation::kRunnerFailure:
    case PromptResultMutation::kMissingTiming:
      break;
    case PromptResultMutation::kWrongLogicalPanelCount:
      ++prompt.logical_panel_count;
      break;
    case PromptResultMutation::kMissingPanel:
      prompt.panels.pop_back();
      break;
    case PromptResultMutation::kWrongPanelOffset:
      ++prompt.panels.back().prompt_token_offset;
      break;
    case PromptResultMutation::kWrongPanelEnd:
      ++prompt.panels.back().end_position;
      break;
    case PromptResultMutation::kWrongStepPosition:
      ++prompt.panels.back().steps.back().position;
      break;
    case PromptResultMutation::kWrongStepToken:
      ++prompt.panels.back().steps.back().input_token_id;
      break;
    case PromptResultMutation::kStepTiming:
      prompt.panels.back().steps.back().timing.emplace();
      break;
    case PromptResultMutation::kStepLogits:
      prompt.panels.back().steps.back().logits.emplace();
      break;
    case PromptResultMutation::kStepPrediction:
      prompt.panels.back().steps.back().prediction.emplace();
      break;
    case PromptResultMutation::kProgressMissingLayer:
      --prompt.progress.completed_panels.back();
      break;
    case PromptResultMutation::kProgressFinalHiddenNotReady:
      prompt.progress.final_hidden_ready = false;
      break;
    case PromptResultMutation::kProgressAlreadyCommitted:
      prompt.progress.prefill_state_committed = true;
      break;
    case PromptResultMutation::kProgressWrongDomain:
      prompt.progress.gdn_advanced_end[3U] =
          unbound_immutable_topology.final_position;
      break;
    case PromptResultMutation::kProgressWrongCursor:
      --prompt.progress.next_layer;
      prompt.progress.next_panel =
          unbound_immutable_topology.panel_count - 1U;
      break;
    case PromptResultMutation::kThrowException:
      break;
  }

  outcome.value.emplace(std::move(prompt));
  return outcome;
}

runtime::ReferenceStepOutcome
fake_finish_whole_request_from_uncommitted_retained(
    void* const context, const std::uint32_t input_token,
    const runtime::ReferenceStepOptions& options) {
  FakeRunner& fake = *static_cast<PhaseContext*>(context)->runner;
  fake.phase_calls.push_back(
      PhaseCall::kFinishWholeRequestFromUncommittedRetained);
  ++fake.whole_finalizer_call_count;
  fake.inputs.push_back(input_token);
  fake.options.push_back(options);

  runtime::ReferenceStepOutcome outcome;
  if (fake.whole_finalizer_fail) {
    outcome.status.error = runtime::ReferenceRunnerError::kPoisoned;
    outcome.status.operation = "fake_whole_request_finalizer";
    return outcome;
  }
  if (fake.whole_finalizer_throw) {
    throw std::runtime_error("fake whole-request finalizer exception");
  }

  runtime::ReferenceStepResult step;
  step.position = fake.whole_retained_position;
  step.input_token_id = input_token;
  runtime::ReferenceStepTiming timing;
  timing.elapsed_milliseconds = fake.whole_finalizer_elapsed_milliseconds;
  step.timing.emplace(timing);
  if (options.compute_logits) {
    const std::uint32_t predicted =
        fake.predictions.at(fake.next_prediction++);
    if (options.logits_mode ==
        runtime::ReferenceLogitsMode::kPredictedTokenOnly) {
      step.prediction.emplace(runtime::ReferenceStepPrediction{predicted});
    } else {
      runtime::ReferenceStepLogits logits;
      logits.predicted_token_id = predicted;
      logits.chosen_logit = 1.0F;
      logits.max_log_probability = -0.5;
      logits.logsumexp = 1.5;
      step.logits.emplace(logits);
    }
  }
  if (fake.whole_finalizer_advance_state) {
    ++fake.next_position;
  }
  outcome.value.emplace(std::move(step));
  return outcome;
}

runtime::ReferenceRunnerStatus fake_commit_whole_request(
    void* const context,
    const runtime::PrefillExecutionPlan& unbound_immutable_topology,
    const runtime::PrefillExecutionProgress&
        completed_uncommitted_progress) noexcept {
  FakeRunner& fake = *static_cast<PhaseContext*>(context)->runner;
  fake.phase_calls.push_back(PhaseCall::kWholeRequestCommit);
  ++fake.prompt_commit_call_count;

  runtime::ReferenceRunnerStatus status;
  if (fake.prompt_commit_force_failure ||
      fake.next_position != unbound_immutable_topology.first_position ||
      completed_uncommitted_progress.prefill_state_committed ||
      !runtime::prefill_final_commit_ready(
          unbound_immutable_topology, completed_uncommitted_progress)) {
    status.error = runtime::ReferenceRunnerError::kStateCommitFailure;
    status.operation = "fake_whole_request_commit";
    return status;
  }

  const runtime::PrefillFinalCommitPlan& expected =
      unbound_immutable_topology.final_commit;
  if (expected.expected_initial_sequence_length != fake.next_position ||
      expected.committed_sequence_length !=
          unbound_immutable_topology.final_position ||
      expected.commit_count != 1U) {
    status.error = runtime::ReferenceRunnerError::kStateCommitFailure;
    status.operation = "fake_whole_request_commit_plan";
    return status;
  }
  fake.next_position = expected.committed_sequence_length;
  ++fake.prompt_commit_success_count;
  return status;
}

detail::GenerationControlOptions options(const std::uint32_t max_new_tokens,
                                         const std::uint32_t capacity,
                                         const bool trace = false,
                                         const std::uint32_t chunk_size = 1U,
                                         const runtime::ReferenceLogitsMode
                                             logits_mode = runtime::
                                                 ReferenceLogitsMode::
                                                     kFullStatistics) {
  detail::GenerationControlOptions result;
  result.max_new_tokens = max_new_tokens;
  result.max_sequence_length = capacity;
  result.capture_trace = trace;
  result.prefill_chunk_size = chunk_size;
  result.stop_token_id = runtime::kQwen36ImEndTokenId;
  result.logits_mode = logits_mode;
  return result;
}

void test_prediction_only_control(TestContext& test) {
  FakeRunner fake;
  fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
  const auto prediction_options = options(
      2U, 3U, false, 1U,
      runtime::ReferenceLogitsMode::kPredictedTokenOnly);
  auto result = detail::run_generation_control(
      {10U, 11U}, prediction_options, &fake, fake_step);
  bool mode_propagated = fake.options.size() == 3U;
  for (const runtime::ReferenceStepOptions& step_options : fake.options) {
    mode_propagated =
        mode_propagated &&
        step_options.logits_mode ==
            runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  }
  test.expect(result && mode_propagated &&
                  result.value->generated_token_ids ==
                      std::vector<std::uint32_t>(
                          {42U, runtime::kQwen36ImEndTokenId}) &&
                  !result.value->steps[0].logits.has_value() &&
                  !result.value->steps[0].prediction.has_value() &&
                  !result.value->steps[1].logits.has_value() &&
                  result.value->steps[1].prediction.has_value() &&
                  result.value->steps[1].prediction->predicted_token_id ==
                      42U,
              "prediction-only control propagates mode and preserves result arms");

  fake = {};
  fake.predictions = {42U};
  fake.force_full_logits_arm = true;
  result = detail::run_generation_control(
      {10U}, options(1U, 1U, false, 1U,
                     runtime::ReferenceLogitsMode::kPredictedTokenOnly),
      &fake, fake_step);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kUnexpectedStep,
              "prediction-only control rejects a full-statistics result arm");

  fake = {};
  fake.predictions = {42U};
  fake.omit_logits = true;
  result = detail::run_generation_control(
      {10U}, options(1U, 1U, false, 1U,
                     runtime::ReferenceLogitsMode::kPredictedTokenOnly),
      &fake, fake_step);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kMissingPrediction &&
                  detail::to_string(result.error) == "missing_prediction",
              "prediction-only control rejects a missing prediction arm");

  fake = {};
  fake.predictions = {
      static_cast<std::uint32_t>(runtime::kReferenceVocabularySize)};
  result = detail::run_generation_control(
      {10U}, options(1U, 1U, false, 1U,
                     runtime::ReferenceLogitsMode::kPredictedTokenOnly),
      &fake, fake_step);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kUnexpectedStep,
              "prediction-only control rejects an out-of-vocabulary token");
}

void test_result_arm_validation(TestContext& test) {
  FakeRunner fake;
  fake.predictions = {42U};
  fake.force_prediction_arm = true;
  auto result = detail::run_generation_control(
      {10U}, options(1U, 1U), &fake, fake_step);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kUnexpectedStep,
              "full-statistics control rejects a prediction-only result arm");

  fake = {};
  fake.predictions = {42U};
  fake.add_opposite_result_arm = true;
  result = detail::run_generation_control(
      {10U}, options(1U, 1U), &fake, fake_step);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kUnexpectedStep,
              "full-statistics control rejects simultaneous result arms");

  fake = {};
  fake.predictions = {42U};
  fake.add_opposite_result_arm = true;
  result = detail::run_generation_control(
      {10U}, options(1U, 1U, false, 1U,
                     runtime::ReferenceLogitsMode::kPredictedTokenOnly),
      &fake, fake_step);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kUnexpectedStep,
              "prediction-only control rejects simultaneous result arms");
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
                  result.value->timing.prefix_execution_milliseconds ==
                      std::vector<double>({1.0, 2.0}) &&
                  result.value->timing.finish_prefill_milliseconds == 3.0 &&
                  result.value->timing.prompt_prefill_milliseconds == 6.0 &&
                  result.value->timing.time_to_first_token_milliseconds == 6.0 &&
                  result.value->timing.subsequent_token_milliseconds ==
                      std::vector<double>({4.0}) &&
                  result.value->timing.decode_after_first_milliseconds == 4.0 &&
                  has_consistent_prefill_timing(result.value->timing) &&
                  result.value->timing.total_generation_milliseconds == 10.0,
              "TTFT includes all prefill and later token timing is separate");
}

void test_explicit_phase_plans(TestContext& test) {
  FakeRunner fake;
  fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
  PhaseContext prefill_context{&fake};
  PhaseContext decode_context{&fake};

  detail::PrefillPlan prefill_plan;
  prefill_plan.context = &prefill_context;
  prefill_plan.prefix_step = fake_prefix_step;
  prefill_plan.finish_prefill = fake_finish_prefill;
  prefill_plan.prefix_tile = fake_prefix_tile;

  detail::DecodePlan decode_plan;
  decode_plan.context = &decode_context;
  decode_plan.decode_step = fake_decode_step;

  auto result = detail::run_generation_control(
      {10U, 11U, 12U}, options(2U, 4U), prefill_plan, decode_plan);
  test.expect(result &&
                  fake.phase_calls ==
                      std::vector<PhaseCall>(
                          {PhaseCall::kPrefixStep,
                           PhaseCall::kPrefixStep,
                           PhaseCall::kFinishPrefill,
                           PhaseCall::kDecodeStep}) &&
                  fake.inputs ==
                      std::vector<std::uint32_t>({10U, 11U, 12U, 42U}),
              "explicit plans route prefix, final prompt, and decode steps in order");

  std::vector<std::uint32_t> canonical_prompt;
  for (std::uint32_t token = 100U; token < 119U; ++token) {
    canonical_prompt.push_back(token);
  }

  fake = {};
  fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
  result = detail::run_generation_control(
      canonical_prompt, options(2U, 20U, false, 8U),
      prefill_plan, decode_plan);
  test.expect(result &&
                  fake.phase_calls ==
                      std::vector<PhaseCall>(
                          {PhaseCall::kPrefixTile,
                           PhaseCall::kPrefixTile,
                           PhaseCall::kPrefixTile,
                           PhaseCall::kFinishPrefill,
                           PhaseCall::kDecodeStep}) &&
                  fake.tile_inputs.size() == 3U &&
                  fake.tile_inputs[0] ==
                      std::vector<std::uint32_t>(canonical_prompt.begin(),
                                                 canonical_prompt.begin() + 8) &&
                  fake.tile_inputs[1] ==
                      std::vector<std::uint32_t>(canonical_prompt.begin() + 8,
                                                 canonical_prompt.begin() + 16) &&
                  fake.tile_inputs[2] ==
                      std::vector<std::uint32_t>(canonical_prompt.begin() + 16,
                                                 canonical_prompt.begin() + 18) &&
                  fake.inputs == std::vector<std::uint32_t>({118U, 42U}),
              "C8 plan routes the canonical 18-token prefix as 8+8+2 "
              "before finish and decode");

  fake = {};
  fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
  result = detail::run_generation_control(
      canonical_prompt, options(2U, 20U, false, 16U),
      prefill_plan, decode_plan);
  test.expect(result &&
                  fake.phase_calls ==
                      std::vector<PhaseCall>(
                          {PhaseCall::kPrefixTile,
                           PhaseCall::kPrefixTile,
                           PhaseCall::kFinishPrefill,
                           PhaseCall::kDecodeStep}) &&
                  fake.tile_inputs.size() == 2U &&
                  fake.tile_inputs[0] ==
                      std::vector<std::uint32_t>(canonical_prompt.begin(),
                                                 canonical_prompt.begin() + 16) &&
                  fake.tile_inputs[1] ==
                      std::vector<std::uint32_t>(canonical_prompt.begin() + 16,
                                                 canonical_prompt.begin() + 18) &&
                  fake.inputs == std::vector<std::uint32_t>({118U, 42U}) &&
                  result.value->timing.prefix_execution_milliseconds ==
                      std::vector<double>({10.0, 10.0}) &&
                  result.value->timing.finish_prefill_milliseconds == 1.0 &&
                  result.value->timing.prompt_prefill_milliseconds == 21.0 &&
                  has_consistent_prefill_timing(result.value->timing) &&
                  result.value->timing.subsequent_token_milliseconds ==
                      std::vector<double>({2.0}),
              "C16 plan routes 16+2, final prompt TTFT, and subsequent "
              "decode timing to separate phases");

  fake = {};
  fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
  result = detail::run_generation_control(
      canonical_prompt, options(2U, 20U, false, 32U),
      prefill_plan, decode_plan);
  test.expect(result &&
                  fake.phase_calls ==
                      std::vector<PhaseCall>(
                          {PhaseCall::kPrefixTile,
                           PhaseCall::kFinishPrefill,
                           PhaseCall::kDecodeStep}) &&
                  fake.tile_inputs.size() == 1U &&
                  fake.tile_inputs[0] ==
                      std::vector<std::uint32_t>(canonical_prompt.begin(),
                                                 canonical_prompt.begin() + 18) &&
                  fake.inputs == std::vector<std::uint32_t>({118U, 42U}) &&
                  result.value->timing.prefix_execution_milliseconds ==
                      std::vector<double>({10.0}) &&
                  result.value->timing.finish_prefill_milliseconds == 1.0 &&
                  result.value->timing.prompt_prefill_milliseconds == 11.0 &&
                  has_consistent_prefill_timing(result.value->timing) &&
                  result.value->timing.subsequent_token_milliseconds ==
                      std::vector<double>({2.0}),
              "C32 plan routes the 18-token prefix as one tile while keeping "
              "finish and decode in separate phases");

  fake = {};
  fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
  result = detail::run_generation_control(
      canonical_prompt, options(2U, 20U, true, 32U),
      prefill_plan, decode_plan);
  const bool trace_route =
      fake.phase_calls.size() == 20U &&
      std::all_of(fake.phase_calls.begin(),
                  fake.phase_calls.begin() + 18,
                  [](const PhaseCall call) {
                    return call == PhaseCall::kPrefixStep;
                  }) &&
      fake.phase_calls[18] == PhaseCall::kFinishPrefill &&
      fake.phase_calls[19] == PhaseCall::kDecodeStep;
  test.expect(result && trace_route && fake.tile_inputs.empty(),
              "trace capture keeps explicit phase routing while forcing "
              "the prefix to C1");

  detail::PrefillPlan invalid_prefill = prefill_plan;
  invalid_prefill.finish_prefill = nullptr;
  fake = {};
  result = detail::run_generation_control(
      {1U}, options(1U, 1U), invalid_prefill, decode_plan);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kInvalidArgument &&
                  fake.inputs.empty(),
              "an incomplete prefill plan fails before phase execution");

  detail::DecodePlan invalid_decode = decode_plan;
  invalid_decode.decode_step = nullptr;
  result = detail::run_generation_control(
      {1U}, options(1U, 1U), prefill_plan, invalid_decode);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kInvalidArgument &&
                  fake.inputs.empty(),
              "an incomplete decode plan fails before phase execution");
}

void test_all_prompt_tile_admission(TestContext& test) {
  const auto run_shape = [&test](
                             const std::size_t prompt_size,
                             const std::vector<std::size_t>& expected_tiles,
                             const runtime::ReferenceLogitsMode logits_mode) {
    FakeRunner fake;
    fake.predictions = {runtime::kQwen36ImEndTokenId};
    PhaseContext context{&fake};

    detail::PrefillPlan prefill_plan;
    prefill_plan.context = &context;
    prefill_plan.prefix_step = fake_prefix_step;
    prefill_plan.finish_prefill = fake_finish_prefill;
    prefill_plan.prefix_tile = fake_prefix_tile;
    prefill_plan.finish_prefill_from_tile =
        fake_finish_prefill_from_tile;
    detail::DecodePlan decode_plan;
    decode_plan.context = &context;
    decode_plan.decode_step = fake_decode_step;

    std::vector<std::uint32_t> prompt(prompt_size);
    for (std::size_t index = 0U; index < prompt.size(); ++index) {
      prompt[index] = static_cast<std::uint32_t>(1'000U + index);
    }
    detail::GenerationControlOptions control_options =
        options(1U, static_cast<std::uint32_t>(prompt_size), false, 512U,
                logits_mode);
    control_options.prefill_all_prompt_tokens = true;
    const auto result = detail::run_generation_control(
        prompt, control_options, prefill_plan, decode_plan);

    std::vector<std::size_t> actual_tiles;
    actual_tiles.reserve(fake.tile_inputs.size());
    for (const auto& tile : fake.tile_inputs) {
      actual_tiles.push_back(tile.size());
    }
    bool only_final_tile_retains =
        fake.tile_options.size() == expected_tiles.size() &&
        !fake.tile_options.empty();
    for (std::size_t index = 0U; index < fake.tile_options.size(); ++index) {
      only_final_tile_retains =
          only_final_tile_retains &&
          fake.tile_options[index].retain_last_hidden_for_logits ==
              (index + 1U == fake.tile_options.size());
    }
    bool phase_order =
        fake.phase_calls.size() == expected_tiles.size() + 1U &&
        !fake.phase_calls.empty() &&
        fake.phase_calls.back() == PhaseCall::kFinishPrefillFromTile;
    for (std::size_t index = 0U; index < expected_tiles.size(); ++index) {
      phase_order = phase_order &&
                    fake.phase_calls[index] == PhaseCall::kPrefixTile;
    }
    const bool correct_result_arm =
        result &&
        (logits_mode == runtime::ReferenceLogitsMode::kFullStatistics
             ? result.value->steps.back().logits.has_value() &&
                   !result.value->steps.back().prediction.has_value()
             : !result.value->steps.back().logits.has_value() &&
                   result.value->steps.back().prediction.has_value());
    test.expect(
        result && actual_tiles == expected_tiles && only_final_tile_retains &&
            phase_order && fake.next_position == prompt_size &&
            fake.inputs == std::vector<std::uint32_t>({prompt.back()}) &&
            result.value->steps.size() == prompt_size &&
            result.value->steps.back().position == prompt_size - 1U &&
            result.value->steps.back().input_token_id == prompt.back() &&
            correct_result_arm &&
            result.value->timing.prefix_execution_milliseconds.size() ==
                expected_tiles.size() &&
            result.value->timing.finish_prefill_milliseconds == 1.0 &&
            has_consistent_prefill_timing(result.value->timing),
        "all-prompt admission tiles every prompt token and finalizes the "
        "already-committed last step without advancing state");
  };

  constexpr auto kFull = runtime::ReferenceLogitsMode::kFullStatistics;
  constexpr auto kPredicted =
      runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  run_shape(32U, {32U}, kFull);
  run_shape(64U, {64U}, kPredicted);
  run_shape(256U, {256U}, kPredicted);
  run_shape(407U, {256U, 64U, 64U, 23U}, kPredicted);
  run_shape(481U, {256U, 64U, 64U, 64U, 32U, 1U}, kPredicted);
  run_shape(512U, {512U}, kPredicted);
  run_shape(513U, {512U, 1U}, kPredicted);
  run_shape(564U, {512U, 32U, 20U}, kPredicted);
  run_shape(695U, {512U, 64U, 64U, 32U, 23U}, kPredicted);
  run_shape(713U, {512U, 64U, 64U, 64U, 9U}, kPredicted);
  run_shape(1'025U, {512U, 512U, 1U}, kPredicted);

  FakeRunner fake;
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  PhaseContext context{&fake};
  detail::PrefillPlan incomplete_plan;
  incomplete_plan.context = &context;
  incomplete_plan.prefix_step = fake_prefix_step;
  incomplete_plan.finish_prefill = fake_finish_prefill;
  incomplete_plan.prefix_tile = fake_prefix_tile;
  detail::DecodePlan decode_plan;
  decode_plan.context = &context;
  decode_plan.decode_step = fake_decode_step;
  detail::GenerationControlOptions admitted =
      options(1U, 32U, false, 32U);
  admitted.prefill_all_prompt_tokens = true;
  auto result = detail::run_generation_control(
      std::vector<std::uint32_t>(32U, 7U), admitted, incomplete_plan,
      decode_plan);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kInvalidArgument,
              "all-prompt admission fails closed without a retained-hidden "
              "finalizer");

  incomplete_plan.finish_prefill_from_tile =
      fake_finish_prefill_from_tile;
  admitted.capture_trace = true;
  result = detail::run_generation_control(
      std::vector<std::uint32_t>(32U, 7U), admitted, incomplete_plan,
      decode_plan);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kInvalidArgument,
              "all-prompt admission rejects incompatible trace capture");
}

void test_whole_request_layer_major_admission(TestContext& test) {
  const auto run_shape = [&test](
                             const std::size_t prompt_size,
                             const std::size_t expected_panel_count,
                             const std::size_t expected_tail_tokens,
                             const runtime::ReferenceLogitsMode logits_mode,
                             const std::uint32_t max_new_tokens) {
    FakeRunner fake;
    fake.predictions = max_new_tokens > 1U
                           ? std::vector<std::uint32_t>(
                                 {42U, runtime::kQwen36ImEndTokenId})
                           : std::vector<std::uint32_t>(
                                 {runtime::kQwen36ImEndTokenId});
    PhaseContext context{&fake};

    detail::PrefillPlan prefill_plan;
    prefill_plan.context = &context;
    prefill_plan.prefix_step = fake_prefix_step;
    prefill_plan.finish_prefill = fake_finish_prefill;
    // Keep the legacy tile callback present so the test detects accidental
    // dispatch rather than merely proving that a null callback was tolerated.
    prefill_plan.prefix_tile = fake_prefix_tile;
    prefill_plan.finish_prefill_from_tile =
        fake_finish_prefill_from_tile;
    prefill_plan.whole_request = fake_whole_request_prefill;
    prefill_plan.finish_whole_request_from_uncommitted_retained =
        fake_finish_whole_request_from_uncommitted_retained;
    prefill_plan.commit_whole_request = fake_commit_whole_request;
    detail::DecodePlan decode_plan;
    decode_plan.context = &context;
    decode_plan.decode_step = fake_decode_step;

    std::vector<std::uint32_t> prompt(prompt_size);
    for (std::size_t index = 0U; index < prompt.size(); ++index) {
      prompt[index] = static_cast<std::uint32_t>(1'000U + index);
    }
    detail::GenerationControlOptions control_options = options(
        max_new_tokens,
        static_cast<std::uint32_t>(prompt_size + max_new_tokens - 1U), false,
        512U, logits_mode);
    control_options.prefill_all_prompt_tokens = true;
    control_options.prefill_whole_request_layer_major = true;
    const detail::GenerationControlResult result =
        detail::run_generation_control(prompt, control_options, prefill_plan,
                                       decode_plan);

    bool transcript_exact =
        result && result.value->steps.size() ==
                      prompt.size() + max_new_tokens - 1U;
    if (result) {
      for (std::size_t index = 0U; index < prompt.size(); ++index) {
        const runtime::ReferenceStepResult& step =
            result.value->steps[index];
        const bool final_step = index + 1U == prompt.size();
        transcript_exact =
            transcript_exact && step.position == index &&
            step.input_token_id == prompt[index] &&
            step.timing.has_value() == final_step &&
            (final_step
                 ? (logits_mode ==
                            runtime::ReferenceLogitsMode::kFullStatistics
                        ? step.logits.has_value() &&
                              !step.prediction.has_value()
                        : !step.logits.has_value() &&
                              step.prediction.has_value())
                 : !step.logits.has_value() &&
                       !step.prediction.has_value());
      }
    }
    const bool decode_exact =
        max_new_tokens == 1U ||
        (result && result.value->steps.back().position == prompt.size() &&
         result.value->steps.back().input_token_id == 42U &&
         result.value->steps.back().timing.has_value() &&
         result.value->generated_token_ids ==
             std::vector<std::uint32_t>(
                 {42U, runtime::kQwen36ImEndTokenId}));

    const bool callback_contract =
        fake.prompt_call_count == 1U &&
        fake.prompt_token_count == prompt_size &&
        fake.prompt_panel_count == expected_panel_count &&
        fake.prompt_tail_token_count == expected_tail_tokens &&
        fake.prompt_topology_was_unbound &&
        fake.prompt_options.size() == 1U &&
        fake.prompt_options[0].measure_timing &&
        fake.prompt_options[0].retain_last_hidden_for_logits &&
        fake.whole_finalizer_call_count == 1U &&
        fake.prompt_commit_call_count == 1U &&
        fake.prompt_commit_success_count == 1U;
    std::vector<PhaseCall> expected_phases{
        PhaseCall::kWholeRequestPrefill,
        PhaseCall::kFinishWholeRequestFromUncommittedRetained,
        PhaseCall::kWholeRequestCommit};
    std::vector<std::uint32_t> expected_scalar_inputs{prompt.back()};
    if (max_new_tokens > 1U) {
      expected_phases.push_back(PhaseCall::kDecodeStep);
      expected_scalar_inputs.push_back(42U);
    }
    const bool phase_contract =
        fake.tile_inputs.empty() && fake.tile_options.empty() &&
        fake.phase_calls == expected_phases &&
        fake.inputs == expected_scalar_inputs &&
        fake.next_position == prompt_size + max_new_tokens - 1U;
    const bool one_aggregate_prompt_timing =
        result &&
        result.value->prefill_execution_mode ==
            runtime::ReferencePrefillExecutionMode::
                kWholeRequestLayerMajor &&
        result.value->prefill_logical_panel_count == expected_panel_count &&
        result.value->timing.prefix_execution_milliseconds ==
            std::vector<double>({fake.prompt_elapsed_milliseconds}) &&
        result.value->timing.finish_prefill_milliseconds ==
            fake.whole_finalizer_elapsed_milliseconds &&
        std::isfinite(
            result.value->timing.commit_prefill_milliseconds) &&
        result.value->timing.commit_prefill_milliseconds >= 0.0 &&
        result.value->timing.prompt_prefill_milliseconds ==
            fake.prompt_elapsed_milliseconds +
                fake.whole_finalizer_elapsed_milliseconds +
                result.value->timing.commit_prefill_milliseconds &&
        has_consistent_prefill_timing(result.value->timing);

    test.expect(
        result && transcript_exact && decode_exact && callback_contract &&
            phase_contract && one_aggregate_prompt_timing,
        "whole-request layer-major admission invokes one unbound-topology "
        "callback and replaces the final placeholder without advancing "
        "request state twice");
  };

  run_shape(32U, 1U, 32U,
            runtime::ReferenceLogitsMode::kFullStatistics, 1U);
  run_shape(513U, 1U, 513U,
            runtime::ReferenceLogitsMode::kPredictedTokenOnly, 1U);
  run_shape(8'193U, 2U, 4'096U,
            runtime::ReferenceLogitsMode::kPredictedTokenOnly, 1U);
  run_shape(40'000U, 5U, 7'712U,
            runtime::ReferenceLogitsMode::kFullStatistics, 1U);
  run_shape(60'000U, 8U, 5'424U,
            runtime::ReferenceLogitsMode::kPredictedTokenOnly, 2U);
  run_shape(130'000U, 16U, 7'656U,
            runtime::ReferenceLogitsMode::kPredictedTokenOnly, 1U);
}

void test_whole_request_layer_major_fail_closed(TestContext& test) {
  const auto malformed_result_is_rejected =
      [](const PromptResultMutation mutation,
         const detail::GenerationControlError expected_error) {
        FakeRunner fake;
        fake.prompt_result_mutation = mutation;
        PhaseContext context{&fake};
        detail::PrefillPlan prefill_plan;
        prefill_plan.context = &context;
        prefill_plan.prefix_step = fake_prefix_step;
        prefill_plan.finish_prefill = fake_finish_prefill;
        prefill_plan.prefix_tile = fake_prefix_tile;
        prefill_plan.finish_prefill_from_tile =
            fake_finish_prefill_from_tile;
        prefill_plan.whole_request = fake_whole_request_prefill;
        prefill_plan.finish_whole_request_from_uncommitted_retained =
            fake_finish_whole_request_from_uncommitted_retained;
        prefill_plan.commit_whole_request = fake_commit_whole_request;
        detail::DecodePlan decode_plan;
        decode_plan.context = &context;
        decode_plan.decode_step = fake_decode_step;
        detail::GenerationControlOptions control_options =
            options(1U, 8'193U, false, 512U);
        control_options.prefill_all_prompt_tokens = true;
        control_options.prefill_whole_request_layer_major = true;
        std::vector<std::uint32_t> prompt(8'193U);
        for (std::size_t index = 0U; index < prompt.size(); ++index) {
          prompt[index] = static_cast<std::uint32_t>(3'000U + index);
        }
        const detail::GenerationControlResult result =
            detail::run_generation_control(prompt, control_options,
                                           prefill_plan, decode_plan);
        const bool status_exact = [&]() {
          if (mutation != PromptResultMutation::kRunnerFailure &&
              mutation != PromptResultMutation::kThrowException) {
            return true;
          }
          const std::string_view expected_operation =
              mutation == PromptResultMutation::kRunnerFailure
                  ? "fake_whole_request_prefill"
                  : "whole_request_callback_exception";
          return result.runner_status.error ==
                     runtime::ReferenceRunnerError::kPoisoned &&
                 result.runner_status.operation != nullptr &&
                 std::string_view(result.runner_status.operation) ==
                     expected_operation;
        }();
        return !result && result.error == expected_error && status_exact &&
               fake.prompt_call_count == 1U && fake.tile_inputs.empty() &&
               fake.inputs.empty() &&
               fake.whole_finalizer_call_count == 0U &&
               fake.prompt_commit_call_count == 0U &&
               fake.prompt_commit_success_count == 0U &&
               fake.next_position == 0U;
      };

  const std::array malformed_cases{
      std::pair{PromptResultMutation::kMissingTiming,
                detail::GenerationControlError::kMissingTiming},
      std::pair{PromptResultMutation::kWrongLogicalPanelCount,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kMissingPanel,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kWrongPanelOffset,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kWrongPanelEnd,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kWrongStepPosition,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kWrongStepToken,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kStepTiming,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kStepLogits,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kStepPrediction,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kProgressMissingLayer,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kProgressFinalHiddenNotReady,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kProgressAlreadyCommitted,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kProgressWrongDomain,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kProgressWrongCursor,
                detail::GenerationControlError::kUnexpectedStep},
      std::pair{PromptResultMutation::kRunnerFailure,
                detail::GenerationControlError::kRunnerFailure},
      std::pair{PromptResultMutation::kThrowException,
                detail::GenerationControlError::kRunnerFailure},
  };
  bool malformed_cases_rejected = true;
  for (const auto& [mutation, expected_error] : malformed_cases) {
    malformed_cases_rejected =
        malformed_cases_rejected &&
        malformed_result_is_rejected(mutation, expected_error);
  }
  test.expect(
      malformed_cases_rejected,
      "whole-request results fail closed on callback errors, aggregate "
      "timing, panel count, token/offset continuity, and state progress");

  const auto invalid_prompt_elapsed_is_rejected = [](const double elapsed) {
    FakeRunner fake;
    fake.prompt_elapsed_milliseconds = elapsed;
    PhaseContext context{&fake};
    detail::PrefillPlan prefill_plan;
    prefill_plan.context = &context;
    prefill_plan.prefix_step = fake_prefix_step;
    prefill_plan.finish_prefill = fake_finish_prefill;
    prefill_plan.prefix_tile = fake_prefix_tile;
    prefill_plan.finish_prefill_from_tile =
        fake_finish_prefill_from_tile;
    prefill_plan.whole_request = fake_whole_request_prefill;
    prefill_plan.finish_whole_request_from_uncommitted_retained =
        fake_finish_whole_request_from_uncommitted_retained;
    prefill_plan.commit_whole_request = fake_commit_whole_request;
    detail::DecodePlan decode_plan;
    decode_plan.context = &context;
    decode_plan.decode_step = fake_decode_step;
    detail::GenerationControlOptions control_options =
        options(1U, 32U, false, 512U);
    control_options.prefill_all_prompt_tokens = true;
    control_options.prefill_whole_request_layer_major = true;
    const detail::GenerationControlResult result =
        detail::run_generation_control(
            std::vector<std::uint32_t>(32U, 9U), control_options,
            prefill_plan, decode_plan);
    return !result &&
           result.error == detail::GenerationControlError::kUnexpectedStep &&
           fake.prompt_call_count == 1U &&
           fake.whole_finalizer_call_count == 0U &&
           fake.prompt_commit_call_count == 0U &&
           fake.prompt_commit_success_count == 0U &&
           fake.next_position == 0U;
  };

  test.expect(
      invalid_prompt_elapsed_is_rejected(
          std::numeric_limits<double>::quiet_NaN()) &&
          invalid_prompt_elapsed_is_rejected(
              std::numeric_limits<double>::infinity()) &&
          invalid_prompt_elapsed_is_rejected(-1.0),
      "whole-request aggregate prompt timing rejects NaN, infinity, and "
      "negative elapsed values before finalization or state commit");

  const auto invalid_configuration_is_rejected =
      [](const bool all_prompt, const bool capture_trace,
         const bool single_arbitrary, const bool provide_callback,
         const bool provide_finalizer, const bool provide_commit) {
        FakeRunner fake;
        PhaseContext context{&fake};
        detail::PrefillPlan prefill_plan;
        prefill_plan.context = &context;
        prefill_plan.prefix_step = fake_prefix_step;
        prefill_plan.finish_prefill = fake_finish_prefill;
        prefill_plan.prefix_tile = fake_prefix_tile;
        prefill_plan.finish_prefill_from_tile =
            fake_finish_prefill_from_tile;
        prefill_plan.whole_request =
            provide_callback ? fake_whole_request_prefill : nullptr;
        prefill_plan.finish_whole_request_from_uncommitted_retained =
            provide_finalizer
                ? fake_finish_whole_request_from_uncommitted_retained
                : nullptr;
        prefill_plan.commit_whole_request =
            provide_commit ? fake_commit_whole_request : nullptr;
        detail::DecodePlan decode_plan;
        decode_plan.context = &context;
        decode_plan.decode_step = fake_decode_step;
        detail::GenerationControlOptions control_options =
            options(1U, 32U, capture_trace, 512U);
        control_options.prefill_all_prompt_tokens = all_prompt;
        control_options.prefill_single_arbitrary_tile = single_arbitrary;
        control_options.prefill_whole_request_layer_major = true;
        const detail::GenerationControlResult result =
            detail::run_generation_control(
                std::vector<std::uint32_t>(32U, 7U), control_options,
                prefill_plan, decode_plan);
        return !result &&
               result.error ==
                   detail::GenerationControlError::kInvalidArgument &&
               fake.phase_calls.empty() && fake.prompt_call_count == 0U &&
               fake.tile_inputs.empty() && fake.inputs.empty();
      };

  test.expect(
      invalid_configuration_is_rejected(false, false, false, true, true,
                                        true) &&
          invalid_configuration_is_rejected(true, true, false, true, true,
                                            true) &&
          invalid_configuration_is_rejected(true, false, true, true, true,
                                            true) &&
          invalid_configuration_is_rejected(true, false, false, false,
                                             true, true) &&
          invalid_configuration_is_rejected(true, false, false, true,
                                             false, true) &&
          invalid_configuration_is_rejected(true, false, false, true, true,
                                            false),
      "whole-request mode requires explicit all-prompt, non-trace, "
      "non-arbitrary admission with prompt, uncommitted finalizer, and "
      "commit callbacks");

  const auto finalizer_or_commit_failure_is_rejected =
      [](const double finalizer_elapsed, const bool finalizer_failure,
         const bool finalizer_advances,
         const detail::GenerationControlError expected_error,
         const std::size_t expected_commit_calls,
         const std::size_t expected_position) {
        FakeRunner fake;
        fake.predictions = {runtime::kQwen36ImEndTokenId};
        fake.whole_finalizer_elapsed_milliseconds = finalizer_elapsed;
        fake.whole_finalizer_fail = finalizer_failure;
        fake.whole_finalizer_advance_state = finalizer_advances;
        PhaseContext context{&fake};
        detail::PrefillPlan prefill_plan;
        prefill_plan.context = &context;
        prefill_plan.prefix_step = fake_prefix_step;
        prefill_plan.finish_prefill = fake_finish_prefill;
        prefill_plan.prefix_tile = fake_prefix_tile;
        prefill_plan.finish_prefill_from_tile =
            fake_finish_prefill_from_tile;
        prefill_plan.whole_request = fake_whole_request_prefill;
        prefill_plan.finish_whole_request_from_uncommitted_retained =
            fake_finish_whole_request_from_uncommitted_retained;
        prefill_plan.commit_whole_request = fake_commit_whole_request;
        detail::DecodePlan decode_plan;
        decode_plan.context = &context;
        decode_plan.decode_step = fake_decode_step;
        detail::GenerationControlOptions control_options =
            options(1U, 32U, false, 512U);
        control_options.prefill_all_prompt_tokens = true;
        control_options.prefill_whole_request_layer_major = true;
        const detail::GenerationControlResult result =
            detail::run_generation_control(
                std::vector<std::uint32_t>(32U, 9U), control_options,
                prefill_plan, decode_plan);
        return !result && result.error == expected_error &&
               fake.prompt_call_count == 1U &&
               fake.whole_finalizer_call_count == 1U &&
               fake.prompt_commit_call_count == expected_commit_calls &&
               fake.prompt_commit_success_count == 0U &&
               fake.next_position == expected_position;
      };

  test.expect(
      finalizer_or_commit_failure_is_rejected(
          1.0, true, false,
          detail::GenerationControlError::kRunnerFailure, 0U, 0U) &&
          finalizer_or_commit_failure_is_rejected(
              std::numeric_limits<double>::quiet_NaN(), false, false,
              detail::GenerationControlError::kUnexpectedStep, 0U, 0U) &&
          finalizer_or_commit_failure_is_rejected(
              std::numeric_limits<double>::infinity(), false, false,
              detail::GenerationControlError::kUnexpectedStep, 0U, 0U) &&
          finalizer_or_commit_failure_is_rejected(
              -1.0, false, false,
              detail::GenerationControlError::kUnexpectedStep, 0U, 0U) &&
          finalizer_or_commit_failure_is_rejected(
              1.0, false, true,
              detail::GenerationControlError::kRunnerFailure, 1U, 1U),
      "whole-request finalizer failures and invalid timings never call "
      "commit, while an advancing finalizer makes the atomic commit fail");

  {
    FakeRunner fake;
    fake.predictions = {runtime::kQwen36ImEndTokenId};
    fake.whole_finalizer_throw = true;
    PhaseContext context{&fake};
    detail::PrefillPlan prefill_plan;
    prefill_plan.context = &context;
    prefill_plan.prefix_step = fake_prefix_step;
    prefill_plan.finish_prefill = fake_finish_prefill;
    prefill_plan.prefix_tile = fake_prefix_tile;
    prefill_plan.finish_prefill_from_tile =
        fake_finish_prefill_from_tile;
    prefill_plan.whole_request = fake_whole_request_prefill;
    prefill_plan.finish_whole_request_from_uncommitted_retained =
        fake_finish_whole_request_from_uncommitted_retained;
    prefill_plan.commit_whole_request = fake_commit_whole_request;
    detail::DecodePlan decode_plan;
    decode_plan.context = &context;
    decode_plan.decode_step = fake_decode_step;
    detail::GenerationControlOptions control_options =
        options(1U, 32U, false, 512U);
    control_options.prefill_all_prompt_tokens = true;
    control_options.prefill_whole_request_layer_major = true;
    const detail::GenerationControlResult result =
        detail::run_generation_control(
            std::vector<std::uint32_t>(32U, 9U), control_options,
            prefill_plan, decode_plan);
    test.expect(
        !result &&
            result.error == detail::GenerationControlError::kRunnerFailure &&
            result.runner_status.error ==
                runtime::ReferenceRunnerError::kPoisoned &&
            result.runner_status.operation != nullptr &&
            std::string_view(result.runner_status.operation) ==
                "whole_request_finalizer_exception" &&
            fake.prompt_call_count == 1U &&
            fake.whole_finalizer_call_count == 1U &&
            fake.prompt_commit_call_count == 0U &&
            fake.prompt_commit_success_count == 0U &&
            fake.next_position == 0U,
        "whole-request dedicated finalizer exceptions map to a poisoned "
        "runner failure without committing staged state");
  }

  {
    FakeRunner fake;
    fake.predictions = {runtime::kQwen36ImEndTokenId};
    fake.prompt_commit_force_failure = true;
    PhaseContext context{&fake};
    detail::PrefillPlan prefill_plan;
    prefill_plan.context = &context;
    prefill_plan.prefix_step = fake_prefix_step;
    prefill_plan.finish_prefill = fake_finish_prefill;
    prefill_plan.whole_request = fake_whole_request_prefill;
    prefill_plan.finish_whole_request_from_uncommitted_retained =
        fake_finish_whole_request_from_uncommitted_retained;
    prefill_plan.commit_whole_request = fake_commit_whole_request;
    detail::DecodePlan decode_plan;
    decode_plan.context = &context;
    decode_plan.decode_step = fake_decode_step;
    detail::GenerationControlOptions control_options =
        options(1U, 32U, false, 512U);
    control_options.prefill_all_prompt_tokens = true;
    control_options.prefill_whole_request_layer_major = true;
    const detail::GenerationControlResult result =
        detail::run_generation_control(
            std::vector<std::uint32_t>(32U, 9U), control_options,
            prefill_plan, decode_plan);
    test.expect(
        !result &&
            result.error == detail::GenerationControlError::kRunnerFailure &&
            result.runner_status.error ==
                runtime::ReferenceRunnerError::kStateCommitFailure &&
            result.runner_status.operation != nullptr &&
            std::string_view(result.runner_status.operation) ==
                "fake_whole_request_commit" &&
            fake.prompt_call_count == 1U &&
            fake.whole_finalizer_call_count == 1U &&
            fake.prompt_commit_call_count == 1U &&
            fake.prompt_commit_success_count == 0U &&
            fake.next_position == 0U,
        "whole-request commit failure leaves staged state unpublished and "
        "preserves the callback's exact failure status");
  }

  FakeRunner isolated;
  isolated.predictions = {runtime::kQwen36ImEndTokenId};
  PhaseContext isolated_context{&isolated};
  detail::PrefillPlan isolated_prefill;
  isolated_prefill.context = &isolated_context;
  isolated_prefill.prefix_step = fake_prefix_step;
  isolated_prefill.finish_prefill = fake_finish_prefill;
  isolated_prefill.prefix_tile = fake_prefix_tile;
  isolated_prefill.finish_prefill_from_tile =
      fake_finish_prefill_from_tile;
  isolated_prefill.whole_request = fake_whole_request_prefill;
  isolated_prefill.finish_whole_request_from_uncommitted_retained =
      fake_finish_whole_request_from_uncommitted_retained;
  isolated_prefill.commit_whole_request = fake_commit_whole_request;
  detail::DecodePlan isolated_decode;
  isolated_decode.context = &isolated_context;
  isolated_decode.decode_step = fake_decode_step;
  const detail::GenerationControlResult isolated_result =
      detail::run_generation_control(
          {10U, 11U}, options(1U, 2U), isolated_prefill,
          isolated_decode);
  test.expect(
      isolated_result && isolated.prompt_call_count == 0U &&
          isolated_result.value->prefill_execution_mode ==
              runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled &&
          isolated_result.value->prefill_logical_panel_count == 2U &&
          isolated.whole_finalizer_call_count == 0U &&
          isolated.prompt_commit_call_count == 0U &&
          isolated.phase_calls ==
              std::vector<PhaseCall>({PhaseCall::kPrefixStep,
                                      PhaseCall::kFinishPrefill}) &&
          isolated.next_position == 2U,
      "whole-request callbacks remain isolated when the explicit flag is "
      "false");
}

void test_single_arbitrary_tile_admission(TestContext& test) {
  const auto run_shape = [&test](
                             const std::size_t prompt_size,
                             const std::uint32_t chunk_size,
                             const std::vector<std::size_t>& expected_tiles,
                             const runtime::ReferenceLogitsMode logits_mode) {
    FakeRunner fake;
    fake.predictions = {runtime::kQwen36ImEndTokenId};
    PhaseContext context{&fake};

    detail::PrefillPlan prefill_plan;
    prefill_plan.context = &context;
    prefill_plan.prefix_step = fake_prefix_step;
    prefill_plan.finish_prefill = fake_finish_prefill;
    prefill_plan.prefix_tile = fake_prefix_tile;
    prefill_plan.finish_prefill_from_tile =
        fake_finish_prefill_from_tile;
    detail::DecodePlan decode_plan;
    decode_plan.context = &context;
    decode_plan.decode_step = fake_decode_step;

    std::vector<std::uint32_t> prompt(prompt_size);
    for (std::size_t index = 0U; index < prompt.size(); ++index) {
      prompt[index] = static_cast<std::uint32_t>(2'000U + index);
    }
    detail::GenerationControlOptions control_options =
        options(1U, static_cast<std::uint32_t>(prompt_size), false,
                chunk_size, logits_mode);
    control_options.prefill_all_prompt_tokens = true;
    control_options.prefill_single_arbitrary_tile = true;
    const auto result = detail::run_generation_control(
        prompt, control_options, prefill_plan, decode_plan);

    std::vector<std::size_t> actual_tiles;
    std::vector<std::uint32_t> reconstructed_prompt;
    for (const auto& tile : fake.tile_inputs) {
      actual_tiles.push_back(tile.size());
      reconstructed_prompt.insert(reconstructed_prompt.end(), tile.begin(),
                                  tile.end());
    }
    bool only_final_tile_retains =
        fake.tile_options.size() == expected_tiles.size() &&
        !fake.tile_options.empty();
    for (std::size_t index = 0U; index < fake.tile_options.size(); ++index) {
      only_final_tile_retains =
          only_final_tile_retains &&
          fake.tile_options[index].retain_last_hidden_for_logits ==
              (index + 1U == fake.tile_options.size());
    }
    const bool correct_result_arm =
        result &&
        (logits_mode == runtime::ReferenceLogitsMode::kFullStatistics
             ? result.value->steps.back().logits.has_value() &&
                   !result.value->steps.back().prediction.has_value()
             : !result.value->steps.back().logits.has_value() &&
                   result.value->steps.back().prediction.has_value());
    test.expect(
        result && actual_tiles == expected_tiles &&
            reconstructed_prompt == prompt && only_final_tile_retains &&
            fake.phase_calls.size() == expected_tiles.size() + 1U &&
            fake.phase_calls.back() == PhaseCall::kFinishPrefillFromTile &&
            fake.next_position == prompt_size &&
            fake.inputs == std::vector<std::uint32_t>({prompt.back()}) &&
            result.value->steps.size() == prompt_size &&
            result.value->timing.prefix_execution_milliseconds.size() ==
                expected_tiles.size() &&
            correct_result_arm &&
            has_consistent_prefill_timing(result.value->timing),
        "single-arbitrary admission preserves ordered whole-prompt state "
        "while eliminating canonical scheduler splits");
  };

  constexpr auto kFull = runtime::ReferenceLogitsMode::kFullStatistics;
  constexpr auto kPredicted =
      runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  run_shape(32U, 512U, {32U}, kFull);
  run_shape(64U, 512U, {64U}, kPredicted);
  run_shape(256U, 512U, {256U}, kPredicted);
  run_shape(407U, 512U, {407U}, kFull);
  run_shape(481U, 512U, {481U}, kPredicted);
  run_shape(512U, 512U, {512U}, kPredicted);
  run_shape(513U, 512U, {512U, 1U}, kPredicted);
  run_shape(1'025U, 512U, {512U, 512U, 1U}, kPredicted);
  run_shape(407U, 320U, {320U, 87U}, kPredicted);

  test.expect(
      detail::next_single_arbitrary_prefix_tile_token_count(407U, 512U) ==
              407U &&
          detail::next_single_arbitrary_prefix_tile_token_count(481U,
                                                                 512U) ==
              481U &&
          detail::next_single_arbitrary_prefix_tile_token_count(513U,
                                                                 512U) ==
              512U &&
          detail::single_arbitrary_prefix_execution_count(407U, 512U) ==
              1U &&
          detail::single_arbitrary_prefix_execution_count(1'025U, 512U) ==
              3U &&
          detail::single_arbitrary_prefix_execution_count(407U, 0U) == 0U,
      "single-arbitrary scheduler boundaries and cardinalities are exact");

  FakeRunner fake;
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  PhaseContext context{&fake};
  detail::PrefillPlan prefill_plan;
  prefill_plan.context = &context;
  prefill_plan.prefix_step = fake_prefix_step;
  prefill_plan.finish_prefill = fake_finish_prefill;
  prefill_plan.prefix_tile = fake_prefix_tile;
  prefill_plan.finish_prefill_from_tile =
      fake_finish_prefill_from_tile;
  detail::DecodePlan decode_plan;
  decode_plan.context = &context;
  decode_plan.decode_step = fake_decode_step;
  detail::GenerationControlOptions invalid = options(1U, 407U, false, 512U);
  invalid.prefill_single_arbitrary_tile = true;
  const auto result = detail::run_generation_control(
      std::vector<std::uint32_t>(407U, 7U), invalid, prefill_plan,
      decode_plan);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kInvalidArgument &&
                  fake.phase_calls.empty(),
              "single-arbitrary admission fails closed unless whole-prompt "
              "admission is explicit");
}

std::vector<std::size_t> prefix_schedule(
    std::size_t remaining_tokens,
    const std::size_t requested_chunk_size) {
  std::vector<std::size_t> result;
  while (remaining_tokens != 0U) {
    const std::size_t tile = detail::next_prefix_tile_token_count(
        remaining_tokens, requested_chunk_size);
    if (tile == 0U || tile > remaining_tokens) {
      return {};
    }
    result.push_back(tile);
    remaining_tokens -= tile;
  }
  return result;
}

void test_explicit_c512_prefill_schedule(TestContext& test) {
  constexpr std::size_t kC512 =
      runtime::kMaximumRequestPrefillChunkSize;
  const bool boundaries_exact =
      prefix_schedule(31U, kC512) == std::vector<std::size_t>({31U}) &&
      prefix_schedule(32U, kC512) == std::vector<std::size_t>({32U}) &&
      prefix_schedule(33U, kC512) ==
          std::vector<std::size_t>({32U, 1U}) &&
      prefix_schedule(63U, kC512) ==
          std::vector<std::size_t>({32U, 31U}) &&
      prefix_schedule(64U, kC512) == std::vector<std::size_t>({64U}) &&
      prefix_schedule(65U, kC512) ==
          std::vector<std::size_t>({64U, 1U}) &&
      prefix_schedule(255U, kC512) ==
          std::vector<std::size_t>({64U, 64U, 64U, 32U, 31U}) &&
      prefix_schedule(256U, kC512) == std::vector<std::size_t>({256U}) &&
      prefix_schedule(257U, kC512) ==
          std::vector<std::size_t>({256U, 1U}) &&
      prefix_schedule(511U, kC512) ==
          std::vector<std::size_t>({256U, 64U, 64U, 64U, 32U, 31U}) &&
      prefix_schedule(512U, kC512) == std::vector<std::size_t>({512U}) &&
      prefix_schedule(513U, kC512) ==
          std::vector<std::size_t>({512U, 1U}) &&
      prefix_schedule(769U, kC512) ==
          std::vector<std::size_t>({512U, 256U, 1U});
  test.expect(boundaries_exact,
              "C512 scheduler boundaries use only 512/256/64/32/tail tiles");

  const bool noncanonical_caps_exact =
      prefix_schedule(128U, 128U) ==
          std::vector<std::size_t>({64U, 64U}) &&
      prefix_schedule(192U, 192U) ==
          std::vector<std::size_t>({64U, 64U, 64U}) &&
      prefix_schedule(320U, 320U) ==
          std::vector<std::size_t>({256U, 64U});
  test.expect(noncanonical_caps_exact,
              "C128/C192/C320 caps decompose into explicit production tiles");

  test.expect(detail::next_prefix_tile_token_count(1U, 0U) == 0U &&
                  detail::next_prefix_tile_token_count(0U, kC512) == 0U &&
                  detail::prefix_execution_count(769U, kC512) == 3U &&
                  detail::prefix_execution_count(320U, 320U) == 2U,
              "scheduler zero guards and execution cardinalities are exact");
}

void test_explicit_phase_plan_shape_matrix(TestContext& test) {
  constexpr std::array<std::size_t, 14U> kPromptSizes = {
      1U,  2U,  7U,  8U,  9U,  15U, 16U,
      17U, 31U, 32U, 33U, 63U, 64U, 65U};
  constexpr std::array<std::uint32_t, 8U> kChunkSizes = {
      1U, 2U, 8U, 16U, 32U, 33U, 63U, 64U};

  for (const std::size_t prompt_size : kPromptSizes) {
    std::vector<std::uint32_t> prompt;
    prompt.reserve(prompt_size);
    for (std::size_t index = 0U; index < prompt_size; ++index) {
      prompt.push_back(static_cast<std::uint32_t>(100U + index));
    }

    for (const std::uint32_t chunk_size : kChunkSizes) {
      FakeRunner fake;
      fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
      PhaseContext prefill_context{&fake};
      PhaseContext decode_context{&fake};

      detail::PrefillPlan prefill_plan;
      prefill_plan.context = &prefill_context;
      prefill_plan.prefix_step = fake_prefix_step;
      prefill_plan.finish_prefill = fake_finish_prefill;
      prefill_plan.prefix_tile = fake_prefix_tile;

      detail::DecodePlan decode_plan;
      decode_plan.context = &decode_context;
      decode_plan.decode_step = fake_decode_step;

      const auto result = detail::run_generation_control(
          prompt,
          options(2U, static_cast<std::uint32_t>(prompt_size + 1U),
                  false, chunk_size),
          prefill_plan, decode_plan);

      const std::size_t prefix_size = prompt_size - 1U;
      const bool tiled = chunk_size > 1U && prefix_size != 0U;
      const std::size_t expected_prefix_calls =
          detail::prefix_execution_count(prefix_size, chunk_size);
      bool route_matches =
          result && fake.phase_calls.size() == expected_prefix_calls + 2U;
      for (std::size_t index = 0U;
           route_matches && index < expected_prefix_calls; ++index) {
        route_matches =
            fake.phase_calls[index] ==
            (tiled ? PhaseCall::kPrefixTile : PhaseCall::kPrefixStep);
      }
      route_matches =
          route_matches &&
          fake.phase_calls[expected_prefix_calls] ==
              PhaseCall::kFinishPrefill &&
          fake.phase_calls[expected_prefix_calls + 1U] ==
              PhaseCall::kDecodeStep;

      bool input_matches = false;
      if (tiled) {
        std::vector<std::uint32_t> reconstructed_prefix;
        for (const auto& tile : fake.tile_inputs) {
          reconstructed_prefix.insert(reconstructed_prefix.end(),
                                      tile.begin(), tile.end());
        }
        input_matches =
            reconstructed_prefix ==
                std::vector<std::uint32_t>(prompt.begin(),
                                           prompt.end() - 1) &&
            fake.inputs ==
                std::vector<std::uint32_t>({prompt.back(), 42U});
      } else {
        std::vector<std::uint32_t> expected_inputs = prompt;
        expected_inputs.push_back(42U);
        input_matches = fake.tile_inputs.empty() &&
                        fake.inputs == expected_inputs;
      }

      const bool case_passed =
          route_matches && input_matches && result.value->steps.size() ==
                                               prompt_size + 1U &&
          result.value->timing.prefix_execution_milliseconds.size() ==
              expected_prefix_calls &&
          has_consistent_prefill_timing(result.value->timing);
      if (!case_passed) {
        std::cerr << "  phase matrix mismatch: prompt_size=" << prompt_size
                  << " chunk_size=" << chunk_size << '\n';
      }
      test.expect(case_passed,
                  "phase plans preserve routing across prompt and chunk boundaries");
    }
  }
}

void test_nvtx_phase_ranges_preserve_control_semantics(TestContext& test) {
  const auto run = [](FakeRunner& fake, const bool emit_nvtx_ranges,
                      const std::uint32_t chunk_size) {
    fake.predictions = {42U, runtime::kQwen36ImEndTokenId};
    PhaseContext prefill_context{&fake};
    PhaseContext decode_context{&fake};

    detail::PrefillPlan prefill_plan;
    prefill_plan.context = &prefill_context;
    prefill_plan.prefix_step = fake_prefix_step;
    prefill_plan.finish_prefill = fake_finish_prefill;
    prefill_plan.prefix_tile = fake_prefix_tile;

    detail::DecodePlan decode_plan;
    decode_plan.context = &decode_context;
    decode_plan.decode_step = fake_decode_step;

    detail::GenerationControlOptions control_options =
        options(2U, 4U, false, chunk_size);
    control_options.emit_nvtx_phase_ranges = emit_nvtx_ranges;
    return detail::run_generation_control(
        {10U, 11U, 12U}, control_options, prefill_plan, decode_plan);
  };

  FakeRunner baseline_fake;
  FakeRunner nvtx_fake;
  const auto baseline = run(baseline_fake, false, 1U);
  const auto with_nvtx = run(nvtx_fake, true, 1U);
  const runtime::ReferenceGenerateOptions default_generate_options;
  const detail::GenerationControlOptions default_control_options;
  test.expect(!default_generate_options.emit_nvtx_phase_ranges &&
                  !default_control_options.emit_nvtx_phase_ranges &&
                  default_generate_options.prefill_progress_observer ==
                      nullptr &&
                  default_generate_options.prefill_progress_context ==
                      nullptr,
              "NVTX phase ranges are disabled by default");
  test.expect(baseline && with_nvtx &&
                  baseline_fake.phase_calls == nvtx_fake.phase_calls &&
                  baseline_fake.inputs == nvtx_fake.inputs &&
                  baseline.value->generated_token_ids ==
                      with_nvtx.value->generated_token_ids &&
                  baseline.value->stop_reason == with_nvtx.value->stop_reason &&
                  baseline.value->timing.prefix_execution_milliseconds ==
                      with_nvtx.value->timing.prefix_execution_milliseconds &&
                  baseline.value->timing.finish_prefill_milliseconds ==
                      with_nvtx.value->timing.finish_prefill_milliseconds &&
                  baseline.value->timing.prompt_prefill_milliseconds ==
                      with_nvtx.value->timing.prompt_prefill_milliseconds &&
                  baseline.value->timing.subsequent_token_milliseconds ==
                      with_nvtx.value->timing.subsequent_token_milliseconds &&
                  baseline.value->timing.decode_after_first_milliseconds ==
                      with_nvtx.value->timing.decode_after_first_milliseconds &&
                  baseline.value->timing.total_generation_milliseconds ==
                      with_nvtx.value->timing.total_generation_milliseconds,
              "enabled NVTX ranges preserve phase ordering, output, and "
              "timing semantics");

  FakeRunner tiled_baseline_fake;
  FakeRunner tiled_nvtx_fake;
  const auto tiled_baseline = run(tiled_baseline_fake, false, 2U);
  const auto tiled_with_nvtx = run(tiled_nvtx_fake, true, 2U);
  test.expect(tiled_baseline && tiled_with_nvtx &&
                  tiled_baseline_fake.phase_calls ==
                      tiled_nvtx_fake.phase_calls &&
                  tiled_baseline_fake.tile_inputs ==
                      tiled_nvtx_fake.tile_inputs &&
                  tiled_baseline.value->timing.prefix_execution_milliseconds ==
                      tiled_with_nvtx.value->timing
                          .prefix_execution_milliseconds &&
                  tiled_baseline.value->timing.finish_prefill_milliseconds ==
                      tiled_with_nvtx.value->timing
                          .finish_prefill_milliseconds,
              "enabled NVTX ranges preserve tiled-prefix semantics");

  FakeRunner failure_fake;
  failure_fake.fail_at = 0U;
  const auto failed_with_nvtx = run(failure_fake, true, 1U);
  test.expect(!failed_with_nvtx &&
                  failed_with_nvtx.error ==
                      detail::GenerationControlError::kRunnerFailure,
              "enabled NVTX ranges preserve runner-failure propagation");
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
                  result.value->timing.prefix_execution_milliseconds ==
                      std::vector<double>({10.0, 10.0, 10.0}) &&
                  result.value->timing.finish_prefill_milliseconds == 1.0 &&
                  result.value->timing.time_to_first_token_milliseconds ==
                      31.0 &&
                  has_consistent_prefill_timing(result.value->timing) &&
                  result.value->timing.subsequent_token_milliseconds ==
                      std::vector<double>({2.0}) &&
                  result.value->timing.decode_after_first_milliseconds == 2.0 &&
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

  std::vector<std::uint32_t> c32_prompt;
  for (std::uint32_t token = 300U; token < 333U; ++token) {
    c32_prompt.push_back(token);
  }
  fake = {};
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  const auto c32_result = detail::run_generation_control(
      c32_prompt, options(1U, 33U, false, 32U), &fake, fake_step,
      fake_prefill_tile);
  test.expect(c32_result && fake.tile_inputs.size() == 1U &&
                  fake.tile_inputs[0] ==
                      std::vector<std::uint32_t>(c32_prompt.begin(),
                                                 c32_prompt.begin() + 32) &&
                  fake.inputs == std::vector<std::uint32_t>({332U}),
              "chunk thirty-two routes a P33 prompt as one 32-token prefix "
              "tile before the scalar final prompt token");

  std::vector<std::uint32_t> c64_prompt;
  for (std::uint32_t token = 400U; token < 465U; ++token) {
    c64_prompt.push_back(token);
  }
  fake = {};
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  const auto c64_result = detail::run_generation_control(
      c64_prompt, options(1U, 65U, false, 64U), &fake, fake_step,
      fake_prefill_tile);
  test.expect(c64_result && fake.tile_inputs.size() == 1U &&
                  fake.tile_inputs[0].size() == 64U &&
                  fake.inputs == std::vector<std::uint32_t>({464U}),
              "chunk sixty-four routes a P65 prompt through one exact C64 "
              "prefix tile");

  std::vector<std::uint32_t> c64_tail_prompt;
  for (std::uint32_t token = 500U; token < 628U; ++token) {
    c64_tail_prompt.push_back(token);
  }
  fake = {};
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  const auto c64_tail_result = detail::run_generation_control(
      c64_tail_prompt, options(1U, 128U, false, 64U), &fake, fake_step,
      fake_prefill_tile);
  test.expect(c64_tail_result && fake.tile_inputs.size() == 3U &&
                  fake.tile_inputs[0].size() == 64U &&
                  fake.tile_inputs[1].size() == 32U &&
                  fake.tile_inputs[2].size() == 31U &&
                  fake.inputs == std::vector<std::uint32_t>({627U}),
              "chunk sixty-four keeps a 63-token remainder on optimized "
              "C32+M31 runner calls");

  std::vector<std::uint32_t> c512_prompt;
  for (std::uint32_t token = 1'000U; token < 1'770U; ++token) {
    c512_prompt.push_back(token);
  }
  fake = {};
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  const auto c512_result = detail::run_generation_control(
      c512_prompt, options(1U, 770U, false, 512U), &fake, fake_step,
      fake_prefill_tile);
  test.expect(c512_result && fake.tile_inputs.size() == 3U &&
                  fake.tile_inputs[0].size() == 512U &&
                  fake.tile_inputs[1].size() == 256U &&
                  fake.tile_inputs[2].size() == 1U &&
                  fake.inputs == std::vector<std::uint32_t>({1'769U}),
              "C512 routes a 769-token prefix as C512+C256+tail before the "
              "scalar final prompt token");
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
      {1U}, options(1U, 1U, false,
                    runtime::kMaximumRequestPrefillChunkSize + 1U),
      &fake, fake_step);
  test.expect(!result && result.error == detail::GenerationControlError::kInvalidArgument,
              "prefill chunk size above fixed capacity is rejected");
  detail::GenerationControlOptions invalid_mode = options(1U, 1U);
  invalid_mode.logits_mode =
      static_cast<runtime::ReferenceLogitsMode>(255U);
  result = detail::run_generation_control(
      {1U}, invalid_mode, &fake, fake_step);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kInvalidArgument,
              "unknown logits mode is rejected before callback");

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

  for (const double invalid_timing : {
           -1.0, std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity()}) {
    fake = {};
    fake.predictions = {1U};
    fake.override_timing = true;
    fake.timing_milliseconds = invalid_timing;
    result = detail::run_generation_control(
        {1U}, options(1U, 1U), &fake, fake_step);
    test.expect(!result &&
                    result.error ==
                        detail::GenerationControlError::kUnexpectedStep,
                "scalar step rejects negative and non-finite timing");
  }

  fake = {};
  fake.predictions = {1U, runtime::kQwen36ImEndTokenId};
  fake.override_timing = true;
  fake.timing_milliseconds = std::numeric_limits<double>::max();
  result = detail::run_generation_control(
      {1U}, options(2U, 2U), &fake, fake_step);
  test.expect(!result &&
                  result.error ==
                      detail::GenerationControlError::kUnexpectedStep,
              "decode timing accumulation rejects finite-value overflow");

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

void test_committed_token_observer_and_cancellation(TestContext& test) {
  FakeRunner fake;
  fake.predictions = {42U, 43U, 44U};
  CommittedTokenRecorder recorder;
  recorder.cancel_after = 2U;
  detail::GenerationControlOptions observed = options(3U, 3U);
  observed.committed_token_context = &recorder;
  observed.committed_token = record_committed_token;
  auto result = detail::run_generation_control(
      {7U}, observed, &fake, fake_step);
  test.expect(result &&
                  result.value->generated_token_ids ==
                      std::vector<std::uint32_t>({42U, 43U}) &&
                  result.value->stop_reason ==
                      runtime::ReferenceStopReason::kCancelled &&
                  fake.inputs == std::vector<std::uint32_t>({7U, 42U}) &&
                  recorder.count == 2U &&
                  recorder.token_ids[0] == 42U &&
                  recorder.token_ids[1] == 43U &&
                  recorder.indices[0] == 0U &&
                  recorder.indices[1] == 1U &&
                  recorder.elapsed_milliseconds[0] == 1.0 &&
                  recorder.elapsed_milliseconds[1] == 2.0,
              "observer sees committed tokens and cancellation prevents the "
              "next Decode step");

  fake = {};
  fake.predictions = {runtime::kQwen36ImEndTokenId};
  recorder = {};
  recorder.cancel_after = 1U;
  observed = options(3U, 3U);
  observed.committed_token_context = &recorder;
  observed.committed_token = record_committed_token;
  result = detail::run_generation_control(
      {7U}, observed, &fake, fake_step);
  test.expect(result && recorder.count == 1U &&
                  result.value->generated_token_ids ==
                      std::vector<std::uint32_t>(
                          {runtime::kQwen36ImEndTokenId}) &&
                  result.value->stop_reason ==
                      runtime::ReferenceStopReason::kImEnd &&
                  fake.inputs == std::vector<std::uint32_t>({7U}),
              "terminal stop takes precedence over simultaneous observer "
              "cancellation");
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

  runtime::ReferenceOneShotOptions one_shot_options;
  test.expect(one_shot_options.overlap_tokenizer_and_resident_load,
              "one-shot startup overlaps independent asset loads by default");
  one_shot_options.generation.logits_mode =
      static_cast<runtime::ReferenceLogitsMode>(255U);
  const runtime::ReferenceOneShotResult generated =
      runtime::generate_reference("unused-model-directory", "prompt",
                                  one_shot_options);
  test.expect(!generated &&
                  generated.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  generated.diagnostic.stage == "one_shot_options",
              "one-shot generation rejects an unknown logits mode before I/O");
}

void test_target_aot_device_preparation_fails_closed_before_io(
    TestContext& test) {
  runtime::ReferenceEngineOptions defaults;
  test.expect(
      !defaults.prepare_sm87_target_aot_projection_device_assets &&
          defaults.load_sm87_target_aot_projection_bundle.empty() &&
          defaults.create_sm87_target_aot_projection_bundle.empty() &&
          defaults
              .expected_sm87_target_aot_projection_payload_catalog_sha256
              .empty(),
      "all target-AOT online, offline-create, and direct-load modes default "
      "off");

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  const auto expect_invalid_configuration = [&test](
                                                runtime::ReferenceEngineOptions
                                                    options,
                                                const char* const label) {
    const runtime::ReferenceEngineCreateResult invalid =
        runtime::create_reference_engine("/path/must/not/be/read", options);
    test.expect(!invalid &&
                    invalid.diagnostic.code ==
                        runtime::ReferenceEngineError::kInvalidArgument &&
                    invalid.diagnostic.stage ==
                        "target_aot_projection_device_assets" &&
                    invalid.diagnostic.message.find(
                        "startup requires one mode") != std::string::npos,
                label);
  };
  constexpr std::string_view kCatalog =
      "367572d8f5aab87c655695fc621562e0e88cb5d1a9656370353d55ab1c4ebdbe";

  runtime::ReferenceEngineOptions invalid;
  invalid.load_sm87_target_aot_projection_bundle = "/workspace/assets.aot";
  expect_invalid_configuration(
      invalid, "persisted target-AOT load rejects a missing trust root before "
               "model I/O");

  invalid = {};
  invalid.expected_sm87_target_aot_projection_payload_catalog_sha256 =
      std::string(kCatalog);
  expect_invalid_configuration(
      invalid, "target-AOT rejects a digest without a load/create mode before "
               "model I/O");

  invalid = {};
  invalid.prepare_sm87_target_aot_projection_device_assets = true;
  invalid.create_sm87_target_aot_projection_bundle = "relative/assets.aot";
  invalid.expected_sm87_target_aot_projection_payload_catalog_sha256 =
      std::string(kCatalog);
  expect_invalid_configuration(
      invalid, "target-AOT offline creation rejects a relative output before "
               "model I/O");

  invalid = {};
  invalid.load_sm87_target_aot_projection_bundle = "/workspace/load.aot";
  invalid.create_sm87_target_aot_projection_bundle =
      "/workspace/create.aot";
  invalid.expected_sm87_target_aot_projection_payload_catalog_sha256 =
      std::string(kCatalog);
  expect_invalid_configuration(
      invalid, "target-AOT rejects simultaneous direct-load and offline-create "
               "paths before model I/O");

  invalid = {};
  invalid.load_sm87_target_aot_projection_bundle = "/workspace/assets.aot";
  invalid.expected_sm87_target_aot_projection_payload_catalog_sha256 =
      std::string(64U, 'A');
  expect_invalid_configuration(
      invalid, "target-AOT rejects a non-lowercase external trust root before "
               "model I/O");

  invalid.expected_sm87_target_aot_projection_payload_catalog_sha256 =
      std::string(64U, '0');
  expect_invalid_configuration(
      invalid, "target-AOT rejects an all-zero external trust root before "
               "model I/O");

  const auto expect_valid_mode_then_incompatible_engine =
      [&test](runtime::ReferenceEngineOptions options,
              const char* const label) {
        const runtime::ReferenceEngineCreateResult invalid =
            runtime::create_reference_engine("/path/must/not/be/read",
                                             options);
        test.expect(!invalid &&
                        invalid.diagnostic.code ==
                            runtime::ReferenceEngineError::kInvalidArgument &&
                        invalid.diagnostic.stage ==
                            "target_aot_projection_device_assets" &&
                        invalid.diagnostic.message.find(
                            "asset admission requires") != std::string::npos,
                    label);
      };
  runtime::ReferenceEngineOptions valid_create;
  valid_create.prepare_sm87_target_aot_projection_device_assets = true;
  valid_create.create_sm87_target_aot_projection_bundle =
      "/workspace/create.aot";
  valid_create.expected_sm87_target_aot_projection_payload_catalog_sha256 =
      std::string(kCatalog);
  expect_valid_mode_then_incompatible_engine(
      valid_create, "offline-create mode passes source classification before "
                    "engine compatibility validation");

  runtime::ReferenceEngineOptions valid_load;
  valid_load.load_sm87_target_aot_projection_bundle =
      "/workspace/load.aot";
  valid_load.expected_sm87_target_aot_projection_payload_catalog_sha256 =
      std::string(kCatalog);
  expect_valid_mode_then_incompatible_engine(
      valid_load, "direct-load mode passes source classification before "
                  "engine compatibility validation");
#endif

  runtime::ReferenceEngineOptions requested;
  requested.prepare_sm87_target_aot_projection_device_assets = true;
  const runtime::ReferenceEngineCreateResult result =
      runtime::create_reference_engine("/path/must/not/be/read", requested);
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  test.expect(!result &&
                  result.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  result.diagnostic.stage ==
                      "target_aot_projection_device_assets",
              "compiled target-AOT preparation rejects an incompatible "
              "engine before model I/O");
#else
  test.expect(!result &&
                  result.diagnostic.code ==
                      runtime::ReferenceEngineError::kPrefillPlanUnavailable &&
                  result.diagnostic.stage ==
                      "target_aot_projection_device_assets",
              "uncompiled target-AOT preparation fails closed before model "
              "I/O");
#endif
}

void test_bulk_v2_route_fails_closed_before_io(TestContext& test) {
  runtime::ReferenceEngineOptions options;
  options.generation_route =
      runtime::ReferenceGenerationRoute::kSm87BulkV2P40;
  if (!runtime::is_reference_generation_route_compiled(
          options.generation_route)) {
    const runtime::ReferenceEngineCreateResult result =
        runtime::create_reference_engine("/path/must/not/be/read", options);
    test.expect(
        !result &&
            result.diagnostic.code ==
                runtime::ReferenceEngineError::kPrefillPlanUnavailable &&
            result.diagnostic.stage == "generation_route",
        "an uncompiled bulk-v2 engine route fails before tokenizer/model I/O "
        "without reference or v1 fallback");
  }
}

}  // namespace

int main() {
  TestContext test;
  test_prefill_decode_and_stop(test);
  test_explicit_phase_plans(test);
  test_all_prompt_tile_admission(test);
  test_whole_request_layer_major_admission(test);
  test_whole_request_layer_major_fail_closed(test);
  test_single_arbitrary_tile_admission(test);
  test_explicit_c512_prefill_schedule(test);
  test_explicit_phase_plan_shape_matrix(test);
  test_nvtx_phase_ranges_preserve_control_semantics(test);
  test_prediction_only_control(test);
  test_result_arm_validation(test);
  test_max_tokens_and_first_stop(test);
  test_chunked_prefix_tiles(test);
  test_chunk_fallbacks_and_callback_requirement(test);
  test_tile_failures_and_malformed_results(test);
  test_validation_and_runner_failures(test);
  test_generated_text_stop_semantics(test);
  test_committed_token_observer_and_cancellation(test);
  test_engine_backend_validation(test);
  test_target_aot_device_preparation_fails_closed_before_io(test);
  test_bulk_v2_route_fails_closed_before_io(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " reference engine control test(s) failed\n";
    return 1;
  }
  std::cout << "All reference engine control tests passed\n";
  return 0;
}
