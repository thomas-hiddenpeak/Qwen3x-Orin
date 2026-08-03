#include "q3x/runtime/reference_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

bool has_consistent_prefill_timing(
    const runtime::ReferenceGenerationTiming& timing) {
  double prefix_sum = 0.0;
  for (const double elapsed : timing.prefix_execution_milliseconds) {
    prefix_sum += elapsed;
  }
  return prefix_sum + timing.finish_prefill_milliseconds ==
             timing.prompt_prefill_milliseconds &&
         timing.prompt_prefill_milliseconds ==
             timing.time_to_first_token_milliseconds;
}

enum class PhaseCall : std::uint8_t {
  kPrefixStep,
  kPrefixTile,
  kLayerMajorPrompt,
  kFinishPrefill,
  kFinishPrefillFromTile,
  kDecodeStep,
};

struct FakeRunner {
  std::vector<std::uint32_t> predictions;
  std::vector<std::uint32_t> inputs;
  std::vector<runtime::ReferenceStepOptions> options;
  std::vector<std::vector<std::uint32_t>> tile_inputs;
  std::vector<runtime::ReferencePrefillTileOptions> tile_options;
  std::vector<std::uint32_t> long_prefill_inputs;
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
  bool long_prefill_fail = false;
  bool long_prefill_omit_timing = false;
  bool long_prefill_wrong_interval = false;
  double long_prefill_elapsed_milliseconds = 40.0;
  std::size_t long_prefill_attention_k256_incumbent_launch_hits = 128U;
  std::size_t
      long_prefill_attention_k256_incumbent_logical_projection_hits = 208U;
  std::size_t long_prefill_attention_k256_a_exchange_b4_launch_hits = 129U;
  std::size_t
      long_prefill_attention_k256_a_exchange_b4_logical_projection_hits =
          209U;
  std::size_t long_prefill_gateup_alternating_launch_hits = 64U;
  std::size_t long_prefill_gateup_ldmatrix_pairfeed_launch_hits = 0U;
  std::size_t long_prefill_gateup_m128n64_same_cta_launch_hits = 64U;
  std::size_t long_prefill_gateup_m128n512_fused_quantize_launch_hits = 64U;
  std::size_t long_prefill_gateup_m128n512_paired_ldmatrix_launch_hits = 64U;
  std::size_t long_prefill_gateup_m64n128_register_pipeline_launch_hits = 64U;
  std::size_t
      long_prefill_gateup_m64n8_paired_warp_register_pipeline_launch_hits =
          64U;
  std::size_t long_prefill_down_m128n128_ldmatrix_pairring_launch_hits = 64U;
  std::size_t long_prefill_gdn_chunk64_native_launch_hits = 192U;
  std::size_t long_prefill_gdn_chunk64_native_logical_token_hits = 88'944U;
  std::size_t long_prefill_gdn_prompt_span_macro_launch_hits = 48U;
  std::size_t long_prefill_gdn_prompt_span_macro_logical_token_hits = 88'944U;
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

runtime::ReferenceLongPrefillOutcome fake_layer_major_prompt(
    void* const context, const std::uint32_t* const input_tokens,
    const std::size_t token_count, const bool measure_timing) {
  FakeRunner& fake = *static_cast<PhaseContext*>(context)->runner;
  fake.phase_calls.push_back(PhaseCall::kLayerMajorPrompt);
  fake.long_prefill_inputs.assign(input_tokens,
                                  input_tokens + token_count);
  runtime::ReferenceLongPrefillOutcome outcome;
  if (fake.long_prefill_fail) {
    outcome.status.error = runtime::ReferenceRunnerError::kPoisoned;
    outcome.status.layer = 31U;
    outcome.status.operation = "fake_layer_major_prompt";
    return outcome;
  }
  runtime::ReferenceLongPrefillResult value;
  value.first_position = static_cast<std::uint32_t>(
      fake.next_position + (fake.long_prefill_wrong_interval ? 1U : 0U));
  value.token_count =
      token_count + (fake.long_prefill_wrong_interval ? 1U : 0U);
  value.attention_k256_m128n256_incumbent_launch_hits =
      fake.long_prefill_attention_k256_incumbent_launch_hits;
  value.attention_k256_m128n256_incumbent_logical_projection_hits =
      fake.long_prefill_attention_k256_incumbent_logical_projection_hits;
  value.attention_k256_m128n256_a_exchange_b4_launch_hits =
      fake.long_prefill_attention_k256_a_exchange_b4_launch_hits;
  value.attention_k256_m128n256_a_exchange_b4_logical_projection_hits =
      fake.long_prefill_attention_k256_a_exchange_b4_logical_projection_hits;
  value.gateup_alternating_launch_hits =
      fake.long_prefill_gateup_alternating_launch_hits;
  value.gateup_ldmatrix_pairfeed_launch_hits =
      fake.long_prefill_gateup_ldmatrix_pairfeed_launch_hits;
  value.gateup_m128n64_same_cta_launch_hits =
      fake.long_prefill_gateup_m128n64_same_cta_launch_hits;
  value.gateup_m128n512_fused_quantize_launch_hits =
      fake.long_prefill_gateup_m128n512_fused_quantize_launch_hits;
  value.gateup_m128n512_paired_ldmatrix_launch_hits =
      fake.long_prefill_gateup_m128n512_paired_ldmatrix_launch_hits;
  value.gateup_m64n128_register_pipeline_launch_hits =
      fake.long_prefill_gateup_m64n128_register_pipeline_launch_hits;
  value.gateup_m64n8_paired_warp_register_pipeline_launch_hits =
      fake.long_prefill_gateup_m64n8_paired_warp_register_pipeline_launch_hits;
  value.down_m128n128_ldmatrix_pairring_launch_hits =
      fake.long_prefill_down_m128n128_ldmatrix_pairring_launch_hits;
  value.gdn_chunk64_native_launch_hits =
      fake.long_prefill_gdn_chunk64_native_launch_hits;
  value.gdn_chunk64_native_logical_token_hits =
      fake.long_prefill_gdn_chunk64_native_logical_token_hits;
  value.gdn_prompt_span_macro_launch_hits =
      fake.long_prefill_gdn_prompt_span_macro_launch_hits;
  value.gdn_prompt_span_macro_logical_token_hits =
      fake.long_prefill_gdn_prompt_span_macro_logical_token_hits;
  if (measure_timing && !fake.long_prefill_omit_timing) {
    value.timing.emplace(runtime::ReferenceStepTiming{
        fake.long_prefill_elapsed_milliseconds});
  }
  fake.next_position += token_count;
  outcome.value.emplace(std::move(value));
  return outcome;
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

void test_layer_major_prompt_admission(TestContext& test) {
  const auto run_shape = [&test](
                             const std::size_t prompt_size,
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
    prefill_plan.layer_major_prompt = fake_layer_major_prompt;
    detail::DecodePlan decode_plan;
    decode_plan.context = &context;
    decode_plan.decode_step = fake_decode_step;

    std::vector<std::uint32_t> prompt(prompt_size);
    for (std::size_t index = 0U; index < prompt.size(); ++index) {
      prompt[index] = static_cast<std::uint32_t>(3'000U + index);
    }
    detail::GenerationControlOptions control_options =
        options(1U, static_cast<std::uint32_t>(prompt.size()), false,
                runtime::kLongPrefillLayerMajorTileTokens, logits_mode);
    control_options.prefill_all_prompt_tokens = true;
    control_options.prefill_layer_major_prompt = true;
    const auto result = detail::run_generation_control(
        prompt, control_options, prefill_plan, decode_plan);

    const bool correct_result_arm =
        result &&
        (logits_mode == runtime::ReferenceLogitsMode::kFullStatistics
             ? result.value->steps.back().logits.has_value() &&
                   !result.value->steps.back().prediction.has_value()
             : !result.value->steps.back().logits.has_value() &&
                   result.value->steps.back().prediction.has_value());
    test.expect(
        result && fake.long_prefill_inputs == prompt &&
            fake.tile_inputs.empty() && fake.inputs.size() == 1U &&
            fake.inputs.front() == prompt.back() &&
            fake.next_position == prompt.size() &&
            fake.phase_calls ==
                std::vector<PhaseCall>({PhaseCall::kLayerMajorPrompt,
                                        PhaseCall::kFinishPrefillFromTile}) &&
            result.value->steps.size() == prompt.size() &&
            result.value->steps.front().position == 0U &&
            result.value->steps.back().position == prompt.size() - 1U &&
            result.value->steps.back().input_token_id == prompt.back() &&
            result.value->timing.prefix_execution_milliseconds ==
                std::vector<double>({
                    fake.long_prefill_elapsed_milliseconds}) &&
            result.value->timing.finish_prefill_milliseconds == 1.0 &&
            result.value->timing
                    .attention_k256_m128n256_incumbent_launch_hits ==
                fake.long_prefill_attention_k256_incumbent_launch_hits &&
            result.value->timing
                    .attention_k256_m128n256_incumbent_logical_projection_hits ==
                fake.long_prefill_attention_k256_incumbent_logical_projection_hits &&
            result.value->timing
                    .attention_k256_m128n256_a_exchange_b4_launch_hits ==
                fake.long_prefill_attention_k256_a_exchange_b4_launch_hits &&
            result.value->timing
                    .attention_k256_m128n256_a_exchange_b4_logical_projection_hits ==
                fake.long_prefill_attention_k256_a_exchange_b4_logical_projection_hits &&
            result.value->timing.gateup_alternating_launch_hits ==
                fake.long_prefill_gateup_alternating_launch_hits &&
            result.value->timing.gateup_ldmatrix_pairfeed_launch_hits ==
                fake.long_prefill_gateup_ldmatrix_pairfeed_launch_hits &&
            result.value->timing.gateup_m128n64_same_cta_launch_hits ==
                fake.long_prefill_gateup_m128n64_same_cta_launch_hits &&
            result.value->timing.gateup_m128n512_fused_quantize_launch_hits ==
                fake.long_prefill_gateup_m128n512_fused_quantize_launch_hits &&
            result.value->timing
                    .gateup_m128n512_paired_ldmatrix_launch_hits ==
                fake.long_prefill_gateup_m128n512_paired_ldmatrix_launch_hits &&
            result.value->timing
                    .gateup_m64n128_register_pipeline_launch_hits ==
                fake.long_prefill_gateup_m64n128_register_pipeline_launch_hits &&
            result.value->timing
                    .gateup_m64n8_paired_warp_register_pipeline_launch_hits ==
                fake.long_prefill_gateup_m64n8_paired_warp_register_pipeline_launch_hits &&
            result.value->timing
                    .down_m128n128_ldmatrix_pairring_launch_hits ==
                fake.long_prefill_down_m128n128_ldmatrix_pairring_launch_hits &&
            result.value->timing.gdn_chunk64_native_launch_hits ==
                fake.long_prefill_gdn_chunk64_native_launch_hits &&
            result.value->timing.gdn_chunk64_native_logical_token_hits ==
                fake.long_prefill_gdn_chunk64_native_logical_token_hits &&
            result.value->timing.gdn_prompt_span_macro_launch_hits ==
                fake.long_prefill_gdn_prompt_span_macro_launch_hits &&
            result.value->timing.gdn_prompt_span_macro_logical_token_hits ==
                fake.long_prefill_gdn_prompt_span_macro_logical_token_hits &&
            has_consistent_prefill_timing(result.value->timing) &&
            correct_result_arm,
        "layer-major admission submits one whole prompt, materializes its "
        "ordered transcript, and finalizes the retained last row once");
  };

  run_shape(480U, runtime::ReferenceLogitsMode::kPredictedTokenOnly);
  run_shape(481U, runtime::ReferenceLogitsMode::kFullStatistics);
  run_shape(512U, runtime::ReferenceLogitsMode::kPredictedTokenOnly);
  run_shape(513U, runtime::ReferenceLogitsMode::kFullStatistics);
  run_shape(4'096U, runtime::ReferenceLogitsMode::kPredictedTokenOnly);

  const auto run_invalid = [&test](const bool install_callback,
                                   const bool all_prompt,
                                   const bool single_arbitrary,
                                   const bool capture_trace,
                                   const std::uint32_t chunk_size,
                                   const bool fail_callback,
                                   const bool omit_timing,
                                   const bool wrong_interval,
                                   const detail::GenerationControlError
                                       expected_error) {
    FakeRunner fake;
    fake.predictions = {runtime::kQwen36ImEndTokenId};
    fake.long_prefill_fail = fail_callback;
    fake.long_prefill_omit_timing = omit_timing;
    fake.long_prefill_wrong_interval = wrong_interval;
    PhaseContext context{&fake};
    detail::PrefillPlan prefill_plan;
    prefill_plan.context = &context;
    prefill_plan.prefix_step = fake_prefix_step;
    prefill_plan.finish_prefill = fake_finish_prefill;
    prefill_plan.prefix_tile = fake_prefix_tile;
    prefill_plan.finish_prefill_from_tile =
        fake_finish_prefill_from_tile;
    prefill_plan.layer_major_prompt =
        install_callback ? fake_layer_major_prompt : nullptr;
    detail::DecodePlan decode_plan;
    decode_plan.context = &context;
    decode_plan.decode_step = fake_decode_step;
    detail::GenerationControlOptions control_options =
        options(1U, 513U, capture_trace, chunk_size);
    control_options.prefill_all_prompt_tokens = all_prompt;
    control_options.prefill_single_arbitrary_tile = single_arbitrary;
    control_options.prefill_layer_major_prompt = true;
    const auto result = detail::run_generation_control(
        std::vector<std::uint32_t>(513U, 7U), control_options,
        prefill_plan, decode_plan);
    test.expect(!result && result.error == expected_error,
                "layer-major admission fails closed on an invalid host "
                "contract or malformed runner result");
  };

  run_invalid(false, true, false, false, 512U, false, false, false,
              detail::GenerationControlError::kInvalidArgument);
  run_invalid(true, false, false, false, 512U, false, false, false,
              detail::GenerationControlError::kInvalidArgument);
  run_invalid(true, true, true, false, 512U, false, false, false,
              detail::GenerationControlError::kInvalidArgument);
  run_invalid(true, true, false, true, 512U, false, false, false,
              detail::GenerationControlError::kInvalidArgument);
  run_invalid(true, true, false, false, 256U, false, false, false,
              detail::GenerationControlError::kInvalidArgument);
  run_invalid(true, true, false, false, 512U, true, false, false,
              detail::GenerationControlError::kRunnerFailure);
  run_invalid(true, true, false, false, 512U, false, true, false,
              detail::GenerationControlError::kMissingTiming);
  run_invalid(true, true, false, false, 512U, false, false, true,
              detail::GenerationControlError::kUnexpectedStep);
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
                  !default_control_options.emit_nvtx_phase_ranges,
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

  runtime::ReferenceEngineOptions partial_a4;
  partial_a4.prefill_a4_payload_path = "payload.a4";
  const runtime::ReferenceEngineCreateResult partial_created =
      runtime::create_reference_engine("unused-model-directory", partial_a4);
  test.expect(!partial_created &&
                  partial_created.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  partial_created.diagnostic.stage ==
                      "prefill_a4_sidecar_options",
              "engine rejects a partial A4 publication request before I/O");

  runtime::ReferenceOneShotOptions partial_one_shot_a4;
  partial_one_shot_a4.prefill_a4_payload_path = "payload.a4";
  const runtime::ReferenceOneShotResult partial_one_shot =
      runtime::generate_reference("unused-model-directory", "prompt",
                                  partial_one_shot_a4);
  test.expect(!partial_one_shot &&
                  partial_one_shot.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  partial_one_shot.diagnostic.stage == "one_shot_options",
              "one-shot rejects a partial A4 publication before asset I/O");

  runtime::ReferenceEngineOptions partial_hybrid;
  partial_hybrid
      .prefill_mlp_k512_paired_gateup_canonical_down_payload_path =
      "hybrid.bin";
  const runtime::ReferenceEngineCreateResult partial_hybrid_created =
      runtime::create_reference_engine("unused-model-directory",
                                       partial_hybrid);
  test.expect(
      !partial_hybrid_created &&
          partial_hybrid_created.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          partial_hybrid_created.diagnostic.stage ==
              "prefill_mlp_k512_paired_gateup_canonical_down_options",
      "engine rejects a partial hybrid publication before asset I/O");

  runtime::ReferenceOneShotOptions partial_one_shot_hybrid;
  partial_one_shot_hybrid
      .prefill_mlp_k512_paired_gateup_canonical_down_receipt_path =
      "hybrid.receipt.json";
  const runtime::ReferenceOneShotResult partial_hybrid_one_shot =
      runtime::generate_reference("unused-model-directory", "prompt",
                                  partial_one_shot_hybrid);
  test.expect(!partial_hybrid_one_shot &&
                  partial_hybrid_one_shot.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  partial_hybrid_one_shot.diagnostic.stage ==
                      "one_shot_options",
              "one-shot rejects a partial hybrid publication before asset "
              "I/O");

  constexpr const char* kHybridSelector =
      "Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_ADMISSION";
  (void)::setenv(kHybridSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult missing_hybrid_paths =
      runtime::create_reference_engine("unused-model-directory", {});
  (void)::unsetenv(kHybridSelector);
  test.expect(
      !missing_hybrid_paths &&
          missing_hybrid_paths.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_hybrid_paths.diagnostic.stage ==
              "prefill_mlp_k512_paired_gateup_canonical_down_options",
      "hybrid runtime selector fails closed when its triplet is absent");

  runtime::ReferenceEngineOptions missing_hybrid_selector;
  missing_hybrid_selector.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  missing_hybrid_selector.prefill_a4_payload_path = "base.bin";
  missing_hybrid_selector.prefill_a4_calibration_policy_path = "base.json";
  missing_hybrid_selector.prefill_a4_receipt_path = "base.receipt.json";
  missing_hybrid_selector
      .prefill_mlp_k512_paired_gateup_canonical_down_payload_path =
      "hybrid.bin";
  missing_hybrid_selector
      .prefill_mlp_k512_paired_gateup_canonical_down_policy_path =
      "hybrid.policy.json";
  missing_hybrid_selector
      .prefill_mlp_k512_paired_gateup_canonical_down_receipt_path =
      "hybrid.receipt.json";
  const runtime::ReferenceEngineCreateResult missing_hybrid_master =
      runtime::create_reference_engine("unused-model-directory",
                                       missing_hybrid_selector);
  test.expect(
      !missing_hybrid_master &&
          missing_hybrid_master.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_hybrid_master.diagnostic.stage ==
              "prefill_mlp_k512_paired_gateup_canonical_down_options",
      "complete hybrid publication fails before asset I/O when its runtime "
      "master selector is absent");

  runtime::ReferenceOneShotOptions missing_one_shot_hybrid_selector;
  missing_one_shot_hybrid_selector.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  missing_one_shot_hybrid_selector.prefill_a4_payload_path = "base.bin";
  missing_one_shot_hybrid_selector.prefill_a4_calibration_policy_path =
      "base.json";
  missing_one_shot_hybrid_selector.prefill_a4_receipt_path =
      "base.receipt.json";
  missing_one_shot_hybrid_selector
      .prefill_mlp_k512_paired_gateup_canonical_down_payload_path =
      "hybrid.bin";
  missing_one_shot_hybrid_selector
      .prefill_mlp_k512_paired_gateup_canonical_down_policy_path =
      "hybrid.policy.json";
  missing_one_shot_hybrid_selector
      .prefill_mlp_k512_paired_gateup_canonical_down_receipt_path =
      "hybrid.receipt.json";
  const runtime::ReferenceOneShotResult missing_one_shot_hybrid_master =
      runtime::generate_reference("unused-model-directory", "prompt",
                                  missing_one_shot_hybrid_selector);
  test.expect(
      !missing_one_shot_hybrid_master &&
          missing_one_shot_hybrid_master.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_one_shot_hybrid_master.diagnostic.stage ==
              "one_shot_options",
      "one-shot complete hybrid publication fails before resident loading "
      "when its runtime master selector is absent");

  runtime::ReferenceEngineOptions partial_projection_major;
  partial_projection_major
      .prefill_mlp_k512_projection_major_gateup_canonical_down_payload_path =
      "projection-major.bin";
  const runtime::ReferenceEngineCreateResult partial_projection_major_created =
      runtime::create_reference_engine("unused-model-directory",
                                       partial_projection_major);
  test.expect(
      !partial_projection_major_created &&
          partial_projection_major_created.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          partial_projection_major_created.diagnostic.stage ==
              "prefill_mlp_k512_projection_major_gateup_canonical_down_options",
      "engine rejects a partial projection-major publication before asset "
      "I/O");

  runtime::ReferenceOneShotOptions partial_one_shot_projection_major;
  partial_one_shot_projection_major
      .prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_path =
      "projection-major.receipt.json";
  const runtime::ReferenceOneShotResult partial_projection_major_one_shot =
      runtime::generate_reference("unused-model-directory", "prompt",
                                  partial_one_shot_projection_major);
  test.expect(!partial_projection_major_one_shot &&
                  partial_projection_major_one_shot.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  partial_projection_major_one_shot.diagnostic.stage ==
                      "one_shot_options",
              "one-shot rejects a partial projection-major publication "
              "before asset I/O");

  constexpr const char* kProjectionMajorSelector =
      "Q3X_RUN_A4W4_MLP_K512_PROJECTION_MAJOR_GATEUP_CANONICAL_DOWN_ADMISSION";
  constexpr const char* kRegisterPipelineSelector =
      "Q3X_RUN_A4W4_GATEUP_K512_M64N128_REGISTER_PIPELINE_ADMISSION";
  (void)::setenv(kProjectionMajorSelector, "1", 1);
  (void)::setenv(kRegisterPipelineSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult missing_projection_major_paths =
      runtime::create_reference_engine("unused-model-directory", {});
  (void)::unsetenv(kRegisterPipelineSelector);
  (void)::unsetenv(kProjectionMajorSelector);
  test.expect(
      !missing_projection_major_paths &&
          missing_projection_major_paths.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_projection_major_paths.diagnostic.stage ==
              "prefill_mlp_k512_projection_major_gateup_canonical_down_options",
      "projection-major runtime selectors fail closed when their triplet "
      "is absent");

  runtime::ReferenceEngineOptions missing_projection_major_selector;
  missing_projection_major_selector.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  missing_projection_major_selector.prefill_a4_payload_path = "base.bin";
  missing_projection_major_selector.prefill_a4_calibration_policy_path =
      "base.json";
  missing_projection_major_selector.prefill_a4_receipt_path =
      "base.receipt.json";
  missing_projection_major_selector
      .prefill_mlp_k512_projection_major_gateup_canonical_down_payload_path =
      "projection-major.bin";
  missing_projection_major_selector
      .prefill_mlp_k512_projection_major_gateup_canonical_down_policy_path =
      "projection-major.policy.json";
  missing_projection_major_selector
      .prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_path =
      "projection-major.receipt.json";
  const runtime::ReferenceEngineCreateResult missing_projection_major_master =
      runtime::create_reference_engine("unused-model-directory",
                                       missing_projection_major_selector);
  test.expect(
      !missing_projection_major_master &&
          missing_projection_major_master.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_projection_major_master.diagnostic.stage ==
              "prefill_mlp_k512_projection_major_gateup_canonical_down_options",
      "complete projection-major publication fails before asset I/O when "
      "its runtime selectors are absent");

  runtime::ReferenceEngineOptions aliased_projection_major_paths =
      missing_projection_major_selector;
  aliased_projection_major_paths
      .prefill_mlp_k512_projection_major_gateup_canonical_down_payload_path =
      "projection-major.same";
  aliased_projection_major_paths
      .prefill_mlp_k512_projection_major_gateup_canonical_down_policy_path =
      "projection-major.same";
  aliased_projection_major_paths
      .prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_path =
      "projection-major.same";
  const runtime::ReferenceEngineCreateResult aliased_projection_major =
      runtime::create_reference_engine("unused-model-directory",
                                       aliased_projection_major_paths);
  test.expect(
      !aliased_projection_major &&
          aliased_projection_major.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          aliased_projection_major.diagnostic.stage ==
              "prefill_mlp_k512_projection_major_gateup_canonical_down_options",
      "projection-major payload, policy, and receipt paths must remain "
      "distinct");

  (void)::setenv(kProjectionMajorSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult missing_register_pipeline_leaf =
      runtime::create_reference_engine("unused-model-directory",
                                       missing_projection_major_selector);
  (void)::unsetenv(kProjectionMajorSelector);
  test.expect(!missing_register_pipeline_leaf &&
                  missing_register_pipeline_leaf.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  missing_register_pipeline_leaf.diagnostic.stage ==
                      "prefill_mlp_k512_leaf_selectors",
              "projection-major master cannot run without the independent "
              "register-pipeline leaf selector");

  runtime::ReferenceOneShotOptions
      missing_one_shot_projection_major_selector;
  missing_one_shot_projection_major_selector.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  missing_one_shot_projection_major_selector.prefill_a4_payload_path =
      "base.bin";
  missing_one_shot_projection_major_selector
      .prefill_a4_calibration_policy_path = "base.json";
  missing_one_shot_projection_major_selector.prefill_a4_receipt_path =
      "base.receipt.json";
  missing_one_shot_projection_major_selector
      .prefill_mlp_k512_projection_major_gateup_canonical_down_payload_path =
      "projection-major.bin";
  missing_one_shot_projection_major_selector
      .prefill_mlp_k512_projection_major_gateup_canonical_down_policy_path =
      "projection-major.policy.json";
  missing_one_shot_projection_major_selector
      .prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_path =
      "projection-major.receipt.json";
  const runtime::ReferenceOneShotResult
      missing_one_shot_projection_major_master =
          runtime::generate_reference(
              "unused-model-directory", "prompt",
              missing_one_shot_projection_major_selector);
  test.expect(
      !missing_one_shot_projection_major_master &&
          missing_one_shot_projection_major_master.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_one_shot_projection_major_master.diagnostic.stage ==
              "one_shot_options",
      "one-shot complete projection-major publication fails before resident "
      "loading when its runtime selectors are absent");

  runtime::ReferenceOneShotOptions one_shot_projection_major_without_base;
  one_shot_projection_major_without_base.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  one_shot_projection_major_without_base
      .prefill_mlp_k512_projection_major_gateup_canonical_down_payload_path =
      "projection-major.bin";
  one_shot_projection_major_without_base
      .prefill_mlp_k512_projection_major_gateup_canonical_down_policy_path =
      "projection-major.policy.json";
  one_shot_projection_major_without_base
      .prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_path =
      "projection-major.receipt.json";
  (void)::setenv(kProjectionMajorSelector, "1", 1);
  (void)::setenv(kRegisterPipelineSelector, "1", 1);
  const runtime::ReferenceOneShotResult
      one_shot_projection_major_missing_base = runtime::generate_reference(
          "unused-model-directory", "prompt",
          one_shot_projection_major_without_base);
  (void)::unsetenv(kRegisterPipelineSelector);
  (void)::unsetenv(kProjectionMajorSelector);
  test.expect(!one_shot_projection_major_missing_base &&
                  one_shot_projection_major_missing_base.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  one_shot_projection_major_missing_base.diagnostic.stage ==
                      "one_shot_options",
              "one-shot projection-major publication requires the explicit "
              "K256 A4 base before asset loading");

  (void)::setenv(kRegisterPipelineSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult orphan_register_pipeline =
      runtime::create_reference_engine("unused-model-directory", {});
  (void)::unsetenv(kRegisterPipelineSelector);
  test.expect(!orphan_register_pipeline &&
                  orphan_register_pipeline.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  orphan_register_pipeline.diagnostic.stage ==
                      "prefill_mlp_k512_leaf_selectors",
              "register-pipeline leaf selector cannot run without its "
              "projection-major publication master");

  (void)::setenv(kProjectionMajorSelector, "1", 1);
  (void)::setenv(kRegisterPipelineSelector, "1", 1);
  runtime::ReferenceEngineOptions projection_major_without_base_options;
  projection_major_without_base_options.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  projection_major_without_base_options
      .prefill_mlp_k512_projection_major_gateup_canonical_down_payload_path =
      "projection-major.bin";
  projection_major_without_base_options
      .prefill_mlp_k512_projection_major_gateup_canonical_down_policy_path =
      "projection-major.policy.json";
  projection_major_without_base_options
      .prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_path =
      "projection-major.receipt.json";
  const runtime::ReferenceEngineCreateResult projection_major_without_base =
      runtime::create_reference_engine("unused-model-directory",
                                       projection_major_without_base_options);
  (void)::unsetenv(kRegisterPipelineSelector);
  (void)::unsetenv(kProjectionMajorSelector);
  test.expect(
      !projection_major_without_base &&
          projection_major_without_base.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          projection_major_without_base.diagnostic.stage ==
              "prefill_mlp_k512_projection_major_gateup_canonical_down_options",
      "projection-major publication requires an explicit K256 A4 base "
      "before model I/O");

  runtime::ReferenceEngineOptions conflicting_projection_publications =
      missing_projection_major_selector;
  conflicting_projection_publications.prefill_mlp_k512_payload_path =
      "v1.bin";
  conflicting_projection_publications.prefill_mlp_k512_policy_path =
      "v1.policy.json";
  conflicting_projection_publications.prefill_mlp_k512_receipt_path =
      "v1.receipt.json";
  (void)::setenv(kProjectionMajorSelector, "1", 1);
  (void)::setenv(kRegisterPipelineSelector, "1", 1);
  (void)::setenv("Q3X_RUN_A4W4_MLP_K512_ADMISSION", "1", 1);
  const runtime::ReferenceEngineCreateResult conflicting_projection_layouts =
      runtime::create_reference_engine("unused-model-directory",
                                       conflicting_projection_publications);
  (void)::unsetenv("Q3X_RUN_A4W4_MLP_K512_ADMISSION");
  (void)::unsetenv(kRegisterPipelineSelector);
  (void)::unsetenv(kProjectionMajorSelector);
  test.expect(
      !conflicting_projection_layouts &&
          conflicting_projection_layouts.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          (conflicting_projection_layouts.diagnostic.stage ==
               "prefill_mlp_k512_leaf_selectors" ||
           conflicting_projection_layouts.diagnostic.stage ==
               "prefill_mlp_k512_overlay_options"),
      "projection-major and v1 K512 MLP publications are mutually "
      "exclusive before model I/O");

  constexpr const char* kPairringSelector =
      "Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION";
  constexpr const char* kMlpK512V1Selector =
      "Q3X_RUN_A4W4_MLP_K512_ADMISSION";
  constexpr const char* kDownM16N64V2Selector =
      "Q3X_RUN_A4W4_DOWN_K512_M16N64_V2_ADMISSION";
  runtime::ReferenceEngineOptions pairring_without_v1;
  pairring_without_v1.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  pairring_without_v1.prefill_a4_payload_path = "base.bin";
  pairring_without_v1.prefill_a4_calibration_policy_path = "base.json";
  pairring_without_v1.prefill_a4_receipt_path = "base.receipt.json";
  pairring_without_v1.prefill_mlp_k512_payload_path = "mlp.bin";
  pairring_without_v1.prefill_mlp_k512_policy_path = "mlp.policy.json";
  pairring_without_v1.prefill_mlp_k512_receipt_path =
      "mlp.receipt.json";
  (void)::setenv(kPairringSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult missing_pairring_v1_master =
      runtime::create_reference_engine("unused-model-directory",
                                       pairring_without_v1);
  (void)::unsetenv(kPairringSelector);
  test.expect(
      !missing_pairring_v1_master &&
          missing_pairring_v1_master.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_pairring_v1_master.diagnostic.stage ==
              "prefill_mlp_k512_leaf_selectors",
      "independent pair-ring Down rejects a complete v1 triplet without "
      "the v1 runtime master before model I/O");

  runtime::ReferenceOneShotOptions one_shot_pairring_without_v1;
  one_shot_pairring_without_v1.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  one_shot_pairring_without_v1.prefill_a4_payload_path = "base.bin";
  one_shot_pairring_without_v1.prefill_a4_calibration_policy_path =
      "base.json";
  one_shot_pairring_without_v1.prefill_a4_receipt_path =
      "base.receipt.json";
  one_shot_pairring_without_v1.prefill_mlp_k512_payload_path = "mlp.bin";
  one_shot_pairring_without_v1.prefill_mlp_k512_policy_path =
      "mlp.policy.json";
  one_shot_pairring_without_v1.prefill_mlp_k512_receipt_path =
      "mlp.receipt.json";
  (void)::setenv(kPairringSelector, "1", 1);
  const runtime::ReferenceOneShotResult one_shot_missing_pairring_v1_master =
      runtime::generate_reference("unused-model-directory", "prompt",
                                  one_shot_pairring_without_v1);
  (void)::unsetenv(kPairringSelector);
  test.expect(
      !one_shot_missing_pairring_v1_master &&
          one_shot_missing_pairring_v1_master.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          one_shot_missing_pairring_v1_master.diagnostic.stage ==
              "one_shot_options",
      "one-shot independent pair-ring Down rejects a missing v1 master "
      "before tokenizer or resident loading");

  (void)::setenv(kPairringSelector, "1", 1);
  (void)::setenv(kMlpK512V1Selector, "1", 1);
  (void)::setenv(kDownM16N64V2Selector, "1", 1);
  const runtime::ReferenceEngineCreateResult conflicting_pairring_down =
      runtime::create_reference_engine("unused-model-directory",
                                       pairring_without_v1);
  (void)::unsetenv(kDownM16N64V2Selector);
  (void)::unsetenv(kMlpK512V1Selector);
  (void)::unsetenv(kPairringSelector);
  test.expect(
      !conflicting_pairring_down &&
          conflicting_pairring_down.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          conflicting_pairring_down.diagnostic.stage ==
              "prefill_mlp_k512_leaf_selectors",
      "pair-ring and M16N64 v2 Down selectors conflict before model I/O");

  constexpr const char* kGateupAlternatingSelector =
      "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION";
  constexpr const char* kGateupLdmatrixPairfeedSelector =
      "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION";
  constexpr const char* kGateupM128N64SameCtaSelector =
      "Q3X_RUN_A4W4_GATEUP_K512_M128N64_SAME_CTA_ADMISSION";
  constexpr const char* kGateupM128N512FusedQuantizeSelector =
      "Q3X_RUN_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_ADMISSION";
  (void)::setenv(kGateupLdmatrixPairfeedSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult missing_pairfeed_v1_master =
      runtime::create_reference_engine("unused-model-directory",
                                       pairring_without_v1);
  (void)::unsetenv(kGateupLdmatrixPairfeedSelector);
  test.expect(
      !missing_pairfeed_v1_master &&
          missing_pairfeed_v1_master.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_pairfeed_v1_master.diagnostic.stage ==
              "prefill_mlp_k512_leaf_selectors",
      "LDSM pair-feed Gate+Up rejects a complete v1 triplet without the "
      "v1 runtime master before model I/O");

  (void)::setenv(kGateupM128N64SameCtaSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult missing_same_cta_v1_master =
      runtime::create_reference_engine("unused-model-directory",
                                       pairring_without_v1);
  (void)::unsetenv(kGateupM128N64SameCtaSelector);
  test.expect(
      !missing_same_cta_v1_master &&
          missing_same_cta_v1_master.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_same_cta_v1_master.diagnostic.stage ==
              "prefill_mlp_k512_leaf_selectors",
      "M128N64 same-CTA Gate+Up rejects a complete v1 triplet without the "
      "v1 runtime master before model I/O");

  (void)::setenv(kGateupM128N512FusedQuantizeSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult missing_fused_quantize_v1_master =
      runtime::create_reference_engine("unused-model-directory",
                                       pairring_without_v1);
  (void)::unsetenv(kGateupM128N512FusedQuantizeSelector);
  test.expect(
      !missing_fused_quantize_v1_master &&
          missing_fused_quantize_v1_master.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_fused_quantize_v1_master.diagnostic.stage ==
              "prefill_mlp_k512_leaf_selectors",
      "M128N512 fused-quantize Gate+Up rejects a missing v1 runtime master "
      "before model I/O");

  (void)::setenv(kMlpK512V1Selector, "1", 1);
  (void)::setenv(kGateupLdmatrixPairfeedSelector, "1", 1);
  (void)::setenv(kGateupM128N512FusedQuantizeSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult conflicting_fused_quantize_gate =
      runtime::create_reference_engine("unused-model-directory",
                                       pairring_without_v1);
  (void)::unsetenv(kGateupM128N512FusedQuantizeSelector);
  (void)::unsetenv(kGateupLdmatrixPairfeedSelector);
  (void)::unsetenv(kMlpK512V1Selector);
  test.expect(
      !conflicting_fused_quantize_gate &&
          conflicting_fused_quantize_gate.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          conflicting_fused_quantize_gate.diagnostic.stage ==
              "prefill_mlp_k512_leaf_selectors",
      "M128N512 fused-quantize and pair-feed Gate+Up selectors conflict "
      "before model I/O");

  (void)::setenv(kMlpK512V1Selector, "1", 1);
  (void)::setenv(kGateupLdmatrixPairfeedSelector, "1", 1);
  (void)::setenv(kGateupM128N64SameCtaSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult conflicting_same_cta_gate =
      runtime::create_reference_engine("unused-model-directory",
                                       pairring_without_v1);
  (void)::unsetenv(kGateupM128N64SameCtaSelector);
  (void)::unsetenv(kGateupLdmatrixPairfeedSelector);
  (void)::unsetenv(kMlpK512V1Selector);
  test.expect(
      !conflicting_same_cta_gate &&
          conflicting_same_cta_gate.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          conflicting_same_cta_gate.diagnostic.stage ==
              "prefill_mlp_k512_leaf_selectors",
      "M128N64 same-CTA and pair-feed Gate+Up selectors conflict before "
      "model I/O");

  (void)::setenv(kMlpK512V1Selector, "1", 1);
  (void)::setenv(kGateupLdmatrixPairfeedSelector, "1", 1);
  (void)::setenv(kGateupM128N64SameCtaSelector, "enabled", 1);
  const runtime::ReferenceEngineCreateResult non_exact_same_cta_ignored =
      runtime::create_reference_engine("unused-model-directory",
                                       pairring_without_v1);
  (void)::unsetenv(kGateupM128N64SameCtaSelector);
  (void)::unsetenv(kGateupLdmatrixPairfeedSelector);
  (void)::unsetenv(kMlpK512V1Selector);
  test.expect(
      !non_exact_same_cta_ignored &&
          non_exact_same_cta_ignored.diagnostic.stage !=
              "prefill_mlp_k512_leaf_selectors",
      "M128N64 same-CTA selector accepts only the exact value '1'");

  (void)::setenv(kGateupAlternatingSelector, "1", 1);
  (void)::setenv(kGateupLdmatrixPairfeedSelector, "1", 1);
  const runtime::ReferenceEngineCreateResult conflicting_gateup_k256 =
      runtime::create_reference_engine("unused-model-directory", {});
  (void)::unsetenv(kGateupLdmatrixPairfeedSelector);
  (void)::unsetenv(kGateupAlternatingSelector);
  test.expect(
      !conflicting_gateup_k256 &&
          conflicting_gateup_k256.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          conflicting_gateup_k256.diagnostic.stage ==
              "prefill_mlp_k512_leaf_selectors",
      "alternating and LDSM pair-feed Gate+Up selectors conflict before "
      "model I/O");

  constexpr const char* kAttentionK256IncumbentSelector =
      "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION";
  constexpr const char* kAttentionK256AExchangeB4Selector =
      "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_A_EXCHANGE_B4_ADMISSION";
  (void)::setenv(kAttentionK256IncumbentSelector, "1", 1);
  (void)::setenv(kAttentionK256AExchangeB4Selector, "1", 1);
  const runtime::ReferenceEngineCreateResult conflicting_attention_k256 =
      runtime::create_reference_engine("unused-model-directory", {});
  (void)::unsetenv(kAttentionK256AExchangeB4Selector);
  (void)::unsetenv(kAttentionK256IncumbentSelector);
  test.expect(
      !conflicting_attention_k256 &&
          conflicting_attention_k256.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          conflicting_attention_k256.diagnostic.stage ==
              "prefill_attention_k256_leaf_selectors",
      "incumbent and A-exchange/B4 Attention selectors conflict before "
      "model I/O");

  runtime::ReferenceEngineOptions attention_k256_without_consumer;
  attention_k256_without_consumer.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  attention_k256_without_consumer.prefill_a4_payload_path = "base.bin";
  attention_k256_without_consumer.prefill_a4_calibration_policy_path =
      "base.policy.json";
  attention_k256_without_consumer.prefill_a4_receipt_path =
      "base.receipt.json";
  (void)::setenv(kAttentionK256AExchangeB4Selector, "1", 1);
  const runtime::ReferenceEngineCreateResult missing_attention_consumer =
      runtime::create_reference_engine("unused-model-directory",
                                       attention_k256_without_consumer);
  (void)::unsetenv(kAttentionK256AExchangeB4Selector);
  test.expect(
      !missing_attention_consumer &&
          missing_attention_consumer.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          missing_attention_consumer.diagnostic.stage ==
              "prefill_attention_k256_leaf_selectors" &&
          missing_attention_consumer.diagnostic.message.find(
              "complete K256 consumer contract") != std::string::npos,
      "A-exchange/B4 Attention rejects an incomplete K256 consumer "
      "contract before model I/O");

  (void)::setenv(kAttentionK256IncumbentSelector, "1", 1);
  (void)::setenv(kAttentionK256AExchangeB4Selector, "1", 1);
  const runtime::ReferenceOneShotResult
      conflicting_one_shot_attention_k256 = runtime::generate_reference(
          "unused-model-directory", "prompt", {});
  (void)::unsetenv(kAttentionK256AExchangeB4Selector);
  (void)::unsetenv(kAttentionK256IncumbentSelector);
  test.expect(
      !conflicting_one_shot_attention_k256 &&
          conflicting_one_shot_attention_k256.diagnostic.code ==
              runtime::ReferenceEngineError::kInvalidArgument &&
          conflicting_one_shot_attention_k256.diagnostic.stage ==
              "one_shot_options",
      "one-shot Attention selector conflict fails before tokenizer or "
      "overlapped resident loading");

  runtime::ReferenceEngineOptions wrong_backend_a4;
  wrong_backend_a4.prefill_a4_payload_path = "payload.a4";
  wrong_backend_a4.prefill_a4_calibration_policy_path = "policy.json";
  const runtime::ReferenceEngineCreateResult wrong_backend_created =
      runtime::create_reference_engine("unused-model-directory",
                                       wrong_backend_a4);
  test.expect(!wrong_backend_created &&
                  wrong_backend_created.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  wrong_backend_created.diagnostic.stage ==
                      "prefill_a4_sidecar_options",
              "engine rejects A4 residency without the SM87 backend before "
              "model I/O");

  runtime::ReferenceEngineOptions wrong_chunk_a4;
  wrong_chunk_a4.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  wrong_chunk_a4.prefill_a4_payload_path = "payload.a4";
  wrong_chunk_a4.prefill_a4_calibration_policy_path = "policy.json";
  wrong_chunk_a4.request_options.prefill_chunk_size = 1U;
  const runtime::ReferenceEngineCreateResult wrong_chunk_created =
      runtime::create_reference_engine("unused-model-directory",
                                       wrong_chunk_a4);
  test.expect(!wrong_chunk_created &&
                  wrong_chunk_created.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  wrong_chunk_created.diagnostic.stage ==
                      "prefill_a4_sidecar_options",
              "engine rejects a non-C512 A4 request before model I/O");

  runtime::ReferenceOneShotOptions wrong_chunk_one_shot_a4;
  wrong_chunk_one_shot_a4.projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
  wrong_chunk_one_shot_a4.prefill_a4_payload_path = "payload.a4";
  wrong_chunk_one_shot_a4.prefill_a4_calibration_policy_path =
      "policy.json";
  wrong_chunk_one_shot_a4.generation.prefill_chunk_size = 1U;
  const runtime::ReferenceOneShotResult wrong_chunk_one_shot =
      runtime::generate_reference("unused-model-directory", "prompt",
                                  wrong_chunk_one_shot_a4);
  test.expect(!wrong_chunk_one_shot &&
                  wrong_chunk_one_shot.diagnostic.code ==
                      runtime::ReferenceEngineError::kInvalidArgument &&
                  wrong_chunk_one_shot.diagnostic.stage ==
                      "one_shot_options",
              "one-shot rejects a non-C512 A4 request before asset I/O");

  const runtime::ReferenceEngineLoadStats empty_load;
  test.expect(!empty_load.prefill_a4_sidecars_requested &&
                  !empty_load.prefill_a4_sidecars_enabled &&
                  empty_load.prefill_a4_sidecar_projections == 0U &&
                  empty_load.prefill_a4_sidecar_bytes == 0U &&
                  empty_load.prefill_a4_sidecar_copy_chunks == 0U &&
                  empty_load.prefill_a4_physical_layout.empty() &&
                  empty_load.prefill_a4_manifest_sha256.empty() &&
                  empty_load.prefill_a4_policy_sha256.empty() &&
                  empty_load.prefill_a4_payload_sha256.empty() &&
                  !empty_load
                       .prefill_mlp_k512_paired_gateup_canonical_down_overlay_requested &&
                  !empty_load
                       .prefill_mlp_k512_paired_gateup_canonical_down_overlay_enabled &&
                  empty_load
                          .prefill_mlp_k512_paired_gateup_canonical_down_overlay_layers ==
                      0U &&
                  empty_load
                      .prefill_mlp_k512_paired_gateup_canonical_down_overlay_layout
                      .empty() &&
                  empty_load
                      .prefill_mlp_k512_paired_gateup_canonical_down_overlay_receipt_sha256
                      .empty() &&
                  !empty_load
                       .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_requested &&
                  !empty_load
                       .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_enabled &&
                  empty_load
                          .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_layers ==
                      0U &&
                  empty_load
                          .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_bytes ==
                      0U &&
                  empty_load
                          .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_copy_chunks ==
                      0U &&
                  empty_load
                      .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_layout
                      .empty() &&
                  empty_load
                      .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_manifest_sha256
                      .empty() &&
                  empty_load
                      .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_policy_sha256
                      .empty() &&
                  empty_load
                      .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_payload_sha256
                      .empty() &&
                  empty_load
                      .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_receipt_sha256
                      .empty() &&
                  empty_load
                      .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_source_v1_receipt_sha256
                      .empty() &&
                  empty_load
                          .prefill_mlp_k512_projection_major_gateup_canonical_down_overlay_milliseconds ==
                      0.0 &&
                  empty_load.request_long_prefill_token_capacity == 0U &&
                  empty_load
                          .request_long_prefill_projection_span_capacity ==
                      0U &&
                  !empty_load.optimized_prefill_disabled,
              "A4 load statistics default to an unrequested empty route");
}

void test_optimized_prefill_engine_derivation(TestContext& test) {
  struct Case {
    std::uint32_t max_sequence_length;
    std::uint32_t expected_long_capacity;
    std::uint32_t expected_span_capacity;
    std::uint64_t expected_arena_bytes;
    bool expected_layer_major;
    bool expected_projection_span;
  };
  constexpr std::array<Case, 7U> kCases = {{
      {512U, 0U, 0U, 207'486'976U, false, false},
      {513U, 513U, 512U, 236'933'632U, true, true},
      {2'048U, 2'048U, 2'048U, 444'366'848U, true, true},
      {4'095U, 4'095U, 3'584U, 695'971'840U, true, true},
      {4'096U, 4'096U, 4'096U, 721'059'840U, true, true},
      {40'960U, 40'960U, 4'096U, 3'904'274'432U, true, true},
      {40'961U, 40'960U, 4'096U, 3'904'340'736U, false, false},
  }};

  for (const Case& item : kCases) {
    runtime::RequestMemoryOptions requested;
    requested.max_sequence_length = item.max_sequence_length;
    requested.prefill_chunk_size =
        runtime::kLongPrefillLayerMajorTileTokens;
    requested.max_arena_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    const runtime::RequestMemoryOptions derived =
        detail::derive_optimized_prefill_request_memory_options(
            requested, true);
    const runtime::RequestPlanResult plan =
        runtime::build_request_memory_plan(derived);

    runtime::LongPrefillLayerMajorRouteQuery query;
    query.runtime_enabled = true;
    query.projection_backend =
        runtime::ProjectionBackend::kSm87WeightOnly;
    query.prompt_token_count = item.max_sequence_length;
    query.prefill_chunk_size = derived.prefill_chunk_size;
    query.hidden_token_capacity = derived.long_prefill_token_capacity;
    query.hidden_buffer_count =
        derived.long_prefill_token_capacity == 0U
            ? 0U
            : runtime::kRequestLongPrefillHiddenBufferCount;
    const bool layer_major =
        runtime::select_long_prefill_layer_major_route(query) ==
        runtime::LongPrefillLayerMajorRoute::kLayerMajorAdmission;
    const bool projection_span =
        runtime::use_long_prefill_projection_span_route(
            item.max_sequence_length,
            derived.long_prefill_projection_span_capacity, true, false);

    test.expect(
        derived.enable_a4_prefill_workspace &&
            derived.long_prefill_token_capacity ==
                item.expected_long_capacity &&
            derived.long_prefill_projection_span_capacity ==
                item.expected_span_capacity &&
            plan && plan.value->arena_bytes == item.expected_arena_bytes &&
            layer_major ==
                (item.expected_layer_major &&
                 runtime::long_prefill_layer_major_build_enabled()) &&
            projection_span == item.expected_projection_span,
        "central A4 Prefill derivation preserves exact boundary capacities, "
        "arena bytes, and safe route selection");

    query.capture_trace = true;
    test.expect(
        runtime::select_long_prefill_layer_major_route(query) ==
                runtime::LongPrefillLayerMajorRoute::kTileMajorFallback &&
            !runtime::use_long_prefill_projection_span_route(
                item.max_sequence_length,
                derived.long_prefill_projection_span_capacity, true, true),
        "trace and unified-disable comparators retain safe Prefill fallback");
  }

  test.expect(
      !runtime::use_long_prefill_projection_span_route(
          512U, 512U, true, false) &&
          runtime::use_long_prefill_projection_span_route(
              513U, 512U, true, false) &&
          runtime::use_long_prefill_projection_span_route(
              1'804U, 1'536U, true, false) &&
          runtime::use_long_prefill_projection_span_route(
              3'987U, 3'584U, true, false) &&
          runtime::use_long_prefill_projection_span_route(
              40'960U, 4'096U, true, false) &&
          !runtime::use_long_prefill_projection_span_route(
              40'961U, 4'096U, true, false),
      "whole-M selector admits arbitrary P513..P40960 boundaries");
  test.expect(
      !runtime::use_long_prefill_projection_span_route(
          479U, 512U, true, false, true, true) &&
          runtime::use_long_prefill_projection_span_route(
              480U, 512U, true, false, true, true) &&
          runtime::use_long_prefill_projection_span_route(
              481U, 512U, true, false, true, true) &&
          runtime::use_long_prefill_projection_span_route(
              512U, 512U, true, false, true, true) &&
          !runtime::use_long_prefill_projection_span_route(
              512U, 512U, true, false, false, true) &&
          !runtime::use_long_prefill_projection_span_route(
              512U, 512U, true, false, true, false) &&
          !runtime::use_long_prefill_projection_span_route(
              512U, 512U, true, true, true, true),
      "short whole-M selector requires P480..P512, explicit admission, "
      "authenticated K128, and enabled optimized dispatch");
  test.expect(
      !runtime::use_long_prefill_projection_span_route(
          1'804U, 0U, true, false) &&
          !runtime::use_long_prefill_projection_span_route(
              1'804U, 511U, true, false) &&
          !runtime::use_long_prefill_projection_span_route(
              1'804U, 513U, true, false),
      "whole-M selector rejects missing, undersized, and non-C512 spans");

  runtime::RequestMemoryOptions ordinary;
  ordinary.max_sequence_length = 4'096U;
  ordinary.prefill_chunk_size =
      runtime::kLongPrefillLayerMajorTileTokens;
  ordinary.long_prefill_token_capacity = 777U;
  const runtime::RequestMemoryOptions preserved =
      detail::derive_optimized_prefill_request_memory_options(
          ordinary, false);
  test.expect(!preserved.enable_a4_prefill_workspace &&
                  preserved.long_prefill_token_capacity == 777U &&
                  preserved.long_prefill_projection_span_capacity == 0U,
              "central derivation leaves non-A4 callers byte-for-byte "
              "unchanged");

  runtime::RequestMemoryOptions short_requested;
  short_requested.max_sequence_length = 512U;
  short_requested.prefill_chunk_size =
      runtime::kLongPrefillLayerMajorTileTokens;
  short_requested.max_arena_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
  const runtime::RequestMemoryOptions short_default =
      detail::derive_optimized_prefill_request_memory_options(
          short_requested, true, false);
  const runtime::RequestMemoryOptions short_admitted =
      detail::derive_optimized_prefill_request_memory_options(
          short_requested, true, true);
  const runtime::RequestPlanResult short_default_plan =
      runtime::build_request_memory_plan(short_default);
  const runtime::RequestPlanResult short_admitted_plan =
      runtime::build_request_memory_plan(short_admitted);
  test.expect(
      short_default.long_prefill_token_capacity == 0U &&
          short_default.long_prefill_projection_span_capacity == 0U &&
          short_admitted.long_prefill_token_capacity == 512U &&
          short_admitted.long_prefill_projection_span_capacity == 512U &&
          short_default_plan && short_admitted_plan &&
          short_admitted_plan.value->arena_bytes >
              short_default_plan.value->arena_bytes,
      "short selector alone reserves exact P512 hidden and projection "
      "capacity while the default arena remains unchanged");

  short_requested.max_sequence_length = 511U;
  const runtime::RequestMemoryOptions below_short_capacity =
      detail::derive_optimized_prefill_request_memory_options(
          short_requested, true, true);
  test.expect(below_short_capacity.long_prefill_token_capacity == 0U &&
                  below_short_capacity
                          .long_prefill_projection_span_capacity == 0U,
              "max-sequence P511 cannot reserve the padded P512 contract");
}

}  // namespace

int main() {
  TestContext test;
  test_prefill_decode_and_stop(test);
  test_explicit_phase_plans(test);
  test_all_prompt_tile_admission(test);
  test_layer_major_prompt_admission(test);
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
  test_optimized_prefill_engine_derivation(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " reference engine control test(s) failed\n";
    return 1;
  }
  std::cout << "All reference engine control tests passed\n";
  return 0;
}
