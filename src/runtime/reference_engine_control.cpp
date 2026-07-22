#include "q3x/runtime/reference_engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace q3x::runtime::reference_engine_detail {
namespace {

GenerationControlResult failure(const GenerationControlError error,
                                const ReferenceRunnerStatus status = {}) {
  GenerationControlResult result;
  result.error = error;
  result.runner_status = status;
  return result;
}

bool checked_required_steps(const std::size_t prompt_tokens,
                            const std::uint32_t max_new_tokens,
                            std::uint64_t& output) noexcept {
  const std::uint64_t prompt = static_cast<std::uint64_t>(prompt_tokens);
  const std::uint64_t additional =
      static_cast<std::uint64_t>(max_new_tokens) - 1U;
  if (additional > std::numeric_limits<std::uint64_t>::max() - prompt) {
    return false;
  }
  output = prompt + additional;
  return true;
}

}  // namespace

GenerationControlResult run_generation_control(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationControlOptions& options,
    const PrefillPlan& prefill_plan,
    const DecodePlan& decode_plan) {
  if (prompt_token_ids.empty() || options.max_new_tokens == 0U ||
      options.max_sequence_length == 0U ||
      prefill_plan.prefix_step == nullptr ||
      prefill_plan.finish_prefill == nullptr ||
      decode_plan.decode_step == nullptr ||
      options.prefill_chunk_size == 0U ||
      options.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
      options.stop_token_id >= kReferenceVocabularySize ||
      !is_valid_reference_logits_mode(options.logits_mode)) {
    return failure(GenerationControlError::kInvalidArgument);
  }
  for (const std::uint32_t token : prompt_token_ids) {
    if (token >= kReferenceVocabularySize) {
      return failure(GenerationControlError::kInvalidArgument);
    }
  }

  std::uint64_t required_steps = 0U;
  if (!checked_required_steps(prompt_token_ids.size(),
                              options.max_new_tokens,
                              required_steps)) {
    return failure(GenerationControlError::kArithmeticOverflow);
  }
  if (required_steps > options.max_sequence_length) {
    return failure(GenerationControlError::kCapacityExceeded);
  }

  try {
    GenerationControl control;
    control.generated_token_ids.reserve(options.max_new_tokens);
    control.steps.reserve(static_cast<std::size_t>(required_steps));

    auto execute = [&](void* const phase_context,
                       const StepFunction phase_step,
                       const std::uint32_t input_token,
                       const bool compute_logits,
                       double& elapsed,
                       std::uint32_t& predicted_token,
                       ReferenceRunnerStatus& failed_status)
        -> GenerationControlError {
      ReferenceStepOptions step_options;
      step_options.compute_logits = compute_logits;
      step_options.capture_trace = options.capture_trace;
      step_options.measure_timing = true;
      step_options.logits_mode = options.logits_mode;
      ReferenceStepOutcome outcome =
          phase_step(phase_context, input_token, step_options);
      if (!outcome) {
        failed_status = outcome.status;
        return GenerationControlError::kRunnerFailure;
      }
      const std::size_t expected_position = control.steps.size();
      if (outcome.value->position !=
              static_cast<std::uint32_t>(expected_position) ||
          outcome.value->input_token_id != input_token ||
          (!compute_logits &&
           (outcome.value->logits.has_value() ||
            outcome.value->prediction.has_value()))) {
        return GenerationControlError::kUnexpectedStep;
      }
      if (!outcome.value->timing.has_value()) {
        return GenerationControlError::kMissingTiming;
      }
      if (compute_logits) {
        if (options.logits_mode == ReferenceLogitsMode::kFullStatistics) {
          if (outcome.value->prediction.has_value()) {
            return GenerationControlError::kUnexpectedStep;
          }
          if (!outcome.value->logits.has_value()) {
            return GenerationControlError::kMissingLogits;
          }
        } else {
          if (outcome.value->logits.has_value()) {
            return GenerationControlError::kUnexpectedStep;
          }
          if (!outcome.value->prediction.has_value()) {
            return GenerationControlError::kMissingPrediction;
          }
        }
      }
      elapsed = outcome.value->timing->elapsed_milliseconds;
      if (!std::isfinite(elapsed) || elapsed < 0.0) {
        return GenerationControlError::kUnexpectedStep;
      }
      if (compute_logits) {
        predicted_token =
            options.logits_mode == ReferenceLogitsMode::kFullStatistics
                ? outcome.value->logits->predicted_token_id
                : outcome.value->prediction->predicted_token_id;
        if (predicted_token >= kReferenceVocabularySize) {
          return GenerationControlError::kUnexpectedStep;
        }
      }
      control.steps.emplace_back(std::move(*outcome.value));
      return GenerationControlError::kNone;
    };

    ReferenceRunnerStatus runner_failure_status;
    std::uint32_t predicted_token = 0U;

    const std::size_t prefix_token_count = prompt_token_ids.size() - 1U;
    const std::size_t effective_prefill_chunk_size =
        options.capture_trace ? 1U : options.prefill_chunk_size;
    if (effective_prefill_chunk_size > 1U && prefix_token_count != 0U &&
        prefill_plan.prefix_tile == nullptr) {
      return failure(GenerationControlError::kInvalidArgument);
    }

    std::size_t prefix_index = 0U;
    if (effective_prefill_chunk_size > 1U) {
      while (prefix_index < prefix_token_count) {
        const std::size_t tile_token_count =
            std::min(effective_prefill_chunk_size,
                     prefix_token_count - prefix_index);
        ReferencePrefillTileOptions tile_options;
        tile_options.measure_timing = true;
        ReferencePrefillTileOutcome outcome = prefill_plan.prefix_tile(
            prefill_plan.context,
            prompt_token_ids.data() + prefix_index,
            tile_token_count, tile_options);
        if (!outcome) {
          return failure(GenerationControlError::kRunnerFailure,
                         outcome.status);
        }
        if (outcome.value->step_count != tile_token_count ||
            !outcome.value->timing.has_value()) {
          return failure(outcome.value->timing.has_value()
                             ? GenerationControlError::kUnexpectedStep
                             : GenerationControlError::kMissingTiming);
        }
        const double tile_elapsed =
            outcome.value->timing->elapsed_milliseconds;
        if (!std::isfinite(tile_elapsed) || tile_elapsed < 0.0) {
          return failure(GenerationControlError::kUnexpectedStep);
        }

        const std::size_t expected_first_position = control.steps.size();
        for (std::size_t tile_index = 0U;
             tile_index < tile_token_count; ++tile_index) {
          const ReferenceStepResult& step = outcome.value->steps[tile_index];
          if (step.position != static_cast<std::uint32_t>(
                                   expected_first_position + tile_index) ||
              step.input_token_id !=
                  prompt_token_ids[prefix_index + tile_index] ||
              step.logits.has_value() ||
              step.prediction.has_value() ||
              (step.timing.has_value() &&
               (!std::isfinite(step.timing->elapsed_milliseconds) ||
                step.timing->elapsed_milliseconds < 0.0))) {
            return failure(GenerationControlError::kUnexpectedStep);
          }
        }

        for (std::size_t tile_index = 0U;
             tile_index < tile_token_count; ++tile_index) {
          control.steps.emplace_back(
              std::move(outcome.value->steps[tile_index]));
        }
        const double accumulated_prefill =
            control.timing.prompt_prefill_milliseconds + tile_elapsed;
        if (!std::isfinite(accumulated_prefill)) {
          return failure(GenerationControlError::kUnexpectedStep);
        }
        control.timing.prompt_prefill_milliseconds = accumulated_prefill;
        prefix_index += tile_token_count;
      }
    } else {
      while (prefix_index < prefix_token_count) {
        double elapsed = 0.0;
        const GenerationControlError error = execute(
            prefill_plan.context, prefill_plan.prefix_step,
            prompt_token_ids[prefix_index], false, elapsed,
            predicted_token, runner_failure_status);
        if (error != GenerationControlError::kNone) {
          return failure(error, runner_failure_status);
        }
        control.timing.prompt_prefill_milliseconds += elapsed;
        ++prefix_index;
      }
    }

    {
      double elapsed = 0.0;
      const GenerationControlError error = execute(
          prefill_plan.context, prefill_plan.finish_prefill,
          prompt_token_ids.back(), true, elapsed,
          predicted_token, runner_failure_status);
      if (error != GenerationControlError::kNone) {
        return failure(error, runner_failure_status);
      }
      control.timing.prompt_prefill_milliseconds += elapsed;
    }

    control.generated_token_ids.push_back(predicted_token);
    control.timing.time_to_first_token_milliseconds =
        control.timing.prompt_prefill_milliseconds;
    control.timing.total_generation_milliseconds =
        control.timing.prompt_prefill_milliseconds;
    if (predicted_token == options.stop_token_id) {
      control.stop_reason = ReferenceStopReason::kImEnd;
    } else {
      while (control.generated_token_ids.size() < options.max_new_tokens) {
        const std::uint32_t input_token =
            control.generated_token_ids.back();
        double elapsed = 0.0;
        const GenerationControlError error = execute(
            decode_plan.context, decode_plan.decode_step,
            input_token, true, elapsed, predicted_token,
            runner_failure_status);
        if (error != GenerationControlError::kNone) {
          return failure(error, runner_failure_status);
        }
        control.generated_token_ids.push_back(predicted_token);
        control.timing.subsequent_token_milliseconds.push_back(elapsed);
        control.timing.decode_after_first_milliseconds += elapsed;
        control.timing.total_generation_milliseconds += elapsed;
        if (predicted_token == options.stop_token_id) {
          control.stop_reason = ReferenceStopReason::kImEnd;
          break;
        }
      }
    }

    GenerationControlResult result;
    result.value.emplace(std::move(control));
    return result;
  } catch (const std::bad_alloc&) {
    return failure(GenerationControlError::kAllocationFailure);
  } catch (const std::length_error&) {
    return failure(GenerationControlError::kAllocationFailure);
  }
}

GenerationControlResult run_generation_control(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationControlOptions& options,
    void* const step_context,
    const StepFunction step_function,
    const PrefillTileFunction prefill_tile_function) {
  PrefillPlan prefill_plan;
  prefill_plan.context = step_context;
  prefill_plan.prefix_step = step_function;
  prefill_plan.finish_prefill = step_function;
  prefill_plan.prefix_tile = prefill_tile_function;

  DecodePlan decode_plan;
  decode_plan.context = step_context;
  decode_plan.decode_step = step_function;
  return run_generation_control(prompt_token_ids, options,
                                prefill_plan, decode_plan);
}

std::size_t generated_text_token_count(
    const std::vector<std::uint32_t>& generated_token_ids,
    const ReferenceStopReason stop_reason,
    const std::uint32_t stop_token_id) noexcept {
  if (stop_reason == ReferenceStopReason::kImEnd &&
      !generated_token_ids.empty() &&
      generated_token_ids.back() == stop_token_id) {
    return generated_token_ids.size() - 1U;
  }
  return generated_token_ids.size();
}

std::string_view to_string(const GenerationControlError error) noexcept {
  switch (error) {
    case GenerationControlError::kNone:
      return "none";
    case GenerationControlError::kInvalidArgument:
      return "invalid_argument";
    case GenerationControlError::kCapacityExceeded:
      return "capacity_exceeded";
    case GenerationControlError::kArithmeticOverflow:
      return "arithmetic_overflow";
    case GenerationControlError::kRunnerFailure:
      return "runner_failure";
    case GenerationControlError::kUnexpectedStep:
      return "unexpected_step";
    case GenerationControlError::kMissingLogits:
      return "missing_logits";
    case GenerationControlError::kMissingTiming:
      return "missing_timing";
    case GenerationControlError::kAllocationFailure:
      return "allocation_failure";
    case GenerationControlError::kMissingPrediction:
      return "missing_prediction";
  }
  return "unknown";
}

}  // namespace q3x::runtime::reference_engine_detail
