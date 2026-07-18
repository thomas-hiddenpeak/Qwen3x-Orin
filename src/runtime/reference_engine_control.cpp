#include "q3x/runtime/reference_engine.h"

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
    void* const step_context,
    const StepFunction step_function) {
  if (prompt_token_ids.empty() || options.max_new_tokens == 0U ||
      options.max_sequence_length == 0U || step_function == nullptr ||
      options.stop_token_id >= kReferenceVocabularySize) {
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

    auto execute = [&](const std::uint32_t input_token,
                       const bool compute_logits,
                       double& elapsed,
                       std::uint32_t& predicted_token,
                       ReferenceRunnerStatus& failed_status)
        -> GenerationControlError {
      ReferenceStepOptions step_options;
      step_options.compute_logits = compute_logits;
      step_options.capture_trace = options.capture_trace;
      step_options.measure_timing = true;
      ReferenceStepOutcome outcome =
          step_function(step_context, input_token, step_options);
      if (!outcome) {
        failed_status = outcome.status;
        return GenerationControlError::kRunnerFailure;
      }
      const std::size_t expected_position = control.steps.size();
      if (outcome.value->position !=
              static_cast<std::uint32_t>(expected_position) ||
          outcome.value->input_token_id != input_token ||
          (!compute_logits && outcome.value->logits.has_value())) {
        return GenerationControlError::kUnexpectedStep;
      }
      if (!outcome.value->timing.has_value()) {
        return GenerationControlError::kMissingTiming;
      }
      if (compute_logits && !outcome.value->logits.has_value()) {
        return GenerationControlError::kMissingLogits;
      }
      elapsed = outcome.value->timing->elapsed_milliseconds;
      if (!(elapsed >= 0.0)) {
        return GenerationControlError::kUnexpectedStep;
      }
      if (compute_logits) {
        predicted_token = outcome.value->logits->predicted_token_id;
        if (predicted_token >= kReferenceVocabularySize) {
          return GenerationControlError::kUnexpectedStep;
        }
      }
      control.steps.emplace_back(std::move(*outcome.value));
      return GenerationControlError::kNone;
    };

    ReferenceRunnerStatus runner_failure_status;
    std::uint32_t predicted_token = 0U;
    for (std::size_t index = 0U; index < prompt_token_ids.size(); ++index) {
      const bool last_prompt_token = index + 1U == prompt_token_ids.size();
      double elapsed = 0.0;
      const GenerationControlError error = execute(prompt_token_ids[index],
                                                   last_prompt_token,
                                                   elapsed,
                                                   predicted_token,
                                                   runner_failure_status);
      if (error != GenerationControlError::kNone) {
        GenerationControlResult result = failure(error, runner_failure_status);
        return result;
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
        const GenerationControlError error = execute(input_token,
                                                     true,
                                                     elapsed,
                                                     predicted_token,
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
  }
  return "unknown";
}

}  // namespace q3x::runtime::reference_engine_detail
