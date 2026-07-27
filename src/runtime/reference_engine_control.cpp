#include "q3x/runtime/reference_engine.h"

#include <nvtx3/nvToolsExt.h>

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

inline constexpr char kNvtxGeneration[] = "q3x.generation";
inline constexpr char kNvtxPrefixStep[] = "q3x.prefill.prefix_step";
inline constexpr char kNvtxPrefixTile[] = "q3x.prefill.prefix_tile";
inline constexpr char kNvtxFinishPrefill[] = "q3x.prefill.finish";
inline constexpr char kNvtxDecodeStep[] = "q3x.decode.step";

enum class NvtxRangeName : std::uint8_t {
  kGeneration,
  kPrefixStep,
  kPrefixTile,
  kFinishPrefill,
  kDecodeStep,
};

struct RegisteredNvtxStrings {
  nvtxStringHandle_t generation = nullptr;
  nvtxStringHandle_t prefix_step = nullptr;
  nvtxStringHandle_t prefix_tile = nullptr;
  nvtxStringHandle_t finish_prefill = nullptr;
  nvtxStringHandle_t decode_step = nullptr;
};

[[nodiscard]] const RegisteredNvtxStrings& registered_nvtx_strings() {
  static const RegisteredNvtxStrings strings{
      nvtxDomainRegisterStringA(nullptr, kNvtxGeneration),
      nvtxDomainRegisterStringA(nullptr, kNvtxPrefixStep),
      nvtxDomainRegisterStringA(nullptr, kNvtxPrefixTile),
      nvtxDomainRegisterStringA(nullptr, kNvtxFinishPrefill),
      nvtxDomainRegisterStringA(nullptr, kNvtxDecodeStep)};
  return strings;
}

[[nodiscard]] nvtxStringHandle_t registered_nvtx_string(
    const NvtxRangeName name) {
  const RegisteredNvtxStrings& strings = registered_nvtx_strings();
  switch (name) {
    case NvtxRangeName::kGeneration:
      return strings.generation;
    case NvtxRangeName::kPrefixStep:
      return strings.prefix_step;
    case NvtxRangeName::kPrefixTile:
      return strings.prefix_tile;
    case NvtxRangeName::kFinishPrefill:
      return strings.finish_prefill;
    case NvtxRangeName::kDecodeStep:
      return strings.decode_step;
  }
  return strings.generation;
}

class ScopedNvtxRange final {
 public:
  explicit ScopedNvtxRange(const nvtxStringHandle_t message) noexcept {
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attributes.message.registered = message;
    (void)nvtxRangePushEx(&attributes);
  }

  ~ScopedNvtxRange() { (void)nvtxRangePop(); }

  ScopedNvtxRange(const ScopedNvtxRange&) = delete;
  ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
};

template <bool EmitRange>
class GenerationNvtxRange final {};

template <>
class GenerationNvtxRange<true> final {
 public:
  GenerationNvtxRange() noexcept
      : range_(registered_nvtx_string(NvtxRangeName::kGeneration)) {}

 private:
  ScopedNvtxRange range_;
};

template <bool EmitRange, typename Callback>
auto invoke_with_optional_nvtx(const NvtxRangeName range_name,
                               Callback&& callback) -> decltype(callback()) {
  if constexpr (!EmitRange) {
    return callback();
  } else {
    const ScopedNvtxRange range(registered_nvtx_string(range_name));
    return callback();
  }
}

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

template <bool EmitNvtx>
GenerationControlResult run_generation_control_impl(
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

  [[maybe_unused]] const GenerationNvtxRange<EmitNvtx> generation_range;

  try {
    GenerationControl control;
    control.generated_token_ids.reserve(options.max_new_tokens);
    control.steps.reserve(static_cast<std::size_t>(required_steps));

    auto execute = [&](void* const phase_context,
                       const StepFunction phase_step,
                       const NvtxRangeName nvtx_range_name,
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
      ReferenceStepOutcome outcome = invoke_with_optional_nvtx<EmitNvtx>(
          nvtx_range_name,
          [&]() {
            return phase_step(phase_context, input_token, step_options);
          });
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
    const std::size_t prefix_execution_count =
        reference_engine_detail::prefix_execution_count(
            prefix_token_count, effective_prefill_chunk_size);
    control.timing.prefix_execution_milliseconds.reserve(
        prefix_execution_count);

    std::size_t prefix_index = 0U;
    if (effective_prefill_chunk_size > 1U) {
      while (prefix_index < prefix_token_count) {
        const std::size_t tile_token_count =
            reference_engine_detail::next_prefix_tile_token_count(
                prefix_token_count - prefix_index,
                effective_prefill_chunk_size);
        ReferencePrefillTileOptions tile_options;
        tile_options.measure_timing = true;
        const auto invoke_prefix_tile = [&]() {
          return prefill_plan.prefix_tile(
              prefill_plan.context,
              prompt_token_ids.data() + prefix_index,
              tile_token_count, tile_options);
        };
        ReferencePrefillTileOutcome outcome =
            invoke_with_optional_nvtx<EmitNvtx>(NvtxRangeName::kPrefixTile,
                                                invoke_prefix_tile);
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
        control.timing.prefix_execution_milliseconds.push_back(tile_elapsed);
        control.timing.prompt_prefill_milliseconds = accumulated_prefill;
        prefix_index += tile_token_count;
      }
    } else {
      while (prefix_index < prefix_token_count) {
        double elapsed = 0.0;
        const GenerationControlError error = execute(
            prefill_plan.context, prefill_plan.prefix_step,
            NvtxRangeName::kPrefixStep,
            prompt_token_ids[prefix_index], false, elapsed,
            predicted_token, runner_failure_status);
        if (error != GenerationControlError::kNone) {
          return failure(error, runner_failure_status);
        }
        const double accumulated_prefill =
            control.timing.prompt_prefill_milliseconds + elapsed;
        if (!std::isfinite(accumulated_prefill)) {
          return failure(GenerationControlError::kUnexpectedStep);
        }
        control.timing.prefix_execution_milliseconds.push_back(elapsed);
        control.timing.prompt_prefill_milliseconds = accumulated_prefill;
        ++prefix_index;
      }
    }

    {
      double elapsed = 0.0;
      const GenerationControlError error = execute(
          prefill_plan.context, prefill_plan.finish_prefill,
          NvtxRangeName::kFinishPrefill,
          prompt_token_ids.back(), true, elapsed,
          predicted_token, runner_failure_status);
      if (error != GenerationControlError::kNone) {
        return failure(error, runner_failure_status);
      }
      const double prompt_prefill =
          control.timing.prompt_prefill_milliseconds + elapsed;
      if (!std::isfinite(prompt_prefill)) {
        return failure(GenerationControlError::kUnexpectedStep);
      }
      control.timing.finish_prefill_milliseconds = elapsed;
      control.timing.prompt_prefill_milliseconds = prompt_prefill;
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
            NvtxRangeName::kDecodeStep,
            input_token, true, elapsed, predicted_token,
            runner_failure_status);
        if (error != GenerationControlError::kNone) {
          return failure(error, runner_failure_status);
        }
        const double decode_after_first =
            control.timing.decode_after_first_milliseconds + elapsed;
        const double total_generation =
            control.timing.total_generation_milliseconds + elapsed;
        if (!std::isfinite(decode_after_first) ||
            !std::isfinite(total_generation)) {
          return failure(GenerationControlError::kUnexpectedStep);
        }
        control.generated_token_ids.push_back(predicted_token);
        control.timing.subsequent_token_milliseconds.push_back(elapsed);
        control.timing.decode_after_first_milliseconds = decode_after_first;
        control.timing.total_generation_milliseconds = total_generation;
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

}  // namespace

GenerationControlResult run_generation_control(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationControlOptions& options,
    const PrefillPlan& prefill_plan,
    const DecodePlan& decode_plan) {
  if (options.emit_nvtx_phase_ranges) {
    return run_generation_control_impl<true>(
        prompt_token_ids, options, prefill_plan, decode_plan);
  }
  return run_generation_control_impl<false>(
      prompt_token_ids, options, prefill_plan, decode_plan);
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
