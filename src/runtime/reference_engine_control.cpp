#include "q3x/runtime/reference_engine.h"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
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
inline constexpr char kNvtxWholeRequestPrefill[] =
    "q3x.prefill.whole_request";
inline constexpr char kNvtxFinishPrefill[] = "q3x.prefill.finish";
inline constexpr char kNvtxCommitPrefill[] = "q3x.prefill.commit";
inline constexpr char kNvtxDecodeStep[] = "q3x.decode.step";

enum class NvtxRangeName : std::uint8_t {
  kGeneration,
  kPrefixStep,
  kPrefixTile,
  kWholeRequestPrefill,
  kFinishPrefill,
  kCommitPrefill,
  kDecodeStep,
};

struct RegisteredNvtxStrings {
  nvtxStringHandle_t generation = nullptr;
  nvtxStringHandle_t prefix_step = nullptr;
  nvtxStringHandle_t prefix_tile = nullptr;
  nvtxStringHandle_t whole_request_prefill = nullptr;
  nvtxStringHandle_t finish_prefill = nullptr;
  nvtxStringHandle_t commit_prefill = nullptr;
  nvtxStringHandle_t decode_step = nullptr;
};

[[nodiscard]] const RegisteredNvtxStrings& registered_nvtx_strings() {
  static const RegisteredNvtxStrings strings{
      nvtxDomainRegisterStringA(nullptr, kNvtxGeneration),
      nvtxDomainRegisterStringA(nullptr, kNvtxPrefixStep),
      nvtxDomainRegisterStringA(nullptr, kNvtxPrefixTile),
      nvtxDomainRegisterStringA(nullptr, kNvtxWholeRequestPrefill),
      nvtxDomainRegisterStringA(nullptr, kNvtxFinishPrefill),
      nvtxDomainRegisterStringA(nullptr, kNvtxCommitPrefill),
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
    case NvtxRangeName::kWholeRequestPrefill:
      return strings.whole_request_prefill;
    case NvtxRangeName::kFinishPrefill:
      return strings.finish_prefill;
    case NvtxRangeName::kCommitPrefill:
      return strings.commit_prefill;
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

[[nodiscard]] bool complete_uncommitted_whole_request_prefill_progress(
    const PrefillExecutionPlan& topology,
    const PrefillExecutionProgress& progress) noexcept {
  if (progress.next_layer != topology.layers.size() ||
      progress.next_panel != 0U || !progress.final_hidden_ready ||
      progress.prefill_state_committed ||
      !prefill_final_commit_ready(topology, progress)) {
    return false;
  }
  for (std::size_t layer_index = 0U;
       layer_index < topology.layers.size(); ++layer_index) {
    if (progress.completed_panels[layer_index] != topology.panel_count) {
      return false;
    }
    switch (topology.layers[layer_index].progress_domain) {
      case PrefillProgressDomain::kKvCache:
        if (progress.kv_visible_end[layer_index] !=
                topology.final_position ||
            progress.gdn_advanced_end[layer_index] !=
                topology.first_position) {
          return false;
        }
        break;
      case PrefillProgressDomain::kGdnState:
        if (progress.gdn_advanced_end[layer_index] !=
                topology.final_position ||
            progress.kv_visible_end[layer_index] !=
                topology.first_position) {
          return false;
        }
        break;
    }
  }
  return true;
}

template <bool EmitNvtx>
GenerationControlResult run_generation_control_impl(
    const std::vector<std::uint32_t>& prompt_token_ids,
    const GenerationControlOptions& options,
    const PrefillPlan& prefill_plan,
    const DecodePlan& decode_plan) {
  const bool use_whole_request_prefill =
      options.prefill_whole_request_layer_major;
  const bool invalid_whole_request_prefill =
      use_whole_request_prefill &&
      (!options.prefill_all_prompt_tokens || options.capture_trace ||
       options.prefill_single_arbitrary_tile ||
       prefill_plan.whole_request == nullptr ||
       prefill_plan.finish_whole_request_from_uncommitted_retained ==
           nullptr ||
       prefill_plan.commit_whole_request == nullptr);
  const bool invalid_legacy_all_prompt_admission =
      options.prefill_all_prompt_tokens && !use_whole_request_prefill &&
      (options.capture_trace || options.prefill_chunk_size <= 1U ||
       prefill_plan.finish_prefill_from_tile == nullptr);
  if (prompt_token_ids.empty() || options.max_new_tokens == 0U ||
      options.max_sequence_length == 0U ||
      prefill_plan.prefix_step == nullptr ||
      prefill_plan.finish_prefill == nullptr ||
      decode_plan.decode_step == nullptr ||
      options.prefill_chunk_size == 0U ||
      options.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
      options.stop_token_id >= kReferenceVocabularySize ||
      !is_valid_reference_logits_mode(options.logits_mode) ||
      (options.prefill_single_arbitrary_tile &&
       !options.prefill_all_prompt_tokens) ||
      invalid_whole_request_prefill ||
      invalid_legacy_all_prompt_admission) {
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

  const PrefillExecutionPlanResult whole_request_topology_result = [&]() {
    if (!use_whole_request_prefill) {
      return PrefillExecutionPlanResult{};
    }
    PrefillExecutionPlanOptions topology_options;
    topology_options.first_position = 0U;
    topology_options.prompt_token_count = prompt_token_ids.size();
    topology_options.max_sequence_length = options.max_sequence_length;
    return build_unbound_layer_major_prefill_execution_plan(topology_options);
  }();
  if (use_whole_request_prefill) {
    if (!whole_request_topology_result) {
      switch (whole_request_topology_result.error) {
        case PrefillExecutionPlanError::kInvalidArgument:
          return failure(GenerationControlError::kInvalidArgument);
        case PrefillExecutionPlanError::kArithmeticOverflow:
          return failure(GenerationControlError::kArithmeticOverflow);
        case PrefillExecutionPlanError::kCapacityExceeded:
          return failure(GenerationControlError::kCapacityExceeded);
        case PrefillExecutionPlanError::kInvalidTopology:
        case PrefillExecutionPlanError::kNone:
          return failure(GenerationControlError::kUnexpectedStep);
      }
    }
    // This seam passes topology only. A bound/executable plan here would
    // silently turn a host scaffold into a production selector.
    if (whole_request_topology_result.value->executable()) {
      return failure(GenerationControlError::kUnexpectedStep);
    }
  }

  [[maybe_unused]] const GenerationNvtxRange<EmitNvtx> generation_range;

  try {
    GenerationControl control;
    control.generated_token_ids.reserve(options.max_new_tokens);
    control.steps.reserve(static_cast<std::size_t>(required_steps));

    auto commit_token = [&](const std::uint32_t token_id,
                            const double elapsed_milliseconds) {
      const std::size_t token_index = control.generated_token_ids.size();
      control.generated_token_ids.push_back(token_id);
      return options.committed_token == nullptr ||
             options.committed_token(options.committed_token_context,
                                     token_id, token_index,
                                     elapsed_milliseconds);
    };

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
    PrefillExecutionProgress whole_request_progress;
    bool whole_request_progress_ready = false;

    const bool use_all_prompt_admission =
        options.prefill_all_prompt_tokens;
    const std::size_t prefix_token_count =
        use_all_prompt_admission
            ? prompt_token_ids.size()
            : prompt_token_ids.size() - 1U;
    const std::size_t effective_prefill_chunk_size =
        options.capture_trace ? 1U : options.prefill_chunk_size;
    if (!use_whole_request_prefill && effective_prefill_chunk_size > 1U &&
        prefix_token_count != 0U &&
        prefill_plan.prefix_tile == nullptr) {
      return failure(GenerationControlError::kInvalidArgument);
    }
    const std::size_t prefix_execution_count =
        use_whole_request_prefill
            ? 1U
            : options.prefill_single_arbitrary_tile
            ? reference_engine_detail::
                  single_arbitrary_prefix_execution_count(
                      prefix_token_count, effective_prefill_chunk_size)
            : reference_engine_detail::prefix_execution_count(
                  prefix_token_count, effective_prefill_chunk_size);
    control.prefill_execution_mode =
        use_whole_request_prefill
            ? ReferencePrefillExecutionMode::kWholeRequestLayerMajor
            : ReferencePrefillExecutionMode::kLegacyC512Tiled;
    // The whole-request callback has one aggregate timing regardless of
    // prompt length. Route coverage remains panel-granular and therefore
    // follows the immutable C8192 topology, including an M1 tail panel.
    control.prefill_logical_panel_count =
        use_whole_request_prefill
            ? static_cast<std::uint64_t>(
                  whole_request_topology_result.value->panel_count)
            : static_cast<std::uint64_t>(prefix_execution_count) +
                  (use_all_prompt_admission ? 0U : 1U);
    control.timing.prefix_execution_milliseconds.reserve(
        prefix_execution_count);

    std::size_t prefix_index = 0U;
    if (use_whole_request_prefill) {
      const PrefillExecutionPlan& unbound_immutable_topology =
          *whole_request_topology_result.value;
      PrefillPromptOptions prompt_options;
      prompt_options.measure_timing = true;
      prompt_options.retain_last_hidden_for_logits = true;
      const auto invoke_whole_request = [&]() {
        return prefill_plan.whole_request(
            prefill_plan.context, prompt_token_ids.data(),
            prompt_token_ids.size(), unbound_immutable_topology,
            prompt_options);
      };
      PrefillPromptOutcome outcome;
      try {
        outcome = invoke_with_optional_nvtx<EmitNvtx>(
            NvtxRangeName::kWholeRequestPrefill, invoke_whole_request);
      } catch (const std::bad_alloc&) {
        throw;
      } catch (const std::length_error&) {
        throw;
      } catch (...) {
        ReferenceRunnerStatus exception_status;
        exception_status.error = ReferenceRunnerError::kPoisoned;
        exception_status.operation = "whole_request_callback_exception";
        return failure(GenerationControlError::kRunnerFailure,
                       exception_status);
      }
      if (!outcome) {
        return failure(GenerationControlError::kRunnerFailure,
                       outcome.status);
      }
      if (outcome.value->logical_panel_count !=
              unbound_immutable_topology.panel_count ||
          outcome.value->panels.size() !=
              unbound_immutable_topology.panel_count ||
          outcome.value->prompt_token_count != prompt_token_ids.size() ||
          !complete_uncommitted_whole_request_prefill_progress(
              unbound_immutable_topology, outcome.value->progress)) {
        return failure(GenerationControlError::kUnexpectedStep);
      }
      if (!outcome.value->timing.has_value()) {
        return failure(GenerationControlError::kMissingTiming);
      }
      const double prompt_elapsed =
          outcome.value->timing->elapsed_milliseconds;
      if (!std::isfinite(prompt_elapsed) || prompt_elapsed < 0.0) {
        return failure(GenerationControlError::kUnexpectedStep);
      }

      std::size_t expected_prompt_offset = 0U;
      for (std::size_t panel_index = 0U;
           panel_index < unbound_immutable_topology.panel_count;
           ++panel_index) {
        const PrefillOperatorPanel& expected_panel =
            unbound_immutable_topology.panels[panel_index];
        const PrefillPromptPanelResult& panel =
            outcome.value->panels[panel_index];
        if (panel.logical_panel_ordinal != panel_index ||
            panel.prompt_token_offset != expected_prompt_offset ||
            panel.first_position != expected_panel.first_position ||
            panel.end_position != expected_panel.end_position ||
            panel.steps.size() != expected_panel.token_count) {
          return failure(GenerationControlError::kUnexpectedStep);
        }
        for (std::size_t panel_step = 0U;
             panel_step < panel.steps.size(); ++panel_step) {
          const ReferenceStepResult& step = panel.steps[panel_step];
          if (step.position != expected_panel.first_position + panel_step ||
              step.input_token_id !=
                  prompt_token_ids[expected_prompt_offset + panel_step] ||
              step.logits.has_value() || step.prediction.has_value() ||
              step.timing.has_value()) {
            return failure(GenerationControlError::kUnexpectedStep);
          }
        }
        expected_prompt_offset += expected_panel.token_count;
      }
      if (expected_prompt_offset != prompt_token_ids.size()) {
        return failure(GenerationControlError::kUnexpectedStep);
      }

      for (PrefillPromptPanelResult& panel : outcome.value->panels) {
        for (ReferenceStepResult& step : panel.steps) {
          control.steps.emplace_back(std::move(step));
        }
      }
      whole_request_progress = outcome.value->progress;
      whole_request_progress_ready = true;
      control.timing.prefix_execution_milliseconds.push_back(prompt_elapsed);
      control.timing.prompt_prefill_milliseconds = prompt_elapsed;
      prefix_index = prefix_token_count;
    } else if (effective_prefill_chunk_size > 1U) {
      while (prefix_index < prefix_token_count) {
        const std::size_t tile_token_count =
            options.prefill_single_arbitrary_tile
                ? reference_engine_detail::
                      next_single_arbitrary_prefix_tile_token_count(
                          prefix_token_count - prefix_index,
                          effective_prefill_chunk_size)
                : reference_engine_detail::next_prefix_tile_token_count(
                      prefix_token_count - prefix_index,
                      effective_prefill_chunk_size);
        ReferencePrefillTileOptions tile_options;
        tile_options.measure_timing = true;
        tile_options.retain_last_hidden_for_logits =
            use_all_prompt_admission &&
            prefix_index + tile_token_count == prefix_token_count;
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
      const StepFunction finish_prefill =
          use_whole_request_prefill
              ? prefill_plan
                    .finish_whole_request_from_uncommitted_retained
              : use_all_prompt_admission
                    ? prefill_plan.finish_prefill_from_tile
                    : prefill_plan.finish_prefill;
      // The final prompt step already exists in the transcript when every
      // prompt token was committed by legacy tiles or staged by the
      // whole-request callback. Replace only that placeholder with the
      // logits-bearing finalizer result. The dedicated whole-request
      // finalizer must leave the staged state uncommitted and must not advance
      // logical position.
      if (use_all_prompt_admission) {
        if (control.steps.empty()) {
          return failure(GenerationControlError::kUnexpectedStep);
        }
        control.steps.pop_back();
      }
      double elapsed = 0.0;
      GenerationControlError error = GenerationControlError::kNone;
      if (use_whole_request_prefill) {
        try {
          error = execute(prefill_plan.context, finish_prefill,
                          NvtxRangeName::kFinishPrefill,
                          prompt_token_ids.back(), true, elapsed,
                          predicted_token, runner_failure_status);
        } catch (const std::bad_alloc&) {
          throw;
        } catch (const std::length_error&) {
          throw;
        } catch (...) {
          runner_failure_status.error = ReferenceRunnerError::kPoisoned;
          runner_failure_status.operation =
              "whole_request_finalizer_exception";
          error = GenerationControlError::kRunnerFailure;
        }
      } else {
        error = execute(prefill_plan.context, finish_prefill,
                        NvtxRangeName::kFinishPrefill,
                        prompt_token_ids.back(), true, elapsed,
                        predicted_token, runner_failure_status);
      }
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

      if (use_whole_request_prefill) {
        if (!whole_request_progress_ready ||
            whole_request_progress.prefill_state_committed) {
          return failure(GenerationControlError::kUnexpectedStep);
        }
        const PrefillExecutionPlan& unbound_immutable_topology =
            *whole_request_topology_result.value;
        // The callback receives an actually const snapshot. A const reference
        // to whole_request_progress itself would still permit legal mutation
        // through const_cast because that underlying object is mutable and is
        // retained for the controller's local publication afterward.
        const PrefillExecutionProgress completed_uncommitted_progress =
            whole_request_progress;
        const auto commit_begin = std::chrono::steady_clock::now();
        const auto invoke_commit = [&]() noexcept {
          return prefill_plan.commit_whole_request(
              prefill_plan.context, unbound_immutable_topology,
              completed_uncommitted_progress);
        };
        const ReferenceRunnerStatus commit_status =
            invoke_with_optional_nvtx<EmitNvtx>(NvtxRangeName::kCommitPrefill,
                                                invoke_commit);
        if (!commit_status) {
          return failure(GenerationControlError::kRunnerFailure,
                         commit_status);
        }
        const auto commit_end = std::chrono::steady_clock::now();
        const double commit_elapsed =
            std::chrono::duration<double, std::milli>(commit_end -
                                                       commit_begin)
                .count();
        const double committed_prompt_prefill =
            control.timing.prompt_prefill_milliseconds + commit_elapsed;
        if (publish_prefill_state_committed(
                unbound_immutable_topology, whole_request_progress) !=
                PrefillExecutionProgressError::kNone ||
            !whole_request_progress.prefill_state_committed) {
          return failure(GenerationControlError::kUnexpectedStep);
        }
        control.timing.commit_prefill_milliseconds = commit_elapsed;
        control.timing.prompt_prefill_milliseconds =
            committed_prompt_prefill;
      }
    }

    control.timing.time_to_first_token_milliseconds =
        control.timing.prompt_prefill_milliseconds;
    control.timing.total_generation_milliseconds =
        control.timing.prompt_prefill_milliseconds;
    const bool continue_after_first = commit_token(
        predicted_token, control.timing.prompt_prefill_milliseconds);
    if (predicted_token == options.stop_token_id) {
      control.stop_reason = ReferenceStopReason::kImEnd;
    } else if (!continue_after_first) {
      control.stop_reason = ReferenceStopReason::kCancelled;
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
        control.timing.subsequent_token_milliseconds.push_back(elapsed);
        control.timing.decode_after_first_milliseconds = decode_after_first;
        control.timing.total_generation_milliseconds = total_generation;
        const bool continue_generation = commit_token(predicted_token,
                                                      elapsed);
        if (predicted_token == options.stop_token_id) {
          control.stop_reason = ReferenceStopReason::kImEnd;
          break;
        }
        if (!continue_generation) {
          control.stop_reason = ReferenceStopReason::kCancelled;
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
