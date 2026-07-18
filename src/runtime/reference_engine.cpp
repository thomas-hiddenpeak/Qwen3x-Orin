#include "q3x/runtime/reference_engine.h"

#include "q3x/core/sha256.h"
#include "q3x/text/tokenizer.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_milliseconds(
    const Clock::time_point begin) noexcept {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

[[nodiscard]] ReferenceEngineDiagnostic engine_diagnostic(
    const ReferenceEngineError code, std::string stage,
    std::string message, std::string context = {}) {
  ReferenceEngineDiagnostic diagnostic;
  diagnostic.code = code;
  diagnostic.stage = std::move(stage);
  diagnostic.message = std::move(message);
  diagnostic.context = std::move(context);
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic tokenizer_diagnostic(
    const std::string_view stage, const text::TokenizerError& error) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      ReferenceEngineError::kTokenizerFailure, std::string(stage),
      error.message, "byte_offset=" + std::to_string(error.offset));
  diagnostic.dependency_error = static_cast<int>(error.code);
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic resident_diagnostic(
    const ResidentLoadDiagnostic& error) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      ReferenceEngineError::kResidentLoadFailure, "resident_load",
      error.message, error.context);
  diagnostic.dependency_error = static_cast<int>(error.code);
  diagnostic.cuda_error = error.cuda_error;
  if (!error.shard.empty()) {
    diagnostic.context = error.shard +
                         (diagnostic.context.empty()
                              ? std::string{}
                              : ":" + diagnostic.context);
  }
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic binding_diagnostic(
    const WeightBindDiagnostic& error) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      ReferenceEngineError::kWeightBindFailure, "weight_bind",
      error.message, error.tensor);
  diagnostic.dependency_error = static_cast<int>(error.code);
  diagnostic.cuda_error = error.cuda_error;
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic request_diagnostic(
    const RequestDiagnostic& error) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      ReferenceEngineError::kRequestStateFailure, "request_state",
      error.message, error.context);
  diagnostic.dependency_error = static_cast<int>(error.code);
  diagnostic.cuda_error = error.cuda_error;
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic runner_diagnostic(
    const ReferenceEngineError code, const std::string_view stage,
    const ReferenceRunnerStatus& status) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      code, std::string(stage), reference_runner_error_string(status.error));
  diagnostic.dependency_error = static_cast<int>(status.error);
  diagnostic.cuda_error = status.cuda_error;
  diagnostic.layer = status.layer;
  if (status.operation != nullptr) {
    diagnostic.operation = status.operation;
  }
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic control_diagnostic(
    const reference_engine_detail::GenerationControlResult& result) {
  using ControlError = reference_engine_detail::GenerationControlError;
  ReferenceEngineError code = ReferenceEngineError::kInvalidArgument;
  switch (result.error) {
    case ControlError::kNone:
      code = ReferenceEngineError::kNone;
      break;
    case ControlError::kInvalidArgument:
      code = ReferenceEngineError::kInvalidArgument;
      break;
    case ControlError::kCapacityExceeded:
      code = ReferenceEngineError::kCapacityExceeded;
      break;
    case ControlError::kArithmeticOverflow:
      code = ReferenceEngineError::kArithmeticOverflow;
      break;
    case ControlError::kRunnerFailure:
      if (result.runner_status.error ==
          ReferenceRunnerError::kTraceUnavailable) {
        code = ReferenceEngineError::kTraceFailure;
      } else if (result.runner_status.error ==
                 ReferenceRunnerError::kAllocationFailure) {
        code = ReferenceEngineError::kAllocationFailure;
      } else {
        code = ReferenceEngineError::kRunnerStepFailure;
      }
      break;
    case ControlError::kUnexpectedStep:
      code = ReferenceEngineError::kRunnerStepFailure;
      break;
    case ControlError::kMissingLogits:
      code = ReferenceEngineError::kMissingLogits;
      break;
    case ControlError::kMissingTiming:
      code = ReferenceEngineError::kMissingTiming;
      break;
    case ControlError::kAllocationFailure:
      code = ReferenceEngineError::kAllocationFailure;
      break;
  }
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      code, "generation_control",
      std::string(reference_engine_detail::to_string(result.error)));
  if (result.error == ControlError::kRunnerFailure) {
    diagnostic.dependency_error =
        static_cast<int>(result.runner_status.error);
    diagnostic.cuda_error = result.runner_status.cuda_error;
    diagnostic.layer = result.runner_status.layer;
    if (result.runner_status.operation != nullptr) {
      diagnostic.operation = result.runner_status.operation;
    }
  }
  return diagnostic;
}

[[nodiscard]] text::ChatResult format_single_user_prompt(
    const text::Tokenizer& tokenizer, const std::string_view user_prompt) {
  std::vector<text::ChatMessage> messages;
  messages.reserve(1U);
  messages.push_back({"user", std::string(user_prompt)});
  text::Qwen36ChatOptions options;
  options.add_generation_prompt = true;
  options.enable_thinking = false;
  return tokenizer.format_qwen36_chat(messages, options);
}

[[nodiscard]] bool hash_span(const ConstBf16Span span,
                             const std::size_t expected_elements,
                             std::string& output) {
  if (span.data == nullptr || span.size != expected_elements ||
      span.size > std::numeric_limits<std::size_t>::max() /
                      sizeof(std::uint16_t)) {
    return false;
  }
  core::Sha256 hash;
  if (!hash.update(span.data, span.size * sizeof(std::uint16_t))) {
    return false;
  }
  output = hash.finalize().hex();
  return true;
}

struct EngineStepContext {
  ReferenceRunner* runner = nullptr;
  std::vector<ReferenceTraceDigest>* traces = nullptr;
  bool capture_trace = false;
};

[[nodiscard]] ReferenceStepOutcome step_with_trace(
    void* const opaque_context, const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) {
  auto& context = *static_cast<EngineStepContext*>(opaque_context);
  ReferenceStepOutcome outcome = context.runner->step(input_token_id, options);
  if (!outcome || !context.capture_trace) {
    return outcome;
  }

  try {
    const std::optional<ReferenceTraceView> view =
        context.runner->last_trace();
    if (!view.has_value() || view->position != outcome.value->position ||
        view->input_token_id != outcome.value->input_token_id ||
        view->element_count != kReferenceTraceElements) {
      return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                   kReferenceNoLayer, "engine_trace_view"}};
    }

    ReferenceTraceDigest digest;
    digest.position = view->position;
    digest.input_token_id = view->input_token_id;
    digest.element_count = view->element_count;
    if (!hash_span(view->raw(), kReferenceTraceElements,
                   digest.full_sha256) ||
        !hash_span(view->embedding(), kReferenceHiddenSize,
                   digest.embedding_sha256)) {
      return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                   kReferenceNoLayer, "engine_trace_hash"}};
    }
    for (std::size_t layer = 0U;
         layer < kReferenceDecoderLayerCount; ++layer) {
      if (!hash_span(view->layer_hidden(layer), kReferenceHiddenSize,
                     digest.layer_hidden_sha256[layer]) ||
          !hash_span(view->layer_residual(layer), kReferenceHiddenSize,
                     digest.layer_residual_sha256[layer])) {
        return {{}, {ReferenceRunnerError::kTraceUnavailable, 0, layer,
                     "engine_trace_boundary_hash"}};
      }
    }
    if (!hash_span(view->final_norm(), kReferenceHiddenSize,
                   digest.final_norm_sha256)) {
      return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                   kReferenceNoLayer, "engine_trace_final_norm_hash"}};
    }
    context.traces->emplace_back(std::move(digest));
    return outcome;
  } catch (const std::bad_alloc&) {
    return {{}, {ReferenceRunnerError::kAllocationFailure, 0,
                 kReferenceNoLayer, "engine_trace_allocation"}};
  } catch (const std::length_error&) {
    return {{}, {ReferenceRunnerError::kAllocationFailure, 0,
                 kReferenceNoLayer, "engine_trace_allocation"}};
  } catch (...) {
    return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                 kReferenceNoLayer, "engine_trace_exception"}};
  }
}

[[nodiscard]] bool checked_required_steps(
    const std::size_t prompt_tokens, const std::uint32_t max_new_tokens,
    std::uint64_t& result) noexcept {
  if (max_new_tokens == 0U) {
    return false;
  }
  const std::uint64_t prompt = static_cast<std::uint64_t>(prompt_tokens);
  const std::uint64_t additional =
      static_cast<std::uint64_t>(max_new_tokens) - 1U;
  if (additional > std::numeric_limits<std::uint64_t>::max() - prompt) {
    return false;
  }
  result = prompt + additional;
  return true;
}

}  // namespace

struct ReferenceEngine::Impl {
  // Declaration order is part of the safety contract. Destruction is exactly
  // runner -> request_state -> model_weights -> resident_weights -> tokenizer.
  std::unique_ptr<text::Tokenizer> tokenizer;
  std::optional<ResidentWeights> resident_weights;
  std::optional<ModelWeights> model_weights;
  std::optional<RequestState> request_state;
  std::optional<ReferenceRunner> runner;
  ReferenceEngineLoadStats load;
  bool trace_enabled = false;

  struct BuildResult {
    std::unique_ptr<Impl> value;
    ReferenceEngineDiagnostic diagnostic;

    [[nodiscard]] explicit operator bool() const noexcept {
      return value != nullptr &&
             diagnostic.code == ReferenceEngineError::kNone;
    }
  };

  [[nodiscard]] static BuildResult build(
      const std::filesystem::path& model_directory,
      const ReferenceEngineOptions& options,
      std::unique_ptr<text::Tokenizer> prepared_tokenizer = {},
      const double prepared_tokenizer_milliseconds = 0.0) {
    BuildResult result;
    if (model_directory.empty()) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kInvalidArgument, "model_directory",
          "model directory must not be empty");
      return result;
    }
    if (!is_valid_projection_backend(options.projection_backend)) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kInvalidArgument, "projection_backend",
          "unknown projection backend");
      return result;
    }
    if (options.projection_backend == ProjectionBackend::kSm87WeightOnly) {
      int device = 0;
      cudaError_t cuda_status = cudaGetDevice(&device);
      cudaDeviceProp properties{};
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaGetDeviceProperties(&properties, device);
      }
      if (cuda_status != cudaSuccess) {
        result.diagnostic = engine_diagnostic(
            ReferenceEngineError::kRunnerFactoryFailure,
            "projection_backend_device",
            "failed to inspect the active CUDA device before model load");
        result.diagnostic.cuda_error = static_cast<int>(cuda_status);
        return result;
      }
      if (properties.major != 8 || properties.minor != 7) {
        result.diagnostic = engine_diagnostic(
            ReferenceEngineError::kInvalidArgument,
            "projection_backend_device",
            "sm87 projection backend requires compute capability 8.7",
            "active_device=sm_" + std::to_string(properties.major) +
                std::to_string(properties.minor));
        return result;
      }
    }

    try {
      const bool tokenizer_was_prepared = prepared_tokenizer != nullptr;
      const Clock::time_point build_begin = Clock::now();
      auto impl = std::make_unique<Impl>();
      impl->trace_enabled = options.enable_trace;
      if (prepared_tokenizer != nullptr) {
        impl->tokenizer = std::move(prepared_tokenizer);
        impl->load.tokenizer_milliseconds =
            prepared_tokenizer_milliseconds;
      } else {
        const Clock::time_point begin = Clock::now();
        const std::filesystem::path tokenizer_path =
            model_directory / "tokenizer.json";
        text::TokenizerLoadResult tokenizer =
            text::Tokenizer::load_file(tokenizer_path.string());
        impl->load.tokenizer_milliseconds = elapsed_milliseconds(begin);
        if (!tokenizer) {
          result.diagnostic =
              tokenizer_diagnostic("tokenizer_load", tokenizer.error);
          return result;
        }
        impl->tokenizer = std::move(tokenizer.tokenizer);
      }

      {
        const Clock::time_point begin = Clock::now();
        ResidentLoadResult resident = load_pinned_qwen36_27b(
            model_directory, options.resident_options);
        impl->load.resident_load_milliseconds = elapsed_milliseconds(begin);
        if (!resident) {
          result.diagnostic = resident_diagnostic(resident.diagnostic);
          return result;
        }
        impl->resident_weights.emplace(std::move(*resident.value));
        impl->load.resident = impl->resident_weights->stats();
      }

      {
        const Clock::time_point begin = Clock::now();
        WeightBindResult weights =
            bind_qwen36_27b_weights(*impl->resident_weights);
        impl->load.weight_bind_milliseconds = elapsed_milliseconds(begin);
        if (!weights) {
          result.diagnostic = binding_diagnostic(weights.diagnostic);
          return result;
        }
        impl->model_weights.emplace(std::move(*weights.value));
        impl->load.binding = impl->model_weights->stats();
      }

      {
        const Clock::time_point begin = Clock::now();
        RequestStateResult request =
            create_request_state(options.request_options);
        impl->load.request_state_milliseconds = elapsed_milliseconds(begin);
        if (!request) {
          result.diagnostic = request_diagnostic(request.diagnostic);
          return result;
        }
        impl->request_state.emplace(std::move(*request.value));
        impl->load.request_arena_bytes =
            impl->request_state->arena_bytes();
        impl->load.request_max_sequence_length =
            impl->request_state->max_sequence_length();
      }

      {
        ReferenceRunnerOptions runner_options;
        runner_options.enable_trace = options.enable_trace;
        runner_options.projection_backend = options.projection_backend;
        const Clock::time_point begin = Clock::now();
        ReferenceRunnerFactoryResult runner = create_reference_runner(
            &*impl->model_weights, &*impl->request_state, runner_options);
        impl->load.runner_factory_milliseconds = elapsed_milliseconds(begin);
        if (!runner) {
          result.diagnostic = runner_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure,
              "runner_factory", runner.diagnostic);
          return result;
        }
        impl->runner.emplace(std::move(*runner.value));
      }

      impl->load.total_milliseconds =
          elapsed_milliseconds(build_begin) +
          (tokenizer_was_prepared ? prepared_tokenizer_milliseconds : 0.0);
      result.value = std::move(impl);
      return result;
    } catch (const std::bad_alloc&) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kAllocationFailure, "engine_create",
          "host allocation failed while creating the reference engine");
      return result;
    } catch (const std::length_error& error) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kAllocationFailure, "engine_create",
          error.what());
      return result;
    } catch (const std::exception& error) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kInvalidArgument, "engine_create",
          error.what());
      return result;
    }
  }
};

ReferenceEngine::ReferenceEngine(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ReferenceEngine::~ReferenceEngine() = default;
ReferenceEngine::ReferenceEngine(ReferenceEngine&&) noexcept = default;
ReferenceEngine& ReferenceEngine::operator=(ReferenceEngine&&) noexcept =
    default;

ReferenceEngine::operator bool() const noexcept {
  return impl_ != nullptr && impl_->tokenizer != nullptr &&
         impl_->resident_weights.has_value() &&
         impl_->model_weights.has_value() &&
         impl_->request_state.has_value() && impl_->runner.has_value();
}

const ReferenceEngineLoadStats& ReferenceEngine::load_stats() const noexcept {
  static const ReferenceEngineLoadStats empty;
  return impl_ != nullptr ? impl_->load : empty;
}

std::uint32_t ReferenceEngine::max_sequence_length() const noexcept {
  return impl_ != nullptr && impl_->request_state.has_value()
             ? impl_->request_state->max_sequence_length()
             : 0U;
}

ReferenceGenerateResult ReferenceEngine::generate(
    const std::string_view user_prompt,
    const ReferenceGenerateOptions& options) {
  ReferenceGenerateResult result;
  if (!*this) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "engine",
        "reference engine is empty");
    return result;
  }
  if (user_prompt.empty() || options.max_new_tokens == 0U ||
      options.stop_token_id >= kReferenceVocabularySize) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "generation_options",
        "prompt must be non-empty, max_new_tokens must be positive, and "
        "stop_token_id must be in the pinned vocabulary");
    return result;
  }
  if (options.capture_trace && !impl_->trace_enabled) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "generation_options",
        "capture_trace requires an engine created with enable_trace=true");
    return result;
  }

  try {
    text::ChatResult chat =
        format_single_user_prompt(*impl_->tokenizer, user_prompt);
    if (!chat) {
      result.diagnostic = tokenizer_diagnostic("chat_encode", chat.error);
      return result;
    }
    if (chat.token_ids.empty()) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kTokenizerFailure, "chat_encode",
          "the rendered chat prompt encoded to zero tokens");
      return result;
    }

    const ReferenceRunnerStatus reset = impl_->runner->reset();
    if (!reset) {
      result.diagnostic = runner_diagnostic(
          ReferenceEngineError::kRunnerResetFailure, "runner_reset", reset);
      return result;
    }

    std::vector<ReferenceTraceDigest> traces;
    EngineStepContext step_context;
    step_context.runner = &*impl_->runner;
    step_context.traces = &traces;
    step_context.capture_trace = options.capture_trace;

    reference_engine_detail::GenerationControlOptions control_options;
    control_options.max_new_tokens = options.max_new_tokens;
    control_options.stop_token_id = options.stop_token_id;
    control_options.max_sequence_length =
        impl_->request_state->max_sequence_length();
    control_options.capture_trace = options.capture_trace;
    reference_engine_detail::GenerationControlResult control =
        reference_engine_detail::run_generation_control(
            chat.token_ids, control_options, &step_context, step_with_trace);
    if (!control) {
      result.diagnostic = control_diagnostic(control);
      return result;
    }

    const std::size_t text_token_count =
        reference_engine_detail::generated_text_token_count(
            control.value->generated_token_ids,
            control.value->stop_reason, options.stop_token_id);
    const auto text_end = control.value->generated_token_ids.begin() +
                          static_cast<std::ptrdiff_t>(text_token_count);
    const std::vector<std::uint32_t> text_token_ids(
        control.value->generated_token_ids.begin(), text_end);
    text::DecodeOptions decode_options;
    // The exact generated-id sequence is authoritative. Hide only the stop
    // token explicitly removed above; do not silently discard any other
    // generated special id under a max-token termination.
    decode_options.skip_special_tokens = false;
    text::DecodeResult decoded =
        impl_->tokenizer->decode(text_token_ids, decode_options);
    if (!decoded) {
      result.diagnostic = tokenizer_diagnostic("generated_decode",
                                               decoded.error);
      result.diagnostic.code = ReferenceEngineError::kDecodeFailure;
      return result;
    }

    ReferenceGeneration generation;
    generation.rendered_prompt = std::move(chat.rendered);
    generation.prompt_token_ids = std::move(chat.token_ids);
    generation.generated_token_ids =
        std::move(control.value->generated_token_ids);
    generation.generated_text = std::move(decoded.text);
    generation.stop_reason = control.value->stop_reason;
    generation.timing = std::move(control.value->timing);
    generation.steps = std::move(control.value->steps);
    generation.traces = std::move(traces);
    result.value.emplace(std::move(generation));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure, "generate",
        "host allocation failed during generation");
    return result;
  } catch (const std::length_error& error) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure, "generate", error.what());
    return result;
  } catch (const std::exception& error) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kTokenizerFailure, "generate", error.what());
    return result;
  }
}

ReferenceEngineCreateResult create_reference_engine(
    const std::filesystem::path& model_directory,
    const ReferenceEngineOptions& options) {
  ReferenceEngineCreateResult result;
  ReferenceEngine::Impl::BuildResult built =
      ReferenceEngine::Impl::build(model_directory, options);
  if (!built) {
    result.diagnostic = std::move(built.diagnostic);
    return result;
  }
  result.value.emplace(ReferenceEngine(std::move(built.value)));
  return result;
}

ReferenceOneShotResult generate_reference(
    const std::filesystem::path& model_directory,
    const std::string_view user_prompt,
    const ReferenceOneShotOptions& options) {
  ReferenceOneShotResult result;
  if (model_directory.empty() || user_prompt.empty() ||
      options.generation.max_new_tokens == 0U ||
      options.generation.stop_token_id >= kReferenceVocabularySize ||
      !is_valid_projection_backend(options.projection_backend)) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "one_shot_options",
        "model directory and prompt must be non-empty; generation options "
        "must be valid");
    return result;
  }

  try {
    const Clock::time_point tokenizer_begin = Clock::now();
    const std::filesystem::path tokenizer_path =
        model_directory / "tokenizer.json";
    text::TokenizerLoadResult tokenizer =
        text::Tokenizer::load_file(tokenizer_path.string());
    const double tokenizer_milliseconds =
        elapsed_milliseconds(tokenizer_begin);
    if (!tokenizer) {
      result.diagnostic =
          tokenizer_diagnostic("tokenizer_load", tokenizer.error);
      return result;
    }

    text::ChatResult preflight =
        format_single_user_prompt(*tokenizer.tokenizer, user_prompt);
    if (!preflight) {
      result.diagnostic = tokenizer_diagnostic("chat_encode", preflight.error);
      return result;
    }
    if (preflight.token_ids.empty()) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kTokenizerFailure, "chat_encode",
          "the rendered chat prompt encoded to zero tokens");
      return result;
    }

    std::uint64_t required_steps = 0U;
    if (!checked_required_steps(preflight.token_ids.size(),
                                options.generation.max_new_tokens,
                                required_steps)) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kArithmeticOverflow, "request_capacity",
          "prompt plus decode capacity overflows uint64");
      return result;
    }
    if (required_steps == 0U ||
        required_steps > kAbsoluteRequestMaxSequenceLength ||
        required_steps > std::numeric_limits<std::uint32_t>::max()) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kCapacityExceeded, "request_capacity",
          "prompt plus decode steps exceed the request-state limit",
          "required_steps=" + std::to_string(required_steps));
      return result;
    }

    ReferenceEngineOptions engine_options;
    engine_options.resident_options = options.resident_options;
    engine_options.request_options.batch_size = 1U;
    engine_options.request_options.max_sequence_length =
        static_cast<std::uint32_t>(required_steps);
    engine_options.request_options.max_arena_bytes =
        options.request_max_arena_bytes;
    engine_options.request_options.min_free_bytes_after_create =
        options.request_min_free_bytes_after_create;
    engine_options.enable_trace = options.generation.capture_trace;
    engine_options.projection_backend = options.projection_backend;

    ReferenceEngine::Impl::BuildResult built = ReferenceEngine::Impl::build(
        model_directory, engine_options, std::move(tokenizer.tokenizer),
        tokenizer_milliseconds);
    if (!built) {
      result.diagnostic = std::move(built.diagnostic);
      return result;
    }

    ReferenceEngine engine(std::move(built.value));
    ReferenceGenerateResult generated =
        engine.generate(user_prompt, options.generation);
    if (!generated) {
      result.diagnostic = std::move(generated.diagnostic);
      return result;
    }

    ReferenceOneShotGeneration value;
    value.load = engine.load_stats();
    value.generation = std::move(*generated.value);
    result.value.emplace(std::move(value));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure, "one_shot",
        "host allocation failed during one-shot generation");
    return result;
  } catch (const std::length_error& error) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure, "one_shot", error.what());
    return result;
  } catch (const std::exception& error) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "one_shot", error.what());
    return result;
  }
}

std::string_view to_string(const ReferenceEngineError error) noexcept {
  switch (error) {
    case ReferenceEngineError::kNone:
      return "none";
    case ReferenceEngineError::kInvalidArgument:
      return "invalid_argument";
    case ReferenceEngineError::kCapacityExceeded:
      return "capacity_exceeded";
    case ReferenceEngineError::kArithmeticOverflow:
      return "arithmetic_overflow";
    case ReferenceEngineError::kTokenizerFailure:
      return "tokenizer_failure";
    case ReferenceEngineError::kResidentLoadFailure:
      return "resident_load_failure";
    case ReferenceEngineError::kWeightBindFailure:
      return "weight_bind_failure";
    case ReferenceEngineError::kRequestStateFailure:
      return "request_state_failure";
    case ReferenceEngineError::kRunnerFactoryFailure:
      return "runner_factory_failure";
    case ReferenceEngineError::kRunnerStepFailure:
      return "runner_step_failure";
    case ReferenceEngineError::kRunnerResetFailure:
      return "runner_reset_failure";
    case ReferenceEngineError::kMissingLogits:
      return "missing_logits";
    case ReferenceEngineError::kMissingTiming:
      return "missing_timing";
    case ReferenceEngineError::kDecodeFailure:
      return "decode_failure";
    case ReferenceEngineError::kTraceFailure:
      return "trace_failure";
    case ReferenceEngineError::kAllocationFailure:
      return "allocation_failure";
  }
  return "unknown";
}

std::string_view to_string(const ReferenceStopReason reason) noexcept {
  switch (reason) {
    case ReferenceStopReason::kImEnd:
      return "im_end";
    case ReferenceStopReason::kMaxNewTokens:
      return "max_new_tokens";
  }
  return "unknown";
}

}  // namespace q3x::runtime
