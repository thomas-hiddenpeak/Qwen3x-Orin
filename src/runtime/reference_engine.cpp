#include "q3x/runtime/reference_engine.h"

#include "q3x/core/sha256.h"
#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "q3x/text/tokenizer.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

using Clock = std::chrono::steady_clock;

struct Sm87Fp8OutputProjectionSidecars {
  std::uint8_t* data = nullptr;
  std::size_t bytes = 0U;

  Sm87Fp8OutputProjectionSidecars() noexcept = default;
  Sm87Fp8OutputProjectionSidecars(
      const Sm87Fp8OutputProjectionSidecars&) = delete;
  Sm87Fp8OutputProjectionSidecars& operator=(
      const Sm87Fp8OutputProjectionSidecars&) = delete;

  ~Sm87Fp8OutputProjectionSidecars() { release(); }

  void release() noexcept {
    if (data != nullptr) {
      (void)cudaFree(data);
    }
    data = nullptr;
    bytes = 0U;
  }
};

struct Sm87Fp8OutputSidecarPreparation {
  bool enabled = false;
  bool hard_failure = false;
  std::size_t layers = 0U;
  std::uint64_t bytes = 0U;
  int cuda_error = 0;
  std::string message;
  std::string fallback_reason;
};

struct Sm87NvFp4DownScale6Sidecars {
  std::uint8_t* data = nullptr;
  std::size_t bytes = 0U;
  std::vector<NvFp4DownScale6SidecarDescriptor> descriptors;

  Sm87NvFp4DownScale6Sidecars() noexcept = default;
  Sm87NvFp4DownScale6Sidecars(
      const Sm87NvFp4DownScale6Sidecars&) = delete;
  Sm87NvFp4DownScale6Sidecars& operator=(
      const Sm87NvFp4DownScale6Sidecars&) = delete;

  ~Sm87NvFp4DownScale6Sidecars() { release(); }

  void release() noexcept {
    if (data != nullptr) {
      (void)cudaFree(data);
    }
    data = nullptr;
    bytes = 0U;
    descriptors.clear();
  }
};

struct Sm87NvFp4DownScale6Preparation {
  bool enabled = false;
  bool hard_failure = false;
  std::size_t eligible_layers = 0U;
  std::size_t fallback_layers = 0U;
  std::uint64_t bytes = 0U;
  int cuda_error = 0;
  std::string message;
  std::string fallback_reason;
};

constexpr std::size_t kNvFp4DownScale6GroupSize = 16U;
constexpr std::size_t kNvFp4DownScale6ColumnsPerTile = 512U;
constexpr std::size_t kNvFp4DownScale6RowsPerQuad = 4U;
constexpr std::size_t kNvFp4DownScale6BytesPerTile = 96U;
constexpr unsigned int kNvFp4DownScale6MaximumBase = 192U;
constexpr std::size_t kNvFp4DownScale6RequiredAlignment = 32U;
constexpr std::size_t kNvFp4DownCanonicalScaleBytesPerLayer =
    kNvFp4DownScale6Rows *
    (kNvFp4DownScale6Columns / kNvFp4DownScale6GroupSize);
constexpr std::size_t kNvFp4DownDerivedScale6BytesPerLayer =
    (kNvFp4DownScale6Rows / kNvFp4DownScale6RowsPerQuad) *
    (kNvFp4DownScale6Columns / kNvFp4DownScale6ColumnsPerTile) *
    kNvFp4DownScale6BytesPerTile;
static_assert(kNvFp4DownDerivedScale6BytesPerLayer ==
              kNvFp4DownScale6SidecarBytesPerProjection);
static_assert((kNvFp4DownScale6SidecarBytesPerProjection %
               kNvFp4DownScale6RequiredAlignment) == 0U);

struct NvFp4DownScale6LayerPlan {
  std::size_t layer_index = 0U;
  const NvFp4LinearWeight* down = nullptr;
  unsigned int scale_base = 0U;
};

[[nodiscard]] const Fp8LinearWeight* attention_output_projection(
    const DecoderLayerWeights& layer) noexcept {
  if (const auto* const linear =
          std::get_if<LinearAttentionWeights>(&layer.attention)) {
    return std::get_if<Fp8LinearWeight>(&linear->out_proj);
  }
  if (const auto* const full =
          std::get_if<FullAttentionWeights>(&layer.attention)) {
    return std::get_if<Fp8LinearWeight>(&full->o_proj);
  }
  return nullptr;
}

[[nodiscard]] const NvFp4LinearWeight* exact_nvfp4_down_projection(
    const DecoderLayerWeights& layer) noexcept {
  if (!supports_nvfp4_down_residual_norm_fusion(
          ProjectionBackend::kSm87WeightOnly, layer.mlp.down_proj)) {
    return nullptr;
  }
  return std::get_if<NvFp4LinearWeight>(&layer.mlp.down_proj);
}

[[nodiscard]] bool derive_nvfp4_down_scale6_base(
    const std::vector<std::uint8_t>& canonical_scales,
    unsigned int& scale_base) noexcept {
  if (canonical_scales.size() !=
      kNvFp4DownCanonicalScaleBytesPerLayer) {
    return false;
  }
  std::uint8_t minimum = std::numeric_limits<std::uint8_t>::max();
  std::uint8_t maximum = 0U;
  for (const std::uint8_t scale : canonical_scales) {
    minimum = std::min(minimum, scale);
    maximum = std::max(maximum, scale);
  }
  scale_base = std::min<unsigned int>(minimum,
                                      kNvFp4DownScale6MaximumBase);
  return static_cast<unsigned int>(maximum) - scale_base <= 63U;
}

[[nodiscard]] bool pack_nvfp4_down_scale6_sidecar(
    const std::vector<std::uint8_t>& canonical_scales,
    const unsigned int scale_base,
    std::vector<std::uint8_t>& packed) noexcept {
  if (canonical_scales.size() !=
          kNvFp4DownCanonicalScaleBytesPerLayer ||
      packed.size() != kNvFp4DownScale6SidecarBytesPerProjection ||
      scale_base > kNvFp4DownScale6MaximumBase) {
    return false;
  }
  std::fill(packed.begin(), packed.end(), 0U);
  constexpr std::size_t kScaleColumns =
      kNvFp4DownScale6Columns / kNvFp4DownScale6GroupSize;
  constexpr std::size_t kRowQuads =
      kNvFp4DownScale6Rows / kNvFp4DownScale6RowsPerQuad;
  constexpr std::size_t kTilesPerRowQuad =
      kNvFp4DownScale6Columns / kNvFp4DownScale6ColumnsPerTile;
  for (std::size_t row_quad = 0U; row_quad < kRowQuads; ++row_quad) {
    for (std::size_t tile = 0U; tile < kTilesPerRowQuad; ++tile) {
      const std::size_t tile_byte =
          (row_quad * kTilesPerRowQuad + tile) *
          kNvFp4DownScale6BytesPerTile;
      for (std::size_t lane_pair = 0U; lane_pair < 16U; ++lane_pair) {
        for (std::size_t phase = 0U; phase < 2U; ++phase) {
          for (std::size_t row = 0U; row < 4U; ++row) {
            const std::size_t code =
                8U * lane_pair + 4U * phase + row;
            const std::size_t canonical_column =
                32U * tile + lane_pair + 16U * phase;
            const unsigned int scale = canonical_scales[
                (4U * row_quad + row) * kScaleColumns +
                canonical_column];
            if (scale < scale_base || scale - scale_base > 63U) {
              return false;
            }
            const unsigned int delta = scale - scale_base;
            const std::size_t first_bit = code * 6U;
            const std::size_t byte = tile_byte + first_bit / 8U;
            const unsigned int shift =
                static_cast<unsigned int>(first_bit % 8U);
            const unsigned int payload = delta << shift;
            packed[byte] |= static_cast<std::uint8_t>(payload);
            if (shift > 2U) {
              packed[byte + 1U] |=
                  static_cast<std::uint8_t>(payload >> 8U);
            }
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool verify_nvfp4_down_scale6_sidecar(
    const std::vector<std::uint8_t>& canonical_scales,
    const unsigned int scale_base,
    const std::vector<std::uint8_t>& packed) noexcept {
  if (canonical_scales.size() !=
          kNvFp4DownCanonicalScaleBytesPerLayer ||
      packed.size() != kNvFp4DownScale6SidecarBytesPerProjection ||
      scale_base > kNvFp4DownScale6MaximumBase) {
    return false;
  }
  constexpr std::size_t kScaleColumns =
      kNvFp4DownScale6Columns / kNvFp4DownScale6GroupSize;
  constexpr std::size_t kRowQuads =
      kNvFp4DownScale6Rows / kNvFp4DownScale6RowsPerQuad;
  constexpr std::size_t kTilesPerRowQuad =
      kNvFp4DownScale6Columns / kNvFp4DownScale6ColumnsPerTile;
  for (std::size_t row_quad = 0U; row_quad < kRowQuads; ++row_quad) {
    for (std::size_t tile = 0U; tile < kTilesPerRowQuad; ++tile) {
      const std::size_t tile_byte =
          (row_quad * kTilesPerRowQuad + tile) *
          kNvFp4DownScale6BytesPerTile;
      for (std::size_t lane_pair = 0U; lane_pair < 16U; ++lane_pair) {
        for (std::size_t phase = 0U; phase < 2U; ++phase) {
          for (std::size_t row = 0U; row < 4U; ++row) {
            const std::size_t code =
                8U * lane_pair + 4U * phase + row;
            const std::size_t first_bit = code * 6U;
            const std::size_t byte = tile_byte + first_bit / 8U;
            const unsigned int shift =
                static_cast<unsigned int>(first_bit % 8U);
            unsigned int window = packed[byte];
            if (shift > 2U) {
              window |= static_cast<unsigned int>(packed[byte + 1U])
                        << 8U;
            }
            const unsigned int reconstructed =
                scale_base + ((window >> shift) & 63U);
            const std::size_t canonical_column =
                32U * tile + lane_pair + 16U * phase;
            const unsigned int canonical = canonical_scales[
                (4U * row_quad + row) * kScaleColumns +
                canonical_column];
            if (reconstructed != canonical) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

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

[[nodiscard]] std::optional<ReferenceEngineDiagnostic>
sm87_device_diagnostic(const ProjectionBackend backend,
                       int* const active_device = nullptr) {
  if (backend != ProjectionBackend::kSm87WeightOnly) {
    return std::nullopt;
  }

  int device = 0;
  // Match the resident loader's error isolation: a stale caller launch error
  // must not be attributed to this startup device probe.
  (void)cudaGetLastError();
  cudaError_t cuda_status = cudaGetDevice(&device);
  cudaDeviceProp properties{};
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaGetDeviceProperties(&properties, device);
  }
  if (cuda_status != cudaSuccess) {
    ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
        ReferenceEngineError::kRunnerFactoryFailure,
        "projection_backend_device",
        "failed to inspect the active CUDA device before model load");
    diagnostic.cuda_error = static_cast<int>(cuda_status);
    return diagnostic;
  }
  if (properties.major != 8 || properties.minor != 7) {
    return engine_diagnostic(
        ReferenceEngineError::kInvalidArgument,
        "projection_backend_device",
        "sm87 projection backend requires compute capability 8.7",
        "active_device=sm_" + std::to_string(properties.major) +
            std::to_string(properties.minor));
  }
  if (active_device != nullptr) {
    *active_device = device;
  }
  return std::nullopt;
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
    case ControlError::kMissingPrediction:
      code = ReferenceEngineError::kMissingPrediction;
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

[[nodiscard]] ReferencePrefillTileOutcome prefill_prefix_tile(
    void* const opaque_context,
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const ReferencePrefillTileOptions& options) {
  auto& context = *static_cast<EngineStepContext*>(opaque_context);
  if (context.capture_trace) {
    return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                 kReferenceNoLayer, "engine_prefill_tile_trace"}};
  }
  return context.runner->prefill_prefix_tile(input_token_ids, token_count,
                                             options);
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

struct TimedResidentLoad {
  ResidentLoadResult result;
  double milliseconds = 0.0;
};

[[nodiscard]] TimedResidentLoad load_resident_on_device(
    const std::filesystem::path& model_directory,
    const ResidentLoadOptions& options, const int device) {
  TimedResidentLoad timed;
  const Clock::time_point begin = Clock::now();
  const cudaError_t cuda_status = cudaSetDevice(device);
  if (cuda_status != cudaSuccess) {
    timed.result.diagnostic.code = ResidentLoadErrorCode::kCudaFailure;
    timed.result.diagnostic.message =
        "failed to select the caller CUDA device in the startup worker";
    timed.result.diagnostic.context = "cudaSetDevice(startup worker)";
    timed.result.diagnostic.cuda_error = static_cast<int>(cuda_status);
  } else {
    (void)cudaGetLastError();
    timed.result = load_pinned_qwen36_27b(model_directory, options);
  }
  timed.milliseconds = elapsed_milliseconds(begin);
  return timed;
}

[[nodiscard]] Sm87Fp8OutputSidecarPreparation
prepare_sm87_fp8_output_projection_sidecars(
    ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    Sm87Fp8OutputProjectionSidecars& owner) {
  Sm87Fp8OutputSidecarPreparation result;
  if (owner.data != nullptr || owner.bytes != 0U) {
    result.hard_failure = true;
    result.message = "FP8 output sidecar owner was not empty before prepare";
    return result;
  }

  for (const DecoderLayerWeights& layer : model_weights.layers()) {
    const Fp8LinearWeight* const output =
        attention_output_projection(layer);
    if (output == nullptr || output->weight == nullptr ||
        output->output_size != kFp8M1OutputProjectionRows ||
        output->input_size != kFp8M1OutputProjectionColumns ||
        output->m1_aosoa4_preswizzled_weight != nullptr ||
        (reinterpret_cast<std::uintptr_t>(output->weight) %
         alignof(std::uint32_t)) != 0U) {
      result.fallback_reason = "ineligible_model_weights";
      return result;
    }
  }

  constexpr std::size_t kSidecarBytes =
      kQwen36Fp8M1OutputProjectionAosoa4PreswizzledBytes;
  (void)cudaGetLastError();
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaMemGetInfo failed before FP8 output sidecar prepare";
    return result;
  }
  const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
  if (kSidecarBytes > free_u64 ||
      minimum_free_bytes_after_prepare > free_u64 - kSidecarBytes) {
    result.fallback_reason = "insufficient_device_memory_margin";
    return result;
  }

  void* allocation = nullptr;
  status = cudaMalloc(&allocation, kSidecarBytes);
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    result.fallback_reason = "cuda_memory_allocation";
    return result;
  }
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaMalloc failed while creating FP8 output sidecars";
    return result;
  }
  owner.data = static_cast<std::uint8_t*>(allocation);
  owner.bytes = kSidecarBytes;

  // Recheck the configured safety margin after the allocation. The first
  // query prevents a predictably unsafe allocation; this second query closes
  // the normal race with other device allocations and accounts for the
  // driver's actual allocation granularity. A margin miss is optional-path
  // fallback, while inability to query CUDA state remains a hard failure.
  std::size_t remaining_free_bytes = 0U;
  std::size_t remaining_total_bytes = 0U;
  status = cudaMemGetInfo(&remaining_free_bytes, &remaining_total_bytes);
  (void)remaining_total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMemGetInfo failed after FP8 output sidecar allocation";
    owner.release();
    return result;
  }
  if (static_cast<std::uint64_t>(remaining_free_bytes) <
      minimum_free_bytes_after_prepare) {
    result.fallback_reason =
        "insufficient_device_memory_margin_after_allocation";
    owner.release();
    return result;
  }

  cudaStream_t stream = nullptr;
  status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaStreamCreateWithFlags failed for FP8 output sidecar prepare";
    owner.release();
    return result;
  }

  std::size_t layer_index = 0U;
  for (const DecoderLayerWeights& layer : model_weights.layers()) {
    const Fp8LinearWeight* const output =
        attention_output_projection(layer);
    std::uint8_t* const destination =
        owner.data +
        layer_index *
            kFp8M1OutputProjectionAosoa4PreswizzledBytesPerLayer;
    status = static_cast<cudaError_t>(
        kernels::
            launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_pack_cuda(
                output->weight, destination, output->output_size,
                output->input_size, static_cast<void*>(stream)));
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "FP8 output sidecar pack launch failed at layer " +
          std::to_string(layer_index);
      (void)cudaStreamDestroy(stream);
      owner.release();
      return result;
    }
    ++layer_index;
  }

  status = cudaStreamSynchronize(stream);
  const cudaError_t destroy_status = cudaStreamDestroy(stream);
  if (status != cudaSuccess || destroy_status != cudaSuccess) {
    const cudaError_t failure =
        status != cudaSuccess ? status : destroy_status;
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(failure);
    result.message =
        "FP8 output sidecar pack stream failed to synchronize or destroy";
    owner.release();
    return result;
  }

  if (!model_weights.attach_fp8_m1_output_projection_sidecars(
          owner.data, owner.bytes)) {
    result.hard_failure = true;
    result.message =
        "ModelWeights rejected the complete FP8 output sidecar arena";
    owner.release();
    return result;
  }

  result.enabled = true;
  result.layers = kQwen36DenseLayerCount;
  result.bytes = kSidecarBytes;
  return result;
}

[[nodiscard]] Sm87NvFp4DownScale6Preparation
prepare_sm87_nvfp4_down_scale6_sidecars(
    ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    Sm87NvFp4DownScale6Sidecars& owner) {
  Sm87NvFp4DownScale6Preparation result;
  if (owner.data != nullptr || owner.bytes != 0U ||
      !owner.descriptors.empty()) {
    result.hard_failure = true;
    result.message =
        "NVFP4 down scale6 sidecar owner was not empty before prepare";
    return result;
  }

  // Phase one validates the complete 64-layer inventory and derives the
  // compact descriptor count before any device allocation or ModelWeights
  // mutation. A layer that is not an exact NVFP4 down projection, or whose
  // raw E4M3 code span cannot fit one base plus six-bit deltas, remains on the
  // canonical route rather than disabling eligible peers.
  std::vector<std::uint8_t> canonical_scales(
      kNvFp4DownCanonicalScaleBytesPerLayer);
  std::vector<NvFp4DownScale6LayerPlan> plans;
  plans.reserve(kQwen36DenseLayerCount);
  (void)cudaGetLastError();
  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    const NvFp4LinearWeight* const down =
        exact_nvfp4_down_projection(model_weights.layer(layer_index));
    if (down == nullptr) {
      continue;
    }
    const cudaError_t status = cudaMemcpy(
        canonical_scales.data(), down->block_scale,
        kNvFp4DownCanonicalScaleBytesPerLayer, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "cudaMemcpy failed while scanning NVFP4 down scales at layer " +
          std::to_string(layer_index);
      return result;
    }
    unsigned int scale_base = 0U;
    if (derive_nvfp4_down_scale6_base(canonical_scales, scale_base)) {
      plans.push_back({layer_index, down, scale_base});
    }
  }
  result.eligible_layers = plans.size();
  result.fallback_layers = kQwen36DenseLayerCount - plans.size();
  if (plans.empty()) {
    result.fallback_reason = "no_eligible_nvfp4_down_scale6_layers";
    return result;
  }
  if (plans.size() >
      std::numeric_limits<std::size_t>::max() /
          kNvFp4DownScale6SidecarBytesPerProjection) {
    result.hard_failure = true;
    result.message = "NVFP4 down scale6 sidecar byte count overflowed";
    return result;
  }
  const std::size_t arena_bytes =
      plans.size() * kNvFp4DownScale6SidecarBytesPerProjection;

  // Allocate all host staging before the device admission check. A host
  // allocation exception is handled by the engine's existing allocation
  // diagnostic, while an optional CUDA capacity miss remains a canonical
  // fallback with no partially owned device arena.
  std::vector<std::uint8_t> packed(
      kNvFp4DownScale6SidecarBytesPerProjection);
  std::vector<NvFp4DownScale6SidecarDescriptor> descriptors;
  descriptors.reserve(plans.size());

  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMemGetInfo failed before NVFP4 down scale6 sidecar prepare";
    return result;
  }
  const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
  const std::uint64_t arena_u64 = static_cast<std::uint64_t>(arena_bytes);
  if (arena_u64 > free_u64 ||
      minimum_free_bytes_after_prepare > free_u64 - arena_u64) {
    result.fallback_reason = "insufficient_device_memory_margin";
    return result;
  }

  void* allocation = nullptr;
  status = cudaMalloc(&allocation, arena_bytes);
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    result.fallback_reason = "cuda_memory_allocation";
    return result;
  }
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMalloc failed while creating NVFP4 down scale6 sidecars";
    return result;
  }
  owner.data = static_cast<std::uint8_t*>(allocation);
  owner.bytes = arena_bytes;
  if ((reinterpret_cast<std::uintptr_t>(owner.data) %
       kNvFp4DownScale6RequiredAlignment) != 0U) {
    result.hard_failure = true;
    result.message =
        "cudaMalloc returned a misaligned NVFP4 down scale6 arena";
    owner.release();
    return result;
  }

  std::size_t remaining_free_bytes = 0U;
  std::size_t remaining_total_bytes = 0U;
  status =
      cudaMemGetInfo(&remaining_free_bytes, &remaining_total_bytes);
  (void)remaining_total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMemGetInfo failed after NVFP4 down scale6 sidecar allocation";
    owner.release();
    return result;
  }
  if (static_cast<std::uint64_t>(remaining_free_bytes) <
      minimum_free_bytes_after_prepare) {
    result.fallback_reason =
        "insufficient_device_memory_margin_after_allocation";
    owner.release();
    return result;
  }

  // Phase two rereads every eligible canonical tensor, verifies that its
  // immutable code span still matches phase one, packs and independently
  // reconstructs every six-bit code, then copies the complete compact arena.
  // ModelWeights is attached only after every layer succeeds.
  for (std::size_t descriptor_index = 0U;
       descriptor_index < plans.size(); ++descriptor_index) {
    const NvFp4DownScale6LayerPlan& plan = plans[descriptor_index];
    status = cudaMemcpy(canonical_scales.data(), plan.down->block_scale,
                        kNvFp4DownCanonicalScaleBytesPerLayer,
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "cudaMemcpy failed while packing NVFP4 down scales at layer " +
          std::to_string(plan.layer_index);
      owner.release();
      return result;
    }
    unsigned int verified_base = 0U;
    if (!derive_nvfp4_down_scale6_base(canonical_scales, verified_base) ||
        verified_base != plan.scale_base ||
        !pack_nvfp4_down_scale6_sidecar(
            canonical_scales, plan.scale_base, packed) ||
        !verify_nvfp4_down_scale6_sidecar(
            canonical_scales, plan.scale_base, packed)) {
      result.hard_failure = true;
      result.message =
          "NVFP4 down scale6 pack validation failed at layer " +
          std::to_string(plan.layer_index);
      owner.release();
      return result;
    }
    std::uint8_t* const destination =
        owner.data +
        descriptor_index * kNvFp4DownScale6SidecarBytesPerProjection;
    if ((reinterpret_cast<std::uintptr_t>(destination) %
         kNvFp4DownScale6RequiredAlignment) != 0U) {
      result.hard_failure = true;
      result.message =
          "NVFP4 down scale6 layer destination was misaligned";
      owner.release();
      return result;
    }
    status = cudaMemcpy(destination, packed.data(), packed.size(),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "cudaMemcpy failed while uploading NVFP4 down scale6 layer " +
          std::to_string(plan.layer_index);
      owner.release();
      return result;
    }
    descriptors.push_back(
        {plan.layer_index, destination,
         kNvFp4DownScale6SidecarBytesPerProjection, plan.scale_base,
         kNvFp4DownScale6Rows, kNvFp4DownScale6Columns});
  }

  owner.descriptors = std::move(descriptors);
  if (!model_weights.attach_nvfp4_down_scale6_sidecars(
          owner.data, owner.bytes, owner.descriptors.data(),
          owner.descriptors.size())) {
    result.hard_failure = true;
    result.message =
        "ModelWeights rejected the complete NVFP4 down scale6 arena";
    owner.release();
    return result;
  }

  result.enabled = true;
  result.bytes = arena_u64;
  return result;
}

}  // namespace

struct ReferenceEngine::Impl {
  // Declaration order is part of the safety contract. Destruction is exactly
  // runner -> request_state -> model_weights -> down scale6 sidecars ->
  // output sidecars -> resident_weights -> tokenizer.
  std::unique_ptr<text::Tokenizer> tokenizer;
  std::optional<ResidentWeights> resident_weights;
  Sm87Fp8OutputProjectionSidecars fp8_output_sidecars;
  Sm87NvFp4DownScale6Sidecars nvfp4_down_scale6_sidecars;
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
      const double prepared_tokenizer_milliseconds = 0.0,
      std::optional<ResidentWeights> prepared_resident = std::nullopt,
      const double prepared_resident_milliseconds = 0.0,
      const double prepared_work_wall_milliseconds = 0.0) {
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
    const Clock::time_point build_begin = Clock::now();
    if (std::optional<ReferenceEngineDiagnostic> diagnostic =
            sm87_device_diagnostic(options.projection_backend)) {
      result.diagnostic = std::move(*diagnostic);
      return result;
    }

    try {
      const bool tokenizer_was_prepared = prepared_tokenizer != nullptr;
      const bool resident_was_prepared = prepared_resident.has_value();
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

      if (prepared_resident.has_value()) {
        impl->resident_weights.emplace(std::move(*prepared_resident));
        impl->load.resident_load_milliseconds =
            prepared_resident_milliseconds;
        impl->load.resident = impl->resident_weights->stats();
      } else {
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
        impl->load.request_prefill_chunk_size =
            impl->request_state->plan().prefill_chunk_size;
      }

      if (options.projection_backend ==
          ProjectionBackend::kSm87WeightOnly) {
        const Clock::time_point begin = Clock::now();
        const Sm87Fp8OutputSidecarPreparation preparation =
            prepare_sm87_fp8_output_projection_sidecars(
                *impl->model_weights,
                options.request_options.min_free_bytes_after_create,
                impl->fp8_output_sidecars);
        impl->load.fp8_output_sidecar_milliseconds =
            elapsed_milliseconds(begin);
        if (preparation.hard_failure) {
          result.diagnostic = engine_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure,
              "fp8_output_sidecar_prepare", preparation.message);
          result.diagnostic.cuda_error = preparation.cuda_error;
          return result;
        }
        impl->load.fp8_output_sidecars_enabled = preparation.enabled;
        impl->load.fp8_output_sidecar_layers = preparation.layers;
        impl->load.fp8_output_sidecar_bytes = preparation.bytes;
        impl->load.fp8_output_sidecar_fallback_reason =
            preparation.fallback_reason;

        const Clock::time_point down_scale6_begin = Clock::now();
        const Sm87NvFp4DownScale6Preparation down_scale6_preparation =
            prepare_sm87_nvfp4_down_scale6_sidecars(
                *impl->model_weights,
                options.request_options.min_free_bytes_after_create,
                impl->nvfp4_down_scale6_sidecars);
        impl->load.nvfp4_down_scale6_sidecar_milliseconds =
            elapsed_milliseconds(down_scale6_begin);
        if (down_scale6_preparation.hard_failure) {
          result.diagnostic = engine_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure,
              "nvfp4_down_scale6_sidecar_prepare",
              down_scale6_preparation.message);
          result.diagnostic.cuda_error =
              down_scale6_preparation.cuda_error;
          return result;
        }
        impl->load.nvfp4_down_scale6_sidecars_enabled =
            down_scale6_preparation.enabled;
        impl->load.nvfp4_down_scale6_sidecar_eligible_layers =
            down_scale6_preparation.eligible_layers;
        impl->load.nvfp4_down_scale6_sidecar_fallback_layers =
            down_scale6_preparation.fallback_layers;
        impl->load.nvfp4_down_scale6_sidecar_bytes =
            down_scale6_preparation.bytes;
        impl->load.nvfp4_down_scale6_sidecar_fallback_reason =
            down_scale6_preparation.fallback_reason;
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

      double prepared_milliseconds = 0.0;
      if (prepared_work_wall_milliseconds > 0.0) {
        prepared_milliseconds = prepared_work_wall_milliseconds;
      } else {
        prepared_milliseconds =
            (tokenizer_was_prepared ? prepared_tokenizer_milliseconds : 0.0) +
            (resident_was_prepared ? prepared_resident_milliseconds : 0.0);
      }
      impl->load.total_milliseconds =
          elapsed_milliseconds(build_begin) + prepared_milliseconds;
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
      options.prefill_chunk_size == 0U ||
      options.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
      options.stop_token_id >= kReferenceVocabularySize ||
      !is_valid_reference_logits_mode(options.logits_mode)) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "generation_options",
        "prompt must be non-empty, max_new_tokens must be positive, "
        "prefill_chunk_size must be in [1,32], and stop_token_id must be in "
        "the pinned vocabulary");
    return result;
  }
  if (options.capture_trace && !impl_->trace_enabled) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "generation_options",
        "capture_trace requires an engine created with enable_trace=true");
    return result;
  }
  if (!options.capture_trace &&
      options.prefill_chunk_size >
          impl_->request_state->plan().prefill_chunk_size) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "generation_options",
        "requested prefill chunk exceeds the engine workspace capacity",
        "requested=" + std::to_string(options.prefill_chunk_size) +
            " capacity=" +
            std::to_string(
                impl_->request_state->plan().prefill_chunk_size));
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
    control_options.prefill_chunk_size = options.prefill_chunk_size;
    control_options.capture_trace = options.capture_trace;
    control_options.logits_mode = options.logits_mode;
    control_options.emit_nvtx_phase_ranges = options.emit_nvtx_phase_ranges;

    reference_engine_detail::PrefillPlan prefill_plan;
    prefill_plan.context = &step_context;
    prefill_plan.prefix_step = step_with_trace;
    prefill_plan.finish_prefill = step_with_trace;
    prefill_plan.prefix_tile = prefill_prefix_tile;

    reference_engine_detail::DecodePlan decode_plan;
    decode_plan.context = &step_context;
    decode_plan.decode_step = step_with_trace;

    reference_engine_detail::GenerationControlResult control =
        reference_engine_detail::run_generation_control(
            chat.token_ids, control_options, prefill_plan, decode_plan);
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
    generation.requested_prefill_chunk_size = options.prefill_chunk_size;
    generation.effective_prefill_chunk_size =
        options.capture_trace ? kDefaultRequestPrefillChunkSize
                              : options.prefill_chunk_size;
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
      options.generation.prefill_chunk_size == 0U ||
      options.generation.prefill_chunk_size >
          kMaximumRequestPrefillChunkSize ||
      options.generation.stop_token_id >= kReferenceVocabularySize ||
      !is_valid_reference_logits_mode(options.generation.logits_mode) ||
      !is_valid_projection_backend(options.projection_backend)) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "one_shot_options",
        "model directory and prompt must be non-empty; generation options "
        "must be valid");
    return result;
  }

  try {
    const Clock::time_point one_shot_prepare_begin = Clock::now();
    std::optional<std::future<TimedResidentLoad>> resident_future;
    std::optional<ReferenceEngineDiagnostic> startup_device_diagnostic;
    if (options.overlap_tokenizer_and_resident_load) {
      int active_device = 0;
      startup_device_diagnostic =
          sm87_device_diagnostic(options.projection_backend, &active_device);
      if (!startup_device_diagnostic.has_value() &&
          options.projection_backend != ProjectionBackend::kSm87WeightOnly) {
        (void)cudaGetLastError();
        const cudaError_t cuda_status = cudaGetDevice(&active_device);
        if (cuda_status != cudaSuccess) {
          startup_device_diagnostic = engine_diagnostic(
              ReferenceEngineError::kResidentLoadFailure, "resident_load",
              "failed to inspect the active CUDA device before concurrent "
              "model load",
              "cudaGetDevice(startup)");
          startup_device_diagnostic->cuda_error =
              static_cast<int>(cuda_status);
        }
      }

      if (!startup_device_diagnostic.has_value()) {
        try {
          resident_future.emplace(std::async(
              std::launch::async,
              [directory = std::filesystem::path(model_directory),
               resident_options = options.resident_options,
               active_device]() {
                return load_resident_on_device(directory, resident_options,
                                               active_device);
              }));
        } catch (const std::system_error&) {
          // Resource exhaustion in the host scheduler must not make a model
          // unloadable. Fall back to the existing serial build path.
          resident_future.reset();
        }
      }
    }

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
    // Preserve the established tokenizer/chat/capacity diagnostic priority.
    // A device probe is needed before starting the worker, but its failure is
    // reported only after all CPU-only preflight checks succeed.
    if (startup_device_diagnostic.has_value()) {
      result.diagnostic = std::move(*startup_device_diagnostic);
      return result;
    }

    ReferenceEngineOptions engine_options;
    engine_options.resident_options = options.resident_options;
    engine_options.request_options.batch_size = 1U;
    engine_options.request_options.prefill_chunk_size =
        options.generation.prefill_chunk_size;
    engine_options.request_options.max_sequence_length =
        static_cast<std::uint32_t>(required_steps);
    engine_options.request_options.max_arena_bytes =
        options.request_max_arena_bytes;
    engine_options.request_options.min_free_bytes_after_create =
        options.request_min_free_bytes_after_create;
    engine_options.enable_trace = options.generation.capture_trace;
    engine_options.projection_backend = options.projection_backend;

    ReferenceEngine::Impl::BuildResult built;
    if (resident_future.has_value()) {
      TimedResidentLoad resident = resident_future->get();
      if (!resident.result) {
        result.diagnostic = resident_diagnostic(resident.result.diagnostic);
        return result;
      }
      std::optional<ResidentWeights> prepared_resident;
      prepared_resident.emplace(std::move(*resident.result.value));
      built = ReferenceEngine::Impl::build(
          model_directory, engine_options, std::move(tokenizer.tokenizer),
          tokenizer_milliseconds, std::move(prepared_resident),
          resident.milliseconds,
          elapsed_milliseconds(one_shot_prepare_begin));
    } else {
      built = ReferenceEngine::Impl::build(
          model_directory, engine_options, std::move(tokenizer.tokenizer),
          tokenizer_milliseconds, std::nullopt, 0.0,
          elapsed_milliseconds(one_shot_prepare_begin));
    }
    if (!built) {
      result.diagnostic = std::move(built.diagnostic);
      return result;
    }
    built.value->load.tokenizer_resident_overlap =
        resident_future.has_value();

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
    case ReferenceEngineError::kMissingPrediction:
      return "missing_prediction";
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
