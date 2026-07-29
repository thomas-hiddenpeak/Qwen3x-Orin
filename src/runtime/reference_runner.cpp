#include "q3x/runtime/reference_runner.h"

#include "q3x/kernels/sm87_fp8_prefill_supermatrix.h"
#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_nvfp4_marlin.h"
#endif

#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
#include "../kernels/reference/gdn_prefill_b8_sequential_sm87.h"
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
#include "../kernels/reference/gdn_prefill_chunk64_cublas_reference_sm87.h"
#include "reference_runner_gdn_chunk64_reference_admission.h"
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
#include "../kernels/reference/gdn_prefill_whole_span_conv_sm87.h"
#include "../kernels/sm87/gdn_prefill_chunk64_native_sm87.h"
#include "reference_runner_gdn_chunk64_native_admission.h"
#endif
#include "../kernels/reference/gdn_prefill_c16_norm_gate_sm87.h"
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
#include "reference_runner_gdn_c16_norm_gate_admission.h"
#endif
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/gdn_decode.h"
#include "q3x/runtime/layout_ops.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>

namespace q3x::runtime {
namespace {

constexpr std::size_t kLinearQkvElements = 10'240U;
constexpr std::size_t kLinearValueElements = 6'144U;
constexpr std::size_t kLinearScalarElements = 48U;
constexpr std::size_t kFullQueryHeads = 24U;
constexpr std::size_t kFullKvHeads = 4U;
constexpr std::size_t kFullHeadDimension = 256U;
constexpr std::size_t kFullQueryElements =
    kFullQueryHeads * kFullHeadDimension;
constexpr std::size_t kFullQGateElements = 2U * kFullQueryElements;
constexpr std::size_t kFullKvElements =
    kFullKvHeads * kFullHeadDimension;
constexpr std::size_t kRopePairs = 32U;
constexpr std::size_t kPrefillKernelTileMaximumTokens = 16U;
constexpr std::size_t kProductionProjectionSubtileTokens = 32U;
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kAttentionScale = 1.0F / 16.0F;

#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
[[nodiscard]] bool nvfp4_marlin_prefill_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_NVFP4_MARLIN_PREFILL_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_nvfp4_marlin_prefill_admission =
    nvfp4_marlin_prefill_environment_enabled();
thread_local std::size_t g_nvfp4_marlin_prefill_admission_hits = 0U;
#endif

#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
// Admission-only switch. It has internal storage and defaults off in every
// thread, so the public runner and CLI retain the production M16 route. The
// matching test-only setter is deliberately absent from the public header.
thread_local bool g_enable_prefill_gdn_b8_admission = false;
thread_local std::size_t g_prefill_gdn_b8_admission_hits = 0U;
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
// Native C64/WY architecture admission. This build remains test-only and the
// worker-local route is enabled only by the exact value "1". It has no
// external-library context, workspace, fallback, or default-route authority.
[[nodiscard]] bool gdn_chunk64_native_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_prefill_gdn_chunk64_native_admission =
    gdn_chunk64_native_environment_enabled();
thread_local std::size_t g_prefill_gdn_chunk64_native_admission_hits = 0U;
thread_local reference_runner_detail::PrefillGdnChunk64NativeSnapshotHook
    g_prefill_gdn_chunk64_native_snapshot_hook{};
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
// Compile-time-isolated architecture switch. It stays false unless the
// admission-only environment value is exactly "1"; the external reference
// context has no fallback or production authority.
// The evaluation server runs generation on a dedicated worker. Initialize
// that worker's private switch from the same explicit environment gate used
// by the test harness; synchronous tests may still override it through the
// private exchange accessor between runner calls.
[[nodiscard]] bool gdn_chunk64_reference_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_GDN_CHUNK64_REFERENCE_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_prefill_gdn_chunk64_reference_admission =
    gdn_chunk64_reference_environment_enabled();
thread_local std::size_t g_prefill_gdn_chunk64_reference_admission_hits = 0U;
thread_local reference_runner_detail::
    PrefillGdnChunk64ReferenceSnapshotHook
        g_prefill_gdn_chunk64_reference_snapshot_hook{};
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
// The opt-in test build can still select the legacy route (false) or the
// production fused route (true) inside one ELF. Release builds do not expose
// this switch and always use the production selector below.
thread_local bool g_enable_prefill_gdn_c16_norm_gate_admission = false;
thread_local std::size_t g_prefill_gdn_c16_norm_gate_admission_hits = 0U;
thread_local reference_runner_detail::
    PrefillGdnC16NormGateAdmissionSnapshotHook
        g_prefill_gdn_c16_norm_gate_admission_snapshot_hook{};
#endif

static_assert(kLinearQkvElements <= kReferenceIntermediateSize);
static_assert(kLinearValueElements <= kReferenceIntermediateSize);
static_assert(kFullQGateElements <= kReferenceIntermediateSize);
static_assert(kFullQueryElements <= kReferenceIntermediateSize);
static_assert(kFullKvElements <= kReferenceIntermediateSize);
static_assert(kPrefillKernelTileMaximumTokens ==
              kQkRopeTileMaximumTokens);
static_assert(kMaximumRequestPrefillChunkSize == 512U);
static_assert(kReferenceHiddenSize == 5'120U);
static_assert(kProductionProjectionSubtileTokens <=
              kMaximumProjectionTileTokenCount);
static_assert(kMaximumProjectionTileTokenCount <=
              kMaximumRequestPrefillChunkSize);

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return pointer == nullptr ||
         bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

// Treat address overflow as overlap so a malformed span can never select a
// concurrent writer. Exactly adjacent half-open ranges remain disjoint.
[[nodiscard]] bool byte_ranges_are_disjoint(
    const void* const first, const std::size_t first_bytes,
    const void* const second, const std::size_t second_bytes) noexcept {
  if (byte_range_overflows(first, first_bytes) ||
      byte_range_overflows(second, second_bytes)) {
    return false;
  }
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_end <= second_begin || second_end <= first_begin;
}

struct ByteSpan {
  const void* data = nullptr;
  std::size_t bytes = 0U;
};

template <std::size_t SpanCount>
[[nodiscard]] bool byte_ranges_are_pairwise_disjoint(
    const std::array<ByteSpan, SpanCount>& spans) noexcept {
  for (std::size_t left = 0U; left < spans.size(); ++left) {
    for (std::size_t right = left + 1U; right < spans.size(); ++right) {
      if (!byte_ranges_are_disjoint(
              spans[left].data, spans[left].bytes,
              spans[right].data, spans[right].bytes)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] ReferenceRunnerStatus runner_status(
    const ReferenceRunnerError error, const char* const operation,
    const std::size_t layer = kReferenceNoLayer,
    const int cuda_error = 0) noexcept {
  return {error, cuda_error, layer, operation};
}

[[nodiscard]] bool valid_vector(const Bf16VectorWeight& weight,
                                const std::size_t elements) noexcept {
  return weight.data != nullptr && weight.element_count == elements;
}

[[nodiscard]] bool valid_linear_payload(const LinearWeight& weight,
                                        const std::size_t output_size,
                                        const std::size_t input_size) noexcept {
  if (linear_output_size(weight) != output_size ||
      linear_input_size(weight) != input_size) {
    return false;
  }
  return std::visit(
      [](const auto& selected) noexcept {
        using Selected = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<Selected, Bf16LinearWeight>) {
          return selected.weight != nullptr;
        } else if constexpr (std::is_same_v<Selected, Fp8LinearWeight>) {
          return selected.weight != nullptr &&
                 selected.weight_scale_device != nullptr &&
                 selected.input_scale_device != nullptr &&
                 std::isfinite(selected.weight_scale) &&
                 selected.weight_scale >= 0.0F &&
                 std::isfinite(selected.input_scale) &&
                 selected.input_scale > 0.0F;
        } else {
          return selected.packed_weight != nullptr &&
                 selected.block_scale != nullptr &&
                 selected.weight_scale_2_device != nullptr &&
                 selected.input_scale_device != nullptr &&
                 std::isfinite(selected.weight_scale_2) &&
                 selected.weight_scale_2 >= 0.0F &&
                 std::isfinite(selected.input_scale) &&
                 selected.input_scale >= 0.0F &&
                 selected.input_size % 16U == 0U;
        }
      },
      weight);
}

[[nodiscard]] ReferenceRunnerStatus validate_model_weights(
    const ModelWeights* const weights) noexcept {
  if (weights == nullptr) {
    return runner_status(ReferenceRunnerError::kInvalidDependency,
                         "model_weights");
  }
  const Bf16LinearWeight& embedding = weights->embed_tokens();
  if (embedding.weight == nullptr ||
      embedding.output_size != kReferenceVocabularySize ||
      embedding.input_size != kReferenceHiddenSize) {
    return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                         "embed_tokens");
  }
  if (!valid_vector(weights->final_norm(), kReferenceHiddenSize) ||
      !valid_linear_payload(weights->lm_head(), kReferenceVocabularySize,
                            kReferenceHiddenSize)) {
    return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                         "final_norm_or_lm_head");
  }

  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& selected = weights->layer(layer);
    if (!valid_vector(selected.input_layernorm, kReferenceHiddenSize) ||
        !valid_vector(selected.post_attention_layernorm,
                      kReferenceHiddenSize) ||
        !valid_linear_payload(selected.mlp.gate_proj,
                              kReferenceIntermediateSize,
                              kReferenceHiddenSize) ||
        !valid_linear_payload(selected.mlp.up_proj,
                              kReferenceIntermediateSize,
                              kReferenceHiddenSize) ||
        !valid_linear_payload(selected.mlp.down_proj,
                              kReferenceHiddenSize,
                              kReferenceIntermediateSize)) {
      return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                           "decoder_common_weights", layer);
    }

    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      const auto* const attention =
          std::get_if<LinearAttentionWeights>(&selected.attention);
      if (attention == nullptr) {
        return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                             "linear_attention_variant", layer);
      }
      const bool conv_shape = attention->conv1d.data != nullptr &&
                              attention->conv1d.shape[0] ==
                                  kLinearQkvElements &&
                              attention->conv1d.shape[1] == 1U &&
                              attention->conv1d.shape[2] == 4U;
      if (!valid_linear_payload(attention->in_proj_qkv,
                                kLinearQkvElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->in_proj_z,
                                kLinearValueElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->in_proj_a,
                                kLinearScalarElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->in_proj_b,
                                kLinearScalarElements,
                                kReferenceHiddenSize) ||
          !conv_shape ||
          !valid_vector(attention->a_log, kLinearScalarElements) ||
          !valid_vector(attention->dt_bias, kLinearScalarElements) ||
          !valid_vector(attention->norm, kGdnHeadDimension) ||
          !valid_linear_payload(attention->out_proj,
                                kReferenceHiddenSize,
                                kLinearValueElements)) {
        return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                             "linear_attention_weights", layer);
      }
    } else if (expected == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&selected.attention);
      if (attention == nullptr) {
        return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                             "full_attention_variant", layer);
      }
      if (!valid_linear_payload(attention->q_proj, kFullQGateElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->k_proj, kFullKvElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->v_proj, kFullKvElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->o_proj, kReferenceHiddenSize,
                                kFullQueryElements) ||
          !valid_vector(attention->q_norm, kFullHeadDimension) ||
          !valid_vector(attention->k_norm, kFullHeadDimension)) {
        return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                             "full_attention_weights", layer);
      }
    } else {
      return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                           "layer_index", layer);
    }
  }
  return {};
}

[[nodiscard]] bool valid_view(const RequestViewResult& view,
                              const std::uint64_t minimum_elements,
                              const std::uint32_t element_size) noexcept {
  return view && view.value->device_data != nullptr &&
         view.value->element_capacity >= minimum_elements &&
         view.value->element_size_bytes == element_size;
}

[[nodiscard]] bool valid_const_view(const RequestConstViewResult& view,
                                    const std::uint64_t minimum_elements,
                                    const std::uint32_t element_size) noexcept {
  return view && view.value->device_data != nullptr &&
         view.value->element_capacity >= minimum_elements &&
         view.value->element_size_bytes == element_size;
}

#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
[[nodiscard]] bool use_prefill_gdn_b8_admission(
    const bool enabled, const ProjectionBackend backend,
    const std::uint32_t first_position,
    const std::size_t token_count) noexcept {
  return enabled && backend == ProjectionBackend::kSm87WeightOnly &&
         (token_count == 256U || token_count == 512U) &&
         first_position % 8U == 0U;
}
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
[[nodiscard]] bool use_prefill_gdn_chunk64_native_admission(
    const bool enabled, const ProjectionBackend backend,
    const std::uint32_t /*first_position*/,
    const std::size_t token_count, const void* const workspace,
    const std::size_t workspace_bytes) noexcept {
  // The prefix scheduler emits C512/C256/C64 tiles plus C32/<=31 tails.
  // Admit only complete C64 blocks at any sequence position; every tail
  // remains on the established ordered path, so no padded token can mutate
  // the recurrent state.
  return enabled && backend == ProjectionBackend::kSm87WeightOnly &&
         token_count >= 64U && token_count <= 512U &&
         token_count % 64U == 0U &&
         workspace != nullptr &&
         workspace_bytes >=
             gdn_prefill_chunk64_native_detail::workspace_bytes();
}
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
[[nodiscard]] bool use_prefill_gdn_chunk64_reference_admission(
    const bool enabled, const ProjectionBackend backend,
    const std::uint32_t first_position, const std::size_t token_count,
    const void* const context, const void* const workspace,
    const std::size_t workspace_bytes) noexcept {
  return enabled && backend == ProjectionBackend::kSm87WeightOnly &&
         first_position == 0U && token_count == 512U && context != nullptr &&
         workspace != nullptr &&
         workspace_bytes >=
             gdn_prefill_chunk64_reference_detail::workspace_bytes();
}
#endif
[[nodiscard]] bool should_use_prefill_gdn_c16_norm_gate(
    const bool enabled, const ProjectionBackend backend,
    const std::uint32_t first_position,
    const std::size_t token_count) noexcept {
  return enabled && backend == ProjectionBackend::kSm87WeightOnly &&
         (token_count == 256U || token_count == 512U) &&
         first_position % kPrefillKernelTileMaximumTokens == 0U;
}

}  // namespace

ReferenceRunnerStatus ReferenceRunner::collect_request_views(
    RequestState* const state, Views& views) noexcept {
  if (state == nullptr) {
    return runner_status(ReferenceRunnerError::kInvalidDependency,
                         "request_state");
  }
  if (!static_cast<bool>(*state)) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "empty_request_state");
  }
  const ReferenceRunnerError plan_error =
      reference_runner_detail::validate_reference_workspace_plan(state->plan());
  if (plan_error != ReferenceRunnerError::kNone) {
    return runner_status(plan_error, "request_memory_plan");
  }
  if (state->sequence_length() > state->max_sequence_length()) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "request_sequence_length");
  }
  const std::uint64_t workspace_tokens = state->plan().prefill_chunk_size;

  for (std::size_t index = 0U; index < 3U; ++index) {
    RequestViewResult view = state->hidden_buffer(index);
    if (!valid_view(view, workspace_tokens * kReferenceHiddenSize,
                    sizeof(std::uint16_t))) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "hidden_workspace");
    }
    views.hidden[index] =
        static_cast<std::uint16_t*>(view.value->device_data);
  }
  for (std::size_t index = 0U; index < 4U; ++index) {
    RequestViewResult view = state->projection_buffer(index);
    if (!valid_view(view, workspace_tokens * kReferenceIntermediateSize,
                    sizeof(std::uint16_t))) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "projection_workspace");
    }
    views.projection[index] =
        static_cast<std::uint16_t*>(view.value->device_data);
  }

  RequestViewResult linear_a = state->linear_a_buffer();
  RequestViewResult linear_b = state->linear_b_buffer();
  RequestViewResult scratch = state->fp32_scratch();
  if (!valid_view(linear_a, workspace_tokens * kLinearScalarElements,
                  sizeof(std::uint16_t)) ||
      !valid_view(linear_b, workspace_tokens * kLinearScalarElements,
                  sizeof(std::uint16_t)) ||
      !valid_view(scratch, kReferenceVocabularySize, sizeof(float))) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "scalar_or_fp32_workspace");
  }
  views.linear_a = static_cast<std::uint16_t*>(linear_a.value->device_data);
  views.linear_b = static_cast<std::uint16_t*>(linear_b.value->device_data);
  views.fp32_scratch = static_cast<float*>(scratch.value->device_data);
  views.fp32_scratch_elements =
      static_cast<std::size_t>(scratch.value->element_capacity);
  const RequestConstViewResult rope_cos = state->rope_cos(0U);
  const RequestConstViewResult rope_sin = state->rope_sin(0U);
  if (!valid_const_view(rope_cos, kRopePairs, sizeof(float)) ||
      !valid_const_view(rope_sin, kRopePairs, sizeof(float))) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "rope_workspace");
  }
  views.rope_cos = static_cast<const float*>(rope_cos.value->device_data);
  views.rope_sin = static_cast<const float*>(rope_sin.value->device_data);

  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      RequestViewResult conv = state->conv_state(layer);
      RequestViewResult gdn = state->gdn_state(layer);
      if (!valid_view(conv, kGdnQkvChannels * kGdnConvHistoryWidth,
                      sizeof(std::uint16_t)) ||
          !valid_view(gdn, kGdnStateElements, sizeof(std::uint16_t))) {
        return runner_status(ReferenceRunnerError::kInvalidRequestState,
                             "linear_persistent_state", layer);
      }
      views.conv_state[layer] =
          static_cast<std::uint16_t*>(conv.value->device_data);
      views.gdn_state[layer] =
          static_cast<std::uint16_t*>(gdn.value->device_data);
    } else if (expected == model::LayerType::kFullAttention) {
      RequestViewResult key = state->key_cache(layer);
      RequestViewResult value = state->value_cache(layer);
      const std::uint64_t required =
          static_cast<std::uint64_t>(state->max_sequence_length()) *
          kFullKvElements;
      if (!valid_view(key, required, sizeof(std::uint16_t)) ||
          !valid_view(value, required, sizeof(std::uint16_t))) {
        return runner_status(ReferenceRunnerError::kInvalidRequestState,
                             "full_attention_cache", layer);
      }
      views.key_cache[layer] =
          static_cast<std::uint16_t*>(key.value->device_data);
      views.value_cache[layer] =
          static_cast<std::uint16_t*>(value.value->device_data);
    } else {
      return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                           "request_layer_schedule", layer);
    }
  }
  return {};
}

namespace {

[[nodiscard]] ConstBf16Span trace_span(
    const ReferenceTraceView& trace, const std::size_t offset) noexcept {
  if (trace.data == nullptr || trace.element_count < kReferenceTraceElements ||
      offset > kReferenceTraceElements - kReferenceHiddenSize) {
    return {};
  }
  return {trace.data + offset, kReferenceHiddenSize};
}

}  // namespace

const char* reference_runner_error_string(
    const ReferenceRunnerError error) noexcept {
  switch (error) {
    case ReferenceRunnerError::kNone:
      return "none";
    case ReferenceRunnerError::kInvalidDependency:
      return "invalid_dependency";
    case ReferenceRunnerError::kInvalidModelWeights:
      return "invalid_model_weights";
    case ReferenceRunnerError::kInvalidRequestState:
      return "invalid_request_state";
    case ReferenceRunnerError::kInvalidLayerSchedule:
      return "invalid_layer_schedule";
    case ReferenceRunnerError::kCudaFailure:
      return "cuda_failure";
    case ReferenceRunnerError::kAllocationFailure:
      return "allocation_failure";
    case ReferenceRunnerError::kInvalidRunner:
      return "invalid_runner";
    case ReferenceRunnerError::kPoisoned:
      return "poisoned";
    case ReferenceRunnerError::kTokenOutOfRange:
      return "token_out_of_range";
    case ReferenceRunnerError::kCapacityExceeded:
      return "capacity_exceeded";
    case ReferenceRunnerError::kTraceUnavailable:
      return "trace_unavailable";
    case ReferenceRunnerError::kNonFiniteLogits:
      return "nonfinite_logits";
    case ReferenceRunnerError::kStateCommitFailure:
      return "state_commit_failure";
    case ReferenceRunnerError::kInvalidStepOptions:
      return "invalid_step_options";
  }
  return "unknown";
}

ConstBf16Span ReferenceTraceView::raw() const noexcept {
  if (data == nullptr || element_count < kReferenceTraceElements) {
    return {};
  }
  return {data, kReferenceTraceElements};
}

ConstBf16Span ReferenceTraceView::embedding() const noexcept {
  return trace_span(*this, 0U);
}

ConstBf16Span ReferenceTraceView::layer_hidden(
    const std::size_t layer) const noexcept {
  if (layer >= kReferenceDecoderLayerCount) {
    return {};
  }
  const std::size_t offset =
      kReferenceHiddenSize + 2U * layer * kReferenceHiddenSize;
  return trace_span(*this, offset);
}

ConstBf16Span ReferenceTraceView::layer_residual(
    const std::size_t layer) const noexcept {
  if (layer >= kReferenceDecoderLayerCount) {
    return {};
  }
  const std::size_t offset =
      kReferenceHiddenSize + (2U * layer + 1U) * kReferenceHiddenSize;
  return trace_span(*this, offset);
}

ConstBf16Span ReferenceTraceView::final_norm() const noexcept {
  const std::size_t offset =
      (1U + 2U * kReferenceDecoderLayerCount) * kReferenceHiddenSize;
  return trace_span(*this, offset);
}

namespace reference_runner_detail {

bool exchange_nvfp4_marlin_prefill_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
  return std::exchange(g_enable_nvfp4_marlin_prefill_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

std::size_t exchange_nvfp4_marlin_prefill_admission_test_hits(
    const std::size_t hits) noexcept {
#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
  return std::exchange(g_nvfp4_marlin_prefill_admission_hits, hits);
#else
  (void)hits;
  return 0U;
#endif
}

#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
// Private test hook for real-checkpoint admission. Keeping this declaration
// out of the installed header prevents the numerically distinct B8 recurrence
// from becoming a supported runtime option before its state/Decode gates pass.
bool exchange_prefill_gdn_b8_admission_test_enabled(
    const bool enabled) noexcept {
  return std::exchange(g_enable_prefill_gdn_b8_admission, enabled);
}

std::size_t exchange_prefill_gdn_b8_admission_test_hits(
    const std::size_t hits) noexcept {
  return std::exchange(g_prefill_gdn_b8_admission_hits, hits);
}
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
bool exchange_prefill_gdn_chunk64_native_admission_test_enabled(
    const bool enabled) noexcept {
  return std::exchange(g_enable_prefill_gdn_chunk64_native_admission,
                       enabled);
}

std::size_t exchange_prefill_gdn_chunk64_native_admission_test_hits(
    const std::size_t hits) noexcept {
  return std::exchange(g_prefill_gdn_chunk64_native_admission_hits, hits);
}

PrefillGdnChunk64NativeSnapshotHook
exchange_prefill_gdn_chunk64_native_snapshot_hook(
    const PrefillGdnChunk64NativeSnapshotHook hook) noexcept {
  return std::exchange(g_prefill_gdn_chunk64_native_snapshot_hook, hook);
}
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
bool exchange_prefill_gdn_chunk64_reference_admission_test_enabled(
    const bool enabled) noexcept {
  return std::exchange(
      g_enable_prefill_gdn_chunk64_reference_admission, enabled);
}

std::size_t exchange_prefill_gdn_chunk64_reference_admission_test_hits(
    const std::size_t hits) noexcept {
  return std::exchange(
      g_prefill_gdn_chunk64_reference_admission_hits, hits);
}

PrefillGdnChunk64ReferenceSnapshotHook
exchange_prefill_gdn_chunk64_reference_snapshot_hook(
    const PrefillGdnChunk64ReferenceSnapshotHook hook) noexcept {
  return std::exchange(g_prefill_gdn_chunk64_reference_snapshot_hook, hook);
}
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
bool exchange_prefill_gdn_c16_norm_gate_admission_test_enabled(
    const bool enabled) noexcept {
  return std::exchange(g_enable_prefill_gdn_c16_norm_gate_admission,
                       enabled);
}

std::size_t exchange_prefill_gdn_c16_norm_gate_admission_test_hits(
    const std::size_t hits) noexcept {
  return std::exchange(g_prefill_gdn_c16_norm_gate_admission_hits, hits);
}

PrefillGdnC16NormGateAdmissionSnapshotHook
exchange_prefill_gdn_c16_norm_gate_admission_snapshot_hook(
    const PrefillGdnC16NormGateAdmissionSnapshotHook hook) noexcept {
  return std::exchange(
      g_prefill_gdn_c16_norm_gate_admission_snapshot_hook, hook);
}
#endif

std::uint16_t float_to_bf16_rne(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t exponent = bits & 0x7F80'0000U;
  const std::uint32_t mantissa = bits & 0x007F'FFFFU;
  if (exponent == 0x7F80'0000U && mantissa != 0U) {
    // Preserve sign/payload high bits and force a quiet BF16 NaN.
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t rounding_bias =
      0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

float bf16_to_float(const std::uint16_t bits) noexcept {
  const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

float round_float_to_bf16(const float value) noexcept {
  return bf16_to_float(float_to_bf16_rne(value));
}

LogitsAnalysis analyze_bf16_argmax_in_place(
    float* const logits, const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  std::size_t maximum_index = 0U;
  float maximum = 0.0F;
  bool all_finite = true;
  for (std::size_t index = 0U; index < element_count; ++index) {
    const float value = round_float_to_bf16(logits[index]);
    logits[index] = value;
    all_finite = all_finite && std::isfinite(value);
    if (index == 0U || value > maximum) {
      maximum = value;
      maximum_index = index;
    }
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  return result;
}

LogitsAnalysis analyze_bf16_argmax_bits(
    const std::uint16_t* const logits,
    const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  std::size_t maximum_index = 0U;
  float maximum = bf16_to_float(logits[0]);
  bool all_finite = std::isfinite(maximum);
  for (std::size_t index = 1U; index < element_count; ++index) {
    const float value = bf16_to_float(logits[index]);
    all_finite = all_finite && std::isfinite(value);
    if (value > maximum) {
      maximum = value;
      maximum_index = index;
    }
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  return result;
}

namespace {

constexpr std::size_t kBf16CodeCount = 1U << 16U;

struct Bf16ExpMemoCache {
  std::array<double, kBf16CodeCount> values{};
  std::array<std::uint8_t, kBf16CodeCount> seen_stamps{};
  std::uint8_t generation = 0U;
  bool in_use = false;
};

thread_local Bf16ExpMemoCache bf16_exp_memo_cache{};

class ScopedBf16ExpMemoUse {
 public:
  explicit ScopedBf16ExpMemoUse(Bf16ExpMemoCache& cache) noexcept
      : cache_(cache) {
    cache_.in_use = true;
  }

  ScopedBf16ExpMemoUse(const ScopedBf16ExpMemoUse&) = delete;
  ScopedBf16ExpMemoUse& operator=(const ScopedBf16ExpMemoUse&) = delete;

  ~ScopedBf16ExpMemoUse() noexcept { cache_.in_use = false; }

 private:
  Bf16ExpMemoCache& cache_;
};

[[nodiscard]] LogitsAnalysis analyze_bf16_logits_bits_scalar(
    const std::uint16_t* const logits,
    const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  std::size_t maximum_index = 0U;
  float maximum = bf16_to_float(logits[0]);
  bool all_finite = std::isfinite(maximum);
  for (std::size_t index = 1U; index < element_count; ++index) {
    const float value = bf16_to_float(logits[index]);
    all_finite = all_finite && std::isfinite(value);
    if (value > maximum) {
      maximum = value;
      maximum_index = index;
    }
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }

  double exponential_sum = 0.0;
  for (std::size_t index = 0U; index < element_count; ++index) {
    exponential_sum +=
        std::exp(static_cast<double>(bf16_to_float(logits[index])) -
                 static_cast<double>(maximum));
  }
  const double logsumexp =
      static_cast<double>(maximum) + std::log(exponential_sum);
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  result.logsumexp = logsumexp;
  result.max_log_probability = static_cast<double>(maximum) - logsumexp;
  return result;
}

}  // namespace

LogitsAnalysis analyze_bf16_logits_in_place(
    float* const logits, const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  bool all_finite = true;
  for (std::size_t index = 0U; index < element_count; ++index) {
    logits[index] = round_float_to_bf16(logits[index]);
    all_finite = all_finite && std::isfinite(logits[index]);
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }

  std::size_t maximum_index = 0U;
  float maximum = logits[0];
  for (std::size_t index = 1U; index < element_count; ++index) {
    // Strict comparison preserves the earliest (smallest-id) tie.
    if (logits[index] > maximum) {
      maximum = logits[index];
      maximum_index = index;
    }
  }
  double exponential_sum = 0.0;
  for (std::size_t index = 0U; index < element_count; ++index) {
    exponential_sum +=
        std::exp(static_cast<double>(logits[index]) -
                 static_cast<double>(maximum));
  }
  const double logsumexp =
      static_cast<double>(maximum) + std::log(exponential_sum);
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  result.logsumexp = logsumexp;
  result.max_log_probability = static_cast<double>(maximum) - logsumexp;
  return result;
}

LogitsAnalysis analyze_bf16_logits_bits(
    const std::uint16_t* const logits,
    const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  Bf16ExpMemoCache& cache = bf16_exp_memo_cache;
  if (cache.in_use) {
    // A same-thread reentrant call must not overwrite the outer invocation's
    // stamps or memoized values.
    return analyze_bf16_logits_bits_scalar(logits, element_count);
  }
  const ScopedBf16ExpMemoUse cache_use(cache);
  std::size_t maximum_index = 0U;
  float maximum = bf16_to_float(logits[0]);
  bool all_finite = std::isfinite(maximum);
  for (std::size_t index = 1U; index < element_count; ++index) {
    const float value = bf16_to_float(logits[index]);
    all_finite = all_finite && std::isfinite(value);
    if (value > maximum) {
      maximum = value;
      maximum_index = index;
    }
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }

  if (cache.generation == std::numeric_limits<std::uint8_t>::max()) {
    cache.seen_stamps.fill(0U);
    cache.generation = 1U;
  } else {
    cache.generation =
        static_cast<std::uint8_t>(cache.generation + 1U);
  }
  const double maximum_double = static_cast<double>(maximum);
  double exponential_sum = 0.0;
  for (std::size_t index = 0U; index < element_count; ++index) {
    const std::uint16_t code = logits[index];
    if (cache.seen_stamps[code] != cache.generation) {
      cache.seen_stamps[code] = cache.generation;
      cache.values[code] =
          std::exp(static_cast<double>(bf16_to_float(code)) -
                   maximum_double);
    }
    // Preserve the scalar oracle's original index order and double-addition
    // order. Only the repeated, deterministic exp evaluation is memoized.
    exponential_sum += cache.values[code];
  }
  const double logsumexp = maximum_double + std::log(exponential_sum);
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  result.logsumexp = logsumexp;
  result.max_log_probability = static_cast<double>(maximum) - logsumexp;
  return result;
}

bool valid_reference_linear_weight_contract(
    const LinearWeight& weight, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  return valid_linear_payload(weight, output_size, input_size);
}

model::LayerType expected_reference_layer_type(
    const std::size_t layer) noexcept {
  if (layer >= kReferenceDecoderLayerCount) {
    return model::LayerType::kInvalid;
  }
  return ((layer + 1U) % 4U) == 0U
             ? model::LayerType::kFullAttention
             : model::LayerType::kLinearAttention;
}

bool use_fused_gqa_sigmoid_gate_tile(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  return token_count != 0U &&
         fused_gqa_sigmoid_gate_prefix_token_count(first_position,
                                                   token_count) == token_count;
}

std::size_t fused_gqa_sigmoid_gate_prefix_token_count(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  if (token_count == 0U ||
      first_position >= kFusedGqaMaximumSequenceLength) {
    return 0U;
  }
  return std::min(token_count,
                  kFusedGqaMaximumSequenceLength - first_position);
}

bool use_bulk_causal_gqa_sigmoid_gate_prefill(
    const ProjectionBackend backend, const model::LayerType layer_type,
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  return backend == ProjectionBackend::kSm87WeightOnly &&
         layer_type == model::LayerType::kFullAttention &&
         (token_count == 256U || token_count == 512U) &&
         first_position <=
             kBulkCausalGqaMaximumSequenceLength - token_count;
}

bool use_qk_rope_tile(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  if (token_count == 0U || token_count > kQkRopeTileMaximumTokens ||
      first_position >
          std::numeric_limits<std::size_t>::max() - token_count) {
    return false;
  }
  return first_position + token_count <=
         std::numeric_limits<std::size_t>::max() /
             (kRopePairs * sizeof(float));
}

bool use_m32_prefill_residual_rms_fusion(
    const std::size_t token_count,
    const std::size_t hidden_size) noexcept {
  return token_count != 0U &&
         token_count <= kMaximumRequestPrefillChunkSize &&
         token_count % kProductionProjectionSubtileTokens == 0U &&
         hidden_size == kReferenceHiddenSize;
}

bool use_fp8_whole_chunk_prefill_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kQkvRows = 10'240U;
  constexpr std::size_t kZRows = 6'144U;
  constexpr std::size_t kHiddenSize = 5'120U;
  constexpr std::size_t kFullQueryGateRows = 12'288U;
  constexpr std::size_t kFullKvRows = 1'024U;
  const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      (token_count != 256U && token_count != 512U) ||
      selected == nullptr) {
    return false;
  }
  const bool qkv_shape = selected->output_size == kQkvRows &&
                         selected->input_size == kHiddenSize;
  const bool z_shape = selected->output_size == kZRows &&
                       selected->input_size == kHiddenSize;
  const bool attention_output_shape =
      selected->output_size == kHiddenSize &&
      selected->input_size == kZRows;
  const bool full_query_shape =
      selected->output_size == kFullQueryGateRows &&
      selected->input_size == kHiddenSize;
  const bool full_kv_shape = selected->output_size == kFullKvRows &&
                             selected->input_size == kHiddenSize;
  return (qkv_shape || z_shape || attention_output_shape ||
          full_query_shape || full_kv_shape) &&
         aligned(selected->weight, 16U) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(output, alignof(std::uint16_t));
}

bool use_fp8_m64_prefill_attention_output_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kRows = 5'120U;
  constexpr std::size_t kColumns = 6'144U;
  const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  return backend == ProjectionBackend::kSm87WeightOnly &&
         token_count == kMaximumProjectionTileTokenCount &&
         selected != nullptr && selected->output_size == kRows &&
         selected->input_size == kColumns &&
         aligned(selected->weight, 16U) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(output, alignof(std::uint16_t));
}

bool use_nvfp4_whole_chunk_prefill_down_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kRows = 5'120U;
  constexpr std::size_t kColumns = 17'408U;
  const auto* const selected = std::get_if<NvFp4LinearWeight>(&weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  return backend == ProjectionBackend::kSm87WeightOnly &&
         (token_count == 256U || token_count == 512U) &&
         selected != nullptr && selected->output_size == kRows &&
         selected->input_size == kColumns &&
         aligned(selected->packed_weight, 16U) &&
         aligned(selected->block_scale, alignof(std::uint16_t)) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(output, alignof(std::uint16_t));
}

bool use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* const input,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kRows = 17'408U;
  constexpr std::size_t kColumns = 5'120U;
  const auto* const gate = std::get_if<NvFp4LinearWeight>(&gate_weight);
  const auto* const up = std::get_if<NvFp4LinearWeight>(&up_weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  if (token_count != 256U && token_count != 512U) {
    return false;
  }
  const std::size_t output_bytes =
      token_count * kRows * sizeof(std::uint16_t);
  return backend == ProjectionBackend::kSm87WeightOnly &&
         gate != nullptr && up != nullptr &&
         gate->output_size == kRows && gate->input_size == kColumns &&
         up->output_size == kRows && up->input_size == kColumns &&
         aligned(gate->packed_weight, 16U) &&
         aligned(up->packed_weight, 16U) &&
         aligned(gate->block_scale, alignof(std::uint16_t)) &&
         aligned(up->block_scale, alignof(std::uint16_t)) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(gate_output, alignof(std::uint16_t)) &&
         aligned(up_output, alignof(std::uint16_t)) &&
         byte_ranges_are_disjoint(gate_output, output_bytes, up_output,
                                  output_bytes);
}

bool use_nvfp4_m32_prefill_gate_up_dual_stream(
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* const input,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kRows = 17'408U;
  constexpr std::size_t kColumns = 5'120U;
  const auto* const gate = std::get_if<NvFp4LinearWeight>(&gate_weight);
  const auto* const up = std::get_if<NvFp4LinearWeight>(&up_weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  if (token_count != kProductionProjectionSubtileTokens &&
      token_count != 2U * kProductionProjectionSubtileTokens) {
    return false;
  }
  const std::size_t output_bytes =
      token_count * kRows * sizeof(std::uint16_t);
  return backend == ProjectionBackend::kSm87WeightOnly &&
         gate != nullptr && up != nullptr &&
         gate->output_size == kRows && gate->input_size == kColumns &&
         up->output_size == kRows && up->input_size == kColumns &&
         aligned(gate->packed_weight, 16U) &&
         aligned(up->packed_weight, 16U) &&
         aligned(gate->block_scale, alignof(std::uint16_t)) &&
         aligned(up->block_scale, alignof(std::uint16_t)) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(gate_output, alignof(std::uint16_t)) &&
         aligned(up_output, alignof(std::uint16_t)) &&
         byte_ranges_are_disjoint(gate_output, output_bytes, up_output,
                                  output_bytes);
}

ReferenceRunnerError validate_reference_workspace_plan(
    const RequestMemoryPlan& plan) noexcept {
  if (plan.batch_size != 1U || plan.prefill_chunk_size == 0U ||
      plan.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
      plan.max_sequence_length == 0U ||
      plan.max_sequence_length > kAbsoluteRequestMaxSequenceLength ||
      plan.hidden_bf16.size() != 3U ||
      plan.projection_bf16.size() != 4U) {
    return ReferenceRunnerError::kInvalidRequestState;
  }
  const auto bf16_region_at_least = [](const RequestRegion& region,
                                       const std::uint64_t elements) noexcept {
    return region.element_size_bytes == sizeof(std::uint16_t) &&
           region.element_capacity >= elements &&
           region.byte_size >= elements * sizeof(std::uint16_t);
  };
  const auto fp32_region_at_least = [](const RequestRegion& region,
                                       const std::uint64_t elements) noexcept {
    return region.element_size_bytes == sizeof(float) &&
           region.element_capacity >= elements &&
           region.byte_size >= elements * sizeof(float);
  };

  for (const RequestRegion& region : plan.hidden_bf16) {
    if (!bf16_region_at_least(
            region, static_cast<std::uint64_t>(plan.prefill_chunk_size) *
                        kReferenceHiddenSize)) {
      return ReferenceRunnerError::kInvalidRequestState;
    }
  }
  for (const RequestRegion& region : plan.projection_bf16) {
    if (!bf16_region_at_least(
            region, static_cast<std::uint64_t>(plan.prefill_chunk_size) *
                        kReferenceIntermediateSize)) {
      return ReferenceRunnerError::kInvalidRequestState;
    }
  }
  if (!bf16_region_at_least(
          plan.linear_a_bf16,
          static_cast<std::uint64_t>(plan.prefill_chunk_size) *
              kLinearScalarElements) ||
      !bf16_region_at_least(
          plan.linear_b_bf16,
          static_cast<std::uint64_t>(plan.prefill_chunk_size) *
              kLinearScalarElements) ||
      !fp32_region_at_least(plan.fp32_scratch,
                            kReferenceVocabularySize) ||
      plan.gqa_probability_scratch.arena_offset !=
          plan.fp32_scratch.arena_offset ||
      !fp32_region_at_least(
          plan.gqa_probability_scratch,
          static_cast<std::uint64_t>(kFullQueryHeads) *
              plan.max_sequence_length) ||
      !fp32_region_at_least(
          plan.rope_cos_fp32,
          static_cast<std::uint64_t>(kRopePairs) *
              plan.max_sequence_length) ||
      !fp32_region_at_least(
          plan.rope_sin_fp32,
          static_cast<std::uint64_t>(kRopePairs) *
              plan.max_sequence_length) ||
      !bf16_region_at_least(
          plan.conv_state,
          static_cast<std::uint64_t>(kRequestLinearLayerCount) *
              kGdnQkvChannels * kGdnConvHistoryWidth) ||
      !bf16_region_at_least(
          plan.gdn_state,
          static_cast<std::uint64_t>(kRequestLinearLayerCount) *
              kGdnStateElements)) {
    return ReferenceRunnerError::kInvalidRequestState;
  }

  for (std::size_t slot = 0U; slot < kRequestFullLayerCount; ++slot) {
    const std::uint64_t required =
        static_cast<std::uint64_t>(plan.max_sequence_length) *
        kFullKvElements;
    if (!bf16_region_at_least(plan.key_cache[slot], required) ||
        !bf16_region_at_least(plan.value_cache[slot], required)) {
      return ReferenceRunnerError::kInvalidRequestState;
    }
  }

  std::size_t linear_slot = 0U;
  std::size_t full_slot = 0U;
  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const model::LayerType expected = expected_reference_layer_type(layer);
    const RequestLayerSlot& actual = plan.layers[layer];
    const std::size_t expected_slot =
        expected == model::LayerType::kFullAttention ? full_slot++
                                                     : linear_slot++;
    if (actual.type != expected || actual.slot != expected_slot) {
      return ReferenceRunnerError::kInvalidLayerSchedule;
    }
  }
  if (linear_slot != kRequestLinearLayerCount ||
      full_slot != kRequestFullLayerCount) {
    return ReferenceRunnerError::kInvalidLayerSchedule;
  }
  return ReferenceRunnerError::kNone;
}

}  // namespace reference_runner_detail

ReferenceRunner::~ReferenceRunner() { release(); }

ReferenceRunner::ReferenceRunner(ReferenceRunner&& other) noexcept
    : ReferenceRunner() {
  *this = std::move(other);
}

ReferenceRunner& ReferenceRunner::operator=(ReferenceRunner&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  weights_ = std::exchange(other.weights_, nullptr);
  state_ = std::exchange(other.state_, nullptr);
  stream_ = std::exchange(other.stream_, nullptr);
  prefill_auxiliary_stream_ =
      std::exchange(other.prefill_auxiliary_stream_, nullptr);
  prefill_branch_ready_event_ =
      std::exchange(other.prefill_branch_ready_event_, nullptr);
  prefill_branch_done_event_ =
      std::exchange(other.prefill_branch_done_event_, nullptr);
  prefill_gdn_chunk64_reference_context_ =
      std::exchange(other.prefill_gdn_chunk64_reference_context_, nullptr);
  prefill_gdn_chunk64_reference_workspace_ =
      std::exchange(other.prefill_gdn_chunk64_reference_workspace_, nullptr);
  prefill_gdn_chunk64_reference_workspace_bytes_ = std::exchange(
      other.prefill_gdn_chunk64_reference_workspace_bytes_, 0U);
  prefill_gdn_chunk64_native_workspace_ =
      std::exchange(other.prefill_gdn_chunk64_native_workspace_, nullptr);
  prefill_gdn_chunk64_native_workspace_bytes_ = std::exchange(
      other.prefill_gdn_chunk64_native_workspace_bytes_, 0U);
  pinned_logits_ = std::exchange(other.pinned_logits_, nullptr);
  pinned_trace_ = std::exchange(other.pinned_trace_, nullptr);
  decode_graph_p1_slots_ = other.decode_graph_p1_slots_;
  other.decode_graph_p1_slots_ = {};
  decode_graph_capture_active_ =
      std::exchange(other.decode_graph_capture_active_, false);
  views_ = other.views_;
  other.views_ = {};
  projection_backend_ = std::exchange(
      other.projection_backend_, ProjectionBackend::kReference);
  trace_enabled_ = std::exchange(other.trace_enabled_, false);
  trace_valid_ = std::exchange(other.trace_valid_, false);
  poisoned_ = std::exchange(other.poisoned_, false);
  retained_prefill_hidden_valid_ =
      std::exchange(other.retained_prefill_hidden_valid_, false);
  retained_prefill_position_ =
      std::exchange(other.retained_prefill_position_, 0U);
  retained_prefill_input_token_ =
      std::exchange(other.retained_prefill_input_token_, 0U);
  retained_prefill_hidden_row_ =
      std::exchange(other.retained_prefill_hidden_row_, 0U);
  trace_position_ = std::exchange(other.trace_position_, 0U);
  trace_input_token_ = std::exchange(other.trace_input_token_, 0U);
  return *this;
}

ReferenceRunner::operator bool() const noexcept {
  bool ready = weights_ != nullptr && state_ != nullptr &&
               stream_ != nullptr && pinned_logits_ != nullptr;
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  ready = ready && prefill_gdn_chunk64_reference_context_ != nullptr &&
          prefill_gdn_chunk64_reference_workspace_ != nullptr &&
          prefill_gdn_chunk64_reference_workspace_bytes_ >=
              gdn_prefill_chunk64_reference_detail::workspace_bytes();
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  ready = ready && prefill_gdn_chunk64_native_workspace_ != nullptr &&
          prefill_gdn_chunk64_native_workspace_bytes_ >=
              gdn_prefill_chunk64_native_detail::workspace_bytes();
#endif
  return ready;
}

std::uint32_t ReferenceRunner::current_position() const noexcept {
  return state_ == nullptr ? 0U : state_->current_position();
}

void ReferenceRunner::release() noexcept {
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    (void)cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
  }
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  if (prefill_gdn_chunk64_reference_context_ != nullptr) {
    (void)gdn_prefill_chunk64_reference_detail::destroy_context(
        prefill_gdn_chunk64_reference_context_);
  }
  if (prefill_gdn_chunk64_reference_workspace_ != nullptr) {
    (void)cudaFree(prefill_gdn_chunk64_reference_workspace_);
  }
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  if (prefill_gdn_chunk64_native_workspace_ != nullptr) {
    (void)cudaFree(prefill_gdn_chunk64_native_workspace_);
  }
#endif
  destroy_decode_graph_p1();
  if (prefill_branch_ready_event_ != nullptr) {
    (void)cudaEventDestroy(
        reinterpret_cast<cudaEvent_t>(prefill_branch_ready_event_));
  }
  if (prefill_branch_done_event_ != nullptr) {
    (void)cudaEventDestroy(
        reinterpret_cast<cudaEvent_t>(prefill_branch_done_event_));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    (void)cudaStreamDestroy(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
  }
  if (stream_ != nullptr) {
    (void)cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
  }
  if (pinned_trace_ != nullptr) {
    (void)cudaFreeHost(pinned_trace_);
  }
  if (pinned_logits_ != nullptr) {
    (void)cudaFreeHost(pinned_logits_);
  }
  weights_ = nullptr;
  state_ = nullptr;
  stream_ = nullptr;
  prefill_auxiliary_stream_ = nullptr;
  prefill_branch_ready_event_ = nullptr;
  prefill_branch_done_event_ = nullptr;
  prefill_gdn_chunk64_reference_context_ = nullptr;
  prefill_gdn_chunk64_reference_workspace_ = nullptr;
  prefill_gdn_chunk64_reference_workspace_bytes_ = 0U;
  prefill_gdn_chunk64_native_workspace_ = nullptr;
  prefill_gdn_chunk64_native_workspace_bytes_ = 0U;
  pinned_logits_ = nullptr;
  pinned_trace_ = nullptr;
  decode_graph_capture_active_ = false;
  views_ = {};
  projection_backend_ = ProjectionBackend::kReference;
  trace_enabled_ = false;
  trace_valid_ = false;
  poisoned_ = false;
  retained_prefill_hidden_valid_ = false;
  retained_prefill_position_ = 0U;
  retained_prefill_input_token_ = 0U;
  retained_prefill_hidden_row_ = 0U;
  trace_position_ = 0U;
  trace_input_token_ = 0U;
}

int ReferenceRunner::destroy_decode_graph_p1_slot(
    DecodeGraphP1Slot& slot) noexcept {
  int first_error = static_cast<int>(cudaSuccess);
  if (slot.exec != nullptr) {
    const cudaError_t status = cudaGraphExecDestroy(
        reinterpret_cast<cudaGraphExec_t>(slot.exec));
    if (status != cudaSuccess) {
      first_error = static_cast<int>(status);
    }
  }
  if (slot.graph != nullptr) {
    const cudaError_t status =
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(slot.graph));
    if (status != cudaSuccess && first_error == static_cast<int>(cudaSuccess)) {
      first_error = static_cast<int>(status);
    }
  }
  slot = {};
  return first_error;
}

void ReferenceRunner::destroy_decode_graph_p1_slot(
    const std::size_t position) noexcept {
  if (position >= decode_graph_p1_slots_.size()) {
    return;
  }
  (void)destroy_decode_graph_p1_slot(decode_graph_p1_slots_[position]);
}

void ReferenceRunner::destroy_decode_graph_p1() noexcept {
  for (std::size_t position = 0U;
       position < decode_graph_p1_slots_.size(); ++position) {
    destroy_decode_graph_p1_slot(position);
  }
}

ReferenceStepOutcome ReferenceRunner::fail_step(
    const ReferenceRunnerStatus status) noexcept {
  // A failed launch may follow earlier successful launches in this token.
  // Drain the owned stream before returning so every step has a synchronous
  // completion boundary even though its mutated device state is not committed
  // and cannot be reused until reset.
  if (decode_graph_capture_active_ && stream_ != nullptr) {
    cudaGraph_t discarded_graph = nullptr;
    (void)cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(stream_),
                               &discarded_graph);
    decode_graph_capture_active_ = false;
    if (discarded_graph != nullptr) {
      (void)cudaGraphDestroy(discarded_graph);
    }
    (void)cudaGetLastError();
  }
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    (void)cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
  }
  poisoned_ = true;
  trace_valid_ = false;
  retained_prefill_hidden_valid_ = false;
  ReferenceStepOutcome outcome;
  outcome.status = status;
  return outcome;
}

ReferencePrefillTileOutcome ReferenceRunner::fail_prefill_tile(
    const ReferenceRunnerStatus status) noexcept {
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    (void)cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
  }
  poisoned_ = true;
  trace_valid_ = false;
  retained_prefill_hidden_valid_ = false;
  ReferencePrefillTileOutcome outcome;
  outcome.status = status;
  return outcome;
}

std::optional<ReferenceTraceView> ReferenceRunner::last_trace() const noexcept {
  if (!trace_valid_ || pinned_trace_ == nullptr) {
    return std::nullopt;
  }
  return ReferenceTraceView{trace_position_, trace_input_token_, pinned_trace_,
                            kReferenceTraceElements};
}

ReferenceRunnerStatus ReferenceRunner::reset() noexcept {
  if (!static_cast<bool>(*this)) {
    return runner_status(ReferenceRunnerError::kInvalidRunner, "reset");
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    const cudaError_t auxiliary_sync_status = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
    if (auxiliary_sync_status != cudaSuccess) {
      poisoned_ = true;
      trace_valid_ = false;
      return runner_status(ReferenceRunnerError::kCudaFailure,
                           "reset_auxiliary_synchronize",
                           kReferenceNoLayer,
                           static_cast<int>(auxiliary_sync_status));
    }
  }
  const RequestOperationStatus reset_status = state_->reset_async(stream_);
  if (!reset_status) {
    poisoned_ = true;
    trace_valid_ = false;
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "request_state_reset", kReferenceNoLayer,
                         reset_status.cuda_error);
  }
  const cudaError_t sync_status = cudaStreamSynchronize(
      reinterpret_cast<cudaStream_t>(stream_));
  if (sync_status != cudaSuccess) {
    poisoned_ = true;
    trace_valid_ = false;
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "reset_synchronize", kReferenceNoLayer,
                         static_cast<int>(sync_status));
  }
  poisoned_ = false;
  trace_valid_ = false;
  retained_prefill_hidden_valid_ = false;
  retained_prefill_position_ = 0U;
  retained_prefill_input_token_ = 0U;
  retained_prefill_hidden_row_ = 0U;
  trace_position_ = 0U;
  trace_input_token_ = 0U;
  return {};
}

ReferenceStepOutcome ReferenceRunner::step(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) noexcept {
  return step_impl(input_token_id, options,
                   DecodeGraphP1Action::kDisabled);
}

ReferenceDecodeGraphP1PrepareOutcome
ReferenceRunner::prepare_fixed_position_decode_graph_p1(
    const std::uint32_t input_token_id) noexcept {
  const std::uint32_t position = current_position();
  ReferenceStepOptions options;
  options.compute_logits = true;
  options.capture_trace = false;
  options.measure_timing = false;
  options.logits_mode = ReferenceLogitsMode::kPredictedTokenOnly;
  ReferenceStepOutcome captured =
      step_impl(input_token_id, options,
                DecodeGraphP1Action::kCaptureOnly);
  ReferenceDecodeGraphP1PrepareOutcome outcome;
  outcome.status = captured.status;
  const std::optional<ReferenceDecodeGraphP1Stats> stats =
      fixed_position_decode_graph_p1_stats(position);
  if (captured && stats.has_value()) {
    outcome.value.emplace(*stats);
  }
  return outcome;
}

ReferenceDecodeGraphCachePrepareOutcome
ReferenceRunner::prepare_fixed_position_decode_graph_cache(
    const std::uint32_t first_position,
    const std::uint32_t last_position,
    const std::uint32_t input_token_id) noexcept {
  ReferenceDecodeGraphCachePrepareOutcome outcome;
  if (!static_cast<bool>(*this)) {
    outcome.status = runner_status(ReferenceRunnerError::kInvalidRunner,
                                   "decode_graph_cache_prepare");
    return outcome;
  }
  if (poisoned_) {
    outcome.status = runner_status(ReferenceRunnerError::kPoisoned,
                                   "decode_graph_cache_prepare");
    return outcome;
  }
  if (first_position > last_position ||
      last_position >= decode_graph_p1_slots_.size()) {
    outcome.status = runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "decode_graph_cache_range");
    return outcome;
  }
  if (last_position >= state_->max_sequence_length()) {
    outcome.status = runner_status(
        ReferenceRunnerError::kCapacityExceeded,
        "decode_graph_cache_capacity");
    return outcome;
  }
  if (input_token_id >= kReferenceVocabularySize) {
    outcome.status = runner_status(ReferenceRunnerError::kTokenOutOfRange,
                                   "decode_graph_cache_input_token");
    return outcome;
  }
  if (projection_backend_ != ProjectionBackend::kSm87WeightOnly ||
      linear_weight_kind(weights_->lm_head()) == LinearWeightKind::kBf16) {
    outcome.status = runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "decode_graph_cache_contract");
    return outcome;
  }
  for (std::uint32_t position = first_position;
       position <= last_position; ++position) {
    if (has_fixed_position_decode_graph_p1(position)) {
      outcome.status = runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_cache_range_not_empty");
      return outcome;
    }
  }

  const std::uint32_t entry_position = state_->current_position();
  std::array<DecodeGraphP1Slot, kReferenceDecodeGraphP2MaximumSlots>
      staged_slots{};
  ReferenceDecodeGraphCachePrepareResult prepared;
  ReferenceStepOptions options;
  options.compute_logits = true;
  options.capture_trace = false;
  options.measure_timing = false;
  options.logits_mode = ReferenceLogitsMode::kPredictedTokenOnly;

  const auto destroy_staged = [&staged_slots]() noexcept {
    int first_error = static_cast<int>(cudaSuccess);
    for (DecodeGraphP1Slot& slot : staged_slots) {
      const int status = ReferenceRunner::destroy_decode_graph_p1_slot(slot);
      if (status != static_cast<int>(cudaSuccess) &&
          first_error == static_cast<int>(cudaSuccess)) {
        first_error = status;
      }
    }
    return first_error;
  };
  const auto restore_entry_position = [this, entry_position]() noexcept {
    return state_->set_sequence_length(entry_position);
  };

  for (std::uint32_t position = first_position;
       position <= last_position; ++position) {
    const RequestOperationStatus position_status =
        state_->set_sequence_length(position);
    if (!position_status) {
      const RequestOperationStatus restore_status = restore_entry_position();
      const int cleanup_status = destroy_staged();
      poisoned_ = true;
      trace_valid_ = false;
      if (!restore_status) {
        outcome.status = runner_status(
            ReferenceRunnerError::kStateCommitFailure,
            "decode_graph_cache_restore_position", kReferenceNoLayer,
            restore_status.cuda_error);
      } else if (cleanup_status != static_cast<int>(cudaSuccess)) {
        outcome.status = runner_status(
            ReferenceRunnerError::kCudaFailure,
            "decode_graph_cache_rollback_destroy", kReferenceNoLayer,
            cleanup_status);
      } else {
        outcome.status = runner_status(
            ReferenceRunnerError::kStateCommitFailure,
            "decode_graph_cache_set_position", kReferenceNoLayer,
            position_status.cuda_error);
      }
      return outcome;
    }
    ReferenceStepOutcome captured = step_impl(
        input_token_id, options, DecodeGraphP1Action::kCaptureOnly,
        &staged_slots[position]);
    if (!captured) {
      const RequestOperationStatus restore_status = restore_entry_position();
      const int cleanup_status = destroy_staged();
      if (!restore_status) {
        poisoned_ = true;
        trace_valid_ = false;
        outcome.status = runner_status(
            ReferenceRunnerError::kStateCommitFailure,
            "decode_graph_cache_restore_position", kReferenceNoLayer,
            restore_status.cuda_error);
      } else if (cleanup_status != static_cast<int>(cudaSuccess)) {
        poisoned_ = true;
        trace_valid_ = false;
        outcome.status = runner_status(
            ReferenceRunnerError::kCudaFailure,
            "decode_graph_cache_rollback_destroy", kReferenceNoLayer,
            cleanup_status);
      } else {
        outcome.status = captured.status;
      }
      return outcome;
    }
    prepared.graphs[prepared.graph_count++] =
        staged_slots[position].stats;
    prepared.prepared_mask |= std::uint64_t{1U} << position;
  }

  const RequestOperationStatus restore_status = restore_entry_position();
  if (!restore_status) {
    const int cleanup_status = destroy_staged();
    poisoned_ = true;
    trace_valid_ = false;
    outcome.status = runner_status(
        cleanup_status == static_cast<int>(cudaSuccess)
            ? ReferenceRunnerError::kStateCommitFailure
            : ReferenceRunnerError::kCudaFailure,
        cleanup_status == static_cast<int>(cudaSuccess)
            ? "decode_graph_cache_restore_position"
            : "decode_graph_cache_rollback_destroy",
        kReferenceNoLayer,
        cleanup_status == static_cast<int>(cudaSuccess)
            ? restore_status.cuda_error
            : cleanup_status);
    return outcome;
  }

  for (std::uint32_t position = first_position;
       position <= last_position; ++position) {
    decode_graph_p1_slots_[position] = staged_slots[position];
    staged_slots[position] = {};
  }
  outcome.value.emplace(std::move(prepared));
  return outcome;
}

std::uint64_t
ReferenceRunner::fixed_position_decode_graph_cache_mask() const noexcept {
  std::uint64_t mask = 0U;
  for (std::uint32_t position = 0U;
       position < decode_graph_p1_slots_.size(); ++position) {
    if (has_fixed_position_decode_graph_p1(position)) {
      mask |= std::uint64_t{1U} << position;
    }
  }
  return mask;
}

ReferenceRunnerStatus
ReferenceRunner::clear_fixed_position_decode_graph_cache() noexcept {
  if (!static_cast<bool>(*this)) {
    return runner_status(ReferenceRunnerError::kInvalidRunner,
                         "decode_graph_cache_clear");
  }
  const cudaError_t main_sync_status =
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  if (main_sync_status != cudaSuccess) {
    poisoned_ = true;
    trace_valid_ = false;
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "decode_graph_cache_clear_main_synchronize",
                         kReferenceNoLayer,
                         static_cast<int>(main_sync_status));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    const cudaError_t auxiliary_sync_status = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
    if (auxiliary_sync_status != cudaSuccess) {
      poisoned_ = true;
      trace_valid_ = false;
      return runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_cache_clear_auxiliary_synchronize",
          kReferenceNoLayer, static_cast<int>(auxiliary_sync_status));
    }
  }

  std::array<DecodeGraphP1Slot, kReferenceDecodeGraphP2MaximumSlots>
      detached_slots = decode_graph_p1_slots_;
  decode_graph_p1_slots_ = {};
  int first_error = static_cast<int>(cudaSuccess);
  for (DecodeGraphP1Slot& slot : detached_slots) {
    const int status = destroy_decode_graph_p1_slot(slot);
    if (status != static_cast<int>(cudaSuccess) &&
        first_error == static_cast<int>(cudaSuccess)) {
      first_error = status;
    }
  }
  if (first_error != static_cast<int>(cudaSuccess)) {
    poisoned_ = true;
    trace_valid_ = false;
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "decode_graph_cache_clear_destroy",
                         kReferenceNoLayer, first_error);
  }
  return {};
}

bool ReferenceRunner::has_fixed_position_decode_graph_p1(
    const std::uint32_t position) const noexcept {
  if (position >= decode_graph_p1_slots_.size()) {
    return false;
  }
  const DecodeGraphP1Slot& slot = decode_graph_p1_slots_[position];
  return slot.graph != nullptr && slot.exec != nullptr &&
         slot.embedding_node != nullptr &&
         slot.embedding_launch.function != nullptr &&
         slot.stats.position == position;
}

std::optional<ReferenceDecodeGraphP1Stats>
ReferenceRunner::fixed_position_decode_graph_p1_stats(
    const std::uint32_t position) const noexcept {
  if (!has_fixed_position_decode_graph_p1(position)) {
    return std::nullopt;
  }
  return decode_graph_p1_slots_[position].stats;
}

ReferenceStepOutcome
ReferenceRunner::replay_fixed_position_decode_graph_p1(
    const std::uint32_t input_token_id,
    const bool measure_timing) noexcept {
  ReferenceStepOptions options;
  options.compute_logits = true;
  options.capture_trace = false;
  options.measure_timing = measure_timing;
  options.logits_mode = ReferenceLogitsMode::kPredictedTokenOnly;
  return step_impl(input_token_id, options,
                   DecodeGraphP1Action::kReplay);
}

ReferenceStepOutcome ReferenceRunner::step_impl(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options,
    const DecodeGraphP1Action graph_action,
    DecodeGraphP1Slot* const capture_destination) noexcept {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }
  // Any ordinary step may overwrite the shared hidden workspace. Retention is
  // a one-call hand-off from a marked prefill tile to its dedicated finalizer.
  retained_prefill_hidden_valid_ = false;
  if (!static_cast<bool>(*this)) {
    return fail_step(
        runner_status(ReferenceRunnerError::kInvalidRunner, "step"));
  }
  if (poisoned_) {
    ReferenceStepOutcome outcome;
    outcome.status =
        runner_status(ReferenceRunnerError::kPoisoned, "step");
    return outcome;
  }
  if (!is_valid_reference_logits_mode(options.logits_mode)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions, "logits_mode"));
  }
  if (input_token_id >= kReferenceVocabularySize) {
    return fail_step(runner_status(ReferenceRunnerError::kTokenOutOfRange,
                                   "input_token"));
  }
  if (state_->remaining_capacity() == 0U) {
    return fail_step(runner_status(ReferenceRunnerError::kCapacityExceeded,
                                   "request_capacity"));
  }
  if (options.capture_trace &&
      (!trace_enabled_ || pinned_trace_ == nullptr)) {
    return fail_step(runner_status(ReferenceRunnerError::kTraceUnavailable,
                                   "trace_not_reserved"));
  }

  const bool use_decode_graph_p1 =
      graph_action != DecodeGraphP1Action::kDisabled;
  if (use_decode_graph_p1 &&
      (!options.compute_logits || options.capture_trace ||
       options.logits_mode != ReferenceLogitsMode::kPredictedTokenOnly ||
       projection_backend_ != ProjectionBackend::kSm87WeightOnly ||
       linear_weight_kind(weights_->lm_head()) == LinearWeightKind::kBf16)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "decode_graph_p1_contract"));
  }
  if (graph_action == DecodeGraphP1Action::kCaptureOnly) {
    int device = -1;
    cudaDeviceProp properties{};
    cudaError_t cuda_status = cudaGetDevice(&device);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaGetDeviceProperties(&properties, device);
    }
    if (cuda_status != cudaSuccess) {
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_device", kReferenceNoLayer,
          static_cast<int>(cuda_status)));
    }
    if (properties.major != 8 || properties.minor != 7) {
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_requires_sm87"));
    }
  }

  const std::uint32_t position = state_->current_position();
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
  DecodeGraphP1Slot* decode_graph_slot = nullptr;
  Clock::time_point decode_graph_prepare_started{};
  if (graph_action == DecodeGraphP1Action::kCaptureOnly) {
    if (position >= decode_graph_p1_slots_.size()) {
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_position_not_cacheable"));
    }
    decode_graph_slot = capture_destination == nullptr
                            ? &decode_graph_p1_slots_[position]
                            : capture_destination;
    decode_graph_prepare_started = Clock::now();
    const cudaError_t begin_status = cudaStreamBeginCapture(
        stream, cudaStreamCaptureModeThreadLocal);
    if (begin_status != cudaSuccess) {
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_begin_capture", kReferenceNoLayer,
          static_cast<int>(begin_status)));
    }
    decode_graph_capture_active_ = true;
  } else if (graph_action == DecodeGraphP1Action::kReplay) {
    if (!has_fixed_position_decode_graph_p1(position)) {
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_not_prepared"));
    }
    decode_graph_slot = &decode_graph_p1_slots_[position];
  }
  ReferenceRunnerStatus launch_failure{};
  const auto check_cuda = [&launch_failure](
                              const int status, const char* const operation,
                              const std::size_t layer) noexcept {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    launch_failure = runner_status(ReferenceRunnerError::kCudaFailure,
                                   operation, layer, status);
    return false;
  };
  const auto project = [this, &check_cuda](
                           const LinearWeight& weight,
                           const std::uint16_t* const input,
                           std::uint16_t* const output,
                           const char* const operation,
                           const std::size_t layer) noexcept {
    return check_cuda(launch_projection_to_bf16_cuda(
                          projection_backend_, weight, input,
                          views_.fp32_scratch,
                          views_.fp32_scratch_elements, output, stream_),
                      operation, layer);
  };
  const auto copy_trace = [this, stream, &check_cuda](
                              const std::uint16_t* const source,
                              const std::size_t offset,
                              const char* const operation,
                              const std::size_t layer) noexcept {
    if (source == nullptr || offset >
                                 kReferenceTraceElements -
                                     kReferenceHiddenSize) {
      return check_cuda(static_cast<int>(cudaErrorInvalidValue), operation,
                        layer);
    }
    return check_cuda(
        static_cast<int>(cudaMemcpyAsync(
            pinned_trace_ + offset, source,
            kReferenceHiddenSize * sizeof(std::uint16_t),
            cudaMemcpyDeviceToHost, stream)),
        operation, layer);
  };
  const bool prediction_only =
      options.compute_logits &&
      options.logits_mode == ReferenceLogitsMode::kPredictedTokenOnly;
  const bool use_sm87_bf16_logits =
      options.compute_logits &&
      projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
      linear_weight_kind(weights_->lm_head()) != LinearWeightKind::kBf16;

  if (graph_action != DecodeGraphP1Action::kReplay) {
  if (!check_cuda(launch_embedding_gather_reference_cuda(
                      weights_->embed_tokens().weight,
                      kReferenceVocabularySize, kReferenceHiddenSize,
                      input_token_id, views_.hidden[0], stream_),
                  "embedding_gather", kReferenceNoLayer)) {
    return fail_step(launch_failure);
  }
  if (options.capture_trace &&
      !copy_trace(views_.hidden[0], 0U, "trace_embedding",
                  kReferenceNoLayer)) {
    return fail_step(launch_failure);
  }
  if (!check_cuda(launch_centered_rms_norm_reference_cuda(
                      views_.hidden[0],
                      weights_->layer(0U).input_layernorm.data,
                      kReferenceHiddenSize, kRmsEpsilon, views_.hidden[1],
                      stream_),
                  "input_layernorm", 0U)) {
    return fail_step(launch_failure);
  }

  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& layer_weights = weights_->layer(layer);
    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      const auto* const attention =
          std::get_if<LinearAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_step(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "linear_attention_variant", layer));
      }
      const int composite_status =
          launch_linear_attention_qkv_z_ab_to_bf16_cuda(
              projection_backend_, attention->in_proj_qkv,
              attention->in_proj_z, attention->in_proj_a,
              attention->in_proj_b, views_.hidden[1], views_.projection[0],
              views_.projection[1], views_.linear_a, views_.linear_b,
              stream_);
      const bool used_projection_composite =
          composite_status == static_cast<int>(cudaSuccess);
      if (!used_projection_composite &&
          composite_status != static_cast<int>(cudaErrorNotSupported)) {
        if (!check_cuda(composite_status,
                        "linear_qkv_z_a_b_projection", layer)) {
          return fail_step(launch_failure);
        }
      }
      if (!used_projection_composite) {
        if (!check_cuda(launch_projection_pair_tile_to_bf16_cuda(
                            projection_backend_, attention->in_proj_qkv,
                            attention->in_proj_z, views_.hidden[1], 1U,
                            views_.fp32_scratch,
                            views_.fp32_scratch_elements,
                            views_.projection[0], views_.projection[1],
                            stream_),
                        "linear_qkv_z_projection", layer)) {
          return fail_step(launch_failure);
        }
        if (supports_bf16_projection_pair(
                projection_backend_, attention->in_proj_a,
                attention->in_proj_b)) {
          if (!check_cuda(launch_projection_pair_tile_to_bf16_cuda(
                              projection_backend_, attention->in_proj_a,
                              attention->in_proj_b, views_.hidden[1], 1U,
                              views_.fp32_scratch,
                              views_.fp32_scratch_elements, views_.linear_a,
                              views_.linear_b, stream_),
                          "linear_a_b_projection", layer)) {
            return fail_step(launch_failure);
          }
        } else if (!project(attention->in_proj_a, views_.hidden[1],
                            views_.linear_a, "linear_a_projection", layer) ||
                   !project(attention->in_proj_b, views_.hidden[1],
                            views_.linear_b, "linear_b_projection", layer)) {
          return fail_step(launch_failure);
        }
      }
      if (!check_cuda(launch_causal_conv1d_silu_update_reference_cuda(
                          views_.projection[0], attention->conv1d.data,
                          views_.conv_state[layer], views_.projection[0], {},
                          stream_),
                      "linear_causal_conv", layer)) {
        return fail_step(launch_failure);
      }
      const bool use_gdn_norm_gate_composite =
          supports_gated_delta_net_update_plain_rms_norm_silu_gate_fusion(
              1U, {}, kGdnValueHeadCount, kGdnHeadDimension);
      if (use_gdn_norm_gate_composite) {
        if (!check_cuda(
                launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
                    views_.projection[0], views_.linear_a,
                    views_.linear_b, attention->a_log.data,
                    attention->dt_bias.data, views_.gdn_state[layer],
                    views_.gdn_state[layer], kRmsEpsilon,
                    attention->norm.data, views_.projection[1],
                    kGdnValueHeadCount, kGdnHeadDimension, kRmsEpsilon,
                    views_.projection[2], {}, stream_),
                "linear_gdn_output_norm_gate", layer)) {
          return fail_step(launch_failure);
        }
      } else if (!check_cuda(
                     launch_gated_delta_net_update_warp_parallel_cuda(
                         views_.projection[0], views_.linear_a,
                         views_.linear_b, attention->a_log.data,
                         attention->dt_bias.data, views_.gdn_state[layer],
                         views_.gdn_state[layer], kRmsEpsilon,
                         views_.projection[2], {}, stream_),
                     "linear_gdn", layer) ||
                 !check_cuda(
                     launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                         views_.projection[2], attention->norm.data,
                         views_.projection[1], kGdnValueHeadCount,
                         kGdnHeadDimension, kRmsEpsilon,
                         views_.projection[2], stream_),
                     "linear_output_norm_gate", layer)) {
        return fail_step(launch_failure);
      }
      if (!project(attention->out_proj, views_.projection[2],
                   views_.hidden[1], "linear_output_projection", layer)) {
        return fail_step(launch_failure);
      }
    } else if (expected == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_step(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "full_attention_variant", layer));
      }
      std::uint16_t* const current_key =
          views_.key_cache[layer] +
          static_cast<std::size_t>(position) * kFullKvElements;
      std::uint16_t* const current_value =
          views_.value_cache[layer] +
          static_cast<std::size_t>(position) * kFullKvElements;
      std::uint16_t* const packed_gates =
          views_.projection[3] + kFullQueryElements;
      std::uint16_t* full_query = views_.projection[0];
      const std::size_t rope_first_position =
          static_cast<std::size_t>(position);
      if (!check_cuda(launch_full_attention_q_kv_to_bf16_cuda(
                          projection_backend_, attention->q_proj,
                          attention->k_proj, attention->v_proj,
                          views_.hidden[1], views_.fp32_scratch,
                          views_.fp32_scratch_elements, views_.projection[0],
                          current_key, current_value, stream_),
                      "full_q_k_v_projection", layer)) {
        return fail_step(launch_failure);
      }

      if (reference_runner_detail::use_qk_rope_tile(rope_first_position,
                                                     1U)) {
        full_query = views_.projection[3];
        if (!check_cuda(launch_full_attention_preprocess_24_4_256_64_cuda(
                            views_.projection[0], current_key,
                            attention->q_norm.data, attention->k_norm.data,
                            kRmsEpsilon, full_query, packed_gates,
                            views_.rope_cos, views_.rope_sin,
                            rope_first_position, 1U, stream_),
                        "full_preprocess", layer)) {
          return fail_step(launch_failure);
        }
      } else {
        if (!check_cuda(launch_split_interleaved_q_gate_reference_cuda(
                            views_.projection[0], kFullQueryHeads,
                            kFullHeadDimension, views_.projection[3],
                            packed_gates, stream_),
                        "full_split_q_gate_fallback", layer) ||
            !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                            views_.projection[3], attention->q_norm.data,
                            kFullQueryHeads, kFullHeadDimension, kRmsEpsilon,
                            full_query, stream_),
                        "full_q_norm_fallback", layer) ||
            !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                            current_key, attention->k_norm.data,
                            kFullKvHeads, kFullHeadDimension, kRmsEpsilon,
                            current_key, stream_),
                        "full_k_norm_fallback", layer)) {
          return fail_step(launch_failure);
        }
        const float* const cosines =
            views_.rope_cos + rope_first_position * kRopePairs;
        const float* const sines =
            views_.rope_sin + rope_first_position * kRopePairs;
        if (!check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                            full_query, cosines, sines, kFullQueryHeads,
                            full_query, stream_),
                        "full_q_rope_fallback", layer) ||
            !check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                            current_key, cosines, sines, kFullKvHeads,
                            current_key, stream_),
                        "full_k_rope_fallback", layer)) {
          return fail_step(launch_failure);
        }
      }

      const std::size_t sequence_length =
          static_cast<std::size_t>(position) + 1U;
      if (reference_runner_detail::use_fused_gqa_sigmoid_gate_tile(
              position, 1U)) {
        if (!check_cuda(
                launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                    full_query, views_.key_cache[layer],
                    views_.value_cache[layer], sequence_length,
                    kAttentionScale, views_.fp32_scratch,
                    views_.fp32_scratch_elements, packed_gates,
                    views_.projection[1], stream_),
                "full_gqa_output_gate", layer)) {
          return fail_step(launch_failure);
        }
      } else if (!check_cuda(launch_gqa_attention_reference_cuda(
                                 full_query, views_.key_cache[layer],
                                 views_.value_cache[layer], kFullQueryHeads,
                                 kFullKvHeads, sequence_length,
                                 kFullHeadDimension, kAttentionScale,
                                 views_.fp32_scratch,
                                 views_.fp32_scratch_elements,
                                 views_.projection[1], stream_),
                             "full_gqa", layer) ||
                 !check_cuda(launch_sigmoid_gate_reference_cuda(
                                 views_.projection[1], packed_gates,
                                 kFullQueryElements, views_.projection[1],
                                 stream_),
                             "full_output_gate", layer)) {
        return fail_step(launch_failure);
      }
      if (!project(attention->o_proj, views_.projection[1], views_.hidden[1],
                   "full_output_projection", layer)) {
        return fail_step(launch_failure);
      }
    } else {
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidLayerSchedule, "layer_schedule",
          layer));
    }

    if (!check_cuda(
            launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                projection_backend_, layer_weights.mlp.gate_proj,
                layer_weights.mlp.up_proj, views_.hidden[0],
                views_.hidden[1],
                layer_weights.post_attention_layernorm.data, kRmsEpsilon,
                views_.fp32_scratch, views_.fp32_scratch_elements,
                views_.hidden[2], views_.projection[0], views_.projection[1],
                stream_),
            "attention_residual_norm_mlp_gate_up_silu", layer)) {
      return fail_step(launch_failure);
    }
    const std::size_t trace_base =
        kReferenceHiddenSize + 2U * layer * kReferenceHiddenSize;
    if (options.capture_trace &&
        !copy_trace(views_.hidden[2], trace_base + kReferenceHiddenSize,
                    "trace_layer_residual", layer)) {
      return fail_step(launch_failure);
    }

    const bool is_final_layer =
        layer + 1U == kReferenceDecoderLayerCount;
    const std::uint16_t* const next_norm_weight =
        is_final_layer
            ? weights_->final_norm().data
            : weights_->layer(layer + 1U).input_layernorm.data;
    const char* const residual_norm_operation =
        is_final_layer ? "mlp_down_residual_final_norm"
                       : "mlp_down_residual_input_layernorm";
    if (!check_cuda(launch_mlp_down_residual_norm_to_bf16_cuda(
                        projection_backend_, layer_weights.mlp.down_proj,
                        views_.projection[0], views_.hidden[2],
                        next_norm_weight, kRmsEpsilon, views_.fp32_scratch,
                        views_.fp32_scratch_elements, views_.projection[1],
                        views_.hidden[0], views_.hidden[1], stream_),
                    residual_norm_operation, layer)) {
      return fail_step(launch_failure);
    }
    if (options.capture_trace &&
        !copy_trace(views_.projection[1], trace_base, "trace_layer_hidden",
                    layer)) {
      return fail_step(launch_failure);
    }
  }

  if (options.capture_trace &&
      !copy_trace(views_.hidden[1],
                  (1U + 2U * kReferenceDecoderLayerCount) *
                      kReferenceHiddenSize,
                  "trace_final_norm", kReferenceNoLayer)) {
    return fail_step(launch_failure);
  }

  if (options.compute_logits) {
    if (use_sm87_bf16_logits) {
      auto* const device_bf16_logits =
          reinterpret_cast<std::uint16_t*>(views_.fp32_scratch);
      if (!check_cuda(launch_projection_to_bf16_cuda(
                          projection_backend_, weights_->lm_head(),
                          views_.hidden[1], nullptr, 0U,
                          device_bf16_logits, stream_),
                      "lm_head_sm87_bf16", kReferenceNoLayer)) {
        return fail_step(launch_failure);
      }
      if (prediction_only) {
        constexpr std::size_t kGreedyWorkspaceBytes =
            kReferenceVocabularySize * sizeof(std::uint16_t) +
            kBf16GreedyArgmaxWorkspaceResults *
                sizeof(Bf16GreedyArgmaxResult);
        static_assert((kReferenceVocabularySize * sizeof(std::uint16_t)) %
                              alignof(Bf16GreedyArgmaxResult) ==
                          0U);
        if (views_.fp32_scratch_elements <
            (kGreedyWorkspaceBytes + sizeof(float) - 1U) / sizeof(float)) {
          return fail_step(runner_status(
              ReferenceRunnerError::kInvalidRequestState,
              "bf16_greedy_argmax_workspace"));
        }
        auto* const greedy_workspace =
            reinterpret_cast<Bf16GreedyArgmaxResult*>(
                device_bf16_logits + kReferenceVocabularySize);
        if (!check_cuda(launch_bf16_greedy_argmax_cuda(
                            device_bf16_logits, kReferenceVocabularySize,
                            greedy_workspace, stream_),
                        "bf16_greedy_argmax", kReferenceNoLayer) ||
            !check_cuda(
                static_cast<int>(cudaMemcpyAsync(
                    pinned_logits_, greedy_workspace,
                    sizeof(Bf16GreedyArgmaxResult),
                    cudaMemcpyDeviceToHost, stream)),
                "logits_prediction_d2h", kReferenceNoLayer)) {
          return fail_step(launch_failure);
        }
      } else if (!check_cuda(
                     static_cast<int>(cudaMemcpyAsync(
                         pinned_logits_, device_bf16_logits,
                         kReferenceVocabularySize * sizeof(std::uint16_t),
                         cudaMemcpyDeviceToHost, stream)),
                     "logits_bf16_d2h", kReferenceNoLayer)) {
        return fail_step(launch_failure);
      }
    } else if (!check_cuda(launch_projection_reference_cuda(
                               weights_->lm_head(), views_.hidden[1],
                               views_.fp32_scratch, stream_),
                           "lm_head", kReferenceNoLayer) ||
               !check_cuda(
                   static_cast<int>(cudaMemcpyAsync(
                       pinned_logits_, views_.fp32_scratch,
                       kReferenceVocabularySize * sizeof(float),
                       cudaMemcpyDeviceToHost, stream)),
                   "logits_d2h", kReferenceNoLayer)) {
      return fail_step(launch_failure);
    }
  }
  }

  if (graph_action == DecodeGraphP1Action::kCaptureOnly) {
    cudaGraph_t captured_graph = nullptr;
    const cudaError_t end_status =
        cudaStreamEndCapture(stream, &captured_graph);
    decode_graph_capture_active_ = false;
    if (end_status != cudaSuccess || captured_graph == nullptr) {
      if (captured_graph != nullptr) {
        (void)cudaGraphDestroy(captured_graph);
      }
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_end_capture", kReferenceNoLayer,
          static_cast<int>(end_status)));
    }
    const Clock::time_point capture_finished = Clock::now();

    constexpr std::size_t kMaximumDecodeGraphP1Nodes = 1'024U;
    std::size_t node_count = 0U;
    cudaError_t graph_status =
        cudaGraphGetNodes(captured_graph, nullptr, &node_count);
    if (graph_status != cudaSuccess || node_count == 0U ||
        node_count > kMaximumDecodeGraphP1Nodes) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          graph_status == cudaSuccess
              ? ReferenceRunnerError::kInvalidStepOptions
              : ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_node_capacity", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    std::array<cudaGraphNode_t, kMaximumDecodeGraphP1Nodes> nodes{};
    std::size_t returned_node_count = node_count;
    graph_status = cudaGraphGetNodes(captured_graph, nodes.data(),
                                     &returned_node_count);
    if (graph_status != cudaSuccess || returned_node_count != node_count) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_get_nodes", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }

    ReferenceDecodeGraphP1Stats stats;
    stats.position = position;
    stats.input_token_id = input_token_id;
    stats.node_count = node_count;
    stats.capture_enqueue_milliseconds =
        std::chrono::duration<double, std::milli>(
            capture_finished - decode_graph_prepare_started)
            .count();
    for (std::size_t index = 0U; index < node_count; ++index) {
      cudaGraphNodeType type{};
      graph_status = cudaGraphNodeGetType(nodes[index], &type);
      if (graph_status != cudaSuccess) {
        break;
      }
      if (type == cudaGraphNodeTypeKernel) {
        ++stats.kernel_node_count;
      } else if (type == cudaGraphNodeTypeMemcpy) {
        ++stats.memcpy_node_count;
      } else {
        ++stats.other_node_count;
      }
    }
    if (graph_status != cudaSuccess) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_classify_nodes", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }

    std::size_t root_count = 0U;
    graph_status =
        cudaGraphGetRootNodes(captured_graph, nullptr, &root_count);
    if (graph_status != cudaSuccess || root_count != 1U) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          graph_status == cudaSuccess
              ? ReferenceRunnerError::kInvalidStepOptions
              : ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_root_count", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    cudaGraphNode_t embedding_node = nullptr;
    std::size_t returned_root_count = 1U;
    graph_status = cudaGraphGetRootNodes(
        captured_graph, &embedding_node, &returned_root_count);
    cudaGraphNodeType root_type{};
    if (graph_status == cudaSuccess && returned_root_count == 1U &&
        embedding_node != nullptr) {
      graph_status = cudaGraphNodeGetType(embedding_node, &root_type);
    }
    if (graph_status != cudaSuccess || returned_root_count != 1U ||
        embedding_node == nullptr || root_type != cudaGraphNodeTypeKernel) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          graph_status == cudaSuccess
              ? ReferenceRunnerError::kInvalidStepOptions
              : ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_embedding_root", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }

    cudaKernelNodeParams embedding_launch{};
    graph_status =
        cudaGraphKernelNodeGetParams(embedding_node, &embedding_launch);
    if (graph_status != cudaSuccess) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_embedding_params", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    const bool valid_embedding_shape =
        embedding_launch.func != nullptr &&
        embedding_launch.gridDim.x == 20U &&
        embedding_launch.gridDim.y == 1U &&
        embedding_launch.gridDim.z == 1U &&
        embedding_launch.blockDim.x == 256U &&
        embedding_launch.blockDim.y == 1U &&
        embedding_launch.blockDim.z == 1U &&
        embedding_launch.sharedMemBytes == 0U &&
        embedding_launch.kernelParams != nullptr &&
        embedding_launch.extra == nullptr &&
        embedding_launch.kernelParams[0] != nullptr &&
        embedding_launch.kernelParams[1] != nullptr &&
        embedding_launch.kernelParams[2] != nullptr &&
        embedding_launch.kernelParams[3] != nullptr;
    if (!valid_embedding_shape) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_embedding_shape"));
    }
    const auto captured_table =
        *static_cast<const std::uint16_t* const*>(
            embedding_launch.kernelParams[0]);
    const std::size_t captured_offset =
        *static_cast<const std::size_t*>(
            embedding_launch.kernelParams[1]);
    const std::size_t captured_hidden_size =
        *static_cast<const std::size_t*>(
            embedding_launch.kernelParams[2]);
    const auto captured_output =
        *static_cast<std::uint16_t* const*>(
            embedding_launch.kernelParams[3]);
    if (captured_table != weights_->embed_tokens().weight ||
        captured_offset !=
            static_cast<std::size_t>(input_token_id) *
                kReferenceHiddenSize ||
        captured_hidden_size != kReferenceHiddenSize ||
        captured_output != views_.hidden[0]) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_embedding_arguments"));
    }
    const Clock::time_point topology_finished = Clock::now();
    stats.topology_inspection_milliseconds =
        std::chrono::duration<double, std::milli>(
            topology_finished - capture_finished)
            .count();

    cudaGraphExec_t captured_exec = nullptr;
    const Clock::time_point instantiate_started = Clock::now();
    graph_status = cudaGraphInstantiate(
        &captured_exec, captured_graph, nullptr, nullptr, 0U);
    if (graph_status != cudaSuccess || captured_exec == nullptr) {
      if (captured_exec != nullptr) {
        (void)cudaGraphExecDestroy(captured_exec);
      }
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_instantiate", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    const Clock::time_point instantiate_finished = Clock::now();
    stats.instantiate_milliseconds =
        std::chrono::duration<double, std::milli>(
            instantiate_finished - instantiate_started)
            .count();

    const Clock::time_point upload_ready_started = Clock::now();
    graph_status = cudaGraphUpload(captured_exec, stream);
    if (graph_status == cudaSuccess) {
      graph_status = cudaStreamSynchronize(stream);
    }
    if (graph_status != cudaSuccess || decode_graph_slot == nullptr) {
      (void)cudaGraphExecDestroy(captured_exec);
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          graph_status == cudaSuccess
              ? ReferenceRunnerError::kInvalidRunner
              : ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_upload", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    const Clock::time_point upload_ready_finished = Clock::now();
    stats.upload_ready_milliseconds =
        std::chrono::duration<double, std::milli>(
            upload_ready_finished - upload_ready_started)
            .count();

    DecodeGraphP1Slot prepared_slot;
    prepared_slot.graph = captured_graph;
    prepared_slot.exec = captured_exec;
    prepared_slot.embedding_node = embedding_node;
    prepared_slot.embedding_launch.function = embedding_launch.func;
    prepared_slot.embedding_launch.grid = {
        embedding_launch.gridDim.x, embedding_launch.gridDim.y,
        embedding_launch.gridDim.z};
    prepared_slot.embedding_launch.block = {
        embedding_launch.blockDim.x, embedding_launch.blockDim.y,
        embedding_launch.blockDim.z};
    prepared_slot.embedding_launch.shared_memory_bytes =
        embedding_launch.sharedMemBytes;
    stats.total_prepare_milliseconds =
        std::chrono::duration<double, std::milli>(
            Clock::now() - decode_graph_prepare_started)
            .count();
    prepared_slot.stats = stats;
    // Publish only after the replacement is completely uploaded and ready.
    // A failed recapture retains the prior slot handles; the runner's normal
    // failure contract still requires reset before any later reuse.
    (void)destroy_decode_graph_p1_slot(*decode_graph_slot);
    *decode_graph_slot = prepared_slot;
    ReferenceStepOutcome outcome;
    ReferenceStepResult result;
    result.position = position;
    result.input_token_id = input_token_id;
    outcome.value.emplace(std::move(result));
    return outcome;
  }

  if (graph_action == DecodeGraphP1Action::kReplay) {
    cudaKernelNodeParams embedding_params{};
    embedding_params.func = decode_graph_slot->embedding_launch.function;
    embedding_params.gridDim =
        dim3(decode_graph_slot->embedding_launch.grid[0],
             decode_graph_slot->embedding_launch.grid[1],
             decode_graph_slot->embedding_launch.grid[2]);
    embedding_params.blockDim =
        dim3(decode_graph_slot->embedding_launch.block[0],
             decode_graph_slot->embedding_launch.block[1],
             decode_graph_slot->embedding_launch.block[2]);
    embedding_params.sharedMemBytes =
        decode_graph_slot->embedding_launch.shared_memory_bytes;
    const std::uint16_t* embedding_table =
        weights_->embed_tokens().weight;
    std::size_t embedding_offset =
        static_cast<std::size_t>(input_token_id) * kReferenceHiddenSize;
    std::size_t embedding_hidden_size = kReferenceHiddenSize;
    std::uint16_t* embedding_output = views_.hidden[0];
    void* embedding_arguments[] = {
        &embedding_table, &embedding_offset, &embedding_hidden_size,
        &embedding_output};
    embedding_params.kernelParams = embedding_arguments;
    embedding_params.extra = nullptr;
    cudaError_t graph_status = cudaGraphExecKernelNodeSetParams(
        reinterpret_cast<cudaGraphExec_t>(decode_graph_slot->exec),
        reinterpret_cast<cudaGraphNode_t>(decode_graph_slot->embedding_node),
        &embedding_params);
    if (graph_status == cudaSuccess) {
      graph_status = cudaGraphLaunch(
          reinterpret_cast<cudaGraphExec_t>(decode_graph_slot->exec), stream);
    }
    if (graph_status != cudaSuccess) {
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_update_or_launch", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
  }

  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return fail_step(runner_status(ReferenceRunnerError::kCudaFailure,
                                   "step_synchronize", kReferenceNoLayer,
                                   static_cast<int>(sync_status)));
  }

  ReferenceStepResult result;
  result.position = position;
  result.input_token_id = input_token_id;
  if (options.compute_logits) {
    if (prediction_only && use_sm87_bf16_logits) {
      const auto& greedy =
          *static_cast<const Bf16GreedyArgmaxResult*>(pinned_logits_);
      if (greedy.has_nonfinite != 0U) {
        return fail_step(runner_status(
            ReferenceRunnerError::kNonFiniteLogits,
            "bf16_greedy_argmax"));
      }
      if (greedy.index >= kReferenceVocabularySize) {
        return fail_step(runner_status(
            ReferenceRunnerError::kCudaFailure,
            "bf16_greedy_argmax_result"));
      }
      result.prediction.emplace(
          ReferenceStepPrediction{greedy.index});
    } else {
      const reference_runner_detail::LogitsAnalysis analysis =
          use_sm87_bf16_logits
              ? reference_runner_detail::analyze_bf16_logits_bits(
                    static_cast<const std::uint16_t*>(pinned_logits_),
                    kReferenceVocabularySize)
              : (prediction_only
                     ? reference_runner_detail::analyze_bf16_argmax_in_place(
                           static_cast<float*>(pinned_logits_),
                           kReferenceVocabularySize)
                     : reference_runner_detail::
                           analyze_bf16_logits_in_place(
                               static_cast<float*>(pinned_logits_),
                               kReferenceVocabularySize));
      if (!analysis.ok()) {
        return fail_step(runner_status(
            ReferenceRunnerError::kNonFiniteLogits,
            "bf16_logits_analysis"));
      }
      if (prediction_only) {
        result.prediction.emplace(ReferenceStepPrediction{
            static_cast<std::uint32_t>(analysis.predicted_index)});
      } else {
        ReferenceStepLogits logits;
        logits.predicted_token_id =
            static_cast<std::uint32_t>(analysis.predicted_index);
        logits.chosen_logit = analysis.maximum;
        logits.max_log_probability = analysis.max_log_probability;
        logits.logsumexp = analysis.logsumexp;
        result.logits.emplace(logits);
      }
    }
  }

  const RequestOperationStatus commit_status = state_->commit_token();
  if (!commit_status) {
    return fail_step(runner_status(
        ReferenceRunnerError::kStateCommitFailure, "commit_token",
        kReferenceNoLayer, commit_status.cuda_error));
  }
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
  // step_synchronize above makes every recurrent-state write visible before
  // this admission-only observer runs. The callback has no return channel so
  // test instrumentation can never change the runner's error contract.
  const auto snapshot_hook =
      g_prefill_gdn_c16_norm_gate_admission_snapshot_hook;
  if (snapshot_hook.callback != nullptr) {
    snapshot_hook.callback(
        *state_, reference_runner_detail::
                     PrefillGdnC16NormGateAdmissionSnapshotStage::kStep,
        snapshot_hook.context);
  }
#endif
  if (options.capture_trace) {
    trace_valid_ = true;
    trace_position_ = position;
    trace_input_token_ = input_token_id;
  }
  if (options.measure_timing) {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - started;
    result.timing.emplace(
        ReferenceStepTiming{elapsed.count()});
  }

  ReferenceStepOutcome outcome;
  outcome.value.emplace(std::move(result));
  return outcome;
}

ReferencePrefillTileOutcome ReferenceRunner::prefill_prefix_tile(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const ReferencePrefillTileOptions& options) noexcept {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }
  // A new tile overwrites the prior layer-major workspace even if validation
  // later rejects the call, so stale retained rows must fail closed.
  retained_prefill_hidden_valid_ = false;
  if (!static_cast<bool>(*this)) {
    return fail_prefill_tile(
        runner_status(ReferenceRunnerError::kInvalidRunner,
                      "prefill_prefix_tile"));
  }
  if (poisoned_) {
    ReferencePrefillTileOutcome outcome;
    outcome.status = runner_status(ReferenceRunnerError::kPoisoned,
                                   "prefill_prefix_tile");
    return outcome;
  }
  if (input_token_ids == nullptr || token_count == 0U ||
      token_count > kMaximumRequestPrefillChunkSize) {
    return fail_prefill_tile(
        runner_status(ReferenceRunnerError::kTokenOutOfRange,
                      "prefill_tile_tokens"));
  }
  for (std::size_t token = 0U; token < token_count; ++token) {
    if (input_token_ids[token] >= kReferenceVocabularySize) {
      return fail_prefill_tile(
          runner_status(ReferenceRunnerError::kTokenOutOfRange,
                        "prefill_tile_token"));
    }
  }
  if (token_count > state_->remaining_capacity() ||
      token_count > state_->plan().prefill_chunk_size) {
    return fail_prefill_tile(
        runner_status(ReferenceRunnerError::kCapacityExceeded,
                      "prefill_tile_capacity"));
  }

  if (token_count == 1U) {
    ReferenceStepOptions step_options;
    step_options.compute_logits = false;
    step_options.capture_trace = false;
    step_options.measure_timing = options.measure_timing;
    ReferenceStepOutcome step_outcome = step(input_token_ids[0], step_options);
    if (!step_outcome) {
      ReferencePrefillTileOutcome outcome;
      outcome.status = step_outcome.status;
      return outcome;
    }
    ReferencePrefillTileResult tile;
    tile.step_count = 1U;
    tile.steps[0] = std::move(*step_outcome.value);
    tile.timing = tile.steps[0].timing;
    if (options.retain_last_hidden_for_logits) {
      retained_prefill_hidden_valid_ = true;
      retained_prefill_position_ = tile.steps[0].position;
      retained_prefill_input_token_ = tile.steps[0].input_token_id;
      retained_prefill_hidden_row_ = 0U;
    }
    ReferencePrefillTileOutcome outcome;
    outcome.value.emplace(std::move(tile));
    return outcome;
  }

  const std::uint32_t first_position = state_->current_position();
#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
  // Snapshot the admission switch once at the tile boundary. Tests may only
  // change it between synchronous public runner calls.
  const bool enable_gdn_b8_admission =
      g_enable_prefill_gdn_b8_admission;
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  const bool enable_gdn_chunk64_native_admission =
      g_enable_prefill_gdn_chunk64_native_admission;
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  const bool enable_gdn_chunk64_reference_admission =
      g_enable_prefill_gdn_chunk64_reference_admission;
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
  // Snapshot the private switch at the synchronous public-call boundary.
  const bool enable_gdn_c16_norm_gate_admission =
      g_enable_prefill_gdn_c16_norm_gate_admission;
#else
  constexpr bool enable_gdn_c16_norm_gate_admission = true;
#endif
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
  ReferenceRunnerStatus launch_failure{};
  const auto check_cuda = [&launch_failure](
                              const int status,
                              const char* const operation,
                              const std::size_t layer) noexcept {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    launch_failure = runner_status(ReferenceRunnerError::kCudaFailure,
                                   operation, layer, status);
    return false;
  };
  const auto project_on_stream =
      [this, token_count, &check_cuda](
          const LinearWeight& weight, const std::uint16_t* const input,
          std::uint16_t* const output, const char* const operation,
          const std::size_t layer, void* const launch_stream) noexcept {
        if (reference_runner_detail::use_fp8_whole_chunk_prefill_projection(
                projection_backend_, weight, input, output, token_count)) {
          const int status =
              launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                  projection_backend_, weight, input, token_count, output,
                  launch_stream);
          if (status == static_cast<int>(cudaSuccess)) {
            return true;
          }
          if (status != static_cast<int>(cudaErrorNotSupported)) {
            return check_cuda(status, operation, layer);
          }
        }
        const std::size_t columns = linear_input_size(weight);
        const std::size_t rows = linear_output_size(weight);
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kProductionProjectionSubtileTokens) {
          const std::size_t remaining = token_count - token_offset;
          const std::size_t subtile_tokens =
              remaining < kProductionProjectionSubtileTokens
                  ? remaining
                  : kProductionProjectionSubtileTokens;
          if (!check_cuda(launch_projection_tile_to_bf16_cuda(
                              projection_backend_, weight,
                              input + token_offset * columns, subtile_tokens,
                              views_.fp32_scratch,
                              views_.fp32_scratch_elements,
                              output + token_offset * rows, launch_stream),
                          operation, layer)) {
            return false;
          }
        }
        return true;
      };
  const auto project = [this, &project_on_stream](
                           const LinearWeight& weight,
                           const std::uint16_t* const input,
                           std::uint16_t* const output,
                           const char* const operation,
                           const std::size_t layer) noexcept {
    return project_on_stream(weight, input, output, operation, layer, stream_);
  };
  const auto has_fp8_prefill_supermatrix_sidecar =
      [this, token_count](const LinearWeight& weight) noexcept {
        const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
        return projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
               token_count == kMaximumRequestPrefillChunkSize &&
               fp8 != nullptr &&
               fp8->prefill_supermatrix_sidecar != nullptr;
      };
  const auto project_fp8_prefill_supermatrix =
      [this, token_count, &check_cuda](
          const LinearWeight* const* const group_weights,
          std::uint16_t* const* const group_outputs,
          const std::size_t group_size,
          const std::uint16_t* const input,
          const char* const operation,
          const std::size_t layer) noexcept {
        if (group_weights == nullptr || group_outputs == nullptr ||
            group_size == 0U || group_size > 3U) {
          return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                            operation, layer);
        }
        std::array<kernels::Sm87Fp8PrefillSupermatrixPartition, 3U>
            partitions{};
        std::size_t columns = 0U;
        for (std::size_t index = 0U; index < group_size; ++index) {
          if (group_weights[index] == nullptr ||
              group_outputs[index] == nullptr) {
            return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                              operation, layer);
          }
          const auto* const fp8 =
              std::get_if<Fp8LinearWeight>(group_weights[index]);
          if (fp8 == nullptr ||
              fp8->prefill_supermatrix_sidecar == nullptr ||
              (index != 0U && fp8->input_size != columns)) {
            return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                              operation, layer);
          }
          columns = fp8->input_size;
          partitions[index] =
              kernels::Sm87Fp8PrefillSupermatrixPartition{
                  fp8->prefill_supermatrix_sidecar, fp8->weight_scale,
                  fp8->output_size, group_outputs[index]};
        }
        return check_cuda(
            kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
                partitions.data(), group_size, input, token_count, columns,
                stream_),
            operation, layer);
      };
  const auto project_attention_output =
      [this, token_count, &check_cuda, &project,
       &has_fp8_prefill_supermatrix_sidecar,
       &project_fp8_prefill_supermatrix](
          const LinearWeight& weight, const std::uint16_t* const input,
          std::uint16_t* const output, const char* const operation,
          const std::size_t layer) noexcept {
        if (has_fp8_prefill_supermatrix_sidecar(weight)) {
          const LinearWeight* const group_weights[1U] = {&weight};
          std::uint16_t* const group_outputs[1U] = {output};
          return project_fp8_prefill_supermatrix(
              group_weights, group_outputs, 1U, input, operation, layer);
        }
        if (!reference_runner_detail::
                use_fp8_m64_prefill_attention_output_projection(
                    projection_backend_, weight, input, output,
                    token_count)) {
          return project(weight, input, output, operation, layer);
        }
        return check_cuda(launch_projection_tile_to_bf16_cuda(
                              projection_backend_, weight, input,
                              token_count, views_.fp32_scratch,
                              views_.fp32_scratch_elements, output, stream_),
                          operation, layer);
      };
  const auto project_pair = [this, token_count, &check_cuda](
                                const LinearWeight& first_weight,
                                const LinearWeight& second_weight,
                                const std::uint16_t* const input,
                                std::uint16_t* const first_output,
                                std::uint16_t* const second_output,
                                const char* const operation,
                                const std::size_t layer) noexcept {
    const std::size_t columns = linear_input_size(first_weight);
    const std::size_t first_rows = linear_output_size(first_weight);
    const std::size_t second_rows = linear_output_size(second_weight);
    for (std::size_t token_offset = 0U; token_offset < token_count;
         token_offset += kProductionProjectionSubtileTokens) {
      const std::size_t remaining = token_count - token_offset;
      const std::size_t subtile_tokens =
          remaining < kProductionProjectionSubtileTokens
              ? remaining
              : kProductionProjectionSubtileTokens;
      if (!check_cuda(launch_projection_pair_tile_to_bf16_cuda(
                          projection_backend_, first_weight, second_weight,
                          input + token_offset * columns, subtile_tokens,
                          views_.fp32_scratch, views_.fp32_scratch_elements,
                          first_output + token_offset * first_rows,
                          second_output + token_offset * second_rows, stream_),
                      operation, layer)) {
        return false;
      }
    }
    return true;
  };
  const auto project_down = [this, token_count, &check_cuda,
                             &project](const LinearWeight& weight,
                                       const std::uint16_t* const input,
                                       std::uint16_t* const output,
                                       const char* const operation,
                                       const std::size_t layer) noexcept {
    if (reference_runner_detail::
            use_nvfp4_whole_chunk_prefill_down_projection(
                projection_backend_, weight, input, output, token_count)) {
      const int status = launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
          projection_backend_, weight, input, token_count, output, stream_);
      if (status == static_cast<int>(cudaErrorNotSupported)) {
        return project(weight, input, output, operation, layer);
      }
      return check_cuda(status, operation, layer);
    }
    if (token_count != kMaximumProjectionTileTokenCount) {
      return project(weight, input, output, operation, layer);
    }
    return check_cuda(launch_projection_tile_to_bf16_cuda(
                          projection_backend_, weight, input, token_count,
                          views_.fp32_scratch,
                          views_.fp32_scratch_elements, output, stream_),
                      operation, layer);
  };
  const auto project_nvfp4_whole_chunk_on_stream =
      [this, token_count, &check_cuda, &project_on_stream](
          const LinearWeight& weight, const std::uint16_t* const input,
          std::uint16_t* const output, const char* const operation,
          const std::size_t layer, void* const launch_stream) noexcept {
        const int status =
            launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
                projection_backend_, weight, input, token_count, output,
                launch_stream);
        if (status == static_cast<int>(cudaSuccess)) {
          return true;
        }
        if (status == static_cast<int>(cudaErrorNotSupported)) {
          return project_on_stream(weight, input, output, operation, layer,
                                   launch_stream);
        }
        return check_cuda(status, operation, layer);
      };
  const auto residual_norm_m32_tiles =
      [this, token_count, &check_cuda](
          const std::uint16_t* const left,
          const std::uint16_t* const right,
          const std::uint16_t* const norm_weight,
          std::uint16_t* const residual_output,
          std::uint16_t* const normalized_output,
          const char* const operation, const std::size_t layer) noexcept {
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kProductionProjectionSubtileTokens) {
          const std::size_t element_offset =
              token_offset * kReferenceHiddenSize;
          if (!check_cuda(
                  launch_residual_add_headwise_centered_rms_norm_m32_5120_cuda(
                      left + element_offset, right + element_offset,
                      norm_weight, kProductionProjectionSubtileTokens,
                      kReferenceHiddenSize, kRmsEpsilon,
                      residual_output + element_offset,
                      normalized_output + element_offset, stream_),
                  operation, layer)) {
            return false;
          }
        }
        return true;
      };
  const bool use_m32_residual_rms_fusion =
      reference_runner_detail::use_m32_prefill_residual_rms_fusion(
          token_count, kReferenceHiddenSize);

  for (std::size_t token = 0U; token < token_count; ++token) {
    if (!check_cuda(launch_embedding_gather_reference_cuda(
                        weights_->embed_tokens().weight,
                        kReferenceVocabularySize, kReferenceHiddenSize,
                        input_token_ids[token],
                        views_.hidden[0] + token * kReferenceHiddenSize,
                        stream_),
                    "prefill_embedding_gather", kReferenceNoLayer)) {
      return fail_prefill_tile(launch_failure);
    }
  }

  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& layer_weights = weights_->layer(layer);
    // The M32-tiled C32/C64 layer-(N-1) MLP boundary already produced layer
    // N's normalized input. Layer 0 still normalizes the embedding explicitly.
    if ((!use_m32_residual_rms_fusion || layer == 0U) &&
        !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                        views_.hidden[0],
                        layer_weights.input_layernorm.data, token_count,
                        kReferenceHiddenSize, kRmsEpsilon, views_.hidden[1],
                        stream_),
                    "prefill_input_layernorm", layer)) {
      return fail_prefill_tile(launch_failure);
    }

    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      const auto* const attention =
          std::get_if<LinearAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_prefill_tile(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "prefill_linear_attention_variant", layer));
      }
      bool linear_qkvz_projected = false;
      if (has_fp8_prefill_supermatrix_sidecar(attention->in_proj_qkv) &&
          has_fp8_prefill_supermatrix_sidecar(attention->in_proj_z)) {
        const LinearWeight* const group_weights[2U] = {
            &attention->in_proj_qkv, &attention->in_proj_z};
        std::uint16_t* const group_outputs[2U] = {
            views_.projection[0], views_.projection[1]};
        if (!project_fp8_prefill_supermatrix(
                group_weights, group_outputs, 2U, views_.hidden[1],
                "prefill_linear_qkvz_supermatrix_projection", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        linear_qkvz_projected = true;
      }
      if (!linear_qkvz_projected &&
          (!project(attention->in_proj_qkv, views_.hidden[1],
                    views_.projection[0], "prefill_linear_qkv_projection",
                    layer) ||
           !project(attention->in_proj_z, views_.hidden[1],
                    views_.projection[1], "prefill_linear_z_projection",
                    layer))) {
        return fail_prefill_tile(launch_failure);
      }
      if (supports_bf16_projection_pair(
              projection_backend_, attention->in_proj_a,
              attention->in_proj_b)) {
        if (!project_pair(attention->in_proj_a, attention->in_proj_b,
                          views_.hidden[1], views_.linear_a, views_.linear_b,
                          "prefill_linear_a_b_projection", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      } else if (!project(attention->in_proj_a, views_.hidden[1],
                          views_.linear_a, "prefill_linear_a_projection",
                          layer) ||
                 !project(attention->in_proj_b, views_.hidden[1],
                          views_.linear_b, "prefill_linear_b_projection",
                          layer)) {
        return fail_prefill_tile(launch_failure);
      }
      const bool use_gdn_c16_norm_gate =
          should_use_prefill_gdn_c16_norm_gate(
              enable_gdn_c16_norm_gate_admission, projection_backend_,
              first_position, token_count);
      bool gdn_output_is_normalized = false;
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
      const bool use_gdn_chunk64_native =
          use_prefill_gdn_chunk64_native_admission(
              enable_gdn_chunk64_native_admission, projection_backend_,
              first_position, token_count,
              prefill_gdn_chunk64_native_workspace_,
              prefill_gdn_chunk64_native_workspace_bytes_);
      if (use_gdn_chunk64_native) {
        if (!check_cuda(
                gdn_prefill_whole_span_conv_detail::
                    launch_causal_conv1d_silu_update_whole_span_exact_cuda(
                        views_.projection[0], token_count,
                        attention->conv1d.data, views_.conv_state[layer],
                        views_.projection[0], stream_),
                "prefill_linear_causal_conv_whole_span", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        ++g_prefill_gdn_chunk64_native_admission_hits;
        if (!check_cuda(
                gdn_prefill_chunk64_native_detail::launch(
                    prefill_gdn_chunk64_native_workspace_,
                    prefill_gdn_chunk64_native_workspace_bytes_,
                    views_.projection[0], token_count, views_.linear_a,
                    views_.linear_b, attention->a_log.data,
                    attention->dt_bias.data, views_.gdn_state[layer],
                    views_.gdn_state[layer], kRmsEpsilon,
                    attention->norm.data, views_.projection[1],
                    kRmsEpsilon, views_.projection[2], stream_),
                "prefill_linear_gdn_chunk64_native", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        gdn_output_is_normalized = true;
      } else
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
      const bool use_gdn_chunk64_reference =
          use_prefill_gdn_chunk64_reference_admission(
              enable_gdn_chunk64_reference_admission, projection_backend_,
              first_position, token_count,
              prefill_gdn_chunk64_reference_context_,
              prefill_gdn_chunk64_reference_workspace_,
              prefill_gdn_chunk64_reference_workspace_bytes_);
      if (use_gdn_chunk64_reference) {
        // Keep the established causal-convolution transition, then execute
        // the complete C64x8 WY hierarchy as one test-only architecture
        // reference. No external-library failure may fall back after state
        // mutation; the public call is poisoned through the normal failure
        // path instead.
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kPrefillKernelTileMaximumTokens) {
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data, views_.conv_state[layer],
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer)) {
            return fail_prefill_tile(launch_failure);
          }
        }
        ++g_prefill_gdn_chunk64_reference_admission_hits;
        if (!check_cuda(
                gdn_prefill_chunk64_reference_detail::launch(
                    prefill_gdn_chunk64_reference_context_,
                    prefill_gdn_chunk64_reference_workspace_,
                    prefill_gdn_chunk64_reference_workspace_bytes_,
                    token_count, views_.projection[0], views_.linear_a,
                    views_.linear_b,
                    attention->a_log.data, attention->dt_bias.data,
                    views_.gdn_state[layer], views_.gdn_state[layer],
                    kRmsEpsilon, attention->norm.data,
                    views_.projection[1], kRmsEpsilon,
                    views_.projection[2], stream_),
                "prefill_linear_gdn_chunk64_reference", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        gdn_output_is_normalized = true;
      } else
#endif
#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
      const bool use_gdn_b8 = use_prefill_gdn_b8_admission(
          enable_gdn_b8_admission, projection_backend_, first_position,
          token_count);
      if (use_gdn_b8) {
        // The B8 recurrence consumes the complete convolved QKV chunk. Keep
        // the causal-convolution state transition ordered in existing M16
        // subtiles, then replace only the GDN chain with one exact C256/C512
        // admission kernel. No workspace, synchronization, or Decode path is
        // added.
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kPrefillKernelTileMaximumTokens) {
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data, views_.conv_state[layer],
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer)) {
            return fail_prefill_tile(launch_failure);
          }
        }
        ++g_prefill_gdn_b8_admission_hits;
        if (!check_cuda(
                gdn_prefill_b8_detail::
                    launch_gated_delta_net_update_sequential_fp32_b8_exact_cuda(
                        views_.projection[0], token_count, views_.linear_a,
                        views_.linear_b, attention->a_log.data,
                        attention->dt_bias.data, views_.gdn_state[layer],
                        views_.gdn_state[layer], kRmsEpsilon,
                        views_.projection[2], stream_),
                "prefill_linear_gdn_b8_admission", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      } else
#endif
      if (use_gdn_c16_norm_gate) {
        // Preserve the established causal-convolution state order, then
        // consume every exact C16 slice in the C256/C512 composite
        // GDN/norm/gate route. Any launch error terminates the tile; the
        // production route never falls back after partially updating state.
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kPrefillKernelTileMaximumTokens) {
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data, views_.conv_state[layer],
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer) ||
              !check_cuda(
                  gdn_prefill_c16_norm_gate_detail::
                      launch_shared_boundary(
                          views_.projection[0] +
                              token_offset * kLinearQkvElements,
                          kPrefillKernelTileMaximumTokens,
                          views_.linear_a +
                              token_offset * kLinearScalarElements,
                          views_.linear_b +
                              token_offset * kLinearScalarElements,
                          attention->a_log.data, attention->dt_bias.data,
                          views_.gdn_state[layer], views_.gdn_state[layer],
                          kRmsEpsilon, attention->norm.data,
                          views_.projection[1] +
                              token_offset * kLinearValueElements,
                          kRmsEpsilon,
                          views_.projection[2] +
                              token_offset * kLinearValueElements,
                          stream_),
                  "prefill_linear_gdn_c16_norm_gate", layer)) {
            return fail_prefill_tile(launch_failure);
          }
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
          ++g_prefill_gdn_c16_norm_gate_admission_hits;
#endif
        }
        gdn_output_is_normalized = true;
      } else {
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kPrefillKernelTileMaximumTokens) {
          const std::size_t remaining = token_count - token_offset;
          const std::size_t subtile_tokens =
              remaining < kPrefillKernelTileMaximumTokens
                  ? remaining
                  : kPrefillKernelTileMaximumTokens;
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      subtile_tokens, attention->conv1d.data,
                      views_.conv_state[layer],
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer) ||
              !check_cuda(
                  launch_gated_delta_net_update_tile_warp_parallel_cuda(
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      subtile_tokens,
                      views_.linear_a +
                          token_offset * kLinearScalarElements,
                      views_.linear_b +
                          token_offset * kLinearScalarElements,
                      attention->a_log.data, attention->dt_bias.data,
                      views_.gdn_state[layer], views_.gdn_state[layer],
                      kRmsEpsilon,
                      views_.projection[2] +
                          token_offset * kLinearValueElements,
                      {}, stream_),
                  "prefill_linear_gdn", layer)) {
            return fail_prefill_tile(launch_failure);
          }
        }
      }
      if (!gdn_output_is_normalized &&
          !check_cuda(
              launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                  views_.projection[2], attention->norm.data,
                  views_.projection[1], token_count * kGdnValueHeadCount,
                  kGdnHeadDimension, kRmsEpsilon, views_.projection[2],
                  stream_),
              "prefill_linear_output_norm_gate", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      if (!project_attention_output(
              attention->out_proj, views_.projection[2], views_.hidden[1],
              "prefill_linear_output_projection", layer)) {
        return fail_prefill_tile(launch_failure);
      }
    } else if (expected == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_prefill_tile(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "prefill_full_attention_variant", layer));
      }
      std::uint16_t* const packed_gates =
          views_.projection[3] + token_count * kFullQueryElements;
      std::uint16_t* const tile_key =
          views_.key_cache[layer] +
          static_cast<std::size_t>(first_position) * kFullKvElements;
      std::uint16_t* const tile_value =
          views_.value_cache[layer] +
          static_cast<std::size_t>(first_position) * kFullKvElements;
      const std::size_t rope_first_position =
          static_cast<std::size_t>(first_position);
      bool full_qkv_projected = false;
      if (has_fp8_prefill_supermatrix_sidecar(attention->q_proj) &&
          has_fp8_prefill_supermatrix_sidecar(attention->k_proj) &&
          has_fp8_prefill_supermatrix_sidecar(attention->v_proj)) {
        const LinearWeight* const group_weights[3U] = {
            &attention->q_proj, &attention->k_proj, &attention->v_proj};
        std::uint16_t* const group_outputs[3U] = {
            views_.projection[0], tile_key, tile_value};
        if (!project_fp8_prefill_supermatrix(
                group_weights, group_outputs, 3U, views_.hidden[1],
                "prefill_full_qkv_supermatrix_projection", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        full_qkv_projected = true;
      }
      if (!full_qkv_projected &&
          (!project(attention->q_proj, views_.hidden[1],
                    views_.projection[0], "prefill_full_q_gate_projection",
                    layer) ||
           !project(attention->k_proj, views_.hidden[1], tile_key,
                    "prefill_full_k_projection", layer) ||
           !project(attention->v_proj, views_.hidden[1], tile_value,
                    "prefill_full_v_projection", layer))) {
        return fail_prefill_tile(launch_failure);
      }

      bool use_fused_preprocess = true;
      for (std::size_t token_offset = 0U; token_offset < token_count;
           token_offset += kPrefillKernelTileMaximumTokens) {
        const std::size_t remaining = token_count - token_offset;
        const std::size_t subtile_tokens =
            remaining < kPrefillKernelTileMaximumTokens
                ? remaining
                : kPrefillKernelTileMaximumTokens;
        if (!reference_runner_detail::use_qk_rope_tile(
                rope_first_position + token_offset, subtile_tokens)) {
          use_fused_preprocess = false;
          break;
        }
      }
      std::uint16_t* const tile_query =
          use_fused_preprocess ? views_.projection[3]
                               : views_.projection[0];
      for (std::size_t token_offset = 0U; token_offset < token_count;
           token_offset += kPrefillKernelTileMaximumTokens) {
        const std::size_t remaining = token_count - token_offset;
        const std::size_t subtile_tokens =
            remaining < kPrefillKernelTileMaximumTokens
                ? remaining
                : kPrefillKernelTileMaximumTokens;
        std::uint16_t* const raw_query_gate =
            views_.projection[0] + token_offset * kFullQGateElements;
        std::uint16_t* const subtile_query =
            tile_query + token_offset * kFullQueryElements;
        std::uint16_t* const split_query =
            views_.projection[3] + token_offset * kFullQueryElements;
        std::uint16_t* const subtile_gates =
            packed_gates + token_offset * kFullQueryElements;
        std::uint16_t* const subtile_key =
            tile_key + token_offset * kFullKvElements;
        const std::size_t subtile_first_position =
            rope_first_position + token_offset;
        if (use_fused_preprocess) {
          if (!check_cuda(
                  launch_full_attention_preprocess_24_4_256_64_cuda(
                      raw_query_gate, subtile_key, attention->q_norm.data,
                      attention->k_norm.data, kRmsEpsilon, subtile_query,
                      subtile_gates, views_.rope_cos, views_.rope_sin,
                      subtile_first_position, subtile_tokens, stream_),
                  "prefill_full_preprocess", layer)) {
            return fail_prefill_tile(launch_failure);
          }
          continue;
        }
        if (!check_cuda(launch_split_interleaved_q_gate_reference_cuda(
                            raw_query_gate,
                            subtile_tokens * kFullQueryHeads,
                            kFullHeadDimension, split_query, subtile_gates,
                            stream_),
                        "prefill_full_split_q_gate_fallback", layer) ||
            !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                            split_query, attention->q_norm.data,
                            subtile_tokens * kFullQueryHeads,
                            kFullHeadDimension, kRmsEpsilon, subtile_query,
                            stream_),
                        "prefill_full_q_norm_fallback", layer) ||
            !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                            subtile_key, attention->k_norm.data,
                            subtile_tokens * kFullKvHeads,
                            kFullHeadDimension, kRmsEpsilon, subtile_key,
                            stream_),
                        "prefill_full_k_norm_fallback", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        for (std::size_t token = 0U; token < subtile_tokens; ++token) {
          const std::size_t position = subtile_first_position + token;
          const float* const cosines =
              views_.rope_cos + position * kRopePairs;
          const float* const sines =
              views_.rope_sin + position * kRopePairs;
          if (!check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                              subtile_query + token * kFullQueryElements,
                              cosines, sines, kFullQueryHeads,
                              subtile_query + token * kFullQueryElements,
                              stream_),
                          "prefill_full_q_rope_fallback", layer) ||
              !check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                              subtile_key + token * kFullKvElements,
                              cosines, sines, kFullKvHeads,
                              subtile_key + token * kFullKvElements,
                              stream_),
                          "prefill_full_k_rope_fallback", layer)) {
            return fail_prefill_tile(launch_failure);
          }
        }
      }
      const bool use_bulk_gqa_gate =
          reference_runner_detail::
              use_bulk_causal_gqa_sigmoid_gate_prefill(
                  projection_backend_, expected, first_position,
                  token_count);
      if (use_bulk_gqa_gate) {
        if (!check_cuda(
                launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
                    tile_query, views_.key_cache[layer],
                    views_.value_cache[layer], packed_gates,
                    first_position, token_count, views_.projection[1],
                    stream_),
                "prefill_full_bulk_gqa_output_gate", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      } else {
        const std::size_t fused_gqa_gate_prefix_tokens =
            reference_runner_detail::
                fused_gqa_sigmoid_gate_prefix_token_count(first_position,
                                                          token_count);
        for (std::size_t token = 0U; token < token_count; ++token) {
          const std::size_t sequence_length =
              static_cast<std::size_t>(first_position) + token + 1U;
          const std::uint16_t* const token_query =
              tile_query + token * kFullQueryElements;
          std::uint16_t* const token_output =
              views_.projection[1] + token * kFullQueryElements;
          const bool fuse_gqa_gate_token =
              token < fused_gqa_gate_prefix_tokens;
          const int gqa_status =
              fuse_gqa_gate_token
                  ? launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                        token_query, views_.key_cache[layer],
                        views_.value_cache[layer], sequence_length,
                        kAttentionScale, views_.fp32_scratch,
                        views_.fp32_scratch_elements,
                        packed_gates + token * kFullQueryElements,
                        token_output, stream_)
                  : launch_gqa_attention_reference_cuda(
                        token_query, views_.key_cache[layer],
                        views_.value_cache[layer], kFullQueryHeads,
                        kFullKvHeads, sequence_length, kFullHeadDimension,
                        kAttentionScale, views_.fp32_scratch,
                        views_.fp32_scratch_elements, token_output, stream_);
          if (!check_cuda(
                  gqa_status,
                  fuse_gqa_gate_token ? "prefill_full_gqa_output_gate"
                                      : "prefill_full_gqa",
                  layer)) {
            return fail_prefill_tile(launch_failure);
          }
        }
        if (fused_gqa_gate_prefix_tokens < token_count &&
            !check_cuda(launch_sigmoid_gate_reference_cuda(
                            views_.projection[1] +
                                fused_gqa_gate_prefix_tokens *
                                    kFullQueryElements,
                            packed_gates +
                                fused_gqa_gate_prefix_tokens *
                                    kFullQueryElements,
                            (token_count - fused_gqa_gate_prefix_tokens) *
                                kFullQueryElements,
                            views_.projection[1] +
                                fused_gqa_gate_prefix_tokens *
                                    kFullQueryElements,
                            stream_),
                        "prefill_full_output_gate", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      }
      if (!project_attention_output(
              attention->o_proj, views_.projection[1], views_.hidden[1],
              "prefill_full_output_projection", layer)) {
        return fail_prefill_tile(launch_failure);
      }
    } else {
      return fail_prefill_tile(runner_status(
          ReferenceRunnerError::kInvalidLayerSchedule,
          "prefill_layer_schedule", layer));
    }

    if (use_m32_residual_rms_fusion) {
      if (!residual_norm_m32_tiles(
              views_.hidden[0], views_.hidden[1],
              layer_weights.post_attention_layernorm.data, views_.hidden[2],
              views_.hidden[1],
              "prefill_attention_residual_post_attention_layernorm", layer)) {
        return fail_prefill_tile(launch_failure);
      }
    } else {
      if (!check_cuda(launch_residual_add_reference_cuda(
                          views_.hidden[0], views_.hidden[1],
                          token_count * kReferenceHiddenSize,
                          views_.hidden[2], stream_),
                      "prefill_attention_residual", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      if (!check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                          views_.hidden[2],
                          layer_weights.post_attention_layernorm.data,
                          token_count, kReferenceHiddenSize, kRmsEpsilon,
                          views_.hidden[1], stream_),
                      "prefill_post_attention_layernorm", layer)) {
        return fail_prefill_tile(launch_failure);
      }
    }
    bool marlin_mlp_completed = false;
#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
    const auto* const marlin_gate =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.gate_proj);
    const auto* const marlin_up =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.up_proj);
    const auto* const marlin_down =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.down_proj);
    const std::size_t marlin_branch_elements =
        token_count * kReferenceIntermediateSize;
    const std::size_t marlin_workspace_branch_elements =
        static_cast<std::size_t>(state_->plan().prefill_chunk_size) *
        kReferenceIntermediateSize;
    const bool use_marlin_mlp =
        g_enable_nvfp4_marlin_prefill_admission &&
        kernels::sm87_nvfp4_marlin_supports_token_count(token_count) &&
        projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
        marlin_gate != nullptr && marlin_up != nullptr &&
        marlin_down != nullptr &&
        marlin_gate->prefill_marlin_weight != nullptr &&
        marlin_gate->prefill_marlin_scales != nullptr &&
        marlin_gate->prefill_marlin_global_scale != nullptr &&
        marlin_gate->prefill_marlin_weight ==
            marlin_up->prefill_marlin_weight &&
        marlin_gate->prefill_marlin_scales ==
            marlin_up->prefill_marlin_scales &&
        marlin_gate->prefill_marlin_global_scale ==
            marlin_up->prefill_marlin_global_scale &&
        marlin_down->prefill_marlin_weight != nullptr &&
        marlin_down->prefill_marlin_scales != nullptr &&
        marlin_down->prefill_marlin_global_scale != nullptr &&
        views_.projection[1] ==
            views_.projection[0] + marlin_workspace_branch_elements &&
        marlin_branch_elements <= marlin_workspace_branch_elements &&
        views_.fp32_scratch_elements >=
            kernels::kSm87NvFp4MarlinReductionElements;
    if (use_marlin_mlp) {
      auto* const locks =
          reinterpret_cast<std::int32_t*>(views_.projection[3]);
      const auto stream = reinterpret_cast<cudaStream_t>(stream_);
      if (!check_cuda(
              static_cast<int>(cudaMemsetAsync(
                  locks, 0, kernels::kSm87NvFp4MarlinLockBytes, stream)),
              "prefill_marlin_clear_locks", layer) ||
          !check_cuda(
              kernels::launch_sm87_nvfp4_marlin_gate_up_cuda(
                  views_.hidden[1], marlin_gate->prefill_marlin_weight,
                  marlin_gate->prefill_marlin_scales,
                  marlin_gate->prefill_marlin_global_scale,
                  token_count, views_.projection[0], views_.fp32_scratch,
                  locks, stream_),
              "prefill_marlin_gate_up", layer) ||
          !check_cuda(
              kernels::launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
                  views_.projection[0], token_count, views_.projection[2],
                  stream_),
              "prefill_marlin_gate_up_silu", layer) ||
          !check_cuda(
              kernels::launch_sm87_nvfp4_marlin_down_cuda(
                  views_.projection[2], marlin_down->prefill_marlin_weight,
                  marlin_down->prefill_marlin_scales,
                  marlin_down->prefill_marlin_global_scale, token_count,
                  views_.hidden[1], views_.fp32_scratch, locks, stream_),
              "prefill_marlin_down", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      ++g_nvfp4_marlin_prefill_admission_hits;
      marlin_mlp_completed = true;
    }
#endif

    if (!marlin_mlp_completed) {
      // Production Prefill is self-hosted only. External-library comparison
      // modules are compiled exclusively into benchmark targets.
      {
      const bool gate_up_fork_join_available =
          prefill_auxiliary_stream_ != nullptr &&
          prefill_branch_ready_event_ != nullptr &&
          prefill_branch_done_event_ != nullptr;
      const bool use_gate_up_whole_chunk_dual_stream =
          gate_up_fork_join_available &&
          reference_runner_detail::
              use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
                  projection_backend_, layer_weights.mlp.gate_proj,
                  layer_weights.mlp.up_proj, views_.hidden[1],
                  views_.projection[0], views_.projection[1], token_count);
      const bool use_gate_up_m32_dual_stream =
          gate_up_fork_join_available &&
          reference_runner_detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              projection_backend_, layer_weights.mlp.gate_proj,
              layer_weights.mlp.up_proj, views_.hidden[1],
              views_.projection[0], views_.projection[1], token_count);
      const bool use_gate_up_dual_stream =
          use_gate_up_whole_chunk_dual_stream ||
          use_gate_up_m32_dual_stream;
      if (use_gate_up_dual_stream) {
        const auto auxiliary_stream =
            reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_);
        const auto branch_ready =
            reinterpret_cast<cudaEvent_t>(prefill_branch_ready_event_);
        const auto branch_done =
            reinterpret_cast<cudaEvent_t>(prefill_branch_done_event_);
        const auto project_gate = [&]() noexcept {
          if (use_gate_up_whole_chunk_dual_stream) {
            return project_nvfp4_whole_chunk_on_stream(
                layer_weights.mlp.gate_proj, views_.hidden[1],
                views_.projection[0], "prefill_mlp_gate_projection", layer,
                stream_);
          }
          return project(layer_weights.mlp.gate_proj, views_.hidden[1],
                         views_.projection[0], "prefill_mlp_gate_projection",
                         layer);
        };
        const auto project_up = [&]() noexcept {
          if (use_gate_up_whole_chunk_dual_stream) {
            return project_nvfp4_whole_chunk_on_stream(
                layer_weights.mlp.up_proj, views_.hidden[1],
                views_.projection[1], "prefill_mlp_up_projection_auxiliary",
                layer, prefill_auxiliary_stream_);
          }
          return project_on_stream(
              layer_weights.mlp.up_proj, views_.hidden[1],
              views_.projection[1], "prefill_mlp_up_projection_auxiliary",
              layer, prefill_auxiliary_stream_);
        };
        if (!check_cuda(
                static_cast<int>(cudaEventRecord(branch_ready, stream)),
                "prefill_mlp_branch_ready_record", layer) ||
            !check_cuda(static_cast<int>(cudaStreamWaitEvent(
                            auxiliary_stream, branch_ready, 0U)),
                        "prefill_mlp_branch_ready_wait", layer) ||
            !project_gate() || !project_up() ||
            !check_cuda(
                static_cast<int>(cudaEventRecord(branch_done,
                                                 auxiliary_stream)),
                "prefill_mlp_branch_done_record", layer) ||
            !check_cuda(static_cast<int>(
                            cudaStreamWaitEvent(stream, branch_done, 0U)),
                        "prefill_mlp_branch_done_wait", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      } else if (!project(layer_weights.mlp.gate_proj, views_.hidden[1],
                          views_.projection[0],
                          "prefill_mlp_gate_projection", layer) ||
                 !project(layer_weights.mlp.up_proj, views_.hidden[1],
                          views_.projection[1],
                          "prefill_mlp_up_projection", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      }
      if (!check_cuda(launch_silu_mul_reference_cuda(
                        views_.projection[0], views_.projection[1],
                        token_count * kReferenceIntermediateSize,
                        views_.projection[0], stream_),
                    "prefill_mlp_silu_mul", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      if (!project_down(layer_weights.mlp.down_proj, views_.projection[0],
                        views_.hidden[1], "prefill_mlp_down_projection",
                        layer)) {
        return fail_prefill_tile(launch_failure);
      }
    }
    if (use_m32_residual_rms_fusion) {
      const bool is_final_layer =
          layer + 1U == kReferenceDecoderLayerCount;
      const std::uint16_t* const next_norm_weight =
          is_final_layer
              ? weights_->final_norm().data
              : weights_->layer(layer + 1U).input_layernorm.data;
      const char* const operation =
          is_final_layer ? "prefill_layer_residual_final_norm"
                         : "prefill_layer_residual_next_input_layernorm";
      if (!residual_norm_m32_tiles(
              views_.hidden[2], views_.hidden[1], next_norm_weight,
              views_.hidden[0], views_.hidden[1], operation, layer)) {
        return fail_prefill_tile(launch_failure);
      }
    } else if (!check_cuda(launch_residual_add_reference_cuda(
                               views_.hidden[2], views_.hidden[1],
                               token_count * kReferenceHiddenSize,
                               views_.hidden[0], stream_),
                           "prefill_layer_residual", layer)) {
      return fail_prefill_tile(launch_failure);
    }
  }

  // Match the non-logit step boundary even though this output is not consumed
  // by the following layer-major tile or by persistent state.
  // The C32/C64 M32-tiled path folded this norm into the final layer's MLP
  // residual boundary.
  if (!use_m32_residual_rms_fusion &&
      !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                      views_.hidden[0], weights_->final_norm().data,
                      token_count, kReferenceHiddenSize, kRmsEpsilon,
                      views_.hidden[1], stream_),
                  "prefill_final_norm", kReferenceNoLayer)) {
    return fail_prefill_tile(launch_failure);
  }

  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kCudaFailure, "prefill_tile_synchronize",
        kReferenceNoLayer, static_cast<int>(sync_status)));
  }
  const std::uint32_t committed_length =
      first_position + static_cast<std::uint32_t>(token_count);
  const RequestOperationStatus commit_status =
      state_->set_sequence_length(committed_length);
  if (!commit_status) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kStateCommitFailure,
        "prefill_tile_commit", kReferenceNoLayer,
        commit_status.cuda_error));
  }
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  const auto native_chunk64_snapshot_hook =
      g_prefill_gdn_chunk64_native_snapshot_hook;
  if (native_chunk64_snapshot_hook.callback != nullptr) {
    native_chunk64_snapshot_hook.callback(
        *state_, native_chunk64_snapshot_hook.context);
  }
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  const auto chunk64_snapshot_hook =
      g_prefill_gdn_chunk64_reference_snapshot_hook;
  if (chunk64_snapshot_hook.callback != nullptr) {
    chunk64_snapshot_hook.callback(*state_, chunk64_snapshot_hook.context);
  }
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
  // prefill_tile_synchronize above completed the full C512 state transition;
  // observe it only after the logical sequence length was committed.
  const auto snapshot_hook =
      g_prefill_gdn_c16_norm_gate_admission_snapshot_hook;
  if (snapshot_hook.callback != nullptr) {
    snapshot_hook.callback(
        *state_, reference_runner_detail::
                     PrefillGdnC16NormGateAdmissionSnapshotStage::kPrefixTile,
        snapshot_hook.context);
  }
#endif

  ReferencePrefillTileResult tile;
  tile.step_count = token_count;
  for (std::size_t token = 0U; token < token_count; ++token) {
    tile.steps[token].position =
        first_position + static_cast<std::uint32_t>(token);
    tile.steps[token].input_token_id = input_token_ids[token];
  }
  if (options.measure_timing) {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - started;
    tile.timing.emplace(ReferenceStepTiming{elapsed.count()});
  }
  if (options.retain_last_hidden_for_logits) {
    retained_prefill_hidden_valid_ = true;
    retained_prefill_position_ =
        committed_length - 1U;
    retained_prefill_input_token_ = input_token_ids[token_count - 1U];
    retained_prefill_hidden_row_ = token_count - 1U;
  }
  ReferencePrefillTileOutcome outcome;
  outcome.value.emplace(std::move(tile));
  return outcome;
}

ReferenceStepOutcome ReferenceRunner::finish_prefill_from_retained_tile(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) noexcept {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }

  // Consume the hand-off before validation or launch. A failed finalization
  // poisons the request exactly like a failed ordinary step and can never
  // accidentally reuse a workspace row after another operation.
  const bool retained_valid = retained_prefill_hidden_valid_;
  const std::uint32_t retained_position = retained_prefill_position_;
  const std::uint32_t retained_input_token =
      retained_prefill_input_token_;
  const std::size_t retained_hidden_row = retained_prefill_hidden_row_;
  retained_prefill_hidden_valid_ = false;

  if (!static_cast<bool>(*this)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "finish_prefill_from_retained_tile"));
  }
  if (poisoned_) {
    ReferenceStepOutcome outcome;
    outcome.status = runner_status(
        ReferenceRunnerError::kPoisoned,
        "finish_prefill_from_retained_tile");
    return outcome;
  }
  if (!options.compute_logits || options.capture_trace ||
      !is_valid_reference_logits_mode(options.logits_mode)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "retained_prefill_logits_options"));
  }
  if (!retained_valid || state_->current_position() == 0U ||
      retained_position + 1U != state_->current_position() ||
      retained_input_token != input_token_id ||
      retained_hidden_row >= kMaximumRequestPrefillChunkSize) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "retained_prefill_hidden_contract"));
  }

  ReferenceRunnerStatus launch_failure{};
  const auto check_cuda = [&launch_failure](
                              const int status,
                              const char* const operation) noexcept {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    launch_failure = runner_status(ReferenceRunnerError::kCudaFailure,
                                   operation, kReferenceNoLayer, status);
    return false;
  };
  const bool prediction_only =
      options.logits_mode == ReferenceLogitsMode::kPredictedTokenOnly;
  const bool use_sm87_bf16_logits =
      projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
      linear_weight_kind(weights_->lm_head()) != LinearWeightKind::kBf16;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
  const std::uint16_t* const final_normalized_hidden =
      views_.hidden[1] + retained_hidden_row * kReferenceHiddenSize;

  if (use_sm87_bf16_logits) {
    auto* const device_bf16_logits =
        reinterpret_cast<std::uint16_t*>(views_.fp32_scratch);
    if (!check_cuda(
            launch_projection_to_bf16_cuda(
                projection_backend_, weights_->lm_head(),
                final_normalized_hidden, nullptr, 0U, device_bf16_logits,
                stream_),
            "retained_prefill_lm_head_sm87_bf16")) {
      return fail_step(launch_failure);
    }
    if (prediction_only) {
      constexpr std::size_t kGreedyWorkspaceBytes =
          kReferenceVocabularySize * sizeof(std::uint16_t) +
          kBf16GreedyArgmaxWorkspaceResults *
              sizeof(Bf16GreedyArgmaxResult);
      static_assert((kReferenceVocabularySize * sizeof(std::uint16_t)) %
                            alignof(Bf16GreedyArgmaxResult) ==
                        0U);
      if (views_.fp32_scratch_elements <
          (kGreedyWorkspaceBytes + sizeof(float) - 1U) / sizeof(float)) {
        return fail_step(runner_status(
            ReferenceRunnerError::kInvalidRequestState,
            "retained_prefill_bf16_greedy_argmax_workspace"));
      }
      auto* const greedy_workspace =
          reinterpret_cast<Bf16GreedyArgmaxResult*>(
              device_bf16_logits + kReferenceVocabularySize);
      if (!check_cuda(
              launch_bf16_greedy_argmax_cuda(
                  device_bf16_logits, kReferenceVocabularySize,
                  greedy_workspace, stream_),
              "retained_prefill_bf16_greedy_argmax") ||
          !check_cuda(
              static_cast<int>(cudaMemcpyAsync(
                  pinned_logits_, greedy_workspace,
                  sizeof(Bf16GreedyArgmaxResult), cudaMemcpyDeviceToHost,
                  stream)),
              "retained_prefill_logits_prediction_d2h")) {
        return fail_step(launch_failure);
      }
    } else if (!check_cuda(
                   static_cast<int>(cudaMemcpyAsync(
                       pinned_logits_, device_bf16_logits,
                       kReferenceVocabularySize * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost, stream)),
                   "retained_prefill_logits_bf16_d2h")) {
      return fail_step(launch_failure);
    }
  } else if (!check_cuda(
                 launch_projection_reference_cuda(
                     weights_->lm_head(), final_normalized_hidden,
                     views_.fp32_scratch, stream_),
                 "retained_prefill_lm_head") ||
             !check_cuda(
                 static_cast<int>(cudaMemcpyAsync(
                     pinned_logits_, views_.fp32_scratch,
                     kReferenceVocabularySize * sizeof(float),
                     cudaMemcpyDeviceToHost, stream)),
                 "retained_prefill_logits_d2h")) {
    return fail_step(launch_failure);
  }

  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return fail_step(runner_status(
        ReferenceRunnerError::kCudaFailure,
        "retained_prefill_logits_synchronize", kReferenceNoLayer,
        static_cast<int>(sync_status)));
  }

  ReferenceStepResult result;
  result.position = retained_position;
  result.input_token_id = input_token_id;
  if (prediction_only && use_sm87_bf16_logits) {
    const auto& greedy =
        *static_cast<const Bf16GreedyArgmaxResult*>(pinned_logits_);
    if (greedy.has_nonfinite != 0U) {
      return fail_step(runner_status(
          ReferenceRunnerError::kNonFiniteLogits,
          "retained_prefill_bf16_greedy_argmax"));
    }
    if (greedy.index >= kReferenceVocabularySize) {
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "retained_prefill_bf16_greedy_argmax_result"));
    }
    result.prediction.emplace(ReferenceStepPrediction{greedy.index});
  } else {
    const reference_runner_detail::LogitsAnalysis analysis =
        use_sm87_bf16_logits
            ? reference_runner_detail::analyze_bf16_logits_bits(
                  static_cast<const std::uint16_t*>(pinned_logits_),
                  kReferenceVocabularySize)
            : (prediction_only
                   ? reference_runner_detail::analyze_bf16_argmax_in_place(
                         static_cast<float*>(pinned_logits_),
                         kReferenceVocabularySize)
                   : reference_runner_detail::analyze_bf16_logits_in_place(
                         static_cast<float*>(pinned_logits_),
                         kReferenceVocabularySize));
    if (!analysis.ok()) {
      return fail_step(runner_status(
          ReferenceRunnerError::kNonFiniteLogits,
          "retained_prefill_bf16_logits_analysis"));
    }
    if (prediction_only) {
      result.prediction.emplace(ReferenceStepPrediction{
          static_cast<std::uint32_t>(analysis.predicted_index)});
    } else {
      ReferenceStepLogits logits;
      logits.predicted_token_id =
          static_cast<std::uint32_t>(analysis.predicted_index);
      logits.chosen_logit = analysis.maximum;
      logits.max_log_probability = analysis.max_log_probability;
      logits.logsumexp = analysis.logsumexp;
      result.logits.emplace(logits);
    }
  }
  if (options.measure_timing) {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - started;
    result.timing.emplace(ReferenceStepTiming{elapsed.count()});
  }

  ReferenceStepOutcome outcome;
  outcome.value.emplace(std::move(result));
  return outcome;
}

ReferenceRunnerFactoryResult create_reference_runner(
    const ModelWeights* const weights, RequestState* const state,
    const ReferenceRunnerOptions& options) noexcept {
  ReferenceRunnerFactoryResult result;
  if (!is_valid_projection_backend(options.projection_backend)) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency, "projection_backend");
    return result;
  }
  const ReferenceRunnerStatus weights_status = validate_model_weights(weights);
  if (!weights_status) {
    result.diagnostic = weights_status;
    return result;
  }

  ReferenceRunner runner;
  runner.weights_ = weights;
  runner.state_ = state;
  runner.projection_backend_ = options.projection_backend;
  const ReferenceRunnerStatus state_status =
      ReferenceRunner::collect_request_views(state, runner.views_);
  if (!state_status) {
    result.diagnostic = state_status;
    return result;
  }

  cudaStream_t stream = nullptr;
  cudaError_t status =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kCudaFailure, "cudaStreamCreateWithFlags",
        kReferenceNoLayer, static_cast<int>(status));
    return result;
  }
  runner.stream_ = reinterpret_cast<void*>(stream);

#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  runner.prefill_gdn_chunk64_native_workspace_bytes_ =
      gdn_prefill_chunk64_native_detail::workspace_bytes();
  status = cudaMalloc(
      &runner.prefill_gdn_chunk64_native_workspace_,
      runner.prefill_gdn_chunk64_native_workspace_bytes_);
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kAllocationFailure,
        "cudaMalloc(gdn_chunk64_native_workspace)", kReferenceNoLayer,
        static_cast<int>(status));
    return result;
  }
#endif

#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  status = static_cast<cudaError_t>(
      gdn_prefill_chunk64_reference_detail::create_context(
          &runner.prefill_gdn_chunk64_reference_context_));
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kAllocationFailure,
        "create_gdn_chunk64_reference_context", kReferenceNoLayer,
        static_cast<int>(status));
    return result;
  }
  runner.prefill_gdn_chunk64_reference_workspace_bytes_ =
      gdn_prefill_chunk64_reference_detail::workspace_bytes();
  status = cudaMalloc(
      &runner.prefill_gdn_chunk64_reference_workspace_,
      runner.prefill_gdn_chunk64_reference_workspace_bytes_);
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kAllocationFailure,
        "cudaMalloc(gdn_chunk64_reference_workspace)", kReferenceNoLayer,
        static_cast<int>(status));
    return result;
  }
#endif

  if (options.projection_backend == ProjectionBackend::kSm87WeightOnly) {
    cudaStream_t auxiliary_stream = nullptr;
    cudaEvent_t branch_ready = nullptr;
    cudaEvent_t branch_done = nullptr;
    const bool auxiliary_ready =
        cudaStreamCreateWithFlags(&auxiliary_stream,
                                  cudaStreamNonBlocking) == cudaSuccess &&
        cudaEventCreateWithFlags(&branch_ready,
                                 cudaEventDisableTiming) == cudaSuccess &&
        cudaEventCreateWithFlags(&branch_done,
                                 cudaEventDisableTiming) == cudaSuccess;
    if (auxiliary_ready) {
      runner.prefill_auxiliary_stream_ =
          reinterpret_cast<void*>(auxiliary_stream);
      runner.prefill_branch_ready_event_ =
          reinterpret_cast<void*>(branch_ready);
      runner.prefill_branch_done_event_ =
          reinterpret_cast<void*>(branch_done);
    } else {
      // The auxiliary branch is a latency optimization, not a runner
      // dependency. Preserve the serial SM87 path if any control resource is
      // unavailable, including after a partially successful allocation.
      if (branch_ready != nullptr) {
        (void)cudaEventDestroy(branch_ready);
      }
      if (branch_done != nullptr) {
        (void)cudaEventDestroy(branch_done);
      }
      if (auxiliary_stream != nullptr) {
        (void)cudaStreamDestroy(auxiliary_stream);
      }
      // This failure is deliberately downgraded to a serial fallback. Do not
      // leak its thread-local CUDA last-error state to the returned runner.
      (void)cudaGetLastError();
    }
  }
  status = cudaHostAlloc(&runner.pinned_logits_,
                         kReferenceVocabularySize * sizeof(float),
                         cudaHostAllocDefault);
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kAllocationFailure, "cudaHostAlloc(logits)",
        kReferenceNoLayer, static_cast<int>(status));
    return result;
  }

  runner.trace_enabled_ = options.enable_trace;
  if (options.enable_trace) {
    status = cudaHostAlloc(reinterpret_cast<void**>(&runner.pinned_trace_),
                           kReferenceTraceElements * sizeof(std::uint16_t),
                           cudaHostAllocDefault);
    if (status != cudaSuccess) {
      result.diagnostic = runner_status(
          ReferenceRunnerError::kAllocationFailure,
          "cudaHostAlloc(trace)", kReferenceNoLayer,
          static_cast<int>(status));
      return result;
    }
  }

  result.value.emplace(std::move(runner));
  return result;
}

}  // namespace q3x::runtime
