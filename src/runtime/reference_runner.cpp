#include "q3x/runtime/reference_runner.h"

#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/gdn_decode.h"
#include "q3x/runtime/layout_ops.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kAttentionScale = 1.0F / 16.0F;

static_assert(kLinearQkvElements <= kReferenceIntermediateSize);
static_assert(kLinearValueElements <= kReferenceIntermediateSize);
static_assert(kFullQGateElements <= kReferenceIntermediateSize);
static_assert(kFullQueryElements <= kReferenceIntermediateSize);
static_assert(kFullKvElements <= kReferenceIntermediateSize);

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
  pinned_logits_ = std::exchange(other.pinned_logits_, nullptr);
  pinned_trace_ = std::exchange(other.pinned_trace_, nullptr);
  views_ = other.views_;
  other.views_ = {};
  projection_backend_ = std::exchange(
      other.projection_backend_, ProjectionBackend::kReference);
  trace_enabled_ = std::exchange(other.trace_enabled_, false);
  trace_valid_ = std::exchange(other.trace_valid_, false);
  poisoned_ = std::exchange(other.poisoned_, false);
  trace_position_ = std::exchange(other.trace_position_, 0U);
  trace_input_token_ = std::exchange(other.trace_input_token_, 0U);
  return *this;
}

ReferenceRunner::operator bool() const noexcept {
  return weights_ != nullptr && state_ != nullptr && stream_ != nullptr &&
         pinned_logits_ != nullptr;
}

std::uint32_t ReferenceRunner::current_position() const noexcept {
  return state_ == nullptr ? 0U : state_->current_position();
}

void ReferenceRunner::release() noexcept {
  if (stream_ != nullptr) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    (void)cudaStreamSynchronize(stream);
    (void)cudaStreamDestroy(stream);
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
  pinned_logits_ = nullptr;
  pinned_trace_ = nullptr;
  views_ = {};
  projection_backend_ = ProjectionBackend::kReference;
  trace_enabled_ = false;
  trace_valid_ = false;
  poisoned_ = false;
  trace_position_ = 0U;
  trace_input_token_ = 0U;
}

ReferenceStepOutcome ReferenceRunner::fail_step(
    const ReferenceRunnerStatus status) noexcept {
  // A failed launch may follow earlier successful launches in this token.
  // Drain the owned stream before returning so every step has a synchronous
  // completion boundary even though its mutated device state is not committed
  // and cannot be reused until reset.
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  poisoned_ = true;
  trace_valid_ = false;
  ReferenceStepOutcome outcome;
  outcome.status = status;
  return outcome;
}

ReferencePrefillTileOutcome ReferenceRunner::fail_prefill_tile(
    const ReferenceRunnerStatus status) noexcept {
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  poisoned_ = true;
  trace_valid_ = false;
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
  trace_position_ = 0U;
  trace_input_token_ = 0U;
  return {};
}

ReferenceStepOutcome ReferenceRunner::step(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) noexcept {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }
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

  const std::uint32_t position = state_->current_position();
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
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

  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& layer_weights = weights_->layer(layer);
    if (!check_cuda(launch_centered_rms_norm_reference_cuda(
                        views_.hidden[0], layer_weights.input_layernorm.data,
                        kReferenceHiddenSize, kRmsEpsilon, views_.hidden[1],
                        stream_),
                    "input_layernorm", layer)) {
      return fail_step(launch_failure);
    }

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
      if (!project(attention->in_proj_qkv, views_.hidden[1],
                   views_.projection[0], "linear_qkv_projection", layer) ||
          !project(attention->in_proj_z, views_.hidden[1],
                   views_.projection[1], "linear_z_projection", layer) ||
          !project(attention->in_proj_a, views_.hidden[1], views_.linear_a,
                   "linear_a_projection", layer) ||
          !project(attention->in_proj_b, views_.hidden[1], views_.linear_b,
                   "linear_b_projection", layer)) {
        return fail_step(launch_failure);
      }
      if (!check_cuda(launch_causal_conv1d_silu_update_reference_cuda(
                          views_.projection[0], attention->conv1d.data,
                          views_.conv_state[layer], views_.projection[0], {},
                          stream_),
                      "linear_causal_conv", layer) ||
          !check_cuda(launch_gated_delta_net_update_reference_cuda(
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
                  kGdnHeadDimension, kRmsEpsilon, views_.projection[2],
                  stream_),
              "linear_output_norm_gate", layer) ||
          !project(attention->out_proj, views_.projection[2],
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
      if (!project(attention->q_proj, views_.hidden[1], views_.projection[0],
                   "full_q_gate_projection", layer) ||
          !project(attention->k_proj, views_.hidden[1], views_.projection[1],
                   "full_k_projection", layer) ||
          !project(attention->v_proj, views_.hidden[1], views_.projection[2],
                   "full_v_projection", layer) ||
          !check_cuda(launch_split_interleaved_q_gate_reference_cuda(
                          views_.projection[0], kFullQueryHeads,
                          kFullHeadDimension, views_.projection[3],
                          views_.projection[3] + kFullQueryElements, stream_),
                      "full_split_q_gate", layer) ||
          !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                          views_.projection[3], attention->q_norm.data,
                          kFullQueryHeads, kFullHeadDimension, kRmsEpsilon,
                          views_.projection[0], stream_),
                      "full_q_norm", layer) ||
          !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                          views_.projection[1], attention->k_norm.data,
                          kFullKvHeads, kFullHeadDimension, kRmsEpsilon,
                          views_.projection[1], stream_),
                      "full_k_norm", layer)) {
        return fail_step(launch_failure);
      }

      const float* const cosines =
          views_.rope_cos + static_cast<std::size_t>(position) * kRopePairs;
      const float* const sines =
          views_.rope_sin + static_cast<std::size_t>(position) * kRopePairs;
      if (!check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                          views_.projection[0], cosines, sines,
                          kFullQueryHeads, views_.projection[0], stream_),
                      "full_q_rope", layer) ||
          !check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                          views_.projection[1], cosines, sines, kFullKvHeads,
                          views_.projection[1], stream_),
                      "full_k_rope", layer)) {
        return fail_step(launch_failure);
      }

      std::uint16_t* const current_key =
          views_.key_cache[layer] +
          static_cast<std::size_t>(position) * kFullKvElements;
      std::uint16_t* const current_value =
          views_.value_cache[layer] +
          static_cast<std::size_t>(position) * kFullKvElements;
      if (!check_cuda(
              static_cast<int>(cudaMemcpyAsync(
                  current_key, views_.projection[1],
                  kFullKvElements * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToDevice, stream)),
              "full_key_cache_write", layer) ||
          !check_cuda(
              static_cast<int>(cudaMemcpyAsync(
                  current_value, views_.projection[2],
                  kFullKvElements * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToDevice, stream)),
              "full_value_cache_write", layer) ||
          !check_cuda(launch_gqa_attention_reference_cuda(
                          views_.projection[0], views_.key_cache[layer],
                          views_.value_cache[layer], kFullQueryHeads,
                          kFullKvHeads,
                          static_cast<std::size_t>(position) + 1U,
                          kFullHeadDimension, kAttentionScale,
                          views_.fp32_scratch,
                          views_.fp32_scratch_elements, views_.projection[1],
                          stream_),
                      "full_gqa", layer) ||
          !check_cuda(launch_sigmoid_gate_reference_cuda(
                          views_.projection[1],
                          views_.projection[3] + kFullQueryElements,
                          kFullQueryElements, views_.projection[1], stream_),
                      "full_output_gate", layer) ||
          !project(attention->o_proj, views_.projection[1], views_.hidden[1],
                   "full_output_projection", layer)) {
        return fail_step(launch_failure);
      }
    } else {
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidLayerSchedule, "layer_schedule",
          layer));
    }

    if (!check_cuda(launch_residual_add_reference_cuda(
                        views_.hidden[0], views_.hidden[1],
                        kReferenceHiddenSize, views_.hidden[2], stream_),
                    "attention_residual", layer)) {
      return fail_step(launch_failure);
    }
    const std::size_t trace_base =
        kReferenceHiddenSize + 2U * layer * kReferenceHiddenSize;
    if (options.capture_trace &&
        !copy_trace(views_.hidden[2], trace_base + kReferenceHiddenSize,
                    "trace_layer_residual", layer)) {
      return fail_step(launch_failure);
    }

    if (!check_cuda(launch_centered_rms_norm_reference_cuda(
                        views_.hidden[2],
                        layer_weights.post_attention_layernorm.data,
                        kReferenceHiddenSize, kRmsEpsilon, views_.hidden[1],
                        stream_),
                    "post_attention_layernorm", layer) ||
        !project(layer_weights.mlp.gate_proj, views_.hidden[1],
                 views_.projection[0], "mlp_gate_projection", layer) ||
        !project(layer_weights.mlp.up_proj, views_.hidden[1],
                 views_.projection[1], "mlp_up_projection", layer) ||
        !check_cuda(launch_silu_mul_reference_cuda(
                        views_.projection[0], views_.projection[1],
                        kReferenceIntermediateSize, views_.projection[0],
                        stream_),
                    "mlp_silu_mul", layer) ||
        !project(layer_weights.mlp.down_proj, views_.projection[0],
                 views_.hidden[1], "mlp_down_projection", layer)) {
      return fail_step(launch_failure);
    }
    if (options.capture_trace &&
        !copy_trace(views_.hidden[1], trace_base, "trace_layer_hidden",
                    layer)) {
      return fail_step(launch_failure);
    }
    if (!check_cuda(launch_residual_add_reference_cuda(
                        views_.hidden[2], views_.hidden[1],
                        kReferenceHiddenSize, views_.hidden[0], stream_),
                    "layer_residual", layer)) {
      return fail_step(launch_failure);
    }
  }

  if (!check_cuda(launch_centered_rms_norm_reference_cuda(
                      views_.hidden[0], weights_->final_norm().data,
                      kReferenceHiddenSize, kRmsEpsilon, views_.hidden[1],
                      stream_),
                  "final_norm", kReferenceNoLayer)) {
    return fail_step(launch_failure);
  }
  if (options.capture_trace &&
      !copy_trace(views_.hidden[1],
                  (1U + 2U * kReferenceDecoderLayerCount) *
                      kReferenceHiddenSize,
                  "trace_final_norm", kReferenceNoLayer)) {
    return fail_step(launch_failure);
  }

  if (options.compute_logits) {
    const bool use_sm87_bf16_logits =
        projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
        linear_weight_kind(weights_->lm_head()) != LinearWeightKind::kBf16;
    if (use_sm87_bf16_logits) {
      auto* const device_bf16_logits =
          reinterpret_cast<std::uint16_t*>(views_.fp32_scratch);
      if (!check_cuda(launch_projection_to_bf16_cuda(
                          projection_backend_, weights_->lm_head(),
                          views_.hidden[1], nullptr, 0U,
                          device_bf16_logits, stream_),
                      "lm_head_sm87_bf16", kReferenceNoLayer) ||
          !check_cuda(
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
    const bool used_sm87_bf16_logits =
        projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
        linear_weight_kind(weights_->lm_head()) != LinearWeightKind::kBf16;
    const reference_runner_detail::LogitsAnalysis analysis =
        used_sm87_bf16_logits
            ? reference_runner_detail::analyze_bf16_logits_bits(
                  static_cast<const std::uint16_t*>(pinned_logits_),
                  kReferenceVocabularySize)
            : reference_runner_detail::analyze_bf16_logits_in_place(
                  static_cast<float*>(pinned_logits_),
                  kReferenceVocabularySize);
    if (!analysis.ok()) {
      return fail_step(runner_status(
          ReferenceRunnerError::kNonFiniteLogits, "bf16_logits_analysis"));
    }
    ReferenceStepLogits logits;
    logits.predicted_token_id =
        static_cast<std::uint32_t>(analysis.predicted_index);
    logits.chosen_logit = analysis.maximum;
    logits.max_log_probability = analysis.max_log_probability;
    logits.logsumexp = analysis.logsumexp;
    result.logits.emplace(logits);
  }

  const RequestOperationStatus commit_status = state_->commit_token();
  if (!commit_status) {
    return fail_step(runner_status(
        ReferenceRunnerError::kStateCommitFailure, "commit_token",
        kReferenceNoLayer, commit_status.cuda_error));
  }
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
    ReferencePrefillTileOutcome outcome;
    outcome.value.emplace(std::move(tile));
    return outcome;
  }

  const std::uint32_t first_position = state_->current_position();
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
  const auto project = [this, token_count, &check_cuda](
                           const LinearWeight& weight,
                           const std::uint16_t* const input,
                           std::uint16_t* const output,
                           const char* const operation,
                           const std::size_t layer) noexcept {
    return check_cuda(launch_projection_tile_to_bf16_cuda(
                          projection_backend_, weight, input, token_count,
                          views_.fp32_scratch,
                          views_.fp32_scratch_elements, output, stream_),
                      operation, layer);
  };

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
    if (!check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
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
      if (!project(attention->in_proj_qkv, views_.hidden[1],
                   views_.projection[0], "prefill_linear_qkv_projection",
                   layer) ||
          !project(attention->in_proj_z, views_.hidden[1],
                   views_.projection[1], "prefill_linear_z_projection",
                   layer) ||
          !project(attention->in_proj_a, views_.hidden[1], views_.linear_a,
                   "prefill_linear_a_projection", layer) ||
          !project(attention->in_proj_b, views_.hidden[1], views_.linear_b,
                   "prefill_linear_b_projection", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      if (!check_cuda(
              launch_causal_conv1d_silu_update_tile_reference_cuda(
                  views_.projection[0], token_count, attention->conv1d.data,
                  views_.conv_state[layer], views_.projection[0], {},
                  stream_),
              "prefill_linear_causal_conv", layer) ||
          !check_cuda(launch_gated_delta_net_update_tile_reference_cuda(
                          views_.projection[0], token_count, views_.linear_a,
                          views_.linear_b, attention->a_log.data,
                          attention->dt_bias.data, views_.gdn_state[layer],
                          views_.gdn_state[layer], kRmsEpsilon,
                          views_.projection[2], {}, stream_),
                      "prefill_linear_gdn", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      if (!check_cuda(
              launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                  views_.projection[2], attention->norm.data,
                  views_.projection[1], token_count * kGdnValueHeadCount,
                  kGdnHeadDimension, kRmsEpsilon, views_.projection[2],
                  stream_),
              "prefill_linear_output_norm_gate", layer) ||
          !project(attention->out_proj, views_.projection[2],
                   views_.hidden[1], "prefill_linear_output_projection",
                   layer)) {
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
      if (!project(attention->q_proj, views_.hidden[1],
                   views_.projection[0], "prefill_full_q_gate_projection",
                   layer) ||
          !project(attention->k_proj, views_.hidden[1],
                   views_.projection[1], "prefill_full_k_projection",
                   layer) ||
          !project(attention->v_proj, views_.hidden[1],
                   views_.projection[2], "prefill_full_v_projection",
                   layer) ||
          !check_cuda(launch_split_interleaved_q_gate_reference_cuda(
                          views_.projection[0], token_count * kFullQueryHeads,
                          kFullHeadDimension, views_.projection[3],
                          packed_gates, stream_),
                      "prefill_full_split_q_gate", layer) ||
          !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                          views_.projection[3], attention->q_norm.data,
                          token_count * kFullQueryHeads, kFullHeadDimension,
                          kRmsEpsilon, views_.projection[0], stream_),
                      "prefill_full_q_norm", layer) ||
          !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                          views_.projection[1], attention->k_norm.data,
                          token_count * kFullKvHeads, kFullHeadDimension,
                          kRmsEpsilon, views_.projection[1], stream_),
                      "prefill_full_k_norm", layer)) {
        return fail_prefill_tile(launch_failure);
      }

      for (std::size_t token = 0U; token < token_count; ++token) {
        const std::size_t position =
            static_cast<std::size_t>(first_position) + token;
        const float* const cosines = views_.rope_cos + position * kRopePairs;
        const float* const sines = views_.rope_sin + position * kRopePairs;
        if (!check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                            views_.projection[0] +
                                token * kFullQueryElements,
                            cosines, sines, kFullQueryHeads,
                            views_.projection[0] +
                                token * kFullQueryElements,
                            stream_),
                        "prefill_full_q_rope", layer) ||
            !check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                            views_.projection[1] + token * kFullKvElements,
                            cosines, sines, kFullKvHeads,
                            views_.projection[1] + token * kFullKvElements,
                            stream_),
                        "prefill_full_k_rope", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      }
      // Populate the complete tile's cache before GQA overwrites projection[1]
      // with query-sized outputs. Each query still uses only its causal prefix.
      for (std::size_t token = 0U; token < token_count; ++token) {
        const std::size_t position =
            static_cast<std::size_t>(first_position) + token;
        std::uint16_t* const current_key =
            views_.key_cache[layer] + position * kFullKvElements;
        std::uint16_t* const current_value =
            views_.value_cache[layer] + position * kFullKvElements;
        if (!check_cuda(
                static_cast<int>(cudaMemcpyAsync(
                    current_key,
                    views_.projection[1] + token * kFullKvElements,
                    kFullKvElements * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToDevice, stream)),
                "prefill_full_key_cache_write", layer) ||
            !check_cuda(
                static_cast<int>(cudaMemcpyAsync(
                    current_value,
                    views_.projection[2] + token * kFullKvElements,
                    kFullKvElements * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToDevice, stream)),
                "prefill_full_value_cache_write", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      }
      for (std::size_t token = 0U; token < token_count; ++token) {
        const std::size_t sequence_length =
            static_cast<std::size_t>(first_position) + token + 1U;
        if (!check_cuda(launch_gqa_attention_reference_cuda(
                            views_.projection[0] +
                                token * kFullQueryElements,
                            views_.key_cache[layer], views_.value_cache[layer],
                            kFullQueryHeads, kFullKvHeads, sequence_length,
                            kFullHeadDimension, kAttentionScale,
                            views_.fp32_scratch,
                            views_.fp32_scratch_elements,
                            views_.projection[1] +
                                token * kFullQueryElements,
                            stream_),
                        "prefill_full_gqa", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      }
      if (!check_cuda(launch_sigmoid_gate_reference_cuda(
                          views_.projection[1], packed_gates,
                          token_count * kFullQueryElements,
                          views_.projection[1], stream_),
                      "prefill_full_output_gate", layer) ||
          !project(attention->o_proj, views_.projection[1],
                   views_.hidden[1], "prefill_full_output_projection",
                   layer)) {
        return fail_prefill_tile(launch_failure);
      }
    } else {
      return fail_prefill_tile(runner_status(
          ReferenceRunnerError::kInvalidLayerSchedule,
          "prefill_layer_schedule", layer));
    }

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
    if (!project(layer_weights.mlp.gate_proj, views_.hidden[1],
                 views_.projection[0], "prefill_mlp_gate_projection", layer) ||
        !project(layer_weights.mlp.up_proj, views_.hidden[1],
                 views_.projection[1], "prefill_mlp_up_projection", layer) ||
        !check_cuda(launch_silu_mul_reference_cuda(
                        views_.projection[0], views_.projection[1],
                        token_count * kReferenceIntermediateSize,
                        views_.projection[0], stream_),
                    "prefill_mlp_silu_mul", layer) ||
        !project(layer_weights.mlp.down_proj, views_.projection[0],
                 views_.hidden[1], "prefill_mlp_down_projection", layer) ||
        !check_cuda(launch_residual_add_reference_cuda(
                        views_.hidden[2], views_.hidden[1],
                        token_count * kReferenceHiddenSize,
                        views_.hidden[0], stream_),
                    "prefill_layer_residual", layer)) {
      return fail_prefill_tile(launch_failure);
    }
  }

  // Match the non-logit step boundary even though this output is not consumed
  // by the following layer-major tile or by persistent state.
  if (!check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
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
  ReferencePrefillTileOutcome outcome;
  outcome.value.emplace(std::move(tile));
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
