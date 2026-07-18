#include "q3x/runtime/gdn_decode.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace q3x::runtime {
namespace {

static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "GDN references require IEEE 754 binary32 float");

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] bool multiply_overflows(const std::size_t left,
                                      const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
}

[[nodiscard]] GdnStatus validate_dimensions(
    const GdnDimensions dimensions) noexcept {
  if (multiply_overflows(dimensions.qk_head_count,
                         dimensions.head_dimension) ||
      multiply_overflows(dimensions.value_head_count,
                         dimensions.head_dimension)) {
    return GdnStatus::kSizeOverflow;
  }
  const std::size_t q_elements =
      dimensions.qk_head_count * dimensions.head_dimension;
  const std::size_t v_elements =
      dimensions.value_head_count * dimensions.head_dimension;
  if (q_elements >
          (std::numeric_limits<std::size_t>::max() - v_elements) / 2U ||
      multiply_overflows(v_elements, dimensions.head_dimension)) {
    return GdnStatus::kSizeOverflow;
  }
  if (dimensions.qk_head_count != kGdnQkHeadCount ||
      dimensions.value_head_count != kGdnValueHeadCount ||
      dimensions.head_dimension != kGdnHeadDimension) {
    return GdnStatus::kInvalidDimension;
  }
  return GdnStatus::kSuccess;
}

[[nodiscard]] float stable_softplus(const float value) noexcept {
  constexpr float kThreshold = 20.0F;
  return value > kThreshold ? value : std::log1p(std::exp(value));
}

[[nodiscard]] float stable_sigmoid(const float value) noexcept {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

[[nodiscard]] float silu(const float value) noexcept {
  return value / (1.0F + std::exp(-value));
}

[[nodiscard]] bool invalid_conv_alias(
    const std::uint16_t* const raw_qkv,
    const std::uint16_t* const conv_weight,
    const std::uint16_t* const history,
    const std::uint16_t* const output) noexcept {
  return raw_qkv == conv_weight || raw_qkv == history ||
         conv_weight == history || conv_weight == output ||
         history == output;
}

[[nodiscard]] bool invalid_gdn_alias(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    const std::uint16_t* const state_output,
    const std::uint16_t* const output) noexcept {
  const bool output_alias =
      output == conv_qkv || output == a || output == b || output == A_log ||
      output == dt_bias || output == state_input || output == state_output;
  const bool state_input_alias =
      state_input == conv_qkv || state_input == a || state_input == b ||
      state_input == A_log || state_input == dt_bias;
  const bool state_output_alias =
      state_output == conv_qkv || state_output == a || state_output == b ||
      state_output == A_log || state_output == dt_bias;
  return output_alias || state_input_alias || state_output_alias;
}

}  // namespace

const char* gdn_status_string(const GdnStatus status) noexcept {
  switch (status) {
    case GdnStatus::kSuccess:
      return "success";
    case GdnStatus::kInvalidArgument:
      return "invalid argument";
    case GdnStatus::kInvalidDimension:
      return "dimensions do not match Qwen3.6-27B GDN";
    case GdnStatus::kSizeOverflow:
      return "GDN tensor element count overflows size_t";
    case GdnStatus::kInvalidAlias:
      return "unsupported exact buffer alias";
  }
  return "unknown GDN status";
}

GdnStatus causal_conv1d_silu_update_reference_cpu(
    const std::uint16_t* const raw_qkv,
    const std::uint16_t* const conv_weight,
    std::uint16_t* const history_in_out,
    std::uint16_t* const conv_qkv_output,
    const GdnDimensions dimensions) noexcept {
  const GdnStatus dimension_status = validate_dimensions(dimensions);
  if (dimension_status != GdnStatus::kSuccess) {
    return dimension_status;
  }
  if (raw_qkv == nullptr || conv_weight == nullptr ||
      history_in_out == nullptr || conv_qkv_output == nullptr) {
    return GdnStatus::kInvalidArgument;
  }
  if (invalid_conv_alias(raw_qkv, conv_weight, history_in_out,
                         conv_qkv_output)) {
    return GdnStatus::kInvalidAlias;
  }

  for (std::size_t channel = 0; channel < kGdnQkvChannels; ++channel) {
    const std::size_t history_offset = channel * kGdnConvHistoryWidth;
    const std::size_t weight_offset = channel * kGdnConvKernelWidth;
    const std::uint16_t current_bits = raw_qkv[channel];
    float convolution = 0.0F;
    convolution = std::fma(decode_bf16(history_in_out[history_offset]),
                           decode_bf16(conv_weight[weight_offset]),
                           convolution);
    convolution = std::fma(decode_bf16(history_in_out[history_offset + 1U]),
                           decode_bf16(conv_weight[weight_offset + 1U]),
                           convolution);
    convolution = std::fma(decode_bf16(history_in_out[history_offset + 2U]),
                           decode_bf16(conv_weight[weight_offset + 2U]),
                           convolution);
    convolution = std::fma(decode_bf16(current_bits),
                           decode_bf16(conv_weight[weight_offset + 3U]),
                           convolution);
    conv_qkv_output[channel] = encode_bf16(silu(convolution));
    history_in_out[history_offset] = history_in_out[history_offset + 1U];
    history_in_out[history_offset + 1U] =
        history_in_out[history_offset + 2U];
    history_in_out[history_offset + 2U] = current_bits;
  }
  return GdnStatus::kSuccess;
}

GdnStatus gated_delta_net_update_reference_cpu(
    const std::uint16_t* const conv_qkv,
    const std::uint16_t* const a,
    const std::uint16_t* const b,
    const std::uint16_t* const A_log,
    const std::uint16_t* const dt_bias,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const float l2_epsilon,
    std::uint16_t* const output,
    const GdnDimensions dimensions) noexcept {
  const GdnStatus dimension_status = validate_dimensions(dimensions);
  if (dimension_status != GdnStatus::kSuccess) {
    return dimension_status;
  }
  if (!std::isfinite(l2_epsilon) || l2_epsilon <= 0.0F) {
    return GdnStatus::kInvalidArgument;
  }
  if (conv_qkv == nullptr || a == nullptr || b == nullptr || A_log == nullptr ||
      dt_bias == nullptr || state_input == nullptr || state_output == nullptr ||
      output == nullptr) {
    return GdnStatus::kInvalidArgument;
  }
  if (invalid_gdn_alias(conv_qkv, a, b, A_log, dt_bias, state_input,
                        state_output, output)) {
    return GdnStatus::kInvalidAlias;
  }

  constexpr std::size_t kQOffset = 0U;
  constexpr std::size_t kKOffset = kGdnQElements;
  constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;
  const float inverse_sqrt_dimension =
      1.0F / std::sqrt(static_cast<float>(kGdnHeadDimension));

  std::array<float, kGdnHeadDimension> normalized_q{};
  std::array<float, kGdnHeadDimension> normalized_k{};
  for (std::size_t value_head = 0; value_head < kGdnValueHeadCount;
       ++value_head) {
    const std::size_t qk_head = value_head / 3U;
    const std::size_t q_offset =
        kQOffset + qk_head * kGdnHeadDimension;
    const std::size_t k_offset =
        kKOffset + qk_head * kGdnHeadDimension;
    float q_sum_of_squares = 0.0F;
    float k_sum_of_squares = 0.0F;
    for (std::size_t dimension = 0; dimension < kGdnHeadDimension;
         ++dimension) {
      const float q_value = decode_bf16(conv_qkv[q_offset + dimension]);
      const float k_value = decode_bf16(conv_qkv[k_offset + dimension]);
      q_sum_of_squares = std::fma(q_value, q_value, q_sum_of_squares);
      k_sum_of_squares = std::fma(k_value, k_value, k_sum_of_squares);
    }
    const float q_scale =
        1.0F / std::sqrt(q_sum_of_squares + l2_epsilon) *
        inverse_sqrt_dimension;
    const float k_scale =
        1.0F / std::sqrt(k_sum_of_squares + l2_epsilon);
    for (std::size_t dimension = 0; dimension < kGdnHeadDimension;
         ++dimension) {
      normalized_q[dimension] =
          decode_bf16(conv_qkv[q_offset + dimension]) * q_scale;
      normalized_k[dimension] =
          decode_bf16(conv_qkv[k_offset + dimension]) * k_scale;
    }

    const float gate_input =
        decode_bf16(a[value_head]) + decode_bf16(dt_bias[value_head]);
    const float g = -std::exp(decode_bf16(A_log[value_head])) *
                    stable_softplus(gate_input);
    const float alpha = std::exp(g);
    const float beta = stable_sigmoid(decode_bf16(b[value_head]));
    const std::size_t value_offset =
        kVOffset + value_head * kGdnHeadDimension;
    const std::size_t state_head_offset =
        value_head * kGdnHeadDimension * kGdnHeadDimension;
    const std::size_t output_offset = value_head * kGdnHeadDimension;

    for (std::size_t value_dimension = 0;
         value_dimension < kGdnHeadDimension; ++value_dimension) {
      const std::size_t state_row_offset =
          state_head_offset + value_dimension * kGdnHeadDimension;
      float prediction = 0.0F;
      for (std::size_t key_dimension = 0;
           key_dimension < kGdnHeadDimension; ++key_dimension) {
        const float decayed_state =
            alpha * decode_bf16(state_input[state_row_offset + key_dimension]);
        prediction = std::fma(decayed_state, normalized_k[key_dimension],
                              prediction);
      }
      const float delta =
          (decode_bf16(conv_qkv[value_offset + value_dimension]) - prediction) *
          beta;
      float result = 0.0F;
      for (std::size_t key_dimension = 0;
           key_dimension < kGdnHeadDimension; ++key_dimension) {
        const std::size_t state_index = state_row_offset + key_dimension;
        const float decayed_state =
            alpha * decode_bf16(state_input[state_index]);
        const float updated_state =
            std::fma(delta, normalized_k[key_dimension], decayed_state);
        state_output[state_index] = encode_bf16(updated_state);
        result = std::fma(updated_state, normalized_q[key_dimension], result);
      }
      output[output_offset + value_dimension] = encode_bf16(result);
    }
  }
  return GdnStatus::kSuccess;
}

}  // namespace q3x::runtime
