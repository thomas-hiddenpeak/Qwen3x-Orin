#include "q3x/runtime/decode_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace q3x::runtime {
namespace {

static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "decode references require IEEE 754 binary32 float");

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fffffffU;
  if (magnitude > 0x7f800000U) {
    // Preserve the sign/top payload and force a quiet BF16 NaN rather than
    // allowing a tiny FP32 payload to round into infinity.
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

[[nodiscard]] bool product3_overflows(const std::size_t first,
                                      const std::size_t second,
                                      const std::size_t third) noexcept {
  return multiply_overflows(first, second) ||
         multiply_overflows(first * second, third);
}

[[nodiscard]] bool valid_epsilon(const float epsilon) noexcept {
  return std::isfinite(epsilon) && epsilon > 0.0F;
}

template <bool kCentered>
[[nodiscard]] DecodeOpStatus rms_norm(
    const std::uint16_t* const input,
    const std::uint16_t* const weight,
    const std::size_t hidden_size,
    const float epsilon,
    std::uint16_t* const output) noexcept {
  if (!valid_epsilon(epsilon)) {
    return DecodeOpStatus::kInvalidArgument;
  }
  if (hidden_size == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (input == nullptr || weight == nullptr || output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }

  float sum_of_squares = 0.0F;
  for (std::size_t index = 0; index < hidden_size; ++index) {
    const float value = decode_bf16(input[index]);
    sum_of_squares += value * value;
  }
  const float inverse_rms =
      1.0F / std::sqrt(sum_of_squares / static_cast<float>(hidden_size) +
                       epsilon);
  for (std::size_t index = 0; index < hidden_size; ++index) {
    const float gamma = decode_bf16(weight[index]) +
                        (kCentered ? 1.0F : 0.0F);
    output[index] = encode_bf16(decode_bf16(input[index]) * inverse_rms *
                                gamma);
  }
  return DecodeOpStatus::kSuccess;
}

template <bool kCentered, bool kApplySiluGate>
[[nodiscard]] DecodeOpStatus headwise_rms_norm(
    const std::uint16_t* const input,
    const std::uint16_t* const shared_weight,
    const std::uint16_t* const gate,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output) noexcept {
  if (!valid_epsilon(epsilon)) {
    return DecodeOpStatus::kInvalidArgument;
  }
  if (multiply_overflows(head_count, head_dimension)) {
    return DecodeOpStatus::kSizeOverflow;
  }
  if (head_count == 0U || head_dimension == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (input == nullptr || shared_weight == nullptr || output == nullptr ||
      (kApplySiluGate && gate == nullptr)) {
    return DecodeOpStatus::kInvalidArgument;
  }

  for (std::size_t head = 0; head < head_count; ++head) {
    const std::size_t offset = head * head_dimension;
    float sum_of_squares = 0.0F;
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      const float value = decode_bf16(input[offset + dimension]);
      sum_of_squares += value * value;
    }
    const float inverse_rms =
        1.0F /
        std::sqrt(sum_of_squares / static_cast<float>(head_dimension) +
                  epsilon);
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      const float gamma = decode_bf16(shared_weight[dimension]) +
                          (kCentered ? 1.0F : 0.0F);
      float value = decode_bf16(input[offset + dimension]) * inverse_rms *
                    gamma;
      if constexpr (kApplySiluGate) {
        const float gate_value = decode_bf16(gate[offset + dimension]);
        value *= gate_value / (1.0F + std::exp(-gate_value));
      }
      output[offset + dimension] = encode_bf16(value);
    }
  }
  return DecodeOpStatus::kSuccess;
}

[[nodiscard]] DecodeOpStatus validate_attention(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    const float attention_scale,
    float* const probabilities_scratch,
    const std::size_t scratch_elements,
    std::uint16_t* const output) noexcept {
  if (query_head_count == 0U || kv_head_count == 0U ||
      sequence_length == 0U || head_dimension == 0U ||
      query_head_count % kv_head_count != 0U) {
    return DecodeOpStatus::kInvalidDimension;
  }
  if (!std::isfinite(attention_scale) || attention_scale < 0.0F) {
    return DecodeOpStatus::kInvalidArgument;
  }
  if (multiply_overflows(query_head_count, head_dimension) ||
      product3_overflows(sequence_length, kv_head_count, head_dimension) ||
      multiply_overflows(query_head_count, sequence_length)) {
    return DecodeOpStatus::kSizeOverflow;
  }
  const std::size_t required_scratch = query_head_count * sequence_length;
  if (scratch_elements < required_scratch) {
    return DecodeOpStatus::kInsufficientScratch;
  }
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      probabilities_scratch == nullptr || output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }
  return DecodeOpStatus::kSuccess;
}

}  // namespace

const char* decode_op_status_string(const DecodeOpStatus status) noexcept {
  switch (status) {
    case DecodeOpStatus::kSuccess:
      return "success";
    case DecodeOpStatus::kInvalidArgument:
      return "invalid argument";
    case DecodeOpStatus::kInvalidDimension:
      return "invalid dimension relationship";
    case DecodeOpStatus::kSizeOverflow:
      return "tensor element count overflows size_t";
    case DecodeOpStatus::kTokenOutOfRange:
      return "token id is outside the embedding vocabulary";
    case DecodeOpStatus::kInsufficientScratch:
      return "attention probability scratch is too small";
  }
  return "unknown decode-op status";
}

DecodeOpStatus embedding_gather_reference_cpu(
    const std::uint16_t* const embedding_table,
    const std::size_t vocabulary_size,
    const std::size_t hidden_size,
    const std::size_t token_id,
    std::uint16_t* const output) noexcept {
  if (multiply_overflows(vocabulary_size, hidden_size)) {
    return DecodeOpStatus::kSizeOverflow;
  }
  if (token_id >= vocabulary_size) {
    return DecodeOpStatus::kTokenOutOfRange;
  }
  if (hidden_size == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (embedding_table == nullptr || output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }
  const std::size_t offset = token_id * hidden_size;
  std::copy_n(embedding_table + offset, hidden_size, output);
  return DecodeOpStatus::kSuccess;
}

DecodeOpStatus centered_rms_norm_reference_cpu(
    const std::uint16_t* const input,
    const std::uint16_t* const weight,
    const std::size_t hidden_size,
    const float epsilon,
    std::uint16_t* const output) noexcept {
  return rms_norm<true>(input, weight, hidden_size, epsilon, output);
}

DecodeOpStatus plain_rms_norm_reference_cpu(
    const std::uint16_t* const input,
    const std::uint16_t* const weight,
    const std::size_t hidden_size,
    const float epsilon,
    std::uint16_t* const output) noexcept {
  return rms_norm<false>(input, weight, hidden_size, epsilon, output);
}

DecodeOpStatus headwise_centered_rms_norm_reference_cpu(
    const std::uint16_t* const input,
    const std::uint16_t* const shared_weight,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output) noexcept {
  return headwise_rms_norm<true, false>(
      input, shared_weight, nullptr, head_count, head_dimension, epsilon,
      output);
}

DecodeOpStatus headwise_plain_rms_norm_reference_cpu(
    const std::uint16_t* const input,
    const std::uint16_t* const shared_weight,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output) noexcept {
  return headwise_rms_norm<false, false>(
      input, shared_weight, nullptr, head_count, head_dimension, epsilon,
      output);
}

DecodeOpStatus headwise_plain_rms_norm_silu_gate_reference_cpu(
    const std::uint16_t* const input,
    const std::uint16_t* const shared_weight,
    const std::uint16_t* const gate,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output) noexcept {
  return headwise_rms_norm<false, true>(
      input, shared_weight, gate, head_count, head_dimension, epsilon, output);
}

DecodeOpStatus residual_add_reference_cpu(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::size_t element_count,
    std::uint16_t* const output) noexcept {
  if (element_count == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (left == nullptr || right == nullptr || output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }
  for (std::size_t index = 0; index < element_count; ++index) {
    output[index] =
        encode_bf16(decode_bf16(left[index]) + decode_bf16(right[index]));
  }
  return DecodeOpStatus::kSuccess;
}

DecodeOpStatus fp32_to_bf16_reference_cpu(
    const float* const input,
    const std::size_t element_count,
    std::uint16_t* const output) noexcept {
  if (element_count == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (input == nullptr || output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }
  for (std::size_t index = 0; index < element_count; ++index) {
    output[index] = encode_bf16(input[index]);
  }
  return DecodeOpStatus::kSuccess;
}

DecodeOpStatus silu_mul_reference_cpu(
    const std::uint16_t* const gate,
    const std::uint16_t* const up,
    const std::size_t element_count,
    std::uint16_t* const output) noexcept {
  if (element_count == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (gate == nullptr || up == nullptr || output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }
  for (std::size_t index = 0; index < element_count; ++index) {
    const float gate_value = decode_bf16(gate[index]);
    const float silu = gate_value / (1.0F + std::exp(-gate_value));
    output[index] = encode_bf16(silu * decode_bf16(up[index]));
  }
  return DecodeOpStatus::kSuccess;
}

DecodeOpStatus sigmoid_gate_reference_cpu(
    const std::uint16_t* const value,
    const std::uint16_t* const gate,
    const std::size_t element_count,
    std::uint16_t* const output) noexcept {
  if (element_count == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (value == nullptr || gate == nullptr || output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }
  for (std::size_t index = 0; index < element_count; ++index) {
    const float gate_value = decode_bf16(gate[index]);
    const float sigmoid =
        gate_value >= 0.0F
            ? 1.0F / (1.0F + std::exp(-gate_value))
            : std::exp(gate_value) / (1.0F + std::exp(gate_value));
    output[index] =
        encode_bf16(decode_bf16(value[index]) * sigmoid);
  }
  return DecodeOpStatus::kSuccess;
}

DecodeOpStatus l2_normalize_heads_reference_cpu(
    const std::uint16_t* const input,
    const std::size_t head_count,
    const std::size_t head_dimension,
    const float epsilon,
    std::uint16_t* const output) noexcept {
  if (!valid_epsilon(epsilon)) {
    return DecodeOpStatus::kInvalidArgument;
  }
  if (multiply_overflows(head_count, head_dimension)) {
    return DecodeOpStatus::kSizeOverflow;
  }
  if (head_count == 0U || head_dimension == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (input == nullptr || output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }
  for (std::size_t head = 0; head < head_count; ++head) {
    const std::size_t offset = head * head_dimension;
    float sum_of_squares = 0.0F;
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      const float value = decode_bf16(input[offset + dimension]);
      sum_of_squares += value * value;
    }
    const float inverse_norm = 1.0F / std::sqrt(sum_of_squares + epsilon);
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      output[offset + dimension] = encode_bf16(
          decode_bf16(input[offset + dimension]) * inverse_norm);
    }
  }
  return DecodeOpStatus::kSuccess;
}

DecodeOpStatus partial_neox_rope_256_64_reference_cpu(
    const std::uint16_t* const input,
    const float* const cosines,
    const float* const sines,
    const std::size_t head_count,
    std::uint16_t* const output) noexcept {
  if (multiply_overflows(head_count, kFullAttentionHeadDimension)) {
    return DecodeOpStatus::kSizeOverflow;
  }
  if (head_count == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (input == nullptr || cosines == nullptr || sines == nullptr ||
      output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }
  constexpr std::size_t kHalfRotary = kQwenRotaryDimension / 2U;
  for (std::size_t head = 0; head < head_count; ++head) {
    const std::size_t offset = head * kFullAttentionHeadDimension;
    for (std::size_t dimension = 0; dimension < kHalfRotary; ++dimension) {
      const float first = decode_bf16(input[offset + dimension]);
      const float second =
          decode_bf16(input[offset + dimension + kHalfRotary]);
      output[offset + dimension] = encode_bf16(
          first * cosines[dimension] - second * sines[dimension]);
      output[offset + dimension + kHalfRotary] = encode_bf16(
          second * cosines[dimension] + first * sines[dimension]);
    }
    std::copy_n(input + offset + kQwenRotaryDimension,
                kFullAttentionHeadDimension - kQwenRotaryDimension,
                output + offset + kQwenRotaryDimension);
  }
  return DecodeOpStatus::kSuccess;
}

DecodeOpStatus softmax_reference_cpu(
    const float* const input,
    const std::size_t rows,
    const std::size_t columns,
    float* const output) noexcept {
  if (multiply_overflows(rows, columns)) {
    return DecodeOpStatus::kSizeOverflow;
  }
  if (rows == 0U || columns == 0U) {
    return DecodeOpStatus::kSuccess;
  }
  if (input == nullptr || output == nullptr) {
    return DecodeOpStatus::kInvalidArgument;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t offset = row * columns;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t column = 0; column < columns; ++column) {
      maximum = std::max(maximum, input[offset + column]);
    }
    float denominator = 0.0F;
    for (std::size_t column = 0; column < columns; ++column) {
      denominator += std::exp(input[offset + column] - maximum);
    }
    for (std::size_t column = 0; column < columns; ++column) {
      output[offset + column] =
          std::exp(input[offset + column] - maximum) / denominator;
    }
  }
  return DecodeOpStatus::kSuccess;
}

DecodeOpStatus gqa_attention_reference_cpu(
    const std::uint16_t* const query,
    const std::uint16_t* const key_cache,
    const std::uint16_t* const value_cache,
    const std::size_t query_head_count,
    const std::size_t kv_head_count,
    const std::size_t sequence_length,
    const std::size_t head_dimension,
    const float attention_scale,
    float* const probabilities_scratch,
    const std::size_t scratch_elements,
    std::uint16_t* const output) noexcept {
  const DecodeOpStatus validation = validate_attention(
      query, key_cache, value_cache, query_head_count, kv_head_count,
      sequence_length, head_dimension, attention_scale,
      probabilities_scratch, scratch_elements, output);
  if (validation != DecodeOpStatus::kSuccess) {
    return validation;
  }

  const std::size_t queries_per_kv = query_head_count / kv_head_count;
  for (std::size_t query_head = 0; query_head < query_head_count;
       ++query_head) {
    const std::size_t kv_head = query_head / queries_per_kv;
    const std::size_t query_offset = query_head * head_dimension;
    const std::size_t score_offset = query_head * sequence_length;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const std::size_t key_offset =
          (position * kv_head_count + kv_head) * head_dimension;
      float score = 0.0F;
      for (std::size_t dimension = 0; dimension < head_dimension;
           ++dimension) {
        score += decode_bf16(query[query_offset + dimension]) *
                 decode_bf16(key_cache[key_offset + dimension]);
      }
      probabilities_scratch[score_offset + position] =
          score * attention_scale;
    }
  }
  const DecodeOpStatus softmax_status = softmax_reference_cpu(
      probabilities_scratch, query_head_count, sequence_length,
      probabilities_scratch);
  if (softmax_status != DecodeOpStatus::kSuccess) {
    return softmax_status;
  }

  for (std::size_t query_head = 0; query_head < query_head_count;
       ++query_head) {
    const std::size_t kv_head = query_head / queries_per_kv;
    const std::size_t score_offset = query_head * sequence_length;
    const std::size_t output_offset = query_head * head_dimension;
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      float value = 0.0F;
      for (std::size_t position = 0; position < sequence_length; ++position) {
        const std::size_t value_offset =
            (position * kv_head_count + kv_head) * head_dimension;
        value += probabilities_scratch[score_offset + position] *
                 decode_bf16(value_cache[value_offset + dimension]);
      }
      output[output_offset + dimension] = encode_bf16(value);
    }
  }
  return DecodeOpStatus::kSuccess;
}

}  // namespace q3x::runtime
