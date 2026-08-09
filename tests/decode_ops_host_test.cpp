#include "q3x/runtime/decode_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using FullAttentionPreprocessReference256Launch = int (*)(
    const std::uint16_t*, std::uint16_t*, const std::uint16_t*,
    const std::uint16_t*, float, std::uint16_t*, std::uint16_t*,
    const float*, const float*, std::size_t, std::size_t, void*) noexcept;

using FixedBulkCausalGqaLaunch = int (*)(
    const std::uint16_t*, const std::uint16_t*, const std::uint16_t*,
    const std::uint16_t*, std::size_t, std::size_t, std::uint16_t*,
    void*) noexcept;

static_assert(std::is_same_v<
              decltype(&q3x::runtime::
                           launch_full_attention_preprocess_24_4_256_64_reference_256_cuda),
              FullAttentionPreprocessReference256Launch>);
static_assert(std::is_same_v<
              decltype(&q3x::runtime::
                           launch_bulk_causal_gqa_sigmoid_gate_24_4_256_fixed_cuda),
              FixedBulkCausalGqaLaunch>);

class TestContext {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  void expect_near(const double actual, const double expected,
                   const double tolerance, const std::string& message) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
      ++failures_;
      std::cerr << "FAIL: " << message << ": expected " << expected
                << ", got " << actual << ", tolerance " << tolerance << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] float float_from_bits(const std::uint32_t bits) {
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void expect_bf16_near(TestContext& test, const std::uint16_t actual,
                      const double expected, const std::string& message) {
  const double tolerance =
      std::max(2.0e-3, std::fabs(expected) * 8.0e-3);
  test.expect_near(static_cast<double>(decode_bf16(actual)), expected,
                   tolerance, message);
}

void test_fixed_bulk_causal_gqa_contract(TestContext& test) {
  using q3x::runtime::FixedBulkCausalGqaPrefillTactic;
  using q3x::runtime::select_fixed_bulk_causal_gqa_prefill_tactic;
  constexpr std::size_t kMaximum =
      q3x::runtime::kBulkCausalGqaMaximumSequenceLength;

  test.expect(
      select_fixed_bulk_causal_gqa_prefill_tactic(0U, 2U) ==
              FixedBulkCausalGqaPrefillTactic::kGroupQ64V3 &&
          select_fixed_bulk_causal_gqa_prefill_tactic(0U, 512U) ==
              FixedBulkCausalGqaPrefillTactic::kGroupQ64V3 &&
          select_fixed_bulk_causal_gqa_prefill_tactic(512U, 2U) ==
              FixedBulkCausalGqaPrefillTactic::kGroupQ64V3 &&
          select_fixed_bulk_causal_gqa_prefill_tactic(512U, 512U) ==
              FixedBulkCausalGqaPrefillTactic::kGroupQ64V3,
      "sealed bulk GQA fixes every legal P0/P512 tile to V3 group-Q64");
  test.expect(
      select_fixed_bulk_causal_gqa_prefill_tactic(1U, 2U) ==
              FixedBulkCausalGqaPrefillTactic::kGenericQt2 &&
          select_fixed_bulk_causal_gqa_prefill_tactic(257U, 256U) ==
              FixedBulkCausalGqaPrefillTactic::kGenericQt2 &&
          select_fixed_bulk_causal_gqa_prefill_tactic(511U, 512U) ==
              FixedBulkCausalGqaPrefillTactic::kGenericQt2 &&
          select_fixed_bulk_causal_gqa_prefill_tactic(513U, 512U) ==
              FixedBulkCausalGqaPrefillTactic::kGenericQt2 &&
          select_fixed_bulk_causal_gqa_prefill_tactic(kMaximum - 512U,
                                                      512U) ==
              FixedBulkCausalGqaPrefillTactic::kGenericQt2,
      "sealed bulk GQA fixes every other legal append to generic QT2");
  test.expect(
      select_fixed_bulk_causal_gqa_prefill_tactic(0U, 1U) ==
              FixedBulkCausalGqaPrefillTactic::kInvalid &&
          select_fixed_bulk_causal_gqa_prefill_tactic(0U, 513U) ==
              FixedBulkCausalGqaPrefillTactic::kInvalid &&
          select_fixed_bulk_causal_gqa_prefill_tactic(kMaximum - 511U,
                                                      512U) ==
              FixedBulkCausalGqaPrefillTactic::kInvalid &&
          select_fixed_bulk_causal_gqa_prefill_tactic(
              std::numeric_limits<std::size_t>::max(), 2U) ==
              FixedBulkCausalGqaPrefillTactic::kInvalid,
      "sealed bulk GQA rejects single-token, oversized, and causal-range "
      "overflow geometry");
}

void test_embedding(TestContext& test) {
  constexpr std::size_t kVocabulary = 5U;
  constexpr std::size_t kHidden = 7U;
  std::array<std::uint16_t, kVocabulary * kHidden> table{};
  for (std::size_t index = 0; index < table.size(); ++index) {
    table[index] = encode_bf16(static_cast<float>(index) * 0.125F - 2.0F);
  }
  std::array<std::uint16_t, kHidden> output{};
  test.expect(q3x::runtime::embedding_gather_reference_cpu(
                  table.data(), kVocabulary, kHidden, 3U, output.data()) ==
                  q3x::runtime::DecodeOpStatus::kSuccess,
              "embedding gather succeeds");
  for (std::size_t index = 0; index < kHidden; ++index) {
    test.expect(output[index] == table[3U * kHidden + index],
                "embedding gather element " + std::to_string(index));
  }
}

void test_norms(TestContext& test) {
  constexpr std::size_t kHidden = 13U;
  constexpr float kEpsilon = 1.0e-6F;
  std::array<std::uint16_t, kHidden> input{};
  std::array<std::uint16_t, kHidden> weight{};
  for (std::size_t index = 0; index < kHidden; ++index) {
    input[index] = encode_bf16(
        static_cast<float>(static_cast<int>(index) - 6) * 0.375F);
    weight[index] = encode_bf16(
        static_cast<float>(static_cast<int>(index % 5U) - 2) * 0.125F);
  }
  std::array<std::uint16_t, kHidden> centered{};
  std::array<std::uint16_t, kHidden> plain{};
  test.expect(q3x::runtime::centered_rms_norm_reference_cpu(
                  input.data(), weight.data(), kHidden, kEpsilon,
                  centered.data()) == q3x::runtime::DecodeOpStatus::kSuccess,
              "centered RMSNorm succeeds");
  test.expect(q3x::runtime::plain_rms_norm_reference_cpu(
                  input.data(), weight.data(), kHidden, kEpsilon,
                  plain.data()) == q3x::runtime::DecodeOpStatus::kSuccess,
              "plain RMSNorm succeeds");

  double sum = 0.0;
  for (const std::uint16_t value : input) {
    const double decoded = static_cast<double>(decode_bf16(value));
    sum += decoded * decoded;
  }
  const double inverse_rms =
      1.0 / std::sqrt(sum / static_cast<double>(kHidden) + kEpsilon);
  for (std::size_t index = 0; index < kHidden; ++index) {
    const double value = static_cast<double>(decode_bf16(input[index]));
    const double gamma = static_cast<double>(decode_bf16(weight[index]));
    expect_bf16_near(test, centered[index],
                     value * inverse_rms * (1.0 + gamma),
                     "centered RMSNorm element " + std::to_string(index));
    expect_bf16_near(test, plain[index], value * inverse_rms * gamma,
                     "plain RMSNorm element " + std::to_string(index));
  }

  // Exact input/output aliasing is supported after the reduction is complete.
  auto in_place = input;
  (void)q3x::runtime::centered_rms_norm_reference_cpu(
      in_place.data(), weight.data(), kHidden, kEpsilon, in_place.data());
  test.expect(in_place == centered, "centered RMSNorm supports in-place input");

  // Full-attention Q/K norms call the same centered primitive independently
  // for each 256-dimensional head; this must not be confused with the
  // 128-dimensional GDN L2 normalization.
  std::array<std::uint16_t,
             q3x::runtime::kFullAttentionHeadDimension> head_input{};
  std::array<std::uint16_t,
             q3x::runtime::kFullAttentionHeadDimension> head_weight{};
  std::array<std::uint16_t,
             q3x::runtime::kFullAttentionHeadDimension> head_output{};
  for (std::size_t index = 0; index < head_input.size(); ++index) {
    head_input[index] = encode_bf16(
        static_cast<float>(static_cast<int>(index % 13U) - 6) / 8.0F);
    head_weight[index] = encode_bf16(0.0F);
  }
  test.expect(q3x::runtime::centered_rms_norm_reference_cpu(
                  head_input.data(), head_weight.data(), head_input.size(),
                  kEpsilon, head_output.data()) ==
                  q3x::runtime::DecodeOpStatus::kSuccess,
              "centered RMSNorm supports one full-attention 256D head");
}

void test_headwise_norms(TestContext& test) {
  constexpr float kEpsilon = 1.0e-6F;
  constexpr std::size_t kCenteredHeads = 3U;
  constexpr std::size_t kFullDimension =
      q3x::runtime::kFullAttentionHeadDimension;
  std::vector<std::uint16_t> centered_input(kCenteredHeads * kFullDimension);
  std::vector<std::uint16_t> centered_weight(kFullDimension);
  std::vector<std::uint16_t> centered_output(centered_input.size());
  for (std::size_t index = 0; index < centered_input.size(); ++index) {
    centered_input[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 7U) % 41U) - 20) /
        static_cast<float>(8U + index / kFullDimension));
  }
  for (std::size_t dimension = 0; dimension < kFullDimension; ++dimension) {
    centered_weight[dimension] = encode_bf16(
        static_cast<float>(static_cast<int>(dimension % 9U) - 4) / 32.0F);
  }
  test.expect(q3x::runtime::headwise_centered_rms_norm_reference_cpu(
                  centered_input.data(), centered_weight.data(),
                  kCenteredHeads, kFullDimension, kEpsilon,
                  centered_output.data()) ==
                  q3x::runtime::DecodeOpStatus::kSuccess,
              "headwise centered RMSNorm succeeds");
  for (std::size_t head = 0; head < kCenteredHeads; ++head) {
    double sum = 0.0;
    for (std::size_t dimension = 0; dimension < kFullDimension; ++dimension) {
      const double value = decode_bf16(
          centered_input[head * kFullDimension + dimension]);
      sum += value * value;
    }
    const double inverse =
        1.0 / std::sqrt(sum / static_cast<double>(kFullDimension) + kEpsilon);
    for (std::size_t dimension = 0; dimension < kFullDimension; ++dimension) {
      const double value = decode_bf16(
          centered_input[head * kFullDimension + dimension]);
      const double gamma = 1.0 + decode_bf16(centered_weight[dimension]);
      expect_bf16_near(
          test, centered_output[head * kFullDimension + dimension],
          value * inverse * gamma,
          "headwise centered RMSNorm oracle head " + std::to_string(head));
    }
  }

  constexpr std::size_t kGdnHeads = 5U;
  constexpr std::size_t kLinearDimension =
      q3x::runtime::kLinearAttentionHeadDimension;
  std::vector<std::uint16_t> plain_input(kGdnHeads * kLinearDimension);
  std::vector<std::uint16_t> gate(plain_input.size());
  std::vector<std::uint16_t> plain_weight(kLinearDimension);
  std::vector<std::uint16_t> plain_output(plain_input.size());
  std::vector<std::uint16_t> fused_output(plain_input.size());
  for (std::size_t index = 0; index < plain_input.size(); ++index) {
    plain_input[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 11U) % 37U) - 18) /
        16.0F);
    gate[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 5U) % 23U) - 11) /
        16.0F);
  }
  for (std::size_t dimension = 0; dimension < kLinearDimension; ++dimension) {
    plain_weight[dimension] = encode_bf16(
        0.75F + static_cast<float>(dimension % 7U) / 32.0F);
  }
  (void)q3x::runtime::headwise_plain_rms_norm_reference_cpu(
      plain_input.data(), plain_weight.data(), kGdnHeads, kLinearDimension,
      kEpsilon, plain_output.data());
  (void)q3x::runtime::headwise_plain_rms_norm_silu_gate_reference_cpu(
      plain_input.data(), plain_weight.data(), gate.data(), kGdnHeads,
      kLinearDimension, kEpsilon, fused_output.data());
  for (std::size_t head = 0; head < kGdnHeads; ++head) {
    double sum = 0.0;
    for (std::size_t dimension = 0; dimension < kLinearDimension; ++dimension) {
      const double value =
          decode_bf16(plain_input[head * kLinearDimension + dimension]);
      sum += value * value;
    }
    const double inverse =
        1.0 /
        std::sqrt(sum / static_cast<double>(kLinearDimension) + kEpsilon);
    for (std::size_t dimension = 0; dimension < kLinearDimension; ++dimension) {
      const std::size_t index = head * kLinearDimension + dimension;
      const double normalized = decode_bf16(plain_input[index]) * inverse *
                                decode_bf16(plain_weight[dimension]);
      const double gate_value = decode_bf16(gate[index]);
      const double silu = gate_value / (1.0 + std::exp(-gate_value));
      expect_bf16_near(test, plain_output[index], normalized,
                       "headwise plain RMSNorm oracle");
      expect_bf16_near(test, fused_output[index], normalized * silu,
                       "headwise plain RMSNorm SiLU gate oracle");
    }
  }
}

void test_pointwise(TestContext& test) {
  constexpr std::size_t kCount = 17U;
  std::array<std::uint16_t, kCount> left{};
  std::array<std::uint16_t, kCount> right{};
  std::array<std::uint16_t, kCount> residual{};
  std::array<std::uint16_t, kCount> silu{};
  std::array<std::uint16_t, kCount> sigmoid_gate{};
  for (std::size_t index = 0; index < kCount; ++index) {
    left[index] = encode_bf16(
        static_cast<float>(static_cast<int>(index) - 8) * 0.5F);
    right[index] = encode_bf16(
        static_cast<float>(static_cast<int>(index % 7U) - 3) * 0.25F);
  }
  (void)q3x::runtime::residual_add_reference_cpu(
      left.data(), right.data(), kCount, residual.data());
  (void)q3x::runtime::silu_mul_reference_cpu(
      left.data(), right.data(), kCount, silu.data());
  (void)q3x::runtime::sigmoid_gate_reference_cpu(
      left.data(), right.data(), kCount, sigmoid_gate.data());
  for (std::size_t index = 0; index < kCount; ++index) {
    const float gate = decode_bf16(left[index]);
    const float up = decode_bf16(right[index]);
    test.expect(residual[index] == encode_bf16(gate + up),
                "residual element " + std::to_string(index));
    test.expect(silu[index] == encode_bf16(
                                   gate / (1.0F + std::exp(-gate)) * up),
                "SiLU multiply element " + std::to_string(index));
    const float sigmoid =
        up >= 0.0F ? 1.0F / (1.0F + std::exp(-up))
                   : std::exp(up) / (1.0F + std::exp(up));
    test.expect(sigmoid_gate[index] == encode_bf16(gate * sigmoid),
                "sigmoid gate element " + std::to_string(index));
  }

  auto in_place_gate = left;
  (void)q3x::runtime::sigmoid_gate_reference_cpu(
      in_place_gate.data(), right.data(), kCount, in_place_gate.data());
  test.expect(in_place_gate == sigmoid_gate,
              "sigmoid gate supports value/output aliasing");

  const std::array<float, 7> fp32 = {
      float_from_bits(0x3f808000U),  // midpoint, even lower BF16 -> down
      float_from_bits(0x3f818000U),  // midpoint, odd lower BF16 -> up
      0.0F,
      -0.0F,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      float_from_bits(0x7f800001U),  // tiny signaling NaN payload
  };
  std::array<std::uint16_t, fp32.size()> converted{};
  (void)q3x::runtime::fp32_to_bf16_reference_cpu(
      fp32.data(), fp32.size(), converted.data());
  test.expect(converted[0] == 0x3f80U,
              "FP32->BF16 midpoint rounds to even lower value");
  test.expect(converted[1] == 0x3f82U,
              "FP32->BF16 midpoint rounds away from odd lower value");
  test.expect(converted[2] == 0x0000U && converted[3] == 0x8000U,
              "FP32->BF16 preserves signed zero");
  test.expect(converted[4] == 0x7f80U && converted[5] == 0xff80U,
              "FP32->BF16 preserves infinities");
  test.expect(std::isnan(decode_bf16(converted[6])) &&
                  (converted[6] & 0x0040U) != 0U,
              "FP32->BF16 preserves and quiets tiny-payload NaN");
}

void test_l2_norm(TestContext& test) {
  constexpr std::size_t kHeads = 5U;
  constexpr std::size_t kDimension = 11U;
  constexpr float kEpsilon = 1.0e-6F;
  std::array<std::uint16_t, kHeads * kDimension> input{};
  std::array<std::uint16_t, kHeads * kDimension> output{};
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 7U) % 19U) - 9) /
        16.0F);
  }
  (void)q3x::runtime::l2_normalize_heads_reference_cpu(
      input.data(), kHeads, kDimension, kEpsilon, output.data());
  for (std::size_t head = 0; head < kHeads; ++head) {
    double input_sum = 0.0;
    double output_sum = 0.0;
    for (std::size_t dimension = 0; dimension < kDimension; ++dimension) {
      const double input_value = static_cast<double>(
          decode_bf16(input[head * kDimension + dimension]));
      const double output_value = static_cast<double>(
          decode_bf16(output[head * kDimension + dimension]));
      input_sum += input_value * input_value;
      output_sum += output_value * output_value;
    }
    const double expected = input_sum / (input_sum + kEpsilon);
    test.expect_near(output_sum, expected, 1.5e-2,
                     "L2 normalized head " + std::to_string(head));
  }
}

void test_rope(TestContext& test) {
  constexpr std::size_t kHeads = 3U;
  constexpr std::size_t kHeadDimension =
      q3x::runtime::kFullAttentionHeadDimension;
  constexpr std::size_t kHalfRotary =
      q3x::runtime::kQwenRotaryDimension / 2U;
  std::array<std::uint16_t, kHeads * kHeadDimension> input{};
  std::array<std::uint16_t, kHeads * kHeadDimension> output{};
  std::array<float, kHalfRotary> cosines{};
  std::array<float, kHalfRotary> sines{};
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = encode_bf16(
        static_cast<float>(static_cast<int>(index % 31U) - 15) / 32.0F);
  }
  for (std::size_t index = 0; index < kHalfRotary; ++index) {
    const float angle = static_cast<float>(index) * 0.03125F;
    cosines[index] = std::cos(angle);
    sines[index] = std::sin(angle);
  }
  (void)q3x::runtime::partial_neox_rope_256_64_reference_cpu(
      input.data(), cosines.data(), sines.data(), kHeads, output.data());
  for (std::size_t head = 0; head < kHeads; ++head) {
    const std::size_t offset = head * kHeadDimension;
    for (std::size_t dimension = 0; dimension < kHalfRotary; ++dimension) {
      const double first = decode_bf16(input[offset + dimension]);
      const double second =
          decode_bf16(input[offset + dimension + kHalfRotary]);
      expect_bf16_near(test, output[offset + dimension],
                       first * cosines[dimension] -
                           second * sines[dimension],
                       "RoPE first half");
      expect_bf16_near(test, output[offset + dimension + kHalfRotary],
                       second * cosines[dimension] +
                           first * sines[dimension],
                       "RoPE second half");
    }
    for (std::size_t dimension = q3x::runtime::kQwenRotaryDimension;
         dimension < kHeadDimension; ++dimension) {
      test.expect(output[offset + dimension] == input[offset + dimension],
                  "RoPE pass-through dimension");
    }
  }
}

void test_softmax(TestContext& test) {
  constexpr std::size_t kRows = 3U;
  constexpr std::size_t kColumns = 7U;
  std::array<float, kRows * kColumns> values{};
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t column = 0; column < kColumns; ++column) {
      values[row * kColumns + column] =
          1000.0F + static_cast<float>(row * 11U + column) * 0.25F;
    }
  }
  auto output = values;
  test.expect(q3x::runtime::softmax_reference_cpu(
                  output.data(), kRows, kColumns, output.data()) ==
                  q3x::runtime::DecodeOpStatus::kSuccess,
              "stable in-place softmax succeeds");
  for (std::size_t row = 0; row < kRows; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0; column < kColumns; ++column) {
      const float probability = output[row * kColumns + column];
      test.expect(std::isfinite(probability) && probability >= 0.0F,
                  "softmax finite probability");
      sum += probability;
    }
    test.expect_near(sum, 1.0, 2.0e-6,
                     "softmax row sum " + std::to_string(row));
  }
}

void test_gqa_attention(TestContext& test) {
  constexpr std::size_t kQueryHeads = 6U;
  constexpr std::size_t kKvHeads = 2U;
  constexpr std::size_t kSequence = 5U;
  constexpr std::size_t kDimension = 7U;
  constexpr float kScale = 0.25F;
  std::array<std::uint16_t, kQueryHeads * kDimension> query{};
  std::array<std::uint16_t, kSequence * kKvHeads * kDimension> key{};
  std::array<std::uint16_t, kSequence * kKvHeads * kDimension> value{};
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 5U) % 17U) - 8) /
        16.0F);
  }
  for (std::size_t index = 0; index < key.size(); ++index) {
    key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 3U) % 19U) - 9) /
        16.0F);
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 7U) % 23U) - 11) /
        16.0F);
  }
  std::array<float, kQueryHeads * kSequence> scratch{};
  std::array<std::uint16_t, kQueryHeads * kDimension> output{};
  test.expect(q3x::runtime::gqa_attention_reference_cpu(
                  query.data(), key.data(), value.data(), kQueryHeads,
                  kKvHeads, kSequence, kDimension, kScale, scratch.data(),
                  scratch.size(), output.data()) ==
                  q3x::runtime::DecodeOpStatus::kSuccess,
              "GQA attention succeeds");

  constexpr std::size_t kQueriesPerKv = kQueryHeads / kKvHeads;
  for (std::size_t query_head = 0; query_head < kQueryHeads; ++query_head) {
    const std::size_t kv_head = query_head / kQueriesPerKv;
    std::array<double, kSequence> scores{};
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t position = 0; position < kSequence; ++position) {
      double score = 0.0;
      for (std::size_t dimension = 0; dimension < kDimension; ++dimension) {
        score += static_cast<double>(decode_bf16(
                     query[query_head * kDimension + dimension])) *
                 static_cast<double>(decode_bf16(
                     key[(position * kKvHeads + kv_head) * kDimension +
                         dimension]));
      }
      scores[position] = score * kScale;
      maximum = std::max(maximum, scores[position]);
    }
    double denominator = 0.0;
    for (double& score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (double& score : scores) {
      score /= denominator;
    }
    double probability_sum = 0.0;
    for (std::size_t position = 0; position < kSequence; ++position) {
      probability_sum += scratch[query_head * kSequence + position];
      test.expect_near(scratch[query_head * kSequence + position],
                       scores[position], 2.0e-6,
                       "GQA probability oracle");
    }
    test.expect_near(probability_sum, 1.0, 2.0e-6,
                     "GQA probability row sum");
    for (std::size_t dimension = 0; dimension < kDimension; ++dimension) {
      double expected = 0.0;
      for (std::size_t position = 0; position < kSequence; ++position) {
        expected +=
            scores[position] *
            static_cast<double>(decode_bf16(
                value[(position * kKvHeads + kv_head) * kDimension +
                      dimension]));
      }
      expect_bf16_near(test,
                       output[query_head * kDimension + dimension], expected,
                       "GQA output oracle");
    }
  }
}

void test_validation_and_nonfinite(TestContext& test) {
  using q3x::runtime::DecodeOpStatus;
  constexpr std::size_t kMaximum = std::numeric_limits<std::size_t>::max();
  std::uint16_t one = encode_bf16(1.0F);
  std::uint16_t output = 0U;
  float float_value = 0.0F;

  test.expect(q3x::runtime::embedding_gather_reference_cpu(
                  &one, 1U, 1U, 1U, &output) ==
                  DecodeOpStatus::kTokenOutOfRange,
              "embedding rejects out-of-range token");
  test.expect(q3x::runtime::embedding_gather_reference_cpu(
                  &one, kMaximum, 2U, 0U, &output) ==
                  DecodeOpStatus::kSizeOverflow,
              "embedding rejects size overflow");
  test.expect(q3x::runtime::centered_rms_norm_reference_cpu(
                  &one, &one, 1U,
                  std::numeric_limits<float>::quiet_NaN(), &output) ==
                  DecodeOpStatus::kInvalidArgument,
              "RMSNorm rejects NaN epsilon");
  test.expect(q3x::runtime::plain_rms_norm_reference_cpu(
                  &one, &one, 1U, 0.0F, &output) ==
                  DecodeOpStatus::kInvalidArgument,
              "RMSNorm rejects zero epsilon");
  test.expect(q3x::runtime::headwise_centered_rms_norm_reference_cpu(
                  &one, &one, kMaximum, 2U, 1.0e-6F, &output) ==
                  DecodeOpStatus::kSizeOverflow,
              "headwise RMSNorm rejects size overflow");
  test.expect(q3x::runtime::headwise_plain_rms_norm_silu_gate_reference_cpu(
                  &one, &one, nullptr, 1U, 1U, 1.0e-6F, &output) ==
                  DecodeOpStatus::kInvalidArgument,
              "fused headwise RMSNorm rejects null gate");
  test.expect(q3x::runtime::residual_add_reference_cpu(
                  nullptr, nullptr, 0U, nullptr) == DecodeOpStatus::kSuccess,
              "empty residual is a no-op");
  test.expect(q3x::runtime::l2_normalize_heads_reference_cpu(
                  &one, kMaximum, 2U, 1.0e-6F, &output) ==
                  DecodeOpStatus::kSizeOverflow,
              "L2 norm rejects size overflow");
  test.expect(q3x::runtime::partial_neox_rope_256_64_reference_cpu(
                  &one, &float_value, &float_value,
                  kMaximum /
                          q3x::runtime::kFullAttentionHeadDimension +
                      1U,
                  &output) == DecodeOpStatus::kSizeOverflow,
              "RoPE rejects size overflow");
  test.expect(q3x::runtime::softmax_reference_cpu(
                  &float_value, kMaximum, 2U, &float_value) ==
                  DecodeOpStatus::kSizeOverflow,
              "softmax rejects size overflow");

  std::array<float, 6> scratch{};
  test.expect(q3x::runtime::gqa_attention_reference_cpu(
                  &one, &one, &one, 3U, 2U, 1U, 1U, 1.0F, scratch.data(),
                  scratch.size(), &output) == DecodeOpStatus::kInvalidDimension,
              "GQA rejects non-divisible head mapping");
  test.expect(q3x::runtime::gqa_attention_reference_cpu(
                  &one, &one, &one, 2U, 1U, 3U, 1U, 1.0F, scratch.data(),
                  5U, &output) == DecodeOpStatus::kInsufficientScratch,
              "GQA rejects undersized scratch");
  test.expect(q3x::runtime::gqa_attention_reference_cpu(
                  &one, &one, &one, 1U, 1U, 1U, 1U,
                  std::numeric_limits<float>::infinity(), scratch.data(), 1U,
                  &output) == DecodeOpStatus::kInvalidArgument,
              "GQA rejects infinite attention scale");
  test.expect(std::string(q3x::runtime::decode_op_status_string(
                  DecodeOpStatus::kInsufficientScratch)) ==
                  "attention probability scratch is too small",
              "decode-op status text is stable");

  const std::uint16_t infinity =
      encode_bf16(std::numeric_limits<float>::infinity());
  (void)q3x::runtime::residual_add_reference_cpu(
      &infinity, &one, 1U, &output);
  test.expect(std::isinf(decode_bf16(output)),
              "residual propagates infinity");
  const std::uint16_t nan =
      encode_bf16(std::numeric_limits<float>::quiet_NaN());
  (void)q3x::runtime::residual_add_reference_cpu(&nan, &one, 1U, &output);
  test.expect(std::isnan(decode_bf16(output)), "residual propagates NaN");
  std::array<float, 3> nan_softmax = {
      0.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F};
  (void)q3x::runtime::softmax_reference_cpu(
      nan_softmax.data(), 1U, nan_softmax.size(), nan_softmax.data());
  test.expect(std::all_of(nan_softmax.begin(), nan_softmax.end(),
                          [](const float value) { return std::isnan(value); }),
              "softmax propagates NaN across its row");
}

}  // namespace

int main() {
  TestContext test;
  test_fixed_bulk_causal_gqa_contract(test);
  test_embedding(test);
  test_norms(test);
  test_headwise_norms(test);
  test_pointwise(test);
  test_l2_norm(test);
  test_rope(test);
  test_softmax(test);
  test_gqa_attention(test);
  test_validation_and_nonfinite(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " decode-op host assertion(s) failed\n";
    return 1;
  }
  std::cout << "Decode-op host reference tests passed\n";
  return 0;
}
