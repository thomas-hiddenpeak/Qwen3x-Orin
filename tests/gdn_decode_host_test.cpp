#include "q3x/runtime/gdn_decode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

class TestContext {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  void expect_near(const float actual, const float expected,
                   const float tolerance, const std::string& message) {
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

[[nodiscard]] float silu(const float value) {
  return value / (1.0F + std::exp(-value));
}

[[nodiscard]] std::size_t state_index(const std::size_t value_head,
                                      const std::size_t value_dimension,
                                      const std::size_t key_dimension) {
  return (value_head * q3x::runtime::kGdnHeadDimension + value_dimension) *
             q3x::runtime::kGdnHeadDimension +
         key_dimension;
}

void test_dimensions_and_validation(TestContext& test) {
  using q3x::runtime::GdnDimensions;
  using q3x::runtime::GdnStatus;
  test.expect(q3x::runtime::kGdnQElements == 2048U &&
                  q3x::runtime::kGdnKElements == 2048U &&
                  q3x::runtime::kGdnVElements == 6144U &&
                  q3x::runtime::kGdnQkvChannels == 10240U &&
                  q3x::runtime::kGdnStateElements == 786432U,
              "GDN exact logical constants");

  std::uint16_t value = 0U;
  const GdnDimensions wrong{15U, 48U, 128U};
  const GdnDimensions overflow{
      std::numeric_limits<std::size_t>::max(), 48U, 128U};
  test.expect(q3x::runtime::causal_conv1d_silu_update_reference_cpu(
                  &value, &value, &value, &value, wrong) ==
                  GdnStatus::kInvalidDimension,
              "conv rejects non-Qwen dimensions before pointers are used");
  test.expect(q3x::runtime::causal_conv1d_silu_update_reference_cpu(
                  &value, &value, &value, &value, overflow) ==
                  GdnStatus::kSizeOverflow,
              "conv rejects overflowing dimensions");
  test.expect(q3x::runtime::gated_delta_net_update_reference_cpu(
                  &value, &value, &value, &value, &value, &value, &value,
                  std::numeric_limits<float>::quiet_NaN(), &value) ==
                  GdnStatus::kInvalidArgument,
              "GDN rejects NaN epsilon");
  test.expect(q3x::runtime::gated_delta_net_update_reference_cpu(
                  &value, &value, &value, &value, &value, &value, &value,
                  std::numeric_limits<float>::infinity(), &value) ==
                  GdnStatus::kInvalidArgument,
              "GDN rejects infinite epsilon");
  test.expect(q3x::runtime::gated_delta_net_update_reference_cpu(
                  &value, &value, &value, &value, &value, &value, &value,
                  0.0F, &value) == GdnStatus::kInvalidArgument,
              "GDN rejects zero epsilon");
  test.expect(std::string(q3x::runtime::gdn_status_string(
                  GdnStatus::kInvalidAlias)) ==
                  "unsupported exact buffer alias",
              "GDN status text is stable");
  test.expect(
      q3x::runtime::
          supports_gated_delta_net_update_plain_rms_norm_silu_gate_fusion(
              1U, {}, q3x::runtime::kGdnValueHeadCount,
              q3x::runtime::kGdnHeadDimension),
      "GDN norm/gate selector accepts exact M1 48x128");
  test.expect(
      !q3x::runtime::
           supports_gated_delta_net_update_plain_rms_norm_silu_gate_fusion(
               2U, {}, q3x::runtime::kGdnValueHeadCount,
               q3x::runtime::kGdnHeadDimension) &&
          !q3x::runtime::
               supports_gated_delta_net_update_plain_rms_norm_silu_gate_fusion(
                   1U, wrong, q3x::runtime::kGdnValueHeadCount,
                   q3x::runtime::kGdnHeadDimension) &&
          !q3x::runtime::
               supports_gated_delta_net_update_plain_rms_norm_silu_gate_fusion(
                   1U, {}, 24U, 256U),
      "GDN norm/gate selector rejects M2, wrong GDN, and norm near miss");
}

void test_conv_cold_and_alias(TestContext& test) {
  std::vector<std::uint16_t> raw(q3x::runtime::kGdnQkvChannels);
  std::vector<std::uint16_t> weight(
      q3x::runtime::kGdnQkvChannels * q3x::runtime::kGdnConvKernelWidth,
      encode_bf16(0.0F));
  std::vector<std::uint16_t> history(
      q3x::runtime::kGdnQkvChannels * q3x::runtime::kGdnConvHistoryWidth,
      encode_bf16(0.0F));
  std::vector<std::uint16_t> output(q3x::runtime::kGdnQkvChannels);
  for (std::size_t channel = 0; channel < raw.size(); ++channel) {
    raw[channel] = encode_bf16(
        static_cast<float>(static_cast<int>(channel % 17U) - 8) / 8.0F);
    weight[channel * q3x::runtime::kGdnConvKernelWidth + 3U] =
        encode_bf16(1.0F);
  }
  const auto status = q3x::runtime::causal_conv1d_silu_update_reference_cpu(
      raw.data(), weight.data(), history.data(), output.data());
  test.expect(status == q3x::runtime::GdnStatus::kSuccess,
              "cold causal conv succeeds");
  for (std::size_t channel = 0; channel < raw.size(); ++channel) {
    const float expected = silu(decode_bf16(raw[channel]));
    test.expect(output[channel] == encode_bf16(expected),
                "cold conv current*w3 at channel " +
                    std::to_string(channel));
    const std::size_t offset =
        channel * q3x::runtime::kGdnConvHistoryWidth;
    test.expect(history[offset] == encode_bf16(0.0F) &&
                    history[offset + 1U] == encode_bf16(0.0F) &&
                    history[offset + 2U] == raw[channel],
                "cold conv stores raw current in history");
  }

  std::fill(history.begin(), history.end(), encode_bf16(0.0F));
  auto aliased_raw_output = raw;
  const auto expected_output = output;
  test.expect(q3x::runtime::causal_conv1d_silu_update_reference_cpu(
                  aliased_raw_output.data(), weight.data(), history.data(),
                  aliased_raw_output.data()) ==
                  q3x::runtime::GdnStatus::kSuccess,
              "conv supports exact raw/output alias");
  test.expect(aliased_raw_output == expected_output,
              "aliased conv output matches disjoint output");
  test.expect(q3x::runtime::causal_conv1d_silu_update_reference_cpu(
                  raw.data(), weight.data(), history.data(), history.data()) ==
                  q3x::runtime::GdnStatus::kInvalidAlias,
              "conv rejects exact history/output alias");
}

void test_conv_multistep_orientation(TestContext& test) {
  std::vector<std::uint16_t> raw(q3x::runtime::kGdnQkvChannels,
                                 encode_bf16(0.0F));
  std::vector<std::uint16_t> weight(
      q3x::runtime::kGdnQkvChannels * q3x::runtime::kGdnConvKernelWidth,
      encode_bf16(0.0F));
  std::vector<std::uint16_t> history(
      q3x::runtime::kGdnQkvChannels * q3x::runtime::kGdnConvHistoryWidth,
      encode_bf16(0.0F));
  std::vector<std::uint16_t> output(q3x::runtime::kGdnQkvChannels);
  constexpr std::size_t kChannel = 173U;
  const std::size_t weight_offset =
      kChannel * q3x::runtime::kGdnConvKernelWidth;
  weight[weight_offset] = encode_bf16(1.0F);
  weight[weight_offset + 1U] = encode_bf16(2.0F);
  weight[weight_offset + 2U] = encode_bf16(4.0F);
  weight[weight_offset + 3U] = encode_bf16(8.0F);
  std::array<float, 3> expected_history = {0.0F, 0.0F, 0.0F};

  for (std::size_t step = 0; step < 5U; ++step) {
    const float current = static_cast<float>(step + 1U) * 0.25F;
    raw[kChannel] = encode_bf16(current);
    const float convolution = expected_history[0] +
                              2.0F * expected_history[1] +
                              4.0F * expected_history[2] + 8.0F * current;
    (void)q3x::runtime::causal_conv1d_silu_update_reference_cpu(
        raw.data(), weight.data(), history.data(), output.data());
    test.expect(output[kChannel] == encode_bf16(silu(convolution)),
                "multistep conv oldest-to-newest weight orientation");
    expected_history = {expected_history[1], expected_history[2], current};
    const std::size_t history_offset =
        kChannel * q3x::runtime::kGdnConvHistoryWidth;
    for (std::size_t slot = 0; slot < expected_history.size(); ++slot) {
      test.expect(history[history_offset + slot] ==
                      encode_bf16(expected_history[slot]),
                  "multistep raw history shift slot " +
                      std::to_string(slot));
    }
  }
}

void fill_aligned_and_orthogonal_qk(std::vector<std::uint16_t>& conv_qkv) {
  std::fill(conv_qkv.begin(), conv_qkv.end(), encode_bf16(0.0F));
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  for (std::size_t head = 0; head < q3x::runtime::kGdnQkHeadCount; ++head) {
    for (std::size_t dimension = 0;
         dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
      conv_qkv[head * q3x::runtime::kGdnHeadDimension + dimension] =
          encode_bf16(1.0F);
      const float key = head == 1U && (dimension & 1U) != 0U ? -1.0F : 1.0F;
      conv_qkv[kKOffset + head * q3x::runtime::kGdnHeadDimension + dimension] =
          encode_bf16(key);
    }
  }
  for (std::size_t index = 0; index < q3x::runtime::kGdnVElements; ++index) {
    conv_qkv[kVOffset + index] = encode_bf16(1.0F);
  }
}

void test_gdn_cold_mapping_and_orientation(TestContext& test) {
  std::vector<std::uint16_t> conv_qkv(q3x::runtime::kGdnQkvChannels);
  fill_aligned_and_orthogonal_qk(conv_qkv);
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> a{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> b{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> A_log{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> dt_bias{};
  a.fill(encode_bf16(0.0F));
  b.fill(encode_bf16(0.0F));
  A_log.fill(encode_bf16(0.0F));
  dt_bias.fill(encode_bf16(0.0F));
  std::vector<std::uint16_t> state(q3x::runtime::kGdnStateElements,
                                   encode_bf16(0.0F));
  std::vector<std::uint16_t> output(q3x::runtime::kGdnVElements);

  test.expect(q3x::runtime::gated_delta_net_update_reference_cpu(
                  conv_qkv.data(), a.data(), b.data(), A_log.data(),
                  dt_bias.data(), state.data(), state.data(), 1.0e-6F,
                  output.data()) == q3x::runtime::GdnStatus::kSuccess,
              "cold in-place GDN update succeeds");
  const float expected_state = 0.5F / std::sqrt(128.0F + 1.0e-6F);
  const float expected_output = expected_state;
  test.expect_near(decode_bf16(output[0]), expected_output, 8.0e-4F,
                   "aligned head cold output formula");
  test.expect_near(decode_bf16(output[2U * 128U + 17U]), expected_output,
                   8.0e-4F, "three value heads share Q/K head zero");
  test.expect(std::fabs(decode_bf16(output[3U * 128U])) < 2.0e-4F,
              "value head three maps to orthogonal Q/K head one");
  test.expect_near(decode_bf16(state[state_index(0U, 0U, 0U)]),
                   expected_state, 8.0e-4F,
                   "canonical state layout [head][value][key]");
  test.expect(decode_bf16(state[state_index(3U, 7U, 0U)]) > 0.0F &&
                  decode_bf16(state[state_index(3U, 7U, 1U)]) < 0.0F,
              "state K axis follows normalized alternating key orientation");
}

void test_gdn_gate_extremes(TestContext& test) {
  std::vector<std::uint16_t> conv_qkv(q3x::runtime::kGdnQkvChannels);
  fill_aligned_and_orthogonal_qk(conv_qkv);
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> a{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> b{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> A_log{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> dt_bias{};
  a.fill(encode_bf16(0.0F));
  b.fill(encode_bf16(0.0F));
  A_log.fill(encode_bf16(0.0F));
  dt_bias.fill(encode_bf16(0.0F));
  b[0] = encode_bf16(20.0F);
  b[1] = encode_bf16(-20.0F);
  a[2] = encode_bf16(25.0F);  // stable softplus linear branch
  A_log[2] = encode_bf16(4.0F);
  b[2] = encode_bf16(-20.0F);
  b[3] = encode_bf16(-20.0F);
  std::vector<std::uint16_t> initial_state(q3x::runtime::kGdnStateElements,
                                           encode_bf16(0.0F));
  for (std::size_t value_head : {2U, 3U}) {
    const std::size_t begin =
        value_head * q3x::runtime::kGdnHeadDimension *
        q3x::runtime::kGdnHeadDimension;
    std::fill_n(initial_state.begin() + static_cast<std::ptrdiff_t>(begin),
                q3x::runtime::kGdnHeadDimension *
                    q3x::runtime::kGdnHeadDimension,
                encode_bf16(0.25F));
  }
  std::vector<std::uint16_t> state(initial_state.size());
  std::vector<std::uint16_t> output(q3x::runtime::kGdnVElements);
  (void)q3x::runtime::gated_delta_net_update_reference_cpu(
      conv_qkv.data(), a.data(), b.data(), A_log.data(), dt_bias.data(),
      initial_state.data(), state.data(), 1.0e-6F, output.data());
  test.expect(std::fabs(decode_bf16(output[0])) >
                  1000.0F * std::fabs(decode_bf16(output[128U])),
              "sigmoid beta +20 versus -20 extreme");
  test.expect(std::isfinite(decode_bf16(output[2U * 128U])),
              "A_log/softplus threshold extreme remains defined");
  test.expect(std::fabs(decode_bf16(state[state_index(2U, 0U, 0U)])) <
                  std::fabs(decode_bf16(
                      state[state_index(3U, 0U, 0U)])),
              "large A_log and softplus input strongly decay prior state");
}

void test_output_uses_fp32_updated_state(TestContext& test) {
  std::vector<std::uint16_t> conv_qkv(q3x::runtime::kGdnQkvChannels,
                                      encode_bf16(0.0F));
  constexpr std::size_t kKOffset = q3x::runtime::kGdnQElements;
  constexpr std::size_t kVOffset =
      q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
  for (std::size_t dimension = 0;
       dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
    const int q_centered = static_cast<int>((dimension * 7U) % 29U) - 14;
    const int k_centered = static_cast<int>((dimension * 5U) % 31U) - 15;
    conv_qkv[dimension] =
        encode_bf16(static_cast<float>(q_centered) / 16.0F);
    conv_qkv[kKOffset + dimension] =
        encode_bf16(static_cast<float>(k_centered) / 16.0F);
  }
  for (std::size_t dimension = 0;
       dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
    conv_qkv[kVOffset + dimension] = encode_bf16(-4.0F);
  }
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> a{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> b{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> A_log{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> dt_bias{};
  a.fill(encode_bf16(0.0F));
  b.fill(encode_bf16(0.0F));
  A_log.fill(encode_bf16(0.0F));
  dt_bias.fill(encode_bf16(0.0F));
  b[0] = encode_bf16(-7.5F);
  std::vector<std::uint16_t> state(q3x::runtime::kGdnStateElements,
                                   encode_bf16(0.0F));
  std::vector<std::uint16_t> output(q3x::runtime::kGdnVElements);
  (void)q3x::runtime::gated_delta_net_update_reference_cpu(
      conv_qkv.data(), a.data(), b.data(), A_log.data(), dt_bias.data(),
      state.data(), state.data(), 1.0e-6F, output.data());

  float q_sum = 0.0F;
  float k_sum = 0.0F;
  for (std::size_t dimension = 0;
       dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
    const float q = decode_bf16(conv_qkv[dimension]);
    const float k = decode_bf16(conv_qkv[kKOffset + dimension]);
    q_sum = std::fma(q, q, q_sum);
    k_sum = std::fma(k, k, k_sum);
  }
  const float q_scale =
      1.0F / std::sqrt(q_sum + 1.0e-6F) / std::sqrt(128.0F);
  const float k_scale = 1.0F / std::sqrt(k_sum + 1.0e-6F);
  const float beta =
      std::exp(-7.5F) / (1.0F + std::exp(-7.5F));
  float fp32_output = 0.0F;
  float quantized_reread_output = 0.0F;
  for (std::size_t dimension = 0;
       dimension < q3x::runtime::kGdnHeadDimension; ++dimension) {
    const float normalized_q =
        decode_bf16(conv_qkv[dimension]) * q_scale;
    const float normalized_k =
        decode_bf16(conv_qkv[kKOffset + dimension]) * k_scale;
    const float updated = -4.0F * beta * normalized_k;
    fp32_output = std::fma(updated, normalized_q, fp32_output);
    quantized_reread_output =
        std::fma(decode_bf16(encode_bf16(updated)), normalized_q,
                 quantized_reread_output);
  }
  const std::uint16_t fp32_expected = encode_bf16(fp32_output);
  const std::uint16_t quantized_reread = encode_bf16(quantized_reread_output);
  test.expect(fp32_expected != quantized_reread,
              "fixture distinguishes FP32 updated state from BF16 reread");
  test.expect(output[0] == fp32_expected,
              "GDN output uses FP32 updated state before BF16 persistence");
}

void test_gdn_inplace_multistep(TestContext& test) {
  std::vector<std::uint16_t> conv_qkv(q3x::runtime::kGdnQkvChannels);
  fill_aligned_and_orthogonal_qk(conv_qkv);
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> a{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> b{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> A_log{};
  std::array<std::uint16_t, q3x::runtime::kGdnValueHeadCount> dt_bias{};
  for (std::size_t head = 0; head < a.size(); ++head) {
    a[head] = encode_bf16(static_cast<float>(head % 5U) * 0.125F);
    b[head] = encode_bf16(
        static_cast<float>(static_cast<int>(head % 7U) - 3) * 0.25F);
    A_log[head] = encode_bf16(-1.0F + static_cast<float>(head % 3U) * 0.25F);
    dt_bias[head] = encode_bf16(-0.5F);
  }
  std::vector<std::uint16_t> initial(q3x::runtime::kGdnStateElements);
  for (std::size_t index = 0; index < initial.size(); ++index) {
    initial[index] = encode_bf16(
        static_cast<float>(static_cast<int>(index % 13U) - 6) / 256.0F);
  }
  auto in_place = initial;
  auto separate_input = initial;
  std::vector<std::uint16_t> separate_output(initial.size());
  std::vector<std::uint16_t> output_in_place(q3x::runtime::kGdnVElements);
  std::vector<std::uint16_t> output_separate(q3x::runtime::kGdnVElements);
  for (std::size_t step = 0; step < 2U; ++step) {
    constexpr std::size_t kVOffset =
        q3x::runtime::kGdnQElements + q3x::runtime::kGdnKElements;
    for (std::size_t index = 0; index < q3x::runtime::kGdnVElements; ++index) {
      conv_qkv[kVOffset + index] = encode_bf16(
          0.5F + static_cast<float>((index + step) % 11U) / 32.0F);
    }
    (void)q3x::runtime::gated_delta_net_update_reference_cpu(
        conv_qkv.data(), a.data(), b.data(), A_log.data(), dt_bias.data(),
        separate_input.data(), separate_output.data(), 1.0e-6F,
        output_separate.data());
    (void)q3x::runtime::gated_delta_net_update_reference_cpu(
        conv_qkv.data(), a.data(), b.data(), A_log.data(), dt_bias.data(),
        in_place.data(), in_place.data(), 1.0e-6F, output_in_place.data());
    test.expect(in_place == separate_output,
                "in-place BF16 state matches separate state at step " +
                    std::to_string(step));
    test.expect(output_in_place == output_separate,
                "in-place output matches separate state at step " +
                    std::to_string(step));
    separate_input.swap(separate_output);
  }
  test.expect(q3x::runtime::gated_delta_net_update_reference_cpu(
                  conv_qkv.data(), a.data(), b.data(), A_log.data(),
                  dt_bias.data(), in_place.data(), in_place.data(), 1.0e-6F,
                  conv_qkv.data()) == q3x::runtime::GdnStatus::kInvalidAlias,
              "GDN rejects output alias with conv QKV");
}

}  // namespace

int main() {
  TestContext test;
  test_dimensions_and_validation(test);
  test_conv_cold_and_alias(test);
  test_conv_multistep_orientation(test);
  test_gdn_cold_mapping_and_orientation(test);
  test_gdn_gate_extremes(test);
  test_output_uses_fp32_updated_state(test);
  test_gdn_inplace_multistep(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " GDN host assertion(s) failed\n";
    return 1;
  }
  std::cout << "GDN host reference tests passed\n";
  return 0;
}
