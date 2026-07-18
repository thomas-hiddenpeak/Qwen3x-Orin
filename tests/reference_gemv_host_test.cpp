#include "q3x/kernels/reference_gemv.h"

#include "q3x/quantization/nvfp4.h"

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

  void expect_near(const float actual, const double expected,
                   const double tolerance, const std::string& message) {
    if (!(std::fabs(static_cast<double>(actual) - expected) <= tolerance)) {
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
  static_assert(sizeof(bits) == sizeof(value), "float must be binary32");
  std::memcpy(&bits, &value, sizeof(bits));
  // Round-to-nearest-even, matching ordinary BF16 conversion.
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  bits += rounding;
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] double tolerance_for(const double expected,
                                   const std::size_t columns) {
  return 2.0e-6 * static_cast<double>(columns) +
         3.0e-6 * std::fabs(expected);
}

class DeterministicRng {
 public:
  [[nodiscard]] std::uint32_t next() noexcept {
    state_ ^= state_ << 13U;
    state_ ^= state_ >> 17U;
    state_ ^= state_ << 5U;
    return state_;
  }

  [[nodiscard]] std::size_t bounded(const std::size_t bound) noexcept {
    return static_cast<std::size_t>(next()) % bound;
  }

  [[nodiscard]] float symmetric() noexcept {
    const int value = static_cast<int>(next() % 2049U) - 1024;
    return static_cast<float>(value) / 1024.0F;
  }

 private:
  std::uint32_t state_ = 0x9e3779b9U;
};

void test_bf16_reference(TestContext& test) {
  constexpr std::size_t kRows = 3U;
  constexpr std::size_t kColumns = 5U;
  const std::array<float, kColumns> activation_values = {
      1.0F, -2.0F, 0.5F, 4.0F, -0.25F};
  const std::array<float, kRows * kColumns> weight_values = {
      0.0F, 1.0F, -1.0F, 2.0F, -2.0F,
      0.5F, 0.25F, -0.5F, -0.25F, 1.5F,
      8.0F, -4.0F, 2.0F, -1.0F, 0.125F};

  std::array<std::uint16_t, kColumns> activation{};
  std::array<std::uint16_t, kRows * kColumns> weights{};
  std::transform(activation_values.begin(), activation_values.end(),
                 activation.begin(), encode_bf16);
  std::transform(weight_values.begin(), weight_values.end(), weights.begin(),
                 encode_bf16);
  std::array<float, kRows> output{};

  test.expect(q3x::kernels::bf16_gemv_reference_cpu(
                  weights.data(), activation.data(), kRows, kColumns,
                  output.data()) == q3x::kernels::GemvStatus::kSuccess,
              "BF16 host GEMV succeeds");
  for (std::size_t row = 0; row < kRows; ++row) {
    double expected = 0.0;
    for (std::size_t column = 0; column < kColumns; ++column) {
      expected += static_cast<double>(decode_bf16(
                      weights[row * kColumns + column])) *
                  static_cast<double>(decode_bf16(activation[column]));
    }
    test.expect_near(output[row], expected, tolerance_for(expected, kColumns),
                     "BF16 double reference row " + std::to_string(row));
  }

  std::fill(weights.begin(), weights.end(), encode_bf16(0.0F));
  std::fill(output.begin(), output.end(), 9.0F);
  (void)q3x::kernels::bf16_gemv_reference_cpu(
      weights.data(), activation.data(), kRows, kColumns, output.data());
  test.expect(std::all_of(output.begin(), output.end(),
                          [](const float value) { return value == 0.0F; }),
              "all-zero BF16 weights produce zero");
}

void test_fp8_reference(TestContext& test) {
  constexpr std::size_t kRows = 4U;
  constexpr std::size_t kColumns = 7U;
  constexpr float kWeightScale = 0.125F;
  constexpr std::array<std::uint8_t, 12> kCodes = {
      0x00U, 0x80U, 0x01U, 0x30U, 0x38U, 0x3cU,
      0x40U, 0xb8U, 0x70U, 0x78U, 0x7eU, 0xfeU};
  std::array<std::uint16_t, kColumns> activation{};
  std::array<std::uint8_t, kRows * kColumns> weights{};
  for (std::size_t column = 0; column < kColumns; ++column) {
    activation[column] = encode_bf16(
        static_cast<float>(static_cast<int>(column) - 3) * 0.25F);
  }
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = kCodes[index % kCodes.size()];
  }

  std::array<float, kRows> output{};
  test.expect(q3x::kernels::fp8_gemv_reference_cpu(
                  weights.data(), kWeightScale, activation.data(), kRows,
                  kColumns, output.data()) ==
                  q3x::kernels::GemvStatus::kSuccess,
              "FP8 host GEMV succeeds");
  for (std::size_t row = 0; row < kRows; ++row) {
    double expected = 0.0;
    for (std::size_t column = 0; column < kColumns; ++column) {
      expected +=
          static_cast<double>(q3x::quantization::decode_e4m3fn(
              weights[row * kColumns + column])) *
          static_cast<double>(kWeightScale) *
          static_cast<double>(decode_bf16(activation[column]));
    }
    test.expect_near(output[row], expected, tolerance_for(expected, kColumns),
                     "FP8 double reference row " + std::to_string(row));
  }
}

void test_fp8_static_reference(TestContext& test) {
  constexpr std::size_t kRows = 2U;
  constexpr std::size_t kColumns = 5U;
  constexpr float kWeightScale = 0.5F;
  constexpr float kInputScale = 1.0F;
  // All weights decode to +1 before the 0.5 weight scale. The activations
  // cover ordinary rounding, a nearest-even tie, high saturation, sign, and
  // the minimum E4M3FN subnormal.
  std::array<std::uint8_t, kRows * kColumns> weights{};
  std::fill(weights.begin(), weights.begin() + kColumns, 0x38U);
  std::fill(weights.begin() + kColumns, weights.end(), 0xb8U);
  const std::array<float, kColumns> activation_values = {
      0.26F, 1.0625F, 500.0F, -0.26F, 0.001953125F};
  std::array<std::uint16_t, kColumns> activation{};
  std::transform(activation_values.begin(), activation_values.end(),
                 activation.begin(), encode_bf16);
  std::array<float, kRows> output{};

  test.expect(q3x::kernels::fp8_static_gemv_reference_cpu(
                  weights.data(), kWeightScale, kInputScale,
                  activation.data(), kRows, kColumns, output.data()) ==
                  q3x::kernels::GemvStatus::kSuccess,
              "static W8A8 host GEMV succeeds");
  // torch.float8_e4m3fn oracle values after BF16 storage are
  // [0.25, 1.0, 448.0, -0.25, 0.001953125].
  constexpr double kExpected =
      0.5 * (0.25 + 1.0 + 448.0 - 0.25 + 0.001953125);
  test.expect_near(output[0], kExpected, 1.0e-6,
                   "static W8A8 finite/saturation oracle");
  test.expect_near(output[1], -kExpected, 1.0e-6,
                   "static W8A8 signed weight oracle");

  test.expect(q3x::kernels::fp8_static_gemv_reference_cpu(
                  weights.data(), kWeightScale, 0.0F, activation.data(),
                  kRows, kColumns, output.data()) ==
                  q3x::kernels::GemvStatus::kInvalidArgument,
              "static W8A8 rejects zero input scale");
}

void test_nvfp4_reference(TestContext& test) {
  constexpr std::size_t kRows = 3U;
  constexpr std::size_t kColumns = 48U;
  constexpr float kScale2 = 0.0625F;
  std::vector<std::uint16_t> activation(kColumns);
  std::vector<std::uint8_t> packed(kRows * kColumns / 2U);
  std::vector<std::uint8_t> scales(kRows * kColumns / 16U);
  for (std::size_t column = 0; column < kColumns; ++column) {
    const int centered = static_cast<int>(column % 11U) - 5;
    activation[column] = encode_bf16(static_cast<float>(centered) * 0.125F);
  }
  for (std::size_t index = 0; index < packed.size(); ++index) {
    const std::uint8_t low = static_cast<std::uint8_t>(index & 0x0fU);
    const std::uint8_t high = static_cast<std::uint8_t>(15U - low);
    packed[index] = static_cast<std::uint8_t>(low | (high << 4U));
  }
  constexpr std::array<std::uint8_t, 5> kScaleCodes = {
      0x30U, 0x38U, 0x3cU, 0x40U, 0x44U};
  for (std::size_t index = 0; index < scales.size(); ++index) {
    scales[index] = kScaleCodes[index % kScaleCodes.size()];
  }

  std::vector<float> output(kRows);
  test.expect(q3x::kernels::nvfp4_gemv_reference_cpu(
                  packed.data(), scales.data(), kScale2, activation.data(),
                  kRows, kColumns, output.data()) ==
                  q3x::kernels::GemvStatus::kSuccess,
              "NVFP4 host GEMV succeeds");
  for (std::size_t row = 0; row < kRows; ++row) {
    double expected = 0.0;
    for (std::size_t column = 0; column < kColumns; ++column) {
      const std::uint8_t byte = packed[row * (kColumns / 2U) + column / 2U];
      const std::uint8_t scale =
          scales[row * (kColumns / 16U) + column / 16U];
      expected += static_cast<double>(q3x::quantization::dequantize_nvfp4_value(
                      byte, (column & 1U) != 0U, scale, kScale2)) *
                  static_cast<double>(decode_bf16(activation[column]));
    }
    test.expect_near(output[row], expected, tolerance_for(expected, kColumns),
                     "NVFP4 double reference row " + std::to_string(row));
  }
}

void test_randomized_references(TestContext& test) {
  DeterministicRng rng;
  constexpr std::array<std::uint8_t, 14> kFiniteFp8 = {
      0x00U, 0x80U, 0x01U, 0x07U, 0x08U, 0x30U, 0x38U,
      0x3cU, 0x40U, 0xb8U, 0x70U, 0x78U, 0x7eU, 0xfeU};
  constexpr std::array<std::uint8_t, 6> kFiniteScales = {
      0x28U, 0x30U, 0x38U, 0x3cU, 0x40U, 0x48U};

  for (std::size_t trial = 0; trial < 24U; ++trial) {
    const std::size_t rows = 1U + rng.bounded(8U);
    const std::size_t columns = 1U + rng.bounded(79U);
    std::vector<std::uint16_t> activation(columns);
    std::vector<std::uint16_t> bf16_weights(rows * columns);
    std::vector<std::uint8_t> fp8_weights(rows * columns);
    for (std::uint16_t& value : activation) {
      value = encode_bf16(rng.symmetric());
    }
    for (std::uint16_t& value : bf16_weights) {
      value = encode_bf16(rng.symmetric());
    }
    for (std::uint8_t& value : fp8_weights) {
      value = kFiniteFp8[rng.bounded(kFiniteFp8.size())];
    }

    std::vector<float> bf16_output(rows);
    std::vector<float> fp8_output(rows);
    constexpr float kFp8Scale = 1.0F / 256.0F;
    (void)q3x::kernels::bf16_gemv_reference_cpu(
        bf16_weights.data(), activation.data(), rows, columns,
        bf16_output.data());
    (void)q3x::kernels::fp8_gemv_reference_cpu(
        fp8_weights.data(), kFp8Scale, activation.data(), rows, columns,
        fp8_output.data());
    for (std::size_t row = 0; row < rows; ++row) {
      double bf16_expected = 0.0;
      double fp8_expected = 0.0;
      for (std::size_t column = 0; column < columns; ++column) {
        const double input =
            static_cast<double>(decode_bf16(activation[column]));
        bf16_expected +=
            static_cast<double>(decode_bf16(
                bf16_weights[row * columns + column])) *
            input;
        fp8_expected +=
            static_cast<double>(q3x::quantization::decode_e4m3fn(
                fp8_weights[row * columns + column])) *
            static_cast<double>(kFp8Scale) * input;
      }
      test.expect_near(
          bf16_output[row], bf16_expected,
          tolerance_for(bf16_expected, columns),
          "random BF16 trial " + std::to_string(trial) + " row " +
              std::to_string(row));
      test.expect_near(
          fp8_output[row], fp8_expected,
          tolerance_for(fp8_expected, columns),
          "random FP8 trial " + std::to_string(trial) + " row " +
              std::to_string(row));
    }

    const std::size_t nv_columns = 16U * (1U + rng.bounded(5U));
    std::vector<std::uint16_t> nv_activation(nv_columns);
    std::vector<std::uint8_t> packed(rows * nv_columns / 2U);
    std::vector<std::uint8_t> scales(rows * nv_columns / 16U);
    for (std::uint16_t& value : nv_activation) {
      value = encode_bf16(rng.symmetric());
    }
    for (std::uint8_t& value : packed) {
      value = static_cast<std::uint8_t>(rng.next() & 0xffU);
    }
    for (std::uint8_t& value : scales) {
      value = kFiniteScales[rng.bounded(kFiniteScales.size())];
    }
    constexpr float kScale2 = 1.0F / 128.0F;
    std::vector<float> nv_output(rows);
    (void)q3x::kernels::nvfp4_gemv_reference_cpu(
        packed.data(), scales.data(), kScale2, nv_activation.data(), rows,
        nv_columns, nv_output.data());
    for (std::size_t row = 0; row < rows; ++row) {
      double expected = 0.0;
      for (std::size_t column = 0; column < nv_columns; ++column) {
        const std::uint8_t byte =
            packed[row * (nv_columns / 2U) + column / 2U];
        const std::uint8_t scale =
            scales[row * (nv_columns / 16U) + column / 16U];
        expected += static_cast<double>(
                        q3x::quantization::dequantize_nvfp4_value(
                            byte, (column & 1U) != 0U, scale, kScale2)) *
                    static_cast<double>(decode_bf16(nv_activation[column]));
      }
      test.expect_near(
          nv_output[row], expected, tolerance_for(expected, nv_columns),
          "random NVFP4 trial " + std::to_string(trial) + " row " +
              std::to_string(row));
    }
  }
}

void test_validation_and_nonfinite_values(TestContext& test) {
  using q3x::kernels::GemvStatus;
  constexpr std::size_t kMaximum = std::numeric_limits<std::size_t>::max();
  std::uint16_t bf16 = encode_bf16(1.0F);
  std::uint8_t byte = 0x38U;
  float output = 0.0F;

  test.expect(q3x::kernels::bf16_gemv_reference_cpu(
                  nullptr, nullptr, 0U, 9U, nullptr) == GemvStatus::kSuccess,
              "empty BF16 shape is a no-op");
  test.expect(q3x::kernels::bf16_gemv_reference_cpu(
                  nullptr, &bf16, 1U, 1U, &output) ==
                  GemvStatus::kInvalidArgument,
              "non-empty BF16 GEMV rejects null weights");
  test.expect(q3x::kernels::bf16_gemv_reference_cpu(
                  &bf16, &bf16, kMaximum, 2U, &output) ==
                  GemvStatus::kSizeOverflow,
              "BF16 GEMV rejects dimension overflow");
  test.expect(q3x::kernels::fp8_gemv_reference_cpu(
                  &byte, std::numeric_limits<float>::quiet_NaN(), &bf16, 1U,
                  1U, &output) == GemvStatus::kInvalidArgument,
              "FP8 GEMV rejects NaN scale");
  test.expect(q3x::kernels::fp8_gemv_reference_cpu(
                  &byte, std::numeric_limits<float>::infinity(), &bf16, 1U,
                  1U, &output) == GemvStatus::kInvalidArgument,
              "FP8 GEMV rejects infinite scale");
  test.expect(q3x::kernels::fp8_gemv_reference_cpu(
                  &byte, -1.0F, &bf16, 1U, 1U, &output) ==
                  GemvStatus::kInvalidArgument,
              "FP8 GEMV rejects negative scale");
  test.expect(q3x::kernels::nvfp4_gemv_reference_cpu(
                  &byte, &byte, 1.0F, &bf16, 1U, 17U, &output) ==
                  GemvStatus::kInvalidColumnCount,
              "NVFP4 GEMV rejects non-group-aligned K");
  test.expect(q3x::kernels::nvfp4_gemv_reference_cpu(
                  &byte, &byte, std::numeric_limits<float>::infinity(), &bf16,
                  1U, 16U, &output) == GemvStatus::kInvalidArgument,
              "NVFP4 GEMV rejects infinite scale2");
  test.expect(q3x::kernels::nvfp4_gemv_reference_cpu(
                  &byte, &byte, 1.0F, &bf16, kMaximum / 16U + 1U, 16U,
                  &output) == GemvStatus::kSizeOverflow,
              "NVFP4 GEMV rejects dimension overflow");
  test.expect(std::string(q3x::kernels::gemv_status_string(
                  GemvStatus::kInvalidColumnCount)) ==
                  "NVFP4 column count must be a multiple of 16",
              "GEMV status text is stable");

  const std::uint16_t positive_infinity = encode_bf16(
      std::numeric_limits<float>::infinity());
  const std::uint16_t quiet_nan =
      encode_bf16(std::numeric_limits<float>::quiet_NaN());
  (void)q3x::kernels::bf16_gemv_reference_cpu(
      &bf16, &positive_infinity, 1U, 1U, &output);
  test.expect(std::isinf(output) && output > 0.0F,
              "BF16 infinity propagates through GEMV");
  (void)q3x::kernels::bf16_gemv_reference_cpu(
      &bf16, &quiet_nan, 1U, 1U, &output);
  test.expect(std::isnan(output), "BF16 NaN propagates through GEMV");

  byte = 0x7fU;
  (void)q3x::kernels::fp8_gemv_reference_cpu(
      &byte, 1.0F, &bf16, 1U, 1U, &output);
  test.expect(std::isnan(output), "FP8 encoded NaN propagates through GEMV");

  std::array<std::uint8_t, 8> packed{};
  std::array<std::uint16_t, 16> nv_activation{};
  nv_activation.fill(bf16);
  byte = 0x7fU;
  (void)q3x::kernels::nvfp4_gemv_reference_cpu(
      packed.data(), &byte, 1.0F, nv_activation.data(), 1U, 16U, &output);
  test.expect(std::isnan(output),
              "NVFP4 encoded NaN block scale propagates through GEMV");
}

}  // namespace

int main() {
  TestContext test;
  test_bf16_reference(test);
  test_fp8_reference(test);
  test_fp8_static_reference(test);
  test_nvfp4_reference(test);
  test_randomized_references(test);
  test_validation_and_nonfinite_values(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " host GEMV assertion(s) failed\n";
    return 1;
  }
  std::cout << "Host reference GEMV tests passed\n";
  return 0;
}
