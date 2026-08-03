#include "q3x/runtime/prefill_mlp_factorized_lane_r4_converter.h"

#include "q3x/core/sha256.h"
#include "q3x/quantization/nvfp4.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

namespace quantization = q3x::quantization;
namespace runtime = q3x::runtime;

class Test final {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int result() const noexcept {
    return failures_ == 0 ? 0 : 1;
  }

 private:
  int failures_ = 0;
};

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] float bits_float(const std::uint32_t bits) noexcept {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = float_bits(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  return bits_float(static_cast<std::uint32_t>(value) << 16U);
}

void write_u16_le(const std::uint16_t value,
                  std::uint8_t* const output) noexcept {
  output[0] = static_cast<std::uint8_t>(value);
  output[1] = static_cast<std::uint8_t>(value >> 8U);
}

[[nodiscard]] std::uint16_t read_u16_le(
    const std::uint8_t* const input) noexcept {
  return static_cast<std::uint16_t>(input[0]) |
         static_cast<std::uint16_t>(
             static_cast<std::uint16_t>(input[1]) << 8U);
}

[[nodiscard]] int nearest_even(const float value) noexcept {
  const float base = std::floor(value);
  const float fraction = value - base;
  if (fraction < 0.5F) {
    return static_cast<int>(base);
  }
  if (fraction > 0.5F) {
    return static_cast<int>(base) + 1;
  }
  const int integer = static_cast<int>(base);
  return (integer & 1) == 0 ? integer : integer + 1;
}

void set_source_nibble(std::vector<std::uint8_t>& packed,
                       const std::size_t input_size,
                       const std::size_t row, const std::size_t k,
                       const std::uint8_t nibble) {
  const std::size_t index = row * (input_size / 2U) + k / 2U;
  const unsigned int shift = static_cast<unsigned int>((k & 1U) * 4U);
  const std::uint8_t mask = static_cast<std::uint8_t>(0x0fU << shift);
  packed[index] = static_cast<std::uint8_t>(
      (packed[index] & static_cast<std::uint8_t>(~mask)) |
      static_cast<std::uint8_t>((nibble & 0x0fU) << shift));
}

[[nodiscard]] int signed_nibble(const std::uint8_t raw) noexcept {
  return raw < 8U ? static_cast<int>(raw)
                  : static_cast<int>(raw) - 16;
}

struct Fixture final {
  std::size_t rows = 0U;
  std::size_t input_size = 0U;
  float weight_scale_2 = 0.5F;
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;
  std::vector<float> alpha;
};

[[nodiscard]] Fixture make_fixture(const std::size_t rows,
                                   const std::size_t input_size) {
  Fixture fixture;
  fixture.rows = rows;
  fixture.input_size = input_size;
  fixture.packed.assign(rows * input_size / 2U, 0U);
  fixture.scales.resize(rows * input_size / 16U);
  fixture.alpha.resize(input_size);

  constexpr std::uint8_t kFiniteScales[4] = {
      0x28U,  // 0.25
      0x30U,  // 0.5
      0x38U,  // 1.0
      0x40U,  // 2.0
  };
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t k16 = 0U; k16 < input_size / 16U; ++k16) {
      fixture.scales[row * (input_size / 16U) + k16] =
          kFiniteScales[(row * 5U + k16 * 3U) % 4U];
    }
  }

  const std::size_t lane_input_size =
      input_size / runtime::kPrefillMLPFactorizedLaneR4LaneCount;
  constexpr float kLaneAlpha[4] = {0.5F, 1.0F, 2.0F, 4.0F};
  for (std::size_t k = 0U; k < input_size; ++k) {
    const std::size_t lane = k / lane_input_size;
    fixture.alpha[k] =
        kLaneAlpha[lane] *
        (1.0F + 0.125F * static_cast<float>((k * 11U + 3U) % 7U));
  }

  constexpr std::uint8_t kNvFp4Codes[14] = {
      0x0U, 0x1U, 0x2U, 0x3U, 0x4U, 0x5U, 0x6U,
      0x7U, 0x9U, 0xaU, 0xbU, 0xcU, 0xdU, 0xfU,
  };
  for (std::size_t row = 0U; row < rows; ++row) {
    // The first row of every N64 block is exactly zero.  It probes the
    // positive BF16-one zero-lane convention and N-block boundaries.
    if ((row % 64U) == 0U) {
      continue;
    }
    for (std::size_t k = 0U; k < input_size; ++k) {
      const std::uint8_t nibble =
          kNvFp4Codes[(row * 17U + k * 13U + (k / 64U) * 5U) % 14U];
      set_source_nibble(fixture.packed, input_size, row, k, nibble);
    }
  }
  return fixture;
}

struct Oracle final {
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;
  std::size_t saturated = 0U;
};

[[nodiscard]] Oracle make_oracle(const Fixture& fixture,
                                 const float clip_ratio) {
  Oracle result;
  result.packed.assign(fixture.rows * fixture.input_size / 2U, 0U);
  result.scales.resize(
      fixture.rows * runtime::kPrefillMLPFactorizedLaneR4LaneCount * 2U);
  const std::size_t source_weight_stride = fixture.input_size / 2U;
  const std::size_t source_scale_stride = fixture.input_size / 16U;
  const std::size_t lane_input_size =
      fixture.input_size /
      runtime::kPrefillMLPFactorizedLaneR4LaneCount;
  const std::size_t k64_blocks = fixture.input_size / 64U;

  std::vector<int> logical_codes(fixture.rows * fixture.input_size, 0);
  std::vector<std::uint16_t> logical_scales(
      fixture.rows * runtime::kPrefillMLPFactorizedLaneR4LaneCount);
  for (std::size_t row = 0U; row < fixture.rows; ++row) {
    for (std::size_t lane = 0U;
         lane < runtime::kPrefillMLPFactorizedLaneR4LaneCount; ++lane) {
      const std::size_t lane_begin = lane * lane_input_size;
      float maximum = 0.0F;
      for (std::size_t inner = 0U; inner < lane_input_size; ++inner) {
        const std::size_t k = lane_begin + inner;
        const float value = quantization::dequantize_nvfp4_value(
                                fixture.packed[row * source_weight_stride +
                                               k / 2U],
                                (k & 1U) != 0U,
                                fixture.scales[row * source_scale_stride +
                                               k / 16U],
                                fixture.weight_scale_2) *
                            fixture.alpha[k];
        maximum = std::max(maximum, std::fabs(value));
      }
      const float threshold = maximum * clip_ratio;
      std::uint16_t scale_bits =
          encode_bf16(maximum == 0.0F ? 1.0F : threshold / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      logical_scales[row *
                         runtime::kPrefillMLPFactorizedLaneR4LaneCount +
                     lane] = scale_bits;
      for (std::size_t inner = 0U; inner < lane_input_size; ++inner) {
        const std::size_t k = lane_begin + inner;
        const float value = quantization::dequantize_nvfp4_value(
                                fixture.packed[row * source_weight_stride +
                                               k / 2U],
                                (k & 1U) != 0U,
                                fixture.scales[row * source_scale_stride +
                                               k / 16U],
                                fixture.weight_scale_2) *
                            fixture.alpha[k];
        const float clipped =
            std::max(-threshold, std::min(threshold, value));
        int code = maximum == 0.0F
                       ? 0
                       : nearest_even(clipped / stored_scale);
        code = std::max(-7, std::min(7, code));
        logical_codes[row * fixture.input_size + k] = code;
        if (maximum != 0.0F && std::fabs(value) > threshold &&
            std::abs(code) == 7) {
          ++result.saturated;
        }
      }
    }
  }

  for (std::size_t row = 0U; row < fixture.rows; ++row) {
    const std::size_t n64 = row / 64U;
    const std::size_t local_n = row % 64U;
    for (std::size_t k = 0U; k < fixture.input_size; k += 2U) {
      const int low = logical_codes[row * fixture.input_size + k];
      const int high = logical_codes[row * fixture.input_size + k + 1U];
      const std::uint8_t encoded = static_cast<std::uint8_t>(
          (static_cast<unsigned int>(low) & 0x0fU) |
          ((static_cast<unsigned int>(high) & 0x0fU) << 4U));
      const std::size_t output_index =
          ((n64 * k64_blocks + k / 64U) * 64U + local_n) * 32U +
          (k % 64U) / 2U;
      result.packed[output_index] = encoded;
    }
    for (std::size_t lane = 0U;
         lane < runtime::kPrefillMLPFactorizedLaneR4LaneCount; ++lane) {
      const std::size_t output_index =
          ((n64 * runtime::kPrefillMLPFactorizedLaneR4LaneCount + lane) *
               64U +
           local_n) *
          2U;
      write_u16_le(
          logical_scales[row *
                             runtime::kPrefillMLPFactorizedLaneR4LaneCount +
                         lane],
          result.scales.data() + output_index);
    }
  }
  return result;
}

[[nodiscard]] runtime::PrefillMLPFactorizedLaneR4ConverterDiagnostic
run_transform(const Fixture& fixture, const double clip_ratio,
              std::vector<std::uint8_t>& packed,
              std::vector<std::uint8_t>& scales) {
  packed.assign(fixture.rows * fixture.input_size / 2U, 0xa5U);
  scales.assign(
      fixture.rows * runtime::kPrefillMLPFactorizedLaneR4LaneCount * 2U,
      0x5aU);
  return runtime::
      transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
          fixture.packed.data(), fixture.packed.size(), fixture.scales.data(),
          fixture.scales.size(), fixture.weight_scale_2, fixture.rows,
          fixture.input_size, fixture.alpha.data(), fixture.alpha.size(),
          clip_ratio, packed.data(), packed.size(), scales.data(),
          scales.size());
}

void test_direct_layout_and_lane_boundaries(Test& test) {
  const Fixture fixture = make_fixture(128U, 256U);
  constexpr float kClip = 0.75F;
  const Oracle expected = make_oracle(fixture, kClip);
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;
  const auto diagnostic = run_transform(fixture, kClip, packed, scales);
  test.expect(diagnostic.ok(),
              "direct original-NVFP4 R4 transform succeeds");
  test.expect(packed == expected.packed,
              "R4 codes exactly preserve [N/64][K/64][64][32] order");
  test.expect(scales == expected.scales,
              "R4 scales exactly preserve [N/64][4][64] order");
  test.expect(expected.saturated != 0U,
              "explicit clip produces signed A4 saturation probes");

  for (const std::size_t row : {0U, 64U}) {
    const std::size_t n64 = row / 64U;
    for (std::size_t lane = 0U;
         lane < runtime::kPrefillMLPFactorizedLaneR4LaneCount; ++lane) {
      const std::size_t scale_index =
          ((n64 * runtime::kPrefillMLPFactorizedLaneR4LaneCount + lane) *
               64U) *
          2U;
      test.expect(read_u16_le(scales.data() + scale_index) == 0x3f80U,
                  "zero lane stores positive BF16 one");
    }
    for (std::size_t k64 = 0U; k64 < fixture.input_size / 64U; ++k64) {
      const std::size_t begin =
          ((n64 * (fixture.input_size / 64U) + k64) * 64U) * 32U;
      test.expect(
          std::all_of(packed.begin() + static_cast<std::ptrdiff_t>(begin),
                      packed.begin() +
                          static_cast<std::ptrdiff_t>(begin + 32U),
                      [](const std::uint8_t value) { return value == 0U; }),
          "the zero row stays zero across every consumer K64 block");
    }
  }

  // Adjacent coordinates on both sides of all three R4 lane seams must map
  // to the exact oracle bytes; no lane may borrow its neighbor's scale.
  for (const std::size_t seam : {64U, 128U, 192U}) {
    const std::size_t row = 1U;
    const std::size_t n64 = row / 64U;
    const std::size_t local_n = row % 64U;
    for (const std::size_t k : {seam - 2U, seam}) {
      const std::size_t offset =
          ((n64 * (fixture.input_size / 64U) + k / 64U) * 64U +
           local_n) *
              32U +
          (k % 64U) / 2U;
      test.expect(packed[offset] == expected.packed[offset],
                  "consumer bytes are exact at an R4 lane boundary");
    }
  }
}

void test_nearest_even_and_clip(Test& test) {
  Fixture fixture = make_fixture(64U, 256U);
  std::fill(fixture.packed.begin(), fixture.packed.end(), 0U);
  std::fill(fixture.scales.begin(), fixture.scales.end(), 0x38U);
  std::fill(fixture.alpha.begin(), fixture.alpha.end(), 1.0F);
  fixture.weight_scale_2 = 1.0F;

  // E2M1(+/-1) times alpha creates exact .5 ties while +7 establishes an
  // exactly representable BF16 scale of one for lane zero.
  constexpr float kAlpha[9] = {7.0F, 0.5F, 1.5F, 2.5F, 3.5F,
                                0.5F, 1.5F, 2.5F, 3.5F};
  constexpr std::uint8_t kSource[9] = {
      0x2U, 0x2U, 0x2U, 0x2U, 0x2U,
      0xaU, 0xaU, 0xaU, 0xaU,
  };
  for (std::size_t k = 0U; k < 9U; ++k) {
    fixture.alpha[k] = kAlpha[k];
    set_source_nibble(fixture.packed, fixture.input_size, 0U, k,
                      kSource[k]);
  }

  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;
  auto diagnostic = run_transform(fixture, 1.0, packed, scales);
  test.expect(diagnostic.ok() && read_u16_le(scales.data()) == 0x3f80U,
              "unclipped tie probe uses exact BF16-one lane scale");
  constexpr int kExpectedCodes[9] = {7, 0, 2, 2, 4, 0, -2, -2, -4};
  for (std::size_t k = 0U; k < 9U; ++k) {
    const std::uint8_t byte = packed[k / 2U];
    const std::uint8_t raw = static_cast<std::uint8_t>(
        (byte >> ((k & 1U) * 4U)) & 0x0fU);
    test.expect(signed_nibble(raw) == kExpectedCodes[k],
                "signed A4 uses deterministic ties-to-even");
  }

  diagnostic = run_transform(fixture, 0.5, packed, scales);
  test.expect(diagnostic.ok() && read_u16_le(scales.data()) == 0x3f00U,
              "explicit 0.5 clip halves the exact lane scale");
  test.expect(signed_nibble(packed[0] & 0x0fU) == 7,
              "explicit clipping saturates the lane maximum to +7");
}

void test_pinned_model_shapes(Test& test) {
  {
    Fixture gate = make_fixture(64U, 5'120U);
    std::fill(gate.alpha.begin(), gate.alpha.end(), 1.0F);
    const Oracle expected = make_oracle(gate, 1.0F);
    std::vector<std::uint8_t> packed;
    std::vector<std::uint8_t> scales;
    const auto diagnostic = run_transform(gate, 1.0, packed, scales);
    test.expect(diagnostic.ok() && packed == expected.packed &&
                    scales == expected.scales,
                "Gate K5120 direct R4 conversion is exact with alpha=1");
    test.expect(gate.input_size / 4U == 1'280U,
                "Gate R4 lanes close at K1280 boundaries");
    const std::size_t row = 1U;
    for (const std::size_t seam : {1'280U, 2'560U, 3'840U}) {
      for (const std::size_t k : {seam - 2U, seam}) {
        const std::size_t offset =
            ((k / 64U) * 64U + row) * 32U + (k % 64U) / 2U;
        test.expect(packed[offset] == expected.packed[offset],
                    "Gate codes are exact across each K1280 lane seam");
      }
    }
  }

  {
    Fixture down = make_fixture(64U, 17'408U);
    const std::size_t lane_input_size = down.input_size / 4U;
    const std::size_t zero_lane = 2U;
    const std::size_t zero_begin = zero_lane * lane_input_size;
    for (std::size_t k = zero_begin; k < zero_begin + lane_input_size; ++k) {
      set_source_nibble(down.packed, down.input_size, 1U, k, 0U);
    }
    const Oracle expected = make_oracle(down, 0.875F);
    std::vector<std::uint8_t> packed;
    std::vector<std::uint8_t> scales;
    const auto diagnostic = run_transform(down, 0.875, packed, scales);
    test.expect(diagnostic.ok() && packed == expected.packed &&
                    scales == expected.scales,
                "Down K17408 direct R4 conversion is exact with nonuniform alpha");
    test.expect(lane_input_size == 4'352U,
                "Down R4 lanes close at K4352 boundaries");
    const std::size_t zero_scale_offset =
        (zero_lane * 64U + 1U) * sizeof(std::uint16_t);
    test.expect(read_u16_le(scales.data() + zero_scale_offset) == 0x3f80U,
                "a single zero Down lane stores BF16 one independently");
    for (std::size_t k64 = zero_begin / 64U;
         k64 < (zero_begin + lane_input_size) / 64U; ++k64) {
      const std::size_t begin = (k64 * 64U + 1U) * 32U;
      test.expect(
          std::all_of(packed.begin() + static_cast<std::ptrdiff_t>(begin),
                      packed.begin() +
                          static_cast<std::ptrdiff_t>(begin + 32U),
                      [](const std::uint8_t value) { return value == 0U; }),
          "a zero Down lane cannot borrow a neighboring lane scale");
    }
    for (const std::size_t seam : {4'352U, 8'704U, 13'056U}) {
      for (const std::size_t k : {seam - 2U, seam}) {
        const std::size_t offset =
            ((k / 64U) * 64U + 1U) * 32U + (k % 64U) / 2U;
        test.expect(packed[offset] == expected.packed[offset],
                    "Down codes are exact across each K4352 lane seam");
      }
    }
  }
}

void test_inverse_alpha_metadata(Test& test) {
  const Fixture fixture = make_fixture(64U, 256U);
  const auto result = runtime::build_prefill_mlp_factorized_lane_r4_metadata(
      fixture.alpha.data(), fixture.alpha.size());
  test.expect(static_cast<bool>(result),
              "R4 alpha builds authenticated inverse-alpha metadata");
  if (!result) {
    return;
  }
  test.expect(
      result.metadata.bytes.size() ==
          runtime::kPrefillA4FactorizedLaneMetadataHeaderBytes +
              fixture.input_size * sizeof(float),
      "R4 metadata has the exact v4 header plus FP32 inverse-alpha payload");
  std::vector<std::uint8_t> expected_inverse_bytes(
      fixture.input_size * sizeof(float));
  for (std::size_t k = 0U; k < fixture.input_size; ++k) {
    const std::uint32_t bits = float_bits(1.0F / fixture.alpha[k]);
    expected_inverse_bytes[k * 4U] = static_cast<std::uint8_t>(bits);
    expected_inverse_bytes[k * 4U + 1U] =
        static_cast<std::uint8_t>(bits >> 8U);
    expected_inverse_bytes[k * 4U + 2U] =
        static_cast<std::uint8_t>(bits >> 16U);
    expected_inverse_bytes[k * 4U + 3U] =
        static_cast<std::uint8_t>(bits >> 24U);
  }
  test.expect(
      std::equal(expected_inverse_bytes.begin(),
                 expected_inverse_bytes.end(),
                 result.metadata.bytes.begin() +
                     static_cast<std::ptrdiff_t>(
                         runtime::kPrefillA4FactorizedLaneMetadataHeaderBytes)),
      "metadata payload is the exact independent FP32-LE reciprocal byte sequence");
  q3x::core::Sha256 hasher;
  const bool hash_updated = hasher.update(expected_inverse_bytes.data(),
                                          expected_inverse_bytes.size());
  const auto expected_digest = hasher.finalize();
  test.expect(
      hash_updated &&
          expected_digest.bytes == result.metadata.inverse_alpha_sha256 &&
          std::equal(
              expected_digest.bytes.begin(), expected_digest.bytes.end(),
              result.metadata.bytes.begin() +
                  static_cast<std::ptrdiff_t>(
                      runtime::kPrefillA4FactorizedLaneMetadataDigestOffset)),
      "metadata digest independently binds the exact reciprocal bytes");
  const auto parsed = runtime::parse_prefill_mlp_factorized_lane_metadata(
      result.metadata.bytes.data(), result.metadata.bytes.size(),
      runtime::kPrefillMLPFactorizedLaneR4LaneCount, fixture.input_size);
  test.expect(parsed && parsed.lane_count == 4U &&
                  parsed.input_size == fixture.input_size,
              "R4 metadata parses only with the authenticated lane/K shape");
  if (parsed) {
    for (std::size_t k = 0U; k < fixture.input_size; ++k) {
      test.expect(parsed.inverse_alpha[k] == 1.0F / fixture.alpha[k],
                  "metadata stores the exact FP32 reciprocal of alpha");
    }
    for (const std::size_t k : {63U, 64U, 127U, 128U, 191U, 192U}) {
      test.expect(parsed.inverse_alpha[k] == 1.0F / fixture.alpha[k],
                  "inverse-alpha remains exact across every R4 lane seam");
    }
  }
  test.expect(
      !runtime::parse_prefill_mlp_factorized_lane_metadata(
          result.metadata.bytes.data(), result.metadata.bytes.size(), 2U,
          fixture.input_size),
      "R4 metadata rejects the wrong expected lane count");
  test.expect(
      !runtime::parse_prefill_mlp_factorized_lane_metadata(
          result.metadata.bytes.data(), result.metadata.bytes.size(), 4U,
          fixture.input_size * 2U),
      "R4 metadata rejects the wrong expected K");
}

void test_fail_closed(Test& test) {
  Fixture fixture = make_fixture(64U, 256U);
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;
  auto diagnostic = run_transform(fixture, 1.0, packed, scales);
  test.expect(diagnostic.ok(), "failure fixture starts valid");

  auto invoke = [&](const float* alpha, const std::size_t alpha_count,
                    const float tensor_scale, const double clip,
                    const std::size_t packed_output_bytes) {
    return runtime::
        transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
            fixture.packed.data(), fixture.packed.size(),
            fixture.scales.data(), fixture.scales.size(), tensor_scale,
            fixture.rows, fixture.input_size, alpha, alpha_count, clip,
            packed.data(), packed_output_bytes, scales.data(), scales.size());
  };

  diagnostic = invoke(fixture.alpha.data(), fixture.alpha.size() - 1U,
                      fixture.weight_scale_2, 1.0, packed.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidAlpha,
              "short alpha fails closed");
  fixture.alpha[63U] = 0.0F;
  diagnostic = invoke(fixture.alpha.data(), fixture.alpha.size(),
                      fixture.weight_scale_2, 1.0, packed.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidAlpha &&
                  diagnostic.index == 63U,
              "zero alpha fails at its exact channel");
  test.expect(!runtime::build_prefill_mlp_factorized_lane_r4_metadata(
                   fixture.alpha.data(), fixture.alpha.size()),
              "metadata applies the same positive-alpha gate");
  fixture.alpha[63U] = 1.0F;
  fixture.alpha[64U] = std::numeric_limits<float>::denorm_min();
  diagnostic = invoke(fixture.alpha.data(), fixture.alpha.size(),
                      fixture.weight_scale_2, 1.0, packed.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidAlpha &&
                  diagnostic.index == 64U,
              "alpha whose reciprocal overflows fails at the lane boundary");
  fixture.alpha[64U] = 1.0F;

  diagnostic = invoke(fixture.alpha.data(), fixture.alpha.size(), -1.0F,
                      1.0, packed.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidSourceValue,
              "negative original tensor scale fails closed");
  diagnostic = invoke(fixture.alpha.data(), fixture.alpha.size(),
                      fixture.weight_scale_2, 0.0, packed.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidArgument,
              "implicit/zero clip ratio fails closed");
  diagnostic = invoke(fixture.alpha.data(), fixture.alpha.size(),
                      fixture.weight_scale_2, 1.0, packed.size() - 1U);
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kBufferSizeMismatch,
              "short consumer output fails closed");

  diagnostic = runtime::
      transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
          fixture.packed.data(), fixture.packed.size(), fixture.scales.data(),
          fixture.scales.size(), fixture.weight_scale_2, fixture.rows,
          fixture.input_size, fixture.alpha.data(), fixture.alpha.size(), 1.0,
          fixture.packed.data(), fixture.packed.size(), scales.data(),
          scales.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidArgument &&
                  diagnostic.context == "r4_direct.alias",
              "source/output alias fails closed before conversion");
  diagnostic = runtime::
      transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
          fixture.packed.data(), fixture.packed.size(), fixture.packed.data(),
          fixture.scales.size(), fixture.weight_scale_2, fixture.rows,
          fixture.input_size, fixture.alpha.data(), fixture.alpha.size(), 1.0,
          packed.data(), packed.size(), scales.data(), scales.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidArgument &&
                  diagnostic.context == "r4_direct.alias",
              "the two original-checkpoint input spans must be disjoint");
  diagnostic = runtime::
      transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
          fixture.packed.data(), fixture.packed.size(), fixture.scales.data(),
          fixture.scales.size(), fixture.weight_scale_2, fixture.rows,
          fixture.input_size,
          reinterpret_cast<const float*>(fixture.packed.data()),
          fixture.alpha.size(), 1.0, packed.data(), packed.size(),
          scales.data(), scales.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidArgument &&
                  diagnostic.context == "r4_direct.alias",
              "original-checkpoint input/alpha alias fails closed");
  diagnostic = runtime::
      transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
          fixture.packed.data(), fixture.packed.size(), fixture.scales.data(),
          fixture.scales.size(), fixture.weight_scale_2, fixture.rows,
          fixture.input_size, fixture.alpha.data(), fixture.alpha.size(), 1.0,
          packed.data(), packed.size(), packed.data(), scales.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidArgument &&
                  diagnostic.context == "r4_direct.alias",
              "packed/scale output alias fails closed before conversion");

  fixture.scales[0] = 0x7fU;
  diagnostic = invoke(fixture.alpha.data(), fixture.alpha.size(),
                      fixture.weight_scale_2, 1.0, packed.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidSourceValue &&
                  diagnostic.index == 0U,
              "NaN original E4M3 block scale fails at its exact group");
  fixture.scales[0] = 0x38U;

  diagnostic = runtime::
      transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
          fixture.packed.data(), fixture.packed.size(), fixture.scales.data(),
          fixture.scales.size(), fixture.weight_scale_2, 63U,
          fixture.input_size, fixture.alpha.data(), fixture.alpha.size(), 1.0,
          packed.data(), packed.size(), scales.data(), scales.size());
  test.expect(diagnostic.code == runtime::
                                     PrefillMLPFactorizedLaneR4ConverterErrorCode::
                                         kInvalidShape,
              "partial N64 source fails closed before byte interpretation");

  std::vector<float> bad_k_alpha(192U, 1.0F);
  test.expect(!runtime::build_prefill_mlp_factorized_lane_r4_metadata(
                   bad_k_alpha.data(), bad_k_alpha.size()),
              "metadata rejects four lanes that are not individually K64 aligned");
  test.expect(runtime::to_string(
                  runtime::PrefillMLPFactorizedLaneR4ConverterErrorCode::
                      kInvalidAlpha) == "invalid_alpha",
              "R4 diagnostics have a stable textual code");
}

}  // namespace

int main() {
  static_assert(runtime::kPrefillMLPFactorizedLaneR4LaneCount == 4U);
  static_assert(runtime::kPrefillMLPFactorizedLaneR4FactorScheme ==
                "calibrated_alpha_f32_v1");

  Test test;
  test_direct_layout_and_lane_boundaries(test);
  test_nearest_even_and_clip(test);
  test_pinned_model_shapes(test);
  test_inverse_alpha_metadata(test);
  test_fail_closed(test);
  if (test.result() != 0) {
    return test.result();
  }
  std::cout << "direct checkpoint factorized-lane R4 converter passed\n";
  return 0;
}
