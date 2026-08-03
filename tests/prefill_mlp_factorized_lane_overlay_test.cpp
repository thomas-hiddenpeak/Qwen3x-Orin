#include "q3x/runtime/prefill_mlp_factorized_lane_overlay.h"

#include "q3x/core/sha256.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace runtime = q3x::runtime;

class Test final {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int result() const noexcept { return failures_ == 0 ? 0 : 1; }

 private:
  int failures_ = 0;
};

constexpr auto kR1Plan =
    runtime::prefill_mlp_factorized_lane_overlay_layout_plan(1U);
constexpr auto kR2Plan =
    runtime::prefill_mlp_factorized_lane_overlay_layout_plan(2U);
constexpr auto kR4Plan =
    runtime::prefill_mlp_factorized_lane_overlay_layout_plan(4U);
static_assert(kR1Plan.valid() && kR2Plan.valid() && kR4Plan.valid());
static_assert(kR1Plan.gate.projection_bytes == 44'620'032ULL);
static_assert(kR1Plan.down.projection_bytes == 44'644'608ULL);
static_assert(kR1Plan.layer_bytes == 133'884'672ULL);
static_assert(kR1Plan.payload_bytes == 8'568'619'008ULL);
static_assert(kR2Plan.gate.projection_bytes == 44'654'848ULL);
static_assert(kR2Plan.down.projection_bytes == 44'654'848ULL);
static_assert(kR2Plan.layer_bytes == 133'964'544ULL);
static_assert(kR2Plan.payload_bytes == 8'573'730'816ULL);
static_assert(kR4Plan.gate.projection_bytes == 44'724'480ULL);
static_assert(kR4Plan.down.projection_bytes == 44'675'328ULL);
static_assert(kR4Plan.layer_bytes == 134'124'288ULL);
static_assert(kR4Plan.payload_bytes == 8'583'954'432ULL);
static_assert(
    runtime::prefill_mlp_factorized_lane_overlay_layout_plan(3U).error ==
    runtime::PrefillMLPFactorizedLaneOverlayPlanError::
        kUnsupportedLaneCount);
static_assert(
    runtime::prefill_mlp_factorized_lane_overlay_layout_plan(1U, 128U)
        .error ==
    runtime::PrefillMLPFactorizedLaneOverlayPlanError::kInvalidAlignment);
static_assert(
    runtime::prefill_mlp_factorized_lane_overlay_layout_plan(
        1U, std::uint64_t{1U} << 63U)
        .error ==
    runtime::PrefillMLPFactorizedLaneOverlayPlanError::kArithmeticOverflow);
static_assert(runtime::prefill_mlp_factorized_lane_qualification_role(1U) ==
              runtime::PrefillMLPFactorizedLaneQualificationRole::
                  kPerformanceUpperBound);
static_assert(runtime::prefill_mlp_factorized_lane_qualification_role(2U) ==
              runtime::PrefillMLPFactorizedLaneQualificationRole::
                  kStructuralOnly);
static_assert(runtime::prefill_mlp_factorized_lane_qualification_role(4U) ==
              runtime::PrefillMLPFactorizedLaneQualificationRole::
                  kQualityCandidate);

void write_u32_le(const std::uint32_t value,
                  std::uint8_t* const output) noexcept {
  output[0] = static_cast<std::uint8_t>(value);
  output[1] = static_cast<std::uint8_t>(value >> 8U);
  output[2] = static_cast<std::uint8_t>(value >> 16U);
  output[3] = static_cast<std::uint8_t>(value >> 24U);
}

void write_u64_le(const std::uint64_t value,
                  std::uint8_t* const output) noexcept {
  for (std::size_t index = 0U; index < 8U; ++index) {
    output[index] =
        static_cast<std::uint8_t>(value >> (index * 8U));
  }
}

[[nodiscard]] std::uint32_t read_u32_le(
    const std::uint8_t* const input) noexcept {
  return static_cast<std::uint32_t>(input[0]) |
         (static_cast<std::uint32_t>(input[1]) << 8U) |
         (static_cast<std::uint32_t>(input[2]) << 16U) |
         (static_cast<std::uint32_t>(input[3]) << 24U);
}

[[nodiscard]] std::uint64_t read_u64_le(
    const std::uint8_t* const input) noexcept {
  std::uint64_t result = 0U;
  for (std::size_t index = 0U; index < 8U; ++index) {
    result |= static_cast<std::uint64_t>(input[index]) << (index * 8U);
  }
  return result;
}

void rehash_inverse_alpha(std::vector<std::uint8_t>& bytes) {
  q3x::core::Sha256 hasher;
  const std::size_t payload_offset = static_cast<std::size_t>(
      runtime::kPrefillA4FactorizedLaneMetadataHeaderBytes);
  (void)hasher.update(bytes.data() + payload_offset,
                      bytes.size() - payload_offset);
  const auto digest = hasher.finalize();
  std::copy(
      digest.bytes.begin(), digest.bytes.end(),
      bytes.begin() + static_cast<std::ptrdiff_t>(
                          runtime::kPrefillA4FactorizedLaneMetadataDigestOffset));
}

void test_layout(Test& test,
                 const runtime::PrefillMLPFactorizedLaneOverlayLayoutPlan&
                     plan) {
  test.expect(plan.valid() && plan.projection_count == 192U,
              "layout fixes the complete 192-projection inventory");
  test.expect(plan.gate_offset_in_layer == 0U &&
                  plan.up_offset_in_layer == plan.gate.projection_bytes &&
                  plan.down_offset_in_layer ==
                      plan.gate.projection_bytes + plan.up.projection_bytes,
              "layer-major Gate/Up/Down offsets are gap-free and explicit");
  for (std::uint32_t layer = 0U;
       layer < runtime::kPrefillMLPFactorizedLaneLayerCount; ++layer) {
    for (std::uint32_t position = 0U;
         position < runtime::kPrefillMLPFactorizedLaneProjectionsPerLayer;
         ++position) {
      const auto family =
          static_cast<runtime::PrefillMLPFactorizedLaneProjectionFamily>(
              position);
      const std::uint64_t offset =
          runtime::prefill_mlp_factorized_lane_projection_absolute_offset(
              plan, layer, family);
      test.expect(offset !=
                          runtime::kPrefillA4FactorizedLaneInvalidOffset &&
                      offset % plan.alignment == 0U &&
                      offset < plan.payload_bytes,
                  "every projection has an in-range 256-byte-aligned offset");
      test.expect(
          runtime::prefill_mlp_factorized_lane_projection_ordinal(layer,
                                                                  family) ==
              layer * 3U + position,
          "ordinal is fixed layer-major Gate/Up/Down");
    }
  }
  const std::uint64_t final_down =
      runtime::prefill_mlp_factorized_lane_projection_absolute_offset(
          plan, 63U,
          runtime::PrefillMLPFactorizedLaneProjectionFamily::kDown);
  test.expect(final_down + plan.down.projection_bytes == plan.payload_bytes,
              "last Down projection closes the exact aggregate payload");
  test.expect(
      runtime::prefill_mlp_factorized_lane_projection_absolute_offset(
          plan, 64U,
          runtime::PrefillMLPFactorizedLaneProjectionFamily::kGate) ==
          runtime::kPrefillA4FactorizedLaneInvalidOffset,
      "out-of-range layer fails closed");
  test.expect(
      runtime::prefill_mlp_factorized_lane_projection_absolute_offset(
          plan, 0U,
          static_cast<runtime::PrefillMLPFactorizedLaneProjectionFamily>(
              0xffU)) == runtime::kPrefillA4FactorizedLaneInvalidOffset,
      "unknown projection family fails closed");
}

[[nodiscard]] std::vector<float> make_inverse_alpha(
    const std::size_t count) {
  std::vector<float> values(count);
  for (std::size_t index = 0U; index < count; ++index) {
    values[index] = 0.25F +
                    static_cast<float>((index * 37U) % 257U) / 128.0F;
  }
  return values;
}

void test_metadata_round_trip(Test& test, const std::uint32_t lane_count,
                              const std::size_t input_size) {
  const std::vector<float> values = make_inverse_alpha(input_size);
  const auto serialized =
      runtime::serialize_prefill_mlp_factorized_lane_metadata(
          lane_count, values.data(), values.size());
  test.expect(serialized.valid(), "valid inverse-alpha metadata serializes");
  if (!serialized) {
    return;
  }
  test.expect(
      serialized.bytes.size() ==
          runtime::kPrefillA4FactorizedLaneMetadataHeaderBytes +
              input_size * sizeof(float),
      "serializer emits exact metadata bytes without projection padding");
  test.expect(
      std::equal(
          runtime::kPrefillA4FactorizedLaneMetadataMagic,
          runtime::kPrefillA4FactorizedLaneMetadataMagic + 8U,
          reinterpret_cast<const char*>(serialized.bytes.data())) &&
          read_u32_le(serialized.bytes.data() + 8U) == 4U &&
          read_u32_le(serialized.bytes.data() + 12U) == 0U &&
          read_u32_le(serialized.bytes.data() + 16U) == lane_count &&
          read_u32_le(serialized.bytes.data() + 20U) == 1U &&
          read_u64_le(serialized.bytes.data() + 24U) == input_size,
      "metadata header is canonical little-endian v4");

  const auto parsed = runtime::parse_prefill_mlp_factorized_lane_metadata(
      serialized.bytes.data(), serialized.bytes.size(), lane_count,
      input_size);
  test.expect(parsed.valid() && parsed.lane_count == lane_count &&
                  parsed.input_size == input_size &&
                  parsed.inverse_alpha_sha256 ==
                      serialized.inverse_alpha_sha256 &&
                  parsed.inverse_alpha == values,
              "parser validates the digest and restores exact FP32 values");
}

void test_metadata_rejections(Test& test) {
  std::vector<float> values = make_inverse_alpha(17'408U);
  const auto original =
      runtime::serialize_prefill_mlp_factorized_lane_metadata(
          4U, values.data(), values.size());
  test.expect(original.valid(), "mutation fixture serializes");
  if (!original) {
    return;
  }
  const auto parse = [](const std::vector<std::uint8_t>& bytes) {
    return runtime::parse_prefill_mlp_factorized_lane_metadata(
        bytes.data(), bytes.size(), 4U, 17'408U);
  };

  std::vector<std::uint8_t> mutated = original.bytes;
  mutated[0] ^= 1U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kInvalidMagic,
              "magic mutation fails closed");

  mutated = original.bytes;
  mutated[8] = 5U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kUnsupportedVersion,
              "version mutation fails closed");

  mutated = original.bytes;
  mutated[20] = 2U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kUnsupportedEncoding,
              "encoding mutation fails closed");

  mutated = original.bytes;
  write_u32_le(2U, mutated.data() + 16U);
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kExpectedShapeMismatch,
              "structurally valid lane header mutation fails expected binding");

  mutated = original.bytes;
  write_u64_le(17'664U, mutated.data() + 24U);
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kExpectedShapeMismatch,
              "structurally valid K header mutation fails expected binding");

  mutated = original.bytes;
  mutated[32] ^= 0x80U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kDigestMismatch,
              "digest mutation fails closed");

  mutated = original.bytes;
  mutated[64U + 17U] ^= 1U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kDigestMismatch,
              "inverse-alpha byte mutation fails closed");

  mutated = original.bytes;
  mutated.pop_back();
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kInvalidByteLength,
              "truncated metadata fails closed");
  mutated = original.bytes;
  mutated.push_back(0U);
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kInvalidByteLength,
              "trailing metadata bytes fail closed");

  // Construct an internally digest-consistent zero.  Digest integrity alone
  // must not bypass the finite-positive numerical contract.
  mutated = original.bytes;
  std::fill_n(mutated.begin() + 64, sizeof(float), 0U);
  rehash_inverse_alpha(mutated);
  test.expect(parse(mutated).error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kInvalidInverseAlpha,
              "digest-consistent nonpositive alpha still fails closed");

  const auto null_input =
      runtime::serialize_prefill_mlp_factorized_lane_metadata(1U, nullptr,
                                                              5'120U);
  test.expect(null_input.error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::kNullInput,
              "null non-empty inverse-alpha input is rejected");
  const auto bad_lane =
      runtime::serialize_prefill_mlp_factorized_lane_metadata(
          3U, values.data(), 5'120U);
  test.expect(
      bad_lane.error ==
          runtime::PrefillMLPFactorizedLaneMetadataError::
              kUnsupportedLaneCount,
      "unsupported lane count is rejected");
  const auto bad_shape =
      runtime::serialize_prefill_mlp_factorized_lane_metadata(
          4U, values.data(), 64U);
  test.expect(bad_shape.error ==
                  runtime::PrefillMLPFactorizedLaneMetadataError::
                      kInvalidInputSize,
              "lane segment not divisible by K64 is rejected");

  values[0] = std::numeric_limits<float>::infinity();
  test.expect(
      runtime::serialize_prefill_mlp_factorized_lane_metadata(
          4U, values.data(), values.size())
              .error == runtime::PrefillMLPFactorizedLaneMetadataError::
                            kInvalidInverseAlpha,
      "nonfinite inverse alpha is rejected before publication");
  values[0] = -1.0F;
  test.expect(
      runtime::serialize_prefill_mlp_factorized_lane_metadata(
          4U, values.data(), values.size())
              .error == runtime::PrefillMLPFactorizedLaneMetadataError::
                            kInvalidInverseAlpha,
      "nonpositive inverse alpha is rejected before publication");

  std::array<std::uint8_t, 64U> huge_header{};
  std::copy_n(
      reinterpret_cast<const std::uint8_t*>(
          runtime::kPrefillA4FactorizedLaneMetadataMagic),
      8U, huge_header.data());
  write_u32_le(4U, huge_header.data() + 8U);
  write_u32_le(0U, huge_header.data() + 12U);
  write_u32_le(1U, huge_header.data() + 16U);
  write_u32_le(1U, huge_header.data() + 20U);
  write_u64_le(std::numeric_limits<std::uint64_t>::max() - 63U,
               huge_header.data() + 24U);
  test.expect(
      runtime::parse_prefill_mlp_factorized_lane_metadata(
          huge_header.data(), huge_header.size(), 1U,
          std::numeric_limits<std::uint64_t>::max() - 63U)
              .error == runtime::PrefillMLPFactorizedLaneMetadataError::
                            kArithmeticOverflow,
      "metadata byte-count overflow fails closed before allocation");
}

void test_publication_binding_vocabulary(Test& test) {
  runtime::PrefillMLPFactorizedLaneBaseK256Binding base;
  base.physical_layout =
      std::string(runtime::kPrefillMLPFactorizedLaneRequiredBaseK256Layout);
  base.manifest_sha256 = std::string(64U, '1');
  base.policy_sha256 = std::string(64U, '2');
  base.payload_sha256 = std::string(64U, '3');

  runtime::PrefillMLPFactorizedLaneProjectionCalibrationBinding calibration;
  calibration.ordinal = 191U;
  calibration.weight_clip_ratio = 0.75;
  calibration.activation_clip_ratio = 1.0;
  calibration.factor_source.path = "calibration/down.63.inverse-alpha.f32";
  calibration.factor_source.sha256 = std::string(64U, '4');
  calibration.factor_source.element_count = 17'408U;

  runtime::PrefillMLPFactorizedLaneOverlayPolicyBinding policy;
  policy.physical_layout =
      std::string(runtime::kPrefillMLPFactorizedLaneOverlayLayout);
  policy.required_base_k256 = base;
  policy.lane_count = 4U;
  policy.projections.push_back(calibration);

  runtime::PrefillMLPFactorizedLaneOverlayReceiptBinding receipt;
  receipt.physical_layout = policy.physical_layout;
  receipt.required_base_k256 = base;
  receipt.lane_count = policy.lane_count;
  receipt.payload.path = "weights-mlp-factorized-r4-v4.bin";
  receipt.payload.sha256 = std::string(64U, '5');
  receipt.payload.bytes = kR4Plan.payload_bytes;
  receipt.projection_count =
      runtime::kPrefillMLPFactorizedLaneProjectionCount;

  test.expect(
      policy.required_base_k256.physical_layout ==
              runtime::kPrefillMLPFactorizedLaneRequiredBaseK256Layout &&
          policy.projections.front().weight_clip_ratio == 0.75 &&
          policy.projections.front().activation_clip_ratio == 1.0 &&
          policy.projections.front().factor_source.path ==
              "calibration/down.63.inverse-alpha.f32" &&
          policy.projections.front().factor_source.sha256.size() == 64U &&
          policy.projections.front().factor_source.element_count == 17'408U &&
          !receipt.production_residency_eligible &&
          receipt.payload.bytes == kR4Plan.payload_bytes,
      "future policy/receipt vocabulary binds base, clips, factors, and payload");
}

}  // namespace

int main() {
  Test test;
  test_layout(test, kR1Plan);
  test_layout(test, kR2Plan);
  test_layout(test, kR4Plan);
  test_metadata_round_trip(test, 1U, 5'120U);
  test_metadata_round_trip(test, 2U, 5'120U);
  test_metadata_round_trip(test, 4U, 17'408U);
  test_metadata_rejections(test);
  test_publication_binding_vocabulary(test);
  return test.result();
}
