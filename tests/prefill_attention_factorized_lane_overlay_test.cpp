#include "q3x/runtime/prefill_attention_factorized_lane_overlay.h"

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
    runtime::prefill_attention_factorized_lane_overlay_layout_plan(1U);
constexpr auto kR2Plan =
    runtime::prefill_attention_factorized_lane_overlay_layout_plan(2U);
constexpr auto kR4Plan =
    runtime::prefill_attention_factorized_lane_overlay_layout_plan(4U);
static_assert(kR1Plan.valid() && kR2Plan.valid() && kR4Plan.valid());
static_assert(kR1Plan.linear_qkv.projection_bytes == 26'255'616ULL);
static_assert(kR1Plan.linear_z.projection_bytes == 15'761'664ULL);
static_assert(kR1Plan.linear_o.projection_bytes == 15'763'712ULL);
static_assert(kR1Plan.full_q.projection_bytes == 31'502'592ULL);
static_assert(kR1Plan.full_k.projection_bytes == 2'644'224ULL);
static_assert(kR1Plan.full_v.projection_bytes == 2'644'224ULL);
static_assert(kR1Plan.full_o.projection_bytes == 15'763'712ULL);
static_assert(kR1Plan.linear_layer_bytes == 57'780'992ULL);
static_assert(kR1Plan.full_layer_bytes == 52'554'752ULL);
static_assert(kR1Plan.payload_bytes == 3'614'363'648ULL);
static_assert(
    runtime::prefill_attention_factorized_lane_overlay_layout_plan(3U).error ==
    runtime::PrefillAttentionFactorizedLaneOverlayPlanError::
        kUnsupportedLaneCount);
static_assert(
    runtime::prefill_attention_factorized_lane_overlay_layout_plan(1U, 128U)
        .error ==
    runtime::PrefillAttentionFactorizedLaneOverlayPlanError::kInvalidAlignment);
static_assert(
    runtime::prefill_attention_factorized_lane_overlay_layout_plan(
        1U, std::uint64_t{1U} << 63U)
        .error ==
    runtime::PrefillAttentionFactorizedLaneOverlayPlanError::kArithmeticOverflow);
static_assert(runtime::prefill_attention_factorized_lane_qualification_role(1U) ==
              runtime::PrefillAttentionFactorizedLaneQualificationRole::
                  kPerformanceUpperBound);
static_assert(runtime::prefill_attention_factorized_lane_qualification_role(2U) ==
              runtime::PrefillAttentionFactorizedLaneQualificationRole::
                  kStructuralOnly);
static_assert(runtime::prefill_attention_factorized_lane_qualification_role(4U) ==
              runtime::PrefillAttentionFactorizedLaneQualificationRole::
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
                 const runtime::PrefillAttentionFactorizedLaneOverlayLayoutPlan&
                     plan) {
  test.expect(plan.valid() && plan.projection_count == 208U,
              "layout fixes the complete 208-projection inventory");
  test.expect(
      plan.linear_qkv_offset_in_layer == 0U &&
          plan.linear_z_offset_in_layer == plan.linear_qkv.projection_bytes &&
          plan.linear_o_offset_in_layer ==
              plan.linear_qkv.projection_bytes + plan.linear_z.projection_bytes &&
          plan.full_q_offset_in_layer == 0U &&
          plan.full_k_offset_in_layer == plan.full_q.projection_bytes &&
          plan.full_v_offset_in_layer ==
              plan.full_q.projection_bytes + plan.full_k.projection_bytes &&
          plan.full_o_offset_in_layer ==
              plan.full_q.projection_bytes + plan.full_k.projection_bytes +
                  plan.full_v.projection_bytes,
      "mixed linear/full layer offsets are gap-free and explicit");
  std::uint32_t visited = 0U;
  for (std::uint32_t layer = 0U;
       layer < runtime::kPrefillAttentionFactorizedLaneLayerCount; ++layer) {
    const bool full =
        runtime::prefill_attention_factorized_lane_is_full_layer(layer);
    const std::array linear_families{
        runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv,
        runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ,
        runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearO};
    const std::array full_families{
        runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullQ,
        runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullK,
        runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullV,
        runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullO};
    const std::size_t count = full ? full_families.size() : linear_families.size();
    for (std::size_t position = 0U; position < count; ++position) {
      const auto family = full ? full_families[position]
                               : linear_families[position];
      const std::uint64_t offset =
          runtime::prefill_attention_factorized_lane_projection_absolute_offset(
              plan, layer, family);
      test.expect(offset !=
                          runtime::kPrefillA4FactorizedLaneInvalidOffset &&
                      offset % plan.alignment == 0U &&
                      offset < plan.payload_bytes,
                  "every projection has an in-range 256-byte-aligned offset");
      test.expect(
          runtime::prefill_attention_factorized_lane_projection_ordinal(layer,
                                                                  family) ==
              visited,
          "ordinal follows the fixed mixed Attention layer inventory");
      ++visited;
    }
    const auto wrong_family =
        full ? runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv
             : runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullQ;
    test.expect(
        runtime::prefill_attention_factorized_lane_projection_absolute_offset(
            plan, layer, wrong_family) ==
            runtime::kPrefillA4FactorizedLaneInvalidOffset &&
            runtime::prefill_attention_factorized_lane_projection_ordinal(
                layer, wrong_family) ==
                runtime::kPrefillAttentionFactorizedLaneInvalidOrdinal,
        "projection family from the wrong layer topology fails closed");
  }
  test.expect(visited == 208U, "mixed layer walk covers all 208 projections");
  const std::uint64_t final_o =
      runtime::prefill_attention_factorized_lane_projection_absolute_offset(
          plan, 63U,
          runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullO);
  test.expect(final_o + plan.full_o.projection_bytes == plan.payload_bytes,
              "last Full-O projection closes the exact aggregate payload");
  test.expect(
      runtime::prefill_attention_factorized_lane_projection_absolute_offset(
          plan, 64U,
          runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv) ==
          runtime::kPrefillA4FactorizedLaneInvalidOffset,
      "out-of-range layer fails closed");
  test.expect(
      runtime::prefill_attention_factorized_lane_projection_absolute_offset(
          plan, 0U,
          static_cast<runtime::PrefillAttentionFactorizedLaneProjectionFamily>(
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
      runtime::serialize_prefill_attention_factorized_lane_metadata(
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

  const auto parsed = runtime::parse_prefill_attention_factorized_lane_metadata(
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
  std::vector<float> values = make_inverse_alpha(6'144U);
  const auto original =
      runtime::serialize_prefill_attention_factorized_lane_metadata(
          4U, values.data(), values.size());
  test.expect(original.valid(), "mutation fixture serializes");
  if (!original) {
    return;
  }
  const auto parse = [](const std::vector<std::uint8_t>& bytes) {
    return runtime::parse_prefill_attention_factorized_lane_metadata(
        bytes.data(), bytes.size(), 4U, 6'144U);
  };

  std::vector<std::uint8_t> mutated = original.bytes;
  mutated[0] ^= 1U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kInvalidMagic,
              "magic mutation fails closed");

  mutated = original.bytes;
  mutated[8] = 5U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kUnsupportedVersion,
              "version mutation fails closed");

  mutated = original.bytes;
  mutated[20] = 2U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kUnsupportedEncoding,
              "encoding mutation fails closed");

  mutated = original.bytes;
  write_u32_le(2U, mutated.data() + 16U);
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kExpectedShapeMismatch,
              "structurally valid lane header mutation fails expected binding");

  mutated = original.bytes;
  write_u64_le(6'400U, mutated.data() + 24U);
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kExpectedShapeMismatch,
              "structurally valid K header mutation fails expected binding");

  mutated = original.bytes;
  mutated[32] ^= 0x80U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kDigestMismatch,
              "digest mutation fails closed");

  mutated = original.bytes;
  mutated[64U + 17U] ^= 1U;
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kDigestMismatch,
              "inverse-alpha byte mutation fails closed");

  mutated = original.bytes;
  mutated.pop_back();
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kInvalidByteLength,
              "truncated metadata fails closed");
  mutated = original.bytes;
  mutated.push_back(0U);
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kInvalidByteLength,
              "trailing metadata bytes fail closed");

  // Construct an internally digest-consistent zero.  Digest integrity alone
  // must not bypass the finite-positive numerical contract.
  mutated = original.bytes;
  std::fill_n(mutated.begin() + 64, sizeof(float), 0U);
  rehash_inverse_alpha(mutated);
  test.expect(parse(mutated).error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kInvalidInverseAlpha,
              "digest-consistent nonpositive alpha still fails closed");

  const auto null_input =
      runtime::serialize_prefill_attention_factorized_lane_metadata(1U, nullptr,
                                                              5'120U);
  test.expect(null_input.error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::kNullInput,
              "null non-empty inverse-alpha input is rejected");
  const auto bad_lane =
      runtime::serialize_prefill_attention_factorized_lane_metadata(
          3U, values.data(), 5'120U);
  test.expect(
      bad_lane.error ==
          runtime::PrefillAttentionFactorizedLaneMetadataError::
              kUnsupportedLaneCount,
      "unsupported lane count is rejected");
  const auto bad_shape =
      runtime::serialize_prefill_attention_factorized_lane_metadata(
          4U, values.data(), 64U);
  test.expect(bad_shape.error ==
                  runtime::PrefillAttentionFactorizedLaneMetadataError::
                      kInvalidInputSize,
              "lane segment not divisible by K64 is rejected");

  values[0] = std::numeric_limits<float>::infinity();
  test.expect(
      runtime::serialize_prefill_attention_factorized_lane_metadata(
          4U, values.data(), values.size())
              .error == runtime::PrefillAttentionFactorizedLaneMetadataError::
                            kInvalidInverseAlpha,
      "nonfinite inverse alpha is rejected before publication");
  values[0] = -1.0F;
  test.expect(
      runtime::serialize_prefill_attention_factorized_lane_metadata(
          4U, values.data(), values.size())
              .error == runtime::PrefillAttentionFactorizedLaneMetadataError::
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
      runtime::parse_prefill_attention_factorized_lane_metadata(
          huge_header.data(), huge_header.size(), 1U,
          std::numeric_limits<std::uint64_t>::max() - 63U)
              .error == runtime::PrefillAttentionFactorizedLaneMetadataError::
                            kArithmeticOverflow,
      "metadata byte-count overflow fails closed before allocation");
}

void test_publication_binding_vocabulary(Test& test) {
  runtime::PrefillAttentionFactorizedLaneBaseK256Binding base;
  base.physical_layout =
      std::string(runtime::kPrefillAttentionFactorizedLaneRequiredBaseK256Layout);
  base.manifest_sha256 = std::string(64U, '1');
  base.policy_sha256 = std::string(64U, '2');
  base.payload_sha256 = std::string(64U, '3');

  runtime::PrefillAttentionFactorizedLaneProjectionCalibrationBinding calibration;
  calibration.ordinal = 207U;
  calibration.weight_clip_ratio = 0.75;
  calibration.activation_clip_ratio = 1.0;
  calibration.factor_source.path = "calibration/full_o.63.inverse-alpha.f32";
  calibration.factor_source.sha256 = std::string(64U, '4');
  calibration.factor_source.element_count = 6'144U;

  runtime::PrefillAttentionFactorizedLaneOverlayPolicyBinding policy;
  policy.physical_layout =
      std::string(runtime::kPrefillAttentionFactorizedLaneOverlayLayout);
  policy.required_base_k256 = base;
  policy.lane_count = 4U;
  policy.projections.push_back(calibration);

  runtime::PrefillAttentionFactorizedLaneOverlayReceiptBinding receipt;
  receipt.physical_layout = policy.physical_layout;
  receipt.required_base_k256 = base;
  receipt.lane_count = policy.lane_count;
  receipt.payload.path = "weights-attention-factorized-r4-v4.bin";
  receipt.payload.sha256 = std::string(64U, '5');
  receipt.payload.bytes = kR4Plan.payload_bytes;
  receipt.projection_count =
      runtime::kPrefillAttentionFactorizedLaneProjectionCount;

  test.expect(
      policy.required_base_k256.physical_layout ==
              runtime::kPrefillAttentionFactorizedLaneRequiredBaseK256Layout &&
          policy.projections.front().weight_clip_ratio == 0.75 &&
          policy.projections.front().activation_clip_ratio == 1.0 &&
          policy.projections.front().factor_source.path ==
              "calibration/full_o.63.inverse-alpha.f32" &&
          policy.projections.front().factor_source.sha256.size() == 64U &&
          policy.projections.front().factor_source.element_count == 6'144U &&
          !receipt.production_residency_eligible &&
          !receipt.quality_production_eligible &&
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
  test_metadata_round_trip(test, 4U, 6'144U);
  test_metadata_rejections(test);
  test_publication_binding_vocabulary(test);
  return test.result();
}
