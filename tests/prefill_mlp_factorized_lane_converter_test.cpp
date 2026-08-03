#include "q3x/runtime/prefill_mlp_factorized_lane_converter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

static_assert(runtime::kPrefillMLPFactorizedLaneR1LaneCount == 1U);
static_assert(runtime::kPrefillMLPFactorizedLaneR1PayloadBytes ==
              8'568'619'008ULL);
static_assert(
    runtime::prefill_mlp_factorized_lane_overlay_layout_plan(1U)
        .payload_bytes ==
    runtime::kPrefillMLPFactorizedLaneR1PayloadBytes);

void write_u16_le(const std::uint16_t value,
                  std::uint8_t* const output) {
  output[0] = static_cast<std::uint8_t>(value);
  output[1] = static_cast<std::uint8_t>(value >> 8U);
}

[[nodiscard]] std::uint16_t read_u16_le(
    const std::uint8_t* const input) {
  return static_cast<std::uint16_t>(input[0]) |
         static_cast<std::uint16_t>(
             static_cast<std::uint16_t>(input[1]) << 8U);
}

[[nodiscard]] runtime::PrefillMLPFactorizedLaneOverlayManifestBinding
make_manifest() {
  runtime::PrefillMLPFactorizedLaneOverlayManifestBinding manifest;
  manifest.physical_layout =
      std::string(runtime::kPrefillMLPFactorizedLaneOverlayLayout);
  manifest.source_checkpoint_id = "pinned-qwen36-27b";
  manifest.source_config_sha256 = std::string(64U, '1');
  manifest.source_index_sha256 = std::string(64U, '2');
  manifest.required_base_k256.physical_layout = std::string(
      runtime::kPrefillMLPFactorizedLaneRequiredBaseK256Layout);
  manifest.required_base_k256.manifest_sha256 = std::string(64U, '3');
  manifest.required_base_k256.policy_sha256 = std::string(64U, '4');
  manifest.required_base_k256.payload_sha256 = std::string(64U, '5');
  manifest.required_base_k256.receipt_sha256 = std::string(64U, '6');
  manifest.lane_count = 1U;
  const auto plan =
      runtime::prefill_mlp_factorized_lane_overlay_layout_plan(1U);
  manifest.payload_bytes = plan.payload_bytes;
  manifest.projections.reserve(
      runtime::kPrefillMLPFactorizedLaneProjectionCount);
  for (std::uint32_t layer = 0U;
       layer < runtime::kPrefillMLPFactorizedLaneLayerCount; ++layer) {
    for (std::uint32_t position = 0U; position < 3U; ++position) {
      const auto family =
          static_cast<runtime::PrefillMLPFactorizedLaneProjectionFamily>(
              position);
      const bool down =
          family ==
          runtime::PrefillMLPFactorizedLaneProjectionFamily::kDown;
      runtime::PrefillMLPFactorizedLaneManifestProjection entry;
      entry.ordinal = layer * 3U + position;
      entry.layer_index = layer;
      entry.family = family;
      entry.source_module =
          "model.language_model.layers." + std::to_string(layer) +
          ".mlp." +
          (position == 0U
               ? "gate_proj"
               : (position == 1U ? "up_proj" : "down_proj"));
      entry.source_sha256 = std::string(64U, "abcdef"[position]);
      entry.output_size =
          down ? runtime::kPrefillMLPFactorizedLaneDownOutputSize
               : runtime::kPrefillMLPFactorizedLaneGateUpOutputSize;
      entry.input_size =
          down ? runtime::kPrefillMLPFactorizedLaneDownInputSize
               : runtime::kPrefillMLPFactorizedLaneGateUpInputSize;
      entry.payload_offset =
          runtime::prefill_mlp_factorized_lane_projection_absolute_offset(
              plan, layer, family);
      entry.payload_bytes =
          down ? plan.down.projection_bytes : plan.gate.projection_bytes;
      manifest.projections.emplace_back(std::move(entry));
    }
  }
  manifest.manifest_sha256 =
      runtime::prefill_mlp_factorized_lane_r1_manifest_sha256(manifest);
  return manifest;
}

[[nodiscard]] bool replace_once(std::string& text,
                                const std::string_view from,
                                const std::string_view to) {
  const std::size_t position = text.find(from);
  if (position == std::string::npos) {
    return false;
  }
  text.replace(position, from.size(), to);
  return true;
}

void test_transform(Test& test) {
  constexpr std::size_t kRows = 128U;
  constexpr std::size_t kInput = 256U;
  std::vector<std::uint8_t> base_packed(kRows * kInput / 2U, 0x97U);
  std::vector<std::uint8_t> base_scales(kRows * kInput / 256U * 2U);
  for (std::size_t row = 0U; row < kRows; ++row) {
    // BF16(0.5) == 0x3f00.
    write_u16_le(0x3f00U, base_scales.data() + row * 2U);
  }
  std::vector<std::uint8_t> r1_packed(base_packed.size());
  std::vector<std::uint8_t> r1_scales(kRows * 2U);
  auto diagnostic =
      runtime::transform_prefill_mlp_k256_to_factorized_r1_consumer_blocks(
          base_packed.data(), base_packed.size(), base_scales.data(),
          base_scales.size(), kRows, kInput, 1.0, r1_packed.data(),
          r1_packed.size(), r1_scales.data(), r1_scales.size());
  test.expect(diagnostic.ok() && r1_packed == base_packed,
              "full-K clip=1 preserves uniform +/-7 K256 codes");
  for (std::size_t row = 0U; row < kRows; ++row) {
    test.expect(read_u16_le(r1_scales.data() + row * 2U) == 0x3f00U,
                "R1 full-K scale is BF16 0.5");
  }

  std::fill(base_packed.begin(), base_packed.end(), 0U);
  std::fill(r1_packed.begin(), r1_packed.end(), 0xffU);
  diagnostic =
      runtime::transform_prefill_mlp_k256_to_factorized_r1_consumer_blocks(
          base_packed.data(), base_packed.size(), base_scales.data(),
          base_scales.size(), kRows, kInput, 0.75, r1_packed.data(),
          r1_packed.size(), r1_scales.data(), r1_scales.size());
  test.expect(diagnostic.ok() &&
                  std::all_of(r1_packed.begin(), r1_packed.end(),
                              [](const std::uint8_t value) {
                                return value == 0U;
                              }),
              "zero rows emit zero codes");
  for (std::size_t row = 0U; row < kRows; ++row) {
    test.expect(read_u16_le(r1_scales.data() + row * 2U) == 0x3f80U,
                "zero rows emit BF16 one scale");
  }

  base_packed[0] = 0x08U;
  diagnostic =
      runtime::transform_prefill_mlp_k256_to_factorized_r1_consumer_blocks(
          base_packed.data(), base_packed.size(), base_scales.data(),
          base_scales.size(), kRows, kInput, 1.0, r1_packed.data(),
          r1_packed.size(), r1_scales.data(), r1_scales.size());
  test.expect(diagnostic.code ==
                  runtime::PrefillMLPFactorizedLaneConverterErrorCode::
                      kQuantizationFailure,
              "reserved signed-W4 -8 code fails closed");
  base_packed[0] = 0U;
  write_u16_le(0U, base_scales.data());
  diagnostic =
      runtime::transform_prefill_mlp_k256_to_factorized_r1_consumer_blocks(
          base_packed.data(), base_packed.size(), base_scales.data(),
          base_scales.size(), kRows, kInput, 1.0, r1_packed.data(),
          r1_packed.size(), r1_scales.data(), r1_scales.size());
  test.expect(diagnostic.code ==
                  runtime::PrefillMLPFactorizedLaneConverterErrorCode::
                      kQuantizationFailure,
              "nonpositive K256 BF16 scale fails closed");
  write_u16_le(0x3f00U, base_scales.data());
  diagnostic =
      runtime::transform_prefill_mlp_k256_to_factorized_r1_consumer_blocks(
          base_packed.data(), base_packed.size() - 1U, base_scales.data(),
          base_scales.size(), kRows, kInput, 1.0, r1_packed.data(),
          r1_packed.size(), r1_scales.data(), r1_scales.size());
  test.expect(diagnostic.code ==
                  runtime::PrefillMLPFactorizedLaneConverterErrorCode::
                      kInvalidOption,
              "consumer buffer size mismatch fails closed");
}

void test_policy_and_receipt(Test& test) {
  auto manifest = make_manifest();
  test.expect(
      runtime::validate_prefill_mlp_factorized_lane_r1_manifest(manifest)
          .ok(),
      "complete synthetic R1 manifest validates");
  auto policy = runtime::build_prefill_mlp_factorized_lane_r1_policy(
      manifest, 0.75, 1.0);
  test.expect(policy && policy.value->performance_upper_bound_only &&
                  !policy.value->quality_production_eligible &&
                  policy.value->binding.projections.size() == 192U &&
                  policy.value->binding.projections.front()
                          .factor_source.scheme ==
                      runtime::kPrefillMLPFactorizedLaneR1FactorScheme &&
                  policy.value->binding.projections.front()
                      .factor_source.path.empty(),
              "canonical policy is explicit upper-bound-only identity alpha");
  if (!policy) {
    return;
  }
  auto reparsed = runtime::parse_prefill_mlp_factorized_lane_r1_policy(
      policy.canonical_document, manifest);
  test.expect(reparsed &&
                  reparsed.value->binding.policy_sha256 ==
                      policy.value->binding.policy_sha256 &&
                  reparsed.value->binding.policy_bytes ==
                      policy.value->binding.policy_bytes,
              "canonical policy strict round-trip preserves exact identity");

  std::string mutated = policy.canonical_document;
  test.expect(replace_once(mutated, "\"quality_production_eligible\":false",
                           "\"quality_production_eligible\":true") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r1_policy(
                      mutated, manifest),
              "policy cannot claim quality-production eligibility");
  mutated = policy.canonical_document;
  test.expect(replace_once(mutated, "identity_alpha_f32_v1",
                           "calibrated_alpha_f32_v1") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r1_policy(
                      mutated, manifest),
              "non-identity factor scheme fails R1 parser");
  mutated = policy.canonical_document;
  test.expect(replace_once(mutated, "\"count\":5120", "\"count\":5119") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r1_policy(
                      mutated, manifest),
              "factor count mutation fails closed");
  mutated = policy.canonical_document;
  test.expect(replace_once(mutated, "{\"schema\":",
                           "{\"unknown\":0,\"schema\":") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r1_policy(
                      mutated, manifest),
              "unknown policy field fails closed");

  const auto receipt =
      runtime::build_prefill_mlp_factorized_lane_r1_receipt(
          manifest, *policy.value, std::string(64U, 'f'));
  test.expect(receipt &&
                  !receipt.value->binding.production_residency_eligible &&
                  receipt.value->residency_eligibility_scope ==
                      runtime::kPrefillMLPFactorizedLaneR1EligibilityScope &&
                  receipt.value->performance_upper_bound_only &&
                  !receipt.value->quality_production_eligible &&
                  receipt.value->binding.payload.bytes ==
                      runtime::kPrefillMLPFactorizedLaneR1PayloadBytes,
              "receipt scopes eligibility to authenticated ABI only");
  if (!receipt) {
    return;
  }
  const auto receipt_reparsed =
      runtime::parse_prefill_mlp_factorized_lane_r1_receipt(
          receipt.canonical_document, manifest, *policy.value);
  test.expect(static_cast<bool>(receipt_reparsed),
              "canonical receipt strict round-trip succeeds");

  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated, "\"production_residency_eligible\":false",
                           "\"production_residency_eligible\":true") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r1_receipt(
                      mutated, manifest, *policy.value),
              "receipt ABI eligibility mutation fails closed");
  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated, "\"quality_production_eligible\":false",
                           "\"quality_production_eligible\":true") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r1_receipt(
                      mutated, manifest, *policy.value),
              "receipt cannot claim quality-production eligibility");
  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated, "\"payload_bytes\":8568619008",
                           "\"payload_bytes\":8568619009") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r1_receipt(
                      mutated, manifest, *policy.value),
              "payload byte mutation fails closed");
  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated, "{\"schema\":",
                           "{\"unknown\":0,\"schema\":") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r1_receipt(
                      mutated, manifest, *policy.value),
              "unknown receipt field fails closed");
}

void test_invalid_conversion_options(Test& test) {
  const runtime::PrefillMLPFactorizedLaneR1ConversionOptions options;
  const auto result =
      runtime::convert_authenticated_k256_to_prefill_mlp_factorized_lane_r1(
          options);
  test.expect(result.diagnostic.code ==
                  runtime::PrefillMLPFactorizedLaneConverterErrorCode::
                      kInvalidOption,
              "empty full-conversion options fail before filesystem access");
}

}  // namespace

int main() {
  Test test;
  test_transform(test);
  test_policy_and_receipt(test);
  test_invalid_conversion_options(test);
  return test.result();
}
