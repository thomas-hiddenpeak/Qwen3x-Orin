#include "q3x/runtime/prefill_attention_factorized_lane_converter.h"

#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_converter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace checkpoint = q3x::model::checkpoint;
namespace mw = q3x::model::weights;
namespace runtime = q3x::runtime;
namespace st = q3x::io::safetensors;

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

struct SyntheticSource final {
  mw::WeightManifest manifest;
  std::vector<runtime::ShardIdentity> shards;
  std::size_t active_shard = 0U;
  std::uint64_t next_offset = 4'096U;
};

[[nodiscard]] std::uint64_t tensor_bytes(
    const st::DType dtype, const std::vector<std::uint64_t>& shape) {
  std::uint64_t elements = 1U;
  for (const std::uint64_t dimension : shape) {
    elements *= dimension;
  }
  return elements * st::bit_width(dtype) / 8U;
}

void add_tensor(SyntheticSource& source, std::string name,
                const st::DType dtype,
                std::vector<std::uint64_t> shape) {
  const std::uint64_t bytes = tensor_bytes(dtype, shape);
  source.next_offset = (source.next_offset + 255U) & ~255ULL;
  while (source.active_shard < source.shards.size() &&
         (source.next_offset > source.shards[source.active_shard].file_size ||
          bytes > source.shards[source.active_shard].file_size -
                      source.next_offset)) {
    ++source.active_shard;
    source.next_offset = 4'096U;
  }
  if (source.active_shard >= source.shards.size()) {
    return;
  }
  const runtime::ShardIdentity& shard = source.shards[source.active_shard];
  mw::TensorLocator locator;
  locator.category = mw::TensorCategory::kText;
  locator.shard = shard.filename;
  locator.file = shard.filename;
  locator.file_begin = source.next_offset;
  locator.file_end = source.next_offset + bytes;
  locator.byte_size = bytes;
  locator.dtype = dtype;
  locator.shape = std::move(shape);
  source.manifest.tensors.emplace(std::move(name), std::move(locator));
  source.next_offset += bytes;
}

void add_nvfp4_projection(SyntheticSource& source, const std::string& module,
                          const std::uint64_t output_size,
                          const std::uint64_t input_size) {
  add_tensor(source, module + ".weight", st::DType::kU8,
             {output_size, input_size / 2U});
  add_tensor(source, module + ".weight_scale", st::DType::kF8E4M3,
             {output_size, input_size / 16U});
  add_tensor(source, module + ".weight_scale_2", st::DType::kF32, {});
  add_tensor(source, module + ".input_scale", st::DType::kF32, {});
}

void add_fp8_projection(SyntheticSource& source, const std::string& module,
                        const std::uint64_t output_size,
                        const std::uint64_t input_size) {
  add_tensor(source, module + ".weight", st::DType::kF8E4M3,
             {output_size, input_size});
  add_tensor(source, module + ".weight_scale", st::DType::kF32, {});
  add_tensor(source, module + ".input_scale", st::DType::kF32, {});
}

[[nodiscard]] SyntheticSource make_source() {
  SyntheticSource source;
  for (const checkpoint::KnownCheckpointDescriptor& descriptor :
       checkpoint::known_checkpoint_catalog()) {
    if (descriptor.model == q3x::model::KnownModel::kQwen36_27B) {
      source.manifest.checkpoint = descriptor;
      break;
    }
  }
  source.shards = runtime::pinned_qwen36_27b_shards();
  for (std::uint32_t layer = 0U; layer < 64U; ++layer) {
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer) + ".";
    add_nvfp4_projection(source, prefix + "mlp.gate_proj", 17'408U, 5'120U);
    add_nvfp4_projection(source, prefix + "mlp.up_proj", 17'408U, 5'120U);
    add_nvfp4_projection(source, prefix + "mlp.down_proj", 5'120U, 17'408U);
    if (((layer + 1U) % 4U) != 0U) {
      add_fp8_projection(source, prefix + "linear_attn.in_proj_qkv", 10'240U,
                         5'120U);
      add_fp8_projection(source, prefix + "linear_attn.in_proj_z", 6'144U,
                         5'120U);
      add_fp8_projection(source, prefix + "linear_attn.out_proj", 5'120U,
                         6'144U);
    } else {
      add_fp8_projection(source, prefix + "self_attn.q_proj", 12'288U,
                         5'120U);
      add_fp8_projection(source, prefix + "self_attn.k_proj", 1'024U,
                         5'120U);
      add_fp8_projection(source, prefix + "self_attn.v_proj", 1'024U,
                         5'120U);
      add_fp8_projection(source, prefix + "self_attn.o_proj", 5'120U,
                         6'144U);
    }
  }
  return source;
}

[[nodiscard]] bool is_attention_family(
    const runtime::PrefillProjectionFamily family) noexcept {
  return family >= runtime::PrefillProjectionFamily::kLinearQkv &&
         family <= runtime::PrefillProjectionFamily::kFullO;
}

[[nodiscard]] runtime::PrefillAttentionFactorizedLaneProjectionFamily
overlay_family(const runtime::PrefillProjectionFamily family) noexcept {
  return static_cast<runtime::PrefillAttentionFactorizedLaneProjectionFamily>(
      static_cast<std::uint8_t>(family) -
      static_cast<std::uint8_t>(
          runtime::PrefillProjectionFamily::kLinearQkv));
}

[[nodiscard]] runtime::PrefillA4PublicationReceipt make_receipt(
    const runtime::PrefillSidecarManifest& manifest) {
  runtime::PrefillA4PublicationReceipt receipt;
  receipt.version_major = runtime::kPrefillA4K256PublicationVersionMajor;
  receipt.version_minor = runtime::kPrefillA4K256PublicationVersionMinor;
  receipt.mode = runtime::PrefillA4ConversionMode::kProductionCalibrated;
  receipt.production_residency_eligible = true;
  receipt.sidecar_kind = runtime::PrefillSidecarKind::kA4K256;
  receipt.packed_k_group_size = runtime::kPrefillA4PackedKGroupSize;
  receipt.scale_group_size = runtime::kPrefillA4K256WeightGroupSize;
  receipt.physical_layout = std::string(runtime::kPrefillA4K256PhysicalLayout);
  receipt.source_checkpoint_id = manifest.source_checkpoint_id;
  receipt.source_config_sha256 = manifest.source_config_sha256;
  receipt.source_index_sha256 = manifest.source_index_sha256;
  receipt.manifest_sha256 = manifest.manifest_sha256;
  receipt.policy_sha256 = std::string(64U, 'c');
  receipt.policy_bytes = 1U;
  receipt.payload_sha256 = std::string(64U, '7');
  receipt.payload_bytes = manifest.summary.arena_bytes;
  receipt.projection_count = manifest.summary.projection_count;
  return receipt;
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

[[nodiscard]] bool replace_projection_field(
    std::string& document, const std::uint32_t ordinal,
    const std::string_view from, const std::string_view to) {
  const std::string marker =
      "{\"ordinal\":" + std::to_string(ordinal) + ",";
  const std::size_t begin = document.find(marker);
  if (begin == std::string::npos) {
    return false;
  }
  const std::size_t end = document.find("},{\"ordinal\":", begin);
  const std::size_t found = document.find(from, begin);
  if (found == std::string::npos ||
      (end != std::string::npos && found >= end)) {
    return false;
  }
  document.replace(found, from.size(), to);
  return true;
}

void write_u16_le(const std::uint16_t value, std::uint8_t* const output) {
  output[0] = static_cast<std::uint8_t>(value);
  output[1] = static_cast<std::uint8_t>(value >> 8U);
}

void test_manifest(
    Test& test, const runtime::PrefillSidecarManifest& base_manifest,
    const runtime::PrefillA4PublicationReceipt& base_receipt,
    runtime::PrefillAttentionFactorizedLaneOverlayManifestBinding& output) {
  const auto built =
      runtime::build_prefill_attention_factorized_lane_r1_manifest(
          base_manifest, base_receipt, std::string(64U, '6'));
  test.expect(static_cast<bool>(built),
              "authenticated 400-entry K256 manifest filters to Attention");
  if (!built) {
    std::cerr << built.diagnostic.context << ": "
              << built.diagnostic.message << '\n';
    return;
  }
  output = *built.value;
  std::vector<const runtime::PrefillProjectionSidecarEntry*> filtered;
  for (const auto& entry : base_manifest.projections) {
    if (is_attention_family(entry.family)) {
      filtered.push_back(&entry);
    }
  }
  test.expect(filtered.size() == 208U && output.projections.size() == 208U,
              "exact 208-entry Attention inventory is retained");
  const auto plan =
      runtime::prefill_attention_factorized_lane_overlay_layout_plan(1U);
  for (std::size_t index = 0U; index < filtered.size(); ++index) {
    const auto& source = *filtered[index];
    const auto& derived = output.projections[index];
    test.expect(derived.ordinal == index &&
                    derived.layer_index == source.layer_index &&
                    derived.family == overlay_family(source.family) &&
                    derived.source_module == source.source_module &&
                    derived.source_sha256 == source.source_sha256 &&
                    derived.output_size == source.output_size &&
                    derived.input_size == source.input_size &&
                    derived.payload_offset ==
                        runtime::
                            prefill_attention_factorized_lane_projection_absolute_offset(
                                plan, derived.layer_index, derived.family) &&
                    runtime::prefill_attention_factorized_lane_projection_ordinal(
                        derived.layer_index, derived.family) == index,
                "derived entry preserves exact base order and fixed ABI");
  }
  test.expect(output.projections[9U].family ==
                      runtime::PrefillAttentionFactorizedLaneProjectionFamily::
                          kFullQ &&
                  output.projections[12U].family ==
                      runtime::PrefillAttentionFactorizedLaneProjectionFamily::
                          kFullO &&
                  output.projections[13U].family ==
                      runtime::PrefillAttentionFactorizedLaneProjectionFamily::
                          kLinearQkv,
              "mixed layer boundary is Full Q/K/V/O then Linear QKV/Z/O");
  const auto& last = output.projections.back();
  test.expect(last.ordinal == 207U && last.layer_index == 63U &&
                  last.family ==
                      runtime::PrefillAttentionFactorizedLaneProjectionFamily::
                          kFullO &&
                  last.payload_offset + last.payload_bytes ==
                      runtime::kPrefillAttentionFactorizedLaneR1PayloadBytes,
              "last Full O closes the exact 3.614GB payload");
  test.expect(output.required_base_k256.manifest_sha256 ==
                      base_receipt.manifest_sha256 &&
                  output.required_base_k256.policy_sha256 ==
                      base_receipt.policy_sha256 &&
                  output.required_base_k256.payload_sha256 ==
                      base_receipt.payload_sha256 &&
                  output.required_base_k256.receipt_sha256 ==
                      std::string(64U, '6'),
              "derivative binds exact K256 manifest/policy/payload/receipt");

  auto invalid_receipt = base_receipt;
  invalid_receipt.production_residency_eligible = false;
  test.expect(!runtime::build_prefill_attention_factorized_lane_r1_manifest(
                   base_manifest, invalid_receipt, std::string(64U, '6')),
              "non-resident base receipt fails closed");
  invalid_receipt = base_receipt;
  invalid_receipt.scale_group_size = runtime::kPrefillA4K128WeightGroupSize;
  test.expect(!runtime::build_prefill_attention_factorized_lane_r1_manifest(
                   base_manifest, invalid_receipt, std::string(64U, '6')),
              "non-K256 base scale grouping fails closed");
  invalid_receipt = base_receipt;
  invalid_receipt.payload_sha256 = "invalid";
  test.expect(!runtime::build_prefill_attention_factorized_lane_r1_manifest(
                   base_manifest, invalid_receipt, std::string(64U, '6')),
              "invalid base payload identity fails closed");
  test.expect(!runtime::build_prefill_attention_factorized_lane_r1_manifest(
                   base_manifest, base_receipt, "invalid"),
              "invalid exact base receipt identity fails closed");

  const auto expect_structure_rejected = [&test](auto mutated,
                                                  const char* message) {
    mutated.manifest_sha256 =
        runtime::prefill_attention_factorized_lane_r1_manifest_sha256(mutated);
    test.expect(!runtime::validate_prefill_attention_factorized_lane_r1_manifest(
                     mutated),
                message);
  };
  auto mutated = output;
  mutated.projections[0U].family =
      runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullQ;
  expect_structure_rejected(mutated, "topology family mutation fails closed");
  mutated = output;
  ++mutated.projections[0U].input_size;
  expect_structure_rejected(mutated, "shape mutation fails closed");
  mutated = output;
  mutated.projections[9U].source_module += ".wrong";
  expect_structure_rejected(mutated, "source module mutation fails closed");
  mutated = output;
  ++mutated.projections[9U].payload_offset;
  expect_structure_rejected(mutated, "payload offset mutation fails closed");
  mutated = output;
  ++mutated.projections[9U].ordinal;
  expect_structure_rejected(mutated, "ordinal mutation fails closed");
  mutated = output;
  mutated.projections[9U].source_sha256 = "ABC";
  expect_structure_rejected(mutated, "source digest mutation fails closed");
  mutated = output;
  mutated.manifest_sha256 = std::string(64U, '0');
  test.expect(runtime::validate_prefill_attention_factorized_lane_r1_manifest(
                  mutated)
                  .code == runtime::
                               PrefillAttentionFactorizedLaneConverterErrorCode::
                                   kDigestMismatch,
              "manifest body digest mutation is distinguished");
}

void test_policy_and_receipt(
    Test& test,
    const runtime::PrefillAttentionFactorizedLaneOverlayManifestBinding&
        manifest) {
  const auto policy =
      runtime::build_prefill_attention_factorized_lane_r1_policy(
          manifest, 0.75, 0.75);
  test.expect(policy && policy.value->binding.projections.size() == 208U &&
                  policy.value->performance_upper_bound_only &&
                  !policy.value->quality_production_eligible,
              "R1 policy is explicit upper-bound-only and complete");
  if (!policy) {
    return;
  }
  const auto reparsed =
      runtime::parse_prefill_attention_factorized_lane_r1_policy(
          policy.canonical_document, manifest);
  test.expect(reparsed && reparsed.value->binding.policy_sha256 ==
                                policy.value->binding.policy_sha256,
              "canonical policy strict round-trip succeeds");
  const auto& bindings = policy.value->binding.projections;
  test.expect(bindings[0U].factor_source.sha256 ==
                      bindings[1U].factor_source.sha256 &&
                  bindings[0U].factor_source.element_count == 5'120U &&
                  bindings[2U].factor_source.element_count == 6'144U &&
                  bindings[2U].factor_source.sha256 !=
                      bindings[0U].factor_source.sha256,
              "Linear QKV/Z share K5120 identity while O uses K6144");
  test.expect(bindings[9U].factor_source.sha256 ==
                      bindings[10U].factor_source.sha256 &&
                  bindings[9U].factor_source.sha256 ==
                      bindings[11U].factor_source.sha256 &&
                  bindings[12U].factor_source.element_count == 6'144U,
              "Full Q/K/V share K5120 identity and O uses K6144");

  std::string policy_document = policy.canonical_document;
  test.expect(replace_projection_field(
                  policy_document, 1U,
                  "\"activation_clip_ratio\":0.75",
                  "\"activation_clip_ratio\":0.5") &&
                  !runtime::parse_prefill_attention_factorized_lane_r1_policy(
                      policy_document, manifest),
              "Linear QKV/Z group mismatch is rejected by strict parser");
  policy_document = policy.canonical_document;
  test.expect(replace_projection_field(
                  policy_document, 10U,
                  "\"activation_clip_ratio\":0.75",
                  "\"activation_clip_ratio\":0.5") &&
                  !runtime::parse_prefill_attention_factorized_lane_r1_policy(
                      policy_document, manifest),
              "Full Q/K/V group mismatch is rejected by strict parser");
  policy_document = policy.canonical_document;
  test.expect(replace_projection_field(
                  policy_document, 2U,
                  "\"activation_clip_ratio\":0.75",
                  "\"activation_clip_ratio\":0.5") &&
                  runtime::parse_prefill_attention_factorized_lane_r1_policy(
                      policy_document, manifest),
              "Linear O activation clip remains an independent group");
  policy_document = policy.canonical_document;
  test.expect(replace_projection_field(policy_document, 2U,
                                       "\"count\":6144",
                                       "\"count\":5120") &&
                  !runtime::parse_prefill_attention_factorized_lane_r1_policy(
                      policy_document, manifest),
              "O identity factor is strictly K6144");
  policy_document = policy.canonical_document;
  test.expect(replace_projection_field(
                  policy_document, 1U, bindings[1U].factor_source.sha256,
                  bindings[2U].factor_source.sha256) &&
                  !runtime::parse_prefill_attention_factorized_lane_r1_policy(
                      policy_document, manifest),
              "shared Linear factor digest mutation fails closed");

  auto mutated_policy = *policy.value;
  mutated_policy.binding.projections[1U].activation_clip_ratio = 0.5;
  test.expect(!runtime::build_prefill_attention_factorized_lane_r1_receipt(
                   manifest, mutated_policy, std::string(64U, 'f')),
              "Linear QKV/Z activation clip mismatch fails closed");
  mutated_policy = *policy.value;
  mutated_policy.binding.projections[10U].activation_clip_ratio = 0.5;
  test.expect(!runtime::build_prefill_attention_factorized_lane_r1_receipt(
                   manifest, mutated_policy, std::string(64U, 'f')),
              "Full Q/K/V activation clip mismatch fails closed");
  mutated_policy = *policy.value;
  mutated_policy.binding.projections[1U].factor_source.sha256 =
      bindings[2U].factor_source.sha256;
  test.expect(!runtime::build_prefill_attention_factorized_lane_r1_receipt(
                   manifest, mutated_policy, std::string(64U, 'f')),
              "Linear group identity-factor mutation fails closed");
  mutated_policy = *policy.value;
  mutated_policy.binding.projections[2U].activation_clip_ratio = 0.5;
  test.expect(!runtime::build_prefill_attention_factorized_lane_r1_receipt(
                   manifest, mutated_policy, std::string(64U, 'f')),
              "mutated policy cannot reuse stale canonical digest");

  const auto receipt =
      runtime::build_prefill_attention_factorized_lane_r1_receipt(
          manifest, *policy.value, std::string(64U, 'f'));
  test.expect(receipt &&
                  !receipt.value->binding.production_residency_eligible &&
                  !receipt.value->binding.quality_production_eligible &&
                  receipt.value->performance_upper_bound_only &&
                  !receipt.value->quality_production_eligible &&
                  receipt.value->residency_eligibility_scope ==
                      runtime::
                          kPrefillAttentionFactorizedLaneR1EligibilityScope &&
                  receipt.value->binding.payload.bytes ==
                      runtime::kPrefillAttentionFactorizedLaneR1PayloadBytes &&
                  receipt.value->binding.projection_count == 208U,
              "receipt independently locks residency/performance/quality");
  if (!receipt) {
    return;
  }
  test.expect(static_cast<bool>(
                  runtime::parse_prefill_attention_factorized_lane_r1_receipt(
                      receipt.canonical_document, manifest, *policy.value)),
              "canonical receipt strict round-trip succeeds");
  const std::array<std::pair<std::string_view, std::string_view>, 4U>
      eligibility_mutations = {{
          {"\"production_residency_eligible\":false",
           "\"production_residency_eligible\":true"},
          {"\"performance_upper_bound_only\":true",
           "\"performance_upper_bound_only\":false"},
          {"\"quality_production_eligible\":false",
           "\"quality_production_eligible\":true"},
          {"\"residency_eligibility_scope\":\"authenticated_abi_only\"",
           "\"residency_eligibility_scope\":\"production\""},
      }};
  for (const auto& mutation : eligibility_mutations) {
    std::string document = receipt.canonical_document;
    test.expect(replace_once(document, mutation.first, mutation.second) &&
                    !runtime::
                        parse_prefill_attention_factorized_lane_r1_receipt(
                            document, manifest, *policy.value),
                "receipt eligibility mutation fails closed");
  }
}

void test_transform_shape(Test& test, const std::size_t input_size) {
  constexpr std::size_t kRows = 64U;
  const std::size_t packed_bytes = kRows * input_size / 2U;
  const std::size_t base_scale_bytes = kRows * (input_size / 256U) * 2U;
  std::vector<std::uint8_t> packed(packed_bytes);
  for (std::size_t index = 0U; index < packed.size(); ++index) {
    const std::uint8_t low = static_cast<std::uint8_t>(index % 8U);
    const std::uint8_t high = static_cast<std::uint8_t>(9U + index % 7U);
    packed[index] = static_cast<std::uint8_t>(low | (high << 4U));
  }
  std::vector<std::uint8_t> scales(base_scale_bytes);
  for (std::size_t index = 0U; index < scales.size() / 2U; ++index) {
    write_u16_le(0x3f00U, scales.data() + index * 2U);
  }
  std::vector<std::uint8_t> expected_packed(packed_bytes);
  std::vector<std::uint8_t> actual_packed(packed_bytes);
  std::vector<std::uint8_t> expected_scales(kRows * 2U);
  std::vector<std::uint8_t> actual_scales(kRows * 2U);
  const auto expected =
      runtime::transform_prefill_mlp_k256_to_factorized_r1_consumer_blocks(
          packed.data(), packed.size(), scales.data(), scales.size(), kRows,
          input_size, 0.75, expected_packed.data(), expected_packed.size(),
          expected_scales.data(), expected_scales.size());
  const auto actual =
      runtime::
          transform_prefill_attention_k256_to_factorized_r1_consumer_blocks(
              packed.data(), packed.size(), scales.data(), scales.size(),
              kRows, input_size, 0.75, actual_packed.data(),
              actual_packed.size(), actual_scales.data(),
              actual_scales.size());
  test.expect(expected.ok() && actual.ok() &&
                  expected_packed == actual_packed &&
                  expected_scales == actual_scales,
              "Attention wrapper is byte-identical to audited generic transform");
  packed[0U] = 0x08U;
  const auto invalid =
      runtime::
          transform_prefill_attention_k256_to_factorized_r1_consumer_blocks(
              packed.data(), packed.size(), scales.data(), scales.size(),
              kRows, input_size, 0.75, actual_packed.data(),
              actual_packed.size(), actual_scales.data(),
              actual_scales.size());
  test.expect(invalid.code ==
                  runtime::PrefillAttentionFactorizedLaneConverterErrorCode::
                      kQuantizationFailure,
              "reserved -8 source code fails closed through wrapper");
}

}  // namespace

int main() {
  Test test;
  const SyntheticSource source = make_source();
  runtime::PrefillSidecarManifestOptions options;
  options.kind = runtime::PrefillSidecarKind::kA4K256;
  const auto base = runtime::build_qwen36_27b_prefill_sidecar_manifest(
      source.manifest, source.shards, options);
  test.expect(static_cast<bool>(base),
              "metadata-only fixture builds a valid 400-entry K256 manifest");
  if (!base) {
    return test.result();
  }
  const auto receipt = make_receipt(*base.value);
  runtime::PrefillAttentionFactorizedLaneOverlayManifestBinding manifest;
  test_manifest(test, *base.value, receipt, manifest);
  if (!manifest.projections.empty()) {
    test_policy_and_receipt(test, manifest);
  }
  test_transform_shape(test, 5'120U);
  test_transform_shape(test, 6'144U);
  const runtime::PrefillAttentionFactorizedLaneR1ConversionOptions empty;
  test.expect(
      runtime::
          convert_authenticated_k256_to_prefill_attention_factorized_lane_r1(
              empty)
              .diagnostic.code ==
          runtime::PrefillAttentionFactorizedLaneConverterErrorCode::
              kInvalidOption,
      "empty publication options fail before filesystem access");
  return test.result();
}
