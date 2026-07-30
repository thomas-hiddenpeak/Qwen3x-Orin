#include "q3x/runtime/prefill_a4_sidecar_converter.h"

#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"
#include "q3x/runtime/resident_weights.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

namespace checkpoint = q3x::model::checkpoint;
namespace fs = std::filesystem;
namespace mw = q3x::model::weights;
namespace runtime = q3x::runtime;
namespace st = q3x::io::safetensors;

class TestContext {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }
  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

struct SyntheticSource {
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

[[nodiscard]] runtime::PrefillSidecarManifest build_manifest(
    const SyntheticSource& source) {
  runtime::PrefillSidecarManifestOptions options;
  options.kind = runtime::PrefillSidecarKind::kA4K64;
  auto result = runtime::build_qwen36_27b_prefill_sidecar_manifest(
      source.manifest, source.shards, options);
  return result ? std::move(*result.value)
                : runtime::PrefillSidecarManifest{};
}

[[nodiscard]] std::string policy_json(
    const runtime::PrefillSidecarManifest& manifest,
    const std::size_t projection_count = runtime::kQwen36PrefillProjectionCount,
    const std::string_view mode = "production_calibrated",
    const bool equalize_first = false,
    const bool corrupt_first_source = false) {
  std::string output;
  output.reserve(256U * projection_count);
  output += "{\"schema\":\"q3x.prefill.a4.calibration-policy\","
            "\"version\":{\"major\":1,\"minor\":0},\"mode\":\"";
  output += mode;
  output += "\",\"sidecar_kind\":\"a4_k64\","
            "\"physical_layout\":\"sm87_s4_n64_k64_consumer_v1\","
            "\"source_checkpoint_id\":\"";
  output += manifest.source_checkpoint_id;
  output += "\",\"source_config_sha256\":\"";
  output += manifest.source_config_sha256;
  output += "\",\"source_index_sha256\":\"";
  output += manifest.source_index_sha256;
  output += "\",\"manifest_sha256\":\"";
  output += manifest.manifest_sha256;
  output += "\",\"projections\":[";
  for (std::size_t index = 0U; index < projection_count; ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    const auto& entry = manifest.projections[index];
    output += "{\"ordinal\":" + std::to_string(entry.ordinal) +
              ",\"source_module\":\"" + entry.source_module +
              "\",\"source_sha256\":\"";
    output += corrupt_first_source && index == 0U
                  ? std::string(64U, '0')
                  : entry.source_sha256;
    output += "\",\"weight_clip_ratio\":0.9375,"
              "\"activation_clip_ratio\":0.96875,"
              "\"activation_scale_group_size\":64,"
              "\"rounding\":\"nearest_even_v1\","
              "\"channel_equalization\":";
    if (equalize_first && index <= 1U) {
      output += "{\"scheme\":\"input_channel_multiply_f32_v1\","
                "\"factors_path\":\"calibration/layer0-gate.f32\","
                "\"factors_sha256\":\"" +
                std::string(64U, 'a') + "\",\"factor_count\":" +
                std::to_string(entry.input_size) + "}";
    } else {
      output += "null";
    }
    output.push_back('}');
  }
  output += "]}";
  return output;
}

void test_policy_gate(TestContext& test,
                      const runtime::PrefillSidecarManifest& manifest) {
  const auto parsed = runtime::parse_prefill_a4_calibration_policy(
      policy_json(manifest), manifest);
  test.expect(parsed && parsed.value->projections.size() == 400U &&
                  parsed.value->physical_layout ==
                      runtime::kPrefillA4PhysicalLayout &&
                  parsed.value->policy_sha256.size() == 64U,
              "production policy covers and binds all 400 projections");

  std::string canonical_layout = policy_json(manifest);
  constexpr std::string_view kPhysicalLayout =
      "sm87_s4_n64_k64_consumer_v1";
  const std::size_t physical_layout = canonical_layout.find(kPhysicalLayout);
  if (physical_layout != std::string::npos) {
    canonical_layout.replace(physical_layout, kPhysicalLayout.size(),
                             "canonical_row_major");
  }
  const auto canonical = runtime::parse_prefill_a4_calibration_policy(
      canonical_layout, manifest);
  test.expect(!canonical &&
                  canonical.diagnostic.code ==
                      runtime::PrefillA4ConverterErrorCode::kInvalidPolicy,
              "legacy canonical row-major policy fails closed");

  const auto equalized = runtime::parse_prefill_a4_calibration_policy(
      policy_json(manifest, 400U, "production_calibrated", true), manifest);
  test.expect(equalized &&
                  equalized.value->projections[0]
                      .channel_equalization.has_value() &&
                  equalized.value->projections[0]
                          .channel_equalization->factor_count == 5'120U,
              "optional versioned channel equalization metadata is retained");

  const auto incomplete = runtime::parse_prefill_a4_calibration_policy(
      policy_json(manifest, 399U), manifest);
  test.expect(!incomplete &&
                  incomplete.diagnostic.code == runtime::
                                                    PrefillA4ConverterErrorCode::
                                                        kPolicyCoverageMismatch,
              "399-projection policy fails closed");

  const auto wrong_source = runtime::parse_prefill_a4_calibration_policy(
      policy_json(manifest, 400U, "production_calibrated", false, true),
      manifest);
  test.expect(!wrong_source &&
                  wrong_source.diagnostic.code == runtime::
                                                      PrefillA4ConverterErrorCode::
                                                          kPolicyCoverageMismatch,
              "per-projection source SHA mismatch fails closed");

  std::string mismatched_shared_activation = policy_json(manifest);
  constexpr std::string_view kActivation =
      "\"activation_clip_ratio\":0.96875";
  const std::size_t first = mismatched_shared_activation.find(kActivation);
  const std::size_t second = first == std::string::npos
                                 ? std::string::npos
                                 : mismatched_shared_activation.find(
                                       kActivation, first + kActivation.size());
  if (second != std::string::npos) {
    mismatched_shared_activation.replace(
        second, kActivation.size(), "\"activation_clip_ratio\":0.875");
  }
  const auto mismatched = runtime::parse_prefill_a4_calibration_policy(
      mismatched_shared_activation, manifest);
  test.expect(!mismatched &&
                  mismatched.diagnostic.code == runtime::
                                                   PrefillA4ConverterErrorCode::
                                                       kPolicyCoverageMismatch,
              "Gate and Up cannot disagree on their shared activation policy");

  const auto experimental = runtime::parse_prefill_a4_calibration_policy(
      policy_json(manifest, 400U, "experimental_nearest_even_smoke"),
      manifest);
  test.expect(!experimental &&
                  experimental.diagnostic.code == runtime::
                                                       PrefillA4ConverterErrorCode::
                                                           kPublicationRejected,
              "experimental nearest-round policy cannot become production");

  std::string underflow_ratio = policy_json(manifest);
  const std::size_t ratio = underflow_ratio.find("0.9375");
  if (ratio != std::string::npos) {
    underflow_ratio.replace(ratio, 6U, "1e-300");
  }
  const auto underflow = runtime::parse_prefill_a4_calibration_policy(
      underflow_ratio, manifest);
  test.expect(!underflow &&
                  underflow.diagnostic.code == runtime::
                                                 PrefillA4ConverterErrorCode::
                                                     kUnsupportedCalibration,
              "double-to-float underflow clip policy fails closed");
}

void test_consumer_prepack_quantization_and_io(TestContext& test) {
  constexpr std::size_t kRows = 128U;
  constexpr std::size_t kColumns = 128U;
  std::vector<float> source(kRows * kColumns, 0.0F);
  const std::array<float, 16U> values = {
      -7.0F, -6.0F, -5.0F, -4.0F, -3.0F, -2.0F, -1.0F, 0.0F,
      1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  0.5F,
  };
  std::copy(values.begin(), values.end(), source.begin());
  source[16] = 1.5F;
  source[17] = 2.5F;
  source[18] = -0.5F;
  source[19] = -1.5F;
  source[64] = 7.0F;
  source[65] = -7.0F;
  // Raw FP32 scale makes the second value an exact -6.5 tie (-6). The stored
  // BF16 scale is slightly smaller, so the required stored-scale path emits
  // -7 and distinguishes the two numerical contracts.
  const float noninteger_scale = 0.1F / 7.0F;
  source[2U * kColumns] = 0.1F;
  source[2U * kColumns + 1U] = -6.5F * noninteger_scale;
  source[64U * kColumns] = 7.0F;
  source[64U * kColumns + 1U] = -7.0F;
  runtime::PrefillA4ProjectionCalibration calibration;
  calibration.weight_clip_ratio = 1.0;
  calibration.rounding = runtime::PrefillA4Rounding::kNearestEvenV1;
  std::vector<std::uint8_t> packed(kRows * kColumns / 2U, 0U);
  std::vector<std::uint8_t> scales(
      kRows * kColumns / 64U * 2U, 0U);
  const auto diagnostic = runtime::quantize_prefill_a4_k64_consumer_blocks(
      source.data(), kRows, kColumns, calibration,
      runtime::PrefillA4ConversionMode::kExperimentalNearestEvenSmoke,
      packed.data(), packed.size(), scales.data(), scales.size());
  test.expect(diagnostic.ok(), "bounded synthetic nearest-even smoke quantizes");
  const auto short_buffer = runtime::quantize_prefill_a4_k64_consumer_blocks(
      source.data(), kRows, kColumns, calibration,
      runtime::PrefillA4ConversionMode::kExperimentalNearestEvenSmoke,
      packed.data(), packed.size() - 1U, scales.data(), scales.size());
  test.expect(!short_buffer &&
                  short_buffer.code ==
                      runtime::PrefillA4ConverterErrorCode::kInvalidOption,
              "public quantizer rejects undersized destination buffers");
  const std::array<std::uint8_t, 10U> expected = {
      0xa9U, 0xcbU, 0xedU, 0x0fU, 0x21U,
      0x43U, 0x65U, 0x07U, 0x22U, 0xe0U,
  };
  test.expect(std::equal(expected.begin(), expected.end(), packed.begin()),
              "signed W4 is two's-complement packed low-even/high-odd");
  test.expect(scales[0] == 0x80U && scales[1] == 0x3fU &&
                  scales[2] == 0x80U && scales[3] == 0x3fU &&
                  scales[4] == 0x6aU && scales[5] == 0x3cU &&
                  scales[128] == 0x80U && scales[129] == 0x3fU &&
                  scales[256] == 0x80U && scales[257] == 0x3fU,
              "consumer scales are BF16 LE and zero groups store one");
  test.expect(packed[32] == 0U && packed[2'048] == 0x97U &&
                  packed[4'096] == 0x97U,
              "K64 and N64 consumer block ordering is exact");
  test.expect(packed[64] == 0x97U,
              "codes divide by decoded BF16 scale rather than raw FP32 scale");

  const auto missing_activation =
      runtime::quantize_prefill_a4_k64_consumer_blocks(
          source.data(), kRows, kColumns, calibration,
          runtime::PrefillA4ConversionMode::kProductionCalibrated,
          packed.data(), packed.size(), scales.data(), scales.size());
  test.expect(!missing_activation &&
                  missing_activation.code ==
                      runtime::PrefillA4ConverterErrorCode::
                          kUnsupportedCalibration,
              "production primitive cannot infer activation clipping");
  calibration.activation_clip_ratio = 0.96875;
  calibration.activation_scale_group_size = 64U;
  const auto production = runtime::quantize_prefill_a4_k64_consumer_blocks(
      source.data(), kRows, kColumns, calibration,
      runtime::PrefillA4ConversionMode::kProductionCalibrated, packed.data(),
      packed.size(), scales.data(), scales.size());
  test.expect(production.ok(),
              "production primitive accepts explicit weight/activation K64 policy");

  const fs::path directory =
      fs::temp_directory_path() /
      ("q3x-a4-sidecar-io-" + std::to_string(::getpid()));
  std::error_code ignored;
  fs::remove_all(directory, ignored);
  fs::create_directories(directory, ignored);
  const fs::path path = directory / "smoke.bin";
  {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    std::array<char, 256U> alignment{};
    output.write(alignment.data(), alignment.size());
    output.write(reinterpret_cast<const char*>(packed.data()),
                 static_cast<std::streamsize>(packed.size()));
    output.write(reinterpret_cast<const char*>(scales.data()),
                 static_cast<std::streamsize>(scales.size()));
  }
  std::vector<std::uint8_t> round_trip(packed.size() + scales.size());
  {
    std::ifstream input(path, std::ios::binary);
    input.seekg(256);
    input.read(reinterpret_cast<char*>(round_trip.data()),
               static_cast<std::streamsize>(round_trip.size()));
  }
  test.expect(std::equal(packed.begin(), packed.end(), round_trip.begin()) &&
                  std::equal(scales.begin(), scales.end(),
                             round_trip.begin() + packed.size()),
              "consumer-prepacked payload survives aligned host I/O round trip");
  fs::remove_all(directory, ignored);
}

void test_publication_gate(TestContext& test,
                           const runtime::PrefillSidecarManifest& manifest) {
  const std::string experimental_receipt =
      "{\"schema\":\"q3x.prefill.a4.publication-receipt\","
      "\"version\":{\"major\":1,\"minor\":0},"
      "\"mode\":\"experimental_nearest_even_smoke\","
      "\"production_residency_eligible\":false,"
      "\"physical_layout\":\"sm87_s4_n64_k64_consumer_v1\","
      "\"source_checkpoint_id\":\"" +
      manifest.source_checkpoint_id + "\",\"source_config_sha256\":\"" +
      manifest.source_config_sha256 + "\",\"source_index_sha256\":\"" +
      manifest.source_index_sha256 + "\",\"manifest_sha256\":\"" +
      manifest.manifest_sha256 + "\",\"policy_sha256\":\"" +
      std::string(64U, 'a') + "\",\"policy_bytes\":123,"
      "\"payload_sha256\":\"" +
      std::string(64U, 'b') + "\",\"payload_bytes\":" +
      std::to_string(manifest.summary.arena_bytes) +
      ",\"projection_count\":400}";
  runtime::PrefillA4ConverterDiagnostic parse_diagnostic;
  const auto receipt = runtime::parse_prefill_a4_publication_receipt(
      experimental_receipt, parse_diagnostic);
  test.expect(receipt.has_value() && parse_diagnostic.ok(),
              "strict experimental receipt parses for audit visibility");

  std::string canonical_receipt = experimental_receipt;
  constexpr std::string_view kPhysicalLayout =
      "sm87_s4_n64_k64_consumer_v1";
  const std::size_t physical_layout = canonical_receipt.find(kPhysicalLayout);
  if (physical_layout != std::string::npos) {
    canonical_receipt.replace(physical_layout, kPhysicalLayout.size(),
                              "canonical_row_major");
  }
  runtime::PrefillA4ConverterDiagnostic canonical_diagnostic;
  const auto canonical = runtime::parse_prefill_a4_publication_receipt(
      canonical_receipt, canonical_diagnostic);
  test.expect(!canonical &&
                  canonical_diagnostic.code ==
                      runtime::PrefillA4ConverterErrorCode::
                          kPublicationRejected,
              "legacy canonical row-major receipt fails closed");
  if (receipt.has_value()) {
    const auto rejected =
        runtime::authenticate_prefill_a4_publication_for_residency(
            manifest, *receipt, "/does/not/exist",
            "/does/not/exist-policy");
    test.expect(!rejected &&
                    rejected.diagnostic.code ==
                        runtime::PrefillA4ConverterErrorCode::
                            kPublicationRejected,
                "experimental receipt is rejected before payload residency");
  }

  const std::string production_policy = policy_json(manifest);
  const auto parsed_policy = runtime::parse_prefill_a4_calibration_policy(
      production_policy, manifest);
  test.expect(parsed_policy.ok(), "production publication policy fixture parses");
  if (!parsed_policy) {
    return;
  }
  const fs::path directory =
      fs::temp_directory_path() /
      ("q3x-a4-publication-gate-" + std::to_string(::getpid()));
  std::error_code ignored;
  fs::remove_all(directory, ignored);
  fs::create_directories(directory, ignored);
  const fs::path policy_path = directory / "policy.json";
  {
    std::ofstream output(policy_path, std::ios::binary | std::ios::trunc);
    output.write(production_policy.data(),
                 static_cast<std::streamsize>(production_policy.size()));
  }
  runtime::PrefillA4PublicationReceipt production_receipt;
  production_receipt.mode =
      runtime::PrefillA4ConversionMode::kProductionCalibrated;
  production_receipt.production_residency_eligible = true;
  production_receipt.physical_layout =
      std::string(runtime::kPrefillA4PhysicalLayout);
  production_receipt.source_checkpoint_id = manifest.source_checkpoint_id;
  production_receipt.source_config_sha256 = manifest.source_config_sha256;
  production_receipt.source_index_sha256 = manifest.source_index_sha256;
  production_receipt.manifest_sha256 = manifest.manifest_sha256;
  production_receipt.policy_sha256 = parsed_policy.value->policy_sha256;
  production_receipt.policy_bytes = parsed_policy.value->policy_bytes;
  production_receipt.payload_sha256.assign(64U, 'b');
  production_receipt.payload_bytes = manifest.summary.arena_bytes;
  production_receipt.projection_count = manifest.summary.projection_count;
  const auto payload_missing =
      runtime::authenticate_prefill_a4_publication_for_residency(
          manifest, production_receipt, directory / "missing.bin", policy_path);
  test.expect(!payload_missing &&
                  payload_missing.diagnostic.code ==
                      runtime::PrefillA4ConverterErrorCode::kOpenFailed,
              "valid policy bytes/SHA pass before missing payload is reported");
  {
    std::ofstream output(policy_path, std::ios::binary | std::ios::app);
    output.put('\n');
  }
  const auto mutated_policy =
      runtime::authenticate_prefill_a4_publication_for_residency(
          manifest, production_receipt, directory / "missing.bin", policy_path);
  test.expect(!mutated_policy &&
                  mutated_policy.diagnostic.code !=
                      runtime::PrefillA4ConverterErrorCode::kOpenFailed,
              "receipt rejects changed policy length/SHA before payload open");
  fs::remove_all(directory, ignored);
}

}  // namespace

int main() {
  TestContext test;
  const SyntheticSource source = make_source();
  const runtime::PrefillSidecarManifest manifest = build_manifest(source);
  test.expect(manifest.projections.size() == 400U &&
                  manifest.kind == runtime::PrefillSidecarKind::kA4K64 &&
                  runtime::validate_prefill_sidecar_manifest(manifest).ok(),
              "valid full-model A4-K64 fixture builds");
  if (manifest.projections.size() == 400U) {
    test_policy_gate(test, manifest);
    test_publication_gate(test, manifest);
  }
  test_consumer_prepack_quantization_and_io(test);

  if (test.failures() != 0) {
    std::cerr << test.failures() << " A4 sidecar converter checks failed\n";
    return 1;
  }
  std::cout << "A4-K64 sidecar converter host contract passed\n";
  return 0;
}
