#include "q3x/runtime/prefill_mlp_k512_overlay.h"

#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace checkpoint = q3x::model::checkpoint;
namespace fs = std::filesystem;
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
                const st::DType dtype, std::vector<std::uint64_t> shape) {
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

void add_nvfp4(SyntheticSource& source, const std::string& module,
               const std::uint64_t n, const std::uint64_t k) {
  add_tensor(source, module + ".weight", st::DType::kU8, {n, k / 2U});
  add_tensor(source, module + ".weight_scale", st::DType::kF8E4M3,
             {n, k / 16U});
  add_tensor(source, module + ".weight_scale_2", st::DType::kF32, {});
  add_tensor(source, module + ".input_scale", st::DType::kF32, {});
}

void add_fp8(SyntheticSource& source, const std::string& module,
             const std::uint64_t n, const std::uint64_t k) {
  add_tensor(source, module + ".weight", st::DType::kF8E4M3, {n, k});
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
    add_nvfp4(source, prefix + "mlp.gate_proj", 17'408U, 5'120U);
    add_nvfp4(source, prefix + "mlp.up_proj", 17'408U, 5'120U);
    add_nvfp4(source, prefix + "mlp.down_proj", 5'120U, 17'408U);
    if (((layer + 1U) % 4U) != 0U) {
      add_fp8(source, prefix + "linear_attn.in_proj_qkv", 10'240U, 5'120U);
      add_fp8(source, prefix + "linear_attn.in_proj_z", 6'144U, 5'120U);
      add_fp8(source, prefix + "linear_attn.out_proj", 5'120U, 6'144U);
    } else {
      add_fp8(source, prefix + "self_attn.q_proj", 12'288U, 5'120U);
      add_fp8(source, prefix + "self_attn.k_proj", 1'024U, 5'120U);
      add_fp8(source, prefix + "self_attn.v_proj", 1'024U, 5'120U);
      add_fp8(source, prefix + "self_attn.o_proj", 5'120U, 6'144U);
    }
  }
  return source;
}

[[nodiscard]] runtime::PrefillMLPK512BaseBinding make_base(
    const SyntheticSource& source, const bool k256 = false) {
  runtime::PrefillMLPK512BaseBinding base;
  base.physical_layout = std::string(
      k256 ? runtime::kPrefillA4K256PhysicalLayout
           : runtime::kPrefillA4K128PhysicalLayout);
  runtime::PrefillSidecarManifestOptions options;
  options.kind = k256 ? runtime::PrefillSidecarKind::kA4K256
                      : runtime::PrefillSidecarKind::kA4K128;
  const auto full = runtime::build_qwen36_27b_prefill_sidecar_manifest(
      source.manifest, source.shards, options);
  base.manifest_sha256 =
      full ? full.value->manifest_sha256 : std::string(64U, '1');
  base.policy_sha256 = std::string(64U, '2');
  base.payload_sha256 = std::string(64U, '3');
  return base;
}

[[nodiscard]] std::string read_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void test_manifest(Test& test,
                   const runtime::PrefillMLPK512OverlayManifest& m) {
  test.expect(m.projections.size() == 192U &&
                  m.payload_bytes == 8'623'226'880ULL &&
                  m.physical_layout ==
                      runtime::kPrefillMLPK512OverlayLayout,
              "fixed 192-entry manifest and exact payload size");
  std::size_t gate = 0U;
  std::size_t up = 0U;
  std::size_t down = 0U;
  for (std::size_t index = 0U; index < m.projections.size(); ++index) {
    const auto& entry = m.projections[index];
    const std::size_t position = index % 3U;
    gate += entry.family == "gate" ? 1U : 0U;
    up += entry.family == "up" ? 1U : 0U;
    down += entry.family == "down" ? 1U : 0U;
    test.expect(entry.layer_index == index / 3U && entry.ordinal == index &&
                    entry.family ==
                        (position == 0U ? "gate"
                                        : (position == 1U ? "up" : "down")) &&
                    entry.sidecar_offset == index * 44'912'640ULL &&
                    entry.weight_bytes == 44'564'480ULL &&
                    entry.scale_bytes == 348'160ULL &&
                    entry.output_size ==
                        (position == 2U ? 5'120U : 17'408U) &&
                    entry.input_size ==
                        (position == 2U ? 17'408U : 5'120U),
                "each MLP entry follows layer-local gate/up/down order and exact layout");
  }
  test.expect(gate == 64U && up == 64U && down == 64U,
              "manifest has exactly 64 gate, 64 up, and 64 down projections");
  auto corrupted = m;
  ++corrupted.projections[3U].sidecar_offset;
  test.expect(
      !runtime::validate_prefill_mlp_k512_overlay_manifest(corrupted),
      "manifest offset tampering fails closed");
}

void test_quantizer(Test& test) {
  constexpr std::size_t kRows = 64U;
  constexpr std::size_t kColumns = 512U;
  std::vector<float> source(kRows * kColumns);
  for (std::size_t row = 0U; row < kRows; ++row) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      const float amplitude =
          static_cast<float>(1U + column / 128U) +
          static_cast<float>(row) * 0.00390625F;
      source[row * kColumns + column] =
          ((row + column) & 1U) == 0U ? amplitude : -amplitude;
    }
  }
  std::vector<std::uint8_t> packed(kRows * kColumns / 2U);
  std::vector<std::uint8_t> packed_repeat(packed.size());
  std::vector<std::uint8_t> scales(kRows * kColumns / 512U * 2U);
  std::vector<std::uint8_t> scales_repeat(scales.size());
  const auto first = runtime::quantize_prefill_mlp_k512_consumer_blocks(
      source.data(), kRows, kColumns, 0.75, packed.data(), packed.size(),
      scales.data(), scales.size());
  const auto second =
      runtime::quantize_prefill_mlp_k512_consumer_blocks(
          source.data(), kRows, kColumns, 0.75, packed_repeat.data(),
          packed_repeat.size(), scales_repeat.data(), scales_repeat.size());
  test.expect(first.ok() && second.ok() && packed == packed_repeat &&
                  scales == scales_repeat,
              "K512 nearest-even quantization is deterministic");

  runtime::PrefillA4ProjectionCalibration k128_calibration;
  k128_calibration.source_module = "fixture";
  k128_calibration.source_sha256 = std::string(64U, '4');
  k128_calibration.weight_clip_ratio = 0.75;
  k128_calibration.activation_clip_ratio = 1.0;
  k128_calibration.activation_scale_group_size = 128U;
  std::vector<std::uint8_t> k128_packed(packed.size());
  std::vector<std::uint8_t> k128_scales(kRows * kColumns / 128U * 2U);
  const auto k128 = runtime::quantize_prefill_a4_k128_consumer_blocks(
      source.data(), kRows, kColumns, k128_calibration,
      runtime::PrefillA4ConversionMode::kProductionCalibrated,
      k128_packed.data(), k128_packed.size(), k128_scales.data(),
      k128_scales.size());
  test.expect(k128.ok() && packed != k128_packed,
              "K512 codes are independently requantized, not copied from K128");

  source[17U] = std::numeric_limits<float>::quiet_NaN();
  const auto nonfinite =
      runtime::quantize_prefill_mlp_k512_consumer_blocks(
          source.data(), kRows, kColumns, 0.75, packed.data(), packed.size(),
          scales.data(), scales.size());
  test.expect(nonfinite.code == runtime::
                                    PrefillMLPK512OverlayErrorCode::
                                        kQuantizationFailure,
              "non-finite original source fails closed");
}

void test_policy_publication(
    Test& test,
    const runtime::PrefillMLPK512OverlayManifest& manifest) {
  const fs::path path = fs::temp_directory_path() /
                        ("q3x-mlp-k512-policy-" +
                         std::to_string(::getpid()) + ".json");
  std::error_code ignored;
  fs::remove(path, ignored);
  const auto first =
      runtime::write_prefill_mlp_k512_overlay_policy_template(
          manifest, path, 0.75, 1.0);
  test.expect(static_cast<bool>(first) &&
                  first.value->projections.size() == 192U &&
                  first.value->projections.front().weight_clip_ratio == 0.75 &&
                  first.value->projections.front().activation_clip_ratio ==
                      1.0,
              "policy explicitly retains independent 0.75/1.0 clips");
  struct stat status {};
  test.expect(::lstat(path.c_str(), &status) == 0 &&
                  (status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0,
              "policy is atomically published read-only");
  const auto second =
      runtime::write_prefill_mlp_k512_overlay_policy_template(
          manifest, path, 0.75, 1.0);
  test.expect(!second && second.diagnostic.code ==
                             runtime::
                                 PrefillMLPK512OverlayErrorCode::
                                     kPublicationConflict,
              "policy publication never replaces an existing target");
  const std::string document = read_file(path);
  const auto parsed = runtime::parse_prefill_mlp_k512_overlay_policy(
      document, manifest);
  test.expect(static_cast<bool>(parsed) &&
                  parsed.value->policy_sha256.size() == 64U,
              "published policy round-trips through strict parser");

  const std::string activation_key = "\"activation_clip_ratio\":";
  const std::size_t gate_clip_key = document.find(activation_key);
  const std::size_t up_clip_key =
      gate_clip_key == std::string::npos
          ? std::string::npos
          : document.find(activation_key,
                          gate_clip_key + activation_key.size());
  test.expect(gate_clip_key != std::string::npos &&
                  up_clip_key != std::string::npos,
              "policy fixture exposes layer-zero Gate/Up activation clips");
  if (up_clip_key != std::string::npos) {
    const std::size_t value_begin = up_clip_key + activation_key.size();
    const std::size_t value_end = document.find(',', value_begin);
    if (value_end != std::string::npos) {
      std::string mismatched = document;
      mismatched.replace(value_begin, value_end - value_begin, "0.5");
      const auto mismatch =
          runtime::parse_prefill_mlp_k512_overlay_policy(mismatched,
                                                         manifest);
      test.expect(!mismatch &&
                      mismatch.diagnostic.code ==
                          runtime::PrefillMLPK512OverlayErrorCode::
                              kInvalidPolicy &&
                      mismatch.diagnostic.context ==
                          "policy.projections[0,1]",
                  "unequal layer-zero Gate/Up activation clips fail at "
                  "policy parse");

      std::string same_narrowed = document;
      same_narrowed.replace(value_begin, value_end - value_begin,
                            "0.99999999");
      test.expect(
          static_cast<bool>(runtime::parse_prefill_mlp_k512_overlay_policy(
              same_narrowed, manifest)),
          "Gate/Up activation clips equal after float narrowing are "
          "admitted");
    }
  }
  std::string corrupted = document;
  const std::size_t source = corrupted.find(manifest.projections[0].source_sha256);
  if (source != std::string::npos) {
    corrupted[source] = corrupted[source] == 'a' ? 'b' : 'a';
  }
  test.expect(!runtime::parse_prefill_mlp_k512_overlay_policy(
                  corrupted, manifest),
              "source binding corruption is rejected");
  fs::remove(path, ignored);
}

void test_receipt_parser(
    Test& test,
    const runtime::PrefillMLPK512OverlayManifest& manifest) {
  const std::string_view base_kind =
      manifest.required_base.physical_layout ==
              runtime::kPrefillA4K256PhysicalLayout
          ? "a4_k256"
          : "a4_k128";
  std::string document =
      "{\"schema\":\"q3x.prefill.mlp-k512.publication-receipt\","
      "\"version\":{\"major\":1,\"minor\":0},"
      "\"mode\":\"production_calibrated\","
      "\"production_residency_eligible\":true,"
      "\"physical_layout\":\"sm87_s4_n64_packed_k64_scale_k512_mlp_v1\","
      "\"packed_k_group_size\":64,\"scale_group_size\":512,"
      "\"source_checkpoint_id\":\"" +
      manifest.source_checkpoint_id + "\",\"source_config_sha256\":\"" +
      manifest.source_config_sha256 +
      "\",\"source_index_sha256\":\"" + manifest.source_index_sha256 +
      "\",\"manifest_sha256\":\"" + manifest.manifest_sha256 +
      "\",\"policy_sha256\":\"" + std::string(64U, '4') +
      "\",\"policy_bytes\":1234,\"required_base\":{"
      "\"sidecar_kind\":\"" + std::string(base_kind) +
      "\",\"physical_layout\":\"" +
      manifest.required_base.physical_layout +
      "\",\"manifest_sha256\":\"" +
      manifest.required_base.manifest_sha256 +
      "\",\"policy_sha256\":\"" + manifest.required_base.policy_sha256 +
      "\",\"payload_sha256\":\"" + manifest.required_base.payload_sha256 +
      "\"},\"payload_sha256\":\"" + std::string(64U, '5') +
      "\",\"payload_bytes\":8623226880,\"projection_count\":192}\n";
  runtime::PrefillMLPK512OverlayDiagnostic diagnostic;
  const auto receipt =
      runtime::parse_prefill_mlp_k512_overlay_receipt(document,
                                                               diagnostic);
  test.expect(receipt.has_value() && diagnostic.ok() &&
                  receipt->required_base.payload_sha256 ==
                      manifest.required_base.payload_sha256,
              "strict receipt retains the exact authenticated base binding");
  document.insert(document.size() - 2U, ",\"unknown\":0");
  test.expect(!runtime::parse_prefill_mlp_k512_overlay_receipt(
                  document, diagnostic),
              "receipt unknown fields fail closed");
}

}  // namespace

int main() {
  Test test;
  const SyntheticSource source = make_source();
  const auto built =
      runtime::build_qwen36_27b_prefill_mlp_k512_overlay_manifest(
          source.manifest, source.shards, make_base(source));
  test.expect(static_cast<bool>(built),
              "synthetic pinned source builds K512 MLP overlay manifest");
  if (!built) {
    std::cerr << runtime::to_string(built.diagnostic.code) << ' '
              << built.diagnostic.context << ' '
              << built.diagnostic.message << '\n';
    return 1;
  }
  test_manifest(test, *built.value);
  test_quantizer(test);
  test_policy_publication(test, *built.value);
  test_receipt_parser(test, *built.value);
  const auto built_k256 =
      runtime::build_qwen36_27b_prefill_mlp_k512_overlay_manifest(
          source.manifest, source.shards, make_base(source, true));
  test.expect(
      static_cast<bool>(built_k256) &&
          built_k256.value->required_base.physical_layout ==
              runtime::kPrefillA4K256PhysicalLayout &&
          built_k256.value->projections.size() == 192U,
      "K512 MLP overlay binds an authenticated K256 base without changing payload ABI");
  if (built_k256) {
    test_receipt_parser(test, *built_k256.value);
  }
  test.expect(
      runtime::prefill_mlp_k512_base_layout_matches_contract(
          runtime::kPrefillA4K128PhysicalLayout,
          runtime::PrefillSidecarKind::kA4K128, 64U, 128U) &&
          runtime::prefill_mlp_k512_base_layout_matches_contract(
              runtime::kPrefillA4K256PhysicalLayout,
              runtime::PrefillSidecarKind::kA4K256, 64U, 256U) &&
          !runtime::prefill_mlp_k512_base_layout_matches_contract(
              runtime::kPrefillA4K128PhysicalLayout,
              runtime::PrefillSidecarKind::kA4K256, 64U, 256U) &&
          !runtime::prefill_mlp_k512_base_layout_matches_contract(
              runtime::kPrefillA4K256PhysicalLayout,
              runtime::PrefillSidecarKind::kA4K128, 64U, 128U),
      "K128 and K256 base layouts cannot cross-bind the same digest tuple");
  if (test.result() == 0) {
    std::cout << "MLP K512 overlay host contract passed\n";
  }
  return test.result();
}
