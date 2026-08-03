#include "q3x/runtime/prefill_mlp_factorized_lane_r4_publication.h"

#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"

#include <algorithm>
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

  [[nodiscard]] int result() const noexcept {
    return failures_ == 0 ? 0 : 1;
  }

 private:
  int failures_ = 0;
};

static_assert(runtime::kPrefillMLPFactorizedLaneR4PublicationLaneCount ==
              4U);
static_assert(
    runtime::kPrefillMLPFactorizedLaneR4PublicationPayloadBytes ==
    8'583'954'432ULL);
static_assert(
    runtime::prefill_mlp_factorized_lane_overlay_layout_plan(4U)
        .payload_bytes == 8'583'954'432ULL);
static_assert(runtime::kPrefillMLPFactorizedLaneR4PublicationPayloadBytes !=
              8'568'619'008ULL);

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
  const auto& shard = source.shards[source.active_shard];
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

void add_nvfp4_projection(SyntheticSource& source,
                          const std::string& module,
                          const std::uint64_t output_size,
                          const std::uint64_t input_size) {
  add_tensor(source, module + ".weight", st::DType::kU8,
             {output_size, input_size / 2U});
  add_tensor(source, module + ".weight_scale", st::DType::kF8E4M3,
             {output_size, input_size / 16U});
  add_tensor(source, module + ".weight_scale_2", st::DType::kF32, {});
  add_tensor(source, module + ".input_scale", st::DType::kF32, {});
}

void add_fp8_projection(SyntheticSource& source,
                        const std::string& module,
                        const std::uint64_t output_size,
                        const std::uint64_t input_size) {
  add_tensor(source, module + ".weight", st::DType::kF8E4M3,
             {output_size, input_size});
  add_tensor(source, module + ".weight_scale", st::DType::kF32, {});
  add_tensor(source, module + ".input_scale", st::DType::kF32, {});
}

[[nodiscard]] SyntheticSource make_source() {
  SyntheticSource source;
  for (const auto& descriptor : checkpoint::known_checkpoint_catalog()) {
    if (descriptor.model == q3x::model::KnownModel::kQwen36_27B) {
      source.manifest.checkpoint = descriptor;
      break;
    }
  }
  source.shards = runtime::pinned_qwen36_27b_shards();
  for (std::uint32_t layer = 0U; layer < 64U; ++layer) {
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer) + ".";
    add_nvfp4_projection(source, prefix + "mlp.gate_proj", 17'408U,
                         5'120U);
    add_nvfp4_projection(source, prefix + "mlp.up_proj", 17'408U,
                         5'120U);
    add_nvfp4_projection(source, prefix + "mlp.down_proj", 5'120U,
                         17'408U);
    if (((layer + 1U) % 4U) != 0U) {
      add_fp8_projection(source, prefix + "linear_attn.in_proj_qkv",
                         10'240U, 5'120U);
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

[[nodiscard]] runtime::PrefillSidecarManifest build_exact_source(
    const SyntheticSource& source,
    const runtime::PrefillSidecarKind kind =
        runtime::PrefillSidecarKind::kExact) {
  runtime::PrefillSidecarManifestOptions options;
  options.kind = kind;
  auto result = runtime::build_qwen36_27b_prefill_sidecar_manifest(
      source.manifest, source.shards, options);
  return result ? std::move(*result.value)
                : runtime::PrefillSidecarManifest{};
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

[[nodiscard]] std::size_t object_end(const std::string_view text,
                                     const std::size_t begin) {
  std::size_t depth = 0U;
  bool in_string = false;
  bool escaped = false;
  for (std::size_t index = begin; index < text.size(); ++index) {
    const char character = text[index];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        in_string = false;
      }
      continue;
    }
    if (character == '"') {
      in_string = true;
    } else if (character == '{') {
      ++depth;
    } else if (character == '}') {
      if (--depth == 0U) {
        return index + 1U;
      }
    }
  }
  return std::string_view::npos;
}

[[nodiscard]] bool swap_first_two_projections(std::string& document) {
  const std::size_t array = document.find("\"projections\":[");
  if (array == std::string::npos) {
    return false;
  }
  const std::size_t first = document.find('{', array);
  const std::size_t first_end = object_end(document, first);
  if (first == std::string::npos || first_end == std::string::npos ||
      first_end >= document.size() || document[first_end] != ',') {
    return false;
  }
  const std::size_t second = first_end + 1U;
  const std::size_t second_end = object_end(document, second);
  if (second_end == std::string::npos) {
    return false;
  }
  const std::string first_object =
      document.substr(first, first_end - first);
  const std::string second_object =
      document.substr(second, second_end - second);
  document.replace(first, second_end - first,
                   second_object + "," + first_object);
  return true;
}

[[nodiscard]] std::vector<
    runtime::PrefillMLPFactorizedLaneR4CalibrationSpec>
make_calibration(
    const runtime::PrefillMLPFactorizedLaneR4Manifest& manifest) {
  std::vector<runtime::PrefillMLPFactorizedLaneR4CalibrationSpec>
      calibration(manifest.projections.size());
  for (std::size_t index = 0U; index < calibration.size(); ++index) {
    const std::size_t layer = index / 3U;
    const bool down = (index % 3U) == 2U;
    auto& spec = calibration[index];
    spec.weight_clip_ratio = down ? 0.75 : 0.875;
    spec.activation_clip_ratio = down ? 0.8125 : 0.9375;
    spec.alpha_path = "alpha/layer_" + std::to_string(layer) +
                      (down ? "_down.f32le" : "_gate_up.f32le");
    const char digest_character =
        static_cast<char>('a' + static_cast<int>(layer % 6U));
    spec.alpha_sha256 = std::string(64U, digest_character);
    spec.alpha_element_count = manifest.projections[index].input_size;
  }
  return calibration;
}

void test_manifest(Test& test,
                   const runtime::PrefillSidecarManifest& exact_source,
                   const runtime::PrefillSidecarManifest& derivative_source) {
  const auto built =
      runtime::build_prefill_mlp_factorized_lane_r4_direct_manifest(
          exact_source);
  test.expect(built && built.value->lane_count == 4U &&
                  built.value->payload_bytes == 8'583'954'432ULL &&
                  built.value->projections.size() == 192U &&
                  built.value->direct_source.source_manifest_kind ==
                      "exact" &&
                  built.value->direct_source.source_manifest_sha256 ==
                      exact_source.manifest_sha256 &&
                  built.value->manifest_bytes ==
                      built.canonical_document.size(),
              "Exact source builds independent fixed R4 manifest");
  test.expect(
      !runtime::build_prefill_mlp_factorized_lane_r4_direct_manifest(
          derivative_source),
      "derivative sidecar kind cannot impersonate direct Exact source");
  if (!built) {
    return;
  }

  test.expect(
      static_cast<bool>(
          runtime::parse_prefill_mlp_factorized_lane_r4_direct_manifest(
              built.canonical_document, exact_source)),
      "canonical manifest strict round-trip succeeds");

  std::string mutated = built.canonical_document;
  test.expect(
      replace_once(mutated, "{\"schema\":",
                   "{\"unknown\":0,\"schema\":") &&
          !runtime::parse_prefill_mlp_factorized_lane_r4_direct_manifest(
              mutated, exact_source),
      "unknown manifest field fails closed");
  mutated = built.canonical_document;
  test.expect(
      replace_once(mutated, "\"physical_layout\":",
                   "\"physical_layout\":\"duplicate\",\"physical_layout\":") &&
          !runtime::parse_prefill_mlp_factorized_lane_r4_direct_manifest(
              mutated, exact_source),
      "duplicate manifest field fails closed");
  mutated = built.canonical_document;
  test.expect(replace_once(mutated, "\"lane_count\":4,", "") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_direct_manifest(
                      mutated, exact_source),
              "missing manifest field fails closed");
  mutated = built.canonical_document;
  test.expect(swap_first_two_projections(mutated) &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_direct_manifest(
                      mutated, exact_source),
              "reordered manifest projections fail closed");
  mutated = built.canonical_document;
  test.expect(replace_once(mutated, exact_source.projections.front().source_sha256,
                           std::string(64U, 'f')) &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_direct_manifest(
                      mutated, exact_source),
              "source projection mutation fails closed");
  mutated = built.canonical_document;
  test.expect(replace_once(mutated, built.value->manifest_sha256,
                           std::string(64U, '0')) &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_direct_manifest(
                      mutated, exact_source),
              "manifest digest mismatch fails closed");
  test.expect(!runtime::parse_prefill_mlp_factorized_lane_r4_direct_manifest(
                  " " + built.canonical_document, exact_source),
              "noncanonical manifest bytes fail closed");
}

void test_policy_and_receipt(
    Test& test, const runtime::PrefillSidecarManifest& exact_source) {
  const auto manifest_result =
      runtime::build_prefill_mlp_factorized_lane_r4_direct_manifest(
          exact_source);
  if (!manifest_result) {
    test.expect(false, "manifest prerequisite builds");
    return;
  }
  const auto& manifest = *manifest_result.value;
  auto calibration = make_calibration(manifest);
  const auto policy = runtime::build_prefill_mlp_factorized_lane_r4_policy(
      manifest, calibration);
  test.expect(policy && policy.value->performance_candidate_only &&
                  !policy.value->production_residency_eligible &&
                  !policy.value->quality_production_eligible &&
                  policy.value->projections.size() == 192U &&
                  policy.value->projections.front().factor_scheme ==
                      "calibrated_alpha_f32_v1" &&
                  policy.value->policy_bytes ==
                      policy.canonical_document.size(),
              "strict R4 policy binds calibrated alpha and candidate scope");
  if (!policy) {
    return;
  }
  test.expect(static_cast<bool>(
                  runtime::parse_prefill_mlp_factorized_lane_r4_policy(
                      policy.canonical_document, manifest)),
              "canonical policy strict round-trip succeeds");

  auto bad_calibration = calibration;
  bad_calibration[1].alpha_path += ".different";
  test.expect(!runtime::build_prefill_mlp_factorized_lane_r4_policy(
                  manifest, bad_calibration),
              "Gate/Up alpha identity mismatch fails builder");
  bad_calibration = calibration;
  bad_calibration[1].activation_clip_ratio = 0.875;
  test.expect(!runtime::build_prefill_mlp_factorized_lane_r4_policy(
                  manifest, bad_calibration),
              "Gate/Up activation clip mismatch fails builder");
  bad_calibration = calibration;
  bad_calibration[2].alpha_path = "../unbound-alpha.f32le";
  test.expect(!runtime::build_prefill_mlp_factorized_lane_r4_policy(
                  manifest, bad_calibration),
              "noncanonical relative alpha path fails builder");

  std::string mutated = policy.canonical_document;
  test.expect(replace_once(mutated, "{\"schema\":",
                           "{\"unknown\":0,\"schema\":") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_policy(
                      mutated, manifest),
              "unknown policy field fails closed");
  mutated = policy.canonical_document;
  test.expect(replace_once(mutated, "\"mode\":",
                           "\"mode\":\"duplicate\",\"mode\":") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_policy(
                      mutated, manifest),
              "duplicate policy field fails closed");
  mutated = policy.canonical_document;
  test.expect(replace_once(mutated,
                           ",\"quality_production_eligible\":false", "") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_policy(
                      mutated, manifest),
              "missing policy field fails closed");
  mutated = policy.canonical_document;
  test.expect(swap_first_two_projections(mutated) &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_policy(
                      mutated, manifest),
              "reordered policy projections fail closed");
  mutated = policy.canonical_document;
  test.expect(replace_once(mutated, calibration[0].alpha_sha256,
                           std::string(64U, 'f')) &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_policy(
                      mutated, manifest),
              "Gate/Up factor digest mismatch fails closed");
  mutated = policy.canonical_document;
  test.expect(replace_once(mutated,
                           manifest.projections.front().source_sha256,
                           std::string(64U, '0')) &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_policy(
                      mutated, manifest),
              "policy source identity mutation fails closed");

  const auto receipt = runtime::build_prefill_mlp_factorized_lane_r4_receipt(
      manifest, *policy.value, std::string(64U, 'e'));
  test.expect(receipt && receipt.value->performance_candidate_only &&
                  !receipt.value->production_residency_eligible &&
                  !receipt.value->quality_production_eligible &&
                  receipt.value->manifest_sha256 ==
                      manifest.manifest_sha256 &&
                  receipt.value->manifest_bytes == manifest.manifest_bytes &&
                  receipt.value->policy_sha256 ==
                      policy.value->policy_sha256 &&
                  receipt.value->policy_bytes == policy.value->policy_bytes &&
                  receipt.value->payload_bytes == 8'583'954'432ULL,
              "receipt binds manifest/policy/payload SHA and byte counts");
  if (!receipt) {
    return;
  }
  test.expect(static_cast<bool>(
                  runtime::parse_prefill_mlp_factorized_lane_r4_receipt(
                      receipt.canonical_document, manifest, *policy.value)),
              "canonical receipt strict round-trip succeeds");

  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated,
                           "\"production_residency_eligible\":false",
                           "\"production_residency_eligible\":true") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_receipt(
                      mutated, manifest, *policy.value),
              "receipt cannot claim production residency");
  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated,
                           "\"quality_production_eligible\":false",
                           "\"quality_production_eligible\":true") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_receipt(
                      mutated, manifest, *policy.value),
              "receipt cannot claim quality eligibility");
  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated, policy.value->policy_sha256,
                           std::string(64U, '0')) &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_receipt(
                      mutated, manifest, *policy.value),
              "receipt policy digest mismatch fails closed");
  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated, "\"payload_bytes\":8583954432",
                           "\"payload_bytes\":8583954431") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_receipt(
                      mutated, manifest, *policy.value),
              "receipt payload byte mismatch fails closed");
  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated, "{\"schema\":",
                           "{\"unknown\":0,\"schema\":") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_receipt(
                      mutated, manifest, *policy.value),
              "unknown receipt field fails closed");
  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated,
                           ",\"performance_candidate_only\":true", "") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_receipt(
                      mutated, manifest, *policy.value),
              "missing receipt field fails closed");
  mutated = receipt.canonical_document;
  test.expect(replace_once(mutated, "\"mode\":",
                           "\"mode\":\"duplicate\",\"mode\":") &&
                  !runtime::parse_prefill_mlp_factorized_lane_r4_receipt(
                      mutated, manifest, *policy.value),
              "duplicate receipt field fails closed");
}

}  // namespace

int main() {
  Test test;
  const SyntheticSource source = make_source();
  const auto exact_source = build_exact_source(source);
  const auto derivative_source =
      build_exact_source(source, runtime::PrefillSidecarKind::kA4K256);
  test.expect(runtime::validate_prefill_sidecar_manifest(exact_source).ok(),
              "synthetic Exact source prerequisite validates");
  test.expect(
      runtime::validate_prefill_sidecar_manifest(derivative_source).ok(),
      "synthetic derivative source prerequisite validates");
  test_manifest(test, exact_source, derivative_source);
  test_policy_and_receipt(test, exact_source);
  return test.result();
}
