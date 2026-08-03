#include "q3x/runtime/prefill_r1_projection_plane_v2.h"

#include "q3x/core/sha256.h"
#include "q3x/runtime/prefill_attention_factorized_lane_converter.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_converter.h"

#include <algorithm>
#include <array>
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

static_assert(runtime::kPrefillR1ProjectionPlaneV2PayloadBytes ==
              12'182'982'656ULL);
static_assert(runtime::kPrefillR1ProjectionPlaneV2LogicalProjectionCount ==
              400U);
static_assert(runtime::kPrefillR1ProjectionPlaneV2PhysicalProjectionCount ==
              336U);

void fill_base(runtime::PrefillMLPFactorizedLaneBaseK256Binding& base) {
  base.physical_layout =
      std::string(runtime::kPrefillMLPFactorizedLaneRequiredBaseK256Layout);
  base.manifest_sha256 = std::string(64U, '2');
  base.policy_sha256 = std::string(64U, '3');
  base.payload_sha256 = std::string(64U, '4');
  base.receipt_sha256 = std::string(64U, '5');
}

void fill_base(runtime::PrefillAttentionFactorizedLaneBaseK256Binding& base) {
  base.physical_layout = std::string(
      runtime::kPrefillAttentionFactorizedLaneRequiredBaseK256Layout);
  base.manifest_sha256 = std::string(64U, '2');
  base.policy_sha256 = std::string(64U, '3');
  base.payload_sha256 = std::string(64U, '4');
  base.receipt_sha256 = std::string(64U, '5');
}

[[nodiscard]] runtime::PrefillMLPFactorizedLaneOverlayManifestBinding
make_mlp_manifest() {
  runtime::PrefillMLPFactorizedLaneOverlayManifestBinding manifest;
  manifest.physical_layout =
      std::string(runtime::kPrefillMLPFactorizedLaneOverlayLayout);
  manifest.source_checkpoint_id = "pinned-qwen36-27b";
  manifest.source_config_sha256 = std::string(64U, '0');
  manifest.source_index_sha256 = std::string(64U, '1');
  fill_base(manifest.required_base_k256);
  manifest.lane_count = 1U;
  const auto plan =
      runtime::prefill_mlp_factorized_lane_overlay_layout_plan(1U);
  manifest.payload_bytes = plan.payload_bytes;
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
      entry.source_sha256 = std::string(64U, "678"[position]);
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
      manifest.projections.push_back(std::move(entry));
    }
  }
  manifest.manifest_sha256 =
      runtime::prefill_mlp_factorized_lane_r1_manifest_sha256(manifest);
  return manifest;
}

[[nodiscard]] std::string attention_suffix(
    const runtime::PrefillAttentionFactorizedLaneProjectionFamily family) {
  switch (family) {
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      return "linear_attn.in_proj_qkv";
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      return "linear_attn.in_proj_z";
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      return "linear_attn.out_proj";
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      return "self_attn.q_proj";
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      return "self_attn.k_proj";
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      return "self_attn.v_proj";
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      return "self_attn.o_proj";
  }
  return "invalid";
}

[[nodiscard]] const runtime::PrefillA4FactorizedLaneProjectionLayoutPlan&
attention_plan(
    const runtime::PrefillAttentionFactorizedLaneOverlayLayoutPlan& plan,
    const runtime::PrefillAttentionFactorizedLaneProjectionFamily family) {
  switch (family) {
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      return plan.linear_qkv;
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      return plan.linear_z;
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      return plan.linear_o;
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      return plan.full_q;
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      return plan.full_k;
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      return plan.full_v;
    case runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      return plan.full_o;
  }
  return plan.linear_qkv;
}

[[nodiscard]]
runtime::PrefillAttentionFactorizedLaneOverlayManifestBinding
make_attention_manifest() {
  runtime::PrefillAttentionFactorizedLaneOverlayManifestBinding manifest;
  manifest.physical_layout =
      std::string(runtime::kPrefillAttentionFactorizedLaneOverlayLayout);
  manifest.source_checkpoint_id = "pinned-qwen36-27b";
  manifest.source_config_sha256 = std::string(64U, '0');
  manifest.source_index_sha256 = std::string(64U, '1');
  fill_base(manifest.required_base_k256);
  manifest.lane_count = 1U;
  const auto plan =
      runtime::prefill_attention_factorized_lane_overlay_layout_plan(1U);
  manifest.payload_bytes = plan.payload_bytes;
  for (std::uint32_t layer = 0U;
       layer < runtime::kPrefillAttentionFactorizedLaneLayerCount; ++layer) {
    const bool full =
        runtime::prefill_attention_factorized_lane_is_full_layer(layer);
    const std::array<
        runtime::PrefillAttentionFactorizedLaneProjectionFamily, 4U>
        full_families = {
            runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullQ,
            runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullK,
            runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullV,
            runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullO,
        };
    const std::array<
        runtime::PrefillAttentionFactorizedLaneProjectionFamily, 3U>
        linear_families = {
            runtime::PrefillAttentionFactorizedLaneProjectionFamily::
                kLinearQkv,
            runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ,
            runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearO,
        };
    const std::size_t count = full ? 4U : 3U;
    for (std::size_t position = 0U; position < count; ++position) {
      const auto family =
          full ? full_families[position] : linear_families[position];
      const auto& projection_plan = attention_plan(plan, family);
      runtime::PrefillAttentionFactorizedLaneManifestProjection entry;
      entry.ordinal = static_cast<std::uint32_t>(manifest.projections.size());
      entry.layer_index = layer;
      entry.family = family;
      entry.source_module =
          "model.language_model.layers." + std::to_string(layer) + "." +
          attention_suffix(family);
      entry.source_sha256 =
          std::string(64U, "9abcdef"[static_cast<std::size_t>(family)]);
      entry.output_size = projection_plan.output_size;
      entry.input_size = projection_plan.input_size;
      entry.payload_offset =
          runtime::
              prefill_attention_factorized_lane_projection_absolute_offset(
                  plan, layer, family);
      entry.payload_bytes = projection_plan.projection_bytes;
      manifest.projections.push_back(std::move(entry));
    }
  }
  manifest.manifest_sha256 =
      runtime::prefill_attention_factorized_lane_r1_manifest_sha256(manifest);
  return manifest;
}

[[nodiscard]]
runtime::PrefillR1ProjectionPlaneV2AuthenticatedPayloadView source_view(
    const runtime::PrefillMLPFactorizedLaneOverlayManifestBinding& manifest) {
  runtime::PrefillR1ProjectionPlaneV2AuthenticatedPayloadView view;
  view.bytes = manifest.payload_bytes;
  view.version_major = manifest.version_major;
  view.version_minor = manifest.version_minor;
  view.physical_layout = manifest.physical_layout;
  view.manifest_sha256 = manifest.manifest_sha256;
  view.policy_sha256 = std::string(64U, 'a');
  view.payload_sha256 = std::string(64U, 'b');
  view.receipt_sha256 = std::string(64U, 'c');
  view.authenticated = true;
  return view;
}

[[nodiscard]]
runtime::PrefillR1ProjectionPlaneV2AuthenticatedPayloadView source_view(
    const runtime::PrefillAttentionFactorizedLaneOverlayManifestBinding&
        manifest) {
  runtime::PrefillR1ProjectionPlaneV2AuthenticatedPayloadView view;
  view.bytes = manifest.payload_bytes;
  view.version_major = manifest.version_major;
  view.version_minor = manifest.version_minor;
  view.physical_layout = manifest.physical_layout;
  view.manifest_sha256 = manifest.manifest_sha256;
  view.policy_sha256 = std::string(64U, 'd');
  view.payload_sha256 = std::string(64U, 'e');
  view.receipt_sha256 = std::string(64U, 'f');
  view.authenticated = true;
  return view;
}

[[nodiscard]] std::string sha256_text(const std::string_view document) {
  q3x::core::Sha256 hash;
  if (!hash.update(document.data(), document.size())) {
    return {};
  }
  return hash.finalize().hex();
}

void seed_projection(std::vector<std::uint8_t>& bytes,
                     const runtime::PrefillA4FactorizedLaneProjectionLayoutPlan&
                         plan,
                     const std::uint8_t bias) {
  std::fill(bytes.begin(), bytes.end(), 0x7bU);
  for (std::size_t index = 0U;
       index < static_cast<std::size_t>(plan.packed_weight_bytes); ++index) {
    bytes[static_cast<std::size_t>(plan.packed_weight_offset) + index] =
        static_cast<std::uint8_t>((index * 29U + bias) & 0xffU);
  }
  for (std::size_t index = 0U;
       index < static_cast<std::size_t>(plan.weight_scale_bytes); ++index) {
    bytes[static_cast<std::size_t>(plan.weight_scale_offset) + index] =
        static_cast<std::uint8_t>((index * 11U + bias + 3U) & 0xffU);
  }
  for (std::size_t index = 0U;
       index < static_cast<std::size_t>(plan.metadata_bytes); ++index) {
    bytes[static_cast<std::size_t>(plan.metadata_offset) + index] =
        static_cast<std::uint8_t>((index * 7U + bias + 5U) & 0xffU);
  }
}

[[nodiscard]] std::size_t canonical_offset(const std::size_t row,
                                           const std::size_t k64,
                                           const std::size_t byte_in_k64,
                                           const std::size_t input_size) {
  return (((row / 64U) * (input_size / 64U) + k64) * 64U + row % 64U) *
             32U +
         byte_in_k64;
}

void test_permutations(Test& test) {
  constexpr std::size_t gate_n = 64U;
  constexpr std::size_t gate_k = 64U;
  const auto gate_plan =
      runtime::prefill_a4_factorized_lane_projection_layout_plan(
          gate_n, gate_k, 1U);
  std::vector<std::uint8_t> gate(gate_plan.projection_bytes);
  std::vector<std::uint8_t> up(gate_plan.projection_bytes);
  seed_projection(gate, gate_plan, 13U);
  seed_projection(up, gate_plan, 71U);
  std::vector<std::uint8_t> paired(2U * gate_plan.projection_bytes, 0xffU);
  const auto gate_result =
      runtime::permute_prefill_r1_gate_up_projection_pair_v2(
          gate.data(), gate.size(), up.data(), up.size(), gate_n, gate_k,
          paired.data(), paired.size());
  test.expect(gate_result.ok(),
              "small Gate+Up v4 payloads permute without requantization");
  for (std::size_t n8 = 0U; n8 < 8U; ++n8) {
    for (std::size_t lane = 0U; lane < 32U; ++lane) {
      const std::size_t slot = static_cast<std::size_t>(
          runtime::prefill_r1_projection_plane_v2_gate_up_code_slot_offset(
              n8 * 8U, 0U, n8, lane, gate_n, gate_k));
      const std::size_t row = n8 * 8U + lane / 4U;
      for (std::size_t word = 0U; word < 2U; ++word) {
        const std::size_t source = canonical_offset(
            row, 0U, word * 16U + 4U * (lane % 4U), gate_k);
        test.expect(
            std::memcmp(paired.data() + slot + word * 4U,
                        gate.data() + source, 4U) == 0 &&
                std::memcmp(paired.data() + slot + 8U + word * 4U,
                            up.data() + source, 4U) == 0,
            "Gate8/Up8 fragment slot preserves exact source code bytes");
      }
    }
  }
  const std::size_t pair_scale_offset =
      static_cast<std::size_t>(2U * gate_plan.packed_weight_bytes);
  for (std::size_t row = 0U; row < gate_n; ++row) {
    test.expect(
        std::memcmp(paired.data() + pair_scale_offset + row * 4U,
                    gate.data() + gate_plan.weight_scale_offset + row * 2U,
                    2U) == 0 &&
            std::memcmp(
                paired.data() + pair_scale_offset + row * 4U + 2U,
                up.data() + gate_plan.weight_scale_offset + row * 2U,
                2U) == 0,
        "Gate/Up R1 scales are interleaved bit-exactly");
  }
  const auto align256 = [](const std::size_t value) {
    return (value + 255U) & ~std::size_t{255U};
  };
  const std::size_t gate_metadata_offset =
      align256(pair_scale_offset + 2U * gate_plan.weight_scale_bytes);
  const std::size_t up_metadata_offset =
      align256(gate_metadata_offset + gate_plan.metadata_bytes);
  test.expect(
      std::memcmp(paired.data() + gate_metadata_offset,
                  gate.data() + gate_plan.metadata_offset,
                  gate_plan.metadata_bytes) == 0 &&
          std::memcmp(paired.data() + up_metadata_offset,
                      up.data() + gate_plan.metadata_offset,
                      gate_plan.metadata_bytes) == 0,
      "Gate/Up inverse-alpha and factor metadata copy bit-exactly");

  constexpr std::size_t adjacent_n = 128U;
  constexpr std::size_t adjacent_k = 64U;
  const auto adjacent_plan =
      runtime::prefill_a4_factorized_lane_projection_layout_plan(
          adjacent_n, adjacent_k, 1U);
  std::vector<std::uint8_t> source(adjacent_plan.projection_bytes);
  std::vector<std::uint8_t> adjacent(adjacent_plan.projection_bytes, 0xffU);
  seed_projection(source, adjacent_plan, 31U);
  const auto adjacent_result =
      runtime::permute_prefill_r1_adjacent_n8_projection_v2(
          source.data(), source.size(), adjacent_n, adjacent_k,
          adjacent.data(), adjacent.size());
  test.expect(adjacent_result.ok(),
              "small single R1 payload permutes to adjacent-N8 order");
  for (std::size_t n16 = 0U; n16 < 8U; ++n16) {
    for (std::size_t lane = 0U; lane < 32U; ++lane) {
      const std::size_t slot = static_cast<std::size_t>(
          runtime::
              prefill_r1_projection_plane_v2_adjacent_n8_code_slot_offset(
                  0U, 0U, n16, lane, adjacent_n, adjacent_k));
      for (std::size_t word = 0U; word < 4U; ++word) {
        const std::size_t row =
            n16 * 16U + (word / 2U) * 8U + lane / 4U;
        const std::size_t source_offset = canonical_offset(
            row, 0U, (word % 2U) * 16U + 4U * (lane % 4U), adjacent_k);
        test.expect(std::memcmp(adjacent.data() + slot + word * 4U,
                                source.data() + source_offset, 4U) == 0,
                    "adjacent-N8 slot preserves the exact four u32 words");
      }
    }
  }
  test.expect(
      std::memcmp(adjacent.data() + adjacent_plan.weight_scale_offset,
                  source.data() + adjacent_plan.weight_scale_offset,
                  adjacent_plan.weight_scale_bytes) == 0 &&
          std::memcmp(adjacent.data() + adjacent_plan.metadata_offset,
                      source.data() + adjacent_plan.metadata_offset,
                      adjacent_plan.metadata_bytes) == 0,
      "single-projection scales, inverse alpha, and factor metadata copy "
      "bit-exactly");
}

void test_manifest_receipt_and_atomic_install(Test& test) {
  auto mlp = make_mlp_manifest();
  auto attention = make_attention_manifest();
  const auto mlp_source_policy =
      runtime::build_prefill_mlp_factorized_lane_r1_policy(
          mlp, 0.75, 1.0);
  const auto attention_source_policy =
      runtime::build_prefill_attention_factorized_lane_r1_policy(
          attention, 0.875, 0.9375);
  test.expect(mlp_source_policy && attention_source_policy,
              "authenticated v4 source policy fixtures build");
  if (!mlp_source_policy || !attention_source_policy) return;
  auto mlp_view = source_view(mlp);
  auto attention_view = source_view(attention);
  mlp_view.policy_sha256 =
      mlp_source_policy.value->binding.policy_sha256;
  attention_view.policy_sha256 =
      attention_source_policy.value->binding.policy_sha256;
  const auto built = runtime::build_prefill_r1_projection_plane_v2_manifest(
      mlp, mlp_view, attention, attention_view);
  test.expect(
      built && built.value->projections.size() == 336U &&
          built.value->logical_projection_count == 400U &&
          built.value->payload_bytes == 12'182'982'656ULL &&
          built.value->required_base_k256.payload_sha256 ==
              mlp.required_base_k256.payload_sha256 &&
          built.value->required_base_k256.receipt_sha256 ==
              attention.required_base_k256.receipt_sha256 &&
          built.value->projections[3U].family ==
              runtime::PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpGateUp &&
          built.value->projections[3U].logical_projection_count == 2U &&
          built.value->projections[4U].family ==
              runtime::PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpDown,
      "unified layer-major manifest closes 400 logical/336 physical entries");
  if (!built) return;

  const auto policy =
      runtime::build_prefill_r1_projection_plane_v2_policy(
          *built.value, *mlp_source_policy.value,
          *attention_source_policy.value);
  const auto receipt =
      policy ? runtime::build_prefill_r1_projection_plane_v2_receipt(
                   *built.value, *policy.value, std::string(64U, '6'))
             : runtime::PrefillR1ProjectionPlaneV2ReceiptResult{};
  test.expect(
      policy && receipt &&
          policy.value->projections.size() == 400U &&
          policy.value->projections[3U].source_module.find("gate_proj") !=
              std::string::npos &&
          policy.value->projections[4U].source_module.find("up_proj") !=
              std::string::npos &&
          receipt.value->mlp_v4.payload_sha256 ==
              mlp_view.payload_sha256 &&
          receipt.value->attention_v4.payload_sha256 ==
              attention_view.payload_sha256 &&
          receipt.value->mlp_v4.receipt_sha256 ==
              mlp_view.receipt_sha256 &&
          receipt.value->attention_v4.receipt_sha256 ==
              attention_view.receipt_sha256 &&
          receipt.value->atomic_installation_required &&
          !receipt.value->legacy_r1_co_residency_allowed &&
          receipt.canonical_document.find(mlp_view.payload_sha256) !=
              std::string::npos &&
          receipt.canonical_document.find(attention_view.payload_sha256) !=
              std::string::npos,
      "one package receipt binds both source payload and receipt identities");
  if (!policy || !receipt) return;

  const auto parsed_manifest =
      runtime::parse_prefill_r1_projection_plane_v2_manifest(
          built.canonical_document);
  const auto parsed_policy =
      parsed_manifest
          ? runtime::parse_prefill_r1_projection_plane_v2_policy(
                policy.canonical_document, *parsed_manifest.value)
          : runtime::PrefillR1ProjectionPlaneV2PolicyResult{};
  const auto parsed_receipt =
      parsed_manifest && parsed_policy
          ? runtime::parse_prefill_r1_projection_plane_v2_receipt(
                receipt.canonical_document, *parsed_manifest.value,
                *parsed_policy.value)
          : runtime::PrefillR1ProjectionPlaneV2ReceiptResult{};
  test.expect(parsed_manifest && parsed_policy && parsed_receipt,
              "canonical v2 manifest/policy/receipt strict-parse as one set");

  const auto replace_one = [](std::string document,
                              const std::string_view needle,
                              const std::string_view replacement) {
    const std::size_t position = document.find(needle);
    if (position != std::string::npos) {
      document.replace(position, needle.size(), replacement);
    }
    return document;
  };
  std::string unknown_manifest = built.canonical_document;
  unknown_manifest.insert(unknown_manifest.size() - 1U, ",\"unknown\":0");
  test.expect(!runtime::parse_prefill_r1_projection_plane_v2_manifest(
                  unknown_manifest),
              "unknown manifest fields fail closed");
  std::string duplicate_manifest = built.canonical_document;
  duplicate_manifest.insert(duplicate_manifest.size() - 1U,
                            ",\"lane_count\":1");
  test.expect(!runtime::parse_prefill_r1_projection_plane_v2_manifest(
                  duplicate_manifest),
              "duplicate manifest fields fail closed in the JSON parser");
  const std::string missing_manifest = replace_one(
      built.canonical_document, "\"lane_count\":1,", "");
  test.expect(!runtime::parse_prefill_r1_projection_plane_v2_manifest(
                  missing_manifest),
              "missing manifest fields fail closed");
  test.expect(!runtime::parse_prefill_r1_projection_plane_v2_manifest(
                  " " + built.canonical_document),
              "non-canonical manifest spelling/order boundary fails closed");

  const std::string policy_first =
      "\"attention_source_policy_sha256\":\"" +
      policy.value->attention_source_policy_sha256 + "\"";
  const std::string policy_second = "\"converter_abi\":\"" +
                                    policy.value->converter_abi + "\"";
  const std::string reordered_policy = replace_one(
      policy.canonical_document, "{" + policy_first + "," + policy_second,
      "{" + policy_second + "," + policy_first);
  test.expect(!runtime::parse_prefill_r1_projection_plane_v2_policy(
                  reordered_policy, *built.value),
              "reordered otherwise-valid policy fields fail closed");

  const std::string policy_hash_mutation = replace_one(
      policy.canonical_document, built.value->manifest_sha256,
      std::string(64U, '7'));
  test.expect(!runtime::parse_prefill_r1_projection_plane_v2_policy(
                  policy_hash_mutation, *built.value),
              "policy hash-binding mutation fails closed");
  const std::string policy_eligibility_mutation = replace_one(
      policy.canonical_document,
      "\"production_residency_eligible\":false",
      "\"production_residency_eligible\":true");
  test.expect(!runtime::parse_prefill_r1_projection_plane_v2_policy(
                  policy_eligibility_mutation, *built.value),
              "policy eligibility mutation fails closed");
  const std::string receipt_hash_mutation = replace_one(
      receipt.canonical_document, policy.value->policy_sha256,
      std::string(64U, '8'));
  test.expect(!runtime::parse_prefill_r1_projection_plane_v2_receipt(
                  receipt_hash_mutation, *built.value, *policy.value),
              "receipt hash-binding mutation fails closed");
  test.expect(!runtime::parse_prefill_r1_projection_plane_v2_receipt(
                  receipt.canonical_document + "\n", *built.value,
                  *policy.value),
              "non-canonical receipt spelling/order boundary fails closed");

  auto mutated_policy = *policy.value;
  mutated_policy.projections[3U].activation_clip_ratio = 0.5;
  test.expect(
      !runtime::validate_prefill_r1_projection_plane_v2_policy_sources(
          mutated_policy, *built.value, *mlp_source_policy.value,
          *attention_source_policy.value),
      "Gate logical calibration cannot differ from authenticated v4 policy");
  auto mutated_mlp_source_policy = *mlp_source_policy.value;
  mutated_mlp_source_policy.binding.projections[0U]
      .activation_clip_ratio = 0.5;
  test.expect(
      !runtime::validate_prefill_r1_projection_plane_v2_policy_sources(
          *policy.value, *built.value, mutated_mlp_source_policy,
          *attention_source_policy.value),
      "source-policy mutation is caught by the 400-entry equality gate");

  auto mutated_receipt = *receipt.value;
  mutated_receipt.attention_v4.payload_sha256 = std::string(64U, '7');
  test.expect(
      runtime::validate_prefill_r1_projection_plane_v2_receipt(
          mutated_receipt, *built.value, *policy.value)
              .code ==
          runtime::PrefillR1ProjectionPlaneV2ErrorCode::kInvalidReceipt,
      "changing either source payload hash invalidates the package receipt");
  mutated_receipt = *receipt.value;
  mutated_receipt.required_base_k256.payload_sha256 =
      std::string(64U, '7');
  test.expect(
      !runtime::validate_prefill_r1_projection_plane_v2_receipt(
          mutated_receipt, *built.value, *policy.value),
      "package receipt cannot detach v2 from the authenticated K256 base");

  std::uint8_t dummy = 0U;
  runtime::PrefillR1ProjectionPlaneV2AuthenticatedPayloadView output_view;
  output_view.data = &dummy;
  output_view.bytes = built.value->payload_bytes;
  output_view.version_major =
      runtime::kPrefillR1ProjectionPlaneV2VersionMajor;
  output_view.version_minor =
      runtime::kPrefillR1ProjectionPlaneV2VersionMinor;
  output_view.physical_layout =
      std::string(runtime::kPrefillR1ProjectionPlaneV2Layout);
  output_view.manifest_sha256 = built.value->manifest_sha256;
  output_view.policy_sha256 = policy.value->policy_sha256;
  output_view.payload_sha256 = receipt.value->payload_sha256;
  output_view.receipt_sha256 = sha256_text(receipt.canonical_document);
  output_view.authenticated = true;
  runtime::PrefillR1ProjectionPlaneV2Installation installation{
      &*built.value, &*policy.value, &*receipt.value, &output_view, false,
      false};
  test.expect(
      runtime::validate_prefill_r1_projection_plane_v2_atomic_installation(
          installation)
          .ok(),
      "complete single v2 triplet passes atomic installation validation");

  auto partial = installation;
  partial.policy = nullptr;
  test.expect(
      runtime::validate_prefill_r1_projection_plane_v2_atomic_installation(
          partial)
              .code ==
          runtime::PrefillR1ProjectionPlaneV2ErrorCode::kPartialInstallation,
      "partial v2 installation fails closed");
  auto mixed = installation;
  mixed.legacy_mlp_r1_installed = true;
  test.expect(
      runtime::validate_prefill_r1_projection_plane_v2_atomic_installation(
          mixed)
              .code ==
          runtime::PrefillR1ProjectionPlaneV2ErrorCode::kLegacyCoResidency,
      "split R1 v1 and unified v2 residency cannot be mixed");
  auto wrong_receipt_view = output_view;
  wrong_receipt_view.receipt_sha256 = std::string(64U, '0');
  auto wrong_receipt_install = installation;
  wrong_receipt_install.payload = &wrong_receipt_view;
  test.expect(
      !runtime::validate_prefill_r1_projection_plane_v2_atomic_installation(
          wrong_receipt_install),
      "payload authentication binds the exact package receipt bytes");

  auto v2_as_source = mlp_view;
  v2_as_source.version_major =
      runtime::kPrefillR1ProjectionPlaneV2VersionMajor;
  v2_as_source.physical_layout =
      std::string(runtime::kPrefillR1ProjectionPlaneV2Layout);
  test.expect(
      !runtime::build_prefill_r1_projection_plane_v2_manifest(
          mlp, v2_as_source, attention, attention_view),
      "v1/v2 source publication mixing is rejected before conversion");
}

}  // namespace

int main() {
  Test test;
  test_permutations(test);
  test_manifest_receipt_and_atomic_install(test);
  return test.result();
}
