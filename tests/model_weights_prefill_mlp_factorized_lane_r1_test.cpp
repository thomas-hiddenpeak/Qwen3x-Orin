#include "q3x/runtime/model_weights.h"

#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"
#include "q3x/model/model_config.h"
#include "q3x/runtime/prefill_a4_sidecar_converter.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_converter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace checkpoint = q3x::model::checkpoint;
namespace model_weights = q3x::model::weights;
namespace runtime = q3x::runtime;
namespace st = q3x::io::safetensors;

class TestContext final {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

enum class SyntheticLinearKind : std::uint8_t {
  kBf16,
  kFp8,
  kNvFp4,
};

[[nodiscard]] constexpr std::uint64_t align256(
    const std::uint64_t value) noexcept {
  return (value + runtime::kResidentTensorAlignment - 1U) &
         ~(runtime::kResidentTensorAlignment - 1U);
}

[[nodiscard]] std::uint64_t tensor_bytes(
    const st::DType dtype, const std::vector<std::uint64_t>& shape) noexcept {
  std::uint64_t elements = 1U;
  for (const std::uint64_t dimension : shape) {
    elements *= dimension;
  }
  return elements * st::bit_width(dtype) / 8U;
}

// Builds only the views requested by ModelWeightBinder.  Fake device
// addresses keep this host-only test allocation-free with respect to the
// multi-gigabyte checkpoint while preserving the exact shape/range ABI.
class SyntheticArena final {
 public:
  [[nodiscard]] runtime::WeightBindingSource source() const noexcept {
    runtime::WeightBindingSource result;
    result.lookup_context = this;
    result.lookup = &lookup;
    result.arena_data = reinterpret_cast<const void*>(kBaseAddress);
    result.arena_bytes = kArenaBytes;
    result.scalar_read_context = this;
    result.scalar_read = &read_scalar;
    return result;
  }

  [[nodiscard]] const auto& tensors() const noexcept { return tensors_; }

 private:
  static constexpr std::uintptr_t kBaseAddress = 0x0000010000000000ULL;
  static constexpr std::uint64_t kArenaBytes = 64ULL * 1024ULL * 1024ULL *
                                               1024ULL;

  [[nodiscard]] static bool ends_with(const std::string_view value,
                                      const std::string_view suffix) noexcept {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
  }

  [[nodiscard]] static int read_scalar(const void* const context,
                                       const float* const device_value,
                                       float* const host_value) noexcept {
    if (context == nullptr || device_value == nullptr || host_value == nullptr) {
      return 1;
    }
    *host_value = 0.5F;
    return 0;
  }

  [[nodiscard]] bool describe_linear(
      const std::string_view name, const std::string_view module,
      const SyntheticLinearKind kind, const std::uint64_t output_size,
      const std::uint64_t input_size, st::DType& dtype,
      std::vector<std::uint64_t>& shape) const {
    if (name.find(module) == std::string_view::npos) {
      return false;
    }
    if (ends_with(name, ".weight")) {
      if (kind == SyntheticLinearKind::kBf16) {
        dtype = st::DType::kBf16;
        shape = {output_size, input_size};
      } else if (kind == SyntheticLinearKind::kFp8) {
        dtype = st::DType::kF8E4M3;
        shape = {output_size, input_size};
      } else {
        dtype = st::DType::kU8;
        shape = {output_size, input_size / 2U};
      }
      return true;
    }
    if (kind == SyntheticLinearKind::kNvFp4 &&
        ends_with(name, ".weight_scale")) {
      dtype = st::DType::kF8E4M3;
      shape = {output_size, input_size / 16U};
      return true;
    }
    if ((kind == SyntheticLinearKind::kFp8 &&
         ends_with(name, ".weight_scale")) ||
        (kind == SyntheticLinearKind::kNvFp4 &&
         ends_with(name, ".weight_scale_2")) ||
        ((kind == SyntheticLinearKind::kFp8 ||
          kind == SyntheticLinearKind::kNvFp4) &&
         ends_with(name, ".input_scale"))) {
      dtype = st::DType::kF32;
      shape.clear();
      return true;
    }
    return false;
  }

  [[nodiscard]] bool describe(const std::string_view name, st::DType& dtype,
                              std::vector<std::uint64_t>& shape) const {
    if (name == "model.language_model.embed_tokens.weight") {
      dtype = st::DType::kBf16;
      shape = {248'320U, 5'120U};
      return true;
    }
    if (name == "model.language_model.norm.weight" ||
        ends_with(name, ".input_layernorm.weight") ||
        ends_with(name, ".post_attention_layernorm.weight")) {
      dtype = st::DType::kBf16;
      shape = {5'120U};
      return true;
    }
    if (name == "lm_head.weight") {
      dtype = st::DType::kBf16;
      shape = {248'320U, 5'120U};
      return true;
    }

    if (describe_linear(name, ".mlp.gate_proj.",
                        SyntheticLinearKind::kNvFp4, 17'408U, 5'120U,
                        dtype, shape) ||
        describe_linear(name, ".mlp.up_proj.",
                        SyntheticLinearKind::kNvFp4, 17'408U, 5'120U,
                        dtype, shape) ||
        describe_linear(name, ".mlp.down_proj.",
                        SyntheticLinearKind::kNvFp4, 5'120U, 17'408U,
                        dtype, shape) ||
        describe_linear(name, ".linear_attn.in_proj_qkv.",
                        SyntheticLinearKind::kFp8, 10'240U, 5'120U, dtype,
                        shape) ||
        describe_linear(name, ".linear_attn.in_proj_z.",
                        SyntheticLinearKind::kFp8, 6'144U, 5'120U, dtype,
                        shape) ||
        describe_linear(name, ".linear_attn.in_proj_a.",
                        SyntheticLinearKind::kBf16, 48U, 5'120U, dtype,
                        shape) ||
        describe_linear(name, ".linear_attn.in_proj_b.",
                        SyntheticLinearKind::kBf16, 48U, 5'120U, dtype,
                        shape) ||
        describe_linear(name, ".linear_attn.out_proj.",
                        SyntheticLinearKind::kFp8, 5'120U, 6'144U, dtype,
                        shape) ||
        describe_linear(name, ".self_attn.q_proj.",
                        SyntheticLinearKind::kFp8, 12'288U, 5'120U, dtype,
                        shape) ||
        describe_linear(name, ".self_attn.k_proj.",
                        SyntheticLinearKind::kFp8, 1'024U, 5'120U, dtype,
                        shape) ||
        describe_linear(name, ".self_attn.v_proj.",
                        SyntheticLinearKind::kFp8, 1'024U, 5'120U, dtype,
                        shape) ||
        describe_linear(name, ".self_attn.o_proj.",
                        SyntheticLinearKind::kFp8, 5'120U, 6'144U, dtype,
                        shape)) {
      return true;
    }

    if (ends_with(name, ".linear_attn.conv1d.weight")) {
      dtype = st::DType::kBf16;
      shape = {10'240U, 1U, 4U};
      return true;
    }
    if (ends_with(name, ".linear_attn.A_log") ||
        ends_with(name, ".linear_attn.dt_bias")) {
      dtype = st::DType::kBf16;
      shape = {48U};
      return true;
    }
    if (ends_with(name, ".linear_attn.norm.weight")) {
      dtype = st::DType::kBf16;
      shape = {128U};
      return true;
    }
    if (ends_with(name, ".self_attn.q_norm.weight") ||
        ends_with(name, ".self_attn.k_norm.weight")) {
      dtype = st::DType::kBf16;
      shape = {256U};
      return true;
    }
    return false;
  }

  [[nodiscard]] static const runtime::DeviceTensorView* lookup(
      const void* const context, const std::string_view name) noexcept {
    auto* const self = const_cast<SyntheticArena*>(
        static_cast<const SyntheticArena*>(context));
    if (self == nullptr) {
      return nullptr;
    }
    const auto existing = self->tensors_.find(name);
    if (existing != self->tensors_.end()) {
      return &existing->second;
    }
    try {
      runtime::DeviceTensorView view;
      if (!self->describe(name, view.dtype, view.shape)) {
        return nullptr;
      }
      self->cursor_ = align256(self->cursor_);
      view.arena_offset = self->cursor_;
      view.byte_size = tensor_bytes(view.dtype, view.shape);
      view.device_data = reinterpret_cast<const void*>(
          kBaseAddress + static_cast<std::uintptr_t>(view.arena_offset));
      if (view.byte_size > kArenaBytes - view.arena_offset) {
        return nullptr;
      }
      self->cursor_ += view.byte_size;
      const auto inserted = self->tensors_.emplace(std::string(name),
                                                    std::move(view));
      return &inserted.first->second;
    } catch (...) {
      return nullptr;
    }
  }

  mutable std::map<std::string, runtime::DeviceTensorView, std::less<>>
      tensors_;
  mutable std::uint64_t cursor_ = 0U;
};

struct SyntheticA4ManifestSource final {
  model_weights::WeightManifest manifest;
  std::vector<runtime::ShardIdentity> shards;
  bool complete = false;
};

[[nodiscard]] bool is_a4_projection_component(
    const std::string_view name) noexcept {
  constexpr std::array<std::string_view, 10U> kModules = {
      ".mlp.gate_proj.",       ".mlp.up_proj.",
      ".mlp.down_proj.",       ".linear_attn.in_proj_qkv.",
      ".linear_attn.in_proj_z.", ".linear_attn.out_proj.",
      ".self_attn.q_proj.",    ".self_attn.k_proj.",
      ".self_attn.v_proj.",    ".self_attn.o_proj."};
  return std::any_of(kModules.begin(), kModules.end(), [name](const auto part) {
    return name.find(part) != std::string_view::npos;
  });
}

[[nodiscard]] SyntheticA4ManifestSource make_a4_manifest_source(
    const SyntheticArena& arena) {
  SyntheticA4ManifestSource source;
  for (const checkpoint::KnownCheckpointDescriptor& descriptor :
       checkpoint::known_checkpoint_catalog()) {
    if (descriptor.model == q3x::model::KnownModel::kQwen36_27B) {
      source.manifest.checkpoint = descriptor;
      break;
    }
  }
  source.shards = runtime::pinned_qwen36_27b_shards();
  std::size_t active_shard = 0U;
  std::uint64_t next_offset = 4'096U;
  std::size_t component_count = 0U;
  for (const auto& [name, view] : arena.tensors()) {
    if (!is_a4_projection_component(name)) {
      continue;
    }
    next_offset = align256(next_offset);
    while (active_shard < source.shards.size() &&
           (next_offset > source.shards[active_shard].file_size ||
            view.byte_size >
                source.shards[active_shard].file_size - next_offset)) {
      ++active_shard;
      next_offset = 4'096U;
    }
    if (active_shard >= source.shards.size()) {
      return source;
    }
    const runtime::ShardIdentity& shard = source.shards[active_shard];
    model_weights::TensorLocator locator;
    locator.category = model_weights::TensorCategory::kText;
    locator.shard = shard.filename;
    locator.file = shard.filename;
    locator.file_begin = next_offset;
    locator.file_end = next_offset + view.byte_size;
    locator.byte_size = view.byte_size;
    locator.dtype = view.dtype;
    locator.shape = view.shape;
    source.manifest.tensors.emplace(name, std::move(locator));
    next_offset += view.byte_size;
    ++component_count;
  }
  source.complete = component_count == 1'392U;
  return source;
}

[[nodiscard]] runtime::PrefillA4CalibrationPolicy make_a4_policy(
    const runtime::PrefillSidecarManifest& manifest) {
  runtime::PrefillA4CalibrationPolicy policy;
  policy.version_major = runtime::kPrefillA4K256CalibrationPolicyVersionMajor;
  policy.version_minor = runtime::kPrefillA4K256CalibrationPolicyVersionMinor;
  policy.sidecar_kind = runtime::PrefillSidecarKind::kA4K256;
  policy.physical_layout = std::string(runtime::kPrefillA4K256PhysicalLayout);
  policy.source_checkpoint_id = manifest.source_checkpoint_id;
  policy.source_config_sha256 = manifest.source_config_sha256;
  policy.source_index_sha256 = manifest.source_index_sha256;
  policy.manifest_sha256 = manifest.manifest_sha256;
  policy.policy_sha256.assign(64U, 'c');
  policy.policy_bytes = 1U;
  policy.projections.reserve(manifest.projections.size());
  for (const runtime::PrefillProjectionSidecarEntry& entry :
       manifest.projections) {
    runtime::PrefillA4ProjectionCalibration calibration;
    calibration.ordinal = entry.ordinal;
    calibration.source_module = entry.source_module;
    calibration.source_sha256 = entry.source_sha256;
    calibration.weight_clip_ratio = 1.0;
    calibration.activation_clip_ratio = 1.0;
    calibration.activation_scale_group_size =
        runtime::kPrefillA4K256WeightGroupSize;
    policy.projections.emplace_back(std::move(calibration));
  }
  return policy;
}

[[nodiscard]] bool same_linear_view(
    const runtime::PrefillMLPFactorizedLaneLinearSidecarView& left,
    const runtime::PrefillMLPFactorizedLaneLinearSidecarView& right) noexcept {
  return left.packed_weight == right.packed_weight &&
         left.packed_weight_capacity_bytes ==
             right.packed_weight_capacity_bytes &&
         left.lane_scales == right.lane_scales &&
         left.lane_scale_capacity_elements ==
             right.lane_scale_capacity_elements &&
         left.inverse_alpha == right.inverse_alpha &&
         left.inverse_alpha_capacity_elements ==
             right.inverse_alpha_capacity_elements &&
         left.output_size == right.output_size &&
         left.input_size == right.input_size &&
         left.lane_count == right.lane_count &&
         left.activation_clip_ratio == right.activation_clip_ratio;
}

using FactorizedSnapshot =
    std::array<runtime::PrefillMLPFactorizedLaneLayerView,
               runtime::kQwen36DenseLayerCount>;

[[nodiscard]] FactorizedSnapshot snapshot(const runtime::ModelWeights& weights) {
  FactorizedSnapshot result{};
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[index] = weights.layer(index).prefill_mlp_factorized_lane_r1;
  }
  return result;
}

[[nodiscard]] bool matches_snapshot(const runtime::ModelWeights& weights,
                                    const FactorizedSnapshot& expected) {
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    const auto& actual =
        weights.layer(index).prefill_mlp_factorized_lane_r1;
    if (!same_linear_view(actual.gate, expected[index].gate) ||
        !same_linear_view(actual.up, expected[index].up) ||
        !same_linear_view(actual.down, expected[index].down)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool all_factorized_empty(
    const runtime::ModelWeights& weights) noexcept {
  return std::all_of(
      weights.layers().begin(), weights.layers().end(),
      [](const runtime::DecoderLayerWeights& layer) {
        return layer.prefill_mlp_factorized_lane_r1.empty();
      });
}

[[nodiscard]] bool factorized_layout_matches(
    const runtime::ModelWeights& weights,
    const std::uintptr_t arena_address, const float activation_clip) {
  const auto plan = runtime::prefill_mlp_factorized_lane_overlay_layout_plan(
      runtime::kPrefillMLPFactorizedLaneR1LaneCount);
  if (!plan) {
    return false;
  }
  for (std::size_t layer_index = 0U; layer_index < weights.layers().size();
       ++layer_index) {
    const auto& layer =
        weights.layer(layer_index).prefill_mlp_factorized_lane_r1;
    if (!layer.attached()) {
      return false;
    }
    const std::array<runtime::PrefillMLPFactorizedLaneProjectionFamily, 3U>
        families = {
            runtime::PrefillMLPFactorizedLaneProjectionFamily::kGate,
            runtime::PrefillMLPFactorizedLaneProjectionFamily::kUp,
            runtime::PrefillMLPFactorizedLaneProjectionFamily::kDown};
    const std::array<const runtime::PrefillMLPFactorizedLaneLinearSidecarView*,
                     3U>
        views = {&layer.gate, &layer.up, &layer.down};
    for (std::size_t family_index = 0U; family_index < families.size();
         ++family_index) {
      const auto family = families[family_index];
      const auto& projection_plan =
          family == runtime::PrefillMLPFactorizedLaneProjectionFamily::kDown
              ? plan.down
              : family == runtime::PrefillMLPFactorizedLaneProjectionFamily::kUp
                    ? plan.up
                    : plan.gate;
      const std::uint64_t projection_offset =
          runtime::prefill_mlp_factorized_lane_projection_absolute_offset(
              plan, static_cast<std::uint32_t>(layer_index), family);
      const auto& view = *views[family_index];
      if (reinterpret_cast<std::uintptr_t>(view.packed_weight) !=
              arena_address + projection_offset +
                  projection_plan.packed_weight_offset ||
          reinterpret_cast<std::uintptr_t>(view.lane_scales) !=
              arena_address + projection_offset +
                  projection_plan.weight_scale_offset ||
          reinterpret_cast<std::uintptr_t>(view.inverse_alpha) !=
              arena_address + projection_offset +
                  projection_plan.inverse_alpha_offset ||
          view.packed_weight_capacity_bytes !=
              projection_plan.packed_weight_bytes ||
          view.lane_scale_capacity_elements !=
              projection_plan.weight_scale_elements ||
          view.inverse_alpha_capacity_elements !=
              projection_plan.inverse_alpha_elements ||
          view.activation_clip_ratio != activation_clip) {
        return false;
      }
    }
  }
  return true;
}

void test_factorized_r1_attachment(TestContext& test) {
  SyntheticArena source_arena;
  runtime::WeightBindResult bound =
      runtime::bind_qwen36_27b_weights(source_arena.source());
  test.expect(bound.ok(), "exact synthetic Qwen3.6 binding succeeds");
  if (!bound) {
    std::cerr << bound.diagnostic.message << " (" << bound.diagnostic.tensor
              << ")\n";
    return;
  }
  runtime::ModelWeights& weights = *bound.value;

  const SyntheticA4ManifestSource source =
      make_a4_manifest_source(source_arena);
  test.expect(source.complete, "synthetic source covers 400 A4 projections");
  if (!source.complete) {
    return;
  }
  runtime::PrefillSidecarManifestOptions options;
  options.kind = runtime::PrefillSidecarKind::kA4K256;
  const runtime::PrefillSidecarManifestResult base_result =
      runtime::build_qwen36_27b_prefill_sidecar_manifest(
          source.manifest, source.shards, options);
  test.expect(base_result.ok(), "strict K256 base manifest builds");
  if (!base_result) {
    return;
  }
  const runtime::PrefillSidecarManifest& base_manifest = *base_result.value;
  const runtime::PrefillA4CalibrationPolicy base_policy =
      make_a4_policy(base_manifest);
  const std::string base_payload_sha256(64U, '7');

  runtime::PrefillA4PublicationReceipt base_receipt;
  base_receipt.version_major = runtime::kPrefillA4K256PublicationVersionMajor;
  base_receipt.version_minor = runtime::kPrefillA4K256PublicationVersionMinor;
  base_receipt.mode = runtime::PrefillA4ConversionMode::kProductionCalibrated;
  base_receipt.production_residency_eligible = true;
  base_receipt.sidecar_kind = runtime::PrefillSidecarKind::kA4K256;
  base_receipt.packed_k_group_size =
      runtime::kPrefillMLPFactorizedLaneRequiredBasePackedK;
  base_receipt.scale_group_size =
      runtime::kPrefillMLPFactorizedLaneRequiredBaseScaleK;
  base_receipt.physical_layout =
      std::string(runtime::kPrefillMLPFactorizedLaneRequiredBaseK256Layout);
  base_receipt.source_checkpoint_id = base_manifest.source_checkpoint_id;
  base_receipt.source_config_sha256 = base_manifest.source_config_sha256;
  base_receipt.source_index_sha256 = base_manifest.source_index_sha256;
  base_receipt.manifest_sha256 = base_manifest.manifest_sha256;
  base_receipt.policy_sha256 = base_policy.policy_sha256;
  base_receipt.policy_bytes = base_policy.policy_bytes;
  base_receipt.payload_sha256 = base_payload_sha256;
  base_receipt.payload_bytes = base_manifest.summary.arena_bytes;
  base_receipt.projection_count = base_manifest.summary.projection_count;

  const runtime::PrefillMLPFactorizedLaneManifestResult manifest_result =
      runtime::build_prefill_mlp_factorized_lane_r1_manifest(
          base_manifest, base_receipt, std::string(64U, '8'));
  test.expect(static_cast<bool>(manifest_result),
              "strict factorized R1 manifest builds");
  if (!manifest_result) {
    return;
  }
  const runtime::PrefillMLPFactorizedLaneOverlayManifestBinding manifest =
      *manifest_result.value;
  const runtime::PrefillMLPFactorizedLaneR1PolicyResult policy_result =
      runtime::build_prefill_mlp_factorized_lane_r1_policy(manifest, 1.0,
                                                           0.875);
  test.expect(static_cast<bool>(policy_result),
              "strict identity-factor R1 policy builds");
  if (!policy_result) {
    return;
  }
  const runtime::PrefillMLPFactorizedLaneOverlayPolicyBinding policy =
      policy_result.value->binding;

  constexpr std::uintptr_t kBaseArenaAddress = 0x0000100000000000ULL;
  constexpr std::uintptr_t kR1ArenaAddress = 0x0000140000000000ULL;
  const auto pointer = [](const std::uintptr_t address) {
    return reinterpret_cast<const std::uint8_t*>(address);
  };
  test.expect(weights.attach_prefill_a4_sidecars(
                  pointer(kBaseArenaAddress),
                  static_cast<std::size_t>(base_manifest.summary.arena_bytes),
                  &base_manifest, &base_policy, base_payload_sha256),
              "authenticated K256 base attaches before derivative R1");

  const std::size_t r1_bytes = static_cast<std::size_t>(manifest.payload_bytes);
  test.expect(weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
                  pointer(kR1ArenaAddress), r1_bytes, &manifest, &policy) &&
                  factorized_layout_matches(weights, kR1ArenaAddress,
                                            0.875F),
              "complete R1 transaction publishes 192 fixed-layout views");
  const FactorizedSnapshot attached = snapshot(weights);

  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress + 1U), r1_bytes, &manifest, &policy) &&
          matches_snapshot(weights, attached),
      "misaligned arena fails and preserves every resident view");
  const std::uintptr_t overflow_address =
      std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{255U};
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(overflow_address), r1_bytes, &manifest, &policy) &&
          matches_snapshot(weights, attached),
      "overflowing arena range fails transactionally");
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes - 1U, &manifest, &policy) &&
          matches_snapshot(weights, attached),
      "noncanonical arena capacity preserves prior publication");

  runtime::PrefillMLPFactorizedLaneOverlayManifestBinding bad_shape =
      manifest;
  ++bad_shape.projections.back().output_size;
  bad_shape.manifest_sha256 =
      runtime::prefill_mlp_factorized_lane_r1_manifest_sha256(bad_shape);
  runtime::PrefillMLPFactorizedLaneOverlayPolicyBinding bad_shape_policy =
      policy;
  bad_shape_policy.manifest_sha256 = bad_shape.manifest_sha256;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &bad_shape,
          &bad_shape_policy) &&
          matches_snapshot(weights, attached),
      "fixed Down shape mutation fails before publication");

  runtime::PrefillMLPFactorizedLaneOverlayManifestBinding bad_digest =
      manifest;
  bad_digest.manifest_sha256[0] =
      bad_digest.manifest_sha256[0] == 'a' ? 'b' : 'a';
  runtime::PrefillMLPFactorizedLaneOverlayPolicyBinding bad_digest_policy =
      policy;
  bad_digest_policy.manifest_sha256 = bad_digest.manifest_sha256;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &bad_digest,
          &bad_digest_policy) &&
          matches_snapshot(weights, attached),
      "manifest digest mutation fails transactionally");

  runtime::PrefillMLPFactorizedLaneOverlayManifestBinding bad_identity =
      manifest;
  bad_identity.required_base_k256.payload_sha256.assign(64U, '9');
  bad_identity.manifest_sha256 =
      runtime::prefill_mlp_factorized_lane_r1_manifest_sha256(bad_identity);
  runtime::PrefillMLPFactorizedLaneOverlayPolicyBinding bad_identity_policy =
      policy;
  bad_identity_policy.required_base_k256 = bad_identity.required_base_k256;
  bad_identity_policy.manifest_sha256 = bad_identity.manifest_sha256;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &bad_identity,
          &bad_identity_policy) &&
          matches_snapshot(weights, attached),
      "different authenticated K256 payload identity cannot be substituted");

  runtime::PrefillMLPFactorizedLaneOverlayManifestBinding bad_offset =
      manifest;
  bad_offset.projections.back().payload_offset -= 256U;
  bad_offset.manifest_sha256 =
      runtime::prefill_mlp_factorized_lane_r1_manifest_sha256(bad_offset);
  runtime::PrefillMLPFactorizedLaneOverlayPolicyBinding bad_offset_policy =
      policy;
  bad_offset_policy.manifest_sha256 = bad_offset.manifest_sha256;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &bad_offset,
          &bad_offset_policy) &&
          matches_snapshot(weights, attached),
      "fixed projection range mutation preserves the old transaction");

  runtime::PrefillMLPFactorizedLaneOverlayPolicyBinding bad_factor = policy;
  bad_factor.projections[1U].factor_source.sha256.assign(64U, 'a');
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &manifest, &bad_factor) &&
          matches_snapshot(weights, attached),
      "Gate/Up inverse-alpha identity mismatch fails closed");
  runtime::PrefillMLPFactorizedLaneOverlayPolicyBinding bad_clip = policy;
  bad_clip.projections[1U].activation_clip_ratio = 0.75;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &manifest, &bad_clip) &&
          matches_snapshot(weights, attached),
      "Gate/Up shared activation clip mismatch fails closed");

  test.expect(weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
                  nullptr, 0U, nullptr, nullptr) &&
                  all_factorized_empty(weights),
              "canonical null/zero call detaches only factorized R1");

  auto& first_layer = const_cast<runtime::DecoderLayerWeights&>(
      weights.layer(0U));
  auto* const first_gate =
      std::get_if<runtime::NvFp4LinearWeight>(&first_layer.mlp.gate_proj);
  test.expect(first_gate != nullptr, "first Gate has exact NVFP4 binding");
  if (first_gate != nullptr) {
    first_gate->prefill_mlp_k512_weight = pointer(0x0000180000000000ULL);
    test.expect(
        !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
            pointer(kR1ArenaAddress), r1_bytes, &manifest, &policy) &&
            all_factorized_empty(weights),
        "canonical MLP K512 conflict rejects the entire R1 transaction");
    first_gate->prefill_mlp_k512_weight = nullptr;
  }
  first_layer.prefill_mlp_k512_fragment_native
      .gateup_code_capacity_bytes = 1U;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &manifest, &policy) &&
          all_factorized_empty(weights),
      "partial composite K512 conflict also rejects R1");
  first_layer.prefill_mlp_k512_fragment_native = {};

  test.expect(weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
                  pointer(kR1ArenaAddress), r1_bytes, &manifest, &policy),
              "R1 reattaches after explicit conflict removal");
  test.expect(weights.attach_prefill_a4_sidecars(nullptr, 0U, nullptr,
                                                  nullptr) &&
                  all_factorized_empty(weights),
              "detaching the K256 base also clears every R1 view");
  test.expect(!weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
                  pointer(kR1ArenaAddress), r1_bytes, &manifest, &policy) &&
                  all_factorized_empty(weights),
              "R1 cannot attach without its exact authenticated K256 base");
}

}  // namespace

int main() {
  TestContext test;
  test_factorized_r1_attachment(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " factorized R1 ModelWeights assertion(s) failed\n";
    return 1;
  }
  std::cout << "factorized R1 ModelWeights tests passed\n";
  return 0;
}
