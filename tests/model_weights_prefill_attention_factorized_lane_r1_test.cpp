#include "q3x/runtime/model_weights.h"

#include "q3x/core/sha256.h"
#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"
#include "q3x/model/model_config.h"
#include "q3x/runtime/prefill_a4_sidecar_converter.h"
#include "q3x/runtime/prefill_attention_factorized_lane_overlay.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
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

// Host-only exact-shape binder fixture.  Fake device addresses avoid a
// multi-gigabyte allocation while still exercising the production binding,
// K256 attachment, and derivative publication contracts.
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
  static constexpr std::uint64_t kArenaBytes =
      64ULL * 1024ULL * 1024ULL * 1024ULL;

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
      const auto inserted =
          self->tensors_.emplace(std::string(name), std::move(view));
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
      ".mlp.gate_proj.",          ".mlp.up_proj.",
      ".mlp.down_proj.",          ".linear_attn.in_proj_qkv.",
      ".linear_attn.in_proj_z.",  ".linear_attn.out_proj.",
      ".self_attn.q_proj.",       ".self_attn.k_proj.",
      ".self_attn.v_proj.",       ".self_attn.o_proj."};
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

[[nodiscard]] bool is_attention_family(
    const runtime::PrefillProjectionFamily family) noexcept {
  switch (family) {
    case runtime::PrefillProjectionFamily::kLinearQkv:
    case runtime::PrefillProjectionFamily::kLinearZ:
    case runtime::PrefillProjectionFamily::kLinearO:
    case runtime::PrefillProjectionFamily::kFullQ:
    case runtime::PrefillProjectionFamily::kFullK:
    case runtime::PrefillProjectionFamily::kFullV:
    case runtime::PrefillProjectionFamily::kFullO:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] runtime::PrefillAttentionFactorizedLaneProjectionFamily
overlay_family(const runtime::PrefillProjectionFamily family) noexcept {
  using Base = runtime::PrefillProjectionFamily;
  using Overlay = runtime::PrefillAttentionFactorizedLaneProjectionFamily;
  switch (family) {
    case Base::kLinearQkv: return Overlay::kLinearQkv;
    case Base::kLinearZ: return Overlay::kLinearZ;
    case Base::kLinearO: return Overlay::kLinearO;
    case Base::kFullQ: return Overlay::kFullQ;
    case Base::kFullK: return Overlay::kFullK;
    case Base::kFullV: return Overlay::kFullV;
    case Base::kFullO: return Overlay::kFullO;
    default: return static_cast<Overlay>(0xffU);
  }
}

[[nodiscard]] const runtime::PrefillA4FactorizedLaneProjectionLayoutPlan*
projection_plan(
    const runtime::PrefillAttentionFactorizedLaneOverlayLayoutPlan& plan,
    const runtime::PrefillAttentionFactorizedLaneProjectionFamily family) {
  using Family = runtime::PrefillAttentionFactorizedLaneProjectionFamily;
  switch (family) {
    case Family::kLinearQkv: return &plan.linear_qkv;
    case Family::kLinearZ: return &plan.linear_z;
    case Family::kLinearO: return &plan.linear_o;
    case Family::kFullQ: return &plan.full_q;
    case Family::kFullK: return &plan.full_k;
    case Family::kFullV: return &plan.full_v;
    case Family::kFullO: return &plan.full_o;
  }
  return nullptr;
}

[[nodiscard]] std::string_view family_name(
    const runtime::PrefillAttentionFactorizedLaneProjectionFamily family) {
  using Family = runtime::PrefillAttentionFactorizedLaneProjectionFamily;
  switch (family) {
    case Family::kLinearQkv: return "linear_qkv";
    case Family::kLinearZ: return "linear_z";
    case Family::kLinearO: return "linear_o";
    case Family::kFullQ: return "full_q";
    case Family::kFullK: return "full_k";
    case Family::kFullV: return "full_v";
    case Family::kFullO: return "full_o";
  }
  return "invalid";
}

[[nodiscard]] std::string manifest_sha256(
    const runtime::PrefillAttentionFactorizedLaneOverlayManifestBinding&
        manifest) {
  std::ostringstream output;
  output << "schema=q3x.prefill.attention-factorized-r1.overlay-manifest"
         << "\nversion=" << manifest.version_major << '.'
         << manifest.version_minor << "\nlayout=" << manifest.physical_layout
         << "\ncheckpoint=" << manifest.source_checkpoint_id << "\nconfig="
         << manifest.source_config_sha256 << "\nindex="
         << manifest.source_index_sha256 << "\nlane=" << manifest.lane_count
         << "\nbase=" << manifest.required_base_k256.physical_layout << ':'
         << manifest.required_base_k256.packed_k_group_size << ':'
         << manifest.required_base_k256.scale_group_size << ':'
         << manifest.required_base_k256.manifest_sha256 << ':'
         << manifest.required_base_k256.policy_sha256 << ':'
         << manifest.required_base_k256.payload_sha256 << ':'
         << manifest.required_base_k256.receipt_sha256 << "\npayload="
         << manifest.payload_bytes << '\n';
  for (const auto& entry : manifest.projections) {
    output << entry.ordinal << ':' << entry.layer_index << ':'
           << family_name(entry.family) << ':' << entry.source_module << ':'
           << entry.source_sha256 << ':' << entry.output_size << ':'
           << entry.input_size << ':' << entry.payload_offset << ':'
           << entry.payload_bytes << '\n';
  }
  const std::string bytes = output.str();
  q3x::core::Sha256 hasher;
  (void)hasher.update(bytes.data(), bytes.size());
  return hasher.finalize().hex();
}

struct AttentionPublication final {
  runtime::PrefillAttentionFactorizedLaneOverlayManifestBinding manifest;
  runtime::PrefillAttentionFactorizedLaneOverlayPolicyBinding policy;
};

[[nodiscard]] AttentionPublication make_attention_publication(
    const runtime::PrefillSidecarManifest& base_manifest,
    const runtime::PrefillA4CalibrationPolicy& base_policy,
    const std::string& base_payload_sha256) {
  AttentionPublication result;
  const auto plan =
      runtime::prefill_attention_factorized_lane_overlay_layout_plan(1U);
  auto& manifest = result.manifest;
  manifest.physical_layout =
      std::string(runtime::kPrefillAttentionFactorizedLaneOverlayLayout);
  manifest.source_checkpoint_id = base_manifest.source_checkpoint_id;
  manifest.source_config_sha256 = base_manifest.source_config_sha256;
  manifest.source_index_sha256 = base_manifest.source_index_sha256;
  manifest.required_base_k256.physical_layout = std::string(
      runtime::kPrefillAttentionFactorizedLaneRequiredBaseK256Layout);
  manifest.required_base_k256.packed_k_group_size =
      runtime::kPrefillAttentionFactorizedLaneRequiredBasePackedK;
  manifest.required_base_k256.scale_group_size =
      runtime::kPrefillAttentionFactorizedLaneRequiredBaseScaleK;
  manifest.required_base_k256.manifest_sha256 = base_manifest.manifest_sha256;
  manifest.required_base_k256.policy_sha256 = base_policy.policy_sha256;
  manifest.required_base_k256.payload_sha256 = base_payload_sha256;
  manifest.required_base_k256.receipt_sha256.assign(64U, '8');
  manifest.lane_count = 1U;
  manifest.payload_bytes = plan.payload_bytes;
  for (const runtime::PrefillProjectionSidecarEntry& source :
       base_manifest.projections) {
    if (!is_attention_family(source.family)) {
      continue;
    }
    const auto family = overlay_family(source.family);
    const auto* const layout = projection_plan(plan, family);
    runtime::PrefillAttentionFactorizedLaneManifestProjection entry;
    entry.ordinal = static_cast<std::uint32_t>(manifest.projections.size());
    entry.layer_index = source.layer_index;
    entry.family = family;
    entry.source_module = source.source_module;
    entry.source_sha256 = source.source_sha256;
    entry.output_size = source.output_size;
    entry.input_size = source.input_size;
    entry.payload_offset =
        runtime::prefill_attention_factorized_lane_projection_absolute_offset(
            plan, source.layer_index, family);
    entry.payload_bytes = layout->projection_bytes;
    manifest.projections.emplace_back(std::move(entry));
  }
  manifest.manifest_sha256 = manifest_sha256(manifest);

  auto& policy = result.policy;
  policy.physical_layout = manifest.physical_layout;
  policy.source_checkpoint_id = manifest.source_checkpoint_id;
  policy.source_config_sha256 = manifest.source_config_sha256;
  policy.source_index_sha256 = manifest.source_index_sha256;
  policy.manifest_sha256 = manifest.manifest_sha256;
  policy.required_base_k256 = manifest.required_base_k256;
  policy.lane_count = 1U;
  policy.policy_sha256.assign(64U, 'd');
  policy.policy_bytes = 1U;
  for (const auto& entry : manifest.projections) {
    runtime::PrefillAttentionFactorizedLaneProjectionCalibrationBinding
        calibration;
    calibration.ordinal = entry.ordinal;
    calibration.source_module = entry.source_module;
    calibration.source_sha256 = entry.source_sha256;
    calibration.weight_clip_ratio = 1.0;
    calibration.activation_clip_ratio = 0.875;
    calibration.factor_source.scheme = "identity_alpha_f32_v1";
    calibration.factor_source.sha256 =
        entry.input_size == 6'144U
            ? "08f46753296b40512f918614f5b0be6f15e4a95fb0aeeff6c9026be0a396c4a7"
            : "42010c1c68b632e2ab15c82bca6edef2cac2026c889dd0202d609602b756f568";
    calibration.factor_source.element_count = entry.input_size;
    policy.projections.emplace_back(std::move(calibration));
  }
  return result;
}

[[nodiscard]] bool same_view(
    const runtime::PrefillAttentionFactorizedLaneR1LinearSidecarView& left,
    const runtime::PrefillAttentionFactorizedLaneR1LinearSidecarView& right) {
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

using Snapshot =
    std::array<runtime::PrefillAttentionFactorizedLaneR1LayerView,
               runtime::kQwen36DenseLayerCount>;

[[nodiscard]] Snapshot snapshot(const runtime::ModelWeights& weights) {
  Snapshot result{};
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[index] =
        weights.layer(index).prefill_attention_factorized_lane_r1;
  }
  return result;
}

[[nodiscard]] bool matches_snapshot(const runtime::ModelWeights& weights,
                                    const Snapshot& expected) {
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    const auto& actual =
        weights.layer(index).prefill_attention_factorized_lane_r1;
    const std::array<const runtime::PrefillAttentionFactorizedLaneR1LinearSidecarView*,
                     7U>
        actual_views = {&actual.linear_qkv, &actual.linear_z, &actual.linear_o,
                        &actual.full_q, &actual.full_k, &actual.full_v,
                        &actual.full_o};
    const auto& wanted = expected[index];
    const std::array<const runtime::PrefillAttentionFactorizedLaneR1LinearSidecarView*,
                     7U>
        wanted_views = {&wanted.linear_qkv, &wanted.linear_z, &wanted.linear_o,
                        &wanted.full_q, &wanted.full_k, &wanted.full_v,
                        &wanted.full_o};
    for (std::size_t view = 0U; view < actual_views.size(); ++view) {
      if (!same_view(*actual_views[view], *wanted_views[view])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool all_empty(const runtime::ModelWeights& weights) {
  return std::all_of(
      weights.layers().begin(), weights.layers().end(),
      [](const runtime::DecoderLayerWeights& layer) {
        return layer.prefill_attention_factorized_lane_r1.empty();
      });
}

[[nodiscard]] bool layout_matches(const runtime::ModelWeights& weights,
                                  const std::uintptr_t arena_address,
                                  const float activation_clip) {
  const auto plan =
      runtime::prefill_attention_factorized_lane_overlay_layout_plan(1U);
  for (std::uint32_t layer_index = 0U;
       layer_index < runtime::kQwen36DenseLayerCount; ++layer_index) {
    const auto& layer =
        weights.layer(layer_index).prefill_attention_factorized_lane_r1;
    if (!layer.attached()) {
      return false;
    }
    const bool full =
        runtime::prefill_attention_factorized_lane_is_full_layer(layer_index);
    const std::array<runtime::PrefillAttentionFactorizedLaneProjectionFamily,
                     4U>
        families = full
                       ? std::array{
                             runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullQ,
                             runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullK,
                             runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullV,
                             runtime::PrefillAttentionFactorizedLaneProjectionFamily::kFullO}
                       : std::array{
                             runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv,
                             runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ,
                             runtime::PrefillAttentionFactorizedLaneProjectionFamily::kLinearO,
                             static_cast<runtime::PrefillAttentionFactorizedLaneProjectionFamily>(0xffU)};
    const std::array<const runtime::PrefillAttentionFactorizedLaneR1LinearSidecarView*,
                     4U>
        views = full ? std::array{&layer.full_q, &layer.full_k, &layer.full_v,
                                  &layer.full_o}
                     : std::array{&layer.linear_qkv, &layer.linear_z,
                                  &layer.linear_o,
                                  static_cast<const runtime::PrefillAttentionFactorizedLaneR1LinearSidecarView*>(nullptr)};
    const std::size_t count = full ? 4U : 3U;
    for (std::size_t position = 0U; position < count; ++position) {
      const auto* const layout = projection_plan(plan, families[position]);
      const std::uint64_t projection_offset =
          runtime::prefill_attention_factorized_lane_projection_absolute_offset(
              plan, layer_index, families[position]);
      const auto& view = *views[position];
      if (reinterpret_cast<std::uintptr_t>(view.packed_weight) !=
              arena_address + projection_offset +
                  layout->packed_weight_offset ||
          reinterpret_cast<std::uintptr_t>(view.lane_scales) !=
              arena_address + projection_offset +
                  layout->weight_scale_offset ||
          reinterpret_cast<std::uintptr_t>(view.inverse_alpha) !=
              arena_address + projection_offset +
                  layout->inverse_alpha_offset ||
          view.packed_weight_capacity_bytes != layout->packed_weight_bytes ||
          view.lane_scale_capacity_elements !=
              layout->weight_scale_elements ||
          view.inverse_alpha_capacity_elements !=
              layout->inverse_alpha_elements ||
          view.activation_clip_ratio != activation_clip) {
        return false;
      }
    }
  }
  return true;
}

void test_attention_r1_attachment(TestContext& test) {
  SyntheticArena source_arena;
  runtime::WeightBindResult bound =
      runtime::bind_qwen36_27b_weights(source_arena.source());
  test.expect(bound.ok(), "exact synthetic Qwen3.6 binding succeeds");
  if (!bound) {
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
  const auto& base_manifest = *base_result.value;
  const runtime::PrefillA4CalibrationPolicy base_policy =
      make_a4_policy(base_manifest);
  const std::string base_payload_sha256(64U, '7');
  AttentionPublication publication = make_attention_publication(
      base_manifest, base_policy, base_payload_sha256);
  test.expect(publication.manifest.projections.size() == 208U,
              "Attention publication contains exactly 208 projections");

  constexpr std::uintptr_t kBaseArenaAddress = 0x0000100000000000ULL;
  constexpr std::uintptr_t kR1ArenaAddress = 0x0000140000000000ULL;
  const auto pointer = [](const std::uintptr_t address) {
    return reinterpret_cast<const std::uint8_t*>(address);
  };
  test.expect(weights.attach_prefill_a4_sidecars(
                  pointer(kBaseArenaAddress),
                  static_cast<std::size_t>(base_manifest.summary.arena_bytes),
                  &base_manifest, &base_policy, base_payload_sha256),
              "authenticated K256 base attaches first");
  const std::size_t r1_bytes =
      static_cast<std::size_t>(publication.manifest.payload_bytes);
  test.expect(
      weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &publication.manifest,
          &publication.policy) &&
          layout_matches(weights, kR1ArenaAddress, 0.875F),
      "complete transaction publishes the exact 208-view Attention R1 ABI");
  const Snapshot attached = snapshot(weights);

  test.expect(
      !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress + 1U), r1_bytes, &publication.manifest,
          &publication.policy) &&
          matches_snapshot(weights, attached),
      "misaligned arena fails without changing resident views");
  test.expect(
      !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes - 1U, &publication.manifest,
          &publication.policy) &&
          matches_snapshot(weights, attached),
      "noncanonical capacity fails transactionally");
  const std::uintptr_t overflow_address =
      std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{255U};
  test.expect(
      !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(overflow_address), r1_bytes, &publication.manifest,
          &publication.policy) &&
          matches_snapshot(weights, attached),
      "overflowing arena range fails transactionally");

  AttentionPublication bad_shape = publication;
  ++bad_shape.manifest.projections.back().input_size;
  bad_shape.manifest.manifest_sha256 = manifest_sha256(bad_shape.manifest);
  bad_shape.policy.manifest_sha256 = bad_shape.manifest.manifest_sha256;
  test.expect(
      !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &bad_shape.manifest,
          &bad_shape.policy) &&
          matches_snapshot(weights, attached),
      "fixed Attention-O shape mutation is rejected");

  AttentionPublication bad_offset = publication;
  bad_offset.manifest.projections[9U].payload_offset += 256U;
  bad_offset.manifest.manifest_sha256 = manifest_sha256(bad_offset.manifest);
  bad_offset.policy.manifest_sha256 = bad_offset.manifest.manifest_sha256;
  test.expect(
      !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &bad_offset.manifest,
          &bad_offset.policy) &&
          matches_snapshot(weights, attached),
      "fixed full-Q offset mutation is rejected");

  AttentionPublication bad_base = publication;
  bad_base.manifest.required_base_k256.payload_sha256.assign(64U, '9');
  bad_base.manifest.manifest_sha256 = manifest_sha256(bad_base.manifest);
  bad_base.policy.required_base_k256 = bad_base.manifest.required_base_k256;
  bad_base.policy.manifest_sha256 = bad_base.manifest.manifest_sha256;
  test.expect(
      !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &bad_base.manifest,
          &bad_base.policy) &&
          matches_snapshot(weights, attached),
      "different authenticated K256 payload identity cannot substitute");

  AttentionPublication bad_linear_group = publication;
  bad_linear_group.policy.projections[1U].activation_clip_ratio = 0.75;
  test.expect(
      !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &bad_linear_group.manifest,
          &bad_linear_group.policy) &&
          matches_snapshot(weights, attached),
      "linear QKV/Z activation clips must be shared");
  AttentionPublication bad_full_group = publication;
  bad_full_group.policy.projections[11U].factor_source.sha256.assign(64U, 'a');
  test.expect(
      !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &bad_full_group.manifest,
          &bad_full_group.policy) &&
          matches_snapshot(weights, attached),
      "full Q/K/V factors must be shared and exact");

  AttentionPublication independent_o = publication;
  independent_o.policy.projections[2U].activation_clip_ratio = 0.75;
  test.expect(
      weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &independent_o.manifest,
          &independent_o.policy) &&
          weights.layer(0U)
                  .prefill_attention_factorized_lane_r1.linear_o
                  .activation_clip_ratio == 0.75F,
      "linear O owns an independent K6144 activation clip contract");

  test.expect(weights.attach_prefill_attention_factorized_lane_r1_sidecars(
                  nullptr, 0U, nullptr, nullptr) &&
                  all_empty(weights),
              "canonical null/zero call detaches only Attention R1");
  test.expect(runtime::prefill_a4_sidecar_view(
                  std::get<runtime::LinearAttentionWeights>(
                      weights.layer(0U).attention)
                      .in_proj_qkv)
                  .attached(),
              "R1 detach preserves the authenticated K256 base");

  auto& layer0 = const_cast<runtime::DecoderLayerWeights&>(weights.layer(0U));
  auto& linear = std::get<runtime::LinearAttentionWeights>(layer0.attention);
  auto* const output = std::get_if<runtime::Fp8LinearWeight>(&linear.out_proj);
  test.expect(output != nullptr, "linear Attention-O retains exact FP8 weight");
  if (output != nullptr) {
    output->prefill_attention_o_k512_weight =
        pointer(0x0000180000000000ULL);
    test.expect(
        !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
            pointer(kR1ArenaAddress), r1_bytes, &publication.manifest,
            &publication.policy) &&
            all_empty(weights),
        "partial legacy Attention-O K512 state conflicts with R1");
    output->prefill_attention_o_k512_weight = nullptr;
  }

  test.expect(weights.attach_prefill_attention_factorized_lane_r1_sidecars(
                  pointer(kR1ArenaAddress), r1_bytes, &publication.manifest,
                  &publication.policy),
              "R1 reattaches after explicit conflict removal");
  test.expect(weights.attach_prefill_a4_sidecars(nullptr, 0U, nullptr,
                                                  nullptr) &&
                  all_empty(weights),
              "detaching K256 clears all Attention R1 views");
  test.expect(
      !weights.attach_prefill_attention_factorized_lane_r1_sidecars(
          pointer(kR1ArenaAddress), r1_bytes, &publication.manifest,
          &publication.policy) &&
          all_empty(weights),
      "Attention R1 cannot attach without its exact authenticated base");
  test.expect(
      runtime::prefill_attention_factorized_lane_qualification_role(1U) ==
          runtime::PrefillAttentionFactorizedLaneQualificationRole::
              kPerformanceUpperBound,
      "lane1 remains an upper-bound role rather than production quality");
}

}  // namespace

int main() {
  TestContext test;
  test_attention_r1_attachment(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " Attention R1 ModelWeights assertion(s) failed\n";
    return 1;
  }
  std::cout << "Attention R1 ModelWeights tests passed\n";
  return 0;
}
