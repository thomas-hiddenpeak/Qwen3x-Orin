#include "q3x/runtime/model_weights.h"

#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"
#include "q3x/model/model_config.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_r4_publication.h"

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
    const st::DType dtype,
    const std::vector<std::uint64_t>& shape) noexcept {
  std::uint64_t elements = 1U;
  for (const std::uint64_t dimension : shape) {
    elements *= dimension;
  }
  return elements * st::bit_width(dtype) / 8U;
}

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

  [[nodiscard]] static bool ends_with(
      const std::string_view value,
      const std::string_view suffix) noexcept {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
  }

  [[nodiscard]] static int read_scalar(const void* const context,
                                       const float* const device_value,
                                       float* const host_value) noexcept {
    if (context == nullptr || device_value == nullptr ||
        host_value == nullptr) {
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

struct ManifestSource final {
  model_weights::WeightManifest manifest;
  std::vector<runtime::ShardIdentity> shards;
  bool complete = false;
};

[[nodiscard]] bool is_projection_component(
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

[[nodiscard]] ManifestSource make_manifest_source(
    const SyntheticArena& arena) {
  ManifestSource source;
  for (const auto& descriptor : checkpoint::known_checkpoint_catalog()) {
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
    if (!is_projection_component(name)) {
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
    const auto& shard = source.shards[active_shard];
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

[[nodiscard]] std::vector<
    runtime::PrefillMLPFactorizedLaneR4CalibrationSpec>
make_calibration(
    const runtime::PrefillMLPFactorizedLaneR4Manifest& manifest) {
  std::vector<runtime::PrefillMLPFactorizedLaneR4CalibrationSpec>
      result(manifest.projections.size());
  for (std::size_t index = 0U; index < result.size(); ++index) {
    const std::size_t layer = index / 3U;
    const bool down = index % 3U == 2U;
    result[index].weight_clip_ratio = down ? 0.75 : 0.875;
    result[index].activation_clip_ratio = down ? 0.8125 : 0.9375;
    result[index].alpha_path =
        "alpha/layer_" + std::to_string(layer) +
        (down ? "_down.f32le" : "_gate_up.f32le");
    result[index].alpha_sha256 =
        std::string(64U, static_cast<char>('a' + layer % 6U));
    result[index].alpha_element_count =
        manifest.projections[index].input_size;
  }
  return result;
}

[[nodiscard]] bool same_view(
    const runtime::PrefillMLPFactorizedLaneR4LinearSidecarView& left,
    const runtime::PrefillMLPFactorizedLaneR4LinearSidecarView&
        right) noexcept {
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
    std::array<runtime::PrefillMLPFactorizedLaneR4LayerView,
               runtime::kQwen36DenseLayerCount>;

[[nodiscard]] Snapshot snapshot(const runtime::ModelWeights& weights) {
  Snapshot result{};
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[index] = weights.layer(index).prefill_mlp_factorized_lane_r4;
  }
  return result;
}

[[nodiscard]] bool matches_snapshot(const runtime::ModelWeights& weights,
                                    const Snapshot& expected) {
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    const auto& actual =
        weights.layer(index).prefill_mlp_factorized_lane_r4;
    if (!same_view(actual.gate, expected[index].gate) ||
        !same_view(actual.up, expected[index].up) ||
        !same_view(actual.down, expected[index].down)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool all_r4_empty(
    const runtime::ModelWeights& weights) noexcept {
  return std::all_of(
      weights.layers().begin(), weights.layers().end(),
      [](const runtime::DecoderLayerWeights& layer) {
        return layer.prefill_mlp_factorized_lane_r4.empty();
      });
}

[[nodiscard]] bool r4_layout_matches(const runtime::ModelWeights& weights,
                                     const std::uintptr_t arena_address) {
  const auto plan = runtime::prefill_mlp_factorized_lane_overlay_layout_plan(
      runtime::kPrefillMLPFactorizedLaneR4PublicationLaneCount);
  if (!plan) {
    return false;
  }
  for (std::size_t layer_index = 0U; layer_index < weights.layers().size();
       ++layer_index) {
    const auto& layer =
        weights.layer(layer_index).prefill_mlp_factorized_lane_r4;
    if (!layer.attached()) {
      return false;
    }
    const std::array<runtime::PrefillMLPFactorizedLaneProjectionFamily, 3U>
        families = {
            runtime::PrefillMLPFactorizedLaneProjectionFamily::kGate,
            runtime::PrefillMLPFactorizedLaneProjectionFamily::kUp,
            runtime::PrefillMLPFactorizedLaneProjectionFamily::kDown};
    const std::array<
        const runtime::PrefillMLPFactorizedLaneR4LinearSidecarView*, 3U>
        views = {&layer.gate, &layer.up, &layer.down};
    for (std::size_t family_index = 0U; family_index < families.size();
         ++family_index) {
      const auto family = families[family_index];
      const auto& projection_plan =
          family == runtime::PrefillMLPFactorizedLaneProjectionFamily::kDown
              ? plan.down
              : family ==
                        runtime::PrefillMLPFactorizedLaneProjectionFamily::kUp
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
          view.lane_count != 4U) {
        return false;
      }
    }
  }
  return true;
}

void test_attachment(Test& test) {
  runtime::PrefillMLPFactorizedLaneR4LinearSidecarView wrong_r4;
  wrong_r4.packed_weight = reinterpret_cast<const std::uint8_t*>(0x1000U);
  wrong_r4.packed_weight_capacity_bytes = 17'408U * 5'120U / 2U;
  wrong_r4.lane_scales =
      reinterpret_cast<const std::uint16_t*>(0x2000U);
  wrong_r4.lane_scale_capacity_elements = 17'408U;
  wrong_r4.inverse_alpha = reinterpret_cast<const float*>(0x3000U);
  wrong_r4.inverse_alpha_capacity_elements = 5'120U;
  wrong_r4.output_size = 17'408U;
  wrong_r4.input_size = 5'120U;
  wrong_r4.lane_count = 1U;
  wrong_r4.activation_clip_ratio = 1.0F;
  test.expect(!wrong_r4.attached(),
              "R4 view never accepts an R1 lane/capacity contract");

  SyntheticArena source_arena;
  runtime::WeightBindResult bound =
      runtime::bind_qwen36_27b_weights(source_arena.source());
  test.expect(bound.ok(), "exact synthetic Qwen3.6 binding succeeds");
  if (!bound) {
    return;
  }
  runtime::ModelWeights& weights = *bound.value;
  const ManifestSource source = make_manifest_source(source_arena);
  test.expect(source.complete, "synthetic source covers 400 projections");
  if (!source.complete) {
    return;
  }
  runtime::PrefillSidecarManifestOptions options;
  options.kind = runtime::PrefillSidecarKind::kExact;
  const auto exact = runtime::build_qwen36_27b_prefill_sidecar_manifest(
      source.manifest, source.shards, options);
  test.expect(exact.ok(), "pinned Exact source manifest builds");
  if (!exact) {
    return;
  }
  const auto manifest_result =
      runtime::build_prefill_mlp_factorized_lane_r4_direct_manifest(
          *exact.value);
  test.expect(static_cast<bool>(manifest_result),
              "strict direct R4 manifest builds");
  if (!manifest_result) {
    return;
  }
  const auto& manifest = *manifest_result.value;
  const auto calibration = make_calibration(manifest);
  const auto policy_result =
      runtime::build_prefill_mlp_factorized_lane_r4_policy(manifest,
                                                           calibration);
  test.expect(static_cast<bool>(policy_result),
              "strict calibrated R4 policy builds");
  if (!policy_result) {
    return;
  }
  const auto& policy = *policy_result.value;

  constexpr std::uintptr_t kArenaAddress = 0x0000140000000000ULL;
  constexpr std::uintptr_t kReplacementAddress = 0x0000180000000000ULL;
  const auto pointer = [](const std::uintptr_t address) {
    return reinterpret_cast<const std::uint8_t*>(address);
  };
  const std::size_t payload_bytes =
      static_cast<std::size_t>(manifest.payload_bytes);
  test.expect(weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
                  pointer(kArenaAddress), payload_bytes, &manifest, &policy) &&
                  r4_layout_matches(weights, kArenaAddress),
              "R4 transaction publishes 192 direct-checkpoint views");
  const Snapshot attached = snapshot(weights);

  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(kArenaAddress + 1U), payload_bytes, &manifest, &policy) &&
          matches_snapshot(weights, attached),
      "misaligned arena fails and preserves old R4 transaction");
  const std::uintptr_t overflow_address =
      std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{255U};
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(overflow_address), payload_bytes, &manifest, &policy) &&
          matches_snapshot(weights, attached),
      "overflowing range fails and preserves old R4 transaction");
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(kArenaAddress), payload_bytes - 1U, &manifest, &policy) &&
          matches_snapshot(weights, attached),
      "payload capacity mismatch preserves old R4 transaction");

  auto bad_manifest = manifest;
  ++bad_manifest.projections.back().output_size;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(kArenaAddress), payload_bytes, &bad_manifest, &policy) &&
          matches_snapshot(weights, attached),
      "shape mutation fails transactionally");
  bad_manifest = manifest;
  bad_manifest.projections.back().payload_offset -= 256U;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(kArenaAddress), payload_bytes, &bad_manifest, &policy) &&
          matches_snapshot(weights, attached),
      "offset mutation fails transactionally");
  bad_manifest = manifest;
  bad_manifest.projections.pop_back();
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(kArenaAddress), payload_bytes, &bad_manifest, &policy) &&
          matches_snapshot(weights, attached),
      "incomplete projection inventory fails transactionally");
  bad_manifest = manifest;
  bad_manifest.manifest_sha256[0] =
      bad_manifest.manifest_sha256[0] == 'a' ? 'b' : 'a';
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(kArenaAddress), payload_bytes, &bad_manifest, &policy) &&
          matches_snapshot(weights, attached),
      "manifest digest mutation fails transactionally");

  auto bad_policy = policy;
  bad_policy.projections[1U].factor_sha256.assign(64U, 'f');
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(kArenaAddress), payload_bytes, &manifest, &bad_policy) &&
          matches_snapshot(weights, attached),
      "Gate/Up alpha identity mismatch preserves old R4 transaction");
  bad_policy = policy;
  bad_policy.projections[1U].activation_clip_ratio = 0.75;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(kArenaAddress), payload_bytes, &manifest, &bad_policy) &&
          matches_snapshot(weights, attached),
      "Gate/Up clip mismatch preserves old R4 transaction");
  bad_policy = policy;
  bad_policy.lane_count = 1U;
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
          pointer(kArenaAddress), payload_bytes, &manifest, &bad_policy) &&
          matches_snapshot(weights, attached),
      "policy lane mutation preserves old R4 transaction");

  test.expect(weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
                  pointer(kReplacementAddress), payload_bytes, &manifest,
                  &policy) &&
                  r4_layout_matches(weights, kReplacementAddress),
              "validated replacement atomically updates all R4 views");
  const Snapshot replacement = snapshot(weights);

  test.expect(
      !weights.attach_prefill_mlp_k512_sidecars(
          pointer(kArenaAddress), 1U, nullptr, nullptr) &&
          matches_snapshot(weights, replacement),
      "K512 v1 cannot attach over R4");
  test.expect(
      !weights.attach_prefill_mlp_k512_fragment_native_sidecars(
          pointer(kArenaAddress), 1U, nullptr, nullptr, nullptr) &&
          matches_snapshot(weights, replacement),
      "K512 fragment-native cannot attach over R4");
  test.expect(
      !weights.attach_prefill_mlp_k512_paired_gateup_canonical_down_sidecars(
          pointer(kArenaAddress), 1U, nullptr, nullptr, nullptr) &&
          matches_snapshot(weights, replacement),
      "K512 paired hybrid cannot attach over R4");
  test.expect(
      !weights
           .attach_prefill_mlp_k512_projection_major_gateup_canonical_down_sidecars(
               pointer(kArenaAddress), 1U, nullptr, nullptr, nullptr) &&
          matches_snapshot(weights, replacement),
      "K512 projection-major hybrid cannot attach over R4");
  test.expect(
      !weights.attach_prefill_mlp_factorized_lane_r1_sidecars(
          pointer(kArenaAddress), 1U, nullptr, nullptr) &&
          matches_snapshot(weights, replacement),
      "R1 cannot attach over R4");

  test.expect(weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
                  nullptr, 0U, nullptr, nullptr) &&
                  all_r4_empty(weights),
              "canonical null call detaches only R4");
  auto& first_layer = const_cast<runtime::DecoderLayerWeights&>(
      weights.layer(0U));
  first_layer.prefill_mlp_factorized_lane_r1.gate.lane_count = 1U;
  test.expect(!weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
                  pointer(kArenaAddress), payload_bytes, &manifest, &policy) &&
                  all_r4_empty(weights),
              "partial R1 state blocks R4 transaction");
  first_layer.prefill_mlp_factorized_lane_r1 = {};

  auto* const first_gate =
      std::get_if<runtime::NvFp4LinearWeight>(&first_layer.mlp.gate_proj);
  test.expect(first_gate != nullptr, "first Gate is NVFP4");
  if (first_gate != nullptr) {
    first_gate->prefill_mlp_k512_weight = pointer(kReplacementAddress);
    test.expect(!weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
                    pointer(kArenaAddress), payload_bytes, &manifest,
                    &policy) &&
                    all_r4_empty(weights),
                "partial canonical K512 state blocks R4 transaction");
    first_gate->prefill_mlp_k512_weight = nullptr;
  }
  first_layer.prefill_mlp_k512_fragment_native
      .gateup_code_capacity_bytes = 1U;
  test.expect(!weights.attach_prefill_mlp_factorized_lane_r4_sidecars(
                  pointer(kArenaAddress), payload_bytes, &manifest, &policy) &&
                  all_r4_empty(weights),
              "partial composite K512 state blocks R4 transaction");
  first_layer.prefill_mlp_k512_fragment_native = {};
}

}  // namespace

int main() {
  Test test;
  test_attachment(test);
  return test.result();
}
