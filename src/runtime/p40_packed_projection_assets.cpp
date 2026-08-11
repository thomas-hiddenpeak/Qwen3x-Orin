#include "q3x/runtime/p40_packed_projection_assets.h"

#include "q3x/core/sha256.h"

#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace q3x::runtime {
namespace {

#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)

namespace st = io::safetensors;

constexpr std::string_view kModelRepository =
    "nvidia/Qwen3.6-27B-NVFP4";
constexpr std::string_view kModelRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";
constexpr std::string_view kModelHashDomain =
    "q3x.sm87.p40.packed-projection.model.v1";
constexpr std::string_view kCheckpointHashDomain =
    "q3x.sm87.p40.packed-projection.checkpoint.v1";
constexpr std::string_view kTensorIdentityHashDomain =
    "q3x.sm87.p40.packed-projection.tensor-identity.v1";
constexpr std::string_view kWeightHashDomain =
    "q3x.sm87.p40.packed-projection.weight-source.v1";
constexpr std::string_view kScaleHashDomain =
    "q3x.sm87.p40.packed-projection.scale-source.v1";
constexpr std::string_view kPayloadHashDomain =
    "q3x.sm87.p40.packed-projection.payload-transform.v1";
constexpr std::string_view kArtifactIdentityHashDomain =
    "q3x.sm87.p40.packed-projection.artifact-identity.v1";

struct PlannedSource {
  kernels::Sm87P40PackedCanonicalSource canonical{};
  std::string module_name;
  float scalar_scale = 0.0F;
  bool nvfp4 = false;
};

struct PlannedArtifact {
  std::size_t layer_index = 0U;
  kernels::Sm87P40PackedProjectionRole role =
      kernels::Sm87P40PackedProjectionRole::kInvalid;
  std::size_t source_count = 0U;
  std::array<PlannedSource,
             kernels::kSm87P40PackedProjectionMaximumSources>
      sources{};
};

enum class PreparationScope : std::uint8_t {
  kCompleteProjection = 0U,
  kNvfp4Only,
};

[[nodiscard]] constexpr std::size_t expected_artifact_count(
    const PreparationScope scope) noexcept {
  return scope == PreparationScope::kNvfp4Only
             ? kP40PackedNvfp4ArtifactCount
             : kernels::kSm87P40PackedProjectionArtifactCount;
}

[[nodiscard]] constexpr std::size_t expected_source_count(
    const PreparationScope scope) noexcept {
  return scope == PreparationScope::kNvfp4Only
             ? kP40PackedNvfp4SourceCount
             : kernels::kSm87P40PackedProjectionSourceIdentityCount;
}

[[nodiscard]] constexpr std::uint64_t expected_arena_bytes(
    const PreparationScope scope) noexcept {
  return scope == PreparationScope::kNvfp4Only
             ? kP40PackedNvfp4ArenaBytes
             : kP40PackedProjectionArenaBytes;
}

using ArtifactPlans =
    std::array<PlannedArtifact,
               kernels::kSm87P40PackedProjectionArtifactCount>;

template <typename Unsigned>
[[nodiscard]] bool hash_unsigned(core::Sha256& hasher,
                                 Unsigned value) noexcept {
  static_assert(std::is_unsigned_v<Unsigned>);
  std::array<std::uint8_t, sizeof(Unsigned)> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
  }
  return hasher.update(bytes.data(), bytes.size());
}

[[nodiscard]] bool hash_string(core::Sha256& hasher,
                               const std::string_view value) noexcept {
  return hash_unsigned(hasher, static_cast<std::uint64_t>(value.size())) &&
         hasher.update(value.data(), value.size());
}

[[nodiscard]] bool hash_digest(
    core::Sha256& hasher,
    const kernels::Sm87P40PackedDigest& digest) noexcept {
  return hasher.update(digest.bytes.data(), digest.bytes.size());
}

[[nodiscard]] kernels::Sm87P40PackedDigest finish_digest(
    core::Sha256& hasher) noexcept {
  kernels::Sm87P40PackedDigest result;
  result.bytes = hasher.finalize().bytes;
  return result;
}

[[nodiscard]] std::uint64_t digest_identity(
    const kernels::Sm87P40PackedDigest& digest) noexcept {
  std::uint64_t value = 0U;
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    value |= static_cast<std::uint64_t>(digest.bytes[index]) <<
             (index * 8U);
  }
  return value;
}

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] bool empty_packed_view(
    const kernels::Sm87P40PackedProjectionDeviceView& view) noexcept {
  return view.payload == nullptr && view.payload_bytes == 0U &&
         view.artifact_identity == 0U &&
         view.role == kernels::Sm87P40PackedProjectionRole::kInvalid &&
         view.tactic == kernels::Sm87P40PackedTactic::kInvalid &&
         view.source_count == 0U &&
         std::all_of(view.scalar_scales.begin(), view.scalar_scales.end(),
                     [](const float value) { return value == 0.0F; });
}

[[nodiscard]] bool resident_range_contains(
    const ResidentWeights& resident, const DeviceTensorView& view) noexcept {
  if (!resident || resident.arena_data() == nullptr ||
      resident.size_bytes() == 0U || view.device_data == nullptr ||
      view.byte_size == 0U || view.arena_offset > resident.size_bytes() ||
      view.byte_size > resident.size_bytes() - view.arena_offset) {
    return false;
  }
  const auto base = reinterpret_cast<std::uintptr_t>(resident.arena_data());
  if (resident.size_bytes() >
          std::numeric_limits<std::uintptr_t>::max() - base ||
      view.arena_offset >
          std::numeric_limits<std::uintptr_t>::max() - base) {
    return false;
  }
  return reinterpret_cast<std::uintptr_t>(view.device_data) ==
         base + static_cast<std::uintptr_t>(view.arena_offset);
}

[[nodiscard]] bool resident_tensor_matches(
    const ResidentWeights& resident, const std::string& name,
    const void* const expected_pointer, const st::DType expected_dtype,
    const std::initializer_list<std::uint64_t> expected_shape,
    const std::uint64_t expected_bytes) noexcept {
  const DeviceTensorView* const view = resident.find(name);
  if (view == nullptr || view->device_data != expected_pointer ||
      view->dtype != expected_dtype || view->byte_size != expected_bytes ||
      view->shape.size() != expected_shape.size() ||
      !resident_range_contains(resident, *view)) {
    return false;
  }
  return std::equal(view->shape.begin(), view->shape.end(),
                    expected_shape.begin(), expected_shape.end());
}

[[nodiscard]] bool verify_observed_checkpoint(
    const ResidentWeights& resident,
    kernels::Sm87P40PackedDigest& checkpoint_digest,
    std::string& message) {
  if (!resident || resident.arena_data() == nullptr ||
      resident.size_bytes() != kPinnedQwen36_27BArenaBytes) {
    message = "resident arena is not the exact pinned Qwen3.6-27B arena";
    return false;
  }
  const auto& pinned = pinned_qwen36_27b_shards();
  const auto& observed = resident.stats().shards;
  if (pinned.empty() || observed.size() != pinned.size()) {
    message = "resident loader did not report the complete pinned shard set";
    return false;
  }

  core::Sha256 hasher;
  bool hash_ok = hash_string(hasher, kCheckpointHashDomain) &&
                 hash_string(hasher, kModelRepository) &&
                 hash_string(hasher, kModelRevision) &&
                 hash_unsigned(hasher,
                               static_cast<std::uint64_t>(pinned.size()));
  std::array<bool, 16U> observed_used{};
  if (observed.size() > observed_used.size()) {
    message = "pinned shard set exceeds the fixed provenance bound";
    return false;
  }
  for (const ShardIdentity& expected : pinned) {
    std::size_t match = observed.size();
    for (std::size_t index = 0U; index < observed.size(); ++index) {
      if (!observed_used[index] &&
          observed[index].filename == expected.filename) {
        match = index;
        break;
      }
    }
    if (match == observed.size() ||
        observed[match].sha256 != expected.sha256 ||
        observed[match].bytes_read != expected.file_size) {
      message = "resident loader shard identity differs from the pinned "
                "checkpoint at " +
                expected.filename;
      return false;
    }
    observed_used[match] = true;
    hash_ok = hash_ok && hash_string(hasher, expected.filename) &&
              hash_unsigned(hasher, expected.file_size) &&
              hash_string(hasher, observed[match].sha256);
  }
  checkpoint_digest = finish_digest(hasher);
  if (!hash_ok ||
      kernels::sm87_p40_packed_digest_is_zero(checkpoint_digest)) {
    checkpoint_digest = {};
    message = "failed to derive a nonzero observed checkpoint digest";
    return false;
  }
  return true;
}

[[nodiscard]] bool derive_model_digest(
    kernels::Sm87P40PackedDigest& model_digest) noexcept {
  core::Sha256 hasher;
  const bool ok =
      hash_string(hasher, kModelHashDomain) &&
      hash_string(hasher, kModelRepository) &&
      hash_string(hasher, kModelRevision) &&
      hash_unsigned(hasher,
                    kernels::kSm87P40PackedProjectionLayerCount) &&
      hash_unsigned(hasher, kernels::kSm87P40PackedProjectionHidden) &&
      hash_unsigned(hasher,
                    kernels::kSm87P40PackedProjectionIntermediate) &&
      hash_unsigned(hasher,
                    kernels::kSm87P40PackedProjectionLinearLayerCount) &&
      hash_unsigned(hasher,
                    kernels::kSm87P40PackedProjectionFullLayerCount);
  model_digest = finish_digest(hasher);
  return ok && !kernels::sm87_p40_packed_digest_is_zero(model_digest);
}

[[nodiscard]] bool verify_nvfp4_source(
    const ResidentWeights& resident, const NvFp4LinearWeight& projection,
    const std::string& module_name,
    const kernels::Sm87P40PackedLogicalRole logical_role,
    const std::size_t expected_output, const std::size_t expected_input,
    PlannedSource& source) {
  if (projection.packed_weight == nullptr ||
      projection.block_scale == nullptr ||
      projection.weight_scale_2_device == nullptr ||
      projection.output_size != expected_output ||
      projection.input_size != expected_input ||
      !std::isfinite(projection.weight_scale_2) ||
      projection.weight_scale_2 <= 0.0F ||
      reinterpret_cast<std::uintptr_t>(projection.packed_weight) % 16U !=
          0U ||
      reinterpret_cast<std::uintptr_t>(projection.block_scale) % 16U !=
          0U ||
      reinterpret_cast<std::uintptr_t>(
          projection.weight_scale_2_device) % alignof(float) != 0U ||
      projection.prefill_marlin_weight != nullptr ||
      projection.prefill_marlin_scales != nullptr ||
      projection.prefill_marlin_global_scale != nullptr ||
      !empty_packed_view(projection.prefill_p40_packed_artifact)) {
    return false;
  }
  const std::uint64_t output = expected_output;
  const std::uint64_t input = expected_input;
  const std::string weight_name = module_name + ".weight";
  const std::string block_scale_name = module_name + ".weight_scale";
  const std::string global_scale_name = module_name + ".weight_scale_2";
  if (!resident_tensor_matches(resident, weight_name,
                               projection.packed_weight, st::DType::kU8,
                               {output, input / 2U}, output * input / 2U) ||
      !resident_tensor_matches(
          resident, block_scale_name, projection.block_scale,
          st::DType::kF8E4M3, {output, input / 16U}, output * input / 16U) ||
      !resident_tensor_matches(
          resident, global_scale_name, projection.weight_scale_2_device,
          st::DType::kF32, {}, sizeof(float))) {
    return false;
  }
  source.canonical = {logical_role,
                      projection.packed_weight,
                      projection.block_scale,
                      projection.weight_scale_2_device,
                      expected_output,
                      expected_input};
  source.module_name = module_name;
  source.scalar_scale = projection.weight_scale_2;
  source.nvfp4 = true;
  return true;
}

[[nodiscard]] bool verify_fp8_source(
    const ResidentWeights& resident, const Fp8LinearWeight& projection,
    const std::string& module_name,
    const kernels::Sm87P40PackedLogicalRole logical_role,
    const std::size_t expected_output, const std::size_t expected_input,
    PlannedSource& source) {
  if (projection.weight == nullptr ||
      projection.weight_scale_device == nullptr ||
      projection.output_size != expected_output ||
      projection.input_size != expected_input ||
      !std::isfinite(projection.weight_scale) ||
      projection.weight_scale <= 0.0F ||
      reinterpret_cast<std::uintptr_t>(projection.weight) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(
          projection.weight_scale_device) % alignof(float) != 0U ||
      projection.prefill_qkv_register_feed_sidecar != nullptr ||
      projection.prefill_supermatrix_sidecar != nullptr ||
      projection.prefill_marlin_weight != nullptr ||
      projection.prefill_marlin_scales != nullptr ||
      !empty_packed_view(projection.prefill_p40_packed_artifact)) {
    return false;
  }
  const std::uint64_t output = expected_output;
  const std::uint64_t input = expected_input;
  const std::string weight_name = module_name + ".weight";
  const std::string scale_name = module_name + ".weight_scale";
  if (!resident_tensor_matches(resident, weight_name, projection.weight,
                               st::DType::kF8E4M3, {output, input},
                               output * input) ||
      !resident_tensor_matches(resident, scale_name,
                               projection.weight_scale_device,
                               st::DType::kF32, {}, sizeof(float))) {
    return false;
  }
  source.canonical = {logical_role,
                      projection.weight,
                      nullptr,
                      projection.weight_scale_device,
                      expected_output,
                      expected_input};
  source.module_name = module_name;
  source.scalar_scale = projection.weight_scale;
  source.nvfp4 = false;
  return true;
}

[[nodiscard]] bool append_artifact(
    ArtifactPlans& plans, std::size_t& plan_count,
    PlannedArtifact artifact) noexcept {
  if (plan_count >= plans.size()) {
    return false;
  }
  plans[plan_count++] = std::move(artifact);
  return true;
}

[[nodiscard]] bool inventory_model(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    ArtifactPlans& plans, std::size_t& plan_count,
    const PreparationScope scope, std::string& message) {
  using kernels::Sm87P40PackedLogicalRole;
  using kernels::Sm87P40PackedProjectionRole;
  plan_count = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kernels::kSm87P40PackedProjectionLayerCount;
       ++layer_index) {
    const DecoderLayerWeights& layer = model_weights.layer(layer_index);
    const std::string prefix = "model.language_model.layers." +
                               std::to_string(layer_index) + ".";
    const auto* const gate =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.gate_proj);
    const auto* const up =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.up_proj);
    const auto* const down =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.down_proj);
    PlannedArtifact gate_up;
    gate_up.layer_index = layer_index;
    gate_up.role = Sm87P40PackedProjectionRole::kNvFp4GateUp;
    gate_up.source_count = 2U;
    PlannedArtifact down_artifact;
    down_artifact.layer_index = layer_index;
    down_artifact.role = Sm87P40PackedProjectionRole::kNvFp4Down;
    down_artifact.source_count = 1U;
    if (gate == nullptr || up == nullptr || down == nullptr ||
        !verify_nvfp4_source(
            resident, *gate, prefix + "mlp.gate_proj",
            Sm87P40PackedLogicalRole::kNvFp4Gate, 17'408U, 5'120U,
            gate_up.sources[0U]) ||
        !verify_nvfp4_source(
            resident, *up, prefix + "mlp.up_proj",
            Sm87P40PackedLogicalRole::kNvFp4Up, 17'408U, 5'120U,
            gate_up.sources[1U]) ||
        !verify_nvfp4_source(
            resident, *down, prefix + "mlp.down_proj",
            Sm87P40PackedLogicalRole::kNvFp4Down, 5'120U, 17'408U,
            down_artifact.sources[0U])) {
      message = "ineligible or non-resident NVFP4 projection inventory at "
                "layer " +
                std::to_string(layer_index);
      return false;
    }

    if (scope == PreparationScope::kNvfp4Only) {
      if (!append_artifact(plans, plan_count, std::move(gate_up)) ||
          !append_artifact(plans, plan_count, std::move(down_artifact))) {
        message = "P40 NVFP4 physical artifact inventory overflowed its "
                  "fixed bound";
        return false;
      }
      continue;
    }

    PlannedArtifact fp8_input;
    fp8_input.layer_index = layer_index;
    PlannedArtifact fp8_output;
    fp8_output.layer_index = layer_index;
    fp8_output.role = Sm87P40PackedProjectionRole::kFp8AttentionOutput;
    fp8_output.source_count = 1U;
    const bool expected_full =
        kernels::sm87_p40_packed_is_full_layer(layer_index);
    if (!expected_full) {
      const auto* const linear =
          std::get_if<LinearAttentionWeights>(&layer.attention);
      if (linear == nullptr) {
        message = "attention layer schedule differs from the fixed P40 "
                  "linear/full pattern at layer " +
                  std::to_string(layer_index);
        return false;
      }
      const auto* const qkv =
          std::get_if<Fp8LinearWeight>(&linear->in_proj_qkv);
      const auto* const z =
          std::get_if<Fp8LinearWeight>(&linear->in_proj_z);
      const auto* const output =
          std::get_if<Fp8LinearWeight>(&linear->out_proj);
      fp8_input.role = Sm87P40PackedProjectionRole::kFp8LinearQkvZ;
      fp8_input.source_count = 2U;
      if (qkv == nullptr || z == nullptr || output == nullptr ||
          !verify_fp8_source(
              resident, *qkv, prefix + "linear_attn.in_proj_qkv",
              Sm87P40PackedLogicalRole::kFp8LinearQkv, 10'240U, 5'120U,
              fp8_input.sources[0U]) ||
          !verify_fp8_source(
              resident, *z, prefix + "linear_attn.in_proj_z",
              Sm87P40PackedLogicalRole::kFp8LinearZ, 6'144U, 5'120U,
              fp8_input.sources[1U]) ||
          !verify_fp8_source(
              resident, *output, prefix + "linear_attn.out_proj",
              Sm87P40PackedLogicalRole::kFp8AttentionOutput, 5'120U,
              6'144U, fp8_output.sources[0U])) {
        message = "ineligible or non-resident linear-attention FP8 "
                  "inventory at layer " +
                  std::to_string(layer_index);
        return false;
      }
    } else {
      const auto* const full =
          std::get_if<FullAttentionWeights>(&layer.attention);
      if (full == nullptr) {
        message = "attention layer schedule differs from the fixed P40 "
                  "linear/full pattern at layer " +
                  std::to_string(layer_index);
        return false;
      }
      const auto* const query =
          std::get_if<Fp8LinearWeight>(&full->q_proj);
      const auto* const key = std::get_if<Fp8LinearWeight>(&full->k_proj);
      const auto* const value =
          std::get_if<Fp8LinearWeight>(&full->v_proj);
      const auto* const output =
          std::get_if<Fp8LinearWeight>(&full->o_proj);
      fp8_input.role = Sm87P40PackedProjectionRole::kFp8FullQkv;
      fp8_input.source_count = 3U;
      if (query == nullptr || key == nullptr || value == nullptr ||
          output == nullptr ||
          !verify_fp8_source(
              resident, *query, prefix + "self_attn.q_proj",
              Sm87P40PackedLogicalRole::kFp8FullQ, 12'288U, 5'120U,
              fp8_input.sources[0U]) ||
          !verify_fp8_source(
              resident, *key, prefix + "self_attn.k_proj",
              Sm87P40PackedLogicalRole::kFp8FullK, 1'024U, 5'120U,
              fp8_input.sources[1U]) ||
          !verify_fp8_source(
              resident, *value, prefix + "self_attn.v_proj",
              Sm87P40PackedLogicalRole::kFp8FullV, 1'024U, 5'120U,
              fp8_input.sources[2U]) ||
          !verify_fp8_source(
              resident, *output, prefix + "self_attn.o_proj",
              Sm87P40PackedLogicalRole::kFp8AttentionOutput, 5'120U,
              6'144U, fp8_output.sources[0U])) {
        message = "ineligible or non-resident full-attention FP8 inventory "
                  "at layer " +
                  std::to_string(layer_index);
        return false;
      }
    }
    if (!append_artifact(plans, plan_count, std::move(gate_up)) ||
        !append_artifact(plans, plan_count, std::move(down_artifact)) ||
        !append_artifact(plans, plan_count, std::move(fp8_input)) ||
        !append_artifact(plans, plan_count, std::move(fp8_output))) {
      message = "P40 physical artifact inventory overflowed its fixed bound";
      return false;
    }
  }
  if (plan_count != expected_artifact_count(scope)) {
    message = scope == PreparationScope::kNvfp4Only
                  ? "P40 NVFP4 inventory did not cover all 128 physical "
                    "artifacts"
                  : "P40 inventory did not cover all 256 physical artifacts";
    return false;
  }
  return true;
}

[[nodiscard]] bool derive_source_identity(
    const PlannedSource& planned,
    const kernels::Sm87P40PackedDigest& model_digest,
    const kernels::Sm87P40PackedDigest& checkpoint_digest,
    kernels::Sm87P40PackedSourceIdentity& source) noexcept {
  const auto& canonical = planned.canonical;
  const std::string weight_name = planned.module_name + ".weight";
  const std::string block_scale_name =
      planned.module_name + ".weight_scale";
  const std::string global_scale_name =
      planned.module_name +
      (planned.nvfp4 ? ".weight_scale_2" : ".weight_scale");
  const std::uint8_t role = static_cast<std::uint8_t>(canonical.role);
  const std::uint32_t scale_bits = float_bits(planned.scalar_scale);

  core::Sha256 identity_hasher;
  bool ok = hash_string(identity_hasher, kTensorIdentityHashDomain) &&
            hash_digest(identity_hasher, model_digest) &&
            hash_digest(identity_hasher, checkpoint_digest) &&
            hash_string(identity_hasher, planned.module_name) &&
            hash_unsigned(identity_hasher, role);
  const auto identity_digest = finish_digest(identity_hasher);

  core::Sha256 weight_hasher;
  ok = ok && hash_string(weight_hasher, kWeightHashDomain) &&
       hash_digest(weight_hasher, model_digest) &&
       hash_digest(weight_hasher, checkpoint_digest) &&
       hash_string(weight_hasher, weight_name) &&
       hash_unsigned(weight_hasher, role) &&
       hash_unsigned(weight_hasher,
                     static_cast<std::uint64_t>(canonical.output_features)) &&
       hash_unsigned(weight_hasher,
                     static_cast<std::uint64_t>(canonical.input_features));
  const auto weight_digest = finish_digest(weight_hasher);

  core::Sha256 scale_hasher;
  ok = ok && hash_string(scale_hasher, kScaleHashDomain) &&
       hash_digest(scale_hasher, model_digest) &&
       hash_digest(scale_hasher, checkpoint_digest) &&
       hash_string(scale_hasher, block_scale_name) &&
       hash_string(scale_hasher, global_scale_name) &&
       hash_unsigned(scale_hasher, scale_bits) &&
       hash_unsigned(scale_hasher, role);
  const auto scale_digest = finish_digest(scale_hasher);

  source.role = canonical.role;
  source.tensor_identity = digest_identity(identity_digest);
  source.weight_digest = weight_digest;
  source.scale_digest = scale_digest;
  source.global_scale_bits = scale_bits;
  return ok && source.tensor_identity != 0U &&
         !kernels::sm87_p40_packed_digest_is_zero(weight_digest) &&
         !kernels::sm87_p40_packed_digest_is_zero(scale_digest);
}

[[nodiscard]] bool derive_payload_digest(
    const kernels::Sm87P40PackedArtifactManifest& manifest,
    const kernels::Sm87P40PackedProjectionPlan& plan,
    kernels::Sm87P40PackedDigest& payload_digest) noexcept {
  core::Sha256 hasher;
  bool ok = hash_string(hasher, kPayloadHashDomain) &&
            hash_unsigned(hasher,
                          kernels::kSm87P40PackedProjectionAbiMajor) &&
            hash_unsigned(hasher,
                          kernels::kSm87P40PackedProjectionAbiMinor) &&
            hash_digest(hasher, manifest.model_digest) &&
            hash_digest(hasher, manifest.checkpoint_digest) &&
            hash_unsigned(hasher, manifest.layer_index) &&
            hash_unsigned(hasher,
                          static_cast<std::uint8_t>(manifest.role)) &&
            hash_unsigned(hasher,
                          static_cast<std::uint8_t>(manifest.tactic)) &&
            hash_unsigned(hasher, plan.policy) &&
            hash_unsigned(hasher, plan.token_count) &&
            hash_unsigned(hasher, plan.tile_m) &&
            hash_unsigned(hasher, plan.grid_m) &&
            hash_unsigned(hasher, plan.grid_n) &&
            hash_unsigned(hasher, plan.group_m) &&
            hash_unsigned(hasher, plan.persistent_ctas) &&
            hash_unsigned(hasher, plan.logical_tasks) &&
            hash_unsigned(hasher, plan.payload_bytes);
  for (std::size_t index = 0U; index < plan.source_count; ++index) {
    const auto& partition = plan.partitions[index];
    const auto& source = manifest.sources[index];
    ok = ok && hash_unsigned(hasher,
                             static_cast<std::uint8_t>(partition.role)) &&
         hash_unsigned(hasher,
                       static_cast<std::uint8_t>(partition.tactic)) &&
         hash_unsigned(hasher, partition.output_features) &&
         hash_unsigned(hasher, partition.input_features) &&
         hash_unsigned(hasher, partition.first_task_n_tile) &&
         hash_unsigned(hasher, partition.task_n_tiles) &&
         hash_unsigned(hasher, partition.tile_n) &&
         hash_unsigned(hasher, partition.tile_k) &&
         hash_unsigned(hasher, partition.pipeline_stages) &&
         hash_unsigned(hasher, partition.payload_offset) &&
         hash_unsigned(hasher, partition.payload_bytes) &&
         hash_unsigned(hasher, source.tensor_identity) &&
         hash_digest(hasher, source.weight_digest) &&
         hash_digest(hasher, source.scale_digest) &&
         hash_unsigned(hasher, source.global_scale_bits);
  }
  payload_digest = finish_digest(hasher);
  return ok &&
         !kernels::sm87_p40_packed_digest_is_zero(payload_digest);
}

[[nodiscard]] bool derive_artifact_identity(
    const kernels::Sm87P40PackedArtifactManifest& manifest,
    std::uint64_t& identity) noexcept {
  core::Sha256 hasher;
  const bool ok = hash_string(hasher, kArtifactIdentityHashDomain) &&
                  hash_digest(hasher, manifest.model_digest) &&
                  hash_digest(hasher, manifest.checkpoint_digest) &&
                  hash_digest(hasher, manifest.payload_digest) &&
                  hash_unsigned(hasher, manifest.layer_index) &&
                  hash_unsigned(hasher,
                                static_cast<std::uint8_t>(manifest.role));
  identity = digest_identity(finish_digest(hasher));
  return ok && identity != 0U;
}

[[nodiscard]] bool valid_nvfp4_only_manifest_inventory(
    const std::array<kernels::Sm87P40PackedArtifactManifest,
                     kernels::kSm87P40PackedProjectionArtifactCount>&
        manifests,
    const std::size_t manifest_count) noexcept {
  if (manifest_count != kP40PackedNvfp4ArtifactCount) {
    return false;
  }
  std::array<std::array<bool, 2U>,
             kernels::kSm87P40PackedProjectionLayerCount>
      seen{};
  std::array<std::uint64_t, kP40PackedNvfp4ArtifactCount>
      artifact_identities{};
  std::array<std::uint64_t, kP40PackedNvfp4SourceCount>
      source_identities{};
  std::size_t source_count = 0U;
  const auto role_slot = [](const kernels::Sm87P40PackedProjectionRole role)
      noexcept -> std::size_t {
    if (role ==
        kernels::Sm87P40PackedProjectionRole::kNvFp4GateUp) {
      return 0U;
    }
    if (role == kernels::Sm87P40PackedProjectionRole::kNvFp4Down) {
      return 1U;
    }
    return 2U;
  };

  for (std::size_t index = 0U; index < manifest_count; ++index) {
    const kernels::Sm87P40PackedArtifactManifest& manifest = manifests[index];
    const std::size_t slot = role_slot(manifest.role);
    const auto plan = kernels::sm87_p40_packed_projection_plan(manifest.role);
    if (slot >= 2U ||
        manifest.layer_index >=
            kernels::kSm87P40PackedProjectionLayerCount ||
        seen[manifest.layer_index][slot] || !plan.valid() ||
        manifest.source_count != plan.source_count ||
        !kernels::validate_sm87_p40_packed_artifact_manifest(manifest)
             .valid() ||
        manifest.model_digest != manifests[0U].model_digest ||
        manifest.checkpoint_digest != manifests[0U].checkpoint_digest ||
        std::find(artifact_identities.begin(),
                  artifact_identities.begin() + index,
                  manifest.artifact_identity) !=
            artifact_identities.begin() + index) {
      return false;
    }
    artifact_identities[index] = manifest.artifact_identity;
    seen[manifest.layer_index][slot] = true;
    for (std::size_t source = 0U; source < manifest.source_count; ++source) {
      const std::uint64_t identity =
          manifest.sources[source].tensor_identity;
      if (source_count >= source_identities.size() || identity == 0U ||
          std::find(source_identities.begin(),
                    source_identities.begin() + source_count,
                    identity) !=
              source_identities.begin() + source_count) {
        return false;
      }
      source_identities[source_count++] = identity;
    }
  }
  for (const auto& layer : seen) {
    if (!layer[0U] || !layer[1U]) {
      return false;
    }
  }
  return source_count == kP40PackedNvfp4SourceCount;
}

[[nodiscard]] bool build_manifests(
    const ArtifactPlans& plans, const std::size_t plan_count,
    const kernels::Sm87P40PackedDigest& model_digest,
    const kernels::Sm87P40PackedDigest& checkpoint_digest,
    const PreparationScope scope,
    std::array<kernels::Sm87P40PackedArtifactManifest,
               kernels::kSm87P40PackedProjectionArtifactCount>& manifests,
    std::string& message) {
  std::array<std::uint64_t,
             kernels::kSm87P40PackedProjectionSourceIdentityCount>
      source_identities{};
  std::size_t source_identity_count = 0U;
  std::array<std::uint64_t,
             kernels::kSm87P40PackedProjectionArtifactCount>
      artifact_identities{};
  for (std::size_t index = 0U; index < plan_count; ++index) {
    const PlannedArtifact& planned = plans[index];
    const auto plan =
        kernels::sm87_p40_packed_projection_plan(planned.role);
    if (!plan.valid() || planned.source_count != plan.source_count) {
      message = "P40 artifact plan disagrees with the fixed kernel ABI";
      return false;
    }
    auto manifest = kernels::make_sm87_p40_packed_artifact_manifest(
        planned.role, planned.layer_index);
    manifest.model_digest = model_digest;
    manifest.checkpoint_digest = checkpoint_digest;
    for (std::size_t source_index = 0U;
         source_index < planned.source_count; ++source_index) {
      if (!derive_source_identity(planned.sources[source_index], model_digest,
                                  checkpoint_digest,
                                  manifest.sources[source_index])) {
        message = "failed to derive an authenticated P40 source identity";
        return false;
      }
      const std::uint64_t identity =
          manifest.sources[source_index].tensor_identity;
      if (source_identity_count >= source_identities.size() ||
          std::find(source_identities.begin(),
                    source_identities.begin() + source_identity_count,
                    identity) !=
              source_identities.begin() + source_identity_count) {
        message = "P40 source identities are not globally unique";
        return false;
      }
      source_identities[source_identity_count++] = identity;
    }
    if (!derive_payload_digest(manifest, plan, manifest.payload_digest) ||
        !derive_artifact_identity(manifest, manifest.artifact_identity)) {
      message = "failed to derive a nonzero P40 payload/artifact identity";
      return false;
    }
    if (std::find(artifact_identities.begin(),
                  artifact_identities.begin() + index,
                  manifest.artifact_identity) !=
        artifact_identities.begin() + index) {
      message = "P40 physical artifact identities are not globally unique";
      return false;
    }
    artifact_identities[index] = manifest.artifact_identity;
    if (!kernels::seal_sm87_p40_packed_artifact_manifest(&manifest) ||
        !kernels::validate_sm87_p40_packed_artifact_manifest(manifest)
             .valid()) {
      message = "P40 artifact manifest failed seal/validation at layer " +
                std::to_string(planned.layer_index);
      return false;
    }
    manifests[index] = manifest;
  }
  if (source_identity_count != expected_source_count(scope)) {
    message = scope == PreparationScope::kNvfp4Only
                  ? "P40 NVFP4 manifests did not derive all 192 source "
                    "identities"
                  : "P40 manifests did not derive all 400 source identities";
    return false;
  }
  const bool inventory_valid =
      scope == PreparationScope::kNvfp4Only
          ? valid_nvfp4_only_manifest_inventory(manifests, plan_count)
          : kernels::validate_sm87_p40_packed_artifact_inventory(
                manifests.data(), plan_count)
                .valid();
  if (!inventory_valid) {
    message = scope == PreparationScope::kNvfp4Only
                  ? "sealed P40 NVFP4 manifest inventory failed exact "
                    "validation"
                  : "sealed P40 manifest inventory failed complete "
                    "validation";
    return false;
  }
  return true;
}

[[nodiscard]] std::string_view role_name(
    const kernels::Sm87P40PackedProjectionRole role) noexcept {
  using kernels::Sm87P40PackedProjectionRole;
  switch (role) {
    case Sm87P40PackedProjectionRole::kNvFp4GateUp:
      return "nvfp4-gate-up";
    case Sm87P40PackedProjectionRole::kNvFp4Down:
      return "nvfp4-down";
    case Sm87P40PackedProjectionRole::kFp8LinearQkvZ:
      return "fp8-linear-qkv-z";
    case Sm87P40PackedProjectionRole::kFp8FullQkv:
      return "fp8-full-qkv";
    case Sm87P40PackedProjectionRole::kFp8AttentionOutput:
      return "fp8-attention-output";
    case Sm87P40PackedProjectionRole::kCount:
    case Sm87P40PackedProjectionRole::kInvalid:
      return "invalid";
  }
  return "invalid";
}

[[nodiscard]] P40PackedProjectionPreparationStats prepare_impl(
    const ResidentWeights& resident, ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    const PreparationScope scope, P40PackedProjectionAssets& owner) {
  P40PackedProjectionPreparationStats result;
  if (!owner.empty()) {
    result.hard_failure = true;
    result.message = "P40 packed projection owner was not empty before "
                     "preparation";
    return result;
  }

  // Capability is deliberately established before model inventory or any
  // allocation. Each transaction checks every physical role it can execute;
  // neither scope has request-time fallback authority.
  constexpr std::array<kernels::Sm87P40PackedProjectionRole, 5U> kRoles{
      kernels::Sm87P40PackedProjectionRole::kNvFp4GateUp,
      kernels::Sm87P40PackedProjectionRole::kNvFp4Down,
      kernels::Sm87P40PackedProjectionRole::kFp8LinearQkvZ,
      kernels::Sm87P40PackedProjectionRole::kFp8FullQkv,
      kernels::Sm87P40PackedProjectionRole::kFp8AttentionOutput};
  (void)cudaGetLastError();
  for (const auto role : kRoles) {
    if (scope == PreparationScope::kNvfp4Only &&
        role != kernels::Sm87P40PackedProjectionRole::kNvFp4GateUp &&
        role != kernels::Sm87P40PackedProjectionRole::kNvFp4Down) {
      continue;
    }
    kernels::Sm87P40PackedProjectionResources resources{};
    const int status =
        kernels::query_sm87_p40_packed_projection_resources_cuda(
            role, &resources);
    if (status != static_cast<int>(cudaSuccess) ||
        resources.active_blocks_per_sm < 2 || resources.local_bytes != 0U) {
      result.hard_failure = true;
      result.cuda_error = status;
      result.message = "P40 packed kernel resource admission failed for " +
                       std::string(role_name(role));
      return result;
    }
  }

  kernels::Sm87P40PackedDigest checkpoint_digest{};
  if (!verify_observed_checkpoint(resident, checkpoint_digest,
                                  result.message)) {
    result.hard_failure = true;
    return result;
  }
  kernels::Sm87P40PackedDigest model_digest{};
  if (!derive_model_digest(model_digest)) {
    result.hard_failure = true;
    result.message = "failed to derive the fixed model identity";
    return result;
  }

  ArtifactPlans plans{};
  std::size_t plan_count = 0U;
  if (!inventory_model(resident, model_weights, plans, plan_count, scope,
                       result.message)) {
    result.hard_failure = true;
    return result;
  }
  std::array<kernels::Sm87P40PackedArtifactManifest,
             kernels::kSm87P40PackedProjectionArtifactCount>
      manifests{};
  if (!build_manifests(plans, plan_count, model_digest, checkpoint_digest,
                       scope, manifests, result.message)) {
    result.hard_failure = true;
    return result;
  }

  static_assert(sizeof(std::size_t) >= sizeof(std::uint64_t));
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaMemGetInfo failed before P40 packed arena "
                     "allocation";
    return result;
  }
  const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
  const std::uint64_t arena_bytes = expected_arena_bytes(scope);
  if (arena_bytes > free_u64 ||
      minimum_free_bytes_after_prepare >
          free_u64 - arena_bytes) {
    result.hard_failure = true;
    result.message = scope == PreparationScope::kNvfp4Only
                         ? "insufficient device-memory margin for the exact "
                           "P40 packed NVFP4 arena"
                         : "insufficient device-memory margin for the "
                           "complete P40 packed projection arena";
    return result;
  }

  void* allocation = nullptr;
  status = cudaMalloc(&allocation, static_cast<std::size_t>(arena_bytes));
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = scope == PreparationScope::kNvfp4Only
                         ? "cudaMalloc failed for the exact P40 packed "
                           "NVFP4 arena"
                         : "cudaMalloc failed for the complete P40 packed "
                           "projection arena";
    return result;
  }
  owner.arena = static_cast<std::uint8_t*>(allocation);
  owner.bytes = arena_bytes;
  if (reinterpret_cast<std::uintptr_t>(owner.arena) %
          kernels::kSm87P40PackedProjectionPayloadAlignment !=
      0U) {
    result.hard_failure = true;
    result.message = "cudaMalloc returned a misaligned P40 packed arena";
    owner.release();
    return result;
  }

  std::size_t remaining_free = 0U;
  status = cudaMemGetInfo(&remaining_free, &total_bytes);
  if (status != cudaSuccess ||
      static_cast<std::uint64_t>(remaining_free) <
          minimum_free_bytes_after_prepare) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "P40 packed arena allocation violated the "
                           "post-allocation device-memory margin"
                         : "cudaMemGetInfo failed after P40 packed arena "
                           "allocation";
    owner.release();
    return result;
  }

  cudaStream_t stream = nullptr;
  status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "failed to create the nonblocking P40 pack stream";
    owner.release();
    return result;
  }

  std::uint64_t offset = 0U;
  for (std::size_t index = 0U; index < plan_count; ++index) {
    const PlannedArtifact& planned = plans[index];
    const auto plan =
        kernels::sm87_p40_packed_projection_plan(planned.role);
    if (offset % kernels::kSm87P40PackedProjectionPayloadAlignment != 0U ||
        plan.payload_bytes > owner.bytes - offset) {
      result.hard_failure = true;
      result.message = "P40 packed artifact offsets did not fit the exact "
                       "aligned arena";
      (void)cudaStreamSynchronize(stream);
      (void)cudaStreamDestroy(stream);
      owner.release();
      return result;
    }
    std::array<kernels::Sm87P40PackedCanonicalSource,
               kernels::kSm87P40PackedProjectionMaximumSources>
        canonical_sources{};
    for (std::size_t source_index = 0U;
         source_index < planned.source_count; ++source_index) {
      canonical_sources[source_index] =
          planned.sources[source_index].canonical;
    }
    std::uint8_t* const destination =
        owner.arena + static_cast<std::size_t>(offset);
    const int prepare_status =
        kernels::prepare_sm87_p40_packed_projection_cuda(
            planned.role, canonical_sources.data(), planned.source_count,
            destination, static_cast<std::size_t>(plan.payload_bytes),
            static_cast<void*>(stream));
    if (prepare_status != static_cast<int>(cudaSuccess)) {
      result.hard_failure = true;
      result.cuda_error = prepare_status;
      result.message = "P40 packed projection preparation failed at layer " +
                       std::to_string(planned.layer_index) + " for " +
                       std::string(role_name(planned.role));
      (void)cudaStreamSynchronize(stream);
      (void)cudaStreamDestroy(stream);
      owner.release();
      return result;
    }

    P40PackedProjectionSidecarDescriptor descriptor;
    descriptor.layer_index = planned.layer_index;
    descriptor.view.payload = destination;
    descriptor.view.payload_bytes =
        static_cast<std::size_t>(plan.payload_bytes);
    descriptor.view.artifact_identity =
        manifests[index].artifact_identity;
    descriptor.view.role = planned.role;
    descriptor.view.tactic = plan.tactic;
    descriptor.view.source_count = plan.source_count;
    for (std::size_t source_index = 0U;
         source_index < planned.source_count; ++source_index) {
      descriptor.view.scalar_scales[source_index] =
          planned.sources[source_index].scalar_scale;
    }
    owner.descriptors[index] = descriptor;
    owner.manifests[index] = manifests[index];
    offset += plan.payload_bytes;
  }
  owner.descriptor_count = plan_count;
  owner.manifest_count = plan_count;
  if (offset != owner.bytes) {
    result.hard_failure = true;
    result.message = scope == PreparationScope::kNvfp4Only
                         ? "P40 packed NVFP4 payloads did not close the exact "
                           "9,625,927,680-byte arena"
                         : "P40 packed projection payloads did not close the "
                           "exact 16,840,130,560-byte arena";
    (void)cudaStreamSynchronize(stream);
    (void)cudaStreamDestroy(stream);
    owner.release();
    return result;
  }

  status = cudaStreamSynchronize(stream);
  const cudaError_t destroy_status = cudaStreamDestroy(stream);
  if (status != cudaSuccess || destroy_status != cudaSuccess) {
    const cudaError_t failure =
        status != cudaSuccess ? status : destroy_status;
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(failure);
    result.message = "P40 nonblocking pack stream failed to synchronize or "
                     "destroy";
    owner.release();
    return result;
  }

  const bool manifest_inventory_valid =
      scope == PreparationScope::kNvfp4Only
          ? valid_nvfp4_only_manifest_inventory(owner.manifests,
                                                owner.manifest_count)
          : kernels::validate_sm87_p40_packed_artifact_inventory(
                owner.manifests.data(), owner.manifest_count)
                .valid();
  const bool attachment_valid =
      manifest_inventory_valid &&
      (scope == PreparationScope::kNvfp4Only
           ? model_weights.attach_p40_packed_nvfp4_sidecars(
                 owner.descriptors.data(), owner.descriptor_count)
           : model_weights.attach_p40_packed_projection_sidecars(
                 owner.descriptors.data(), owner.descriptor_count));
  if (!manifest_inventory_valid || !attachment_valid) {
    result.hard_failure = true;
    result.message = !manifest_inventory_valid
                         ? "retained P40 manifest inventory failed final "
                           "validation"
                     : scope == PreparationScope::kNvfp4Only
                         ? "ModelWeights rejected the exact P40 packed "
                           "NVFP4-only transaction"
                         : "ModelWeights rejected the complete P40 packed "
                           "projection transaction";
    owner.release();
    return result;
  }

  result.enabled = true;
  if (scope == PreparationScope::kNvfp4Only) {
    result.artifacts = kP40PackedNvfp4ArtifactCount;
    result.sources = kP40PackedNvfp4SourceCount;
    result.fp8_logical = 0U;
    result.fp8_physical = 0U;
    result.nvfp4_physical = kP40PackedNvfp4ArtifactCount;
  } else {
    const auto inventory =
        kernels::validate_sm87_p40_packed_artifact_inventory(
            owner.manifests.data(), owner.manifest_count);
    result.artifacts = inventory.artifact_count;
    result.sources = inventory.source_identities;
    result.fp8_logical = inventory.fp8_logical_roles;
    result.fp8_physical = inventory.fp8_artifacts;
    result.nvfp4_physical =
        inventory.gate_up_artifacts + inventory.down_artifacts;
  }
  result.bytes = owner.bytes;
  return result;
}

#endif  // Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION

}  // namespace

P40PackedProjectionAssets::~P40PackedProjectionAssets() { release(); }

P40PackedProjectionAssets::P40PackedProjectionAssets(
    P40PackedProjectionAssets&& other) noexcept
    : arena(std::exchange(other.arena, nullptr)),
      bytes(std::exchange(other.bytes, 0U)),
      descriptors(other.descriptors),
      descriptor_count(std::exchange(other.descriptor_count, 0U)),
      manifests(other.manifests),
      manifest_count(std::exchange(other.manifest_count, 0U)) {
  other.descriptors = {};
  other.manifests = {};
}

P40PackedProjectionAssets& P40PackedProjectionAssets::operator=(
    P40PackedProjectionAssets&& other) noexcept {
  if (this != &other) {
    release();
    arena = std::exchange(other.arena, nullptr);
    bytes = std::exchange(other.bytes, 0U);
    descriptors = other.descriptors;
    descriptor_count = std::exchange(other.descriptor_count, 0U);
    manifests = other.manifests;
    manifest_count = std::exchange(other.manifest_count, 0U);
    other.descriptors = {};
    other.manifests = {};
  }
  return *this;
}

void P40PackedProjectionAssets::release() noexcept {
#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
  if (arena != nullptr) {
    (void)cudaFree(arena);
  }
#endif
  arena = nullptr;
  bytes = 0U;
  descriptors = {};
  descriptor_count = 0U;
  manifests = {};
  manifest_count = 0U;
}

P40PackedProjectionPreparationStats prepare_p40_packed_projection_assets(
    const ResidentWeights& resident, ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    P40PackedProjectionAssets& owner) {
#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
  try {
    return prepare_impl(resident, model_weights,
                        minimum_free_bytes_after_prepare,
                        PreparationScope::kCompleteProjection, owner);
  } catch (const std::exception& error) {
    owner.release();
    P40PackedProjectionPreparationStats result;
    result.hard_failure = true;
    result.message =
        std::string("exception during P40 packed projection preparation: ") +
        error.what();
    return result;
  } catch (...) {
    owner.release();
    P40PackedProjectionPreparationStats result;
    result.hard_failure = true;
    result.message =
        "unknown exception during P40 packed projection preparation";
    return result;
  }
#else
  (void)resident;
  (void)model_weights;
  (void)minimum_free_bytes_after_prepare;
  (void)owner;
  P40PackedProjectionPreparationStats result;
  result.hard_failure = true;
  result.message = "P40 packed projection asset preparation is not compiled";
  return result;
#endif
}

P40PackedProjectionPreparationStats prepare_p40_packed_nvfp4_assets(
    const ResidentWeights& resident, ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    P40PackedProjectionAssets& owner) {
#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
  try {
    return prepare_impl(resident, model_weights,
                        minimum_free_bytes_after_prepare,
                        PreparationScope::kNvfp4Only, owner);
  } catch (const std::exception& error) {
    owner.release();
    P40PackedProjectionPreparationStats result;
    result.hard_failure = true;
    result.message =
        std::string("exception during P40 packed NVFP4 preparation: ") +
        error.what();
    return result;
  } catch (...) {
    owner.release();
    P40PackedProjectionPreparationStats result;
    result.hard_failure = true;
    result.message = "unknown exception during P40 packed NVFP4 preparation";
    return result;
  }
#else
  (void)resident;
  (void)model_weights;
  (void)minimum_free_bytes_after_prepare;
  (void)owner;
  P40PackedProjectionPreparationStats result;
  result.hard_failure = true;
  result.message = "P40 packed NVFP4 asset preparation is not compiled";
  return result;
#endif
}

}  // namespace q3x::runtime
