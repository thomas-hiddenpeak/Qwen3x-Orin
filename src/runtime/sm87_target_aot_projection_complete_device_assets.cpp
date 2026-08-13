#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"

#include "q3x/core/sha256.h"

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)

namespace st = io::safetensors;

constexpr std::string_view kModelRepository =
    "nvidia/Qwen3.6-27B-NVFP4";
constexpr std::string_view kModelRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";
constexpr std::string_view kCheckpointIdentityDomain =
    "q3x.sm87.target-aot.complete-assets.checkpoint.v2";
constexpr std::string_view kSourceIdentityDomain =
    "q3x.sm87.target-aot.complete-assets.source.v2";
constexpr std::string_view kInventoryIdentityDomain =
    "q3x.sm87.target-aot.complete-assets.inventory.v2";
constexpr std::string_view kArtifactIdentityDomain =
    "q3x.sm87.target-aot.complete-assets.artifact.v2";
constexpr std::string_view kOwnerIdentityDomain =
    "q3x.sm87.target-aot.complete-assets.owner.v2";
constexpr std::string_view kAllocationIdentityDomain =
    "q3x.sm87.target-aot.complete-assets.allocation.v2";
constexpr std::string_view kStreamIdentityDomain =
    "q3x.sm87.target-aot.complete-assets.stream.v2";
constexpr std::string_view kEventIdentityDomain =
    "q3x.sm87.target-aot.complete-assets.event.v2";
constexpr std::string_view kVerifiedPayloadCatalogDomain =
    "q3x.sm87.target-aot.complete-assets.catalog.v2";

std::atomic<std::uint64_t> g_complete_asset_transaction_serial{1U};

enum class PlannedSourceEncoding : std::uint8_t {
  kInvalid = 0U,
  kNvFp4,
  kFp8,
};

struct PlannedSource final {
  PlannedSourceEncoding encoding = PlannedSourceEncoding::kInvalid;
  kernels::Sm87TargetAotLogicalRole logical_role =
      kernels::Sm87TargetAotLogicalRole::kInvalid;
  std::string module_name;
  const NvFp4LinearWeight* nvfp4 = nullptr;
  const Fp8LinearWeight* fp8 = nullptr;
  const DeviceTensorView* weight = nullptr;
  const DeviceTensorView* block_scale = nullptr;
  const DeviceTensorView* tensor_scale = nullptr;
  const DeviceTensorView* input_scale = nullptr;
  std::uint64_t tensor_identity = 0U;
  float scalar_scale = 0.0F;
};

struct PlannedArtifact final {
  std::size_t layer_index = 0U;
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  std::uint64_t inventory_identity = 0U;
  std::uint64_t artifact_identity = 0U;
  std::uint64_t device_arena_offset = 0U;
  std::uint32_t source_count = 0U;
  std::array<PlannedSource,
             kernels::kSm87TargetAotProjectionPackedMaxPartitions>
      sources{};
};

using ArtifactPlans =
    std::array<PlannedArtifact,
               kSm87TargetAotCompleteProjectionDeviceArtifactCount>;

template <typename Unsigned>
[[nodiscard]] bool hash_unsigned(core::Sha256& hasher,
                                 const Unsigned value) noexcept {
  static_assert(std::is_unsigned_v<Unsigned>);
  std::array<std::uint8_t, sizeof(Unsigned)> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
  return hasher.update(bytes.data(), bytes.size());
}

[[nodiscard]] bool hash_string(core::Sha256& hasher,
                               const std::string_view value) noexcept {
  return hash_unsigned(hasher, static_cast<std::uint64_t>(value.size())) &&
         hasher.update(value.data(), value.size());
}

[[nodiscard]] std::uint64_t digest_identity(
    const core::Sha256Digest& digest) noexcept {
  std::uint64_t value = 0U;
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    value |= static_cast<std::uint64_t>(digest.bytes[index]) <<
             (8U * index);
  }
  return value;
}

[[nodiscard]] std::uint64_t finish_identity(core::Sha256& hasher) noexcept {
  return digest_identity(hasher.finalize());
}

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] bool digest_empty(
    const NvFp4MarlinP40ParityDigest& digest) noexcept {
  return std::all_of(digest.begin(), digest.end(),
                     [](const std::uint8_t byte) { return byte == 0U; });
}

[[nodiscard]] bool empty_p40_view(
    const kernels::Sm87P40PackedProjectionDeviceView& view) noexcept {
  return view.payload == nullptr && view.payload_bytes == 0U &&
         view.artifact_identity == 0U &&
         view.role == kernels::Sm87P40PackedProjectionRole::kInvalid &&
         view.tactic == kernels::Sm87P40PackedTactic::kInvalid &&
         view.source_count == 0U &&
         std::all_of(view.scalar_scales.begin(), view.scalar_scales.end(),
                     [](const float scale) { return scale == 0.0F; });
}

[[nodiscard]] bool empty_parity_view(
    const NvFp4MarlinP40ParityDeviceView& view) noexcept {
  const auto source_empty = [](const NvFp4MarlinP40ParitySourceManifest& source) {
    return source.role == NvFp4MarlinP40ParitySourceRole::kInvalid &&
           source.tensor_identity == 0U &&
           digest_empty(source.weight_digest) &&
           digest_empty(source.scale_digest) &&
           source.global_scale_bits == 0U;
  };
  const auto& manifest = view.manifest;
  return view.weight == nullptr && view.scales == nullptr &&
         view.global_scale == nullptr &&
         manifest.version == kNvFp4MarlinP40ParityManifestVersion &&
         manifest.layer_index == 0U &&
         manifest.role == NvFp4MarlinP40ParityRole::kInvalid &&
         manifest.layout == NvFp4MarlinP40ParityLayout::kInvalid &&
         manifest.output_features == 0U && manifest.input_features == 0U &&
         manifest.weight_bytes == 0U && manifest.scale_bytes == 0U &&
         manifest.artifact_identity == 0U &&
         digest_empty(manifest.transformation_digest) &&
         manifest.source_count == 0U &&
         std::all_of(manifest.sources.begin(), manifest.sources.end(),
                     source_empty);
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
  return resident.size_bytes() <=
             std::numeric_limits<std::uintptr_t>::max() - base &&
         view.arena_offset <=
             std::numeric_limits<std::uintptr_t>::max() - base &&
         reinterpret_cast<std::uintptr_t>(view.device_data) ==
             base + static_cast<std::uintptr_t>(view.arena_offset);
}

[[nodiscard]] const DeviceTensorView* exact_resident_tensor(
    const ResidentWeights& resident, const std::string& name,
    const void* const expected_pointer, const st::DType expected_dtype,
    const std::initializer_list<std::uint64_t> expected_shape,
    const std::uint64_t expected_bytes) noexcept {
  const DeviceTensorView* const view = resident.find(name);
  if (view == nullptr || view->device_data != expected_pointer ||
      view->dtype != expected_dtype || view->byte_size != expected_bytes ||
      view->shape.size() != expected_shape.size() ||
      !resident_range_contains(resident, *view) ||
      !std::equal(view->shape.begin(), view->shape.end(),
                  expected_shape.begin(), expected_shape.end())) {
    return nullptr;
  }
  return view;
}

[[nodiscard]] bool verify_observed_checkpoint(
    const ResidentWeights& resident, core::Sha256Digest& checkpoint_digest,
    std::string& message) {
  if (!resident || resident.arena_data() == nullptr ||
      resident.size_bytes() != kPinnedQwen36_27BArenaBytes) {
    message = "resident arena is not the exact pinned Qwen3.6-27B arena";
    return false;
  }
  const auto& pinned = pinned_qwen36_27b_shards();
  const auto& observed = resident.stats().shards;
  std::array<bool, 16U> used{};
  if (pinned.empty() || observed.size() != pinned.size() ||
      observed.size() > used.size()) {
    message = "resident loader did not report the complete pinned shard set";
    return false;
  }
  core::Sha256 hasher;
  bool ok = hash_string(hasher, kCheckpointIdentityDomain) &&
            hash_string(hasher, kModelRepository) &&
            hash_string(hasher, kModelRevision) &&
            hash_unsigned(hasher, static_cast<std::uint64_t>(pinned.size()));
  for (const ShardIdentity& expected : pinned) {
    std::size_t match = observed.size();
    for (std::size_t index = 0U; index < observed.size(); ++index) {
      if (!used[index] && observed[index].filename == expected.filename) {
        match = index;
        break;
      }
    }
    if (match == observed.size() ||
        observed[match].sha256 != expected.sha256 ||
        observed[match].bytes_read != expected.file_size) {
      message = "resident shard identity differs from the pinned checkpoint at " +
                expected.filename;
      return false;
    }
    used[match] = true;
    ok = ok && hash_string(hasher, expected.filename) &&
         hash_unsigned(hasher, expected.file_size) &&
         hash_string(hasher, observed[match].sha256);
  }
  checkpoint_digest = hasher.finalize();
  if (!ok || digest_identity(checkpoint_digest) == 0U) {
    checkpoint_digest = {};
    message = "failed to derive the pinned checkpoint identity";
    return false;
  }
  return true;
}

[[nodiscard]] std::uint64_t make_source_identity(
    const core::Sha256Digest& checkpoint_digest,
    const std::string_view module_name,
    const kernels::Sm87TargetAotLogicalRole role) noexcept {
  core::Sha256 hasher;
  const bool ok = hash_string(hasher, kSourceIdentityDomain) &&
                  hasher.update(checkpoint_digest.bytes.data(),
                                checkpoint_digest.bytes.size()) &&
                  hash_string(hasher, module_name) &&
                  hash_unsigned(hasher, static_cast<std::uint8_t>(role));
  const std::uint64_t result = finish_identity(hasher);
  return ok ? result : 0U;
}

[[nodiscard]] std::uint64_t make_inventory_identity(
    const core::Sha256Digest& checkpoint_digest,
    const PlannedArtifact& planned) noexcept {
  core::Sha256 hasher;
  bool ok = hash_string(hasher, kInventoryIdentityDomain) &&
            hasher.update(checkpoint_digest.bytes.data(),
                          checkpoint_digest.bytes.size()) &&
            hash_unsigned(hasher,
                          static_cast<std::uint64_t>(planned.layer_index)) &&
            hash_unsigned(hasher, static_cast<std::uint8_t>(planned.role)) &&
            hash_unsigned(hasher, planned.source_count);
  for (std::size_t index = 0U; index < planned.source_count; ++index) {
    ok = ok && hash_unsigned(hasher, planned.sources[index].tensor_identity);
  }
  const std::uint64_t result = finish_identity(hasher);
  return ok ? result : 0U;
}

[[nodiscard]] std::uint64_t make_artifact_identity(
    const core::Sha256Digest& checkpoint_digest,
    const PlannedArtifact& planned) noexcept {
  core::Sha256 hasher;
  const bool ok = hash_string(hasher, kArtifactIdentityDomain) &&
                  hasher.update(checkpoint_digest.bytes.data(),
                                checkpoint_digest.bytes.size()) &&
                  hash_unsigned(
                      hasher,
                      static_cast<std::uint64_t>(planned.layer_index)) &&
                  hash_unsigned(hasher,
                                static_cast<std::uint8_t>(planned.role)) &&
                  hash_unsigned(hasher, planned.inventory_identity) &&
                  hash_unsigned(
                      hasher, static_cast<std::uint16_t>(
                                  kernels::Sm87TargetAotProjectionPackedTransformIdentity::
                                      kCanonicalNkToConsumerN64K16LaneComponentV1));
  const std::uint64_t result = finish_identity(hasher);
  return ok ? result : 0U;
}

[[nodiscard]] std::uint64_t make_runtime_identity(
    const std::string_view domain,
    const core::Sha256Digest& checkpoint_digest,
    const std::uint64_t transaction_serial, const std::uint64_t first,
    const std::uint64_t second = 0U,
    const std::uint64_t third = 0U) noexcept {
  core::Sha256 hasher;
  const bool ok = hash_string(hasher, domain) &&
                  hasher.update(checkpoint_digest.bytes.data(),
                                checkpoint_digest.bytes.size()) &&
                  hash_unsigned(hasher, transaction_serial) &&
                  hash_unsigned(hasher, first) &&
                  hash_unsigned(hasher, second) &&
                  hash_unsigned(hasher, third);
  const std::uint64_t result = finish_identity(hasher);
  return ok ? result : 0U;
}

[[nodiscard]] bool fp8_sidecars_empty(
    const Fp8LinearWeight& projection) noexcept {
  return projection.m1_aosoa4_preswizzled_weight == nullptr &&
         projection.prefill_qkv_register_feed_sidecar == nullptr &&
         projection.prefill_supermatrix_sidecar == nullptr &&
         projection.prefill_marlin_weight == nullptr &&
         projection.prefill_marlin_scales == nullptr &&
         empty_p40_view(projection.prefill_p40_packed_artifact);
}

[[nodiscard]] bool nvfp4_sidecars_empty(
    const NvFp4LinearWeight& projection) noexcept {
  return projection.down_scale6_sidecar == nullptr &&
         projection.down_scale6_base == 0U &&
         projection.down_consumer_order_weight == nullptr &&
         projection.decode_gate_up_coupled_feed_sidecar == nullptr &&
         projection.prefill_marlin_gate_up_layout ==
             NvFp4MarlinGateUpLayout::kUnbound &&
         projection.prefill_marlin_weight == nullptr &&
         projection.prefill_marlin_scales == nullptr &&
         projection.prefill_marlin_global_scale == nullptr &&
         empty_p40_view(projection.prefill_p40_packed_artifact) &&
         empty_parity_view(projection.prefill_p40_vllm_marlin_parity);
}

[[nodiscard]] bool plan_nvfp4_source(
    const ResidentWeights& resident, const NvFp4LinearWeight& projection,
    const std::string& module_name,
    const kernels::Sm87TargetAotLogicalRole logical_role,
    const std::uint64_t expected_output, const std::uint64_t expected_input,
    const core::Sha256Digest& checkpoint_digest, PlannedSource& source) {
  if (projection.packed_weight == nullptr ||
      projection.block_scale == nullptr ||
      projection.weight_scale_2_device == nullptr ||
      projection.input_scale_device == nullptr ||
      projection.output_size != expected_output ||
      projection.input_size != expected_input ||
      !std::isfinite(projection.weight_scale_2) ||
      projection.weight_scale_2 <= 0.0F ||
      !std::isfinite(projection.input_scale) || projection.input_scale < 0.0F ||
      reinterpret_cast<std::uintptr_t>(projection.packed_weight) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(projection.block_scale) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(projection.weight_scale_2_device) %
              alignof(float) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(projection.input_scale_device) %
              alignof(float) !=
          0U ||
      !nvfp4_sidecars_empty(projection)) {
    return false;
  }
  const std::uint64_t values = expected_output * expected_input;
  const DeviceTensorView* const weight = exact_resident_tensor(
      resident, module_name + ".weight", projection.packed_weight,
      st::DType::kU8, {expected_output, expected_input / 2U}, values / 2U);
  const DeviceTensorView* const block_scale = exact_resident_tensor(
      resident, module_name + ".weight_scale", projection.block_scale,
      st::DType::kF8E4M3, {expected_output, expected_input / 16U},
      values / 16U);
  const DeviceTensorView* const tensor_scale = exact_resident_tensor(
      resident, module_name + ".weight_scale_2",
      projection.weight_scale_2_device, st::DType::kF32, {}, sizeof(float));
  const DeviceTensorView* const input_scale = exact_resident_tensor(
      resident, module_name + ".input_scale", projection.input_scale_device,
      st::DType::kF32, {}, sizeof(float));
  const std::uint64_t tensor_identity =
      make_source_identity(checkpoint_digest, module_name, logical_role);
  if (weight == nullptr || block_scale == nullptr || tensor_scale == nullptr ||
      input_scale == nullptr || tensor_identity == 0U) {
    return false;
  }
  source.encoding = PlannedSourceEncoding::kNvFp4;
  source.logical_role = logical_role;
  source.module_name = module_name;
  source.nvfp4 = &projection;
  source.weight = weight;
  source.block_scale = block_scale;
  source.tensor_scale = tensor_scale;
  source.input_scale = input_scale;
  source.tensor_identity = tensor_identity;
  source.scalar_scale = projection.weight_scale_2;
  return true;
}

[[nodiscard]] bool plan_fp8_source(
    const ResidentWeights& resident, const Fp8LinearWeight& projection,
    const std::string& module_name,
    const kernels::Sm87TargetAotLogicalRole logical_role,
    const std::uint64_t expected_output, const std::uint64_t expected_input,
    const core::Sha256Digest& checkpoint_digest, PlannedSource& source) {
  if (projection.weight == nullptr ||
      projection.weight_scale_device == nullptr ||
      projection.input_scale_device == nullptr ||
      projection.output_size != expected_output ||
      projection.input_size != expected_input ||
      !std::isfinite(projection.weight_scale) ||
      projection.weight_scale <= 0.0F ||
      !std::isfinite(projection.input_scale) || projection.input_scale < 0.0F ||
      reinterpret_cast<std::uintptr_t>(projection.weight) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(projection.weight_scale_device) %
              alignof(float) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(projection.input_scale_device) %
              alignof(float) !=
          0U ||
      !fp8_sidecars_empty(projection)) {
    return false;
  }
  const std::uint64_t values = expected_output * expected_input;
  const DeviceTensorView* const weight = exact_resident_tensor(
      resident, module_name + ".weight", projection.weight,
      st::DType::kF8E4M3, {expected_output, expected_input}, values);
  const DeviceTensorView* const tensor_scale = exact_resident_tensor(
      resident, module_name + ".weight_scale",
      projection.weight_scale_device, st::DType::kF32, {}, sizeof(float));
  const DeviceTensorView* const input_scale = exact_resident_tensor(
      resident, module_name + ".input_scale", projection.input_scale_device,
      st::DType::kF32, {}, sizeof(float));
  const std::uint64_t tensor_identity =
      make_source_identity(checkpoint_digest, module_name, logical_role);
  if (weight == nullptr || tensor_scale == nullptr || input_scale == nullptr ||
      tensor_identity == 0U) {
    return false;
  }
  source.encoding = PlannedSourceEncoding::kFp8;
  source.logical_role = logical_role;
  source.module_name = module_name;
  source.fp8 = &projection;
  source.weight = weight;
  source.tensor_scale = tensor_scale;
  source.input_scale = input_scale;
  source.tensor_identity = tensor_identity;
  source.scalar_scale = projection.weight_scale;
  return true;
}

[[nodiscard]] bool append_planned_artifact(
    ArtifactPlans& plans, std::size_t& count,
    PlannedArtifact artifact) noexcept {
  if (count >= plans.size()) {
    return false;
  }
  plans[count++] = std::move(artifact);
  return true;
}

[[nodiscard]] bool plan_inventory(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const core::Sha256Digest& checkpoint_digest, ArtifactPlans& plans,
    std::string& message) {
  std::array<std::uint64_t,
             kSm87TargetAotCompleteProjectionDeviceSourceCount>
      source_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      inventory_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      artifact_identities{};
  std::size_t source_count = 0U;
  std::size_t artifact_count = 0U;
  std::uint64_t arena_offset = 0U;

  for (std::size_t layer_index = 0U;
       layer_index < kSm87TargetAotCompleteProjectionDeviceLayerCount;
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
    gate_up.role = kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp;
    gate_up.source_count = 2U;
    PlannedArtifact down_artifact;
    down_artifact.layer_index = layer_index;
    down_artifact.role = kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
    down_artifact.source_count = 1U;
    if (gate == nullptr || up == nullptr || down == nullptr ||
        !plan_nvfp4_source(
            resident, *gate, prefix + "mlp.gate_proj",
            kernels::Sm87TargetAotLogicalRole::kNvFp4Gate, 17'408U, 5'120U,
            checkpoint_digest, gate_up.sources[0U]) ||
        !plan_nvfp4_source(
            resident, *up, prefix + "mlp.up_proj",
            kernels::Sm87TargetAotLogicalRole::kNvFp4Up, 17'408U, 5'120U,
            checkpoint_digest, gate_up.sources[1U]) ||
        !plan_nvfp4_source(
            resident, *down, prefix + "mlp.down_proj",
            kernels::Sm87TargetAotLogicalRole::kNvFp4Down, 5'120U, 17'408U,
            checkpoint_digest, down_artifact.sources[0U])) {
      message = "ineligible or non-resident NVFP4 inventory at layer " +
                std::to_string(layer_index);
      return false;
    }

    PlannedArtifact fp8_input;
    fp8_input.layer_index = layer_index;
    PlannedArtifact fp8_output;
    fp8_output.layer_index = layer_index;
    fp8_output.role =
        kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput;
    fp8_output.source_count = 1U;
    if (!sm87_target_aot_complete_is_full_layer(layer_index)) {
      const auto* const linear =
          std::get_if<LinearAttentionWeights>(&layer.attention);
      if (linear == nullptr) {
        message = "attention schedule is not linear at layer " +
                  std::to_string(layer_index);
        return false;
      }
      const auto* const qkv =
          std::get_if<Fp8LinearWeight>(&linear->in_proj_qkv);
      const auto* const z = std::get_if<Fp8LinearWeight>(&linear->in_proj_z);
      const auto* const output =
          std::get_if<Fp8LinearWeight>(&linear->out_proj);
      fp8_input.role = kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
      fp8_input.source_count = 2U;
      if (qkv == nullptr || z == nullptr || output == nullptr ||
          !plan_fp8_source(
              resident, *qkv, prefix + "linear_attn.in_proj_qkv",
              kernels::Sm87TargetAotLogicalRole::kFp8GdnQkv, 10'240U,
              5'120U, checkpoint_digest, fp8_input.sources[0U]) ||
          !plan_fp8_source(
              resident, *z, prefix + "linear_attn.in_proj_z",
              kernels::Sm87TargetAotLogicalRole::kFp8GdnZ, 6'144U, 5'120U,
              checkpoint_digest, fp8_input.sources[1U]) ||
          !plan_fp8_source(
              resident, *output, prefix + "linear_attn.out_proj",
              kernels::Sm87TargetAotLogicalRole::kFp8AttentionOutput,
              5'120U, 6'144U, checkpoint_digest,
              fp8_output.sources[0U])) {
        message = "ineligible or non-resident GDN FP8 inventory at layer " +
                  std::to_string(layer_index);
        return false;
      }
    } else {
      const auto* const full =
          std::get_if<FullAttentionWeights>(&layer.attention);
      if (full == nullptr) {
        message = "attention schedule is not full at layer " +
                  std::to_string(layer_index);
        return false;
      }
      const auto* const query = std::get_if<Fp8LinearWeight>(&full->q_proj);
      const auto* const key = std::get_if<Fp8LinearWeight>(&full->k_proj);
      const auto* const value = std::get_if<Fp8LinearWeight>(&full->v_proj);
      const auto* const output = std::get_if<Fp8LinearWeight>(&full->o_proj);
      fp8_input.role = kernels::Sm87TargetAotProjectionRole::kFp8FullQkv;
      fp8_input.source_count = 3U;
      if (query == nullptr || key == nullptr || value == nullptr ||
          output == nullptr ||
          !plan_fp8_source(
              resident, *query, prefix + "self_attn.q_proj",
              kernels::Sm87TargetAotLogicalRole::kFp8FullQGate, 12'288U,
              5'120U, checkpoint_digest, fp8_input.sources[0U]) ||
          !plan_fp8_source(
              resident, *key, prefix + "self_attn.k_proj",
              kernels::Sm87TargetAotLogicalRole::kFp8FullK, 1'024U, 5'120U,
              checkpoint_digest, fp8_input.sources[1U]) ||
          !plan_fp8_source(
              resident, *value, prefix + "self_attn.v_proj",
              kernels::Sm87TargetAotLogicalRole::kFp8FullV, 1'024U, 5'120U,
              checkpoint_digest, fp8_input.sources[2U]) ||
          !plan_fp8_source(
              resident, *output, prefix + "self_attn.o_proj",
              kernels::Sm87TargetAotLogicalRole::kFp8AttentionOutput,
              5'120U, 6'144U, checkpoint_digest,
              fp8_output.sources[0U])) {
        message = "ineligible or non-resident full-attention FP8 inventory at layer " +
                  std::to_string(layer_index);
        return false;
      }
    }

    std::array<PlannedArtifact, 4U> layer_artifacts{{
        std::move(gate_up), std::move(down_artifact), std::move(fp8_input),
        std::move(fp8_output)}};
    for (PlannedArtifact& artifact : layer_artifacts) {
      const auto layout =
          kernels::sm87_target_aot_projection_packed_layout(artifact.role);
      const std::size_t expected_ordinal =
          sm87_target_aot_complete_descriptor_ordinal(layer_index,
                                                       artifact.role);
      const std::uint64_t expected_arena_offset =
          sm87_target_aot_complete_expected_arena_offset(layer_index,
                                                          artifact.role);
      if (!layout.valid() || layout.partition_count != artifact.source_count ||
          expected_ordinal != artifact_count ||
          expected_arena_offset != arena_offset ||
          arena_offset % layout.payload_alignment != 0U ||
          arena_offset > kSm87TargetAotCompleteProjectionDeviceArenaBytes ||
          layout.payload_bytes >
              kSm87TargetAotCompleteProjectionDeviceArenaBytes - arena_offset) {
        message = "complete target-AOT layout/order escaped its fixed arena";
        return false;
      }
      artifact.device_arena_offset = arena_offset;
      for (std::size_t source_index = 0U;
           source_index < artifact.source_count; ++source_index) {
        const std::uint64_t identity =
            artifact.sources[source_index].tensor_identity;
        if (source_count >= source_identities.size() || identity == 0U ||
            std::find(source_identities.begin(),
                      source_identities.begin() + source_count,
                      identity) != source_identities.begin() + source_count) {
          message = "complete target-AOT source identities are not unique";
          return false;
        }
        source_identities[source_count++] = identity;
      }
      artifact.inventory_identity =
          make_inventory_identity(checkpoint_digest, artifact);
      artifact.artifact_identity =
          make_artifact_identity(checkpoint_digest, artifact);
      if (artifact.inventory_identity == 0U ||
          artifact.artifact_identity == 0U ||
          std::find(inventory_identities.begin(),
                    inventory_identities.begin() + artifact_count,
                    artifact.inventory_identity) !=
              inventory_identities.begin() + artifact_count ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + artifact_count,
                    artifact.artifact_identity) !=
              artifact_identities.begin() + artifact_count ||
          !append_planned_artifact(plans, artifact_count,
                                   std::move(artifact))) {
        message = "complete target-AOT artifact identities are invalid";
        return false;
      }
      inventory_identities[artifact_count - 1U] =
          plans[artifact_count - 1U].inventory_identity;
      artifact_identities[artifact_count - 1U] =
          plans[artifact_count - 1U].artifact_identity;
      arena_offset += layout.payload_bytes;
    }
  }
  if (artifact_count != plans.size() ||
      source_count != source_identities.size() ||
      arena_offset != kSm87TargetAotCompleteProjectionDeviceArenaBytes) {
    message = "complete target-AOT inventory did not close 256 artifacts, "
              "400 sources, and the exact arena";
    return false;
  }
  return true;
}

[[nodiscard]] bool copy_and_bind_sources(
    const PlannedArtifact& planned, const cudaStream_t stream,
    std::vector<std::uint8_t>& canonical,
    Sm87TargetAotProjectionSourceSet& sources, std::uint64_t& copied_bytes,
    int& cuda_error, std::string& message) {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(planned.role);
  if (!layout.valid() || planned.source_count != layout.partition_count ||
      canonical.size() < layout.payload_bytes) {
    message = "invalid complete target-AOT source staging contract";
    return false;
  }
  sources = {};
  sources.role = planned.role;
  sources.inventory_identity = planned.inventory_identity;
  sources.source_count = planned.source_count;
  std::array<std::uint32_t,
             kernels::kSm87TargetAotProjectionPackedMaxPartitions>
      observed_scale_bits{};
  std::uint64_t cursor = 0U;
  for (std::size_t index = 0U; index < planned.source_count; ++index) {
    const PlannedSource& source = planned.sources[index];
    const auto& partition = layout.partitions[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t weight_bytes = values * partition.weight_bits / 8U;
    const std::uint64_t scale_bytes =
        partition.block_scale_group_k == 0U
            ? 0U
            : values / partition.block_scale_group_k;
    const bool fp8 = source.encoding == PlannedSourceEncoding::kFp8;
    if (source.weight == nullptr || source.tensor_scale == nullptr ||
        source.input_scale == nullptr ||
        (fp8 ? source.fp8 == nullptr || source.nvfp4 != nullptr ||
                   source.block_scale != nullptr || scale_bytes != 0U
             : source.nvfp4 == nullptr || source.fp8 != nullptr ||
                   source.block_scale == nullptr || scale_bytes == 0U) ||
        cursor > layout.payload_bytes ||
        weight_bytes > layout.payload_bytes - cursor) {
      message = "planned complete target-AOT source is incomplete";
      return false;
    }
    std::uint8_t* const weight_host =
        canonical.data() + static_cast<std::size_t>(cursor);
    cudaError_t status = cudaMemcpyAsync(
        weight_host, source.weight->device_data,
        static_cast<std::size_t>(weight_bytes), cudaMemcpyDeviceToHost,
        stream);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      message = "D2H failed for " + source.module_name + ".weight";
      return false;
    }
    cursor += weight_bytes;
    const std::uint8_t* scale_host = nullptr;
    if (scale_bytes != 0U) {
      if (scale_bytes > layout.payload_bytes - cursor) {
        message = "complete target-AOT block-scale staging overflow";
        return false;
      }
      std::uint8_t* const mutable_scale_host =
          canonical.data() + static_cast<std::size_t>(cursor);
      status = cudaMemcpyAsync(
          mutable_scale_host, source.block_scale->device_data,
          static_cast<std::size_t>(scale_bytes), cudaMemcpyDeviceToHost,
          stream);
      if (status != cudaSuccess) {
        cuda_error = static_cast<int>(status);
        message = "D2H failed for " + source.module_name + ".weight_scale";
        return false;
      }
      scale_host = mutable_scale_host;
      cursor += scale_bytes;
    }
    status = cudaMemcpyAsync(&observed_scale_bits[index],
                             source.tensor_scale->device_data,
                             sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                             stream);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      message = "D2H failed for scalar scale at " + source.module_name;
      return false;
    }
    auto& output = sources.sources[index];
    output.logical_role = source.logical_role;
    output.tensor_identity = source.tensor_identity;
    output.output_features = partition.output_features;
    output.input_features = partition.input_features;
    output.packed_weight = {weight_host,
                            static_cast<std::size_t>(weight_bytes)};
    output.block_scale = {scale_host,
                          static_cast<std::size_t>(scale_bytes)};
  }
  const cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    cuda_error = static_cast<int>(status);
    message = "complete target-AOT source D2H synchronization failed";
    return false;
  }
  for (std::size_t index = 0U; index < planned.source_count; ++index) {
    const std::uint32_t expected_bits =
        float_bits(planned.sources[index].scalar_scale);
    if (observed_scale_bits[index] != expected_bits ||
        !kernels::sm87_target_aot_projection_scale_bits_valid(
            observed_scale_bits[index])) {
      message = "device scalar scale differs from ModelWeights at " +
                planned.sources[index].module_name;
      return false;
    }
    sources.sources[index].tensor_scale_bits = observed_scale_bits[index];
  }
  if (cursor != layout.payload_bytes) {
    message = "complete target-AOT canonical bytes did not close artifact";
    return false;
  }
  copied_bytes += cursor + planned.source_count * sizeof(std::uint32_t);
  return true;
}

[[nodiscard]] bool append_catalog_entry(
    core::Sha256& hasher, const PlannedArtifact& planned,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt& transform,
    const kernels::Sm87TargetAotProjectionSha256Digest& readback) noexcept {
  bool ok =
      hash_unsigned(hasher,
                    static_cast<std::uint64_t>(planned.layer_index)) &&
      hash_unsigned(hasher, static_cast<std::uint8_t>(planned.role)) &&
      hash_unsigned(hasher, planned.device_arena_offset) &&
      hash_unsigned(hasher, manifest.artifact_identity) &&
      hash_unsigned(hasher, inventory.identity) &&
      hash_unsigned(hasher, static_cast<std::uint16_t>(manifest.plan_identity)) &&
      hash_unsigned(hasher,
                    static_cast<std::uint16_t>(manifest.layout_identity)) &&
      hash_unsigned(hasher, static_cast<std::uint8_t>(manifest.encoding)) &&
      hash_unsigned(hasher,
                    static_cast<std::uint16_t>(transform.transform_identity)) &&
      hash_unsigned(hasher, manifest.payload_bytes) &&
      hasher.update(manifest.payload_digest.bytes.data(),
                    manifest.payload_digest.bytes.size()) &&
      hasher.update(readback.bytes.data(), readback.bytes.size()) &&
      hash_unsigned(hasher, manifest.seal.value) &&
      hash_unsigned(hasher, inventory.source_count);
  for (std::size_t index = 0U; index < inventory.source_count; ++index) {
    const auto& source = inventory.sources[index];
    ok = ok &&
         hash_unsigned(hasher, static_cast<std::uint8_t>(source.logical_role)) &&
         hash_unsigned(hasher, source.partition_index) &&
         hash_unsigned(hasher, source.tensor_identity) &&
         hasher.update(source.weight_digest.bytes.data(),
                       source.weight_digest.bytes.size()) &&
         hasher.update(source.scale_digest.bytes.data(),
                       source.scale_digest.bytes.size()) &&
         hash_unsigned(hasher, source.tensor_scale_bits) &&
         hash_unsigned(hasher, source.payload_offset) &&
         hash_unsigned(hasher, source.payload_bytes);
  }
  return ok;
}

struct UploadContext final {
  std::uint8_t* arena = nullptr;
  std::uint64_t arena_bytes = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::uint64_t stream_identity = 0U;
  std::uint64_t transaction_serial = 0U;
  const core::Sha256Digest* checkpoint_digest = nullptr;
};

[[nodiscard]] bool make_descriptor(
    const UploadContext& context, const PlannedArtifact& planned,
    const std::size_t ordinal,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt& transform,
    const kernels::Sm87TargetAotProjectionSha256Digest& readback,
    Sm87TargetAotCompleteDeviceAssetDescriptor& descriptor) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(planned.role);
  if (!layout.valid() || context.arena == nullptr ||
      context.arena_bytes !=
          kSm87TargetAotCompleteProjectionDeviceArenaBytes ||
      context.allocation_identity == 0U || context.owner_identity == 0U ||
      context.device_ordinal < 0 || context.stream_identity == 0U ||
      context.transaction_serial == 0U ||
      context.checkpoint_digest == nullptr ||
      planned.device_arena_offset > context.arena_bytes ||
      layout.payload_bytes > context.arena_bytes - planned.device_arena_offset ||
      manifest.role != planned.role || manifest.payload_digest != readback ||
      !kernels::sm87_target_aot_projection_validate_transform_receipt(
          manifest, inventory, transform)) {
    return false;
  }
  const auto allocation_begin =
      reinterpret_cast<std::uintptr_t>(context.arena);
  if (context.arena_bytes >
      std::numeric_limits<std::uintptr_t>::max() - allocation_begin) {
    return false;
  }
  std::uint8_t* const destination =
      context.arena + static_cast<std::size_t>(planned.device_arena_offset);
  const std::uint64_t upload_event_identity = make_runtime_identity(
      kEventIdentityDomain, *context.checkpoint_digest,
      context.transaction_serial, planned.artifact_identity, ordinal, 1U);
  const std::uint64_t verification_event_identity = make_runtime_identity(
      kEventIdentityDomain, *context.checkpoint_digest,
      context.transaction_serial, planned.artifact_identity, ordinal, 2U);
  if (upload_event_identity == 0U || verification_event_identity == 0U ||
      upload_event_identity == verification_event_identity) {
    return false;
  }
  descriptor = {};
  descriptor.layer_index = planned.layer_index;
  descriptor.role = planned.role;
  descriptor.device_arena_offset = planned.device_arena_offset;
  descriptor.source_inventory = inventory;
  descriptor.manifest = manifest;
  descriptor.transform_receipt = transform;
  descriptor.encoding = manifest.encoding;

  const auto fill_common = [&](auto& upload) noexcept {
    upload.artifact_identity = manifest.artifact_identity;
    upload.source_inventory_identity = inventory.identity;
    upload.role = manifest.role;
    upload.plan_identity = manifest.plan_identity;
    upload.layout_identity = manifest.layout_identity;
    upload.transform_identity = transform.transform_identity;
    upload.host_payload_offset = manifest.payload_offset;
    upload.host_payload_bytes = manifest.payload_bytes;
    upload.host_payload_digest = manifest.payload_digest;
    upload.host_manifest_seal = manifest.seal;
    upload.tensor_scale_count = manifest.source_count;
    for (std::size_t index = 0U; index < manifest.source_count; ++index) {
      upload.tensor_scale_bits[index] =
          manifest.sources[index].tensor_scale_bits;
    }
    upload.device_allocation_identity = context.allocation_identity;
    upload.device_allocation_owner_identity = context.owner_identity;
    upload.device_ordinal = context.device_ordinal;
    upload.device_allocation_begin = allocation_begin;
    upload.device_allocation_end = allocation_begin + context.arena_bytes;
    upload.device_allocation_bytes = context.arena_bytes;
    upload.device_payload_begin = reinterpret_cast<std::uintptr_t>(destination);
    upload.device_payload_end =
        upload.device_payload_begin + layout.payload_bytes;
    upload.device_payload_bytes = layout.payload_bytes;
    upload.upload_stream_owner_identity = context.owner_identity;
    upload.upload_stream_identity = context.stream_identity;
    upload.upload_completion_event_identity = upload_event_identity;
    upload.verification_stream_owner_identity = context.owner_identity;
    upload.verification_stream_identity = context.stream_identity;
    upload.verification_completion_event_identity =
        verification_event_identity;
    upload.verification_readback_bytes = layout.payload_bytes;
    upload.verification_readback_digest = readback;
    upload.host_payload_digest_verified_before_copy = true;
    upload.host_payload_immutable_until_completion = true;
    upload.copy_enqueued_to_exact_payload_range = true;
    upload.completion_event_recorded_after_copy = true;
    upload.completion_event_observed = true;
    upload.upload_completed = true;
    upload.verification_copy_enqueued_from_exact_payload_range = true;
    upload.verification_event_recorded_after_copy = true;
    upload.verification_event_observed = true;
    upload.verification_completed = true;
    upload.device_payload_matches_host_payload = true;
    upload.allocation_retained_for_asset_lifetime = true;
  };

  if (sm87_target_aot_complete_role_is_nvfp4(planned.role)) {
    auto& upload = descriptor.nvfp4_upload_receipt;
    fill_common(upload);
    upload.receipt_identity =
        kernels::sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(
            upload);
    descriptor.nvfp4_view = kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
        manifest, inventory, transform, upload);
    if (upload.receipt_identity == 0U ||
        !kernels::
            sm87_target_aot_nvfp4_cuda_device_upload_receipt_structurally_valid(
                manifest, upload) ||
        !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(
            descriptor.nvfp4_view) ||
        !sm87_target_aot_complete_fp8_upload_receipt_zero(
            descriptor.fp8_upload_receipt) ||
        !sm87_target_aot_complete_fp8_view_zero(descriptor.fp8_view)) {
      return false;
    }
  } else if (sm87_target_aot_complete_role_is_fp8(planned.role)) {
    auto& upload = descriptor.fp8_upload_receipt;
    fill_common(upload);
    for (std::size_t index = 0U; index < manifest.source_count; ++index) {
      upload.compensated_tensor_scale_bf16_bits[index] =
          kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
              upload.tensor_scale_bits[index]);
      if (upload.compensated_tensor_scale_bf16_bits[index] == 0U) {
        return false;
      }
    }
    upload.receipt_identity =
        kernels::sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(
            upload);
    descriptor.fp8_view = kernels::sm87_target_aot_bind_fp8_cuda_asset(
        manifest, inventory, transform, upload);
    if (upload.receipt_identity == 0U ||
        !kernels::
            sm87_target_aot_fp8_cuda_device_upload_receipt_structurally_valid(
                manifest, upload) ||
        !kernels::sm87_target_aot_fp8_cuda_asset_valid(descriptor.fp8_view) ||
        !sm87_target_aot_complete_nvfp4_upload_receipt_zero(
            descriptor.nvfp4_upload_receipt) ||
        !sm87_target_aot_complete_nvfp4_view_zero(descriptor.nvfp4_view)) {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

[[nodiscard]] bool descriptor_valid(
    const Sm87TargetAotCompleteDeviceAssetDescriptor& descriptor,
    const std::size_t ordinal, const std::uintptr_t arena_begin,
    const std::uint64_t arena_bytes, const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::int32_t device_ordinal) noexcept {
  const auto expected = sm87_target_aot_complete_descriptor_ordinal(
      descriptor.layer_index, descriptor.role);
  return expected == ordinal &&
         sm87_target_aot_complete_device_descriptor_valid(
             descriptor, descriptor.layer_index, descriptor.role,
             descriptor.device_arena_offset, arena_begin, arena_bytes,
             owner_identity, allocation_identity, device_ordinal);
}

class PendingOnlinePreparation final {
 public:
  PendingOnlinePreparation() noexcept = default;
  ~PendingOnlinePreparation() { reset(); }
  PendingOnlinePreparation(const PendingOnlinePreparation&) = delete;
  PendingOnlinePreparation& operator=(const PendingOnlinePreparation&) =
      delete;

  void reset() noexcept {
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
    }
    if (upload_event != nullptr) {
      (void)cudaEventDestroy(upload_event);
      upload_event = nullptr;
    }
    if (verification_event != nullptr) {
      (void)cudaEventDestroy(verification_event);
      verification_event = nullptr;
    }
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }
    if (arena != nullptr) {
      (void)cudaFree(arena);
      arena = nullptr;
    }
    descriptors = {};
    bytes = 0U;
  }

  std::uint8_t* arena = nullptr;
  std::uint64_t bytes = 0U;
  cudaStream_t stream = nullptr;
  cudaEvent_t upload_event = nullptr;
  cudaEvent_t verification_event = nullptr;
  std::array<Sm87TargetAotCompleteDeviceAssetDescriptor,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      descriptors{};
};

#endif

}  // namespace

Sm87TargetAotCompleteProjectionDeviceAssets::
    ~Sm87TargetAotCompleteProjectionDeviceAssets() {
  // This admission has no executable launcher yet, so there can be no queued
  // consumer.  Still close both legal declaration orders: if the owner is
  // destroyed before its bound ModelWeights view, clear that view's private
  // reverse attachment before releasing the arena.  When ModelWeights was
  // destroyed first its detach path already cleared execution_bound_, and we
  // deliberately do not dereference the retained diagnostic pointer here.
  if (execution_bound_ && prepared_model_weights_ != nullptr) {
    auto* const model_weights =
        const_cast<ModelWeights*>(prepared_model_weights_);
    auto& attachment =
        model_weights->target_aot_complete_projection_attachment_;
    if (attachment.owner == this &&
        attachment.owner_identity == owner_identity_ &&
        attachment.allocation_identity == allocation_identity_) {
      attachment = {};
    }
    execution_bound_ = false;
  }
  release_unconditionally();
}

bool Sm87TargetAotCompleteProjectionDeviceAssets::release() noexcept {
  if (execution_bound_) {
    return false;
  }
  release_unconditionally();
  return true;
}

void Sm87TargetAotCompleteProjectionDeviceAssets::
    release_unconditionally() noexcept {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  if (arena_ != nullptr) {
    (void)cudaFree(arena_);
  }
#endif
  arena_ = nullptr;
  bytes_ = 0U;
  allocation_identity_ = 0U;
  owner_identity_ = 0U;
  device_ordinal_ = -1;
  descriptors_ = {};
  descriptor_count_ = 0U;
  prepared_resident_ = nullptr;
  prepared_model_weights_ = nullptr;
  execution_bound_ = false;
}

const Sm87TargetAotCompleteDeviceAssetDescriptor*
Sm87TargetAotCompleteProjectionDeviceAssets::find(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionRole role) const noexcept {
  const std::size_t ordinal = sm87_target_aot_complete_descriptor_ordinal(
      layer_index, role);
  if (ordinal >= descriptor_count_ || ordinal >= descriptors_.size()) {
    return nullptr;
  }
  const auto& descriptor = descriptors_[ordinal];
  const std::uint64_t expected_offset =
      sm87_target_aot_complete_expected_arena_offset(layer_index, role);
  return sm87_target_aot_complete_device_descriptor_valid(
             descriptor, layer_index, role, expected_offset,
             reinterpret_cast<std::uintptr_t>(arena_), bytes_,
             owner_identity_, allocation_identity_, device_ordinal_)
             ? &descriptor
             : nullptr;
}

Sm87TargetAotCompleteDevicePreparationStats
Sm87TargetAotCompleteProjectionDeviceAssets::prepare_online(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const Sm87TargetAotCompleteOnlinePreparationRequest& request) {
  Sm87TargetAotCompleteDevicePreparationStats result;
  if (!request.create_bundle_path.empty()) {
    result.bundle_create_rejected = true;
  }
  if (!request.load_bundle_path.empty()) {
    result.bundle_load_rejected = true;
  }
  if (!sm87_target_aot_complete_online_request_valid(request)) {
    result.hard_failure = true;
    result.message =
        "complete target-AOT v2 admits only online preparation during Engine "
        "startup; bundle create/load and request-time preparation are forbidden";
    return result;
  }
#if !defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  (void)resident;
  (void)model_weights;
  result.hard_failure = true;
  result.message =
      "this binary does not contain complete target-AOT v2 online preparation";
  return result;
#else
  if (!empty()) {
    result.hard_failure = true;
    result.message = "complete target-AOT v2 owner was not empty";
    return result;
  }
  if (model_weights.target_aot_projection_attachment_.owner != nullptr ||
      model_weights.target_aot_complete_projection_attachment_.owner !=
          nullptr) {
    result.hard_failure = true;
    result.message =
        "ModelWeights already has a mutually exclusive target-AOT attachment";
    return result;
  }
  core::Sha256Digest checkpoint_digest{};
  if (!verify_observed_checkpoint(resident, checkpoint_digest,
                                  result.message)) {
    result.hard_failure = true;
    return result;
  }
  ArtifactPlans plans{};
  if (!plan_inventory(resident, model_weights, checkpoint_digest, plans,
                      result.message)) {
    result.hard_failure = true;
    return result;
  }

  int device_ordinal = -1;
  cudaError_t status = cudaGetDevice(&device_ordinal);
  if (status != cudaSuccess || device_ordinal < 0) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaGetDevice failed before complete target-AOT prepare";
    return result;
  }
  cudaPointerAttributes resident_attributes{};
  status = cudaPointerGetAttributes(&resident_attributes,
                                    resident.arena_data());
  if (status != cudaSuccess ||
      resident_attributes.type != cudaMemoryTypeDevice ||
      resident_attributes.device != device_ordinal) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "resident checkpoint is not device memory on the current CUDA ordinal";
    return result;
  }
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  const std::uint64_t required_before =
      kSm87TargetAotCompleteProjectionDeviceArenaBytes +
      kSm87TargetAotCompleteProjectionMaximumHostStagingBytes;
  if (status != cudaSuccess || required_before > free_bytes ||
      request.minimum_free_bytes_after_prepare > free_bytes - required_before) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "insufficient memory for complete target-AOT arena, "
                           "bounded staging, and retained margin"
                         : "cudaMemGetInfo failed before complete target-AOT prepare";
    return result;
  }

  std::vector<std::uint8_t> canonical(
      static_cast<std::size_t>(
          kSm87TargetAotCompleteProjectionMaximumArtifactPayloadBytes));
  std::vector<std::uint8_t> payload(
      static_cast<std::size_t>(
          kSm87TargetAotCompleteProjectionMaximumArtifactPayloadBytes));
  result.host_staging_peak_bytes =
      kSm87TargetAotCompleteProjectionMaximumHostStagingBytes;
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      kSm87TargetAotCompleteProjectionDeviceArenaBytes > free_bytes ||
      request.minimum_free_bytes_after_prepare >
          free_bytes - kSm87TargetAotCompleteProjectionDeviceArenaBytes) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "host staging consumed complete target-AOT retained margin"
                         : "cudaMemGetInfo failed after complete target-AOT staging";
    return result;
  }

  PendingOnlinePreparation pending;
  void* allocation = nullptr;
  status = cudaMalloc(
      &allocation,
      static_cast<std::size_t>(
          kSm87TargetAotCompleteProjectionDeviceArenaBytes));
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaMalloc failed for complete target-AOT arena";
    return result;
  }
  pending.arena = static_cast<std::uint8_t*>(allocation);
  pending.bytes = kSm87TargetAotCompleteProjectionDeviceArenaBytes;
  cudaPointerAttributes allocation_attributes{};
  status = cudaPointerGetAttributes(&allocation_attributes, pending.arena);
  if (status != cudaSuccess ||
      allocation_attributes.type != cudaMemoryTypeDevice ||
      allocation_attributes.device != device_ordinal ||
      reinterpret_cast<std::uintptr_t>(pending.arena) %
              kernels::kSm87TargetAotProjectionPackedAlignment !=
          0U) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "complete target-AOT allocation identity/alignment failed";
    return result;
  }
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      static_cast<std::uint64_t>(free_bytes) <
          request.minimum_free_bytes_after_prepare) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "complete target-AOT allocation violated retained margin"
                         : "cudaMemGetInfo failed after complete target-AOT allocation";
    return result;
  }

  status = cudaStreamCreateWithFlags(&pending.stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "failed to create complete target-AOT loader stream";
    return result;
  }
  const std::uint64_t transaction_serial =
      g_complete_asset_transaction_serial.fetch_add(1U,
                                                     std::memory_order_relaxed);
  const std::uint64_t owner_identity = make_runtime_identity(
      kOwnerIdentityDomain, checkpoint_digest, transaction_serial,
      static_cast<std::uint64_t>(device_ordinal), pending.bytes);
  const std::uint64_t allocation_identity = make_runtime_identity(
      kAllocationIdentityDomain, checkpoint_digest, transaction_serial,
      owner_identity, pending.bytes,
      static_cast<std::uint64_t>(device_ordinal));
  const std::uint64_t stream_identity = make_runtime_identity(
      kStreamIdentityDomain, checkpoint_digest, transaction_serial,
      owner_identity, 1U);
  if (owner_identity == 0U || allocation_identity == 0U ||
      stream_identity == 0U) {
    result.hard_failure = true;
    result.message = "failed to derive complete target-AOT runtime identities";
    return result;
  }
  const UploadContext upload_context{
      pending.arena, pending.bytes, allocation_identity, owner_identity,
      device_ordinal, stream_identity, transaction_serial, &checkpoint_digest};
  core::Sha256 catalog;
  bool catalog_ok = hash_string(catalog, kVerifiedPayloadCatalogDomain);
  std::uint64_t verified_bytes = 0U;

  for (std::size_t index = 0U; index < plans.size(); ++index) {
    const PlannedArtifact& planned = plans[index];
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(planned.role);
    if (!layout.valid() || layout.payload_bytes > canonical.size() ||
        layout.payload_bytes > payload.size() ||
        planned.device_arena_offset != verified_bytes ||
        planned.device_arena_offset > pending.bytes ||
        layout.payload_bytes > pending.bytes - planned.device_arena_offset) {
      result.hard_failure = true;
      result.message = "complete target-AOT artifact escaped bounded storage";
      return result;
    }
    Sm87TargetAotProjectionSourceSet sources;
    if (!copy_and_bind_sources(planned, pending.stream, canonical, sources,
                               result.source_d2h_bytes, result.cuda_error,
                               result.message)) {
      result.hard_failure = true;
      return result;
    }
    const auto inspection =
        sm87_target_aot_projection_inspect_sources(sources);
    if (!inspection) {
      result.hard_failure = true;
      result.message = "complete target-AOT source inspection failed at layer " +
                       std::to_string(planned.layer_index) + ": " +
                       sm87_target_aot_projection_asset_error_string(
                           inspection.error);
      return result;
    }
    const auto build = sm87_target_aot_projection_build_asset(
        planned.artifact_identity, sources, inspection.inventory,
        {payload.data(), static_cast<std::size_t>(layout.payload_bytes)});
    if (!build ||
        sm87_target_aot_projection_validate_asset(
            build.manifest, build.transform_receipt, sources,
            inspection.inventory,
            {payload.data(), static_cast<std::size_t>(layout.payload_bytes)}) !=
            Sm87TargetAotProjectionAssetError::kSuccess) {
      result.hard_failure = true;
      result.message = "complete target-AOT host build failed at layer " +
                       std::to_string(planned.layer_index) + ": " +
                       sm87_target_aot_projection_asset_error_string(
                           build.error);
      return result;
    }
    kernels::Sm87TargetAotProjectionSha256Digest before_copy{};
    if (!sm87_target_aot_projection_sha256(
            {payload.data(), static_cast<std::size_t>(layout.payload_bytes)},
            &before_copy) ||
        before_copy != build.manifest.payload_digest) {
      result.hard_failure = true;
      result.message = "complete target-AOT payload changed before H2D";
      return result;
    }

    status = cudaEventCreateWithFlags(&pending.upload_event,
                                      cudaEventDisableTiming);
    if (status == cudaSuccess) {
      status = cudaEventCreateWithFlags(&pending.verification_event,
                                        cudaEventDisableTiming);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "failed to create complete target-AOT events";
      return result;
    }
    std::uint8_t* const destination =
        pending.arena + static_cast<std::size_t>(planned.device_arena_offset);
    status = cudaMemcpyAsync(destination, payload.data(),
                             static_cast<std::size_t>(layout.payload_bytes),
                             cudaMemcpyHostToDevice, pending.stream);
    if (status == cudaSuccess) {
      status = cudaEventRecord(pending.upload_event, pending.stream);
    }
    if (status == cudaSuccess) {
      status = cudaEventSynchronize(pending.upload_event);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "complete target-AOT H2D/event failed";
      return result;
    }
    result.payload_h2d_bytes += layout.payload_bytes;
    status = cudaMemcpyAsync(canonical.data(), destination,
                             static_cast<std::size_t>(layout.payload_bytes),
                             cudaMemcpyDeviceToHost, pending.stream);
    if (status == cudaSuccess) {
      status = cudaEventRecord(pending.verification_event, pending.stream);
    }
    if (status == cudaSuccess) {
      status = cudaEventSynchronize(pending.verification_event);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "complete target-AOT verification D2H/event failed";
      return result;
    }
    kernels::Sm87TargetAotProjectionSha256Digest readback{};
    if (!sm87_target_aot_projection_sha256(
            {canonical.data(), static_cast<std::size_t>(layout.payload_bytes)},
            &readback) ||
        readback != build.manifest.payload_digest) {
      result.hard_failure = true;
      result.message = "complete target-AOT device readback digest mismatch";
      return result;
    }
    result.verification_d2h_bytes += layout.payload_bytes;
    catalog_ok = catalog_ok && append_catalog_entry(
                                   catalog, planned, build.manifest,
                                   inspection.inventory,
                                   build.transform_receipt, readback);
    if (!catalog_ok ||
        !make_descriptor(upload_context, planned, index, build.manifest,
                         inspection.inventory, build.transform_receipt,
                         readback, pending.descriptors[index])) {
      result.hard_failure = true;
      result.message = "complete target-AOT descriptor/receipt binding failed";
      return result;
    }
    verified_bytes += layout.payload_bytes;
    const cudaError_t upload_destroy =
        cudaEventDestroy(pending.upload_event);
    pending.upload_event = nullptr;
    const cudaError_t verification_destroy =
        cudaEventDestroy(pending.verification_event);
    pending.verification_event = nullptr;
    if (upload_destroy != cudaSuccess || verification_destroy != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(
          upload_destroy != cudaSuccess ? upload_destroy
                                        : verification_destroy);
      result.message = "complete target-AOT event destruction failed";
      return result;
    }
  }
  status = cudaStreamDestroy(pending.stream);
  pending.stream = nullptr;
  if (status != cudaSuccess || !catalog_ok || verified_bytes != pending.bytes ||
      result.source_d2h_bytes !=
          kSm87TargetAotCompleteProjectionCanonicalSourceD2hBytes ||
      result.payload_h2d_bytes != pending.bytes ||
      result.verification_d2h_bytes != pending.bytes) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "complete target-AOT accounting did not close";
    return result;
  }
  result.verified_payload_catalog_sha256 = catalog.finalize().hex();
  if (result.verified_payload_catalog_sha256.size() != 64U ||
      result.verified_payload_catalog_sha256 == std::string(64U, '0')) {
    result.hard_failure = true;
    result.message = "complete target-AOT catalog digest is invalid";
    return result;
  }
  const std::uintptr_t arena_begin =
      reinterpret_cast<std::uintptr_t>(pending.arena);
  std::uint64_t expected_offset = 0U;
  for (std::size_t index = 0U; index < pending.descriptors.size(); ++index) {
    const auto& descriptor = pending.descriptors[index];
    if (descriptor.device_arena_offset != expected_offset ||
        !descriptor_valid(descriptor, index, arena_begin, pending.bytes,
                          owner_identity, allocation_identity,
                          device_ordinal)) {
      result.hard_failure = true;
      result.message = "complete target-AOT retained descriptor catalog failed";
      return result;
    }
    expected_offset += descriptor.manifest.payload_bytes;
  }
  if (expected_offset != pending.bytes) {
    result.hard_failure = true;
    result.message = "complete target-AOT retained descriptor ranges did not close";
    return result;
  }

  arena_ = pending.arena;
  bytes_ = pending.bytes;
  descriptors_ = std::move(pending.descriptors);
  descriptor_count_ = descriptors_.size();
  allocation_identity_ = allocation_identity;
  owner_identity_ = owner_identity;
  device_ordinal_ = device_ordinal;
  prepared_resident_ = &resident;
  prepared_model_weights_ = &model_weights;
  pending.arena = nullptr;
  pending.bytes = 0U;
  result.enabled = true;
  result.artifacts = descriptor_count_;
  result.sources = kSm87TargetAotCompleteProjectionDeviceSourceCount;
  result.arena_bytes = bytes_;
  result.owner_identity = owner_identity_;
  result.allocation_identity = allocation_identity_;
  result.device_ordinal = device_ordinal_;
  result.message =
      "prepared complete target-AOT v2 online owner; bundle operations remain disabled";
  return result;
#endif
}

}  // namespace q3x::runtime
