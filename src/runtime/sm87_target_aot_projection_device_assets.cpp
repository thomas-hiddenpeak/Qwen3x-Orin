#include "q3x/runtime/sm87_target_aot_projection_device_assets.h"

#include "q3x/core/sha256.h"

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
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
#include <vector>

namespace q3x::runtime {
namespace {

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)

namespace st = io::safetensors;

constexpr std::string_view kModelRepository =
    "nvidia/Qwen3.6-27B-NVFP4";
constexpr std::string_view kModelRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";
constexpr std::string_view kCheckpointIdentityDomain =
    "q3x.sm87.target-aot.device-assets.checkpoint.v1";
constexpr std::string_view kSourceIdentityDomain =
    "q3x.sm87.target-aot.device-assets.source.v1";
constexpr std::string_view kInventoryIdentityDomain =
    "q3x.sm87.target-aot.device-assets.inventory.v1";
constexpr std::string_view kArtifactIdentityDomain =
    "q3x.sm87.target-aot.device-assets.artifact.v1";
constexpr std::string_view kOwnerIdentityDomain =
    "q3x.sm87.target-aot.device-assets.owner.v1";
constexpr std::string_view kAllocationIdentityDomain =
    "q3x.sm87.target-aot.device-assets.allocation.v1";
constexpr std::string_view kStreamIdentityDomain =
    "q3x.sm87.target-aot.device-assets.stream.v1";
constexpr std::string_view kEventIdentityDomain =
    "q3x.sm87.target-aot.device-assets.event.v1";
constexpr std::string_view kVerifiedPayloadCatalogDomain =
    "q3x.sm87.target-aot.device-assets.verified-payload-catalog.v1";

std::atomic<std::uint64_t> g_device_asset_transaction_serial{1U};

struct PlannedSource final {
  kernels::Sm87TargetAotLogicalRole logical_role =
      kernels::Sm87TargetAotLogicalRole::kInvalid;
  std::string module_name;
  const NvFp4LinearWeight* projection = nullptr;
  const DeviceTensorView* weight = nullptr;
  const DeviceTensorView* block_scale = nullptr;
  const DeviceTensorView* tensor_scale = nullptr;
  std::uint64_t tensor_identity = 0U;
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
               kSm87TargetAotProjectionDeviceArtifactCount>;

template <typename Unsigned>
[[nodiscard]] bool hash_unsigned(core::Sha256& hasher,
                                 Unsigned value) noexcept {
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

[[nodiscard]] bool empty_p40_packed_artifact_view(
    const kernels::Sm87P40PackedProjectionDeviceView& view) noexcept {
  return view.payload == nullptr && view.payload_bytes == 0U &&
         view.artifact_identity == 0U &&
         view.role == kernels::Sm87P40PackedProjectionRole::kInvalid &&
         view.tactic == kernels::Sm87P40PackedTactic::kInvalid &&
         view.source_count == 0U &&
         std::all_of(view.scalar_scales.begin(), view.scalar_scales.end(),
                     [](const float scale) { return scale == 0.0F; });
}

[[nodiscard]] bool empty_nvfp4_marlin_p40_parity_view(
    const NvFp4MarlinP40ParityDeviceView& view) noexcept {
  const auto digest_empty = [](const NvFp4MarlinP40ParityDigest& digest) {
    return std::all_of(digest.begin(), digest.end(),
                       [](const std::uint8_t byte) { return byte == 0U; });
  };
  const auto source_empty = [&digest_empty](
                                const NvFp4MarlinP40ParitySourceManifest&
                                    source) {
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
  if (resident.size_bytes() >
          std::numeric_limits<std::uintptr_t>::max() - base ||
      view.arena_offset >
          std::numeric_limits<std::uintptr_t>::max() - base) {
    return false;
  }
  return reinterpret_cast<std::uintptr_t>(view.device_data) ==
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
  if (pinned.empty() || observed.size() != pinned.size()) {
    message = "resident loader did not report the complete pinned shard set";
    return false;
  }
  std::array<bool, 16U> used{};
  if (observed.size() > used.size()) {
    message = "pinned shard set exceeds the fixed provenance bound";
    return false;
  }
  core::Sha256 hasher;
  bool ok = hash_string(hasher, kCheckpointIdentityDomain) &&
            hash_string(hasher, kModelRepository) &&
            hash_string(hasher, kModelRevision) &&
            hash_unsigned(hasher,
                          static_cast<std::uint64_t>(pinned.size()));
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
      message = "resident loader shard identity differs from the pinned "
                "checkpoint at " +
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
  const std::uint64_t identity = finish_identity(hasher);
  return ok ? identity : 0U;
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
  const std::uint64_t identity = finish_identity(hasher);
  return ok ? identity : 0U;
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
                      hasher,
                      static_cast<std::uint16_t>(
                          kernels::Sm87TargetAotProjectionPackedTransformIdentity::
                              kCanonicalNkToConsumerN64K16LaneComponentV1));
  const std::uint64_t identity = finish_identity(hasher);
  return ok ? identity : 0U;
}

[[nodiscard]] std::uint64_t make_runtime_identity(
    const std::string_view domain, const core::Sha256Digest& checkpoint_digest,
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
  const std::uint64_t identity = finish_identity(hasher);
  return ok ? identity : 0U;
}

[[nodiscard]] bool plan_source(
    const ResidentWeights& resident, const NvFp4LinearWeight& projection,
    const std::string& module_name,
    const kernels::Sm87TargetAotLogicalRole logical_role,
    const std::uint64_t expected_output,
    const std::uint64_t expected_input,
    const core::Sha256Digest& checkpoint_digest, PlannedSource& source) {
  if (projection.packed_weight == nullptr ||
      projection.block_scale == nullptr ||
      projection.weight_scale_2_device == nullptr ||
      projection.output_size != expected_output ||
      projection.input_size != expected_input ||
      !std::isfinite(projection.weight_scale_2) ||
      projection.weight_scale_2 <= 0.0F ||
      reinterpret_cast<std::uintptr_t>(projection.packed_weight) % 16U !=
          0U ||
      reinterpret_cast<std::uintptr_t>(projection.block_scale) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(projection.weight_scale_2_device) %
              alignof(float) !=
          0U ||
      projection.prefill_marlin_gate_up_layout !=
          NvFp4MarlinGateUpLayout::kUnbound ||
      projection.prefill_marlin_weight != nullptr ||
      projection.prefill_marlin_scales != nullptr ||
      projection.prefill_marlin_global_scale != nullptr ||
      !empty_p40_packed_artifact_view(
          projection.prefill_p40_packed_artifact) ||
      !empty_nvfp4_marlin_p40_parity_view(
          projection.prefill_p40_vllm_marlin_parity)) {
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
  const std::uint64_t tensor_identity =
      make_source_identity(checkpoint_digest, module_name, logical_role);
  if (weight == nullptr || block_scale == nullptr || tensor_scale == nullptr ||
      tensor_identity == 0U) {
    return false;
  }
  source.logical_role = logical_role;
  source.module_name = module_name;
  source.projection = &projection;
  source.weight = weight;
  source.block_scale = block_scale;
  source.tensor_scale = tensor_scale;
  source.tensor_identity = tensor_identity;
  return true;
}

[[nodiscard]] bool plan_inventory(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const core::Sha256Digest& checkpoint_digest, ArtifactPlans& plans,
    std::string& message) {
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceSourceCount>
      source_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceArtifactCount>
      inventory_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceArtifactCount>
      artifact_identities{};
  std::size_t source_identity_count = 0U;
  std::size_t plan_count = 0U;
  std::uint64_t arena_offset = 0U;

  for (std::size_t layer_index = 0U;
       layer_index < kSm87TargetAotProjectionDeviceLayerCount;
       ++layer_index) {
    const DecoderLayerWeights& layer = model_weights.layer(layer_index);
    const auto* const gate =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.gate_proj);
    const auto* const up =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.up_proj);
    const auto* const down =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.down_proj);
    if (gate == nullptr || up == nullptr || down == nullptr) {
      message = "non-NVFP4 MLP projection at layer " +
                std::to_string(layer_index);
      return false;
    }
    const std::string prefix = "model.language_model.layers." +
                               std::to_string(layer_index) + ".mlp.";
    PlannedArtifact gate_up;
    gate_up.layer_index = layer_index;
    gate_up.role = kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp;
    gate_up.source_count = 2U;
    PlannedArtifact down_artifact;
    down_artifact.layer_index = layer_index;
    down_artifact.role = kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
    down_artifact.source_count = 1U;
    if (!plan_source(resident, *gate, prefix + "gate_proj",
                     kernels::Sm87TargetAotLogicalRole::kNvFp4Gate, 17'408U,
                     5'120U, checkpoint_digest, gate_up.sources[0U]) ||
        !plan_source(resident, *up, prefix + "up_proj",
                     kernels::Sm87TargetAotLogicalRole::kNvFp4Up, 17'408U,
                     5'120U, checkpoint_digest, gate_up.sources[1U]) ||
        !plan_source(resident, *down, prefix + "down_proj",
                     kernels::Sm87TargetAotLogicalRole::kNvFp4Down, 5'120U,
                     17'408U, checkpoint_digest,
                     down_artifact.sources[0U])) {
      message = "ineligible or non-resident target-AOT NVFP4 inventory at "
                "layer " +
                std::to_string(layer_index);
      return false;
    }

    for (PlannedArtifact* const artifact :
         std::array<PlannedArtifact*, 2U>{{&gate_up, &down_artifact}}) {
      const auto layout =
          kernels::sm87_target_aot_projection_packed_layout(artifact->role);
      if (!layout.valid() || layout.partition_count != artifact->source_count ||
          arena_offset % layout.payload_alignment != 0U ||
          layout.payload_bytes >
              kSm87TargetAotProjectionDeviceArenaBytes - arena_offset ||
          plan_count >= plans.size()) {
        message = "target-AOT layout does not fit the fixed device arena";
        return false;
      }
      artifact->device_arena_offset = arena_offset;
      for (std::size_t source_index = 0U;
           source_index < artifact->source_count; ++source_index) {
        const std::uint64_t identity =
            artifact->sources[source_index].tensor_identity;
        if (source_identity_count >= source_identities.size() ||
            identity == 0U ||
            std::find(source_identities.begin(),
                      source_identities.begin() + source_identity_count,
                      identity) !=
                source_identities.begin() + source_identity_count) {
          message = "target-AOT source identities are not globally unique";
          return false;
        }
        source_identities[source_identity_count++] = identity;
      }
      artifact->inventory_identity =
          make_inventory_identity(checkpoint_digest, *artifact);
      artifact->artifact_identity =
          make_artifact_identity(checkpoint_digest, *artifact);
      if (artifact->inventory_identity == 0U ||
          artifact->artifact_identity == 0U ||
          std::find(inventory_identities.begin(),
                    inventory_identities.begin() + plan_count,
                    artifact->inventory_identity) !=
              inventory_identities.begin() + plan_count ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + plan_count,
                    artifact->artifact_identity) !=
              artifact_identities.begin() + plan_count) {
        message = "target-AOT artifact identities are zero or duplicated";
        return false;
      }
      inventory_identities[plan_count] = artifact->inventory_identity;
      artifact_identities[plan_count] = artifact->artifact_identity;
      plans[plan_count++] = std::move(*artifact);
      arena_offset += layout.payload_bytes;
    }
  }
  if (plan_count != plans.size() ||
      source_identity_count != source_identities.size() ||
      arena_offset != kSm87TargetAotProjectionDeviceArenaBytes) {
    message = "target-AOT inventory did not close 128 artifacts, 192 "
              "sources, and the exact arena";
    return false;
  }
  return true;
}

[[nodiscard]] bool copy_and_bind_sources(
    const PlannedArtifact& planned, cudaStream_t stream,
    std::vector<std::uint8_t>& canonical,
    Sm87TargetAotProjectionSourceSet& sources,
    std::uint64_t& copied_bytes, int& cuda_error,
    std::string& message) {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(planned.role);
  if (!layout.valid() || planned.source_count != layout.partition_count ||
      canonical.size() < layout.payload_bytes) {
    message = "invalid bounded source staging contract";
    return false;
  }
  sources = {};
  sources.role = planned.role;
  sources.inventory_identity = planned.inventory_identity;
  sources.source_count = planned.source_count;
  std::array<std::uint32_t,
             kernels::kSm87TargetAotProjectionPackedMaxPartitions>
      observed_tensor_scale_bits{};
  std::uint64_t cursor = 0U;
  for (std::size_t index = 0U; index < planned.source_count; ++index) {
    const PlannedSource& planned_source = planned.sources[index];
    const auto& partition = layout.partitions[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t weight_bytes = values / 2U;
    const std::uint64_t scale_bytes = values / 16U;
    if (planned_source.weight == nullptr ||
        planned_source.block_scale == nullptr ||
        planned_source.tensor_scale == nullptr ||
        planned_source.projection == nullptr ||
        weight_bytes > layout.payload_bytes - cursor) {
      message = "planned target-AOT source is incomplete";
      return false;
    }
    std::uint8_t* const weight_host =
        canonical.data() + static_cast<std::size_t>(cursor);
    cudaError_t status = cudaMemcpyAsync(
        weight_host, planned_source.weight->device_data,
        static_cast<std::size_t>(weight_bytes), cudaMemcpyDeviceToHost,
        stream);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      message = "D2H failed for " + planned_source.module_name + ".weight";
      return false;
    }
    cursor += weight_bytes;
    if (scale_bytes > layout.payload_bytes - cursor) {
      message = "planned target-AOT scale staging overflow";
      return false;
    }
    std::uint8_t* const scale_host =
        canonical.data() + static_cast<std::size_t>(cursor);
    status = cudaMemcpyAsync(
        scale_host, planned_source.block_scale->device_data,
        static_cast<std::size_t>(scale_bytes), cudaMemcpyDeviceToHost,
        stream);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      message = "D2H failed for " + planned_source.module_name +
                ".weight_scale";
      return false;
    }
    cursor += scale_bytes;
    status = cudaMemcpyAsync(&observed_tensor_scale_bits[index],
                             planned_source.tensor_scale->device_data,
                             sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                             stream);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      message = "D2H failed for scalar scales at " +
                planned_source.module_name;
      return false;
    }
    auto& source = sources.sources[index];
    source.logical_role = planned_source.logical_role;
    source.tensor_identity = planned_source.tensor_identity;
    source.output_features = partition.output_features;
    source.input_features = partition.input_features;
    source.packed_weight = {weight_host,
                            static_cast<std::size_t>(weight_bytes)};
    source.block_scale = {scale_host,
                          static_cast<std::size_t>(scale_bytes)};
  }
  cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    cuda_error = static_cast<int>(status);
    message = "source D2H stream synchronization failed";
    return false;
  }
  for (std::size_t index = 0U; index < planned.source_count; ++index) {
    const NvFp4LinearWeight& projection =
        *planned.sources[index].projection;
    const std::uint32_t expected_tensor = float_bits(projection.weight_scale_2);
    if (observed_tensor_scale_bits[index] != expected_tensor ||
        !kernels::sm87_target_aot_projection_scale_bits_valid(
            observed_tensor_scale_bits[index])) {
      message = "device scalar scale bits differ from ModelWeights at " +
                planned.sources[index].module_name;
      return false;
    }
    sources.sources[index].tensor_scale_bits =
        observed_tensor_scale_bits[index];
  }
  if (cursor != layout.payload_bytes) {
    message = "canonical source bytes did not equal the artifact payload";
    return false;
  }
  copied_bytes +=
      cursor + sizeof(std::uint32_t) * planned.source_count;
  return true;
}

void cleanup_stream(cudaStream_t& stream, cudaEvent_t& upload_event,
                    cudaEvent_t& verification_event) noexcept {
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
}

[[nodiscard]] Sm87TargetAotProjectionDevicePreparationStats prepare_impl(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    Sm87TargetAotProjectionDeviceAssets& owner,
    std::uint8_t*& owner_arena, std::uint64_t& owner_bytes,
    std::uint64_t& allocation_identity, std::uint64_t& owner_identity,
    std::int32_t& owner_device_ordinal,
    std::array<Sm87TargetAotProjectionDeviceAssetDescriptor,
               kSm87TargetAotProjectionDeviceArtifactCount>& descriptors,
    std::size_t& descriptor_count) {
  Sm87TargetAotProjectionDevicePreparationStats result;
  if (!owner.empty()) {
    result.hard_failure = true;
    result.message = "target-AOT device asset owner was not empty";
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
    result.message = "cudaGetDevice failed before target-AOT preparation";
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
    result.message = "resident checkpoint arena is not device memory on the "
                     "current CUDA ordinal";
    return result;
  }
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  const std::uint64_t required_before =
      kSm87TargetAotProjectionDeviceArenaBytes +
      kSm87TargetAotProjectionMaximumHostStagingBytes;
  if (status != cudaSuccess || required_before > free_bytes ||
      minimum_free_bytes_after_prepare > free_bytes - required_before) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "insufficient memory for the target-AOT arena, "
                           "bounded dual staging, and retained margin"
                         : "cudaMemGetInfo failed before target-AOT staging";
    return result;
  }

  std::vector<std::uint8_t> canonical(
      static_cast<std::size_t>(
          kSm87TargetAotProjectionMaximumArtifactPayloadBytes));
  std::vector<std::uint8_t> payload(
      static_cast<std::size_t>(
          kSm87TargetAotProjectionMaximumArtifactPayloadBytes));
  result.host_staging_peak_bytes =
      kSm87TargetAotProjectionMaximumHostStagingBytes;
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      kSm87TargetAotProjectionDeviceArenaBytes > free_bytes ||
      minimum_free_bytes_after_prepare >
          free_bytes - kSm87TargetAotProjectionDeviceArenaBytes) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "bounded host staging consumed the target-AOT "
                           "post-allocation margin"
                         : "cudaMemGetInfo failed after host staging";
    return result;
  }

  void* allocation = nullptr;
  status = cudaMalloc(
      &allocation,
      static_cast<std::size_t>(kSm87TargetAotProjectionDeviceArenaBytes));
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaMalloc failed for the exact target-AOT arena";
    return result;
  }
  owner_arena = static_cast<std::uint8_t*>(allocation);
  owner_bytes = kSm87TargetAotProjectionDeviceArenaBytes;
  owner_device_ordinal = device_ordinal;
  cudaPointerAttributes allocation_attributes{};
  status = cudaPointerGetAttributes(&allocation_attributes, owner_arena);
  if (status != cudaSuccess ||
      allocation_attributes.type != cudaMemoryTypeDevice ||
      allocation_attributes.device != device_ordinal) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "target-AOT allocation is not device memory on the "
                     "current CUDA ordinal";
    owner.release();
    return result;
  }
  const std::uint64_t transaction_serial =
      g_device_asset_transaction_serial.fetch_add(1U,
                                                   std::memory_order_relaxed);
  owner_identity = make_runtime_identity(
      kOwnerIdentityDomain, checkpoint_digest, transaction_serial,
      static_cast<std::uint64_t>(device_ordinal), owner_bytes);
  allocation_identity = make_runtime_identity(
      kAllocationIdentityDomain, checkpoint_digest, transaction_serial,
      owner_identity, owner_bytes,
      static_cast<std::uint64_t>(device_ordinal));
  if (reinterpret_cast<std::uintptr_t>(owner_arena) %
              kernels::kSm87TargetAotProjectionPackedAlignment !=
          0U ||
      owner_identity == 0U || allocation_identity == 0U) {
    result.hard_failure = true;
    result.message = "target-AOT allocation alignment or identity failed";
    owner.release();
    return result;
  }
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      static_cast<std::uint64_t>(free_bytes) <
          minimum_free_bytes_after_prepare) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "target-AOT allocation violated retained margin"
                         : "cudaMemGetInfo failed after target-AOT allocation";
    owner.release();
    return result;
  }

  cudaStream_t stream = nullptr;
  status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "failed to create target-AOT loader stream";
    owner.release();
    return result;
  }
  cudaEvent_t upload_event = nullptr;
  cudaEvent_t verification_event = nullptr;
  core::Sha256 verified_payload_catalog_hasher;
  bool verified_payload_catalog_hash_ok =
      hash_string(verified_payload_catalog_hasher,
                  kVerifiedPayloadCatalogDomain);
  std::uint64_t verified_payload_catalog_bytes = 0U;
  const std::uint64_t stream_identity = make_runtime_identity(
      kStreamIdentityDomain, checkpoint_digest, transaction_serial,
      owner_identity, 1U);
  if (stream_identity == 0U) {
    result.hard_failure = true;
    result.message = "failed to derive target-AOT stream identity";
    cleanup_stream(stream, upload_event, verification_event);
    owner.release();
    return result;
  }

  for (std::size_t index = 0U; index < plans.size(); ++index) {
    const PlannedArtifact& planned = plans[index];
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(planned.role);
    if (!layout.valid() || layout.payload_bytes > canonical.size() ||
        layout.payload_bytes > payload.size() ||
        planned.device_arena_offset > owner_bytes ||
        layout.payload_bytes > owner_bytes - planned.device_arena_offset) {
      result.hard_failure = true;
      result.message = "target-AOT artifact escaped bounded storage";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }

    Sm87TargetAotProjectionSourceSet sources;
    if (!copy_and_bind_sources(planned, stream, canonical, sources,
                               result.source_d2h_bytes, result.cuda_error,
                               result.message)) {
      result.hard_failure = true;
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    const auto inspection =
        sm87_target_aot_projection_inspect_sources(sources);
    if (!inspection) {
      result.hard_failure = true;
      result.message = "target-AOT source inspection failed at layer " +
                       std::to_string(planned.layer_index) + ": " +
                       sm87_target_aot_projection_asset_error_string(
                           inspection.error);
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
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
      result.message = "target-AOT host build/validation failed at layer " +
                       std::to_string(planned.layer_index) + ": " +
                       sm87_target_aot_projection_asset_error_string(
                           build.error);
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    kernels::Sm87TargetAotProjectionSha256Digest before_copy_digest{};
    if (!sm87_target_aot_projection_sha256(
            {payload.data(), static_cast<std::size_t>(layout.payload_bytes)},
            &before_copy_digest) ||
        before_copy_digest != build.manifest.payload_digest) {
      result.hard_failure = true;
      result.message = "target-AOT payload changed before H2D";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }

    status = cudaEventCreateWithFlags(&upload_event, cudaEventDisableTiming);
    if (status == cudaSuccess) {
      status = cudaEventCreateWithFlags(&verification_event,
                                        cudaEventDisableTiming);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "failed to create target-AOT completion events";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    std::uint8_t* const destination =
        owner_arena + static_cast<std::size_t>(planned.device_arena_offset);
    status = cudaMemcpyAsync(destination, payload.data(),
                             static_cast<std::size_t>(layout.payload_bytes),
                             cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) {
      status = cudaEventRecord(upload_event, stream);
    }
    if (status == cudaSuccess) {
      status = cudaEventSynchronize(upload_event);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "target-AOT H2D or upload event failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    result.payload_h2d_bytes += layout.payload_bytes;

    status = cudaMemcpyAsync(canonical.data(), destination,
                             static_cast<std::size_t>(layout.payload_bytes),
                             cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) {
      status = cudaEventRecord(verification_event, stream);
    }
    if (status == cudaSuccess) {
      status = cudaEventSynchronize(verification_event);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "target-AOT verification D2H or event failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    kernels::Sm87TargetAotProjectionSha256Digest readback_digest{};
    if (!sm87_target_aot_projection_sha256(
            {canonical.data(),
             static_cast<std::size_t>(layout.payload_bytes)},
            &readback_digest) ||
        readback_digest != build.manifest.payload_digest) {
      result.hard_failure = true;
      result.message = "target-AOT device readback SHA-256 mismatch";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    if (planned.device_arena_offset != verified_payload_catalog_bytes) {
      result.hard_failure = true;
      result.message =
          "target-AOT verified-payload catalog order failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    verified_payload_catalog_hash_ok =
        verified_payload_catalog_hash_ok &&
        hash_unsigned(verified_payload_catalog_hasher,
                      static_cast<std::uint64_t>(planned.layer_index)) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      static_cast<std::uint8_t>(planned.role)) &&
        verified_payload_catalog_hasher.update(
            build.manifest.magic.data(), build.manifest.magic.size()) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      build.manifest.abi_major) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      build.manifest.abi_minor) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      build.manifest.header_bytes) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      planned.device_arena_offset) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      build.manifest.artifact_identity) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      inspection.inventory.identity) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      static_cast<std::uint16_t>(
                          build.manifest.plan_identity)) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      static_cast<std::uint16_t>(
                          build.manifest.layout_identity)) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      static_cast<std::uint8_t>(
                          build.manifest.encoding)) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      static_cast<std::uint16_t>(
                          build.transform_receipt.transform_identity)) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      build.manifest.payload_offset) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      build.manifest.payload_bytes) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      build.manifest.artifact_bytes) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      build.manifest.payload_alignment) &&
        verified_payload_catalog_hasher.update(
            build.manifest.payload_digest.bytes.data(),
            build.manifest.payload_digest.bytes.size()) &&
        verified_payload_catalog_hasher.update(
            readback_digest.bytes.data(), readback_digest.bytes.size()) &&
        hash_unsigned(
            verified_payload_catalog_hasher,
            static_cast<std::uint8_t>(
                build.manifest.token_count_independent)) &&
        hash_unsigned(
            verified_payload_catalog_hasher,
            static_cast<std::uint8_t>(
                build.manifest.cuda_implementation_present)) &&
        hash_unsigned(
            verified_payload_catalog_hasher,
            static_cast<std::uint8_t>(
                build.manifest.static_resources_qualified)) &&
        hash_unsigned(
            verified_payload_catalog_hasher,
            static_cast<std::uint8_t>(
                build.manifest.numerical_contract_qualified)) &&
        hash_unsigned(
            verified_payload_catalog_hasher,
            static_cast<std::uint8_t>(
                build.manifest.production_dispatch_eligible)) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      build.manifest.seal.value) &&
        hash_unsigned(verified_payload_catalog_hasher,
                      inspection.inventory.source_count);
    for (std::size_t source_index = 0U;
         source_index < inspection.inventory.source_count; ++source_index) {
      const auto& source = inspection.inventory.sources[source_index];
      verified_payload_catalog_hash_ok =
          verified_payload_catalog_hash_ok &&
          hash_unsigned(verified_payload_catalog_hasher,
                        static_cast<std::uint8_t>(source.logical_role)) &&
          hash_unsigned(verified_payload_catalog_hasher,
                        source.partition_index) &&
          hash_unsigned(verified_payload_catalog_hasher,
                        source.tensor_identity) &&
          verified_payload_catalog_hasher.update(
              source.weight_digest.bytes.data(),
              source.weight_digest.bytes.size()) &&
          verified_payload_catalog_hasher.update(
              source.scale_digest.bytes.data(),
              source.scale_digest.bytes.size()) &&
          hash_unsigned(verified_payload_catalog_hasher,
                        source.output_features) &&
          hash_unsigned(verified_payload_catalog_hasher,
                        source.input_features) &&
          hash_unsigned(verified_payload_catalog_hasher,
                        source.tensor_scale_bits) &&
          hash_unsigned(verified_payload_catalog_hasher,
                        source.payload_offset) &&
          hash_unsigned(verified_payload_catalog_hasher,
                        source.payload_bytes);
    }
    if (!verified_payload_catalog_hash_ok) {
      result.hard_failure = true;
      result.message = "target-AOT verified-payload catalog hashing failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    verified_payload_catalog_bytes += layout.payload_bytes;
    result.verification_d2h_bytes += layout.payload_bytes;

    const std::uint64_t upload_event_identity = make_runtime_identity(
        kEventIdentityDomain, checkpoint_digest, transaction_serial,
        planned.artifact_identity, index, 1U);
    const std::uint64_t verification_event_identity = make_runtime_identity(
        kEventIdentityDomain, checkpoint_digest, transaction_serial,
        planned.artifact_identity, index, 2U);
    kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt upload;
    upload.artifact_identity = build.manifest.artifact_identity;
    upload.source_inventory_identity = inspection.inventory.identity;
    upload.role = build.manifest.role;
    upload.plan_identity = build.manifest.plan_identity;
    upload.layout_identity = build.manifest.layout_identity;
    upload.transform_identity = build.transform_receipt.transform_identity;
    upload.host_payload_offset = build.manifest.payload_offset;
    upload.host_payload_bytes = build.manifest.payload_bytes;
    upload.host_payload_digest = build.manifest.payload_digest;
    upload.host_manifest_seal = build.manifest.seal;
    upload.tensor_scale_count = build.manifest.source_count;
    for (std::size_t source_index = 0U;
         source_index < build.manifest.source_count; ++source_index) {
      upload.tensor_scale_bits[source_index] =
          build.manifest.sources[source_index].tensor_scale_bits;
    }
    upload.device_allocation_identity = allocation_identity;
    upload.device_allocation_owner_identity = owner_identity;
    upload.device_ordinal = owner_device_ordinal;
    upload.device_allocation_begin =
        reinterpret_cast<std::uintptr_t>(owner_arena);
    upload.device_allocation_bytes = owner_bytes;
    upload.device_allocation_end =
        upload.device_allocation_begin + static_cast<std::uintptr_t>(owner_bytes);
    upload.device_payload_begin = reinterpret_cast<std::uintptr_t>(destination);
    upload.device_payload_bytes = layout.payload_bytes;
    upload.device_payload_end =
        upload.device_payload_begin +
        static_cast<std::uintptr_t>(layout.payload_bytes);
    upload.upload_stream_owner_identity = owner_identity;
    upload.upload_stream_identity = stream_identity;
    upload.upload_completion_event_identity = upload_event_identity;
    upload.verification_stream_owner_identity = owner_identity;
    upload.verification_stream_identity = stream_identity;
    upload.verification_completion_event_identity =
        verification_event_identity;
    upload.verification_readback_bytes = layout.payload_bytes;
    upload.verification_readback_digest = readback_digest;
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
    upload.receipt_identity =
        kernels::sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(
            upload);
    const auto view = kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
        build.manifest, inspection.inventory, build.transform_receipt, upload);
    if (upload.receipt_identity == 0U || !view.valid ||
        !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(view)) {
      result.hard_failure = true;
      result.message = "target-AOT receipt or bound device view failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }

    Sm87TargetAotProjectionDeviceAssetDescriptor descriptor;
    descriptor.layer_index = planned.layer_index;
    descriptor.role = planned.role;
    descriptor.device_arena_offset = planned.device_arena_offset;
    descriptor.source_inventory = inspection.inventory;
    descriptor.manifest = build.manifest;
    descriptor.transform_receipt = build.transform_receipt;
    descriptor.upload_receipt = upload;
    descriptor.view = view;
    descriptors[index] = descriptor;

    const cudaError_t upload_destroy = cudaEventDestroy(upload_event);
    upload_event = nullptr;
    const cudaError_t verification_destroy =
        cudaEventDestroy(verification_event);
    verification_event = nullptr;
    if (upload_destroy != cudaSuccess ||
        verification_destroy != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(
          upload_destroy != cudaSuccess ? upload_destroy
                                        : verification_destroy);
      result.message = "target-AOT completion event destruction failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
  }

  status = cudaStreamDestroy(stream);
  stream = nullptr;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "target-AOT loader stream destruction failed";
    owner.release();
    return result;
  }
  if (verified_payload_catalog_bytes != owner_bytes ||
      !verified_payload_catalog_hash_ok) {
    result.hard_failure = true;
    result.message =
        "target-AOT verified-payload catalog did not close the exact arena";
    owner.release();
    return result;
  }
  result.verified_payload_catalog_sha256 =
      verified_payload_catalog_hasher.finalize().hex();
  if (result.verified_payload_catalog_sha256.size() != 64U ||
      result.verified_payload_catalog_sha256 == std::string(64U, '0')) {
    result.hard_failure = true;
    result.message = "target-AOT verified-payload catalog digest was invalid";
    owner.release();
    return result;
  }
  descriptor_count = descriptors.size();
  for (std::size_t layer = 0U;
       layer < kSm87TargetAotProjectionDeviceLayerCount; ++layer) {
    if (!owner.has_asset(
            layer,
            kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp) ||
        !owner.has_asset(
            layer, kernels::Sm87TargetAotProjectionRole::kNvFp4Down)) {
      result.hard_failure = true;
      result.message = "retained target-AOT inventory is incomplete";
      owner.release();
      return result;
    }
  }
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      static_cast<std::uint64_t>(free_bytes) <
          minimum_free_bytes_after_prepare) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "completed target-AOT preparation violated the "
                           "retained memory margin"
                         : "cudaMemGetInfo failed after target-AOT "
                           "preparation";
    owner.release();
    return result;
  }
  result.enabled = true;
  result.artifacts = descriptor_count;
  result.sources = kSm87TargetAotProjectionDeviceSourceCount;
  result.arena_bytes = owner_bytes;
  result.owner_identity = owner_identity;
  result.allocation_identity = allocation_identity;
  result.device_ordinal = owner_device_ordinal;
  return result;
}

#endif  // Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION

}  // namespace

Sm87TargetAotProjectionDeviceAssets::~Sm87TargetAotProjectionDeviceAssets() {
  release_unconditionally();
}

bool Sm87TargetAotProjectionDeviceAssets::release() noexcept {
  if (attached_model_weights_ != nullptr) {
    return false;
  }
  release_unconditionally();
  return true;
}

void Sm87TargetAotProjectionDeviceAssets::release_unconditionally() noexcept {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
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
  prepared_model_weights_ = nullptr;
  attached_model_weights_ = nullptr;
}

const Sm87TargetAotProjectionDeviceAssetDescriptor*
Sm87TargetAotProjectionDeviceAssets::find(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionRole role) const noexcept {
  if (layer_index >= kSm87TargetAotProjectionDeviceLayerCount ||
      (role != kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp &&
       role != kernels::Sm87TargetAotProjectionRole::kNvFp4Down)) {
    return nullptr;
  }
  for (std::size_t index = 0U; index < descriptor_count_; ++index) {
    if (descriptors_[index].layer_index == layer_index &&
        descriptors_[index].role == role) {
      return &descriptors_[index];
    }
  }
  return nullptr;
}

Sm87TargetAotProjectionDevicePreparationStats
Sm87TargetAotProjectionDeviceAssets::prepare(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare) {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  try {
    Sm87TargetAotProjectionDevicePreparationStats result = prepare_impl(
        resident, model_weights, minimum_free_bytes_after_prepare, *this,
        arena_, bytes_, allocation_identity_, owner_identity_, device_ordinal_,
        descriptors_, descriptor_count_);
    if (result.enabled && !result.hard_failure) {
      prepared_model_weights_ = &model_weights;
    } else {
      (void)release();
    }
    return result;
  } catch (const std::exception& error) {
    (void)release();
    Sm87TargetAotProjectionDevicePreparationStats result;
    result.hard_failure = true;
    result.message =
        std::string("exception during target-AOT device preparation: ") +
        error.what();
    return result;
  } catch (...) {
    (void)release();
    Sm87TargetAotProjectionDevicePreparationStats result;
    result.hard_failure = true;
    result.message =
        "unknown exception during target-AOT device preparation";
    return result;
  }
#else
  (void)resident;
  (void)model_weights;
  (void)minimum_free_bytes_after_prepare;
  Sm87TargetAotProjectionDevicePreparationStats result;
  result.hard_failure = true;
  result.message = "target-AOT device asset preparation is not compiled";
  return result;
#endif
}

}  // namespace q3x::runtime
