#pragma once

#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"
#include "../../src/runtime/sm87_target_aot_projection_complete_execution_access_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
namespace q3x::runtime::target_aot_complete_execution_detail {
namespace {

using Role = kernels::Sm87TargetAotProjectionRole;
using Descriptor = Sm87TargetAotCompleteDeviceAssetDescriptor;
using Owner = Sm87TargetAotCompleteProjectionDeviceAssets;

inline constexpr std::uintptr_t kHostTestArenaBegin =
    0x0000'0030'0000'0000ULL;
inline constexpr std::uintptr_t kHostTestResidentAddress =
    0x0000'0020'0000'0000ULL;
inline constexpr std::uint64_t kHostTestOwnerIdentity =
    0x7133'7843'4f4d'504fULL;
inline constexpr std::uint64_t kHostTestAllocationIdentity =
    0x7133'7843'4f4d'5041ULL;
inline constexpr std::uint64_t kHostTestDeviceIdentity =
    0x7133'7843'5058'4f52ULL;

[[nodiscard]] kernels::Sm87TargetAotProjectionSha256Digest host_test_digest(
    const std::uint64_t seed) noexcept {
  kernels::Sm87TargetAotProjectionSha256Digest digest;
  std::uint64_t state = seed | 1U;
  for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    digest.bytes[index] = static_cast<std::uint8_t>(state >> 24U);
  }
  if (kernels::sm87_target_aot_projection_digest_is_zero(digest)) {
    digest.bytes[0U] = 1U;
  }
  return digest;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedTransformReceipt
make_host_test_transform_receipt(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest) noexcept {
  kernels::Sm87TargetAotProjectionPackedTransformReceipt receipt;
  receipt.artifact_identity = manifest.artifact_identity;
  receipt.source_inventory_identity = inventory.identity;
  receipt.role = layout.role;
  receipt.plan_identity = layout.plan_identity;
  receipt.layout_identity = layout.layout_identity;
  receipt.encoding = layout.encoding;
  receipt.transform_identity = kernels::
      Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.partition_count = layout.partition_count;
  receipt.deterministic_transform = true;
  receipt.no_arithmetic_conversion = true;
  receipt.no_request_time_repacking = true;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const auto& partition = layout.partitions[index];
    const auto& source = inventory.sources[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t weight_bytes =
        values * partition.weight_bits / 8U;
    const std::uint64_t scale_values =
        partition.block_scale_group_k == 0U
            ? 0U
            : values / partition.block_scale_group_k;
    receipt.partitions[index] = {
        partition.logical_role,
        static_cast<std::uint32_t>(index),
        source.tensor_identity,
        source.weight_digest,
        source.scale_digest,
        weight_bytes,
        scale_values + sizeof(std::uint32_t),
        values,
        scale_values,
        scale_values,
        scale_values,
        0U,
        0U,
        partition.payload_offset,
        partition.payload_bytes,
        true,
        true,
        true,
        scale_values != 0U,
        true,
    };
  }
  receipt.payload = {manifest.artifact_identity,
                     manifest.payload_offset,
                     manifest.payload_bytes,
                     manifest.payload_digest,
                     true};
  return receipt;
}

template <typename Receipt>
void fill_common_host_test_upload(
    const std::size_t ordinal, const std::uint64_t arena_offset,
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt& transform,
    Receipt& upload) noexcept {
  upload.artifact_identity = manifest.artifact_identity;
  upload.source_inventory_identity = inventory.identity;
  upload.role = layout.role;
  upload.plan_identity = layout.plan_identity;
  upload.layout_identity = layout.layout_identity;
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
  upload.device_allocation_identity = kHostTestAllocationIdentity;
  upload.device_allocation_owner_identity = kHostTestOwnerIdentity;
  upload.device_ordinal = 0;
  upload.device_allocation_begin = kHostTestArenaBegin;
  upload.device_allocation_end =
      kHostTestArenaBegin + static_cast<std::uintptr_t>(
                                kSm87TargetAotCompleteProjectionDeviceArenaBytes);
  upload.device_allocation_bytes =
      kSm87TargetAotCompleteProjectionDeviceArenaBytes;
  upload.device_payload_begin =
      kHostTestArenaBegin + static_cast<std::uintptr_t>(arena_offset);
  upload.device_payload_end =
      upload.device_payload_begin +
      static_cast<std::uintptr_t>(manifest.payload_bytes);
  upload.device_payload_bytes = manifest.payload_bytes;
  upload.upload_stream_owner_identity = kHostTestOwnerIdentity;
  upload.upload_stream_identity = 0x4100'0000'0000'0000ULL + ordinal + 1U;
  upload.upload_completion_event_identity =
      0x4200'0000'0000'0000ULL + ordinal + 1U;
  upload.verification_stream_owner_identity = kHostTestOwnerIdentity;
  upload.verification_stream_identity =
      0x4300'0000'0000'0000ULL + ordinal + 1U;
  upload.verification_completion_event_identity =
      0x4400'0000'0000'0000ULL + ordinal + 1U;
  upload.verification_readback_bytes = manifest.payload_bytes;
  upload.verification_readback_digest = manifest.payload_digest;
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
}

[[nodiscard]] bool make_host_test_descriptor(
    const std::size_t layer_index, const Role role,
    const std::uint64_t arena_offset, Descriptor& descriptor) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(role);
  const std::size_t ordinal =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (!layout.valid() ||
      ordinal >= kSm87TargetAotCompleteProjectionDeviceArtifactCount ||
      arena_offset !=
          sm87_target_aot_complete_expected_arena_offset(layer_index, role)) {
    return false;
  }

  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = 0x1100'0000'0000'0000ULL + ordinal + 1U;
  inventory.role = role;
  inventory.source_count = layout.partition_count;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const std::uint64_t source_ordinal = 3U * ordinal + index + 1U;
    inventory.sources[index] =
        kernels::sm87_target_aot_projection_packed_source_binding(
            layout, index, 0x2100'0000'0000'0000ULL + source_ordinal,
            host_test_digest(0x1000U + 2U * source_ordinal),
            host_test_digest(0x1001U + 2U * source_ordinal),
            0x3f80'0000U);
  }
  if (!inventory.valid(layout)) {
    return false;
  }

  const auto manifest =
      kernels::sm87_target_aot_projection_make_packed_manifest(
          role, 0x3100'0000'0000'0000ULL + ordinal + 1U, inventory,
          host_test_digest(0x8000U + ordinal));
  const auto transform =
      make_host_test_transform_receipt(layout, inventory, manifest);
  if (!kernels::sm87_target_aot_projection_validate_packed_manifest(
          manifest, inventory) ||
      !kernels::sm87_target_aot_projection_validate_transform_receipt(
          manifest, inventory, transform)) {
    return false;
  }

  descriptor = {};
  descriptor.layer_index = layer_index;
  descriptor.role = role;
  descriptor.device_arena_offset = arena_offset;
  descriptor.source_inventory = inventory;
  descriptor.manifest = manifest;
  descriptor.transform_receipt = transform;
  descriptor.encoding = layout.encoding;
  if (sm87_target_aot_complete_role_is_nvfp4(role)) {
    auto& upload = descriptor.nvfp4_upload_receipt;
    fill_common_host_test_upload(ordinal, arena_offset, layout, manifest,
                                 inventory, transform, upload);
    upload.receipt_identity =
        kernels::sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(
            upload);
    descriptor.nvfp4_view = kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
        manifest, inventory, transform, upload);
  } else if (sm87_target_aot_complete_role_is_fp8(role)) {
    auto& upload = descriptor.fp8_upload_receipt;
    fill_common_host_test_upload(ordinal, arena_offset, layout, manifest,
                                 inventory, transform, upload);
    for (std::size_t index = 0U; index < upload.tensor_scale_count; ++index) {
      upload.compensated_tensor_scale_bf16_bits[index] =
          kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
              upload.tensor_scale_bits[index]);
    }
    upload.receipt_identity =
        kernels::sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(
            upload);
    descriptor.fp8_view = kernels::sm87_target_aot_bind_fp8_cuda_asset(
        manifest, inventory, transform, upload);
  } else {
    return false;
  }
  return sm87_target_aot_complete_device_descriptor_valid(
      descriptor, layer_index, role, arena_offset, kHostTestArenaBegin,
      kSm87TargetAotCompleteProjectionDeviceArenaBytes,
      kHostTestOwnerIdentity, kHostTestAllocationIdentity, 0);
}

}  // namespace

std::optional<ModelWeights>
Sm87TargetAotCompleteProjectionExecutionAccess::
    make_complete_host_test_fixture(Owner& owner) noexcept {
  if (!owner.empty()) {
    return std::nullopt;
  }
  ModelWeights model_weights;
  owner.arena_ = reinterpret_cast<std::uint8_t*>(kHostTestArenaBegin);
  owner.bytes_ = kSm87TargetAotCompleteProjectionDeviceArenaBytes;
  owner.allocation_identity_ = kHostTestAllocationIdentity;
  owner.owner_identity_ = kHostTestOwnerIdentity;
  owner.device_identity_ = kHostTestDeviceIdentity;
  owner.device_ordinal_ = 0;
  std::uint64_t offset = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87TargetAotCompleteProjectionDeviceLayerCount;
       ++layer_index) {
    const std::array<Role, 4U> roles{{
        Role::kNvFp4GateUp,
        Role::kNvFp4Down,
        sm87_target_aot_complete_is_full_layer(layer_index)
            ? Role::kFp8FullQkv
            : Role::kFp8GdnQkvZ,
        Role::kFp8AttentionOutput}};
    for (const Role role : roles) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index >= owner.descriptors_.size() ||
          index != owner.descriptor_count_ ||
          !make_host_test_descriptor(layer_index, role, offset,
                                     owner.descriptors_[index])) {
        owner.arena_ = nullptr;
        owner.bytes_ = 0U;
        owner.allocation_identity_ = 0U;
        owner.owner_identity_ = 0U;
        owner.device_identity_ = 0U;
        owner.device_ordinal_ = -1;
        owner.descriptors_ = {};
        owner.descriptor_count_ = 0U;
        return std::nullopt;
      }
      offset += owner.descriptors_[index].manifest.payload_bytes;
      ++owner.descriptor_count_;
    }
  }
  if (offset != owner.bytes_ ||
      owner.descriptor_count_ != owner.descriptors_.size()) {
    owner.arena_ = nullptr;
    owner.bytes_ = 0U;
    owner.allocation_identity_ = 0U;
    owner.owner_identity_ = 0U;
    owner.device_identity_ = 0U;
    owner.device_ordinal_ = -1;
    owner.descriptors_ = {};
    owner.descriptor_count_ = 0U;
    return std::nullopt;
  }
  owner.prepared_resident_ =
      reinterpret_cast<const ResidentWeights*>(kHostTestResidentAddress);
  owner.prepared_model_weights_ = &model_weights;
  owner.execution_bound_ = true;
  model_weights.target_aot_complete_projection_attachment_.owner = &owner;
  model_weights.target_aot_complete_projection_attachment_.owner_identity =
      owner.owner_identity_;
  model_weights.target_aot_complete_projection_attachment_
      .allocation_identity = owner.allocation_identity_;
  model_weights.target_aot_complete_projection_attachment_.device_identity =
      owner.device_identity_;
  model_weights.target_aot_complete_projection_attachment_.arena_begin =
      kHostTestArenaBegin;
  model_weights.target_aot_complete_projection_attachment_.arena_bytes =
      owner.bytes_;
  model_weights.target_aot_complete_projection_attachment_.device_ordinal =
      owner.device_ordinal_;
  model_weights.target_aot_complete_projection_attachment_.artifact_count =
      owner.descriptor_count_;
  return std::optional<ModelWeights>(std::in_place,
                                     std::move(model_weights));
}

bool Sm87TargetAotCompleteProjectionExecutionAccess::
    poison_host_test_fixture_receipt(
        Owner& owner, const std::size_t layer_index,
        const Role role) noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (owner.arena_ != reinterpret_cast<std::uint8_t*>(kHostTestArenaBegin) ||
      owner.owner_identity_ != kHostTestOwnerIdentity ||
      owner.allocation_identity_ != kHostTestAllocationIdentity ||
      !owner.execution_bound_ || index >= owner.descriptor_count_) {
    return false;
  }
  auto& descriptor = owner.descriptors_[index];
  if (sm87_target_aot_complete_role_is_nvfp4(role)) {
    descriptor.nvfp4_upload_receipt.verification_event_observed = false;
  } else if (sm87_target_aot_complete_role_is_fp8(role)) {
    descriptor.fp8_upload_receipt.verification_event_observed = false;
  } else {
    return false;
  }
  return true;
}

bool Sm87TargetAotCompleteProjectionExecutionAccess::
    tamper_host_test_fixture_source_identity(
        Owner& owner, const std::size_t layer_index, const Role role,
        const std::size_t source_index) noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (owner.arena_ != reinterpret_cast<std::uint8_t*>(kHostTestArenaBegin) ||
      owner.owner_identity_ != kHostTestOwnerIdentity ||
      owner.allocation_identity_ != kHostTestAllocationIdentity ||
      !owner.execution_bound_ || index >= owner.descriptor_count_) {
    return false;
  }
  auto& inventory = owner.descriptors_[index].source_inventory;
  if (source_index >= inventory.source_count) {
    return false;
  }
  inventory.sources[source_index].tensor_identity ^=
      0x0100'0000'0000'0000ULL;
  return inventory.sources[source_index].tensor_identity != 0U;
}

bool Sm87TargetAotCompleteProjectionExecutionAccess::
    tamper_host_test_fixture_scale_bits(
        Owner& owner, const std::size_t layer_index, const Role role,
        const std::size_t source_index) noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (owner.arena_ != reinterpret_cast<std::uint8_t*>(kHostTestArenaBegin) ||
      owner.owner_identity_ != kHostTestOwnerIdentity ||
      owner.allocation_identity_ != kHostTestAllocationIdentity ||
      !owner.execution_bound_ || index >= owner.descriptor_count_) {
    return false;
  }
  auto& inventory = owner.descriptors_[index].source_inventory;
  if (source_index >= inventory.source_count) {
    return false;
  }
  inventory.sources[source_index].tensor_scale_bits ^= 1U;
  return true;
}

bool Sm87TargetAotCompleteProjectionExecutionAccess::
    tamper_host_test_fixture_device_identity(Owner& owner) noexcept {
  if (owner.arena_ != reinterpret_cast<std::uint8_t*>(kHostTestArenaBegin) ||
      owner.owner_identity_ != kHostTestOwnerIdentity ||
      owner.allocation_identity_ != kHostTestAllocationIdentity ||
      owner.device_identity_ != kHostTestDeviceIdentity ||
      !owner.execution_bound_) {
    return false;
  }
  owner.device_identity_ ^= 0x0000'0000'0000'0100ULL;
  return owner.device_identity_ != 0U;
}

bool Sm87TargetAotCompleteProjectionExecutionAccess::
    clear_host_test_fixture(Owner& owner) noexcept {
  if (owner.arena_ != reinterpret_cast<std::uint8_t*>(kHostTestArenaBegin) ||
      owner.bytes_ != kSm87TargetAotCompleteProjectionDeviceArenaBytes ||
      owner.owner_identity_ != kHostTestOwnerIdentity ||
      owner.allocation_identity_ != kHostTestAllocationIdentity ||
      owner.device_ordinal_ != 0 || owner.execution_bound_) {
    return false;
  }
  owner.arena_ = nullptr;
  owner.bytes_ = 0U;
  owner.allocation_identity_ = 0U;
  owner.owner_identity_ = 0U;
  owner.device_identity_ = 0U;
  owner.device_ordinal_ = -1;
  owner.descriptors_ = {};
  owner.descriptor_count_ = 0U;
  owner.prepared_resident_ = nullptr;
  owner.prepared_model_weights_ = nullptr;
  return true;
}

}  // namespace q3x::runtime::target_aot_complete_execution_detail
#endif
