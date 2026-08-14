#include "../src/runtime/sm87_macrofeed_v3_p40_execution_package_internal.h"
#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
#include "../src/runtime/sm87_target_aot_projection_complete_execution_access_internal.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string_view>
#include <type_traits>
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
    upload.receipt_identity = kernels::
        sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(upload);
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
  model_weights.target_aot_complete_projection_attachment_.allocation_identity =
      owner.allocation_identity_;
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
    poison_host_test_fixture_receipt(Owner& owner,
                                     const std::size_t layer_index,
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

bool Sm87TargetAotCompleteProjectionExecutionAccess::clear_host_test_fixture(
    Owner& owner) noexcept {
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

namespace q3x::kernels {
namespace {

bool g_fail_gate_up_seal = false;
std::int32_t g_fp8_seal_device_ordinal = 0;

[[nodiscard]] Sm87MacroFeedV3NvFp4GateUpStartupSeal
host_test_gate_up_seal() noexcept {
  Sm87MacroFeedV3NvFp4GateUpStartupSeal seal;
  seal.plan_identity = kSm87MacroFeedV3NvFp4GateUpIdentity;
  seal.kernel_symbol_identity =
      kSm87MacroFeedV3NvFp4GateUpKernelSymbolIdentity;
  seal.device_ordinal = 0;
  seal.compute_major = 8;
  seal.compute_minor = 7;
  seal.sm_count = 16;
  seal.binary_version = 87;
  seal.registers_per_thread = 220;
  seal.static_shared_bytes = 0U;
  seal.dynamic_shared_bytes =
      kSm87MacroFeedV3NvFp4GateUpDynamicSharedBytes;
  seal.local_bytes = 0U;
  seal.maximum_threads_per_block = 1'024;
  seal.active_blocks_per_sm = 1;
  seal.optin_shared_bytes_per_block = 102'400U;
  seal.dynamic_shared_attribute_configured = true;
  seal.static_resource_gate_passed = true;
  seal.request_hot_static_queries_forbidden = true;
  seal.t0_t1_only = true;
  seal.production_dispatch_eligible = false;
  seal.seal_identity =
      sm87_macrofeed_v3_nvfp4_gate_up_compute_startup_seal_identity(seal);
  return seal;
}

[[nodiscard]] Sm87MacroFeedV3NvFp4DownCudaResources
host_test_down_resources() noexcept {
  Sm87MacroFeedV3NvFp4DownCudaResources resources;
  resources.identity = kSm87MacroFeedV3NvFp4DownIdentity;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.binary_version = 87;
  resources.registers_per_thread = 218;
  resources.static_shared_bytes = 0U;
  resources.dynamic_shared_bytes =
      kSm87MacroFeedV3NvFp4DownDynamicSharedBytes;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = 1;
  resources.optin_shared_bytes_per_block = 102'400U;
  resources.kernel_compiled = true;
  resources.static_resource_gate_passed = true;
  resources.numerical_contract_qualified = false;
  resources.production_dispatch_eligible = false;
  return resources;
}

[[nodiscard]] Sm87MacrofeedV3GdnResources
host_test_gdn_resources() noexcept {
  Sm87MacrofeedV3GdnResources resources;
  resources.binary_version = 87;
  resources.convolution = {
      29,
      4U,
      0U,
      1'024,
      6,
      static_cast<int>(kSm87MacrofeedV3GdnConvThreads),
      static_cast<int>(kSm87MacrofeedV3GdnConvCtas),
  };
  resources.recurrence_epilogue = {
      80,
      34'316U,
      0U,
      1'024,
      3,
      static_cast<int>(kSm87MacrofeedV3GdnRecurrenceThreads),
      static_cast<int>(kSm87MacrofeedV3GdnRecurrenceCtas),
  };
  resources.kernels_compiled = true;
  resources.exact_geometry = true;
  resources.resource_gate_passed = true;
  resources.numerical_contract_qualified = false;
  resources.production_dispatch_eligible = false;
  return resources;
}

[[nodiscard]] Sm87MacroFeedV3Fp8StartupSeal host_test_fp8_seal(
    const Sm87TargetAotProjectionRole role) noexcept {
  Sm87MacroFeedV3Fp8StartupSeal seal;
  seal.resources.identity = sm87_macrofeed_v3_fp8_identity(role);
  seal.resources.role = role;
  seal.resources.device_ordinal = g_fp8_seal_device_ordinal;
  seal.resources.compute_major = 8;
  seal.resources.compute_minor = 7;
  seal.resources.sm_count = 16;
  seal.resources.binary_version = 87;
  seal.resources.registers_per_thread =
      role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ
          ? 224
          : (role == Sm87TargetAotProjectionRole::kFp8FullQkv ? 228 : 220);
  seal.resources.static_shared_bytes = 0U;
  seal.resources.dynamic_shared_bytes =
      kSm87MacroFeedV3Fp8DynamicSharedBytes;
  seal.resources.local_bytes = 0U;
  seal.resources.maximum_threads_per_block = 1'024;
  seal.resources.active_blocks_per_sm = 1;
  seal.resources.optin_shared_bytes_per_block = 102'400U;
  seal.resources.kernel_compiled = true;
  seal.resources.static_resource_gate_passed = true;
  seal.resources.numerical_contract_qualified = false;
  seal.resources.production_dispatch_eligible = false;
  seal.dynamic_shared_attribute_set = true;
  seal.tactic_frozen_before_requests = true;
  seal.no_hot_device_queries = true;
  seal.no_hot_function_queries = true;
  seal.no_hot_occupancy_queries = true;
  seal.no_hot_pointer_queries = true;
  seal.no_hot_error_state_clear = true;
  seal.t0_t1_only = true;
  seal.production_dispatch_eligible = false;
  seal.seal_identity =
      sm87_macrofeed_v3_fp8_compute_startup_seal_identity(seal);
  return seal;
}

}  // namespace

int seal_sm87_macrofeed_v3_nvfp4_gate_up_startup(
    Sm87MacroFeedV3NvFp4GateUpStartupSeal* const seal) noexcept {
  if (seal == nullptr || g_fail_gate_up_seal) {
    if (seal != nullptr) {
      *seal = {};
    }
    return 1;
  }
  *seal = host_test_gate_up_seal();
  return 0;
}

int query_sm87_macrofeed_v3_nvfp4_down_cuda_resources(
    Sm87MacroFeedV3NvFp4DownCudaResources* const resources) noexcept {
  if (resources == nullptr) {
    return 1;
  }
  *resources = host_test_down_resources();
  return 0;
}

int query_sm87_macrofeed_v3_gdn_p40_resources_cuda(
    Sm87MacrofeedV3GdnResources* const resources) noexcept {
  if (resources == nullptr) {
    return 1;
  }
  *resources = host_test_gdn_resources();
  return 0;
}

int seal_sm87_macrofeed_v3_fp8_startup_cuda(
    const Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV3Fp8StartupSeal* const seal) noexcept {
  if (seal == nullptr || !sm87_macrofeed_v3_fp8_role(role)) {
    if (seal != nullptr) {
      *seal = {};
    }
    return 1;
  }
  *seal = host_test_fp8_seal(role);
  return 0;
}

}  // namespace q3x::kernels

namespace {

namespace package =
    q3x::runtime::sm87_macrofeed_v3_p40_execution_package_detail;
namespace execution =
    q3x::runtime::target_aot_complete_execution_detail;
using Package = package::Sm87MacroFeedV3P40ExecutionPackage;
using Binding = package::Sm87MacroFeedV3P40ProjectionStartupBinding;
using PackageError = package::Sm87MacroFeedV3P40ExecutionPackageError;
using Access = execution::Sm87TargetAotCompleteProjectionExecutionAccess;
using Owner = q3x::runtime::Sm87TargetAotCompleteProjectionDeviceAssets;
using Role = q3x::kernels::Sm87TargetAotProjectionRole;
using q3x::runtime::ModelWeights;

void require_package(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
[[nodiscard]] bool clear_fixture(
    std::optional<ModelWeights>& model_weights, Owner& owner) {
  model_weights.reset();
  return Access::clear_host_test_fixture(owner) && owner.empty();
}
#endif

#if defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
void test_complete_private_package() {
  static_assert(package::kSm87MacroFeedV3P40ExecutionPackageCompiled);
  static_assert(!std::is_default_constructible_v<Package>);
  static_assert(!std::is_copy_constructible_v<Package>);
  static_assert(!std::is_move_constructible_v<Package>);
  static_assert(!std::is_default_constructible_v<Binding>);
  static_assert(std::is_copy_constructible_v<Binding>);
  static_assert(std::is_move_constructible_v<Binding>);
  static_assert(!std::is_copy_assignable_v<Binding>);
  static_assert(!std::is_move_assignable_v<Binding>);

  Owner owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct complete private package fixture");

  auto first = Package::create(*model_weights);
  require_package(static_cast<bool>(first),
                  "complete private authority did not create package");
  require_package(first.audit.valid() && first.package->audit().valid(),
                  "package audit is not closed");
  require_package(first.audit.layers == 64U &&
                      first.audit.device_identity ==
                          execution::kHostTestDeviceIdentity &&
                      first.audit.device_ordinal == 0 &&
                      first.audit.artifacts == 256U &&
                      first.audit.sources == 400U &&
                      first.audit.gate_up_assets == 64U &&
                      first.audit.down_assets == 64U &&
                      first.audit.gdn_projection_assets == 48U &&
                      first.audit.full_projection_assets == 16U &&
                      first.audit.attention_output_assets == 64U,
                  "package catalog cardinality drifted");
  require_package(!first.audit.caller_raw_receipts_accepted &&
                      !first.audit.v2_owner_or_executor_reused &&
                      !first.audit.request_time_repack_jit_or_fallback_permitted &&
                      first.audit.authenticated_source_manifests_retained &&
                      first.audit.t0_t1_only &&
                      !first.audit.production_dispatch_eligible,
                  "package fail-closed policy drifted");

  auto second = Package::create(*model_weights);
  require_package(static_cast<bool>(second) &&
                      second.audit.package_identity ==
                          first.audit.package_identity,
                  "package identity is not stable for one startup authority");

  std::size_t nvfp4 = 0U;
  std::size_t fp8 = 0U;
  for (std::size_t layer = 0U; layer < 64U; ++layer) {
    require_package(
        first.package->borrow_nvfp4_asset(layer, Role::kNvFp4GateUp) !=
                nullptr &&
            first.package->borrow_nvfp4_asset(layer, Role::kNvFp4Down) !=
                nullptr,
        "package did not retain both typed NVFP4 capabilities");
    nvfp4 += 2U;
    const Role primary =
        q3x::runtime::sm87_target_aot_complete_is_full_layer(layer)
            ? Role::kFp8FullQkv
            : Role::kFp8GdnQkvZ;
    require_package(
        first.package->borrow_fp8_asset(layer, primary) != nullptr &&
            first.package->borrow_fp8_asset(
                layer, Role::kFp8AttentionOutput) != nullptr,
        "package did not retain both typed FP8 capabilities");
    fp8 += 2U;
    require_package(
        first.package->borrow_fp8_asset(
            layer, primary == Role::kFp8FullQkv
                       ? Role::kFp8GdnQkvZ
                       : Role::kFp8FullQkv) == nullptr,
        "package exposed a role absent from the layer catalog");

    for (const Role role : {Role::kNvFp4GateUp, Role::kNvFp4Down,
                            primary, Role::kFp8AttentionOutput}) {
      const auto* const issued =
          first.package->borrow_projection_startup_binding(layer, role);
      require_package(
          issued != nullptr && issued->valid() &&
              issued->valid_for(layer, role,
                                first.audit.package_identity) &&
              issued->binding_identity() != 0U &&
              issued->package_identity() == first.audit.package_identity &&
              issued->device_identity() == first.audit.device_identity &&
              issued->layer_index() == layer && issued->role() == role &&
              issued->artifact_identity() != 0U &&
              issued->source_inventory_identity() != 0U &&
              issued->payload_begin() != 0U &&
              issued->payload_bytes() != 0U,
          "startup binding did not retain the exact package/layer/role");
      Binding copied(*issued);
      require_package(
          copied.valid() &&
              copied.binding_identity() == issued->binding_identity() &&
              !copied.valid_for(
                  layer == 0U ? 1U : 0U, role,
                  first.audit.package_identity) &&
              !copied.valid_for(
                  layer,
                  role == Role::kNvFp4GateUp ? Role::kNvFp4Down
                                             : Role::kNvFp4GateUp,
                  first.audit.package_identity) &&
              !copied.valid_for(layer, role,
                                first.audit.package_identity ^ 1U),
          "startup binding accepted a changed layer, role, or package");

      if (role == Role::kNvFp4GateUp) {
        const auto* const gate = copied.source(0U);
        const auto* const up = copied.source(1U);
        const auto* const receipt = copied.gate_up_payload_receipt();
        require_package(
            copied.source_count() == 2U && gate != nullptr && up != nullptr &&
                gate->tensor_identity != 0U && up->tensor_identity != 0U &&
                gate->tensor_identity != up->tensor_identity &&
                std::isfinite(copied.tensor_scale(0U)) &&
                copied.tensor_scale(0U) > 0.0F &&
                std::isfinite(copied.tensor_scale(1U)) &&
                copied.tensor_scale(1U) > 0.0F && receipt != nullptr &&
                receipt->gate_source_identity == gate->tensor_identity &&
                receipt->up_source_identity == up->tensor_identity &&
                copied.down_payload_receipt() == nullptr &&
                copied.borrow_nvfp4_asset() != nullptr &&
                copied.borrow_fp8_asset() == nullptr,
            "Gate+Up binding lost its two source/scale/receipt facts");
      } else if (role == Role::kNvFp4Down) {
        const auto* const down = copied.source(0U);
        require_package(
            copied.source_count() == 1U && down != nullptr &&
                down->tensor_identity != 0U &&
                std::isfinite(copied.tensor_scale(0U)) &&
                copied.tensor_scale(0U) > 0.0F &&
                copied.down_payload_receipt() != nullptr &&
                copied.gate_up_payload_receipt() == nullptr &&
                copied.borrow_nvfp4_asset() != nullptr &&
                copied.borrow_fp8_asset() == nullptr,
            "Down binding lost its source/scale/receipt facts");
      } else {
        require_package(
            copied.source_count() != 0U &&
                copied.borrow_fp8_asset() != nullptr &&
                copied.borrow_nvfp4_asset() == nullptr &&
                copied.gate_up_payload_receipt() == nullptr &&
                copied.down_payload_receipt() == nullptr,
            "FP8 binding did not remain role-specific and receipt-free");
      }
      require_package(copied.source(copied.source_count()) == nullptr &&
                          copied.tensor_scale(copied.source_count()) == 0.0F,
                      "startup binding accepted an out-of-range source");
    }
  }
  require_package(
      nvfp4 == 128U && fp8 == 128U &&
          first.package->borrow_nvfp4_asset(64U, Role::kNvFp4GateUp) ==
              nullptr &&
          first.package->borrow_nvfp4_asset(0U, Role::kFp8GdnQkvZ) ==
              nullptr &&
          first.package->borrow_fp8_asset(0U, Role::kNvFp4Down) ==
              nullptr,
      "typed capability boundary accepted an invalid key");
  require_package(
      first.package->borrow_projection_startup_binding(
          64U, Role::kNvFp4GateUp) == nullptr &&
          first.package->borrow_projection_startup_binding(
              0U, Role::kInvalid) == nullptr &&
          first.package->borrow_projection_startup_binding(
              0U, Role::kFp8FullQkv) == nullptr,
      "startup binding borrow accepted an absent or invalid catalog key");

  require_package(
      q3x::kernels::sm87_macrofeed_v3_nvfp4_gate_up_startup_seal_valid(
          first.package->gate_up_startup_seal()) &&
          q3x::kernels::sm87_macrofeed_v3_nvfp4_down_resource_gate(
              first.package->down_startup_seal()) &&
          q3x::kernels::sm87_macrofeed_v3_gdn_resources_valid(
              first.package->gdn_startup_seal()),
      "non-FP8 startup seal set is invalid");
  for (const Role role :
       {Role::kFp8GdnQkvZ, Role::kFp8FullQkv,
        Role::kFp8AttentionOutput}) {
    const auto* const seal = first.package->fp8_startup_seal(role);
    require_package(
        seal != nullptr &&
            q3x::kernels::sm87_macrofeed_v3_fp8_startup_seal_valid(*seal),
        "FP8 startup seal set is incomplete");
  }
  require_package(
      first.package->fp8_startup_seal(Role::kNvFp4GateUp) == nullptr,
      "FP8 seal lookup accepted a non-FP8 role");

  const auto* const poisoned_borrow =
      first.package->borrow_projection_startup_binding(
          63U, Role::kFp8AttentionOutput);
  require_package(poisoned_borrow != nullptr && poisoned_borrow->valid(),
                  "could not retain binding for receipt poison test");
  Binding poisoned_binding(*poisoned_borrow);
  const bool poisoned = Access::poison_host_test_fixture_receipt(
      owner, 63U, Role::kFp8AttentionOutput);
  require_package(
      poisoned && !first.package->valid() && !second.package->valid() &&
          !poisoned_binding.valid() &&
          poisoned_binding.borrow_fp8_asset() == nullptr &&
          first.package->borrow_fp8_asset(
              63U, Role::kFp8AttentionOutput) == nullptr,
      "package did not revalidate a poisoned live private receipt");

  first.package.reset();
  second.package.reset();
  require_package(clear_fixture(model_weights, owner),
                  "complete package fixture cleanup failed");
}

void test_binding_owner_lifetime_is_closed() {
  Owner owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct binding owner fixture");
  auto result = Package::create(*model_weights);
  require_package(static_cast<bool>(result),
                  "could not create binding owner package");
  const auto* const borrowed = result.package->borrow_projection_startup_binding(
      7U, Role::kNvFp4GateUp);
  require_package(borrowed != nullptr && borrowed->valid(),
                  "could not issue owner-backed binding");
  Binding binding(*borrowed);

  std::optional<ModelWeights> moved(std::in_place,
                                    std::move(*model_weights));
  require_package(!binding.valid() && !result.package->valid() &&
                      binding.borrow_nvfp4_asset() == nullptr &&
                      binding.gate_up_payload_receipt() == nullptr,
                  "binding survived replacement of its ModelWeights owner");

  result.package.reset();
  model_weights.reset();
  moved.reset();
  require_package(Access::clear_host_test_fixture(owner) && owner.empty(),
                  "binding owner fixture cleanup failed");
}

void test_binding_source_identity_tamper_is_closed() {
  Owner owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct source identity fixture");
  auto result = Package::create(*model_weights);
  require_package(static_cast<bool>(result),
                  "could not create source identity package");
  const auto* const borrowed = result.package->borrow_projection_startup_binding(
      11U, Role::kNvFp4GateUp);
  require_package(borrowed != nullptr && borrowed->valid(),
                  "could not issue source identity binding");
  Binding binding(*borrowed);
  require_package(
      Access::tamper_host_test_fixture_source_identity(
          owner, 11U, Role::kNvFp4GateUp, 1U) &&
          !binding.valid() && !result.package->valid() &&
          binding.gate_up_payload_receipt() == nullptr,
      "binding accepted a changed live source identity");
  result.package.reset();
  require_package(clear_fixture(model_weights, owner),
                  "source identity fixture cleanup failed");
}

void test_binding_scale_tamper_is_closed() {
  Owner owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct scale fixture");
  auto result = Package::create(*model_weights);
  require_package(static_cast<bool>(result),
                  "could not create scale package");
  const auto* const borrowed = result.package->borrow_projection_startup_binding(
      19U, Role::kNvFp4Down);
  require_package(borrowed != nullptr && borrowed->valid(),
                  "could not issue scale binding");
  Binding binding(*borrowed);
  require_package(Access::tamper_host_test_fixture_scale_bits(
                      owner, 19U, Role::kNvFp4Down, 0U) &&
                      !binding.valid() && !result.package->valid() &&
                      binding.down_payload_receipt() == nullptr,
                  "binding accepted changed live tensor scale bits");
  result.package.reset();
  require_package(clear_fixture(model_weights, owner),
                  "scale fixture cleanup failed");
}

void test_binding_device_identity_tamper_is_closed() {
  Owner owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct device identity fixture");
  auto result = Package::create(*model_weights);
  require_package(static_cast<bool>(result) &&
                      result.audit.device_identity ==
                          execution::kHostTestDeviceIdentity &&
                      result.audit.device_ordinal == 0,
                  "could not create physical-device-bound package");
  const auto* const borrowed =
      result.package->borrow_projection_startup_binding(
          23U, Role::kNvFp4GateUp);
  require_package(borrowed != nullptr && borrowed->valid() &&
                      borrowed->device_identity() ==
                          execution::kHostTestDeviceIdentity,
                  "could not borrow physical-device-bound binding");
  Binding binding(*borrowed);
  require_package(
      Access::tamper_host_test_fixture_device_identity(owner) &&
          !binding.valid() && !result.package->valid() &&
          result.package->borrow_projection_startup_binding(
              23U, Role::kNvFp4GateUp) == nullptr &&
          binding.borrow_nvfp4_asset() == nullptr &&
          binding.gate_up_payload_receipt() == nullptr &&
          !Access::bind(*model_weights).has_value(),
      "ordinal-only match survived a physical device identity change");
  std::optional<ModelWeights> moved(std::in_place,
                                    std::move(*model_weights));
  require_package(!Access::bind(*moved).has_value() && !binding.valid() &&
                      !result.package->valid(),
                  "ModelWeights move retargeted a poisoned device snapshot");
  result.package.reset();
  model_weights.reset();
  moved.reset();
  require_package(Access::clear_host_test_fixture(owner) && owner.empty(),
                  "device identity fixture cleanup failed");
}

void test_gate_seal_failure_is_closed() {
  Owner owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct gate failure fixture");
  q3x::kernels::g_fail_gate_up_seal = true;
  auto result = Package::create(*model_weights);
  q3x::kernels::g_fail_gate_up_seal = false;
  require_package(
      !result && result.package == nullptr &&
          result.status.error == PackageError::kGateUpStartupSeal &&
          result.status.cuda_error == 1,
      "gate startup failure did not fail closed");
  require_package(clear_fixture(model_weights, owner),
                  "gate failure fixture cleanup failed");
}

void test_cross_device_seal_is_closed() {
  Owner owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct device mismatch fixture");
  q3x::kernels::g_fp8_seal_device_ordinal = 1;
  auto result = Package::create(*model_weights);
  q3x::kernels::g_fp8_seal_device_ordinal = 0;
  require_package(
      !result && result.package == nullptr &&
          result.status.error == PackageError::kDeviceMismatch,
      "cross-device startup seals did not fail closed");
  require_package(clear_fixture(model_weights, owner),
                  "device mismatch fixture cleanup failed");
}
#elif defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
void test_default_off_package_is_closed() {
  static_assert(!package::kSm87MacroFeedV3P40ExecutionPackageCompiled);
  Owner owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct default-off package fixture");
  auto result = Package::create(*model_weights);
  require_package(
      !result && result.package == nullptr &&
          result.status.error == PackageError::kAdmissionDisabled,
      "default-off execution package did not fail closed");
  require_package(clear_fixture(model_weights, owner),
                  "default-off package fixture cleanup failed");
}
#else
void test_default_off_package_linkage_is_closed() {
  static_assert(!package::kSm87MacroFeedV3P40ExecutionPackageCompiled);
  package::Sm87MacroFeedV3P40ExecutionPackageCreateResult result;
  require_package(!result && result.package == nullptr,
                  "default-off package result was unexpectedly valid");
}
#endif

}  // namespace

int main() {
#if defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  test_complete_private_package();
  test_binding_owner_lifetime_is_closed();
  test_binding_source_identity_tamper_is_closed();
  test_binding_scale_tamper_is_closed();
  test_binding_device_identity_tamper_is_closed();
  test_gate_seal_failure_is_closed();
  test_cross_device_seal_is_closed();
#elif defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  test_default_off_package_is_closed();
#else
  test_default_off_package_linkage_is_closed();
#endif
  return 0;
}
