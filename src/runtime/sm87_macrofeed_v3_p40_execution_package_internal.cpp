#include "sm87_macrofeed_v3_p40_execution_package_internal.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace q3x::runtime::sm87_macrofeed_v3_p40_execution_package_detail {
namespace {

using Role = kernels::Sm87TargetAotProjectionRole;
using Encoding = kernels::Sm87TargetAotProjectionEncoding;
using Error = Sm87MacroFeedV3P40ExecutionPackageError;
using Status = Sm87MacroFeedV3P40ExecutionPackageStatus;
using Package = Sm87MacroFeedV3P40ExecutionPackage;
using CreateResult = Sm87MacroFeedV3P40ExecutionPackageCreateResult;

[[nodiscard]] constexpr std::array<Role, 4U> layer_roles(
    const std::size_t layer_index) noexcept {
  return {{Role::kNvFp4GateUp, Role::kNvFp4Down,
           sm87_target_aot_complete_is_full_layer(layer_index)
               ? Role::kFp8FullQkv
               : Role::kFp8GdnQkvZ,
           Role::kFp8AttentionOutput}};
}

[[nodiscard]] constexpr std::size_t fp8_seal_index(
    const Role role) noexcept {
  return role == Role::kFp8GdnQkvZ
             ? 0U
             : (role == Role::kFp8FullQkv
                    ? 1U
                    : (role == Role::kFp8AttentionOutput ? 2U : 3U));
}

[[nodiscard]] Status failure(const Error error, const char* const context,
                             const int cuda_error = 0,
                             const std::size_t layer =
                                 kSm87MacroFeedV3P40PackageLayers,
                             const Role role = Role::kInvalid) noexcept {
  return {error, context, layer, role, cuda_error};
}

[[nodiscard]] constexpr std::uint64_t mix(std::uint64_t hash,
                                          const std::uint64_t value) noexcept {
  hash ^= value + 0x9e37'79b9'7f4a'7c15ULL + (hash << 6U) + (hash >> 2U);
  return hash;
}

}  // namespace

Sm87MacroFeedV3P40ExecutionPackage::Sm87MacroFeedV3P40ExecutionPackage(
    ProjectionAccess access,
    std::array<AssetCapability, kSm87MacroFeedV3P40PackageArtifacts>
        capabilities,
    StartupSeals seals,
    Sm87MacroFeedV3P40ExecutionPackageAudit audit) noexcept
    : projection_access_(std::move(access)),
      capabilities_(std::move(capabilities)),
      seals_(std::move(seals)),
      audit_(audit) {}

Sm87MacroFeedV3P40ProjectionStartupBinding::
    Sm87MacroFeedV3P40ProjectionStartupBinding(
        ProjectionAccess access, ProjectionAsset asset,
        Snapshot snapshot) noexcept
    : projection_access_(std::move(access)),
      asset_(std::move(asset)),
      snapshot_(std::move(snapshot)) {}

Sm87MacroFeedV3P40ExecutionPackageCreateResult::operator bool()
    const noexcept {
  return package != nullptr && static_cast<bool>(status) && audit.valid() &&
         package->valid() &&
         audit.package_identity == package->audit().package_identity;
}

Sm87MacroFeedV3P40ExecutionPackageCreateResult
Sm87MacroFeedV3P40ExecutionPackage::create(
    const ModelWeights& model_weights) noexcept {
#if !defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  (void)model_weights;
  CreateResult result;
  result.status = failure(Error::kAdmissionDisabled, "admission_disabled");
  return result;
#else
  auto access = ProjectionAccess::bind(model_weights);
  if (!access) {
    CreateResult result;
    result.status =
        failure(Error::kProjectionAccessBind, "projection_access_bind");
    return result;
  }

  StartupSeals seals;
  int status = kernels::seal_sm87_macrofeed_v3_nvfp4_gate_up_startup(
      &seals.gate_up);
  if (status != static_cast<int>(cudaSuccess) ||
      !kernels::sm87_macrofeed_v3_nvfp4_gate_up_startup_seal_valid(
          seals.gate_up)) {
    CreateResult result;
    result.status = failure(Error::kGateUpStartupSeal, "gate_up_startup_seal",
                            status);
    return result;
  }
  status = kernels::query_sm87_macrofeed_v3_nvfp4_down_cuda_resources(
      &seals.down);
  if (status != static_cast<int>(cudaSuccess) ||
      !seals.down.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v3_nvfp4_down_resource_gate(seals.down)) {
    CreateResult result;
    result.status = failure(Error::kDownStartupSeal, "down_startup_seal",
                            status);
    return result;
  }
  status = kernels::query_sm87_macrofeed_v3_gdn_p40_resources_cuda(
      &seals.gdn);
  if (status != static_cast<int>(cudaSuccess) ||
      !kernels::sm87_macrofeed_v3_gdn_resources_valid(seals.gdn)) {
    CreateResult result;
    result.status = failure(Error::kGdnStartupSeal, "gdn_startup_seal",
                            status);
    return result;
  }

  constexpr std::array<Role, 3U> kFp8Roles{{
      Role::kFp8GdnQkvZ,
      Role::kFp8FullQkv,
      Role::kFp8AttentionOutput,
  }};
  constexpr std::array<Error, 3U> kFp8Errors{{
      Error::kFp8GdnStartupSeal,
      Error::kFp8FullStartupSeal,
      Error::kFp8OutputStartupSeal,
  }};
  constexpr std::array<const char*, 3U> kFp8Contexts{{
      "fp8_gdn_startup_seal",
      "fp8_full_startup_seal",
      "fp8_output_startup_seal",
  }};
  for (std::size_t index = 0U; index < kFp8Roles.size(); ++index) {
    status = kernels::seal_sm87_macrofeed_v3_fp8_startup_cuda(
        kFp8Roles[index], &seals.fp8[index]);
    if (status != static_cast<int>(cudaSuccess) ||
        !kernels::sm87_macrofeed_v3_fp8_startup_seal_valid(
            seals.fp8[index])) {
      CreateResult result;
      result.status =
          failure(kFp8Errors[index], kFp8Contexts[index], status);
      return result;
    }
  }
  return build_from_private_authority(std::move(*access), std::move(seals));
#endif
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
namespace {

using Binding = Sm87MacroFeedV3P40ProjectionStartupBinding;
using LogicalRole = kernels::Sm87TargetAotLogicalRole;

[[nodiscard]] float decode_fp32_bits(const std::uint32_t bits) noexcept {
  static_assert(sizeof(float) == sizeof(bits));
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] bool same_source_inventory(
    const Binding::SourceInventory& left,
    const Binding::SourceInventory& right) noexcept {
  if (left.identity != right.identity || left.role != right.role ||
      left.source_count != right.source_count) {
    return false;
  }
  for (std::size_t index = 0U; index < left.sources.size(); ++index) {
    if (!kernels::sm87_target_aot_projection_same_source_binding(
            left.sources[index], right.sources[index])) {
      return false;
    }
  }
  return true;
}

template <typename UploadReceipt>
[[nodiscard]] bool authenticated_upload_complete(
    const UploadReceipt& upload) noexcept {
  return upload.receipt_identity != 0U &&
         upload.host_payload_digest_verified_before_copy &&
         upload.host_payload_immutable_until_completion &&
         upload.copy_enqueued_to_exact_payload_range &&
         upload.completion_event_recorded_after_copy &&
         upload.completion_event_observed && upload.upload_completed &&
         upload.verification_copy_enqueued_from_exact_payload_range &&
         upload.verification_event_recorded_after_copy &&
         upload.verification_event_observed && upload.verification_completed &&
         upload.device_payload_matches_host_payload &&
         upload.allocation_retained_for_asset_lifetime;
}

[[nodiscard]] constexpr bool gate_up_receipt_zero(
    const kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt& receipt) noexcept {
  return receipt.receipt_identity == 0U &&
         receipt.plan_identity ==
             kernels::Sm87MacroFeedV3NvFp4GateUpIdentity::kInvalid &&
         receipt.payload_identity == 0U &&
         receipt.gate_source_identity == 0U &&
         receipt.up_source_identity == 0U && receipt.device_ordinal == -1 &&
         receipt.payload_begin == 0U && receipt.payload_end == 0U &&
         receipt.payload_bytes == 0U && receipt.gate_partition_bytes == 0U &&
         receipt.up_partition_bytes == 0U &&
         !receipt.canonical_consumer_n64_k16_lane_component_v1 &&
         !receipt.canonical_gate_then_up_partition_order &&
         !receipt.independent_tensor_scales &&
         !receipt.host_bytes_authenticated_before_copy &&
         !receipt.device_readback_authenticated &&
         !receipt.allocation_retained_for_launch;
}

[[nodiscard]] constexpr bool down_receipt_zero(
    const kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt& receipt) noexcept {
  return receipt.receipt_identity == 0U &&
         receipt.plan_identity ==
             kernels::Sm87MacroFeedV3NvFp4DownIdentity::kInvalid &&
         receipt.payload_identity == 0U && receipt.device_ordinal == -1 &&
         receipt.payload_begin == 0U && receipt.payload_end == 0U &&
         receipt.payload_bytes == 0U &&
         !receipt.canonical_consumer_n64_k16_lane_component_v1 &&
         !receipt.host_bytes_authenticated_before_copy &&
         !receipt.device_readback_authenticated &&
         !receipt.allocation_retained_for_launch;
}

}  // namespace

std::uint64_t
Sm87MacroFeedV3P40ProjectionStartupBinding::compute_binding_identity(
    const Snapshot& snapshot) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(snapshot.role);
  if (snapshot.package_identity == 0U || snapshot.owner_identity == 0U ||
      snapshot.allocation_identity == 0U || snapshot.catalog_identity == 0U ||
      snapshot.device_identity == 0U ||
      snapshot.artifact_identity == 0U || snapshot.manifest_seal == 0U ||
      snapshot.upload_receipt_identity == 0U ||
      snapshot.device_ordinal < 0 ||
      snapshot.layer_index >= kSm87MacroFeedV3P40PackageLayers ||
      !layout.valid() || snapshot.encoding != layout.encoding ||
      !snapshot.source_inventory.valid(layout) ||
      snapshot.payload_begin == 0U ||
      snapshot.payload_bytes != layout.payload_bytes ||
      snapshot.payload_begin >
          std::numeric_limits<std::uintptr_t>::max() -
              snapshot.payload_bytes ||
      snapshot.payload_end != snapshot.payload_begin + snapshot.payload_bytes ||
      !snapshot.issued_from_live_private_descriptor ||
      snapshot.caller_receipt_accepted || !snapshot.t0_t1_only ||
      snapshot.production_dispatch_eligible) {
    return 0U;
  }

  const bool gate_up = snapshot.role == Role::kNvFp4GateUp;
  const bool down = snapshot.role == Role::kNvFp4Down;
  if ((gate_up &&
       (!kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
            snapshot.gate_up_receipt) ||
        !down_receipt_zero(snapshot.down_receipt))) ||
      (down &&
       (!kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
            snapshot.down_receipt) ||
        !gate_up_receipt_zero(snapshot.gate_up_receipt))) ||
      (!gate_up && !down &&
       (!sm87_target_aot_complete_role_is_fp8(snapshot.role) ||
        !gate_up_receipt_zero(snapshot.gate_up_receipt) ||
        !down_receipt_zero(snapshot.down_receipt)))) {
    return 0U;
  }

  std::uint64_t identity = 0x5133'4d46'5633'424eULL;
  identity = mix(identity, snapshot.package_identity);
  identity = mix(identity, snapshot.owner_identity);
  identity = mix(identity, snapshot.allocation_identity);
  identity = mix(identity, snapshot.catalog_identity);
  identity = mix(identity, snapshot.device_identity);
  identity = mix(identity, snapshot.artifact_identity);
  identity = mix(identity, snapshot.manifest_seal);
  identity = mix(identity, snapshot.upload_receipt_identity);
  identity = mix(identity, snapshot.payload_begin);
  identity = mix(identity, snapshot.payload_end);
  identity = mix(identity, snapshot.payload_bytes);
  identity = mix(
      identity, static_cast<std::uint64_t>(snapshot.device_ordinal + 1));
  identity = mix(identity, snapshot.layer_index + 1U);
  identity = mix(identity, static_cast<std::uint64_t>(snapshot.role));
  identity = mix(identity, static_cast<std::uint64_t>(snapshot.encoding));
  identity = mix(identity, snapshot.source_inventory.identity);
  identity = mix(identity, snapshot.source_inventory.source_count);
  for (std::size_t index = 0U;
       index < snapshot.source_inventory.source_count; ++index) {
    const auto& source = snapshot.source_inventory.sources[index];
    identity = mix(identity, index + 1U);
    identity = mix(identity,
                   static_cast<std::uint64_t>(source.logical_role));
    identity = mix(identity, source.partition_index);
    identity = mix(identity, source.tensor_identity);
    for (const std::uint8_t byte : source.weight_digest.bytes) {
      identity = mix(identity, byte);
    }
    for (const std::uint8_t byte : source.scale_digest.bytes) {
      identity = mix(identity, byte);
    }
    identity = mix(identity, source.output_features);
    identity = mix(identity, source.input_features);
    identity = mix(identity, source.tensor_scale_bits);
    identity = mix(identity, source.payload_offset);
    identity = mix(identity, source.payload_bytes);
  }
  identity = mix(identity, snapshot.gate_up_receipt.receipt_identity);
  identity = mix(identity, snapshot.down_receipt.receipt_identity);
  identity = mix(identity, snapshot.issued_from_live_private_descriptor);
  identity = mix(identity, snapshot.caller_receipt_accepted);
  identity = mix(identity, snapshot.t0_t1_only);
  identity = mix(identity, snapshot.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5633'424eULL : identity;
}

const Sm87MacroFeedV3P40ProjectionStartupBinding::SourceBinding*
Sm87MacroFeedV3P40ProjectionStartupBinding::source(
    const std::size_t source_index) const noexcept {
  return source_index < snapshot_.source_inventory.source_count
             ? &snapshot_.source_inventory.sources[source_index]
             : nullptr;
}

float Sm87MacroFeedV3P40ProjectionStartupBinding::tensor_scale(
    const std::size_t source_index) const noexcept {
  const SourceBinding* const item = source(source_index);
  return item != nullptr &&
                 kernels::sm87_target_aot_projection_scale_bits_valid(
                     item->tensor_scale_bits)
             ? decode_fp32_bits(item->tensor_scale_bits)
             : 0.0F;
}

bool Sm87MacroFeedV3P40ProjectionStartupBinding::
    valid_with_authenticated_catalog(
        const std::uint64_t catalog_identity) const noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(snapshot_.role);
  if (!layout.valid() || snapshot_.binding_identity == 0U ||
      snapshot_.binding_identity != compute_binding_identity(snapshot_) ||
      !projection_access_.attached() ||
      projection_access_.owner_identity() != snapshot_.owner_identity ||
      projection_access_.allocation_identity() !=
          snapshot_.allocation_identity ||
      projection_access_.device_identity() != snapshot_.device_identity ||
      projection_access_.device_ordinal() != snapshot_.device_ordinal ||
      catalog_identity != snapshot_.catalog_identity ||
      asset_.layer_index() != snapshot_.layer_index ||
      asset_.role() != snapshot_.role ||
      asset_.encoding() != snapshot_.encoding ||
      asset_.artifact_identity() != snapshot_.artifact_identity ||
      asset_.source_inventory_identity() !=
          snapshot_.source_inventory.identity ||
      asset_.payload_bytes() != snapshot_.payload_bytes) {
    return false;
  }

  auto fresh = projection_access_.resolve(snapshot_.layer_index,
                                          snapshot_.role);
  const auto* const descriptor = asset_.descriptor_;
  if (!fresh || descriptor == nullptr ||
      fresh->descriptor_ != descriptor ||
      descriptor->layer_index != snapshot_.layer_index ||
      descriptor->role != snapshot_.role ||
      descriptor->encoding != snapshot_.encoding ||
      descriptor->manifest.artifact_identity != snapshot_.artifact_identity ||
      descriptor->manifest.seal.value != snapshot_.manifest_seal ||
      !same_source_inventory(descriptor->source_inventory,
                             snapshot_.source_inventory) ||
      !kernels::sm87_target_aot_projection_validate_packed_manifest(
          descriptor->manifest, descriptor->source_inventory)) {
    return false;
  }

  const auto scales_match = [&](const auto& view) noexcept {
    if (view.tensor_scale_count != snapshot_.source_inventory.source_count) {
      return false;
    }
    for (std::size_t index = 0U; index < view.tensor_scale_count; ++index) {
      if (view.tensor_scale_bits[index] !=
              snapshot_.source_inventory.sources[index].tensor_scale_bits ||
          !std::isfinite(tensor_scale(index)) || tensor_scale(index) <= 0.0F) {
        return false;
      }
    }
    return true;
  };
  const auto common_view_matches = [&](const auto& view) noexcept {
    const auto& upload = view.device_upload_receipt;
    return view.artifact_identity == snapshot_.artifact_identity &&
           view.source_inventory_identity ==
               snapshot_.source_inventory.identity &&
           view.host_manifest_seal.value == snapshot_.manifest_seal &&
           view.payload.role == snapshot_.role &&
           view.payload.begin == snapshot_.payload_begin &&
           view.payload.end == snapshot_.payload_end &&
           view.payload.bytes == snapshot_.payload_bytes &&
           upload.receipt_identity == snapshot_.upload_receipt_identity &&
           upload.device_allocation_owner_identity ==
               snapshot_.owner_identity &&
           upload.device_allocation_identity ==
               snapshot_.allocation_identity &&
           upload.device_ordinal == snapshot_.device_ordinal &&
           upload.device_payload_begin == snapshot_.payload_begin &&
           upload.device_payload_end == snapshot_.payload_end &&
           upload.device_payload_bytes == snapshot_.payload_bytes &&
           upload.host_manifest_seal.value == snapshot_.manifest_seal &&
           authenticated_upload_complete(upload) && scales_match(view);
  };

  if (snapshot_.role == Role::kNvFp4GateUp ||
      snapshot_.role == Role::kNvFp4Down) {
    const auto* const view = asset_.borrow_nvfp4_cuda_asset();
    if (view == nullptr || asset_.borrow_fp8_cuda_asset() != nullptr ||
        !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) ||
        !common_view_matches(*view)) {
      return false;
    }
  } else {
    const auto* const view = asset_.borrow_fp8_cuda_asset();
    if (view == nullptr || asset_.borrow_nvfp4_cuda_asset() != nullptr ||
        !kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) ||
        !common_view_matches(*view)) {
      return false;
    }
  }

  if (snapshot_.role == Role::kNvFp4GateUp) {
    return snapshot_.source_inventory.source_count == 2U &&
           snapshot_.source_inventory.sources[0U].logical_role ==
               LogicalRole::kNvFp4Gate &&
           snapshot_.source_inventory.sources[1U].logical_role ==
               LogicalRole::kNvFp4Up &&
           snapshot_.gate_up_receipt.payload_identity ==
               snapshot_.artifact_identity &&
           snapshot_.gate_up_receipt.gate_source_identity ==
               snapshot_.source_inventory.sources[0U].tensor_identity &&
           snapshot_.gate_up_receipt.up_source_identity ==
               snapshot_.source_inventory.sources[1U].tensor_identity &&
           snapshot_.gate_up_receipt.device_ordinal ==
               snapshot_.device_ordinal &&
           snapshot_.gate_up_receipt.payload_begin ==
               snapshot_.payload_begin &&
           snapshot_.gate_up_receipt.payload_end == snapshot_.payload_end &&
           snapshot_.gate_up_receipt.payload_bytes ==
               snapshot_.payload_bytes &&
           kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
               snapshot_.gate_up_receipt) &&
           down_receipt_zero(snapshot_.down_receipt);
  }
  if (snapshot_.role == Role::kNvFp4Down) {
    return snapshot_.source_inventory.source_count == 1U &&
           snapshot_.source_inventory.sources[0U].logical_role ==
               LogicalRole::kNvFp4Down &&
           snapshot_.down_receipt.payload_identity ==
               snapshot_.artifact_identity &&
           snapshot_.down_receipt.device_ordinal ==
               snapshot_.device_ordinal &&
           snapshot_.down_receipt.payload_begin == snapshot_.payload_begin &&
           snapshot_.down_receipt.payload_end == snapshot_.payload_end &&
           snapshot_.down_receipt.payload_bytes == snapshot_.payload_bytes &&
           kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
               snapshot_.down_receipt) &&
           gate_up_receipt_zero(snapshot_.gate_up_receipt);
  }
  return sm87_target_aot_complete_role_is_fp8(snapshot_.role) &&
         gate_up_receipt_zero(snapshot_.gate_up_receipt) &&
         down_receipt_zero(snapshot_.down_receipt);
}

bool Sm87MacroFeedV3P40ProjectionStartupBinding::valid() const noexcept {
  if (!projection_access_.attached()) {
    return false;
  }
  const std::uint64_t catalog_identity =
      projection_access_.catalog_identity();
  return catalog_identity != 0U &&
         valid_with_authenticated_catalog(catalog_identity);
}

bool Sm87MacroFeedV3P40ProjectionStartupBinding::valid_for(
    const std::size_t layer_index, const Role role,
    const std::uint64_t package_identity) const noexcept {
  return layer_index == snapshot_.layer_index && role == snapshot_.role &&
         package_identity != 0U &&
         package_identity == snapshot_.package_identity && valid();
}

const kernels::Sm87TargetAotNvFp4CudaAssetView*
Sm87MacroFeedV3P40ProjectionStartupBinding::borrow_nvfp4_asset()
    const noexcept {
  return valid() && sm87_target_aot_complete_role_is_nvfp4(snapshot_.role)
             ? asset_.borrow_nvfp4_cuda_asset()
             : nullptr;
}

const kernels::Sm87TargetAotFp8CudaAssetView*
Sm87MacroFeedV3P40ProjectionStartupBinding::borrow_fp8_asset()
    const noexcept {
  return valid() && sm87_target_aot_complete_role_is_fp8(snapshot_.role)
             ? asset_.borrow_fp8_cuda_asset()
             : nullptr;
}

const kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt*
Sm87MacroFeedV3P40ProjectionStartupBinding::gate_up_payload_receipt()
    const noexcept {
  return valid() && snapshot_.role == Role::kNvFp4GateUp
             ? &snapshot_.gate_up_receipt
             : nullptr;
}

const kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt*
Sm87MacroFeedV3P40ProjectionStartupBinding::down_payload_receipt()
    const noexcept {
  return valid() && snapshot_.role == Role::kNvFp4Down
             ? &snapshot_.down_receipt
             : nullptr;
}

bool Sm87MacroFeedV3P40ExecutionPackage::startup_seals_valid(
    const StartupSeals& seals, const std::int32_t device_ordinal) noexcept {
  if (device_ordinal < 0 ||
      !kernels::sm87_macrofeed_v3_nvfp4_gate_up_startup_seal_valid(
          seals.gate_up) ||
      seals.gate_up.device_ordinal != device_ordinal ||
      !seals.down.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v3_nvfp4_down_resource_gate(seals.down) ||
      seals.down.device_ordinal != device_ordinal ||
      !kernels::sm87_macrofeed_v3_gdn_resources_valid(seals.gdn)) {
    return false;
  }
  constexpr std::array<Role, 3U> kRoles{{
      Role::kFp8GdnQkvZ,
      Role::kFp8FullQkv,
      Role::kFp8AttentionOutput,
  }};
  for (std::size_t index = 0U; index < kRoles.size(); ++index) {
    if (!kernels::sm87_macrofeed_v3_fp8_startup_seal_valid(
            seals.fp8[index]) ||
        seals.fp8[index].resources.role != kRoles[index] ||
        seals.fp8[index].resources.device_ordinal != device_ordinal) {
      return false;
    }
  }
  return true;
}

std::uint64_t Sm87MacroFeedV3P40ExecutionPackage::compute_package_identity(
    const ProjectionAccess& access,
    const std::array<AssetCapability,
                     kSm87MacroFeedV3P40PackageArtifacts>& capabilities,
    const StartupSeals& seals, const std::size_t sources) noexcept {
  const std::uint64_t catalog = access.catalog_identity();
  if (!access.attached() || access.owner_identity() == 0U ||
      access.allocation_identity() == 0U ||
      access.device_identity() == 0U || access.device_ordinal() < 0 ||
      catalog == 0U || sources != kSm87MacroFeedV3P40PackageSources ||
      !startup_seals_valid(seals, access.device_ordinal())) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5633'504bULL;
  identity = mix(identity, access.owner_identity());
  identity = mix(identity, access.allocation_identity());
  identity = mix(identity, access.device_identity());
  identity = mix(identity, catalog);
  identity = mix(identity,
                 static_cast<std::uint64_t>(access.device_ordinal() + 1));
  identity = mix(identity, capabilities.size());
  identity = mix(identity, sources);
  for (std::size_t index = 0U; index < capabilities.size(); ++index) {
    const auto& capability = capabilities[index];
    if (!capability.asset || capability.artifact_identity == 0U ||
        capability.source_inventory_identity == 0U ||
        capability.source_manifest_seal == 0U ||
        capability.payload_bytes == 0U || capability.source_count == 0U) {
      return 0U;
    }
    identity = mix(identity, index + 1U);
    identity = mix(identity, capability.layer_index + 1U);
    identity = mix(identity, static_cast<std::uint64_t>(capability.role));
    identity = mix(identity,
                   static_cast<std::uint64_t>(capability.encoding));
    identity = mix(identity, capability.artifact_identity);
    identity = mix(identity, capability.source_inventory_identity);
    identity = mix(identity, capability.source_manifest_seal);
    identity = mix(identity, capability.payload_bytes);
    identity = mix(identity, capability.source_count);
  }
  identity = mix(identity, seals.gate_up.seal_identity);
  const auto& down = seals.down;
  identity = mix(identity, static_cast<std::uint64_t>(down.identity));
  identity = mix(identity,
                 static_cast<std::uint64_t>(down.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(down.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(down.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(down.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(down.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(down.registers_per_thread));
  identity = mix(identity, down.static_shared_bytes);
  identity = mix(identity, down.dynamic_shared_bytes);
  identity = mix(identity, down.local_bytes);
  identity = mix(
      identity, static_cast<std::uint64_t>(down.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(down.active_blocks_per_sm));
  identity = mix(identity, down.optin_shared_bytes_per_block);
  identity = mix(identity, down.kernel_compiled);
  identity = mix(identity, down.static_resource_gate_passed);
  identity = mix(identity, down.numerical_contract_qualified);
  identity = mix(identity, down.production_dispatch_eligible);

  const auto mix_gdn_kernel = [&identity](
                                  const kernels::
                                      Sm87MacrofeedV3GdnKernelResources&
                                          resources) noexcept {
    identity = mix(identity,
                   static_cast<std::uint64_t>(resources.registers_per_thread));
    identity = mix(identity, resources.static_shared_bytes);
    identity = mix(identity, resources.local_bytes);
    identity = mix(
        identity,
        static_cast<std::uint64_t>(resources.maximum_threads_per_block));
    identity = mix(identity,
                   static_cast<std::uint64_t>(resources.active_blocks_per_sm));
    identity = mix(identity,
                   static_cast<std::uint64_t>(resources.threads_per_block));
    identity = mix(identity,
                   static_cast<std::uint64_t>(resources.physical_grid_ctas));
  };
  identity = mix(identity,
                 static_cast<std::uint64_t>(seals.gdn.binary_version));
  mix_gdn_kernel(seals.gdn.convolution);
  mix_gdn_kernel(seals.gdn.recurrence_epilogue);
  identity = mix(identity, seals.gdn.kernels_compiled);
  identity = mix(identity, seals.gdn.exact_geometry);
  identity = mix(identity, seals.gdn.resource_gate_passed);
  identity = mix(identity, seals.gdn.numerical_contract_qualified);
  identity = mix(identity, seals.gdn.production_dispatch_eligible);
  for (const auto& seal : seals.fp8) {
    identity = mix(identity, seal.seal_identity);
  }
  return identity == 0U ? 0x5133'4d46'5633'504bULL : identity;
}

Sm87MacroFeedV3P40ExecutionPackageCreateResult
Sm87MacroFeedV3P40ExecutionPackage::build_from_private_authority(
    ProjectionAccess access, StartupSeals seals) noexcept {
  CreateResult result;
  if (!access.attached() || access.artifact_count() !=
                                kSm87MacroFeedV3P40PackageArtifacts) {
    result.status =
        failure(Error::kProjectionAttachment, "projection_attachment");
    return result;
  }
  const std::uint64_t owner_identity = access.owner_identity();
  const std::uint64_t allocation_identity = access.allocation_identity();
  const std::uint64_t catalog_identity = access.catalog_identity();
  const std::uint64_t device_identity = access.device_identity();
  const std::int32_t device_ordinal = access.device_ordinal();
  if (owner_identity == 0U || allocation_identity == 0U ||
      catalog_identity == 0U || device_identity == 0U ||
      device_ordinal < 0) {
    result.status =
        failure(Error::kProjectionCatalog, "projection_catalog");
    return result;
  }
  if (!startup_seals_valid(seals, device_ordinal)) {
    result.status = failure(Error::kDeviceMismatch, "startup_seal_device");
    return result;
  }

  std::array<AssetCapability, kSm87MacroFeedV3P40PackageArtifacts>
      capabilities{};
  std::array<std::uint64_t, kSm87MacroFeedV3P40PackageArtifacts>
      artifact_identities{};
  std::array<std::uint64_t, kSm87MacroFeedV3P40PackageArtifacts>
      inventory_identities{};
  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  std::size_t gate_up_assets = 0U;
  std::size_t down_assets = 0U;
  std::size_t gdn_assets = 0U;
  std::size_t full_assets = 0U;
  std::size_t output_assets = 0U;

  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV3P40PackageLayers; ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != artifacts || index >= capabilities.size()) {
        result.status = failure(Error::kProjectionInventory,
                                "projection_descriptor_order", 0,
                                layer_index, role);
        return result;
      }
      auto asset = access.resolve(layer_index, role);
      if (!asset) {
        result.status = failure(Error::kProjectionAssetResolve,
                                "projection_asset_resolve", 0,
                                layer_index, role);
        return result;
      }
      const auto layout = kernels::sm87_target_aot_projection_packed_layout(role);
      if (!layout.valid() || asset->layer_index() != layer_index ||
          asset->role() != role ||
          asset->artifact_identity() == 0U ||
          asset->source_inventory_identity() == 0U ||
          asset->payload_bytes() != layout.payload_bytes ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + artifacts,
                    asset->artifact_identity()) !=
              artifact_identities.begin() + artifacts ||
          std::find(inventory_identities.begin(),
                    inventory_identities.begin() + artifacts,
                    asset->source_inventory_identity()) !=
              inventory_identities.begin() + artifacts) {
        result.status = failure(Error::kProjectionInventory,
                                "projection_asset_identity", 0,
                                layer_index, role);
        return result;
      }
      bool typed_borrow_valid = false;
      std::uint64_t source_manifest_seal = 0U;
      std::uint32_t source_count = 0U;
      if (sm87_target_aot_complete_role_is_nvfp4(role)) {
        const auto* const view = asset->borrow_nvfp4_cuda_asset();
        typed_borrow_valid =
            view != nullptr &&
            kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) &&
            view->artifact_identity == asset->artifact_identity() &&
            view->source_inventory_identity ==
                asset->source_inventory_identity();
        if (typed_borrow_valid) {
          source_manifest_seal = view->host_manifest_seal.value;
          source_count = view->tensor_scale_count;
        }
      } else {
        const auto* const view = asset->borrow_fp8_cuda_asset();
        typed_borrow_valid =
            view != nullptr &&
            kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) &&
            view->artifact_identity == asset->artifact_identity() &&
            view->source_inventory_identity ==
                asset->source_inventory_identity();
        if (typed_borrow_valid) {
          source_manifest_seal = view->host_manifest_seal.value;
          source_count = view->tensor_scale_count;
        }
      }
      if (!typed_borrow_valid || source_manifest_seal == 0U ||
          source_count != layout.partition_count) {
        result.status = failure(Error::kProjectionAssetBorrow,
                                "projection_asset_borrow", 0,
                                layer_index, role);
        return result;
      }

      auto& capability = capabilities[index];
      capability.layer_index = layer_index;
      capability.role = role;
      capability.encoding = asset->encoding();
      capability.artifact_identity = asset->artifact_identity();
      capability.source_inventory_identity =
          asset->source_inventory_identity();
      capability.source_manifest_seal = source_manifest_seal;
      capability.payload_bytes = asset->payload_bytes();
      capability.source_count = source_count;
      capability.asset = std::move(*asset);
      artifact_identities[artifacts] = capability.artifact_identity;
      inventory_identities[artifacts] =
          capability.source_inventory_identity;
      sources += capability.source_count;
      ++artifacts;
      if (role == Role::kNvFp4GateUp) {
        ++gate_up_assets;
      } else if (role == Role::kNvFp4Down) {
        ++down_assets;
      } else if (role == Role::kFp8GdnQkvZ) {
        ++gdn_assets;
      } else if (role == Role::kFp8FullQkv) {
        ++full_assets;
      } else if (role == Role::kFp8AttentionOutput) {
        ++output_assets;
      }
    }
  }
  if (artifacts != kSm87MacroFeedV3P40PackageArtifacts ||
      sources != kSm87MacroFeedV3P40PackageSources ||
      gate_up_assets != kSm87MacroFeedV3P40PackageLayers ||
      down_assets != kSm87MacroFeedV3P40PackageLayers ||
      gdn_assets != kSm87MacroFeedV3P40PackageGdnLayers ||
      full_assets != kSm87MacroFeedV3P40PackageFullLayers ||
      output_assets != kSm87MacroFeedV3P40PackageLayers ||
      access.catalog_identity() != catalog_identity) {
    result.status =
        failure(Error::kProjectionInventory, "projection_inventory");
    return result;
  }

  Sm87MacroFeedV3P40ExecutionPackageAudit audit;
  audit.owner_identity = owner_identity;
  audit.allocation_identity = allocation_identity;
  audit.catalog_identity = catalog_identity;
  audit.device_identity = device_identity;
  audit.device_ordinal = device_ordinal;
  audit.layers = kSm87MacroFeedV3P40PackageLayers;
  audit.artifacts = artifacts;
  audit.sources = sources;
  audit.gate_up_assets = gate_up_assets;
  audit.down_assets = down_assets;
  audit.gdn_projection_assets = gdn_assets;
  audit.full_projection_assets = full_assets;
  audit.attention_output_assets = output_assets;
  audit.complete_projection_access_retained = true;
  audit.catalog_revalidated = true;
  audit.typed_capabilities_retained = true;
  audit.authenticated_source_manifests_retained = true;
  audit.startup_seals_complete = true;
  audit.caller_raw_receipts_accepted = false;
  audit.v2_owner_or_executor_reused = false;
  audit.request_time_repack_jit_or_fallback_permitted = false;
  audit.t0_t1_only = true;
  audit.production_dispatch_eligible = false;
  audit.package_identity = compute_package_identity(
      access, capabilities, seals, sources);
  if (!audit.valid()) {
    result.status = failure(Error::kPackageIdentity, "package_identity");
    return result;
  }

  auto package = std::unique_ptr<Package>(new (std::nothrow) Package(
      std::move(access), std::move(capabilities), std::move(seals), audit));
  if (!package) {
    result.status = failure(Error::kAllocationFailure, "package_allocation");
    return result;
  }
  if (!package->populate_projection_startup_bindings() ||
      !package->valid()) {
    result.status = failure(Error::kPackageIdentity, "package_revalidation");
    return result;
  }
  result.audit = package->audit();
  result.package = std::move(package);
  result.status = {};
  return result;
}

const Sm87MacroFeedV3P40ExecutionPackage::AssetCapability*
Sm87MacroFeedV3P40ExecutionPackage::capability(
    const std::size_t layer_index, const Role role) const noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (index >= capabilities_.size() || !projection_access_.attached()) {
    return nullptr;
  }
  const auto& capability = capabilities_[index];
  return capability.asset && capability.layer_index == layer_index &&
                 capability.role == role &&
                 capability.artifact_identity != 0U &&
                 capability.source_inventory_identity != 0U &&
                 capability.source_manifest_seal != 0U &&
                 capability.payload_bytes != 0U &&
                 capability.source_count != 0U
             ? &capability
             : nullptr;
}

const kernels::Sm87TargetAotNvFp4CudaAssetView*
Sm87MacroFeedV3P40ExecutionPackage::borrow_nvfp4_asset(
    const std::size_t layer_index, const Role role) const noexcept {
  if (!sm87_target_aot_complete_role_is_nvfp4(role)) {
    return nullptr;
  }
  const AssetCapability* const item = capability(layer_index, role);
  if (item == nullptr ||
      item->encoding !=
          Encoding::kNvFp4E2M1Block16E4M3FnScale) {
    return nullptr;
  }
  const auto* const view = item->asset->borrow_nvfp4_cuda_asset();
  return view != nullptr &&
                 kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) &&
                 view->artifact_identity == item->artifact_identity &&
                 view->source_inventory_identity ==
                     item->source_inventory_identity &&
                 view->host_manifest_seal.value ==
                     item->source_manifest_seal &&
                 view->tensor_scale_count == item->source_count
             ? view
             : nullptr;
}

const kernels::Sm87TargetAotFp8CudaAssetView*
Sm87MacroFeedV3P40ExecutionPackage::borrow_fp8_asset(
    const std::size_t layer_index, const Role role) const noexcept {
  if (!sm87_target_aot_complete_role_is_fp8(role)) {
    return nullptr;
  }
  const AssetCapability* const item = capability(layer_index, role);
  if (item == nullptr ||
      item->encoding != Encoding::kFp8E4M3FnTensorScale) {
    return nullptr;
  }
  const auto* const view = item->asset->borrow_fp8_cuda_asset();
  return view != nullptr &&
                 kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) &&
                 view->artifact_identity == item->artifact_identity &&
                 view->source_inventory_identity ==
                     item->source_inventory_identity &&
                 view->host_manifest_seal.value ==
                     item->source_manifest_seal &&
                 view->tensor_scale_count == item->source_count
             ? view
             : nullptr;
}

std::optional<Sm87MacroFeedV3P40ProjectionStartupBinding>
Sm87MacroFeedV3P40ExecutionPackage::make_projection_startup_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  if (!audit_.valid() || !projection_access_.attached() ||
      projection_access_.owner_identity() != audit_.owner_identity ||
      projection_access_.allocation_identity() !=
          audit_.allocation_identity ||
      projection_access_.device_identity() != audit_.device_identity ||
      projection_access_.device_ordinal() != audit_.device_ordinal) {
    return std::nullopt;
  }
  const AssetCapability* const retained = capability(layer_index, role);
  auto asset = projection_access_.resolve(layer_index, role);
  if (retained == nullptr || !asset || !retained->asset ||
      asset->artifact_identity() != retained->artifact_identity ||
      asset->source_inventory_identity() !=
          retained->source_inventory_identity ||
      asset->payload_bytes() != retained->payload_bytes ||
      asset->encoding() != retained->encoding) {
    return std::nullopt;
  }

  const auto* const descriptor = asset->descriptor_;
  const auto layout = kernels::sm87_target_aot_projection_packed_layout(role);
  if (descriptor == nullptr || !layout.valid() ||
      descriptor->layer_index != layer_index || descriptor->role != role ||
      descriptor->encoding != retained->encoding ||
      descriptor->manifest.artifact_identity !=
          retained->artifact_identity ||
      descriptor->manifest.seal.value != retained->source_manifest_seal ||
      descriptor->source_inventory.identity !=
          retained->source_inventory_identity ||
      descriptor->source_inventory.source_count != retained->source_count ||
      !descriptor->source_inventory.valid(layout) ||
      !kernels::sm87_target_aot_projection_validate_packed_manifest(
          descriptor->manifest, descriptor->source_inventory)) {
    return std::nullopt;
  }

  Sm87MacroFeedV3P40ProjectionStartupBinding::Snapshot snapshot;
  snapshot.package_identity = audit_.package_identity;
  snapshot.owner_identity = audit_.owner_identity;
  snapshot.allocation_identity = audit_.allocation_identity;
  snapshot.catalog_identity = audit_.catalog_identity;
  snapshot.device_identity = audit_.device_identity;
  snapshot.artifact_identity = retained->artifact_identity;
  snapshot.manifest_seal = retained->source_manifest_seal;
  snapshot.device_ordinal = audit_.device_ordinal;
  snapshot.layer_index = layer_index;
  snapshot.role = role;
  snapshot.encoding = retained->encoding;
  snapshot.source_inventory = descriptor->source_inventory;
  snapshot.issued_from_live_private_descriptor = true;
  snapshot.caller_receipt_accepted = false;
  snapshot.t0_t1_only = true;
  snapshot.production_dispatch_eligible = false;

  const auto set_common_view = [&](const auto& view) noexcept {
    snapshot.upload_receipt_identity =
        view.device_upload_receipt.receipt_identity;
    snapshot.payload_begin = view.payload.begin;
    snapshot.payload_end = view.payload.end;
    snapshot.payload_bytes = view.payload.bytes;
  };
  if (sm87_target_aot_complete_role_is_nvfp4(role)) {
    const auto* const view = asset->borrow_nvfp4_cuda_asset();
    if (view == nullptr || asset->borrow_fp8_cuda_asset() != nullptr ||
        !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) ||
        view->tensor_scale_count != snapshot.source_inventory.source_count ||
        !authenticated_upload_complete(view->device_upload_receipt)) {
      return std::nullopt;
    }
    for (std::size_t index = 0U; index < view->tensor_scale_count; ++index) {
      if (view->tensor_scale_bits[index] !=
          snapshot.source_inventory.sources[index].tensor_scale_bits) {
        return std::nullopt;
      }
    }
    set_common_view(*view);
    const auto& upload = view->device_upload_receipt;
    if (role == Role::kNvFp4GateUp) {
      if (snapshot.source_inventory.source_count != 2U ||
          snapshot.source_inventory.sources[0U].logical_role !=
              LogicalRole::kNvFp4Gate ||
          snapshot.source_inventory.sources[1U].logical_role !=
              LogicalRole::kNvFp4Up ||
          !layout.partitions[0U].independent_tensor_scale ||
          !layout.partitions[1U].independent_tensor_scale) {
        return std::nullopt;
      }
      auto& receipt = snapshot.gate_up_receipt;
      receipt.plan_identity =
          kernels::kSm87MacroFeedV3NvFp4GateUpIdentity;
      receipt.payload_identity = snapshot.artifact_identity;
      receipt.gate_source_identity =
          snapshot.source_inventory.sources[0U].tensor_identity;
      receipt.up_source_identity =
          snapshot.source_inventory.sources[1U].tensor_identity;
      receipt.device_ordinal = snapshot.device_ordinal;
      receipt.payload_begin = snapshot.payload_begin;
      receipt.payload_end = snapshot.payload_end;
      receipt.payload_bytes = snapshot.payload_bytes;
      receipt.gate_partition_bytes = layout.partitions[0U].payload_bytes;
      receipt.up_partition_bytes = layout.partitions[1U].payload_bytes;
      receipt.canonical_consumer_n64_k16_lane_component_v1 =
          view->transform_identity ==
          kernels::Sm87TargetAotProjectionPackedTransformIdentity::
              kCanonicalNkToConsumerN64K16LaneComponentV1;
      receipt.canonical_gate_then_up_partition_order = true;
      receipt.independent_tensor_scales = true;
      receipt.host_bytes_authenticated_before_copy =
          upload.host_payload_digest_verified_before_copy &&
          upload.host_payload_immutable_until_completion;
      receipt.device_readback_authenticated =
          upload.verification_event_observed &&
          upload.verification_completed &&
          upload.device_payload_matches_host_payload;
      receipt.allocation_retained_for_launch =
          upload.allocation_retained_for_asset_lifetime;
      receipt.receipt_identity = kernels::
          sm87_macrofeed_v3_nvfp4_gate_up_compute_payload_receipt_identity(
              receipt);
    } else if (role == Role::kNvFp4Down) {
      if (snapshot.source_inventory.source_count != 1U ||
          snapshot.source_inventory.sources[0U].logical_role !=
              LogicalRole::kNvFp4Down) {
        return std::nullopt;
      }
      auto& receipt = snapshot.down_receipt;
      receipt.plan_identity = kernels::kSm87MacroFeedV3NvFp4DownIdentity;
      receipt.payload_identity = snapshot.artifact_identity;
      receipt.device_ordinal = snapshot.device_ordinal;
      receipt.payload_begin = snapshot.payload_begin;
      receipt.payload_end = snapshot.payload_end;
      receipt.payload_bytes = snapshot.payload_bytes;
      receipt.canonical_consumer_n64_k16_lane_component_v1 =
          view->transform_identity ==
          kernels::Sm87TargetAotProjectionPackedTransformIdentity::
              kCanonicalNkToConsumerN64K16LaneComponentV1;
      receipt.host_bytes_authenticated_before_copy =
          upload.host_payload_digest_verified_before_copy &&
          upload.host_payload_immutable_until_completion;
      receipt.device_readback_authenticated =
          upload.verification_event_observed &&
          upload.verification_completed &&
          upload.device_payload_matches_host_payload;
      receipt.allocation_retained_for_launch =
          upload.allocation_retained_for_asset_lifetime;
      receipt.receipt_identity =
          kernels::sm87_macrofeed_v3_nvfp4_down_compute_payload_receipt_identity(
              receipt);
    } else {
      return std::nullopt;
    }
  } else if (sm87_target_aot_complete_role_is_fp8(role)) {
    const auto* const view = asset->borrow_fp8_cuda_asset();
    if (view == nullptr || asset->borrow_nvfp4_cuda_asset() != nullptr ||
        !kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) ||
        view->tensor_scale_count != snapshot.source_inventory.source_count ||
        !authenticated_upload_complete(view->device_upload_receipt)) {
      return std::nullopt;
    }
    for (std::size_t index = 0U; index < view->tensor_scale_count; ++index) {
      if (view->tensor_scale_bits[index] !=
          snapshot.source_inventory.sources[index].tensor_scale_bits) {
        return std::nullopt;
      }
    }
    set_common_view(*view);
  } else {
    return std::nullopt;
  }

  snapshot.binding_identity =
      Sm87MacroFeedV3P40ProjectionStartupBinding::compute_binding_identity(
          snapshot);
  if (snapshot.binding_identity == 0U) {
    return std::nullopt;
  }
  Sm87MacroFeedV3P40ProjectionStartupBinding binding(
      projection_access_, std::move(*asset), std::move(snapshot));
  if (!binding.valid_with_authenticated_catalog(audit_.catalog_identity)) {
    return std::nullopt;
  }
  return std::optional<Sm87MacroFeedV3P40ProjectionStartupBinding>(
      std::move(binding));
}

const kernels::Sm87MacroFeedV3Fp8StartupSeal*
Sm87MacroFeedV3P40ExecutionPackage::fp8_startup_seal(
    const Role role) const noexcept {
  const std::size_t index = fp8_seal_index(role);
  return index < seals_.fp8.size() &&
                 kernels::sm87_macrofeed_v3_fp8_startup_seal_valid(
                     seals_.fp8[index])
             ? &seals_.fp8[index]
             : nullptr;
}

bool Sm87MacroFeedV3P40ExecutionPackage::base_valid() const noexcept {
  if (!audit_.valid() || !projection_access_.attached() ||
      projection_access_.owner_identity() != audit_.owner_identity ||
      projection_access_.allocation_identity() !=
          audit_.allocation_identity ||
      projection_access_.device_identity() != audit_.device_identity ||
      projection_access_.device_ordinal() != audit_.device_ordinal ||
      projection_access_.artifact_count() != audit_.artifacts ||
      projection_access_.catalog_identity() != audit_.catalog_identity ||
      !startup_seals_valid(seals_, audit_.device_ordinal)) {
    return false;
  }
  std::size_t sources = 0U;
  std::size_t artifacts = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV3P40PackageLayers; ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != artifacts || index >= capabilities_.size()) {
        return false;
      }
      const auto& item = capabilities_[index];
      const auto layout = kernels::sm87_target_aot_projection_packed_layout(role);
      if (!item.asset || !layout.valid() ||
          item.layer_index != layer_index || item.role != role ||
          item.artifact_identity != item.asset->artifact_identity() ||
          item.source_inventory_identity !=
              item.asset->source_inventory_identity() ||
          item.source_manifest_seal == 0U ||
          item.payload_bytes != layout.payload_bytes ||
          item.source_count != layout.partition_count) {
        return false;
      }
      if (sm87_target_aot_complete_role_is_nvfp4(role)) {
        if (borrow_nvfp4_asset(layer_index, role) == nullptr) {
          return false;
        }
      } else if (borrow_fp8_asset(layer_index, role) == nullptr) {
        return false;
      }
      sources += item.source_count;
      ++artifacts;
    }
  }
  return artifacts == audit_.artifacts && sources == audit_.sources &&
         audit_.package_identity == compute_package_identity(
                                        projection_access_, capabilities_,
                                        seals_, sources);
}

bool Sm87MacroFeedV3P40ExecutionPackage::
    populate_projection_startup_bindings() noexcept {
  if (!base_valid()) {
    return false;
  }
  std::size_t bindings = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV3P40PackageLayers; ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != bindings || index >= startup_bindings_.size() ||
          startup_bindings_[index].has_value()) {
        return false;
      }
      auto binding = make_projection_startup_binding(layer_index, role);
      if (!binding) {
        return false;
      }
      startup_bindings_[index].emplace(std::move(*binding));
      ++bindings;
    }
  }
  return bindings == startup_bindings_.size();
}

bool Sm87MacroFeedV3P40ExecutionPackage::valid() const noexcept {
  if (!base_valid()) {
    return false;
  }
  std::size_t bindings = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV3P40PackageLayers; ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != bindings || index >= startup_bindings_.size() ||
          !startup_bindings_[index] ||
          startup_bindings_[index]->package_identity() !=
              audit_.package_identity ||
          startup_bindings_[index]->layer_index() != layer_index ||
          startup_bindings_[index]->role() != role ||
          !startup_bindings_[index]->valid_with_authenticated_catalog(
              audit_.catalog_identity)) {
        return false;
      }
      ++bindings;
    }
  }
  return bindings == startup_bindings_.size();
}

const Sm87MacroFeedV3P40ProjectionStartupBinding*
Sm87MacroFeedV3P40ExecutionPackage::borrow_projection_startup_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (index >= startup_bindings_.size() || !startup_bindings_[index] ||
      !audit_.valid() || !projection_access_.attached() ||
      projection_access_.owner_identity() != audit_.owner_identity ||
      projection_access_.allocation_identity() !=
          audit_.allocation_identity ||
      projection_access_.device_identity() != audit_.device_identity ||
      projection_access_.device_ordinal() != audit_.device_ordinal ||
      !startup_seals_valid(seals_, audit_.device_ordinal)) {
    return nullptr;
  }
  const auto& binding = *startup_bindings_[index];
  return binding.layer_index() == layer_index && binding.role() == role &&
                 binding.package_identity() == audit_.package_identity &&
                 binding.valid_with_authenticated_catalog(
                     audit_.catalog_identity)
             ? &binding
             : nullptr;
}
#else
const Sm87MacroFeedV3P40ProjectionStartupBinding::SourceBinding*
Sm87MacroFeedV3P40ProjectionStartupBinding::source(
    const std::size_t source_index) const noexcept {
  (void)source_index;
  return nullptr;
}

float Sm87MacroFeedV3P40ProjectionStartupBinding::tensor_scale(
    const std::size_t source_index) const noexcept {
  (void)source_index;
  return 0.0F;
}

bool Sm87MacroFeedV3P40ProjectionStartupBinding::valid() const noexcept {
  return false;
}

bool Sm87MacroFeedV3P40ProjectionStartupBinding::
    valid_with_authenticated_catalog(
        const std::uint64_t catalog_identity) const noexcept {
  (void)catalog_identity;
  return false;
}

bool Sm87MacroFeedV3P40ProjectionStartupBinding::valid_for(
    const std::size_t layer_index, const Role role,
    const std::uint64_t package_identity) const noexcept {
  (void)layer_index;
  (void)role;
  (void)package_identity;
  return false;
}

const kernels::Sm87TargetAotNvFp4CudaAssetView*
Sm87MacroFeedV3P40ProjectionStartupBinding::borrow_nvfp4_asset()
    const noexcept {
  return nullptr;
}

const kernels::Sm87TargetAotFp8CudaAssetView*
Sm87MacroFeedV3P40ProjectionStartupBinding::borrow_fp8_asset()
    const noexcept {
  return nullptr;
}

const kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt*
Sm87MacroFeedV3P40ProjectionStartupBinding::gate_up_payload_receipt()
    const noexcept {
  return nullptr;
}

const kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt*
Sm87MacroFeedV3P40ProjectionStartupBinding::down_payload_receipt()
    const noexcept {
  return nullptr;
}

const kernels::Sm87TargetAotNvFp4CudaAssetView*
Sm87MacroFeedV3P40ExecutionPackage::borrow_nvfp4_asset(
    const std::size_t layer_index, const Role role) const noexcept {
  (void)layer_index;
  (void)role;
  return nullptr;
}

const kernels::Sm87TargetAotFp8CudaAssetView*
Sm87MacroFeedV3P40ExecutionPackage::borrow_fp8_asset(
    const std::size_t layer_index, const Role role) const noexcept {
  (void)layer_index;
  (void)role;
  return nullptr;
}

const Sm87MacroFeedV3P40ProjectionStartupBinding*
Sm87MacroFeedV3P40ExecutionPackage::borrow_projection_startup_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  (void)layer_index;
  (void)role;
  return nullptr;
}

const kernels::Sm87MacroFeedV3Fp8StartupSeal*
Sm87MacroFeedV3P40ExecutionPackage::fp8_startup_seal(
    const Role role) const noexcept {
  (void)role;
  return nullptr;
}

bool Sm87MacroFeedV3P40ExecutionPackage::valid() const noexcept {
  return false;
}

bool Sm87MacroFeedV3P40ExecutionPackage::base_valid() const noexcept {
  return false;
}

bool Sm87MacroFeedV3P40ExecutionPackage::
    populate_projection_startup_bindings() noexcept {
  return false;
}

std::optional<Sm87MacroFeedV3P40ProjectionStartupBinding>
Sm87MacroFeedV3P40ExecutionPackage::make_projection_startup_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  (void)layer_index;
  (void)role;
  return std::nullopt;
}
#endif

}  // namespace q3x::runtime::sm87_macrofeed_v3_p40_execution_package_detail
