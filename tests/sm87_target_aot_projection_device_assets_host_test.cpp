#include "q3x/runtime/sm87_target_aot_projection_device_assets.h"
#include "../src/runtime/sm87_target_aot_projection_execution_access_internal.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>

namespace kernels = q3x::kernels;
namespace runtime = q3x::runtime;
namespace execution = q3x::runtime::target_aot_execution_detail;

namespace q3x::runtime::target_aot_execution_detail {
namespace {

using Role = kernels::Sm87TargetAotProjectionRole;
using Descriptor = Sm87TargetAotProjectionDeviceAssetDescriptor;
using Owner = Sm87TargetAotProjectionDeviceAssets;

inline constexpr std::uintptr_t kHostTestArenaBegin =
    0x0000'0040'0000'0000ULL;
inline constexpr std::uint64_t kHostTestOwnerIdentity =
    0x7133'7854'4553'544fULL;
inline constexpr std::uint64_t kHostTestAllocationIdentity =
    0x7133'7854'4553'5441ULL;

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
  receipt.payload = {manifest.artifact_identity,
                     manifest.payload_offset,
                     manifest.payload_bytes,
                     manifest.payload_digest,
                     true};
  receipt.deterministic_transform = true;
  receipt.no_arithmetic_conversion = true;
  receipt.no_request_time_repacking = true;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const auto& partition = layout.partitions[index];
    const auto& source = inventory.sources[index];
    auto& observed = receipt.partitions[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t block_scale_values =
        partition.block_scale_group_k == 0U
            ? 0U
            : values / partition.block_scale_group_k;
    observed.logical_role = partition.logical_role;
    observed.partition_index = static_cast<std::uint32_t>(index);
    observed.tensor_identity = source.tensor_identity;
    observed.observed_source_weight_digest = source.weight_digest;
    observed.observed_source_scale_digest = source.scale_digest;
    observed.source_weight_bytes_hashed =
        values * partition.weight_bits / 8U;
    observed.source_scale_bytes_hashed =
        block_scale_values + sizeof(std::uint32_t);
    observed.repacked_weight_values = values;
    observed.repacked_block_scale_values = block_scale_values;
    observed.source_block_scale_e4m3fn_bytes_scanned = block_scale_values;
    observed.payload_block_scale_e4m3fn_bytes_scanned = block_scale_values;
    observed.payload_offset = partition.payload_offset;
    observed.payload_bytes = partition.payload_bytes;
    observed.source_digests_computed_from_tensor_bytes = true;
    observed.canonical_address_bijection_applied = true;
    observed.bit_exact_weight_permutation = true;
    observed.bit_exact_block_scale_permutation = block_scale_values != 0U;
    observed.tensor_scale_kept_external = true;
  }
  return receipt;
}

[[nodiscard]] bool make_host_test_descriptor(
    const std::size_t layer_index, const Role role,
    const std::uint64_t arena_offset, Descriptor& descriptor) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(role);
  const std::size_t role_index = role == Role::kNvFp4GateUp ? 0U : 1U;
  const std::size_t ordinal = 2U * layer_index + role_index;
  if (!layout.valid() ||
      (role != Role::kNvFp4GateUp && role != Role::kNvFp4Down) ||
      ordinal >= kSm87TargetAotProjectionDeviceArtifactCount) {
    return false;
  }

  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = 0x1000'0000'0000'0000ULL + ordinal + 1U;
  inventory.role = role;
  inventory.source_count = layout.partition_count;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const std::uint64_t source_ordinal = 3U * ordinal + index + 1U;
    inventory.sources[index] =
        kernels::sm87_target_aot_projection_packed_source_binding(
            layout, index, 0x2000'0000'0000'0000ULL + source_ordinal,
            host_test_digest(0x1000U + 2U * source_ordinal),
            host_test_digest(0x1001U + 2U * source_ordinal),
            0x3f80'0000U);
  }
  if (!inventory.valid(layout)) {
    return false;
  }

  const auto manifest =
      kernels::sm87_target_aot_projection_make_packed_manifest(
          role, 0x3000'0000'0000'0000ULL + ordinal + 1U, inventory,
          host_test_digest(0x8000U + ordinal));
  if (!kernels::sm87_target_aot_projection_validate_packed_manifest(
          manifest, inventory)) {
    return false;
  }
  const auto transform =
      make_host_test_transform_receipt(layout, inventory, manifest);
  if (!kernels::sm87_target_aot_projection_validate_transform_receipt(
          manifest, inventory, transform)) {
    return false;
  }

  kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt upload;
  upload.artifact_identity = manifest.artifact_identity;
  upload.source_inventory_identity = inventory.identity;
  upload.role = role;
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
      kHostTestArenaBegin +
      static_cast<std::uintptr_t>(kSm87TargetAotProjectionDeviceArenaBytes);
  upload.device_allocation_bytes =
      kSm87TargetAotProjectionDeviceArenaBytes;
  upload.device_payload_begin =
      kHostTestArenaBegin + static_cast<std::uintptr_t>(arena_offset);
  upload.device_payload_end =
      upload.device_payload_begin +
      static_cast<std::uintptr_t>(manifest.payload_bytes);
  upload.device_payload_bytes = manifest.payload_bytes;
  upload.upload_stream_owner_identity = kHostTestOwnerIdentity;
  upload.upload_stream_identity = 0x4000'0000'0000'0000ULL + ordinal + 1U;
  upload.upload_completion_event_identity =
      0x5000'0000'0000'0000ULL + ordinal + 1U;
  upload.verification_stream_owner_identity = kHostTestOwnerIdentity;
  upload.verification_stream_identity =
      0x6000'0000'0000'0000ULL + ordinal + 1U;
  upload.verification_completion_event_identity =
      0x7000'0000'0000'0000ULL + ordinal + 1U;
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
  upload.receipt_identity =
      kernels::sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(
          upload);
  if (!kernels::
          sm87_target_aot_nvfp4_cuda_device_upload_receipt_structurally_valid(
              manifest, upload)) {
    return false;
  }

  const auto view = kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
      manifest, inventory, transform, upload);
  if (!kernels::sm87_target_aot_nvfp4_cuda_asset_valid(view)) {
    return false;
  }
  descriptor = {layer_index, role, arena_offset, inventory, manifest,
                transform, upload, view};
  return true;
}

}  // namespace

std::optional<ModelWeights>
Sm87TargetAotProjectionExecutionAccess::make_complete_host_test_fixture(
    Owner& owner) noexcept {
  if (!owner.empty()) {
    return std::nullopt;
  }
  ModelWeights model_weights;
  owner.arena_ = reinterpret_cast<std::uint8_t*>(kHostTestArenaBegin);
  owner.bytes_ = kSm87TargetAotProjectionDeviceArenaBytes;
  owner.allocation_identity_ = kHostTestAllocationIdentity;
  owner.owner_identity_ = kHostTestOwnerIdentity;
  owner.device_ordinal_ = 0;
  std::uint64_t offset = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87TargetAotProjectionDeviceLayerCount;
       ++layer_index) {
    for (const Role role : {Role::kNvFp4GateUp, Role::kNvFp4Down}) {
      const std::size_t index = ordinal(layer_index, role);
      if (index >= owner.descriptors_.size() ||
          !make_host_test_descriptor(layer_index, role, offset,
                                     owner.descriptors_[index])) {
        owner.arena_ = nullptr;
        owner.bytes_ = 0U;
        owner.allocation_identity_ = 0U;
        owner.owner_identity_ = 0U;
        owner.device_ordinal_ = -1;
        owner.descriptors_ = {};
        return std::nullopt;
      }
      offset += owner.descriptors_[index].manifest.payload_bytes;
    }
  }
  if (offset != owner.bytes_) {
    owner.arena_ = nullptr;
    owner.bytes_ = 0U;
    owner.allocation_identity_ = 0U;
    owner.owner_identity_ = 0U;
    owner.device_ordinal_ = -1;
    owner.descriptors_ = {};
    return std::nullopt;
  }
  owner.descriptor_count_ = owner.descriptors_.size();
  owner.prepared_model_weights_ = &model_weights;
  owner.attached_model_weights_ = &model_weights;
  model_weights.target_aot_projection_attachment_.owner = &owner;
  model_weights.target_aot_projection_attachment_.owner_identity =
      owner.owner_identity_;
  model_weights.target_aot_projection_attachment_.allocation_identity =
      owner.allocation_identity_;
  model_weights.target_aot_projection_attachment_.arena_begin =
      kHostTestArenaBegin;
  model_weights.target_aot_projection_attachment_.arena_bytes = owner.bytes_;
  model_weights.target_aot_projection_attachment_.device_ordinal =
      owner.device_ordinal_;
  model_weights.target_aot_projection_attachment_.artifact_count =
      owner.descriptor_count_;
  return std::optional<ModelWeights>(std::in_place,
                                     std::move(model_weights));
}

bool Sm87TargetAotProjectionExecutionAccess::clear_host_test_fixture(
    Owner& owner) noexcept {
  if (owner.arena_ != reinterpret_cast<std::uint8_t*>(kHostTestArenaBegin) ||
      owner.bytes_ != kSm87TargetAotProjectionDeviceArenaBytes ||
      owner.owner_identity_ != kHostTestOwnerIdentity ||
      owner.allocation_identity_ != kHostTestAllocationIdentity ||
      owner.prepared_model_weights_ != nullptr ||
      owner.attached_model_weights_ != nullptr) {
    return false;
  }
  owner.arena_ = nullptr;
  owner.bytes_ = 0U;
  owner.allocation_identity_ = 0U;
  owner.owner_identity_ = 0U;
  owner.device_ordinal_ = -1;
  owner.descriptors_ = {};
  owner.descriptor_count_ = 0U;
  return true;
}

}  // namespace q3x::runtime::target_aot_execution_detail

static_assert(runtime::sm87_target_aot_projection_device_assets_compiled());
static_assert(runtime::kSm87TargetAotProjectionDeviceArtifactCount == 128U);
static_assert(runtime::kSm87TargetAotProjectionDeviceSourceCount == 192U);
static_assert(runtime::kSm87TargetAotProjectionDeviceArenaBytes ==
              9'625'927'680ULL);
static_assert(runtime::kSm87TargetAotProjectionMaximumHostStagingBytes ==
              200'540'160ULL);
static_assert(
    runtime::kSm87TargetAotProjectionPersistentDirectLoadFileBytesRead ==
    19'252'912'128ULL);
static_assert(!std::is_default_constructible_v<
              execution::Sm87TargetAotProjectionExecutionAccess>);
static_assert(!std::is_default_constructible_v<
              execution::Sm87TargetAotProjectionExecutionAsset>);
static_assert(!std::is_assignable_v<
              execution::Sm87TargetAotProjectionExecutionAccess&,
              const execution::Sm87TargetAotProjectionExecutionAccess&>);

[[nodiscard]] bool test_private_execution_catalog() {
  using Access = execution::Sm87TargetAotProjectionExecutionAccess;
  using Role = kernels::Sm87TargetAotProjectionRole;

  runtime::Sm87TargetAotProjectionDeviceAssets owner;
  std::optional<runtime::ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  if (!model_weights.has_value()) {
    std::cerr << "could not make the source-private complete host fixture\n";
    return false;
  }
  auto access = Access::bind(*model_weights);
  if (!access.has_value() || !access->attached() ||
      access->artifact_count() !=
          runtime::kSm87TargetAotProjectionDeviceArtifactCount) {
    std::cerr << "complete owner attachment did not issue one stable "
                 "execution capability\n";
    model_weights.reset();
    (void)Access::clear_host_test_fixture(owner);
    return false;
  }

  std::array<std::uint64_t,
             runtime::kSm87TargetAotProjectionDeviceArtifactCount>
      artifact_identities{};
  std::size_t artifact_count = 0U;
  std::uint64_t payload_bytes = 0U;
  std::optional<execution::Sm87TargetAotProjectionExecutionAsset>
      retained_layer0_gate_up;
  for (std::size_t layer_index = 0U;
       layer_index < runtime::kSm87TargetAotProjectionDeviceLayerCount;
       ++layer_index) {
    for (const Role role : {Role::kNvFp4GateUp, Role::kNvFp4Down}) {
      auto asset = access->resolve(layer_index, role);
      if (!asset.has_value() || asset->layer_index() != layer_index ||
          asset->role() != role || asset->artifact_identity() == 0U ||
          asset->source_inventory_identity() == 0U ||
          asset->payload_bytes() !=
              kernels::sm87_target_aot_projection_packed_layout(role)
                  .payload_bytes ||
          asset->borrow_cuda_asset() == nullptr ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + artifact_count,
                    asset->artifact_identity()) !=
              artifact_identities.begin() + artifact_count) {
        std::cerr << "layer/role resolver did not authenticate artifact "
                  << artifact_count << '\n';
        model_weights.reset();
        (void)Access::clear_host_test_fixture(owner);
        return false;
      }
      artifact_identities[artifact_count++] = asset->artifact_identity();
      payload_bytes += asset->payload_bytes();
      if (layer_index == 0U && role == Role::kNvFp4GateUp) {
        retained_layer0_gate_up = asset;
      }
    }
  }
  if (artifact_count != artifact_identities.size() ||
      payload_bytes != runtime::kSm87TargetAotProjectionDeviceArenaBytes ||
      !retained_layer0_gate_up.has_value() ||
      access->resolve(runtime::kSm87TargetAotProjectionDeviceLayerCount,
                      Role::kNvFp4Down)
          .has_value() ||
      access->resolve(0U, Role::kInvalid).has_value() ||
      access->resolve(0U, Role::kFp8FullQkv).has_value()) {
    std::cerr << "complete catalog closure or invalid-key rejection failed\n";
    model_weights.reset();
    (void)Access::clear_host_test_fixture(owner);
    return false;
  }

  std::optional<runtime::ModelWeights> moved_model_weights;
  moved_model_weights.emplace(std::move(*model_weights));
  const bool old_capability_invalidated =
      !access->attached() &&
      retained_layer0_gate_up->borrow_cuda_asset() == nullptr &&
      !Access::bind(*model_weights).has_value();
  auto rebound = Access::bind(*moved_model_weights);
  std::optional<execution::Sm87TargetAotProjectionExecutionAsset>
      rebound_last;
  if (rebound.has_value()) {
    rebound_last = rebound->resolve(
        runtime::kSm87TargetAotProjectionDeviceLayerCount - 1U,
        Role::kNvFp4Down);
  }
  const bool moved_capability_valid =
      rebound.has_value() && rebound->attached() &&
      rebound_last.has_value() &&
      rebound_last->borrow_cuda_asset() != nullptr;

  model_weights.reset();
  moved_model_weights.reset();
  const bool cleared = Access::clear_host_test_fixture(owner);
  if (!old_capability_invalidated || !moved_capability_valid || !cleared ||
      !owner.empty()) {
    std::cerr << "ModelWeights move did not invalidate/rebind the private "
                 "execution capability exactly once\n";
    return false;
  }
  return true;
}

int main() {
  const runtime::Sm87TargetAotProjectionDevicePreparationStats empty_stats;
  if (empty_stats.persistent_bundle_file_bytes_read != 0U ||
      empty_stats.persistent_bundle_host_authentication_passes != 0U) {
    std::cerr << "default persistence authentication accounting is not zero\n";
    return 1;
  }
  kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt receipt;
  receipt.artifact_identity = 1U;
  receipt.source_inventory_identity = 2U;
  receipt.role = kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp;
  const auto receipt_layout =
      kernels::sm87_target_aot_projection_packed_layout(receipt.role);
  receipt.plan_identity = receipt_layout.plan_identity;
  receipt.layout_identity = receipt_layout.layout_identity;
  receipt.transform_identity =
      kernels::Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.device_ordinal = 0;
  receipt.device_allocation_owner_identity = 3U;
  receipt.device_allocation_begin = 0x1000U;
  receipt.device_payload_begin = 0x2000U;
  receipt.upload_stream_identity = 4U;
  receipt.upload_completion_event_identity = 5U;
  receipt.verification_completion_event_identity = 6U;
  receipt.verification_readback_digest.bytes[0U] = 7U;
  receipt.verification_completed = true;
  const std::uint64_t base_identity =
      kernels::sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(
          receipt);
  const auto identity_changes = [base_identity](auto changed) {
    return kernels::
               sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(
                   changed) != base_identity;
  };
  auto changed = receipt;
  changed.device_ordinal = 1;
  if (!identity_changes(changed)) {
    std::cerr << "receipt identity does not seal device ordinal\n";
    return 1;
  }
  changed = receipt;
  ++changed.device_allocation_owner_identity;
  if (!identity_changes(changed)) {
    std::cerr << "receipt identity does not seal allocation owner\n";
    return 1;
  }
  changed = receipt;
  ++changed.device_allocation_begin;
  if (!identity_changes(changed)) {
    std::cerr << "receipt identity does not seal allocation range\n";
    return 1;
  }
  changed = receipt;
  ++changed.device_payload_begin;
  if (!identity_changes(changed)) {
    std::cerr << "receipt identity does not seal payload range\n";
    return 1;
  }
  changed = receipt;
  ++changed.upload_stream_identity;
  if (!identity_changes(changed)) {
    std::cerr << "receipt identity does not seal stream identity\n";
    return 1;
  }
  changed = receipt;
  ++changed.verification_completion_event_identity;
  if (!identity_changes(changed)) {
    std::cerr << "receipt identity does not seal verification event\n";
    return 1;
  }
  changed = receipt;
  changed.verification_readback_digest.bytes[0U] ^= 1U;
  if (!identity_changes(changed)) {
    std::cerr << "receipt identity does not seal readback digest\n";
    return 1;
  }
  changed = receipt;
  changed.verification_completed = false;
  if (!identity_changes(changed)) {
    std::cerr << "receipt identity does not seal terminal facts\n";
    return 1;
  }

  runtime::Sm87TargetAotProjectionDeviceAssets owner;
  if (!owner.empty() ||
      owner.has_asset(
          0U, kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp)) {
    std::cerr << "default target-AOT device owner is not empty\n";
    return 1;
  }

  if (owner.has_asset(64U,
                      kernels::Sm87TargetAotProjectionRole::kNvFp4Down) ||
      owner.has_asset(0U,
                      kernels::Sm87TargetAotProjectionRole::kFp8FullQkv)) {
    std::cerr << "target-AOT empty owner getters are not fail-closed\n";
    return 1;
  }

  if (!owner.release() || !owner.empty()) {
    std::cerr << "target-AOT owner release did not clear state\n";
    return 1;
  }

  static_assert(!std::is_copy_constructible_v<
                runtime::Sm87TargetAotProjectionDeviceAssets>);
  static_assert(!std::is_copy_assignable_v<
                runtime::Sm87TargetAotProjectionDeviceAssets>);
  static_assert(!std::is_move_constructible_v<
                runtime::Sm87TargetAotProjectionDeviceAssets>);
  static_assert(!std::is_move_assignable_v<
                runtime::Sm87TargetAotProjectionDeviceAssets>);

  if (!test_private_execution_catalog()) {
    return 1;
  }

  std::cout << "SM87 target-AOT device owner host checks passed\n";
  return 0;
}
