#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
#include "../src/runtime/sm87_target_aot_projection_complete_execution_access_internal.h"
#endif

#include <algorithm>
#include <array>
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

namespace {

using q3x::kernels::Sm87TargetAotProjectionRole;
using namespace q3x::runtime;

void require(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void test_exact_catalog_geometry() {
  require(kSm87TargetAotCompleteProjectionDeviceArtifactCount == 256U,
          "complete artifact count drifted");
  require(kSm87TargetAotCompleteProjectionDeviceSourceCount == 400U,
          "complete source count drifted");
  require(kSm87TargetAotCompleteProjectionDeviceArenaBytes ==
              16'840'130'560ULL,
          "complete arena size drifted");
  require(kSm87TargetAotCompleteProjectionCanonicalSourceD2hBytes ==
              16'840'132'160ULL,
          "complete canonical D2H accounting drifted");

  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  std::size_t linear_layers = 0U;
  std::size_t full_layers = 0U;
  std::uint64_t offset = 0U;
  for (std::size_t layer = 0U;
       layer < kSm87TargetAotCompleteProjectionDeviceLayerCount; ++layer) {
    const bool full = sm87_target_aot_complete_is_full_layer(layer);
    full ? ++full_layers : ++linear_layers;
    const Sm87TargetAotProjectionRole input_role =
        full ? Sm87TargetAotProjectionRole::kFp8FullQkv
             : Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
    const std::array<Sm87TargetAotProjectionRole, 4U> roles{{
        Sm87TargetAotProjectionRole::kNvFp4GateUp,
        Sm87TargetAotProjectionRole::kNvFp4Down, input_role,
        Sm87TargetAotProjectionRole::kFp8AttentionOutput}};
    for (const Sm87TargetAotProjectionRole role : roles) {
      const auto ordinal =
          sm87_target_aot_complete_descriptor_ordinal(layer, role);
      const auto layout =
          q3x::kernels::sm87_target_aot_projection_packed_layout(role);
      require(ordinal == artifacts, "layer-major O(1) ordinal drifted");
      require(sm87_target_aot_complete_expected_arena_offset(layer, role) ==
                  offset,
              "layer/role O(1) arena offset drifted");
      require(layout.valid(), "complete role layout is invalid");
      require(offset % layout.payload_alignment == 0U,
              "complete artifact offset lost alignment");
      offset += layout.payload_bytes;
      sources += layout.partition_count;
      ++artifacts;
    }
    require(sm87_target_aot_complete_descriptor_ordinal(
                layer, full ? Sm87TargetAotProjectionRole::kFp8GdnQkvZ
                            : Sm87TargetAotProjectionRole::kFp8FullQkv) ==
                kSm87TargetAotCompleteProjectionDeviceArtifactCount,
            "wrong attention-input role did not fail closed");
  }
  require(linear_layers == 48U && full_layers == 16U,
          "linear/full layer schedule drifted");
  require(artifacts == kSm87TargetAotCompleteProjectionDeviceArtifactCount &&
              sources == kSm87TargetAotCompleteProjectionDeviceSourceCount &&
              offset == kSm87TargetAotCompleteProjectionDeviceArenaBytes,
          "complete catalog did not close exact counts and bytes");
}

void test_online_only_request_contract() {
  Sm87TargetAotCompleteOnlinePreparationRequest request;
  require(!sm87_target_aot_complete_online_request_valid(request),
          "invalid preparation phase was accepted");
  request.phase = Sm87TargetAotCompletePreparationPhase::kEngineStartup;
  require(sm87_target_aot_complete_online_request_valid(request),
          "engine-startup online preparation was rejected");
  request.create_bundle_path = std::filesystem::path("bundle-v2.bin");
  require(!sm87_target_aot_complete_online_request_valid(request),
          "bundle creation was accepted by online-only v2");
  request.create_bundle_path.clear();
  request.load_bundle_path = std::filesystem::path("bundle-v1.bin");
  require(!sm87_target_aot_complete_online_request_valid(request),
          "bundle direct loading was accepted by online-only v2");
}

void test_typed_zero_domains_are_complete() {
  q3x::kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt nvfp4;
  q3x::kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt fp8;
  q3x::kernels::Sm87TargetAotNvFp4CudaAssetView nvfp4_view;
  q3x::kernels::Sm87TargetAotFp8CudaAssetView fp8_view;
  require(sm87_target_aot_complete_nvfp4_upload_receipt_zero(nvfp4) &&
              sm87_target_aot_complete_fp8_upload_receipt_zero(fp8) &&
              sm87_target_aot_complete_nvfp4_view_zero(nvfp4_view) &&
              sm87_target_aot_complete_fp8_view_zero(fp8_view),
          "canonical typed zero domains were rejected");

  const auto nvfp4_copy = nvfp4;
  const auto fp8_copy = fp8;
  require(sm87_target_aot_complete_same_nvfp4_upload_receipt(
              nvfp4, nvfp4_copy) &&
              sm87_target_aot_complete_same_fp8_upload_receipt(fp8,
                                                               fp8_copy),
          "identical typed receipts were not equal");

  nvfp4.verification_event_observed = true;
  require(!sm87_target_aot_complete_nvfp4_upload_receipt_zero(nvfp4),
          "deep NVFP4 receipt mutation survived zero validation");
  fp8.compensated_tensor_scale_bf16_bits[2U] = 1U;
  require(!sm87_target_aot_complete_fp8_upload_receipt_zero(fp8),
          "compensated FP8 scale mutation survived zero validation");
  require(!sm87_target_aot_complete_same_fp8_upload_receipt(fp8, fp8_copy),
          "FP8 receipt/view equality ignored compensated scales");
  nvfp4_view.tensor_scale_bits[1U] = 1U;
  require(!sm87_target_aot_complete_nvfp4_view_zero(nvfp4_view),
          "deep NVFP4 view mutation survived zero validation");
  fp8_view.no_request_time_scale_conversion = true;
  require(!sm87_target_aot_complete_fp8_view_zero(fp8_view),
          "FP8 conversion flag mutation survived zero validation");
}

void test_owner_default_state() {
  Sm87TargetAotCompleteProjectionDeviceAssets owner;
  require(owner.empty(), "default complete owner is not empty");
  require(!owner.has_asset(0U, Sm87TargetAotProjectionRole::kNvFp4GateUp),
          "empty complete owner exposed an asset");
  require(!owner.has_asset(3U, Sm87TargetAotProjectionRole::kFp8FullQkv),
          "empty complete owner exposed a full-attention asset");
  require(owner.release() && owner.empty(),
          "empty complete owner did not release idempotently");
#if defined(Q3X_EXPECT_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  require(sm87_target_aot_complete_projection_device_assets_compiled(),
          "complete admission build reported unavailable");
#else
  require(!sm87_target_aot_complete_projection_device_assets_compiled(),
          "default-off build reported complete admission available");
#endif
}

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
[[nodiscard]] bool test_source_private_complete_execution_catalog() {
  namespace execution =
      q3x::runtime::target_aot_complete_execution_detail;
  using Access = execution::Sm87TargetAotCompleteProjectionExecutionAccess;
  using Asset = execution::Sm87TargetAotCompleteProjectionExecutionAsset;
  using Role = q3x::kernels::Sm87TargetAotProjectionRole;

  static_assert(!std::is_default_constructible_v<Access>);
  static_assert(!std::is_default_constructible_v<Asset>);
  static_assert(!std::is_assignable_v<Access&, const Access&>);

  Sm87TargetAotCompleteProjectionDeviceAssets owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  if (!model_weights.has_value()) {
    std::cerr << "could not make complete target-AOT host fixture\n";
    return false;
  }
  std::optional<ModelWeights> moved_model_weights;
  const auto cleanup = [&]() {
    model_weights.reset();
    moved_model_weights.reset();
    return Access::clear_host_test_fixture(owner);
  };

  auto access = Access::bind(*model_weights);
  if (!access.has_value() || !access->attached() ||
      access->artifact_count() !=
          kSm87TargetAotCompleteProjectionDeviceArtifactCount) {
    std::cerr << "complete owner did not issue the private catalog access\n";
    (void)cleanup();
    return false;
  }

  std::array<std::uint64_t,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      artifact_identities{};
  std::size_t artifact_count = 0U;
  std::size_t nvfp4_count = 0U;
  std::size_t fp8_count = 0U;
  std::uint64_t payload_bytes = 0U;
  std::optional<Asset> retained_layer0_gate_up;
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
      auto asset = access->resolve(layer_index, role);
      const bool nvfp4 = sm87_target_aot_complete_role_is_nvfp4(role);
      const bool fp8 = sm87_target_aot_complete_role_is_fp8(role);
      const auto layout =
          q3x::kernels::sm87_target_aot_projection_packed_layout(role);
      const bool typed_borrow_valid =
          asset.has_value() &&
          (nvfp4 ? asset->borrow_nvfp4_cuda_asset() != nullptr &&
                        asset->borrow_fp8_cuda_asset() == nullptr
                  : fp8 && asset->borrow_fp8_cuda_asset() != nullptr &&
                        asset->borrow_nvfp4_cuda_asset() == nullptr);
      if (!asset.has_value() || asset->layer_index() != layer_index ||
          asset->role() != role || asset->encoding() != layout.encoding ||
          asset->artifact_identity() == 0U ||
          asset->source_inventory_identity() == 0U ||
          asset->payload_bytes() != layout.payload_bytes ||
          !typed_borrow_valid ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + artifact_count,
                    asset->artifact_identity()) !=
              artifact_identities.begin() + artifact_count) {
        std::cerr << "complete typed resolver rejected layer/role artifact "
                  << artifact_count << '\n';
        (void)cleanup();
        return false;
      }
      artifact_identities[artifact_count++] = asset->artifact_identity();
      payload_bytes += asset->payload_bytes();
      nvfp4 ? ++nvfp4_count : ++fp8_count;
      if (layer_index == 0U && role == Role::kNvFp4GateUp) {
        retained_layer0_gate_up = asset;
      }
    }
  }
  if (artifact_count != artifact_identities.size() || nvfp4_count != 128U ||
      fp8_count != 128U ||
      payload_bytes != kSm87TargetAotCompleteProjectionDeviceArenaBytes ||
      !retained_layer0_gate_up.has_value() ||
      access->resolve(kSm87TargetAotCompleteProjectionDeviceLayerCount,
                      Role::kNvFp4Down)
          .has_value() ||
      access->resolve(0U, Role::kInvalid).has_value() ||
      access->resolve(0U, Role::kFp8FullQkv).has_value() ||
      access->resolve(3U, Role::kFp8GdnQkvZ).has_value()) {
    std::cerr << "complete catalog closure or invalid-key rejection failed\n";
    (void)cleanup();
    return false;
  }

  moved_model_weights.emplace(std::move(*model_weights));
  const bool old_capability_invalidated =
      !access->attached() &&
      retained_layer0_gate_up->borrow_nvfp4_cuda_asset() == nullptr &&
      !Access::bind(*model_weights).has_value();
  auto rebound = Access::bind(*moved_model_weights);
  std::optional<Asset> retained_fp8_output;
  if (rebound.has_value()) {
    retained_fp8_output = rebound->resolve(
        kSm87TargetAotCompleteProjectionDeviceLayerCount - 1U,
        Role::kFp8AttentionOutput);
  }
  const bool moved_capability_valid =
      rebound.has_value() && rebound->attached() &&
      retained_fp8_output.has_value() &&
      retained_fp8_output->borrow_fp8_cuda_asset() != nullptr &&
      retained_fp8_output->borrow_nvfp4_cuda_asset() == nullptr;
  const bool poisoned = Access::poison_host_test_fixture_receipt(
      owner, kSm87TargetAotCompleteProjectionDeviceLayerCount - 1U,
      Role::kFp8AttentionOutput);
  const bool live_receipt_revalidated =
      poisoned && rebound.has_value() && rebound->attached() &&
      !rebound
           ->resolve(kSm87TargetAotCompleteProjectionDeviceLayerCount - 1U,
                     Role::kFp8AttentionOutput)
           .has_value() &&
      retained_fp8_output.has_value() &&
      retained_fp8_output->borrow_fp8_cuda_asset() == nullptr;

  const bool cleared = cleanup();
  if (!old_capability_invalidated || !moved_capability_valid ||
      !live_receipt_revalidated || !cleared || !owner.empty()) {
    std::cerr << "complete private access did not revalidate move/receipt "
                 "lifetime exactly\n";
    return false;
  }
  return true;
}
#endif

}  // namespace

int main() {
  test_exact_catalog_geometry();
  test_online_only_request_contract();
  test_typed_zero_domains_are_complete();
  test_owner_default_state();
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  require(test_source_private_complete_execution_catalog(),
          "source-private complete execution catalog failed");
#endif
  return 0;
}
