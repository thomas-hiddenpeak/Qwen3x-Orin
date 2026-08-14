#pragma once

#include "q3x/kernels/sm87_macrofeed_v4_fp8.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace q3x::tests::support {

namespace kernels = q3x::kernels;

class Sm87MacroFeedV4LiveFp8Allocation final {
 public:
  Sm87MacroFeedV4LiveFp8Allocation() = default;
  Sm87MacroFeedV4LiveFp8Allocation(
      const Sm87MacroFeedV4LiveFp8Allocation&) = delete;
  Sm87MacroFeedV4LiveFp8Allocation& operator=(
      const Sm87MacroFeedV4LiveFp8Allocation&) = delete;

  ~Sm87MacroFeedV4LiveFp8Allocation() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  [[nodiscard]] bool allocate_zeroed(const std::size_t bytes) noexcept {
    if (bytes == 0U || pointer_ != nullptr ||
        cudaMalloc(&pointer_, bytes) != cudaSuccess || pointer_ == nullptr) {
      return false;
    }
    bytes_ = bytes;
    return cudaMemset(pointer_, 0, bytes_) == cudaSuccess;
  }

  [[nodiscard]] void* data() noexcept { return pointer_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0U;
};

[[nodiscard]] inline kernels::Sm87TargetAotProjectionSha256Digest
sm87_macrofeed_v4_test_digest(const std::uint64_t seed) noexcept {
  kernels::Sm87TargetAotProjectionSha256Digest digest;
  std::uint64_t state = seed | 1U;
  for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    digest.bytes[index] = static_cast<std::uint8_t>(state >> 24U);
  }
  digest.bytes[0U] = 0x7fU;
  digest.bytes[1U] = 0xffU;
  return digest;
}

[[nodiscard]] inline kernels::
    Sm87TargetAotProjectionPackedSourceInventory
sm87_macrofeed_v4_test_inventory(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout) noexcept {
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = 0x5133'4650'3849'4e56ULL;
  inventory.role = layout.role;
  inventory.source_count = layout.partition_count;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    inventory.sources[index] =
        kernels::sm87_target_aot_projection_packed_source_binding(
            layout, index, 0x5133'4650'3853'0001ULL + index,
            sm87_macrofeed_v4_test_digest(2U * index + 1U),
            sm87_macrofeed_v4_test_digest(2U * index + 2U),
            0x0380'0000U +
                static_cast<std::uint32_t>(index) * 0x0001'0000U);
  }
  return inventory;
}

[[nodiscard]] inline kernels::Sm87TargetAotProjectionPackedTransformReceipt
sm87_macrofeed_v4_test_transform(
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
  receipt.transform_identity =
      kernels::Sm87TargetAotProjectionPackedTransformIdentity::
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
    observed.logical_role = partition.logical_role;
    observed.partition_index = static_cast<std::uint32_t>(index);
    observed.tensor_identity = source.tensor_identity;
    observed.observed_source_weight_digest = source.weight_digest;
    observed.observed_source_scale_digest = source.scale_digest;
    observed.source_weight_bytes_hashed = values;
    observed.source_scale_bytes_hashed = sizeof(std::uint32_t);
    observed.repacked_weight_values = values;
    observed.payload_offset = partition.payload_offset;
    observed.payload_bytes = partition.payload_bytes;
    observed.source_digests_computed_from_tensor_bytes = true;
    observed.canonical_address_bijection_applied = true;
    observed.bit_exact_weight_permutation = true;
    observed.tensor_scale_kept_external = true;
  }
  return receipt;
}

// Honest, isolated live-CUDA asset used only to execute a synthetic T1
// composition.  It neither aliases nor upgrades the fake complete-catalog
// fixture and carries no real-checkpoint or production-performance meaning.
struct Sm87MacroFeedV4LiveFp8AssetFixture final {
  Sm87MacroFeedV4LiveFp8Allocation payload_allocation;
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory{};
  kernels::Sm87TargetAotProjectionPackedManifest manifest{};
  kernels::Sm87TargetAotProjectionPackedTransformReceipt transform{};
  kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt upload{};
  kernels::Sm87TargetAotFp8CudaAssetView asset{};

  [[nodiscard]] bool initialize(
      const kernels::Sm87TargetAotProjectionRole role,
      const int device_ordinal) noexcept {
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(role);
    if (!layout.valid() || device_ordinal < 0 ||
        !payload_allocation.allocate_zeroed(layout.payload_bytes)) {
      return false;
    }
    inventory = sm87_macrofeed_v4_test_inventory(layout);
    manifest = kernels::sm87_target_aot_projection_make_packed_manifest(
        role, 0x5133'4650'3841'5353ULL, inventory,
        sm87_macrofeed_v4_test_digest(0x5133U));
    transform =
        sm87_macrofeed_v4_test_transform(layout, inventory, manifest);
    if (!inventory.valid(layout) ||
        !kernels::sm87_target_aot_projection_validate_packed_manifest(
            manifest, inventory) ||
        !kernels::sm87_target_aot_projection_validate_transform_receipt(
            manifest, inventory, transform)) {
      return false;
    }

    upload.artifact_identity = manifest.artifact_identity;
    upload.source_inventory_identity = manifest.source_inventory_identity;
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
      upload.compensated_tensor_scale_bf16_bits[index] =
          kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
              upload.tensor_scale_bits[index]);
    }
    upload.device_allocation_identity = 0x5133'4650'3841'4c4cULL;
    upload.device_allocation_owner_identity = 0x5133'4650'384f'574eULL;
    upload.device_ordinal = device_ordinal;
    upload.device_allocation_begin = reinterpret_cast<std::uintptr_t>(
        payload_allocation.data());
    upload.device_allocation_bytes = payload_allocation.bytes();
    upload.device_allocation_end =
        upload.device_allocation_begin + upload.device_allocation_bytes;
    upload.device_payload_begin = upload.device_allocation_begin;
    upload.device_payload_bytes = manifest.payload_bytes;
    upload.device_payload_end =
        upload.device_payload_begin + upload.device_payload_bytes;
    upload.upload_stream_owner_identity =
        upload.device_allocation_owner_identity;
    upload.upload_stream_identity = 0x5133'4650'3855'5053ULL;
    upload.upload_completion_event_identity = 0x5133'4650'3855'5045ULL;
    upload.verification_stream_owner_identity =
        upload.device_allocation_owner_identity;
    upload.verification_stream_identity = 0x5133'4650'3856'5353ULL;
    upload.verification_completion_event_identity =
        0x5133'4650'3856'4556ULL;
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
        kernels::sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(
            upload);
    asset = kernels::sm87_target_aot_bind_fp8_cuda_asset(
        manifest, inventory, transform, upload);
    return kernels::sm87_target_aot_fp8_cuda_asset_valid(asset);
  }
};

}  // namespace q3x::tests::support
