#include "q3x/runtime/sm87_target_aot_projection_device_assets.h"

#include <iostream>
#include <type_traits>
#include <utility>

namespace kernels = q3x::kernels;
namespace runtime = q3x::runtime;

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

  std::cout << "SM87 target-AOT device owner host checks passed\n";
  return 0;
}
