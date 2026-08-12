#include "q3x/kernels/sm87_target_aot_projection_cuda.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace kernels = q3x::kernels;

namespace {

constexpr auto kGateP40 = kernels::sm87_target_aot_projection_plan(
    kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 40'000U);
constexpr auto kGateP60 = kernels::sm87_target_aot_projection_plan(
    kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 60'000U);
constexpr auto kGateP130 = kernels::sm87_target_aot_projection_plan(
    kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 130'000U);
constexpr auto kDownP40 = kernels::sm87_target_aot_projection_plan(
    kernels::Sm87TargetAotProjectionRole::kNvFp4Down, 40'000U);
constexpr auto kDownP60 = kernels::sm87_target_aot_projection_plan(
    kernels::Sm87TargetAotProjectionRole::kNvFp4Down, 60'000U);
constexpr auto kDownP130 = kernels::sm87_target_aot_projection_plan(
    kernels::Sm87TargetAotProjectionRole::kNvFp4Down, 130'000U);

static_assert(kGateP40.valid() && kGateP60.valid() && kGateP130.valid());
static_assert(kDownP40.valid() && kDownP60.valid() && kDownP130.valid());
static_assert(kGateP40.dynamic_shared_bytes == 76'800U);
static_assert(kDownP130.dynamic_shared_bytes == 76'800U);
static_assert(kGateP40.physical_ctas == 16U &&
              kGateP40.ctas_per_sm == 1U);
static_assert(kGateP40.mma_partitions_per_task == 2U);
static_assert(kDownP40.mma_partitions_per_task == 1U);
static_assert(!kGateP40.cuda_implementation_present &&
              !kGateP40.static_resources_qualified &&
              !kGateP40.numerical_contract_qualified &&
              !kGateP40.production_dispatch_eligible);

constexpr auto kGateLayout =
    kernels::sm87_target_aot_projection_packed_layout(
        kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp);
constexpr auto kDownLayout =
    kernels::sm87_target_aot_projection_packed_layout(
        kernels::Sm87TargetAotProjectionRole::kNvFp4Down);
static_assert(kGateLayout.valid() && kDownLayout.valid());
static_assert(kGateLayout.partition_count == 2U);
static_assert(kDownLayout.partition_count == 1U);
static_assert(kGateLayout.layout_identity ==
              kernels::Sm87TargetAotProjectionPackedLayoutIdentity::
                  kConsumerN64K16LaneComponentV1);
static_assert(!kGateLayout.cuda_implementation_present &&
              !kGateLayout.static_resources_qualified &&
              !kGateLayout.numerical_contract_qualified &&
              !kGateLayout.production_dispatch_eligible);

constexpr kernels::Sm87TargetAotProjectionPackedManifest kEmptyManifest{};
constexpr kernels::Sm87TargetAotProjectionPackedSourceInventory
    kEmptyInventory{};
constexpr kernels::Sm87TargetAotProjectionPackedTransformReceipt
    kEmptyTransform{};
constexpr kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt
    kEmptyDeviceUpload{};
constexpr auto kRejectedEmptyAsset =
    kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
        kEmptyManifest, kEmptyInventory, kEmptyTransform,
        kEmptyDeviceUpload);
static_assert(!kRejectedEmptyAsset.valid);

struct Test final {
  int failures = 0;

  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
};

[[nodiscard]] kernels::Sm87TargetAotProjectionSha256Digest make_digest(
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

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedSourceInventory
make_inventory(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const std::uint64_t seed) noexcept {
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = 0x5100'0000'0000'0000ULL + seed;
  inventory.role = layout.role;
  inventory.source_count = layout.partition_count;
  constexpr std::uint32_t kScaleBits[2U]{0x3f80'0000U,
                                        0x3f00'0000U};
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    inventory.sources[index] =
        kernels::sm87_target_aot_projection_packed_source_binding(
            layout, index,
            0x6100'0000'0000'0000ULL + seed * 4U + index,
            make_digest(seed * 17U + index * 2U + 1U),
            make_digest(seed * 17U + index * 2U + 2U),
            kScaleBits[index]);
  }
  return inventory;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedTransformReceipt
make_transform_receipt(
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
    observed.bit_exact_block_scale_permutation =
        block_scale_values != 0U;
    observed.tensor_scale_kept_external = true;
  }
  return receipt;
}

[[nodiscard]] kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt
make_device_upload_receipt(
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const std::uintptr_t payload_address) noexcept {
  // T0 schema fixture only: these are deliberately non-dereferenced opaque
  // addresses and identities. This test never claims to authenticate device
  // bytes and the launcher remains fail-closed before any enqueue.
  constexpr std::uintptr_t kAllocationGuard = 4'096U;
  kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt receipt;
  receipt.artifact_identity = manifest.artifact_identity;
  receipt.source_inventory_identity = manifest.source_inventory_identity;
  receipt.role = manifest.role;
  receipt.plan_identity = manifest.plan_identity;
  receipt.layout_identity = manifest.layout_identity;
  receipt.transform_identity = kernels::
      Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.host_payload_offset = manifest.payload_offset;
  receipt.host_payload_bytes = manifest.payload_bytes;
  receipt.host_payload_digest = manifest.payload_digest;
  receipt.host_manifest_seal = manifest.seal;
  receipt.tensor_scale_count = manifest.source_count;
  for (std::size_t index = 0U; index < manifest.source_count; ++index) {
    receipt.tensor_scale_bits[index] =
        manifest.sources[index].tensor_scale_bits;
  }
  receipt.device_allocation_identity = 0x7200'0000'0000'0001ULL;
  receipt.device_allocation_owner_identity = 0x7300'0000'0000'0001ULL;
  receipt.device_ordinal = 0;
  receipt.device_allocation_begin = payload_address - kAllocationGuard;
  receipt.device_allocation_bytes =
      manifest.payload_bytes + 2U * kAllocationGuard;
  receipt.device_allocation_end =
      receipt.device_allocation_begin +
      static_cast<std::uintptr_t>(receipt.device_allocation_bytes);
  receipt.device_payload_begin = payload_address;
  receipt.device_payload_bytes = manifest.payload_bytes;
  receipt.device_payload_end =
      payload_address + static_cast<std::uintptr_t>(manifest.payload_bytes);
  receipt.upload_stream_owner_identity =
      receipt.device_allocation_owner_identity;
  receipt.upload_stream_identity = 0x7400'0000'0000'0001ULL;
  receipt.upload_completion_event_identity = 0x7500'0000'0000'0001ULL;
  receipt.verification_stream_owner_identity =
      receipt.device_allocation_owner_identity;
  receipt.verification_stream_identity = receipt.upload_stream_identity;
  receipt.verification_completion_event_identity =
      0x7600'0000'0000'0001ULL;
  receipt.verification_readback_bytes = manifest.payload_bytes;
  receipt.verification_readback_digest = manifest.payload_digest;
  receipt.host_payload_digest_verified_before_copy = true;
  receipt.host_payload_immutable_until_completion = true;
  receipt.copy_enqueued_to_exact_payload_range = true;
  receipt.completion_event_recorded_after_copy = true;
  receipt.completion_event_observed = true;
  receipt.upload_completed = true;
  receipt.verification_copy_enqueued_from_exact_payload_range = true;
  receipt.verification_event_recorded_after_copy = true;
  receipt.verification_event_observed = true;
  receipt.verification_completed = true;
  receipt.device_payload_matches_host_payload = true;
  receipt.allocation_retained_for_asset_lifetime = true;
  receipt.receipt_identity =
      kernels::sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(
          receipt);
  return receipt;
}

struct FakeBinding final {
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory{};
  kernels::Sm87TargetAotProjectionPackedManifest manifest{};
  kernels::Sm87TargetAotProjectionPackedTransformReceipt transform{};
  kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt upload{};
};

[[nodiscard]] FakeBinding make_fake_binding(
    const kernels::Sm87TargetAotProjectionRole role,
    const std::uintptr_t address) {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(role);
  const auto seed = static_cast<std::uint64_t>(role) + 1U;
  FakeBinding binding;
  binding.inventory = make_inventory(layout, seed);
  binding.manifest = kernels::sm87_target_aot_projection_make_packed_manifest(
      role, 0x3141'5926'0000'0000ULL + seed, binding.inventory,
      make_digest(0x8100U + seed));
  binding.transform =
      make_transform_receipt(layout, binding.inventory, binding.manifest);
  binding.upload = make_device_upload_receipt(binding.manifest, address);
  return binding;
}

[[nodiscard]] kernels::Sm87TargetAotNvFp4CudaAssetView fake_asset(
    const kernels::Sm87TargetAotProjectionRole role,
    const std::uintptr_t address) {
  const auto binding = make_fake_binding(role, address);
  return kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
      binding.manifest, binding.inventory, binding.transform,
      binding.upload);
}

[[nodiscard]] kernels::Sm87TargetAotNvFp4CudaArguments fake_arguments(
    const kernels::Sm87TargetAotProjectionRole role,
    const std::size_t token_count) {
  constexpr std::uintptr_t kInput = 0x0000'0010'0000'0000ULL;
  constexpr std::uintptr_t kPayload = 0x0000'0020'0000'0000ULL;
  constexpr std::uintptr_t kOutput = 0x0000'0030'0000'0000ULL;
  kernels::Sm87TargetAotNvFp4CudaArguments arguments;
  arguments.role = role;
  arguments.input = reinterpret_cast<const std::uint16_t*>(kInput);
  arguments.asset = fake_asset(role, kPayload);
  arguments.token_count = token_count;
  arguments.output_or_residual =
      reinterpret_cast<std::uint16_t*>(kOutput);
  return arguments;
}

}  // namespace

int main() {
  Test test;
  constexpr std::uintptr_t kReceiptPayload =
      0x0000'0020'0000'0000ULL;
  const auto valid_binding = make_fake_binding(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp,
      kReceiptPayload);
  test.expect(
      kernels::
          sm87_target_aot_nvfp4_cuda_device_upload_receipt_structurally_valid(
          valid_binding.manifest, valid_binding.upload),
      "coherent upload receipt schema must validate structurally");
  test.expect(
      kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
          valid_binding.manifest, valid_binding.inventory,
          valid_binding.transform, valid_binding.upload)
          .valid,
      "binder must accept a coherent host transform and device upload receipt");

  auto bad_upload = valid_binding.upload;
  bad_upload.host_payload_digest.bytes[0U] ^= 1U;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "device upload receipt must bind the authenticated host payload digest");

  bad_upload = valid_binding.upload;
  ++bad_upload.host_payload_offset;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "device upload receipt must bind the manifest host payload offset");

  bad_upload = valid_binding.upload;
  bad_upload.device_allocation_end = bad_upload.device_payload_end - 1U;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "payload range outside its exact allocation range must fail closed");

  bad_upload = valid_binding.upload;
  ++bad_upload.upload_stream_owner_identity;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "upload stream owner must match the device allocation owner");

  bad_upload = valid_binding.upload;
  bad_upload.completion_event_observed = false;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "unobserved upload completion event must fail closed");

  bad_upload = valid_binding.upload;
  bad_upload.upload_completed = false;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "incomplete upload state must fail closed");

  bad_upload = valid_binding.upload;
  bad_upload.device_ordinal = -1;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "upload receipt must name a concrete CUDA device ordinal");

  bad_upload = valid_binding.upload;
  bad_upload.verification_completed = false;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "device readback verification completion must fail closed");

  bad_upload = valid_binding.upload;
  bad_upload.verification_readback_digest.bytes[0U] ^= 1U;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "device readback SHA-256 must equal the authenticated host payload");

  bad_upload = valid_binding.upload;
  bad_upload.device_payload_matches_host_payload = false;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "device payload match attestation must come from the upload receipt");

  bad_upload = valid_binding.upload;
  bad_upload.host_manifest_seal.value ^= 1U;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "device upload receipt must bind the exact manifest seal");

  bad_upload = valid_binding.upload;
  bad_upload.tensor_scale_bits[0U] = 0x4000'0000U;
  test.expect(
      !kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
           valid_binding.manifest, valid_binding.inventory,
           valid_binding.transform, bad_upload)
           .valid,
      "receipt must reject another individually valid tensor-scale value");

  for (const auto role :
       {kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp,
        kernels::Sm87TargetAotProjectionRole::kNvFp4Down}) {
    for (const std::size_t tokens : {40'000U, 60'000U, 130'000U}) {
      const auto arguments = fake_arguments(role, tokens);
      test.expect(kernels::sm87_target_aot_nvfp4_cuda_asset_valid(
                      arguments.asset),
                  "receipt-bound target-AOT asset schema must validate");
      test.expect(kernels::sm87_target_aot_nvfp4_cuda_arguments_valid(
                      arguments),
                  "P40/P60/P130 host arguments must validate");
      test.expect(
          static_cast<cudaError_t>(
              kernels::launch_sm87_target_aot_nvfp4_cuda(arguments)) ==
              cudaErrorNotSupported,
          "valid compile asset must remain explicitly non-executable");
    }
  }

  auto bad = fake_arguments(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 513U);
  test.expect(!kernels::sm87_target_aot_nvfp4_cuda_arguments_valid(bad),
              "non-witness token count must fail closed");
  test.expect(
      static_cast<cudaError_t>(
          kernels::launch_sm87_target_aot_nvfp4_cuda(bad)) ==
          cudaErrorInvalidValue,
      "invalid host arguments must fail before CUDA runtime inspection");

  bad = fake_arguments(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 40'000U);
  bad.output_or_residual = const_cast<std::uint16_t*>(bad.input);
  test.expect(!kernels::sm87_target_aot_nvfp4_cuda_arguments_valid(bad),
              "input/output alias must fail closed");

  bad = fake_arguments(
      kernels::Sm87TargetAotProjectionRole::kNvFp4Down, 40'000U);
  bad.input = reinterpret_cast<const std::uint16_t*>(
      reinterpret_cast<std::uintptr_t>(bad.input) + 2U);
  test.expect(!kernels::sm87_target_aot_nvfp4_cuda_arguments_valid(bad),
              "misaligned activation must fail closed");

  bad = fake_arguments(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 40'000U);
  bad.asset.tensor_scale_bits[1U] = 0U;
  test.expect(!kernels::sm87_target_aot_nvfp4_cuda_arguments_valid(bad),
              "zero partition tensor scale must fail closed");

  bad = fake_arguments(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 40'000U);
  bad.asset.tensor_scale_bits[0U] = 0x4000'0000U;
  test.expect(!kernels::sm87_target_aot_nvfp4_cuda_arguments_valid(bad),
              "post-bind legal tensor-scale mutation must fail closed");

  bad = fake_arguments(
      kernels::Sm87TargetAotProjectionRole::kNvFp4Down, 40'000U);
  bad.asset.no_request_time_repacking = false;
  test.expect(!kernels::sm87_target_aot_nvfp4_cuda_arguments_valid(bad),
              "request-time repack asset must fail closed");

  bad = fake_arguments(
      kernels::Sm87TargetAotProjectionRole::kNvFp4Down, 40'000U);
  bad.asset.device_upload_receipt.upload_completed = false;
  test.expect(!kernels::sm87_target_aot_nvfp4_cuda_arguments_valid(bad),
              "tampered device upload completion must fail closed");

  bad = fake_arguments(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 40'000U);
  bad.asset.payload.role =
      kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
  test.expect(!kernels::sm87_target_aot_nvfp4_cuda_arguments_valid(bad),
              "payload role mismatch must fail closed");

  test.expect(
      static_cast<cudaError_t>(
          kernels::query_sm87_target_aot_nvfp4_cuda_resources(
              kernels::Sm87TargetAotProjectionRole::kInvalid, 40'000U,
              nullptr)) == cudaErrorInvalidValue,
      "null static-resource destination must fail without device access");

  if (test.failures != 0) {
    return 1;
  }
  std::cout << "sm87 target-AOT NVFP4 CUDA compile/fail-closed tests passed\n";
  return 0;
}
