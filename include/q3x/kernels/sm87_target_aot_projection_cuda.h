#pragma once

#include "q3x/kernels/sm87_target_aot_projection_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Compile-only CUDA feed for the NVFP4 projection constituent of
// AC-PREFILL-SM87-AOT-SYSTEM-v1.  This is a distinct interface: it accepts
// only a loader-issued receipt for a ConsumerN64K16LaneComponentV1 artifact
// and never an Sm87P40PackedProjectionDeviceView or a request-time repack.
//
// The canonical plan and packed layout intentionally remain unqualified.
// Consequently a structurally valid launch returns cudaErrorNotSupported
// until a reviewed loader/device-residency implementation, admission-only
// executable launcher, device oracle, resource evidence, and numerical
// qualification all exist. Merely changing qualification flags is not an
// executable admission. Host-structural null/alignment/range-overlap errors,
// identities, receipts, or scales fail earlier with cudaErrorInvalidValue;
// this milestone does not inspect activation/output pointer residency.

[[nodiscard]] constexpr bool sm87_target_aot_nvfp4_cuda_role(
    const Sm87TargetAotProjectionRole role) noexcept {
  return role == Sm87TargetAotProjectionRole::kNvFp4GateUp ||
         role == Sm87TargetAotProjectionRole::kNvFp4Down;
}

// This receipt is issued by the private default-off loader/uploader, never by
// the binder below. The issuing component must own the allocation and stream,
// hash and authenticate the exact host payload before the copy, keep those
// source bytes immutable through completion, record the named event after the copy,
// and observe that event complete before setting the terminal facts.  A T0
// test may construct a receipt to exercise this schema, but doing so does not
// authenticate device memory and grants no execution or production authority.
struct Sm87TargetAotNvFp4CudaDeviceUploadReceipt final {
  std::uint64_t receipt_identity = 0U;
  std::uint64_t artifact_identity = 0U;
  std::uint64_t source_inventory_identity = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  Sm87TargetAotProjectionPackedPlanIdentity plan_identity =
      Sm87TargetAotProjectionPackedPlanIdentity::kInvalid;
  Sm87TargetAotProjectionPackedLayoutIdentity layout_identity =
      Sm87TargetAotProjectionPackedLayoutIdentity::kInvalid;
  Sm87TargetAotProjectionPackedTransformIdentity transform_identity =
      Sm87TargetAotProjectionPackedTransformIdentity::kInvalid;

  // Identity of the host payload authenticated by the transform receipt and
  // used as the upload source.  Offset and byte count bind the digest to the
  // same manifest interval rather than to an untyped digest value.
  std::uint64_t host_payload_offset = 0U;
  std::uint64_t host_payload_bytes = 0U;
  Sm87TargetAotProjectionSha256Digest host_payload_digest{};
  Sm87TargetAotProjectionPackedManifestSeal host_manifest_seal{};
  std::array<std::uint32_t, 2U> tensor_scale_bits{};
  std::uint32_t tensor_scale_count = 0U;

  // The payload can be a subrange of an owned allocation.  Both intervals
  // are explicit so a pointer into an unrelated allocation cannot acquire an
  // authenticated asset view merely by having the expected byte count.
  std::uint64_t device_allocation_identity = 0U;
  std::uint64_t device_allocation_owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::uintptr_t device_allocation_begin = 0U;
  std::uintptr_t device_allocation_end = 0U;
  std::uint64_t device_allocation_bytes = 0U;
  std::uintptr_t device_payload_begin = 0U;
  std::uintptr_t device_payload_end = 0U;
  std::uint64_t device_payload_bytes = 0U;

  // Opaque stable identities supplied by the owning loader.  They are not
  // raw cudaStream_t/cudaEvent_t handles and remain meaningful in retained
  // startup/DeploymentPlan evidence after those handles are destroyed.
  std::uint64_t upload_stream_owner_identity = 0U;
  std::uint64_t upload_stream_identity = 0U;
  std::uint64_t upload_completion_event_identity = 0U;
  std::uint64_t verification_stream_owner_identity = 0U;
  std::uint64_t verification_stream_identity = 0U;
  std::uint64_t verification_completion_event_identity = 0U;
  std::uint64_t verification_readback_bytes = 0U;
  Sm87TargetAotProjectionSha256Digest verification_readback_digest{};

  bool host_payload_digest_verified_before_copy = false;
  bool host_payload_immutable_until_completion = false;
  bool copy_enqueued_to_exact_payload_range = false;
  bool completion_event_recorded_after_copy = false;
  bool completion_event_observed = false;
  bool upload_completed = false;
  bool verification_copy_enqueued_from_exact_payload_range = false;
  bool verification_event_recorded_after_copy = false;
  bool verification_event_observed = false;
  bool verification_completed = false;
  bool device_payload_matches_host_payload = false;
  bool allocation_retained_for_asset_lifetime = false;
};

// Deterministic T0 coherence seal over every receipt fact other than the seal
// itself. This detects field substitution in retained evidence. It is not a
// secret/MAC and therefore grants no execution authority; only an owner-backed
// private Engine capability may do that.
[[nodiscard]] constexpr std::uint64_t
sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(
    const Sm87TargetAotNvFp4CudaDeviceUploadReceipt& receipt) noexcept {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  constexpr std::array<std::uint8_t, 10U> domain{
      {'q', '3', 'x', '.', 'u', 'p', 'l', 'o', 'a', 'd'}};
  for (const std::uint8_t byte : domain) {
    hash = sm87_target_aot_projection_manifest_hash_byte(hash, byte);
  }
  const auto add = [&hash](const std::uint64_t value,
                           const std::size_t bytes) constexpr {
    hash = sm87_target_aot_projection_manifest_hash_u64(hash, value, bytes);
  };
  add(receipt.artifact_identity, sizeof(receipt.artifact_identity));
  add(receipt.source_inventory_identity,
      sizeof(receipt.source_inventory_identity));
  add(static_cast<std::uint8_t>(receipt.role), 1U);
  add(static_cast<std::uint16_t>(receipt.plan_identity), 2U);
  add(static_cast<std::uint16_t>(receipt.layout_identity), 2U);
  add(static_cast<std::uint16_t>(receipt.transform_identity), 2U);
  add(receipt.host_payload_offset, sizeof(receipt.host_payload_offset));
  add(receipt.host_payload_bytes, sizeof(receipt.host_payload_bytes));
  for (const std::uint8_t byte : receipt.host_payload_digest.bytes) {
    add(byte, 1U);
  }
  add(receipt.host_manifest_seal.value,
      sizeof(receipt.host_manifest_seal.value));
  for (const std::uint32_t bits : receipt.tensor_scale_bits) {
    add(bits, sizeof(bits));
  }
  add(receipt.tensor_scale_count, sizeof(receipt.tensor_scale_count));
  add(receipt.device_allocation_identity,
      sizeof(receipt.device_allocation_identity));
  add(receipt.device_allocation_owner_identity,
      sizeof(receipt.device_allocation_owner_identity));
  add(static_cast<std::uint32_t>(receipt.device_ordinal),
      sizeof(receipt.device_ordinal));
  add(receipt.device_allocation_begin,
      sizeof(receipt.device_allocation_begin));
  add(receipt.device_allocation_end, sizeof(receipt.device_allocation_end));
  add(receipt.device_allocation_bytes,
      sizeof(receipt.device_allocation_bytes));
  add(receipt.device_payload_begin, sizeof(receipt.device_payload_begin));
  add(receipt.device_payload_end, sizeof(receipt.device_payload_end));
  add(receipt.device_payload_bytes, sizeof(receipt.device_payload_bytes));
  add(receipt.upload_stream_owner_identity,
      sizeof(receipt.upload_stream_owner_identity));
  add(receipt.upload_stream_identity, sizeof(receipt.upload_stream_identity));
  add(receipt.upload_completion_event_identity,
      sizeof(receipt.upload_completion_event_identity));
  add(receipt.verification_stream_owner_identity,
      sizeof(receipt.verification_stream_owner_identity));
  add(receipt.verification_stream_identity,
      sizeof(receipt.verification_stream_identity));
  add(receipt.verification_completion_event_identity,
      sizeof(receipt.verification_completion_event_identity));
  add(receipt.verification_readback_bytes,
      sizeof(receipt.verification_readback_bytes));
  for (const std::uint8_t byte : receipt.verification_readback_digest.bytes) {
    add(byte, 1U);
  }
  add(receipt.host_payload_digest_verified_before_copy, 1U);
  add(receipt.host_payload_immutable_until_completion, 1U);
  add(receipt.copy_enqueued_to_exact_payload_range, 1U);
  add(receipt.completion_event_recorded_after_copy, 1U);
  add(receipt.completion_event_observed, 1U);
  add(receipt.upload_completed, 1U);
  add(receipt.verification_copy_enqueued_from_exact_payload_range, 1U);
  add(receipt.verification_event_recorded_after_copy, 1U);
  add(receipt.verification_event_observed, 1U);
  add(receipt.verification_completed, 1U);
  add(receipt.device_payload_matches_host_payload, 1U);
  add(receipt.allocation_retained_for_asset_lifetime, 1U);
  return hash;
}

// This is a schema/coherence validator for a receipt supplied by the owning
// loader. It does not query CUDA pointer attributes, prove device residency,
// hash device memory, or establish that the opaque identities correspond to
// live CUDA objects. It is T0 structural evidence only; those facts and all
// execution authority belong to the private owner-backed issuer/Engine path.
[[nodiscard]] constexpr bool
sm87_target_aot_nvfp4_cuda_device_upload_receipt_matches(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::uint64_t expected_artifact_identity,
    const std::uint64_t expected_source_inventory_identity,
    const Sm87TargetAotProjectionPackedTransformIdentity
        expected_transform_identity,
    const std::uint64_t expected_host_payload_offset,
    const std::uint64_t expected_host_payload_bytes,
    const Sm87TargetAotProjectionSha256Digest&
        expected_host_payload_digest,
    const Sm87TargetAotProjectionPackedManifestSeal expected_manifest_seal,
    const std::array<std::uint32_t, 2U>& expected_tensor_scale_bits,
    const std::uint32_t expected_tensor_scale_count,
    const Sm87TargetAotNvFp4CudaDeviceUploadReceipt& receipt) noexcept {
  if (!layout.valid() || !sm87_target_aot_nvfp4_cuda_role(layout.role) ||
      expected_artifact_identity == 0U ||
      expected_source_inventory_identity == 0U ||
      expected_transform_identity !=
          Sm87TargetAotProjectionPackedTransformIdentity::
              kCanonicalNkToConsumerN64K16LaneComponentV1 ||
      expected_host_payload_offset !=
          kSm87TargetAotProjectionPackedHeaderBytes ||
      expected_host_payload_bytes != layout.payload_bytes ||
      sm87_target_aot_projection_digest_is_zero(
          expected_host_payload_digest) ||
      expected_manifest_seal.value == 0U ||
      expected_tensor_scale_count != layout.partition_count ||
      receipt.receipt_identity == 0U ||
      receipt.receipt_identity !=
          sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(receipt) ||
      receipt.artifact_identity != expected_artifact_identity ||
      receipt.source_inventory_identity !=
          expected_source_inventory_identity ||
      receipt.role != layout.role ||
      receipt.plan_identity != layout.plan_identity ||
      receipt.layout_identity != layout.layout_identity ||
      receipt.transform_identity != expected_transform_identity ||
      receipt.host_payload_offset != expected_host_payload_offset ||
      receipt.host_payload_bytes != expected_host_payload_bytes ||
      receipt.host_payload_digest != expected_host_payload_digest ||
      receipt.host_manifest_seal != expected_manifest_seal ||
      receipt.tensor_scale_count != expected_tensor_scale_count ||
      receipt.device_allocation_identity == 0U ||
      receipt.device_allocation_owner_identity == 0U ||
      receipt.device_ordinal < 0 ||
      receipt.device_allocation_begin == 0U ||
      receipt.device_allocation_bytes == 0U ||
      receipt.device_allocation_bytes >
          std::numeric_limits<std::uintptr_t>::max() ||
      receipt.device_allocation_begin >
          std::numeric_limits<std::uintptr_t>::max() -
              static_cast<std::uintptr_t>(
                  receipt.device_allocation_bytes) ||
      receipt.device_allocation_end !=
          receipt.device_allocation_begin +
              static_cast<std::uintptr_t>(
                  receipt.device_allocation_bytes) ||
      receipt.device_payload_begin == 0U ||
      receipt.device_payload_bytes != expected_host_payload_bytes ||
      receipt.device_payload_bytes >
          std::numeric_limits<std::uintptr_t>::max() ||
      receipt.device_payload_begin >
          std::numeric_limits<std::uintptr_t>::max() -
              static_cast<std::uintptr_t>(receipt.device_payload_bytes) ||
      receipt.device_payload_end !=
          receipt.device_payload_begin +
              static_cast<std::uintptr_t>(receipt.device_payload_bytes) ||
      receipt.device_payload_begin < receipt.device_allocation_begin ||
      receipt.device_payload_end > receipt.device_allocation_end ||
      !sm87_target_aot_packed_aligned(receipt.device_payload_begin,
                                      layout.payload_alignment) ||
      receipt.upload_stream_owner_identity == 0U ||
      receipt.upload_stream_owner_identity !=
          receipt.device_allocation_owner_identity ||
      receipt.upload_stream_identity == 0U ||
      receipt.upload_completion_event_identity == 0U ||
      receipt.verification_stream_owner_identity == 0U ||
      receipt.verification_stream_owner_identity !=
          receipt.device_allocation_owner_identity ||
      receipt.verification_stream_identity == 0U ||
      receipt.verification_completion_event_identity == 0U ||
      receipt.verification_completion_event_identity ==
          receipt.upload_completion_event_identity ||
      receipt.verification_readback_bytes != expected_host_payload_bytes ||
      receipt.verification_readback_digest != expected_host_payload_digest ||
      !receipt.host_payload_digest_verified_before_copy ||
      !receipt.host_payload_immutable_until_completion ||
      !receipt.copy_enqueued_to_exact_payload_range ||
      !receipt.completion_event_recorded_after_copy ||
      !receipt.completion_event_observed || !receipt.upload_completed ||
      !receipt.verification_copy_enqueued_from_exact_payload_range ||
      !receipt.verification_event_recorded_after_copy ||
      !receipt.verification_event_observed ||
      !receipt.verification_completed ||
      !receipt.device_payload_matches_host_payload ||
      !receipt.allocation_retained_for_asset_lifetime) {
    return false;
  }
  for (std::size_t index = 0U;
       index < expected_tensor_scale_bits.size(); ++index) {
    if (index < expected_tensor_scale_count) {
      if (!sm87_target_aot_projection_scale_bits_valid(
              expected_tensor_scale_bits[index]) ||
          receipt.tensor_scale_bits[index] !=
              expected_tensor_scale_bits[index]) {
        return false;
      }
    } else if (expected_tensor_scale_bits[index] != 0U ||
               receipt.tensor_scale_bits[index] != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool
sm87_target_aot_nvfp4_cuda_device_upload_receipt_structurally_valid(
    const Sm87TargetAotProjectionPackedManifest& manifest,
    const Sm87TargetAotNvFp4CudaDeviceUploadReceipt& receipt) noexcept {
  if (!sm87_target_aot_projection_packed_manifest_structurally_valid(
          manifest)) {
    return false;
  }
  const auto layout =
      sm87_target_aot_projection_packed_layout(manifest.role);
  std::array<std::uint32_t, 2U> tensor_scale_bits{};
  if (manifest.source_count > tensor_scale_bits.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < manifest.source_count; ++index) {
    tensor_scale_bits[index] = manifest.sources[index].tensor_scale_bits;
  }
  return sm87_target_aot_nvfp4_cuda_device_upload_receipt_matches(
      layout, manifest.artifact_identity, manifest.source_inventory_identity,
      Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1,
      manifest.payload_offset, manifest.payload_bytes, manifest.payload_digest,
      manifest.seal, tensor_scale_bits, manifest.source_count, receipt);
}

struct Sm87TargetAotNvFp4CudaAssetView final {
  Sm87TargetAotProjectionPackedPayloadView payload{};
  std::uint64_t artifact_identity = 0U;
  std::uint64_t source_inventory_identity = 0U;
  Sm87TargetAotProjectionPackedTransformIdentity transform_identity =
      Sm87TargetAotProjectionPackedTransformIdentity::kInvalid;
  Sm87TargetAotProjectionSha256Digest host_payload_digest{};
  Sm87TargetAotProjectionPackedManifestSeal host_manifest_seal{};
  Sm87TargetAotNvFp4CudaDeviceUploadReceipt device_upload_receipt{};
  std::array<std::uint32_t, 2U> tensor_scale_bits{};
  std::uint32_t tensor_scale_count = 0U;
  bool no_request_time_repacking = false;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotNvFp4CudaAssetView
sm87_target_aot_bind_nvfp4_cuda_asset(
    const Sm87TargetAotProjectionPackedManifest& manifest,
    const Sm87TargetAotProjectionPackedSourceInventory& expected,
    const Sm87TargetAotProjectionPackedTransformReceipt& transform_receipt,
    const Sm87TargetAotNvFp4CudaDeviceUploadReceipt&
        device_upload_receipt) noexcept {
  if (!sm87_target_aot_nvfp4_cuda_role(manifest.role) ||
      !sm87_target_aot_projection_validate_transform_receipt(
          manifest, expected, transform_receipt) ||
      !sm87_target_aot_nvfp4_cuda_device_upload_receipt_structurally_valid(
          manifest, device_upload_receipt)) {
    return {};
  }
  const auto layout =
      sm87_target_aot_projection_packed_layout(manifest.role);
  const auto payload = sm87_target_aot_projection_bind_packed_payload(
      layout, device_upload_receipt.device_payload_begin,
      device_upload_receipt.device_payload_bytes);
  if (!payload.valid) {
    return {};
  }

  Sm87TargetAotNvFp4CudaAssetView view;
  view.payload = payload;
  view.artifact_identity = manifest.artifact_identity;
  view.source_inventory_identity = manifest.source_inventory_identity;
  view.transform_identity = transform_receipt.transform_identity;
  view.host_payload_digest = manifest.payload_digest;
  view.host_manifest_seal = manifest.seal;
  view.device_upload_receipt = device_upload_receipt;
  view.tensor_scale_count = manifest.source_count;
  for (std::size_t index = 0U; index < manifest.source_count; ++index) {
    view.tensor_scale_bits[index] = manifest.sources[index].tensor_scale_bits;
  }
  view.no_request_time_repacking = transform_receipt.no_request_time_repacking;
  view.valid = true;
  return view;
}

[[nodiscard]] constexpr bool sm87_target_aot_nvfp4_cuda_asset_valid(
    const Sm87TargetAotNvFp4CudaAssetView& asset) noexcept {
  if (!asset.valid || !asset.payload.valid ||
      !sm87_target_aot_nvfp4_cuda_role(asset.payload.role) ||
      asset.artifact_identity == 0U ||
      asset.source_inventory_identity == 0U ||
      asset.transform_identity !=
          Sm87TargetAotProjectionPackedTransformIdentity::
              kCanonicalNkToConsumerN64K16LaneComponentV1 ||
      sm87_target_aot_projection_digest_is_zero(asset.host_payload_digest) ||
      asset.host_manifest_seal.value == 0U ||
      !asset.no_request_time_repacking) {
    return false;
  }
  const auto layout =
      sm87_target_aot_projection_packed_layout(asset.payload.role);
  if (!layout.valid() || asset.payload.plan_identity != layout.plan_identity ||
      asset.payload.layout_identity != layout.layout_identity ||
      asset.payload.bytes != layout.payload_bytes ||
      asset.payload.end < asset.payload.begin ||
      asset.payload.end - asset.payload.begin != asset.payload.bytes ||
      asset.payload.begin == 0U ||
      !sm87_target_aot_packed_aligned(asset.payload.begin,
                                      layout.payload_alignment) ||
      asset.tensor_scale_count != layout.partition_count) {
    return false;
  }
  const auto& upload = asset.device_upload_receipt;
  if (!sm87_target_aot_nvfp4_cuda_device_upload_receipt_matches(
          layout, asset.artifact_identity, asset.source_inventory_identity,
          asset.transform_identity,
          kSm87TargetAotProjectionPackedHeaderBytes, asset.payload.bytes,
          asset.host_payload_digest, asset.host_manifest_seal,
          asset.tensor_scale_bits, asset.tensor_scale_count, upload) ||
      upload.device_payload_begin != asset.payload.begin ||
      upload.device_payload_end != asset.payload.end) {
    return false;
  }
  for (std::size_t index = 0U; index < asset.tensor_scale_bits.size();
       ++index) {
    if (index < asset.tensor_scale_count) {
      if (!sm87_target_aot_projection_scale_bits_valid(
              asset.tensor_scale_bits[index])) {
        return false;
      }
    } else if (asset.tensor_scale_bits[index] != 0U) {
      return false;
    }
  }
  return true;
}

struct Sm87TargetAotNvFp4CudaArguments final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  Sm87TargetAotNvFp4CudaAssetView asset{};
  std::size_t token_count = 0U;
  // GateUp publishes SiLU(Gate_bf16) * Up_bf16. Down reads the residual and
  // publishes the BF16 residual sum through this same in/out pointer.
  std::uint16_t* output_or_residual = nullptr;
  void* cuda_stream = nullptr;
};

struct Sm87TargetAotNvFp4CudaByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotNvFp4CudaByteRange
sm87_target_aot_nvfp4_cuda_byte_range(const void* const pointer,
                                      const std::uint64_t bytes) noexcept {
  if (pointer == nullptr || bytes == 0U ||
      bytes > std::numeric_limits<std::uintptr_t>::max()) {
    return {};
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() -
                  static_cast<std::uintptr_t>(bytes)) {
    return {};
  }
  return {begin, begin + static_cast<std::uintptr_t>(bytes), true};
}

[[nodiscard]] constexpr bool sm87_target_aot_nvfp4_cuda_ranges_overlap(
    const Sm87TargetAotNvFp4CudaByteRange& left,
    const Sm87TargetAotNvFp4CudaByteRange& right) noexcept {
  return !left.valid || !right.valid ||
         (left.begin < right.end && right.begin < left.end);
}

[[nodiscard]] constexpr bool sm87_target_aot_nvfp4_cuda_arguments_valid(
    const Sm87TargetAotNvFp4CudaArguments& arguments) noexcept {
  const auto plan =
      sm87_target_aot_projection_plan(arguments.role, arguments.token_count);
  if (!plan.valid() || !sm87_target_aot_nvfp4_cuda_role(arguments.role) ||
      arguments.asset.payload.role != arguments.role ||
      !sm87_target_aot_nvfp4_cuda_asset_valid(arguments.asset) ||
      arguments.input == nullptr || arguments.output_or_residual == nullptr ||
      reinterpret_cast<std::uintptr_t>(arguments.input) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(arguments.output_or_residual) % 16U !=
          0U) {
    return false;
  }
  const std::uint64_t input_values =
      static_cast<std::uint64_t>(arguments.token_count) *
      plan.input_features;
  const std::uint64_t output_values =
      static_cast<std::uint64_t>(arguments.token_count) *
      plan.published_output_features;
  if (input_values > std::numeric_limits<std::uint64_t>::max() /
                         sizeof(std::uint16_t) ||
      output_values > std::numeric_limits<std::uint64_t>::max() /
                          sizeof(std::uint16_t)) {
    return false;
  }
  const auto input = sm87_target_aot_nvfp4_cuda_byte_range(
      arguments.input, input_values * sizeof(std::uint16_t));
  const auto payload = Sm87TargetAotNvFp4CudaByteRange{
      arguments.asset.payload.begin, arguments.asset.payload.end,
      arguments.asset.payload.valid};
  const auto output = sm87_target_aot_nvfp4_cuda_byte_range(
      arguments.output_or_residual,
      output_values * sizeof(std::uint16_t));
  return input.valid && payload.valid && output.valid &&
         !sm87_target_aot_nvfp4_cuda_ranges_overlap(input, payload) &&
         !sm87_target_aot_nvfp4_cuda_ranges_overlap(input, output) &&
         !sm87_target_aot_nvfp4_cuda_ranges_overlap(payload, output);
}

struct Sm87TargetAotNvFp4CudaResources final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  bool kernel_compiled = false;
  bool static_resources_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

// This query inspects only the compiled kernel image. It does not confer
// qualification and does not launch work. The three exact witness sizes use
// the same kernel image, so token_count is validated but is not a resource
// selector.
[[nodiscard]] int query_sm87_target_aot_nvfp4_cuda_resources(
    Sm87TargetAotProjectionRole role, std::size_t token_count,
    Sm87TargetAotNvFp4CudaResources* resources) noexcept;

// This v1 entry point is permanently compile-only and returns NotSupported
// for every structurally valid call. Opening execution requires a separately
// reviewed admission-only launcher; changing qualification bits cannot make
// this symbol enqueue work.
[[nodiscard]] int launch_sm87_target_aot_nvfp4_cuda(
    const Sm87TargetAotNvFp4CudaArguments& arguments) noexcept;

}  // namespace q3x::kernels
