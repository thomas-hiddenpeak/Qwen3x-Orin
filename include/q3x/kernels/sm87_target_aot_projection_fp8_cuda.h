#pragma once

#include "q3x/kernels/sm87_target_aot_projection_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Source-independent CUDA contract for the three FP8 constituents of
// AC-PREFILL-SM87-AOT-SYSTEM-v1.  This header authenticates a loader-owned
// device subrange and freezes the pure projection publication boundary.  It
// neither implements a CUDA kernel nor grants numerical or production
// qualification; all executable admission remains permanently default-off.
inline constexpr std::size_t kSm87TargetAotFp8CudaMaximumTensorScales = 3U;

[[nodiscard]] constexpr bool sm87_target_aot_fp8_cuda_role(
    const Sm87TargetAotProjectionRole role) noexcept {
  return role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ ||
         role == Sm87TargetAotProjectionRole::kFp8FullQkv ||
         role == Sm87TargetAotProjectionRole::kFp8AttentionOutput;
}

// Reproduce the admitted FP8 Marlin scale path using only integer operations:
// F32 -> BF16 RNE, exact multiplication by 2^120, then BF16 RNE.  Multiplying
// a finite BF16 by a power of two introduces no additional significand bits;
// the subnormal branch normalizes the exact seven-bit fraction.  Zero means
// that the authenticated raw scale cannot produce a finite nonzero launch
// scale.  FP8 terminal 0x7f/0xff codes are weight bytes, not scale bits, and
// are deliberately outside this validation.
[[nodiscard]] constexpr std::uint16_t
sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
    const std::uint32_t raw_scale_bits) noexcept {
  if (!sm87_target_aot_projection_scale_bits_valid(raw_scale_bits)) {
    return 0U;
  }
  const std::uint32_t upper = raw_scale_bits >> 16U;
  const std::uint32_t rounding_bias = 0x7fffU + (upper & 1U);
  const std::uint32_t rounded = raw_scale_bits + rounding_bias;
  const std::uint16_t bf16 = static_cast<std::uint16_t>(rounded >> 16U);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>((bf16 >> 7U) & 0xffU);
  const std::uint16_t fraction = static_cast<std::uint16_t>(bf16 & 0x7fU);
  if (bf16 == 0U || exponent == 0xffU) {
    return 0U;
  }
  if (exponent != 0U) {
    const std::uint16_t compensated_exponent =
        static_cast<std::uint16_t>(exponent + 120U);
    if (compensated_exponent >= 0xffU) {
      return 0U;
    }
    return static_cast<std::uint16_t>((compensated_exponent << 7U) |
                                      fraction);
  }

  std::uint16_t leading_bit = 0U;
  for (std::uint16_t bit = 1U; bit < 7U; ++bit) {
    if ((fraction & static_cast<std::uint16_t>(1U << bit)) != 0U) {
      leading_bit = bit;
    }
  }
  const std::uint16_t normalized = static_cast<std::uint16_t>(
      fraction << static_cast<std::uint16_t>(7U - leading_bit));
  const std::uint16_t compensated_exponent =
      static_cast<std::uint16_t>(114U + leading_bit);
  return static_cast<std::uint16_t>(
      (compensated_exponent << 7U) | (normalized & 0x7fU));
}

// Issued only by the private owner/uploader after the exact host payload has
// been authenticated, copied into the retained allocation interval, and read
// back from that exact device interval.  This remains a structural receipt:
// only the private owner-backed execution capability can grant authority.
struct Sm87TargetAotFp8CudaDeviceUploadReceipt final {
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

  std::uint64_t host_payload_offset = 0U;
  std::uint64_t host_payload_bytes = 0U;
  Sm87TargetAotProjectionSha256Digest host_payload_digest{};
  Sm87TargetAotProjectionPackedManifestSeal host_manifest_seal{};
  std::array<std::uint32_t, kSm87TargetAotFp8CudaMaximumTensorScales>
      tensor_scale_bits{};
  std::array<std::uint16_t, kSm87TargetAotFp8CudaMaximumTensorScales>
      compensated_tensor_scale_bf16_bits{};
  std::uint32_t tensor_scale_count = 0U;

  std::uint64_t device_allocation_identity = 0U;
  std::uint64_t device_allocation_owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::uintptr_t device_allocation_begin = 0U;
  std::uintptr_t device_allocation_end = 0U;
  std::uint64_t device_allocation_bytes = 0U;
  std::uintptr_t device_payload_begin = 0U;
  std::uintptr_t device_payload_end = 0U;
  std::uint64_t device_payload_bytes = 0U;

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

// Deterministic coherence seal over every receipt fact except the seal itself.
// It is intentionally not a secret or a substitute for private ownership.
[[nodiscard]] constexpr std::uint64_t
sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(
    const Sm87TargetAotFp8CudaDeviceUploadReceipt& receipt) noexcept {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  constexpr std::array<std::uint8_t, 14U> domain{{
      'q', '3', 'x', '.', 'f', 'p', '8', '.', 'u', 'p', 'l', 'o', 'a', 'd'}};
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
  for (const std::uint16_t bits :
       receipt.compensated_tensor_scale_bf16_bits) {
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

[[nodiscard]] constexpr bool
sm87_target_aot_fp8_cuda_device_upload_receipt_matches(
    const Sm87TargetAotProjectionPackedLayout& layout,
    const std::uint64_t expected_artifact_identity,
    const std::uint64_t expected_source_inventory_identity,
    const Sm87TargetAotProjectionPackedTransformIdentity
        expected_transform_identity,
    const std::uint64_t expected_host_payload_offset,
    const std::uint64_t expected_host_payload_bytes,
    const Sm87TargetAotProjectionSha256Digest& expected_host_payload_digest,
    const Sm87TargetAotProjectionPackedManifestSeal expected_manifest_seal,
    const std::array<std::uint32_t,
                     kSm87TargetAotFp8CudaMaximumTensorScales>&
        expected_tensor_scale_bits,
    const std::array<std::uint16_t,
                     kSm87TargetAotFp8CudaMaximumTensorScales>&
        expected_compensated_tensor_scale_bf16_bits,
    const std::uint32_t expected_tensor_scale_count,
    const Sm87TargetAotFp8CudaDeviceUploadReceipt& receipt) noexcept {
  if (!layout.valid() || !sm87_target_aot_fp8_cuda_role(layout.role) ||
      layout.encoding !=
          Sm87TargetAotProjectionEncoding::kFp8E4M3FnTensorScale ||
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
      expected_tensor_scale_count == 0U ||
      expected_tensor_scale_count != layout.partition_count ||
      expected_tensor_scale_count >
          kSm87TargetAotFp8CudaMaximumTensorScales ||
      receipt.receipt_identity == 0U ||
      receipt.receipt_identity !=
          sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(receipt) ||
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
              static_cast<std::uintptr_t>(receipt.device_allocation_bytes) ||
      receipt.device_allocation_end !=
          receipt.device_allocation_begin +
              static_cast<std::uintptr_t>(receipt.device_allocation_bytes) ||
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
       index < kSm87TargetAotFp8CudaMaximumTensorScales; ++index) {
    if (index < expected_tensor_scale_count) {
      const std::uint16_t compensated =
          sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
              expected_tensor_scale_bits[index]);
      if (compensated == 0U ||
          expected_compensated_tensor_scale_bf16_bits[index] != compensated ||
          receipt.tensor_scale_bits[index] !=
              expected_tensor_scale_bits[index] ||
          receipt.compensated_tensor_scale_bf16_bits[index] != compensated) {
        return false;
      }
    } else if (expected_tensor_scale_bits[index] != 0U ||
               expected_compensated_tensor_scale_bf16_bits[index] != 0U ||
               receipt.tensor_scale_bits[index] != 0U ||
               receipt.compensated_tensor_scale_bf16_bits[index] != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool
sm87_target_aot_fp8_cuda_device_upload_receipt_structurally_valid(
    const Sm87TargetAotProjectionPackedManifest& manifest,
    const Sm87TargetAotFp8CudaDeviceUploadReceipt& receipt) noexcept {
  if (!sm87_target_aot_projection_packed_manifest_structurally_valid(
          manifest) ||
      !sm87_target_aot_fp8_cuda_role(manifest.role) ||
      manifest.source_count > kSm87TargetAotFp8CudaMaximumTensorScales) {
    return false;
  }
  std::array<std::uint32_t, kSm87TargetAotFp8CudaMaximumTensorScales>
      tensor_scale_bits{};
  std::array<std::uint16_t, kSm87TargetAotFp8CudaMaximumTensorScales>
      compensated_tensor_scale_bf16_bits{};
  for (std::size_t index = 0U; index < manifest.source_count; ++index) {
    tensor_scale_bits[index] = manifest.sources[index].tensor_scale_bits;
    compensated_tensor_scale_bf16_bits[index] =
        sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
            tensor_scale_bits[index]);
  }
  return sm87_target_aot_fp8_cuda_device_upload_receipt_matches(
      sm87_target_aot_projection_packed_layout(manifest.role),
      manifest.artifact_identity, manifest.source_inventory_identity,
      Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1,
      manifest.payload_offset, manifest.payload_bytes, manifest.payload_digest,
      manifest.seal, tensor_scale_bits,
      compensated_tensor_scale_bf16_bits, manifest.source_count, receipt);
}

struct Sm87TargetAotFp8CudaAssetView final {
  Sm87TargetAotProjectionPackedPayloadView payload{};
  std::uint64_t artifact_identity = 0U;
  std::uint64_t source_inventory_identity = 0U;
  Sm87TargetAotProjectionPackedTransformIdentity transform_identity =
      Sm87TargetAotProjectionPackedTransformIdentity::kInvalid;
  Sm87TargetAotProjectionSha256Digest host_payload_digest{};
  Sm87TargetAotProjectionPackedManifestSeal host_manifest_seal{};
  Sm87TargetAotFp8CudaDeviceUploadReceipt device_upload_receipt{};
  std::array<std::uint32_t, kSm87TargetAotFp8CudaMaximumTensorScales>
      tensor_scale_bits{};
  // CUDA wrappers decode these already authenticated BF16 bit patterns into
  // launch scalars; request-time scale conversion is forbidden.
  std::array<std::uint16_t, kSm87TargetAotFp8CudaMaximumTensorScales>
      compensated_tensor_scale_bf16_bits{};
  std::uint32_t tensor_scale_count = 0U;
  bool no_request_time_repacking = false;
  bool no_request_time_scale_conversion = false;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotFp8CudaAssetView
sm87_target_aot_bind_fp8_cuda_asset(
    const Sm87TargetAotProjectionPackedManifest& manifest,
    const Sm87TargetAotProjectionPackedSourceInventory& expected,
    const Sm87TargetAotProjectionPackedTransformReceipt& transform_receipt,
    const Sm87TargetAotFp8CudaDeviceUploadReceipt&
        device_upload_receipt) noexcept {
  if (!sm87_target_aot_fp8_cuda_role(manifest.role) ||
      !sm87_target_aot_projection_validate_transform_receipt(
          manifest, expected, transform_receipt) ||
      !sm87_target_aot_fp8_cuda_device_upload_receipt_structurally_valid(
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

  Sm87TargetAotFp8CudaAssetView view;
  view.payload = payload;
  view.artifact_identity = manifest.artifact_identity;
  view.source_inventory_identity = manifest.source_inventory_identity;
  view.transform_identity = transform_receipt.transform_identity;
  view.host_payload_digest = manifest.payload_digest;
  view.host_manifest_seal = manifest.seal;
  view.device_upload_receipt = device_upload_receipt;
  view.tensor_scale_count = manifest.source_count;
  for (std::size_t index = 0U; index < manifest.source_count; ++index) {
    view.tensor_scale_bits[index] =
        manifest.sources[index].tensor_scale_bits;
    view.compensated_tensor_scale_bf16_bits[index] =
        sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
            view.tensor_scale_bits[index]);
  }
  view.no_request_time_repacking = transform_receipt.no_request_time_repacking;
  view.no_request_time_scale_conversion = true;
  view.valid = true;
  return view;
}

[[nodiscard]] constexpr bool sm87_target_aot_fp8_cuda_asset_valid(
    const Sm87TargetAotFp8CudaAssetView& asset) noexcept {
  if (!asset.valid || !asset.payload.valid ||
      !sm87_target_aot_fp8_cuda_role(asset.payload.role) ||
      asset.artifact_identity == 0U ||
      asset.source_inventory_identity == 0U ||
      asset.transform_identity !=
          Sm87TargetAotProjectionPackedTransformIdentity::
              kCanonicalNkToConsumerN64K16LaneComponentV1 ||
      sm87_target_aot_projection_digest_is_zero(asset.host_payload_digest) ||
      asset.host_manifest_seal.value == 0U ||
      !asset.no_request_time_repacking ||
      !asset.no_request_time_scale_conversion) {
    return false;
  }
  const auto layout =
      sm87_target_aot_projection_packed_layout(asset.payload.role);
  if (!layout.valid() ||
      layout.encoding !=
          Sm87TargetAotProjectionEncoding::kFp8E4M3FnTensorScale ||
      asset.payload.plan_identity != layout.plan_identity ||
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
  if (!sm87_target_aot_fp8_cuda_device_upload_receipt_matches(
          layout, asset.artifact_identity, asset.source_inventory_identity,
          asset.transform_identity, kSm87TargetAotProjectionPackedHeaderBytes,
          asset.payload.bytes, asset.host_payload_digest,
          asset.host_manifest_seal, asset.tensor_scale_bits,
          asset.compensated_tensor_scale_bf16_bits, asset.tensor_scale_count,
          upload) ||
      upload.device_payload_begin != asset.payload.begin ||
      upload.device_payload_end != asset.payload.end) {
    return false;
  }
  for (std::size_t index = 0U;
       index < kSm87TargetAotFp8CudaMaximumTensorScales; ++index) {
    if (index < asset.tensor_scale_count) {
      if (asset.compensated_tensor_scale_bf16_bits[index] == 0U ||
          asset.compensated_tensor_scale_bf16_bits[index] !=
              sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
                  asset.tensor_scale_bits[index])) {
        return false;
      }
    } else if (asset.tensor_scale_bits[index] != 0U ||
               asset.compensated_tensor_scale_bf16_bits[index] != 0U) {
      return false;
    }
  }
  return true;
}

// Pure projection publication.  In particular, AttentionOutput publishes its
// BF16 projection to a distinct output interval; residual fusion belongs to a
// later system stage and cannot be smuggled through this argument schema.
struct Sm87TargetAotFp8CudaArguments final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  Sm87TargetAotFp8CudaAssetView asset{};
  std::size_t token_count = 0U;
  std::uint16_t* output = nullptr;
  void* cuda_stream = nullptr;
};

struct Sm87TargetAotFp8CudaByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotFp8CudaByteRange
sm87_target_aot_fp8_cuda_byte_range(const void* const pointer,
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

[[nodiscard]] constexpr bool sm87_target_aot_fp8_cuda_ranges_overlap(
    const Sm87TargetAotFp8CudaByteRange& left,
    const Sm87TargetAotFp8CudaByteRange& right) noexcept {
  return !left.valid || !right.valid ||
         (left.begin < right.end && right.begin < left.end);
}

[[nodiscard]] constexpr bool sm87_target_aot_fp8_cuda_arguments_valid(
    const Sm87TargetAotFp8CudaArguments& arguments) noexcept {
  const auto plan =
      sm87_target_aot_projection_plan(arguments.role, arguments.token_count);
  if (!plan.valid() || !sm87_target_aot_fp8_cuda_role(arguments.role) ||
      arguments.asset.payload.role != arguments.role ||
      !sm87_target_aot_fp8_cuda_asset_valid(arguments.asset) ||
      arguments.input == nullptr || arguments.output == nullptr ||
      reinterpret_cast<std::uintptr_t>(arguments.input) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(arguments.output) % 16U != 0U) {
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
  const auto input = sm87_target_aot_fp8_cuda_byte_range(
      arguments.input, input_values * sizeof(std::uint16_t));
  const auto payload = Sm87TargetAotFp8CudaByteRange{
      arguments.asset.payload.begin, arguments.asset.payload.end,
      arguments.asset.payload.valid};
  const auto output = sm87_target_aot_fp8_cuda_byte_range(
      arguments.output, output_values * sizeof(std::uint16_t));
  return input.valid && payload.valid && output.valid &&
         !sm87_target_aot_fp8_cuda_ranges_overlap(input, payload) &&
         !sm87_target_aot_fp8_cuda_ranges_overlap(input, output) &&
         !sm87_target_aot_fp8_cuda_ranges_overlap(payload, output);
}

struct Sm87TargetAotFp8CudaResources final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::size_t token_count = 0U;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  bool kernel_compiled = false;
  bool static_resources_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool
sm87_target_aot_fp8_cuda_resources_structurally_valid(
    const Sm87TargetAotFp8CudaResources& resources) noexcept {
  const auto plan =
      sm87_target_aot_projection_plan(resources.role, resources.token_count);
  if (!plan.valid() || !sm87_target_aot_fp8_cuda_role(resources.role) ||
      resources.static_resources_qualified ||
      resources.numerical_contract_qualified ||
      resources.production_dispatch_eligible) {
    return false;
  }
  if (!resources.kernel_compiled) {
    return resources.binary_version == 0 &&
           resources.registers_per_thread == 0 &&
           resources.static_shared_bytes == 0U &&
           resources.dynamic_shared_bytes == 0U &&
           resources.local_bytes == 0U &&
           resources.maximum_threads_per_block == 0 &&
           resources.active_blocks_per_sm == 0;
  }
  return resources.binary_version > 0 &&
         resources.registers_per_thread > 0 &&
         resources.dynamic_shared_bytes == plan.dynamic_shared_bytes &&
         resources.maximum_threads_per_block >=
             static_cast<int>(kSm87TargetAotProjectionThreads) &&
         resources.active_blocks_per_sm > 0;
}

// These declarations are admission-only.  A compiled body or a successful
// resource query still cannot set qualification or production-dispatch facts.
[[nodiscard]] int query_sm87_target_aot_fp8_cuda_resources(
    Sm87TargetAotProjectionRole role, std::size_t token_count,
    Sm87TargetAotFp8CudaResources* resources) noexcept;

[[nodiscard]] int launch_sm87_target_aot_fp8_cuda(
    const Sm87TargetAotFp8CudaArguments& arguments) noexcept;

}  // namespace q3x::kernels
