#pragma once

#include "q3x/kernels/sm87_target_aot_projection_cuda.h"
#include "q3x/kernels/sm87_target_aot_projection_fp8_cuda.h"
#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/resident_weights.h"
#include "q3x/runtime/sm87_target_aot_projection_assets.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>

namespace q3x::runtime {

class ReferenceEngine;

namespace target_aot_complete_execution_detail {
class Sm87TargetAotCompleteProjectionExecutionAccess;
}

inline constexpr std::size_t
    kSm87TargetAotCompleteProjectionDeviceLayerCount = 64U;
inline constexpr std::size_t
    kSm87TargetAotCompleteProjectionArtifactsPerLayer = 4U;
inline constexpr std::size_t
    kSm87TargetAotCompleteProjectionDeviceArtifactCount =
        kSm87TargetAotCompleteProjectionDeviceLayerCount *
        kSm87TargetAotCompleteProjectionArtifactsPerLayer;
inline constexpr std::size_t
    kSm87TargetAotCompleteProjectionDeviceSourceCount = 400U;
inline constexpr std::uint64_t
    kSm87TargetAotCompleteProjectionDeviceArenaBytes = 16'840'130'560ULL;
inline constexpr std::uint64_t
    kSm87TargetAotCompleteProjectionCanonicalSourceD2hBytes =
        16'840'132'160ULL;
inline constexpr std::uint64_t
    kSm87TargetAotCompleteProjectionMaximumArtifactPayloadBytes =
        100'270'080ULL;
inline constexpr std::uint64_t
    kSm87TargetAotCompleteProjectionMaximumHostStagingBytes =
        2U * kSm87TargetAotCompleteProjectionMaximumArtifactPayloadBytes;

enum class Sm87TargetAotCompletePreparationPhase : std::uint8_t {
  kInvalid = 0U,
  kEngineStartup,
};

// V2 deliberately admits only an engine-startup online transform. Bundle
// creation and direct loading require a future, separately versioned
// persistence ABI; carrying explicit paths here makes accidental reuse of the
// frozen NVFP4-only v1 bundle fail closed instead of silently doing nothing.
struct Sm87TargetAotCompleteOnlinePreparationRequest final {
  Sm87TargetAotCompletePreparationPhase phase =
      Sm87TargetAotCompletePreparationPhase::kInvalid;
  std::uint64_t minimum_free_bytes_after_prepare = 0U;
  std::filesystem::path create_bundle_path;
  std::filesystem::path load_bundle_path;
};

[[nodiscard]] inline bool sm87_target_aot_complete_online_request_valid(
    const Sm87TargetAotCompleteOnlinePreparationRequest& request) noexcept {
  return request.phase == Sm87TargetAotCompletePreparationPhase::kEngineStartup &&
         request.create_bundle_path.empty() && request.load_bundle_path.empty();
}

[[nodiscard]] constexpr bool
sm87_target_aot_complete_role_is_fp8(
    const kernels::Sm87TargetAotProjectionRole role) noexcept {
  return role == kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ ||
         role == kernels::Sm87TargetAotProjectionRole::kFp8FullQkv ||
         role == kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput;
}

[[nodiscard]] constexpr bool
sm87_target_aot_complete_role_is_nvfp4(
    const kernels::Sm87TargetAotProjectionRole role) noexcept {
  return role == kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp ||
         role == kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
}

[[nodiscard]] constexpr bool sm87_target_aot_complete_role_valid(
    const kernels::Sm87TargetAotProjectionRole role) noexcept {
  return sm87_target_aot_complete_role_is_fp8(role) ||
         sm87_target_aot_complete_role_is_nvfp4(role);
}

[[nodiscard]] constexpr bool sm87_target_aot_complete_is_full_layer(
    const std::size_t layer_index) noexcept {
  return layer_index < kSm87TargetAotCompleteProjectionDeviceLayerCount &&
         ((layer_index + 1U) % 4U) == 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_target_aot_complete_descriptor_ordinal(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionRole role) noexcept {
  if (layer_index >= kSm87TargetAotCompleteProjectionDeviceLayerCount) {
    return kSm87TargetAotCompleteProjectionDeviceArtifactCount;
  }
  std::size_t slot = kSm87TargetAotCompleteProjectionArtifactsPerLayer;
  switch (role) {
    case kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp:
      slot = 0U;
      break;
    case kernels::Sm87TargetAotProjectionRole::kNvFp4Down:
      slot = 1U;
      break;
    case kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ:
      if (!sm87_target_aot_complete_is_full_layer(layer_index)) {
        slot = 2U;
      }
      break;
    case kernels::Sm87TargetAotProjectionRole::kFp8FullQkv:
      if (sm87_target_aot_complete_is_full_layer(layer_index)) {
        slot = 2U;
      }
      break;
    case kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput:
      slot = 3U;
      break;
    case kernels::Sm87TargetAotProjectionRole::kInvalid:
    default:
      break;
  }
  return slot < kSm87TargetAotCompleteProjectionArtifactsPerLayer
             ? layer_index *
                       kSm87TargetAotCompleteProjectionArtifactsPerLayer +
                   slot
             : kSm87TargetAotCompleteProjectionDeviceArtifactCount;
}

[[nodiscard]] constexpr std::uint64_t
sm87_target_aot_complete_expected_arena_offset(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionRole role) noexcept {
  if (sm87_target_aot_complete_descriptor_ordinal(layer_index, role) >=
      kSm87TargetAotCompleteProjectionDeviceArtifactCount) {
    return kSm87TargetAotCompleteProjectionDeviceArenaBytes;
  }
  const std::uint64_t gate_up =
      kernels::sm87_target_aot_projection_packed_layout(
          kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp)
          .payload_bytes;
  const std::uint64_t down =
      kernels::sm87_target_aot_projection_packed_layout(
          kernels::Sm87TargetAotProjectionRole::kNvFp4Down)
          .payload_bytes;
  const std::uint64_t gdn =
      kernels::sm87_target_aot_projection_packed_layout(
          kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ)
          .payload_bytes;
  const std::uint64_t full =
      kernels::sm87_target_aot_projection_packed_layout(
          kernels::Sm87TargetAotProjectionRole::kFp8FullQkv)
          .payload_bytes;
  const std::uint64_t output =
      kernels::sm87_target_aot_projection_packed_layout(
          kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput)
          .payload_bytes;
  const std::uint64_t full_layers_before = layer_index / 4U;
  const std::uint64_t linear_layers_before =
      layer_index - full_layers_before;
  std::uint64_t offset =
      layer_index * (gate_up + down + output) +
      linear_layers_before * gdn + full_layers_before * full;
  if (role == kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp) {
    return offset;
  }
  offset += gate_up;
  if (role == kernels::Sm87TargetAotProjectionRole::kNvFp4Down) {
    return offset;
  }
  offset += down;
  const auto input_role = sm87_target_aot_complete_is_full_layer(layer_index)
                              ? kernels::Sm87TargetAotProjectionRole::
                                    kFp8FullQkv
                              : kernels::Sm87TargetAotProjectionRole::
                                    kFp8GdnQkvZ;
  if (role == input_role) {
    return offset;
  }
  offset += sm87_target_aot_complete_is_full_layer(layer_index) ? full : gdn;
  return role ==
                 kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput
             ? offset
             : kSm87TargetAotCompleteProjectionDeviceArenaBytes;
}

[[nodiscard]] constexpr bool sm87_target_aot_complete_payload_view_zero(
    const kernels::Sm87TargetAotProjectionPackedPayloadView& payload) noexcept {
  return payload.role == kernels::Sm87TargetAotProjectionRole::kInvalid &&
         payload.plan_identity ==
             kernels::Sm87TargetAotProjectionPackedPlanIdentity::kInvalid &&
         payload.layout_identity ==
             kernels::Sm87TargetAotProjectionPackedLayoutIdentity::kInvalid &&
         payload.begin == 0U && payload.end == 0U && payload.bytes == 0U &&
         !payload.valid;
}

template <typename Receipt>
[[nodiscard]] constexpr bool
sm87_target_aot_complete_common_upload_receipt_zero(
    const Receipt& receipt) noexcept {
  return receipt.receipt_identity == 0U && receipt.artifact_identity == 0U &&
         receipt.source_inventory_identity == 0U &&
         receipt.role == kernels::Sm87TargetAotProjectionRole::kInvalid &&
         receipt.plan_identity ==
             kernels::Sm87TargetAotProjectionPackedPlanIdentity::kInvalid &&
         receipt.layout_identity ==
             kernels::Sm87TargetAotProjectionPackedLayoutIdentity::kInvalid &&
         receipt.transform_identity ==
             kernels::Sm87TargetAotProjectionPackedTransformIdentity::
                 kInvalid &&
         receipt.host_payload_offset == 0U &&
         receipt.host_payload_bytes == 0U &&
         kernels::sm87_target_aot_projection_digest_is_zero(
             receipt.host_payload_digest) &&
         receipt.host_manifest_seal.value == 0U &&
         receipt.tensor_scale_count == 0U &&
         receipt.device_allocation_identity == 0U &&
         receipt.device_allocation_owner_identity == 0U &&
         receipt.device_ordinal == -1 &&
         receipt.device_allocation_begin == 0U &&
         receipt.device_allocation_end == 0U &&
         receipt.device_allocation_bytes == 0U &&
         receipt.device_payload_begin == 0U &&
         receipt.device_payload_end == 0U &&
         receipt.device_payload_bytes == 0U &&
         receipt.upload_stream_owner_identity == 0U &&
         receipt.upload_stream_identity == 0U &&
         receipt.upload_completion_event_identity == 0U &&
         receipt.verification_stream_owner_identity == 0U &&
         receipt.verification_stream_identity == 0U &&
         receipt.verification_completion_event_identity == 0U &&
         receipt.verification_readback_bytes == 0U &&
         kernels::sm87_target_aot_projection_digest_is_zero(
             receipt.verification_readback_digest) &&
         !receipt.host_payload_digest_verified_before_copy &&
         !receipt.host_payload_immutable_until_completion &&
         !receipt.copy_enqueued_to_exact_payload_range &&
         !receipt.completion_event_recorded_after_copy &&
         !receipt.completion_event_observed && !receipt.upload_completed &&
         !receipt.verification_copy_enqueued_from_exact_payload_range &&
         !receipt.verification_event_recorded_after_copy &&
         !receipt.verification_event_observed &&
         !receipt.verification_completed &&
         !receipt.device_payload_matches_host_payload &&
         !receipt.allocation_retained_for_asset_lifetime;
}

template <typename Receipt>
[[nodiscard]] constexpr bool
sm87_target_aot_complete_same_common_upload_receipt(
    const Receipt& left, const Receipt& right) noexcept {
  return left.receipt_identity == right.receipt_identity &&
         left.artifact_identity == right.artifact_identity &&
         left.source_inventory_identity == right.source_inventory_identity &&
         left.role == right.role && left.plan_identity == right.plan_identity &&
         left.layout_identity == right.layout_identity &&
         left.transform_identity == right.transform_identity &&
         left.host_payload_offset == right.host_payload_offset &&
         left.host_payload_bytes == right.host_payload_bytes &&
         left.host_payload_digest == right.host_payload_digest &&
         left.host_manifest_seal == right.host_manifest_seal &&
         left.tensor_scale_count == right.tensor_scale_count &&
         left.device_allocation_identity == right.device_allocation_identity &&
         left.device_allocation_owner_identity ==
             right.device_allocation_owner_identity &&
         left.device_ordinal == right.device_ordinal &&
         left.device_allocation_begin == right.device_allocation_begin &&
         left.device_allocation_end == right.device_allocation_end &&
         left.device_allocation_bytes == right.device_allocation_bytes &&
         left.device_payload_begin == right.device_payload_begin &&
         left.device_payload_end == right.device_payload_end &&
         left.device_payload_bytes == right.device_payload_bytes &&
         left.upload_stream_owner_identity ==
             right.upload_stream_owner_identity &&
         left.upload_stream_identity == right.upload_stream_identity &&
         left.upload_completion_event_identity ==
             right.upload_completion_event_identity &&
         left.verification_stream_owner_identity ==
             right.verification_stream_owner_identity &&
         left.verification_stream_identity ==
             right.verification_stream_identity &&
         left.verification_completion_event_identity ==
             right.verification_completion_event_identity &&
         left.verification_readback_bytes ==
             right.verification_readback_bytes &&
         left.verification_readback_digest ==
             right.verification_readback_digest &&
         left.host_payload_digest_verified_before_copy ==
             right.host_payload_digest_verified_before_copy &&
         left.host_payload_immutable_until_completion ==
             right.host_payload_immutable_until_completion &&
         left.copy_enqueued_to_exact_payload_range ==
             right.copy_enqueued_to_exact_payload_range &&
         left.completion_event_recorded_after_copy ==
             right.completion_event_recorded_after_copy &&
         left.completion_event_observed == right.completion_event_observed &&
         left.upload_completed == right.upload_completed &&
         left.verification_copy_enqueued_from_exact_payload_range ==
             right.verification_copy_enqueued_from_exact_payload_range &&
         left.verification_event_recorded_after_copy ==
             right.verification_event_recorded_after_copy &&
         left.verification_event_observed ==
             right.verification_event_observed &&
         left.verification_completed == right.verification_completed &&
         left.device_payload_matches_host_payload ==
             right.device_payload_matches_host_payload &&
         left.allocation_retained_for_asset_lifetime ==
             right.allocation_retained_for_asset_lifetime;
}

[[nodiscard]] constexpr bool
sm87_target_aot_complete_same_nvfp4_upload_receipt(
    const kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt& left,
    const kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt& right) noexcept {
  return sm87_target_aot_complete_same_common_upload_receipt(left, right) &&
         left.tensor_scale_bits == right.tensor_scale_bits;
}

[[nodiscard]] constexpr bool
sm87_target_aot_complete_same_fp8_upload_receipt(
    const kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt& left,
    const kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt& right) noexcept {
  return sm87_target_aot_complete_same_common_upload_receipt(left, right) &&
         left.tensor_scale_bits == right.tensor_scale_bits &&
         left.compensated_tensor_scale_bf16_bits ==
             right.compensated_tensor_scale_bf16_bits;
}

[[nodiscard]] constexpr bool
sm87_target_aot_complete_nvfp4_upload_receipt_zero(
    const kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt& receipt) noexcept {
  if (!sm87_target_aot_complete_common_upload_receipt_zero(receipt)) {
    return false;
  }
  for (const std::uint32_t bits : receipt.tensor_scale_bits) {
    if (bits != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_target_aot_complete_fp8_upload_receipt_zero(
    const kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt& receipt) noexcept {
  if (!sm87_target_aot_complete_common_upload_receipt_zero(receipt)) {
    return false;
  }
  for (std::size_t index = 0U; index < receipt.tensor_scale_bits.size();
       ++index) {
    if (receipt.tensor_scale_bits[index] != 0U ||
        receipt.compensated_tensor_scale_bf16_bits[index] != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_target_aot_complete_nvfp4_view_zero(
    const kernels::Sm87TargetAotNvFp4CudaAssetView& view) noexcept {
  return !view.valid && sm87_target_aot_complete_payload_view_zero(view.payload) &&
         view.artifact_identity == 0U && view.source_inventory_identity == 0U &&
         view.transform_identity ==
             kernels::Sm87TargetAotProjectionPackedTransformIdentity::
                 kInvalid &&
         kernels::sm87_target_aot_projection_digest_is_zero(
             view.host_payload_digest) &&
         view.host_manifest_seal.value == 0U &&
         view.tensor_scale_bits[0U] == 0U &&
         view.tensor_scale_bits[1U] == 0U && view.tensor_scale_count == 0U &&
         !view.no_request_time_repacking &&
         sm87_target_aot_complete_nvfp4_upload_receipt_zero(
             view.device_upload_receipt);
}

[[nodiscard]] constexpr bool sm87_target_aot_complete_fp8_view_zero(
    const kernels::Sm87TargetAotFp8CudaAssetView& view) noexcept {
  return !view.valid && sm87_target_aot_complete_payload_view_zero(view.payload) &&
         view.artifact_identity == 0U && view.source_inventory_identity == 0U &&
         view.transform_identity ==
             kernels::Sm87TargetAotProjectionPackedTransformIdentity::
                 kInvalid &&
         kernels::sm87_target_aot_projection_digest_is_zero(
             view.host_payload_digest) &&
         view.host_manifest_seal.value == 0U &&
         view.tensor_scale_bits[0U] == 0U &&
         view.tensor_scale_bits[1U] == 0U &&
         view.tensor_scale_bits[2U] == 0U &&
         view.compensated_tensor_scale_bf16_bits[0U] == 0U &&
         view.compensated_tensor_scale_bf16_bits[1U] == 0U &&
         view.compensated_tensor_scale_bf16_bits[2U] == 0U &&
         view.tensor_scale_count == 0U && !view.no_request_time_repacking &&
         !view.no_request_time_scale_conversion &&
         sm87_target_aot_complete_fp8_upload_receipt_zero(
             view.device_upload_receipt);
}

struct Sm87TargetAotCompleteDeviceAssetDescriptor final {
  std::size_t layer_index = 0U;
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  std::uint64_t device_arena_offset = 0U;
  kernels::Sm87TargetAotProjectionPackedSourceInventory source_inventory{};
  kernels::Sm87TargetAotProjectionPackedManifest manifest{};
  kernels::Sm87TargetAotProjectionPackedTransformReceipt transform_receipt{};
  kernels::Sm87TargetAotProjectionEncoding encoding =
      kernels::Sm87TargetAotProjectionEncoding::kInvalid;
  kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt nvfp4_upload_receipt{};
  kernels::Sm87TargetAotNvFp4CudaAssetView nvfp4_view{};
  kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt fp8_upload_receipt{};
  kernels::Sm87TargetAotFp8CudaAssetView fp8_view{};
};

[[nodiscard]] inline bool
sm87_target_aot_complete_device_descriptor_valid(
    const Sm87TargetAotCompleteDeviceAssetDescriptor& descriptor,
    const std::size_t expected_layer_index,
    const kernels::Sm87TargetAotProjectionRole expected_role,
    const std::uint64_t expected_arena_offset,
    const std::uintptr_t arena_begin, const std::uint64_t arena_bytes,
    const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::int32_t device_ordinal) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(expected_role);
  if (!layout.valid() ||
      sm87_target_aot_complete_descriptor_ordinal(expected_layer_index,
                                                   expected_role) >=
          kSm87TargetAotCompleteProjectionDeviceArtifactCount ||
      descriptor.layer_index != expected_layer_index ||
      descriptor.role != expected_role ||
      descriptor.device_arena_offset != expected_arena_offset ||
      descriptor.encoding != layout.encoding ||
      descriptor.source_inventory.role != expected_role ||
      descriptor.manifest.role != expected_role ||
      descriptor.manifest.encoding != descriptor.encoding ||
      descriptor.manifest.source_inventory_identity !=
          descriptor.source_inventory.identity ||
      descriptor.manifest.payload_bytes != layout.payload_bytes ||
      expected_arena_offset > arena_bytes ||
      layout.payload_bytes > arena_bytes - expected_arena_offset ||
      arena_begin == 0U || arena_bytes == 0U || owner_identity == 0U ||
      allocation_identity == 0U || device_ordinal < 0 ||
      arena_bytes > std::numeric_limits<std::uintptr_t>::max() - arena_begin ||
      !kernels::sm87_target_aot_projection_validate_transform_receipt(
          descriptor.manifest, descriptor.source_inventory,
          descriptor.transform_receipt)) {
    return false;
  }
  const std::uintptr_t payload_begin =
      arena_begin + static_cast<std::uintptr_t>(expected_arena_offset);
  const auto receipt_owner_matches =
      [&](const auto& receipt) noexcept {
        return receipt.device_allocation_identity == allocation_identity &&
               receipt.device_allocation_owner_identity == owner_identity &&
               receipt.device_ordinal == device_ordinal &&
               receipt.device_allocation_begin == arena_begin &&
               receipt.device_allocation_end == arena_begin + arena_bytes &&
               receipt.device_allocation_bytes == arena_bytes &&
               receipt.device_payload_begin == payload_begin &&
               receipt.device_payload_end ==
                   payload_begin + layout.payload_bytes &&
               receipt.device_payload_bytes == layout.payload_bytes;
      };
  if (sm87_target_aot_complete_role_is_nvfp4(expected_role)) {
    return kernels::
               sm87_target_aot_nvfp4_cuda_device_upload_receipt_structurally_valid(
                   descriptor.manifest, descriptor.nvfp4_upload_receipt) &&
           kernels::sm87_target_aot_nvfp4_cuda_asset_valid(
               descriptor.nvfp4_view) &&
           sm87_target_aot_complete_same_nvfp4_upload_receipt(
               descriptor.nvfp4_upload_receipt,
               descriptor.nvfp4_view.device_upload_receipt) &&
           receipt_owner_matches(descriptor.nvfp4_upload_receipt) &&
           descriptor.nvfp4_view.payload.begin == payload_begin &&
           sm87_target_aot_complete_fp8_upload_receipt_zero(
               descriptor.fp8_upload_receipt) &&
           sm87_target_aot_complete_fp8_view_zero(descriptor.fp8_view);
  }
  if (sm87_target_aot_complete_role_is_fp8(expected_role)) {
    return kernels::
               sm87_target_aot_fp8_cuda_device_upload_receipt_structurally_valid(
                   descriptor.manifest, descriptor.fp8_upload_receipt) &&
           kernels::sm87_target_aot_fp8_cuda_asset_valid(descriptor.fp8_view) &&
           sm87_target_aot_complete_same_fp8_upload_receipt(
               descriptor.fp8_upload_receipt,
               descriptor.fp8_view.device_upload_receipt) &&
           receipt_owner_matches(descriptor.fp8_upload_receipt) &&
           descriptor.fp8_view.payload.begin == payload_begin &&
           sm87_target_aot_complete_nvfp4_upload_receipt_zero(
               descriptor.nvfp4_upload_receipt) &&
           sm87_target_aot_complete_nvfp4_view_zero(descriptor.nvfp4_view);
  }
  return false;
}

struct Sm87TargetAotCompleteDevicePreparationStats final {
  bool enabled = false;
  bool hard_failure = false;
  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  std::uint64_t arena_bytes = 0U;
  std::uint64_t host_staging_peak_bytes = 0U;
  std::uint64_t source_d2h_bytes = 0U;
  std::uint64_t payload_h2d_bytes = 0U;
  std::uint64_t verification_d2h_bytes = 0U;
  bool bundle_create_rejected = false;
  bool bundle_load_rejected = false;
  std::string verified_payload_catalog_sha256;
  std::uint64_t owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::int32_t device_ordinal = -1;
  int cuda_error = 0;
  std::string message;
};

// Independent complete-projection owner. It never attaches sidecars to
// ModelWeights and never prepares on a request path. The only preparation
// entry is private to ReferenceEngine startup; execution borrows remain
// source-private and owner-backed.
class Sm87TargetAotCompleteProjectionDeviceAssets final {
 public:
  Sm87TargetAotCompleteProjectionDeviceAssets() noexcept = default;
  ~Sm87TargetAotCompleteProjectionDeviceAssets();

  Sm87TargetAotCompleteProjectionDeviceAssets(
      const Sm87TargetAotCompleteProjectionDeviceAssets&) = delete;
  Sm87TargetAotCompleteProjectionDeviceAssets& operator=(
      const Sm87TargetAotCompleteProjectionDeviceAssets&) = delete;
  Sm87TargetAotCompleteProjectionDeviceAssets(
      Sm87TargetAotCompleteProjectionDeviceAssets&&) = delete;
  Sm87TargetAotCompleteProjectionDeviceAssets& operator=(
      Sm87TargetAotCompleteProjectionDeviceAssets&&) = delete;

  bool release() noexcept;

  [[nodiscard]] bool empty() const noexcept {
    return arena_ == nullptr && bytes_ == 0U && descriptor_count_ == 0U &&
           allocation_identity_ == 0U && owner_identity_ == 0U &&
           prepared_resident_ == nullptr && prepared_model_weights_ == nullptr &&
           !execution_bound_;
  }

  [[nodiscard]] bool has_asset(
      const std::size_t layer_index,
      const kernels::Sm87TargetAotProjectionRole role) const noexcept {
    return find(layer_index, role) != nullptr;
  }

 private:
  friend class ReferenceEngine;
  friend class ModelWeights;
  friend class target_aot_complete_execution_detail::
      Sm87TargetAotCompleteProjectionExecutionAccess;

  [[nodiscard]] Sm87TargetAotCompleteDevicePreparationStats prepare_online(
      const ResidentWeights& resident, const ModelWeights& model_weights,
      const Sm87TargetAotCompleteOnlinePreparationRequest& request);

  [[nodiscard]] const Sm87TargetAotCompleteDeviceAssetDescriptor* find(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;

  void release_unconditionally() noexcept;

  std::uint8_t* arena_ = nullptr;
  std::uint64_t bytes_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  std::uint64_t owner_identity_ = 0U;
  std::int32_t device_ordinal_ = -1;
  std::array<Sm87TargetAotCompleteDeviceAssetDescriptor,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      descriptors_{};
  std::size_t descriptor_count_ = 0U;
  const ResidentWeights* prepared_resident_ = nullptr;
  const ModelWeights* prepared_model_weights_ = nullptr;
  bool execution_bound_ = false;
};

[[nodiscard]] constexpr bool
sm87_target_aot_complete_projection_device_assets_compiled() noexcept {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  return true;
#else
  return false;
#endif
}

static_assert(kSm87TargetAotCompleteProjectionDeviceLayerCount ==
              kQwen36DenseLayerCount);
static_assert(kSm87TargetAotCompleteProjectionDeviceArtifactCount == 256U);
static_assert(kSm87TargetAotCompleteProjectionDeviceSourceCount == 400U);
static_assert(kSm87TargetAotCompleteProjectionCanonicalSourceD2hBytes ==
              kSm87TargetAotCompleteProjectionDeviceArenaBytes +
                  kSm87TargetAotCompleteProjectionDeviceSourceCount *
                      sizeof(float));
static_assert(kSm87TargetAotCompleteProjectionDeviceArenaBytes ==
              64U *
                  (kernels::sm87_target_aot_projection_packed_layout(
                       kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp)
                       .payload_bytes +
                   kernels::sm87_target_aot_projection_packed_layout(
                       kernels::Sm87TargetAotProjectionRole::kNvFp4Down)
                       .payload_bytes +
                   kernels::sm87_target_aot_projection_packed_layout(
                       kernels::Sm87TargetAotProjectionRole::
                           kFp8AttentionOutput)
                       .payload_bytes) +
              48U *
                  kernels::sm87_target_aot_projection_packed_layout(
                      kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ)
                      .payload_bytes +
              16U *
                  kernels::sm87_target_aot_projection_packed_layout(
                      kernels::Sm87TargetAotProjectionRole::kFp8FullQkv)
                      .payload_bytes);
static_assert(kSm87TargetAotCompleteProjectionMaximumArtifactPayloadBytes ==
              kernels::sm87_target_aot_projection_packed_layout(
                  kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp)
                  .payload_bytes);

}  // namespace q3x::runtime
