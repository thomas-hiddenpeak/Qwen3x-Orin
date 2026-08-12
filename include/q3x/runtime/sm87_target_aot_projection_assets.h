#pragma once

#include "q3x/kernels/sm87_target_aot_projection_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

// Explicit host byte views. A zero-length const view may use nullptr; every
// non-empty view and every mutable payload view requires a non-null,
// non-wrapping address interval.
struct Sm87TargetAotProjectionConstBytes {
  const std::uint8_t* data = nullptr;
  std::size_t bytes = 0U;
};

struct Sm87TargetAotProjectionMutableBytes {
  std::uint8_t* data = nullptr;
  std::size_t bytes = 0U;
};

struct Sm87TargetAotProjectionSourceBytes {
  kernels::Sm87TargetAotLogicalRole logical_role =
      kernels::Sm87TargetAotLogicalRole::kInvalid;
  std::uint64_t tensor_identity = 0U;
  std::uint32_t output_features = 0U;
  std::uint32_t input_features = 0U;
  // Canonical checkpoint order. FP8 is [N,K] bytes. NVFP4 is [N,K/2]
  // bytes with even K in the low nibble and odd K in the high nibble.
  Sm87TargetAotProjectionConstBytes packed_weight{};
  // Canonical NVFP4 [N,K/16] E4M3FN bytes. FP8 must supply an empty view.
  Sm87TargetAotProjectionConstBytes block_scale{};
  // Raw little-endian FP32 bits. The scalar remains outside the packed
  // payload and participates in scale_digest authentication.
  std::uint32_t tensor_scale_bits = 0U;
};

struct Sm87TargetAotProjectionSourceSet {
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  std::uint64_t inventory_identity = 0U;
  std::uint32_t source_count = 0U;
  std::array<Sm87TargetAotProjectionSourceBytes,
             kernels::kSm87TargetAotProjectionPackedMaxPartitions>
      sources{};
};

enum class Sm87TargetAotProjectionAssetError : std::uint8_t {
  kSuccess = 0U,
  kInvalidArgument,
  kInvalidLayout,
  kSizeOverflow,
  kSourceCountMismatch,
  kSourceMetadataMismatch,
  kSourceSizeMismatch,
  kSourceRangeOverflow,
  kSourceAliasing,
  kSourceDigestMismatch,
  kForbiddenNvFp4BlockScale,
  kPayloadSizeMismatch,
  kPayloadRangeOverflow,
  kPayloadAliasing,
  kPayloadDigestMismatch,
  kPayloadBijectionMismatch,
  kManifestMismatch,
  kTransformReceiptMismatch,
};

[[nodiscard]] const char* sm87_target_aot_projection_asset_error_string(
    Sm87TargetAotProjectionAssetError error) noexcept;

struct Sm87TargetAotProjectionSourceInspection {
  Sm87TargetAotProjectionAssetError error =
      Sm87TargetAotProjectionAssetError::kInvalidArgument;
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == Sm87TargetAotProjectionAssetError::kSuccess;
  }
};

struct Sm87TargetAotProjectionAssetBuildResult {
  Sm87TargetAotProjectionAssetError error =
      Sm87TargetAotProjectionAssetError::kInvalidArgument;
  kernels::Sm87TargetAotProjectionPackedManifest manifest{};
  kernels::Sm87TargetAotProjectionPackedTransformReceipt transform_receipt{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == Sm87TargetAotProjectionAssetError::kSuccess;
  }
};

// Standards-compliant SHA-256 over the supplied bytes. This is the primitive
// used for source and payload authentication; metadata-only pseudo-digests are
// never accepted by this asset path.
[[nodiscard]] bool sm87_target_aot_projection_sha256(
    Sm87TargetAotProjectionConstBytes bytes,
    kernels::Sm87TargetAotProjectionSha256Digest* digest) noexcept;

// Hash and authenticate every canonical source tensor, including the
// little-endian tensor_scale_bits suffix in scale_digest. No payload bytes are
// written by inspection.
[[nodiscard]] Sm87TargetAotProjectionSourceInspection
sm87_target_aot_projection_inspect_sources(
    const Sm87TargetAotProjectionSourceSet& sources) noexcept;

// Re-hash sources against expected_inventory, perform the bit-exact canonical
// [N,K] -> ConsumerN64K16LaneComponentV1 permutation, hash the resulting
// payload, and construct a sealed manifest plus transform receipt. The output
// span is the payload interval only; the 4096-byte manifest reservation is not
// serialized by this host API.
[[nodiscard]] Sm87TargetAotProjectionAssetBuildResult
sm87_target_aot_projection_build_asset(
    std::uint64_t artifact_identity,
    const Sm87TargetAotProjectionSourceSet& sources,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory&
        expected_inventory,
    Sm87TargetAotProjectionMutableBytes payload) noexcept;

// Re-hash current source and payload bytes, validate manifest/receipt metadata,
// rescan NVFP4 scale domains, and replay the packed-layout bijection. Payload
// corruption therefore fails even if a caller also rewrites and reseals its
// payload digest metadata.
[[nodiscard]] Sm87TargetAotProjectionAssetError
sm87_target_aot_projection_validate_asset(
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt& receipt,
    const Sm87TargetAotProjectionSourceSet& sources,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory&
        expected_inventory,
    Sm87TargetAotProjectionConstBytes payload) noexcept;

}  // namespace q3x::runtime
