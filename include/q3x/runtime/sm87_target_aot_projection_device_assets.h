#pragma once

#include "q3x/kernels/sm87_target_aot_projection_cuda.h"
#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/resident_weights.h"
#include "q3x/runtime/sm87_target_aot_projection_assets.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace q3x::runtime {

class Sm87TargetAotProjectionDeviceAssets;
struct Sm87TargetAotProjectionDevicePreparationStats;
class ReferenceEngine;

inline constexpr std::size_t kSm87TargetAotProjectionDeviceLayerCount = 64U;
inline constexpr std::size_t kSm87TargetAotProjectionDeviceArtifactCount =
    2U * kSm87TargetAotProjectionDeviceLayerCount;
inline constexpr std::size_t kSm87TargetAotProjectionDeviceSourceCount =
    3U * kSm87TargetAotProjectionDeviceLayerCount;
inline constexpr std::uint64_t kSm87TargetAotProjectionDeviceArenaBytes =
    9'625'927'680ULL;
inline constexpr std::uint64_t
    kSm87TargetAotProjectionMaximumArtifactPayloadBytes = 100'270'080ULL;
inline constexpr std::uint64_t
    kSm87TargetAotProjectionMaximumHostStagingBytes =
        2U * kSm87TargetAotProjectionMaximumArtifactPayloadBytes;

struct Sm87TargetAotProjectionDeviceAssetDescriptor final {
  std::size_t layer_index = 0U;
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  std::uint64_t device_arena_offset = 0U;
  kernels::Sm87TargetAotProjectionPackedSourceInventory source_inventory{};
  kernels::Sm87TargetAotProjectionPackedManifest manifest{};
  kernels::Sm87TargetAotProjectionPackedTransformReceipt transform_receipt{};
  kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt upload_receipt{};
  kernels::Sm87TargetAotNvFp4CudaAssetView view{};
};

// Engine-lifetime owner for the exact 64-layer NVFP4 target-AOT projection
// inventory. The device arena contains payload bytes only. Host manifests,
// transform receipts, uploader receipts, and source inventories remain in the
// owner. This slice does not attach views to ModelWeights or any runner.
class Sm87TargetAotProjectionDeviceAssets final {
 public:
  Sm87TargetAotProjectionDeviceAssets() noexcept = default;
  ~Sm87TargetAotProjectionDeviceAssets();

  Sm87TargetAotProjectionDeviceAssets(
      const Sm87TargetAotProjectionDeviceAssets&) = delete;
  Sm87TargetAotProjectionDeviceAssets& operator=(
      const Sm87TargetAotProjectionDeviceAssets&) = delete;
  Sm87TargetAotProjectionDeviceAssets(
      Sm87TargetAotProjectionDeviceAssets&& other) noexcept;
  Sm87TargetAotProjectionDeviceAssets& operator=(
      Sm87TargetAotProjectionDeviceAssets&& other) noexcept;

  void release() noexcept;

  [[nodiscard]] bool empty() const noexcept {
    return arena_ == nullptr && bytes_ == 0U && descriptor_count_ == 0U &&
           allocation_identity_ == 0U && owner_identity_ == 0U;
  }

  // Structural presence only. It deliberately does not expose a naked
  // descriptor, receipt, or device view. Future execution must resolve an
  // asset through the private owner-backed loader/Engine capability.
  [[nodiscard]] bool has_asset(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept {
    return find(layer_index, role) != nullptr;
  }

 private:
  friend class ReferenceEngine;

  // Private loader-issued authority. The next Engine integration slice may
  // call this only through a ReferenceEngine member; there is deliberately no
  // public/free preparation or receipt-issuance function.
  [[nodiscard]] Sm87TargetAotProjectionDevicePreparationStats prepare(
      const ResidentWeights& resident, const ModelWeights& model_weights,
      std::uint64_t minimum_free_bytes_after_prepare);

  [[nodiscard]] const Sm87TargetAotProjectionDeviceAssetDescriptor* find(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;

  [[nodiscard]] const std::uint8_t* data() const noexcept { return arena_; }
  [[nodiscard]] std::uint64_t size_bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::uint64_t allocation_identity() const noexcept {
    return allocation_identity_;
  }
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return owner_identity_;
  }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }
  [[nodiscard]] std::size_t descriptor_count() const noexcept {
    return descriptor_count_;
  }

  std::uint8_t* arena_ = nullptr;
  std::uint64_t bytes_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  std::uint64_t owner_identity_ = 0U;
  std::int32_t device_ordinal_ = -1;
  std::array<Sm87TargetAotProjectionDeviceAssetDescriptor,
             kSm87TargetAotProjectionDeviceArtifactCount>
      descriptors_{};
  std::size_t descriptor_count_ = 0U;
};

struct Sm87TargetAotProjectionDevicePreparationStats final {
  bool enabled = false;
  bool hard_failure = false;
  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  std::uint64_t arena_bytes = 0U;
  std::uint64_t host_staging_peak_bytes = 0U;
  std::uint64_t source_d2h_bytes = 0U;
  std::uint64_t payload_h2d_bytes = 0U;
  std::uint64_t verification_d2h_bytes = 0U;
  int cuda_error = 0;
  std::string message;
};

// Loader-only all-or-nothing preparation. It consumes the exact pinned
// ResidentWeights and its already-bound ModelWeights view, authenticates all
// 192 canonical NVFP4 sources, transforms and validates one bounded artifact
// at a time, uploads into the exact arena, then reads each device payload back
// and compares its SHA-256 before issuing a receipt. It never mutates
// ModelWeights and is not callable from a request path.
[[nodiscard]] constexpr bool
sm87_target_aot_projection_device_assets_compiled() noexcept {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  return true;
#else
  return false;
#endif
}

static_assert(kSm87TargetAotProjectionDeviceLayerCount ==
              kQwen36DenseLayerCount);
static_assert(kSm87TargetAotProjectionDeviceArenaBytes ==
              kSm87TargetAotProjectionDeviceLayerCount *
                  (kernels::sm87_target_aot_projection_packed_layout(
                       kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp)
                       .payload_bytes +
                   kernels::sm87_target_aot_projection_packed_layout(
                       kernels::Sm87TargetAotProjectionRole::kNvFp4Down)
                       .payload_bytes));
static_assert(kSm87TargetAotProjectionMaximumArtifactPayloadBytes ==
              kernels::sm87_target_aot_projection_packed_layout(
                  kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp)
                  .payload_bytes);

}  // namespace q3x::runtime
