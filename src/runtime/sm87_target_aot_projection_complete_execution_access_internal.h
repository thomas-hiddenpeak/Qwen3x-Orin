#pragma once

#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::runtime::target_aot_complete_execution_detail {

// Source-private immutable borrow of one artifact from the complete 256-entry
// target-AOT catalog.  This capability neither owns nor extends the Engine
// lifetime.  Every typed borrow revalidates the live ModelWeights attachment,
// owner/allocation/device snapshot, descriptor slot, encoding, range, and
// upload/readback receipt before exposing a CUDA view.
class Sm87TargetAotCompleteProjectionExecutionAsset final {
 public:
  [[nodiscard]] std::size_t layer_index() const noexcept {
    return layer_index_;
  }
  [[nodiscard]] kernels::Sm87TargetAotProjectionRole role() const noexcept {
    return role_;
  }
  [[nodiscard]] kernels::Sm87TargetAotProjectionEncoding encoding()
      const noexcept {
    return encoding_;
  }
  [[nodiscard]] std::uint64_t artifact_identity() const noexcept {
    return artifact_identity_;
  }
  [[nodiscard]] std::uint64_t source_inventory_identity() const noexcept {
    return source_inventory_identity_;
  }
  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return payload_bytes_;
  }

  // Exactly one typed borrow can succeed for a valid asset.  The other
  // encoding always returns null, as does either borrow after any owner,
  // attachment, descriptor, range, or receipt fact changes.
  [[nodiscard]] const kernels::Sm87TargetAotNvFp4CudaAssetView*
  borrow_nvfp4_cuda_asset() const noexcept;
  [[nodiscard]] const kernels::Sm87TargetAotFp8CudaAssetView*
  borrow_fp8_cuda_asset() const noexcept;

 private:
  friend class Sm87TargetAotCompleteProjectionExecutionAccess;

  Sm87TargetAotCompleteProjectionExecutionAsset(
      const ModelWeights* model_weights,
      const Sm87TargetAotCompleteProjectionDeviceAssets* owner,
      const Sm87TargetAotCompleteDeviceAssetDescriptor* descriptor,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uintptr_t arena_begin, std::uint64_t arena_bytes,
      std::int32_t device_ordinal, std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role,
      kernels::Sm87TargetAotProjectionEncoding encoding,
      std::uint64_t artifact_identity,
      std::uint64_t source_inventory_identity,
      std::uint64_t payload_bytes) noexcept
      : model_weights_(model_weights),
        owner_(owner),
        descriptor_(descriptor),
        owner_identity_(owner_identity),
        allocation_identity_(allocation_identity),
        arena_begin_(arena_begin),
        arena_bytes_(arena_bytes),
        device_ordinal_(device_ordinal),
        layer_index_(layer_index),
        role_(role),
        encoding_(encoding),
        artifact_identity_(artifact_identity),
        source_inventory_identity_(source_inventory_identity),
        payload_bytes_(payload_bytes) {}

  const ModelWeights* model_weights_ = nullptr;
  const Sm87TargetAotCompleteProjectionDeviceAssets* owner_ = nullptr;
  const Sm87TargetAotCompleteDeviceAssetDescriptor* descriptor_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  std::uintptr_t arena_begin_ = 0U;
  std::uint64_t arena_bytes_ = 0U;
  std::int32_t device_ordinal_ = -1;
  std::size_t layer_index_ = 0U;
  kernels::Sm87TargetAotProjectionRole role_ =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  kernels::Sm87TargetAotProjectionEncoding encoding_ =
      kernels::Sm87TargetAotProjectionEncoding::kInvalid;
  std::uint64_t artifact_identity_ = 0U;
  std::uint64_t source_inventory_identity_ = 0U;
  std::uint64_t payload_bytes_ = 0U;
};

// Source-private authority issued only from the complete owner's private,
// authenticated ModelWeights attachment.  No default or caller-populated
// constructor exists, and public CUDA launchers remain fail closed.
class Sm87TargetAotCompleteProjectionExecutionAccess final {
 public:
  Sm87TargetAotCompleteProjectionExecutionAccess(
      const Sm87TargetAotCompleteProjectionExecutionAccess&) = default;
  Sm87TargetAotCompleteProjectionExecutionAccess(
      Sm87TargetAotCompleteProjectionExecutionAccess&&) noexcept = default;
  Sm87TargetAotCompleteProjectionExecutionAccess& operator=(
      const Sm87TargetAotCompleteProjectionExecutionAccess&) = delete;
  Sm87TargetAotCompleteProjectionExecutionAccess& operator=(
      Sm87TargetAotCompleteProjectionExecutionAccess&&) = delete;

  [[nodiscard]] static std::optional<
      Sm87TargetAotCompleteProjectionExecutionAccess>
  bind(const ModelWeights& model_weights) noexcept;

  [[nodiscard]] bool attached() const noexcept;
  [[nodiscard]] constexpr std::size_t artifact_count() const noexcept {
    return descriptors_.size();
  }
  [[nodiscard]] std::optional<Sm87TargetAotCompleteProjectionExecutionAsset>
  resolve(std::size_t layer_index,
          kernels::Sm87TargetAotProjectionRole role) const noexcept;

  // T0-only fixture seam.  Definitions live exclusively in the host-test
  // translation unit, never in libq3x_core.  The synthetic device address is
  // never dereferenced or passed to CUDA, and the fixture still enters through
  // the same private owner/ModelWeights attachment as real execution access.
  [[nodiscard]] static std::optional<ModelWeights>
  make_complete_host_test_fixture(
      Sm87TargetAotCompleteProjectionDeviceAssets& owner) noexcept;
  [[nodiscard]] static bool poison_host_test_fixture_receipt(
      Sm87TargetAotCompleteProjectionDeviceAssets& owner,
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) noexcept;
  [[nodiscard]] static bool clear_host_test_fixture(
      Sm87TargetAotCompleteProjectionDeviceAssets& owner) noexcept;

 private:
  friend class Sm87TargetAotCompleteProjectionExecutionAsset;

  Sm87TargetAotCompleteProjectionExecutionAccess(
      const ModelWeights* model_weights,
      const Sm87TargetAotCompleteProjectionDeviceAssets* owner,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uintptr_t arena_begin, std::uint64_t arena_bytes,
      std::int32_t device_ordinal,
      const std::array<
          const Sm87TargetAotCompleteDeviceAssetDescriptor*,
          kSm87TargetAotCompleteProjectionDeviceArtifactCount>& descriptors)
      noexcept
      : model_weights_(model_weights),
        owner_(owner),
        owner_identity_(owner_identity),
        allocation_identity_(allocation_identity),
        arena_begin_(arena_begin),
        arena_bytes_(arena_bytes),
        device_ordinal_(device_ordinal),
        descriptors_(descriptors) {}

  [[nodiscard]] static bool attachment_matches(
      const ModelWeights* model_weights,
      const Sm87TargetAotCompleteProjectionDeviceAssets* owner,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uintptr_t arena_begin, std::uint64_t arena_bytes,
      std::int32_t device_ordinal) noexcept;
  [[nodiscard]] static bool descriptor_matches(
      const Sm87TargetAotCompleteProjectionDeviceAssets& owner,
      const Sm87TargetAotCompleteDeviceAssetDescriptor& descriptor,
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role,
      std::uint64_t expected_offset) noexcept;

  const ModelWeights* model_weights_ = nullptr;
  const Sm87TargetAotCompleteProjectionDeviceAssets* owner_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  std::uintptr_t arena_begin_ = 0U;
  std::uint64_t arena_bytes_ = 0U;
  std::int32_t device_ordinal_ = -1;
  std::array<const Sm87TargetAotCompleteDeviceAssetDescriptor*,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      descriptors_{};
};

}  // namespace q3x::runtime::target_aot_complete_execution_detail
