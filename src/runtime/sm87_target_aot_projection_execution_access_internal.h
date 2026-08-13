#pragma once

#include "q3x/runtime/sm87_target_aot_projection_device_assets.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::runtime::target_aot_execution_detail {

// Source-private, immutable borrow of one authenticated target-AOT artifact.
// It neither owns nor extends the Engine lifetime: ModelWeights, its attached
// non-movable owner, and every queued consumer must remain alive.  The public
// installed API never exposes this type or the CUDA asset view it gates.
class Sm87TargetAotProjectionExecutionAsset final {
 public:
  [[nodiscard]] std::size_t layer_index() const noexcept {
    return layer_index_;
  }
  [[nodiscard]] kernels::Sm87TargetAotProjectionRole role() const noexcept {
    return role_;
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

  // Returns null if the ModelWeights/owner attachment no longer matches the
  // immutable identity snapshot.  The returned view remains a non-owning
  // source-private borrow and may be consumed only while both owners live.
  [[nodiscard]] const kernels::Sm87TargetAotNvFp4CudaAssetView*
  borrow_cuda_asset() const noexcept;

 private:
  friend class Sm87TargetAotProjectionExecutionAccess;

  Sm87TargetAotProjectionExecutionAsset(
      const ModelWeights* model_weights,
      const Sm87TargetAotProjectionDeviceAssets* owner,
      const Sm87TargetAotProjectionDeviceAssetDescriptor* descriptor,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uintptr_t arena_begin, std::uint64_t arena_bytes,
      std::int32_t device_ordinal,
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role,
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
        artifact_identity_(artifact_identity),
        source_inventory_identity_(source_inventory_identity),
        payload_bytes_(payload_bytes) {}

  const ModelWeights* model_weights_ = nullptr;
  const Sm87TargetAotProjectionDeviceAssets* owner_ = nullptr;
  const Sm87TargetAotProjectionDeviceAssetDescriptor* descriptor_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  std::uintptr_t arena_begin_ = 0U;
  std::uint64_t arena_bytes_ = 0U;
  std::int32_t device_ordinal_ = -1;
  std::size_t layer_index_ = 0U;
  kernels::Sm87TargetAotProjectionRole role_ =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  std::uint64_t artifact_identity_ = 0U;
  std::uint64_t source_inventory_identity_ = 0U;
  std::uint64_t payload_bytes_ = 0U;
};

// Source-private capability issued only from an authenticated private
// ModelWeights attachment. Construction snapshots the exact complete catalog;
// callers cannot construct an empty or caller-populated execution view.
class Sm87TargetAotProjectionExecutionAccess final {
 public:
  Sm87TargetAotProjectionExecutionAccess(
      const Sm87TargetAotProjectionExecutionAccess&) = default;
  Sm87TargetAotProjectionExecutionAccess(
      Sm87TargetAotProjectionExecutionAccess&&) noexcept = default;
  Sm87TargetAotProjectionExecutionAccess& operator=(
      const Sm87TargetAotProjectionExecutionAccess&) = delete;
  Sm87TargetAotProjectionExecutionAccess& operator=(
      Sm87TargetAotProjectionExecutionAccess&&) = delete;

  [[nodiscard]] static std::optional<
      Sm87TargetAotProjectionExecutionAccess>
  bind(const ModelWeights& model_weights) noexcept;

  [[nodiscard]] bool attached() const noexcept;
  [[nodiscard]] constexpr std::size_t artifact_count() const noexcept {
    return descriptors_.size();
  }
  [[nodiscard]] std::optional<Sm87TargetAotProjectionExecutionAsset> resolve(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;

  // T0-only fixture seam: the deliberately private ownership graph has no
  // public constructor, so a no-GPU host test cannot otherwise exercise its
  // complete-catalog and invalidation contracts. Definitions live only in the
  // test translation unit (not libq3x_core); this header is not installed, the
  // synthetic address is never dereferenced, and clear_host_test_fixture must
  // run after ModelWeights detaches so that address is never passed to CUDA.
  [[nodiscard]] static std::optional<ModelWeights>
  make_complete_host_test_fixture(
      Sm87TargetAotProjectionDeviceAssets& owner) noexcept;
  [[nodiscard]] static bool clear_host_test_fixture(
      Sm87TargetAotProjectionDeviceAssets& owner) noexcept;

 private:
  friend class Sm87TargetAotProjectionExecutionAsset;

  Sm87TargetAotProjectionExecutionAccess(
      const ModelWeights* model_weights,
      const Sm87TargetAotProjectionDeviceAssets* owner,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uintptr_t arena_begin, std::uint64_t arena_bytes,
      std::int32_t device_ordinal,
      const std::array<const Sm87TargetAotProjectionDeviceAssetDescriptor*,
                       kSm87TargetAotProjectionDeviceArtifactCount>&
          descriptors) noexcept
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
      const Sm87TargetAotProjectionDeviceAssets* owner,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uintptr_t arena_begin, std::uint64_t arena_bytes,
      std::int32_t device_ordinal) noexcept;
  [[nodiscard]] static bool descriptor_matches(
      const Sm87TargetAotProjectionDeviceAssets& owner,
      const Sm87TargetAotProjectionDeviceAssetDescriptor& descriptor,
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role,
      std::uint64_t expected_offset) noexcept;
  [[nodiscard]] static std::size_t ordinal(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) noexcept;

  const ModelWeights* model_weights_ = nullptr;
  const Sm87TargetAotProjectionDeviceAssets* owner_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  std::uintptr_t arena_begin_ = 0U;
  std::uint64_t arena_bytes_ = 0U;
  std::int32_t device_ordinal_ = -1;
  std::array<const Sm87TargetAotProjectionDeviceAssetDescriptor*,
             kSm87TargetAotProjectionDeviceArtifactCount>
      descriptors_{};
};

}  // namespace q3x::runtime::target_aot_execution_detail
