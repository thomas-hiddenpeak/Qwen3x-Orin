#pragma once

#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::runtime::sm87_macrofeed_v3_p40_execution_package_detail {
class Sm87MacroFeedV3P40ExecutionPackage;
class Sm87MacroFeedV3P40ProjectionStartupBinding;
}  // namespace q3x::runtime::sm87_macrofeed_v3_p40_execution_package_detail

namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail {
class Sm87MacroFeedV4P40StartupPackage;
class Sm87MacroFeedV4P40StartupPackageHostTestFixture;
}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail

namespace q3x::runtime::target_aot_complete_execution_detail {

struct Sm87TargetAotCompleteHostTestBf16AbPair final {
  LinearWeight a;
  LinearWeight b;
};

struct Sm87TargetAotCompleteHostTestLayerNormPair final {
  Bf16VectorWeight input_layernorm;
  Bf16VectorWeight post_attention_layernorm;
};

struct Sm87TargetAotCompleteHostTestFullQkNormPair final {
  Bf16VectorWeight q_norm;
  Bf16VectorWeight k_norm;
};

struct Sm87TargetAotCompleteHostTestRequestBoundary final {
  Bf16LinearWeight embedding;
  Bf16VectorWeight final_norm;
  LinearWeight lm_head;
};

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
  friend class sm87_macrofeed_v3_p40_execution_package_detail::
      Sm87MacroFeedV3P40ExecutionPackage;
  friend class sm87_macrofeed_v3_p40_execution_package_detail::
      Sm87MacroFeedV3P40ProjectionStartupBinding;
  friend class sm87_macrofeed_v4_p40_startup_package_detail::
      Sm87MacroFeedV4P40StartupPackage;

  Sm87TargetAotCompleteProjectionExecutionAsset(
      const ModelWeights* model_weights,
      const Sm87TargetAotCompleteProjectionDeviceAssets* owner,
      const Sm87TargetAotCompleteDeviceAssetDescriptor* descriptor,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uint64_t device_identity,
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
        device_identity_(device_identity),
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
  std::uint64_t device_identity_ = 0U;
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
  // Source-private identity snapshot for a higher-level composition root.
  // These values come from the live, authenticated ModelWeights attachment;
  // they are not caller-provided deployment claims.  catalog_identity()
  // folds every one of the 256 descriptor artifact/source identities after
  // revalidating the complete attachment.
  [[nodiscard]] std::uint64_t owner_identity() const noexcept {
    return attached() ? owner_identity_ : 0U;
  }
  [[nodiscard]] std::uint64_t allocation_identity() const noexcept {
    return attached() ? allocation_identity_ : 0U;
  }
  [[nodiscard]] std::uint64_t device_identity() const noexcept {
    return attached() ? device_identity_ : 0U;
  }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return attached() ? device_ordinal_ : -1;
  }
  [[nodiscard]] std::uint64_t catalog_identity() const noexcept;
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
  // Test-TU definition only.  It issues an explicit projection/test-resident
  // authority for Startup host tests.  It is distinct from bind(): no such
  // access can satisfy the normal ResidentWeights gate or mint a production
  // request-boundary execution catalog.
  [[nodiscard]] static std::optional<
      Sm87TargetAotCompleteProjectionExecutionAccess>
  bind_complete_host_test_fixture(
      const ModelWeights& model_weights, std::uintptr_t resident_arena_begin,
      std::uint64_t resident_arena_bytes) noexcept;
  // Test-TU definition only.  This installs ModelWeights fields before a
  // startup package is constructed; it never mutates an already-issued
  // package and never creates a raw-pointer capability of its own.
  [[nodiscard]] static bool install_complete_host_test_bf16_ab_pairs(
      ModelWeights& model_weights,
      const Sm87TargetAotCompleteHostTestBf16AbPair* pairs,
      std::size_t pair_count) noexcept;
  [[nodiscard]] static bool install_complete_host_test_layer_norm_pairs(
      ModelWeights& model_weights,
      const Sm87TargetAotCompleteHostTestLayerNormPair* pairs,
      std::size_t pair_count) noexcept;
  [[nodiscard]] static bool install_complete_host_test_full_qk_norm_pairs(
      ModelWeights& model_weights,
      const Sm87TargetAotCompleteHostTestFullQkNormPair* pairs,
      std::size_t pair_count) noexcept;
  [[nodiscard]] static bool install_complete_host_test_request_boundary(
      ModelWeights& model_weights,
      const Sm87TargetAotCompleteHostTestRequestBoundary& boundary) noexcept;
  [[nodiscard]] static bool poison_host_test_fixture_receipt(
      Sm87TargetAotCompleteProjectionDeviceAssets& owner,
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) noexcept;
  [[nodiscard]] static bool tamper_host_test_fixture_source_identity(
      Sm87TargetAotCompleteProjectionDeviceAssets& owner,
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role,
      std::size_t source_index) noexcept;
  [[nodiscard]] static bool tamper_host_test_fixture_scale_bits(
      Sm87TargetAotCompleteProjectionDeviceAssets& owner,
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role,
      std::size_t source_index) noexcept;
  [[nodiscard]] static bool tamper_host_test_fixture_device_identity(
      Sm87TargetAotCompleteProjectionDeviceAssets& owner) noexcept;
  [[nodiscard]] static bool clear_host_test_fixture(
      Sm87TargetAotCompleteProjectionDeviceAssets& owner) noexcept;

 private:
  inline static constexpr std::uint64_t kHostTestResidentIssuerNonce =
      0x5133'5848'4f53'5452ULL;
  friend class Sm87TargetAotCompleteProjectionExecutionAsset;
  friend class sm87_macrofeed_v4_p40_startup_package_detail::
      Sm87MacroFeedV4P40StartupPackage;
  friend class sm87_macrofeed_v4_p40_startup_package_detail::
      Sm87MacroFeedV4P40StartupPackageHostTestFixture;

  Sm87TargetAotCompleteProjectionExecutionAccess(
      const ModelWeights* model_weights,
      const Sm87TargetAotCompleteProjectionDeviceAssets* owner,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uint64_t device_identity,
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
        device_identity_(device_identity),
        arena_begin_(arena_begin),
        arena_bytes_(arena_bytes),
        device_ordinal_(device_ordinal),
        descriptors_(descriptors) {}

  [[nodiscard]] static bool attachment_matches(
      const ModelWeights* model_weights,
      const Sm87TargetAotCompleteProjectionDeviceAssets* owner,
      std::uint64_t owner_identity, std::uint64_t allocation_identity,
      std::uint64_t device_identity,
      std::uintptr_t arena_begin, std::uint64_t arena_bytes,
      std::int32_t device_ordinal) noexcept;
  // Strict canonical ResidentWeights authentication for the Startup boundary
  // source catalog only.  The shared bind()/attached()/catalog_identity()
  // contract remains the pure 256-artifact projection domain.
  [[nodiscard]] static std::optional<
      Sm87TargetAotCompleteProjectionExecutionAccess>
  bind_request_boundary_startup(const ModelWeights& model_weights) noexcept;
  [[nodiscard]] static bool descriptor_matches(
      const Sm87TargetAotCompleteProjectionDeviceAssets& owner,
      const Sm87TargetAotCompleteDeviceAssetDescriptor& descriptor,
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role,
      std::uint64_t expected_offset) noexcept;
  [[nodiscard]] bool resident_root_matches() const noexcept;

  const ModelWeights* model_weights_ = nullptr;
  const Sm87TargetAotCompleteProjectionDeviceAssets* owner_ = nullptr;
  std::uint64_t owner_identity_ = 0U;
  std::uint64_t allocation_identity_ = 0U;
  std::uint64_t device_identity_ = 0U;
  std::uintptr_t arena_begin_ = 0U;
  std::uint64_t arena_bytes_ = 0U;
  std::int32_t device_ordinal_ = -1;
  const ResidentWeights* resident_root_ = nullptr;
  std::uint64_t resident_root_identity_ = 0U;
  std::uintptr_t resident_arena_begin_ = 0U;
  std::uint64_t resident_arena_bytes_ = 0U;
  bool host_test_resident_authority_ = false;
  std::uint64_t host_test_issuer_nonce_ = 0U;
  std::array<const Sm87TargetAotCompleteDeviceAssetDescriptor*,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      descriptors_{};
};

}  // namespace q3x::runtime::target_aot_complete_execution_detail
