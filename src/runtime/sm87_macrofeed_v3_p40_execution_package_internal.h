#pragma once

#include "q3x/kernels/sm87_macrofeed_v3_fp8.h"
#include "q3x/kernels/sm87_macrofeed_v3_gdn_p40.h"
#include "q3x/kernels/sm87_macrofeed_v3_nvfp4_down.h"
#include "q3x/kernels/sm87_macrofeed_v3_nvfp4_gate_up.h"
#include "sm87_target_aot_projection_complete_execution_access_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace q3x::runtime::sm87_macrofeed_v3_p40_execution_package_detail {

#if defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
inline constexpr bool kSm87MacroFeedV3P40ExecutionPackageCompiled = true;
#else
inline constexpr bool kSm87MacroFeedV3P40ExecutionPackageCompiled = false;
#endif

inline constexpr std::size_t kSm87MacroFeedV3P40PackageLayers = 64U;
inline constexpr std::size_t kSm87MacroFeedV3P40PackageArtifacts = 256U;
inline constexpr std::size_t kSm87MacroFeedV3P40PackageSources = 400U;
inline constexpr std::size_t kSm87MacroFeedV3P40PackageGdnLayers = 48U;
inline constexpr std::size_t kSm87MacroFeedV3P40PackageFullLayers = 16U;

enum class Sm87MacroFeedV3P40ExecutionPackageError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kProjectionAccessBind,
  kProjectionAttachment,
  kProjectionCatalog,
  kProjectionAssetResolve,
  kProjectionAssetBorrow,
  kProjectionInventory,
  kGateUpStartupSeal,
  kDownStartupSeal,
  kGdnStartupSeal,
  kFp8GdnStartupSeal,
  kFp8FullStartupSeal,
  kFp8OutputStartupSeal,
  kDeviceMismatch,
  kPackageIdentity,
  kAllocationFailure,
};

struct Sm87MacroFeedV3P40ExecutionPackageStatus final {
  Sm87MacroFeedV3P40ExecutionPackageError error =
      Sm87MacroFeedV3P40ExecutionPackageError::kNone;
  const char* context = "none";
  std::size_t layer = kSm87MacroFeedV3P40PackageLayers;
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  int cuda_error = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == Sm87MacroFeedV3P40ExecutionPackageError::kNone;
  }
};

// Capability-free diagnostic record.  Copying it grants no asset or enqueue
// authority; all executable borrows remain private complete-owner
// capabilities retained by the package.
struct Sm87MacroFeedV3P40ExecutionPackageAudit final {
  std::uint64_t package_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t catalog_identity = 0U;
  std::uint64_t device_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::size_t layers = 0U;
  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  std::size_t gate_up_assets = 0U;
  std::size_t down_assets = 0U;
  std::size_t gdn_projection_assets = 0U;
  std::size_t full_projection_assets = 0U;
  std::size_t attention_output_assets = 0U;
  bool complete_projection_access_retained = false;
  bool catalog_revalidated = false;
  bool typed_capabilities_retained = false;
  bool authenticated_source_manifests_retained = false;
  bool startup_seals_complete = false;
  bool caller_raw_receipts_accepted = true;
  bool v2_owner_or_executor_reused = true;
  bool request_time_repack_jit_or_fallback_permitted = true;
  bool t0_t1_only = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return package_identity != 0U && owner_identity != 0U &&
           allocation_identity != 0U && catalog_identity != 0U &&
           device_identity != 0U && device_ordinal >= 0 &&
           layers == kSm87MacroFeedV3P40PackageLayers &&
           artifacts == kSm87MacroFeedV3P40PackageArtifacts &&
           sources == kSm87MacroFeedV3P40PackageSources &&
           gate_up_assets == kSm87MacroFeedV3P40PackageLayers &&
           down_assets == kSm87MacroFeedV3P40PackageLayers &&
           gdn_projection_assets == kSm87MacroFeedV3P40PackageGdnLayers &&
           full_projection_assets == kSm87MacroFeedV3P40PackageFullLayers &&
           attention_output_assets == kSm87MacroFeedV3P40PackageLayers &&
           complete_projection_access_retained && catalog_revalidated &&
           typed_capabilities_retained &&
           authenticated_source_manifests_retained &&
           startup_seals_complete &&
           !caller_raw_receipts_accepted && !v2_owner_or_executor_reused &&
           !request_time_repack_jit_or_fallback_permitted && t0_t1_only &&
           !production_dispatch_eligible;
  }
};

class Sm87MacroFeedV3P40ExecutionPackage;

struct Sm87MacroFeedV3P40ExecutionPackageCreateResult final {
  std::unique_ptr<Sm87MacroFeedV3P40ExecutionPackage> package;
  Sm87MacroFeedV3P40ExecutionPackageAudit audit{};
  Sm87MacroFeedV3P40ExecutionPackageStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept;
};

// Immutable, copy-constructible authority record for one exact layer/role
// projection.  Only the complete owner-backed package can construct it.
// Callers can retain and inspect the issued binding but cannot provide source
// identities, scales, payload receipts, or typed assets to its constructor.
// Every live validation returns through the private complete-owner capability.
class Sm87MacroFeedV3P40ProjectionStartupBinding final {
 public:
  using ProjectionAccess =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAccess;
  using ProjectionAsset =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAsset;
  using SourceInventory =
      kernels::Sm87TargetAotProjectionPackedSourceInventory;
  using SourceBinding = kernels::Sm87TargetAotProjectionPackedSourceBinding;

  Sm87MacroFeedV3P40ProjectionStartupBinding(
      const Sm87MacroFeedV3P40ProjectionStartupBinding&) = default;
  Sm87MacroFeedV3P40ProjectionStartupBinding(
      Sm87MacroFeedV3P40ProjectionStartupBinding&&) noexcept = default;
  Sm87MacroFeedV3P40ProjectionStartupBinding& operator=(
      const Sm87MacroFeedV3P40ProjectionStartupBinding&) = delete;
  Sm87MacroFeedV3P40ProjectionStartupBinding& operator=(
      Sm87MacroFeedV3P40ProjectionStartupBinding&&) = delete;
  ~Sm87MacroFeedV3P40ProjectionStartupBinding() = default;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool valid_for(
      std::size_t layer_index, kernels::Sm87TargetAotProjectionRole role,
      std::uint64_t package_identity) const noexcept;

  [[nodiscard]] std::uint64_t binding_identity() const noexcept {
    return snapshot_.binding_identity;
  }
  [[nodiscard]] std::uint64_t package_identity() const noexcept {
    return snapshot_.package_identity;
  }
  [[nodiscard]] std::uint64_t device_identity() const noexcept {
    return snapshot_.device_identity;
  }
  [[nodiscard]] std::size_t layer_index() const noexcept {
    return snapshot_.layer_index;
  }
  [[nodiscard]] kernels::Sm87TargetAotProjectionRole role() const noexcept {
    return snapshot_.role;
  }
  [[nodiscard]] std::uint64_t artifact_identity() const noexcept {
    return snapshot_.artifact_identity;
  }
  [[nodiscard]] std::uint64_t source_inventory_identity() const noexcept {
    return snapshot_.source_inventory.identity;
  }
  [[nodiscard]] std::uint32_t source_count() const noexcept {
    return snapshot_.source_inventory.source_count;
  }
  [[nodiscard]] const SourceBinding* source(
      std::size_t source_index) const noexcept;
  [[nodiscard]] float tensor_scale(std::size_t source_index) const noexcept;
  [[nodiscard]] std::uintptr_t payload_begin() const noexcept {
    return snapshot_.payload_begin;
  }
  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return snapshot_.payload_bytes;
  }

  [[nodiscard]] const kernels::Sm87TargetAotNvFp4CudaAssetView*
  borrow_nvfp4_asset() const noexcept;
  [[nodiscard]] const kernels::Sm87TargetAotFp8CudaAssetView*
  borrow_fp8_asset() const noexcept;
  [[nodiscard]] const kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt*
  gate_up_payload_receipt() const noexcept;
  [[nodiscard]] const kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt*
  down_payload_receipt() const noexcept;

 private:
  friend class Sm87MacroFeedV3P40ExecutionPackage;

  struct Snapshot final {
    std::uint64_t binding_identity = 0U;
    std::uint64_t package_identity = 0U;
    std::uint64_t owner_identity = 0U;
    std::uint64_t allocation_identity = 0U;
    std::uint64_t catalog_identity = 0U;
    std::uint64_t device_identity = 0U;
    std::uint64_t artifact_identity = 0U;
    std::uint64_t manifest_seal = 0U;
    std::uint64_t upload_receipt_identity = 0U;
    std::uintptr_t payload_begin = 0U;
    std::uintptr_t payload_end = 0U;
    std::uint64_t payload_bytes = 0U;
    std::int32_t device_ordinal = -1;
    std::size_t layer_index = kSm87MacroFeedV3P40PackageLayers;
    kernels::Sm87TargetAotProjectionRole role =
        kernels::Sm87TargetAotProjectionRole::kInvalid;
    kernels::Sm87TargetAotProjectionEncoding encoding =
        kernels::Sm87TargetAotProjectionEncoding::kInvalid;
    SourceInventory source_inventory{};
    kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt gate_up_receipt{};
    kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt down_receipt{};
    bool issued_from_live_private_descriptor = false;
    bool caller_receipt_accepted = true;
    bool t0_t1_only = false;
    bool production_dispatch_eligible = true;
  };

  Sm87MacroFeedV3P40ProjectionStartupBinding(
      ProjectionAccess access, ProjectionAsset asset,
      Snapshot snapshot) noexcept;
  [[nodiscard]] static std::uint64_t compute_binding_identity(
      const Snapshot& snapshot) noexcept;
  [[nodiscard]] bool valid_with_authenticated_catalog(
      std::uint64_t catalog_identity) const noexcept;

  ProjectionAccess projection_access_;
  ProjectionAsset asset_;
  Snapshot snapshot_{};
};

// Startup-only composition root for MacroFeed-v3.  It accepts only
// ModelWeights carrying the private complete-owner attachment.  It never
// accepts a caller receipt, view, pointer, identity, resource record, or V2
// owner/executor.  ModelWeights and its complete owner must outlive this
// package and every capability borrowed from it.
class Sm87MacroFeedV3P40ExecutionPackage final {
 public:
  using ProjectionAccess =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAccess;
  using ProjectionAsset =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAsset;

  Sm87MacroFeedV3P40ExecutionPackage(
      const Sm87MacroFeedV3P40ExecutionPackage&) = delete;
  Sm87MacroFeedV3P40ExecutionPackage& operator=(
      const Sm87MacroFeedV3P40ExecutionPackage&) = delete;
  Sm87MacroFeedV3P40ExecutionPackage(
      Sm87MacroFeedV3P40ExecutionPackage&&) = delete;
  Sm87MacroFeedV3P40ExecutionPackage& operator=(
      Sm87MacroFeedV3P40ExecutionPackage&&) = delete;
  ~Sm87MacroFeedV3P40ExecutionPackage() = default;

  [[nodiscard]] static Sm87MacroFeedV3P40ExecutionPackageCreateResult create(
      const ModelWeights& model_weights) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const Sm87MacroFeedV3P40ExecutionPackageAudit& audit()
      const noexcept {
    return audit_;
  }

  // Typed live borrows revalidate the private complete-owner attachment and
  // exact descriptor before returning.  No raw upload receipt is exposed.
  [[nodiscard]] const kernels::Sm87TargetAotNvFp4CudaAssetView*
  borrow_nvfp4_asset(std::size_t layer_index,
                     kernels::Sm87TargetAotProjectionRole role)
      const noexcept;
  [[nodiscard]] const kernels::Sm87TargetAotFp8CudaAssetView*
  borrow_fp8_asset(std::size_t layer_index,
                   kernels::Sm87TargetAotProjectionRole role) const noexcept;
  // All 256 immutable bindings are constructed once by create().  Runtime
  // consumers may only borrow the exact retained layer/role entry; they have
  // no request-time issuance or caller-populated selection authority.  This
  // pointer borrow revalidates the owner and exact descriptor; executable
  // asset/receipt borrows on the returned binding additionally revalidate the
  // complete live catalog before exposing authority.
  [[nodiscard]] const Sm87MacroFeedV3P40ProjectionStartupBinding*
  borrow_projection_startup_binding(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;

  [[nodiscard]] const kernels::Sm87MacroFeedV3NvFp4GateUpStartupSeal&
  gate_up_startup_seal() const noexcept {
    return seals_.gate_up;
  }
  [[nodiscard]] const kernels::Sm87MacroFeedV3NvFp4DownCudaResources&
  down_startup_seal() const noexcept {
    return seals_.down;
  }
  [[nodiscard]] const kernels::Sm87MacrofeedV3GdnResources&
  gdn_startup_seal() const noexcept {
    return seals_.gdn;
  }
  [[nodiscard]] const kernels::Sm87MacroFeedV3Fp8StartupSeal*
  fp8_startup_seal(kernels::Sm87TargetAotProjectionRole role)
      const noexcept;

 private:
  struct StartupSeals final {
    kernels::Sm87MacroFeedV3NvFp4GateUpStartupSeal gate_up{};
    kernels::Sm87MacroFeedV3NvFp4DownCudaResources down{};
    kernels::Sm87MacrofeedV3GdnResources gdn{};
    std::array<kernels::Sm87MacroFeedV3Fp8StartupSeal, 3U> fp8{};
  };

  struct AssetCapability final {
    std::optional<ProjectionAsset> asset;
    std::size_t layer_index = kSm87MacroFeedV3P40PackageLayers;
    kernels::Sm87TargetAotProjectionRole role =
        kernels::Sm87TargetAotProjectionRole::kInvalid;
    kernels::Sm87TargetAotProjectionEncoding encoding =
        kernels::Sm87TargetAotProjectionEncoding::kInvalid;
    std::uint64_t artifact_identity = 0U;
    std::uint64_t source_inventory_identity = 0U;
    // The authenticated manifest seal transitively folds the exact source
    // count and every tensor identity, logical role/partition, weight/scale
    // digest, shape, scale, and payload interval.  Retaining it alongside the
    // live typed capability makes all 400 actual sources part of the package
    // identity without exposing caller-populated source receipts.
    std::uint64_t source_manifest_seal = 0U;
    std::uint64_t payload_bytes = 0U;
    std::uint32_t source_count = 0U;
  };

  Sm87MacroFeedV3P40ExecutionPackage(
      ProjectionAccess access,
      std::array<AssetCapability, kSm87MacroFeedV3P40PackageArtifacts>
          capabilities,
      StartupSeals seals,
      Sm87MacroFeedV3P40ExecutionPackageAudit audit) noexcept;

  [[nodiscard]] static Sm87MacroFeedV3P40ExecutionPackageCreateResult
  build_from_private_authority(ProjectionAccess access,
                               StartupSeals seals) noexcept;
  [[nodiscard]] static bool startup_seals_valid(
      const StartupSeals& seals, std::int32_t device_ordinal) noexcept;
  [[nodiscard]] static std::uint64_t compute_package_identity(
      const ProjectionAccess& access,
      const std::array<AssetCapability,
                       kSm87MacroFeedV3P40PackageArtifacts>& capabilities,
      const StartupSeals& seals, std::size_t sources) noexcept;
  [[nodiscard]] bool base_valid() const noexcept;
  [[nodiscard]] bool populate_projection_startup_bindings() noexcept;
  [[nodiscard]] std::optional<Sm87MacroFeedV3P40ProjectionStartupBinding>
  make_projection_startup_binding(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;
  [[nodiscard]] const AssetCapability* capability(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;

  ProjectionAccess projection_access_;
  std::array<AssetCapability, kSm87MacroFeedV3P40PackageArtifacts>
      capabilities_{};
  StartupSeals seals_{};
  Sm87MacroFeedV3P40ExecutionPackageAudit audit_{};
  std::array<std::optional<Sm87MacroFeedV3P40ProjectionStartupBinding>,
             kSm87MacroFeedV3P40PackageArtifacts>
      startup_bindings_{};
};

static_assert(kSm87MacroFeedV3P40PackageLayers ==
              kSm87TargetAotCompleteProjectionDeviceLayerCount);
static_assert(kSm87MacroFeedV3P40PackageArtifacts ==
              kSm87TargetAotCompleteProjectionDeviceArtifactCount);
static_assert(kSm87MacroFeedV3P40PackageSources ==
              kSm87TargetAotCompleteProjectionDeviceSourceCount);

}  // namespace q3x::runtime::sm87_macrofeed_v3_p40_execution_package_detail
