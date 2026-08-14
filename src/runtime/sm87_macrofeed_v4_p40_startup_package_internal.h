#pragma once

#include "q3x/kernels/sm87_macrofeed_v4_nvfp4_down.h"
#include "q3x/kernels/sm87_macrofeed_v4_nvfp4_gate_up.h"
#include "q3x/runtime/sm87_macrofeed_v4_panel_wavefront_plan.h"
#include "sm87_target_aot_projection_complete_execution_access_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail {

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
inline constexpr bool kSm87MacroFeedV4P40StartupPackageCompiled = true;
#else
inline constexpr bool kSm87MacroFeedV4P40StartupPackageCompiled = false;
#endif

inline constexpr std::array<std::uint8_t, 8U>
    kSm87MacroFeedV4P40StartupPackageMagic{{'Q', '3', 'X', 'M', '4', 'S',
                                             'P', '1'}};
inline constexpr std::uint16_t kSm87MacroFeedV4P40StartupPackageAbiMajor =
    1U;
inline constexpr std::uint16_t kSm87MacroFeedV4P40StartupPackageAbiMinor =
    0U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageLayers = 64U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageArtifacts =
    256U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageSources = 400U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageGdnLayers =
    48U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageFullLayers =
    16U;
inline constexpr std::size_t kSm87MacroFeedV4MaximumTensorScales =
    kernels::kSm87TargetAotFp8CudaMaximumTensorScales;

enum class Sm87MacroFeedV4P40StartupPackageError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kCanonicalPlan,
  kProjectionAccessBind,
  kProjectionAttachment,
  kProjectionCatalog,
  kProjectionAssetResolve,
  kProjectionAssetBorrow,
  kProjectionInventory,
  kGateUpStartupSeal,
  kDownStartupSeal,
  kDeviceMismatch,
  kPackageIdentity,
  kBindingConstruction,
  kAllocationFailure,
};

struct Sm87MacroFeedV4P40StartupPackageStatus final {
  Sm87MacroFeedV4P40StartupPackageError error =
      Sm87MacroFeedV4P40StartupPackageError::kNone;
  const char* context = "none";
  std::size_t layer = kSm87MacroFeedV4P40StartupPackageLayers;
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  int cuda_error = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == Sm87MacroFeedV4P40StartupPackageError::kNone;
  }
};

class Sm87MacroFeedV4P40StartupPackage;

// These seals contain startup facts only.  They expose no kernel symbol,
// stream, device pointer, launch receipt, or enqueue function.  The package
// never accepts a caller-created instance.
class Sm87MacroFeedV4GateUpStartupSeal final {
 public:
  Sm87MacroFeedV4GateUpStartupSeal(
      const Sm87MacroFeedV4GateUpStartupSeal&) = delete;
  Sm87MacroFeedV4GateUpStartupSeal& operator=(
      const Sm87MacroFeedV4GateUpStartupSeal&) = delete;
  ~Sm87MacroFeedV4GateUpStartupSeal() noexcept { issuer_nonce_ = 0U; }

  std::uint64_t seal_identity = 0U;
  std::uint64_t package_identity = 0U;
  std::uint64_t deployment_plan_identity = 0U;
  kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources resources{};
  bool canonical_c8000_plan = false;
  bool issued_by_v4_package = false;
  bool caller_receipt_accepted = true;
  bool launcher_authority = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool valid() const noexcept;

 private:
  Sm87MacroFeedV4GateUpStartupSeal() = default;
  Sm87MacroFeedV4GateUpStartupSeal(
      Sm87MacroFeedV4GateUpStartupSeal&&) noexcept = default;
  Sm87MacroFeedV4GateUpStartupSeal& operator=(
      Sm87MacroFeedV4GateUpStartupSeal&&) noexcept = default;
  std::uint64_t issuer_nonce_ = 0U;

  friend class Sm87MacroFeedV4P40StartupPackage;
};

class Sm87MacroFeedV4DownStartupSeal final {
 public:
  Sm87MacroFeedV4DownStartupSeal(
      const Sm87MacroFeedV4DownStartupSeal&) = delete;
  Sm87MacroFeedV4DownStartupSeal& operator=(
      const Sm87MacroFeedV4DownStartupSeal&) = delete;
  ~Sm87MacroFeedV4DownStartupSeal() noexcept { issuer_nonce_ = 0U; }

  std::uint64_t seal_identity = 0U;
  std::uint64_t package_identity = 0U;
  std::uint64_t deployment_plan_identity = 0U;
  kernels::Sm87MacroFeedV4NvFp4DownCudaResources resources{};
  bool canonical_c8000_plan = false;
  bool issued_by_v4_package = false;
  bool caller_receipt_accepted = true;
  bool launcher_authority = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool valid() const noexcept;

 private:
  Sm87MacroFeedV4DownStartupSeal() = default;
  Sm87MacroFeedV4DownStartupSeal(
      Sm87MacroFeedV4DownStartupSeal&&) noexcept = default;
  Sm87MacroFeedV4DownStartupSeal& operator=(
      Sm87MacroFeedV4DownStartupSeal&&) noexcept = default;
  std::uint64_t issuer_nonce_ = 0U;

  friend class Sm87MacroFeedV4P40StartupPackage;
};

// Capability-free diagnostic record.  It intentionally makes every missing
// V4 dependency observable; copying it cannot grant an asset or launch.
struct Sm87MacroFeedV4P40StartupPackageAudit final {
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::string_view candidate_id{};
  std::string_view deployment_plan_id{};
  std::uint64_t deployment_plan_identity = 0U;
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
  bool canonical_plan_generated_internally = false;
  bool caller_plan_accepted = true;
  bool complete_projection_access_retained = false;
  bool catalog_revalidated = false;
  bool typed_capabilities_retained = false;
  bool authenticated_source_manifests_retained = false;
  bool authenticated_upload_readback_retained = false;
  bool projection_bindings_complete = false;
  bool nvfp4_startup_seals_complete = false;
  bool caller_raw_receipts_accepted = true;
  bool v3_execution_identity_reused = true;
  bool request_time_repack_jit_autotune_or_fallback_permitted = true;
  bool fp8_executor_bound = true;
  bool gdn_executor_bound = true;
  bool attention_executor_bound = true;
  bool request_state_bound = true;
  bool finalizer_bound = true;
  bool physical_receipt_bound = true;
  bool host_only = false;
  bool default_off = false;
  bool test_only = false;
  bool selector_bound = true;
  bool launcher_present = true;
  bool execution_ready = true;
  bool numerical_qualification_complete = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool valid() const noexcept {
    return magic == kSm87MacroFeedV4P40StartupPackageMagic &&
           abi_major == kSm87MacroFeedV4P40StartupPackageAbiMajor &&
           abi_minor == kSm87MacroFeedV4P40StartupPackageAbiMinor &&
           candidate_id == kSm87MacroFeedV4CandidateId &&
           deployment_plan_id == kSm87MacroFeedV4P40DeploymentPlanId &&
           deployment_plan_identity != 0U && package_identity != 0U &&
           owner_identity != 0U && allocation_identity != 0U &&
           catalog_identity != 0U && device_identity != 0U &&
           device_ordinal >= 0 &&
           layers == kSm87MacroFeedV4P40StartupPackageLayers &&
           artifacts == kSm87MacroFeedV4P40StartupPackageArtifacts &&
           sources == kSm87MacroFeedV4P40StartupPackageSources &&
           gate_up_assets == kSm87MacroFeedV4P40StartupPackageLayers &&
           down_assets == kSm87MacroFeedV4P40StartupPackageLayers &&
           gdn_projection_assets ==
               kSm87MacroFeedV4P40StartupPackageGdnLayers &&
           full_projection_assets ==
               kSm87MacroFeedV4P40StartupPackageFullLayers &&
           attention_output_assets ==
               kSm87MacroFeedV4P40StartupPackageLayers &&
           canonical_plan_generated_internally && !caller_plan_accepted &&
           complete_projection_access_retained && catalog_revalidated &&
           typed_capabilities_retained &&
           authenticated_source_manifests_retained &&
           authenticated_upload_readback_retained &&
           projection_bindings_complete && nvfp4_startup_seals_complete &&
           !caller_raw_receipts_accepted && !v3_execution_identity_reused &&
           !request_time_repack_jit_autotune_or_fallback_permitted &&
           !fp8_executor_bound && !gdn_executor_bound &&
           !attention_executor_bound && !request_state_bound &&
           !finalizer_bound && !physical_receipt_bound && host_only &&
           default_off && test_only && !selector_bound &&
           !launcher_present && !execution_ready &&
           !numerical_qualification_complete &&
           !production_dispatch_eligible;
  }
};

struct Sm87MacroFeedV4P40StartupPackageCreateResult final {
  std::unique_ptr<Sm87MacroFeedV4P40StartupPackage> package;
  Sm87MacroFeedV4P40StartupPackageAudit audit{};
  Sm87MacroFeedV4P40StartupPackageStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept;
};

// Immutable owner-backed startup binding.  It authenticates the exact
// payload interval and scale bits but deliberately exposes neither a raw V3
// receipt nor a typed CUDA asset.  A later executable package must introduce
// a new V4 launcher contract instead of treating this T0 binding as enqueue
// authority.
class Sm87MacroFeedV4ProjectionStartupBinding final {
 public:
  using ProjectionAccess =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAccess;
  using ProjectionAsset =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAsset;

  Sm87MacroFeedV4ProjectionStartupBinding(
      const Sm87MacroFeedV4ProjectionStartupBinding&) = default;
  Sm87MacroFeedV4ProjectionStartupBinding(
      Sm87MacroFeedV4ProjectionStartupBinding&&) noexcept = default;
  Sm87MacroFeedV4ProjectionStartupBinding& operator=(
      const Sm87MacroFeedV4ProjectionStartupBinding&) = delete;
  Sm87MacroFeedV4ProjectionStartupBinding& operator=(
      Sm87MacroFeedV4ProjectionStartupBinding&&) = delete;
  ~Sm87MacroFeedV4ProjectionStartupBinding() = default;

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
  [[nodiscard]] std::uint64_t deployment_plan_identity() const noexcept {
    return snapshot_.deployment_plan_identity;
  }
  [[nodiscard]] std::uint64_t consumer_tactic_identity() const noexcept {
    return snapshot_.consumer_tactic_identity;
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
    return snapshot_.source_inventory_identity;
  }
  [[nodiscard]] std::uint32_t source_count() const noexcept {
    return snapshot_.source_count;
  }
  [[nodiscard]] std::uint32_t tensor_scale_bits(
      std::size_t source_index) const noexcept;
  [[nodiscard]] std::uintptr_t payload_begin() const noexcept {
    return snapshot_.payload_begin;
  }
  [[nodiscard]] std::uintptr_t payload_end() const noexcept {
    return snapshot_.payload_end;
  }
  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return snapshot_.payload_bytes;
  }
  [[nodiscard]] constexpr bool launcher_authority() const noexcept {
    return false;
  }

 private:
  friend class Sm87MacroFeedV4P40StartupPackage;

  struct Snapshot final {
    std::uint64_t binding_identity = 0U;
    std::uint64_t package_identity = 0U;
    std::uint64_t deployment_plan_identity = 0U;
    std::uint64_t owner_identity = 0U;
    std::uint64_t allocation_identity = 0U;
    std::uint64_t catalog_identity = 0U;
    std::uint64_t device_identity = 0U;
    std::uint64_t consumer_tactic_identity = 0U;
    std::uint64_t artifact_identity = 0U;
    std::uint64_t source_inventory_identity = 0U;
    std::uint64_t manifest_seal = 0U;
    std::uint64_t upload_receipt_identity = 0U;
    kernels::Sm87TargetAotProjectionSha256Digest payload_digest{};
    std::uintptr_t payload_begin = 0U;
    std::uintptr_t payload_end = 0U;
    std::uint64_t payload_bytes = 0U;
    std::int32_t device_ordinal = -1;
    std::size_t layer_index = kSm87MacroFeedV4P40StartupPackageLayers;
    kernels::Sm87TargetAotProjectionRole role =
        kernels::Sm87TargetAotProjectionRole::kInvalid;
    kernels::Sm87TargetAotProjectionEncoding encoding =
        kernels::Sm87TargetAotProjectionEncoding::kInvalid;
    std::array<std::uint32_t, kSm87MacroFeedV4MaximumTensorScales>
        tensor_scale_bits{};
    std::uint32_t source_count = 0U;
    bool issued_from_live_complete_asset = false;
    bool canonical_payload_layout_retained = false;
    bool caller_raw_receipt_accepted = true;
    bool v3_execution_identity_reused = true;
    bool t0_only = false;
    bool launcher_authority = true;
    bool production_dispatch_eligible = true;
  };

  Sm87MacroFeedV4ProjectionStartupBinding(
      ProjectionAccess access, ProjectionAsset asset,
      Snapshot snapshot) noexcept;
  [[nodiscard]] static std::uint64_t compute_binding_identity(
      const Snapshot& snapshot) noexcept;
  [[nodiscard]] bool valid_with_catalog(
      std::uint64_t catalog_identity) const noexcept;
  [[nodiscard]] bool valid_with_prevalidated_catalog(
      std::uint64_t catalog_identity) const noexcept;
  [[nodiscard]] bool valid_for_prevalidated_catalog(
      std::size_t layer_index, kernels::Sm87TargetAotProjectionRole role,
      std::uint64_t package_identity,
      std::uint64_t catalog_identity) const noexcept;

  ProjectionAccess projection_access_;
  ProjectionAsset asset_;
  Snapshot snapshot_{};
};

class Sm87MacroFeedV4P40StartupPackage final {
 public:
  using ProjectionAccess =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAccess;
  using ProjectionAsset =
      target_aot_complete_execution_detail::
          Sm87TargetAotCompleteProjectionExecutionAsset;
  using ProjectionStartupBindingCatalog =
      std::array<const Sm87MacroFeedV4ProjectionStartupBinding*,
                 kSm87MacroFeedV4P40StartupPackageArtifacts>;

  Sm87MacroFeedV4P40StartupPackage(
      const Sm87MacroFeedV4P40StartupPackage&) = delete;
  Sm87MacroFeedV4P40StartupPackage& operator=(
      const Sm87MacroFeedV4P40StartupPackage&) = delete;
  Sm87MacroFeedV4P40StartupPackage(
      Sm87MacroFeedV4P40StartupPackage&&) = delete;
  Sm87MacroFeedV4P40StartupPackage& operator=(
      Sm87MacroFeedV4P40StartupPackage&&) = delete;
  ~Sm87MacroFeedV4P40StartupPackage() = default;

  [[nodiscard]] static Sm87MacroFeedV4P40StartupPackageCreateResult create(
      const ModelWeights& model_weights) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const Sm87MacroFeedV4P40StartupPackageAudit& audit()
      const noexcept {
    return audit_;
  }
  [[nodiscard]] const Sm87MacroFeedV4PanelWavefrontPlan& plan()
      const noexcept {
    return plan_;
  }
  [[nodiscard]] const Sm87MacroFeedV4GateUpStartupSeal&
  gate_up_startup_seal() const noexcept {
    return seals_.gate_up;
  }
  [[nodiscard]] const Sm87MacroFeedV4DownStartupSeal&
  down_startup_seal() const noexcept {
    return seals_.down;
  }
  [[nodiscard]] const Sm87MacroFeedV4ProjectionStartupBinding*
  borrow_projection_startup_binding(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;
  [[nodiscard]] bool borrow_projection_startup_catalog(
      ProjectionStartupBindingCatalog* catalog) const noexcept;

 private:
  struct AssetCapability final {
    std::optional<ProjectionAsset> asset;
    std::size_t layer_index = kSm87MacroFeedV4P40StartupPackageLayers;
    kernels::Sm87TargetAotProjectionRole role =
        kernels::Sm87TargetAotProjectionRole::kInvalid;
    kernels::Sm87TargetAotProjectionEncoding encoding =
        kernels::Sm87TargetAotProjectionEncoding::kInvalid;
    std::uint64_t artifact_identity = 0U;
    std::uint64_t source_inventory_identity = 0U;
    std::uint64_t manifest_seal = 0U;
    std::uint64_t upload_receipt_identity = 0U;
    kernels::Sm87TargetAotProjectionSha256Digest payload_digest{};
    std::uintptr_t payload_begin = 0U;
    std::uintptr_t payload_end = 0U;
    std::uint64_t payload_bytes = 0U;
    std::array<std::uint32_t, kSm87MacroFeedV4MaximumTensorScales>
        tensor_scale_bits{};
    std::uint32_t source_count = 0U;
  };

  struct StartupSeals final {
    Sm87MacroFeedV4GateUpStartupSeal gate_up{};
    Sm87MacroFeedV4DownStartupSeal down{};

    StartupSeals() = default;
    StartupSeals(const StartupSeals&) = delete;
    StartupSeals& operator=(const StartupSeals&) = delete;
    StartupSeals(StartupSeals&& other) noexcept
        : gate_up(std::move(other.gate_up)), down(std::move(other.down)) {}
    StartupSeals& operator=(StartupSeals&&) = delete;
  };

  Sm87MacroFeedV4P40StartupPackage(
      ProjectionAccess access,
      std::array<AssetCapability,
                 kSm87MacroFeedV4P40StartupPackageArtifacts> capabilities,
      Sm87MacroFeedV4PanelWavefrontPlan plan, StartupSeals seals,
      Sm87MacroFeedV4P40StartupPackageAudit audit) noexcept;

  [[nodiscard]] static std::uint64_t compute_deployment_plan_identity(
      const Sm87MacroFeedV4PanelWavefrontPlan& plan) noexcept;
  [[nodiscard]] static std::uint64_t compute_package_identity(
      const ProjectionAccess& access,
      const std::array<AssetCapability,
                       kSm87MacroFeedV4P40StartupPackageArtifacts>&
          capabilities,
      const Sm87MacroFeedV4PanelWavefrontPlan& plan,
      const kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources& gate_up,
      const kernels::Sm87MacroFeedV4NvFp4DownCudaResources& down,
      std::size_t sources) noexcept;
  [[nodiscard]] static StartupSeals mint_startup_seals(
      std::uint64_t package_identity, std::uint64_t plan_identity,
      const kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources& gate_up,
      const kernels::Sm87MacroFeedV4NvFp4DownCudaResources& down) noexcept;
  [[nodiscard]] static bool startup_seals_valid(
      const StartupSeals& seals, std::uint64_t package_identity,
      std::uint64_t plan_identity, std::int32_t device_ordinal) noexcept;
  [[nodiscard]] static Sm87MacroFeedV4P40StartupPackageCreateResult
  build_from_private_authority(
      ProjectionAccess access, Sm87MacroFeedV4PanelWavefrontPlan plan,
      kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up,
      kernels::Sm87MacroFeedV4NvFp4DownCudaResources down) noexcept;
  [[nodiscard]] const AssetCapability* capability(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;
  [[nodiscard]] bool base_valid() const noexcept;
  [[nodiscard]] bool populate_projection_bindings() noexcept;
  [[nodiscard]] std::optional<Sm87MacroFeedV4ProjectionStartupBinding>
  make_projection_binding(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;

  ProjectionAccess projection_access_;
  std::array<AssetCapability, kSm87MacroFeedV4P40StartupPackageArtifacts>
      capabilities_{};
  Sm87MacroFeedV4PanelWavefrontPlan plan_{};
  StartupSeals seals_{};
  Sm87MacroFeedV4P40StartupPackageAudit audit_{};
  std::array<std::optional<Sm87MacroFeedV4ProjectionStartupBinding>,
             kSm87MacroFeedV4P40StartupPackageArtifacts>
      projection_bindings_{};
};

static_assert(kSm87MacroFeedV4P40StartupPackageLayers ==
              kSm87TargetAotCompleteProjectionDeviceLayerCount);
static_assert(kSm87MacroFeedV4P40StartupPackageArtifacts ==
              kSm87TargetAotCompleteProjectionDeviceArtifactCount);
static_assert(kSm87MacroFeedV4P40StartupPackageSources ==
              kSm87TargetAotCompleteProjectionDeviceSourceCount);

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail
