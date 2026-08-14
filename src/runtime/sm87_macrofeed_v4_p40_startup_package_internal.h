#pragma once

#include "q3x/kernels/sm87_macrofeed_v4_bf16_ab.h"
#include "q3x/kernels/sm87_macrofeed_v4_fp8.h"
#include "q3x/kernels/sm87_macrofeed_v4_norm_residual.h"
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

namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail {
class Sm87MacroFeedV4P40ExecutionPackage;
}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail

namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail {

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
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
    2U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageLayers = 64U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageArtifacts =
    256U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageSources = 400U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageGdnLayers =
    48U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageFullLayers =
    16U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageBf16AbPairs =
    48U;
inline constexpr std::size_t kSm87MacroFeedV4P40StartupPackageBf16AbTensors =
    2U * kSm87MacroFeedV4P40StartupPackageBf16AbPairs;
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
  kBf16AbModelInventory,
  kBf16AbResourceSeal,
  kBf16AbDeviceRange,
  kGdnQkvZResourceSeal,
  kGdnQkvZCatalogSeal,
  kDeviceMismatch,
  kPackageIdentity,
  kBindingConstruction,
  kAllocationFailure,
};

enum class Sm87MacroFeedV4Bf16AbWeightRole : std::uint8_t {
  kInvalid = 0U,
  kA,
  kB,
};

// Caller-fillable T0 input.  It lets host tests exercise the natural layer,
// dtype, shape, role and range-order rules without pretending that a fake
// address is a CUDA allocation.  The returned audit below is diagnostic only
// and can never construct the package's private startup capability.
struct Sm87MacroFeedV4Bf16AbT0TensorDescriptor final {
  std::size_t gdn_ordinal =
      kSm87MacroFeedV4P40StartupPackageBf16AbPairs;
  std::size_t model_layer = kSm87MacroFeedV4P40StartupPackageLayers;
  Sm87MacroFeedV4Bf16AbWeightRole role =
      Sm87MacroFeedV4Bf16AbWeightRole::kInvalid;
  LinearWeightKind weight_kind = LinearWeightKind::kFp8;
  const std::uint16_t* weight = nullptr;
  std::size_t output_size = 0U;
  std::size_t input_size = 0U;
};

struct Sm87MacroFeedV4Bf16AbT0InventoryAudit final {
  std::uint64_t catalog_identity = 0U;
  std::size_t tensors = 0U;
  std::size_t pairs = 0U;
  std::size_t failure_index =
      kSm87MacroFeedV4P40StartupPackageBf16AbTensors;
  bool canonical_natural_layer_order = false;
  bool canonical_a_then_b_role_order = false;
  bool exact_bf16_shapes = false;
  bool nonnull_16b_aligned_disjoint_ranges = false;
  bool live_cuda_device_ranges_validated = true;
  bool execution_capability = true;

  [[nodiscard]] constexpr bool valid_t0() const noexcept {
    return catalog_identity != 0U &&
           tensors == kSm87MacroFeedV4P40StartupPackageBf16AbTensors &&
           pairs == kSm87MacroFeedV4P40StartupPackageBf16AbPairs &&
           failure_index ==
               kSm87MacroFeedV4P40StartupPackageBf16AbTensors &&
           canonical_natural_layer_order && canonical_a_then_b_role_order &&
           exact_bf16_shapes && nonnull_16b_aligned_disjoint_ranges &&
           !live_cuda_device_ranges_validated && !execution_capability;
  }
};

[[nodiscard]] Sm87MacroFeedV4Bf16AbT0InventoryAudit
inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
    const Sm87MacroFeedV4Bf16AbT0TensorDescriptor* tensors,
    std::size_t tensor_count) noexcept;

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

class Sm87MacroFeedV4Bf16AbStartupSeal final {
 public:
  Sm87MacroFeedV4Bf16AbStartupSeal(
      const Sm87MacroFeedV4Bf16AbStartupSeal&) = delete;
  Sm87MacroFeedV4Bf16AbStartupSeal& operator=(
      const Sm87MacroFeedV4Bf16AbStartupSeal&) = delete;
  ~Sm87MacroFeedV4Bf16AbStartupSeal() noexcept { issuer_nonce_ = 0U; }

  std::uint64_t seal_identity = 0U;
  std::uint64_t package_identity = 0U;
  std::uint64_t deployment_plan_identity = 0U;
  std::uint64_t binding_catalog_identity = 0U;
  std::size_t tensor_count = 0U;
  std::size_t pair_count = 0U;
  kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot resources{};
  bool canonical_natural_layer_order = false;
  bool canonical_a_then_b_role_order = false;
  bool complete_live_device_ranges = false;
  bool issued_by_v4_package = false;
  bool caller_resource_snapshot_accepted = true;
  bool raw_pointer_exposed = true;
  bool launcher_authority = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool valid() const noexcept;

 private:
  Sm87MacroFeedV4Bf16AbStartupSeal() = default;
  Sm87MacroFeedV4Bf16AbStartupSeal(
      Sm87MacroFeedV4Bf16AbStartupSeal&&) noexcept = default;
  Sm87MacroFeedV4Bf16AbStartupSeal& operator=(
      Sm87MacroFeedV4Bf16AbStartupSeal&&) noexcept = default;
  std::uint64_t issuer_nonce_ = 0U;

  friend class Sm87MacroFeedV4P40StartupPackage;
};

// Startup-only seal for the one fixed GDN-QKVZ FP8 tactic.  It deliberately
// contains no packed-payload address or typed asset.  The 48 asset snapshots
// remain private and become available only through the execution package's
// construction-time, live-allocation-validated catalog seal.
class Sm87MacroFeedV4GdnQkvZStartupSeal final {
 public:
  Sm87MacroFeedV4GdnQkvZStartupSeal(
      const Sm87MacroFeedV4GdnQkvZStartupSeal&) = delete;
  Sm87MacroFeedV4GdnQkvZStartupSeal& operator=(
      const Sm87MacroFeedV4GdnQkvZStartupSeal&) = delete;
  ~Sm87MacroFeedV4GdnQkvZStartupSeal() noexcept { issuer_nonce_ = 0U; }

  std::uint64_t seal_identity = 0U;
  std::uint64_t package_identity = 0U;
  std::uint64_t deployment_plan_identity = 0U;
  std::uint64_t binding_catalog_identity = 0U;
  std::size_t binding_count = 0U;
  kernels::Sm87MacroFeedV4Fp8CudaResources resources{};
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  kernels::Sm87MacroFeedV4Fp8InputLayout input_layout =
      kernels::Sm87MacroFeedV4Fp8InputLayout::kInvalid;
  kernels::Sm87MacroFeedV4Fp8Identity tactic_identity =
      kernels::Sm87MacroFeedV4Fp8Identity::kInvalid;
  bool canonical_natural_gdn_layer_order = false;
  bool role_layout_and_tactic_fixed = false;
  bool typed_asset_values_private = false;
  bool caller_resource_snapshot_accepted = true;
  bool raw_pointer_exposed = true;
  bool launcher_authority = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] bool valid() const noexcept;

 private:
  Sm87MacroFeedV4GdnQkvZStartupSeal() = default;
  Sm87MacroFeedV4GdnQkvZStartupSeal(
      Sm87MacroFeedV4GdnQkvZStartupSeal&&) noexcept = default;
  Sm87MacroFeedV4GdnQkvZStartupSeal& operator=(
      Sm87MacroFeedV4GdnQkvZStartupSeal&&) noexcept = default;
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
  std::size_t bf16_ab_tensors = 0U;
  std::size_t bf16_ab_pairs = 0U;
  std::uint64_t bf16_ab_binding_catalog_identity = 0U;
  std::uint64_t bf16_ab_resource_seal_identity = 0U;
  std::size_t gdn_qkvz_bindings = 0U;
  std::uint64_t gdn_qkvz_binding_catalog_identity = 0U;
  std::uint64_t gdn_qkvz_resource_seal_identity = 0U;
  bool canonical_plan_generated_internally = false;
  bool caller_plan_accepted = true;
  bool complete_projection_access_retained = false;
  bool catalog_revalidated = false;
  bool typed_capabilities_retained = false;
  bool authenticated_source_manifests_retained = false;
  bool authenticated_upload_readback_retained = false;
  bool projection_bindings_complete = false;
  bool nvfp4_startup_seals_complete = false;
  bool bf16_ab_nonowning_model_weights_dependency_bound = false;
  bool bf16_ab_projection_owner_identity_retained = false;
  bool bf16_ab_natural_layer_order_complete = false;
  bool bf16_ab_a_then_b_roles_complete = false;
  bool bf16_ab_live_device_ranges_complete = false;
  bool bf16_ab_resource_seal_complete = false;
  bool bf16_ab_private_capability_retained = false;
  bool bf16_ab_raw_pointer_publicly_exposed = true;
  bool gdn_qkvz_natural_layer_order_complete = false;
  bool gdn_qkvz_role_layout_tactic_fixed = false;
  bool gdn_qkvz_asset_value_snapshots_private = false;
  bool gdn_qkvz_resource_seal_complete = false;
  bool gdn_qkvz_raw_pointer_publicly_exposed = true;
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
           bf16_ab_tensors ==
               kSm87MacroFeedV4P40StartupPackageBf16AbTensors &&
           bf16_ab_pairs ==
               kSm87MacroFeedV4P40StartupPackageBf16AbPairs &&
           bf16_ab_binding_catalog_identity != 0U &&
           bf16_ab_resource_seal_identity != 0U &&
           gdn_qkvz_bindings ==
               kSm87MacroFeedV4P40StartupPackageGdnLayers &&
           gdn_qkvz_binding_catalog_identity != 0U &&
           gdn_qkvz_resource_seal_identity != 0U &&
           canonical_plan_generated_internally && !caller_plan_accepted &&
           complete_projection_access_retained && catalog_revalidated &&
           typed_capabilities_retained &&
           authenticated_source_manifests_retained &&
           authenticated_upload_readback_retained &&
           projection_bindings_complete && nvfp4_startup_seals_complete &&
           bf16_ab_nonowning_model_weights_dependency_bound &&
           bf16_ab_projection_owner_identity_retained &&
           bf16_ab_natural_layer_order_complete &&
           bf16_ab_a_then_b_roles_complete &&
           bf16_ab_live_device_ranges_complete &&
           bf16_ab_resource_seal_complete &&
           bf16_ab_private_capability_retained &&
           !bf16_ab_raw_pointer_publicly_exposed &&
           gdn_qkvz_natural_layer_order_complete &&
           gdn_qkvz_role_layout_tactic_fixed &&
           gdn_qkvz_asset_value_snapshots_private &&
           gdn_qkvz_resource_seal_complete &&
           !gdn_qkvz_raw_pointer_publicly_exposed &&
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

// This package is an immutable, non-owning lifetime dependency.  Engine
// teardown must destroy any future execution package first, this startup
// package second, and the exact ModelWeights/ResidentWeights root last.  A
// execution package copies the immutable projection-binding values and seals
// the complete BF16/norm catalogs during construction; request execution
// reuses those snapshots without CUDA queries or catalog rescans.  Device
// payload/weight ownership must remain under the future Engine lifetime root.
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
  [[nodiscard]] const Sm87MacroFeedV4Bf16AbStartupSeal&
  bf16_ab_startup_seal() const noexcept {
    return seals_.bf16_ab;
  }
  [[nodiscard]] const Sm87MacroFeedV4GdnQkvZStartupSeal&
  gdn_qkvz_startup_seal() const noexcept {
    return seals_.gdn_qkvz;
  }
  [[nodiscard]] const Sm87MacroFeedV4ProjectionStartupBinding*
  borrow_projection_startup_binding(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;
  [[nodiscard]] bool borrow_projection_startup_catalog(
      ProjectionStartupBindingCatalog* catalog) const noexcept;

 private:
  struct Bf16AbPair final {
    std::uint32_t model_layer =
        static_cast<std::uint32_t>(
            kSm87MacroFeedV4P40StartupPackageLayers);
    const std::uint16_t* a_weights = nullptr;
    const std::uint16_t* b_weights = nullptr;
    std::uint64_t pair_identity = 0U;
  };

  using Bf16AbExecutionBindingCatalog =
      std::array<Bf16AbPair,
                 kSm87MacroFeedV4P40StartupPackageBf16AbPairs>;

  struct LayerNormExecutionBinding final {
    std::uint32_t model_layer =
        static_cast<std::uint32_t>(
            kSm87MacroFeedV4P40StartupPackageLayers);
    const std::uint16_t* input_layernorm = nullptr;
    const std::uint16_t* post_attention_layernorm = nullptr;
    std::uint64_t input_layernorm_identity = 0U;
    std::uint64_t post_attention_layernorm_identity = 0U;
    std::uint64_t pair_identity = 0U;
    std::uint32_t epsilon_fp32_bits = 0U;
  };

  using LayerNormExecutionBindingCatalog =
      std::array<LayerNormExecutionBinding,
                 kSm87MacroFeedV4P40StartupPackageLayers>;

  // Private execution-package construction value.  The typed asset is copied
  // only after its exact CUDA allocation range is observed.  Requests cannot
  // select or replace its model layer, role, layout, tactic, resource facts,
  // payload, or scales and cannot call the construction seal below.
  struct GdnQkvZExecutionBinding final {
    std::uint32_t gdn_ordinal =
        static_cast<std::uint32_t>(
            kSm87MacroFeedV4P40StartupPackageGdnLayers);
    std::uint32_t model_layer =
        static_cast<std::uint32_t>(
            kSm87MacroFeedV4P40StartupPackageLayers);
    kernels::Sm87TargetAotProjectionRole role =
        kernels::Sm87TargetAotProjectionRole::kInvalid;
    kernels::Sm87MacroFeedV4Fp8InputLayout input_layout =
        kernels::Sm87MacroFeedV4Fp8InputLayout::kInvalid;
    kernels::Sm87MacroFeedV4Fp8Identity tactic_identity =
        kernels::Sm87MacroFeedV4Fp8Identity::kInvalid;
    kernels::Sm87TargetAotFp8CudaAssetView asset{};
    kernels::Sm87MacroFeedV4Fp8CudaResources resources{};
    std::uint64_t package_identity = 0U;
    std::uint64_t deployment_plan_identity = 0U;
    std::uint64_t owner_identity = 0U;
    std::uint64_t allocation_identity = 0U;
    std::uint64_t projection_catalog_identity = 0U;
    std::uint64_t device_identity = 0U;
    std::uint64_t resource_seal_identity = 0U;
    std::uint64_t projection_binding_identity = 0U;
    std::uint64_t asset_value_identity = 0U;
    std::uint64_t binding_identity = 0U;
    bool live_cuda_payload_range_validated = false;
    bool request_selectable = false;
    bool launcher_authority = false;
    bool production_dispatch_eligible = false;
  };

  using GdnQkvZExecutionBindingCatalog =
      std::array<GdnQkvZExecutionBinding,
                 kSm87MacroFeedV4P40StartupPackageGdnLayers>;

  class Bf16AbStartupCapability;

  // Non-owning startup capability.  It is meaningful only while the exact
  // ModelWeights object and its resident allocation remain alive.  No public
  // method returns this object or one of its pointers.  The future execution
  // package may seal the complete canonical catalog once during construction;
  // it cannot choose a caller role, layer, pointer or offset.
  class Bf16AbStartupCapability final {
   public:
    Bf16AbStartupCapability() = default;
    Bf16AbStartupCapability(const Bf16AbStartupCapability&) = delete;
    Bf16AbStartupCapability& operator=(const Bf16AbStartupCapability&) =
        delete;
    Bf16AbStartupCapability(Bf16AbStartupCapability&& other) noexcept;
    Bf16AbStartupCapability& operator=(Bf16AbStartupCapability&&) = delete;
    ~Bf16AbStartupCapability() noexcept;

   private:
    Bf16AbStartupCapability(
        const ModelWeights* model_weights,
        std::array<Bf16AbPair,
                   kSm87MacroFeedV4P40StartupPackageBf16AbPairs>
            pairs,
        kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot resources,
        std::uint64_t catalog_identity, std::uint64_t package_identity,
        std::uint64_t deployment_plan_identity,
        std::uint64_t projection_owner_identity,
        std::uint64_t projection_allocation_identity,
        std::uint64_t projection_catalog_identity,
        std::uint64_t projection_device_identity,
        std::int32_t device_ordinal, std::uint64_t issuer_nonce) noexcept;

    [[nodiscard]] bool valid(const ProjectionAccess& access) const noexcept;
    const ModelWeights* model_weights_ = nullptr;
    std::array<Bf16AbPair,
               kSm87MacroFeedV4P40StartupPackageBf16AbPairs>
        pairs_{};
    kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot resources_{};
    std::uint64_t catalog_identity_ = 0U;
    std::uint64_t package_identity_ = 0U;
    std::uint64_t deployment_plan_identity_ = 0U;
    std::uint64_t projection_owner_identity_ = 0U;
    std::uint64_t projection_allocation_identity_ = 0U;
    std::uint64_t projection_catalog_identity_ = 0U;
    std::uint64_t projection_device_identity_ = 0U;
    std::int32_t device_ordinal_ = -1;
    std::uint64_t issuer_nonce_ = 0U;

    friend class Sm87MacroFeedV4P40StartupPackage;
  };

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
    Sm87MacroFeedV4Bf16AbStartupSeal bf16_ab{};
    Sm87MacroFeedV4GdnQkvZStartupSeal gdn_qkvz{};

    StartupSeals() = default;
    StartupSeals(const StartupSeals&) = delete;
    StartupSeals& operator=(const StartupSeals&) = delete;
    StartupSeals(StartupSeals&& other) noexcept
        : gate_up(std::move(other.gate_up)),
          down(std::move(other.down)),
          bf16_ab(std::move(other.bf16_ab)),
          gdn_qkvz(std::move(other.gdn_qkvz)) {}
    StartupSeals& operator=(StartupSeals&&) = delete;
  };

  Sm87MacroFeedV4P40StartupPackage(
      ProjectionAccess access,
      std::array<AssetCapability,
                 kSm87MacroFeedV4P40StartupPackageArtifacts> capabilities,
      Bf16AbStartupCapability bf16_ab_capability,
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
      std::uint64_t bf16_ab_binding_catalog_identity,
      const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
          bf16_ab,
      std::uint64_t gdn_qkvz_binding_catalog_identity,
      const kernels::Sm87MacroFeedV4Fp8CudaResources& gdn_qkvz,
      std::size_t sources) noexcept;
  [[nodiscard]] static std::uint64_t
  compute_gdn_qkvz_binding_catalog_identity(
      const ProjectionAccess& access,
      const std::array<AssetCapability,
                       kSm87MacroFeedV4P40StartupPackageArtifacts>&
          capabilities,
      std::uint64_t plan_identity,
      const kernels::Sm87MacroFeedV4Fp8CudaResources& resources) noexcept;
  [[nodiscard]] static StartupSeals mint_startup_seals(
      std::uint64_t package_identity, std::uint64_t plan_identity,
      const kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources& gate_up,
      const kernels::Sm87MacroFeedV4NvFp4DownCudaResources& down,
      std::uint64_t bf16_ab_binding_catalog_identity,
      const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
          bf16_ab,
      std::uint64_t gdn_qkvz_binding_catalog_identity,
      const kernels::Sm87MacroFeedV4Fp8CudaResources& gdn_qkvz) noexcept;
  [[nodiscard]] static bool startup_seals_valid(
      const StartupSeals& seals, std::uint64_t package_identity,
      std::uint64_t plan_identity, std::int32_t device_ordinal) noexcept;
  [[nodiscard]] static Sm87MacroFeedV4P40StartupPackageCreateResult
  build_from_private_authority(
      ProjectionAccess access, Sm87MacroFeedV4PanelWavefrontPlan plan,
      kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up,
      kernels::Sm87MacroFeedV4NvFp4DownCudaResources down,
      kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot bf16_ab,
      kernels::Sm87MacroFeedV4Fp8CudaResources gdn_qkvz,
      const ModelWeights& model_weights) noexcept;
  [[nodiscard]] static bool build_bf16_ab_pairs(
      const ModelWeights& model_weights, std::int32_t device_ordinal,
      std::array<Bf16AbPair,
                 kSm87MacroFeedV4P40StartupPackageBf16AbPairs>* pairs,
      Sm87MacroFeedV4Bf16AbT0InventoryAudit* inventory,
      int* cuda_error, std::size_t* failure_layer) noexcept;
  [[nodiscard]] const AssetCapability* capability(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;
  [[nodiscard]] bool base_valid() const noexcept;
  [[nodiscard]] bool populate_projection_bindings() noexcept;
  // Construction-only seam for the Engine-lifetime execution package.
  // It performs the expensive live catalog/resource validation once, then
  // returns all 48 immutable bindings together.  No request path may call it.
  [[nodiscard]] bool seal_bf16_ab_execution_catalog_for_execution_package(
      Bf16AbExecutionBindingCatalog* catalog) const noexcept;
  // Construction-only, all-or-nothing seal of the exact outer decoder-layer
  // RMSNorm weights.  The friend execution package retains this immutable
  // natural-order catalog under the same Engine lifetime root.  Request
  // execution must neither call this method nor rediscover CUDA ranges.
  [[nodiscard]] bool
  seal_layer_norm_execution_catalog_for_execution_package(
      LayerNormExecutionBindingCatalog* catalog,
      std::uint64_t* catalog_identity, std::size_t* failure_layer,
      bool* failure_post_attention, int* cuda_error) const noexcept;
  // Construction-only fixed GDN-QKVZ catalog.  Unlike the T0 host fixture,
  // every returned typed asset must belong to one observed live CUDA
  // allocation; failure clears the entire catalog and grants no capability.
  [[nodiscard]] bool
  seal_gdn_qkvz_execution_catalog_for_execution_package(
      GdnQkvZExecutionBindingCatalog* catalog,
      std::uint64_t* catalog_identity, std::size_t* failure_ordinal,
      int* cuda_error) const noexcept;
  [[nodiscard]] static std::uint64_t
  compute_gdn_qkvz_asset_value_identity(
      const kernels::Sm87TargetAotFp8CudaAssetView& asset) noexcept;
  [[nodiscard]] static std::uint64_t
  compute_gdn_qkvz_execution_binding_identity(
      const GdnQkvZExecutionBinding& binding) noexcept;
  [[nodiscard]] std::optional<Sm87MacroFeedV4ProjectionStartupBinding>
  make_projection_binding(
      std::size_t layer_index,
      kernels::Sm87TargetAotProjectionRole role) const noexcept;

  ProjectionAccess projection_access_;
  std::array<AssetCapability, kSm87MacroFeedV4P40StartupPackageArtifacts>
      capabilities_{};
  Bf16AbStartupCapability bf16_ab_capability_{};
  Sm87MacroFeedV4PanelWavefrontPlan plan_{};
  StartupSeals seals_{};
  Sm87MacroFeedV4P40StartupPackageAudit audit_{};
  std::array<std::optional<Sm87MacroFeedV4ProjectionStartupBinding>,
             kSm87MacroFeedV4P40StartupPackageArtifacts>
      projection_bindings_{};

  friend class sm87_macrofeed_v4_p40_execution_detail::
      Sm87MacroFeedV4P40ExecutionPackage;
};

static_assert(kSm87MacroFeedV4P40StartupPackageLayers ==
              kSm87TargetAotCompleteProjectionDeviceLayerCount);
static_assert(kSm87MacroFeedV4P40StartupPackageArtifacts ==
              kSm87TargetAotCompleteProjectionDeviceArtifactCount);
static_assert(kSm87MacroFeedV4P40StartupPackageSources ==
              kSm87TargetAotCompleteProjectionDeviceSourceCount);
static_assert(kSm87MacroFeedV4P40StartupPackageBf16AbPairs ==
              kSm87MacroFeedV4P40StartupPackageGdnLayers);
static_assert(kSm87MacroFeedV4P40StartupPackageBf16AbTensors == 96U);

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail
