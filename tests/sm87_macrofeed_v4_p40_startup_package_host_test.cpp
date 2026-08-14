#include "../src/runtime/sm87_macrofeed_v4_p40_startup_package_internal.h"
#include "support/sm87_target_aot_complete_host_fixture.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION)
#include <cuda_runtime_api.h>
#endif

namespace q3x::kernels {
namespace {

bool g_fail_v4_gate_up_resource_query = false;
std::int32_t g_v4_down_device_ordinal = 0;
bool g_fail_v4_bf16_ab_resource_query = false;
std::int32_t g_v4_bf16_ab_device_ordinal = 0;
bool g_poison_v4_bf16_ab_resource_identity = false;

[[nodiscard]] Sm87MacroFeedV4NvFp4GateUpCudaResources
host_test_v4_gate_up_resources() noexcept {
  Sm87MacroFeedV4NvFp4GateUpCudaResources resources;
  resources.identity = kSm87MacroFeedV4NvFp4GateUpIdentity;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.binary_version = 87;
  resources.registers_per_thread = 128;
  resources.static_shared_bytes = 0U;
  resources.dynamic_shared_bytes =
      kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = 2;
  resources.kernel_compiled = true;
  resources.static_resource_gate_passed = true;
  resources.numerical_contract_qualified = false;
  resources.production_dispatch_eligible = false;
  return resources;
}

[[nodiscard]] Sm87MacroFeedV4NvFp4DownCudaResources
host_test_v4_down_resources() noexcept {
  Sm87MacroFeedV4NvFp4DownCudaResources resources;
  resources.identity = kSm87MacroFeedV4NvFp4DownIdentity;
  resources.device_ordinal = g_v4_down_device_ordinal;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.binary_version = 87;
  resources.registers_per_thread = 102;
  resources.static_shared_bytes = 0U;
  resources.dynamic_shared_bytes =
      kSm87MacroFeedV4NvFp4DownDynamicSharedBytes;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = 2;
  resources.shared_bytes_per_sm = 102'400U;
  resources.optin_shared_bytes_per_block = 102'400U;
  resources.kernel_compiled = true;
  resources.static_resource_gate_passed = true;
  resources.numerical_contract_qualified = false;
  resources.production_dispatch_eligible = false;
  return resources;
}

[[nodiscard]] Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot
host_test_v4_bf16_ab_resources() noexcept {
  Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot resources;
  resources.identity = g_poison_v4_bf16_ab_resource_identity
                           ? Sm87MacroFeedV4Bf16AbIdentity::kInvalid
                           : kSm87MacroFeedV4Bf16AbIdentity;
  resources.device_ordinal = g_v4_bf16_ab_device_ordinal;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.binary_version = 87;
  resources.registers_per_thread = 88;
  resources.static_shared_bytes = 0U;
  resources.dynamic_shared_bytes =
      kSm87MacroFeedV4Bf16AbDynamicSharedBytes;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = 2;
  resources.threads_per_block =
      static_cast<std::int32_t>(kSm87MacroFeedV4Bf16AbThreads);
  resources.physical_grid_ctas =
      static_cast<std::int32_t>(kSm87MacroFeedV4Bf16AbGridCtas);
  resources.kernel_compiled = true;
  resources.exact_geometry = true;
  resources.static_resource_gate_passed = true;
  resources.numerical_contract_qualified = false;
  resources.production_dispatch_eligible = false;
  resources.startup_package_unbound = true;
  resources.execution_capability = false;
  resources.caller_snapshot_grants_production_authority = false;
  return resources;
}

}  // namespace

int query_sm87_macrofeed_v4_nvfp4_gate_up_cuda_resources(
    Sm87MacroFeedV4NvFp4GateUpCudaResources* const resources) noexcept {
  if (resources == nullptr || g_fail_v4_gate_up_resource_query) {
    if (resources != nullptr) {
      *resources = {};
    }
    return 1;
  }
  *resources = host_test_v4_gate_up_resources();
  return 0;
}

int query_sm87_macrofeed_v4_nvfp4_down_cuda_resources(
    Sm87MacroFeedV4NvFp4DownCudaResources* const resources) noexcept {
  if (resources == nullptr) {
    return 1;
  }
  *resources = host_test_v4_down_resources();
  return 0;
}

int query_sm87_macrofeed_v4_bf16_ab_admission_resource_snapshot_cuda(
    Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot* const resources)
    noexcept {
  if (resources == nullptr || g_fail_v4_bf16_ab_resource_query) {
    if (resources != nullptr) {
      *resources = {};
    }
    return 1;
  }
  *resources = host_test_v4_bf16_ab_resources();
  return 0;
}

}  // namespace q3x::kernels

namespace {

namespace package =
    q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail;
namespace execution =
    q3x::runtime::target_aot_complete_execution_detail;
using Package = package::Sm87MacroFeedV4P40StartupPackage;
using Binding = package::Sm87MacroFeedV4ProjectionStartupBinding;
using PackageError = package::Sm87MacroFeedV4P40StartupPackageError;
using Access = execution::Sm87TargetAotCompleteProjectionExecutionAccess;
using HostBf16AbPair =
    execution::Sm87TargetAotCompleteHostTestBf16AbPair;
using Owner = q3x::runtime::Sm87TargetAotCompleteProjectionDeviceAssets;
using Role = q3x::kernels::Sm87TargetAotProjectionRole;
using q3x::runtime::Bf16LinearWeight;
using q3x::runtime::Fp8LinearWeight;
using q3x::runtime::ModelWeights;

void require_package(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

[[nodiscard]] constexpr std::size_t gdn_model_layer(
    const std::size_t ordinal) noexcept {
  return ordinal + ordinal / 3U;
}

[[nodiscard]] std::array<package::Sm87MacroFeedV4Bf16AbT0TensorDescriptor,
                         package::kSm87MacroFeedV4P40StartupPackageBf16AbTensors>
make_bf16_ab_t0_inventory() noexcept {
  std::array<package::Sm87MacroFeedV4Bf16AbT0TensorDescriptor,
             package::kSm87MacroFeedV4P40StartupPackageBf16AbTensors>
      tensors{};
  constexpr std::uintptr_t kFakeBase = 0x0000'0060'0000'0000ULL;
  for (std::size_t ordinal = 0U;
       ordinal < package::kSm87MacroFeedV4P40StartupPackageBf16AbPairs;
       ++ordinal) {
    for (std::size_t role_index = 0U; role_index < 2U; ++role_index) {
      const std::size_t index = 2U * ordinal + role_index;
      tensors[index] = {
          ordinal,
          gdn_model_layer(ordinal),
          role_index == 0U
              ? package::Sm87MacroFeedV4Bf16AbWeightRole::kA
              : package::Sm87MacroFeedV4Bf16AbWeightRole::kB,
          q3x::runtime::LinearWeightKind::kBf16,
          reinterpret_cast<const std::uint16_t*>(
              kFakeBase + index *
                              q3x::kernels::
                                  kSm87MacroFeedV4Bf16AbWeightBytes),
          q3x::kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
          q3x::kernels::kSm87MacroFeedV4Bf16AbInputFeatures,
      };
    }
  }
  return tensors;
}

void test_bf16_ab_t0_inventory_contract() {
  static_assert(!std::is_default_constructible_v<
                package::Sm87MacroFeedV4Bf16AbStartupSeal>);
  static_assert(!std::is_copy_constructible_v<
                package::Sm87MacroFeedV4Bf16AbStartupSeal>);
  static_assert(!std::is_move_constructible_v<
                package::Sm87MacroFeedV4Bf16AbStartupSeal>);
  static_assert(!std::is_trivially_copyable_v<
                package::Sm87MacroFeedV4Bf16AbStartupSeal>);

  const auto canonical = make_bf16_ab_t0_inventory();
  const auto audit = package::inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
      canonical.data(), canonical.size());
  require_package(
      audit.valid_t0() && audit.tensors == 96U && audit.pairs == 48U &&
          audit.catalog_identity != 0U &&
          !audit.live_cuda_device_ranges_validated &&
          !audit.execution_capability,
      "canonical 48-pair BF16 A/B T0 mapping did not close");

  auto swapped_roles = canonical;
  std::swap(swapped_roles[0U], swapped_roles[1U]);
  require_package(
      !package::inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
           swapped_roles.data(), swapped_roles.size())
           .valid_t0(),
      "BF16 A/B role swap was accepted");
  require_package(
      !package::inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
           canonical.data(), canonical.size() - 1U)
           .valid_t0(),
      "missing BF16 tensor was accepted");

  auto wrong_order = canonical;
  wrong_order[8U].model_layer += 1U;
  require_package(
      !package::inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
           wrong_order.data(), wrong_order.size())
           .valid_t0(),
      "non-canonical GDN layer order was accepted");

  auto wrong_shape = canonical;
  wrong_shape[12U].output_size += 1U;
  require_package(
      !package::inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
           wrong_shape.data(), wrong_shape.size())
           .valid_t0(),
      "wrong BF16 A/B shape was accepted");

  auto wrong_dtype = canonical;
  wrong_dtype[20U].weight_kind = q3x::runtime::LinearWeightKind::kFp8;
  require_package(
      !package::inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
           wrong_dtype.data(), wrong_dtype.size())
           .valid_t0(),
      "non-BF16 A/B tensor was accepted");

  auto unaligned = canonical;
  unaligned[30U].weight = reinterpret_cast<const std::uint16_t*>(
      reinterpret_cast<std::uintptr_t>(unaligned[30U].weight) + 2U);
  require_package(
      !package::inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
           unaligned.data(), unaligned.size())
           .valid_t0(),
      "unaligned BF16 A/B tensor was accepted");

  auto overlap = canonical;
  overlap[41U].weight = overlap[40U].weight;
  require_package(
      !package::inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
           overlap.data(), overlap.size())
           .valid_t0(),
      "overlapping BF16 A/B tensor ranges were accepted");
}

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
[[nodiscard]] bool clear_fixture(
    std::optional<ModelWeights>& model_weights, Owner& owner) {
  model_weights.reset();
  return Access::clear_host_test_fixture(owner) && owner.empty();
}
#endif

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION)

class LiveBf16AbHostFixture final {
 public:
  LiveBf16AbHostFixture() = default;
  LiveBf16AbHostFixture(const LiveBf16AbHostFixture&) = delete;
  LiveBf16AbHostFixture& operator=(const LiveBf16AbHostFixture&) = delete;
  ~LiveBf16AbHostFixture() {
    if (allocation_ != nullptr) {
      (void)cudaFree(allocation_);
      allocation_ = nullptr;
    }
  }

  [[nodiscard]] bool allocate() noexcept {
    if (allocation_ != nullptr) {
      return false;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess ||
        current_device != 0 ||
        cudaMalloc(&allocation_, kAllocationBytes) != cudaSuccess) {
      allocation_ = nullptr;
      return false;
    }
    auto* const bytes = static_cast<std::uint8_t*>(allocation_);
    for (std::size_t ordinal = 0U; ordinal < pairs_.size(); ++ordinal) {
      const auto* const a = reinterpret_cast<const std::uint16_t*>(
          bytes + (2U * ordinal) * kWeightBytes);
      const auto* const b = reinterpret_cast<const std::uint16_t*>(
          bytes + (2U * ordinal + 1U) * kWeightBytes);
      pairs_[ordinal].a = Bf16LinearWeight{
          a, q3x::kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
          q3x::kernels::kSm87MacroFeedV4Bf16AbInputFeatures};
      pairs_[ordinal].b = Bf16LinearWeight{
          b, q3x::kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
          q3x::kernels::kSm87MacroFeedV4Bf16AbInputFeatures};
    }
    return true;
  }

  [[nodiscard]] bool install(ModelWeights& weights,
                             const std::size_t pair_count =
                                 package::kSm87MacroFeedV4P40StartupPackageBf16AbPairs)
      noexcept {
    return allocation_ != nullptr && pair_count <= pairs_.size() &&
           Access::install_complete_host_test_bf16_ab_pairs(
               weights, pairs_.data(), pair_count);
  }

  [[nodiscard]] HostBf16AbPair& pair(const std::size_t ordinal) noexcept {
    return pairs_[ordinal];
  }

  [[nodiscard]] const std::uint16_t* one_past_allocation() const noexcept {
    return reinterpret_cast<const std::uint16_t*>(
        static_cast<const std::uint8_t*>(allocation_) + kAllocationBytes);
  }

 private:
  static constexpr std::size_t kWeightBytes =
      q3x::kernels::kSm87MacroFeedV4Bf16AbWeightBytes;
  static constexpr std::size_t kAllocationBytes =
      package::kSm87MacroFeedV4P40StartupPackageBf16AbTensors *
      kWeightBytes;

  void* allocation_ = nullptr;
  std::array<HostBf16AbPair,
             package::kSm87MacroFeedV4P40StartupPackageBf16AbPairs>
      pairs_{};
};

[[nodiscard]] bool live_sm87_test_device_available() noexcept {
  int device = -1;
  cudaDeviceProp properties{};
  return cudaGetDevice(&device) == cudaSuccess && device == 0 &&
         cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
         properties.major == 8 && properties.minor == 7;
}

void test_complete_v4_foundation_package() {
  static_assert(package::kSm87MacroFeedV4P40StartupPackageCompiled);
  static_assert(!std::is_default_constructible_v<Package>);
  static_assert(!std::is_copy_constructible_v<Package>);
  static_assert(!std::is_move_constructible_v<Package>);
  static_assert(!std::is_default_constructible_v<Binding>);
  static_assert(std::is_copy_constructible_v<Binding>);
  static_assert(std::is_move_constructible_v<Binding>);
  static_assert(!std::is_copy_assignable_v<Binding>);
  static_assert(!std::is_move_assignable_v<Binding>);
  static_assert(!std::is_copy_constructible_v<
                package::Sm87MacroFeedV4GateUpStartupSeal>);
  static_assert(!std::is_default_constructible_v<
                package::Sm87MacroFeedV4GateUpStartupSeal>);
  static_assert(!std::is_move_constructible_v<
                package::Sm87MacroFeedV4GateUpStartupSeal>);
  static_assert(!std::is_trivially_copyable_v<
                package::Sm87MacroFeedV4GateUpStartupSeal>);
  static_assert(!std::is_copy_constructible_v<
                package::Sm87MacroFeedV4DownStartupSeal>);
  static_assert(!std::is_default_constructible_v<
                package::Sm87MacroFeedV4DownStartupSeal>);
  static_assert(!std::is_move_constructible_v<
                package::Sm87MacroFeedV4DownStartupSeal>);
  static_assert(!std::is_trivially_copyable_v<
                package::Sm87MacroFeedV4DownStartupSeal>);
  static_assert(!std::is_copy_constructible_v<
                package::Sm87MacroFeedV4Bf16AbStartupSeal>);
  static_assert(!std::is_default_constructible_v<
                package::Sm87MacroFeedV4Bf16AbStartupSeal>);
  static_assert(!std::is_move_constructible_v<
                package::Sm87MacroFeedV4Bf16AbStartupSeal>);

  Owner owner;
  LiveBf16AbHostFixture bf16_ab;
  require_package(bf16_ab.allocate(),
                  "could not allocate live BF16 A/B fixture");
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct V4 startup fixture");
  require_package(bf16_ab.install(*model_weights),
                  "could not install live BF16 A/B fixture");

  auto first = Package::create(*model_weights);
  if (!first) {
    std::cerr << "V4 startup create error="
              << static_cast<unsigned>(first.status.error)
              << " context=" << first.status.context
              << " layer=" << first.status.layer
              << " role=" << static_cast<unsigned>(first.status.role)
              << " cuda=" << first.status.cuda_error << '\n';
  }
  require_package(static_cast<bool>(first),
                  "complete owner did not create V4 startup package");
  require_package(first.audit.valid() && first.package->audit().valid(),
                  "V4 capability-free audit is not closed");
  require_package(
      first.audit.magic == package::kSm87MacroFeedV4P40StartupPackageMagic &&
          first.audit.abi_major ==
              package::kSm87MacroFeedV4P40StartupPackageAbiMajor &&
          first.audit.candidate_id ==
              q3x::runtime::kSm87MacroFeedV4CandidateId &&
          first.audit.deployment_plan_id ==
              q3x::runtime::kSm87MacroFeedV4P40DeploymentPlanId &&
          first.audit.deployment_plan_identity != 0U &&
          first.audit.device_identity == execution::kHostTestDeviceIdentity &&
          first.audit.device_ordinal == 0 && first.audit.layers == 64U &&
          first.audit.artifacts == 256U && first.audit.sources == 400U &&
          first.audit.gate_up_assets == 64U &&
          first.audit.down_assets == 64U &&
          first.audit.gdn_projection_assets == 48U &&
          first.audit.full_projection_assets == 16U &&
          first.audit.attention_output_assets == 64U &&
          first.audit.bf16_ab_tensors == 96U &&
          first.audit.bf16_ab_pairs == 48U &&
          first.audit.bf16_ab_binding_catalog_identity != 0U &&
          first.audit.bf16_ab_resource_seal_identity != 0U,
      "V4 startup identity or complete asset cardinality drifted");
  require_package(
      first.audit.canonical_plan_generated_internally &&
          !first.audit.caller_plan_accepted &&
          first.audit.complete_projection_access_retained &&
          first.audit.catalog_revalidated &&
          first.audit.typed_capabilities_retained &&
          first.audit.authenticated_source_manifests_retained &&
          first.audit.authenticated_upload_readback_retained &&
          first.audit.projection_bindings_complete &&
          first.audit.nvfp4_startup_seals_complete &&
          first.audit.bf16_ab_nonowning_model_weights_dependency_bound &&
          first.audit.bf16_ab_projection_owner_identity_retained &&
          first.audit.bf16_ab_natural_layer_order_complete &&
          first.audit.bf16_ab_a_then_b_roles_complete &&
          first.audit.bf16_ab_live_device_ranges_complete &&
          first.audit.bf16_ab_resource_seal_complete &&
          first.audit.bf16_ab_private_capability_retained &&
          !first.audit.bf16_ab_raw_pointer_publicly_exposed &&
          !first.audit.caller_raw_receipts_accepted &&
          !first.audit.v3_execution_identity_reused &&
          !first.audit
               .request_time_repack_jit_autotune_or_fallback_permitted,
      "V4 package reused caller or V3 execution authority");
  require_package(
      !first.audit.fp8_executor_bound && !first.audit.gdn_executor_bound &&
          !first.audit.attention_executor_bound &&
          !first.audit.request_state_bound && !first.audit.finalizer_bound &&
          !first.audit.physical_receipt_bound && first.audit.host_only &&
          first.audit.default_off && first.audit.test_only &&
          !first.audit.selector_bound && !first.audit.launcher_present &&
          !first.audit.execution_ready &&
          !first.audit.numerical_qualification_complete &&
          !first.audit.production_dispatch_eligible,
      "V4 T0 foundation concealed an unimplemented dependency");

  require_package(
      q3x::runtime::sm87_macrofeed_v4_p40_panel_wavefront_plan_valid(
          first.package->plan()) &&
          first.package->plan().deployment_plan_id ==
              q3x::runtime::kSm87MacroFeedV4P40DeploymentPlanId &&
          !first.package->plan().route.selector_bound &&
          !first.package->plan().route.launcher_present &&
          !first.package->plan().route.production_dispatch_eligible,
      "V4 package did not retain its internally generated canonical plan");

  const auto& gate_seal = first.package->gate_up_startup_seal();
  const auto& down_seal = first.package->down_startup_seal();
  const auto& bf16_ab_seal = first.package->bf16_ab_startup_seal();
  require_package(
      gate_seal.valid() && down_seal.valid() && bf16_ab_seal.valid() &&
          gate_seal.seal_identity != down_seal.seal_identity &&
          bf16_ab_seal.seal_identity != gate_seal.seal_identity &&
          bf16_ab_seal.seal_identity != down_seal.seal_identity &&
          gate_seal.package_identity == first.audit.package_identity &&
          down_seal.package_identity == first.audit.package_identity &&
          bf16_ab_seal.package_identity == first.audit.package_identity &&
          gate_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          down_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          bf16_ab_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          bf16_ab_seal.binding_catalog_identity ==
              first.audit.bf16_ab_binding_catalog_identity &&
          bf16_ab_seal.tensor_count == 96U &&
          bf16_ab_seal.pair_count == 48U &&
          q3x::kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
              gate_seal.resources) &&
          q3x::kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(
              down_seal.resources) &&
          q3x::kernels::
              sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
                  bf16_ab_seal.resources) &&
          bf16_ab_seal.canonical_natural_layer_order &&
          bf16_ab_seal.canonical_a_then_b_role_order &&
          bf16_ab_seal.complete_live_device_ranges &&
          !bf16_ab_seal.raw_pointer_exposed &&
          !gate_seal.launcher_authority && !down_seal.launcher_authority &&
          !bf16_ab_seal.launcher_authority &&
          !gate_seal.caller_receipt_accepted &&
          !down_seal.caller_receipt_accepted &&
          !bf16_ab_seal.caller_resource_snapshot_accepted &&
          !bf16_ab_seal.production_dispatch_eligible,
      "V4 C8000 startup resource seals are not independently closed");

  auto second = Package::create(*model_weights);
  require_package(static_cast<bool>(second) &&
                      second.audit.package_identity ==
                          first.audit.package_identity &&
                      second.audit.deployment_plan_identity ==
                          first.audit.deployment_plan_identity,
                  "V4 startup identities are unstable for one authority");

  q3x::kernels::g_poison_v4_bf16_ab_resource_identity = true;
  require_package(!first.package->valid() && !second.package->valid(),
                  "V4 package cached a stale/forged BF16 resource seal");
  q3x::kernels::g_poison_v4_bf16_ab_resource_identity = false;
  require_package(first.package->valid() && second.package->valid(),
                  "V4 package did not recover exact observed BF16 resources");

  Package::ProjectionStartupBindingCatalog catalog{};
  require_package(
      !first.package->borrow_projection_startup_catalog(nullptr) &&
          first.package->borrow_projection_startup_catalog(&catalog),
      "V4 startup catalog did not close one authenticated borrow");

  std::size_t observed = 0U;
  std::size_t sources = 0U;
  for (std::size_t layer = 0U; layer < 64U; ++layer) {
    const Role primary =
        q3x::runtime::sm87_target_aot_complete_is_full_layer(layer)
            ? Role::kFp8FullQkv
            : Role::kFp8GdnQkvZ;
    for (const Role role : {Role::kNvFp4GateUp, Role::kNvFp4Down,
                            primary, Role::kFp8AttentionOutput}) {
      const Binding* const binding = catalog[observed];
      require_package(
          binding != nullptr && binding->layer_index() == layer &&
              binding->role() == role &&
              binding->binding_identity() != 0U &&
              binding->deployment_plan_identity() ==
                  first.audit.deployment_plan_identity &&
              binding->device_identity() == first.audit.device_identity &&
              binding->artifact_identity() != 0U &&
              binding->source_inventory_identity() != 0U &&
              binding->payload_begin() != 0U &&
              binding->payload_end() > binding->payload_begin() &&
              binding->payload_end() - binding->payload_begin() ==
                  binding->payload_bytes() &&
              binding->source_count() != 0U &&
              binding->tensor_scale_bits(0U) != 0U &&
              binding->tensor_scale_bits(binding->source_count()) == 0U &&
              !binding->launcher_authority(),
          "V4 binding lost exact owner/plan/payload/scale facts");
      const std::uint64_t expected_tactic =
          role == Role::kNvFp4GateUp
              ? static_cast<std::uint64_t>(
                    q3x::kernels::kSm87MacroFeedV4NvFp4GateUpIdentity)
              : (role == Role::kNvFp4Down
                     ? static_cast<std::uint64_t>(
                           q3x::kernels::kSm87MacroFeedV4NvFp4DownIdentity)
                     : 0U);
      require_package(binding->consumer_tactic_identity() == expected_tactic,
                      "V4 binding reused another execution tactic identity");

      if ((observed % 64U) == 0U) {
        require_package(
            binding->valid() &&
                binding->valid_for(layer, role,
                                   first.audit.package_identity),
            "V4 representative binding lost standalone validation");
        Binding copy(*binding);
        require_package(
            copy.valid() &&
                copy.binding_identity() == binding->binding_identity() &&
                !copy.valid_for(layer == 0U ? 1U : 0U, role,
                                first.audit.package_identity) &&
                !copy.valid_for(
                    layer,
                    role == Role::kNvFp4GateUp ? Role::kNvFp4Down
                                               : Role::kNvFp4GateUp,
                    first.audit.package_identity) &&
                !copy.valid_for(layer, role,
                                first.audit.package_identity ^ 1U),
            "V4 binding accepted a changed package, layer, or role");
      }
      sources += binding->source_count();
      ++observed;
    }
  }
  require_package(
      observed == 256U && sources == 400U &&
          first.package->borrow_projection_startup_binding(
              64U, Role::kNvFp4GateUp) == nullptr &&
          first.package->borrow_projection_startup_binding(
              0U, Role::kInvalid) == nullptr &&
          first.package->borrow_projection_startup_binding(
              0U, Role::kFp8FullQkv) == nullptr,
      "V4 binding catalog accepted an absent key or missed a live asset");

  const Binding* const poisoned_borrow =
      first.package->borrow_projection_startup_binding(
          63U, Role::kFp8AttentionOutput);
  require_package(poisoned_borrow != nullptr && poisoned_borrow->valid(),
                  "could not retain V4 binding for live receipt poison");
  Binding poisoned(*poisoned_borrow);
  require_package(
      Access::poison_host_test_fixture_receipt(
          owner, 63U, Role::kFp8AttentionOutput) &&
          !first.package->valid() && !second.package->valid() &&
          !poisoned.valid() &&
          first.package->borrow_projection_startup_binding(
              63U, Role::kFp8AttentionOutput) == nullptr,
      "V4 package did not revalidate an upload/readback receipt");

  first.package.reset();
  second.package.reset();
  require_package(clear_fixture(model_weights, owner),
                  "V4 complete fixture cleanup failed");
}

void test_source_scale_and_device_tamper_fail_closed() {
  {
    Owner owner;
    LiveBf16AbHostFixture bf16_ab;
    require_package(bf16_ab.allocate(), "source BF16 allocation failed");
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value(), "source fixture construction failed");
    require_package(bf16_ab.install(*weights),
                    "source BF16 fixture installation failed");
    auto result = Package::create(*weights);
    require_package(static_cast<bool>(result), "source package creation failed");
    const Binding* const binding =
        result.package->borrow_projection_startup_binding(
            11U, Role::kNvFp4GateUp);
    require_package(
        binding != nullptr && binding->valid() &&
            Access::tamper_host_test_fixture_source_identity(
                owner, 11U, Role::kNvFp4GateUp, 1U) &&
            !binding->valid() && !result.package->valid(),
        "V4 package accepted a changed authenticated source identity");
    result.package.reset();
    require_package(clear_fixture(weights, owner),
                    "source fixture cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture bf16_ab;
    require_package(bf16_ab.allocate(), "scale BF16 allocation failed");
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value(), "scale fixture construction failed");
    require_package(bf16_ab.install(*weights),
                    "scale BF16 fixture installation failed");
    auto result = Package::create(*weights);
    require_package(static_cast<bool>(result), "scale package creation failed");
    const Binding* const binding =
        result.package->borrow_projection_startup_binding(
            19U, Role::kNvFp4Down);
    require_package(
        binding != nullptr && binding->valid() &&
            Access::tamper_host_test_fixture_scale_bits(
                owner, 19U, Role::kNvFp4Down, 0U) &&
            !binding->valid() && !result.package->valid(),
        "V4 package accepted changed live tensor-scale bits");
    result.package.reset();
    require_package(clear_fixture(weights, owner),
                    "scale fixture cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture bf16_ab;
    require_package(bf16_ab.allocate(), "device BF16 allocation failed");
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value(), "device fixture construction failed");
    require_package(bf16_ab.install(*weights),
                    "device BF16 fixture installation failed");
    auto result = Package::create(*weights);
    require_package(static_cast<bool>(result), "device package creation failed");
    const Binding* const binding =
        result.package->borrow_projection_startup_binding(
            23U, Role::kNvFp4GateUp);
    require_package(
        binding != nullptr && binding->valid() &&
            Access::tamper_host_test_fixture_device_identity(owner) &&
            !binding->valid() && !result.package->valid() &&
            !Access::bind(*weights).has_value(),
        "V4 package survived a physical-device identity change");
    result.package.reset();
    std::optional<ModelWeights> moved(std::in_place, std::move(*weights));
    weights.reset();
    moved.reset();
    require_package(Access::clear_host_test_fixture(owner) && owner.empty(),
                    "device fixture cleanup failed");
  }
}

void test_bf16_ab_model_and_range_failures() {
  {
    Owner owner;
    LiveBf16AbHostFixture bf16_ab;
    require_package(bf16_ab.allocate(), "missing BF16 allocation failed");
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && bf16_ab.install(*weights, 47U),
                    "missing BF16 fixture setup failed");
    auto result = Package::create(*weights);
    require_package(
        !result && result.status.error ==
                       PackageError::kBf16AbModelInventory &&
            result.status.layer == 62U,
        "missing 48th BF16 A/B pair was accepted");
    require_package(clear_fixture(weights, owner),
                    "missing BF16 fixture cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture bf16_ab;
    require_package(bf16_ab.allocate(), "shape BF16 allocation failed");
    auto& malformed =
        std::get<Bf16LinearWeight>(bf16_ab.pair(7U).a);
    malformed.output_size += 1U;
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && bf16_ab.install(*weights),
                    "shape BF16 fixture setup failed");
    auto result = Package::create(*weights);
    require_package(
        !result && result.status.error ==
                       PackageError::kBf16AbModelInventory &&
            result.status.layer == 9U,
        "wrong BF16 A/B shape was accepted by startup package");
    require_package(clear_fixture(weights, owner),
                    "shape BF16 fixture cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture bf16_ab;
    require_package(bf16_ab.allocate(), "dtype BF16 allocation failed");
    const auto* const original =
        std::get<Bf16LinearWeight>(bf16_ab.pair(9U).b).weight;
    bf16_ab.pair(9U).b = Fp8LinearWeight{
        reinterpret_cast<const std::uint8_t*>(original),
        nullptr,
        nullptr,
        1.0F,
        1.0F,
        q3x::kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
        q3x::kernels::kSm87MacroFeedV4Bf16AbInputFeatures};
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && bf16_ab.install(*weights),
                    "dtype BF16 fixture setup failed");
    auto result = Package::create(*weights);
    require_package(
        !result && result.status.error ==
                       PackageError::kBf16AbModelInventory &&
            result.status.layer == 12U,
        "non-BF16 A/B model field was accepted by startup package");
    require_package(clear_fixture(weights, owner),
                    "dtype BF16 fixture cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture bf16_ab;
    require_package(bf16_ab.allocate(), "range BF16 allocation failed");
    bf16_ab.pair(47U).b = Bf16LinearWeight{
        bf16_ab.one_past_allocation(),
        q3x::kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
        q3x::kernels::kSm87MacroFeedV4Bf16AbInputFeatures};
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && bf16_ab.install(*weights),
                    "range BF16 fixture setup failed");
    auto result = Package::create(*weights);
    require_package(
        !result && result.status.error == PackageError::kBf16AbDeviceRange &&
            result.status.layer == 62U && result.status.cuda_error != 0,
        "non-resident BF16 A/B allocation range was accepted");
    require_package(clear_fixture(weights, owner),
                    "range BF16 fixture cleanup failed");
  }
}

void test_model_weights_lifetime_order_fail_closed() {
  Owner owner;
  LiveBf16AbHostFixture bf16_ab;
  require_package(bf16_ab.allocate(), "lifetime BF16 allocation failed");
  std::optional<ModelWeights> weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(weights.has_value() && bf16_ab.install(*weights),
                  "lifetime BF16 fixture setup failed");
  auto result = Package::create(*weights);
  require_package(static_cast<bool>(result),
                  "lifetime startup package creation failed");

  std::optional<ModelWeights> moved(std::in_place, std::move(*weights));
  require_package(!result.package->valid(),
                  "startup package survived its exact ModelWeights root move");
  result.package.reset();
  weights.reset();
  moved.reset();
  require_package(Access::clear_host_test_fixture(owner) && owner.empty(),
                  "lifetime fixture cleanup order failed");
}

void test_resource_failures_fail_closed() {
  Owner owner;
  LiveBf16AbHostFixture bf16_ab;
  require_package(bf16_ab.allocate(), "resource BF16 allocation failed");
  std::optional<ModelWeights> weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(weights.has_value(), "resource fixture construction failed");
  require_package(bf16_ab.install(*weights),
                  "resource BF16 fixture installation failed");

  q3x::kernels::g_fail_v4_gate_up_resource_query = true;
  auto gate_failure = Package::create(*weights);
  q3x::kernels::g_fail_v4_gate_up_resource_query = false;
  require_package(
      !gate_failure && gate_failure.package == nullptr &&
          gate_failure.status.error == PackageError::kGateUpStartupSeal &&
          gate_failure.status.cuda_error == 1,
      "V4 Gate+Up resource query failure did not fail closed");

  q3x::kernels::g_fail_v4_bf16_ab_resource_query = true;
  auto bf16_query_failure = Package::create(*weights);
  q3x::kernels::g_fail_v4_bf16_ab_resource_query = false;
  require_package(
      !bf16_query_failure && bf16_query_failure.package == nullptr &&
          bf16_query_failure.status.error ==
              PackageError::kBf16AbResourceSeal &&
          bf16_query_failure.status.cuda_error == 1,
      "V4 BF16 A/B resource query failure did not fail closed");

  q3x::kernels::g_poison_v4_bf16_ab_resource_identity = true;
  auto bf16_identity_failure = Package::create(*weights);
  q3x::kernels::g_poison_v4_bf16_ab_resource_identity = false;
  require_package(
      !bf16_identity_failure && bf16_identity_failure.package == nullptr &&
          bf16_identity_failure.status.error ==
              PackageError::kBf16AbResourceSeal,
      "caller-forgeable BF16 A/B resource identity was accepted");

  q3x::kernels::g_v4_down_device_ordinal = 1;
  auto device_failure = Package::create(*weights);
  q3x::kernels::g_v4_down_device_ordinal = 0;
  require_package(
      !device_failure && device_failure.package == nullptr &&
          device_failure.status.error == PackageError::kDeviceMismatch,
      "V4 cross-device startup resource set did not fail closed");

  q3x::kernels::g_v4_bf16_ab_device_ordinal = 1;
  auto bf16_device_failure = Package::create(*weights);
  q3x::kernels::g_v4_bf16_ab_device_ordinal = 0;
  require_package(
      !bf16_device_failure && bf16_device_failure.package == nullptr &&
          bf16_device_failure.status.error == PackageError::kDeviceMismatch,
      "V4 BF16 A/B resource seal accepted another device");
  require_package(clear_fixture(weights, owner),
                  "resource fixture cleanup failed");
}

#elif defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)

void test_default_off_package_is_closed() {
  static_assert(!package::kSm87MacroFeedV4P40StartupPackageCompiled);
  Owner owner;
  std::optional<ModelWeights> weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(weights.has_value(),
                  "could not construct default-off V4 fixture");
  auto result = Package::create(*weights);
  require_package(
      !result && result.package == nullptr &&
          result.status.error == PackageError::kAdmissionDisabled,
      "default-off V4 startup package did not fail closed");
  require_package(clear_fixture(weights, owner),
                  "default-off V4 fixture cleanup failed");
}

#else

void test_default_off_package_linkage_is_closed() {
  static_assert(!package::kSm87MacroFeedV4P40StartupPackageCompiled);
  package::Sm87MacroFeedV4P40StartupPackageCreateResult result;
  require_package(!result && result.package == nullptr,
                  "default-off V4 result was unexpectedly valid");
}

#endif

}  // namespace

int main() {
  test_bf16_ab_t0_inventory_contract();
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION)
  if (!live_sm87_test_device_available()) {
    std::cout << "sm87_macrofeed_v4_p40_startup_package_host_test: "
                 "SKIP (T0 PASS; live SM87 T1 unavailable)\n";
    return 77;
  }
  test_complete_v4_foundation_package();
  test_source_scale_and_device_tamper_fail_closed();
  test_bf16_ab_model_and_range_failures();
  test_model_weights_lifetime_order_fail_closed();
  test_resource_failures_fail_closed();
  std::cout << "sm87_macrofeed_v4_p40_startup_package_host_test: PASS\n";
#elif defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  test_default_off_package_is_closed();
#else
  test_default_off_package_linkage_is_closed();
#endif
  return 0;
}
