#include "../src/runtime/sm87_macrofeed_v4_p40_startup_package_internal.h"
#include "support/sm87_target_aot_complete_host_fixture.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace q3x::kernels {
namespace {

bool g_fail_v4_gate_up_resource_query = false;
std::int32_t g_v4_down_device_ordinal = 0;

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
using Owner = q3x::runtime::Sm87TargetAotCompleteProjectionDeviceAssets;
using Role = q3x::kernels::Sm87TargetAotProjectionRole;
using q3x::runtime::ModelWeights;

void require_package(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
[[nodiscard]] bool clear_fixture(
    std::optional<ModelWeights>& model_weights, Owner& owner) {
  model_weights.reset();
  return Access::clear_host_test_fixture(owner) && owner.empty();
}
#endif

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)

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

  Owner owner;
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(model_weights.has_value(),
                  "could not construct V4 startup fixture");

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
          first.audit.attention_output_assets == 64U,
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
  require_package(
      gate_seal.valid() && down_seal.valid() &&
          gate_seal.seal_identity != down_seal.seal_identity &&
          gate_seal.package_identity == first.audit.package_identity &&
          down_seal.package_identity == first.audit.package_identity &&
          gate_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          down_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          q3x::kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
              gate_seal.resources) &&
          q3x::kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(
              down_seal.resources) &&
          !gate_seal.launcher_authority && !down_seal.launcher_authority &&
          !gate_seal.caller_receipt_accepted &&
          !down_seal.caller_receipt_accepted,
      "V4 C8000 startup resource seals are not independently closed");

  auto second = Package::create(*model_weights);
  require_package(static_cast<bool>(second) &&
                      second.audit.package_identity ==
                          first.audit.package_identity &&
                      second.audit.deployment_plan_identity ==
                          first.audit.deployment_plan_identity,
                  "V4 startup identities are unstable for one authority");

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
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value(), "source fixture construction failed");
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
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value(), "scale fixture construction failed");
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
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value(), "device fixture construction failed");
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

void test_resource_failures_fail_closed() {
  Owner owner;
  std::optional<ModelWeights> weights =
      Access::make_complete_host_test_fixture(owner);
  require_package(weights.has_value(), "resource fixture construction failed");

  q3x::kernels::g_fail_v4_gate_up_resource_query = true;
  auto gate_failure = Package::create(*weights);
  q3x::kernels::g_fail_v4_gate_up_resource_query = false;
  require_package(
      !gate_failure && gate_failure.package == nullptr &&
          gate_failure.status.error == PackageError::kGateUpStartupSeal &&
          gate_failure.status.cuda_error == 1,
      "V4 Gate+Up resource query failure did not fail closed");

  q3x::kernels::g_v4_down_device_ordinal = 1;
  auto device_failure = Package::create(*weights);
  q3x::kernels::g_v4_down_device_ordinal = 0;
  require_package(
      !device_failure && device_failure.package == nullptr &&
          device_failure.status.error == PackageError::kDeviceMismatch,
      "V4 cross-device startup resource set did not fail closed");
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
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  test_complete_v4_foundation_package();
  test_source_scale_and_device_tamper_fail_closed();
  test_resource_failures_fail_closed();
  std::cout << "sm87_macrofeed_v4_p40_startup_package_host_test: PASS\n";
#elif defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  test_default_off_package_is_closed();
#else
  test_default_off_package_linkage_is_closed();
#endif
  return 0;
}
