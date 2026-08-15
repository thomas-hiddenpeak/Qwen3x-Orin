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
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace q3x::kernels {
namespace {

bool g_fail_v4_gate_up_resource_query = false;
std::int32_t g_v4_down_device_ordinal = 0;
bool g_fail_v4_bf16_ab_resource_query = false;
std::int32_t g_v4_bf16_ab_device_ordinal = 0;
bool g_poison_v4_bf16_ab_resource_identity = false;
bool g_fail_v4_gdn_qkvz_resource_query = false;
std::int32_t g_v4_gdn_qkvz_device_ordinal = 0;
bool g_poison_v4_gdn_qkvz_resource_identity = false;
bool g_fail_v4_gdn_output_resource_query = false;
std::int32_t g_v4_gdn_output_device_ordinal = 0;
bool g_poison_v4_gdn_output_resource_identity = false;

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

[[nodiscard]] Sm87MacroFeedV4Fp8CudaResources
host_test_v4_gdn_qkvz_resources() noexcept {
  Sm87MacroFeedV4Fp8CudaResources resources;
  resources.identity = g_poison_v4_gdn_qkvz_resource_identity
                           ? Sm87MacroFeedV4Fp8Identity::kInvalid
                           : Sm87MacroFeedV4Fp8Identity::
                                 kGdnQkvZM64N128K64OrdinaryGridV1;
  resources.role = Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
  resources.input_layout =
      Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1;
  resources.device_ordinal = g_v4_gdn_qkvz_device_ordinal;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.binary_version = 87;
  resources.registers_per_thread = 96;
  resources.static_shared_bytes = 0U;
  resources.dynamic_shared_bytes = kSm87MacroFeedV4Fp8DynamicSharedBytes;
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

[[nodiscard]] Sm87MacroFeedV4Fp8CudaResources
host_test_v4_gdn_output_resources() noexcept {
  auto resources = host_test_v4_gdn_qkvz_resources();
  resources.identity = g_poison_v4_gdn_output_resource_identity
                           ? Sm87MacroFeedV4Fp8Identity::kInvalid
                           : Sm87MacroFeedV4Fp8Identity::
                                 kGdnAttentionOutputM64N128K64OrdinaryGridV1;
  resources.role = Sm87TargetAotProjectionRole::kFp8AttentionOutput;
  resources.input_layout =
      Sm87MacroFeedV4Fp8InputLayout::kGdnContiguousVScratchV1;
  resources.device_ordinal = g_v4_gdn_output_device_ordinal;
  return resources;
}

[[nodiscard]] Sm87MacroFeedV4Fp8CudaResources
host_test_v4_full_qkv_resources() noexcept {
  auto resources = host_test_v4_gdn_qkvz_resources();
  resources.identity = Sm87MacroFeedV4Fp8Identity::
      kFullQkvM64N128K64OrdinaryGridV1;
  resources.role = Sm87TargetAotProjectionRole::kFp8FullQkv;
  resources.input_layout =
      Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1;
  resources.device_ordinal = 0;
  return resources;
}

[[nodiscard]] Sm87MacroFeedV4Fp8CudaResources
host_test_v4_full_output_resources() noexcept {
  auto resources = host_test_v4_gdn_qkvz_resources();
  resources.identity = Sm87MacroFeedV4Fp8Identity::
      kAttentionOutputM64N128K64OrdinaryGridV1;
  resources.role = Sm87TargetAotProjectionRole::kFp8AttentionOutput;
  resources.input_layout =
      Sm87MacroFeedV4Fp8InputLayout::kFullAttentionInterleavedQScratchV1;
  resources.device_ordinal = 0;
  return resources;
}

[[nodiscard]] Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot
host_test_v4_full_preprocess_resources() noexcept {
  Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot resources;
  resources.identity = kSm87MacroFeedV4FullAttentionPreprocessIdentity;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessSmCount);
  resources.binary_version = 87;
  resources.kernel.registers_per_thread = 64;
  resources.kernel.static_shared_bytes =
      kSm87MacroFeedV4FullAttentionPreprocessStaticSharedBytes;
  resources.kernel.local_bytes = 0U;
  resources.kernel.maximum_threads_per_block = 1'024;
  resources.kernel.active_blocks_per_sm = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessRequiredCtasPerSm);
  resources.kernel.threads_per_block = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessThreads);
  resources.kernel.grid_x = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessTokens);
  resources.kernel.grid_y = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads);
  resources.kernel.physical_grid_ctas = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessPhysicalCtas);
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

[[nodiscard]] Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot
host_test_v4_full_attention_resources() noexcept {
  Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot resources;
  resources.identity = kSm87MacroFeedV4AttentionC8000Identity;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count =
      static_cast<std::int32_t>(kSm87MacroFeedV4AttentionC8000SmCount);
  resources.binary_version = 87;
  resources.kernel.registers_per_thread = 128;
  resources.kernel.static_shared_bytes = 0U;
  resources.kernel.dynamic_shared_bytes =
      kSm87MacroFeedV4AttentionC8000DynamicSharedBytes;
  resources.kernel.local_bytes = 0U;
  resources.kernel.maximum_threads_per_block = 1'024;
  resources.kernel.active_blocks_per_sm = static_cast<std::int32_t>(
      kSm87MacroFeedV4AttentionC8000RequiredCtasPerSm);
  resources.kernel.threads_per_block =
      static_cast<std::int32_t>(kSm87MacroFeedV4AttentionC8000Threads);
  resources.kernel.grid_x =
      static_cast<std::int32_t>(kSm87MacroFeedV4AttentionC8000GridX);
  resources.kernel.grid_y =
      static_cast<std::int32_t>(kSm87MacroFeedV4AttentionC8000GridY);
  resources.kernel.grid_z =
      static_cast<std::int32_t>(kSm87MacroFeedV4AttentionC8000GridZ);
  resources.kernel.physical_grid_ctas = static_cast<std::int32_t>(
      kSm87MacroFeedV4AttentionC8000PhysicalCtas);
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

[[nodiscard]] sm87_macrofeed_v4_bound_launch_detail::
    Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot
host_test_v4_embedding_resources() noexcept {
  using namespace sm87_macrofeed_v4_bound_launch_detail;
  Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot resources;
  resources.identity = 0x7634'656d'6263'3830ULL;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.gather = {32, 0U, 0U, 1'024, 4, 87, 256, 8'000};
  resources.exact_geometry = true;
  resources.static_resource_gate_passed = true;
  return resources;
}

[[nodiscard]] sm87_macrofeed_v4_bound_launch_detail::
    Sm87MacroFeedV4FinalNormM1ResourceSnapshot
host_test_v4_final_norm_resources() noexcept {
  using namespace sm87_macrofeed_v4_bound_launch_detail;
  Sm87MacroFeedV4FinalNormM1ResourceSnapshot resources;
  resources.identity = 0x7634'666e'6d31'3531ULL;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.centered_rms_norm = {32, 1'024U, 0U, 1'024, 4, 87, 256, 1};
  resources.exact_geometry = true;
  resources.static_resource_gate_passed = true;
  return resources;
}

[[nodiscard]] sm87_macrofeed_v4_bound_launch_detail::
    Sm87MacroFeedV4LmHeadM1ResourceSnapshot
host_test_v4_lm_head_resources() noexcept {
  using namespace sm87_macrofeed_v4_bound_launch_detail;
  Sm87MacroFeedV4LmHeadM1ResourceSnapshot resources;
  resources.identity = 0x7634'6c6d'6831'3234ULL;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.activation_staged = {64, 11'328U, 0U, 256, 4, 87, 256, 64};
  resources.exact_geometry = true;
  resources.static_resource_gate_passed = true;
  return resources;
}

[[nodiscard]] sm87_macrofeed_v4_bound_launch_detail::
    Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot
host_test_v4_greedy_resources() noexcept {
  using namespace sm87_macrofeed_v4_bound_launch_detail;
  Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot resources;
  resources.identity = 0x7634'6172'6731'3234ULL;
  resources.device_ordinal = 0;
  resources.compute_major = 8;
  resources.compute_minor = 7;
  resources.sm_count = 16;
  resources.partial = {32, 3'072U, 0U, 1'024, 4, 87, 256, 32};
  resources.finalize = {32, 0U, 0U, 1'024, 4, 87, 32, 1};
  resources.exact_geometry = true;
  resources.static_resource_gate_passed = true;
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

int query_sm87_macrofeed_v4_fp8_cuda_resources(
    const Sm87TargetAotProjectionRole role,
    const Sm87MacroFeedV4Fp8InputLayout input_layout,
    Sm87MacroFeedV4Fp8CudaResources* const resources) noexcept {
  if (resources == nullptr) {
    return 1;
  }
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ &&
      input_layout ==
          Sm87MacroFeedV4Fp8InputLayout::kHiddenContiguousH5120V1) {
    if (g_fail_v4_gdn_qkvz_resource_query) {
      *resources = {};
      return 1;
    }
    *resources = host_test_v4_gdn_qkvz_resources();
    return 0;
  }
  if (role == Sm87TargetAotProjectionRole::kFp8AttentionOutput &&
      input_layout ==
          Sm87MacroFeedV4Fp8InputLayout::kGdnContiguousVScratchV1) {
    if (g_fail_v4_gdn_output_resource_query) {
      *resources = {};
      return 1;
    }
    *resources = host_test_v4_gdn_output_resources();
    return 0;
  }
  *resources = {};
  return 1;
}

}  // namespace q3x::kernels

namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail {

class Sm87MacroFeedV4P40StartupPackageHostTestFixture final {
 public:
  [[nodiscard]] static Sm87MacroFeedV4P40StartupPackageCreateResult create(
      const ModelWeights& model_weights) noexcept {
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
    CUdeviceptr resident_arena_begin = 0U;
    std::size_t resident_arena_bytes = 0U;
    const auto* const embedding = model_weights.embed_tokens().weight;
    if (embedding == nullptr ||
        cuMemGetAddressRange(
            &resident_arena_begin, &resident_arena_bytes,
            static_cast<CUdeviceptr>(
                reinterpret_cast<std::uintptr_t>(embedding))) !=
            CUDA_SUCCESS ||
        resident_arena_begin == 0U || resident_arena_bytes == 0U) {
      Sm87MacroFeedV4P40StartupPackageCreateResult result;
      result.status.error =
          Sm87MacroFeedV4P40StartupPackageError::kProjectionAccessBind;
      result.status.context = "host_test_resident_allocation_query";
      return result;
    }
    return create(model_weights,
                  static_cast<std::uintptr_t>(resident_arena_begin),
                  resident_arena_bytes);
#else
    return Sm87MacroFeedV4P40StartupPackage::create(model_weights);
#endif
  }

  [[nodiscard]] static Sm87MacroFeedV4P40StartupPackageCreateResult create(
      const ModelWeights& model_weights,
      const std::uintptr_t resident_arena_begin,
      const std::uint64_t resident_arena_bytes) noexcept {
    auto access = target_aot_complete_execution_detail::
        Sm87TargetAotCompleteProjectionExecutionAccess::
            bind_complete_host_test_fixture(
                model_weights, resident_arena_begin, resident_arena_bytes);
    if (!access) {
      Sm87MacroFeedV4P40StartupPackageCreateResult result;
      result.status.error =
          Sm87MacroFeedV4P40StartupPackageError::kProjectionAccessBind;
      result.status.context = "host_test_resident_access_bind";
      return result;
    }
    return Sm87MacroFeedV4P40StartupPackage::create_from_host_test_authority(
        model_weights, std::move(*access));
  }

  [[nodiscard]] static bool typed_catalog_contract_closed(
      const Sm87MacroFeedV4P40StartupPackage& package,
      const ModelWeights& model_weights) noexcept {
    static_assert(std::tuple_size_v<
                      Sm87MacroFeedV4P40StartupPackage::
                          GdnLayerExecutionBindingCatalog> ==
                  kSm87MacroFeedV4P40StartupPackageGdnLayers);
    static_assert(std::tuple_size_v<
                      Sm87MacroFeedV4P40StartupPackage::
                          MlpPairExecutionBindingCatalog> ==
                  kSm87MacroFeedV4P40StartupPackageLayers);
    static_assert(std::tuple_size_v<
                      Sm87MacroFeedV4P40StartupPackage::
                          FullAttentionLayerExecutionBindingCatalog> ==
                  kSm87MacroFeedV4P40StartupPackageFullLayers);
    static_assert(std::tuple_size_v<
                      Sm87MacroFeedV4P40StartupPackage::
                          RequestBoundaryExecutionBindingCatalog> ==
                  kSm87MacroFeedV4P40StartupPackageRequestBoundaryBindings);
    static_assert(!std::is_same_v<
                  Sm87MacroFeedV4P40StartupPackage::GdnOutputExecutionBinding,
                  Sm87MacroFeedV4P40StartupPackage::GdnLayerExecutionBinding>);
    static_assert(!std::is_same_v<
                  Sm87MacroFeedV4P40StartupPackage::GateUpExecutionBinding,
                  Sm87MacroFeedV4P40StartupPackage::DownExecutionBinding>);
    static_assert(!std::is_same_v<
                  Sm87MacroFeedV4P40StartupPackage::
                      FullAttentionQkvExecutionBinding,
                  Sm87MacroFeedV4P40StartupPackage::
                      FullAttentionOutputExecutionBinding>);

    const std::uint64_t mlp_identity =
        Sm87MacroFeedV4P40StartupPackage::
            compute_mlp_pair_binding_catalog_identity(
                package.projection_access_, package.capabilities_,
                package.audit_.deployment_plan_identity);
    const std::uint64_t gdn_identity =
        Sm87MacroFeedV4P40StartupPackage::
            compute_gdn_qkvz_binding_catalog_identity(
                package.projection_access_, package.capabilities_,
                package.audit_.deployment_plan_identity,
                package.seals_.gdn_qkvz.resources,
                package.seals_.gdn_qkvz.output_resources,
                model_weights);

    auto full_output_substitution = package.seals_.gdn_qkvz.output_resources;
    full_output_substitution.identity =
        kernels::Sm87MacroFeedV4Fp8Identity::
            kAttentionOutputM64N128K64OrdinaryGridV1;
    full_output_substitution.input_layout =
        kernels::Sm87MacroFeedV4Fp8InputLayout::
            kFullAttentionInterleavedQScratchV1;
    const std::uint64_t substituted_identity =
        Sm87MacroFeedV4P40StartupPackage::
            compute_gdn_qkvz_binding_catalog_identity(
                package.projection_access_, package.capabilities_,
                package.audit_.deployment_plan_identity,
                package.seals_.gdn_qkvz.resources, full_output_substitution,
                model_weights);

    std::size_t source_failure =
        kSm87MacroFeedV4P40StartupPackageFullLayers;
    bool source_failure_k = false;
    int source_cuda_error = 0;
    const std::uint64_t full_source_identity =
        Sm87MacroFeedV4P40StartupPackage::
            compute_full_attention_source_catalog_identity(
                package.projection_access_, package.capabilities_,
                package.audit_.deployment_plan_identity, model_weights,
                &source_failure, &source_failure_k, &source_cuda_error);
    auto observations = make_full_attention_resource_observations();
    const std::uint64_t resource_bundle_identity =
        Sm87MacroFeedV4P40StartupPackage::
            compute_full_attention_resource_bundle_identity(
                observations, package.audit_.package_identity,
                package.audit_.deployment_plan_identity,
                package.audit_.device_identity,
                package.audit_.device_ordinal);
    for (std::size_t ordinal = 0U;
         ordinal < kSm87MacroFeedV4P40StartupPackageFullLayers; ++ordinal) {
      const std::size_t model_layer = 4U * ordinal + 3U;
      const auto qkv_index = sm87_target_aot_complete_descriptor_ordinal(
          model_layer, kernels::Sm87TargetAotProjectionRole::kFp8FullQkv);
      const auto output_index = sm87_target_aot_complete_descriptor_ordinal(
          model_layer,
          kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput);
      const auto* const full = std::get_if<FullAttentionWeights>(
          &model_weights.layer(model_layer).attention);
      if (qkv_index >= package.projection_bindings_.size() ||
          output_index >= package.projection_bindings_.size() ||
          !package.projection_bindings_[qkv_index] ||
          !package.projection_bindings_[output_index] || full == nullptr ||
          full->q_norm.data == nullptr || full->k_norm.data == nullptr ||
          full->q_norm.element_count != 256U ||
          full->k_norm.element_count != 256U ||
          package.projection_bindings_[qkv_index]->role() !=
              kernels::Sm87TargetAotProjectionRole::kFp8FullQkv ||
          package.projection_bindings_[qkv_index]
                  ->consumer_tactic_identity() !=
              static_cast<std::uint64_t>(
                  kernels::Sm87MacroFeedV4Fp8Identity::
                      kFullQkvM64N128K64OrdinaryGridV1) ||
          package.projection_bindings_[output_index]->role() !=
              kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput ||
          package.projection_bindings_[output_index]
                  ->consumer_tactic_identity() !=
              static_cast<std::uint64_t>(
                  kernels::Sm87MacroFeedV4Fp8Identity::
                      kAttentionOutputM64N128K64OrdinaryGridV1)) {
        return false;
      }
    }

    return mlp_identity != 0U &&
           mlp_identity == package.seals_.gate_up.binding_catalog_identity &&
           mlp_identity == package.seals_.down.binding_catalog_identity &&
           gdn_identity != 0U &&
           gdn_identity == package.seals_.gdn_qkvz.binding_catalog_identity &&
           substituted_identity == 0U && full_source_identity != 0U &&
           full_source_identity ==
               package.audit_.full_attention_source_catalog_identity &&
           source_failure == kSm87MacroFeedV4P40StartupPackageFullLayers &&
           !source_failure_k && source_cuda_error == 0 &&
           resource_bundle_identity != 0U;
  }

  [[nodiscard]] static bool
  request_boundary_resource_and_host_authority_fail_closed(
      const Sm87MacroFeedV4P40StartupPackage& package) noexcept {
    using Catalog = Sm87MacroFeedV4P40StartupPackage::
        RequestBoundaryExecutionBindingCatalog;
    const auto catalog_is_clear = [](const Catalog& catalog) noexcept {
      const auto& binding = catalog[0U];
      return binding.binding_identity == 0U &&
             binding.embedding.table == nullptr &&
             binding.final_norm.centered_weight == nullptr &&
             binding.lm_head.canonical_packed_weight == nullptr &&
             binding.lm_head.canonical_block_scale == nullptr &&
             binding.lm_head.weight_scale_2_device == nullptr &&
             binding.lm_head.input_scale_device == nullptr &&
             binding.resident_root == nullptr &&
             binding.resident_root_identity == 0U &&
             binding.resource_bundle_identity == 0U;
    };
    const auto rejected = [&](const auto& observations,
                              const bool bundle_must_be_zero) noexcept {
      const std::uint64_t bundle_identity =
          Sm87MacroFeedV4P40StartupPackage::
              compute_request_boundary_resource_bundle_identity(
                  observations, package.audit_.package_identity,
                  package.audit_.deployment_plan_identity,
                  package.audit_.device_identity,
                  package.audit_.device_ordinal);
      Catalog catalog{};
      catalog[0U].binding_identity = 1U;
      catalog[0U].resident_root = reinterpret_cast<const ResidentWeights*>(
          static_cast<std::uintptr_t>(16U));
      std::uint64_t identity = 1U;
      std::size_t failure =
          kSm87MacroFeedV4P40StartupPackageRequestBoundaryBindings;
      int cuda_error = 0;
      const bool sealed = package.
          seal_request_boundary_execution_catalog_for_execution_package(
              observations, &catalog, &identity, &failure, &cuda_error);
      return !sealed && identity == 0U && failure == 0U && cuda_error != 0 &&
             catalog_is_clear(catalog) &&
             (bundle_must_be_zero ? bundle_identity == 0U
                                  : bundle_identity != 0U);
    };

    const auto canonical = make_request_boundary_resource_observations();
    if (!rejected(canonical, false)) {
      return false;
    }
    const auto reject_mutation = [&](auto mutate) noexcept {
      auto observations = canonical;
      mutate(observations);
      return rejected(observations, true);
    };
    return reject_mutation([](auto& observations) noexcept {
             observations.embedding.identity = 0U;
           }) &&
           reject_mutation([](auto& observations) noexcept {
             observations.final_norm.exact_geometry = false;
           }) &&
           reject_mutation([](auto& observations) noexcept {
             observations.lm_head.activation_staged.registers_per_thread =
                 63;
           }) &&
           reject_mutation([](auto& observations) noexcept {
             observations.greedy.partial.grid_ctas = 31;
           }) &&
           reject_mutation([](auto& observations) noexcept {
             observations.embedding.device_ordinal = 1;
           }) &&
           reject_mutation([](auto& observations) noexcept {
             observations.source_private_queries_completed = false;
           }) &&
           reject_mutation([](auto& observations) noexcept {
             observations.caller_resource_snapshot_accepted = true;
           });
  }

  [[nodiscard]] static bool full_resource_substitutions_fail_closed(
      const Sm87MacroFeedV4P40StartupPackage& package) noexcept {
    using Catalog = Sm87MacroFeedV4P40StartupPackage::
        FullAttentionLayerExecutionBindingCatalog;
    const auto observations_rejected = [&](const auto& observations) noexcept {
      Catalog catalog{};
      std::uint64_t identity = 1U;
      std::size_t failure = kSm87MacroFeedV4P40StartupPackageFullLayers;
      int cuda_error = 0;
      const std::uint64_t resource_bundle_identity =
          Sm87MacroFeedV4P40StartupPackage::
              compute_full_attention_resource_bundle_identity(
                  observations, package.audit_.package_identity,
                  package.audit_.deployment_plan_identity,
                  package.audit_.device_identity,
                  package.audit_.device_ordinal);
      if (resource_bundle_identity != 0U ||
          package
              .seal_full_attention_execution_catalog_for_execution_package(
                  observations, &catalog, &identity, &failure, &cuda_error) ||
          identity != 0U || failure != 0U || cuda_error == 0) {
        return false;
      }
      for (const auto& binding : catalog) {
        if (binding.binding_identity != 0U ||
            binding.qkv.asset.payload.begin != 0U ||
            binding.output.asset.payload.begin != 0U ||
            binding.qk_norm.q_norm != nullptr ||
            binding.qk_norm.k_norm != nullptr) {
          return false;
        }
      }
      return true;
    };
    const auto rejected = [&](auto mutate) noexcept {
      auto observations = make_full_attention_resource_observations();
      mutate(observations);
      return observations_rejected(observations);
    };
    const Sm87MacroFeedV4P40StartupPackage::
        FullAttentionExecutionResourceObservations defaults{};
    return observations_rejected(defaults) &&
           rejected([](auto& observations) noexcept {
             observations.qkv.identity = kernels::Sm87MacroFeedV4Fp8Identity::
                 kGdnQkvZM64N128K64OrdinaryGridV1;
           }) &&
           rejected([](auto& observations) noexcept {
             observations.output.input_layout =
                 kernels::Sm87MacroFeedV4Fp8InputLayout::
                     kGdnContiguousVScratchV1;
           }) &&
           rejected([](auto& observations) noexcept {
             observations.preprocess.identity = kernels::
                 Sm87MacroFeedV4FullAttentionPreprocessIdentity::kInvalid;
           }) &&
           rejected([](auto& observations) noexcept {
             observations.attention.identity =
                 kernels::Sm87MacroFeedV4AttentionC8000Identity::kInvalid;
           }) &&
           rejected([](auto& observations) noexcept {
             observations.source_private_queries_completed = false;
           }) &&
           rejected([](auto& observations) noexcept {
             observations.caller_resource_snapshot_accepted = true;
           });
  }

 private:
  [[nodiscard]] static Sm87MacroFeedV4P40StartupPackage::
      FullAttentionExecutionResourceObservations
  make_full_attention_resource_observations() noexcept {
    Sm87MacroFeedV4P40StartupPackage::
        FullAttentionExecutionResourceObservations observations;
    observations.qkv = kernels::host_test_v4_full_qkv_resources();
    observations.output = kernels::host_test_v4_full_output_resources();
    observations.preprocess =
        kernels::host_test_v4_full_preprocess_resources();
    observations.attention =
        kernels::host_test_v4_full_attention_resources();
    observations.source_private_queries_completed = true;
    observations.caller_resource_snapshot_accepted = false;
    return observations;
  }

  [[nodiscard]] static Sm87MacroFeedV4P40StartupPackage::
      RequestBoundaryExecutionResourceObservations
      make_request_boundary_resource_observations() noexcept {
    Sm87MacroFeedV4P40StartupPackage::
        RequestBoundaryExecutionResourceObservations observations;
    observations.embedding = kernels::host_test_v4_embedding_resources();
    observations.final_norm = kernels::host_test_v4_final_norm_resources();
    observations.lm_head = kernels::host_test_v4_lm_head_resources();
    observations.greedy = kernels::host_test_v4_greedy_resources();
    observations.source_private_queries_completed = true;
    observations.caller_resource_snapshot_accepted = false;
    return observations;
  }
};

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail

namespace {

namespace package =
    q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail;
namespace execution =
    q3x::runtime::target_aot_complete_execution_detail;
using Package = package::Sm87MacroFeedV4P40StartupPackage;
using PackageFactory =
    package::Sm87MacroFeedV4P40StartupPackageHostTestFixture;
using Binding = package::Sm87MacroFeedV4ProjectionStartupBinding;
using PackageError = package::Sm87MacroFeedV4P40StartupPackageError;
using Access = execution::Sm87TargetAotCompleteProjectionExecutionAccess;
using HostBf16AbPair =
    execution::Sm87TargetAotCompleteHostTestBf16AbPair;
using HostFullQkNormPair =
    execution::Sm87TargetAotCompleteHostTestFullQkNormPair;
using HostRequestBoundary =
    execution::Sm87TargetAotCompleteHostTestRequestBoundary;
using Owner = q3x::runtime::Sm87TargetAotCompleteProjectionDeviceAssets;
using Role = q3x::kernels::Sm87TargetAotProjectionRole;
using q3x::runtime::Bf16LinearWeight;
using q3x::runtime::Bf16VectorWeight;
using q3x::runtime::Fp8LinearWeight;
using q3x::runtime::ModelWeights;
using q3x::runtime::NvFp4LinearWeight;

template <typename T, typename = void>
struct HasPublicStartupCreate : std::false_type {};

template <typename T>
struct HasPublicStartupCreate<
    T, std::void_t<decltype(T::create(
           std::declval<const ModelWeights&>()))>> : std::true_type {};

static_assert(!HasPublicStartupCreate<Package>::value);

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

[[nodiscard]] package::Sm87MacroFeedV4RequestBoundaryT0SourceDescriptor
make_request_boundary_t0_source() noexcept {
  namespace bound =
      q3x::kernels::sm87_macrofeed_v4_bound_launch_detail;
  constexpr std::uintptr_t kBase = 0x0000'0100'0000'0000ULL;
  constexpr std::uintptr_t kEmbedding = kBase;
  constexpr std::uintptr_t kFinalNorm =
      kEmbedding + bound::kSm87MacroFeedV4EmbeddingTableBytes;
  constexpr std::uintptr_t kPacked =
      kFinalNorm + bound::kSm87MacroFeedV4FinalNormBytes;
  constexpr std::uintptr_t kBlockScale =
      kPacked + bound::kSm87MacroFeedV4LmHeadPackedWeightBytes;
  constexpr std::uintptr_t kWeightScale =
      kBlockScale + bound::kSm87MacroFeedV4LmHeadBlockScaleBytes;
  constexpr std::uintptr_t kInputScale = kWeightScale + 16U;

  package::Sm87MacroFeedV4RequestBoundaryT0SourceDescriptor source;
  source.embedding = {
      reinterpret_cast<const std::uint16_t*>(kEmbedding),
      bound::kSm87MacroFeedV4EmbeddingVocabulary,
      bound::kSm87MacroFeedV4Hidden};
  source.final_norm = {
      reinterpret_cast<const std::uint16_t*>(kFinalNorm),
      bound::kSm87MacroFeedV4Hidden};
  source.lm_head = NvFp4LinearWeight{
      reinterpret_cast<const std::uint8_t*>(kPacked),
      reinterpret_cast<const std::uint8_t*>(kBlockScale),
      reinterpret_cast<const float*>(kWeightScale),
      reinterpret_cast<const float*>(kInputScale), 1.0F, 0.5F,
      bound::kSm87MacroFeedV4LmHeadRows,
      bound::kSm87MacroFeedV4LmHeadColumns};
  source.final_norm_epsilon_fp32_bits =
      bound::kSm87MacroFeedV4FinalNormEpsilonFp32Bits;
  source.greedy_vocabulary = bound::kSm87MacroFeedV4Vocabulary;
  source.greedy_workspace_results =
      bound::kSm87MacroFeedV4GreedyWorkspaceResults;
  source.greedy_strict_left_to_right_fp32_order = true;
  source.greedy_smallest_index_tie_break = true;
  source.greedy_nonfinite_reported_and_ignored = true;
  return source;
}

void test_request_boundary_t0_source_contract() {
  const auto canonical = make_request_boundary_t0_source();
  const auto audit =
      package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
          canonical);
  require_package(
      audit.valid_t0() && audit.failure_index == 6U &&
          audit.catalog_identity != 0U && audit.embedding_identity != 0U &&
          audit.final_norm_identity != 0U && audit.lm_head_identity != 0U &&
          audit.greedy_identity != 0U &&
          audit.input_scale_provenance_retained &&
          !audit.input_scale_consumed &&
          !audit.live_cuda_device_ranges_validated &&
          !audit.execution_capability,
      "canonical request-boundary T0 source did not close");

  auto wrong_embedding_shape = canonical;
  wrong_embedding_shape.embedding.output_size -= 1U;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           wrong_embedding_shape)
           .valid_t0(),
      "wrong embedding shape was accepted");
  auto null_embedding = canonical;
  null_embedding.embedding.weight = nullptr;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           null_embedding)
           .valid_t0(),
      "null embedding source was accepted");
  auto unaligned_embedding = canonical;
  unaligned_embedding.embedding.weight =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<std::uintptr_t>(canonical.embedding.weight) + 2U);
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           unaligned_embedding)
           .valid_t0(),
      "unaligned embedding source was accepted");

  auto wrong_norm = canonical;
  wrong_norm.final_norm.element_count -= 1U;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           wrong_norm)
           .valid_t0(),
      "wrong final norm shape was accepted");
  auto wrong_epsilon = canonical;
  wrong_epsilon.final_norm_epsilon_fp32_bits ^= 1U;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           wrong_epsilon)
           .valid_t0(),
      "wrong final norm epsilon bits were accepted");

  auto non_nvfp4 = canonical;
  non_nvfp4.lm_head = Bf16LinearWeight{
      canonical.embedding.weight,
      q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
          kSm87MacroFeedV4LmHeadRows,
      q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
          kSm87MacroFeedV4LmHeadColumns};
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           non_nvfp4)
           .valid_t0(),
      "non-NVFP4 LM head was accepted");
  auto wrong_lm_shape = canonical;
  std::get<NvFp4LinearWeight>(wrong_lm_shape.lm_head).input_size -= 1U;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           wrong_lm_shape)
           .valid_t0(),
      "wrong NVFP4 LM-head shape was accepted");
  auto null_scale_pointer = canonical;
  std::get<NvFp4LinearWeight>(null_scale_pointer.lm_head)
      .weight_scale_2_device = nullptr;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           null_scale_pointer)
           .valid_t0(),
      "null LM-head scale source was accepted");
  auto invalid_weight_scale = canonical;
  std::get<NvFp4LinearWeight>(invalid_weight_scale.lm_head).weight_scale_2 =
      0.0F;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           invalid_weight_scale)
           .valid_t0(),
      "invalid LM-head weight-scale raw bits were accepted");
  auto invalid_input_scale = canonical;
  std::get<NvFp4LinearWeight>(invalid_input_scale.lm_head).input_scale =
      -1.0F;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           invalid_input_scale)
           .valid_t0(),
      "invalid retained input-scale raw bits were accepted");

  auto overlap = canonical;
  auto& overlap_lm = std::get<NvFp4LinearWeight>(overlap.lm_head);
  overlap_lm.block_scale = overlap_lm.packed_weight;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(overlap)
           .valid_t0(),
      "overlapping request-boundary sources were accepted");
  auto scalar_substitution = canonical;
  auto& substituted_lm =
      std::get<NvFp4LinearWeight>(scalar_substitution.lm_head);
  std::swap(substituted_lm.weight_scale_2_device,
            substituted_lm.input_scale_device);
  const auto substituted_audit =
      package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
          scalar_substitution);
  require_package(
      substituted_audit.valid_t0() &&
          substituted_audit.catalog_identity != audit.catalog_identity &&
          substituted_audit.lm_head_identity != audit.lm_head_identity,
      "disjoint scalar-source substitution preserved boundary identity");

  auto wrong_greedy = canonical;
  wrong_greedy.greedy_smallest_index_tie_break = false;
  require_package(
      !package::inspect_sm87_macrofeed_v4_request_boundary_t0_source(
           wrong_greedy)
           .valid_t0(),
      "wrong greedy tie-break spec was accepted");
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
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)

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
    for (std::size_t ordinal = 0U; ordinal < full_qk_norm_pairs_.size();
         ++ordinal) {
      const std::size_t q_offset =
          kBf16AbBytes + (2U * ordinal) * kFullNormWeightBytes;
      const std::size_t k_offset = q_offset + kFullNormWeightBytes;
      full_qk_norm_pairs_[ordinal].q_norm = Bf16VectorWeight{
          reinterpret_cast<const std::uint16_t*>(bytes + q_offset),
          q3x::kernels::
              kSm87MacroFeedV4FullAttentionPreprocessHeadDimension};
      full_qk_norm_pairs_[ordinal].k_norm = Bf16VectorWeight{
          reinterpret_cast<const std::uint16_t*>(bytes + k_offset),
          q3x::kernels::
              kSm87MacroFeedV4FullAttentionPreprocessHeadDimension};
    }
    request_boundary_.embedding = Bf16LinearWeight{
        reinterpret_cast<const std::uint16_t*>(
            bytes + kRequestEmbeddingOffset),
        q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
            kSm87MacroFeedV4EmbeddingVocabulary,
        q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
            kSm87MacroFeedV4Hidden};
    request_boundary_.final_norm = Bf16VectorWeight{
        reinterpret_cast<const std::uint16_t*>(bytes + kRequestFinalNormOffset),
        q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
            kSm87MacroFeedV4Hidden};
    request_boundary_.lm_head = NvFp4LinearWeight{
        bytes + kRequestLmPackedOffset,
        bytes + kRequestLmBlockScaleOffset,
        reinterpret_cast<const float*>(bytes + kRequestWeightScaleOffset),
        reinterpret_cast<const float*>(bytes + kRequestInputScaleOffset),
        1.0F,
        1.0F,
        q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
            kSm87MacroFeedV4LmHeadRows,
        q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
            kSm87MacroFeedV4LmHeadColumns};
    constexpr float kScale = 1.0F;
    if (cudaMemcpy(bytes + kRequestWeightScaleOffset, &kScale, sizeof(kScale),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(bytes + kRequestInputScaleOffset, &kScale, sizeof(kScale),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      (void)cudaFree(allocation_);
      allocation_ = nullptr;
      request_boundary_ = {};
      return false;
    }
    return true;
  }

  [[nodiscard]] bool install(ModelWeights& weights,
                             const std::size_t pair_count =
                                 package::kSm87MacroFeedV4P40StartupPackageBf16AbPairs)
      noexcept {
    return allocation_ != nullptr && pair_count <= pairs_.size() &&
           Access::install_complete_host_test_bf16_ab_pairs(
               weights, pairs_.data(), pair_count) &&
           Access::install_complete_host_test_full_qk_norm_pairs(
               weights, full_qk_norm_pairs_.data(),
               full_qk_norm_pairs_.size()) &&
           Access::install_complete_host_test_request_boundary(
               weights, request_boundary_);
  }

  [[nodiscard]] HostBf16AbPair& pair(const std::size_t ordinal) noexcept {
    return pairs_[ordinal];
  }

  [[nodiscard]] HostFullQkNormPair& full_qk_norm_pair(
      const std::size_t ordinal) noexcept {
    return full_qk_norm_pairs_[ordinal];
  }

  [[nodiscard]] const std::uint16_t* one_past_allocation() const noexcept {
    return reinterpret_cast<const std::uint16_t*>(
        static_cast<const std::uint8_t*>(allocation_) + kAllocationBytes);
  }

  [[nodiscard]] std::uintptr_t arena_begin() const noexcept {
    return reinterpret_cast<std::uintptr_t>(allocation_);
  }

  [[nodiscard]] std::uint64_t arena_bytes() const noexcept {
    return kAllocationBytes;
  }

  [[nodiscard]] HostRequestBoundary& request_boundary() noexcept {
    return request_boundary_;
  }

 private:
  static constexpr std::size_t kWeightBytes =
      q3x::kernels::kSm87MacroFeedV4Bf16AbWeightBytes;
  static constexpr std::size_t kBf16AbBytes =
      package::kSm87MacroFeedV4P40StartupPackageBf16AbTensors *
      kWeightBytes;
  static constexpr std::size_t kFullNormWeightBytes =
      q3x::kernels::
          kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes;
  static constexpr std::size_t kFullNormBytes =
      kBf16AbBytes +
      2U * package::kSm87MacroFeedV4P40StartupPackageFullLayers *
          kFullNormWeightBytes;
  static constexpr std::size_t kRequestEmbeddingOffset = kFullNormBytes;
  static constexpr std::size_t kRequestFinalNormOffset =
      kRequestEmbeddingOffset +
      q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
          kSm87MacroFeedV4EmbeddingTableBytes;
  static constexpr std::size_t kRequestLmPackedOffset =
      kRequestFinalNormOffset +
      q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
          kSm87MacroFeedV4FinalNormBytes;
  static constexpr std::size_t kRequestLmBlockScaleOffset =
      kRequestLmPackedOffset +
      q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
          kSm87MacroFeedV4LmHeadPackedWeightBytes;
  static constexpr std::size_t kRequestWeightScaleOffset =
      kRequestLmBlockScaleOffset +
      q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
          kSm87MacroFeedV4LmHeadBlockScaleBytes;
  static constexpr std::size_t kRequestInputScaleOffset =
      kRequestWeightScaleOffset + 16U;
  static constexpr std::size_t kAllocationBytes =
      kRequestInputScaleOffset + 16U;

  void* allocation_ = nullptr;
  std::array<HostBf16AbPair,
             package::kSm87MacroFeedV4P40StartupPackageBf16AbPairs>
      pairs_{};
  std::array<HostFullQkNormPair,
             package::kSm87MacroFeedV4P40StartupPackageFullLayers>
      full_qk_norm_pairs_{};
  HostRequestBoundary request_boundary_{};
};

[[nodiscard]] bool live_sm87_test_device_available() noexcept {
  int device = -1;
  cudaDeviceProp properties{};
  return cudaGetDevice(&device) == cudaSuccess && device == 0 &&
         cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
         properties.major == 8 && properties.minor == 7;
}

void test_request_boundary_live_source_failures() {
  LiveBf16AbHostFixture fixture;
  require_package(fixture.allocate(),
                  "request-boundary live allocation failed");
  const HostRequestBoundary canonical = fixture.request_boundary();

  const auto reject = [&](const char* const context,
                          const std::size_t expected_failure_index,
                          const bool expect_cuda_error,
                          auto&& mutate) noexcept {
    fixture.request_boundary() = canonical;
    auto& lm = std::get<NvFp4LinearWeight>(fixture.request_boundary().lm_head);
    constexpr float kCanonicalScale = 1.0F;
    require_package(
        cudaMemcpy(const_cast<float*>(lm.weight_scale_2_device),
                   &kCanonicalScale, sizeof(kCanonicalScale),
                   cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(const_cast<float*>(lm.input_scale_device),
                       &kCanonicalScale, sizeof(kCanonicalScale),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            mutate(fixture.request_boundary()),
        context);

    Owner owner;
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && fixture.install(*weights), context);
    auto result = PackageFactory::create(*weights);
    require_package(
        !result && result.package == nullptr &&
            result.status.error ==
                PackageError::kRequestBoundarySourceCatalogSeal &&
            result.status.layer == expected_failure_index &&
            ((result.status.cuda_error != 0) == expect_cuda_error),
        context);
    require_package(clear_fixture(weights, owner), context);
  };

  reject("device weight_scale_2 raw-bit mismatch was accepted", 4U, true,
         [](HostRequestBoundary& boundary) noexcept {
           auto& lm = std::get<NvFp4LinearWeight>(boundary.lm_head);
           constexpr float kDifferentPositiveScale = 2.0F;
           return cudaMemcpy(const_cast<float*>(lm.weight_scale_2_device),
                             &kDifferentPositiveScale,
                             sizeof(kDifferentPositiveScale),
                             cudaMemcpyHostToDevice) == cudaSuccess;
         });
  reject("device input_scale raw-bit mismatch was accepted", 5U, true,
         [](HostRequestBoundary& boundary) noexcept {
           auto& lm = std::get<NvFp4LinearWeight>(boundary.lm_head);
           constexpr float kDifferentPositiveScale = 0.5F;
           return cudaMemcpy(const_cast<float*>(lm.input_scale_device),
                             &kDifferentPositiveScale,
                             sizeof(kDifferentPositiveScale),
                             cudaMemcpyHostToDevice) == cudaSuccess;
         });
  void* foreign_norm_allocation = nullptr;
  require_package(
      cudaMalloc(&foreign_norm_allocation,
                 q3x::kernels::sm87_macrofeed_v4_bound_launch_detail::
                     kSm87MacroFeedV4FinalNormBytes) == cudaSuccess &&
          foreign_norm_allocation != nullptr,
      "foreign request-boundary allocation failed");
  reject("foreign live CUDA allocation substituted the resident source", 1U,
         true, [&](HostRequestBoundary& boundary) noexcept {
           boundary.final_norm.data =
               static_cast<const std::uint16_t*>(foreign_norm_allocation);
           return true;
         });
  require_package(cudaFree(foreign_norm_allocation) == cudaSuccess,
                  "foreign request-boundary allocation cleanup failed");
  foreign_norm_allocation = nullptr;
  reject("one-past resident request-boundary range was accepted", 1U, true,
         [&](HostRequestBoundary& boundary) noexcept {
           boundary.final_norm.data = fixture.one_past_allocation();
           return true;
         });
  reject("overlapping request-boundary source substitution was accepted", 5U,
         false, [](HostRequestBoundary& boundary) noexcept {
           boundary.final_norm.data = boundary.embedding.weight;
           return true;
         });
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
  static_assert(!std::is_default_constructible_v<
                package::Sm87MacroFeedV4GdnQkvZStartupSeal>);
  static_assert(!std::is_copy_constructible_v<
                package::Sm87MacroFeedV4GdnQkvZStartupSeal>);
  static_assert(!std::is_move_constructible_v<
                package::Sm87MacroFeedV4GdnQkvZStartupSeal>);
  static_assert(!std::is_trivially_copyable_v<
                package::Sm87MacroFeedV4GdnQkvZStartupSeal>);
  static_assert(!std::is_default_constructible_v<
                package::Sm87MacroFeedV4FullAttentionStartupSeal>);
  static_assert(!std::is_copy_constructible_v<
                package::Sm87MacroFeedV4FullAttentionStartupSeal>);
  static_assert(!std::is_move_constructible_v<
                package::Sm87MacroFeedV4FullAttentionStartupSeal>);
  static_assert(!std::is_trivially_copyable_v<
                package::Sm87MacroFeedV4FullAttentionStartupSeal>);
  static_assert(!std::is_default_constructible_v<
                package::Sm87MacroFeedV4RequestBoundaryStartupSeal>);
  static_assert(!std::is_copy_constructible_v<
                package::Sm87MacroFeedV4RequestBoundaryStartupSeal>);
  static_assert(!std::is_move_constructible_v<
                package::Sm87MacroFeedV4RequestBoundaryStartupSeal>);
  static_assert(!std::is_trivially_copyable_v<
                package::Sm87MacroFeedV4RequestBoundaryStartupSeal>);

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

  auto first = PackageFactory::create(*model_weights);
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
          first.audit.bf16_ab_resource_seal_identity != 0U &&
          first.audit.gdn_qkvz_bindings == 48U &&
          first.audit.gdn_qkvz_binding_catalog_identity != 0U &&
          first.audit.gdn_qkvz_resource_seal_identity != 0U &&
          first.audit.full_attention_source_bindings == 16U &&
          first.audit.full_attention_source_catalog_identity != 0U &&
          first.audit.full_attention_source_seal_identity != 0U &&
          first.audit.request_boundary_source_bindings == 1U &&
          first.audit.request_boundary_source_catalog_identity != 0U &&
          first.audit.request_boundary_source_seal_identity != 0U &&
          first.audit.request_boundary_resident_root_identity != 0U &&
          first.audit.request_boundary_resident_arena_bytes ==
              bf16_ab.arena_bytes(),
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
          first.audit.gdn_qkvz_natural_layer_order_complete &&
          first.audit.gdn_qkvz_role_layout_tactic_fixed &&
          first.audit.gdn_qkvz_asset_value_snapshots_private &&
          first.audit.gdn_qkvz_resource_seal_complete &&
          !first.audit.gdn_qkvz_raw_pointer_publicly_exposed &&
          first.audit.full_attention_natural_layer_order_complete &&
          first.audit.full_attention_role_layout_tactics_fixed &&
          first.audit.full_attention_qk_norm_shapes_exact &&
          first.audit.full_attention_qk_norm_live_device_ranges_complete &&
          first.audit.full_attention_typed_asset_values_private &&
          !first.audit
               .full_attention_observed_resource_execution_catalog_sealed &&
          first.audit.request_boundary_exact_shapes_and_specs &&
          first.audit.request_boundary_live_device_ranges_complete &&
          first.audit.request_boundary_device_scale_raw_bits_match_host &&
          first.audit.request_boundary_input_scale_provenance_retained &&
          !first.audit.request_boundary_input_scale_consumed &&
          !first.audit
               .request_boundary_observed_resource_execution_catalog_sealed &&
          first.audit
              .request_boundary_final_representation_ready_diagnostic_only &&
          first.audit.pure_prefill_state_committed_endpoint_unchanged &&
          !first.audit.request_boundary_normal_resident_authority &&
          first.audit.request_boundary_host_test_resident_authority &&
          !first.audit.request_boundary_raw_pointer_publicly_exposed &&
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
  const auto& gdn_qkvz_seal = first.package->gdn_qkvz_startup_seal();
  const auto& full_seal = first.package->full_attention_startup_seal();
  const auto& request_seal =
      first.package->request_boundary_startup_seal();
  require_package(
      gate_seal.valid() && down_seal.valid() && bf16_ab_seal.valid() &&
          gdn_qkvz_seal.valid() && full_seal.valid() &&
          request_seal.valid() &&
          gate_seal.seal_identity != down_seal.seal_identity &&
          bf16_ab_seal.seal_identity != gate_seal.seal_identity &&
          bf16_ab_seal.seal_identity != down_seal.seal_identity &&
          gdn_qkvz_seal.seal_identity != gate_seal.seal_identity &&
          gdn_qkvz_seal.seal_identity != down_seal.seal_identity &&
          gdn_qkvz_seal.seal_identity != bf16_ab_seal.seal_identity &&
          full_seal.seal_identity != gate_seal.seal_identity &&
          full_seal.seal_identity != down_seal.seal_identity &&
          full_seal.seal_identity != bf16_ab_seal.seal_identity &&
          full_seal.seal_identity != gdn_qkvz_seal.seal_identity &&
          request_seal.seal_identity != full_seal.seal_identity &&
          gate_seal.package_identity == first.audit.package_identity &&
          down_seal.package_identity == first.audit.package_identity &&
          bf16_ab_seal.package_identity == first.audit.package_identity &&
          gdn_qkvz_seal.package_identity == first.audit.package_identity &&
          full_seal.package_identity == first.audit.package_identity &&
          request_seal.package_identity == first.audit.package_identity &&
          gate_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          down_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          bf16_ab_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          gdn_qkvz_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          full_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          request_seal.deployment_plan_identity ==
              first.audit.deployment_plan_identity &&
          bf16_ab_seal.binding_catalog_identity ==
              first.audit.bf16_ab_binding_catalog_identity &&
          gate_seal.binding_catalog_identity != 0U &&
          gate_seal.binding_catalog_identity ==
              down_seal.binding_catalog_identity &&
          gate_seal.binding_count == 64U && down_seal.binding_count == 64U &&
          gdn_qkvz_seal.binding_catalog_identity ==
              first.audit.gdn_qkvz_binding_catalog_identity &&
          bf16_ab_seal.tensor_count == 96U &&
          bf16_ab_seal.pair_count == 48U &&
          gdn_qkvz_seal.binding_count == 48U &&
          full_seal.source_catalog_identity ==
              first.audit.full_attention_source_catalog_identity &&
          full_seal.binding_count == 16U &&
          request_seal.source_catalog_identity ==
              first.audit.request_boundary_source_catalog_identity &&
          request_seal.resident_root_identity ==
              first.audit.request_boundary_resident_root_identity &&
          request_seal.resident_arena_bytes == bf16_ab.arena_bytes() &&
          request_seal.binding_count == 1U &&
          request_seal.final_norm_epsilon_fp32_bits == 0x3586'37bdU &&
          request_seal.weight_scale_2_fp32_bits == 0x3f80'0000U &&
          request_seal.input_scale_fp32_bits == 0x3f80'0000U &&
          full_seal.qkv_role == Role::kFp8FullQkv &&
          full_seal.qkv_input_layout ==
              q3x::kernels::Sm87MacroFeedV4Fp8InputLayout::
                  kHiddenContiguousH5120V1 &&
          full_seal.qkv_tactic_identity ==
              q3x::kernels::Sm87MacroFeedV4Fp8Identity::
                  kFullQkvM64N128K64OrdinaryGridV1 &&
          full_seal.output_role == Role::kFp8AttentionOutput &&
          full_seal.output_input_layout ==
              q3x::kernels::Sm87MacroFeedV4Fp8InputLayout::
                  kFullAttentionInterleavedQScratchV1 &&
          full_seal.output_tactic_identity ==
              q3x::kernels::Sm87MacroFeedV4Fp8Identity::
                  kAttentionOutputM64N128K64OrdinaryGridV1 &&
          full_seal.q_norm_elements == 256U &&
          full_seal.k_norm_elements == 256U &&
          full_seal.norm_weight_bytes == 512U &&
          q3x::kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
              gate_seal.resources) &&
          q3x::kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(
              down_seal.resources) &&
          q3x::kernels::
              sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
                  bf16_ab_seal.resources) &&
          q3x::kernels::sm87_macrofeed_v4_fp8_resource_gate(
              gdn_qkvz_seal.resources) &&
          q3x::kernels::sm87_macrofeed_v4_fp8_resource_gate(
              gdn_qkvz_seal.output_resources) &&
          gdn_qkvz_seal.role == Role::kFp8GdnQkvZ &&
          gdn_qkvz_seal.input_layout ==
              q3x::kernels::Sm87MacroFeedV4Fp8InputLayout::
                  kHiddenContiguousH5120V1 &&
          gdn_qkvz_seal.tactic_identity ==
              q3x::kernels::Sm87MacroFeedV4Fp8Identity::
                  kGdnQkvZM64N128K64OrdinaryGridV1 &&
          gdn_qkvz_seal.output_role == Role::kFp8AttentionOutput &&
          gdn_qkvz_seal.output_input_layout ==
              q3x::kernels::Sm87MacroFeedV4Fp8InputLayout::
                  kGdnContiguousVScratchV1 &&
          gdn_qkvz_seal.output_tactic_identity ==
              q3x::kernels::Sm87MacroFeedV4Fp8Identity::
                  kGdnAttentionOutputM64N128K64OrdinaryGridV1 &&
          bf16_ab_seal.canonical_natural_layer_order &&
          bf16_ab_seal.canonical_a_then_b_role_order &&
          bf16_ab_seal.complete_live_device_ranges &&
          !bf16_ab_seal.raw_pointer_exposed &&
          gdn_qkvz_seal.canonical_natural_gdn_layer_order &&
          gdn_qkvz_seal.role_layout_and_tactic_fixed &&
          gdn_qkvz_seal.output_role_layout_and_tactic_fixed &&
          gdn_qkvz_seal.continuation_weights_execution_seal_required &&
          gdn_qkvz_seal.typed_asset_values_private &&
          !gdn_qkvz_seal.raw_pointer_exposed &&
          full_seal.canonical_natural_full_layer_order &&
          full_seal.role_layout_and_tactics_fixed &&
          full_seal.qk_norm_exact_shapes &&
          full_seal.qk_norm_live_device_ranges &&
          full_seal.typed_asset_values_private &&
          full_seal.observed_resource_execution_seal_deferred &&
          !full_seal.raw_pointer_exposed &&
          request_seal.embedding_exact_bf16_shape &&
          request_seal.final_norm_exact_bf16_shape_and_epsilon &&
          request_seal.lm_head_exact_canonical_nvfp4_shape &&
          request_seal.device_scale_raw_bits_match_host &&
          request_seal.input_scale_provenance_retained &&
          !request_seal.input_scale_consumed && request_seal.greedy_spec_exact &&
          request_seal.complete_live_device_ranges &&
          request_seal.observed_resource_execution_seal_deferred &&
          request_seal.final_representation_ready_diagnostic_only &&
          request_seal.pure_prefill_state_committed_endpoint_unchanged &&
          !request_seal.normal_resident_authority &&
          request_seal.host_test_resident_authority &&
          !request_seal.raw_pointer_exposed &&
          !gate_seal.launcher_authority && !down_seal.launcher_authority &&
          !bf16_ab_seal.launcher_authority &&
          !gdn_qkvz_seal.launcher_authority &&
          !full_seal.launcher_authority &&
          !request_seal.launcher_authority &&
          !gate_seal.caller_receipt_accepted &&
          !down_seal.caller_receipt_accepted &&
          !bf16_ab_seal.caller_resource_snapshot_accepted &&
          !gdn_qkvz_seal.caller_resource_snapshot_accepted &&
          !full_seal.caller_resource_snapshot_accepted &&
          !request_seal.caller_resource_snapshot_accepted &&
          !bf16_ab_seal.production_dispatch_eligible &&
          !gdn_qkvz_seal.production_dispatch_eligible &&
          !full_seal.production_dispatch_eligible,
      "V4 C8000 startup resource seals are not independently closed");
  require_package(PackageFactory::typed_catalog_contract_closed(
                      *first.package, *model_weights),
                  "V4 typed GDN/MLP/Full catalogs or role isolation drifted");
  require_package(
      PackageFactory::full_resource_substitutions_fail_closed(*first.package),
      "V4 Full observed-resource substitution did not fail all-or-nothing");
  require_package(
      PackageFactory::request_boundary_resource_and_host_authority_fail_closed(
          *first.package),
      "V4 request-boundary resources or host authority did not fail closed");

  auto second = PackageFactory::create(*model_weights);
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
                     : (role == Role::kFp8GdnQkvZ
                            ? static_cast<std::uint64_t>(
                                  q3x::kernels::Sm87MacroFeedV4Fp8Identity::
                                      kGdnQkvZM64N128K64OrdinaryGridV1)
                            : (role == Role::kFp8FullQkv
                                   ? static_cast<std::uint64_t>(
                                         q3x::kernels::
                                             Sm87MacroFeedV4Fp8Identity::
                                                 kFullQkvM64N128K64OrdinaryGridV1)
                                   : static_cast<std::uint64_t>(
                                         (q3x::runtime::
                                                  sm87_target_aot_complete_is_full_layer(
                                                      layer)
                                              ? q3x::kernels::
                                                    Sm87MacroFeedV4Fp8Identity::
                                                        kAttentionOutputM64N128K64OrdinaryGridV1
                                              : q3x::kernels::
                                                    Sm87MacroFeedV4Fp8Identity::
                                                        kGdnAttentionOutputM64N128K64OrdinaryGridV1)))));
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
    auto result = PackageFactory::create(*weights);
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
    auto result = PackageFactory::create(*weights);
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
    auto result = PackageFactory::create(*weights);
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
    auto result = PackageFactory::create(*weights);
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
    auto result = PackageFactory::create(*weights);
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
    auto result = PackageFactory::create(*weights);
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
    auto result = PackageFactory::create(*weights);
    require_package(
        !result && result.status.error == PackageError::kBf16AbDeviceRange &&
            result.status.layer == 62U && result.status.cuda_error != 0,
        "non-resident BF16 A/B allocation range was accepted");
    require_package(clear_fixture(weights, owner),
                    "range BF16 fixture cleanup failed");
  }
}

void test_full_attention_qk_norm_failures() {
  {
    Owner owner;
    LiveBf16AbHostFixture fixture;
    require_package(fixture.allocate(), "missing Full Q norm allocation failed");
    fixture.full_qk_norm_pair(0U).q_norm = {};
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && fixture.install(*weights),
                    "missing Full Q norm fixture setup failed");
    auto result = PackageFactory::create(*weights);
    require_package(
        !result && result.status.error ==
                       PackageError::kFullAttentionSourceCatalogSeal &&
            result.status.layer == 3U && result.status.cuda_error == 0,
        "missing Full Q norm was accepted");
    require_package(clear_fixture(weights, owner),
                    "missing Full Q norm fixture cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture fixture;
    require_package(fixture.allocate(), "Full K norm shape allocation failed");
    fixture.full_qk_norm_pair(4U).k_norm.element_count -= 1U;
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && fixture.install(*weights),
                    "Full K norm shape fixture setup failed");
    auto result = PackageFactory::create(*weights);
    require_package(
        !result && result.status.error ==
                       PackageError::kFullAttentionSourceCatalogSeal &&
            result.status.layer == 19U && result.status.cuda_error == 0,
        "wrong Full K norm shape was accepted");
    require_package(clear_fixture(weights, owner),
                    "Full K norm shape fixture cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture fixture;
    require_package(fixture.allocate(), "Full Q norm range allocation failed");
    fixture.full_qk_norm_pair(15U).q_norm = Bf16VectorWeight{
        fixture.one_past_allocation(),
        q3x::kernels::
            kSm87MacroFeedV4FullAttentionPreprocessHeadDimension};
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && fixture.install(*weights),
                    "Full Q norm range fixture setup failed");
    auto result = PackageFactory::create(*weights);
    require_package(
        !result && result.status.error ==
                       PackageError::kFullAttentionSourceCatalogSeal &&
            result.status.layer == 63U && result.status.cuda_error != 0,
        "non-resident Full Q norm range was accepted");
    require_package(clear_fixture(weights, owner),
                    "Full Q norm range fixture cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture fixture;
    require_package(fixture.allocate(),
                    "Full Q/K identity substitution allocation failed");
    fixture.full_qk_norm_pair(8U).k_norm =
        fixture.full_qk_norm_pair(8U).q_norm;
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && fixture.install(*weights),
                    "Full Q/K identity substitution fixture setup failed");
    auto result = PackageFactory::create(*weights);
    require_package(
        !result && result.status.error ==
                       PackageError::kFullAttentionSourceCatalogSeal &&
            result.status.layer == 35U,
        "Full K norm accepted a substituted Q norm identity");
    require_package(clear_fixture(weights, owner),
                    "Full Q/K identity substitution cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture fixture;
    require_package(fixture.allocate(),
                    "Full Q/K partial-overlap allocation failed");
    const auto q_norm = fixture.full_qk_norm_pair(11U).q_norm;
    fixture.full_qk_norm_pair(11U).k_norm = Bf16VectorWeight{
        q_norm.data + 8U,
        q3x::kernels::
            kSm87MacroFeedV4FullAttentionPreprocessHeadDimension};
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && fixture.install(*weights),
                    "Full Q/K partial-overlap fixture setup failed");
    auto result = PackageFactory::create(*weights);
    require_package(
        !result && result.status.error ==
                       PackageError::kFullAttentionSourceCatalogSeal &&
            result.status.layer == 47U,
        "partially overlapping Full Q/K norm ranges were accepted");
    require_package(clear_fixture(weights, owner),
                    "Full Q/K partial-overlap cleanup failed");
  }
  {
    Owner owner;
    LiveBf16AbHostFixture fixture;
    require_package(fixture.allocate(),
                    "cross-layer Full norm overlap allocation failed");
    const auto prior_k = fixture.full_qk_norm_pair(4U).k_norm;
    fixture.full_qk_norm_pair(5U).q_norm = Bf16VectorWeight{
        prior_k.data + 8U,
        q3x::kernels::
            kSm87MacroFeedV4FullAttentionPreprocessHeadDimension};
    std::optional<ModelWeights> weights =
        Access::make_complete_host_test_fixture(owner);
    require_package(weights.has_value() && fixture.install(*weights),
                    "cross-layer Full norm overlap fixture setup failed");
    auto result = PackageFactory::create(*weights);
    require_package(
        !result && result.status.error ==
                       PackageError::kFullAttentionSourceCatalogSeal &&
            result.status.layer == 23U && result.status.cuda_error != 0,
        "cross-layer partially overlapping Full norm ranges were accepted");
    require_package(clear_fixture(weights, owner),
                    "cross-layer Full norm overlap cleanup failed");
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
  auto result = PackageFactory::create(*weights);
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
  auto gate_failure = PackageFactory::create(*weights);
  q3x::kernels::g_fail_v4_gate_up_resource_query = false;
  require_package(
      !gate_failure && gate_failure.package == nullptr &&
          gate_failure.status.error == PackageError::kGateUpStartupSeal &&
          gate_failure.status.cuda_error == 1,
      "V4 Gate+Up resource query failure did not fail closed");

  q3x::kernels::g_fail_v4_bf16_ab_resource_query = true;
  auto bf16_query_failure = PackageFactory::create(*weights);
  q3x::kernels::g_fail_v4_bf16_ab_resource_query = false;
  require_package(
      !bf16_query_failure && bf16_query_failure.package == nullptr &&
          bf16_query_failure.status.error ==
              PackageError::kBf16AbResourceSeal &&
          bf16_query_failure.status.cuda_error == 1,
      "V4 BF16 A/B resource query failure did not fail closed");

  q3x::kernels::g_poison_v4_bf16_ab_resource_identity = true;
  auto bf16_identity_failure = PackageFactory::create(*weights);
  q3x::kernels::g_poison_v4_bf16_ab_resource_identity = false;
  require_package(
      !bf16_identity_failure && bf16_identity_failure.package == nullptr &&
          bf16_identity_failure.status.error ==
              PackageError::kBf16AbResourceSeal,
      "caller-forgeable BF16 A/B resource identity was accepted");

  q3x::kernels::g_fail_v4_gdn_qkvz_resource_query = true;
  auto gdn_query_failure = PackageFactory::create(*weights);
  q3x::kernels::g_fail_v4_gdn_qkvz_resource_query = false;
  require_package(
      !gdn_query_failure && gdn_query_failure.package == nullptr &&
          gdn_query_failure.status.error ==
              PackageError::kGdnQkvZResourceSeal &&
          gdn_query_failure.status.cuda_error == 1,
      "V4 GDN-QKVZ resource query failure did not fail closed");

  q3x::kernels::g_poison_v4_gdn_qkvz_resource_identity = true;
  auto gdn_identity_failure = PackageFactory::create(*weights);
  q3x::kernels::g_poison_v4_gdn_qkvz_resource_identity = false;
  require_package(
      !gdn_identity_failure && gdn_identity_failure.package == nullptr &&
          gdn_identity_failure.status.error ==
              PackageError::kGdnQkvZResourceSeal,
      "changed GDN-QKVZ role/layout/tactic resource was accepted");

  q3x::kernels::g_fail_v4_gdn_output_resource_query = true;
  auto gdn_output_query_failure = PackageFactory::create(*weights);
  q3x::kernels::g_fail_v4_gdn_output_resource_query = false;
  require_package(
      !gdn_output_query_failure &&
          gdn_output_query_failure.package == nullptr &&
          gdn_output_query_failure.status.error ==
              PackageError::kGdnQkvZResourceSeal &&
          gdn_output_query_failure.status.cuda_error == 1,
      "V4 dedicated GDN-O resource query failure did not fail closed");

  q3x::kernels::g_poison_v4_gdn_output_resource_identity = true;
  auto gdn_output_identity_failure = PackageFactory::create(*weights);
  q3x::kernels::g_poison_v4_gdn_output_resource_identity = false;
  require_package(
      !gdn_output_identity_failure &&
          gdn_output_identity_failure.package == nullptr &&
          gdn_output_identity_failure.status.error ==
              PackageError::kGdnQkvZResourceSeal,
      "Full-O/invalid tactic substitution was accepted as GDN-O");

  q3x::kernels::g_v4_down_device_ordinal = 1;
  auto device_failure = PackageFactory::create(*weights);
  q3x::kernels::g_v4_down_device_ordinal = 0;
  require_package(
      !device_failure && device_failure.package == nullptr &&
          device_failure.status.error == PackageError::kDeviceMismatch,
      "V4 cross-device startup resource set did not fail closed");

  q3x::kernels::g_v4_bf16_ab_device_ordinal = 1;
  auto bf16_device_failure = PackageFactory::create(*weights);
  q3x::kernels::g_v4_bf16_ab_device_ordinal = 0;
  require_package(
      !bf16_device_failure && bf16_device_failure.package == nullptr &&
          bf16_device_failure.status.error == PackageError::kDeviceMismatch,
      "V4 BF16 A/B resource seal accepted another device");

  q3x::kernels::g_v4_gdn_qkvz_device_ordinal = 1;
  auto gdn_device_failure = PackageFactory::create(*weights);
  q3x::kernels::g_v4_gdn_qkvz_device_ordinal = 0;
  require_package(
      !gdn_device_failure && gdn_device_failure.package == nullptr &&
          gdn_device_failure.status.error == PackageError::kDeviceMismatch,
      "V4 GDN-QKVZ resource seal accepted another device");

  q3x::kernels::g_v4_gdn_output_device_ordinal = 1;
  auto gdn_output_device_failure = PackageFactory::create(*weights);
  q3x::kernels::g_v4_gdn_output_device_ordinal = 0;
  require_package(
      !gdn_output_device_failure &&
          gdn_output_device_failure.package == nullptr &&
          gdn_output_device_failure.status.error ==
              PackageError::kDeviceMismatch,
      "V4 GDN-O resource seal accepted another device");
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
  auto result = PackageFactory::create(*weights);
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
  test_request_boundary_t0_source_contract();
  test_bf16_ab_t0_inventory_contract();
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
  if (!live_sm87_test_device_available()) {
    std::cout << "sm87_macrofeed_v4_p40_startup_package_host_test: "
                 "SKIP (T0 PASS; live SM87 T1 unavailable)\n";
    return 77;
  }
  test_complete_v4_foundation_package();
  test_request_boundary_live_source_failures();
  test_source_scale_and_device_tamper_fail_closed();
  test_bf16_ab_model_and_range_failures();
  test_full_attention_qk_norm_failures();
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
